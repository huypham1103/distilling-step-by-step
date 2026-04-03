import argparse
import json
import math
import os
from multiprocessing import get_context
from pathlib import Path
import time
from typing import List, Sequence, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration


def load_test_frame(test_data_path: str) -> pd.DataFrame:
    try:
        dataset = load_dataset('csv', data_files={'test': test_data_path})['test']
        dataframe = pd.DataFrame(dataset)
    except Exception:
        dataframe = pd.read_csv(test_data_path, keep_default_na=False)
    if 'split' in dataframe.columns:
        split = dataframe['split'].astype(str).str.strip().str.lower()
        dataframe = dataframe[split == 'test'].copy()
        if dataframe.empty:
            raise ValueError('The provided test_data_path has a split column but contains no "test" rows.')

    required_columns = {'input', 'label'}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f'Test data is missing required columns: {sorted(missing_columns)}')

    dataframe['input'] = dataframe['input'].fillna('').astype(str)
    dataframe['label'] = dataframe['label'].fillna('').astype(str)
    return dataframe.reset_index(drop=True)


def format_inputs(inputs: Sequence[str], model_type: str, add_task_prefix: bool) -> List[str]:
    if model_type == 'task_prefix' and add_task_prefix:
        return [f'predict: {text}' for text in inputs]
    return list(inputs)


def choose_autocast_dtype(device: str, bf16: bool, fp16: bool):
    if device != 'cuda':
        return None
    if not bf16 and not fp16:
        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if bf16 and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if fp16:
        return torch.float16
    return None


def generate_predictions_single_gpu(
    model_path: str,
    inputs: Sequence[str],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    bf16: bool,
    fp16: bool,
    gpu_id: int | None = None,
) -> List[str]:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    model.config.use_cache = True

    autocast_dtype = choose_autocast_dtype('cuda' if device.startswith('cuda') else device, bf16, fp16)
    predictions: List[str] = []

    iterator = range(0, len(inputs), batch_size)
    if gpu_id is None or gpu_id == 0:
        iterator = tqdm(iterator, desc='Generating')

    for start in iterator:
        batch_inputs = list(inputs[start:start + batch_size])
        tokenized = tokenizer(
            batch_inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}

        with torch.inference_mode():
            if autocast_dtype is not None:
                with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                    output = model.generate(**tokenized, max_new_tokens=max_new_tokens)
            else:
                output = model.generate(**tokenized, max_new_tokens=max_new_tokens)

        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        predictions.extend(text.strip() for text in decoded)

    return predictions


def _worker_generate(payload: Tuple[int, str, List[Tuple[int, str]], int, int, int, bool, bool]):
    gpu_id, model_path, indexed_inputs, batch_size, max_input_length, max_new_tokens, bf16, fp16 = payload
    indices = [index for index, _ in indexed_inputs]
    inputs = [text for _, text in indexed_inputs]
    predictions = generate_predictions_single_gpu(
        model_path=model_path,
        inputs=inputs,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        bf16=bf16,
        fp16=fp16,
        gpu_id=gpu_id,
    )
    return list(zip(indices, predictions))


def generate_predictions_multi_gpu(
    model_path: str,
    inputs: Sequence[str],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    bf16: bool,
    fp16: bool,
    num_gpus: int,
) -> List[str]:
    indexed_inputs = list(enumerate(inputs))
    shards = [indexed_inputs[gpu_index::num_gpus] for gpu_index in range(num_gpus)]
    payloads = [
        (gpu_index, model_path, shard, batch_size, max_input_length, max_new_tokens, bf16, fp16)
        for gpu_index, shard in enumerate(shards)
        if shard
    ]

    ctx = get_context('spawn')
    with ctx.Pool(processes=len(payloads)) as pool:
        results = pool.map(_worker_generate, payloads)

    flattened = [item for shard in results for item in shard]
    flattened.sort(key=lambda item: item[0])
    return [prediction for _, prediction in flattened]


def score_predictions(labels: Sequence[str], predictions: Sequence[str]) -> dict:
    label_norm = pd.Series(labels, dtype='object').astype(str).str.strip().str.lower()
    pred_raw = pd.Series(predictions, dtype='object').astype(str)
    pred_norm = pred_raw.str.strip().str.lower()

    score_1 = float((pred_norm == label_norm).mean())
    score_2 = float(sum(label in output for label, output in zip(label_norm, pred_norm)) / len(label_norm))
    score_3 = float(sum(label in output.lower() for label, output in zip(label_norm, pred_raw)) / len(label_norm))

    return {
        'test_accuracy_exact': score_1,
        'test_accuracy_contains_norm': score_2,
        'test_accuracy_contains_raw_lower': score_3,
        'num_examples': int(len(label_norm)),
    }


def empty_prediction_ratio(predictions: Sequence[str]) -> float:
    if not predictions:
        return 1.0
    return sum(not bool(prediction) for prediction in predictions) / len(predictions)


def run_prediction_pass(
    model_path: str,
    inputs: Sequence[str],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    bf16: bool,
    fp16: bool,
    use_multi_gpu: bool,
    num_gpus: int | None,
) -> tuple[list[str], float]:
    start_time = time.perf_counter()
    if use_multi_gpu and num_gpus and num_gpus > 1:
        predictions = generate_predictions_multi_gpu(
            model_path=model_path,
            inputs=inputs,
            batch_size=batch_size,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            bf16=bf16,
            fp16=fp16,
            num_gpus=num_gpus,
        )
    else:
        predictions = generate_predictions_single_gpu(
            model_path=model_path,
            inputs=inputs,
            batch_size=batch_size,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            bf16=bf16,
            fp16=fp16,
        )
    return predictions, time.perf_counter() - start_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='artifacts/test_eval')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--disable_task_prefix', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--num_gpus', type=int, default=None)
    args = parser.parse_args()

    test_frame = load_test_frame(args.test_data_path)
    add_task_prefix = args.model_type == 'task_prefix' and not args.disable_task_prefix
    inputs = format_inputs(
        test_frame['input'].tolist(),
        model_type=args.model_type,
        add_task_prefix=add_task_prefix,
    )

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    requested_gpus = args.num_gpus or available_gpus
    use_multi_gpu = args.multi_gpu and available_gpus > 1 and requested_gpus > 1

    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'GPU count: {available_gpus}')
    if torch.cuda.is_available():
        print(f'Primary GPU: {torch.cuda.get_device_name(0)}')
    print(f'Using multi_gpu: {use_multi_gpu}')
    print(f'Input formatting: {"predict: <input>" if add_task_prefix else "raw input"}')

    num_gpus = min(requested_gpus, available_gpus) if use_multi_gpu else None
    predictions, prediction_seconds = run_prediction_pass(
        model_path=args.model_path,
        inputs=inputs,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.gen_max_len,
        bf16=args.bf16,
        fp16=args.fp16,
        use_multi_gpu=use_multi_gpu,
        num_gpus=num_gpus,
    )

    fallback_report = []
    current_empty_ratio = empty_prediction_ratio(predictions)
    if current_empty_ratio >= 0.999:
        print('Initial prediction pass returned all-empty outputs. Retrying safer evaluation modes...')
        fallback_candidates = []
        if use_multi_gpu:
            fallback_candidates.append({
                'name': 'single_gpu_same_precision',
                'inputs': inputs,
                'bf16': args.bf16,
                'fp16': args.fp16,
            })
        if args.bf16 or args.fp16:
            fallback_candidates.append({
                'name': 'single_gpu_full_precision',
                'inputs': inputs,
                'bf16': False,
                'fp16': False,
            })
        if args.model_type == 'task_prefix':
            fallback_candidates.append({
                'name': 'single_gpu_raw_input_full_precision',
                'inputs': format_inputs(test_frame['input'].tolist(), args.model_type, add_task_prefix=False),
                'bf16': False,
                'fp16': False,
            })
            fallback_candidates.append({
                'name': 'single_gpu_prefixed_input_full_precision',
                'inputs': format_inputs(test_frame['input'].tolist(), args.model_type, add_task_prefix=True),
                'bf16': False,
                'fp16': False,
            })

        for candidate in fallback_candidates:
            candidate_predictions, candidate_seconds = run_prediction_pass(
                model_path=args.model_path,
                inputs=candidate['inputs'],
                batch_size=args.batch_size,
                max_input_length=args.max_input_length,
                max_new_tokens=args.gen_max_len,
                bf16=candidate['bf16'],
                fp16=candidate['fp16'],
                use_multi_gpu=False,
                num_gpus=None,
            )
            candidate_empty_ratio = empty_prediction_ratio(candidate_predictions)
            fallback_entry = {
                'name': candidate['name'],
                'empty_ratio': candidate_empty_ratio,
                'seconds': candidate_seconds,
                'preview': candidate_predictions[:5],
            }
            fallback_report.append(fallback_entry)
            print('Fallback attempt:', json.dumps(fallback_entry, indent=2))
            if candidate_empty_ratio < current_empty_ratio:
                predictions = candidate_predictions
                prediction_seconds = candidate_seconds
                current_empty_ratio = candidate_empty_ratio
                if candidate['name'] == 'single_gpu_raw_input_full_precision':
                    add_task_prefix = False
                elif candidate['name'] == 'single_gpu_prefixed_input_full_precision':
                    add_task_prefix = True
                if candidate_empty_ratio < 0.999:
                    break

    metrics = score_predictions(test_frame['label'].tolist(), predictions)
    metrics['prediction_seconds'] = float(prediction_seconds)
    metrics['seconds_per_example'] = float(prediction_seconds / max(len(test_frame), 1))
    metrics['prediction_empty_ratio'] = float(current_empty_ratio)
    if fallback_report:
        metrics['fallback_report'] = fallback_report
    normalized_predictions = [prediction if prediction is not None else '' for prediction in predictions]
    result_frame = test_frame.copy()
    result_frame['prediction'] = normalized_predictions
    result_frame['prediction_is_empty'] = [not bool(prediction) for prediction in normalized_predictions]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{Path(args.model_path).name}_test'
    predictions_path = output_dir / f'{stem}_predictions.csv'
    metrics_path = output_dir / f'{stem}_metrics.json'

    result_frame.to_csv(predictions_path, index=False)
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    print(f'Prediction time: {prediction_seconds:.2f}s total, {metrics["seconds_per_example"]:.4f}s/example')
    print('Prediction preview:', normalized_predictions[:5])
    print(json.dumps(metrics, indent=2))
    print(f'Test predictions saved to {predictions_path}')
    print(f'Test metrics saved to {metrics_path}')


if __name__ == '__main__':
    main()
