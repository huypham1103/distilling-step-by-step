import argparse
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5ForConditionalGeneration

from metrics import compute_metrics_equation, compute_metrics_equation_aux, compute_metrics_text, compute_metrics_text_aux
from model_utils import TaskPrefixDataCollator, TaskPrefixTrainer
from run import (
    build_task_prefix_tokenize_function,
    find_rationale_indices,
    load_selected_rationale_datasets,
    map_datasetdict_per_split,
    tokenize_targets,
)


def build_eval_args_kwargs(args, output_dir):
    kwargs = {
        'output_dir': output_dir,
        'remove_unused_columns': False,
        'per_device_eval_batch_size': args.eval_batch_size or args.batch_size,
        'predict_with_generate': True,
        'generation_max_length': args.gen_max_len,
        'bf16': getattr(args, 'bf16', False),
        'fp16': getattr(args, 'fp16', False),
        'dataloader_pin_memory': bool(torch.cuda.is_available()),
        'dataloader_num_workers': getattr(args, 'dataloader_num_workers', 0),
        'tf32': getattr(args, 'tf32', False),
        'report_to': [],
        'do_train': False,
        'do_eval': False,
        'do_predict': True,
    }

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    supported_params = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported_params}


def build_compute_metrics(args, tokenizer):
    if args.model_type == 'standard':
        if args.dataset not in ['svamp', 'asdiv']:
            return compute_metrics_text_aux(tokenizer)
        return compute_metrics_equation_aux(tokenizer)

    if args.dataset not in ['svamp', 'asdiv']:
        return compute_metrics_text(tokenizer)
    return compute_metrics_equation(tokenizer)


def build_tokenized_selected_datasets(args, tokenizer):
    raw_datasets = load_selected_rationale_datasets(args.selected_rationale_path)
    rationale_indices = find_rationale_indices(raw_datasets['train'].column_names) if args.model_type == 'task_prefix' else []

    if args.model_type == 'task_prefix':
        tokenize_function = build_task_prefix_tokenize_function(tokenizer, args, rationale_indices)
        tokenized = map_datasetdict_per_split(raw_datasets, tokenize_function, remove_columns_strategy='all')
    elif args.model_type == 'standard':
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True
            )
            label_output_encodings = tokenize_targets(tokenizer, examples['label'], max_length=256)
            model_inputs['labels'] = label_output_encodings['input_ids']
            return model_inputs

        tokenized = map_datasetdict_per_split(raw_datasets, tokenize_function, remove_columns_strategy='all')
    else:
        raise ValueError(f'Unsupported model_type: {args.model_type}')

    return raw_datasets, tokenized


def build_trainer(args, training_args, model, tokenized_datasets, tokenizer, compute_metrics):
    if args.model_type == 'task_prefix':
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
        trainer_kwargs = {
            'alpha': args.alpha,
            'output_rationale': args.output_rationale,
            'model': model,
            'args': training_args,
            'train_dataset': tokenized_datasets['train'],
            'eval_dataset': {'test': tokenized_datasets['test']},
            'data_collator': data_collator,
            'tokenizer': tokenizer,
            'processing_class': tokenizer,
            'compute_metrics': compute_metrics,
        }
        trainer_signature = inspect.signature(TaskPrefixTrainer.__init__)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in trainer_signature.parameters.values()):
            base_signature = inspect.signature(Seq2SeqTrainer.__init__)
            supported_params = set(base_signature.parameters).union({'alpha', 'output_rationale'})
        else:
            supported_params = set(trainer_signature.parameters)
        trainer_kwargs = {key: value for key, value in trainer_kwargs.items() if key in supported_params}
        return TaskPrefixTrainer(**trainer_kwargs)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets['train'],
        'eval_dataset': tokenized_datasets['test'],
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'processing_class': tokenizer,
        'compute_metrics': compute_metrics,
    }
    signature = inspect.signature(Seq2SeqTrainer.__init__)
    supported_params = set(signature.parameters)
    trainer_kwargs = {key: value for key, value in trainer_kwargs.items() if key in supported_params}
    return Seq2SeqTrainer(**trainer_kwargs)


def select_primary_prediction(predictions):
    if isinstance(predictions, (list, tuple)):
        return predictions[0]
    return predictions


def maybe_select_aux_prediction(predictions):
    if isinstance(predictions, (list, tuple)) and len(predictions) > 1:
        return predictions[1]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--selected_rationale_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='artifacts/test_eval')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=None)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--selection_policy', type=str, default='heuristic')
    parser.add_argument('--num_selected_rationales', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--llm', type=str, default='palm')
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--tf32', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    raw_datasets, tokenized_datasets = build_tokenized_selected_datasets(args, tokenizer)
    compute_metrics = build_compute_metrics(args, tokenizer)

    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    eval_output_dir = Path(args.output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        **build_eval_args_kwargs(args, str(eval_output_dir / 'tmp_eval'))
    )
    trainer = build_trainer(args, training_args, model, tokenized_datasets, tokenizer, compute_metrics)

    prediction_output = trainer.predict(tokenized_datasets['test'], metric_key_prefix='test')
    metrics = {key: float(value) if isinstance(value, (np.floating, np.integer)) else value for key, value in prediction_output.metrics.items()}

    primary_predictions = select_primary_prediction(prediction_output.predictions)
    decoded_predictions = tokenizer.batch_decode(primary_predictions, skip_special_tokens=True)

    result_frame = pd.DataFrame(raw_datasets['test'])
    result_frame['prediction'] = decoded_predictions
    result_frame['correct'] = (result_frame['prediction'] == result_frame['label']).astype(int)

    aux_predictions = maybe_select_aux_prediction(prediction_output.predictions)
    if args.output_rationale and aux_predictions is not None:
        result_frame['predicted_rationale'] = tokenizer.batch_decode(aux_predictions, skip_special_tokens=True)

    stem = f'{Path(args.model_path).name}_test'
    predictions_path = eval_output_dir / f'{stem}_predictions.csv'
    metrics_path = eval_output_dir / f'{stem}_metrics.json'
    result_frame.to_csv(predictions_path, index=False)
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f'Test predictions saved to {predictions_path}')
    print(f'Test metrics saved to {metrics_path}')


if __name__ == '__main__':
    main()
