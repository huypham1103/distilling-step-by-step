# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import re

from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader
from metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux
from selection_utils import resolve_rationale_type_name
from train_utils import train_and_evaluate


def find_rationale_indices(column_names):
    indices = []
    for column_name in column_names:
        match = re.fullmatch(r'rationale_(\d+)', column_name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def tokenize_targets(tokenizer, texts, max_length):
    if hasattr(tokenizer, 'as_target_tokenizer'):
        with tokenizer.as_target_tokenizer():
            return tokenizer(texts, max_length=max_length, truncation=True)
    return tokenizer(text_target=texts, max_length=max_length, truncation=True)


def load_selected_rationale_datasets(selected_rationale_path):
    import pandas as pd

    dataframe = pd.read_csv(selected_rationale_path)
    required_columns = {'input', 'label', 'split'}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f'Selected rationale dataset is missing required columns: {sorted(missing_columns)}')

    dataframe = dataframe.copy()
    dataframe['split'] = (
        dataframe['split']
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({'validation': 'valid', 'dev': 'valid'})
    )

    split_counts = dataframe['split'].value_counts().to_dict()
    if 'valid' not in split_counts or split_counts.get('valid', 0) == 0:
        train_frame = dataframe[dataframe['split'] == 'train'].copy()
        if train_frame.empty:
            raise ValueError('Selected rationale dataset does not contain a usable "train" split to derive validation data from.')
        valid_size = max(1, int(round(len(train_frame) * 0.1)))
        valid_size = min(valid_size, len(train_frame))
        valid_frame = train_frame.sample(n=valid_size, random_state=0)
        dataframe.loc[valid_frame.index, 'split'] = 'valid'

    datasets = {}
    for split_name in ['train', 'valid', 'test']:
        split_frame = dataframe[dataframe['split'] == split_name].copy()
        if split_frame.empty:
            raise ValueError(f'Selected rationale dataset does not contain split "{split_name}"')
        datasets[split_name] = Dataset.from_pandas(split_frame.reset_index(drop=True), preserve_index=False)
    return DatasetDict(datasets)


def build_task_prefix_tokenize_function(tokenizer, args, rationale_indices):
    def tokenize_function(examples):
        model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
        for index in rationale_indices:
            rationale_type_column = f'rationale_type_{index}'
            expl_model_inputs = tokenizer(
                [f'explain {rationale_type}: {text}' for text, rationale_type in zip(examples['input'], examples[rationale_type_column])],
                max_length=args.max_input_length,
                truncation=True
            )
            model_inputs[f'expl_input_ids_{index}'] = expl_model_inputs['input_ids']
            model_inputs[f'expl_attention_mask_{index}'] = expl_model_inputs['attention_mask']

        label_output_encodings = tokenize_targets(tokenizer, examples['label'], max_length=256)
        model_inputs['labels'] = label_output_encodings['input_ids']
        for index in rationale_indices:
            rationale_output_encodings = tokenize_targets(tokenizer, examples[f'rationale_{index}'], max_length=256)
            model_inputs[f'aux_labels_{index}'] = rationale_output_encodings['input_ids']
        return model_inputs

    return tokenize_function


def load_latest_rationales_dataframe(rationale_name):
    import pandas as pd

    resolved_name = resolve_rationale_type_name(rationale_name)
    return pd.read_csv(f'[API] ESNLI/{resolved_name} - full.csv')[['premise', 'hypothesis', 'rationale', 'LLM_answer']]


def map_datasetdict_per_split(datasets, tokenize_function, remove_columns_strategy):
    mapped_splits = {}
    for split_name, split_dataset in datasets.items():
        if remove_columns_strategy == 'all':
            remove_columns = split_dataset.column_names
        elif remove_columns_strategy == 'input_label':
            remove_columns = [column for column in ['input', 'label'] if column in split_dataset.column_names]
        else:
            raise ValueError(f'Unsupported remove_columns_strategy: {remove_columns_strategy}')
        mapped_splits[split_name] = split_dataset.map(
            tokenize_function,
            remove_columns=remove_columns,
            batched=True
        )
    return DatasetDict(mapped_splits)


def run(args):
    if args.selected_rationale_path is not None:
        datasets = load_selected_rationale_datasets(args.selected_rationale_path)
    else:
    #### Prepare datasets
        if args.dataset == 'cqa':
            dataset_loader = CQADatasetLoader()
        elif args.dataset == 'svamp':
            dataset_loader = SVAMPDatasetLoader()
        elif args.dataset == 'esnli':
            dataset_loader = ESNLIDatasetLoader()
        elif args.dataset == 'anli1':
            dataset_loader = ANLI1DatasetLoader()
        elif args.dataset == 'asdiv':  # NOTE: for augmenting SVAMP only
            dataset_loader = SVAMPDatasetLoader()
            dataset_loader_svamp = SVAMPDatasetLoader()
            dataset_loader_asdiv = ASDivDatasetLoader()
        else:
            raise ValueError

        if args.dataset == 'asdiv':
            datasets_svamp = dataset_loader_svamp.load_from_json()
            datasets_asdiv = dataset_loader_asdiv.load_from_json()
            datasets = DatasetDict({
                'train': concatenate_datasets([datasets_svamp['train'], datasets_asdiv['train']]),
                'test': datasets_svamp['test']
            })
        else:
            datasets = dataset_loader.load_from_json()

        if args.llm is None:
            pass
        elif args.llm == 'palm':
            if args.dataset == 'asdiv':
                train_llm_rationales_svamp, train_llm_labels_svamp = dataset_loader_svamp.load_llm_preds(split='train')
                train_llm_rationales_asdiv, train_llm_labels_asdiv = dataset_loader_asdiv.load_llm_preds(split='train')
                train_llm_rationales = train_llm_rationales_svamp + train_llm_rationales_asdiv
                train_llm_labels = train_llm_labels_svamp + train_llm_labels_asdiv
                test_llm_rationales, test_llm_labels = dataset_loader_svamp.load_llm_preds(split='test')
            else:
                train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train')
                test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(split='test')
        elif args.llm == 'gpt':
            train_llm_rationales, train_llm_labels = dataset_loader.load_gpt_preds(split='train')
            test_llm_rationales, test_llm_labels = dataset_loader.load_gpt_preds(split='test')
        else:
            raise ValueError

        if args.llm is not None:
            datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
            datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
            datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
            datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)

        if args.subsample < 1.0:
            datasets['train'] = datasets['train'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']

        if dataset_loader.has_valid:
            if args.llm is None:
                pass
            elif args.llm == 'palm':
                valid_llm_rationales, valid_llm_labels = dataset_loader.load_llm_preds(split='valid')
            elif args.llm == 'gpt':
                valid_llm_rationales, valid_llm_labels = dataset_loader.load_gpt_preds(split='valid')
            else:
                raise ValueError

            datasets['valid'] = datasets['valid'].add_column('llm_label', valid_llm_labels)
            datasets['valid'] = datasets['valid'].add_column('llm_rationale', valid_llm_rationales)
        else:
            train_valid_datasets = datasets['train'].train_test_split(test_size=0.1, seed=0)

            datasets = DatasetDict({
                'train': train_valid_datasets['train'],
                'valid': train_valid_datasets['test'],
                'test': datasets['test'],
            })

        if args.label_type == 'gt':
            pass
        elif args.label_type == 'llm' and args.llm is not None:
            if args.dataset not in ['svamp', 'asdiv']:
                train_label_acc = compute_text_acc(datasets['train']['llm_label'], datasets['train']['label'])
                test_label_acc = compute_text_acc(datasets['test']['llm_label'], datasets['test']['label'])
            else:
                train_label_acc = compute_equation_acc(datasets['train']['llm_label'], datasets['train']['label'])
                test_label_acc = compute_equation_acc(datasets['test']['llm_label'], datasets['test']['label'])

            print(f'LLM Train Acc: {train_label_acc:.4f}')
            print(f'LLM Test Acc: {test_label_acc:.4f}')

            # datasets['train'] = datasets['train'].remove_columns('label')
            # datasets['train'] = datasets['train'].add_column('label', datasets['train']['llm_label'])

        else:
            raise ValueError

        if args.llm is not None:
            if 'rationale' in datasets['train'].column_names:
                datasets = datasets.remove_columns('rationale')
            datasets = datasets.rename_column('llm_rationale', 'rationale')


    #### Prepare datasets Prepare data for training
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    if args.selected_rationale_path is None and 'nli' in args.dataset:
        datasets = datasets.map(
            lambda example: {'input': tokenizer.eos_token.join([example['premise'], example['hypothesis']])},
            # remove_columns=['premise', 'hypothesis'],
        )


    rationale_indices = find_rationale_indices(datasets['train'].column_names) if args.model_type == 'task_prefix' else []
    if args.model_type == 'task_prefix':
        tokenize_function = build_task_prefix_tokenize_function(tokenizer, args, rationale_indices)
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

    else:
        raise ValueError


    if args.selected_rationale_path is not None:
        tokenized_datasets = map_datasetdict_per_split(datasets, tokenize_function, remove_columns_strategy='all')
    elif args.llm is None:
        tokenized_datasets = map_datasetdict_per_split(datasets, tokenize_function, remove_columns_strategy='input_label')
    else:
        # load myself rationales

        import pandas as pd
        from datasets import Dataset 
        test = pd.DataFrame(datasets['test'])
        test = test.set_index('input')
        
        rationales_1 = load_latest_rationales_dataframe(args.extra_rationale_1)
        rationales_2 = load_latest_rationales_dataframe(args.extra_rationale_2)
        rationales_3 = load_latest_rationales_dataframe(args.extra_rationale_3)
        rationales_4 = load_latest_rationales_dataframe(args.extra_rationale_4)
        
        rationales_1['input'] = rationales_1['premise'] + '</s>' + rationales_1['hypothesis']
        rationales_2['input'] = rationales_2['premise'] + '</s>' + rationales_2['hypothesis']
        rationales_3['input'] = rationales_3['premise'] + '</s>' + rationales_3['hypothesis']
        rationales_4['input'] = rationales_4['premise'] + '</s>' + rationales_4['hypothesis']
        
        rationales_1.set_index('input', inplace=True, drop=True)
        rationales_2.set_index('input', inplace=True, drop=True)
        rationales_3.set_index('input', inplace=True, drop=True)
        rationales_4.set_index('input', inplace=True, drop=True)
        
        rationales_1.loc[rationales_2.index, 'rationale_2'] = rationales_2['rationale']
        rationales_1.loc[rationales_2.index, 'label_2'] = rationales_2['LLM_answer']
        rationales_1.loc[rationales_3.index, 'rationale_3'] = rationales_3['rationale']
        rationales_1.loc[rationales_3.index, 'label_3'] = rationales_3['LLM_answer']
        rationales_1.loc[rationales_4.index, 'rationale_4'] = rationales_4['rationale']
        rationales_1.loc[rationales_4.index, 'label_4'] = rationales_4['LLM_answer']
        rationales_1.rename(columns={'LLM_answer': 'label'}, inplace=True)
        # split train, valid
        train = rationales_1.sample(frac=0.8, random_state=0)
        val = rationales_1.drop(train.index)
       
        train.rename(columns={'rationale': 'rationale_1'}, inplace=True)
        val.rename(columns={'rationale': 'rationale_1'}, inplace=True)
        test.rename(columns={'rationale': 'rationale_1'}, inplace=True)
        train['rationale_type_1'] = resolve_rationale_type_name(args.extra_rationale_1)
        val['rationale_type_1'] = resolve_rationale_type_name(args.extra_rationale_1)
        test['rationale_type_1'] = resolve_rationale_type_name(args.extra_rationale_1)
        test['rationale_2'] = test['rationale_1']
        test['rationale_3'] = test['rationale_1']
        test['rationale_4'] = test['rationale_1']
        train['rationale_type_2'] = resolve_rationale_type_name(args.extra_rationale_2)
        val['rationale_type_2'] = resolve_rationale_type_name(args.extra_rationale_2)
        test['rationale_type_2'] = resolve_rationale_type_name(args.extra_rationale_2)
        train['rationale_type_3'] = resolve_rationale_type_name(args.extra_rationale_3)
        val['rationale_type_3'] = resolve_rationale_type_name(args.extra_rationale_3)
        test['rationale_type_3'] = resolve_rationale_type_name(args.extra_rationale_3)
        train['rationale_type_4'] = resolve_rationale_type_name(args.extra_rationale_4)
        val['rationale_type_4'] = resolve_rationale_type_name(args.extra_rationale_4)
        test['rationale_type_4'] = resolve_rationale_type_name(args.extra_rationale_4)

        # if label_2 is different from label, then use the rationale_1 as rationale_2
        train.loc[train['label'] != train['label_2'], 'rationale_2'] = train.loc[train['label'] != train['label_2'], 'rationale_1']
        val.loc[val['label'] != val['label_2'], 'rationale_2'] = val.loc[val['label'] != val['label_2'], 'rationale_1']
        train.drop(columns=['label_2'], inplace=True)
        val.drop(columns=['label_2'], inplace=True)
                
        # if label_3 is different from label, then use the rationale_1 as rationale_3
        train.loc[train['label'] != train['label_3'], 'rationale_3'] = train.loc[train['label'] != train['label_3'], 'rationale_1']
        val.loc[val['label'] != val['label_3'], 'rationale_3'] = val.loc[val['label'] != val['label_3'], 'rationale_1']
        train.drop(columns=['label_3'], inplace=True)
        val.drop(columns=['label_3'], inplace=True)
        
        # if label_4 is different from label, then use the rationale_1 as rationale_4
        train.loc[train['label'] != train['label_4'], 'rationale_4'] = train.loc[train['label'] != train['label_4'], 'rationale_1']
        val.loc[val['label'] != val['label_4'], 'rationale_4'] = val.loc[val['label'] != val['label_4'], 'rationale_1']
        train.drop(columns=['label_4'], inplace=True)
        val.drop(columns=['label_4'], inplace=True)
        
        datasets['train'] = Dataset.from_pandas(train.reset_index())
        datasets['valid'] = Dataset.from_pandas(val.reset_index())
        datasets['test'] = Dataset.from_pandas(test.reset_index())

        tokenized_datasets = map_datasetdict_per_split(datasets, tokenize_function, remove_columns_strategy='all')
    if args.model_type == 'standard':
        if args.dataset not in ['svamp', 'asdiv']:
            compute_metrics = compute_metrics_text_aux(tokenizer)
        else:
            compute_metrics = compute_metrics_equation_aux(tokenizer)

    else:
        if args.dataset not in ['svamp', 'asdiv']:
            compute_metrics = compute_metrics_text(tokenizer)
        else:
            compute_metrics = compute_metrics_equation(tokenizer)


    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--llm', type=str, default='palm')
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    parser.add_argument('--data_size', type=int, default=1)
    parser.add_argument('--selected_rationale_path', type=str, default=None)
    parser.add_argument('--selection_policy', type=str, default='heuristic')
    parser.add_argument('--num_selected_rationales', type=int, default=1)
    parser.add_argument('--extra_rationale_1', type=str, default='if_else')
    parser.add_argument('--extra_rationale_2', type=str, default='neutral')
    parser.add_argument('--extra_rationale_3', type=str, default='neutral')
    parser.add_argument('--extra_rationale_4', type=str, default='neutral')


    args = parser.parse_args()

    # dic = {
    #     'dataset': 'esnli',
    #     'subsample': 1.0,
    #     'alpha': 0.5,
    #     'max_steps': 10000,
    #     'eval_steps': 1,
    #     'batch_size': 2,
    #     'optimizer_name': 'AdamW',
    #     'lr': 5e-05,
    #     'run': 0,
    #     'from_pretrained': 'google/t5-v1_1-base',
    #     'label_type': 'gt',
    #     'llm': 'palm',
    #     'max_input_length': 1024,
    #     'grad_steps': 1,
    #     'local_rank': -1,
    #     'gen_max_len': 64,
    #     'parallelize': False,
    #     'model_type': 'task_prefix',
    #     'bf16': False,
    #     'no_log': False,
    #     'output_rationale': False,
    #     'data_size': 1,
    #     'extra_rationale_1': 'causal',
    #     'extra_rationale_2': 'condition',
    #     'extra_rationale_3': 'causal',
    #     'extra_rationale_4': 'causal'
    # }
    # from types import SimpleNamespace
    # args = SimpleNamespace(**dic)

    run(args)
