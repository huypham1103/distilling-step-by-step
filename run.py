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

from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader
from metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux
from train_utils import train_and_evaluate


def run(args):
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
            # training set = SVAMP training + ASDiv training
            train_llm_rationales_svamp, train_llm_labels_svamp = dataset_loader_svamp.load_llm_preds(split='train')
            train_llm_rationales_asdiv, train_llm_labels_asdiv = dataset_loader_asdiv.load_llm_preds(split='train')
            train_llm_rationales = train_llm_rationales_svamp + train_llm_rationales_asdiv
            train_llm_labels = train_llm_labels_svamp + train_llm_labels_asdiv
            # test set = SVAMP test
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

    if 'nli' in args.dataset:
        datasets = datasets.map(
            lambda example: {'input': tokenizer.eos_token.join([example['premise'], example['hypothesis']])},
            # remove_columns=['premise', 'hypothesis'],
        )


    if args.model_type == 'task_prefix' and args.llm is not None:
        def tokenize_function(examples):
            model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
                rationale_output_encodings_1 = tokenizer(examples['rationale_1'], max_length=256, truncation=True)
                rationale_output_encodings_2 = tokenizer(examples['rationale_2'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels_1'] = rationale_output_encodings_1['input_ids']
            model_inputs['aux_labels_2'] = rationale_output_encodings_2['input_ids']

            return model_inputs

    elif args.model_type == 'standard':
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True
            )

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']

            return model_inputs

    else:
        raise ValueError


    if args.llm is None:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'label'],
            batched=True
        )
    else:
        # load myself rationales
        import pandas as pd
        from datasets import Dataset 
        train = pd.DataFrame(datasets['train'])
        train['question'] = train['input'].apply(lambda x: x.split('\n')[0])
        train = train.set_index('question')
        val = pd.DataFrame(datasets['valid'])
        val['question'] = val['input'].apply(lambda x: x.split('\n')[0])
        val = val.set_index('question')
        test = pd.DataFrame(datasets['test'])
        test['question'] = test['input'].apply(lambda x: x.split('\n')[0])
        test = test.set_index('question')

        if args.data_size:
            train = train.sample(frac=args.data_size/100, random_state=0)
            val = val.sample(frac=args.data_size/100, random_state=0)
            val = pd.concat([val, train[5000:]])
            train = train[:5000]
            # test = test.sample(frac=args.data_size/100, random_state=0)

        rationales_1 = pd.read_csv(f'[API] dataset/after_{args.extra_rationale_1} - full.csv').set_index('question')
        rationales_2 = pd.read_csv(f'[API] dataset/after_{args.extra_rationale_2} - full.csv').set_index('question')

        # modify the encode char
        train['rationale'] = rationales_1.loc[train.index][f'rationales'].values
        val['rationale'] = rationales_1.loc[val.index][f'rationales'].values
        
        train['rationale_2'] = rationales_2.loc[train.index][f'rationales'].values
        val['rationale_2'] = rationales_2.loc[val.index][f'rationales'].values

        train.rename(columns={'rationale': 'rationale_1'}, inplace=True)
        val.rename(columns={'rationale': 'rationale_1'}, inplace=True) 
        test.rename(columns={'rationale': 'rationale_1'}, inplace=True)
        test['rationale_2'] = test['rationale_1']

        train['label'] = rationales_1.loc[train.index]['LLM_answer'].values
        val['label'] = rationales_1.loc[val.index]['LLM_answer'].values
        # test['label'] = rationales.loc[test.index]['LLM_answer'].values
        
        datasets['train'] = Dataset.from_pandas(train.reset_index().drop(columns=['question']))
        datasets['valid'] = Dataset.from_pandas(val.reset_index().drop(columns=['question']))
        datasets['test'] = Dataset.from_pandas(test.reset_index().drop(columns=['question']))

        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'rationale_1', 'rationale_2', 'label', 'llm_label', 'premise', 'hypothesis'],
            batched=True
        )
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
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    # parser.add_argument('--type_rationale', type=str, default='if_else')
    parser.add_argument('--data_size', type=int, default=1)
    parser.add_argument('--extra_rationale_1', type=str, default='if_else')
    parser.add_argument('--extra_rationale_2', type=str, default='neutral')


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
    #     'bf16': True,
    #     'no_log': True,
    #     'output_rationale': True,
    #     # 'type_rationale': 'after_consensus_wucs_score',
    #     'data_size': 1,
    #     'extra_rationale_1': 'if_else',
    #     'extra_rationale_2': 'neutral'
    # }
    # from types import SimpleNamespace
    # args = SimpleNamespace(**dic)

    run(args)