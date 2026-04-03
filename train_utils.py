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


import os
import shutil
import logging
import inspect

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed

from model_utils import TaskPrefixDataCollator, TaskPrefixTrainer


def get_config_dir(args):
    model_name = args.from_pretrained.split("/")[-1]
    selection_tag = args.selection_policy if getattr(args, 'selected_rationale_path', None) else args.llm
    return f'{args.dataset}/{model_name}/{args.model_type}/{selection_tag}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}'


def build_training_args_kwargs(args, output_dir, logging_dir, logging_strategy, run):
    eval_batch_size = getattr(args, 'eval_batch_size', None) or args.batch_size
    kwargs = {
        'output_dir': output_dir,
        'remove_unused_columns': False,
        'eval_steps': args.eval_steps,
        'save_steps': args.eval_steps,
        'logging_dir': logging_dir,
        'logging_steps': args.eval_steps,
        'max_steps': args.max_steps,
        'learning_rate': args.lr,
        'gradient_accumulation_steps': args.grad_steps,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': eval_batch_size,
        'predict_with_generate': True,
        'seed': run,
        'local_rank': args.local_rank,
        'bf16': args.bf16,
        'fp16': getattr(args, 'fp16', False),
        'gradient_checkpointing': getattr(args, 'gradient_checkpointing', False),
        'generation_max_length': args.gen_max_len,
        'prediction_loss_only': False,
        'dataloader_num_workers': getattr(args, 'dataloader_num_workers', 0),
        'tf32': getattr(args, 'tf32', False),
        'eval_accumulation_steps': getattr(args, 'eval_accumulation_steps', None),
        'save_only_model': getattr(args, 'save_only_model', False),
        'ddp_find_unused_parameters': getattr(args, 'ddp_find_unused_parameters', None),
        'torch_compile': getattr(args, 'torch_compile', False),
    }

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    supported_params = set(signature.parameters)
    if 'evaluation_strategy' in supported_params:
        kwargs['evaluation_strategy'] = 'steps'
    elif 'eval_strategy' in supported_params:
        kwargs['eval_strategy'] = 'steps'
    if 'save_strategy' in supported_params:
        kwargs['save_strategy'] = 'no'
    if 'logging_strategy' in supported_params:
        kwargs['logging_strategy'] = logging_strategy
    return {key: value for key, value in kwargs.items() if key in supported_params}


def build_trainer_kwargs(trainer_class, args, training_args, model, tokenized_datasets, data_collator, tokenizer, compute_metrics):
    kwargs = {
        'alpha': args.alpha,
        'output_rationale': args.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': {'test': tokenized_datasets["valid"]},
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'processing_class': tokenizer,
        'compute_metrics': compute_metrics,
    }

    trainer_signature = inspect.signature(trainer_class.__init__)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in trainer_signature.parameters.values()):
        base_signature = inspect.signature(Seq2SeqTrainer.__init__)
        supported_params = set(base_signature.parameters).union({'alpha', 'output_rationale'})
    else:
        supported_params = set(trainer_signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported_params}


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics):
    set_seed(run)

    model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained)
    if getattr(args, 'gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.parallelize:
        model.parallelize()
    
    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/{run}'  # for model ckpts
    logging_dir = f'logs/{config_dir}/{run}'  # for training logs

    if args.no_log:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    # clear output dir if already exists
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    training_args = Seq2SeqTrainingArguments(
        **build_training_args_kwargs(args, output_dir, logging_dir, logging_strategy, run)
    )

    if args.model_type == 'task_prefix':
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    elif args.model_type == 'standard':
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        raise ValueError

    if args.model_type == 'task_prefix':
        trainer_kwargs = build_trainer_kwargs(
            TaskPrefixTrainer,
            args,
            training_args,
            model,
            tokenized_datasets,
            data_collator,
            tokenizer,
            compute_metrics,
        )
        trainer = TaskPrefixTrainer(**trainer_kwargs)
    elif args.model_type == 'standard':
        trainer_kwargs = build_trainer_kwargs(
            Seq2SeqTrainer,
            args,
            training_args,
            model,
            tokenized_datasets,
            data_collator,
            tokenizer,
            compute_metrics,
        )
        trainer = Seq2SeqTrainer(**trainer_kwargs)
    else:
        raise ValueError
    

    trainer.train()
    if getattr(args, 'selected_rationale_path', None):
        output_path = f'../model_path/selected_{args.selection_policy}_{args.num_selected_rationales}'
    else:
        output_path = f'../model_path/{args.extra_rationale_1}_{args.extra_rationale_2}_{args.extra_rationale_3}_{args.extra_rationale_4}'
    trainer.save_model(output_path)
