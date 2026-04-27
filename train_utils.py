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

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed

from model_utils import TaskPrefixDataCollator, TaskPrefixTrainer


def get_config_dir(args):
    return f'{args.dataset}/{args.from_pretrained.split("/")[1]}/{args.model_type}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}'


def _build_training_args(args, output_dir, logging_dir, max_steps, lr):
    if args.no_log:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    return Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns=False,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='no',
        save_steps=args.eval_steps,
        logging_dir=logging_dir,
        logging_strategy=logging_strategy,
        logging_steps=args.eval_steps,
        max_steps=max_steps,
        learning_rate=lr,
        gradient_accumulation_steps=args.grad_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        seed=args.run,
        local_rank=args.local_rank,
        bf16=args.bf16,
        generation_max_length=args.gen_max_len,
        prediction_loss_only=False,
    )


def _build_trainer(args, model, tokenizer, tokenized_datasets, compute_metrics, training_args, label_only=False):
    if args.model_type == 'task_prefix' and not label_only:
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    elif args.model_type in ['task_prefix', 'standard']:
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        raise ValueError

    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': {'test': tokenized_datasets["valid"],},
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }

    if args.model_type == 'task_prefix' and not label_only:
        trainer_kwargs['alpha'] = args.alpha
        trainer_kwargs['output_rationale'] = args.output_rationale
        return TaskPrefixTrainer(**trainer_kwargs)

    return Seq2SeqTrainer(**trainer_kwargs)


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics, gold_tokenized_datasets=None, gold_compute_metrics=None):
    set_seed(run)

    model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained)

    if args.parallelize:
        model.parallelize()

    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/{run}'  # for model ckpts
    logging_dir = f'logs/{config_dir}/{run}'  # for training logs

    # clear output dir if already exists
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    training_args = _build_training_args(args, output_dir, logging_dir, args.max_steps, args.lr)
    trainer = _build_trainer(args, model, tokenizer, tokenized_datasets, compute_metrics, training_args)

    print(f'Stage 1: rationale training for {args.max_steps} steps')
    trainer.train()

    output_type = args.type_rationale
    final_trainer = trainer
    if gold_tokenized_datasets is not None:
        gold_max_steps = args.gold_max_steps
        gold_lr = args.gold_lr if args.gold_lr is not None else args.lr
        gold_output_dir = f'{output_dir}_gold'
        gold_logging_dir = None if logging_dir is None else f'{logging_dir}_gold'

        if os.path.exists(gold_output_dir):
            logging.info('Found existing gold fine-tune ckpt directory. Deleted the old directory for the latest run.')
            shutil.rmtree(gold_output_dir)

        gold_training_args = _build_training_args(args, gold_output_dir, gold_logging_dir, gold_max_steps, gold_lr)
        gold_trainer = _build_trainer(
            args,
            model,
            tokenizer,
            gold_tokenized_datasets,
            gold_compute_metrics if gold_compute_metrics is not None else compute_metrics,
            gold_training_args,
            label_only=True
        )

        print(f'Stage 2: gold fine-tuning for {gold_max_steps} steps with lr={gold_lr}')
        gold_trainer.train()
        output_type = f'{args.type_rationale}{args.gold_output_suffix}'
        final_trainer = gold_trainer

    output_path = f'../model_path/{output_type}'
    final_trainer.save_model(output_path)
