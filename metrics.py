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


import numpy as np


def _extract_primary_token_batch(values):
    current = values

    while True:
        if isinstance(current, (list, tuple)):
            if not current:
                return np.empty((0, 0), dtype=np.int64)
            current = current[0]
            continue

        array = np.asarray(current)

        if array.dtype == object:
            if array.size == 0:
                return np.empty((0, 0), dtype=np.int64)
            current = array.flat[0]
            continue

        if array.ndim >= 3:
            current = array[0]
            continue

        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(1, -1)

        return array.astype(np.int64, copy=False)


def _sanitize_token_batch(token_batch, tokenizer):
    array = np.asarray(token_batch)
    if array.size == 0:
        return array.astype(np.int64, copy=False)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_token_id = max(len(tokenizer) - 1, pad_token_id)

    if np.issubdtype(array.dtype, np.floating):
        array = np.where(np.isfinite(array), array, pad_token_id)
        array = np.rint(array)

    array = array.astype(np.int64, copy=False)
    array = np.where(array < 0, pad_token_id, array)
    array = np.where(array > max_token_id, pad_token_id, array)
    return array


def compute_text_acc(preds, labels):
    return np.mean(np.array(preds) == np.array(labels))


def compute_equation_acc(preds, labels):
    preds = [eval_equation(pred) for pred in preds]
    labels = [eval_equation(label) for label in labels]

    return np.mean(np.array(preds) == np.array(labels))


def eval_equation(equation):
    try:
        answer = eval(equation)
    except:
        answer = np.nan

    return answer


def compute_metrics_text(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        prediction_ids = _sanitize_token_batch(_extract_primary_token_batch(predictions), tokenizer)
        decoded_preds = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)

        label_ids = _sanitize_token_batch(_extract_primary_token_batch(labels), tokenizer)
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics


def compute_metrics_text_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics



def compute_metrics_equation(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        prediction_ids = _sanitize_token_batch(_extract_primary_token_batch(predictions), tokenizer)
        decoded_preds = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)

        label_ids = _sanitize_token_batch(_extract_primary_token_batch(labels), tokenizer)
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        preds = list()
        for pred in decoded_preds:    
            preds.append(eval_equation(pred))

        labels = list()
        for label in decoded_labels:    
            labels.append(eval_equation(label))

        acc = np.mean(np.array(preds) == np.array(labels))

        return {'accuracy': acc}
    
    return compute_metrics


def compute_metrics_equation_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = list()
        for pred in decoded_preds:    
            preds.append(eval_equation(pred))

        labels = list()
        for label in decoded_labels:    
            labels.append(eval_equation(label))

        acc = np.mean(np.array(preds) == np.array(labels))

        return {'accuracy': acc}
    
    return compute_metrics
