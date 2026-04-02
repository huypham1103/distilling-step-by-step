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


import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer


"""T5 Multi-Task by Task Prefix
"""
class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    @staticmethod
    def _get_aux_indices(columns: List[str]) -> List[int]:
        indices = []
        for column in columns:
            if column.startswith('aux_labels_'):
                indices.append(int(column.rsplit('_', 1)[1]))
        return sorted(indices)

    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        aux_indices = self._get_aux_indices(features_df.columns.tolist())
        aux_columns = []
        for index in aux_indices:
            aux_columns.extend([
                f'aux_labels_{index}',
                f'expl_input_ids_{index}',
                f'expl_attention_mask_{index}',
            ])

        batch = {
            'pred': super().__call__(
                features_df.loc[:, ~features_df.columns.isin(aux_columns)].to_dict('records'),
                return_tensors
            ),
        }

        for index in aux_indices:
            excluded_columns = {'labels', 'input_ids', 'attention_mask'}
            for other_index in aux_indices:
                if other_index == index:
                    continue
                excluded_columns.update({
                    f'aux_labels_{other_index}',
                    f'expl_input_ids_{other_index}',
                    f'expl_attention_mask_{other_index}',
                })
            expl_features = features_df.loc[:, ~features_df.columns.isin(excluded_columns)].rename(
                columns={
                    f'aux_labels_{index}': 'labels',
                    f'expl_input_ids_{index}': 'input_ids',
                    f'expl_attention_mask_{index}': 'attention_mask',
                }
            ).to_dict('records')
            batch[f'expl_{index}'] = super().__call__(expl_features, return_tensors)

        return batch


class TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale


    def compute_loss(self, model, inputs, return_outputs=False):
        pred_outputs = model(**inputs['pred'])
        expl_keys = sorted(key for key in inputs if key.startswith('expl_'))
        expl_outputs = {key: model(**inputs[key]) for key in expl_keys}

        if expl_outputs:
            expl_loss = sum(output.loss for output in expl_outputs.values()) / len(expl_outputs)
        else:
            expl_loss = pred_outputs.loss

        loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_loss
        outputs = {'pred': pred_outputs}
        outputs.update(expl_outputs)
        return (loss, outputs) if return_outputs else loss


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        expl_keys = sorted(key for key in inputs if key.startswith('expl_'))
        expl_outputs = []
        if self.output_rationale:
            for key in expl_keys:
                expl_outputs.append(super().prediction_step(model, inputs[key], prediction_loss_only=False, ignore_keys=ignore_keys))

        if expl_outputs:
            expl_loss = sum(output[0] for output in expl_outputs) / len(expl_outputs)
            expl_prediction = expl_outputs[0][1]
            expl_label = expl_outputs[0][2]
        else:
            expl_loss = pred_outputs[0]
            expl_prediction = pred_outputs[1]
            expl_label = pred_outputs[2]

        loss = self.alpha * pred_outputs[0] + (1 - self.alpha) * expl_loss

        return (
            loss,
            [pred_outputs[1], expl_prediction],
            [pred_outputs[2], expl_label],
        )
