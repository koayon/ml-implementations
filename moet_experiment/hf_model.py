from typing import Optional

import torch as t
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel

from moet_experiment.model import MoET


class MoETHFConfig(PretrainedConfig):

    def __init__(
        self,
        block_type="MoE",
        layers: int = 8,
        **kwargs,
    ):

        self.block_type = block_type
        self.layers = layers
        super().__init__(**kwargs)

class MoET_hf(PreTrainedModel):
    def __init__(self, hf_config: MoETHFConfig):
        super().__init__(hf_config)
        self.hf_config = hf_config

        self.model = MoET()

    def forward(self, input_ids: t.Tensor, attention_mask: t.Tensor, return_loss: bool = True, **kwargs):
        logits, _moe_cache = self.model(input_ids, attention_mask)
        if return_loss:
            labels = input_ids[:, 1:]
            pred_logits = logits[:, :-1, :]

            flattened_logits = rearrange(pred_logits, "b s v -> (b s) v")
            flattened_labels = rearrange(labels, "b s -> (b s)")

            loss = F.cross_entropy(flattened_logits, flattened_labels)

            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
