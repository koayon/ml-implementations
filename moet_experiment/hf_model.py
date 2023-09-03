from json import load
from typing import Optional

import torch as t
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel

from general import device
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

        self.lb_coef = self.model.config.lb_coef
        self.z_coef = self.model.config.z_coef

    def forward(self, input_ids: t.Tensor, attention_mask: t.Tensor, return_loss: bool = True, **kwargs):
        """Forward function for hf wrapped model.

        Parameters
        ----------
        input_ids : Int[t.Tensor, "batch_size, seq_len"]
            Input tokens
        attention_mask : t.Tensor
            Attention mask
        return_loss : bool, optional
            Whether to return the model's loss in the output, by default True

        Returns
        -------
        dict
            Output dict
        """
        # Forward pass
        logits, _moe_cache = self.model(input_ids, attention_mask)

        if return_loss:
            labels = input_ids[:, 1:]
            pred_logits = logits[:, :-1, :]

            flattened_logits = rearrange(pred_logits, "b s v -> (b s) v")
            flattened_labels = rearrange(labels, "b s -> (b s)")

            cross_entropy_loss = F.cross_entropy(flattened_logits, flattened_labels)

            load_balancing_aux_loss = 0

            router_z_loss = 0

            loss = cross_entropy_loss + self.lb_coef * load_balancing_aux_loss + self.z_coef * router_z_loss

            return {"loss": loss, "cross_entropy_loss": cross_entropy_loss,
                    "load_balancing_aux_loss": load_balancing_aux_loss,
                    "router_z_loss": router_z_loss,
                    "logits": logits}
        else:
            return {"logits": logits}
