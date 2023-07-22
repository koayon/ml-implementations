import collections
from typing import Any, Optional, OrderedDict, Union

import torch as t
from einops import rearrange, repeat
from fancy_einsum import einsum
from torch import nn


class SparseMoETransformer:
    def __init__(
        self,
        *,
        num_layers: int,
        hidden_size: int,
        moe_block: nn.Module,
        transformer_block: nn.Module,
        attn_dropout: float,
        expert_dropout: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.expert = moe_block
        self.attn_dropout = attn_dropout
        self.expert_dropout = expert_dropout
        self.final_norm = nn.LayerNorm([hidden_size])

        layers: OrderedDict[str, nn.Module] = collections.OrderedDict()
        for i in range(num_layers):
            if i % 2 == 0:
                layers[f"moe_block{i}"] = moe_block
            else:
                layers[f"transformer_block{i}"] = transformer_block

        self.layers = nn.Sequential(layers)

    def forward(self, x: t.Tensor):
        """
        x: batch seq hidden_size (has already been tokenised)
        """
        x = self.layers(x)
        z = self.final_norm(x)

        return z


# TODO: Think about dropped tokens and the expert capacity
