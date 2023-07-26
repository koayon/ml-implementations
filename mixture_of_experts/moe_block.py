from typing import Any, Optional, Union

import torch as t
from torch import nn

from gpt.attention import UnidirectionalAttention
from mixture_of_experts.config import MoEConfig
from mixture_of_experts.expert_choice_layer import ExpertChoiceFFN

device = "cuda" if t.cuda.is_available() else "cpu"


class MoEBlock(nn.Module):
    def __init__(
        self,
        *,
        config: MoEConfig,
        expert_layer: Optional[nn.Module] = None,
        layer_id: str,
    ):
        super().__init__()

        self.expert_layer = (
            expert_layer
            if expert_layer
            else ExpertChoiceFFN(
                config=config,
                layer_id=layer_id,
            )
        )

        self.hidden_size = config.hidden_size
        self.attention_layer = UnidirectionalAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attn_heads,
            dropout=config.attn_dropout,
        )
        self.ln1 = nn.LayerNorm(normalized_shape=(config.hidden_size), device=device)
        self.ln2 = nn.LayerNorm(normalized_shape=(config.hidden_size), device=device)

    def forward(self, x: t.Tensor, cache=None):
        """
        x: batch seq hidden_size

        Return: shape (batch seq hidden_size)
        """
        x = x + self.ln1(self.attention_layer(x))

        y, cache = self.expert_layer(x, cache)

        x = x + self.ln2(y)

        return x, cache
