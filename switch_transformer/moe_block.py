from typing import Any, Optional, Union

import torch as t
from einops import rearrange, repeat
from expert_choice_layer import ExpertChoiceFFN
from fancy_einsum import einsum
from torch import nn

device = "cuda" if t.cuda.is_available() else "cpu"


class MoEBlock:
    def __init__(
        self,
        *,
        hidden_size: int,
        expert_layer: Optional[nn.Module] = None,
        num_attn_heads: int,
        attn_dropout: float,
        expert_dropout: float,
        num_experts: int,
        layer_id: str,
    ):
        super().__init__()
        self.expert_layer = (
            expert_layer
            if expert_layer
            else ExpertChoiceFFN(
                hidden_size=hidden_size,
                num_experts=num_experts,
                dropout=expert_dropout,
                layer_id=layer_id,
            )
        )
        self.hidden_size = hidden_size
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attn_heads,
            dropout=attn_dropout,
            batch_first=True,
            device=device,
        )
        self.ln1 = nn.LayerNorm(normalized_shape=(hidden_size), device=device)
        self.ln2 = nn.LayerNorm(normalized_shape=(hidden_size), device=device)

    def forward(self, x: t.Tensor, cache=None):
        """
        x: batch seq hidden_size

        Return: shape (batch seq hidden_size)
        """
        x = x + self.ln1(self.attention_layer(x))
        y, cache = self.expert_layer(x, cache)
        x = x + self.ln2(y)
        return x, cache
