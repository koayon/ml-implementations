from typing import Any, Optional, Union

import torch as t
from einops import rearrange, repeat
from fancy_einsum import einsum
from torch import nn

device = "cuda" if t.cuda.is_available() else "cpu"


class MoEBlock:
    def __init__(
        self,
        *,
        hidden_size: int,
        expert: nn.Module,
        num_attn_heads: int,
        attn_dropout: float,
    ):
        super().__init__()
        self.expert = expert
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

    def forward(self, x: t.Tensor):
        """
        x: batch seq hidden_size

        Return: shape (batch seq hidden_size)
        """
        x = x + self.ln1(self.attention_layer(x))
        x = x + self.ln2(self.expert(x))
        return x
