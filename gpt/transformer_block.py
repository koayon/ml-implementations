from typing import Any, Optional

import torch as t
from einops import rearrange
from fancy_einsum import einsum
from torch import nn

from gpt.attention import UnidirectionalAttention

ACTIVATION_FUNCTIONS = dict(relu=nn.ReLU(), gelu=nn.GELU())


class GPT2Block(nn.Module):
    """
    GPT2Block is a transformer block with a unidirectional attention layer.
    Based on OpenAI's GPT-2 implementation.
    """

    attn: UnidirectionalAttention
    linear1: nn.Linear
    linear2: nn.Linear
    ln1: nn.LayerNorm
    ln2: nn.LayerNorm

    def __init__(
        self,
        hidden_size: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        activation_function: str = "gelu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = UnidirectionalAttention(hidden_size, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.nonlinearity = ACTIVATION_FUNCTIONS[activation_function]
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: t.Tensor, cache: Optional[Any] = None) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        x = x + self.attn(self.ln1(x), cache=cache)
        x = x + self.dropout(self.linear2(self.nonlinearity(self.linear1(self.ln2(x)))))
        return x
