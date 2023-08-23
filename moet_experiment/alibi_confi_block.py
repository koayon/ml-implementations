from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import torch as t
import transformers
from einops import rearrange
from torch import nn
from transformers.activations import NewGELUActivation

from alibi.attention import AlibiUnidirectionalAttention
from general.confi_ffn import ConfiFFN
from general.swiglu_ffn import SwiGLUFFN
from helpers import einsum


class ALiBiConfiTBlock(nn.Module):
    """
    ALiBiTransformerBlock is a transformer block with a unidirectional attention layer.
    Based on OpenAI's GPT-2 implementation and the ALiBi paper (Train Short, Test Long)
    We're using SwiGLU for the MLP part.
    """

    attn: AlibiUnidirectionalAttention
    confi_mlp: nn.Module
    ln1: nn.LayerNorm
    ln2: nn.LayerNorm

    def __init__(
        self,
        *,
        layer_index: int,
        hidden_size: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
    ):
        super().__init__()

        self.layer_index = layer_index

        # Attention part

        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = AlibiUnidirectionalAttention(
            hidden_size=hidden_size, num_heads=num_heads, dropout=attn_dropout
        )

        # MLP part

        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.ConfiMLP = ConfiFFN(
            hidden_size=hidden_size,
            dropout=mlp_dropout,
            activation_function="silu",
        )

    def forward(self, x: t.Tensor, layer_cache=None) -> Tuple[t.Tensor, Any]:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        y = self.ln1(x)
        y, _layer_cache = self.attn(y, layer_cache=layer_cache)

        x = x + y

        x = x + self.ConfiMLP(x)

        return x, None


if __name__ == "__main__":
    # Test GPT2Block
    block = ALiBiConfiTBlock(layer_index=0, hidden_size=16, num_heads=4)
    x = t.rand(2, 10, 768)
    y = block(x)
    print(y.shape)
    print(y)
