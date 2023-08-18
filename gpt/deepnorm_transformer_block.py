from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Union

import torch as t
import transformers
from einops import rearrange
from torch import nn
from transformers.activations import NewGELUActivation

from gpt.cached_attention import AttentionCache, UnidirectionalAttention
from gpt.group_query_attention import GroupedQueryAttention
from helpers import einsum

ACTIVATION_FUNCTIONS = dict(
    relu=nn.ReLU(),
    gelu=nn.GELU(),
    new_gelu=NewGELUActivation(),
)


class DeepNormBlock(nn.Module):
    """
    Decoder-only Transformer Block with DeepNorm.

    DeepNorm should have the performance of PostNorm (putting LayerNorm after the residual connection) and the training stability of PreNorm (putting LayerNorm before the residual connection).

    Reference: https://arxiv.org/pdf/2203.00555.pdf
    """

    attn: UnidirectionalAttention | GroupedQueryAttention
    linear1: nn.Linear
    linear2: nn.Linear
    ln1: nn.LayerNorm
    ln2: nn.LayerNorm

    def __init__(
        self,
        layer_index: int,
        hidden_size: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        activation_function: str = "gelu",
        group_size: int = 0,
        num_layers: int = 12,
    ):
        super().__init__()

        self.layer_index = layer_index

        # For decoder-only models alpha = (2*num_layers)^(1/4)
        self.alpha = (2 * num_layers) ** 0.25
        # We initialise the weights with xavier_normal gain = beta
        self.beta = (8 * num_layers) ** 0.25

        # Attention part

        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        if group_size > 0:
            assert num_heads % group_size == 0
            self.attn = GroupedQueryAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                num_groups=num_heads // group_size,
            )
        else:
            self.attn = UnidirectionalAttention(hidden_size, num_heads, dropout=dropout)

        # MLP part
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        up_linear = nn.Linear(hidden_size, hidden_size * 4)
        down_linear = nn.Linear(hidden_size * 4, hidden_size)

        t.nn.init.xavier_normal_(up_linear.weight, self.beta)
        t.nn.init.xavier_normal_(down_linear.weight, self.beta)

        self.MLP = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", up_linear),
                    ("activation_function", ACTIVATION_FUNCTIONS[activation_function]),
                    ("linear2", down_linear),
                    ("dropout", nn.Dropout(dropout)),
                ]
            )
        )

    def forward(
        self, x: t.Tensor, layer_cache: Optional[AttentionCache] = None
    ) -> Tuple[t.Tensor, Optional[AttentionCache]]:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        y, layer_cache = self.attn(x, layer_cache=layer_cache)
        # DeepNorm
        x = self.ln1(x * self.alpha + y)

        y = self.MLP(x)
        # DeepNorm
        x = self.ln2(x * self.alpha + y)

        return x, layer_cache


if __name__ == "__main__":
    # Test GPT2Block
    block = DeepNormBlock(layer_index=0)
    x = t.rand(2, 10, 768)
    y: t.Tensor = block(x)
    print(y.shape)
    print(y)
