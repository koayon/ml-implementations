from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Union

import torch as t
import transformers
from einops import rearrange
from torch import nn
from transformers.activations import NewGELUActivation

from gpt.cached_attention import AttentionCache, UnidirectionalAttention
from gpt.cross_attention import CrossAttentionLayer
from gpt.group_query_attention import GroupedQueryAttention
from helpers import einsum

ACTIVATION_FUNCTIONS = dict(
    relu=nn.ReLU(),
    gelu=nn.GELU(),
    new_gelu=NewGELUActivation(),
)


class EncDecTransformerBlock(nn.Module):
    """
    Transformer block on the decoder side of an encoder-decoder transformer.
    Contains cross-attention, self-attention, and MLP layers.
    """

    cross_attn: CrossAttentionLayer
    self_attn: UnidirectionalAttention
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
        activation_function: str = "new_gelu",
    ):
        super().__init__()

        self.layer_index = layer_index

        # Attention part

        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.self_attn = UnidirectionalAttention(
            hidden_size, num_heads, dropout=dropout, autoregressive=True
        )

        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.cross_attn = CrossAttentionLayer(
            encoder_hidden_size=hidden_size,
            decoder_hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            head_size=hidden_size // num_heads,
        )

        # MLP part

        self.MLP = nn.Sequential(
            OrderedDict(
                [
                    ("ln3", nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)),
                    ("linear1", nn.Linear(hidden_size, hidden_size * 4)),
                    ("activation_function", ACTIVATION_FUNCTIONS[activation_function]),
                    ("linear2", nn.Linear(hidden_size * 4, hidden_size)),
                    ("dropout", nn.Dropout(dropout)),
                ]
            )
        )

    def forward(
        self,
        x: t.Tensor,
        encoder_outputs: t.Tensor,
        layer_cache: Optional[AttentionCache] = None,
    ) -> Tuple[t.Tensor, Optional[AttentionCache]]:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        y = self.ln1(x)
        y, layer_cache = self.self_attn(y, layer_cache=layer_cache)

        x = x + y

        y = self.ln2(x)
        y, layer_cache = self.cross_attn(
            y, encoder_outputs=encoder_outputs, layer_cache=layer_cache
        )

        x = x + self.MLP(x)

        return x, layer_cache


if __name__ == "__main__":
    # Test GPT2Block
    block = EncDecTransformerBlock(layer_index=0)
    x = t.rand(2, 10, 768)
    y: t.Tensor = block(x)
    print(y.shape)
    print(y)
