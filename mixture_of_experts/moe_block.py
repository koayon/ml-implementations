from typing import Any, Optional, Union

import torch as t
from config import MoEConfig
from einops import rearrange, repeat
from expert_choice_layer import ExpertChoiceFFN
from fancy_einsum import einsum
from torch import nn

device = "cuda" if t.cuda.is_available() else "cpu"


class MoEBlock(nn.Module):
    def __init__(
        self,
        *,
        config: MoEConfig,
        expert_layer: Optional[nn.Module] = None,
        layer_id: str,
    ):
        hidden_size = config.hidden_size
        num_attn_heads = config.num_attn_heads
        attn_dropout = config.attn_dropout

        super().__init__()
        self.expert_layer = (
            expert_layer
            if expert_layer
            else ExpertChoiceFFN(
                config=config,
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
