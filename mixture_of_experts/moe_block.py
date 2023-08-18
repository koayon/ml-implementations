from typing import Any, Optional, Tuple, Union

import torch as t
from torch import nn

from gpt.cached_attention import UnidirectionalAttention
from mixture_of_experts.cache import MoELayerCache
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
        self.hidden_size = config.hidden_size

        self.ln1 = nn.LayerNorm(normalized_shape=(config.hidden_size), device=device)

        self.attention_layer = UnidirectionalAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attn_heads,
            dropout=config.attn_dropout,
        )

        self.ln2 = nn.LayerNorm(normalized_shape=(config.hidden_size), device=device)

        self.expert_layer = (
            expert_layer
            if expert_layer
            else ExpertChoiceFFN(
                config=config,
                layer_id=layer_id,
            )
        )

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, MoELayerCache]:
        """
        x: batch seq hidden_size

        Return: shape (batch seq hidden_size)
        """
        # Using PreNorm from GPT-3 paper (Brown et al)

        y, _attn_cache = self.attention_layer(self.ln1(x))
        x = x + y

        x = self.ln2(x)

        y, moe_layer_cache = self.expert_layer(x)

        x = x + y

        return x, moe_layer_cache
