from typing import Any, Optional, Tuple, Union

import torch as t
from torch import nn

from alibi.attention import AlibiUnidirectionalAttention
from general.norms import RMSNorm
from mixture_of_experts.cache import MoELayerCache
from moet_experiment.group_moe_layer import GroupExpertChoiceMoELayer
from moet_experiment.moet_config import MoETConfig

device = "cuda" if t.cuda.is_available() else "cpu"


class MoETBlock(nn.Module):
    norm: nn.Module

    def __init__(
        self,
        *,
        config: MoETConfig,
        num_experts: int,
        parallel_ffn: bool,
        group_size: int,
        layer_id: str,
        router_str: str,
        norm_str: str = "rms",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        if norm_str == "rms":
            self.norm1 = RMSNorm(shape_without_batch=(config.hidden_size,))
        else:
            self.norm1 = nn.LayerNorm(config.hidden_size)

        self.attention_layer = AlibiUnidirectionalAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attn_heads,
            dropout=config.attn_dropout,
        )

        if norm_str == "rms":
            self.norm2 = RMSNorm(shape_without_batch=(config.hidden_size,))
        else:
            self.norm2 = nn.LayerNorm(config.hidden_size)

        self.expert_layer = GroupExpertChoiceMoELayer(
            num_experts=num_experts,
            config=config,
            layer_id=layer_id,
            group_size=group_size,  # group_size of 1 means no sharing of upsample parameters
            router_str=router_str,
        )

        if parallel_ffn:
            self.parallel_ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 4),
                nn.SiLU(),
                nn.Linear(config.hidden_size * 4, config.hidden_size),
            )

    def forward(
        self, x: t.Tensor, input_tokens: t.Tensor
    ) -> Tuple[t.Tensor, MoELayerCache]:
        """
        x: batch seq hidden_size

        Return: shape (batch seq hidden_size)
        """
        # Using PreNorm from GPT-3 paper (Brown et al)

        y, _attn_cache = self.attention_layer(self.norm1(x))
        x = x + y

        x = self.norm2(x)

        y, moe_layer_cache = self.expert_layer(x=x, input=input_tokens)

        if hasattr(self, "parallel_ffn"):
            y = y + self.parallel_ffn(y)

        x = x + y

        return x, moe_layer_cache
