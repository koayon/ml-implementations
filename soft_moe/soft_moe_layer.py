from typing import Optional, Tuple, Union

import torch as t
from einops import einsum, rearrange, repeat
from torch import nn

from general import device
from general.norms import L2LayerNorm
from mixture_of_experts.cache import MoELayerCache, SoftTokenMergeLayerCache
from mixture_of_experts.routers import Router
from moet_experiment.group_moe_layer import get_experts
from moet_experiment.moet_config import MoETConfig
from soft_moe.soft_expert_layer import SoftExpertLayer


class SoftMoELayer(nn.Module):
    def __init__(
        self,
        *,
        num_experts,
        layer_id,
        config: MoETConfig = MoETConfig(),
        group_size=1,
        slots_per_expert=1,
        # ffn_dim_multiplier=4,
        # ffn_ratio=2 / 3
    ):
        self.router = Router(
            num_experts=num_experts * slots_per_expert,
            router_str="linear",
            config=config,
        )
        self.soft_expert_layer = SoftExpertLayer(
            num_experts=num_experts,
            layer_id=layer_id,
            group_size=group_size,
            slots_per_expert=slots_per_expert,
        )

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, SoftTokenMergeLayerCache]:
        router_logits = self.router(x)

        y, _cache = self.soft_expert_layer(x, router_logits)

        return y, _cache
