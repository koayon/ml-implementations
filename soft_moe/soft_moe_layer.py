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


class SoftMoERouter(Router):
    def __init__(self, *, num_experts: int, config: MoETConfig):
        super().__init__(num_experts=num_experts, router_str="linear", config=config)
        self.l2_norm_0 = L2LayerNorm(dim=0)
        self.l2_norm_1 = L2LayerNorm(dim=2)
        self.scale = nn.Parameter(t.ones(1))

        self.linear.weight.data = self.l2_norm_0(self.linear.weight)
        self.linear.weight.data *= self.scale
        # TODO: How often to do this scaling and norming?

    def forward(self, x: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        x = self.l2_norm_1(x)
        # Forward the router
        router_logits, uncorrupted_router_logits = super()(x)
        return router_logits, uncorrupted_router_logits


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
        self.router = SoftMoERouter(
            num_experts=num_experts * slots_per_expert,
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
