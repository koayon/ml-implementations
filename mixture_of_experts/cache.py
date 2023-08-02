from dataclasses import dataclass
from typing import Dict

import torch as t
from jaxtyping import Float, Int
from typeguard import typechecked


# Initialise cache for routing and use for MoE layers
@dataclass
class MoELayerCache:
    """G: softmaxed routing weights for the top k experts
    token_assignments: the top k expert ids
    routing_weights: raw outputs of the routing model (before softmax)
    """

    G: Float[t.Tensor, "k num_experts"]
    token_assignments: Int[t.Tensor, "k num_experts"]
    routing_weights: Float[t.Tensor, "batch*seq num_experts"]


# @typechecked
class MoEFullCache:
    def __init__(self, moe_cache_dict: Dict[str, MoELayerCache]):
        self._cache_dict = moe_cache_dict

    def __len__(self):
        return len(self._cache_dict)

    def __getitem__(self, idx):
        return self._cache_dict[idx]

    def __setitem__(self, idx, value):
        self._cache_dict[idx] = value

    def __iter__(self):
        return iter(self._cache_dict)

    @property
    def G(self) -> Dict[str, Float[t.Tensor, "k num_experts"]]:
        return {idx: cache.G for idx, cache in self._cache_dict.items()}

    @property
    def token_assignments(self) -> Dict[str, Int[t.Tensor, "k num_experts"]]:
        return {idx: cache.token_assignments for idx, cache in self._cache_dict.items()}

    @property
    def routing_weights_tensor(self) -> Float[t.Tensor, "layer batch*seq num_experts"]:
        return t.stack(
            [cache.routing_weights for idx, cache in self._cache_dict.items()], dim=0
        )
