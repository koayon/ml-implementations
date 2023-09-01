from dataclasses import dataclass
from typing import Dict

import torch as t
from jaxtyping import Float, Int
from typeguard import typechecked

# Initialise cache for routing and use for MoE layers

@dataclass
class ExpertChoiceLayerCache():
    """G: softmaxed routing weights for the top k experts
    token_assignments: the top k expert ids
    routing_weights: raw outputs of the routing model (before softmax)
    """

    G: Float[t.Tensor, "k num_experts"]
    token_assignments: Int[t.Tensor, "k num_experts"] # token assignments here
    routing_weights: Float[t.Tensor, "batch*seq num_experts"]

@dataclass
class TokenChoiceLayerCache():
    """G: softmaxed routing weights for the top k experts
    token_assignments: the top k expert ids
    routing_weights: raw outputs of the routing model (before softmax)
    """

    G: Float[t.Tensor, "batch*seq k"]
    expert_assignments: Int[t.Tensor, "batch*seq k"] # expert assignments here
    routing_weights: Float[t.Tensor, "batch*seq num_experts"]



# @typechecked
class ExpertChoiceFullCache(Dict[str, ExpertChoiceLayerCache]):
    """Cache containing the G, routing weights and assignments for each layer.

    G is the softmaxed routing weights for the top k experts

    token_assignments is the top k expert ids

    routing_weights is the raw outputs of the routing model (before softmax)
    """

    def __init__(self, moe_cache_dict: Dict[str, ExpertChoiceLayerCache]):
        super().__init__(moe_cache_dict)

    def __setitem__(self, idx: str, cache: ExpertChoiceLayerCache) -> None:
        assert isinstance(cache, ExpertChoiceLayerCache)
        super().__setitem__(idx, cache)

        # Make sure the cache has consistent shapes even when the number of experts per layer varies
        self._pad_with_0s()

    def __getitem__(self, __key: str) -> ExpertChoiceLayerCache:
        return super().__getitem__(__key)

    @property
    def G(self) -> Float[t.Tensor, "layer k num_experts"]:
        return t.stack([cache.G for idx, cache in self.items()], dim=0)

    @property
    def token_assignments(self) -> Int[t.Tensor, "layer k num_experts"]:
        return t.stack([cache.token_assignments for idx, cache in self.items()], dim=0)

    @property
    def routing_weights_tensor(self) -> Float[t.Tensor, "layer batch*seq num_experts"]:
        return t.stack([cache.routing_weights for idx, cache in self.items()], dim=0)

    @property
    def layer_indices(self) -> list[str]:
        return list(self.keys())

    @property
    def num_experts(self) -> int:
        return max([layer_cache.G.shape[1] for idx, layer_cache in self.items()])

    def _pad_with_0s(self) -> None:
        """Some layers of the cache might have half the number of experts. In this case we want to pad this tensor with 0s so that they can stack together nicely"""
        for _, cache in self.items():
            if cache.G.shape[1] < self.num_experts:
                # Get zeros to double the number of experts and pad the cache
                zeros = t.zeros_like(cache.G)
                cache.G = t.cat([cache.G, zeros], dim=1)
                cache.token_assignments = t.cat([cache.token_assignments, zeros], dim=1)

                routing_weight_zeros = t.zeros_like(cache.routing_weights)
                cache.routing_weights = t.cat(
                    [cache.routing_weights, routing_weight_zeros],
                    dim=1,
                )

class TokenChoiceFullCache(Dict[str, ExpertChoiceLayerCache]):
    """Will be similar to the above. Decoupling the two for later processing."""
    raise NotImplementedError
