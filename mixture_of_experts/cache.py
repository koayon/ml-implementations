import re
from dataclasses import dataclass
from typing import Dict

import torch as t
from einops import rearrange
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
    P: Int[t.Tensor, "bs k num_experts"] # one-hot vector of the expert assignments
    token_assignments: Int[t.Tensor, "k num_experts"] # token assignments here
    routing_weights: Float[t.Tensor, "batch*seq num_experts"]

@dataclass
class TokenChoiceLayerCache():
    """G: softmaxed routing weights for the top k experts
    token_assignments: the top k expert ids
    routing_weights: raw outputs of the routing model (before softmax)
    """

    G: Float[t.Tensor, "batch*seq k"]
    P: Int[t.Tensor, "bs k num_experts"] # one-hot vector of the expert assignments
    expert_assignments: Int[t.Tensor, "batch*seq k"] # expert assignments here
    routing_weights: Float[t.Tensor, "batch*seq num_experts"]



# @typechecked
class ExpertChoiceFullCache(Dict[str, ExpertChoiceLayerCache]):
    def __init__(self, moe_cache_dict: Dict[str, ExpertChoiceLayerCache]):
        """Cache containing the G, routing weights and assignments for each layer.

        G is the softmaxed routing weights for the top k experts

        token_assignments is the top k expert ids

        routing_weights is the raw outputs of the routing model (before softmax)

        As this is expert choice, we index dimensions by experts first (layer, expert)
        """
        super().__init__(moe_cache_dict)

    def __setitem__(self, idx: str, cache: ExpertChoiceLayerCache) -> None:
        assert isinstance(cache, ExpertChoiceLayerCache)
        super().__setitem__(idx, cache)

        # Make sure the cache has consistent shapes even when the number of experts per layer varies
        self._pad_with_negative1s()

    def __getitem__(self, __key: str) -> ExpertChoiceLayerCache:
        return super().__getitem__(__key)

    @property
    def G(self) -> Float[t.Tensor, "layer num_experts k"]:
        """G is the softmaxed routing weights for the top k experts

        Returns
        -------
        t.Tensor [layer num_experts k]
        """
        out = t.stack([cache.G for idx, cache in self.items()], dim=0)
        out = rearrange(out, "layer k num_experts -> layer num_experts k")
        return out

    @property
    def token_assignments(self) -> Int[t.Tensor, "layer num_experts k"]:
        out = t.stack([cache.token_assignments for idx, cache in self.items()], dim=0)
        out = rearrange(out, "layer k num_experts -> layer num_experts k")
        return out

    def P(self) -> Int[t.Tensor, "layer num_experts bs k"]:
        out = t.stack([cache.P for idx, cache in self.items()], dim=0)
        out = rearrange(out, "layer bs k num_experts -> layer num_experts bs k")
        return out

    @property
    def routing_weights_tensor(self) -> Float[t.Tensor, "layer num_experts batch*seq"]:
        out = t.stack([cache.routing_weights for idx, cache in self.items()], dim=0)
        out = rearrange(out, "layer batch*seq num_experts -> layer num_experts batch*seq")
        return out

    @property
    def layer_indices(self) -> list[str]:
        return list(self.keys())

    @property
    def num_experts(self) -> int:
        return max([layer_cache.G.shape[1] for idx, layer_cache in self.items()])

    @property
    def num_tokens(self) -> int:
        return self.routing_weights_tensor.shape[-1] # batch*seq


    def _pad_with_negative1s(self) -> None:
        """Some layers of the cache might have half the number of experts. In this case we want to pad this tensor with 0s so that they can stack together nicely"""
        for _, cache in self.items():
            if cache.G.shape[1] < self.num_experts:
                # Get zeros to double the number of experts and pad the cache
                negs = -t.ones_like(cache.G)
                cache.G = t.cat([cache.G, negs], dim=1)
                cache.token_assignments = t.cat([cache.token_assignments, negs], dim=1)

                routing_weight_negs = -t.ones_like(cache.routing_weights)
                cache.routing_weights = t.cat(
                    [cache.routing_weights, routing_weight_negs],
                    dim=1,
                )

class TokenChoiceFullCache(Dict[str, TokenChoiceLayerCache]):
    def __init__(self, moe_cache_dict: Dict[str, TokenChoiceLayerCache]):
        """Cache containing the G, routing weights and assignments for each layer.

        G is the softmaxed routing weights for the top k experts

        token_assignments is the top k expert ids

        routing_weights is the raw outputs of the routing model (before softmax)

        As this is token choice, we index dimensions by tokens first (apart from in P and routing_weights which are consistent with Expert Choice in layer, expert first)
        """
        super().__init__(moe_cache_dict)


    def __setitem__(self, idx: str, cache: TokenChoiceLayerCache) -> None:
        assert isinstance(cache, TokenChoiceLayerCache)
        super().__setitem__(idx, cache)

    def __getitem__(self, __key: str) -> TokenChoiceLayerCache:
        return super().__getitem__(__key)

    @property
    def G(self) -> Float[t.Tensor, "batch_seq layer k"]:
        out = t.stack([cache.G for idx, cache in self.items()], dim=0)
        out = rearrange(out, "layer batch_seq k -> batch_seq layer k")
        return out

    @property
    def expert_assignments(self) -> Int[t.Tensor, "batch_seq layer k"]:
        out = t.stack([cache.expert_assignments for idx, cache in self.items()], dim=0)
        out = rearrange(out, "layer batch_seq k -> batch_seq layer k")
        return out

    def P(self) -> Int[t.Tensor, "layer num_experts bs k"]:
        out = t.stack([cache.P for idx, cache in self.items()], dim=0)
        out = rearrange(out, "layer bs k num_experts -> layer num_experts bs k")
        return out

    @property
    def routing_weights_tensor(self) -> Float[t.Tensor, "layer num_experts batch*seq"]:
        out = t.stack([cache.routing_weights for idx, cache in self.items()], dim=0)
        out = rearrange(out, "layer batch*seq num_experts -> layer num_experts batch*seq")
        return out

    @property
    def layer_indices(self) -> list[str]:
        return list(self.keys())

    @property
    def num_experts(self) -> int:
        return max([layer_cache.routing_weights.shape[-1] for idx, layer_cache in self.items()])

    @property
    def num_tokens(self) -> int:
        return self.routing_weights_tensor.shape[-1] # batch*seq

    def _pad_with_negative1s(self) -> None:
        """Some layers of the cache might have half the number of experts. In this case we want to pad this tensor with 0s so that they can stack together nicely"""
        for _, cache in self.items():
            if cache.routing_weights.shape[1] < self.num_experts:
                # Get zeros to double the number of experts and pad the cache
                routing_weight_negs = -t.ones_like(cache.routing_weights)
                cache.routing_weights = t.cat(
                    [cache.routing_weights, routing_weight_negs],
                    dim=1,
                )
