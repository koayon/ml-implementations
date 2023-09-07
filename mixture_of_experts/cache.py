import re
from dataclasses import dataclass
from typing import Dict

import torch as t
from einops import rearrange, repeat
from jaxtyping import Float, Int
from typeguard import typechecked

# Initialise cache for routing and use for MoE layers

@dataclass
class ExpertChoiceLayerCache():
    """G: softmaxed routing weights for the top k experts
        [k num_experts]
    token_assignments: the top k expert ids
        [k num_experts]

    P: one-hot vector of the expert assignments
        [bs k num_experts]
    routing_weights: raw outputs of the routing model (before softmax)
        [batch_seq num_experts]
    """

    G: Float[t.Tensor, "k num_experts"]
    token_assignments: Int[t.Tensor, "k num_experts"]

    P: Int[t.Tensor, "bs k num_experts"]
    routing_weights: Float[t.Tensor, "batch_seq num_experts"]

    def detach(self) -> None:
        self.G = self.G.detach()
        self.token_assignments = self.token_assignments.detach()

        self.P = self.P.detach()
        self.routing_weights = self.routing_weights.detach()

@dataclass
class TokenChoiceLayerCache():
    """G: softmaxed routing weights for the top k experts
        [batch_seq k]
    token_assignments: the top k expert ids
        [batch_seq k]

    P: one-hot vector of the expert assignments
        [bs k num_experts]
    routing_weights: raw outputs of the routing model (before softmax)
        [batch_seq num_experts]
    """

    G: Float[t.Tensor, "batch_seq k"]
    expert_assignments: Int[t.Tensor, "batch_seq k"]

    P: Int[t.Tensor, "bs k num_experts"]
    routing_weights: Float[t.Tensor, "batch_seq num_experts"]

    def detach(self) -> None:
        self.G = self.G.detach()
        self.token_assignments = self.expert_assignments.detach()

        self.P = self.P.detach()
        self.routing_weights = self.routing_weights.detach()



def pad_with_negs(tensor: t.Tensor) -> t.Tensor:
    # Get -1s to double the number of experts and pad the cache
    negs = -t.ones_like(tensor)
    return t.cat([tensor, negs], dim=1)

# @typechecked
class ExpertChoiceFullCache(Dict[str, ExpertChoiceLayerCache]):
    def __init__(self, moe_cache_dict: Dict[str, ExpertChoiceLayerCache]):
        """Cache containing the G, routing weights and assignments for each layer.

        G is the softmaxed routing weights for the top k experts

        token_assignments is the top k expert ids

        P is the one-hot vector of the assignments

        routing_weights is the raw outputs of the routing model (before softmax)

        As this is expert choice, we index dimensions by experts first (layer, expert)
        """
        super().__init__(moe_cache_dict)

    def __setitem__(self, idx: str, cache: ExpertChoiceLayerCache) -> None:
        assert isinstance(cache, ExpertChoiceLayerCache)
        super().__setitem__(idx, cache)

        # Make sure the cache has consistent shapes even when the number of experts per layer varies
        # self._pad_with_negative1s()
        self._pad_with_duplicates()
        self._pad_k_dim()

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

    @property
    def P(self) -> Int[t.Tensor, "layer num_experts bs k"]:
        out = t.stack([cache.P for idx, cache in self.items()], dim=0)
        out = rearrange(out, "layer bs k num_experts -> layer num_experts bs k")
        return out

    @property
    def routing_weights_tensor(self) -> Float[t.Tensor, "layer num_experts batch_seq"]:
        out = t.stack([cache.routing_weights for idx, cache in self.items()], dim=0)
        out = rearrange(out, "layer batch_seq num_experts -> layer num_experts batch_seq")
        return out

    @property
    def layer_indices(self) -> list[str]:
        return list(self.keys())

    @property
    def num_experts(self) -> int:
        return max([layer_cache.G.shape[1] for idx, layer_cache in self.items()])

    @property
    def k(self) -> int:
        return max([layer_cache.G.shape[0] for idx, layer_cache in self.items()])

    @property
    def num_tokens(self) -> int:
        return self.routing_weights_tensor.shape[-1] # batch_seq


    def _pad_with_negative1s(self) -> None:
        """Some layers of the cache might have half the number of experts. In this case we want to pad this tensor with -1s so that they can stack together nicely"""
        for _, cache in self.items():
            if cache.G.shape[1] < self.num_experts:
                cache.G = pad_with_negs(cache.G)
                cache.token_assignments = pad_with_negs(cache.token_assignments)

                cache.P = pad_with_negs(cache.P)
                cache.routing_weights = pad_with_negs(cache.routing_weights)

    def _pad_with_duplicates(self) -> None:
        """Some layers of the cache might have half the number of experts. In this case we want to pad this tensor by duplicating the first E/2 experts so that they can stack together nicely"""
        for _, cache in self.items():
            if cache.G.shape[1] < self.num_experts:
                cache.G = repeat(cache.G, "k num_experts -> k (2 num_experts)")
                cache.token_assignments = repeat(cache.token_assignments, "k num_experts -> k (2 num_experts)")

                cache.P = repeat(cache.P, "bs k num_experts -> bs k (2 num_experts)")
                cache.routing_weights = repeat(cache.routing_weights, "batch_seq num_experts -> batch_seq (2 num_experts)")

    def _pad_k_dim(self) -> None:
        for _, cache in self.items():
            if cache.G.shape[0] < self.k:
                cache.G = repeat(cache.G, "k num_experts -> (2 k) num_experts")
                cache.token_assignments = repeat(cache.token_assignments, "k num_experts -> (2 k) num_experts")

                cache.P = repeat(cache.P, "bs k num_experts -> bs (2 k) num_experts")


class TokenChoiceFullCache(Dict[str, TokenChoiceLayerCache]):
    def __init__(self, moe_cache_dict: Dict[str, TokenChoiceLayerCache]):
        """Cache containing the G, routing weights and assignments for each layer.

        G is the softmaxed routing weights for the top k experts

        token_assignments is the top k expert ids

        P is the one-hot vector of the assignments

        routing_weights is the raw outputs of the routing model (before softmax)

        As this is token choice, we index dimensions by tokens first (apart from in P and routing_weights which are consistent with Expert Choice in layer, expert first)
        """
        super().__init__(moe_cache_dict)



    def __setitem__(self, idx: str, cache: TokenChoiceLayerCache) -> None:
        assert isinstance(cache, TokenChoiceLayerCache)
        super().__setitem__(idx, cache)

         # Make sure the cache has consistent shapes even when the number of experts per layer varies
        # self._pad_with_negative1s()
        self._pad_with_duplicates()
        self._pad_k_dim()

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

    @property
    def P(self) -> Int[t.Tensor, "layer num_experts bs k"]:
        out = t.stack([cache.P for idx, cache in self.items()], dim=0)
        out = rearrange(out, "layer bs k num_experts -> layer num_experts bs k")
        return out

    @property
    def routing_weights_tensor(self) -> Float[t.Tensor, "layer num_experts batch_seq"]:
        out = t.stack([cache.routing_weights for idx, cache in self.items()], dim=0)
        out = rearrange(out, "layer batch_seq num_experts -> layer num_experts batch_seq")
        return out

    @property
    def layer_indices(self) -> list[str]:
        return list(self.keys())

    @property
    def k(self) -> int:
        return max([layer_cache.G.shape[1] for idx, layer_cache in self.items()])

    @property
    def num_experts(self) -> int:
        return max([layer_cache.routing_weights.shape[-1] for idx, layer_cache in self.items()])

    @property
    def num_tokens(self) -> int:
        return self.routing_weights_tensor.shape[-1] # batch_seq

    def _pad_with_negative1s(self) -> None:
        """Some layers of the cache might have half the number of experts. In this case we want to pad this tensor with 0s so that they can stack together nicely"""
        for _, cache in self.items():
            if cache.routing_weights.shape[1] < self.num_experts:
                # Get zeros to double the number of experts and pad the cache
                cache.P = pad_with_negs(cache.P)
                cache.routing_weights = pad_with_negs(cache.routing_weights)

    def _pad_with_duplicates(self) -> None:
        """Some layers of the cache might have half the number of experts. In this case we want to pad this tensor by duplicating the first E/2 experts so that they can stack together nicely"""
        for _, cache in self.items():
            if cache.routing_weights.shape[1] < self.num_experts:
                cache.P = repeat(cache.P, "bs k num_experts -> bs k (2 num_experts)")
                cache.routing_weights = repeat(cache.routing_weights, "batch_seq num_experts -> batch_seq (2 num_experts)")

    def _pad_k_dim(self) -> None:
        for _, cache in self.items():
            if cache.G.shape[1] < self.k:
                cache.G = repeat(cache.G, "bs k -> bs (2 k)")
                cache.expert_assignments = repeat(cache.expert_assignments, "bs k -> bs (2 k)")

                cache.P = repeat(cache.P, "bs k num_experts -> bs (2 k) num_experts")
