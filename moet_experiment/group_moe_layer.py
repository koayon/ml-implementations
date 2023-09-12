from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Tuple, Union

import torch as t
from einops import rearrange, repeat
from jaxtyping import Float, Int
from numpy import cumsum
from torch import nn
from torch.nn import functional as F

from general.swiglu_ffn import SwiGLUFFN
from helpers import einsum
from mixture_of_experts.cache import (
    ExpertChoiceFullCache,
    ExpertChoiceLayerCache,
    MoELayerCache,
    TokenChoiceFullCache,
    TokenChoiceLayerCache,
)
from mixture_of_experts.routers import HashRouter
from moet_experiment.moet_config import MoETConfig


class RouterEnums(Enum):
    linear = auto()
    learned = auto()
    hash = auto()
    router_weights_passed_separately = auto()


device = "cuda" if t.cuda.is_available() else "cpu"

# config = MoETConfig()


class Router(nn.Module):
    def __init__(self, *, num_experts: int, router_str: str, config: MoETConfig):
        super().__init__()
        self.config = config

        self.router_enum = RouterEnums[router_str]
        self.router_temperature = config.router_temperature
        self.num_experts = num_experts
        self.hidden_size = config.hidden_size

        if self.router_enum in (RouterEnums.linear, RouterEnums.learned):
            # Define the linear router
            self.linear = nn.Linear(self.hidden_size, self.num_experts)
        elif self.router_enum == RouterEnums.hash:
            # Build the hash router
            assert config.num_experts_hash == self.num_experts
            self.hash_router = HashRouter(
                config=config, num_experts=config.num_experts_hash)
            # self.hash_router.build_random_hash()
        elif self.router_enum == RouterEnums.router_weights_passed_separately:
            # Router will be passed in separately
            pass
        else:
            raise ValueError(
                f"Unknown router {router_str}. Please choose from {RouterEnums._member_names_}")

        self.routing_dropout = nn.Dropout(config.routing_dropout)

    def forward(self, x: t.Tensor, input_tokens: Optional[t.IntTensor] = None) -> t.Tensor:
        """
        Parameters:
        x (t.Tensor): Hidden state input tensor. Shape (bs, hidden_size).
        input_tokens (Optional[t.IntTensor]): Original input tokens required if router_str is "hash". Shape (batch_size, hidden_size).

        Returns:
        t.Tensor: Mapping from input tokens to experts. Shape (batch_size, num_experts).

        Raises:
        AssertionError: If router_str is "hash" and input_tokens is None.
        """

        if self.router_enum == RouterEnums["hash"]:
            assert input_tokens is not None

            input_tokens = rearrange(input_tokens, "b s -> (b s)")
            clean_h = self.hash_router(input_tokens).float()  # bs num_experts
        else:
            clean_h = self.linear(x)  # bs num_experts
            clean_h = self.routing_dropout(clean_h)  # bs num_experts

        # Add gumbel noise to the routing logits to encourage exploration during training
        # self.training is inherited from nn.Module and is set by calling model.train() or model.eval()
        if self.training and (self.router_enum in (RouterEnums.linear, RouterEnums.learned)):
            gumbel_noise = -t.log(-t.log(t.rand_like(clean_h) + 1e-10) + 1e-10)
            h = (clean_h + gumbel_noise) / self.router_temperature
            return h  # bs num_experts
        else:
            return clean_h  # bs num_experts


class GroupMoELayer(nn.Module):
    experts: nn.ModuleList
    up_experts: list[nn.Module]
    down_experts: list[nn.Module]

    def __init__(
        self,
        *,
        num_experts: int,
        layer_id: str,
        router_str: str = "linear",
        router_weights_passed_separately: bool = False,
        router: Optional[Router] = None,
        config: MoETConfig = MoETConfig(),
        group_size: int = 1,
        k: int = 0,  # topk
        c: float = 1.0,  # capacity factor
        ffn_dim_multiplier: int = 4,
        ffn_ratio: float = 2 / 3,
        use_expert_choice: Optional[bool] = None,
    ) -> None:
        super().__init__()

        # Either choose k or set it from the capacity factor (c)
        assert (k > 0) or (c > 0)
        assert num_experts % group_size == 0
        assert router_str in RouterEnums._member_names_

        self.num_experts = num_experts
        self.num_expert_groups = num_experts // group_size
        self.use_expert_choice = use_expert_choice if use_expert_choice is not None else config.use_expert_choice

        self.router_weights_passed_separately = router_weights_passed_separately

        self.layer_id = layer_id

        self.hidden_size = config.hidden_size
        self.batch_size = config.batch_size
        self.seq_len = config.max_position_embeddings

        if router:
            # If we've passed in a router then use that
            self.router = router
        elif self.router_weights_passed_separately:
            # If we're passing in the router separately then we don't need to create one here
            self.router = None
        else:  # Otherwise create a new router
            self.router = Router(
                num_experts=num_experts,
                router_str=router_str,
                config=config,
            )

        if group_size > 1:

            # Grouped experts
            up_experts = [
                nn.Linear(
                    in_features=self.hidden_size,
                    out_features=int(self.hidden_size *
                                     ffn_dim_multiplier * ffn_ratio),
                )
                for _ in range(self.num_expert_groups)
            ]
            down_experts = [
                nn.Linear(
                    in_features=int(self.hidden_size *
                                    ffn_dim_multiplier * ffn_ratio),
                    out_features=self.hidden_size,
                )
                for _ in range(self.num_experts)
            ]
            silu = nn.SiLU()
            expert_dropout = nn.Dropout(config.expert_dropout)

            experts = []
            for expert_num in range(self.num_experts):
                # group_size experts share the same up layer. Each has a unique down layer.
                expert_group_num = expert_num // group_size
                experts.append(
                    nn.Sequential(
                        up_experts[expert_group_num],
                        silu,
                        down_experts[expert_num],
                        expert_dropout,
                    )
                )
            self.experts = nn.ModuleList(experts)
        else:

            # Regular FFN experts
            expert = SwiGLUFFN(
                in_features=self.hidden_size, dropout=config.expert_dropout
            )
            self.experts = nn.ModuleList(
                [expert for _ in range(self.num_experts)])

        # Use the capacity factor to set k
        if c > 0:
            self.k = self.batch_size * self.seq_len * c // self.num_experts
        else:
            self.k = k

        # print(layer_id, "- k: ", self.k)

    def forward(
        self, x: t.Tensor, input_tokens: Optional[t.Tensor] = None, router_weights: Optional[t.Tensor] = None
    ) -> Tuple[t.Tensor, MoELayerCache]:
        """
        For interpretability uses we want to hook and cache G (the top k softmaxed router weights - either over tokens or experts).
        We also want the chosen token/expert indices known as the assignments.

        It may also be interesting to look at S which is the same as G but the full matrix, not restricted to the top k.

        Args:
            x: batch seq hidden_size
            router: hidden_size num_experts
            input_tokens: batch seq, the original input tokens

        Returns:
            x: batch, seq, hidden_size
            MoELayerCache
            Either an ExpertChoiceLayerCache or a TokenChoiceLayerCache depending on the value of self.use_expert_choice
            Contains:
                G: (depends on self.use_expert_choice)
                assignments: (depends on self.use_expert_choice)

                routing_weights: (batch seq) num_experts
                    Also called h. These are the logits used in the loss function.

        """

        batch_size, seq_len, _hidden_size = x.shape

        x = rearrange(x, "b s h -> (b s) h")

        if router_weights is not None:
            # Use the precomputed router weights if provided
            assert router_weights.shape == (
                batch_size * seq_len, self.num_experts)

            h = router_weights
        else:
            # Get routing weights, h
            assert self.router
            h = self.router(x, input_tokens)  # (b s) num_experts

        S = t.softmax(h, dim=-1)  # bs num_experts

        if self.use_expert_choice:
            G, chosen_token_index, P = self._expert_choice_routing_matrices(
                S=S, batch_size=batch_size, seq_len=seq_len)

            layer_cache = ExpertChoiceLayerCache(
                G=G,
                P=P,
                token_assignments=chosen_token_index,
                routing_weights=h,
            )

        else:
            G, chosen_expert_index, P = self._token_choice_routing_matrices(
                S=S, batch_size=batch_size)

            layer_cache = TokenChoiceLayerCache(
                G=G,
                P=P,
                expert_assignments=chosen_expert_index,
                routing_weights=h,
            )

        tokens_for_expert = einsum(
            "bs k expert, bs hidden_size -> expert k hidden_size", P.float(), x
        )  # expert k hidden_size

        # USE EXPERTS
        # forward the relevant tokens through the relevant expert
        E_list = [self.experts[expert_num](tokens_for_expert[expert_num]) for expert_num in range(
            self.num_experts)]  # num_experts list[k hidden_size]

        E = t.stack(E_list, dim=0)  # num_experts k hidden_size

        # Put the results back in the right order with the permutation matrix P and weight them correctly with the routing weights G

        if self.use_expert_choice:
            # P [bs k num_experts]
            # G [k num_experts]
            # E [num_experts k hidden_size]
            y = einsum(
                "bs k expert, k expert, expert k hidden_size -> bs hidden_size", P.float(), G, E)
        else:
            # P [bs k num_experts]
            # G [bs k]
            # E [num_experts k hidden_size]
            y = einsum(
                "bs k expert, bs k, expert k hidden_size -> bs hidden_size", P.float(), G, E)

        y = rearrange(
            y, "(batch seq) hidden_size -> batch seq hidden_size", batch=batch_size)

        return y, layer_cache

    # TODO: Decouple MoE layer from Router

    def _expert_choice_routing_matrices(self, S: Float[t.Tensor, "bs num_experts"], batch_size: int, seq_len: int) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        """Expert Choice: Each expert picks the top-k tokens it wants to process. In the moment that we pick the topk across the sequence dimension, we share some information across the time/seq dimension which would be a problem for autoregressive models (it's allowing the model to cheat). This is best used for non-autoregressive models.

        Parameters
        ----------
        S : Float[t.Tensor, "bs num_experts"]
            _description_

        Returns
        -------
        Tuple[t.Tensor, t.Tensor, t.Tensor]
            G, chosen_token_index, P
        """

        # If there aren't enough tokens in the input to select top k, reduce k
        self.k = min(int(self.k), (batch_size * seq_len))

        G, chosen_token_index = t.topk(
            S, k=self.k, dim=0)  # k num_experts each

        # Select top-k expert, with one-hot vector. P is the permutation matrix
        P: t.Tensor = F.one_hot(
            chosen_token_index, num_classes=batch_size * seq_len
        )  # k num_experts bs (one-hot)

        # bs k num_experts
        P = rearrange(P, "k num_experts bs -> bs k num_experts").float()

        return G, chosen_token_index, P

    def _token_choice_routing_matrices(self, S: Float[t.Tensor, "bs num_experts"], batch_size: int) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        """We use left-to-right token dropping.

        Token-choice: Each token picks the top-k experts it wants to process it. This is best used for autoregressive models.
        If we balance the experts enough with our load-balancing and set a sufficiently high k, then there is very little information shared across the sequence dimension.
        The only information that can be shared is that a token is dropped.
        Having a high expert dropout rate also helps here (a token doesn't know if it was dropped because of later tokens also wanting that expert or because of the expert dropout rate).
        Another way to mitigate this is to fill up the experts left to right so any dropped tokens are dropped from the end of the sequence and the knowledge that a token was dropped is passed only backwards not forwards in time.

        TODO: Add random token-dropping and Batch Priority Routing (BPR) for inference time.


        Parameters
        ----------
        S : Float[t.Tensor, "bs num_experts"]
            _description_

        Returns
        -------
        Tuple[t.Tensor, t.Tensor, t.Tensor]
            G, chosen_expert_index, P
        """

        # If there aren't enough tokens in the input to select top k, reduce k
        self.k = min(int(self.k), (self.num_experts))

        G, chosen_expert_index = t.topk(S, k=self.k, dim=1)  # bs k each

        # We now need to decide which experts to drop. Eventually this has to be each expert with k tokens so it should be of G should be of shape (expert, k*bs)

        # Select top-k expert, with one-hot vector
        P = F.one_hot(
            chosen_expert_index, num_classes=self.num_experts
        )  # bs k num_experts (one-hot)

        # We want to rearrange P. Currently we have a long line of bs such that we see all of the first batch before we see any of the second batch. We would like to see the first elements from all batches then second elements etc.
        # This means there's less variance in performance depending on where you happen to be within a batch
        P = rearrange(P, "(b s) k num_experts -> (s b) k num_experts",
                      b=batch_size)  # sb k num_experts

        drop_points = self._get_first_drop_point(P=P, k=self.k)

        for expert_num in range(self.num_experts):
            # Set everything after the drop point to 0
            P[drop_points[expert_num]:, :, expert_num] = 0  # sb k num_experts

        # Now P defines a permutation matrix where each expert gets at most k tokens.

        # Rearrange P back to the usual shape
        P = rearrange(P, "(s b) k num_experts -> (b s) k num_experts",
                      b=batch_size)  # bs k num_experts

        return G, chosen_expert_index, P

    @staticmethod
    def _get_first_drop_point(P: t.Tensor, k: int) -> Int[t.Tensor, "num_experts"]:
        """_summary_

        Parameters
        ----------
        P : t.Tensor
            Permutation matrix [sb k num_experts]
        k : int
            Maximum number of experts per token
        batch_size: int
            Batch size
        seq_len: int
            Current sequence length

        Returns
        -------
        t.Tensor
        Drop points: The number of tokens after which each expert is full and drops any later tokens
        """

        tokens_per_expert = t.sum(P, dim=1)  # sb num_experts

        cumsum_tokens_per_expert = t.cumsum(
            tokens_per_expert, dim=0)  # sb num_experts
        cumsum_tokens_per_expert = rearrange(
            cumsum_tokens_per_expert, "sb num_experts -> num_experts sb")
        # All the indices where we need to drop
        indices = t.nonzero(cumsum_tokens_per_expert >= k)

        # Intialise drop_points as -1s for each expert (no dropping)
        drop_points = t.full((cumsum_tokens_per_expert.shape[0],), -1)

        for expert_num, token_num in indices:
            # If cycling through the points we're needing to drop we see an expert that doesn't have a drop point set yet then set it
            if drop_points[expert_num] == -1:
                drop_points[expert_num] = token_num

        return drop_points  # num_experts


def main():
    expert_layer = GroupMoELayer(
        k=2,
        layer_id="expert-layer-1",
        num_experts=4,
        router_str="linear",
    )

    x = t.rand(size=(3, 4, 16))  # batch seq hidden_size

    y, cache = expert_layer(x, cache={})
    print(cache)


if __name__ == "__main__":
    main()
