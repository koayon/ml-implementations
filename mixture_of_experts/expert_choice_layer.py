from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch as t
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from helpers import einsum
from mixture_of_experts.cache import MoELayerCache
from mixture_of_experts.config import MoEConfig

ROUTERS = ["linear", "hash", ...]

device = "cuda" if t.cuda.is_available() else "cpu"


class ExpertChoiceFFN(nn.Module):
    routing_model: nn.Module
    experts: nn.ModuleList
    expert_dropout: nn.Dropout

    def __init__(
        self,
        *,
        config: MoEConfig,
        expert: Optional[nn.Module] = None,
        k: int = 0,  # topk
        c: float = 1.0,  # capacity factor
        router: Optional[nn.Module] = None,
        layer_id: str,
    ) -> None:
        super().__init__()

        # Either choose k or set it from the capacity factor (c)
        assert (k > 0) or (c > 0)

        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.layer_id = layer_id

        self.router = (
            router
            if router is not None
            else nn.Linear(self.hidden_size, self.num_experts, device=device)
        )

        self.routing_dropout = nn.Dropout(config.routing_dropout)

        self.expert = (
            expert
            if expert is not None
            else nn.Linear(self.hidden_size, self.hidden_size, device=device)
        )

        self.experts = nn.ModuleList([self.expert for _ in range(self.num_experts)])
        self.expert_dropout = nn.Dropout(config.expert_dropout)
        self.k = k
        self.c = c

    def forward_individual_expert_choice(
        self,
        expert_num: int,
        chosen_token_index: t.Tensor,
        G: t.Tensor,
        x: t.Tensor,
    ) -> t.Tensor:
        """
        Set up to be parallelisable
        expert_num: The expert that we're using for forward
        chosen_token_index: k num_experts
        G: k num_experts
        x: bs hidden_size
        """

        batch_seq_size = x.shape[0]

        # Select top-k expert, with one-hot vector
        P: t.Tensor = nn.functional.one_hot(
            chosen_token_index, num_classes=batch_seq_size
        )  # k num_experts (one-hot)

        P = rearrange(P, "k num_experts bs -> bs k num_experts")  # bs k num_experts

        # Extract relevant sections of P, G
        P_expert = P[..., expert_num] * 1.0  # bs k
        G_expert = G[:, expert_num]  # k

        tokens_for_expert = einsum(
            "bs k, bs hidden_size -> k hidden_size", P_expert, x
        )  # k hidden_size

        # Forward pass through the expert network
        E = self.expert_dropout(
            self.experts[expert_num](tokens_for_expert)
        )  # k hidden_size

        x_out = einsum(
            "bs k, k, k hidden_size -> bs hidden_size", P_expert, G_expert, E
        )  # bs hidden_size

        return x_out

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, MoELayerCache]:
        """
        Args:
            x: batch seq hidden_size
            router: hidden_size num_experts

        Returns:
            x: batch, seq, hidden_size
            cache: MoELayerCache: G, token_assignments, routing_weights
        """

        batch_dim, seq_length, _hidden_size = x.shape

        # Use the capacity factor to set k
        if self.c > 0:
            # TODO: Data dependent workflow which breaks TensorBoard
            # self.k = int(batch_dim * seq_length * self.c // self.num_experts)
            pass

        # If there aren't enough tokens in the input to select top k, reduce k
        # self.k = min(int(self.k), (batch_dim * seq_length))
        self.k = 4

        x = rearrange(x, "b s h -> (b s) h")
        h = self.routing_dropout(self.router(x))  # bs num_experts

        # Calculate router score or Gate Value
        S = t.softmax(h, dim=-1)  # bs num_experts
        G, chosen_token_index = t.topk(S, k=self.k, dim=0)  # k num_experts each

        layer_cache = MoELayerCache(
            G=G,
            token_assignments=chosen_token_index,
            routing_weights=h,
        )

        # Collect expert results from parallelised expert forward
        expert_results = [
            self.forward_individual_expert_choice(
                expert_num=expert_num, G=G, x=x, chosen_token_index=chosen_token_index
            )
            for expert_num in range(self.num_experts)
        ]  # list[bs hidden_size]

        # Aggregate expert results together
        expert_results_stack = t.stack(
            expert_results, dim=0
        )  # num_experts bs hidden_size
        y = t.sum(expert_results_stack, dim=0)  # bs hidden_size

        y = rearrange(y, "(b s) h -> b s h", b=batch_dim)  # batch seq hidden_size

        return y, layer_cache


def main():
    expert_layer = ExpertChoiceFFN(
        config=MoEConfig(),
        expert=nn.Linear(16, 16),
        k=2,
        layer_id="expert-layer-1",
    )

    x = t.rand(size=(3, 4, 16))  # batch seq hidden_size

    y, cache = expert_layer(x, cache={})
    print(cache)


if __name__ == "__main__":
    main()
