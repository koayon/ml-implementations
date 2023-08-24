from dataclasses import dataclass
from turtle import down
from typing import Any, Optional, Tuple, Union

import torch as t
from click import group
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from general.swiglu_ffn import SwiGLUFFN
from helpers import einsum
from mixture_of_experts.cache import MoELayerCache
from mixture_of_experts.routers import HashRouter
from moet_experiment.moet_config import MoETConfig

ROUTERS = ["linear", "learned", "hash"]
MULT = 4
RATIO = 2 / 3

device = "cuda" if t.cuda.is_available() else "cpu"


class GroupExpertChoiceMoELayer(nn.Module):
    experts: nn.ModuleList
    up_experts: list[nn.Module]
    down_experts: list[nn.Module]
    router: nn.Module

    def __init__(
        self,
        *,
        config: MoETConfig,
        num_experts: int,
        group_size: int = 1,
        k: int = 0,  # topk
        c: float = 1.0,  # capacity factor
        router_str: str,
        layer_id: str,
    ) -> None:
        super().__init__()

        # Either choose k or set it from the capacity factor (c)
        assert (k > 0) or (c > 0)
        assert num_experts % group_size == 0
        assert router_str in ROUTERS

        self.num_experts = num_experts

        self.hidden_size = config.hidden_size
        self.num_expert_groups = num_experts // group_size

        self.context_size = config.max_position_embeddings
        self.layer_id = layer_id
        self.batch_size = config.batch_size
        self.seq_len = config.max_position_embeddings
        self.router_str = router_str
        self.router_temperature = config.router_temperature

        if router_str in ("linear", "learned"):
            self.router = nn.Linear(self.hidden_size, self.num_experts, device=device)
        elif router_str == "hash":
            self.router = HashRouter(config=config, num_experts=config.num_experts_hash)
            self.router.build_random_hash()

        self.routing_dropout = nn.Dropout(config.routing_dropout)

        if group_size > 1:
            up_experts = [
                nn.Linear(
                    in_features=self.hidden_size,
                    out_features=int(self.hidden_size * MULT * RATIO),
                    device=device,
                )
                for _ in range(self.num_expert_groups)
            ]
            down_experts = [
                nn.Linear(
                    in_features=int(self.hidden_size * MULT * RATIO * self.num_experts),
                    out_features=self.hidden_size,
                    device=device,
                )
                for _ in range(self.num_experts)
            ]
            silu = nn.SiLU()
            expert_dropout = nn.Dropout(config.expert_dropout)

            experts = []
            for expert_num in range(self.num_experts):
                expert_group_num = expert_num // group_size
                experts[expert_num] = nn.Sequential(
                    up_experts[expert_group_num],
                    silu,
                    down_experts[expert_num],
                    expert_dropout,
                )
            self.experts = nn.ModuleList(experts)
        else:
            expert = SwiGLUFFN(
                in_features=self.hidden_size, dropout=config.expert_dropout
            )
            self.experts = nn.ModuleList([expert for _ in range(self.num_experts)])

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

        # Select top-k expert, with one-hot vector. P is the permutation matrix
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
        E = self.experts[expert_num](tokens_for_expert)
        # k hidden_size

        x_out = einsum(
            "bs k, k, k hidden_size -> bs hidden_size", P_expert, G_expert, E
        )  # bs hidden_size

        return x_out

    def forward(
        self, x: t.Tensor, input_tokens: Optional[t.Tensor] = None
    ) -> Tuple[t.Tensor, MoELayerCache]:
        """
        Args:
            x: batch seq hidden_size
            router: hidden_size num_experts
            input_tokens: batch seq, the original input tokens

        Returns:
            x: batch, seq, hidden_size
            cache: MoELayerCache: G, token_assignments, routing_weights
        """

        batch_dim, seq_length, _hidden_size = x.shape

        # Use the capacity factor to set k
        if self.c > 0:
            self.k = int(self.batch_size * self.seq_len * self.c // self.num_experts)
            pass

        # If there aren't enough tokens in the input to select top k, reduce k
        self.k = min(int(self.k), (batch_dim * seq_length))
        # self.k = 4

        x = rearrange(x, "b s h -> (b s) h")
        if self.router_str == "hash":
            assert input_tokens is not None

            input_tokens = rearrange(input_tokens, "b s -> (b s)")
            clean_h = self.router(input_tokens).float()  # bs num_experts
            clean_h = self.routing_dropout(clean_h)  # bs num_experts
        else:
            clean_h = self.router(x).float()  # bs num_experts
            clean_h = self.routing_dropout(clean_h)  # bs num_experts

        # Add gumbel noise to the routing logits to encourage exploration
        gumbel_noise = -t.log(-t.log(t.rand_like(clean_h) + 1e-10) + 1e-10)
        h = (clean_h + gumbel_noise) / self.router_temperature

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
        ]  # expert list[bs hidden_size]

        # Aggregate expert results together
        expert_results_stack = t.stack(
            expert_results, dim=0
        )  # num_experts bs hidden_size
        y = t.sum(expert_results_stack, dim=0)  # bs hidden_size

        y = rearrange(y, "(b s) h -> b s h", b=batch_dim)  # batch seq hidden_size

        return y, layer_cache


def main():
    expert_layer = GroupExpertChoiceMoELayer(
        config=MoETConfig(),
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
