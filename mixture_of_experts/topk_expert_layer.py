from typing import Any, Optional, Union

import torch as t
from einops import rearrange, repeat
from torch import nn

from helpers import einsum
from mixture_of_experts.config import MoEConfig

config = MoEConfig()

ROUTERS = ["linear", "hash", ...]

device = "cuda" if t.cuda.is_available() else "cpu"


class ExpertFFN(nn.Module):
    routing_model: nn.Module
    experts: nn.ModuleList
    expert_dropout: nn.Dropout

    def __init__(
        self,
        *,
        config: MoEConfig,
        expert: nn.Module,
        topk: int,
        router: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts

        self.router = (
            router
            if router is not None
            else nn.Linear(self.hidden_size, self.num_experts, device=device)
        )
        self.experts = nn.ModuleList([expert for _ in range(self.num_experts)])
        self.expert_dropout = nn.Dropout(config.expert_dropout)
        self.topk = topk

    def forward_individual_expert(
        self, expert_num: int, chosen_expert_index: t.Tensor, G: t.Tensor, x: t.Tensor
    ):
        """
        Set up to be parallelisable
        expert_num: The expert that we're using for forward
        chosen_expert_index: bs k
        G: bs k
        x: bs hidden_size
        """

        # Select top-k expert, with one-hot vector
        P = nn.functional.one_hot(
            chosen_expert_index,
        )  # bs k num_experts (one-hot)

        P_expert = P[..., expert_num]  # bs k

        tokens_for_expert = einsum("bs k, bs hidden_size -> k hidden_size", P_expert, x)

        # Forward the relevant tokens through the expert
        E = self.experts[expert_num](tokens_for_expert)  # k hidden_size

        x_out = einsum("bs k, k hidden_size -> bs hidden_size", G, E)  # bs hidden_size

        return x_out

    def forward(self, x: t.Tensor):
        """
        x: batch seq hidden_size
        router: hidden_size num_experts

        Return: shape (batch, seq, hidden_size)
        """
        batch_dim = x.shape[0]

        x = rearrange(x, "b s h -> (b s) h")
        h = self.router(x)  # bs num_experts

        # Calculate router score or Gate Value
        S = t.softmax(h, dim=-1)  # bs num_experts
        G, chosen_expert_index = t.topk(S, k=2, dim=-1)  # bs k each

        # Collect expert results from parallelised expert forward
        expert_results = [
            self.forward_individual_expert(
                expert_num=expert_num, G=G, x=x, chosen_expert_index=chosen_expert_index
            )
            for expert_num in range(self.num_experts)
        ]  # list[bs hidden_size]

        # Aggregate expert results together
        expert_results_stack = t.stack(
            expert_results, dim=0
        )  # num_experts bs hidden_size
        y = t.sum(expert_results_stack, dim=0)  # bs hidden_size

        y = rearrange(y, "(b s) h -> b s h", b=batch_dim)  # batch seq hidden_size

        return y


def main():
    expert_layer = ExpertFFN(config=config, expert=nn.Linear(16, 16), topk=2)

    x = t.rand(size=(3, 4, 16))

    print("x: ", x)
    print("----------------")
    print(f"{expert_layer(x)}")


if __name__ == "__main__":
    main()
