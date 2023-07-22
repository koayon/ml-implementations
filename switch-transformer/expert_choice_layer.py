from typing import Any, Optional, Union

import torch as t
from einops import rearrange, repeat
from fancy_einsum import einsum
from torch import nn

ROUTERS = ["linear", "hash", ...]

device = "cuda" if t.cuda.is_available() else "cpu"


class ExpertChoiceFFN(nn.Module):
    routing_model: nn.Module
    experts: nn.ModuleList
    expert_dropout: nn.Dropout

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        dropout: float,
        expert: nn.Module,
        topk: int = 0,
        c: float = 0,
        router: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        # Either choose k or set it from the capacity factor (c)
        assert (topk > 0) or (c > 0)

        self.hidden_size = hidden_size
        self.num_experts = num_experts

        self.router = (
            router
            if router is not None
            else nn.Linear(hidden_size, num_experts, device=device)
        )
        self.experts = nn.ModuleList([expert for _ in range(num_experts)])
        self.expert_dropout = nn.Dropout(dropout)
        self.topk = topk
        self.c = c

    def forward_individual_expert_choice(
        self,
        expert_num: int,
        chosen_token_index: t.Tensor,
        G: t.Tensor,
        x: t.Tensor,
    ):
        """
        Set up to be parallelisable
        expert_num: The expert that we're using for forward
        chosen_token_index: k num_experts
        G: num_experts k
        x: bs hidden_size
        """

        batch_seq_size = x.shape[0]

        # Select top-k expert, with one-hot vector
        P = nn.functional.one_hot(
            chosen_token_index, num_classes=batch_seq_size
        )  # k num_experts (one-hot)
        print(f"{P.shape=}")

        P = rearrange(P, "k num_experts bs -> bs k num_experts")  # bs k num_experts
        print(f"{P.shape=}")

        # Extract relevant sections of P, G
        P_expert = P[..., expert_num]  # bs k
        G_expert = G[expert_num, :]  # k

        tokens_for_expert = einsum(
            "bs k, bs hidden_size -> k hidden_size", P_expert, x
        )  # k hidden_size

        # Forward pass through the expert network
        E = self.experts[expert_num](tokens_for_expert)  # k hidden_size

        x_out = t.einsum(
            "bs k, k, k hidden_size -> bs hidden_size", P_expert, G_expert, E
        )  # bs hidden_size

        return x_out

    def forward(self, x: t.Tensor):
        """
        x: batch seq hidden_size
        router: hidden_size num_experts

        Return: shape (batch, seq, hidden_size)
        """
        batch_dim, seq_length, _hidden_size = x.shape

        if self.c > 0:
            self.k = batch_dim * seq_length * self.c / self.num_experts

        x = rearrange(x, "b s h -> (b s) h")
        h = self.router(x)  # bs num_experts

        print(h.shape)

        # Calculate router score or Gate Value
        S = t.softmax(h, dim=-1)  # bs num_experts
        G, chosen_token_index = t.topk(S, k=2, dim=0)  # k num_experts each

        print(f"{G.shape=}")
        print(f"{chosen_token_index.shape=}")

        print(chosen_token_index)

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

        return y


def main():
    expert_layer = ExpertChoiceFFN(
        hidden_size=16, num_experts=4, dropout=0.1, expert=nn.Linear(16, 16), topk=2
    )

    x = t.rand(size=(3, 4, 16))  # batch seq hidden_size

    print(f"{expert_layer(x)}")


if __name__ == "__main__":
    main()
