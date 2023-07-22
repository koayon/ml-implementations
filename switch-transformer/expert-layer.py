from typing import Any, Optional

import torch as t
from einops import rearrange
from fancy_einsum import einsum
from torch import nn


class SimpleExpertFFN(nn.Module):
    router: nn.Linear
    experts: nn.ModuleList
    expert_dropout: nn.Dropout

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        dropout: float,
        expert: nn.Module,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([expert for _ in range(num_experts)])
        self.expert_dropout = nn.Dropout(dropout)

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
        G, chosen_expert_index = t.max(S, dim=-1)  # bs each

        # Select top-1 expert, with one-hot vector
        P = nn.functional.one_hot(chosen_expert_index)  # bs num_experts (one-hot)

        # Calculate ExpertFFN result
        E_list = []  # list[batch seq hidden_size]

        for token_index, expert_index in enumerate(chosen_expert_index):
            E_list.append(
                self.expert_dropout(self.experts[expert_index](x[token_index, :]))
            )
        E = t.stack(E_list, dim=0)  # bs hidden_size

        y = einsum(
            "bs_keep num_experts, bs, bs hidden_size -> bs_keep hidden_size",
            P,
            G,
            E,
        )

        y = rearrange(y, "(b s) h -> b s h", b=batch_dim)

        return y


def main():
    expert_layer = SimpleExpertFFN(
        hidden_size=16, num_experts=2, dropout=0.1, expert=nn.Linear(16, 16)
    )

    x = t.rand(size=(2, 4, 16))

    print("x: ", x)
    print("----------------")
    print(f"{expert_layer(x)}")


if __name__ == "__main__":
    main()
