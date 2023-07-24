from typing import Any, Optional

import torch as t
from einops import rearrange
from fancy_einsum import einsum
from torch import nn


class ExpertFFN(nn.Module):
    router: nn.Linear
    experts: nn.ModuleList
    expert_dropout: nn.Dropout

    def __init__(
        self,
        *,
        hidden_size: int = 16,
        num_experts: int = 4,
        dropout: float = 0.4,
        expert: nn.Module = nn.Linear(16, 16),
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([expert for _ in range(num_experts)])
        self.expert_dropout = nn.Dropout(dropout)

    def forward_individual_expert(
        self, expert_num: int, tokens_list: list[t.Tensor], x: t.Tensor
    ):
        """
        Set up to be parallelisable
        tokens: list[bs]
        x: bs hidden_size
        """
        tokens = t.stack(tokens_list, dim=0)
        x_out = {
            token_index: self.experts[expert_num](x[token_index, :])
            for token_index in tokens
        }
        return x_out

    def forward(self, x: t.Tensor, cache=None):
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

        expert_tokens = {expert_num: [] for expert_num in range(self.num_experts)}
        for token_num, expert_num in enumerate(chosen_expert_index):
            expert_tokens[expert_num].append(token_num)

        E_dict = {}

        for expert_num, tokens in expert_tokens.items():
            # Potentially parallelised
            E_dict.update(
                self.forward_individual_expert(
                    expert_num=expert_num, tokens_list=tokens, x=x
                )
            )  # {token_index: X_out}
        E_dict = dict(sorted(E_dict.items()))
        E_list = list(E_dict.values())

        E = t.stack(E_list, dim=0)  # bs hidden_size

        y = einsum(
            "bs_keep num_experts, bs, bs hidden_size -> bs_keep hidden_size",
            P,
            G,
            E,
        )

        y = rearrange(y, "(b s) h -> b s h", b=batch_dim)

        return y, cache


def main():
    expert_layer = ExpertFFN(
        hidden_size=16, num_experts=2, dropout=0.1, expert=nn.Linear(16, 16)
    )

    x = t.rand(size=(2, 4, 16))

    print("x: ", x)
    print("----------------")
    print(f"{expert_layer(x)}")


if __name__ == "__main__":
    main()
