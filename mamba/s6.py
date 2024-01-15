from typing import Any, Callable, List, Optional, Tuple, Union

import torch as t
import torch.nn as nn
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float, Int
from torch.nn import functional as F


class S6(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.input_dim = input_dim

        A = t.empty((input_dim, hidden_dim))
        nn.init.xavier_uniform_(A)
        self.A = nn.Parameter(A)  # input_dim, hidden_dim

        self.W_B = nn.Linear(input_dim, hidden_dim)
        self.W_C = nn.Linear(input_dim, hidden_dim)

        delta_param = t.zeros(input_dim)
        self.delta_param = nn.Parameter(delta_param)  # input_dim

        self.W_delta = nn.Linear(input_dim, 1)

        self.ssm = SSM()

    def discretize(
        self,
        A: Float[t.Tensor, "input_dim hidden_dim"],
        B: Float[t.Tensor, "batch seq_len hidden_dim"],
        delta: Float[t.Tensor, "batch seq_len input_dim"],
    ) -> tuple[
        Float[t.Tensor, "batch seq_len input_dim hidden_dim"],
        Float[t.Tensor, "batch seq_len input_dim hidden_dim"],
    ]:
        raise NotImplementedError

    def forward(
        self,
        x: Float[t.Tensor, "batch seq_len dim"],
    ) -> Float[t.Tensor, "batch seq_len dim"]:
        B = self.W_B(x)  # batch, seq_len, hidden_dim
        C = self.W_C(x)  # batch, seq_len, hidden_dim

        s_delta = repeat(
            self.W_delta(x), "batch seq_len 1 -> batch seq_len dim", dim=self.input_dim
        )  # batch, seq_len, input_dim

        delta = F.softplus(self.delta_param + s_delta)  # batch, seq_len, input_dim

        A_disc, B_disc = self.discretize(
            self.A, B, delta
        )  # batch, seq_len, input_dim, hidden_dim

        y = self.ssm(A_disc, B_disc, C, x)  # batch, seq_len, dim

        return y
