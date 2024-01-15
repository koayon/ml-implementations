import re
from typing import Any, Callable, List, Optional, Tuple, Union

import torch as t
import torch.nn as nn
from einops import einsum, rearrange, repeat
from jax import Array
from jaxtyping import Bool, Float, Int
from torch.nn import functional as F


class SSM(nn.Module):
    def __init__(self):
        super().__init__()

    def _forward_recurrent(
        self,
        A: Float[t.Tensor, "batch seq_len hidden_dim input_dim"],
        B: Float[t.Tensor, "batch seq_len hidden_dim input_dim"],
        C: Float[t.Tensor, "batch seq_len hidden_dim"],
        x: Float[t.Tensor, "batch seq_len input_dim"],
    ) -> Float[t.Tensor, "batch seq_len dim"]:
        batch_size, seq_len, input_dim, hidden_dim = A.shape

        h: Float[t.Tensor, "batch seq_len hidden_dim"] = t.zeros(
            batch_size, seq_len, hidden_dim
        )
        y_list: list[Float[t.Tensor, "batch input_dim"]] = []
        for seq_num in range(seq_len):
            # h_t = A h_{t-1} + B x_t
            # y_t = C h_t

            B_xt = einsum(
                B[:, seq_num, :, :],
                x[:, seq_num, :],
                "batch hidden_dim input_dim, batch hidden_dim -> batch hidden_dim",
            )  # B x_t
            if seq_num:
                A_h_t1 = einsum(
                    A[:, seq_num, :, :],
                    h[:, seq_num - 1, :],
                    "batch hidden_dim input_dim, batch hidden_dim -> batch hidden_dim",
                )  # A h_{t-1}
                h[:, seq_num, :] = A_h_t1 + B_xt  # h_t
            else:
                h[:, seq_num, :] = B_xt  # h_t

            y_t = einsum(
                C[:, seq_num, :],
                h[:, seq_num, :],
                "batch hidden_dim, batch hidden_dim -> batch input_dim",
            )  # C h_t  # batch, input_dim

            y_list.append(y_t)

        y = t.stack(y_list, dim=1)  # batch, seq_len, input_dim

        # TODO: Something off with shape of B I thinkkk

        return y

    def _forward_convolutional_scan(
        self,
        A: Float[t.Tensor, "batch seq_len hidden_dim input_dim"],
        B: Float[t.Tensor, "batch seq_len hidden_dim input_dim"],
        C: Float[t.Tensor, "batch seq_len hidden_dim"],
        x: Float[t.Tensor, "batch seq_len input_dim"],
    ) -> Float[t.Tensor, "batch seq_len dim"]:
        raise NotImplementedError

    def forward(
        self,
        A: Float[t.Tensor, "batch seq_len hidden_dim input_dim"],
        B: Float[t.Tensor, "batch seq_len hidden_dim input_dim"],
        C: Float[t.Tensor, "batch seq_len hidden_dim"],
        x: Float[t.Tensor, "batch seq_len input_dim"],
    ) -> Float[t.Tensor, "batch seq_len dim"]:
        if self.training:
            return self._forward_convolutional_scan(A, B, C, x)
        else:
            return self._forward_recurrent(A, B, C, x)


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
        Float[t.Tensor, "batch seq_len hidden_dim input_dim"],
        Float[t.Tensor, "batch seq_len hidden_dim input_dim"],
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
        )  # batch, seq_len, hidden_dim, input_dim

        y = self.ssm(A_disc, B_disc, C, x)  # batch, seq_len, dim

        return y
