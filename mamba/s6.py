from typing import Optional

import torch as t
import torch.nn as nn
from einops import einsum, repeat
from jaxtyping import Float
from torch.nn import functional as F


class HUpdater(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, Ah: Optional[t.Tensor], Bx: t.Tensor, seq_num: Optional[int] = None
    ) -> t.Tensor:
        """Update the hidden state of the SSM.
        Made into a module for hooking purposes. seq_num variable is used for ablations.

        Returns
        -------
        h : t.Tensor
            h_t
        """
        if Ah is None:
            return Bx
        else:
            return Ah + Bx


class SSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssm_state: Optional[Float[t.Tensor, "batch input_dim hidden_dim"]] = None
        self.h_updater = HUpdater()

    def _forward_recurrent(
        self,
        A: Float[t.Tensor, "batch seq_len input_dim hidden_dim"],
        B: Float[t.Tensor, "batch seq_len input_dim hidden_dim"],
        C: Float[t.Tensor, "batch seq_len hidden_dim"],
        x: Float[t.Tensor, "batch seq_len input_dim"],
    ) -> tuple[
        Float[t.Tensor, "batch seq_len input_dim"],
        Float[t.Tensor, "batch seq_len input_dim hidden_dim"],
    ]:
        """Run the SSM forward in a recurrent manner.

        Equations 2a and 2b in the Mamba paper.

        Returns
        -------
        y : Float[t.Tensor, "batch seq_len input_dim"]
        h : Float[t.Tensor, "batch seq_len input_dim hidden_dim"]
        """
        batch_size, seq_len, input_dim, hidden_dim = A.shape

        h: Float[t.Tensor, "batch seq_len input_dim hidden_dim"] = t.zeros(
            (batch_size, seq_len, input_dim, hidden_dim), device=A.device
        )

        for seq_num in range(seq_len):
            # h_t = A h_{t-1} + B x_t (element-wise multiplication)
            # y_t = C h_t (matrix multiplication)

            B_xt = einsum(
                B[:, seq_num, :, :],
                x[:, seq_num, :],
                "batch input_dim hidden_dim, batch input_dim -> batch input_dim hidden_dim",
            )  # B x_t  # batch, input_dim, hidden_dim

            A_h_t1: Optional[Float[t.Tensor, "batch input_dim hidden_dim"]] = None
            if seq_num:
                A_h_t1 = einsum(
                    A[:, seq_num, :, :],
                    h[:, seq_num - 1, :, :],
                    "batch input_dim hidden_dim, batch input_dim hidden_dim -> batch input_dim hidden_dim",
                )  # batch, input_dim, hidden_dim

            h[:, seq_num, :] = self.h_updater(A_h_t1, B_xt)  # h_t

        self.ssm_state = h[:, -1, ...]  # batch, input_dim, hidden_dim

        y = einsum(
            C,
            h,
            "batch seq_len hidden_dim, batch seq_len input_dim hidden_dim -> batch seq_len input_dim",
        )  # C h_t  # batch, seq_len, input_dim

        return y, h

    def step(
        self,
        A: Float[t.Tensor, "batch input_dim hidden_dim"],
        B: Float[t.Tensor, "batch input_dim hidden_dim"],
        C: Float[t.Tensor, "batch hidden_dim"],
        x: Float[t.Tensor, "batch input_dim"],
    ) -> Float[t.Tensor, "batch seq_len dim"]:
        """Complete one step of the SSM, given the ssm_state vectors.

        Equations 2a and 2b in the Mamba paper.

        Returns
        -------
        y : Float[t.Tensor, "batch seq_len input_dim"]
        h : Float[t.Tensor, "batch seq_len input_dim hidden_dim"]
        """
        # h_t = A h_{t-1} + B x_t (element-wise multiplication)
        # y_t = C h_t (matrix multiplication)

        h = self.ssm_state

        B_xt = einsum(
            B,
            x,
            "batch input_dim hidden_dim, batch input_dim -> batch input_dim hidden_dim",
        )
        A_h_t1 = einsum(
            A,
            h,
            "batch input_dim hidden_dim, batch input_dim hidden_dim -> batch input_dim hidden_dim",
        )
        # TODO: If we're ablating a state - we want to eliminate the B_xt term for sure.
        # Should we also eliminate the A_h_t1 term? Probs yes because there's some dependence on x through A.
        # But also might we want to have SOME dampening of the previous state?
        # Otherwise we might be throwing it off its mental clock/position counter.
        # Similarly C is dependant on x too! So we should probably ablate the output of the layer too.

        h = self.h_updater(A_h_t1, B_xt)  # batch, input_dim, hidden_dim
        self.ssm_state = h

        y = einsum(
            C,
            h,
            "batch hidden_dim, batch input_dim hidden_dim -> batch input_dim",
        )  # C h_t  # batch, input_dim

        return y

    def forward(
        self,
        A: Float[t.Tensor, "batch seq_len input_dim hidden_dim"],
        B: Float[t.Tensor, "batch seq_len input_dim hidden_dim"],
        C: Float[t.Tensor, "batch seq_len hidden_dim"],
        x: Float[t.Tensor, "batch seq_len input_dim"],
    ) -> Float[t.Tensor, "batch seq_len dim"]:
        if self.ssm_state is None:
            y, _ = self._forward_recurrent(A, B, C, x)
            return y

        else:
            x_final = x[:, -1, :]  # batch, input_dim
            A_final = A[:, -1, :, :]  # batch, input_dim, hidden_dim
            B_final = B[:, -1, :, :]  # batch, input_dim, hidden_dim
            C_final = C[:, -1, :]  # batch, hidden_dim
            y = self.step(A_final, B_final, C_final, x_final)
            return y


class S6(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.input_dim = input_dim

        A = t.empty((input_dim, hidden_dim))
        nn.init.xavier_uniform_(A)
        self.A = nn.Parameter(A)  # input_dim, hidden_dim

        self.W_B = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_C = nn.Linear(input_dim, hidden_dim, bias=False)

        self.W_delta = nn.Linear(input_dim, 1, bias=True)

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
        """Discretize the continuous-time SSM parameters A and B using delta.

        A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper)
        We use a simplified Euler discretization for B, which is the less important than A for performance.

        In the paper they use the ZOH discretization for both A and B.

        Returns
        -------
        A_disc : Float[t.Tensor, "batch seq_len input_dim hidden_dim"]
        B_disc : Float[t.Tensor, "batch seq_len input_dim hidden_dim"]
        """
        # Element-wise multiplication and exponentiation
        delta_A = einsum(
            delta,
            A,
            "batch seq_len input_dim, input_dim hidden_dim -> batch seq_len input_dim hidden_dim",
        )  # batch, seq_len, input_dim, hidden_dim

        A_disc = t.exp(delta_A)  # batch, seq_len, input_dim, hidden_dim

        delta_B = einsum(
            delta,
            B,
            "batch seq_len input_dim, batch seq_len hidden_dim -> batch seq_len input_dim hidden_dim",
        )  # batch, seq_len, input_dim, hidden_dim

        return A_disc, delta_B

    def forward(
        self,
        x: Float[t.Tensor, "batch seq_len input_dim"],
    ) -> Float[t.Tensor, "batch seq_len input_dim"]:
        """Run the forward pass of the SSM for Mamba. See Section 1, Figure 1 in the Mamba paper.
        Also see Section 3, Algorithm 2.

        Returns
        -------
        y : Float[t.Tensor, "batch seq_len input_dim"]
        """
        B = self.W_B(x)  # batch, seq_len, hidden_dim
        C = self.W_C(x)  # batch, seq_len, hidden_dim

        s_delta = repeat(
            self.W_delta(x),
            "batch seq_len 1 -> batch seq_len input_dim",
            input_dim=self.input_dim,
        )  # batch, seq_len, input_dim

        delta = F.softplus(s_delta)  # batch, seq_len, input_dim

        A_disc, B_disc = self.discretize(
            self.A, B, delta
        )  # batch, seq_len, hidden_dim, input_dim

        y = self.ssm(A_disc, B_disc, C, x)  # batch, seq_len, input_dim

        return y

    # TODO: Add step method for S6 (don't need to dicretize A and B on the whole sequence)
