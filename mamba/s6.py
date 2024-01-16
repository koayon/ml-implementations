import torch as t
import torch.nn as nn
from einops import einsum, repeat
from jaxtyping import Float
from torch.nn import functional as F


class SSM(nn.Module):
    def __init__(self):
        super().__init__()

    def _forward_recurrent(
        self,
        A: Float[t.Tensor, "batch seq_len input_dim hidden_dim"],
        B: Float[t.Tensor, "batch seq_len input_dim hidden_dim"],
        C: Float[t.Tensor, "batch seq_len hidden_dim"],
        x: Float[t.Tensor, "batch seq_len input_dim"],
    ) -> Float[t.Tensor, "batch seq_len input_dim"]:
        """Run the SSM forward in a recurrent manner.

        Equations 2a and 2b in the Mamba paper.

        Returns
        -------
        y : Float[t.Tensor, "batch seq_len input_dim"]

        # TODO: Amend to enable only final step if you have all the previous h values (hidden states)
        """
        batch_size, seq_len, input_dim, hidden_dim = A.shape

        h: Float[t.Tensor, "batch seq_len input_dim hidden_dim"] = t.zeros(
            batch_size, seq_len, input_dim, hidden_dim
        )
        y_list: list[Float[t.Tensor, "batch input_dim"]] = []
        for seq_num in range(seq_len):
            # h_t = A h_{t-1} + B x_t (element-wise multiplication)
            # y_t = C h_t (matrix multiplication)

            B_xt = einsum(
                B[:, seq_num, :, :],
                x[:, seq_num, :],
                "batch input_dim hidden_dim, batch input_dim -> batch input_dim hidden_dim",
            )  # B x_t  # batch, input_dim, hidden_dim
            if seq_num:
                A_h_t1 = einsum(
                    A[:, seq_num, :, :],
                    h[:, seq_num - 1, :, :],
                    "batch input_dim hidden_dim, batch input_dim hidden_dim -> batch input_dim hidden_dim",
                )  # batch, input_dim, hidden_dim
                h[:, seq_num, :] = A_h_t1 + B_xt  # h_t
            else:
                h[:, seq_num, :] = B_xt  # h_t

            y_t = einsum(
                C[:, seq_num, :],
                h[:, seq_num, :],
                "batch hidden_dim, batch input_dim hidden_dim -> batch input_dim",
            )  # C h_t  # batch, input_dim

            y_list.append(y_t)

        y = t.stack(y_list, dim=1)  # batch, seq_len, input_dim

        return y

    def _forward_convolutional_scan(
        self,
        A: Float[t.Tensor, "batch seq_len input_dim hidden_dim"],
        B: Float[t.Tensor, "batch seq_len input_dim hidden_dim"],
        C: Float[t.Tensor, "batch seq_len hidden_dim"],
        x: Float[t.Tensor, "batch seq_len input_dim"],
    ) -> Float[t.Tensor, "batch seq_len dim"]:
        """Run the SSM forward using the hardware-efficient scan.

        Equations 3a and 3b in the Mamba paper.
        Also see Section 3.3.2.

        Returns
        -------
        y : Float[t.Tensor, "batch seq_len input_dim"]

        # TODO: Amend to enable only final step if you have all the previous h values (hidden states)
        """
        raise NotImplementedError

    def forward(
        self,
        A: Float[t.Tensor, "batch seq_len input_dim hidden_dim"],
        B: Float[t.Tensor, "batch seq_len input_dim hidden_dim"],
        C: Float[t.Tensor, "batch seq_len hidden_dim"],
        x: Float[t.Tensor, "batch seq_len input_dim"],
    ) -> Float[t.Tensor, "batch seq_len dim"]:
        if self.training:
            # return self._forward_convolutional_scan(A, B, C, x)
            return self._forward_recurrent(A, B, C, x)
        else:
            return self._forward_recurrent(A, B, C, x)


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
