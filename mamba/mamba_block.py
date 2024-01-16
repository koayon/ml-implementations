import torch as t
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float

from mamba.config import MambaConfig
from mamba.s6 import S6


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig = MambaConfig()):
        super().__init__()

        self.mamba_block = MambaBlock(
            hidden_dim=config.hidden_dim,
            residual_dim=config.residual_dim,
            expansion_factor=config.expansion_factor,
            conv_kernel_size=config.conv_kernel_size,
        )

        self.layer_norm = nn.LayerNorm(config.residual_dim)

    def forward(
        self, x: Float[t.Tensor, "batch seq_len input_dim"]
    ) -> Float[t.Tensor, "batch seq_len input_dim"]:
        """Run the Mamba Residual Block.
        See Section 3.4 in the Mamba Paper.

        Returns
        -------
        _type_
            _description_
        """
        y = self.layer_norm(x)
        y = self.mamba_block(y)
        return x + y


class MambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        residual_dim: int,
        expansion_factor: int = 2,
        conv_kernel_size: int = 3,
    ):
        super().__init__()

        ssm_input_dim = hidden_dim * expansion_factor

        self.double_up_proj = nn.Linear(residual_dim, ssm_input_dim * 2)

        self.swish = nn.SiLU()

        self.conv = nn.Conv1d(
            ssm_input_dim,
            ssm_input_dim,
            kernel_size=conv_kernel_size,
            groups=ssm_input_dim,
            padding=conv_kernel_size - 1,
        )

        self.s6 = S6(ssm_input_dim, hidden_dim)

        self.down_proj = nn.Linear(ssm_input_dim, residual_dim)

    def convolve_on_seq_dim(
        self, x: Float[t.Tensor, "batch seq_len input_dim"]
    ) -> Float[t.Tensor, "batch seq_len input_dim"]:
        """Convolve on the sequence dimension.

        Returns
        -------
        x : Float[t.Tensor, "batch seq_len input_dim"]
        """
        x = rearrange(
            x, "batch seq_len ssm_input_dim -> batch ssm_input_dim seq_len"
        )  # batch, ssm_input_dim, seq_len
        x = self.conv(x)  # batch, ssm_input_dim, seq_len
        x = rearrange(
            x, "batch ssm_input_dim seq_len -> batch seq_len ssm_input_dim"
        )  # batch, seq_len, ssm_input_dim

        return x

    def forward(
        self, up_x: Float[t.Tensor, "batch seq_len residual_dim"]
    ) -> Float[t.Tensor, "batch seq_len residual_dim"]:
        """Run the MambaBlock forward.
        See Section 3, Figure 3 in the Mamba Paper.

        Returns
        -------
        y : Float[t.Tensor, "batch seq_len residual_dim"]
        """
        # Project up
        up_x = self.double_up_proj(up_x)  # batch, seq_len, ssm_input_dim * 2

        left_branch_x, right_branch_x = t.split(
            up_x, 2, dim=-1
        )  # batch, seq_len, ssm_input_dim

        # LEFT BRANCH

        left_branch_x = self.convolve_on_seq_dim(left_branch_x)
        left_branch_x = self.swish(left_branch_x)
        left_branch_x = self.s6(left_branch_x)  # batch, seq_len, ssm_input_dim

        # RIGHT BRANCH

        right_branch_x = self.swish(right_branch_x)

        # Combine for SwiGLU style gating

        up_x = left_branch_x * right_branch_x  # batch, seq_len, ssm_input_dim

        x = self.down_proj(up_x)  # batch, seq_len, residual_dim

        return x
