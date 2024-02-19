from functools import lru_cache

import torch as t
import torch.nn as nn
from torch.nn import functional as F


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        """
        Parameters
        ----------
        in_dim : int
            in_dim of Linear Layer
        out_dim : int
            out_dim of Linear Layer
        rank : int
            The rank of the low-rank approximation. Proportional to the number of parameters for the LoRA layer.
        """
        super().__init__()
        std_dev = 1 / t.sqrt(t.tensor(rank).float())
        self.A = nn.Parameter(t.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(t.zeros(rank, out_dim))

    def forward(self, x: t.Tensor) -> t.Tensor:
        lora = self.A @ self.B  # in_dim, out_dim
        return x @ lora


class LinearWithLoRA(nn.Module):

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        """
        alpha : float
            How much weight to give to the low-rank approximation vs the pre-trained layer.
        """

        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank)
        self.alpha = alpha

    def forward(self, x) -> t.Tensor:
        return self.linear(x) + self.alpha * self.lora(x)


class LinearWithLoRAMerged(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank)
        self.alpha = alpha

    @lru_cache(maxsize=None)
    def merge_lora(self) -> t.Tensor:
        lora = self.lora.A @ self.lora.B  # Combine LoRA matrices

        # Then combine LoRA with orig. weights
        merged_linear_weights = self.linear.weight + self.alpha * lora.T
        return merged_linear_weights

    def forward(self, x) -> t.Tensor:
        merged_linear_weights = self.merge_lora()
        return F.linear(x, merged_linear_weights, self.linear.bias)
