import torch as t
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        std_dev = 1 / t.sqrt(t.tensor(rank).float())
        self.A = nn.Parameter(t.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(t.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.alpha * (x @ self.A @ self.B)
        return x
