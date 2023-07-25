import numpy as np
import torch as t
from torch import nn


class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.tensor(0))


class GeLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        erf = t.erf(x / np.sqrt(2))
        return x / 2 * (1 + erf)
