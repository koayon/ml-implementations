import math

import numpy as np
import torch as t
from torch import nn


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        """Flattens out dimensions from start_dim to end_dim, inclusive of both."""

        size_list = list(input.shape[0 : self.start_dim])

        if self.end_dim == -1 or self.end_dim == len(input.shape):
            prod_dims = math.prod(input.shape[self.start_dim :])
            size_list = size_list + [prod_dims]
        else:
            prod_dims = math.prod(input.shape[self.start_dim : self.end_dim + 1])
            size_list = (
                size_list + [prod_dims] + (list(input.shape[self.end_dim + 1 :]))
            )

        return t.reshape(input, size_list)

    def extra_repr(self) -> str:
        return f"Flatten - start dim: {self.start_dim}, end dim: {self.end_dim}"
