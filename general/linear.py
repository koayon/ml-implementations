from typing import Optional

import numpy as np
import torch as t
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """A simple linear transformation (or affine with bias)"""
        super().__init__()

        self.weight = nn.Parameter(
            ((t.randn((out_features, in_features)) * 2) - 1) / (np.sqrt(in_features))
        )

        if bias:
            self.bias = nn.Parameter(
                ((t.randn((out_features,)) * 2) - 1) / (np.sqrt(in_features))
            )
        else:
            # If `bias` is False, set `self.bias` to None.
            self.bias = None

        # print(dict(self.named_parameters())  )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        if self.bias is None:
            return x @ self.weight.T
        else:
            return (x @ self.weight.T) + self.bias

    def extra_repr(self) -> str:
        return f"Linear. Weight matrix shape : {self.weight.shape}, (in_features x out_features). Bias: {self.bias != None}"
