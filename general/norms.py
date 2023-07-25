from typing import Optional, Tuple, Union

import numpy as np
import torch as t
from einops import rearrange, repeat
from torch import nn


class BatchNorm2d(nn.Module):
    running_mean: t.Tensor
    "running_mean: shape (num_features,)"
    running_var: t.Tensor
    "running_var: shape (num_features,)"
    num_batches_tracked: t.Tensor
    "num_batches_tracked: shape ()"

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """

        super().__init__()

        self.eps = eps
        self.num_features = num_features
        self.momentum = momentum

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        # Buffers are variables that are part of the model but not trainable parameters.
        # They aren't learned.
        # Each channel (red, blue, green) has its own mean and variance.
        self.register_buffer("running_mean", t.zeros(num_features))  # channels
        self.register_buffer("running_var", t.ones(num_features))  # channels

        self.register_buffer("num_batches_tracked", t.tensor(0))  # scalar

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        batch, _channels, height, width = x.shape

        if self.training:
            mean = t.mean(x, dim=(0, 2, 3))  # aveage over batch and spatial dimensions
            var = t.var(
                x, dim=(0, 2, 3), unbiased=False
            )  # variance of batch and spatial dimensions
        else:
            mean = self.running_mean
            var = self.running_var

        num = x - repeat(
            mean, "channels -> batch channels height width", b=batch, h=height, w=width
        )
        denom = t.sqrt(
            repeat(
                var,
                "channels -> batch channels height width",
                b=batch,
                h=height,
                w=width,
            )
            + self.eps
        )

        weight = repeat(self.weight, "c-> b c h w", b=batch, h=height, w=width)
        bias = repeat(self.bias, "c-> b c h w", b=batch, h=height, w=width)

        y = num / denom * weight + bias

        self.running_mean = (
            1 - self.momentum
        ) * self.running_mean + self.momentum * mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        self.num_batches_tracked += 1

        return y

    def extra_repr(self) -> str:
        return f"BatchNorm2d - eps: {self.eps}, momentum: {self.momentum}, num_features: {self.weight.shape}"
