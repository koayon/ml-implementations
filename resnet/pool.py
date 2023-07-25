from typing import Optional, Tuple, Union

import numpy as np
import torch as t
from conv import IntOrPair, force_pair
from einops import reduce
from pad import pad2d
from torch import nn


def maxpool2d(
    x: t.Tensor,
    kernel_size: IntOrPair,
    stride: Optional[IntOrPair] = None,
    padding: IntOrPair = 0,
) -> t.Tensor:
    """Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, out_height, out_width)
    """
    kernel_height, kernel_width = force_pair(kernel_size)

    if stride == None:
        stride = kernel_size

    stride_height, stride_width = force_pair(stride)

    pad_height, pad_width = force_pair(padding)
    pad_x = pad2d(
        x, pad_height, pad_height, pad_width, pad_width, pad_value=-float("inf")
    )

    x_s = pad_x.stride()

    batch, channels, height, width = x.shape

    output_height = (
        np.floor((height + 2 * pad_height - kernel_height) / stride_height) + 1
    )
    output_width = np.floor((width + 2 * pad_width - kernel_width) / stride_width) + 1

    strided_x_height = t.as_strided(
        pad_x,
        size=(batch, channels, output_height, kernel_height, width),
        stride=(x_s[0], x_s[1], x_s[2] * stride_height, x_s[2], x_s[3]),
    )

    strided_x = t.as_strided(
        strided_x_height,
        size=(
            batch,
            channels,
            output_height,
            kernel_height,
            output_width,
            kernel_width,
        ),
        stride=(
            x_s[0],
            x_s[1],
            x_s[2] * stride_height,
            x_s[2],
            x_s[3] * stride_width,
            x_s[3],
        ),
    )  # batch, channels, output_height, kernel_height, output_width, kernel_width

    return reduce(
        strided_x,
        "batch channel output_height kernel_height output_height kernel_width -> batch channel output_height output_width",
        "max",
    )


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: Optional[IntOrPair] = None,
        padding: IntOrPair = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        if stride == None:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Maxpool2d forward pass."""
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        return f"maxpool2d: Kernel size : {self.kernel_size}, Stride: {self.stride}, Padding: {self.padding}"


class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return reduce(x, " batch channel height width -> batch channel", "mean")
