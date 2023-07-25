from typing import Union

import numpy as np
import torch as t
from fancy_einsum import einsum
from pad import pad1d, pad2d
from torch import nn

IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]


def conv1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    """Replication of torch's conv1d using bias=False

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """

    pad_x = pad1d(x, padding, padding, 0)

    x_s = pad_x.stride()

    batch_num, in_channels, width = x.shape

    _, _, kernel_width = weights.shape

    output_width = np.floor((width + 2 * padding - kernel_width) / stride) + 1

    strided_x = t.as_strided(
        pad_x,
        size=(
            batch_num,
            in_channels,
            output_width,
            kernel_width,
        ),
        stride=(x_s[0], x_s[1], x_s[2] * stride, x_s[2]),
    )  # batch, in_channels, output_width, kernel_width

    return einsum("ijkl, ojl -> iok", strided_x, weights)


def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)


def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    """Replication of torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    """
    stride_height, stride_width = force_pair(stride)

    pad_height, pad_width = force_pair(padding)
    pad_x = pad2d(x, pad_height, pad_height, pad_width, pad_width, pad_value=0)

    x_s = pad_x.stride()

    batch, in_channels, height, width = x.shape
    _out_channels, _in_channels, kernel_height, kernel_width = weights.shape

    output_height = (
        np.floor((height + 2 * pad_height - kernel_height) / stride_height) + 1
    )
    output_width = np.floor((width + 2 * pad_width - kernel_width) / stride_width) + 1

    strided_x = t.as_strided(
        pad_x,
        size=(
            batch,
            in_channels,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
        ),
        stride=(
            x_s[0],
            x_s[1],
            x_s[2] * stride_height,
            x_s[3] * stride_width,
            x_s[2],
            x_s[3],
        ),
    )  # batch, in_channels, output_height, output_width, kernel_height, kernel_width

    return einsum("bihwYX, oiYX -> bohw", strided_x, weights)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
    ):
        """Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        """
        super().__init__()

        kernel_pair = force_pair(kernel_size)
        in_features = in_channels * kernel_pair[0] * kernel_pair[1]

        kernel = t.randn((out_channels, in_channels, kernel_pair[0], kernel_pair[1]))

        self.weight = nn.Parameter((((kernel) * 2) - 1) / (np.sqrt(in_features)))
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Conv2d forward pass"""
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return f"Conv2d module: Kernel shape: {self.weight.shape}, stride: {self.stride}, padding: {self.padding}"
