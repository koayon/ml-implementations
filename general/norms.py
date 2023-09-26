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

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """Like nn.BatchNorm2d with affine=True."""

        super().__init__()

        self.eps = eps
        self.num_features = num_features
        self.momentum = momentum

        # By default the affine transform is the identity (might learn something else tweaked slightly)
        self.weight = nn.Parameter(t.ones(num_features))  # channels
        self.bias = nn.Parameter(t.zeros(num_features))  # channels

        # Buffers are variables that are part of the model but not trainable parameters.
        # They aren't learned.
        # Each channel (red, blue, green) has its own mean and variance.
        self.register_buffer("running_mean", t.zeros(num_features))  # channels
        self.register_buffer("running_var", t.ones(num_features))  # channels

        self.register_buffer("num_batches_tracked", t.tensor(0))  # scalar

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Normalises each channel along the batch.
        To be used at the minibatch level.
        Downside is that it requires large-ish mini-batches to be useful but large batches may require too much memory.
        Generally prefer LayerNorm or RMSNorm

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        _batch, channels, _height, _width = x.shape
        assert channels == self.num_features

        # If training we're going to get the mean, var from our current batch
        if self.training:
            mean = t.mean(
                x, dim=(0, 2, 3)
            )  # average over batch and spatial dimensions shape(channels)
            var = t.var(
                x, dim=(0, 2, 3), unbiased=False
            )  # variance of batch and spatial dimensions shape(channels)

            # Update running mean and var
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var

        # For inference grab the running_mean/var from the training data and use this instead
        else:
            mean = self.running_mean
            var = self.running_var

        # Rearrange shape(channels) tensors to broadcasts well
        # Takes (channels) -> (1, channels, 1, 1)
        broadcast = lambda v: v.reshape(1, self.num_features, 1, 1)

        # Normalise then learned affine transform
        x_norm = (x - broadcast(mean)) / (broadcast(t.sqrt(var)) + self.eps)
        x_norm *= broadcast(self.weight)
        x_norm += broadcast(self.bias)

        return x_norm

    def extra_repr(self) -> str:
        return f"BatchNorm2d - eps: {self.eps}, momentum: {self.momentum}, num_features: {self.num_features}"


class LayerNorm(nn.Module):
    """LayerNorm implementation as given in https://arxiv.org/pdf/1607.06450.pdf."""

    def __init__(self, shape_without_batch: tuple, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(t.ones(shape_without_batch))  # channels
        self.bias = nn.Parameter(t.zeros(shape_without_batch))  # channels

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: (batch channels *other_dims)
        Return: shape(batch channels *other_dims)
        """
        # Only keep the batch dimension
        dims_to_reduce = tuple(range(1, x.ndim))

        mean = t.mean(x, dim=dims_to_reduce, keepdim=True)
        var = t.var(x, dim=dims_to_reduce, keepdim=True, unbiased=False)

        # Normalise
        x_norm = (x - mean) / (t.sqrt(var) + self.eps)
        x_norm *= self.weight
        x_norm += self.bias

        return x_norm


class RMSNorm(nn.Module):
    """RMS Layer Norm Implementation
    Reference: https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(self, shape_without_batch: tuple, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(t.ones(shape_without_batch))  # channels, *other_dims

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: (batch channels *other_dims)
        Return: shape(batch channels *other_dims)
        """
        # Only keep the batch dimension
        dims_to_reduce = tuple(range(1, x.ndim))

        rms = t.sqrt(t.mean(x**2, dim=dims_to_reduce, keepdim=True))

        # Normalise
        x_norm = (x / (rms + self.eps)) * self.weight

        return x_norm


class GroupRMSNorm(nn.Module):
    """Group Normalisation.
    Layer Norm but instead of normalising over all channels, normalise over groups of channels separately.
    We're actually doing Group RMSNorm here.

    group_size = num_channels gives RMS Layer Norm
    group_size = 1 gives RMS Instance Norm

    Reference: https://arxiv.org/abs/1803.08494"""

    def __init__(self, shape_without_batch: tuple, num_groups: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(t.ones(shape_without_batch))  # channels, *other_dims
        self.channels = shape_without_batch[0]
        self.group_size = self.channels // num_groups

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: (batch channels *other_dims)
        Return: shape(batch channels *other_dims)
        """
        assert x.shape[1] == self.channels

        x_groups = t.split(
            x, self.group_size, dim=1
        )  # list[batch group_size *other_dims]

        # Only keep the batch dimension
        dims_to_reduce = tuple(range(1, x.ndim))

        group_norms = []
        for group in x_groups:
            # Calculate the RMS Norm for each group
            rms = t.sqrt(t.mean(group**2, dim=dims_to_reduce, keepdim=True))
            group_norm = (group / (rms + self.eps)) * self.weight
            group_norms.append(group_norm)

        # Concatenate the groups back together
        x_norm = t.cat(group_norms, dim=1)

        return x_norm


def l2_norm(x: t.Tensor, dim: int, eps: float = 1e-6) -> t.Tensor:
    """L2 Norm of a tensor along a dimension.
    x: shape(*other_dims)
    Return: shape(*other_dims)
    """
    sum_of_squares = t.sum(x**2, dim=dim, keepdim=True)
    norm = t.sqrt(sum_of_squares)

    out = x / (norm + eps)
    return out


class L2LayerNorm(nn.Module):
    """LayerNorm implementation as given in Soft-MoE paper.
    This is to help with keeping the same hyperparameters as we scale up the model size.

    In the router logits calculation they replace X with l2_norm(X) and Phi with scale*l2_norm(Phi), where scale is a learned scalar parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: (batch channels *other_dims)
        Return: shape(batch channels *other_dims)
        """
        return l2_norm(x, dim=self.dim, eps=self.eps)
