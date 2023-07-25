import torch as t
from conv import Conv2d
from sequential import Sequential
from torch import nn

from general.activations import ReLU
from general.norms import BatchNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """A single residual block based on ResNet34."""
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.left = Sequential(
            Conv2d(
                in_channels=in_feats,
                out_channels=out_feats,
                kernel_size=3,
                padding=1,
                stride=first_stride,
            ),
            BatchNorm2d(num_features=out_feats),
            ReLU(),
            Conv2d(
                in_channels=out_feats, kernel_size=3, padding=1, out_channels=out_feats
            ),
            BatchNorm2d(num_features=out_feats),
        )

        # If first_stride is > 1, we add the optional (conv + bn) on the right branch.
        if first_stride > 1:
            self.right = Sequential(
                Conv2d(
                    in_channels=in_feats,
                    out_channels=out_feats,
                    kernel_size=1,
                    padding=0,
                    stride=first_stride,
                ),
                BatchNorm2d(num_features=out_feats),
            )

        self.first_stride = first_stride

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)
        """

        if self.first_stride > 1:
            y = self.left(x) + self.right(x)
        else:
            y = self.left(x) + x

        relu = ReLU()
        return relu(y)


class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride. As in ResNet34."""
        super().__init__()
        self.blocks = Sequential(
            ResidualBlock(
                in_feats=in_feats, out_feats=out_feats, first_stride=first_stride
            ),
            *[
                ResidualBlock(in_feats=out_feats, out_feats=out_feats, first_stride=1)
                for i in range(0, n_blocks - 1)
            ]
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        return self.blocks(x)
