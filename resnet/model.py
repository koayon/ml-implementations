import torch as t
from conv import Conv2d
from torch import nn

from general.activations import ReLU
from general.flatten import Flatten
from general.linear import Linear
from general.norms import BatchNorm2d
from resnet.pool import AveragePool, MaxPool2d
from resnet.res_blocks import BlockGroup
from resnet.sequential import Sequential


class ResNet34(nn.Module):
    """ResNet34 model."""

    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        num_blocks = len(n_blocks_per_group)

        in_features_per_group = [64] + out_features_per_group[:-1]

        self.resnet = Sequential(
            Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                padding=3,
                stride=strides_per_group[0],
            ),
            BatchNorm2d(num_features=64),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            *[
                BlockGroup(
                    n_blocks=n_blocks_per_group[i],
                    in_feats=in_features_per_group[i],
                    out_feats=out_features_per_group[i],
                    first_stride=strides_per_group[i],
                )
                for i in range(0, num_blocks)
            ],
        )

        self.final_functions = Sequential(
            AveragePool(),
            Flatten(),
            Linear(
                in_features=out_features_per_group[-1],
                out_features=n_classes,
                bias=True,
            ),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)

        Return: shape (batch, n_classes)
        """
        x = self.resnet(x)
        print(x.shape)

        return self.final_functions(x)
