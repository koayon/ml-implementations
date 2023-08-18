from typing import Optional

import torch as t
from torch import nn
from torch.nn import functional as F


class SwiGLUFeedForward(nn.Module):
    """Implements a SwiGLU feed forward network inspired by Swish paper: https://arxiv.org/pdf/1710.05941v1.pdf

    Llama 2 paper receommends SwiGLU as the activation function for FFN with hidden_dim = 2 / 3 * in_features * mult
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
    ):
        super().__init__()
        out_features = in_features

        self.w1 = nn.Linear(
            in_features=in_features, out_features=hidden_dim, bias=False
        )

        self.w2 = nn.Linear(
            in_features=hidden_dim, out_features=out_features, bias=False
        )

    def forward(self, x):
        # FFN_ReLU would be simply w2(ReLU(w1_x)) instead we use SwiGLU:  w2(Swish(w1_x))
        # Where swish = sigmoid(x) * x (the gating function)

        swish = F.silu(self.w1(x))
        out = self.w2(swish)
        return out
