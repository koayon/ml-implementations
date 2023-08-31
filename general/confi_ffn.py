import torch as t
import torch.nn as nn
from torch.nn import functional as F

from general.basic_ffn import FFN

MULT = 4
RATIO = 2 / 3


class StrengthDial(nn.Module):
    def __init__(self, alpha: float, hidden_size: int, strength_dropout: float, use_softmax: bool = True):
        super().__init__()
        # Two hyperparameters for this paradigm. Alpha is the scaling factor for the sigmoid, and we can initialize the strength to a given value.
        self.alpha = alpha
        INITIAL_STRENGTH_VALUE = -1

        self.use_softmax = use_softmax

        # Conceptually we have an inner strength and an outer strength depending on x and y respectively. But they're really combined into a single strength score.
        if use_softmax:
            self.strength = nn.Linear(hidden_size * 2, 2)
        else:
            self.strength = nn.Linear(hidden_size * 2, 1)
        # Default strengths to -1
        t.nn.init.constant_(self.strength.weight, INITIAL_STRENGTH_VALUE)

        self.strength_dropout = nn.Dropout(strength_dropout)

    def forward(self, x: t.Tensor, y: t.Tensor) -> t.Tensor:
        strength_logits = self.strength(t.concat([x, y], dim=-1))  # (batch, seq, 2) for softmax; (batch, seq, 1) for sigmoid
        if self.use_softmax:
            strength = F.softmax(strength_logits / self.alpha, dim = -1)  # (batch, seq, 2)
            strength = strength[:, :, 0:1]  # (batch, seq, 1)
        else:
            strength = F.sigmoid(strength_logits / self.alpha)
        print(strength.shape)
        # If dropout instead of default dropping the layer, we default to using the layer with the 1 - prob
        strength = 1 - self.strength_dropout(strength)  # (batch, seq, 1)

        return strength


class ConfiFFN(nn.Module):
    """ConfiFFN reminiscient of the Highway Network of Srivastava et al (at least for the inner strength component).
    Previously referred to as SigmoidFFN.

    This approach takes advantage of modern activation functions and better defaults.
    It also always preserves the residual stream rather than possibly wiping it out (as in the case T = 0, C = 1 in the Highway Network approach). This is much more stable for large networks. Here we only affect how much is _written_ to the residual stream.

    Another benefit here is the "outer strength". Here we allow the module to realise that it's output is trash and let the strength multiplier know this so that we can ignore or at least downweight this output.

    This can also be thought of as Soft-MoE with one real expert and one 0 expert.

    https://arxiv.org/pdf/1505.00387.pdf
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        dropout: float,
        activation_function: str,
        strength_dropout: float = 0.2,
        alpha: float = 1.0
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.MLP = FFN(
            hidden_size=hidden_size,
            dropout=dropout,
            activation_function=activation_function,
        )

        self.strength_dial = StrengthDial(
            alpha=alpha, hidden_size=hidden_size, strength_dropout=strength_dropout
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        y = self.MLP(x)  # (batch, seq, hidden_size)

        strength = self.strength_dial(x, y)  # (batch, seq, 1)

        return strength * y  # (batch, seq, hidden_size)
