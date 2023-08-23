import torch as t
import torch.nn as nn
from regex import R

from helpers import ACTIVATION_FUNCTIONS, einsum

MULT = 4
RATIO = 2 / 3


class ConfiFFN(nn.Module):
    """ConfiFFN reminiscient of the Highway Network of Srivastava et al (at least for the inner strength component).
    Previously referred to as SigmoidFFN.

    This approach takes advantage of modern activation functions and better defaults.
    It also always preserves the residual stream rather than possibly wiping it out (as in the case T = 0, C = 1 in the Highway Network approach). This is much more stable for large networks. Here we only affect how much is _written_ to the residual stream.

    Another benefit here is the "outer strength". Here we allow the module to realise that it's output is trash and let the strength multiplier know this so that we can ignore or at least downweight this output.

    https://arxiv.org/pdf/1505.00387.pdf
    """

    linear1: nn.Linear
    linear2: nn.Linear

    strength: nn.Linear

    strength_dropout: nn.Dropout
    ffn_dropout: nn.Dropout
    nonlinearity: nn.Module

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

        # Two hyperparameters for this paradigm. Alpha is the scaling factor for the sigmoid, and we can initialize the strength to a given value.
        self.alpha = alpha
        initial_strength_value = -1

        # Conceptually we have an inner strength and an outer strength depending on x and y respectively. But they're really combined into a single strength score.
        self.strength = nn.Linear(hidden_size * 2, 1)
        # Default strengths to 1
        t.nn.init.constant_(self.strength.weight, initial_strength_value)

        up_dim = int(hidden_size * MULT * RATIO)

        self.linear1 = nn.Linear(hidden_size, up_dim)
        self.nonlinearity = ACTIVATION_FUNCTIONS[activation_function]
        self.linear2 = nn.Linear(up_dim, hidden_size)

        self.ffn_dropout = nn.Dropout(dropout)
        self.strength_dropout = nn.Dropout(strength_dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        y = self.ffn_dropout(
            self.linear2(self.nonlinearity(self.linear1(x)))
        )  # (batch, seq, hidden_size)

        strength_logits = self.strength(t.concat([x, y], dim=-1))  # (batch, seq, 1)
        strength = t.sigmoid(strength_logits / self.alpha)  # (batch, seq, 1)
        # If dropout instead of default dropping the layer, we default to using the layer with the 1 - prob
        strength = 1 - self.strength_dropout(strength)  # (batch, seq, 1)

        return strength * y  # (batch, seq, hidden_size)
