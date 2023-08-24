import torch as t
import torch.nn as nn

from helpers import ACTIVATION_FUNCTIONS


class FFN(nn.Module):
    linear1: nn.Linear
    linear2: nn.Linear
    ffn_dropout: nn.Dropout
    nonlinearity: nn.Module

    def __init__(
        self,
        *,
        hidden_size: int,
        dropout: float,
        activation_function: str,
        multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

        up_dim = hidden_size * multiplier

        self.linear1 = nn.Linear(hidden_size, up_dim)
        self.nonlinearity = ACTIVATION_FUNCTIONS[activation_function]
        self.linear2 = nn.Linear(up_dim, hidden_size)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        return self.ffn_dropout(self.linear2(self.nonlinearity(self.linear1(x))))
