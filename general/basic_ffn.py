import torch as t
import torch.nn as nn

from helpers import ACTIVATION_FUNCTIONS


class BasicFFN(nn.Module):
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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.nonlinearity = ACTIVATION_FUNCTIONS[activation_function]
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        return self.ffn_dropout(self.linear2(self.nonlinearity(self.linear1(x))))
