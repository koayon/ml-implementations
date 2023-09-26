from typing import Union

import torch as t
import torch.nn as nn
from einops import einsum, rearrange


class LinearModel(nn.Module):
    """Linear Model W.T @ W + b with a bias term.
    We're tying the weights of the up and down projections"""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, hidden_dim, bias=False)
        self.W = self.linear.weight
        self.bias = nn.Parameter(t.Tensor(dim))

    def forward(self, x):
        x_prime = self.linear(x)
        x_pred = (
            einsum(
                self.W, x_prime, "hidden_dim dim, batch_dim hidden_dim-> batch_dim dim"
            )
            + self.bias
        )
        return x_pred


class ReLUModel(nn.Module):
    """Model ReLU(W.T @ W + b) with a bias term.
    We are tying the weights of the up and down projections"""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.linear_model = LinearModel(dim=dim, hidden_dim=hidden_dim)
        self.relu = nn.ReLU()
        self.W = self.linear_model.W
        self.bias = self.linear_model.bias

    def forward(self, x):
        x_pred = self.linear_model(x)
        x_pred = self.relu(x_pred)
        return x_pred


class HiddenReLUModel(nn.Module):
    """Adding in a ReLU to the hidden layer in latent space.
    This should force privelleged basis of the neurons rather than any rotation of these.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.W = self.linear.weight
        self.bias = nn.Parameter(t.Tensor(dim))

    def forward(self, x):
        x_prime = self.linear(x)
        x_prime = self.relu(x_prime)
        x_pred = (
            einsum(
                self.W, x_prime, "hidden_dim dim, batch_dim hidden_dim-> batch_dim dim"
            )
            + self.bias
        )
        x_pred = self.relu(x_pred)
        return x_pred


class MLP(nn.Module):
    """A simple MLP with a single hidden layer"""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=True)
        self.relu = nn.ReLU()
        self.W1 = self.linear1.weight
        self.W2 = self.linear2.weight
        self.bias = self.linear2.bias
        self.W = self.W1

    def forward(self, x):
        x_prime = self.linear1(x)
        x_prime = self.relu(x_prime)
        x_pred = self.linear2(x_prime)
        x_pred = self.relu(x_pred)
        return x_pred


class MoE(nn.Module):
    """A minimal MoE with a linear router, two MLP experts batch/2 slots each.
    Here k is such that we can use every expert as much as we want."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        experts: list[nn.Module],
        num_experts: int = 2,
        batch_size=2000,
    ):
        super().__init__()
        self.num_experts = num_experts

        assert self.num_experts == len(experts)

        self.linear_router = nn.Linear(dim, num_experts, bias=True)
        self.experts = experts
        self.mlp2 = MLP(dim, hidden_dim)

        self.W = self.linear_router.weight
        self.bias = self.linear_router.bias

    def forward(self, x):
        router_logits = self.linear_router(x)  # batch, num_experts
        router_probs = t.softmax(router_logits, dim=0)  # batch, num_experts

        expert1_tokens = (router_probs[:, 0].unsqueeze(1) > 0.5) * x
        expert2_tokens = (router_probs[:, 1].unsqueeze(1) > 0.5) * x

        y_prime_1 = self.experts[0](expert1_tokens) * router_probs[:, 0].unsqueeze(1)
        y_prime_2 = self.experts[1](expert2_tokens) * router_probs[:, 1].unsqueeze(1)

        y = y_prime_1 + y_prime_2

        return y
