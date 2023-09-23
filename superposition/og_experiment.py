from typing import Union

import plotly.express as px
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import einsum
from plotly.subplots import make_subplots

# Set consts
SPARSITY = 0.999
DIM = 25
HIDDEN_DIM = 6

BATCH_SIZE = 2000
importances = t.tensor([0.7**i for i in range(1, DIM + 1)])


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(DIM, HIDDEN_DIM, bias=False)
        self.W = self.linear.weight
        self.bias = nn.Parameter(t.Tensor(DIM))

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
    def __init__(self):
        super().__init__()
        self.linear_model = LinearModel()
        self.relu = nn.ReLU()
        self.W = self.linear_model.W
        self.bias = self.linear_model.bias

    def forward(self, x):
        x_pred = self.linear_model(x)
        x_pred = self.relu(x_pred)
        return x_pred


Model = Union[LinearModel, ReLUModel]


def train_model(model: Model, x: t.Tensor) -> Model:
    """Run training loop for model"""

    def importance_loss(x, x_preds, importances=importances):
        return t.sum((x - x_preds) ** 2 * importances)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(200):
        optimizer.zero_grad()
        x_preds = model(x)
        loss = importance_loss(x, x_preds)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item():.2f}")
    return model


def show_heatmap(
    W: t.Tensor, bias: t.Tensor, model_type: str, sparsity: float = SPARSITY
) -> None:
    """Plotting function for W.T @ W and bias

    Parameters
    ----------
    W : t.Tensor
        hidden_dim, dim
    bias : t.Tensor
        dim
    model_type : str
        _description_
    sparsity : float, optional
        _description_, by default SPARSITY
    """
    sup_matrix = W.T @ W  # dim, dim
    bias = bias.unsqueeze(1)  # 1, dim

    # Show W.T @ W
    fig = px.imshow(
        sup_matrix.detach().numpy(),
        title=f"W.T @ W for {model_type} model, S = {sparsity}",
        color_continuous_scale="Portland",
        zmin=-1,
        zmax=1,
    )
    fig.show()

    # Show bias
    fig = px.imshow(
        bias.detach().numpy(),
        title=f"Bias Matrix for {model_type} model, S = {sparsity}",
        color_continuous_scale="Portland",
        zmin=-1,
        zmax=1,
    )
    fig.show()


# Define synthetic data, x
x = t.rand(BATCH_SIZE, DIM)
sparsity_mask = t.rand(BATCH_SIZE, DIM) > SPARSITY
x = x * sparsity_mask

# Linear Model
model = LinearModel()
model = train_model(model, x)
W = model.W
bias = model.bias
show_heatmap(W, bias, "linear")

# ReLU Model
model = ReLUModel()
model = train_model(model, x)
W = model.W
bias = model.bias
show_heatmap(W, bias, "relu")
