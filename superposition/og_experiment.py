import plotly.express as px
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import einsum
from plotly.subplots import make_subplots

SPARSITY = 0.999
DIM = 25
HIDDEN_DIM = 6

BATCH_SIZE = 2000
importances = t.tensor([0.7**i for i in range(1, DIM + 1)])

x = t.rand(BATCH_SIZE, DIM)

sparsity_mask = t.rand(BATCH_SIZE, DIM) > SPARSITY
x = x * sparsity_mask


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

    def forward(self, x):
        x_pred = self.linear_model(x)
        x_pred = self.relu(x_pred)
        return x_pred


def importance_loss(x, x_preds, importances=importances):
    # print(x)
    # print(x_preds)
    return t.sum((x - x_preds) ** 2 * importances)


def train_model(model, x):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(200):
        optimizer.zero_grad()
        x_preds = model(x)
        loss = importance_loss(x, x_preds)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item():.2f}")
    return model


def show_heatmap(W: t.Tensor, bias: t.Tensor, type: str) -> None:
    # W hidden_dim, dim
    # bias dim
    sup_matrix = W.T @ W  # dim, dim
    bias = bias.unsqueeze(1)  # 1, dim
    zeros = t.zeros_like(bias)  # 1, dim
    data = t.cat([sup_matrix, zeros, bias], dim=1)  # dim, dim + 2
    fig = make_subplots(
        rows=1,
        cols=1,
        # row_titles= ...
    )
    fig.add_trace(
        px.imshow(
            data.detach().numpy(),
            # title=f"W.T @ W for {type} model, S = {SPARSITY}",
        ).data[0],
        row=1,
        col=1,
    )
    fig.show()


# Linear Model
model = LinearModel()
model = train_model(model, x)
W = model.W
show_heatmap(W, model.bias, "linear")

# ReLU Model
model = ReLUModel()
model = train_model(model, x)
W = model.linear_model.W
show_heatmap(W, model.linear_model.bias, "relu")

# fig = px.imshow(
#     model.linear_model.bias.unsqueeze(0).detach().numpy(),
#     title=f"Bias for ReLU model, S = {SPARSITY}",
# )
# fig.show()
