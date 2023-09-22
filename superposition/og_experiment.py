import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import einsum

SPARSITY = 0.11
DIM = 25
HIDDEN_DIM = 6

BATCH_SIZE = 500
importances = t.abs(t.randn(DIM))

x = t.randn(BATCH_SIZE, DIM)

sparsity_mask = t.rand(DIM) < SPARSITY
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


def importance_loss(x, x_preds, importances):
    return t.sum((x - x_preds) ** 2 * importances)


# Training Loop
model = ReLUModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = importance_loss

for epoch in range(1000):
    optimizer.zero_grad()
    x_preds = model(x)
    loss = criterion(x, x_preds, importances)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item():.2f}")

W = model.linear_model.W
print(W.T @ W)
