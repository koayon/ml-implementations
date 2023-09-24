from typing import Tuple

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from superposition.models import Model


def train_model(
    model: Model,
    x: t.Tensor,
    importances: t.Tensor,
    verbose: bool = True,
    max_steps=200,
) -> Tuple[Model, float]:
    """Run training loop for model"""

    def importance_loss(x, x_preds, importances=importances):
        return t.sum((x - x_preds) ** 2 * importances)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss = t.empty(1)
    for epoch in range(max_steps):
        optimizer.zero_grad()
        x_preds = model(x)
        loss = importance_loss(x, x_preds)
        loss.backward()
        optimizer.step()
        if verbose or epoch == max_steps - 1:
            print(f"Epoch {epoch} Loss: {loss.item():.2f}")
    return model, loss.item()
