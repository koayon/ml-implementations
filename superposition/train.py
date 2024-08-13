from typing import Tuple

import torch as t
import torch.optim as optim
from tqdm import tqdm

from superposition.models import Model


def train_model(
    model: Model,
    x_BD: t.Tensor,
    importances: t.Tensor,
    verbose: bool = True,
    max_steps=200,
    abs_loss: bool = False,
) -> Tuple[Model, float]:
    """Run training loop for model"""

    def importance_loss(x_BD, x_preds_BD, importances=importances):
        # print((x_BD - x_preds_BD) ** 2 * importances)
        out_B = t.sum((x_BD - x_preds_BD) ** 2 * importances, dim=-1)
        return t.mean(out_B)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss = t.empty(1)
    for epoch in tqdm(range(max_steps)):
        optimizer.zero_grad()
        x_preds_BD = model(x_BD)
        # print(x_preds_BD)
        if abs_loss:
            x_BD = t.abs(x_BD)
        loss = importance_loss(x_BD, x_preds_BD)
        # print(loss)
        loss.backward()
        optimizer.step()
        if verbose or epoch == max_steps - 1:
            print(f"Epoch {epoch} Loss: {loss.item():.2f}")
    return model, loss.item()
