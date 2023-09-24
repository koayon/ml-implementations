import torch as t

from superposition.models import LinearModel, Model, ReLUModel
from superposition.plotting import show_heatmap
from superposition.train import train_model

# Set consts
SPARSITY = 0.999  # feature_importance = 1 - SPARSITY
DIM = 25
HIDDEN_DIM = 6

BATCH_SIZE = 2000
importances = t.tensor([0.7**i for i in range(1, DIM + 1)])


def main():
    # Define synthetic data, x
    x = t.rand(BATCH_SIZE, DIM)
    sparsity_mask = t.rand(BATCH_SIZE, DIM) > SPARSITY
    x = x * sparsity_mask

    # Linear Model
    model = LinearModel(dim=DIM, hidden_dim=HIDDEN_DIM)
    model, _loss = train_model(model, x, importances=importances)
    show_heatmap(W=model.W, bias=model.bias, model_type="linear", sparsity=SPARSITY)

    # ReLU Model
    model = ReLUModel(dim=DIM, hidden_dim=HIDDEN_DIM)
    model, _loss = train_model(model, x, importances=importances)
    show_heatmap(W=model.W, bias=model.bias, model_type="relu", sparsity=SPARSITY)


if __name__ == "__main__":
    main()
