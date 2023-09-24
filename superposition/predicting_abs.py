import torch as t

from superposition.models import MLP
from superposition.plotting import show_heatmap
from superposition.train import train_model

# Set consts
SPARSITY = 0.3  # feature_importance = 1 - SPARSITY
DIM = 10
HIDDEN_DIM = 4

BATCH_SIZE = 2000
importances = t.tensor([0.5**i for i in range(1, DIM + 1)])

# Define synthetic data, x
x_base = t.rand(1, DIM)
x = x_base + t.randn(BATCH_SIZE, DIM) * 0.5
sparsity_mask = t.rand(BATCH_SIZE, DIM) > SPARSITY
x = x * sparsity_mask


# HiddenReLUModel
# Note that with this model we force the basis vector to be the priveleged basis vector so we can interpret W directly
model: MLP = MLP(dim=DIM, hidden_dim=HIDDEN_DIM)
model, _loss = train_model(model, x, importances=importances, max_steps=1000, abs_loss=True)  # type: ignore
show_heatmap(
    W=model.W1,
    bias=model.bias,
    use_sup=False,
    model_type="W1: hidden_relu",
    sparsity=SPARSITY,
    plot_bias=False,
)

show_heatmap(
    W=model.W2,
    bias=model.bias,
    use_sup=False,
    model_type="W2: hidden_relu",
    sparsity=SPARSITY,
    plot_bias=True,
)
