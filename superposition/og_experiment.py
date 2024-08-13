import torch as t

from superposition.models import LinearModel, Model, ReLUModel
from superposition.plotting import make_heatmap
from superposition.train import train_model

# Set consts
SPARSITY = 0.999  # feature_importance = 1 - SPARSITY
DIM = 25
HIDDEN_DIM = 6

BATCH_SIZE = 2000
importances = t.tensor([0.7**i for i in range(1, DIM + 1)])


def main():
    # Define synthetic data, x
    x_BD = t.rand(BATCH_SIZE, DIM)
    sparsity_mask_BD = t.rand(BATCH_SIZE, DIM) > SPARSITY
    x_BD = x_BD * sparsity_mask_BD

    print(x_BD)
    # assert False

    # Linear Model
    model = LinearModel(dim=DIM, hidden_dim=HIDDEN_DIM)
    model, _loss = train_model(model, x_BD, importances=importances)
    make_heatmap(
        W=model.W, bias=model.bias, model_type="linear", sparsity=SPARSITY, plot_heatmap=False
    )

    # ReLU Model
    model = ReLUModel(dim=DIM, hidden_dim=HIDDEN_DIM)
    model, _loss = train_model(model, x_BD, importances=importances)
    weights_fig, bias_fig = make_heatmap(
        W=model.W, bias=model.bias, model_type="relu", sparsity=SPARSITY, plot_heatmap=False
    )

    base_dir = "superposition/figures/"
    weights_fig.write_html(base_dir + "weights_fig.html")
    bias_fig.write_html(base_dir + "bias_fig.html")


if __name__ == "__main__":
    t.manual_seed(42)
    main()
