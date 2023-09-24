import numpy as np
import pandas as pd
import plotly.express as px
import torch as t

from superposition.models import ReLUModel
from superposition.og_experiment import train_model
from superposition.train import train_model

# SPARSITY = 0.7
DIM = 80
HIDDEN_DIM = 20

BATCH_SIZE = 2000
# NUM_IMPORTANT_FEATURES = 4
# importances = t.tensor(
#     [2.0] * NUM_IMPORTANT_FEATURES + [0.2] * (DIM - NUM_IMPORTANT_FEATURES)
# )
importances = t.tensor([0.7**i for i in range(1, DIM + 1)])

d_values = []
frob_norms = []
sparsities = []

for current_sparsity in np.linspace(0, 0.9, num=200):
    x = t.rand(BATCH_SIZE, DIM)

    sparsity_mask = t.abs(t.rand(BATCH_SIZE, DIM)) > current_sparsity
    x = x * sparsity_mask

    model = ReLUModel(DIM, HIDDEN_DIM)
    print(f"Current sparsity: {current_sparsity}, {model}")
    model2, loss = train_model(model, x, importances=importances, verbose=False)
    # print(id(model2))
    W = model2.W
    # show_heatmap(W)
    if loss > 12:
        print("Loss too high, skipping")
        print(f"Sparsity: {current_sparsity}, Loss: {loss}")
        continue

    # Get the frobenius norm of W
    frobenius_norm = t.norm(W, p="fro")

    # d_star = HIDDEN_DIM / frobenius_norm.item() ** 2

    # d_values.append(d_star)
    frob_norms.append(frobenius_norm.item())
    sparsities.append(current_sparsity)

df = pd.DataFrame({"sparsity": sparsities, "frob_norm": frob_norms})
df["dimensions_per_feature"] = HIDDEN_DIM / df["frob_norm"] ** 2
df["1/(1-s)"] = 1 / (1 - df["sparsity"])

print(df)

fig = px.line(df, x="1/(1-s)", y="dimensions_per_feature", log_x=True)
fig.show()
