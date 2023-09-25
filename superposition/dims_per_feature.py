import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from einops import einsum

from superposition.dims_funcs import feature_dimensionalities_to_geometries, get_metrics
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

NUM_SAMPLE_SPARSITIES = 200


d_values = []
frob_norms = []
sparsities = []
avg_feature_dimensionalities = []
feature_dimensionalities_list = []


prev_loss = 12

for current_sparsity in np.linspace(0, 0.9, num=NUM_SAMPLE_SPARSITIES):
    x = t.rand(BATCH_SIZE, DIM)

    sparsity_mask = t.abs(t.rand(BATCH_SIZE, DIM)) > current_sparsity
    x = x * sparsity_mask

    model = ReLUModel(DIM, HIDDEN_DIM)
    print(f"Current sparsity: {current_sparsity}, {model}")

    trained_model, loss = train_model(model, x, importances=importances, verbose=False)
    W = trained_model.W

    if loss > 12 or loss > (prev_loss * 1.5):
        print("Loss too high, skipping")
        print(f"Sparsity: {current_sparsity}, Loss: {loss}")
        continue
    prev_loss = loss

    frobenius_norm, feature_dimensionality_tensor = get_metrics(W)

    avg_feature_dimensionality = t.mean(feature_dimensionality_tensor)

    frob_norms.append(frobenius_norm.item())
    sparsities.append(current_sparsity)
    feature_dimensionalities_list.append(feature_dimensionality_tensor)
    avg_feature_dimensionalities.append(avg_feature_dimensionality.item())

bucketed_feature_dimensionalities = feature_dimensionalities_to_geometries(
    feature_dimensionalities_list=feature_dimensionalities_list
)

# print(frob_norms)
# print(sparsities)

# Build dataframe
df = pd.DataFrame(bucketed_feature_dimensionalities)
df["frob_norm"] = frob_norms
df["sparsity"] = sparsities
df["avg_feature_dimensionality"] = avg_feature_dimensionalities

df["dimensions_per_feature"] = HIDDEN_DIM / df["frob_norm"] ** 2
df["1/(1-s)"] = 1 / (1 - df["sparsity"])

df.set_index("sparsity", inplace=True)

# print(df)


for col in df.columns:
    if col in ("sparsity", "frob_norm", "1/(1-s)"):
        continue
    df[col] = df.rolling(window=10).mean()[col]

# print(df)

fig = px.line(
    df,
    x="1/(1-s)",
    y=[
        "dimensions_per_feature",
        "avg_feature_dimensionality",
        "no_superposition",
        "tetraheadron",
        "triangle",
        "digon",
        "pentagon",
        "square_antiprism",
        "other",
        "not_represented",
    ],
    log_x=True,
)
fig.show()

# Create table
fig = go.Figure(
    data=[
        go.Table(
            header=dict(values=list(df.columns)),
            cells=dict(values=df.T.values.tolist()),
        )
    ]
)
fig.show()
