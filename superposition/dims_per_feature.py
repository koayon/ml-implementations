import numpy as np
import pandas as pd
import plotly.express as px
import torch as t
from einops import rearrange, repeat
from tqdm import tqdm

from superposition.dims_funcs import feature_dimensionalities_to_geometries, get_metrics
from superposition.models import ReLUModel
from superposition.og_experiment import train_model
from superposition.train import train_model

# SPARSITY = 0.7
DIM = 400
HIDDEN_DIM = 30

BATCH_SIZE = 2000
# NUM_IMPORTANT_FEATURES = 4
# importances = t.tensor(
#     [2.0] * NUM_IMPORTANT_FEATURES + [0.2] * (DIM - NUM_IMPORTANT_FEATURES)
# )
# importances = t.tensor([0.7**i for i in range(1, DIM + 1)])
importances = t.tensor([1.0] * DIM)

NUM_SAMPLE_SPARSITIES = 100
# NUM_MODELS_PER_SPARSITY = 1


# d_values = []
frob_norms: list[float] = []
sparsities: list[float] = []
avg_feature_dimensionalities: list[float] = []
feature_dimensionalities_list: list[t.Tensor] = []


prev_loss = initial_prev_loss = 50.0

for current_sparsity in tqdm(np.linspace(0, 0.9, num=NUM_SAMPLE_SPARSITIES)):
    # for _ in range(NUM_MODELS_PER_SPARSITY):
    x = t.rand(BATCH_SIZE, DIM)

    sparsity_mask = t.abs(t.rand(BATCH_SIZE, DIM)) > current_sparsity
    x = x * sparsity_mask

    model = ReLUModel(DIM, HIDDEN_DIM)
    print(f"Current sparsity: {current_sparsity}, {model}")

    trained_model, loss = train_model(
        model, x, importances=importances, verbose=False, max_steps=5_000
    )
    W = trained_model.W

    if loss > initial_prev_loss or loss > (prev_loss * 1.5):
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

feature_dimensionalities_SD = t.stack(
    feature_dimensionalities_list, dim=0
)  # num_sparsities, hidden_dim
feature_dimensionalities_Sd = rearrange(
    feature_dimensionalities_SD, "sparsities feature_dim -> (sparsities feature_dim)"
)
feature_dimensionalities_Sd_list = feature_dimensionalities_Sd.tolist()


def repeat_elements(l: list, repeat_count: int) -> list:
    return [element for element in l for _ in range(repeat_count)]


sparsities_Sd = repeat_elements(sparsities, HIDDEN_DIM)

print(len(sparsities_Sd))
print(len(feature_dimensionalities_Sd_list))
assert len(sparsities_Sd) == len(
    feature_dimensionalities_Sd_list
), f"{len(sparsities_Sd)} != {len(feature_dimensionalities_Sd_list)}"


raw_feature_dims_df = pd.DataFrame(
    data={
        "sparsity": sparsities_Sd,
        "feature_dimensionality": feature_dimensionalities_Sd_list,
    }
)
raw_feature_dims_df["1/(1-s)"] = 1 / (1 - raw_feature_dims_df["sparsity"])

# raw_feature_dims_df["feature_dimensionality"]

fig = px.scatter(
    raw_feature_dims_df,
    x="1/(1-s)",
    y="feature_dimensionality",
    log_x=True,
)
fig.show()

assert False

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
    if col in ("sparsity", "frob_norm"):
        continue
    df[col] = df.rolling(window=10).mean()[col]

print(df)

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
        "four_ninths",
        "pentagon",
        "square_antiprism",
        "other",
        "not_represented",
    ],
    log_x=True,
)
fig.show()

# fig = px.scatter(
#     df,
#     x="1/(1-s)",
#     y=[
#         "dimensions_per_feature",
#         "avg_feature_dimensionality",
#         "no_superposition",
#         "tetraheadron",
#         "triangle",
#         "digon",
#         "four_ninths",
#         "pentagon",
#         "square_antiprism",
#         "other",
#         "not_represented",
#     ],
#     log_x=True,
# )
# fig.show()

# # Create table
# fig = go.Figure(
#     data=[
#         go.Table(
#             header=dict(values=list(df.columns)),
#             cells=dict(values=df.T.values.tolist()),
#         )
#     ]
# )
# fig.show()
