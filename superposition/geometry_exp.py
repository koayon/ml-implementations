import pandas as pd
import plotly.express as px
import torch as t

from superposition.models import ReLUModel
from superposition.train import train_model

SPARSITY = 0.75
DIM = 6
HIDDEN_DIM = 2

BATCH_SIZE = 2000
NUM_IMPORTANT_FEATURES = DIM
# Constant importances
importances = t.tensor([1] * DIM)


# Generate correlated x features like in paper
x_base1 = t.rand(1, 2)
x_base2 = t.rand(1, 2)
x_base3 = t.rand(1, 2)

# Add noise to the correlated features
x_1_2 = x_base1 + t.rand(BATCH_SIZE, 2) * 0.1  # batch_size x 2
x_3_4 = x_base2 + t.rand(BATCH_SIZE, 2) * 0.1  # batch_size x 2
x_5_6 = x_base3 + t.rand(BATCH_SIZE, 2) * 0.1  # batch_size x 2

x = t.cat([x_1_2, x_3_4, x_5_6], dim=1)  # batch_size x 6
sparsity_mask = t.abs(t.rand(BATCH_SIZE, DIM)) > SPARSITY
x = x * sparsity_mask

# Train ReLU model
model = ReLUModel(dim=DIM, hidden_dim=HIDDEN_DIM)
model, _loss = train_model(model, x, importances=importances, verbose=False)

# Plot heatmap of weights in R^2
W = model.W.detach().numpy()
df = pd.DataFrame(
    {
        "x": W[0, :],
        "y": W[1, :],
        "color": ["group0", "group0", "group1", "group1", "group2", "group2"],
    }
)
df["size"] = (df.x**2 + df.y**2) ** (1 / 2) * 5
df["size"] = df["size"].clip(lower=0.5)
fig = px.scatter(df, x="x", y="y", color="color", size="size")
fig.show()

# We can see the features arrange themselves in weight space according to their correlation in the same way as in the paper.
# The arrangements vary with the sparsity present.
