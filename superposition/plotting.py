import plotly.express as px
import torch as t
from plotly import graph_objects as go


def make_heatmap(
    W: t.Tensor,
    bias: t.Tensor,
    model_type: str,
    sparsity: float = 0.7,
    use_sup: bool = True,
    plot_bias: bool = True,
    plot_heatmap: bool = True,
) -> tuple[go.Figure, go.Figure]:
    """Plotting function for W.T @ W and bias

    Parameters
    ----------
    W : t.Tensor
        hidden_dim, dim
    bias : t.Tensor
        dim
    model_type : str
        _description_
    sparsity : float, optional
        _description_, by default SPARSITY
    """
    sup_matrix = W.T @ W  # dim, dim
    bias = bias.unsqueeze(1)  # 1, dim

    # Show W.T @ W
    fig1 = px.imshow(
        sup_matrix.detach().numpy() if use_sup else W.detach().numpy(),
        title=f"W.T @ W for {model_type} model, S = {sparsity}",
        color_continuous_scale="Portland",
        zmin=-1,
        zmax=1,
    )

    # Show bias
    fig2 = px.imshow(
        bias.detach().numpy(),
        title=f"Bias Matrix for {model_type} model, S = {sparsity}",
        color_continuous_scale="Portland",
        zmin=-1,
        zmax=1,
    )
    if plot_heatmap:
        fig1.show()
        if plot_bias:
            fig2.show()

    return fig1, fig2
