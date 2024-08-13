import torch as t


def num_dims_within_2_perc(
    input_tensor_SD: t.Tensor, targets: dict[str, float], tolerance: float = 0.02
) -> dict[str, t.Tensor]:
    """Return the number of dimensions within 2% of target"""
    out = {}
    lower_bound = 1 - tolerance
    upper_bound = 1 + tolerance

    for target_name, target_val in targets.items():
        close_counts_S = t.sum(
            (input_tensor_SD > target_val * lower_bound)
            & (input_tensor_SD < target_val * upper_bound),
            dim=-1,
        )
        out[target_name] = close_counts_S
    return out


def feature_dimensionalities_to_geometries(
    feature_dimensionalities_list: list[t.Tensor],
) -> dict[str, list[int]]:
    """
    Turns feature dimensionalities into geometries counts

    From paper;
    #  No superposition if f = 1
    #  Tetrahedron if f = 3/4
    #  Triangle if f = 2/3
    #  Digon/Antipodal pair if f = 1/2
    #  Pentagon if f = 2/5
    #  Square antiprism if f = 3/8
    """

    feature_dimensionalities_SD = t.stack(
        feature_dimensionalities_list, dim=0
    )  # num_sparsities, hidden_dim

    # All num_sparsities length
    no_superposition = t.sum(feature_dimensionalities_SD > 0.8, dim=1).tolist()
    tetraheadron = t.sum(
        (0.7 < feature_dimensionalities_SD) & (feature_dimensionalities_SD < 0.8), dim=1
    ).tolist()  # 3/4
    triangle = t.sum(
        (0.6 < feature_dimensionalities_SD) & (feature_dimensionalities_SD < 0.7), dim=1
    ).tolist()  # 2/3
    digon = t.sum(
        (0.46 < feature_dimensionalities_SD) & (feature_dimensionalities_SD < 0.6), dim=1
    ).tolist()  # 1/2
    four_ninths = t.sum(
        (0.43 < feature_dimensionalities_SD) & (feature_dimensionalities_SD < 0.46), dim=1
    ).tolist()  # 4/9
    pentagon = t.sum(
        (0.39 < feature_dimensionalities_SD) & (feature_dimensionalities_SD < 0.43), dim=1
    ).tolist()  # 2/5
    square_antiprism = t.sum(
        (0.35 < feature_dimensionalities_SD) & (feature_dimensionalities_SD < 0.39), dim=1
    ).tolist()  # 3/8
    other = t.sum(
        (0.1 < feature_dimensionalities_SD) & (feature_dimensionalities_SD < 0.35), dim=1
    ).tolist()
    not_represented = t.sum(feature_dimensionalities_SD < 0.1, dim=1).tolist()

    return {
        "no_superposition": no_superposition,
        "tetraheadron": tetraheadron,
        "triangle": triangle,
        "digon": digon,
        "four_ninths": four_ninths,
        "pentagon": pentagon,
        "square_antiprism": square_antiprism,
        "other": other,
        "not_represented": not_represented,
    }


def get_metrics(W: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Get Frobenius norm and feature dimensionality tensor from W, the weight matrix

    Parameters
    ----------
    W : t.Tensor
        hidden_dim, dim

    Returns
    ----------
    frobenius_norm : t.Tensor
        scalar
    feature_dimensionality_tensor : t.Tensor
        hidden_dim
    """

    # Get the frobenius norm of W
    frobenius_norm = t.norm(W, p="fro")  # scalar

    # Normalise each column of W
    W_hat = W / t.norm(W, dim=0, keepdim=True)  # hidden_dim, dim

    num = t.norm(W, dim=1, p=2)  # hidden_dim
    denom = t.sum((W_hat @ W.T) ** 2, dim=1)  # hidden_dim

    feature_dimensionality_tensor = num / denom  # hidden_dim

    return frobenius_norm, feature_dimensionality_tensor
