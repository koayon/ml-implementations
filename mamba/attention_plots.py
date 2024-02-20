import os

"""
For the branch with MambaInterp
    pip install git+https://github.com/JadenFiotto-Kaufman/nnsight.git@mambainterp
pip3 install torch --upgrade
pip install causal-conv1d>=1.1.0
pip install mamba-ssm
pip install -U kaleido
pip install loguru
pip install jaxtyping
"""

import argparse

import plotly.express as px
import torch as t
from einops import einsum, rearrange
from jaxtyping import Float
from loguru import logger
from nnsight import NNsightModel
from nnsight.contexts.DirectInvoker import DirectInvoker
from nnsight.models.Mamba import MambaInterp, MambaModuleInterp
from plotly.graph_objs._figure import Figure
from sklearn.decomposition import NMF
from transformers import AutoTokenizer


def mean_abs_aggregation(
    head_values: Float[t.Tensor, "layer source target input_dim"]
) -> Float[t.Tensor, "layer source target"]:
    return t.mean(head_values.abs(), dim=-1)


def nmf_aggregation(
    head_values: Float[t.Tensor, "layer source target input_dim"]
) -> Float[t.Tensor, "component source target"]:
    NUM_COMPONENTS = 16
    nmf_model = NMF(n_components=NUM_COMPONENTS, init="random", random_state=0)

    attn_values = rearrange(
        head_values, "layer source target input_dim -> source target (layer input_dim)"
    )
    components = nmf_model.fit_transform(attn_values)  # source target component

    out = rearrange(components, "source target component -> component source target")
    return t.tensor(out)


AGGREGATION_METHOD = {"mean_abs": mean_abs_aggregation, "nmf": nmf_aggregation}


def linear_normalisation(values: t.Tensor) -> t.Tensor:
    values = t.clip(values, 0)
    return values / values.sum(dim=2, keepdim=True)


def get_layer_values(
    prompt: str, model: MambaInterp
) -> tuple[DirectInvoker, Float[t.Tensor, "layer source target input_dim"]]:
    with model.invoke(prompt, fwd_args={"inference": True}) as invoker:
        layer_values = []

        for layer in model.backbone.layers:
            mixer: MambaModuleInterp = layer.mixer
            x, _delta, _A, _B, C, _ = mixer.ssm.input[0]

            x: Float[t.Tensor, "batch input_dim seq_len"]

            _delta: Float[t.Tensor, "batch input_dim seq_len"]

            _A: Float[t.Tensor, "input_dim state_dim"]
            _B: Float[t.Tensor, "batch state_dim seq_len"]
            C: Float[t.Tensor, "batch state_dim seq_len"]

            discA: Float[t.Tensor, "batch input_dim seq_len state_dim"]
            discB: Float[t.Tensor, "batch input_dim seq_len state_dim"]

            discA: t.Tensor = mixer.ssm.discA.output
            discB: t.Tensor = mixer.ssm.discB.output

            batch, input_dim, _seq_len, _state_dim = discA.shape

            source_values = []

            # Iterate through target tokens.
            for target_token_idx in range(x.shape[2]):
                target_values = []

                # Iterate through source tokens.
                for source_token_idx in range(x.shape[2]):
                    # If source is after target token, it can't "see" it ofc.
                    if target_token_idx < source_token_idx:
                        attention_heads_tensor = -float("inf") * t.ones(
                            (batch, input_dim)
                        )
                    else:
                        discB_source = discB[
                            :, :, source_token_idx, :
                        ]  # batch, input_dim, state_dim
                        C_target = C[:, :, target_token_idx]  # batch, state_dim

                        # Multiply together all As between source and target.
                        discA_multistep = t.prod(
                            discA[
                                :, :, source_token_idx + 1 : target_token_idx + 1, :
                            ],  # TODO: Check off by one error?
                            dim=2,
                        )  # batch, input_dim, state_dim

                        # Apply the multistep A to the B from source.
                        discAB = (
                            discA_multistep * discB_source
                        )  # batch, input_dim, state_dim

                        # Apply C from target.
                        # This sums over the state_dim dimension.
                        attention_heads_tensor = einsum(
                            C_target,
                            discAB,
                            "batch state_dim, batch input_dim state_dim -> batch input_dim",
                        )  # batch, input_dim

                    target_values.append(
                        attention_heads_tensor
                    )  # list[batch, input_dim]

                source_values.append(target_values)  # list[list[batch, input_dim]]

            layer_values.append(source_values)  # list[list[list[batch, input_dim]]]

            full_attention_tensor = t.tensor(
                layer_values
            )  # layer, source, target, batch, input_dim
            output = t.squeeze(
                full_attention_tensor, dim=3
            )  # layer, source, target, input_dim

    return invoker, output


def visualise_attn_patterns(
    values: t.Tensor, token_labels: list[str], name: str, out_path: str
) -> None:
    fig: Figure = px.imshow(
        values.T.flip(0, 1),
        # values,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": "Source Token", "y": "Target Token"},
        x=token_labels,
        y=token_labels,
        title=name,
    )

    fig.write_image(os.path.join(out_path, f"{name}.png"))


def get_token_labels(model: NNsightModel, invoker: DirectInvoker) -> list[str]:
    clean_tokens = [
        model.tokenizer.decode(token) for token in invoker.input["input_ids"][0]
    ]
    token_labels = [f"{token}_{index}" for index, token in enumerate(clean_tokens)]
    return token_labels


def main(
    prompt: str = "The capital of France is Paris",
    out_path: str = "./attn",
    repo_id: str = "state-spaces/mamba-130m",
    aggregation: str = "mean_abs",
    normalisation: str = "linear",
):
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b", padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = MambaInterp(repo_id, device="cuda", tokenizer=tokenizer)

    invoker, full_attention_tensor = get_layer_values(prompt, model)

    # Aggregate attention heads
    grouping = "layer"

    if aggregation in AGGREGATION_METHOD.values():
        aggregation_fn = AGGREGATION_METHOD[aggregation]
        if aggregation == "nmf":
            grouping = "component"
    else:
        print("Invalid aggregation method, using mean_abs instead")
        aggregation_fn = mean_abs_aggregation

    values = aggregation_fn(full_attention_tensor)  # layer|component, source, target

    # Normalise row-wise along the target token dimension
    if normalisation == "linear":
        values = linear_normalisation(values)
    elif normalisation == "softmax":
        values = values.softmax(dim=2)
    else:
        print("Invalid normalisation method, using linear instead")
        values = linear_normalisation(values)

    token_labels = get_token_labels(model, invoker)

    os.makedirs(out_path, exist_ok=True)

    for layer_idx, layer_attn_matrix in enumerate(values):
        visualise_attn_patterns(
            layer_attn_matrix, token_labels, f"{grouping}_{layer_idx}", out_path
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("prompt")

    main(**vars(parser.parse_args()))
