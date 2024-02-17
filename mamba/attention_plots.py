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

import plotly.express as px
import torch
import torch as t
from einops import einsum
from jaxtyping import Float
from loguru import logger
from nnsight import util
from nnsight.models.Mamba import MambaInterp
from nnsight.tracing.Proxy import Proxy
from transformers import AutoTokenizer


def main(
    prompt: str = "The capital of France is Paris",
    out_path: str = "./attn",
    repo_id: str = "state-spaces/mamba-130m",
    use_absolute_value: bool = True,
    use_softmax: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b", padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = MambaInterp(repo_id, device="cuda", tokenizer=tokenizer)

    with model.invoke(prompt, fwd_args={"inference": True}) as invoker:
        layer_values = []

        for layer in model.backbone.layers:
            x, _delta, _A, _B, C, _ = layer.mixer.ssm.input[0]

            x: Float[t.Tensor, "batch input_dim seq_len"]

            _delta: Float[t.Tensor, "batch input_dim seq_len"]

            _A: Float[t.Tensor, "input_dim state_dim"]
            _B: Float[t.Tensor, "batch state_dim seq_len"]
            C: Float[t.Tensor, "batch state_dim seq_len"]

            discA: Float[t.Tensor, "batch input_dim seq_len state_dim"]
            discB: Float[t.Tensor, "batch input_dim seq_len state_dim"]

            discA: t.Tensor = layer.mixer.ssm.discA.output.save()
            discB: t.Tensor = layer.mixer.ssm.discB.output.save()

            source_values = []

            # Iterate through target tokens.
            for target_token_idx in range(x.shape[2]):
                target_values = []

                # Iterate through source tokens.
                for source_token_idx in range(x.shape[2]):
                    # If source is after target token, it can't "see" it ofc.
                    if target_token_idx < source_token_idx:
                        output_value = -float("inf")
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
                        # This sums over all 'd', but we might want a separate attention map per d.
                        head_values = einsum(
                            C_target,
                            discAB,
                            "batch state_dim, batch input_dim state_dim -> batch input_dim",
                        )  # batch, input_dim

                        aggregated_head_values: t.Tensor = t.mean(
                            head_values.abs(), dim=1
                        )  # batch
                        output_value: float = (
                            aggregated_head_values.item().save()  # type: ignore
                        )

                    target_values.append(output_value)

                source_values.append(target_values)

            layer_values.append(source_values)

    # Convert to values and combine to one tensor (n_layer, n_tokens, n_tokens)

    def post(proxy):

        value = proxy.value

        if use_absolute_value:

            value = abs(value)

        return value

    values = util.apply(layer_values, post, Proxy)
    values = torch.tensor(values)  # layer, source, target

    # Normalise row-wise along the target token dimension
    # values = values / values.sum(dim=1, keepdim=True)

    # print(values[0])
    # assert False

    if use_softmax:
        values = values.softmax(dim=2)

    clean_tokens = [
        model.tokenizer.decode(token) for token in invoker.input["input_ids"][0]
    ]
    token_labels = [f"{token}_{index}" for index, token in enumerate(clean_tokens)]

    def vis(values: t.Tensor, token_labels, name, out_path):
        fig = px.imshow(
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

    os.makedirs(out_path, exist_ok=True)

    for layer_idx in range(values.shape[0]):
        vis(values[layer_idx], token_labels, f"layer_{layer_idx}", out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("prompt")

    main(**vars(parser.parse_args()))
