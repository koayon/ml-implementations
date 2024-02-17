import os

"""
For the branch with MambaInterp
pip install git+https://github.com/JadenFiotto-Kaufman/nnsight.git@mambainterp
"""

import plotly.express as px
import torch
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
            x = layer.mixer.ssm.input[0][0].save()

            C = layer.mixer.ssm.input[0][4].save()

            discA = layer.mixer.ssm.discA.output.save()
            discB = layer.mixer.ssm.discB.output.save()

            # Compute Bx for all tokens before.
            Bx = torch.einsum("bdln,bdl->bdln", discB, x)

            source_values = []

            # Iterate through target tokens.
            for target_token_idx in range(x.shape[2]):
                target_values = []

                # Iterate through source tokens.
                for source_token_idx in range(x.shape[2]):
                    # If source is after target token, it can't "see" it ofc.
                    if target_token_idx < source_token_idx:
                        value = -float("inf")
                    else:
                        # Multiply together all As between source and target.
                        discA_multistep = torch.prod(
                            discA[:, :, source_token_idx + 1 : target_token_idx + 1],
                            dim=2,
                        )

                        # Apply the multistep A to the Bx from source.
                        discABx = discA_multistep * Bx[:, :, source_token_idx]

                        # Apply C from target.
                        # This sums over all 'd', but we might want a separate attention map per d.
                        value = (
                            torch.einsum(
                                "bdn,bn->b", discABx, C[:, :, target_token_idx]
                            )
                            .item()
                            .save()  # type: ignore
                        )

                    target_values.append(value)

                source_values.append(target_values)

            layer_values.append(source_values)

    # Convert to values and combine to one tensor (n_layer, n_tokens, n_tokens)

    def post(proxy):

        value = proxy.value

        if use_absolute_value:

            value = abs(value)

        return value

    values = util.apply(layer_values, post, Proxy)
    values = torch.tensor(values)

    if use_softmax:
        values = values.softmax(dim=2)

    clean_tokens = [
        model.tokenizer.decode(token) for token in invoker.input["input_ids"][0]
    ]
    token_labels = [f"{token}_{index}" for index, token in enumerate(clean_tokens)]

    def vis(values, token_labels, name, out_path):
        fig = px.imshow(
            values,
            color_continuous_midpoint=0.0,
            color_continuous_scale="RdBu",
            labels={"y": "Target Token", "x": "Source Token"},
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
    parser.add_argument("--out_path", default="./attn")
    parser.add_argument("--repo_id", default="state-spaces/mamba-130m")
    parser.add_argument("--absv", action="store_true", default=False)
    parser.add_argument("--softmax", action="store_true", default=False)

    main(**vars(parser.parse_args()))
