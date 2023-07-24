import collections
from typing import Any, Optional, OrderedDict, Tuple, Union

import numpy as np
import plotly.express as px
import torch as t
from einops import rearrange, repeat
from fancy_einsum import einsum
from torch import nn

from gpt2.transformer_block import GPT2Block
from switch_transformer.expert_choice_layer import ExpertChoiceFFN


class SparseMoETransformer(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int = 16,
        hidden_size: int = 16,
        # moe_block: nn.Module,
        # transformer_block: nn.Module = GPT2Block()
        attn_dropout: float = 0.1,
        expert_dropout: float = 0.4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.expert = moe_block
        self.attn_dropout = attn_dropout
        self.expert_dropout = expert_dropout
        self.final_norm = nn.LayerNorm([hidden_size])

        transformer_block: nn.Module = GPT2Block(hidden_size=hidden_size)

        layers: OrderedDict[str, nn.Module] = collections.OrderedDict()
        for i in range(num_layers):
            if i % 2 == 0:
                layers[f"moe_block{i}"] = ExpertChoiceFFN(layer_id=f"expert_layer_{i}")
            else:
                layers[f"transformer_block{i}"] = transformer_block

        self.layers = layers

    def forward(
        self, x: t.Tensor
    ) -> Tuple[t.Tensor, Optional[collections.OrderedDict]]:
        """
        x: batch seq hidden_size (has already been tokenised)
        """
        cache: OrderedDict[str, t.Tensor] = collections.OrderedDict()
        for idx, layer in self.layers.items():
            if idx.startswith("moe"):
                x, cache[idx] = layer(x, cache)
            else:
                x = layer(x)
        z = self.final_norm(x)

        print(z)
        print(cache)

        return z, cache

    def generate(self, input: str) -> str:
        raise NotImplementedError
        # Tokenise
        # Embed
        # Forward auto-regressively until stop token


# TODO: Complete generate function
# TODO: Add G to weight how much the paths show up on the plot
# TODO: Add in training/optimisation loop
# TODO: Add activation/attention caching as well as router caching? - a question of hooks essentially, can add these in later.


def token_path(cache: OrderedDict[str, t.Tensor], token_num: int) -> dict:
    """
    cache: OrderedDict[str, t.Tensor]
    """

    filtered_cache = {k: v for k, v in cache.items() if k.startswith("expert_layer_")}

    out = dict()

    print(f"{filtered_cache = }")
    print(len(filtered_cache))
    for layer, token_assignments in filtered_cache.items():
        print(layer, token_assignments)
        bool_token_assignment = token_assignments == token_num
        print(bool_token_assignment)
        out[layer] = bool_token_assignment.max(dim=0)[0].numpy()

    array = np.array(list(out.values()))
    num_experts = array.shape[1]

    fig = px.imshow(
        array,
        x=[f"expert_{i}" for i in range(num_experts)],
        y=list(out.keys()),
        title=f"Token {token_num} path",
    )
    fig.show()

    return out


def main():
    x = t.randn(1, 8, 16)  # batch seq hidden_size
    print(x)
    print(x.shape)
    model = SparseMoETransformer()
    with t.inference_mode():
        with t.no_grad():
            y, cache = model(x)
    token_0_path = token_path(cache=cache, token_num=0)
    return y, cache, token_0_path


if __name__ == "__main__":
    main()
    # cache = OrderedDict(
    #     [
    #         (str("expert_layer_0"), t.Tensor([[7, 2, 2, 0], [4, 4, 1, 5]])),
    #         (str("expert_layer_2"), t.Tensor([[7, 5, 1, 4], [2, 1, 0, 7]])),
    #     ]
    # )
    # token_2_path = token_path(cache=cache, token_num=2)
    # print(token_2_path)
    # plot_token_path()
