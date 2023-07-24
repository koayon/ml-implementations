import collections
from typing import Any, Optional, OrderedDict, Tuple, Union

import torch as t
from einops import rearrange, repeat
from expert_choice_layer import ExpertChoiceFFN
from fancy_einsum import einsum
from torch import nn


class SparseMoETransformer(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int = 2,
        hidden_size: int = 16,
        # moe_block: nn.Module,
        # transformer_block: nn.Module = nn.TransformerDecoderLayer(d_model = hidden_size, nhead = 4),
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

        transformer_block = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4)

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
            x, cache = layer(x, cache)
        z = self.final_norm(x)

        return z, cache

    def generate(self, input: str) -> str:
        raise NotImplementedError


# TODO: Add in interpretability piece - we want to know which tokens where routed where
# TODO: Add in training/optimisation loop
# TODO: Add activation/attention caching as well as router caching? - a question of hooks essentially, can add these in later.
# TODO: Complete generate function


def main():
    x = t.randn(2, 3, 16)
    print(x)
    print(x.shape)
    model = SparseMoETransformer()
    y, cache = model(x)
    return y, cache


if __name__ == "__main__":
    main()
