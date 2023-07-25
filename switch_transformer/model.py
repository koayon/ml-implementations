import collections
from typing import Any, Optional, OrderedDict, Tuple, Union

import numpy as np
import plotly.express as px
import tiktoken
import torch as t
from einops import rearrange, repeat
from fancy_einsum import einsum
from torch import nn

from gpt2.transformer_block import GPT2Block
from switch_transformer.expert_choice_layer import ExpertChoiceFFN


class SparseMoETransformer(nn.Module):
    token_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    transformer_block: nn.Module
    moe_block: nn.Module
    vocab_size: int

    def __init__(
        self,
        *,
        num_layers: int = 16,
        hidden_size: int = 16,
        attn_dropout: float = 0.1,
        expert_dropout: float = 0.4,
        max_position_embeddings: int = 1024,
        vocab_size: int = 50257,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.expert = moe_block
        self.attn_dropout = attn_dropout
        self.expert_dropout = expert_dropout
        self.final_norm = nn.LayerNorm([hidden_size])

        layers: OrderedDict[str, nn.Module] = collections.OrderedDict()
        for i in range(num_layers):
            if i % 2 == 0:
                layers[f"moe_block{i}"] = ExpertChoiceFFN(
                    layer_id=f"expert_layer_{i}",
                    hidden_size=hidden_size,
                    dropout=expert_dropout,
                )
            else:
                layers[f"transformer_block{i}"] = GPT2Block(hidden_size=hidden_size)

        self.layers = layers

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(
        self, x: t.Tensor
    ) -> Tuple[t.Tensor, Optional[collections.OrderedDict]]:
        """
        x: batch seq_length hidden_size (has already been tokenised)
        """
        # Get position of tokens
        seq_length = x.shape[1]
        pos = t.arange(0, seq_length).to(x.device)

        # Combine token and positional embeddings
        x = self.token_embedding(x) + self.pos_embedding(pos)
        # print("x.shape", x.shape)

        # Initialise cache for routing and use for MoE layers
        cache: OrderedDict[str, t.Tensor] = collections.OrderedDict()
        for idx, layer in self.layers.items():
            if idx.startswith("moe"):
                x, cache[idx] = layer(x, cache)
            else:
                x = layer(x)
        z = self.final_norm(x)

        # Unembed to get logits for each token
        out = einsum(
            "b s h, v h -> b s v", z, self.token_embedding.weight
        )  # batch seq vocab_size

        # print(z)
        # print(cache)

        return out, cache


def sample_next_token(input: str, model: nn.Module) -> str:
    # Embed
    # Forward auto-regressively until stop token

    # Tokenise input
    tokenizer = tiktoken.encoding_for_model("gpt2")
    tokens_list = tokenizer.encode(input)
    tokens = t.Tensor(tokens_list).long().unsqueeze(0)  # batch seq
    print(tokens)
    print(len(tokens))

    # Forward pass tokens
    with t.inference_mode():
        with t.no_grad():
            all_logits, _cache = model(t.Tensor(tokens))  # batch seq vocab_size

    # Here we're looking at the next token for the first batch
    logits = all_logits[0, -1, :]  # vocab_size

    # Sample from logits (basic categorical sampling)
    sampled_token = (
        t.distributions.categorical.Categorical(logits=logits).sample().item()
    )
    assert isinstance(sampled_token, int)

    return tokenizer.decode([sampled_token])


# TODO: Add G to weight how much the paths show up on the plot
# TODO: Add in training/optimisation loop
# TODO: Add activation/attention caching as well as router caching? - a question of hooks essentially, can add these in later.
# TODO: Extract variables to config
#


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
    model = SparseMoETransformer()
    x = t.randint(low=0, high=50000, size=(1, 16))
    y, _cache = model(x)
    # y = sample_next_token(model=model, input="Hello")
    # token_0_path = token_path(cache=cache, token_num=0)
    print("cache")
    print(_cache)
    return y, _cache


if __name__ == "__main__":
    main()
