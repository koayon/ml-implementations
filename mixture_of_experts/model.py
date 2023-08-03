import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, OrderedDict, Protocol, Tuple, TypedDict

import numpy as np
import plotly.express as px
import tiktoken
import torch as t
from einops import rearrange, repeat
from fancy_einsum import einsum
from jaxtyping import Float, Int
from tensorboardX import SummaryWriter
from torch import nn
from torch.distributions.categorical import Categorical

from gpt.transformer_block import GPT2Block
from mixture_of_experts.cache import MoEFullCache, MoELayerCache
from mixture_of_experts.config import MoEConfig
from mixture_of_experts.moe_block import MoEBlock

config = MoEConfig()


class SparseMoETransformer(nn.Module):
    token_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    transformer_block: nn.Module
    moe_block: nn.Module
    vocab_size: int
    cache: MoEFullCache

    def __init__(
        self,
        *,
        config: MoEConfig = config,
    ):
        super().__init__()
        self.config = config

        self.num_layers = config.num_layers
        self.attn_dropout = config.attn_dropout
        self.expert_dropout = config.expert_dropout

        self.layers: OrderedDict[str, nn.Module] = collections.OrderedDict()
        for i in range(self.num_layers):
            if i % 2 == 0:
                self.layers[f"moe_block{i}"] = MoEBlock(
                    config=config,
                    layer_id=f"moe_layer_{i}",
                )
            else:
                self.layers[f"transformer_block{i}"] = GPT2Block(
                    layer_index=i,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attn_heads,
                )

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.sequential_layers = nn.Sequential(self.layers)
        self.final_norm = nn.LayerNorm([config.hidden_size])
        self.cache = MoEFullCache({})

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, MoEFullCache]:
        """
        x: batch seq_length
        """

        # Get position of tokens
        seq_length = x.shape[1]
        pos = t.arange(0, seq_length).to(x.device)

        # Combine token and positional embeddings
        x = self.token_embedding(x) + self.pos_embedding(pos)

        for idx, layer in self.sequential_layers.named_children():
            if idx.startswith("moe"):
                x, moe_layer_cache = layer(x)
                self.cache[idx] = moe_layer_cache
            else:
                x, _attention_cache = layer(x)
        z = self.final_norm(x)

        # Unembed to get logits for each token
        out = einsum(
            "b s h, v h -> b s v", z, self.token_embedding.weight
        )  # batch seq vocab_size

        return out, self.cache

    def load_model(self, model_path: str):
        self.load_state_dict(t.load(model_path))


def sample_next_token(input: str, model: nn.Module) -> str:
    # Tokenise input
    tokenizer = tiktoken.encoding_for_model(config.tokeniser_string)
    tokens_list = tokenizer.encode(input)
    tokens = t.Tensor(tokens_list).long().unsqueeze(0)  # batch seq

    # Forward pass tokens
    with t.inference_mode():
        with t.no_grad():
            all_logits, _cache = model(t.Tensor(tokens))  # batch seq vocab_size

    # Here we're looking at the next token for the first batch
    logits = all_logits[0, -1, :]  # vocab_size

    # print(t.sort(logits, descending=True))

    # Sample from logits (basic categorical sampling)
    dist = Categorical(logits=logits)
    sampled_token = dist.sample().item()

    assert isinstance(sampled_token, int)

    return tokenizer.decode([sampled_token])


def token_path(cache: MoEFullCache, token_num: int) -> dict:
    """
    Given the token path cache, return where a given token was routed to.
    Show the path as a heatmap.

    cache: OrderedDict[str, t.Tensor]
    cache_tensor: shape (k, num_experts)
    out: dict[str, np.ndarray]
    """

    G = cache.G  # layer, k, num_experts
    token_assignments = cache.token_assignments  # layer, k, num_experts
    layer_indices = cache.layer_indices

    # Build up array of whether token was routed to each expert on a layer and the routing coefficient

    # Get the mask to index the routing matrix
    bool_token_assignment = token_assignments == token_num  # layer, k, num_experts

    # Get the routing matrix for the token
    weighted_token_assignment = G * bool_token_assignment  # layer, k, num_experts
    out = weighted_token_assignment.max(dim=1)[0].numpy()  # layer, num_experts

    array = np.array(out)  # num_expert_layer, num_experts
    _num_expert_layers, num_experts = array.shape

    fig = px.imshow(
        array,
        x=[f"expert_{i}" for i in range(num_experts)],
        y=list(layer_indices),
        title=f"Token {token_num} path",
    )
    fig.show()

    return out


def compare_models(
    model: SparseMoETransformer,
    new_model_path: str,
    first_model_path: Optional[str] = None,
):
    PROMPT = "Oh, tempest of the heart, yield not to sorrow's art; for love is but a"

    if first_model_path is not None:
        model.load_model(first_model_path)

    first_model_output = sample_next_token(model=model, input=PROMPT)

    print("First model output:", first_model_output)

    model.load_model(new_model_path)

    new_model_output = sample_next_token(model=model, input=PROMPT)
    print("trained model output:", new_model_output)


@t.inference_mode()
def main():
    model = SparseMoETransformer()

    x = t.randint(low=0, high=config.vocab_size, size=(1, 6))  # batch seq
    y, moe_cache = model(x)  # batch seq vocab_size, Cache dict
    print("Cache", moe_cache)

    # with SummaryWriter(comment="ModelArchitecture") as w:
    #     w.add_graph(model, (x,))

    token_0_path = token_path(cache=moe_cache, token_num=0)
    print(token_0_path)

    # compare_models(model, new_model_path="models/sgd_100_2023-07-28_02:51:31.pt")


if __name__ == "__main__":
    main()
