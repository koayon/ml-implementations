import collections
import token
from dataclasses import dataclass
from typing import Iterable, List, Optional, OrderedDict, Protocol, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import tiktoken
import torch as t
from einops import rearrange, repeat
from jaxtyping import Float, Int
from tensorboardX import SummaryWriter
from torch import nn
from torch.distributions.categorical import Categorical

from gpt.transformer_block import GPT2Block
from helpers import einsum, remove_hooks
from mixture_of_experts.cache import MoEFullCache, MoELayerCache
from mixture_of_experts.config import MoEConfig
from mixture_of_experts.moe_block import MoEBlock

config = MoEConfig()
tokeniser = tiktoken.encoding_for_model(config.tokeniser_string)


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
                self.layers[f"transformer_block{i}"] = GPT2Block(
                    layer_index=i,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attn_heads,
                )
            else:
                self.layers[f"moe_block{i}"] = MoEBlock(
                    config=config,
                    layer_id=f"moe_layer_{i}",
                )

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.sequential_layers = nn.Sequential(self.layers)
        self.final_norm = nn.LayerNorm([config.hidden_size])
        self.cache = MoEFullCache({})

    def unembed(self, z: Float[t.Tensor, "batch seq hidden"]) -> t.Tensor:
        out = einsum(
            "b s h, v h -> b s v", z, self.token_embedding.weight
        )  # batch seq vocab_size
        return out

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
        out = self.unembed(z)  # batch seq vocab_size

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
    out = weighted_token_assignment.max(dim=1)[0]  # layer, num_experts

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


def expert_importance(cache: MoEFullCache) -> t.Tensor:
    """
    Calculates the importance of each expert in a layer based on the softmaxed routing weights.

    Parameters
    ----------
    cache : MoEFullCache
        The cache object containing the softmaxed routing weights.

    Returns
    -------
    torch.Tensor
        A tensor of shape (layer, num_experts) representing the importance of each expert in each layer.
    """
    G = cache.G  # layer, k, num_experts
    importance = G.sum(dim=1)  # layer, num_experts
    return importance


def top_tokens_for_expert(
    cache: MoEFullCache,
    layer_index: str,
    expert_num: int,
    input: t.Tensor,
    tokeniser: tiktoken.Encoding = tokeniser,
) -> Tuple[list[str], list[int]]:
    """
    Retrieve the top tokens for a specific expert in a specific layer.

    Parameters
    ----------
    cache : MoEFullCache
        The cache of expert routing.
    layer_index : str
        The index of the layer.
    expert : int
        The index of the expert.
    input : t.Tensor
        The input tensor of tokens - shape (batch, seq).
    tokeniser : tiktoken.Encoding
        The model's tokeniser - used to decode the tokens.

    Returns
    -------
    list[str]
        The top tokens for the specified expert in the specified layer. There should be k tokens
    list[int]
        The indices of the top tokens in the sequence for the specified expert.
    """
    layer_cache = cache[layer_index]
    tokens_indexes = layer_cache.token_assignments[:, expert_num]  # k

    tokens = rearrange(input, "batch seq -> (batch seq)")

    expert_tokens = tokens[tokens_indexes].unsqueeze(dim=1).cpu()  # k, 1

    str_tokens = tokeniser.decode_batch(expert_tokens.tolist())

    return str_tokens, tokens_indexes.tolist()


def expert_token_table(
    cache: MoEFullCache, input: t.Tensor, tokeniser: tiktoken.Encoding = tokeniser
) -> pd.DataFrame:
    layer_indexes = []
    expert_nums = []
    expert_strs = []

    for layer_index in cache.layer_indices:
        for expert_num in range(cache.num_experts):
            top_tokens = top_tokens_for_expert(
                cache=cache,
                layer_index=layer_index,
                expert_num=expert_num,
                input=input,
                tokeniser=tokeniser,
            )
            layer_indexes.append(layer_index)
            expert_nums.append(expert_num)
            expert_strs.append(top_tokens)

    df = pd.DataFrame(
        {
            "layer_index": layer_indexes,
            "expert_num": expert_nums,
            "expert_tokens": expert_strs,
        }
    )
    return df


def add_in_hooks(module: nn.Module, display_name: str, activations: dict) -> None:
    def fwd_hook(mod, input, output):
        activations[f"pre_{display_name}"] = input

    module.register_forward_hook(fwd_hook)


def add_out_hooks(module: nn.Module, display_name: str, activations: dict) -> None:
    def fwd_hook(mod, input, output):
        activations[f"post_{display_name}"] = output

    module.register_forward_hook(fwd_hook)


def add_ablation_hook(module: nn.Module) -> None:
    def fwd_hook(mod, input, output):
        output = t.zeros_like(output)

    module.register_forward_hook(fwd_hook)


def logit_lens_before_and_after_expert(
    input: Int[t.Tensor, "batch seq"],
    # token_indices: Iterable[int],
    layer_index: int,
    expert_num: int,
    model: SparseMoETransformer,
    tokeniser: tiktoken.Encoding = tokeniser,
) -> Tuple[pd.DataFrame, list, list]:
    """Idea: First forward pass the input with hooks to get the activations before and after the expert.
    Then, for each token which is relevant, umembed

    Parameters
    ----------
    input : t.Tensor
        _description_
    token_indices : t.Tensor
        _description_
    layer_index : str
        _description_
    expert_num : int
        _description_
    model : SparseMoETransformer
        _description_
    tokeniser : tiktoken.Encoding, optional
        _description_, by default tokeniser

    Returns
    -------
    pd.DataFrame
        _description_
    """

    assert layer_index % 2 == 0, "Only works for MoE layers"

    # Add hooks
    activations: dict[str, Float[t.Tensor, "batch, seq, hidden"]] = {}
    layer: nn.Module = model.sequential_layers[layer_index]
    # expert: nn.Module = layer.experts[expert_num]  # type: ignore

    # Ablate other experts
    for i, expert in enumerate(layer.expert_layer.experts):  # type: ignore
        if i != expert_num:
            add_ablation_hook(expert)  # type: ignore

    name = f"layer_{layer_index}_expert_{expert_num}"
    add_in_hooks(layer.ln2, name, activations)  # type: ignore
    add_out_hooks(layer.expert_layer, name, activations)  # type: ignore

    # TODO: Confused that even the tokens not routed to our expert seem to have changed?
    # Examine this

    # Forward pass and get activations
    with t.inference_mode():
        with t.no_grad():
            # Getting activations before expert layer and after expert layer where only the chosen expert is active
            model(input)
    model.apply(remove_hooks)

    pre_unmbedded = model.unembed(activations[f"pre_{name}"][0])  # batch seq vocab_size
    post_unembedded = model.unembed(
        activations[f"post_{name}"][0]
    )  # batch seq vocab_size

    pre_tokens = pre_unmbedded.argmax(dim=-1)  # batch seq
    post_tokens = post_unembedded.argmax(dim=-1)  # batch seq

    pre_str_tokens = tokeniser.decode_batch(pre_tokens.tolist())
    post_str_tokens = tokeniser.decode_batch(post_tokens.tolist())

    before_after = {f"Before {name}": pre_str_tokens, f"After {name}": post_str_tokens}

    return pd.DataFrame(before_after), pre_str_tokens, post_str_tokens


def logit_diffs(
    input: Int[t.Tensor, "batch seq"],
    tokens_to_check: list[str],
    layer_index: int,
    expert_num: int,
    model: SparseMoETransformer,
    tokeniser: tiktoken.Encoding = tokeniser,
) -> pd.DataFrame:
    """Returns the difference in logits for given potential tokens before and after the expert layer.

    Parameters
    ----------
    input : Int[t.Tensor, "batch seq"]
        _description_
    tokens_to_check : list[str]
        _description_
    layer_index : int
        _description_
    expert_num : int
        _description_
    model : SparseMoETransformer
        _description_
    tokeniser : tiktoken.Encoding, optional
        _description_, by default tokeniser

    Returns
    -------
    pd.DataFrame
        _description_
    """
    raise NotImplementedError


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
    # print("Cache", moe_cache)

    expert0_0 = top_tokens_for_expert(
        cache=moe_cache, layer_index="moe_block0", expert_num=0, input=x
    )

    print(expert0_0)

    # expert_df = expert_token_table(cache=moe_cache, input=x)
    # print(expert_df)

    before_after, pre, post = logit_lens_before_and_after_expert(
        input=x, layer_index=0, expert_num=0, model=model
    )
    print(before_after)
    print(pre)
    print(post)

    # with SummaryWriter(comment="ModelArchitecture") as w:
    #     w.add_graph(model, (x,))

    # token_0_path = token_path(cache=moe_cache, token_num=0)
    # print(token_0_path)

    # compare_models(model, new_model_path="models/sgd_100_2023-07-28_02:51:31.pt")


if __name__ == "__main__":
    main()
