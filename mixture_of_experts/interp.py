from typing import Iterable, List, Optional, OrderedDict, Protocol, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tiktoken
import torch as t
from einops import rearrange, repeat
from jaxtyping import Float, Int
from plotly.graph_objs._figure import Figure
from torch import nn

from helpers import einsum
from hooks import remove_hooks
from mixture_of_experts.cache import (
    ExpertChoiceFullCache,
    ExpertChoiceLayerCache,
    TokenChoiceFullCache,
    TokenChoiceLayerCache,
)
from mixture_of_experts.config import MoEConfig
from mixture_of_experts.experts import Expert
from mixture_of_experts.model import SparseMoETransformer
from moet_experiment.model import MoET

config = MoEConfig()
tokeniser = tiktoken.encoding_for_model(config.tokenizer_string)


def token_path(cache: ExpertChoiceFullCache, token_num: int) -> dict:
    """
    Given the token path cache, return where a given token was routed to.
    Show the path as a heatmap.

    cache: OrderedDict[str, t.Tensor]
    cache_tensor: shape (k, num_experts)
    out: dict[str, np.ndarray]
    """

    G = cache.G  # layer, num_experts, k
    layer_indices = cache.layer_indices
    P = cache.P  # layer, num_experts, bs, k

    # Build up array of whether token was routed to each expert on a layer and the routing coefficient

    # Get the mask to index the routing matrix
    P_token = P[:, :, token_num, :]  # layer, num_experts, k

    # Get the routing matrix for the token
    weighted_token_assignment = G * P_token  # layer, num_experts, k
    out, _ = weighted_token_assignment.max(dim=1)  # layer, num_experts

    array = np.array(out)  # num_expert_layer, num_experts
    _num_expert_layers, num_experts = array.shape

    assert num_experts == G.shape[1]

    fig = px.imshow(
        array,
        x=[f"expert_{i}" for i in range(num_experts)],
        y=list(layer_indices),
        title=f"Token {token_num} path",
    )
    fig.show()

    return out


def expert_importance(cache: ExpertChoiceFullCache) -> t.Tensor:
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
    G = cache.G  # layer, num_experts, k
    importance = G.sum(dim=-1)  # layer, num_experts
    return importance


def tokens_processed_by_expert(
    cache: ExpertChoiceFullCache,
    layer_index: str,
    expert_num: int,
    input: Optional[t.Tensor] = None,
    tokeniser: tiktoken.Encoding = tokeniser,
) -> Tuple[list[int], Optional[list[str]]]:
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
    list[int]
        The indices of the top tokens in the sequence for the specified expert.
    list[str]
        The top tokens for the specified expert in the specified layer. There should be k tokens
    """
    layer_cache: ExpertChoiceLayerCache = cache[layer_index]
    tokens_indexes = layer_cache.token_assignments[:, expert_num]  # k

    if input is not None:
        tokens = rearrange(input, "batch seq -> (batch seq)")

        expert_tokens = tokens[tokens_indexes].unsqueeze(dim=1)  # k, 1

        str_tokens = tokeniser.decode_batch(expert_tokens.tolist())
    else:
        str_tokens = None

    return tokens_indexes.tolist(), str_tokens


def expert_token_table(
    cache: ExpertChoiceFullCache,
    input: t.Tensor,
    tokeniser: tiktoken.Encoding = tokeniser,
) -> pd.DataFrame:
    layer_indexes = []
    expert_nums = []
    expert_strs = []

    for layer_index in cache.layer_indices:
        for expert_num in range(cache.num_experts):
            _, top_tokens = tokens_processed_by_expert(
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


def expert_affinity(
    expert_1: Tuple[str, int, int],
    expert_2: Tuple[str, int, int],
    cache: ExpertChoiceFullCache,
) -> float:
    """Measures the affinity of two experts by calculating the number of tokens which are routed to both experts.

    Can either pick experts in different layers to understand compositionality or experts within the same layer to understand parallel computation.

    Parameters
    ----------
    expert_1 : Tuple[str, int, int]
        layer_index, layer_num, expert_num
    expert_2 : Tuple[str, int, int]
        layer_index, layer_num, expert_num
    cache : ExpertChoiceFullCache
        Cache

    Returns
    -------
    float
        Affinity score between 0 and 1.
    """
    layer_index_1, _layer_num_1, expert_num_1 = expert_1
    layer_index_2, _layer_num_2, expert_num_2 = expert_2
    expert_1_tokens, _ = tokens_processed_by_expert(cache, layer_index_1, expert_num_1)
    expert_2_tokens, _ = tokens_processed_by_expert(cache, layer_index_2, expert_num_2)

    # Get the number of tokens routed to both experts
    num_tokens_1 = len(expert_1_tokens)
    num_tokens_2 = len(expert_2_tokens)
    num_tokens_both = len(set(expert_1_tokens).intersection(expert_2_tokens))

    # Calculate the affinity
    affinity = 3 * num_tokens_both / (num_tokens_1 + num_tokens_2 - num_tokens_both)
    affinity = affinity ** (1 / 2)

    return affinity


def expert_weights_similarity(
    expert_1: Tuple[str, int, int], expert_2: Tuple[str, int, int], model: MoET
) -> float:
    """Get a similarity score between two experts by comparing their weights directly.
    Typically want to compare experts within the same layer.

    Parameters
    ----------
    expert_1 : Tuple[str, int, int]
        layer_index, layer_num, expert_num
    expert_2 : Tuple[str, int, int]
        layer_index, layer_num, expert_num
    model : MoET
        Model

    Returns
    -------
    float
        Similarity score between 0 and 1.
    """
    layer_index_1, layer_num_1, expert_num_1 = expert_1
    layer_index_2, layer_num_2, expert_num_2 = expert_2

    expert1: Expert = model.sequential_layers[layer_num_1].moe_layer.expert_layer[expert_num_1]  # type: ignore
    expert2: Expert = model.sequential_layers[layer_num_2].moe_layer.expert_layer[expert_num_2]  # type: ignore

    # Get the weights (list of the weights and biases of each linear layer)
    expert1_weights = expert1.all_weights
    expert2_weights = expert2.all_weights

    # Calculate the similarity
    differences = [
        t.norm(a - b, p="fro") for a, b in zip(expert1_weights, expert2_weights)
    ]
    differences = t.stack(differences)
    avg_difference = t.mean(differences).item()

    similarity = 1 - avg_difference

    return similarity


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
    # TODO: ^Generalise

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


@t.inference_mode()
def main():
    model = SparseMoETransformer()

    x = t.randint(low=0, high=config.vocab_size, size=(1, 6))  # batch seq
    y, moe_cache = model(x)  # batch seq vocab_size, Cache dict
    # print("Cache", moe_cache)

    expert0_0 = tokens_processed_by_expert(
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
