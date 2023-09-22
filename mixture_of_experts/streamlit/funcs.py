from typing import List, Tuple

import torch as t
from torch import nn
from transformers import AutoTokenizer

from mixture_of_experts.cache import ExpertChoiceFullCache
from mixture_of_experts.interp import tokens_processed_by_expert

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")


def colour_text(word, colour):
    return f'<u style="color: {colour}">{word}</u>'


def generate_output_visual(
    expert: Tuple[str, int], input_str: str, model: nn.Module
) -> str:
    layer_index, expert_num = expert

    # Forward model
    input_tokens = t.tensor(
        tokenizer(input_str, return_tensors="pt")["input_ids"]
    )  # 1, seq_len

    cache: ExpertChoiceFullCache
    _, cache = model(input_tokens)

    # Get tokens processed by expert
    token_indexes, tokens = tokens_processed_by_expert(
        cache=cache, layer_index=layer_index, expert_num=expert_num
    )
    token_indexes = set(token_indexes)

    # Display the output
    coloured_text = ""

    for i, _ in enumerate(input_tokens.squeeze(0).tolist()):
        if i in token_indexes:
            coloured_text += f"<span style='color: red;'>{tokenizer.decode(input_tokens.squeeze(0)[i])}</span>"
        else:
            coloured_text += tokenizer.decode(input_tokens.squeeze(0)[i])
    return coloured_text
