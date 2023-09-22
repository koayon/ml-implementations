from functools import partial
from typing import List, Optional, Tuple

import torch as t
from torch import nn
from transformers import AutoTokenizer

from mixture_of_experts.cache import ExpertChoiceFullCache
from mixture_of_experts.interp import tokens_processed_by_expert

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")

COLORS = {
    "expert1": "red",
    "expert2": "blue",
    "both": "purple",
}


def colour_text(word, colour) -> str:
    return f"<span style='color: {colour};'>{word}</span>"


@t.no_grad()
def generate_output_visual(
    input_str: str,
    model: nn.Module,
    expert1: Tuple[str, int],
    expert2: Optional[Tuple[str, int]] = None,
) -> str:
    # Forward model
    input_tokens = t.tensor(
        tokenizer(input_str, return_tensors="pt")["input_ids"]
    )  # 1, seq_len

    cache: ExpertChoiceFullCache
    model.train()
    _, cache = model(input_tokens)

    # Get tokens processed by expert
    layer_index1, expert_num1 = expert1
    token_indexes_expert1, _ = tokens_processed_by_expert(
        cache=cache, layer_index=layer_index1, expert_num=expert_num1
    )
    token_indexes_expert1 = set(token_indexes_expert1)

    if expert2 is not None:
        # Get tokens processed by expert2
        layer_index2, expert_num2 = expert2
        token_indexes_expert2, _ = tokens_processed_by_expert(
            cache=cache, layer_index=layer_index2, expert_num=expert_num2
        )
        token_indexes_expert2 = set(token_indexes_expert2)

        token_indexes_both = token_indexes_expert1.intersection(token_indexes_expert2)
    else:
        token_indexes_expert2 = set()
        token_indexes_both = set()

    def get_text_color(
        i: int,
        token_indexes_expert1: set[int] = token_indexes_expert1,
        token_indexes_expert2: set[int] = token_indexes_expert2,
        token_indexes_both: set[int] = token_indexes_both,
    ):
        if i in token_indexes_both:
            return COLORS["both"]
        if i in token_indexes_expert1:
            return COLORS["expert1"]
        if i in token_indexes_expert2:
            return COLORS["expert2"]
        return None

    # Display the output
    coloured_tokens = []

    for i, _ in enumerate(input_tokens.squeeze(0).tolist()):
        token_str = tokenizer.decode(input_tokens.squeeze(0)[i])
        text_col = get_text_color(i=i)

        coloured_tokens.append(
            colour_text(token_str, text_col) if text_col else token_str
        )

    return "".join(coloured_tokens)
