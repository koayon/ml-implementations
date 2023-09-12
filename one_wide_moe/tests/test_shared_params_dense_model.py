import pytest
import torch as t
from einops import repeat
from transformers import AutoTokenizer

from general import device
from one_wide_moe.shared_params_dense_model import SharedParamsDenseModel

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")
model = SharedParamsDenseModel()
model.to(device)

def test_shared_params_dense_model(
    batch_size: int = 2,
):

    input_str = "Hello world"
    tokens_list = tokenizer(input_str)["input_ids"]

    input = repeat(
        t.tensor(tokens_list, device=device),
        "seq_len -> batch seq_len",
        batch=batch_size,
    )  # batch seq
    # input.to(device)

    seq_len = input.shape[1]

    # Check that forward pass works
    y = model(input)
    assert (batch_size, seq_len, model.config.vocab_size) == y.shape
