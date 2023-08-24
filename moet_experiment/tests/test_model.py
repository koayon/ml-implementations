import pytest
import torch as t
from einops import repeat
from numpy import require
from transformers import AutoTokenizer

from moet_experiment.model import MoET

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
DTYPE = t.float32

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")


@pytest.mark.parametrize("batch_size", [2, 4])
def test_moet_model(
    batch_size: int,
):
    model = MoET()

    input_str = "Hello world"
    tokens_list = tokenizer(input_str)["input_ids"]
    input = repeat(
        t.tensor(tokens_list),
        "seq_len -> batch seq_len",
        batch=batch_size,
    )  # batch seq
    seq_len = input.shape[1]

    # Check that forward pass works
    y, _cache = model(input)
    assert y.size(0) == batch_size
    assert y.size(1) == seq_len
    assert y.size(2) == model.config.vocab_size


def test_moet_model_exceptions(batch_size: int = 4, seq_len: int = 8):
    model = MoET()

    # Test invalid number of dimensions
    input = t.randint(high=50_000, size=(batch_size, seq_len, 512), device=DEVICE)
    with pytest.raises(ValueError):
        model(input)
