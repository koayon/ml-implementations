import pytest
import torch as t
from einops import repeat
from transformers import AutoTokenizer

from general import device
from moet_experiment.model import MoET

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")


@pytest.mark.parametrize("batch_size", [2, 4])
def test_moet_model(
    batch_size: int,
):
    model = MoET()
    model.to(device)

    input_str = "Hello world"
    tokens_list = tokenizer(input_str)["input_ids"]

    input = repeat(
        t.tensor(tokens_list, device=device),
        "seq_len -> batch seq_len",
        batch=batch_size,
    )  # batch seq
    # input.to(device)

    seq_len = input.shape[1]

    print(input.device)

    # Check that forward pass works
    y, moe_cache = model(input)
    assert (batch_size, seq_len, model.config.vocab_size) == y.shape

    # Check that backward pass works for loss
    y.sum().backward()
    for p in model.parameters():
        assert p.grad is not None
        assert p.grad.shape == p.shape
        assert p.grad.requires_grad is False
        break

    model.zero_grad()

    y, moe_cache = model(input)

    # Check that backward pass works for cache
    moe_cache.routing_logits_tensor.sum().backward()
    for p in model.parameters():
        assert p.grad is not None
        assert p.grad.shape == p.shape
        assert p.grad.requires_grad is False
        break



def test_moet_model_exceptions(batch_size: int = 4, seq_len: int = 8):
    model = MoET()

    # Test invalid number of dimensions
    input = t.randint(high=50_000, size=(batch_size, seq_len, 512))
    with pytest.raises(ValueError):
        model(input)
