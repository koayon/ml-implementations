import pytest
import torch as t

from gpt.multimodel_sampling.speculative_decoding import sample


@pytest.mark.parametrize("num_samples", [1, 3])
def test_sample(
    num_samples: int, batch_size: int = 3, seq_len: int = 5, vocab_size: int = 10
):
    probs = t.rand(batch_size, seq_len, vocab_size)
    sampled_ids = sample(probs, num_samples)
    assert sampled_ids.shape == (batch_size, seq_len, num_samples)
