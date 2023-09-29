import pytest
import torch as t

from gpt.multimodel_sampling.contrastive_decoding import ContrastiveDecodingWrapper

cd_model = ContrastiveDecodingWrapper()
print("Loaded models")

VOCAB_SIZE = 50257


def test_forward():
    x = t.tensor([[1, 2, 3, 4, 5]])
    logits, _, _ = cd_model(x)
    assert logits.shape == (1, VOCAB_SIZE)
