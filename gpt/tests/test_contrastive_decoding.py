import pytest
import torch as t

from gpt.multimodel_sampling.contrastive_decoding import ContrastiveDecodingWrapper

cd_model = ContrastiveDecodingWrapper()
print("Loaded models")


def test_forward():
    x = t.tensor([[1, 2, 3, 4, 5]])
    logits = cd_model(x)
    assert logits.shape == (1, 5, 50257)
