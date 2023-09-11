import pytest
import torch as t
from einops import einsum
from torch import nn
from torch.nn import functional as F

from alibi.attention import AlibiUnidirectionalAttention
from general import device


def test_alibi_mask(hidden_size: int = 16,
        num_heads: int = 8,
        seq_len: int = 6,
        dropout: float = 0
    ):

    attention = AlibiUnidirectionalAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout = dropout
    )

    regular_mask = attention.regular_mask(seq_len)
    assert regular_mask.shape == (seq_len, seq_len)
    # Check that lower triangular part is 0s
    assert (regular_mask.tril() == 0).all()

    # Check that parts of upper triangular part is -inf
    assert regular_mask[0, seq_len-1] == float("-inf")

    alibi_mask = attention.get_alibi_mask(seq_len)

    # Check that lower triangular part is are <= 0
    assert (alibi_mask.tril() <= 0).all()

    # Check that upper triangular part is are -inf
    assert regular_mask[0, seq_len-1] == float("-inf")

    # Do visual check on the atteniton matrix. Can also do this with hooks.

def test_alibi_attention(hidden_size: int = 16,
        num_heads: int = 8,
        seq_len: int = 6,
        dropout: float = 0
    ):
    attention = AlibiUnidirectionalAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout = dropout
    )

    x = t.randn(1, seq_len, hidden_size)
    out, _ = attention(x)
    assert out.shape == (1, seq_len, hidden_size)

if __name__ == "__main__":
    test_alibi_mask()
