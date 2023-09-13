import pytest
import torch

from general import device
from general.confi_ffn import ConfiFFN


@pytest.mark.parametrize("hidden_size", [8, 16])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 4])
def test_confi_ffn(
    hidden_size: int,
    batch_size: int,
    seq_len: int,
):
    confi_ffn = ConfiFFN(
        hidden_size=hidden_size,
        dropout=0.2,
        activation_function="silu",
    )
    x = torch.randn(
        (batch_size, seq_len, hidden_size),
        requires_grad=True,
    )

    # Check that forward pass works
    y = confi_ffn(x)
    assert y.size(0) == x.size(0)
    assert y.size(1) == x.size(1)
    assert y.size(2) == x.size(2)

    # Check that gradients are propagated
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.requires_grad is False
