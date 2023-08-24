import pytest
import torch

from moet_experiment.alibi_confi_block import ALiBiConfiTBlock

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


@pytest.mark.parametrize("layer_index", [0])
@pytest.mark.parametrize("hidden_size", [8])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 4])
@pytest.mark.parametrize("in_features", [8])
def test_alibi_confi_t_block(
    layer_index: int,
    hidden_size: int,
    num_heads: int,
    batch_size: int,
    seq_len: int,
    in_features: int,
):
    transformer_block = ALiBiConfiTBlock(
        layer_index=layer_index,
        hidden_size=hidden_size,
        num_heads=num_heads,
    )
    x = torch.randn(
        (batch_size, seq_len, in_features),
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    # Check that forward pass works
    y, _cache = transformer_block(x)
    assert y.size(0) == x.size(0)
    assert y.size(1) == x.size(1)
    assert y.size(2) == x.size(2)

    # Check that gradients are propagated
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.requires_grad is False


def test_alibi_confi_t_block_exceptions():
    transformer_block = ALiBiConfiTBlock(
        layer_index=0,
        hidden_size=8,
        num_heads=4,
    )

    # Test wrong input dimension
    x = torch.randn((1, 4, 16), device=DEVICE, dtype=DTYPE)
    with pytest.raises(RuntimeError):
        transformer_block(x)

    # Test invalid number of dimensions
    x = torch.randn((1, 4, 2, 8), device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError):
        transformer_block(x)
