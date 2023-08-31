import pytest
import torch as t
from einops import einsum
from torch import nn
from torch.nn import functional as F

from general import device
from general.ensemble import Ensemble


@pytest.mark.parametrize("hidden_size", [8, 16])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 4])
def test_ensemble(
    hidden_size: int,
    batch_size: int,
    seq_len: int,
):
    l1 = nn.Linear(hidden_size, hidden_size)
    l2 = nn.Linear(hidden_size, hidden_size)
    l3 = nn.Linear(hidden_size, hidden_size)

    ensemble_model = Ensemble(models = [l1, l2, l3])

    x = t.randn(
        (batch_size, seq_len, hidden_size),
        requires_grad=True
    )

    # Check that forward pass works
    y = ensemble_model(x)
    assert y.shape == x.shape

    # Check that gradients are propagated
    t.sum(t.flatten(y)).backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.requires_grad is False

    # Check that model is the same as averaging the outputs of the models
    y1 = F.softmax(l1(x), dim = -1)
    y2 = F.softmax(l2(x), dim = -1)
    y3 = F.softmax(l3(x), dim = -1)
    y_avg = t.log(t.mean(t.stack([y1, y2, y3], dim = 0), dim = 0))

    assert t.allclose(y, y_avg, atol = 1e-4)



def test_weighted_ensemble():
    l1 = nn.Linear(8, 8)
    l2 = nn.Linear(8, 8)
    l3 = nn.Linear(8, 8)

    ensemble_model = Ensemble(models = [l1, l2, l3], model_weighting = t.tensor([1, 2, 3]))

    x = t.randn(
        (1, 1, 8),
        requires_grad=True
    )

    # Check that weights are normalized
    assert t.allclose(ensemble_model.model_weighting, t.tensor([1/6, 2/6, 3/6]))

    # Check that forward pass works
    y = ensemble_model(x)
    assert x.shape == y.shape

    # Check that gradients are propagated
    t.sum(t.flatten(y)).backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.requires_grad is False

    # Check that model is the same as weighted averaging the outputs of the models
    y1 = F.softmax(l1(x), dim = -1)
    y2 = F.softmax(l2(x), dim = -1)
    y3 = F.softmax(l3(x), dim = -1)

    y_weighted_avg = t.log(einsum(ensemble_model.model_weighting, t.stack([y1, y2, y3], dim = 0), "model, model batch seq vocab -> batch seq vocab"))


    assert t.allclose(y, y_weighted_avg, atol = 1e-4)
