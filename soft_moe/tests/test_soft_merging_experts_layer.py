import pytest
import torch as t

from general import device
from moet_experiment.moet_config import MoETConfig
from soft_moe.soft_merging_experts_layer import SoftMergingExpertLayer

config = MoETConfig()

@pytest.mark.parametrize("num_experts", [2])
@pytest.mark.parametrize("group_size", [2])
@pytest.mark.parametrize("seq_len", [6])
@pytest.mark.parametrize("batch_size", [4])
def test_soft_merging_expert_layer(
    num_experts: int,
    group_size: int,
    seq_len: int,
    batch_size: int,
    config: MoETConfig = MoETConfig(),
):
    moe_layer = SoftMergingExpertLayer(
        num_experts=num_experts,
        layer_id="layer1",
        group_size=group_size,
        config=config,
    )
    moe_layer.to(device)

    x = t.randn(
        (batch_size*seq_len, config.hidden_size),
        requires_grad=True,
        device = device
    )
    routing_logits = t.randn(size = (batch_size*seq_len, num_experts), device= device)

    # Check that forward pass works
    y, _cache, merged_expert = moe_layer(x = x, routing_logits = routing_logits)

    assert x.shape == y.shape

    # Check that gradients are propagated
    t.sum(t.flatten(y)).backward()

    first_param = None
    for name, param in moe_layer.named_parameters():
        if param.is_leaf:
            first_param = param
            break

    assert first_param is not None
    assert first_param.grad is not None
    assert first_param.grad.shape == first_param.shape
    assert first_param.grad.requires_grad is False
