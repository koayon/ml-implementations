import pytest
import torch as t
from einops import repeat
from transformers import AutoTokenizer

from general import device
from one_wide_moe.shared_moe_model import SharedParamsMoEModel

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")


@pytest.mark.parametrize("ffn_dim_multiplier", [4])
@pytest.mark.parametrize("num_experts", [4])
@pytest.mark.parametrize("share_attention_layers", [True, False])
@pytest.mark.parametrize("share_moe_layers", [True, False])
@pytest.mark.parametrize("share_routers", [True, False])
def test_shared_params_dense_model(
    ffn_dim_multiplier: int,
    num_experts: int,
    share_attention_layers: bool,
    share_moe_layers: bool,
    share_routers: bool,
    batch_size: int = 2,
):

    model = SharedParamsMoEModel(ffn_dim_multiplier=ffn_dim_multiplier, num_experts = num_experts, share_attention_layers=share_attention_layers, share_expert_layers=share_moe_layers, share_routers = share_routers)
    model.to(device)

    input_str = "Hello world"
    tokens_list = tokenizer(input_str)["input_ids"]

    input_ids = repeat(
        t.tensor(tokens_list, device=device),
        "seq_len -> batch seq_len",
        batch=batch_size,
    )  # batch seq
    # input.to(device)

    seq_len = input_ids.shape[1]

    # Check that forward pass works
    y, _cache = model(input_ids)
    assert (batch_size, seq_len, model.config.vocab_size) == y.shape


def test_group_shared_moe_model(batch_size: int = 2):
    model = SharedParamsMoEModel(ffn_dim_multiplier=4, num_experts=8, share_attention_layers=False, share_expert_layers=True, share_routers=False, group_size=2)
    model.to(device)

    input_str = "Hello world"
    tokens_list = tokenizer(input_str)["input_ids"]

    input_ids = repeat(
        t.tensor(tokens_list, device=device),
        "seq_len -> batch seq_len",
        batch=batch_size,
    )  # batch seq
    # input.to(device)

    seq_len = input_ids.shape[1]

    # Check that forward pass works
    y, _cache = model(input_ids)
    assert (batch_size, seq_len, model.config.vocab_size) == y.shape

    # Check that gradients are propagated
    t.sum(t.flatten(y)).backward()

    first_param = None
    for name, param in model.named_parameters():
        if param.is_leaf:
            first_param = param
            break

    assert first_param is not None and first_param.grad is not None
    assert first_param.grad.shape == first_param.shape
    assert first_param.grad.requires_grad is False
