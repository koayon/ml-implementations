import pytest
import torch as t
from einops import repeat
from transformers import AutoTokenizer

from general import device
from one_wide_moe.shared_moe_model import SharedParamsMoEModel

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")


@pytest.mark.parametrize("ffn_dim_multiplier", [4, 6])
@pytest.mark.parametrize("num_experts", [2, 4])
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
    if (not share_moe_layers) and share_routers:
        with pytest.raises(ValueError):
            model = SharedParamsMoEModel(
                ffn_dim_multiplier=ffn_dim_multiplier,
                num_experts=num_experts,
                share_attention_layers=share_attention_layers,
                share_moe_layers=share_moe_layers,
                share_routers=share_routers,
            )
        return


    model = SharedParamsMoEModel(ffn_dim_multiplier=ffn_dim_multiplier, num_experts = num_experts, share_attention_layers=share_attention_layers, share_moe_layers=share_moe_layers, share_routers = share_routers)
    # model.to(device)

    input_str = "Hello world"
    tokens_list = tokenizer(input_str)["input_ids"]

    input = repeat(
        t.tensor(tokens_list, device=device),
        "seq_len -> batch seq_len",
        batch=batch_size,
    )  # batch seq
    # input.to(device)

    seq_len = input.shape[1]

    # Check that forward pass works
    y, _cache = model(input)
    assert (batch_size, seq_len, model.config.vocab_size) == y.shape
