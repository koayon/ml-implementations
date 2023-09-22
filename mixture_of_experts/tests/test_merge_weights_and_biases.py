import pytest
import torch as t
from torch import nn

from general import device
from mixture_of_experts.experts import Expert, ExpertLinearParams, ExpertList


# Test with multiple experts and valid input.
def test_multiple_experts_valid_input(
    dim: int = 2, up_dim: int = 3, merging_weights=t.tensor([0.4, 0.6])
):
    # Initialize the class object with multiple experts
    expert1 = Expert(
        up_expert=nn.Linear(dim, up_dim),
        down_expert=nn.Linear(up_dim, dim),
        act_fn=nn.ReLU(),
    )
    expert2 = Expert(
        up_expert=nn.Linear(dim, up_dim),
        down_expert=nn.Linear(up_dim, dim),
        act_fn=nn.ReLU(),
    )

    expert1.up_expert_weight

    expert_list = ExpertList([expert1, expert2])

    # Call the method under test
    merged_expert_weights_and_biases = expert_list.merge_weights_and_biases(
        merging_weights
    )

    merged_up_weights = merged_expert_weights_and_biases.up_expert_weight
    merged_down_bias = merged_expert_weights_and_biases.down_expert_bias

    # Assert the output
    assert merged_up_weights.shape == (up_dim, dim)
    assert merged_down_bias.shape == (dim,)
    assert t.allclose(
        merged_up_weights,
        merging_weights[0] * expert1.up_expert_weight
        + merging_weights[1] * expert2.up_expert_weight,
    )


def test_mismatch_number_of_experts(
    dim: int = 2, up_dim: int = 3, merging_weights=t.tensor([0.4, 0.6, 0.3])
):
    # Initialize the class object with multiple experts
    expert1 = Expert(
        up_expert=nn.Linear(dim, up_dim),
        down_expert=nn.Linear(up_dim, dim),
        act_fn=nn.ReLU(),
    )
    expert2 = Expert(
        up_expert=nn.Linear(dim, up_dim),
        down_expert=nn.Linear(up_dim, dim),
        act_fn=nn.ReLU(),
    )

    expert1.up_expert_weight

    expert_list = ExpertList([expert1, expert2])

    # Call the method under test
    with pytest.raises(AssertionError):
        merged_expert_weights_and_biases = expert_list.merge_weights_and_biases(
            merging_weights
        )
