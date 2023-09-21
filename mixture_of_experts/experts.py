from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import torch as t
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import nn
from torch.nn import functional as F

from helpers import einsum


class Expert(nn.Module):
    """Regular FFN expert with up and down projections.

        Parameters
        ----------
        up_expert : nn.Linear
            _description_
        down_expert : nn.Linear
            _description_
        act_fn : nn.Module
            _description_
        dropout : float
            _description_
    """

    def __init__(self, up_expert: nn.Linear, down_expert: nn.Linear, act_fn: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.up_expert_weight: t.Tensor = up_expert.weight
        self.up_expert_bias: t.Tensor = up_expert.bias
        self.down_expert_weight: t.Tensor = down_expert.weight
        self.down_expert_bias: t.Tensor = down_expert.bias
        self.dropout = dropout
        self.act_fn = act_fn

        self.all_weights: list[t.Tensor] = [self.up_expert_weight, self.up_expert_bias, self.down_expert_weight, self.down_expert_bias]

        self.expert = nn.Sequential(
                        OrderedDict(
                            [
                                ("up_expert", up_expert),
                                ("act_fn", act_fn),
                                ("down_expert", down_expert),
                                ("expert_dropout", nn.Dropout(dropout)),
                            ]
                        )
                    )

    def forward(self, x: t.Tensor):
        return self.expert(x)

@dataclass
class ExpertLinearParams:
    up_expert_weight: t.Tensor
    up_expert_bias: t.Tensor
    down_expert_weight: t.Tensor
    down_expert_bias: t.Tensor

class ExpertFromWeights(nn.Module):
    """Expert with up and down projections, created from weights and biases.

        Parameters
        ----------
        up_expert_weight : t.Tensor
            _description_
        up_expert_bias : t.Tensor
            _description_
        down_expert_weight : t.Tensor
            _description_
        down_expert_bias : t.Tensor
            _description_
        act_fn : nn.Module
            _description_
        dropout : nn.Module
            _description_
    """
    def __init__(self, expert_linear_params: ExpertLinearParams, act_fn: nn.Module, dropout: float):
        super().__init__()
        self.up_expert_weight = expert_linear_params.up_expert_weight
        self.up_expert_bias = expert_linear_params.up_expert_bias
        self.down_expert_weight = expert_linear_params.down_expert_weight
        self.down_expert_bias = expert_linear_params.down_expert_bias
        self.dropout = dropout
        self.act_fn = act_fn

    def forward(self, x: t.Tensor):
        x = F.linear(x, weight = self.up_expert_weight, bias = self.up_expert_bias)
        x = self.act_fn(x)
        x = F.linear(x, weight = self.down_expert_weight, bias = self.down_expert_bias)
        x = F.dropout(x, p = self.dropout)
        return x

class ExpertList(nn.ModuleList):
    def __init__(self, experts: list[Expert]):
        super().__init__(experts)
        self.experts = experts

    @property
    def up_expert_weights(self) -> t.Tensor:
        """
        Returns
        -------
        t.Tensor
            num_experts, up_dim, dim
        """
        expert_weights = t.stack([expert.up_expert_weight for expert in self.experts], dim = 0) # num_experts dim up_dim
        return expert_weights

    @property
    def up_expert_biases(self) -> t.Tensor:
        """
        Returns
        -------
        t.Tensor
            num_experts, up_dim
        """
        expert_biases = t.stack([expert.up_expert_bias for expert in self.experts], dim = 0) # num_experts up_dim
        return expert_biases

    @property
    def down_expert_weights(self) -> t.Tensor:
        """
        Returns
        -------
        t.Tensor
            num_experts, dim, up_dim
        """
        expert_weights = t.stack([expert.down_expert_weight for expert in self.experts], dim = 0)
        return expert_weights

    @property
    def down_expert_biases(self) -> t.Tensor:
        """
        Returns
        -------
        t.Tensor
            num_experts, dim
        """
        expert_biases = t.stack([expert.down_expert_bias for expert in self.experts], dim = 0)
        return expert_biases

    def merge_weights_and_biases(self, merging_weights: Float[t.Tensor, "num_experts"]) -> ExpertLinearParams:
        """Merge experts into a single expert for SMEAR method.

        Parameters
        ----------
        merging_weights : Float[t.Tensor, "num_experts"]
            The weights with which to merge the expert, as a weighted sum.

        Returns
        -------
        ExpertLinearParams
            Container for up and down expert weights and biases
        """
        assert merging_weights.shape == (len(self.experts),)

        # Merge weights and biases
        new_up_weights = einsum("num_experts up_dim dim, num_experts -> up_dim dim", self.up_expert_weights, merging_weights)
        new_up_biases = einsum("num_experts up_dim, num_experts -> up_dim", self.up_expert_biases, merging_weights)

        new_down_weights = einsum("num_experts dim up_dim, num_experts -> dim up_dim", self.down_expert_weights, merging_weights)
        new_down_biases = einsum("num_experts dim, num_experts -> dim", self.down_expert_biases, merging_weights)

        return ExpertLinearParams(up_expert_weight = new_up_weights, up_expert_bias = new_up_biases, down_expert_weight = new_down_weights, down_expert_bias = new_down_biases)

def get_expert_linears(expert_linear_params: ExpertLinearParams) -> Tuple[nn.Linear, nn.Linear]:
    """Get up and down projections from expert linear parameters.
    Used for Experts Weights Averaging

    Reference: https://arxiv.org/pdf/2308.06093.pdf"""

    #TODO: Refactor and add tests

    up_dim, dim = expert_linear_params.up_expert_weight.shape

    up_expert = nn.Linear(in_features = dim, out_features = up_dim, bias = True)
    up_expert.weight.data = expert_linear_params.up_expert_weight.T
    up_expert.bias.data = expert_linear_params.up_expert_bias

    down_expert = nn.Linear(in_features = up_dim, out_features = dim, bias = True)
    down_expert.weight.data = expert_linear_params.down_expert_weight.T
    down_expert.bias.data = expert_linear_params.down_expert_bias

    return up_expert, down_expert
