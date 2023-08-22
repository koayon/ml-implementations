from typing import Any, Dict, List, Optional, Tuple

import torch as t
from jaxtyping import Int
from torch import nn

from mixture_of_experts.config import MoEConfig

config = MoEConfig()


class HashRouter(nn.Module):
    """Router for the MoE layer that uses a hash function to assign tokens to experts.

    Reference: https://arxiv.org/pdf/2106.04426.pdf"""

    def __init__(
        self,
        *,
        config: MoEConfig,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.vocab_size = config.vocab_size

    def forward(
        self, input: Int[t.Tensor, "batch, seq"], hash: Int[t.Tensor, "vocab_size"]
    ) -> Int[t.Tensor, "batch, seq"]:
        "Takes in token ids and a hashing function and returns the expert num that each token should be assigned to."
        return hash[input]  # batch, seq

    def build_random_hash(self, seed: Optional[int] = None) -> t.Tensor:
        g = t.Generator()
        if seed is not None:
            g.manual_seed(seed)

        hash = t.randint(high=self.num_experts, size=(self.vocab_size,))
        return hash

    def build_balanced_hash(self) -> t.Tensor:
        raise NotImplementedError
