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
        k: int = 1,
        config: MoEConfig,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.vocab_size = config.vocab_size
        self.k = k

    def forward(
        self, input: Int[t.Tensor, "batch, seq"], hash: Int[t.Tensor, "vocab_size, k"]
    ) -> Int[t.Tensor, "batch, seq, k"]:
        "Takes in token ids and a hashing function and returns the expert num that each token should be assigned to."
        hashes = [hash[:, i][input] for i in range(self.k)]  # k list of seq, batch

        out = t.stack(hashes, dim=-1)  # batch, seq, k
        return out  # batch, seq, k

    def build_random_hash(self, seed: int = 42) -> Int[t.Tensor, "vocab_size, k"]:
        """Randomly assigns each token in vocabularly to k experts.

        Returns
        -------
        t.Tensor
            Hash function that maps tokens to k experts.
            A lookup table which given a token id returns a list of k expert ids.
        """
        g = t.Generator()
        g.manual_seed(seed)

        hash = t.randint(
            high=self.num_experts, size=(self.vocab_size, self.k), generator=g
        )
        return hash

    def build_balanced_hash(
        self, token_frequencies: Int[t.Tensor, "vocab_size"]
    ) -> t.Tensor:
        """Assigns each token to k experts in a way that balances the number of tokens assigned to each expert.

        Parameters
        ----------
        token_frequencies : t.Tensor
            A tensor of shape (vocab_size,) where each element is the number of times a token appears in (a subset of) the training data.

        Returns
        -------
        t.Tensor
            Hash function that maps tokens to k experts.
        """
        raise NotImplementedError
