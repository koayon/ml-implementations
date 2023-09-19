import sys
from typing import Any, Optional

import torch as t
from jaxtyping import Int
from torch import nn
from torch.nn.functional import one_hot

from general import device
from mixture_of_experts.config import MoEConfig
from moet_experiment.moet_config import MoETConfig

config = MoEConfig()


class HashRouter(nn.Module):
    """Router for the MoE layer that uses a hash function to assign tokens to experts.

    Reference: https://arxiv.org/pdf/2106.04426.pdf"""

    hash: Int[t.Tensor, "vocab_size k"]

    def __init__(
        self, *, k: int = 1, config: MoEConfig | MoETConfig = config, num_experts: int
    ):
        super().__init__()
        self.num_experts = num_experts
        self.vocab_size = config.vocab_size
        self.k = k

        # self.generator = t.Generator(device = device)
        self.generator = t.Generator()

        self.hash = self.build_random_hash()
        self.hash = self.hash.to(device)

    def forward(self, input: Int[t.Tensor, "bs"]) -> Int[t.Tensor, "bs k"]:
        "Takes in token ids and a hashing function and returns the expert num that each token should be assigned to."

        hashes = [self.hash[:, i][input]
                  for i in range(self.k)]  # k list of bs

        top_k = t.stack(hashes, dim=-1)  # bs, k
        top_k_one_hot = one_hot(
            top_k, num_classes=self.num_experts
        )  # bs, k, num_experts
        out, _indices = t.max(top_k_one_hot, dim=1)  # bs, num_experts

        return out  # bs, num_experts

    def build_random_hash(self, seed: int = 42) -> t.Tensor:
        """Randomly assigns each token in vocabularly to k experts.

        Returns
        -------
        t.Tensor
            Hash function that maps tokens to k experts.
            A lookup table which given a token id returns a list of k expert ids.
        """
        self.generator.manual_seed(seed)

        hash = t.randint(
            high=self.num_experts, size=(
                self.vocab_size, self.k), generator=self.generator
            # , device = device
        )
        return hash

    def build_balanced_hash(
        self, token_frequencies: Int[t.Tensor, "vocab_size k"]
    ) -> None:
        """Assigns each token to k experts in a way that balances the number of tokens assigned to each expert.
        This is based on the token occurences in the training data.

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


def main():
    hash_router = HashRouter(k=1, num_experts=2)
    hash_router.build_random_hash()
    print(hash_router.hash)
    print(hash_router.forward(t.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))


if __name__ == "__main__":
    main()
