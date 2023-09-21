import sys
from enum import Enum, auto
from typing import Any, Optional, Tuple, Union

import torch as t
from einops import rearrange
from jaxtyping import Int
from torch import nn
from torch.nn.functional import one_hot

from general import device
from mixture_of_experts.config import MoEConfig
from moet_experiment.moet_config import MoETConfig
from one_wide_moe.one_wide_config import OneWideConfig

config = MoEConfig()

class RouterEnums(Enum):
    linear = auto()
    learned = auto()
    hash = auto()

class Router(nn.Module):
    def __init__(self, *, num_experts: int, router_str: str, config: Union[MoETConfig, OneWideConfig]):
        super().__init__()
        self.config = config

        self.router_enum = RouterEnums[router_str]
        self.router_temperature = config.router_temperature
        self.num_experts = num_experts
        self.hidden_size = config.hidden_size

        if self.router_enum in (RouterEnums.linear, RouterEnums.learned):
            # Define the linear router
            self.linear = nn.Linear(self.hidden_size, self.num_experts)
        elif self.router_enum == RouterEnums.hash:
            assert isinstance(config, MoETConfig)
            # Build the hash router
            assert config.num_experts_hash == self.num_experts
            self.hash_router = HashRouter(
                config=config, num_experts=config.num_experts_hash)
            # self.hash_router.build_random_hash()
        else:
            raise ValueError(
                f"Unknown router {router_str}. Please choose from {RouterEnums._member_names_}")

        self.routing_dropout = nn.Dropout(config.routing_dropout)

    def forward(self, x: t.Tensor, input_tokens: Optional[t.IntTensor] = None) -> t.Tensor:
        """
        Parameters:
        x (t.Tensor): Hidden state input tensor. Shape (bs, hidden_size).
        input_tokens (Optional[t.IntTensor]): Original input tokens required if router_str is "hash". Shape (batch_size, hidden_size).

        Returns:
        t.Tensor: Mapping from input tokens to experts. Shape (batch_size, num_experts).

        Raises:
        AssertionError: If router_str is "hash" and input_tokens is None.
        """

        if self.router_enum == RouterEnums["hash"]:
            assert input_tokens is not None

            input_tokens = rearrange(input_tokens, "b s -> (b s)")
            clean_h = self.hash_router(input_tokens).float()  # bs num_experts

            return clean_h # bs num_experts
        else:
            clean_h = self.linear(x)  # bs num_experts
            clean_h = self.routing_dropout(clean_h)  # bs num_experts

            # Add gumbel noise to the routing logits to encourage exploration during training
            # self.training is inherited from nn.Module and is set by calling model.train() or model.eval()
            if self.training:
                gumbel_noise = -t.log(-t.log(t.rand_like(clean_h) + 1e-10) + 1e-10)
                h = (clean_h + gumbel_noise) / self.router_temperature
                return h  # bs num_experts

            return clean_h  # bs num_experts

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
