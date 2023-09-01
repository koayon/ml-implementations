from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch as t
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from general.swiglu_ffn import SwiGLUFFN
from helpers import einsum
from mixture_of_experts.cache import MoELayerCache
from mixture_of_experts.routers import HashRouter
from moet_experiment.moet_config import MoETConfig

ROUTERS = ["linear", "learned", "hash"]
MULT = 4
RATIO = 2 / 3

device = "cuda" if t.cuda.is_available() else "cpu"

# config = MoETConfig()

class Router(nn.Module):
    def __init__(self, *, num_experts: int, router_str: str, config: MoETConfig):
        super().__init__()
        self.config = config

        self.router_str = router_str
        self.router_temperature = config.router_temperature
        self.num_experts = num_experts
        self.hidden_size = config.hidden_size

        if router_str in ("linear", "learned"):
            # Define the linear router
            self.linear = nn.Linear(self.hidden_size, self.num_experts)
        elif router_str == "hash":
            # Build the hash router
            assert config.num_experts_hash == self.num_experts
            self.hash_router = HashRouter(config=config, num_experts=config.num_experts_hash)
            # self.hash_router.build_random_hash()
        else:
            raise ValueError(f"Unknown router {router_str}. Please choose from {ROUTERS}")

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

            if self.router_str == "hash":
                assert input_tokens is not None

                input_tokens = rearrange(input_tokens, "b s -> (b s)")
                clean_h = self.hash_router(input_tokens)  # bs num_experts
            else:
                clean_h = self.linear(x)  # bs num_experts

            clean_h = self.routing_dropout(clean_h)  # bs num_experts

            # Add gumbel noise to the routing logits to encourage exploration during training
            # self.training is inherited from nn.Module and is set by calling model.train() or model.eval()
            if self.training:
                gumbel_noise = -t.log(-t.log(t.rand_like(clean_h) + 1e-10) + 1e-10)
                h = (clean_h + gumbel_noise) / self.router_temperature
                return h # bs num_experts
            else:
                return clean_h # bs num_experts


class GroupExpertChoiceMoELayer(nn.Module):
    experts: nn.ModuleList
    up_experts: list[nn.Module]
    down_experts: list[nn.Module]

    def __init__(
        self,
        *,
        num_experts: int,
        layer_id: str,
        router_str: str = "linear",
        config: MoETConfig = MoETConfig(),
        group_size: int = 1,
        k: int = 0,  # topk
        c: float = 1.0,  # capacity factor
        use_expert_choice: bool = True,
    ) -> None:
        super().__init__()

        # Either choose k or set it from the capacity factor (c)
        assert (k > 0) or (c > 0)
        assert num_experts % group_size == 0
        assert router_str in ROUTERS

        self.num_experts = num_experts
        self.num_expert_groups = num_experts // group_size
        self.use_expert_choice = use_expert_choice

        self.layer_id = layer_id

        self.hidden_size = config.hidden_size
        self.batch_size = config.batch_size
        self.seq_len = config.max_position_embeddings

        self.router = Router(
            num_experts=num_experts,
            router_str=router_str,
            config=config,
        )

        if group_size > 1:

            # Grouped experts
            up_experts = [
                nn.Linear(
                    in_features=self.hidden_size,
                    out_features=int(self.hidden_size * MULT * RATIO),
                )
                for _ in range(self.num_expert_groups)
            ]
            down_experts = [
                nn.Linear(
                    in_features=int(self.hidden_size * MULT * RATIO),
                    out_features=self.hidden_size,
                )
                for _ in range(self.num_experts)
            ]
            silu = nn.SiLU()
            expert_dropout = nn.Dropout(config.expert_dropout)

            experts = []
            for expert_num in range(self.num_experts):
                # group_size experts share the same up layer. Each has a unique down layer.
                expert_group_num = expert_num // group_size
                experts.append(
                    nn.Sequential(
                        up_experts[expert_group_num],
                        silu,
                        down_experts[expert_num],
                        expert_dropout,
                    )
                )
            self.experts = nn.ModuleList(experts)
        else:

            # Regular FFN experts
            expert = SwiGLUFFN(
                in_features=self.hidden_size, dropout=config.expert_dropout
            )
            self.experts = nn.ModuleList([expert for _ in range(self.num_experts)])

        # Use the capacity factor to set k
        if c > 0:
            self.k = int(self.batch_size * self.seq_len * c // self.num_experts)
        else:
            self.k = k

    def forward_individual_expert_choice(
        self,
        expert_num: int,
        chosen_token_index: t.Tensor,
        G: t.Tensor,
        x: t.Tensor,
    ) -> t.Tensor:
        """_summary_

        Parameters
        ----------
        expert_num : int
            The expert that we're using for forward
        chosen_token_index : t.Tensor
            Shape (k, num_experts)
        G : t.Tensor
            Shape (k, num_experts)
        x : t.Tensor
            Shape (bs hidden_size)

        Returns
        -------
        t.Tensor
            Shape (bs hidden_size)
        """

        batch_seq_size = x.shape[0]

        # Select top-k expert, with one-hot vector. P is the permutation matrix
        P: t.Tensor = F.one_hot(
            chosen_token_index, num_classes=batch_seq_size
        )  # k num_experts (one-hot)

        P = rearrange(P, "k num_experts bs -> bs k num_experts")  # bs k num_experts

        # Extract relevant sections of P, G
        P_expert = P[..., expert_num] * 1.0  # bs k
        G_expert = G[:, expert_num]  # k

        tokens_for_expert = einsum(
            "bs k, bs hidden_size -> k hidden_size", P_expert, x
        )  # k hidden_size

        # TODO: All of above could be moved to forward (or another method) so that this
        # function represents the abstraction boundary of "things which happen on the expert GPU"
        # It also seems like this could all be done in one go with tensors rather than per expert

        # Forward pass through the expert network
        E = self.experts[expert_num](tokens_for_expert)
        # k hidden_size

        x_out = einsum(
            "bs k, k, k hidden_size -> bs hidden_size", P_expert, G_expert, E
        )  # bs hidden_size

        return x_out

    def forward(
        self, x: t.Tensor, input_tokens: Optional[t.Tensor] = None
    ) -> Tuple[t.Tensor, MoELayerCache]:
        """
        Args:
            x: batch seq hidden_size
            router: hidden_size num_experts
            input_tokens: batch seq, the original input tokens

        Returns:
            x: batch, seq, hidden_size
            cache: MoELayerCache: G, token_assignments, routing_weights
        """

        batch_size, seq_len, _hidden_size = x.shape

        # If there aren't enough tokens in the input to select top k, reduce k
        self.k = min(int(self.k), (batch_size * seq_len))

        x = rearrange(x, "b s h -> (b s) h")

        # Get routing weights, h
        h = self.router(x, input_tokens)  # (b s) num_experts

        S = t.softmax(h, dim=-1)  # bs num_experts

        if self.use_expert_choice:
            # Expert Choice: Each expert picks the top-k tokens it wants to process. In the moment that we pick the topk across the sequence dimension, we share some information across the time/seq dimension which would be a problem for autoregressive models (it's allowing the model to cheat). This is best used for non-autoregressive models.
            G, chosen_token_index = t.topk(S, k=self.k, dim=0)  # k num_experts each
        else:
            # Token-choice: Each token picks the top-k experts it wants to process it. This is best used for autoregressive models.
            # If we balance the experts enough with our load-balancing and set a sufficiently high k, then there is very little information shared across the sequence dimension.
            # The only information that can be shared is that a token is dropped.
            # Having a high expert dropout rate also helps here (a token doesn't know if it was dropped because of later tokens also wanting that expert or because of the expert dropout rate).
            # Another way to mitigate this is to fill up the experts left to right so any dropped tokens are dropped from the end of the sequence and the knowledge that a token was dropped is passed only backwards not forwards in time.
            G, chosen_expert_index = t.topk(S, k=self.k, dim=1)  # bs k each
            raise NotImplementedError

        # Store cache for interpretability and loss function
        layer_cache = MoELayerCache(
            G=G,
            token_assignments=chosen_token_index,
            routing_weights=h,
        )

        # Collect expert results from parallelised expert forward
        expert_results = [
            self.forward_individual_expert_choice(
                expert_num=expert_num, G=G, x=x, chosen_token_index=chosen_token_index
            )
            for expert_num in range(self.num_experts)
        ]  # expert list[bs hidden_size]

        # Aggregate expert results together
        expert_results_stack = t.stack(
            expert_results, dim=0
        )  # num_experts bs hidden_size
        y = t.sum(expert_results_stack, dim=0)  # bs hidden_size

        y = rearrange(y, "(b s) h -> b s h", b=batch_size)  # batch seq hidden_size

        return y, layer_cache


def main():
    expert_layer = GroupExpertChoiceMoELayer(
        k=2,
        layer_id="expert-layer-1",
        num_experts=4,
        router_str="linear",
    )

    x = t.rand(size=(3, 4, 16))  # batch seq hidden_size

    y, cache = expert_layer(x, cache={})
    print(cache)


if __name__ == "__main__":
    main()
