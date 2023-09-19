from typing import Optional, Tuple

import torch as t
from einops import einsum, rearrange, repeat
from torch import nn

from general import device
from general.norms import RMSNorm
from general.swiglu_ffn import SwiGLUFFN
from mixture_of_experts.cache import MoECache, MoELayerCache
from moet_experiment.moet_config import MoETConfig


class SoftExpertLayer(nn.Module):
    experts: nn.ModuleList
    up_experts: list[nn.Module]
    down_experts: list[nn.Module]

    def __init__(
        self,
        *,
        num_experts: int,
        layer_id: str,
        config: MoETConfig = MoETConfig(),
        group_size: int = 1,
        num_slots: int = 1,
        ffn_dim_multiplier: int = 4,
        ffn_ratio: float = 2 / 3,
        use_expert_choice: Optional[bool] = None,
    ) -> None:
        super().__init__()

        assert num_experts % group_size == 0

        self.num_slots = num_slots

        self.num_experts = num_experts
        self.num_expert_groups = num_experts // group_size
        self.use_expert_choice = use_expert_choice if use_expert_choice is not None else config.use_expert_choice

        self.layer_id = layer_id

        self.hidden_size = config.hidden_size
        self.batch_size = config.batch_size
        self.seq_len = config.max_position_embeddings

        if group_size > 1:

            # Grouped experts
            up_experts = [
                nn.Linear(
                    in_features=self.hidden_size,
                    out_features=int(self.hidden_size * ffn_dim_multiplier * ffn_ratio),
                )
                for _ in range(self.num_expert_groups)
            ]
            down_experts = [
                nn.Linear(
                    in_features=int(self.hidden_size * ffn_dim_multiplier * ffn_ratio),
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

    def forward(
        self, x: t.Tensor, router_weights: t.Tensor, batch_size: int, seq_len: int
    ) -> Tuple[t.Tensor, MoELayerCache]:
        """
        Soft MoE as given in

        Args:
            x: batch seq hidden_size
            router: hidden_size num_experts
            input_tokens: batch seq, the original input tokens

        Returns:
            x: batch, seq, hidden_size
            MoELayerCache
            Either an ExpertChoiceLayerCache or a TokenChoiceLayerCache depending on the value of self.use_expert_choice
            Contains:
                G: (depends on self.use_expert_choice)
                assignments: (depends on self.use_expert_choice)

                routing_weights: (batch seq) num_experts
                    Also called h. These are the logits used in the loss function.

        """

        bs, _hidden_size = x.shape

        # x = rearrange(x, "b s h -> (b s) h")

        assert router_weights.shape == (bs, self.num_experts)
        h = router_weights # (b s) num_experts

        S = t.softmax(h, dim=-1)  # bs num_experts

        G, chosen_token_index, P = self._soft_routing_matrices(S = S, batch_size = batch_size, seq_len = seq_len)

        # Soft MoE cache
        # layer_cache = ExpertChoiceLayerCache(
        #         G=G,
        #         P = P,
        #         token_assignments=chosen_token_index,
        #         routing_weights=h,
        #     )


        tokens_for_expert = einsum(
                P.float(), x, "bs k expert, bs hidden_size -> expert k hidden_size",
            )  # expert k hidden_size

        # USE EXPERTS
        # forward the relevant tokens through the relevant expert
        E_list = [self.experts[expert_num](tokens_for_expert[expert_num]) for expert_num in range(self.num_experts)]  # num_experts list[k hidden_size]

        E = t.stack(E_list, dim=0)  # num_experts k hidden_size

        # Put the results back in the right order with the permutation matrix P and weight them correctly with the routing weights G

        if self.use_expert_choice:
            # P [bs k num_experts]
            # G [k num_experts]
            # E [num_experts k hidden_size]
            y = einsum(
            P.float(), G, E, "bs k expert, k expert, expert k hidden_size -> bs hidden_size",)
        else:
            # P [bs k num_experts]
            # G [bs k]
            # E [num_experts k hidden_size]
            y = einsum(
             P.float(), G, E, "bs k expert, bs k, expert k hidden_size -> bs hidden_size",)

        y = rearrange(y, "(batch seq) hidden_size -> batch seq hidden_size", batch=batch_size)

        return y, None

    def _soft_routing_matrices(self, S: t.Tensor, batch_size: int, seq_len: int) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        """
        S: bs num_experts
        """
        raise NotImplementedError
