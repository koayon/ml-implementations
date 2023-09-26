from typing import Optional, Tuple

import torch as t
from einops import einsum, rearrange, repeat
from torch import nn

from general import device
from mixture_of_experts.cache import MoELayerCache, SoftTokenMergeLayerCache
from moet_experiment.group_moe_layer import get_experts
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
        slots_per_expert: int = 1,
        ffn_dim_multiplier: int = 4,
        ffn_ratio: float = 2 / 3,
    ) -> None:
        super().__init__()

        self.slots_per_expert = slots_per_expert

        self.num_experts = num_experts

        self.layer_id = layer_id

        self.hidden_size = config.hidden_size
        self.batch_size = config.batch_size
        self.seq_len = config.max_position_embeddings

        self.experts = get_experts(
            num_experts=num_experts,
            hidden_size=self.hidden_size,
            ffn_dim_multiplier=ffn_dim_multiplier,
            ffn_ratio=ffn_ratio,
            group_size=group_size,
            dropout=config.expert_dropout,
        )

    def forward(
        self, x: t.Tensor, routing_logits: t.Tensor
    ) -> Tuple[t.Tensor, MoELayerCache]:
        """
        Soft MoE as given in From Sparse to Soft Mixtures of Experts

        Args:
            x: batch_seq hidden_size
            router: hidden_size num_experts
            routing_logits: batch_seq num_experts slots

        Returns:
            x: batch_seq, hidden_size
            MoELayerCache
            Either an ExpertChoiceLayerCache or a TokenChoiceLayerCache depending on the value of self.use_expert_choice
            Contains:
                G: (depends on self.use_expert_choice)
                assignments: (depends on self.use_expert_choice)

                routing_logits: batch_seq num_experts slots
                    Also called phi. These are the logits used in the loss function.

        """

        bs, _hidden_size = x.shape

        assert routing_logits.shape == (bs, self.num_experts, self.slots_per_expert)
        # Routing logits are called phi in the paper # (b s) num_experts slots

        # Rearrange routing logits to ensure we don't share information across the batch dim.
        routing_logits = rearrange(
            routing_logits,
            "(batch seq) num_experts slots -> batch seq (num_experts slots)",
        )

        # Define Dispatch weights (D), softmax over columns of X @ phi, mixing tokens
        dispatch_weights = t.softmax(
            routing_logits, dim=1
        )  # batch seq (num_experts slots)
        dispatch_weights = rearrange(
            dispatch_weights,
            "batch seq (num_experts slots) -> (batch seq) num_experts slots",
            slots=self.slots_per_expert,
        )

        # Define Combine weights (C), softmax over rows of X @ phi, mixing expert slots
        combine_weights = t.softmax(
            routing_logits, dim=-1
        )  # batch seq (num_experts slots)
        combine_weights = rearrange(
            combine_weights,
            "batch seq (num_experts slots) -> (batch seq) num_experts slots",
            slots=self.slots_per_expert,
        )

        layer_cache = SoftTokenMergeLayerCache(
            D=dispatch_weights, C=combine_weights, routing_logits=routing_logits
        )

        # Get the weighted averaged X for each slot. This is the input to the experts
        x_tilda = einsum(
            dispatch_weights,
            x,
            "bs num_experts slots, bs hidden_size -> num_experts slots hidden_size",
        )

        # USE EXPERTS
        # forward the relevant tokens through the relevant expert
        # E_list is denoted y_tilda in the paper
        E_list = [
            self.experts[expert_num](x_tilda[expert_num])
            for expert_num in range(self.num_experts)
        ]  # num_experts list[slots hidden_size]

        E = t.stack(E_list, dim=0)  # num_experts slots hidden_size

        # Combine the outputs of the experts with the weights from the router to get the final output with the correct shapes
        y = einsum(
            combine_weights,
            E,
            "bs num_experts slots, num_experts slots hidden_size -> bs hidden_size",
        )

        return y, layer_cache
