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
        slots_per_expert: int = 1,
        ffn_dim_multiplier: int = 4,
        ffn_ratio: float = 2 / 3,
        use_expert_choice: Optional[bool] = None,
    ) -> None:
        super().__init__()

        assert num_experts % group_size == 0

        self.slots_per_expert = slots_per_expert

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
        h = router_weights # (b s) num_experts slots

        # Dispatch weights
        D = t.softmax(h, dim=-1)  # bs num_experts slots
        # Combine weights
        C = t.softmax(h, dim = 0) # bs num_experts slots

        # Get the weighted averaged X for each slot. This is the input to the experts
        x_tilda = einsum(D, x, "bs num_experts slots, bs hidden_size -> num_experts slots hidden_size")

        # USE EXPERTS
        # forward the relevant tokens through the relevant expert
        # E_list is denoted y_tilda in the paper
        E_list = [self.experts[expert_num](x_tilda[expert_num]) for expert_num in range(self.num_experts)]  # num_experts list[slots hidden_size]

        E = t.stack(E_list, dim=0)  # num_experts slots hidden_size

        # Combine the outputs of the experts with the weights from the router to get the final output with the correct shapes
        y = einsum(C, E, "bs num_experts slots, num_experts slots hidden_size -> bs hidden_size")

        return y, None
