from typing import Optional, Tuple, Union

import torch as t
from einops import einsum, rearrange, repeat
from torch import nn

from general import device
from general.norms import RMSNorm
from general.swiglu_ffn import SwiGLUFFN
from mixture_of_experts.cache import MoECache, MoELayerCache, SMEARLayerCache
from mixture_of_experts.experts import Expert, ExpertFromWeights, ExpertList
from moet_experiment.group_moe_layer import get_experts
from moet_experiment.moet_config import MoETConfig


class SoftMergingExpertLayer(nn.Module):
    experts: ExpertList

    def __init__(
        self,
        *,
        num_experts: int,
        layer_id: str,
        config: MoETConfig = MoETConfig(),
        group_size: int = 1,
        k: int = 1,
        ffn_dim_multiplier: int = 4,
        ffn_ratio: float = 2 / 3,
        act_fn: nn.Module = nn.SiLU(),
    ) -> None:
        """Soft Merging of Experts Layer inspired by Soft Merging of Experts with Adaptive Routing Paper

        Reference: https://arxiv.org/pdf/2306.03745.pdf

        Parameters
        ----------
        num_experts : int
            Number of experts in the layer
        layer_id : str
            Layer id
        config : MoETConfig, optional
            Config dataclass, by default MoETConfig()
        group_size : int, optional
            Number of expert groups - experts in the same group share the same down projection but have different up projections, by default 1
        k : int, optional
            _description_, by default 1
        ffn_dim_multiplier : int, optional
            _description_, by default 4
        ffn_ratio : float, optional
            _description_, by default 2/3
        """

        super().__init__()


        self.k = k

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
            act_fn=act_fn,
        )
        self.act_fn = act_fn
        self.dropout = config.expert_dropout


    def forward(
        self, x: t.Tensor, routing_logits: Optional[t.Tensor] = None, merged_expert: Optional[Union[Expert, ExpertFromWeights]] = None
    ) -> Tuple[t.Tensor, Optional[SMEARLayerCache], Union[Expert, ExpertFromWeights]]:
        """
        Soft MoE as given in From Sparse to Soft Mixtures of Experts.

        Given some experts and routing logits we merge the experts with a weighted average and then forward the input through the merged expert.

        This approach requires less FLOPs/GPUs than an ensemble since we only have to forward one rather than N experts.
        However, the averaging operation takes some FLOPs in itself meaning that it's generally not faster than an ensemble overall unless we use the same expert for multiple tokens in a sequence (and/or over multiple batches). This is potentially okay in a Chat scenario where we can get the routing weights based on the prompt and then use the same expert for all tokens in the response. Overall this would seem to be a weakness of the approach however.

        Args:
            x: batch_seq hidden_size
            routing_logits: hidden_size num_experts
            merged_expert: batch_seq num_experts slots

        Returns:
            x: batch_seq, hidden_size
            layer_cache: SMEARLayerCache
            merged_expert: nn.Module
        """

        bs, _hidden_size = x.shape
        if merged_expert is None:
            assert routing_logits is not None
            assert routing_logits.shape == (bs, self.num_experts)
            # Routing logits are called phi in the paper # (b s) num_experts

            # Define the routing matrix, h
            routing_matrix = t.softmax(routing_logits, dim=-1)  # bs num_experts

            layer_cache = SMEARLayerCache(routing_matrix=routing_matrix, routing_logits=routing_logits)

            # Define merged (smeared) expert module (considering final token)
            merged_expert_weights_and_biases = self.experts.merge_weights_and_biases(merging_weights = routing_matrix[-1])
            merged_expert = ExpertFromWeights(expert_linear_params=merged_expert_weights_and_biases, act_fn=self.act_fn, dropout=self.dropout)
        else:
            layer_cache = None

        # USE EXPERT
        # forward the relevant tokens through the merged expert
        y = merged_expert(x)

        return y, layer_cache, merged_expert
