from json import load
from typing import Optional

import torch as t
from einops import einsum, rearrange, reduce
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel

from general import device
from mixture_of_experts.cache import TokenChoiceFullCache
from moet_experiment.model import MoET


def load_balancing_aux_loss_function(moe_cache: TokenChoiceFullCache) -> float:
    """Load balancing auxiliary loss.

    Reference: Shazeer et al (2017) and ST-MoE: Designing Stable and Transferable Sparse Expert Models, https://arxiv.org/pdf/2202.08906.pdf

    Parameters
    ----------
    moe_cache : MoEFullCache
        MoE cache containing G, assignments and routing logits

    Returns
    -------
    float
        Load balancing auxiliary loss
    """
    num_experts = moe_cache.num_experts
    num_tokens = moe_cache.num_tokens

    total_tokens_per_expert = reduce(moe_cache.P, "layer expert batch_seq k -> layer expert", "sum")  # [layer, expert]
    frac_tokens_per_expert = total_tokens_per_expert / num_tokens

    routing_probs = F.softmax(moe_cache.routing_weights_tensor, dim=-1)  # [layer, num_experts, batch_seq]

    total_router_prob_per_expert = reduce(routing_probs, "layer num_experts batch_seq -> layer num_experts", "sum")  # [layer, num_experts]
    frac_router_prob_per_expert = total_router_prob_per_expert / num_tokens

    # Dot product
    lb_loss = num_experts * einsum(frac_tokens_per_expert, frac_router_prob_per_expert, "layer expert, layer expert ->")

    return lb_loss.item()

def router_z_loss_function(moe_cache: TokenChoiceFullCache) -> float:
    """Router z loss.

    Reference: ST-MoE: Designing Stable and Transferable Sparse Expert Models, https://arxiv.org/pdf/2202.08906.pdf

    Note that we've chosen not to multiply divide through the num_experts here.

    Parameters
    ----------
    moe_cache : MoEFullCache
        MoE cache containing G, assignments and routing logits

    Returns
    -------
    float
        Router z loss
    """
    router_logits = moe_cache.routing_weights_tensor # [layer, num_experts, batch_seq]
    lse_logits = t.logsumexp(router_logits, dim=-1)  # [layer, num_experts]
    squared_lse_logits = lse_logits ** 2
    z_loss = einsum(squared_lse_logits, "layer num_experts ->") / (moe_cache.num_tokens)

    return z_loss.item()

class MoETHFConfig(PretrainedConfig):

    def __init__(
        self,
        block_type="MoE",
        layers: int = 8,
        **kwargs,
    ):

        self.block_type = block_type
        self.layers = layers
        super().__init__(**kwargs)


class MoET_hf(PreTrainedModel):
    def __init__(self, hf_config: MoETHFConfig = MoETHFConfig()):
        super().__init__(hf_config)
        self.hf_config = hf_config

        self.model = MoET()

        self.lb_coef = self.model.config.lb_coef
        self.z_coef = self.model.config.z_coef

    def forward(self, input_ids: t.Tensor, attention_mask: t.Tensor, return_loss: bool = True, **kwargs):
        """Forward function for hf wrapped model.

        Parameters
        ----------
        input_ids : Int[t.Tensor, "batch_size, seq_len"]
            Input tokens
        attention_mask : t.Tensor
            Attention mask
        return_loss : bool, optional
            Whether to return the model's loss in the output, by default True

        Returns
        -------
        dict
            Output dict
        """
        # Forward pass
        logits, moe_cache = self.model(input_ids, attention_mask)

        if return_loss:
            labels = input_ids[:, 1:]
            pred_logits = logits[:, :-1, :]

            flattened_logits = rearrange(pred_logits, "b s v -> (b s) v")
            flattened_labels = rearrange(labels, "b s -> (b s)")

            cross_entropy_loss = F.cross_entropy(flattened_logits, flattened_labels)

            load_balancing_aux_loss = load_balancing_aux_loss_function(moe_cache)
            router_z_loss = router_z_loss_function(moe_cache)

            loss = cross_entropy_loss + self.lb_coef * load_balancing_aux_loss + self.z_coef * router_z_loss

            return {"loss": loss, "cross_entropy_loss": cross_entropy_loss,
                    "load_balancing_aux_loss": load_balancing_aux_loss,
                    "router_z_loss": router_z_loss,
                    "logits": logits}
        else:
            return {"logits": logits}
