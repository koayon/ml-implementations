from typing import Tuple

import torch as t
import transformers
from einops import rearrange
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import (
    MoECausalLMOutputWithPast,
    MoEModelOutput,
    Seq2SeqMoEModelOutput,
)

from mixture_of_experts.cache import (
    ExpertChoiceFullCache,
    ExpertChoiceLayerCache,
    SoftTokenMergeLayerCache,
    TokenChoiceFullCache,
    TokenChoiceLayerCache,
)
from moet_experiment.group_moe_layer import GroupExpertLayer

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-128")


def get_layer_cache(
    layer_router_logits: t.Tensor, layer_str: str
) -> TokenChoiceLayerCache:
    """_summary_

    Parameters
    ----------
    layer_router_logits : t.Tensor
        bs, num_experts
    layer_str : str
        _description_

    Returns
    -------
    TokenChoiceLayerCache
        Layer cache containing G, expert_assignments, P, routing_logits
    """
    batch_size, seq_len, num_experts = layer_router_logits.shape
    layer_router_logits = rearrange(
        layer_router_logits, "batch seq expert ->  (batch seq) expert"
    )
    S = t.softmax(layer_router_logits, dim=-1)

    # TODO: Extract _token_choice_routing_matrices to its own function
    expert_layer = GroupExpertLayer(
        num_experts=num_experts,
        layer_id=layer_str,
        # c=None, k=None
    )
    G, expert_assignments, P = expert_layer._token_choice_routing_matrices(
        S, batch_size, seq_len
    )

    layer_cache = TokenChoiceLayerCache(
        G=G,
        expert_assignments=expert_assignments,
        P=P,
        routing_logits=layer_router_logits,
    )
    return layer_cache


class TokenChoiceModelWrapper(PreTrainedModel):
    def __init__(self, config, model_name: str = "google/switch-base-128"):
        super().__init__(config)
        self.config = config

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        output_router_probs=True,
    ) -> Tuple[Seq2SeqMoEModelOutput, TokenChoiceFullCache]:
        out: Seq2SeqMoEModelOutput = self.model(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict,
            output_router_probs=output_router_probs,
        )

        router_logits_types = []
        if hasattr(out, "encoder_router_logits"):
            router_logits_types.append(out.encoder_router_logits)
            # layers tuple[batch_size, seq len, num_experts]

        if hasattr(out, "decoder_router_logits"):
            router_logits_types.append(out.decoder_router_logits)
            # layers tuple[batch_size, seq len, num_experts]

        if hasattr(out, "router_logits"):
            router_logits_types.append(out.router_logits)  #  type: ignore
            # layers tuple[batch_size, seq_len, num_experts]

        cache = {}

        # Convert this into a TokenChoiceLayerCache
        for router_logits in router_logits_types:
            for layer_idx, layer_router_logits in enumerate(router_logits):
                layer_str = f"encoder_layer_{layer_idx}"

                layer_cache = get_layer_cache(layer_router_logits, layer_str)

                cache[layer_str] = layer_cache

        full_cache = TokenChoiceFullCache(cache)

        return out, full_cache
