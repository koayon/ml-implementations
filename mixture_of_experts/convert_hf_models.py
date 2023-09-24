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
    SoftTokenMergeLayerCache,
    TokenChoiceFullCache,
    TokenChoiceLayerCache,
)
from moet_experiment.group_moe_layer import GroupExpertLayer

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-128")


def get_layer_cache(
    layer_router_logits: t.Tensor, layer_str: str
) -> TokenChoiceLayerCache:
    batch_size, seq_len, num_experts = layer_router_logits.shape
    layer_router_logits = rearrange(
        layer_router_logits, "batch seq expert ->  (batch seq) expert"
    )
    S = t.softmax(layer_router_logits, dim=-1)

    # TODO: Extract to its own function
    GEL = GroupExpertLayer(
        num_experts=num_experts,
        layer_id=layer_str,
        # c=None, k=None
    )
    G, expert_assignments, P = GEL._token_choice_routing_matrices(
        S, batch_size, seq_len
    )

    layer_cache = TokenChoiceLayerCache(
        G=G,
        expert_assignments=expert_assignments,
        P=P,
        routing_logits=layer_router_logits,
    )
    return layer_cache


class TokenChoiceMoE(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "roneneldan/TinyStories-8M"
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-128")

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
    ):
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
        )

        encoder_router_logits = (
            out.encoder_router_logits
        )  # layers tuple[batch_size, seq len, num_experts]
        decoder_router_logits = (
            out.decoder_router_logits
        )  # layers tuple[batch_size, seq_len, num_experts]
        assert encoder_router_logits and decoder_router_logits

        cache = {}

        # Convert this into a TokenChoiceLayerCache
        for layer_idx, layer_router_logits in enumerate(encoder_router_logits):
            layer_str = f"encoder_layer_{layer_idx}"

            layer_cache = get_layer_cache(layer_router_logits, layer_str)

            cache[layer_str] = layer_cache

        for layer_idx, layer_router_logits in enumerate(decoder_router_logits):
            layer_str = f"decoder_layer_{layer_idx}"

            layer_cache = get_layer_cache(layer_router_logits, layer_str)

            cache[layer_str] = layer_cache

        full_cache = TokenChoiceFullCache(cache)

        return {}
