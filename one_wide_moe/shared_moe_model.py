from collections import OrderedDict
from itertools import chain
from typing import Any, Dict, List, Optional, OrderedDict, Protocol, Tuple, Union

import torch as t
import torch.nn as nn
import transformers
from einops import einsum, rearrange
from jaxtyping import Float, Int
from tensorboardX import SummaryWriter
from torch import nn
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as HFGPT2Block

from alibi.attention import AlibiUnidirectionalAttention
from alibi.transformer_block import ALiBiTransformerBlock
from general.basic_ffn import FFN
from general.norms import RMSNorm
from gpt.cached_attention import AttentionCache
from gpt.config import GPTConfig
from gpt.model import FullKeyValueCache, FullKeyValueCacheTensor
from gpt.transformer_block import GPT2Block
from mixture_of_experts.cache import MoECache
from moet_experiment.group_moe_layer import GroupMoELayer
from one_wide_moe.one_wide_config import OneWideConfig

config = OneWideConfig()
# tokeniser = tiktoken.encoding_for_model(config.tokeniser_string)
# Use the tokenizer from the TinyStories models
# Note using this tokenizer we only see the top 10K tokens. Hence the embedding matrix is only 10K x hidden_size really, even though it looks larger and we need to take this into account when counting parameters.
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")


class SharedParamsMoEModel(nn.Module):
    token_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    transformer_block: nn.Module
    moe_block: nn.Module
    vocab_size: int

    def __init__(
        self,
        *,
        ffn_dim_multiplier: int = 4,
        num_experts: int = 8,
        config: OneWideConfig = config,
        share_attention_layers: bool = True,
        share_moe_layers: bool = False,
        share_routers: bool = False,
        group_size: int = 1,
    ):
        super().__init__()
        self.config = config

        self.num_layers = config.num_total_layers

        attn_layers: OrderedDict[str, nn.Module] = OrderedDict()
        routers: OrderedDict[str, nn.Module] = OrderedDict()
        moe_layers: OrderedDict[str, nn.Module] = OrderedDict()

        # If we're sharing the routers we need to pass the router weights separately to the MoE layers
        self.router_weights_passed_separately = share_routers

        if share_attention_layers:
            single_attention_layer = AlibiUnidirectionalAttention(hidden_size=config.hidden_size, num_heads=config.num_attn_heads)
            attn_layers = OrderedDict(
                {
                    f"attn_layer_{i}": single_attention_layer
                    for i in range(self.config.num_total_layers)
                }
            )
        else:
            for i in range(self.config.num_total_layers):
                attn_layers[f"attn_layer_{i}"] = AlibiUnidirectionalAttention(hidden_size=config.hidden_size, num_heads=config.num_attn_heads)

        if share_moe_layers:
            single_moe_layer = GroupMoELayer(num_experts = num_experts, layer_id = f"moe_layer", router_weights_passed_separately=self.router_weights_passed_separately, ffn_dim_multiplier = ffn_dim_multiplier, group_size = group_size
                                             )
            attn_layers = OrderedDict(
                {
                    f"ffn_layer_{i}": single_moe_layer
                    for i in range(self.config.num_total_layers)
                }
            )

        else:
            for i in range(self.config.num_total_layers):
                moe_layers[f"ffn_layer_{i}"] = GroupMoELayer(num_experts = num_experts, layer_id = f"moe_layer_{i}", router_weights_passed_separately=self.router_weights_passed_separately, ffn_dim_multiplier = ffn_dim_multiplier, group_size = group_size
                                             )

        if share_routers:
            single_router = nn.Linear(config.hidden_size, num_experts)
            routers = OrderedDict(
                {
                    f"router_{i}": single_router
                    for i in range(self.config.num_total_layers)
                }
            )
        else:
            # Leave routers empty
            pass

        if (not share_moe_layers) and share_routers:
            raise ValueError("If sharing routers, must also share MoE layers")

        # Interleave the attention and ffn layers
        zipped_dicts = zip(attn_layers.items(), moe_layers.items())
        layers = OrderedDict(chain.from_iterable(zipped_dicts))


        # Define the model layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.sequential_layers = nn.Sequential(layers)
        self.routers = routers
        self.final_norm = RMSNorm(shape_without_batch=(config.hidden_size,))

    def unembed(self, z: Float[t.Tensor, "batch seq hidden"]) -> t.Tensor:
        out = einsum(
            z, self.token_embedding.weight, "b s h, v h -> b s v",
        )  # batch seq vocab_size
        return out

    def forward(self, input_ids: t.Tensor) -> Tuple[t.Tensor, Optional[MoECache]]:
        """
        x: batch seq_length
        """

        # Get token embeddings. Note since we're using ALiBI there are no positional embeddings here
        x = self.token_embedding(input_ids)

        for idx, layer in self.sequential_layers.named_children():
            if idx.startswith("attn_layer"):
                x, _attention_cache = layer(x)
            else:
                # For MoE layers
                if self.routers: # If routers are defined separately
                    router = self.routers[idx]
                    h = router(x)
                    x, _cache = layer(x = x, routing_weights = h)
                else:
                    # If routers are part of the MoE layer
                    x, _cache = layer(x)

        z = self.final_norm(x)

        # Unembed to get logits for each token
        out = self.unembed(z)  # batch seq vocab_size

        return out, None

    def load(self, model_path: str):
        self.load_state_dict(t.load(model_path))

    def save(self, model_path: str):
        t.save(self.state_dict(), model_path)
