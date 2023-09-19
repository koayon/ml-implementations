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
from general import device
from general.basic_ffn import FFN
from general.norms import RMSNorm
from gpt.cached_attention import AttentionCache
from gpt.config import GPTConfig
from gpt.model import FullKeyValueCache, FullKeyValueCacheTensor
from gpt.transformer_block import GPT2Block
from mixture_of_experts.cache import MoECache
from moet_experiment.group_moe_layer import GroupExpertLayer, GroupMoELayer, Router
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
        share_expert_layers: bool = False,
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
        # self.router_weights_passed_separately = share_routers

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

        if share_expert_layers:
            single_moe_layer = GroupExpertLayer(num_experts = num_experts, layer_id = f"moe_layer",  ffn_dim_multiplier = ffn_dim_multiplier, group_size = group_size
                                             )
            moe_layers = OrderedDict(
                {
                    f"moe_layer_{i}": single_moe_layer
                    for i in range(self.config.num_total_layers)
                }
            )

        else:
            for i in range(self.config.num_total_layers):
                moe_layers[f"moe_layer_{i}"] = GroupExpertLayer(num_experts = num_experts, layer_id = f"moe_layer_{i}", ffn_dim_multiplier = ffn_dim_multiplier, group_size = group_size
                                             )

        if share_routers:
            single_router = Router(num_experts = num_experts, router_str="linear", config = config)
            routers = OrderedDict(
                {
                    f"router_{i}": single_router
                    for i in range(self.config.num_total_layers)
                }
            )
        else:
            routers = OrderedDict(
                {
                    f"router_{i}": Router(num_experts = num_experts, router_str="linear", config = config)
                    for i in range(self.config.num_total_layers)
                }
            )

        # Interleave the attention and ffn layers
        zipped_dicts = zip(attn_layers.items(), moe_layers.items())
        layers = OrderedDict(chain.from_iterable(zipped_dicts))

        assert len(layers) == self.config.num_total_layers * 2

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

        batch_size, seq_len = input_ids.shape

        # Get token embeddings. Note since we're using ALiBI there are no positional embeddings here
        x = self.token_embedding(input_ids) # batch seq hidden

        for idx, layer in self.sequential_layers.named_children():
            if idx.startswith("attn_layer"):
                x, _attention_cache = layer(x) # batch seq hidden
            else:
                # MoE layers

                x = rearrange(x, "b s h -> (b s) h")

                # Get the router for this layer
                router_idx = f'router_{int(idx.split("_")[-1])}'
                router = self.routers[router_idx]
                router.to(device)

                # Get the router weights
                h = router(x) # (bs) num_experts

                # Pass the router weights to the MoE layer with x
                x, _cache = layer(x = x, routing_logits = h, batch_size = batch_size, seq_len = seq_len) # (batch seq) hidden

                x = rearrange(x, "(b s) h -> b s h", b = batch_size, s = seq_len)

        z = self.final_norm(x) # batch seq hidden

        # Unembed to get logits for each token
        out = self.unembed(z)  # batch seq vocab_size

        return out, None

    def load(self, model_path: str):
        self.load_state_dict(t.load(model_path))

    def save(self, model_path: str):
        t.save(self.state_dict(), model_path)
