from collections import OrderedDict
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

from alibi.transformer_block import ALiBiTransformerBlock
from general.norms import RMSNorm
from gpt.cached_attention import AttentionCache
from gpt.config import GPTConfig
from gpt.model import FullKeyValueCache, FullKeyValueCacheTensor
from gpt.transformer_block import GPT2Block
from one_wide_moe.one_wide_config import OneWideConfig

config = OneWideConfig()
# tokeniser = tiktoken.encoding_for_model(config.tokeniser_string)
# Use the tokenizer from the TinyStories models
# Note using this tokenizer we only see the top 10K tokens. Hence the embedding matrix is only 10K x hidden_size really, even though it looks larger and we need to take this into account when counting parameters.
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")


class RegularDenseModel(nn.Module):
    token_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    transformer_block: nn.Module
    moe_block: nn.Module
    vocab_size: int

    def __init__(
        self,
        *,
        config: OneWideConfig = config,
    ):
        super().__init__()
        self.config = config

        self.num_layers = config.num_total_layers

        layers: OrderedDict[str, nn.Module] = OrderedDict()

        for i in range(self.config.num_total_layers):
            layers[f"transformer_block_{i}"] = ALiBiTransformerBlock(layer_index=i, hidden_size = config.hidden_size, num_heads = config.num_attn_heads)

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.sequential_layers = nn.Sequential(layers)
        self.final_norm = RMSNorm(shape_without_batch=(config.hidden_size,))

    def unembed(self, z: Float[t.Tensor, "batch seq hidden"]) -> t.Tensor:
        out = einsum(
            z, self.token_embedding.weight, "b s h, v h -> b s v",
        )  # batch seq vocab_size
        return out

    def forward(self, input_ids: t.Tensor) -> t.Tensor:
        """
        x: batch seq_length
        """

        # Get token embeddings. Note since we're using ALiBI there are no positional embeddings here
        x = self.token_embedding(input_ids)

        for idx, layer in self.sequential_layers.named_children():
            # Layer types are AliBiTransformerBlock
            x, _attention_cache = layer(x)
        z = self.final_norm(x)

        # Unembed to get logits for each token
        out = self.unembed(z)  # batch seq vocab_size

        return out

    def load(self, model_path: str):
        self.load_state_dict(t.load(model_path))

    def save(self, model_path: str):
        t.save(self.state_dict(), model_path)
