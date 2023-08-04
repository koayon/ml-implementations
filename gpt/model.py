from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import tiktoken
import torch as t
import transformers
from einops import rearrange
from fancy_einsum import einsum
from jaxtyping import Float, Int
from numpy import isin
from tensorboardX import SummaryWriter
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as HFGPT2Block

import helpers
from gpt.cached_attention import AttentionCache
from gpt.transformer_block import GPT2Block

tokenizer = tiktoken.encoding_for_model("gpt2")

device = "cuda" if t.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class GPTConfig:
    """Constants used throughout the GPT2 model."""

    activation_function: str = "new_gelu"
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50257
    hidden_size: int = 768
    max_position_embeddings: int = 1024
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5


config = GPTConfig()

FullKeyValueCacheTensor = Float[t.Tensor, "layer 2 batch seq dim"]


class FullKeyValueCache(t.Tensor):
    """
    This class holds tensors of key and value vectors, to be used for caching.

    layer 2 batch seq dim
    """

    def __init__(self, cache_list: List[AttentionCache]):
        """Turn a list of AttentionCaches into a single tensor."""
        key_layer_caches = t.stack(
            [layer_cache.k for layer_cache in cache_list]
        )  # (layer, batch, seq, dim)
        value_layer_caches = t.stack([layer_cache.v for layer_cache in cache_list])

        # print("key_layer_caches.shape", key_layer_caches.shape)

        full_tensor = t.stack(
            [key_layer_caches, value_layer_caches], dim=1
        )  # (layer, 2, batch, seq, dim)
        super().__init__(full_tensor)

    def to_cache_list(self) -> List[AttentionCache]:
        """Turns the Full Key Value Cache back into a list of AttentionCaches.
        Input shape: (layer, 2, batch, seq, dim)

        """
        key_layer_caches = self[:, 0]  # layer, batch, seq, dim
        value_layer_caches = self[:, 1]  # layer, batch, seq, dim

        attn_caches = []
        for key_layer_cache, value_layer_cache in zip(
            key_layer_caches, value_layer_caches
        ):
            attn_caches.append(AttentionCache(k=key_layer_cache, v=value_layer_cache))

        return attn_caches  # list of AttentionCaches

    @property
    def k(self) -> t.Tensor:
        return self[:, 0]

    @property
    def v(self) -> t.Tensor:
        return self[:, 1]

    @property
    def batch(self) -> int:
        return self.shape[2]

    @property
    def seq_len(self) -> int:
        return self.shape[3]


class GPT2(nn.Module):
    """GPT-2 model.
    Reference: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    """

    token_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    final_layer_norm: nn.LayerNorm
    blocks: nn.ModuleList  # of GPT2Block

    def __init__(self, config: GPTConfig = config, with_pretrained_weights=True):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                GPT2Block(
                    layer_index=index,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    layer_norm_epsilon=config.layer_norm_epsilon,
                    activation_function=config.activation_function,
                )
                for index in range(config.num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

        if with_pretrained_weights:
            self.load_pretrained_weights()

    def forward(
        self, x: t.Tensor, cache: Optional[FullKeyValueCache] = None
    ) -> Tuple[t.Tensor, FullKeyValueCache]:
        """
        x: shape (batch, seq), dtype t.int64 - the token ids

        Return: shape (batch, seq, vocab_size), dtype t.float32- the output logits
        """
        layer_cache: AttentionCache
        cache_list: List[AttentionCache]

        if cache is None:
            cache_list = [None] * len(self.blocks)  # type: ignore
        else:
            cache_list = cache.to_cache_list()

        _batch_size, seq_len = x.shape

        # Combine the token and position embeddings for the embedding layer
        tokens = self.token_embedding(x)  # (batch, seq, hidden_size)
        positions = self.pos_embedding(
            t.arange(seq_len, device=device)
        )  # (batch, seq, hidden_size)
        x = tokens + positions
        x = self.dropout(x)  # batch, seq, hidden_size

        # Apply transformer blocks
        for layer_index, block in enumerate(self.blocks):
            x, layer_cache = block(
                x, layer_cache=cache_list[layer_index]
            )  # batch, seq, hidden_size
            cache_list[layer_index] = layer_cache

        y = self.final_layer_norm(x)  # batch, seq, hidden_size

        # Umbed the output back to the vocabulary size using the transpose of the token embedding as the umbedding matrix (i.e. tied embeddings)
        logits = einsum(
            "vocab_size hidden_size, batch seq hidden_size -> batch seq vocab_size",
            self.token_embedding.weight,
            y,
        )  # batch, seq, vocab_size

        full_cache = FullKeyValueCache(cache_list=cache_list)

        return logits, full_cache

    def load_pretrained_weights(self):
        """Load weights from OpenAI's pretrained model from HuggingFace."""

        hf_gpt = helpers.load_pretrained_gpt()
        for param in self.parameters():
            param.requires_grad_(False)

        # Embeddings (note the copy_ ensures that weights are copied in_place)
        self.token_embedding.weight.copy_(hf_gpt.transformer.wte.weight)
        self.pos_embedding.weight.copy_(hf_gpt.transformer.wpe.weight)
        self._copy_weight_bias(self.final_layer_norm, hf_gpt.transformer.ln_f)

        for my_block, hf_block in zip(self.blocks, hf_gpt.transformer.h):
            assert isinstance(hf_block, HFGPT2Block)
            assert isinstance(my_block, GPT2Block)

            # Copy attention weights
            self._copy_weight_bias(my_block.ln1, hf_block.ln_1)
            self._copy_weight_bias(
                my_block.attn.qkv_proj, hf_block.attn.c_attn, transpose=True
            )
            self._copy_weight_bias(
                my_block.attn.output_proj,
                hf_block.attn.c_proj,
                transpose=True,
            )

            # Copy MLP weights
            self._copy_weight_bias(my_block.MLP.ln2, hf_block.ln_2)
            self._copy_weight_bias(
                my_block.MLP.linear1, hf_block.mlp.c_fc, transpose=True
            )
            self._copy_weight_bias(
                my_block.MLP.linear2, hf_block.mlp.c_proj, transpose=True
            )

        for p in self.parameters():
            p.requires_grad_(True)

    def _copy_weight_bias(self, mine, theirs, transpose=False):
        mine.weight.copy_(theirs.weight.T if transpose else theirs.weight)
        if mine.bias is not None:
            mine.bias.copy_(theirs.bias)


if __name__ == "__main__":
    model = GPT2(config, with_pretrained_weights=True)
    x = t.randint(0, config.vocab_size, (1, 10))
    logits, _cache = model(x)
    print(logits)
    print(logits.shape)

    print(model.config)

    with SummaryWriter(comment="ModelArchitecture") as w:
        w.add_graph(model, (x,))
