from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Tuple

import tiktoken
import torch as t
import transformers
from einops import rearrange
from fancy_einsum import einsum
from jaxtyping import Float, Int
from tensorboardX import SummaryWriter
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as HFGPT2Block

import helpers
from alibi.transformer_block import GPT2Block
from gpt.config import GPTConfig
from gpt.model import FullKeyValueCache

tokenizer = tiktoken.encoding_for_model("gpt2")

device = "cuda" if t.cuda.is_available() else "cpu"


config = GPTConfig()


class AlibiGPT(nn.Module):
    """GPT-style decoder model with ALiBi (ATTENTION WITH LINEAR BIASES)
    From Train Short, Test Long paper.

    Reference: https://arxiv.org/pdf/2108.12409.pdf
    """

    token_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    final_layer_norm: nn.LayerNorm
    blocks: nn.ModuleList  # of GPT2Block

    def __init__(self, config: GPTConfig = config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

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

    def forward(
        self, x: t.Tensor, cache: Optional[FullKeyValueCache] = None
    ) -> Tuple[t.Tensor, Optional[FullKeyValueCache]]:
        """
        x: shape (batch, seq), dtype t.int64 - the token ids

        Return: shape (batch, seq, vocab_size), dtype t.float32- the output logits
        """

        if cache is None:
            cache_list = [None] * len(self.blocks)
        else:
            cache_list = cache.to_cache_list()

        _batch_size, seq_len = x.shape

        # Combine the token and position embeddings for the embedding layer
        embedding_tokens = self.token_embedding(x)  # (batch, seq, hidden_size)
        x = self.dropout(x=embedding_tokens)  # batch, seq, hidden_size

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

        # full_cache = FullKeyValueCache.from_cache_list(cache_list=cache_list)

        return logits, None


if __name__ == "__main__":
    model = AlibiGPT(config)
    x = t.randint(0, config.vocab_size, (1, 10))
    logits, _cache = model(x)
    print(logits)
    print(logits.shape)

    print(model.config)

    with SummaryWriter(comment="ModelArchitecture") as w:
        w.add_graph(model, (x,))
