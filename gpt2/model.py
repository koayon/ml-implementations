from dataclasses import dataclass
from typing import Any, Optional

import torch as t
from torch import nn
from transformer_block import GPT2Block

import utils


@dataclass(frozen=True)
class GPTConfig:
    """Constants used throughout the GPT2 model."""

    activation_function: str = "gelu"
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50257
    hidden_size: int = 768
    max_position_embeddings: int = 1024
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5


config = GPTConfig()


class GPT2(nn.Module):
    token_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    ln: nn.LayerNorm
    blocks: utils.StaticModuleList[GPT2Block]

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks: utils.StaticModuleList[GPT2Block] = utils.StaticModuleList(
            [
                GPT2Block(
                    config.hidden_size,
                    config.num_heads,
                    config.dropout,
                    config.layer_norm_epsilon,
                    config.activation_function,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, x: t.Tensor, cache: Optional[Any] = None) -> t.Tensor:
        """
        x: shape (batch, seq), dtype t.int64 - the token ids

        Return: shape (batch, seq, vocab_size), dtype t.float32- the output logits
        """
        _, seq_length = x.shape
        if cache is None:
            pos_start = 0
        else:
            _, _, cached_seq_length, _ = cache[0].k.shape
            assert (
                cached_seq_length == 0 or seq_length == 1
            ), "Pass in one seq at a time after loading cache"
            pos_start = cached_seq_length
        pos = t.arange(pos_start, pos_start + seq_length).to(x.device)
        x = self.token_embedding(x) + self.pos_embedding(pos)
        x = self.dropout(x)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=cache[i] if cache else None)
        x = self.ln(x)
        x = t.einsum("bnl, vl -> bnv", x, self.token_embedding.weight)
        return x
