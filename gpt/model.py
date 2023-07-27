from dataclasses import dataclass
from typing import Any, Optional

import tiktoken
import torch as t
import transformers
from fancy_einsum import einsum
from tensorboardX import SummaryWriter
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as HFGPT2Block

import helpers
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


class GPT2(nn.Module):
    """GPT-2 model.
    Reference: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    """

    token_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    final_layer_norm: nn.LayerNorm
    blocks: helpers.StaticModuleList[GPT2Block]

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.dropout = nn.Dropout(config.dropout)

        self.blocks: helpers.StaticModuleList[GPT2Block] = helpers.StaticModuleList(
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

        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

    def forward(self, x: t.Tensor, cache: Optional[Any] = None) -> t.Tensor:
        """
        x: shape (batch, seq), dtype t.int64 - the token ids

        Return: shape (batch, seq, vocab_size), dtype t.float32- the output logits
        """

        batch_size, seq_len = x.shape

        # Combine the token and position embeddings for the embedding layer
        tokens = self.token_embedding(x)  # (batch, seq, hidden_size)
        positions = self.pos_embedding(
            t.arange(seq_len, device=device)
        )  # (batch, seq, hidden_size)
        x = tokens + positions
        x = self.dropout(x)  # batch, seq, hidden_size

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)  # batch, seq, hidden_size

        y = self.final_layer_norm(x)  # batch, seq, hidden_size

        # Umbed the output back to the vocabulary size using the transpose of the token embedding as the umbedding matrix (i.e. tied embeddings)
        logits = einsum(
            "vocab_size hidden_size, batch seq hidden_size -> batch seq vocab_size",
            self.token_embedding.weight,
            y,
        )  # batch, seq, vocab_size

        return logits

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
            self._copy_weight_bias(my_block.attention.ln1, hf_block.ln_1)
            self._copy_weight_bias(
                my_block.attention.attn.qkv_proj, hf_block.attn.c_attn, transpose=True
            )
            self._copy_weight_bias(
                my_block.attention.attn.output_proj,
                hf_block.attn.c_proj,
                transpose=True,
            )
            self._copy_weight_bias(my_block.MLP.ln2, hf_block.ln_2)

            # Copy MLP weights
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
    model = GPT2(config)
    x = t.randint(0, config.vocab_size, (1, 10))
    model.load_pretrained_weights()
    logits = model(x)
    print(logits)
    print(logits.shape)

    with SummaryWriter(comment="ModelArchitecture") as w:
        w.add_graph(model, (x,))
