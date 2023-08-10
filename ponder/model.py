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
from torch.distributions.categorical import Categorical
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as HFGPT2Block

import helpers
from gpt.cached_attention import AttentionCache
from gpt.config import GPTConfig
from gpt.model import FullKeyValueCache, full_kv_cache_from
from gpt.transformer_block import GPT2Block

tokenizer = tiktoken.encoding_for_model("gpt2")

device = "cuda" if t.cuda.is_available() else "cpu"


config = GPTConfig()


@dataclass
class PonderCache:
    lambda_vals: t.Tensor  # num_layers
    intermediate_vals: t.Tensor  # list of num_layers tensors of shape (batch, seq, hidden_size)


class PonderNet(nn.Module):
    """PonderNet-style decoder only transformer, using adaptive computation.

    Inspired by PonderNet from DeepMind paper and GPT-2 architecture.
    Reference:
    """

    token_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    final_layer_norm: nn.LayerNorm
    blocks: nn.ModuleList  # of GPT2Block

    def __init__(
        self,
        config: GPTConfig = config,
        with_pretrained_weights=True,
        training: bool = True,
    ):
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

        self.intermediate_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

        # Map which generates probabilities for each layer
        self.probs_proj = nn.ModuleList(
            [
                nn.Linear(in_features=config.hidden_size, out_features=1, device=device)
                for index in range(config.num_layers)
            ]
        )

        self.unembedding = (
            self.token_embedding.weight.T
        )  # pseudo-inverse of Embedding matrix which is used as the unembed.
        # Note: in the general case we could also have W_y_i as a learned matrix.

        if with_pretrained_weights:
            self.load_pretrained_weights()

        self.training = training

    def forward(
        self, x: t.Tensor, cache: Optional[FullKeyValueCache] = None
    ) -> Tuple[t.Tensor, FullKeyValueCache, PonderCache]:
        """
        Args:
            x: shape (batch, seq), dtype t.int64 - the token ids

        Return:
            output_logits: shape (batch, seq, vocab_size), dtype t.float32- the output logits
            kv_cache
            lambda_vals: list
            intermediate_preds: list of tensors (these are the y_i values)

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

        lambda_vals: list[t.Tensor] = []
        intermediate_pred: list[t.Tensor] = []

        # Apply transformer blocks
        for layer_index, block in enumerate(self.blocks):
            x, layer_cache = block(
                x, layer_cache=cache_list[layer_index]
            )  # batch, seq, hidden_size
            cache_list[layer_index] = layer_cache

            # Calculate lambda_i and y_i for each layer. We will hold these in variables lambda_vals and intermediate_pred
            conditional_prob = self.probs_proj(x)
            lambda_vals.append(conditional_prob)

            current_pred = einsum(
                "hidden_size vocab_size, batch seq hidden_size -> batch seq vocab_size",
                self.unembedding,
                self.intermediate_layer_norm(x),
            )  # batch seq vocab_size
            intermediate_pred.append(current_pred)

            if self.training:
                pass
            else:
                dist = Categorical(probs=[conditional_prob, 1 - conditional_prob])
                # If stopping at this node
                if dist.sample() == 0:
                    x = current_pred
                    break

        y = self.final_layer_norm(x)  # batch, seq, hidden_size

        # Umbed the output back to the vocabulary size using the transpose of the token embedding as the umbedding matrix (i.e. tied embeddings)
        logits = einsum(
            "vocab_size hidden_size, batch seq hidden_size -> batch seq vocab_size",
            self.token_embedding.weight,
            y,
        )  # batch, seq, vocab_size

        full_cache = full_kv_cache_from(cache_list=cache_list)

        ponder_cache = PonderCache(
            intermediate_vals=t.stack(intermediate_pred),
            lambda_vals=t.stack(lambda_vals),
        )

        return logits, full_cache, ponder_cache

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
    model = PonderNet(config, with_pretrained_weights=True, training=True)
    x = t.randint(0, config.vocab_size, (1, 10))
    logits, _cache = model(x)
    print(logits)
    print(logits.shape)

    print(model.config)

    with SummaryWriter(comment="ModelArchitecture") as w:
        w.add_graph(model, (x,))

    # k = t.randn(1, 10, 10)
    # v = t.randn(1, 10, 10)
    # cache = AttentionCache(k, v)

    # a = FullKeyValueCache(cache_list=[cache] * 3)
    # assert isinstance(a, FullKeyValueCache)
