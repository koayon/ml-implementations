from typing import Any, List, Optional, Tuple

import torch as t
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.distributions.categorical import Categorical
from transformers import PretrainedConfig, PreTrainedModel

from arithmetic.config import ArithmeticConfig
from general.character_level_tokenizer import CharTokenizer
from gpt.cached_attention import AttentionCache
from gpt.model import FullKeyValueCache
from gpt.transformer_block import GPT2Block
from helpers import einsum

# Use character level tokenizer
tokenizer = CharTokenizer()
device = "cuda" if t.cuda.is_available() else "cpu"


config = ArithmeticConfig()


class ArithmeticNet(PreTrainedModel):
    """PonderNet-style decoder only transformer, using adaptive computation.
    Instead of pondering on whether to continue to the next layer, it Ponders to decide whether to output the <IDK> token which allows it to continue computing for a small penalty.

    Inspired by PonderNet from DeepMind paper, GPT-2 architecture and Pause Tokens paper

    Reference: https://arxiv.org/pdf/2310.02226.pdf
    """

    token_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    final_layer_norm: nn.LayerNorm
    blocks: nn.ModuleList  # of GPT2Block

    layer_cache: AttentionCache
    cache_list: List[AttentionCache]

    def __init__(
        self,
        config: ArithmeticConfig = config,
        hf_config=PretrainedConfig(),
        training: bool = True,
    ):
        super().__init__(config=hf_config)

        self.config = config
        self.vocab_size = len(tokenizer)
        self.training = training

        self.token_embedding = nn.Embedding(self.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.dropout = nn.Dropout(config.dropout)

        self.encoder_blocks = nn.ModuleList(
            [
                GPT2Block(
                    layer_index=index,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attn_heads,
                    dropout=config.dropout,
                    layer_norm_epsilon=config.layer_norm_epsilon,
                    activation_function=config.activation_function,
                    autoregressive=False,
                )
                for index in range(config.num_layers)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                GPT2Block(
                    layer_index=index,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attn_heads,
                    dropout=config.dropout,
                    layer_norm_epsilon=config.layer_norm_epsilon,
                    activation_function=config.activation_function,
                    autoregressive=True,
                )
                for index in range(config.num_layers)
            ]
        )

        self.confidence_score_linears: nn.ModuleList = nn.ModuleList(
            [nn.Linear(config.hidden_size, 1) for _ in range(config.num_layers)]
        )

        self.confidence_combine = nn.Linear(config.num_layers, 1)

        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

        self.unembedding = (
            self.token_embedding.weight.T
        )  # pseudo-inverse of Embedding matrix which is used as the unembed.
        # Note: in the general case we could also have W_y_i as a learned matrix.

        # TODO: Make model encoder-decoder.

    def forward(
        self,
        input_ids: t.Tensor,
        cache: Optional[FullKeyValueCache] = None,
        confidence_scores: Optional[t.Tensor] = None,
    ) -> Tuple[t.Tensor, FullKeyValueCache, t.Tensor, t.Tensor]:
        """
        Args:
            x (t.Tensor): shape (batch, seq), dtype t.int64 - the token ids
            cache (Optional[FullKeyValueCache], optional): _description_. Defaults to None.

        Returns:
            Tuple[t.Tensor, FullKeyValueCache, PonderCache]:
                output_logits: shape (batch, seq, vocab_size), dtype t.float32- the output logits
                kv_cache
                idk_logits: shape (batch, seq, 1), dtype t.float32 - the confidence scores for each
                pre_idk_logits: shape (batch, seq, vocab_size), dtype t.float32 - the logits before the idk_logits are added
        """
        confidence_scores_list = []

        if cache is None:
            cache_list = [None] * len(self.blocks)  # type: ignore
        else:
            cache_list = cache.to_cache_list()

        _batch_size, seq_len = input_ids.shape

        # Combine the token and position embeddings for the embedding layer
        tokens = self.token_embedding(input_ids)  # (batch, seq, hidden_size)
        positions = self.pos_embedding(t.arange(seq_len))  # (batch, seq, hidden_size)
        x = tokens + positions
        x = self.dropout(x)  # batch, seq, hidden_size

        # Apply transformer blocks
        for layer_index, block in enumerate(self.blocks):
            x, layer_cache = block(
                x, layer_cache=cache_list[layer_index]
            )  # batch, seq, hidden_size
            cache_list[layer_index] = layer_cache

            # Compute confidence scores
            confidence_scores_list.append(
                self.confidence_score_linears[layer_index](x)
            )  # batch, seq, 1

        y = self.final_layer_norm(x)  # batch, seq, hidden_size

        # Get the confidence scores for each layer
        confidence_scores = t.stack(
            confidence_scores_list, dim=-1
        )  # batch, seq, 1, num_layers
        confidence_scores = confidence_scores.squeeze(2)  # batch, seq, num_layers

        print("confidence_scores", confidence_scores.shape)

        idk_logits: t.Tensor = self.confidence_combine(
            confidence_scores
        )  # batch, seq, 1

        print("idk_logits", idk_logits.shape)

        idk_logits = repeat(
            idk_logits,
            "batch seq 1 -> batch seq vocab_size",
            vocab_size=self.vocab_size,
        )  # batch, seq, vocab_size

        idk_token_mask = (
            t.arange(self.vocab_size) == tokenizer.idk_token_id
        )  # vocab_size
        idk_token_mask = repeat(
            idk_token_mask,
            "vocab_size -> batch seq vocab_size",
            batch=_batch_size,
            seq=seq_len,
        )  # batch, seq, vocab_size
        idk_logits = idk_logits * idk_token_mask  # batch, seq, vocab_size

        # Umbed the output back to the vocabulary size using the transpose of the token embedding as the umbedding matrix (i.e. tied embeddings)
        logits = einsum(
            "vocab_size hidden_size, batch seq hidden_size -> batch seq vocab_size",
            self.token_embedding.weight,
            y,
        )  # batch, seq, vocab_size

        pre_idk_logits = logits.clone()

        # Combine the logits with the idk_logits
        logits = logits + idk_logits  # batch, seq, vocab_size

        full_cache = FullKeyValueCache.from_cache_list(cache_list=cache_list)  # type: ignore

        return logits, full_cache, idk_logits, pre_idk_logits


if __name__ == "__main__":
    model = ArithmeticNet(config, training=True)

    text = "2 + 5 = "
    inputs_ids = tokenizer.encode(text)
    input_ids = t.tensor([inputs_ids])

    logits, _kv_cache, idk_logits, pre_idk_logits = model(input_ids)

    print(logits)
    print(logits.shape)

    print(model.config)