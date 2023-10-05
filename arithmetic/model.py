from json import decoder, encoder
from typing import Any, List, Optional, Tuple, Union

import torch as t
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.distributions.categorical import Categorical
from transformers import PretrainedConfig, PreTrainedModel

from arithmetic.config import ArithmeticConfig
from general.character_level_tokenizer import CharTokenizer
from gpt.cached_attention import AttentionCache
from gpt.enc_dec_transformer_block import EncDecTransformerBlock
from gpt.model import FullKeyValueCache
from gpt.transformer_block import GPT2Block
from helpers import einsum

# Use character level tokenizer
tokenizer = CharTokenizer()
device = "cuda" if t.cuda.is_available() else "cpu"


model_config = ArithmeticConfig()


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
        model_config: ArithmeticConfig = model_config,
        hf_config=PretrainedConfig(),
        training: bool = True,
    ):
        super().__init__(config=hf_config)

        self.model_config = model_config
        self.vocab_size = len(tokenizer)
        self.training = training
        self.batch_size = model_config.batch_size

        self.token_embedding = nn.Embedding(self.vocab_size, model_config.hidden_size)
        self.pos_embedding = nn.Embedding(
            model_config.max_position_embeddings, model_config.hidden_size
        )

        self.dropout = nn.Dropout(model_config.dropout)

        self.encoder_blocks = nn.ModuleList(
            [
                GPT2Block(
                    layer_index=index,
                    hidden_size=model_config.hidden_size,
                    num_heads=model_config.num_attn_heads,
                    dropout=model_config.dropout,
                    layer_norm_epsilon=model_config.layer_norm_epsilon,
                    activation_function=model_config.activation_function,
                    autoregressive=False,
                )
                for index in range(model_config.num_layers)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                EncDecTransformerBlock(
                    layer_index=index,
                    hidden_size=model_config.hidden_size,
                    num_heads=model_config.num_attn_heads,
                    dropout=model_config.dropout,
                    layer_norm_epsilon=model_config.layer_norm_epsilon,
                    activation_function=model_config.activation_function,
                )
                for index in range(model_config.num_layers)
            ]
        )

        self.confidence_score_linears: nn.ModuleList = nn.ModuleList(
            [
                nn.Linear(model_config.hidden_size, 1)
                for _ in range(model_config.num_layers)
            ]
        )

        self.confidence_combine = nn.Linear(model_config.num_layers, 1)

        self.final_layer_norm = nn.LayerNorm(
            model_config.hidden_size, eps=model_config.layer_norm_epsilon
        )

        self.unembedding = (
            self.token_embedding.weight.T
        )  # pseudo-inverse of Embedding matrix which is used as the unembed.
        # Note: in the general case we could also have W_y_i as a learned matrix.

    def _encoder_forward(self, encoder_input_ids: t.Tensor) -> t.Tensor:
        _batch_size, seq_len = encoder_input_ids.shape
        # Combine the token and position embeddings for the embedding layer
        tokens = self.token_embedding(encoder_input_ids)  # (batch, seq, hidden_size)
        positions = self.pos_embedding(t.arange(seq_len))  # (batch, seq, hidden_size)
        x = tokens + positions
        x = self.dropout(x)  # batch, seq, hidden_size

        # Apply encoder_transformer blocks
        for layer_index, block in enumerate(self.encoder_blocks):
            x, _ = block(x)  # batch, seq, hidden_size

        return x

    def _get_idk_logits(self, confidence_scores_list: List[t.Tensor]) -> t.Tensor:
        # Get the confidence scores for each layer
        confidence_scores = t.stack(
            confidence_scores_list, dim=-1
        )  # batch, seq, 1, num_layers
        confidence_scores = confidence_scores.squeeze(2)  # batch, seq, num_layers

        batch_size, seq_len, num_layers = confidence_scores.shape

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
            batch=batch_size,
            seq=seq_len,
        )  # batch, seq, vocab_size
        idk_logits = idk_logits * idk_token_mask  # batch, seq, vocab_size

        return idk_logits

    def _decoder_forward(
        self,
        decoder_input_ids: t.Tensor,
        encoder_outputs: t.Tensor,
        cache_list: Union[list[AttentionCache], list[None]],
    ) -> Tuple[t.Tensor, FullKeyValueCache, t.Tensor, t.Tensor]:
        confidence_scores_list = []

        _batch_size, seq_len = decoder_input_ids.shape

        # Combine the token and position embeddings for the embedding layer
        tokens = self.token_embedding(decoder_input_ids)  # (batch, seq, hidden_size)
        positions = self.pos_embedding(t.arange(seq_len))  # (batch, seq, hidden_size)
        x = tokens + positions
        x = self.dropout(x)  # batch, seq, hidden_size

        # Apply decoder transformer blocks
        for layer_index, block in enumerate(self.decoder_blocks):
            x, layer_cache = block(
                x, encoder_outputs=encoder_outputs, layer_cache=cache_list[layer_index]
            )  # batch, seq, hidden_size
            cache_list[layer_index] = layer_cache

            # Compute confidence scores
            confidence_scores_list.append(
                self.confidence_score_linears[layer_index](x)
            )  # batch, seq, 1

        y = self.final_layer_norm(x)  # batch, seq, hidden_size

        # Umbed the output back to the vocabulary size using the transpose of the token embedding as the umbedding matrix (i.e. tied embeddings)
        logits = einsum(
            "vocab_size hidden_size, batch seq hidden_size -> batch seq vocab_size",
            self.token_embedding.weight,
            y,
        )  # batch, seq, vocab_size

        pre_idk_logits = logits.clone()

        idk_logits = self._get_idk_logits(confidence_scores_list)

        # Combine the logits with the idk_logits
        logits = logits + idk_logits  # batch, seq, vocab_size

        full_cache = FullKeyValueCache.from_cache_list(cache_list=cache_list)  # type: ignore

        return logits, full_cache, idk_logits, pre_idk_logits

    def forward(
        self,
        encoder_input_ids: t.Tensor,
        decoder_input_ids: Optional[t.Tensor] = None,
        cache: Optional[FullKeyValueCache] = None,
        encoder_outputs: Optional[t.Tensor] = None,
    ) -> Tuple[t.Tensor, FullKeyValueCache, t.Tensor, t.Tensor]:
        """
        Args:
            x (t.Tensor): shape (batch, seq), dtype t.int64 - the token ids
            cache (Optional[FullKeyValueCache], optional): _description_. Defaults to None.

        Returns:
            Tuple[t.Tensor, FullKeyValueCache, t.Tensor, t.Tensor]:
                output_logits: shape (batch, seq, vocab_size), dtype t.float32- the output logits
                kv_cache
                idk_logits: shape (batch, seq, 1), dtype t.float32 - the confidence scores for each
                pre_idk_logits: shape (batch, seq, vocab_size), dtype t.float32 - the logits before the idk_logits are added
        """
        if encoder_outputs is None:
            encoder_outputs = self._encoder_forward(encoder_input_ids)

        if cache is None:
            cache_list = [None] * len(self.decoder_blocks)  # type: ignore
        else:
            cache_list = cache.to_cache_list()

        if decoder_input_ids is None:
            decoder_input_ids = t.full(
                (self.batch_size, 1),
                tokenizer.sos_token_id,
            )

        print("encoder_outputs", encoder_outputs.shape)

        output_logits, full_cache, idk_logits, pre_idk_logits = self._decoder_forward(
            decoder_input_ids, encoder_outputs, cache_list=cache_list
        )

        return output_logits, full_cache, idk_logits, pre_idk_logits


if __name__ == "__main__":
    model = ArithmeticNet(model_config, training=True)

    text = "2 + 5 = "
    inputs_ids = tokenizer.encode(text)
    input_ids = t.tensor([inputs_ids])

    logits, _kv_cache, idk_logits, pre_idk_logits = model(input_ids)

    print(logits)
    print(logits.shape)

    print(model.model_config)
