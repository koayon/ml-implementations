import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch as t
from einops import rearrange
from jaxtyping import Float, Int
from torch import nn
from torch.nn import functional as F

from helpers import einsum


@dataclass
class AttentionCache:
    k: Float[t.Tensor, "batch seq hidden_dim"]
    v: Float[t.Tensor, "batch seq hidden_dim"]


class UnidirectionalAttention(nn.Module):
    """Unidirectional attention layer based on Attention is all you need.
    Self-attention layer for decoder in transformer model.

    Reference: https://arxiv.org/abs/1706.03762
    """

    qkv_proj: nn.Linear
    output_proj: nn.Linear
    attn_dropout: nn.Dropout
    resid_dropout: nn.Dropout

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_size: Optional[int] = None,
        dropout=0.1,
        autoregressive: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0

        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads if head_size is None else head_size

        self.qkv_proj = nn.Linear(
            hidden_size, (self.num_heads * self.head_size) * 3
        )  # W_qkv
        self.output_proj = nn.Linear(
            (self.num_heads * self.head_size), hidden_size
        )  # W_O

        self.attn_scale = 1.0 / math.sqrt(self.head_size)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.autoregressive = autoregressive

    def forward(
        self, x: t.Tensor, layer_cache: Optional[AttentionCache] = None
    ) -> Tuple[t.Tensor, AttentionCache]:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        _batch, _seq_length, hidden_size = x.shape
        # assert hidden_size == self.hidden_size

        if layer_cache:
            return self._forward_with_cache(x, layer_cache)
        else:
            return self._forward_without_cache(x)

    def _forward_without_cache(self, x: t.Tensor) -> Tuple[t.Tensor, AttentionCache]:
        """First time we forward, before we have a cache."""

        # Apply W_qkv to x to get q, k, v
        qkv = self.qkv_proj(x)  # (batch, seq, 3 * num_heads * head_size)
        q, k, v = t.split(
            qkv, (self.num_heads * self.head_size), dim=-1
        )  # (batch, seq, num_heads * head_size)

        layer_cache = AttentionCache(k=k, v=v)

        q = rearrange(
            q, "batch seq (head dim) -> batch head seq dim", dim=self.head_size
        )
        k = rearrange(
            k, "batch seq (head dim) -> batch head seq dim", dim=self.head_size
        )
        v = rearrange(
            v, "batch seq (head dim) -> batch head seq dim", dim=self.head_size
        )

        # Combine q and k to get attention scores
        q_k = t.einsum("bnih,bnjh->bnij", q, k)  # batch, num_heads, seq, seq
        q_k *= self.attn_scale

        mask = t.tril(t.ones_like(q_k)).to(x.device)  # seq, seq

        # Use lower triangular mask for q_k matrix. Where mask is 0 (i.e. upper triangle), we set the attention score to -inf (which will be 0 post softmax)
        if self.autoregressive:
            masked_attention_scores = q_k.masked_fill(
                mask == 0, float("-inf")
            )  # batch, num_heads, seq, seq

            attn_matrix = self.attn_dropout(
                F.softmax(masked_attention_scores, dim=-1)
            )  # seq, seq
        else:
            attn_matrix = self.attn_dropout(F.softmax(q_k, dim=-1))

        # For each query vector, combine with the weighted average value vector
        combined_with_v = einsum(
            "batch head seq seq_i, batch head seq_i hidden_dim -> batch head seq hidden_dim",
            attn_matrix,
            v,
        )  # batch, num_heads, seq, hidden_size
        combined_with_v = rearrange(
            combined_with_v, "batch head seq hidden_dim -> batch seq (head hidden_dim)"
        )  # batch, seq, hidden_size*num_heads

        out = self.output_proj(combined_with_v)  # batch, seq, embedding_dim
        out = self.resid_dropout(out)  # batch, seq, embedding_dim

        return out, layer_cache

    def _forward_with_cache(
        self, x: t.Tensor, layer_cache: AttentionCache
    ) -> Tuple[t.Tensor, AttentionCache]:
        """Forward using cached attention keys and values."""

        # Grab the cached keys and values
        cached_k, cached_v = (layer_cache.k, layer_cache.v)  # (batch, head, seq, dim)

        # Compute the q,k,v for the new token
        qkv_final_token = self.qkv_proj(
            x[:, -1:, :]
        )  # (batch, 1, 3 * num_heads * head_size)
        q_final_row, k_final_row, v_final_row = t.split(
            qkv_final_token, (self.num_heads * self.head_size), dim=-1
        )  # (batch, 1, num_heads * head_size) Note that num_heads * head_size = dim

        # Concatenate K and V with to get the full matrices
        k_full = t.cat([cached_k, k_final_row], dim=1)  # (batch, seq + 1, dim)
        v_full = t.cat([cached_v, v_final_row], dim=1)  # (batch, seq + 1, dim)

        # Cache the keys and values
        layer_cache.k = k_full
        layer_cache.v = v_full

        # Rearrage q,k,v
        q_final_row = rearrange(
            q_final_row, "batch 1 (head dim) -> batch head 1 dim", dim=self.head_size
        )
        k_full = rearrange(
            k_full, "batch seq (head dim) -> batch head seq dim", dim=self.head_size
        )
        v_full = rearrange(
            v_full, "batch seq (head dim) -> batch head seq dim", dim=self.head_size
        )

        # Compute attention scores
        q_k_final_row = einsum(
            "batch head_num one dim, batch head_num seq dim -> batch head_num one seq",
            q_final_row,
            k_full,
        )  # (batch, head, 1, seq + 1)
        q_k_final_row *= self.attn_scale  # (batch, head_num, 1, seq + 1)

        attn_final_row = F.softmax(
            q_k_final_row, dim=-1
        )  # (batch, head_num, 1, seq + 1)
        attn_final_row = self.attn_dropout(
            attn_final_row
        )  # (batch, head_num, 1, seq + 1)

        combined_with_v_final_row = einsum(
            "batch head one seq_i, batch head seq_i dim -> batch head one dim",
            attn_final_row,
            v_full,
        )  # (batch, head_num, 1, dim)
        combined_with_v_final_row = rearrange(
            combined_with_v_final_row,
            "batch head 1 hidden_size -> batch 1 (head hidden_size)",
        )

        # Apply W_O to the combined vectors
        final_token_output = self.output_proj(
            combined_with_v_final_row
        )  # (batch, 1, hidden_size)

        return final_token_output, layer_cache


if __name__ == "__main__":
    attention = UnidirectionalAttention(512, 8)
    x = t.randn(2, 10, 512)
    out = attention(x)
    print(out)
    print(out.shape)
