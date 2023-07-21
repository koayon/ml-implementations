import math
from typing import Any, Optional

import torch as t
from einops import rearrange
from fancy_einsum import einsum
from torch import nn


class UnidirectionalAttention(nn.Module):
    "Implements the attention mechanism described in"
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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0

        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads if head_size is None else head_size
        self.qkv_proj = nn.Linear(hidden_size, (self.num_heads * self.head_size) * 3)
        self.output_proj = nn.Linear((self.num_heads * self.head_size), hidden_size)
        self.attn_scale = 1.0 / math.sqrt(self.head_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: t.Tensor, cache: Optional[Any] = None) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        _, S, H = x.shape
        assert H == self.hidden_size

        qkv = self.qkv_proj(x)  # (batch, seq, 3 * num_heads * head_size)
        q, k, v = t.split(qkv, (self.num_heads * self.head_size), dim=-1)
        q = rearrange(
            q, "batch seq (head dim) -> batch head seq dim", dim=self.head_size
        )
        k = rearrange(
            k, "batch seq (head dim) -> batch head seq dim", dim=self.head_size
        )
        v = rearrange(
            v, "batch seq (head dim) -> batch head seq dim", dim=self.head_size
        )
        if cache:
            CB, C_HEADS, CS, C_HEADSIZE = cache.k.shape
            assert CB == 1
            assert C_HEADS == self.num_heads
            assert C_HEADSIZE == self.head_size
            if CS != 0:
                assert (
                    S == 1
                ), "The cache is loaded. Should only pass one token at a time."
            cache.k = k = t.cat([cache.k, k], dim=2)
            cache.v = v = t.cat([cache.v, v], dim=2)
        else:
            CS = 0
        # In the cache case, we're processing q from sequence position CS in the full sequence
        q_ind = t.arange(CS, CS + S).unsqueeze(1)
        k_ind = t.arange(0, CS + S).unsqueeze(0)
        mask = (q_ind >= k_ind).to(x.device)
        # For each query vector, a row of scores corresponding to each key vector
        attn_scores = einsum(
            "batch head seq_q dim, batch head seq_k dim -> batch head seq_q seq_k", q, k
        )
        attn_scores = attn_scores * self.attn_scale
        neg_inf = t.tensor(-1e4, dtype=attn_scores.dtype, device=x.device)
        attn_scores = t.where(mask, attn_scores, neg_inf)
        probs = self.attn_dropout(attn_scores.softmax(dim=-1))
        # For each query vector, the weighted average value vector
        combined_v = einsum(
            "batch head seq_q seq_k, batch head seq_k dim -> batch head seq_q dim",
            probs,
            v,
        )
        combined_v = rearrange(combined_v, "batch head seq dim -> batch seq (head dim)")
        out = self.output_proj(combined_v)
        return self.resid_dropout(out)
