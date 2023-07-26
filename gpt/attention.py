import math
from typing import Any, Optional

import torch as t
from einops import rearrange
from fancy_einsum import einsum
from torch import nn
from torch.nn import functional as F


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

    def forward(self, x: t.Tensor, cache: Optional[Any] = None) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        _, seq_length, hidden_size = x.shape
        assert hidden_size == self.hidden_size

        # Apply W_qkv to x to get q, k, v
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

        # Combine q and k to get attention scores
        q_k = t.einsum("bnih,bnjh->bnij", q, k)  # batch, num_heads, seq, seq
        q_k *= self.attn_scale

        mask = t.triu(t.ones_like(q_k)).to(x.device)  # seq, seq

        # Use lower triangular mask for q_k matrix
        masked_attention_scores = q_k.masked_fill(
            mask == 0, float("-inf")
        )  # batch, num_heads, seq, seq

        attn_matrix = self.attn_dropout(
            F.softmax(masked_attention_scores, dim=-1)
        )  # seq, seq

        # For each query vector, combine with the weighted average value vector
        combined_with_v = einsum(
            "batch head seq seq_i, batch head seq_i hidden_dim -> batch head seq hidden_dim",
            attn_matrix,
            v,
        )  # batch, num_heads, seq, hidden_size
        combined_with_v = rearrange(
            combined_with_v, "batch head seq hidden_dim -> batch seq (head hidden_dim)"
        )  # batch, seq, hidden_size*num_heads

        out = self.output_proj(combined_with_v)  # batch, seq, hidden_size
        out = self.resid_dropout(out)

        return out
