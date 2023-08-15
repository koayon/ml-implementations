import math
from typing import Any, Optional, Tuple

import torch as t
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from helpers import einsum


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention layer from GQA paper

    Reference: https://arxiv.org/pdf/2305.13245.pdf
    """

    q_proj: nn.Linear
    kv_proj: nn.Linear
    output_proj: nn.Linear
    attn_dropout: nn.Dropout
    resid_dropout: nn.Dropout

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_size: Optional[int] = None,
        num_groups: int = 1,
        dropout=0.1,
    ):
        """GQA.
        For num_groups=1, this reduces to Multi-Query Attention.
        For num_groups=num_heads, this reduces to regular Multi-Head Attention."""
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % num_groups == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads if head_size is None else head_size
        self.num_groups = num_groups
        self.group_ratio = num_heads // num_groups

        self.q_proj = nn.Linear(hidden_size, (self.num_groups * self.head_size))  # W_q
        self.kv_proj = nn.Linear(
            hidden_size, (self.num_heads * self.head_size) * 2
        )  # W_kv
        self.output_proj = nn.Linear(
            (self.num_heads * self.head_size), hidden_size
        )  # W_O

        self.attn_scale = 1.0 / math.sqrt(self.head_size)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self, x: t.Tensor, layer_cache: Optional[Any] = None
    ) -> Tuple[t.Tensor, None]:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        _batch, _seq_length, hidden_size = x.shape
        assert hidden_size == self.hidden_size

        # Apply W_qkv to x to get q, k, v
        q = self.q_proj(x)  # (batch, seq, num_groups * head_size)
        kv = self.kv_proj(x)  # (batch, seq, 2 * num_heads * head_size)
        k, v = t.split(
            kv, (self.num_heads * self.head_size), dim=-1
        )  # (batch, seq, num_heads * head_size)

        q = rearrange(
            q, "batch seq (group dim) -> batch group seq dim", dim=self.head_size
        )
        q = repeat(
            q,
            "batch group seq dim -> batch (group group_ratio) seq dim",
            group_ratio=self.group_ratio,
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

        return out, None


if __name__ == "__main__":
    attention = GroupedQueryAttention(hidden_size=512, num_heads=8)
    x = t.randn(2, 10, 512)
    out = attention(x)
    print(out)
    print(out.shape)
