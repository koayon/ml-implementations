import math
from functools import lru_cache
from typing import Any, Optional, Tuple, Union

import torch as t
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from gpt.cached_attention import AttentionCache
from helpers import einsum
from mixture_of_experts.config import MoEConfig
from moet_experiment.moet_config import MoETConfig

config = MoETConfig()


class AlibiUnidirectionalAttention(nn.Module):
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
        *,
        # config: MoETConfig | MoEConfig = config,
        hidden_size: int,
        num_heads: int,
        head_size: Optional[int] = None,
        dropout=0.1,
        # max_seq_len: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert self.hidden_size % self.num_heads == 0

        self.head_size = (
            head_size if head_size is not None else self.hidden_size // self.num_heads
        )

        self.qkv_proj = nn.Linear(
            self.hidden_size, (self.num_heads * self.head_size) * 3
        )  # W_qkv
        self.output_proj = nn.Linear(
            (self.num_heads * self.head_size), self.hidden_size
        )  # W_O

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if self.num_heads <= 8:
            self.m_list = [1 / (2**i) for i in range(self.num_heads)]
        else:
            self.m_list = [1 / (2 ** (i / 2)) for i in range(self.num_heads)]

    @lru_cache
    def regular_mask(self, seq_length: int) -> t.Tensor:
        ones = t.ones(seq_length, seq_length)
        infs = ones * float("-inf")
        mask = t.triu(infs, diagonal=1)

        return mask

    @lru_cache
    def get_alibi_mask(self, seq_length: int) -> t.Tensor:
        ones = t.ones(seq_length, seq_length)
        mask = t.triu(ones, diagonal=1)  # seq seq (upper triangular)

        # Get decreasing mask values
        mask = -t.cumsum(mask, dim=1)  # seq seq (upper triangular)
        mask = mask.T  # seq seq (lower triangular)

        mask += self.regular_mask(
            seq_length
        )  # seq seq (lower triangular with content, upper triangular with -inf)

        mask_list = []
        for i in range(self.num_heads):
            mask_list.append(mask * self.m_list[i])
        mask = t.stack(mask_list, dim=0)  # num_heads seq seq

        mask = rearrange(
            mask, "num_heads seq_i seq_j -> 1 num_heads seq_i seq_j"
        )  # 1 num_heads seq seq

        return mask

    def forward(
        self, x: t.Tensor, layer_cache: Optional[Any] = None
    ) -> Tuple[t.Tensor, None]:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        _batch, seq_length, hidden_size = x.shape
        assert hidden_size == self.hidden_size

        # Apply W_qkv to x to get q, k, v
        qkv = self.qkv_proj(x)  # (batch, seq, 3 * num_heads * head_size)
        q, k, v = t.split(
            qkv, (self.num_heads * self.head_size), dim=-1
        )  # (batch, seq, num_heads * head_size)

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

        # Apply mask (note we don't scale by sqrt(d_k) as normal with ALiBi)
        mask = self.get_alibi_mask(seq_length).to(x.device)  # 1 num_heads seq seq

        masked_attention_scores = q_k + mask

        attn_matrix = self.attn_dropout(
            F.softmax(masked_attention_scores, dim=-1)
        )  # seq, seq

        # print(attn_matrix)

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

        # TODO: Cache KV Cache for inference!


if __name__ == "__main__":
    attention = AlibiUnidirectionalAttention(hidden_size=16, num_heads=4)
    x = t.randn(1, 5, 16)
    out = attention(x)
    # print(out)
    # print(out.shape)
    # print(get_alibi_mask(5))
