from functools import lru_cache

import torch as t
import torch.nn as nn
from torch.nn import functional as F

from dora.lora import LinearWithLoRAMerged


class LinearWithDoRAMerged(LinearWithLoRAMerged):

    def __init__(self, linear, rank, alpha):
        super().__init__(linear, rank, alpha)
        self.m = nn.Parameter(self.linear.weight.norm(p=2, dim=0, keepdim=True))

    @lru_cache(maxsize=None)
    def merge_dora(self) -> t.Tensor:
        lora = self.lora.A @ self.lora.B  # Combine LoRA matrices

        numerator = self.linear.weight + self.alpha * lora.T
        norm = numerator.norm(p=2, dim=0, keepdim=True)

        v_prime = numerator / norm
        merged_linear_weights = self.m * v_prime

        return merged_linear_weights

    def forward(self, x):
        merged_linear_weights = self.merge_dora()
        return F.linear(x, merged_linear_weights, self.linear.bias)
