from typing import Any, Dict, List, Optional, Tuple

import torch as t
from torch import nn

from mixture_of_experts.config import MoEConfig

config = MoEConfig()


class HashRouter(nn.Module):
    def __init__(
        self,
        *,
        config: MoEConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.vocab_size = config.vocab_size

        raise NotImplementedError

    def forward(self):
        "Should use the built up hash table for the tokens to assign them"
        raise NotImplementedError
