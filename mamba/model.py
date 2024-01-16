from dataclasses import dataclass

import torch as t
import torch.nn as nn
from jaxtyping import Float

from mamba.config import MambaConfig
from mamba.mamba_block import ResidualBlock


class Mamba(nn.Module):
    def __init__(
        self,
        config: MambaConfig = MambaConfig(),
    ):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.residual_dim)

        self.dropout = nn.Dropout(config.dropout_rate)

        self.residual_mamba_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    config=config,
                )
                for _ in range(config.num_blocks)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(config.residual_dim)

        self.embedding_matrix = self.embedding.weight
        self.unembed = nn.Linear(config.residual_dim, config.vocab_size, bias=False)
        self.unembed.weight = self.embedding_matrix.T  # tie weights

    def noise_embeddings(
        self, x: Float[t.Tensor, "batch seq_len input_dim"], std: float = 0.001
    ) -> Float[t.Tensor, "batch seq_len input_dim"]:
        """Add noise to the embeddings."""
        return x + t.randn(x.shape, device=x.device) * std

    def forward(
        self, input_tokens: Float[t.Tensor, "batch seq_len"]
    ) -> Float[t.Tensor, "batch seq_len input_dim"]:
        x = self.embedding(input_tokens)

        x = self.noise_embeddings(x)
        x = self.dropout(x)

        # TODO: Check for previous cache of h values to do recurrent operation
        for mamba_block in self.residual_mamba_blocks:
            x = mamba_block(x)  # batch, seq_len, input_dim

        y = self.final_layer_norm(x)

        logits = self.unembed(y)  # batch, seq_len, vocab_size

        return logits
