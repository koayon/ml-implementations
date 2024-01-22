import tiktoken as tk
import torch as t
import torch.nn as nn
from jaxtyping import Float

from mamba.config import MambaConfig
from mamba.mamba_block import MambaResidualBlock


class Mamba(nn.Module):
    def __init__(
        self,
        config: MambaConfig = MambaConfig(),
    ):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.residual_dim)

        self.dropout = nn.Dropout(config.dropout_rate)

        self.layers = nn.ModuleList(
            [
                MambaResidualBlock(
                    config=config,
                )
                for _ in range(config.num_blocks)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(config.residual_dim)

        self.unembed = nn.Linear(config.residual_dim, config.vocab_size, bias=False)
        self.unembed.weight = nn.Parameter(self.embedding.weight)  # tie weights

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
        for mamba_block in self.layers:
            x = mamba_block(x)  # batch, seq_len, input_dim

        y = self.final_layer_norm(x)

        logits = self.unembed(y)  # batch, seq_len, vocab_size

        return logits


if __name__ == "__main__":
    prompt = "I am become Death, the destroyer of worlds."
    tokenizer = tk.encoding_for_model("gpt2")
    tokens_list = tokenizer.encode(prompt)
    tokens = t.tensor(tokens_list, device="mps").unsqueeze(0)  # batch, seq_len, 1

    mamba = Mamba().to("mps")
    logits = mamba(tokens)
    print(logits.shape)
    print("Done!")
