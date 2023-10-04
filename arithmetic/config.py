from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ArithmeticConfig:
    "Constants used for the MoE model."

    # Model
    # tokenizer_string: str = "gpt2"
    activation_function: str = "gelu"
    num_layers: int = 2
    num_attn_heads: int = 8

    hidden_size: int = 128
    # vocab_size: int = 50257
    max_position_embeddings: int = 128

    dropout: float = 0.1

    layer_norm_epsilon: float = 1e-05

    # Training parameters
    max_steps: int = 100
    num_epochs: int = 1
    learning_rate: float = 0.001

    batch_size: int = 64
    # train_test_split: float = 0.99
    block_size: int = 64

    def __str__(self) -> str:
        out = "ArithemticConfig:\n\n"
        out += "\n".join(f"{k}={str(v)}" for k, v in asdict(self).items())
        return out


if __name__ == "__main__":
    print(ArithmeticConfig())
