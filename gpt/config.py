from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    """Constants used throughout the GPT2 model."""

    # model_hyperparameters
    activation_function: str = "new_gelu"
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50257
    hidden_size: int = 768
    max_position_embeddings: int = 1024
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5

    # train hyperparameters
    max_iters: int = 1000
    num_epochs: int = 1
    learning_rate: float = 0.001

    batch_size: int = 16
    train_test_split: float = 0.99
    block_size: int = 64

    eval_steps: int = 10
