from dataclasses import asdict, dataclass


# @dataclass(frozen=True)
@dataclass
class MoEConfig:
    "Constants used for the MoE model."

    # Model
    tokeniser_string: str = "gpt2"
    activation_function: str = "gelu"
    num_layers: int = 6
    num_experts: int = 4
    num_attn_heads: int = 8

    hidden_size: int = 256
    vocab_size: int = 50257
    max_position_embeddings: int = 256

    attn_dropout: float = 0.1
    expert_dropout: float = 0.4
    routing_dropout: float = 0.1

    layer_norm_epsilon: float = 1e-05
    train_capacity_factor: float = 1.25
    eval_capacity_factor: float = 2.0

    # Training parameters
    max_iters: int = 100
    num_epochs: int = 1
    learning_rate: float = 0.001

    batch_size: int = 16
    train_test_split: float = 0.99
    block_size: int = 64

    sophia_hessian_update_steps: int = 10
    eval_steps: int = 10

    def __str__(self) -> str:
        out = "MoEConfig:\n\n"
        out += "\n".join(f"{k}={str(v)}" for k, v in asdict(self).items())
        return out


if __name__ == "__main__":
    print(MoEConfig())
