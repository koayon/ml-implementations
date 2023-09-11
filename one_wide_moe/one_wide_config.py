from dataclasses import asdict, dataclass


# @dataclass(frozen=True)
@dataclass
class OneWideConfig:
    "Constants used for the OneWideMoE model."

    model_name: str = "OneWideMoE"

    # Model
    tokenizer_string: str = "roneneldan/TinyStories-8M"
    activation_function: str = "silu"
    # use_confi_mlp: bool = True

    num_total_layers: int = 8
    # num_early_layers: int = 4

    num_experts = 8
    num_expert_groups = 4

    # num_experts_early: int = 4
    # num_experts_late: int = 8
    # num_experts_hash: int = 8

    num_attn_heads: int = 8

    hidden_size: int = 512
    vocab_size: int = 50257
    max_position_embeddings: int = 256

    attn_dropout: float = 0.1
    expert_dropout: float = 0.4
    routing_dropout: float = 0.1
    resid_dropout: float = 0.1

    router_temperature: float = (
        0.3  # we may want to reduce this over time like the learning rate
    )

    layer_norm_epsilon: float = 1e-05
    train_capacity_factor: float = 1.25
    eval_capacity_factor: float = 2.0

    use_expert_choice: bool = False

    # Training parameters
    max_steps: int = 4000 # 2M in TinyStories
    num_epochs: int = 1

    learning_rate: float = 0.001
    weight_decay: float = 0.01

    lb_coef: float = 1.0 # to tune. Load balancing coefficient
    z_coef: float = 0.001 # Router z coefficient

    warmup_steps: int = 100

    batch_size: int = 32
    train_test_split: float = 0.99
    block_size: int = 256

    eval_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 400

    # sophia_hessian_update_steps: int = 10


    def __str__(self) -> str:
        out = "OneWideConfig:\n\n"
        out += "\n".join(f"{k}={str(v)}" for k, v in asdict(self).items())
        return out

    def __dict__(self):
        return asdict(self)

    def to_dict(self):
        return asdict(self)


if __name__ == "__main__":
    print(OneWideConfig())
