from dataclasses import dataclass


@dataclass(frozen=True)
class MoEConfig:
    "Constants used for the MoE model."

    # Model
    tokeniser_string = "gpt2"
    activation_function = "gelu"
    num_layers = 6
    num_experts = 4
    num_attn_heads = 8

    hidden_size = 256
    vocab_size = 50257
    max_position_embeddings = 256

    attn_dropout = 0.1
    expert_dropout = 0.4
    routing_dropout = 0.1

    layer_norm_epsilon = 1e-05
    # capacity_factor = 1.25

    # Training
    max_iters = 100
    num_epochs = 1
    learning_rate = 0.001

    batch_size = 16
    train_test_split = 0.9
    block_size = 64

    sophia_hessian_update_steps = 10
    eval_steps = 10
