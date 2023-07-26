from dataclasses import dataclass


@dataclass(frozen=True)
class MoEConfig:
    "Constants used for the MoE model."
    tokeniser_string = "gpt2"
    activation_function = "gelu"
    num_layers = 6
    hidden_size = 256
    attn_dropout = 0.1
    expert_dropout = 0.4
    max_position_embeddings = 256
    vocab_size = 50257
    batch_size = 16
    train_test_split = 0.9
    num_epochs = 1
    num_experts = 4
    # capacity_factor = 1.25
    routing_dropout = 0.1
    learning_rate = 0.001
    num_attn_heads = 8
    layer_norm_epsilon = 1e-05
