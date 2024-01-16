from dataclasses import dataclass


@dataclass
class MambaConfig:
    vocab_size: int = 50257
    hidden_dim: int = 768
    residual_dim: int = 128
    expansion_factor: int = 2
    conv_kernel_size: int = 3
    num_blocks: int = 1
    dropout_rate: float = 0.1
