from dataclasses import dataclass


@dataclass
class MambaConfig:
    # Model params
    vocab_size: int = 50257
    hidden_dim: int = 768
    residual_dim: int = 128
    expansion_factor: int = 2
    conv_kernel_size: int = 3
    num_blocks: int = 24

    # Training params
    dropout_rate: float = 0.1


official_130m_config = {
    "d_model": 768,
    "n_layer": 24,
    "vocab_size": 50277,
    "ssm_cfg": {},
    "rms_norm": True,
    "residual_in_fp32": True,
    "fused_add_norm": True,
    "pad_vocab_size_multiple": 8,
}
