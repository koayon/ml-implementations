import torch as t
from torch import nn
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel

from mamba.model import Mamba


class MambaHFConfig(PretrainedConfig):
    def __init__(
        self,
        # layers: int = 8,
        **kwargs,
    ):
        # self.block_type = block_type
        # self.layers = layers
        super().__init__(**kwargs)


class MambaHFModel(PreTrainedModel):
    def __init__(
        self,
        model: nn.Module = Mamba(),
        hf_config: MambaHFConfig = MambaHFConfig(),
        *args,
        **kwargs,
    ):
        super().__init__(hf_config, *args, **kwargs)
        self.mamba = model

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer

    def forward(self, input_ids: t.Tensor, **kwargs):
        return self.mamba(input_ids)
