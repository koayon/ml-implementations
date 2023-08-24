import collections
from typing import OrderedDict, Tuple

import tiktoken
import torch as t
from einops import rearrange, repeat
from jaxtyping import Float
from tensorboardX import SummaryWriter
from torch import nn
from transformers import AutoTokenizer

from alibi.transformer_block import ALiBiTransformerBlock
from general.norms import RMSNorm
from helpers import einsum, get_param_count_dict, tiny_stories_true_parameter_count
from hooks import remove_hooks
from mixture_of_experts.cache import MoEFullCache
from moet_experiment.alibi_confi_block import ALiBiConfiTBlock
from moet_experiment.moe_blocks import MoETBlock
from moet_experiment.moet_config import MoETConfig

config = MoETConfig()
# tokeniser = tiktoken.encoding_for_model(config.tokeniser_string)
# Use the tokenizer from the TinyStories models
# Note using this tokenizer we only see the top 10K tokens. Hence the embedding matrix is only 10K x hidden_size really, even though it looks larger and we need to take this into account when counting parameters.
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")


class MoET(nn.Module):
    token_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    transformer_block: nn.Module
    moe_block: nn.Module
    vocab_size: int
    cache: MoEFullCache

    def __init__(
        self,
        *,
        config: MoETConfig = config,
    ):
        super().__init__()
        self.config = config

        self.num_layers = config.num_total_layers
        self.num_early_layers = config.num_early_layers

        layers: OrderedDict[str, nn.Module] = collections.OrderedDict()

        if config.confi_mlp:
            T_Block = ALiBiConfiTBlock
        else:
            T_Block = ALiBiTransformerBlock

        layers["moe_block_hash0"] = MoETBlock(
            config=config,
            layer_id=f"moe_layer_hash0",
            num_experts=config.num_experts_hash,
            parallel_ffn=False,
            group_size=1,
            router_str="hash",
        )
        for i in range(1, self.num_early_layers):
            if i % 2 == 0:
                layers[f"moe_block_early{i}"] = MoETBlock(
                    config=config,
                    layer_id=f"moe_layer_{i}",
                    num_experts=config.num_experts_early,
                    parallel_ffn=True,
                    group_size=1,
                    router_str="learned",
                )
            else:
                layers[f"transformer_block{i}"] = T_Block(
                    layer_index=i,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attn_heads,
                )
                # TODO: Switch out for ConfiNet Block
        for i in range(self.num_early_layers, self.num_layers):
            if i % 2 == 0:
                layers[f"moe_block_late{i}"] = MoETBlock(
                    config=config,
                    layer_id=f"moe_layer_{i}",
                    num_experts=config.num_experts_late,
                    parallel_ffn=False,
                    group_size=1,
                    router_str="learned",
                )
            else:
                layers[f"transformer_block{i}"] = T_Block(
                    layer_index=i,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attn_heads,
                )

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # Using ALiBi rather than learned positional embeddings
        # self.pos_embedding = nn.Embedding(
        #     config.max_position_embeddings, config.hidden_size
        # )
        self.sequential_layers = nn.Sequential(layers)
        self.final_norm = RMSNorm(shape_without_batch=(config.hidden_size,))
        self.cache = MoEFullCache({})

    def unembed(self, z: Float[t.Tensor, "batch seq hidden"]) -> t.Tensor:
        out = einsum(
            "b s h, v h -> b s v", z, self.token_embedding.weight
        )  # batch seq vocab_size
        return out

    def forward(self, input: t.Tensor) -> Tuple[t.Tensor, MoEFullCache]:
        """
        x: batch seq_length
        """

        # Get position of tokens

        # Get token embeddings. Note since we're using ALiBI there are no positional embeddings here
        x = self.token_embedding(input)

        for idx, layer in self.sequential_layers.named_children():
            # Layer types are MoETBlock (hash), MoETBlock (learned), ALiBiTransformerBlock
            if idx.startswith("moe_block_hash"):
                x, moe_layer_cache = layer(x, input)
            elif idx.startswith("moe"):
                x, moe_layer_cache = layer(x)
                self.cache[idx] = moe_layer_cache
            else:
                x, _attention_cache = layer(x)
        z = self.final_norm(x)

        # Unembed to get logits for each token
        out = self.unembed(z)  # batch seq vocab_size

        # Make sure the cache is consistent shapes even when the number of experts per layer varies
        self.cache.pad_with_0s()

        return out, self.cache

    def load_model(self, model_path: str):
        self.load_state_dict(t.load(model_path))


def main():
    model = MoET()
    # print(model)
    # print(tiny_stories_true_parameter_count(model, config.hidden_size))
    # print(get_param_count_dict(model).head(10))

    # Test model
    input_str = "Hello world"
    tokens_list = tokenizer(input_str)["input_ids"]
    tokens = t.tensor(tokens_list).unsqueeze(0)  # batch seq

    print(tokens.shape)
    out, _moe_cache = model(tokens)
    print(out.shape)
    print(out)


if __name__ == "__main__":
    main()
