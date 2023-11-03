from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Union

import torch as t
import torch.nn as nn
import torch.optim as optim
from einops import rearrange, repeat
from transformers.activations import NewGELUActivation

from gpt.cached_attention import AttentionCache, UnidirectionalAttention
from gpt.cross_attention import CrossAttentionLayer
from gpt.transformer_block import GPT2Block

ACTIVATION_FUNCTIONS = dict(
    relu=nn.ReLU(),
    gelu=nn.GELU(),
    new_gelu=NewGELUActivation(),
)


class EncDecTransformerBlock(nn.Module):
    """
    Transformer block on the decoder side of an encoder-decoder transformer.
    Contains cross-attention, self-attention, and MLP layers.
    """

    cross_attn: CrossAttentionLayer
    self_attn: UnidirectionalAttention
    linear1: nn.Linear
    linear2: nn.Linear
    ln1: nn.LayerNorm
    ln2: nn.LayerNorm

    def __init__(
        self,
        layer_index: int,
        hidden_size: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        activation_function: str = "gelu",
    ):
        super().__init__()

        self.layer_index = layer_index

        # Attention part

        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.self_attn = UnidirectionalAttention(
            hidden_size, num_heads, dropout=dropout, autoregressive=True
        )

        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.cross_attn = CrossAttentionLayer(
            encoder_hidden_size=hidden_size,
            decoder_hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            head_size=hidden_size // num_heads,
        )

        # MLP part

        self.MLP = nn.Sequential(
            OrderedDict(
                [
                    ("ln3", nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)),
                    ("linear1", nn.Linear(hidden_size, hidden_size * 4)),
                    ("activation_function", ACTIVATION_FUNCTIONS[activation_function]),
                    ("linear2", nn.Linear(hidden_size * 4, hidden_size)),
                    ("dropout", nn.Dropout(dropout)),
                ]
            )
        )

    def forward(
        self,
        x: t.Tensor,
        encoder_outputs: t.Tensor,
        layer_cache: Optional[AttentionCache] = None,
    ) -> Tuple[t.Tensor, Optional[AttentionCache]]:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        y = self.ln1(x)
        y, _ = self.self_attn(y, layer_cache=layer_cache)

        x = x + y

        y = self.ln2(x)
        y, layer_cache = self.cross_attn(
            y, encoder_outputs=encoder_outputs, layer_cache=layer_cache
        )

        x = x + self.MLP(x)

        return x, layer_cache


def train(batch_size=1_000, num_epochs=1_000, seq_len=2, hidden_size=2, num_heads=1):
    enc_dec_block = EncDecTransformerBlock(
        layer_index=0, hidden_size=hidden_size, num_heads=num_heads
    )
    gpt_block = GPT2Block(layer_index=0, hidden_size=hidden_size, num_heads=num_heads)
    encoder_output = (
        repeat(t.tensor([[1, 3], [-0.3, 0.6]]), "s h -> b s h", b=batch_size)
        # + t.randn(batch_size, seq_len, hidden_size) * 0.2
    )
    decoder_input = encoder_output = (
        repeat(t.tensor([[-4, 1], [-0.8, 0.1]]), "s h -> b s h", b=batch_size)
        # + t.randn(batch_size, seq_len, hidden_size) * 0.2
    )

    # signal_from_encoder = encoder_output * 2 + 5
    signal_from_encoder = 0
    # seq_signal = repeat(t.arange(seq_len), "s -> b s h", b=batch_size,
    # h=hidden_size)
    seq_signal = 0
    signal_from_input = decoder_input * 5.3
    # signal_from_input = 0
    # noise = t.randn(batch_size, seq_len, hidden_size)
    noise = 0

    preds = signal_from_encoder + seq_signal + signal_from_input + noise

    for model in [enc_dec_block, gpt_block]:
        print("-------------------")
        print("model", model)

        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Trying to reconstruct the signal
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            if model == enc_dec_block:
                y, _ = model(decoder_input, encoder_output)
            else:
                y, _ = model(decoder_input)
            loss = t.sum((y - preds) ** 2)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, {loss/batch_size}")

            # print the l2 loss of the weights


if __name__ == "__main__":
    # Test GPT2Block
    train()
    # block = EncDecTransformerBlock(layer_index=0)
    # x = t.rand(2, 10, 768)
    # y: t.Tensor = block(x)
    # print(y.shape)
    # print(y)
