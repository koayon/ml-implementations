import torch as t
import torch.nn as nn
import torch.optim as optim
from einops import rearrange, repeat
from nltk import flatten

from arithmetic.model import ArithmeticNet


def train(batch_size=1000, num_epochs=1000, seq_len=10):
    model = ArithmeticNet()
    # hidden_size = model.model_config.hidden_size

    encoder_input_ids = t.randint(
        low=0, high=10, size=(batch_size, seq_len)
    )  # batch, seq
    decoder_input_ids = repeat(
        t.arange(seq_len), "seq -> batch seq", batch=batch_size
    )  # batch, seq

    sum_of_encoder_ids = t.sum(encoder_input_ids, dim=-1, keepdim=True)  # batch, 1
    decoder_input_ids = decoder_input_ids + sum_of_encoder_ids

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Trying to learn the function.
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits, _, _, _ = model(
            encoder_input_ids, decoder_input_ids
        )  # batch, seq, vocab

        # Shift the decoder input ids
        targets = decoder_input_ids[:, 1:]
        logits = logits[:, :-1]

        flattened_targets = rearrange(targets, "batch seq -> (batch seq)")
        flattened_logits = rearrange(logits, "batch seq vocab -> (batch seq) vocab")
        loss = nn.CrossEntropyLoss()(flattened_logits, flattened_targets)
        loss.sum().backward()
        optimizer.step()
        print(f"Epoch {epoch}, {loss/batch_size}")


if __name__ == "__main__":
    train()
