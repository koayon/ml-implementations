import time
from typing import Tuple

import deepspeed
import tiktoken
import torch as t
import torch.nn as nn
from config import MoEConfig
from einops import rearrange, repeat
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm

from mixture_of_experts.model import SparseMoETransformer

device = "cuda" if t.cuda.is_available() else "cpu"

config = MoEConfig()


def get_text_data(
    data_source: str = "data/tiny_shakespeare.txt",
) -> Tuple[t.Tensor, t.Tensor]:
    """Get the text dataset (Shakespeare)."""

    # Get text from file and convert to tensors for training
    with open(data_source, "r") as f:
        text = f.read()
    tokeniser = tiktoken.encoding_for_model("gpt2")
    tokenised_text = tokeniser.encode(text)  # list of ints
    full_data = t.tensor(tokenised_text, dtype=t.long, device=device)  # len_of_text

    # Split into train and test sets
    train_split = int(len(tokenised_text) * 0.9)

    train_data = full_data[:train_split]
    test_data = full_data[train_split:]

    return train_data, test_data  # vectors of ints


class ShakespeareDataset(Dataset):
    """Train Dataset for Shakespeare data."""

    def __init__(self, data: t.Tensor, block_size: int):
        data.to(device)
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return self.data.shape[0] - self.block_size

    def __getitem__(self, idx):
        return self.data[idx * self.block_size : (idx + 1) * self.block_size]


def evaluate(model: nn.Module, test_dataloader: DataLoader) -> float:
    """Evaluate the model on the test set."""
    with t.inference_mode():
        total_loss = 0
        for _batch_num, batch_data in enumerate(test_dataloader):
            # Predictions are shifted right by one
            target_tokens = batch_data[:, 1:]  # batch, seq_len - 1

            # Run model to get logits, note that we don't have ground truth for the prediction
            logits, _cache = model(batch_data)
            logits = logits[:, :-1, :]  # batch, seq_len - 1, vocab_size

            # Flatten for cross entropy
            flattened_logits = rearrange(logits, "b s v -> (b s) v")
            flattened_targets = rearrange(target_tokens, "b s -> (b s)")

            probs = t.softmax(logits, dim=-1)  # batch, seq_len - 1, vocab_size

            loss = F.cross_entropy(flattened_logits, flattened_targets)
            total_loss += loss.item()

        return total_loss / len(test_dataloader)


def train(model: nn.Module) -> nn.Module:
    """Train the model on the data source."""
    # Get dataset
    train_data, test_data = get_text_data()
    train_dataset = ShakespeareDataset(train_data, block_size=128)
    test_dataset = ShakespeareDataset(test_data, block_size=128)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset, replacement=True),
        batch_size=8,
        shuffle=False,
        num_workers=6,
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=RandomSampler(test_dataset, replacement=True),
        batch_size=8,
        shuffle=False,
        num_workers=6,
    )

    # Set up the optimiser
    optimiser = t.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    # Train the model
    for epoch in range(1, 2):
        for sample_batch_num, batch_data in enumerate(
            train_dataloader
        ):  # batch, seq_len
            model.train()

            optimiser.zero_grad()

            target_tokens = batch_data[:, 1:]  # batch seq_len - 1
            logits, _cache = model(batch_data)
            logits = logits[:, :-1, :]  # batch seq_len - 1, vocab_size

            # Flatten for cross entropy
            flattened_logits = rearrange(logits, "b s v -> (b s) v")  # bs, vocab_size
            flattened_targets = rearrange(target_tokens, "b s -> (b s)")  # bs

            loss = F.cross_entropy(flattened_logits, flattened_targets)

            loss.backward()
            optimiser.step()

            if sample_batch_num % 5 == 0:
                # if True:
                model.eval()
                test_loss = evaluate(model, test_dataloader)
                print(
                    f"Epoch: {epoch}, Batch: {sample_batch_num}, Test Loss: {test_loss}"
                )
                # print(f"Epoch: {epoch}, Batch: {batch_num}, Test Loss: {loss}")

    return model


def save_model(model, model_dest):
    """Save the model to the model_dest."""
    full_dest = f"models/{model_dest}"
    t.save(model.state_dict(), full_dest)
    print(f"Saved model to {full_dest}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Set up the model
    model = SparseMoETransformer(config=config).to(device)
    print(f"{count_parameters(model)=}")  # 26M parameter model

    # Train the model
    trained_model = train(model)

    # Save the model
    save_model(trained_model, "moe.pt")


if __name__ == "__main__":
    main()
