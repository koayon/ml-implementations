import time
from typing import Tuple

import deepspeed
import tiktoken
import torch as t
import torch.nn as nn
import tqdm
from einops import rearrange, repeat
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from switch_transformer.expert_choice_layer import ExpertChoiceFFN

device = "cuda" if t.cuda.is_available() else "cpu"


def get_shakespeare_data() -> Tuple[t.Tensor, t.Tensor]:
    """Get the Shakespeare dataset."""
    data_source = "data/tiny_shakespeare.txt"
    # Get text from file and convert to tensors for training
    with open(data_source, "r") as f:
        text = f.read()
    tokeniser = tiktoken.encoding_for_model("gpt2")
    tokenised_text = tokeniser.encode(text)
    train_split = int(len(tokenised_text) * 0.9)
    full_data = t.tensor(tokenised_text, dtype=t.long, device=device).unsqueeze(0)

    train_data = full_data[:, :train_split]
    test_data = full_data[:, train_split:]
    return train_data, test_data


class ShakespeareDataset(Dataset):
    """Train Dataset for Shakespeare data."""

    def __init__(self, data: t.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return self.data.shape[1] // self.block_size

    def __getitem__(self, idx):
        return self.data[:, idx * self.block_size : (idx + 1) * self.block_size]


def evaluate(model: nn.Module, test_dataloader: DataLoader) -> float:
    """Evaluate the model on the test set."""

    with t.inference_mode():
        total_loss = 0
        for batch, batch_data in enumerate(test_dataloader):
            output = model(batch_data)
            loss = F.nll_loss(output.view(-1, model.vocab_size), batch_data.view(-1))
            total_loss += loss.item()
        return total_loss / len(test_dataloader)


def train(model: nn.Module) -> nn.Module:
    """Train the model on the data source."""
    # Get dataset
    train_data, test_data = get_shakespeare_data()
    train_dataset = ShakespeareDataset(train_data, block_size=128)
    test_dataset = ShakespeareDataset(test_data, block_size=128)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset, replacement=True),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=RandomSampler(train_dataset, replacement=True),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    # Set up the optimiser
    optimiser = t.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    # Train the model
    for epoch in range(1, 2):
        model.train()
        for batch_num, batch_data in enumerate(train_dataloader):
            batch_data = batch_data.squeeze(0)
            batch_data.to(device)  # seq_len

            optimiser.zero_grad()

            target_tokens = batch_data[1:]  # seq_len - 1
            targets = F.one_hot(
                target_tokens, model.vocab_size
            ).float()  # seq_len - 1, vocab_size
            logits = model(batch_data)[:-1]  # seq_len - 1, vocab_size
            probs = F.softmax(logits, dim=-1)  # seq_len - 1, vocab_size

            # loss = F.nll_loss(output.view(-1, model.vocab_size), batch_data.view(-1))
            loss = F.cross_entropy(probs, targets)
            loss.backward()
            optimiser.step()

            test_loss = evaluate(model, test_dataloader)
            if batch_num % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_num}, Test Loss: {test_loss}")

    return model


# TODO: Add deepspeed
