from datetime import datetime
from typing import Optional, Tuple

import tiktoken
import torch as t
import torch.nn as nn
from einops import rearrange, repeat
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm.auto import tqdm
from transformers.models.switch_transformers.modeling_switch_transformers import (
    router_z_loss_func,
)

from mixture_of_experts.config import MoEConfig
from mixture_of_experts.model import SparseMoETransformer
from mixture_of_experts.tiny_stories import TinyStoriesDataset
from optimisers.adam import Adam
from optimisers.sgd import SGD
from optimisers.sophia import Sophia

device = "cuda" if t.cuda.is_available() else "cpu"

config = MoEConfig()

OPTIMISERS = {
    "adam": Adam,
    "sgd": SGD,
    "sophia": Sophia,
    "pytorch-adam": t.optim.Adam,
}


class ShakespeareDataset(Dataset):
    """Train Dataset for Shakespeare data."""

    def __init__(self, data: t.Tensor, block_size: int):
        if block_size <= 0:
            raise ValueError("block_size should be a positive integer")
        if data.shape[0] < block_size:
            raise ValueError(
                "block_size should not be greater than the first dimension of data"
            )

        self.data = data
        self.block_size = block_size

    def __len__(self):
        # This is the number of blocks of size `block_size` in `data`
        return self.data.shape[0] // self.block_size

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError("Index out of bounds")

        return self.data[idx * self.block_size : (idx + 1) * self.block_size]


class Trainer:
    Optimiser: Optimizer

    def __init__(
        self,
        model: nn.Module = SparseMoETransformer(),
        optimiser_string: str = "adam",
        config: MoEConfig = config,
        max_iters: Optional[int] = None,
    ):
        self.model = model
        self.config = config
        self.optimiser_string = optimiser_string
        self.Optimiser = OPTIMISERS[optimiser_string]
        if max_iters:
            self.config.max_iters = max_iters

    def get_text_data(
        self,
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
        train_split = int(len(tokenised_text) * self.config.train_test_split)

        train_data = full_data[:train_split]
        test_data = full_data[train_split:]

        return train_data, test_data  # vectors of ints

    def dataset_to_dataloader(
        self, dataset: Dataset, random_sampler: bool
    ) -> DataLoader:
        """Convert a dataset to a dataloader."""
        return DataLoader(
            dataset,
            sampler=RandomSampler(dataset, replacement=True)
            if random_sampler
            else None,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def get_tiny_stories_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = TinyStoriesDataset(
            split="train", max_seq_len=self.config.block_size
        )
        test_dataset = TinyStoriesDataset(
            split="test", max_seq_len=self.config.block_size
        )
        train_dataloader = self.dataset_to_dataloader(
            train_dataset, random_sampler=False
        )
        test_dataloader = self.dataset_to_dataloader(test_dataset, random_sampler=False)

        # data_iter = iter(train_dataloader)
        # X, y = next(data_iter)  # lis
        # print(X.shape)
        # print(y.shape)
        return train_dataloader, test_dataloader

    def get_tiny_shakespeare_dataset(self) -> Tuple[DataLoader, DataLoader]:
        # Get dataset
        train_data, test_data = self.get_text_data()
        train_dataset = ShakespeareDataset(
            train_data, block_size=self.config.block_size
        )
        test_dataset = ShakespeareDataset(test_data, block_size=self.config.block_size)

        # Create dataloaders

        train_dataloader = self.dataset_to_dataloader(
            train_dataset, random_sampler=True
        )
        test_dataloader = self.dataset_to_dataloader(test_dataset, random_sampler=True)

        return train_dataloader, test_dataloader

    @t.inference_mode()
    def evaluate(self, test_dataloader: DataLoader) -> float:
        """Evaluate the model on the test set."""

        total_loss = 0
        batch_data = next(iter(test_dataloader)).to(device)

        model = self.model.to(device)

        # Predictions are shifted right by one
        target_tokens = batch_data[:, 1:]  # batch, seq_len - 1

        # Run model to get logits, note that we don't have ground truth for the final prediction
        logits, cache = model(batch_data)

        # Extract the router logits from the cache and use for router z-loss
        (G, token_assignments, router_logits) = cache
        # Router logits is shape bs, num_experts
        router_logits = rearrange(
            router_logits, "(bs) e -> b s e", b=self.config.batch_size
        )
        router_z_loss = router_z_loss_func(router_logits=router_logits)

        logits = logits[:, :-1, :]  # batch, seq_len - 1, vocab_size

        # Flatten for cross entropy
        flattened_logits = rearrange(logits, "b s v -> (b s) v")
        flattened_targets = rearrange(target_tokens, "b s -> (b s)")

        _probs = t.softmax(logits, dim=-1)  # batch, seq_len - 1, vocab_size

        loss = F.cross_entropy(flattened_logits, flattened_targets)
        total_loss += loss.item()

        return total_loss / self.config.batch_size

    def estimate_loss_tiny_shakespeare(
        self, sample_batch_num, batch_data, model, optimiser
    ):
        batch_data = batch_data.to(device)

        model.train()
        optimiser.zero_grad()

        # Get targets
        target_tokens = batch_data[:, 1:]  # batch seq_len - 1

        # Forward pass
        logits, _cache = model(batch_data)
        logits = logits[:, :-1, :]  # batch seq_len - 1, vocab_size

        # Flatten for cross entropy
        flattened_logits = rearrange(logits, "b s v -> (b s) v")  # bs, vocab_size
        flattened_targets = rearrange(target_tokens, "b s -> (b s)")  # bs

        # Calculate loss and backprop
        loss = F.cross_entropy(flattened_logits, flattened_targets)
        loss.backward()

        # Step optimiser
        optimiser.step()
        optimiser.zero_grad()

        if (
            sample_batch_num % self.config.sophia_hessian_update_steps
            and self.optimiser_string == "sophia"
        ):
            # Update Hessian

            # Compute forward pass and sample loss by logits to a sample (regularisation)
            logits, _cache = model(batch_data)  # batch, seq_len, vocab_size

            flattened_logits: t.Tensor = rearrange(
                logits, "b s v -> (b s) v"
            )  # bs, vocab_size

            samp_dist = Categorical(logits=flattened_logits)
            y_sample = samp_dist.sample()  # bs

            loss_sampled = F.cross_entropy(flattened_logits, y_sample)
            loss_sampled.backward()

            optimiser.update_hessian()
            optimiser.zero_grad()

    def estimate_loss(
        self, sample_batch_num, inputs, targets, training: bool, model, optimiser
    ):
        if targets is None:
            x = inputs[:, :-1].to(device)
            y = inputs[:, 1:].to(device)
        else:
            x, y = inputs.to(device), targets.to(device)

        model.train()
        optimiser.zero_grad()

        # Forward pass
        logits, cache_dict = model(x)

        # Extract the router logits from the cache and use for router z-loss
        # router_logits_list = []
        # for _layer_name, cache in cache_dict.items():
        #     print(len(cache))
        #     print(cache)
        #     print(cache[2].shape)
        #     (_G, _token_assignments, router_logits) = cache
        #     router_logits_list.append(router_logits)

        # router_logits = t.stack(router_logits_list, dim=0)  # layer bs, num_experts
        # # Router logits is shape bs, num_experts
        # router_logits = rearrange(
        #     router_logits,
        #     "layer (bs) e -> b s (layer e)",
        #     b=self.config.batch_size,
        #     layer=self.config.num_layers,
        # )
        # router_z_loss = router_z_loss_func(router_logits=router_logits)

        # Flatten for cross entropy
        flattened_logits = rearrange(logits, "b s v -> (b s) v")  # bs, vocab_size
        flattened_targets = rearrange(y, "b s -> (b s)")  # bs

        # Calculate loss and backprop
        loss = F.cross_entropy(flattened_logits, flattened_targets)
        # loss += router_z_loss

        if training:
            loss.backward()

            # Step optimiser
            optimiser.step()
            optimiser.zero_grad()

            if (
                sample_batch_num % self.config.sophia_hessian_update_steps
                and self.optimiser_string == "sophia"
            ):
                # Update Hessian

                # Compute forward pass and sample loss by logits to a sample (regularisation)
                logits, _cache = model(x)  # batch, seq_len, vocab_size

                flattened_logits: t.Tensor = rearrange(
                    logits, "b s v -> (b s) v"
                )  # bs, vocab_size

                samp_dist = Categorical(logits=flattened_logits)
                y_sample = samp_dist.sample()  # bs

                loss_sampled = F.cross_entropy(flattened_logits, y_sample)
                loss_sampled.backward()

                optimiser.update_hessian()
                optimiser.zero_grad()

        return loss.item()

    def train(self, data_source: str = "tiny_stories") -> nn.Module:
        """Train the model on the data source."""

        # Print config and model parameters
        print(f"Config: \n {self.config} \n")
        print(f"Number of parameters: {self.count_parameters}")

        if data_source == "tiny_stories":
            train_dataloader, test_dataloader = self.get_tiny_stories_dataloaders()
        elif data_source == "tiny_shakespeare":
            train_dataloader, test_dataloader = self.get_tiny_shakespeare_dataset()
        else:
            raise ValueError("Invalid data source")

        print("Created dataloaders")

        # t.autograd.set_detect_anomaly(True)

        model = self.model.to(device)
        optimiser = self.Optimiser(model.parameters(), lr=self.config.learning_rate)

        # Train the model
        for epoch in range(self.config.num_epochs):
            sample_batch_num = 0
            for batch_data in tqdm(train_dataloader):
                if data_source == "tiny_shakespeare":
                    _loss = self.estimate_loss(
                        sample_batch_num=sample_batch_num,
                        inputs=batch_data,
                        targets=None,
                        training=True,
                        model=model,
                        optimiser=optimiser,
                    )
                elif data_source == "tiny_stories":
                    x, y = batch_data
                    _loss = self.estimate_loss(
                        sample_batch_num=sample_batch_num,
                        inputs=x,
                        targets=y,
                        training=True,
                        model=model,
                        optimiser=optimiser,
                    )

                sample_batch_num += 1
                print(f"Sample batch num: {sample_batch_num}")

                if sample_batch_num % self.config.eval_steps == 0:
                    # if True:
                    model.eval()
                    for batch_data in test_dataloader:
                        test_loss = 0
                        num_batches = 0
                        if data_source == "tiny_shakespeare":
                            test_loss += self.estimate_loss(
                                sample_batch_num=sample_batch_num,
                                inputs=batch_data,
                                targets=None,
                                training=False,
                                model=model,
                                optimiser=optimiser,
                            )
                        elif data_source == "tiny_stories":
                            x, y = batch_data
                            test_loss += self.estimate_loss(
                                sample_batch_num=sample_batch_num,
                                inputs=x,
                                targets=y,
                                training=False,
                                model=model,
                                optimiser=optimiser,
                            )
                        num_batches += 1
                    test_loss /= num_batches
                    print(
                        f"Epoch: {epoch}, Batch: {sample_batch_num}, Test Loss: {test_loss}"
                    )

        return model

    def save_model(self, model_name: str) -> None:
        """Save the model to the model_dest."""

        if model_name == "":
            model_name = "moe"

        # Get the current date
        current_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        full_dest = f"models/{model_name}_{current_date}.pt"

        t.save(self.model.state_dict(), full_dest)
        print(f"Saved model to {full_dest}")

    def load_model(self, model_dest: str) -> None:
        """Load a model from the model_dest."""

        self.model.load_state_dict(t.load(model_dest))
        print(f"Loaded model from {model_dest}")

    @property
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def main():
    # Set up the trainer
    trainer = Trainer(model=SparseMoETransformer(), max_iters=1000)

    # Train and save the model
    trainer.model = trainer.train()
    trainer.save_model("pytorch_adam_1000")


if __name__ == "__main__":
    main()
