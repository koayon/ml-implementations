from datetime import datetime
from functools import lru_cache
from typing import Callable, Optional, Tuple, Union

import tiktoken
import torch as t
import torch.nn as nn
from einops import rearrange, repeat
from torch import optim
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.switch_transformers.modeling_switch_transformers import (
    router_z_loss_func,
)
from typeguard import typechecked

import wandb
from mixture_of_experts.cache import ExpertChoiceFullCache
from mixture_of_experts.config import MoEConfig
from mixture_of_experts.model import SparseMoETransformer, sample_next_token
from mixture_of_experts.tiny_stories import TinyStoriesDataset
from moet_experiment.model import MoET
from moet_experiment.moet_config import MoETConfig
from optimisers.adam import Adam
from optimisers.sgd import SGD
from optimisers.sophia import Sophia

device = "cuda" if t.cuda.is_available() else "cpu"

OPTIMISERS = {
    "my_adam": Adam,
    "my_sgd": SGD,
    "my_sophia": Sophia,
    "adam": t.optim.Adam,
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


# @typechecked
class Trainer:
    Optimiser: Callable
    optimiser: Optimizer

    def __init__(
        self,
        config: Union[MoEConfig, MoETConfig],
        model: nn.Module,
        model_name: str,
        optimiser_string: str = "adam",
        max_iters: Optional[int] = None,
        model_load_path: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.optimiser_string = optimiser_string
        self.Optimiser = OPTIMISERS[optimiser_string]
        if max_iters:
            self.config.max_steps = max_iters
        self.model_name = model_name
        if model_load_path:
            _model, self.optimiser = self.load_model(model_load_path)

    @lru_cache
    def get_text_data(
        self,
        data_source: str = "data/tiny_shakespeare.txt",
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("gpt2")
    ) -> Tuple[t.Tensor, t.Tensor]:
        """Get the text dataset (Shakespeare)."""

        # Get text from file and convert to tensors for training
        with open(data_source, "r") as f:
            text = f.read()


        tokenised_text = tokenizer(text, return_tensors = "pt")  # list of ints
        full_data = t.tensor(tokenised_text, dtype=t.long, device=device)  # len_of_text

        # Split into train and test sets
        train_split = int(len(tokenised_text) * self.config.train_test_split)

        train_data = full_data[:train_split]
        test_data = full_data[train_split:]

        return train_data, test_data  # vectors of ints

    @lru_cache
    def dataset_to_dataloader(
        self, dataset: Dataset, random_sampler: bool
    ) -> DataLoader:
        """Convert a dataset to a dataloader."""
        return DataLoader(
            dataset,
            sampler=RandomSampler(dataset, replacement=True)  #  type: ignore
            if random_sampler
            else None,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    @lru_cache
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

    @lru_cache
    def get_tiny_shakespeare_dataset(self) -> Tuple[DataLoader, DataLoader]:
        # Get dataset
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_string)
        train_data, test_data = self.get_text_data(tokenizer = tokenizer)
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

    def estimate_loss(
        self,
        sample_batch_num,
        inputs: t.Tensor,
        targets: Optional[t.Tensor],
        training: bool,
        model: nn.Module,
        optimiser: Optimizer,
    ):
        MoE_cache: ExpertChoiceFullCache

        if targets is None:
            # Note that we don't have ground truth for the final prediction so we shift along one.
            x = inputs[:, :-1].to(device)
            y = inputs[:, 1:].to(device)
        else:
            x, y = inputs.to(device), targets.to(device)

        model.train()
        optimiser.zero_grad()

        # Forward pass
        # Run model to get logits
        logits, MoE_cache = model(x)  # batch, seq_len, vocab_size

        # Extract the router logits from the cache and use for router z-loss
        router_logits = MoE_cache.routing_logits_tensor  # layer, bs, num_experts

        router_logits = rearrange(
            router_logits, "l (b s) e -> b s (l e)", b=self.config.batch_size
        )
        router_z_loss = router_z_loss_func(router_logits=router_logits)

        # Flatten for cross entropy
        flattened_logits = rearrange(logits, "b s v -> (b s) v")  # bs, vocab_size
        flattened_targets = rearrange(y, "b s -> (b s)")  # bs

        # Calculate loss and backprop
        loss = F.cross_entropy(flattened_logits, flattened_targets)
        loss += router_z_loss

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

                optimiser.update_hessian()  # type: ignore
                optimiser.zero_grad()

        return loss.item()

    def train(
        self, data_source: str = "tiny_stories", optimiser: Optional[Optimizer] = None
    ) -> nn.Module:
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
        optimiser = optimiser or self.Optimiser(
            model.parameters(), lr=self.config.learning_rate
        )
        best_loss = float("inf")
        sample_batch_num = 0

        wandb.init(project=self.model_name)
        wandb_config = wandb.config
        wandb.watch(model)

        # Train the model
        for epoch in range(self.config.num_epochs):
            for batch_data in tqdm(train_dataloader):
                if data_source == "tiny_shakespeare":
                    train_loss = self.estimate_loss(
                        sample_batch_num=sample_batch_num,
                        inputs=batch_data,
                        targets=None,
                        training=True,
                        model=model,
                        optimiser=optimiser,
                    )
                elif data_source == "tiny_stories":
                    x, y = batch_data
                    train_loss = self.estimate_loss(
                        sample_batch_num=sample_batch_num,
                        inputs=x,
                        targets=y,
                        training=True,
                        model=model,
                        optimiser=optimiser,
                    )

                sample_batch_num += 1
                if sample_batch_num > self.config.max_steps:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimiser.state_dict(),
                        "model_config": self.config,
                        "iter_num": -1,
                        "best_val_loss": best_loss,
                    }
                    self.save_model(
                        checkpoint=checkpoint,
                        model_name=f"{self.model_name}/post_training",
                    )
                    break
                # print(f"Sample batch num: {sample_batch_num}/{self.config.max_iters}")

                if sample_batch_num % self.config.eval_steps == 0:
                    # Evaluate model
                    model.eval()
                    test_loss = 0
                    num_batches = 0
                    for batch_data in test_dataloader:
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
                        # print(f"Num test batches: {num_batches}/5")
                        if num_batches > 5:
                            break
                    test_loss /= num_batches
                    print(
                        f"-----\nEpoch: {epoch}, Batch: {sample_batch_num}, Test Loss: {test_loss}\n-----"
                    )
                    # Log to wandb
                    wandb.log(
                        {
                            "iter": sample_batch_num,
                            "loss/train": train_loss,
                            "loss/val": test_loss,
                            "lr": self.config.learning_rate,
                        }
                    )

                    # Save checkpoint
                    if test_loss < best_loss:
                        best_loss = test_loss
                        checkpoint = {
                            "model": model.state_dict(),
                            "optimizer": optimiser.state_dict(),
                            "model_config": self.config,
                            "iter_num": sample_batch_num,
                            "best_val_loss": best_loss,
                        }
                        self.save_model(
                            checkpoint=checkpoint,
                            model_name=f"{self.model_name}/checkpoint_{sample_batch_num}",
                        )
                        print(f"New best loss: {best_loss}. Checkpoint saved")


        return model

    def save_model(self, checkpoint: dict, model_name: str = "moe") -> None:
        """Save the model to the checkpoint."""

        # Get the current date
        current_date = datetime.now().strftime("%Y-%m-%d")

        full_dest = f"models/{model_name}_{current_date}.pt"
        t.save(checkpoint, full_dest)

        print(f"Saved model to {full_dest}")

    def load_model(self, checkpoint_path: str) -> Tuple[nn.Module, Optimizer]:
        """Load a model from the checkpoint."""

        checkpoint = t.load(checkpoint_path)

        self.model.load_state_dict(checkpoint["model"])

        optimiser = self.Optimiser(
            self.model.parameters(), lr=self.config.learning_rate
        )
        optimiser.load_state_dict(checkpoint["optimizer"])

        print(f"Loaded model from {checkpoint_path}")
        print(
            f"Best val loss: {checkpoint['best_val_loss']} for iter {checkpoint['iter_num']}"
        )
        return self.model, optimiser

    @property
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def main():
    # Set up the trainer
    model = MoET()
    model.to(device)

    trainer = Trainer(
        model=MoET(),
        config=MoETConfig(),
        model_name="moet",
        model_load_path="models/moet/post_training_2023-08-31.pt",
    )
    print("Created trainer")

    # Train and save the model
    trainer.model = trainer.train(data_source="tiny_stories")

    # print(trainer.count_parameters)

    # Load model
    model_filepath = ...
    # model = trainer.load_model("models/{model_filepath}")

    # next_token = sample_next_token(
    #     input="One day, I went to meet my friend Jill. She has brown hair and",
    #     model=model,
    # )
    # print(next_token)


if __name__ == "__main__":
    main()
