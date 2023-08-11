"""Training script for PonderNet model."""

from datetime import datetime
from typing import Callable, Optional, Tuple

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
from typeguard import typechecked

import wandb
from gpt.config import GPTConfig
from helpers import check_leaf_nodes
from mixture_of_experts.tiny_stories import TinyStoriesDataset
from mixture_of_experts.train import ShakespeareDataset
from ponder.model import PonderCache, PonderNet

device = "cuda" if t.cuda.is_available() else "cpu"


config = GPTConfig()


# @typechecked
class Trainer:
    Optimiser: Callable
    optimiser: Optimizer

    def __init__(
        self,
        model: nn.Module = PonderNet(),
        config: GPTConfig = config,
        max_iters: Optional[int] = None,
    ):
        self.model = model
        self.config = config
        self.Optimiser = t.optim.SGD
        self.max_iters = max_iters or 1000

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
            sampler=RandomSampler(dataset, replacement=True)  #  type: ignore
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

    def estimate_loss(
        self,
        sample_batch_num,
        inputs: t.Tensor,
        targets: Optional[t.Tensor],
        training: bool,
        model: nn.Module,
        optimiser: Optimizer,
    ):
        ponder_cache: PonderCache

        num_layers = self.config.num_layers

        if targets is None:
            x = inputs[:, :-1].to(device)
            y = inputs[:, 1:].to(device)
        else:
            x, y = inputs.to(device), targets.to(device)

        model.train()
        optimiser.zero_grad()

        # Forward pass
        # Run model to get logits, note that we don't have ground truth for the final prediction
        logits, _kv_cache, ponder_cache = model(x)

        # Extract the Ponder probs (p) from the lambda vals
        lambdas = ponder_cache.lambda_vals  # num_layers batch seq
        lambda_complements = 1 - lambdas  # num_layers batch seq

        # TODO: Fix below
        exit_probs = t.stack(
            [lambdas[i] * t.prod(lambda_complements[:i]) for i in range(num_layers)]
        )  # num_layers batch seq
        flattened_exit_probs = rearrange(
            exit_probs, "layer batch seq -> layer (batch seq)"
        )

        exit_outputs = (
            ponder_cache.intermediate_vals
        )  #  num_layers, batch, seq_len, vocab_size

        flattened_exit_outputs = rearrange(
            exit_outputs, "layer batch seq vocab -> layer (batch seq) vocab"
        )  # num_layers, (batch * seq_len), vocab_size

        # Prepare targets
        flattened_targets = rearrange(y, "b s -> (b s)")  # bs

        # Calculate loss and backprop
        loss_tensor = t.zeros(num_layers)
        for layer_index, layer_output in enumerate(flattened_exit_outputs):
            # print(layer_output.shape)  # bs, vocab_size
            layer_loss = (
                F.cross_entropy(layer_output, flattened_targets)
                # * flattened_exit_probs[layer_index]
            )

            loss_tensor[layer_index] = layer_loss

        _, bs, _ = flattened_exit_outputs.shape

        loss = t.sum(loss_tensor) / bs

        if training:
            loss.backward()

            # Step optimiser
            optimiser.step()
            optimiser.zero_grad()

        return loss.item()

    def train(
        self, data_source: str = "tiny_stories", use_wandb: Optional[bool] = False
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

        # print(check_leaf_nodes(model))

        optimiser: Optimizer = self.Optimiser(
            model.parameters(), lr=self.config.learning_rate
        )
        best_loss = float("inf")
        sample_batch_num = 0

        wandb.init(project="moe")
        wandb_config = wandb.config
        wandb.watch(model)

        # Train the model
        for epoch in range(self.config.num_epochs):
            for batch_data in train_dataloader:
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
                if sample_batch_num > self.config.max_iters:
                    checkpoint = {
                        "model": model.state_dict(),
                        # "optimizer": optimiser.state_dict(),
                        "model_config": self.config,
                        "iter_num": -1,
                        "best_val_loss": best_loss,
                    }
                    self.save_model(
                        checkpoint=checkpoint,
                        model_name="moe_post_training",
                    )
                    break
                print(
                    f"\n\nSample batch num: {sample_batch_num}/{self.config.max_iters}"
                )

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
                        f"Epoch: {epoch}, Batch: {sample_batch_num}, Test Loss: {test_loss}"
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
                            # "optimizer": optimiser.state_dict(),
                            "model_config": self.config,
                            "iter_num": sample_batch_num,
                            "best_val_loss": best_loss,
                        }
                        self.save_model(
                            checkpoint=checkpoint,
                            model_name="moe_checkpoint",
                        )
                        print(f"New best loss: {best_loss}. Checkpoint saved")

        return model

    def save_model(self, checkpoint: dict, model_name: str = "ponder") -> None:
        """Save the model to the checkpoint."""

        # Get the current date
        current_date = datetime.now().strftime("%Y-%m-%d")

        full_dest = f"models/{model_name}_{current_date}.pt"
        t.save(checkpoint, full_dest)

        print(f"Saved model to {full_dest}")

    def load_model(self, checkpoint_path: str) -> nn.Module:
        """Load a model from the checkpoint."""

        checkpoint = t.load(checkpoint_path)

        self.model.load_state_dict(checkpoint["model"])

        print(f"Loaded model from {checkpoint_path}")
        print(
            f"Best val loss: {checkpoint['best_val_loss']} for iter {checkpoint['iter_num']}"
        )
        return self.model

    @property
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def main():
    # Set up the trainer
    trainer = Trainer(model=PonderNet(), max_iters=1000)

    print(trainer.count_parameters)

    # Train and save the model
    trainer.model = trainer.train(use_wandb=False)

    # Load model
    # model = ...

    # next_token = sample_next_token(
    #     input="One day, I went to meet my friend Jill. She has brown hair and",
    #     model=model,
    # )
    # print(next_token)


if __name__ == "__main__":
    main()
