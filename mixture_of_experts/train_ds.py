from datetime import datetime
from json import load
from typing import Tuple

import deepspeed
import tiktoken
import torch as t
import torch.nn as nn
from einops import rearrange, repeat
from torch import mode, optim
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm.auto import tqdm
from transformers.models.switch_transformers.modeling_switch_transformers import (
    router_z_loss_func,
)

from mixture_of_experts.config import MoEConfig
from mixture_of_experts.model import SparseMoETransformer
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

ds_config = {
    "train_micro_batch_size_per_gpu": config.batch_size,
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 1, "offload_optimizer": {"device": "cpu"}},
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
        config: MoEConfig = config,
    ):
        self.model = model
        self.model_engine = None
        self.config = config

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
        loss += router_z_loss

        total_loss += loss.item()

        return total_loss / self.config.batch_size

    def train(self) -> nn.Module:
        """Train the model on the data source."""

        # Print config and model parameters
        print(f"Config: \n {self.config} \n")
        print(f"Number of parameters: {self.count_parameters}")

        # Get dataset
        train_data, test_data = self.get_text_data()
        train_dataset = ShakespeareDataset(
            train_data, block_size=self.config.block_size
        )
        test_dataset = ShakespeareDataset(test_data, block_size=self.config.block_size)

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset, replacement=True),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=6,
        )
        test_dataloader = DataLoader(
            test_dataset,
            sampler=RandomSampler(test_dataset, replacement=True),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=6,
        )

        print("Created dataloaders")

        (
            model_engine,
            _optimizer,
            _training_dataloader,
            _lr_scheduler,
        ) = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=ds_config,
            optimizer=None,
            training_data=None,
            lr_scheduler=None,
        )

        # Train the model
        for epoch in range(self.config.num_epochs):
            data_iter = iter(train_dataloader)
            for sample_batch_num in tqdm(range(self.config.max_iters)):
                # Iterate over batches
                try:
                    batch_data = next(data_iter)
                except:
                    data_iter = iter(train_dataloader)
                    batch_data = next(data_iter)

                batch_data = batch_data.to(device)

                # Get targets
                target_tokens = batch_data[:, 1:]  # batch seq_len - 1

                # Forward pass
                logits, cache = model_engine(batch_data)

                # Extract the router logits from the cache and use for router z-loss
                (G, token_assignments, router_logits) = cache
                # Router logits is shape bs, num_experts
                router_logits = rearrange(
                    router_logits, "(bs) e -> b s e", b=self.config.batch_size
                )
                router_z_loss = router_z_loss_func(router_logits=router_logits)

                logits = logits[:, :-1, :]  # batch seq_len - 1, vocab_size

                # Flatten for cross entropy
                flattened_logits = rearrange(
                    logits, "b s v -> (b s) v"
                )  # bs, vocab_size
                flattened_targets = rearrange(target_tokens, "b s -> (b s)")  # bs

                # Calculate loss and backprop
                loss = F.cross_entropy(flattened_logits, flattened_targets)
                router_z_loss_func
                loss += router_z_loss

                model_engine.backward(loss)

                # Step optimiser
                model_engine.step()

                if sample_batch_num % self.config.eval_steps == 0:
                    # if True:
                    model_engine.eval()
                    test_loss = self.evaluate(test_dataloader)
                    print(
                        f"Epoch: {epoch}, Batch: {sample_batch_num}, Test Loss: {test_loss}"
                    )

        return model_engine

    def save_model_engine(self, model_name: str, step: int = 0) -> None:
        """Save the model to the model_dest."""

        if model_name == "":
            model_name = "moe"

        # Get the current date
        current_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        full_dest = f"models/{model_name}_{current_date}.pt"

        self.model_engine.save_checkpoint(
            save_dir=full_dest, client_state={"checkpoint_step": step}
        )

        print(f"Saved model to {full_dest}")

    def load_model_engine(
        self, model_dest: str, load_optimizer_states: bool = False
    ) -> None:
        """Load a model from the model_dest."""

        _, client_state = self.model_engine.load_checkpoint(
            load_dir=model_dest, load_optimizer_states=load_optimizer_states
        )
        checkpoint_step = client_state["checkpoint_step"]
        print(f"Loaded model from {model_dest}")

    @property
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def main():
    # Set up the trainer
    trainer = Trainer(model=SparseMoETransformer(), optimiser_string="sgd")

    # Train and save the model
    trainer.model = trainer.train()
    trainer.save_model("sophia_100")


if __name__ == "__main__":
    main()
