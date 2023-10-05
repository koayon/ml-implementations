import random

import torch as t
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset

from arithmetic.config import ArithmeticConfig
from general.character_level_tokenizer import CharTokenizer

tokenizer = CharTokenizer()
config = ArithmeticConfig()


def make_input_output_pairs(n: int = 10) -> tuple[list[str], list[str]]:
    """Generate n arithmetic input-output pairs"""
    input_strs = []
    output_strs = []
    for _ in range(n):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        c = a + b
        input_strs.append(f"{a}+{b}=")
        output_strs.append(str(c))
    return input_strs, output_strs


def get_dataset() -> Dataset:
    input_strs, output_strs = make_input_output_pairs(10)
    # print(input_strs)
    # print(output_strs)

    # Tokenize and pad the strings, then convert to PyTorch tensors
    input_tensors = [
        t.tensor(tokenizer.encode(text), dtype=t.long) for text in input_strs
    ]
    target_tensors = [
        t.tensor(tokenizer.encode(text), dtype=t.long) for text in output_strs
    ]

    input_tensors = pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_tensors = pad_sequence(target_tensors, batch_first=True, padding_value=0)

    # Turn into a dataset
    dataset = TensorDataset(input_tensors, target_tensors)

    # Turn into a dataloader
    # dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    return dataset


def main():
    dataloader = get_dataset()

    for batch in dataloader:
        print(batch)
        break


if __name__ == "__main__":
    main()
