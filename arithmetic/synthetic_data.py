import math
import random

import torch as t
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers.data.data_collator import default_data_collator

from arithmetic.config import ArithmeticConfig
from general.character_level_tokenizer import CharTokenizer

tokenizer = CharTokenizer()
config = ArithmeticConfig()


def num_digits_proxy_hardness(num: int) -> int:
    return math.ceil(math.log(num, 10)) if num > 0 else 1


def make_input_output_pairs(n: int = 10) -> tuple[list[str], list[str], list[int]]:
    """Generate n arithmetic input-output pairs"""
    input_strs = []
    output_strs = []
    num_idk_tokens = []
    for _ in range(n):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        c = a + b

        # Calcuate a proxy for hardness of the problem. Here we're using the number of digits in the sum
        proxy_hardness = num_digits_proxy_hardness(a) + num_digits_proxy_hardness(b)

        input_strs.append(f"{a}+{b}=")
        output_strs.append(str(c))
        num_idk_tokens.append(proxy_hardness)
    return input_strs, output_strs, num_idk_tokens


class CustomDataset(Dataset):
    def __init__(self, input_tensors, target_tensors):
        self.input_tensors = input_tensors
        self.target_tensors = target_tensors

        assert len(self.input_tensors) == len(self.target_tensors)

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return {
            "encoder_input_ids": self.input_tensors[idx],
            "label_ids": self.target_tensors[idx],
            "labels": self.target_tensors[idx],
            # "target_ids": self.target_tensors[idx][1:],
            # "decoder_input_ids": self.target_tensors[idx][:-1],
        }


def get_dataset() -> Dataset:
    input_strs, output_strs, num_idk_tokens = make_input_output_pairs(10)
    # print(input_strs)
    # print(output_strs)

    # Tokenize and pad the strings, then convert to PyTorch tensors
    input_tensors = [
        t.tensor(tokenizer.encode(text), dtype=t.long) for text in input_strs
    ]
    target_tensors = [
        t.tensor(
            [tokenizer.sos_token_id]
            + idk_tokens * [tokenizer.idk_token_id]
            + tokenizer.encode(text),
            dtype=t.long,
        )
        for text, idk_tokens in zip(output_strs, num_idk_tokens)
    ]

    input_tensors = pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_tensors = pad_sequence(target_tensors, batch_first=True, padding_value=0)

    # Turn into a dataset
    dataset = CustomDataset(input_tensors, target_tensors)

    # Turn into a dataloader
    # dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    return dataset


def main():
    dataset = get_dataset()

    for row in dataset:
        print(row)
        break

    out = default_data_collator(dataset)  # type: ignore
    print(out)


if __name__ == "__main__":
    main()
