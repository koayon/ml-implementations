"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
Inspired by Tiny Stories and llama2.c
"""

import glob
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
import requests
import tiktoken
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

tokenizer = tiktoken.encoding_for_model("gpt2")

DATA_CACHE_DIR = "data"


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads the dataset to disk."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")


def pretokenize():
    def process_shard(shard):
        with open(shard, "r") as f:
            data = json.load(f)
        all_tokens = []
        for example in tqdm(data):
            text = example["story"]
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = tokenizer.encode(text)  # encode the text
            all_tokens.extend(tokens)
        # convert to uint16 nparray
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        # write to disk
        tokenized_filename = shard.replace(".json", ".bin")
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())
        print(f"Saved {tokenized_filename}")

    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    # process all the shards in a threadpool
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_shard, shard_filenames)

    print("Done.")


class TinyStoriesDataset(Dataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split: str = "train", max_seq_len: int = 1024):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len

    def __len__(self):
        # This is the number of blocks of size `block_size` in `data`
        return 100000 // self.max_seq_len

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0

        # combine the worker_id and worker_rank to create a unique seed for rng
        data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        shard_filenames = (
            shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        )
        while True:
            random.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                print(f"Shard {shard} has {num_batches} batches.")
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                random.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    # y = chunk[1:]
                    yield x


if __name__ == "__main__":
    # download()
    # pretokenize()
    train_dataset = TinyStoriesDataset(split="train", max_seq_len=1024)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset, replacement=True),
        batch_size=12,
        shuffle=False,
        num_workers=0,
    )
    data_iter = iter(train_dataloader)
    X, y = next(data_iter)  # lis
    print(X.shape)
    print(y.shape)
