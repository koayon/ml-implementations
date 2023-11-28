import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Any, Iterable, List, Optional, Tuple, Union

import gym
import gym.envs.registration
import numpy as np
import pandas as pd
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from gym.spaces import Box, Discrete
from matplotlib import pyplot as plt
from numpy.random import Generator
from rl_utils import make_env
from torch.utils.tensorboard import SummaryWriter

os.environ["SDL_VIDEODRIVER"] = "dummy"


class QNetwork(nn.Module):
    def __init__(
        self,
        dim_observation: int,
        num_actions: int,
        hidden_sizes: list[int] = [120, 84],
    ):
        super().__init__()
        l1 = nn.Linear(dim_observation, hidden_sizes[0])
        relu = nn.ReLU()
        l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        l3 = nn.Linear(hidden_sizes[1], num_actions)

        self.seq = nn.Sequential(l1, relu, l2, relu, l3)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.seq(x)


if __name__ == "__main__":
    net = QNetwork(dim_observation=4, num_actions=2)
    n_params = sum((p.nelement() for p in net.parameters()))
    print(net)
    print(f"Total number of parameters: {n_params}")
    assert n_params == 10934
