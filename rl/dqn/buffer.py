import random
from dataclasses import dataclass

import gym
import gym.envs.registration
import numpy as np
import pandas as pd
import plotly.express as px
import torch as t

from rl.rl_utils import make_env


@dataclass
class ReplayBufferSamples:
    """
    Samples from the replay buffer, converted to PyTorch for use in neural network training.
    obs: shape (sample_size, *observation_shape), dtype t.float
    actions: shape (sample_size, ) dtype t.int
    rewards: shape (sample_size, ), dtype t.float
    dones: shape (sample_size, ), dtype t.bool
    next_observations: shape (sample_size, *observation_shape), dtype t.float
    """

    observations: t.Tensor
    actions: t.Tensor
    rewards: t.Tensor
    dones: t.Tensor
    next_observations: t.Tensor

    def to_device(self, device: t.device) -> None:
        self.observations = self.observations.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.next_observations = self.next_observations.to(device)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return ReplayBufferSamples(
            observations=self.observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            dones=self.dones[idx],
            next_observations=self.next_observations[idx],
        )
