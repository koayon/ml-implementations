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

    def __getitem__(self, idx) -> "ReplayBufferSamples":
        return ReplayBufferSamples(
            observations=self.observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            dones=self.dones[idx],
            next_observations=self.next_observations[idx],
        )


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        num_actions: int,
        observation_shape: tuple,
        num_environments: int,
        seed: int,
    ):
        assert (
            num_environments == 1
        ), "This buffer only supports SyncVectorEnv with 1 environment inside."
        self.buffer_size = buffer_size
        self.num_actions = num_actions
        self.observation_shape = observation_shape

        random.seed(seed)

        self.buffer = ReplayBufferSamples(
            t.zeros((0, *observation_shape)),
            t.zeros((0,)),
            t.zeros((0,)),
            t.zeros((0,)),
            t.zeros((0, *observation_shape)),
        )

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        """
        obs: shape (num_environments, *observation_shape) Observation before the action
        actions: shape (num_environments, ) the action chosen by the agent
        rewards: shape (num_environments, ) the reward after the after
        dones: shape (num_environments, ) if True, the episode ended and was reset automatically
        next_obs: shape (num_environments, *observation_shape) Observation after the action. If done is True, this should be the terminal observation, NOT the first observation of the next episode.
        """
        prev_len = len(self.buffer)

        self.buffer.observations = t.cat(
            [self.buffer.observations, t.tensor(obs, dtype=t.float)]
        )
        self.buffer.actions = t.cat(
            [self.buffer.actions, t.tensor(actions, dtype=t.int)]
        )
        self.buffer.rewards = t.cat(
            [self.buffer.rewards, t.tensor(rewards, dtype=t.float)]
        )
        self.buffer.dones = t.cat([self.buffer.dones, t.tensor(dones, dtype=t.bool)])
        self.buffer.next_observations = t.cat(
            [self.buffer.next_observations, t.tensor(next_obs, dtype=t.float)]
        )

        assert len(self.buffer) == prev_len + len(obs)

        if len(self.buffer) >= self.buffer_size:
            self.buffer.observations = self.buffer.observations[-self.buffer_size :]
            self.buffer.actions = self.buffer.actions[-self.buffer_size :]
            self.buffer.rewards = self.buffer.rewards[-self.buffer_size :]
            self.buffer.dones = self.buffer.dones[-self.buffer_size :]
            self.buffer.next_observations = self.buffer.next_observations[
                -self.buffer_size :
            ]

    def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
        """Uniformly sample sample_size entries from the buffer and convert them to PyTorch tensors on device.
        Sampling is with replacement, and sample_size may be larger than the buffer size.
        """
        indices = np.random.randint(len(self.buffer), size=sample_size)
        samples = self.buffer[indices]
        samples.to_device(device)
        return samples


if __name__ == "__main__":
    rb = ReplayBuffer(
        buffer_size=256,
        num_actions=2,
        observation_shape=(4,),
        num_environments=1,
        seed=0,
    )

    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 0, 0, False, "test")])  # type: ignore
    obs = envs.reset()

    for i in range(512):
        actions = np.array([0])
        (next_obs, rewards, dones, infos) = envs.step(actions)
        real_next_obs = next_obs.copy()
        for i, done in enumerate(dones):
            if done:
                real_next_obs[i] = infos[i]["terminal_observation"]
        rb.add(obs, actions, rewards, dones, next_obs)
        obs = next_obs

    sample = rb.sample(128, t.device("cpu"))

    columns = ["cart_pos", "cart_v", "pole_angle", "pole_v"]

    df = pd.DataFrame(rb.buffer.observations.cpu().numpy(), columns=columns)
    print(df.head())
    fig = px.line(df, x=df.index, y="cart_pos", title="Replay Buffer")
    fig.show()

    df2 = pd.DataFrame(sample.observations.cpu().numpy(), columns=columns)
    print(df2.head())
    fig = px.line(df2, x=df2.index, y="cart_pos", title="Sampled Replay Buffer")
    fig.show()
