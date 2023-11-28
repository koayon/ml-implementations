from typing import Optional, Union

import gym
import gym.envs.registration
import numpy as np
from gym.spaces import Box, Discrete

ObsType = np.ndarray
ActType = int


def register_probe_environments():
    gym.envs.registration.register(id="Probe1-v0", entry_point=Probe1)


class Probe1(gym.Env):
    """One action, observation of [0.0], one timestep long, +1 reward.

    We expect the agent to rapidly learn that the value of the constant [0.0] observation is +1.0. Note we're using a continuous observation space for consistency with CartPole.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([0]), np.array([0]))
        self.action_space = Discrete(1)
        self.seed()
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        return (np.array([0]), 1.0, True, {})

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        if return_info:
            return (np.array([0.0]), {})
        return np.array([0.0])


# class Probe2(gym.Env):
#     """One action, observation of [-1.0] or [+1.0], one timestep long, reward equals observation.

#     We expect the agent to rapidly learn the value of each observation is equal to the observation.
#     """

#     action_space: Discrete
#     observation_space: Box

#     def __init__(self):
#         pass

#     def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
#         pass

#     def reset(
#         self, seed: Optional[int] = None, return_info=False, options=None
#     ) -> Union[ObsType, tuple[ObsType, dict]]:
#         pass


# gym.envs.registration.register(id="Probe2-v0", entry_point=Probe2)


# class Probe3(gym.Env):
#     """One action, [0.0] then [1.0] observation, two timesteps, +1 reward at the end.

#     We expect the agent to rapidly learn the discounted value of the initial observation.
#     """

#     action_space: Discrete
#     observation_space: Box

#     def __init__(self):
#         pass

#     def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
#         pass

#     def reset(
#         self, seed: Optional[int] = None, return_info=False, options=None
#     ) -> Union[ObsType, tuple[ObsType, dict]]:
#         pass


# gym.envs.registration.register(id="Probe3-v0", entry_point=Probe3)


# class Probe4(gym.Env):
#     """Two actions, [0.0] observation, one timestep, reward is -1.0 or +1.0 dependent on the action.

#     We expect the agent to learn to choose the +1.0 action.
#     """

#     action_space: Discrete
#     observation_space: Box

#     def __init__(self):
#         pass

#     def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
#         pass

#     def reset(
#         self, seed: Optional[int] = None, return_info=False, options=None
#     ) -> Union[ObsType, tuple[ObsType, dict]]:
#         pass


# gym.envs.registration.register(id="Probe4-v0", entry_point=Probe4)


# class Probe5(gym.Env):
#     """Two actions, random 0/1 observation, one timestep, reward is 1 if action equals observation otherwise -1.

#     We expect the agent to learn to match its action to the observation.
#     """

#     action_space: Discrete
#     observation_space: Box

#     def __init__(self):
#         pass

#     def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
#         pass

#     def reset(
#         self, seed: Optional[int] = None, return_info=False, options=None
#     ) -> Union[ObsType, tuple[ObsType, dict]]:
#         pass


# gym.envs.registration.register(id="Probe5-v0", entry_point=Probe5)
