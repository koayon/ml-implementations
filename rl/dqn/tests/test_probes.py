import gym

from rl.dqn.probes import Probe1, register_probe_environments

register_probe_environments()


def test_probe1():
    env = gym.make("Probe1-v0")
    assert env.observation_space.shape == (1,)
    assert env.action_space.shape == ()
