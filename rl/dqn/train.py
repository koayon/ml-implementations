import random
import time
from typing import Any, Iterable, List, Optional, Tuple, Union

import gym
import numpy as np
import torch as t
from gym.spaces import Box, Discrete
from numpy.random import Generator
from rl_utils import make_env

from helpers import allclose, assert_all_equal
from rl.dqn.args import DQNArgs
from rl.dqn.buffer import ReplayBuffer
from rl.dqn.model import QNetwork


def linear_schedule(
    current_step: int,
    start_e: float,
    end_e: float,
    exploration_fraction: float,
    total_timesteps: int,
) -> float:
    """Return the appropriate epsilon for the current step.
    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).
    """
    proportion_done = current_step / (total_timesteps * exploration_fraction)
    return start_e - proportion_done * (start_e - end_e)


def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv,  # type: ignore
    q_network: QNetwork,
    rng: Generator,
    obs: t.Tensor,
    epsilon: float,
) -> np.ndarray:
    """With probability epsilon, take a random action. Otherwise, take a greedy action according to the q_network.
    Inputs:
        envs : gym.vector.SyncVectorEnv, the family of environments to run against
        q_network : QNetwork, the network used to approximate the Q-value function
        obs : The current observation
        epsilon : exploration percentage
    Outputs:
        actions: (n_environments, ) the sampled action for each environment.
    """
    possible_actions = np.arange(envs.single_action_space.n)
    if rng.random() < epsilon:
        # Explore! Sample a random action.
        actions = rng.choice(possible_actions, size=envs.num_envs)  # [n_environments]
        return actions
    else:
        # Exploit! Use the q-network to predict the q-values for each action,
        # and then take the greedy action.
        with t.no_grad():
            # obs.shape = [n_environments, dim_observation]
            q_values = q_network(obs)  # [n_environments, n_actions]
            actions: np.ndarray = (
                t.argmax(q_values, dim=1).cpu().numpy()
            )  # [n_environments]
            return actions


def setup(
    args: DQNArgs,
) -> Tuple[str, np.random.Generator, t.device, gym.vector.SyncVectorEnv]:  # type: ignore
    """Helper function to set up useful variables for the DQN implementation"""
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    t.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    t.backends.cudnn.deterministic = args.torch_deterministic  # type: ignore

    device = t.device("cuda" if t.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(  # type: ignore
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, Discrete
    ), "only discrete action space is supported"
    return (run_name, rng, device, envs)


def log(
    start_time: float,
    step: int,
    predicted_q_vals: t.Tensor,
    loss: Union[float, t.Tensor],
    infos: Iterable[dict],
    epsilon: float,
):
    """Helper function to write relevant info to logs, and print some things to stdout"""
    if step % 100 == 0:
        print("losses/td_loss", loss, step)
        print("losses/q_values", predicted_q_vals.mean().item(), step)
        print("charts/SPS", int(step / (time.time() - start_time)), step)
        if step % 10000 == 0:
            print("SPS:", int(step / (time.time() - start_time)))
    for info in infos:
        if "episode" in info.keys():
            print(f"global_step={step}, episodic_return={info['episode']['r']}")
            print("charts/episodic_return", info["episode"]["r"], step)
            print("charts/episodic_length", info["episode"]["l"], step)
            print("charts/epsilon", epsilon, step)
            break


def train_dqn(args: DQNArgs):
    (run_name, rng, device, envs) = setup(args)
    print("Set-up complete. Running experiment: ", run_name)

    # Initialise the q-network, optimizer, and replay buffer
    dim_observation = envs.single_observation_space.shape[0]
    num_total_possible_actions = envs.single_action_space.n

    q_network = QNetwork(
        dim_observation=dim_observation,
        num_actions=num_total_possible_actions,
    ).to(device)

    q_target_network = QNetwork(
        dim_observation=dim_observation,
        num_actions=num_total_possible_actions,
    ).to(device)

    optimizer = t.optim.Adam(q_network.parameters(), lr=args.learning_rate)

    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        num_actions=num_total_possible_actions,
        observation_shape=(dim_observation,),
        num_environments=envs.num_envs,
        seed=args.seed,
    )

    start_time = time.time()
    obs = envs.reset()
    for step in range(args.total_timesteps):
        # Sample actions according to the epsilon greedy policy using the linear
        # schedule for epsilon, and then step the environment

        epsilon = linear_schedule(
            current_step=step,
            start_e=args.start_e,
            end_e=args.end_e,
            exploration_fraction=args.exploration_fraction,
            total_timesteps=args.total_timesteps,
        )
        actions = epsilon_greedy_policy(
            envs, q_network=q_network, rng=rng, obs=obs, epsilon=epsilon
        )

        next_obs, rewards, dones, infos = envs.step(actions)

        # Boilerplate to handle the terminal observation case
        real_next_obs = next_obs.copy()
        for i, done in enumerate(dones):
            if done:
                real_next_obs[i] = infos[i]["terminal_observation"]

        # Add state to the replay buffer
        rb.add(obs, actions, rewards, dones, next_obs)

        obs = next_obs

        if step > args.learning_starts and step % args.train_frequency == 0:
            # Sample from the replay buffer, compute the TD target, compute TD
            # loss, and perform an optimizer step.
            replay_buffer_samples = rb.sample(
                sample_size=args.batch_size, device=device
            )
            sampled_observations = (
                replay_buffer_samples.observations
            )  # [num_environments, num_samples, observations_shape]

            predicted_q_values_for_actions = q_network(
                sampled_observations
            )  # [num_environments, num_samples, num_actions]

            # Get targets and loss function
            next_observations = replay_buffer_samples.next_observations
            rewards = replay_buffer_samples.rewards

            with t.no_grad():
                predicted_q_values_after_next_move = q_target_network(
                    next_observations
                )  # [num_environments, num_samples, num_actions]
                max_q_val_after_next_move = t.max(
                    predicted_q_values_after_next_move, dim=-1
                )  # [num_environments, num_samples]

            # Target is r + gamma * max_a' Q(s', a') for non-terminal
            # transitions. If done at that step then there are no future
            # rewards.
            target_q_values = rewards + (
                args.gamma * t.max(max_q_val_after_next_move)
            ) * (1 - replay_buffer_samples.dones)

            # Compute TD-loss normalised by the batch size
            loss: t.Tensor = t.norm(
                target_q_values - predicted_q_values_for_actions
            ) / (t.prod(t.tensor(target_q_values.shape)))
            loss.backward()
            optimizer.step()

            log(start_time, step, predicted_q_values_for_actions, loss, infos, epsilon)

        # Update the target network
        if step % args.target_network_frequency == 0:
            q_target_network.load_state_dict(q_network.state_dict())

    "If running one of the Probe environments, will test if the learned q-values are sensible after training. Useful for debugging."
    if args.env_id == "Probe1-v0":
        batch = t.tensor([[0.0]]).to(device)
        value = q_network(batch)
        print("Value: ", value)
        expected = t.tensor([[1.0]]).to(device)
        allclose(value, expected, 0.0001)
    elif args.env_id == "Probe2-v0":
        batch = t.tensor([[-1.0], [+1.0]]).to(device)
        value = q_network(batch)
        print("Value:", value)
        expected = batch
        allclose(value, expected, 0.0001)
    elif args.env_id == "Probe3-v0":
        batch = t.tensor([[0.0], [1.0]]).to(device)
        value = q_network(batch)
        print("Value: ", value)
        expected = t.tensor([[args.gamma], [1.0]])
        allclose(value, expected, 0.0001)
    elif args.env_id == "Probe4-v0":
        batch = t.tensor([[0.0]]).to(device)
        value = q_network(batch)
        expected = t.tensor([[-1.0, 1.0]]).to(device)
        print("Value: ", value)
        allclose(value, expected, 0.0001)
    elif args.env_id == "Probe5-v0":
        batch = t.tensor([[0.0], [1.0]]).to(device)
        value = q_network(batch)
        expected = t.tensor([[1.0, -1.0], [-1.0, 1.0]]).to(device)
        print("Value: ", value)
        allclose(value, expected, 0.0001)
    envs.close()


if __name__ == "__main__":
    args = DQNArgs()
    args.env_id = "Probe1-v0"

    train_dqn(args)
