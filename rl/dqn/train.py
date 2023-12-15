import random
import time
from typing import Iterable, Tuple, Union

import gym
import gym.vector
import numpy as np
import torch as t
import torch.backends.cudnn
from gym.spaces import Box, Discrete
from numpy.random import Generator

import wandb
from helpers import allclose, allclose_atol
from rl.dqn.args import DQNArgs
from rl.dqn.buffer import ReplayBuffer
from rl.dqn.model import QNetwork
from rl.dqn.probes import PROBE_ENV_CONFIGS, register_probe_environments
from rl.rl_utils import make_env


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
    return max(start_e - proportion_done * (start_e - end_e), end_e)


def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv,
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
) -> Tuple[str, np.random.Generator, t.device, gym.vector.SyncVectorEnv]:
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

    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = t.device("cuda" if t.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, Discrete
    ), "only discrete action space is supported"
    return (run_name, rng, device, envs)


def log(
    step: int,
    predicted_q_vals: t.Tensor,
    loss: t.Tensor,
    infos: Iterable[dict],
    epsilon: float,
    print_to_console: bool = True,
):
    """Helper function to write relevant info to wandb, and print some things to stdout"""
    if step % 10 == 0:
        # Prepare data for logging
        log_data = {
            "losses/td_loss": loss.item() if isinstance(loss, t.Tensor) else loss,
            "losses/q_values": t.mean(predicted_q_vals).item(),
        }

        # Log data to wandb
        wandb.log(log_data, step=step)

    if (step % 1000 == 0) and print_to_console:
        print(f"Step: {step} | loss: {loss.item():2f}")

    for info in infos:
        if "episode" in info.keys():
            episode_reward = info["episode"]["r"]
            episode_length = info["episode"]["l"]

            episodic_data = {
                "charts/episodic_return": episode_reward,
                "charts/episodic_length": episode_length,
                "charts/epsilon": epsilon,
            }

            # Log episodic data to wandb
            wandb.log(episodic_data, step=step)

            if (step % 10 == 0) and print_to_console:
                print(
                    f"Step: {step} | epsilon: {epsilon:2f} | episode reward: {episode_reward}",
                )
            break


def train_dqn(args: DQNArgs, log_to_wandb: bool = True):
    (run_name, rng, device, envs) = setup(args)
    print("Set-up complete. Running experiment: ", run_name)

    # Initialise the q-network, optimizer, and replay buffer
    dim_observation = np.array(envs.single_observation_space.shape).prod()
    num_total_possible_actions = envs.single_action_space.n

    q_network = QNetwork(
        dim_observation=dim_observation,
        hidden_sizes=args.hidden_sizes,
        num_actions=num_total_possible_actions,
        dropout=args.dropout,
    ).to(device)

    # Initialise the target network to be the same as the q-network
    q_target_network = QNetwork(
        dim_observation=dim_observation,
        hidden_sizes=args.hidden_sizes,
        num_actions=num_total_possible_actions,
    ).to(device)
    q_target_network.load_state_dict(q_network.state_dict())

    optimizer = t.optim.Adam(
        q_network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        num_actions=num_total_possible_actions,
        observation_shape=(dim_observation,),
        num_environments=envs.num_envs,
        seed=args.seed,
    )

    start_time = time.time()
    obs: np.ndarray = envs.reset()

    # print(obs)

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
        # if step % 1000:
        #     print(f"Step: {step}, epsilon: {epsilon}")
        # if epsilon < 0:
        #     raise ValueError("Epsilon < 0")
        actions = epsilon_greedy_policy(
            envs, q_network=q_network, rng=rng, obs=t.tensor(obs), epsilon=epsilon
        )

        next_obs, rewards, dones, infos = envs.step(actions)

        # print(next_obs)
        # if step > 50:
        #     break

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
            )  # [num_samples, observations_shape]
            sampled_actions = (replay_buffer_samples.actions).to(
                t.long
            )  # [num_samples]

            predicted_q_values_for_all_actions = q_network(
                sampled_observations
            )  # [num_samples, num_actions]

            predicted_q_values_for_actions_taken = t.gather(
                predicted_q_values_for_all_actions, 1, sampled_actions.unsqueeze(-1)
            ).squeeze(1)

            # Get targets and loss function
            next_observations = replay_buffer_samples.next_observations
            rewards = replay_buffer_samples.rewards

            with t.no_grad():
                predicted_q_values_after_next_move = q_target_network(
                    next_observations
                )  # [num_samples, num_actions]
                max_q_val_after_next_move, _ = t.max(
                    predicted_q_values_after_next_move, dim=-1
                )  # [num_samples]

            # Target is r + gamma * max_a' Q(s', a') for non-terminal
            # transitions. If done at that step then there are no future
            # rewards.
            target_q_values = rewards.flatten() + (
                args.gamma * max_q_val_after_next_move
            ) * (
                1 - replay_buffer_samples.dones.flatten()
            )  # [num_samples]
            if args.target_noise:
                target_q_values += args.target_noise * np.random.randn()

            # Compute TD-loss normalised by the batch size
            loss: t.Tensor = t.norm(
                target_q_values - predicted_q_values_for_actions_taken.squeeze(-1)
            ) / (t.prod(t.tensor(target_q_values.shape)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if log_to_wandb:
                log(
                    step=step,
                    predicted_q_vals=predicted_q_values_for_all_actions,
                    loss=loss,
                    infos=infos,
                    epsilon=epsilon,
                )

        # Update the target network
        if step % args.target_network_frequency == 0:
            q_target_network.load_state_dict(q_network.state_dict())

    # If running one of the Probe environments, will test if the learned
    # q-values are sensible after training. Useful for debugging.
    if args.env_id in PROBE_ENV_CONFIGS:
        probe_env_config = PROBE_ENV_CONFIGS[args.env_id]

        batch = probe_env_config.batch.to(device)
        value = q_network(batch)
        print("Value: ", value)

        # Test if the q-values are close to the expected values
        allclose_atol(value, probe_env_config.expected.to(device), 0.005)

    return q_network


if __name__ == "__main__":
    args = DQNArgs()

    register_probe_environments()

    # args.env_id = "Probe5-v0"
    # args.hidden_sizes = [20, 10]
    # args.weight_decay = 0
    log_to_wandb = True
    # args.total_timesteps = 5000
    if log_to_wandb:
        wandb.init()
    try:
        q_network = train_dqn(args, log_to_wandb=log_to_wandb)
    except AssertionError as e:
        print("Not close enough to expected values:")
        print(e)
    if log_to_wandb:
        wandb.finish()
