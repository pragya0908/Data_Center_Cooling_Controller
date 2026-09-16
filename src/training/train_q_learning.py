"""Training pipeline for tabular Q-learning on the CoolRL DataCenterEnv.

This module implements the complete reproducible training workflow:
- Resets environment and discretizes initial observations.
- Interacts with environment via epsilon-greedy action selection.
- Performs Bellman TD temporal difference updates on experience transitions.
- Records comprehensive per-episode thermal, energy, and learning convergence metrics.
- Decays exploration epsilon per episode.
- Persists trained Q-table and complete training history.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.agents.q_learning import QLearningAgent
from src.agents.state_discretizer import StateDiscretizer
from src.environment.datacenter_env import DataCenterEnv


def train_q_learning(
    env: DataCenterEnv | None = None,
    discretizer: StateDiscretizer | None = None,
    agent: QLearningAgent | None = None,
    scenario: str = "normal",
    split: str = "train",
    num_episodes: int = 1000,
    alpha: float = 0.10,
    gamma: float = 0.95,
    epsilon: float = 1.00,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int = 42,
    rolling_window: int = 50,
    save_dir: str | Path | None = None,
    verbose: bool = True,
    log_interval: int = 100,
) -> tuple[QLearningAgent, pd.DataFrame]:
    """Train a tabular Q-learning agent on DataCenterEnv.

    Args:
        env: Optional pre-constructed DataCenterEnv. If None, created using scenario & split.
        discretizer: Optional StateDiscretizer. If None, initialized with defaults.
        agent: Optional pre-constructed QLearningAgent. If None, initialized with hyperparameters.
        scenario: Scenario name ('normal', 'high_workload', etc.).
        split: Dataset split ('train', 'val', 'all').
        num_episodes: Total training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Initial exploration rate.
        epsilon_min: Minimum exploration floor.
        epsilon_decay: Multiplicative decay per episode.
        seed: Random seed for reproducibility.
        rolling_window: Window size for rolling mean cumulative reward.
        save_dir: Destination folder for trained model (.npz) and history (.csv).
        verbose: Print training progress updates.
        log_interval: Number of episodes between progress prints.

    Returns:
        tuple of (trained_agent, history_dataframe)
    """
    # 1. Initialize components
    if env is None:
        env = DataCenterEnv(scenario=scenario, split=split)
    if discretizer is None:
        discretizer = StateDiscretizer()
    if agent is None:
        agent = QLearningAgent(
            n_states=discretizer.NUM_STATES,
            n_actions=discretizer.NUM_ACTIONS,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed,
        )

    records: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    if verbose:
        print(f"Starting Q-learning training on {scenario} ({split} split):")
        print(f"  Episodes: {num_episodes}, Alpha: {agent.alpha}, Gamma: {agent.gamma}")
        print(f"  Epsilon: {agent.epsilon} -> {agent.epsilon_min} (decay: {agent.epsilon_decay})")
        print(f"  Trajectory length: {env.trajectory_length} steps per episode")

    for episode in range(1, num_episodes + 1):
        # Reset environment with reproducible seed per episode for consistent initial states
        obs, info = env.reset(seed=seed + episode)
        state_id = discretizer.encode(obs)

        done = False
        ep_reward = 0.0
        ep_energy = 0.0
        step_count = 0
        temps: list[float] = []
        steps_gt_27 = 0
        steps_gt_32 = 0
        steps_lt_18 = 0

        # Step through episode until exhaustion / termination
        while not done:
            action = agent.choose_action(state_id, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state_id = discretizer.encode(next_obs)
            agent.update(state_id, action, reward, next_state_id, terminated=done)

            t_int = float(info["internal_temperature"])
            temps.append(t_int)
            if t_int > 27.0:
                steps_gt_27 += 1
            if t_int > 32.0:
                steps_gt_32 += 1
            if t_int < 18.0:
                steps_lt_18 += 1

            ep_reward += reward
            ep_energy += float(info["cooling_energy"])
            step_count += 1

            state_id = next_state_id

        # Record pre-decay epsilon for this episode
        ep_epsilon = agent.epsilon

        # Episode-level exploration decay
        agent.decay_epsilon()

        # Compile episode metrics
        mean_step_r = ep_reward / step_count if step_count > 0 else 0.0
        avg_temp = float(np.mean(temps)) if temps else 0.0
        max_temp = float(np.max(temps)) if temps else 0.0

        rec = {
            "episode": episode,
            "cumulative_reward": float(ep_reward),
            "mean_step_reward": float(mean_step_r),
            "epsilon": float(ep_epsilon),
            "steps": step_count,
            "total_cooling_energy": float(ep_energy),
            "average_internal_temperature": float(avg_temp),
            "maximum_internal_temperature": float(max_temp),
            "overheating_steps_gt_27": steps_gt_27,
            "severe_overheating_steps_gt_32": steps_gt_32,
            "overcooling_steps_lt_18": steps_lt_18,
        }
        records.append(rec)

        if verbose and (episode == 1 or episode % log_interval == 0 or episode == num_episodes):
            recent_rewards = [r["cumulative_reward"] for r in records[-min(rolling_window, len(records)):]]
            rolling_avg = float(np.mean(recent_rewards))
            print(
                f"Episode {episode:4d}/{num_episodes} | "
                f"Reward: {ep_reward:9.2f} | "
                f"Rolling({len(recent_rewards)}): {rolling_avg:9.2f} | "
                f"Eps: {ep_epsilon:.4f} | "
                f"Avg Temp: {avg_temp:5.2f}°C | "
                f">27°C: {steps_gt_27:3d} | "
                f">32°C: {steps_gt_32:3d} | "
                f"Energy: {ep_energy:6.2f}"
            )

    t_total = time.perf_counter() - t_start
    if verbose:
        print(f"Training completed in {t_total:.2f} seconds ({t_total/num_episodes:.4f} s/ep).")

    # Construct history DataFrame and compute rolling mean reward
    df_history = pd.DataFrame(records)
    df_history["rolling_mean_reward"] = (
        df_history["cumulative_reward"].rolling(window=rolling_window, min_periods=1).mean()
    )

    # 2. Persist artifacts if save_dir provided
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = out_dir / f"q_learning_{scenario}_{split}.npz"
        history_path = out_dir / f"q_learning_{scenario}_{split}_history.csv"

        agent.save(model_path)
        df_history.to_csv(history_path, index=False)
        if verbose:
            print(f"Saved trained agent model to: {model_path}")
            print(f"Saved training history to: {history_path}")

    return agent, df_history
