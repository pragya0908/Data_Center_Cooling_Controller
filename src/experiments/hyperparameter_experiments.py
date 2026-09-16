"""Hyperparameter sensitivity experiments for Q-learning in CoolRL.

This module evaluates the sensitivity of tabular Q-learning convergence and performance
to variations in key hyperparameters:
- Learning rate alpha: [0.05, 0.10, 0.20]
- Discount factor gamma: [0.90, 0.95, 0.99]

All experiments are conducted strictly on:
- Scenario: normal
- Split: train (1440 steps per episode)
- Fixed seed: 42
- Fixed episode count: 500 episodes
- Fixed exploration parameters: epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995

Evaluation of each trained model is conducted on the validation split ('val')
without updating the Q-table, preserving test split isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.agents.q_learning import QLearningAgent
from src.agents.state_discretizer import StateDiscretizer
from src.environment.datacenter_env import DataCenterEnv
from src.training.train_q_learning import train_q_learning

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def evaluate_agent_greedy(
    agent: QLearningAgent,
    scenario: str = "normal",
    split: str = "val",
    seed: int = 42,
) -> dict[str, float]:
    """Perform a pure greedy evaluation rollout without updating Q-values.

    Args:
        agent: Trained QLearningAgent.
        scenario: Scenario name.
        split: Dataset split ('val' for sensitivity/tuning isolation).
        seed: Environment reset seed.

    Returns:
        Dictionary of thermal, energy, and reward evaluation metrics.
    """
    env = DataCenterEnv(scenario=scenario, split=split)
    discretizer = StateDiscretizer()
    obs, info = env.reset(seed=seed)

    temps: list[float] = []
    energies: list[float] = []
    rewards: list[float] = []
    coolings: list[float] = []
    actions: list[int] = []

    done = False
    while not done:
        state_id = discretizer.encode(obs)
        action = agent.choose_action(state_id, training=False)
        actions.append(action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        temps.append(float(info["internal_temperature"]))
        energies.append(float(info["cooling_energy"]))
        rewards.append(float(reward))
        coolings.append(float(info["cooling_level"]))
        obs = next_obs

    t_arr = np.array(temps)
    r_arr = np.array(rewards)
    n_steps = len(t_arr)

    steps_gt_27 = int(np.sum(t_arr > 27.0))
    steps_gt_32 = int(np.sum(t_arr > 32.0))
    steps_lt_18 = int(np.sum(t_arr < 18.0))
    steps_rec = int(np.sum((t_arr >= 18.0) & (t_arr <= 27.0)))

    return {
        "eval_total_steps": float(n_steps),
        "eval_cumulative_reward": float(np.sum(r_arr)),
        "eval_mean_step_reward": float(np.mean(r_arr)),
        "eval_total_cooling_energy": float(np.sum(energies)),
        "eval_mean_cooling_level": float(np.mean(coolings)),
        "eval_mean_internal_temp": float(np.mean(t_arr)),
        "eval_max_internal_temp": float(np.max(t_arr)),
        "eval_temp_std": float(np.std(t_arr)),
        "eval_recommended_zone_pct": float((steps_rec / n_steps) * 100.0) if n_steps > 0 else 0.0,
        "eval_steps_gt_27": float(steps_gt_27),
        "eval_pct_gt_27": float((steps_gt_27 / n_steps) * 100.0) if n_steps > 0 else 0.0,
        "eval_steps_gt_32": float(steps_gt_32),
        "eval_pct_gt_32": float((steps_gt_32 / n_steps) * 100.0) if n_steps > 0 else 0.0,
        "eval_steps_lt_18": float(steps_lt_18),
        "eval_pct_lt_18": float((steps_lt_18 / n_steps) * 100.0) if n_steps > 0 else 0.0,
    }


def run_hyperparameter_experiments(
    output_csv: Path | str = PROJECT_ROOT / "results" / "experiments" / "hyperparameter_sensitivity.csv",
    model_dir: Path | str = PROJECT_ROOT / "results" / "experiments" / "models" / "hyperparameters",
    num_episodes: int = 500,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Execute hyperparameter sensitivity matrix across alpha and gamma.

    Matrix:
    - alpha: 0.05, 0.10, 0.20 (at gamma = 0.95)
    - gamma: 0.90, 0.95, 0.99 (at alpha = 0.10)
    Note: (alpha=0.10, gamma=0.95) is the shared baseline configuration.
    Total distinct configurations: 5.

    Returns:
        DataFrame containing training and evaluation metrics for each configuration.
    """
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    m_dir = Path(model_dir)
    m_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        # Varying alpha (gamma fixed at 0.95)
        {"config_id": "alpha_0.05_gamma_0.95", "alpha": 0.05, "gamma": 0.95},
        {"config_id": "alpha_0.10_gamma_0.95", "alpha": 0.10, "gamma": 0.95},
        {"config_id": "alpha_0.20_gamma_0.95", "alpha": 0.20, "gamma": 0.95},
        # Varying gamma (alpha fixed at 0.10)
        {"config_id": "alpha_0.10_gamma_0.90", "alpha": 0.10, "gamma": 0.90},
        {"config_id": "alpha_0.10_gamma_0.99", "alpha": 0.10, "gamma": 0.99},
    ]

    results: list[dict[str, Any]] = []

    for cfg in configs:
        cid = cfg["config_id"]
        a_val = cfg["alpha"]
        g_val = cfg["gamma"]

        if verbose:
            print(f"[HYPERPARAM] Running {cid}: alpha={a_val}, gamma={g_val}, episodes={num_episodes}")

        # Train on train split
        save_path = m_dir / f"q_learning_{cid}.npz"
        trained_agent, history_df = train_q_learning(
            scenario="normal",
            split="train",
            num_episodes=num_episodes,
            alpha=a_val,
            gamma=g_val,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.995,
            seed=seed,
            save_dir=None,  # We save manually below with specific naming
            verbose=False,
        )
        trained_agent.save(save_path)

        # Extract training convergence metrics
        final_row = history_df.iloc[-1]
        final_reward = float(final_row["cumulative_reward"])
        rolling_50_reward = float(final_row["rolling_mean_reward"])
        final_eps = float(final_row["epsilon"])
        final_train_energy = float(final_row["total_cooling_energy"])
        final_train_avg_temp = float(final_row["average_internal_temperature"])
        final_train_max_temp = float(final_row["maximum_internal_temperature"])

        # Q-table statistics
        q_table = trained_agent.q_table
        q_mean = float(np.mean(q_table))
        q_std = float(np.std(q_table))
        q_min = float(np.min(q_table))
        q_max = float(np.max(q_table))
        non_zero_states = int(np.sum(np.any(q_table != 0.0, axis=1)))
        non_zero_pct = float((non_zero_states / q_table.shape[0]) * 100.0)

        # Evaluate on validation split (greedy, read-only)
        eval_metrics = evaluate_agent_greedy(
            agent=trained_agent,
            scenario="normal",
            split="val",
            seed=seed,
        )

        record: dict[str, Any] = {
            "config_id": cid,
            "alpha": a_val,
            "gamma": g_val,
            "episodes": num_episodes,
            "seed": seed,
            "scenario": "normal",
            "train_split": "train",
            "eval_split": "val",
            "final_epsilon": final_eps,
            "final_train_reward": final_reward,
            "rolling_50_train_reward": rolling_50_reward,
            "final_train_cooling_energy": final_train_energy,
            "final_train_avg_temp": final_train_avg_temp,
            "final_train_max_temp": final_train_max_temp,
            "q_table_mean": q_mean,
            "q_table_std": q_std,
            "q_table_min": q_min,
            "q_table_max": q_max,
            "visited_states_count": non_zero_states,
            "visited_states_pct": non_zero_pct,
            "model_path": str(save_path.relative_to(PROJECT_ROOT)),
        }
        record.update(eval_metrics)
        results.append(record)

    df_results = pd.DataFrame(results)
    df_results.to_csv(out_csv, index=False)
    if verbose:
        print(f"[HYPERPARAM] Completed all {len(configs)} configurations -> {out_csv}")
    return df_results


if __name__ == "__main__":
    run_hyperparameter_experiments()
