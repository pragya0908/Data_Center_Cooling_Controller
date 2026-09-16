"""Training seed robustness experiments for Q-learning in CoolRL.

This module evaluates the stability of tabular Q-learning across multiple independent
random training seeds while keeping all environment, model, and hyperparameter
specifications completely invariant:
- alpha = 0.10
- gamma = 0.95
- epsilon = 1.0 -> 0.05 (decay 0.995)
- episodes = 500
- scenario = normal
- train split = train (1440 steps)

Seeds evaluated: [42, 101, 2024, 7, 999] (5 independent seeds).

Each resulting model is evaluated under the SAME greedy validation protocol on 'val'.
Descriptive statistics (mean, std, min, max) are calculated across seeds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.agents.q_learning import QLearningAgent
from src.experiments.hyperparameter_experiments import evaluate_agent_greedy
from src.training.train_q_learning import train_q_learning

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEEDS: list[int] = [42, 101, 2024, 7, 999]


def run_seed_robustness_experiments(
    seeds: list[int] = SEEDS,
    output_csv: Path | str = PROJECT_ROOT / "results" / "experiments" / "seed_robustness.csv",
    model_dir: Path | str = PROJECT_ROOT / "results" / "experiments" / "models" / "seeds",
    num_episodes: int = 500,
    alpha: float = 0.10,
    gamma: float = 0.95,
    eval_split: str = "val",
    verbose: bool = True,
) -> pd.DataFrame:
    """Execute multi-seed training and evaluation runs.

    Args:
        seeds: List of integer seeds.
        output_csv: Path to save result CSV.
        model_dir: Directory to save trained seed model weights.
        num_episodes: Episodes per training run.
        alpha: Learning rate.
        gamma: Discount factor.
        eval_split: Evaluation split ('val').
        verbose: Print progress.

    Returns:
        DataFrame containing seed run rows and summary descriptive statistics.
    """
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    m_dir = Path(model_dir)
    m_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for s in seeds:
        if verbose:
            print(f"[SEED ROBUSTNESS] Training with seed {s} ({num_episodes} episodes)...")

        save_path = m_dir / f"q_learning_seed_{s}.npz"
        trained_agent, history_df = train_q_learning(
            scenario="normal",
            split="train",
            num_episodes=num_episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.995,
            seed=s,
            save_dir=None,
            verbose=False,
        )
        trained_agent.save(save_path)

        final_row = history_df.iloc[-1]
        final_reward = float(final_row["cumulative_reward"])
        rolling_50_reward = float(final_row["rolling_mean_reward"])

        # Q-table metrics
        q_table = trained_agent.q_table
        non_zero_states = int(np.sum(np.any(q_table != 0.0, axis=1)))

        # Greedy evaluation on validation split
        eval_metrics = evaluate_agent_greedy(
            agent=trained_agent,
            scenario="normal",
            split=eval_split,
            seed=42,  # Standardized evaluation seed
        )

        record: dict[str, Any] = {
            "seed": s,
            "alpha": alpha,
            "gamma": gamma,
            "episodes": num_episodes,
            "scenario": "normal",
            "train_split": "train",
            "eval_split": eval_split,
            "final_train_reward": final_reward,
            "rolling_50_train_reward": rolling_50_reward,
            "visited_states_count": non_zero_states,
            "visited_states_pct": float((non_zero_states / q_table.shape[0]) * 100.0),
            "model_path": str(save_path.relative_to(PROJECT_ROOT)),
        }
        record.update(eval_metrics)
        results.append(record)

    df_results = pd.DataFrame(results)
    df_results.to_csv(out_csv, index=False)
    if verbose:
        print(f"[SEED ROBUSTNESS] Completed all {len(seeds)} seeds -> {out_csv}")
    return df_results


def compute_seed_descriptive_statistics(df_seeds: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, min, and max across seed evaluations.

    Args:
        df_seeds: DataFrame from run_seed_robustness_experiments.

    Returns:
        Summary DataFrame containing descriptive statistics.
    """
    metrics = [
        "final_train_reward",
        "rolling_50_train_reward",
        "eval_cumulative_reward",
        "eval_total_cooling_energy",
        "eval_mean_internal_temp",
        "eval_max_internal_temp",
        "eval_recommended_zone_pct",
        "eval_pct_gt_27",
        "eval_pct_gt_32",
        "eval_pct_lt_18",
    ]

    summary_rows = []
    for m in metrics:
        if m in df_seeds.columns:
            vals = df_seeds[m].to_numpy(dtype=float)
            summary_rows.append({
                "metric": m,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            })

    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    df = run_seed_robustness_experiments()
    summary = compute_seed_descriptive_statistics(df)
    print(summary)
