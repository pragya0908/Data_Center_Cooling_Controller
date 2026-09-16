"""Comparative Evaluation Pipeline for CoolRL Controllers.

This module benchmarks three distinct control paradigms across all five
standard experimental scenarios on the held-out test split:
1. Fixed-Cooling Baseline (C = 0.50 static maintain policy).
2. Rule-Based Adaptive Baseline (ASHRAE Class A1 threshold heuristics).
3. Trained Tabular Q-Learning Agent (Greedy policy from Phase 3).

All controllers are evaluated under strictly identical environmental conditions,
workload sequences, ambient weather traces, and physical thermodynamic parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.agents.q_learning import QLearningAgent
from src.agents.state_discretizer import StateDiscretizer
from src.baselines.fixed_cooling import FixedCoolingController
from src.baselines.rule_based import RuleBasedController
from src.environment.datacenter_env import DataCenterEnv

SCENARIOS: list[str] = [
    "normal",
    "high_workload",
    "workload_spikes",
    "high_ambient",
    "combined_stress",
]


def evaluate_single_run(
    controller_name: str,
    controller: Any,
    scenario: str,
    split: str = "test",
    discretizer: StateDiscretizer | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute a deterministic evaluation rollout for a single controller on a scenario.

    Args:
        controller_name: String label ('fixed_cooling_0.50', 'rule_based', 'q_learning').
        controller: The instantiated controller object.
        scenario: Scenario name.
        split: Dataset split ('test' by default).
        discretizer: StateDiscretizer required for Q-learning.
        seed: Environment reset seed.

    Returns:
        Dictionary of comprehensive performance, thermal, and energy metrics.
    """
    env = DataCenterEnv(scenario=scenario, split=split)
    obs, info = env.reset(seed=seed)

    temps: list[float] = []
    energies: list[float] = []
    coolings: list[float] = []
    rewards: list[float] = []
    safety_pens: list[float] = []
    severe_pens: list[float] = []
    overcool_pens: list[float] = []
    churn_pens: list[float] = []
    actions: list[int] = []

    done = False
    step_count = 0

    while not done:
        # Select action using appropriate interface
        if controller_name == "q_learning":
            if discretizer is None:
                discretizer = StateDiscretizer()
            state_id = discretizer.encode(obs)
            # Pure greedy evaluation: training=False guarantees zero exploration
            action = controller.choose_action(state_id, training=False)
        else:
            action = controller.choose_action(obs)

        actions.append(action)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        temps.append(float(info["internal_temperature"]))
        energies.append(float(info["cooling_energy"]))
        coolings.append(float(info["cooling_level"]))
        rewards.append(float(reward))
        safety_pens.append(float(info["safety_penalty"]))
        severe_pens.append(float(info["severe_overheat_penalty"]))
        overcool_pens.append(float(info["overcooling_penalty"]))
        churn_pens.append(float(info["action_churn_penalty"]))

        obs = next_obs

    # Compute aggregate statistical metrics
    t_arr = np.array(temps)
    r_arr = np.array(rewards)
    act_arr = np.array(actions)
    n_steps = len(t_arr)

    steps_gt_27 = int(np.sum(t_arr > 27.0))
    steps_gt_32 = int(np.sum(t_arr > 32.0))
    steps_lt_18 = int(np.sum(t_arr < 18.0))
    steps_in_rec = int(np.sum((t_arr >= 18.0) & (t_arr <= 27.0)))

    act_0_count = int(np.sum(act_arr == 0))
    act_1_count = int(np.sum(act_arr == 1))
    act_2_count = int(np.sum(act_arr == 2))
    act_3_count = int(np.sum(act_arr == 3))
    act_4_count = int(np.sum(act_arr == 4))

    cooling_increases = act_3_count + act_4_count
    cooling_decreases = act_0_count + act_1_count
    action_changes = cooling_increases + cooling_decreases

    return {
        "controller": controller_name,
        "scenario": scenario,
        "split": split,
        "total_steps": n_steps,
        "cumulative_reward": float(np.sum(r_arr)),
        "mean_step_reward": float(np.mean(r_arr)),
        "total_cooling_energy": float(np.sum(energies)),
        "mean_cooling_level": float(np.mean(coolings)),
        "min_internal_temperature": float(np.min(t_arr)),
        "mean_internal_temperature": float(np.mean(t_arr)),
        "max_internal_temperature": float(np.max(t_arr)),
        "temperature_std": float(np.std(t_arr)),
        "recommended_zone_percentage": float((steps_in_rec / n_steps) * 100.0),
        "steps_gt_27": steps_gt_27,
        "percentage_gt_27": float((steps_gt_27 / n_steps) * 100.0),
        "steps_gt_32": steps_gt_32,
        "percentage_gt_32": float((steps_gt_32 / n_steps) * 100.0),
        "steps_lt_18": steps_lt_18,
        "percentage_lt_18": float((steps_lt_18 / n_steps) * 100.0),
        "total_safety_penalty": float(np.sum(safety_pens)),
        "total_severe_overheating_penalty": float(np.sum(severe_pens)),
        "total_overcooling_penalty": float(np.sum(overcool_pens)),
        "total_action_churn_penalty": float(np.sum(churn_pens)),
        "action_0_count": act_0_count,
        "action_1_count": act_1_count,
        "action_2_count": act_2_count,
        "action_3_count": act_3_count,
        "action_4_count": act_4_count,
        "cooling_increase_count": cooling_increases,
        "cooling_decrease_count": cooling_decreases,
        "maintain_action_count": act_2_count,
        "number_of_action_changes": action_changes,
    }


def run_full_comparative_evaluation(
    model_path: str | Path = "results/training/q_learning_normal_train.npz",
    split: str = "test",
    scenarios: list[str] = SCENARIOS,
    output_dir: str | Path = "results/evaluation",
    seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete 3-controller comparative evaluation across all scenarios.

    Args:
        model_path: Path to pre-trained Q-learning model artifact.
        split: Dataset split ('test' by default).
        scenarios: List of scenario names to evaluate.
        output_dir: Destination directory for CSV result tables.
        seed: Random seed for environment resets.
        verbose: Print execution progress.

    Returns:
        tuple of (detailed_comparison_df, summary_comparison_df)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize controllers
    fixed_ctrl = FixedCoolingController(target_cooling=0.50)
    rule_ctrl = RuleBasedController()

    # Load Q-learning agent and verify integrity
    resolved_model_path = Path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Trained Q-learning model not found at: {resolved_model_path}")

    q_agent = QLearningAgent.load(resolved_model_path, seed=seed)
    q_table_backup = q_agent.q_table.copy()
    discretizer = StateDiscretizer()

    controllers: dict[str, Any] = {
        "fixed_cooling_0.50": fixed_ctrl,
        "rule_based": rule_ctrl,
        "q_learning": q_agent,
    }

    detailed_records: list[dict[str, Any]] = []

    if verbose:
        print(f"Executing Phase 4 comparative evaluation on '{split}' split ({len(scenarios)} scenarios, 3 controllers):")

    for sc in scenarios:
        for ctrl_name, ctrl in controllers.items():
            if verbose:
                print(f"  Evaluating {ctrl_name:<20} on scenario: {sc:<18}...", end="", flush=True)

            res = evaluate_single_run(
                controller_name=ctrl_name,
                controller=ctrl,
                scenario=sc,
                split=split,
                discretizer=discretizer,
                seed=seed,
            )
            detailed_records.append(res)
            if verbose:
                print(f" Done! (Reward: {res['cumulative_reward']:8.2f}, Energy: {res['total_cooling_energy']:5.2f}, Max T: {res['max_internal_temperature']:5.2f}°C)")

    # Verify Q-table was strictly unaltered during evaluation
    np.testing.assert_array_equal(
        q_agent.q_table,
        q_table_backup,
        err_msg="CRITICAL: Q-table was modified during evaluation! Greedy evaluation must be read-only.",
    )

    df_detailed = pd.DataFrame(detailed_records)

    # 2. Compute aggregate summary table grouped by controller
    summary_rows: list[dict[str, Any]] = []
    for ctrl_name, group in df_detailed.groupby("controller"):
        summary_rows.append({
            "controller": ctrl_name,
            "mean_cumulative_reward": float(group["cumulative_reward"].mean()),
            "mean_cooling_energy": float(group["total_cooling_energy"].mean()),
            "mean_internal_temperature": float(group["mean_internal_temperature"].mean()),
            "mean_max_temperature": float(group["max_internal_temperature"].mean()),
            "mean_recommended_zone_pct": float(group["recommended_zone_percentage"].mean()),
            "total_steps_gt_27": int(group["steps_gt_27"].sum()),
            "total_steps_gt_32": int(group["steps_gt_32"].sum()),
            "total_steps_lt_18": int(group["steps_lt_18"].sum()),
            "total_action_changes": int(group["number_of_action_changes"].sum()),
        })

    df_summary = pd.DataFrame(summary_rows)

    # 3. Save CSV artifacts
    csv_detailed_path = out_dir / "controller_comparison.csv"
    csv_summary_path = out_dir / "controller_comparison_summary.csv"

    df_detailed.to_csv(csv_detailed_path, index=False)
    df_summary.to_csv(csv_summary_path, index=False)

    if verbose:
        print(f"\nSaved detailed comparison to: {csv_detailed_path}")
        print(f"Saved summary comparison to:  {csv_summary_path}")

    return df_detailed, df_summary
