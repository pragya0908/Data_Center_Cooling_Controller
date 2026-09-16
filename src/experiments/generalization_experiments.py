"""Generalization experiments for CoolRL controllers.

This module implements two rigorous validation protocols:
1. Google Workload Generalization:
   Evaluates the frozen Phase 3 Q-learning agent, fixed cooling, and rule-based
   controllers under the unseen Google Cluster 2019 instance usage dataset.
   Zero retraining, zero fine-tuning, zero parameter changes.

2. Cross-Scenario Generalization:
   Evaluates all three controllers across the 5 standard environmental stress
   scenarios on the held-out test split, compiling cross-scenario comparative metrics.
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
from src.evaluation.evaluate_controllers import SCENARIOS, evaluate_single_run

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def ensure_google_weather_aligned_dataset() -> Path:
    """Ensure temporally-aligned weather dataset for Google 8064-step trace exists."""
    out_csv = PROCESSED_DATA_DIR / "bengaluru_weather_300s_google_aligned.csv"
    if out_csv.exists():
        return out_csv

    google_csv = PROCESSED_DATA_DIR / "google_workload_300s.csv"
    hourly_weather_csv = PROCESSED_DATA_DIR / "bengaluru_weather_hourly.csv"

    if not google_csv.exists() or not hourly_weather_csv.exists():
        raise FileNotFoundError("Missing google_workload_300s.csv or bengaluru_weather_hourly.csv")

    df_google = pd.read_csv(google_csv)
    df_hourly = pd.read_csv(hourly_weather_csv)

    n_steps = len(df_google)
    target_elapsed_hours = np.arange(n_steps) * (300.0 / 3600.0)
    hourly_hours = df_hourly["elapsed_hours"].values
    hourly_temps = df_hourly["ambient_temp_c"].values
    interpolated_temps = np.interp(target_elapsed_hours, hourly_hours, hourly_temps)

    df_aligned = pd.DataFrame({
        "step": np.arange(n_steps),
        "elapsed_seconds": np.arange(n_steps) * 300,
        "elapsed_hours": target_elapsed_hours,
        "ambient_temp_c": np.round(interpolated_temps, 3),
    })
    df_aligned.to_csv(out_csv, index=False)
    return out_csv


def run_google_generalization_experiments(
    model_path: Path | str = PROJECT_ROOT / "results" / "training" / "q_learning_normal_train.npz",
    output_csv: Path | str = PROJECT_ROOT / "results" / "experiments" / "generalization_google.csv",
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate frozen controllers on the independent Google 2019 workload trace.

    Contract:
    - Controllers: Fixed (0.50), Rule-based, Q-learning (Phase 3 model).
    - Workload: Google Cluster Workload Traces 2019 (8,064 300s intervals = 28 days).
    - Ambient: Aligned Bengaluru hourly weather trace interpolated across 28 days.
    - Zero retraining or modification of the Q-table.

    Returns:
        DataFrame containing evaluation results for all three controllers.
    """
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    m_path = Path(model_path)

    weather_path = ensure_google_weather_aligned_dataset()
    workload_path = PROCESSED_DATA_DIR / "google_workload_300s.csv"

    # Compute and verify Google workload statistics
    df_gw = pd.read_csv(workload_path)
    gw_mean = float(df_gw["workload"].mean())
    gw_std = float(df_gw["workload"].std())
    gw_min = float(df_gw["workload"].min())
    gw_max = float(df_gw["workload"].max())

    controllers: dict[str, Any] = {
        "fixed_cooling_0.50": FixedCoolingController(target_cooling=0.50),
        "rule_based": RuleBasedController(),
        "q_learning": QLearningAgent.load(m_path, seed=seed),
    }

    # Verify Q-table backup
    q_table_backup = controllers["q_learning"].q_table.copy()
    discretizer = StateDiscretizer()

    results: list[dict[str, Any]] = []

    for name, ctrl in controllers.items():
        if verbose:
            print(f"[GOOGLE GENERALIZATION] Evaluating {name} on Google workload (8,064 steps)...")

        env = DataCenterEnv(
            scenario="normal",
            split="all",
            workload_csv_path=workload_path,
            weather_csv_path=weather_path,
        )
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
        while not done:
            if name == "q_learning":
                s_id = discretizer.encode(obs)
                act = ctrl.choose_action(s_id, training=False)
            else:
                act = ctrl.choose_action(obs)

            actions.append(act)
            next_obs, r, term, trunc, info = env.step(act)
            done = term or trunc

            temps.append(float(info["internal_temperature"]))
            energies.append(float(info["cooling_energy"]))
            coolings.append(float(info["cooling_level"]))
            rewards.append(float(r))
            safety_pens.append(float(info["safety_penalty"]))
            severe_pens.append(float(info["severe_overheat_penalty"]))
            overcool_pens.append(float(info["overcooling_penalty"]))
            churn_pens.append(float(info["action_churn_penalty"]))
            obs = next_obs

        # Calculate metrics
        t_arr = np.array(temps)
        r_arr = np.array(rewards)
        act_arr = np.array(actions)
        n_steps = len(t_arr)

        steps_gt_27 = int(np.sum(t_arr > 27.0))
        steps_gt_32 = int(np.sum(t_arr > 32.0))
        steps_lt_18 = int(np.sum(t_arr < 18.0))
        steps_rec = int(np.sum((t_arr >= 18.0) & (t_arr <= 27.0)))

        act_0 = int(np.sum(act_arr == 0))
        act_1 = int(np.sum(act_arr == 1))
        act_2 = int(np.sum(act_arr == 2))
        act_3 = int(np.sum(act_arr == 3))
        act_4 = int(np.sum(act_arr == 4))

        rec = {
            "controller": name,
            "workload_source": "google_cluster_2019",
            "weather_source": "nasa_power_bengaluru_may2018",
            "scenario": "normal",
            "split": "all_28_days",
            "total_steps": n_steps,
            "workload_mean": gw_mean,
            "workload_std": gw_std,
            "workload_min": gw_min,
            "workload_max": gw_max,
            "cumulative_reward": float(np.sum(r_arr)),
            "mean_step_reward": float(np.mean(r_arr)),
            "total_cooling_energy": float(np.sum(energies)),
            "mean_cooling_level": float(np.mean(coolings)),
            "min_internal_temperature": float(np.min(t_arr)),
            "mean_internal_temperature": float(np.mean(t_arr)),
            "max_internal_temperature": float(np.max(t_arr)),
            "temperature_std": float(np.std(t_arr)),
            "recommended_zone_percentage": float((steps_rec / n_steps) * 100.0),
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
            "action_0_count": act_0,
            "action_1_count": act_1,
            "action_2_count": act_2,
            "action_3_count": act_3,
            "action_4_count": act_4,
            "cooling_increase_count": act_3 + act_4,
            "cooling_decrease_count": act_0 + act_1,
            "maintain_action_count": act_2,
            "number_of_action_changes": (act_3 + act_4) + (act_0 + act_1),
        }
        results.append(rec)

    # Verify Q-table was not modified during evaluation
    np.testing.assert_array_equal(
        controllers["q_learning"].q_table,
        q_table_backup,
        err_msg="Q-table mutated during Google generalization evaluation!",
    )

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv, index=False)
    if verbose:
        print(f"[GOOGLE GENERALIZATION] Completed -> {out_csv}")
    return df_out


def run_cross_scenario_experiments(
    model_path: Path | str = PROJECT_ROOT / "results" / "training" / "q_learning_normal_train.npz",
    output_csv: Path | str = PROJECT_ROOT / "results" / "experiments" / "cross_scenario.csv",
    scenarios: list[str] = SCENARIOS,
    split: str = "test",
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Execute cross-scenario evaluation across all 5 test scenarios.

    Evaluates Fixed (0.50), Rule-based, and Q-learning (Phase 3 model) across:
    - normal
    - high_workload
    - workload_spikes
    - high_ambient
    - combined_stress

    Returns:
        DataFrame containing 15 rows (3 controllers x 5 scenarios).
    """
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    m_path = Path(model_path)

    controllers: dict[str, Any] = {
        "fixed_cooling_0.50": FixedCoolingController(target_cooling=0.50),
        "rule_based": RuleBasedController(),
        "q_learning": QLearningAgent.load(m_path, seed=seed),
    }

    q_table_backup = controllers["q_learning"].q_table.copy()
    discretizer = StateDiscretizer()

    results: list[dict[str, Any]] = []

    for sc in scenarios:
        for name, ctrl in controllers.items():
            if verbose:
                print(f"[CROSS-SCENARIO] Evaluating {name} on {sc} ({split} split)...")
            res = evaluate_single_run(
                controller_name=name,
                controller=ctrl,
                scenario=sc,
                split=split,
                discretizer=discretizer,
                seed=seed,
            )
            # Add metadata tag
            res["workload_source"] = "alibaba_cluster_2018"
            res["model_source"] = "phase_3_trained_normal_train"
            results.append(res)

    np.testing.assert_array_equal(
        controllers["q_learning"].q_table,
        q_table_backup,
        err_msg="Q-table mutated during cross-scenario evaluation!",
    )

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv, index=False)
    if verbose:
        print(f"[CROSS-SCENARIO] Completed all {len(results)} runs -> {out_csv}")
    return df_out


if __name__ == "__main__":
    run_google_generalization_experiments()
    run_cross_scenario_experiments()
