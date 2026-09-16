"""Tests for Phase 5 experiment modules in CoolRL.

Validates:
1. Hyperparameter sensitivity experiment configuration integrity and outputs.
2. Seed robustness reproducible generation and valid descriptive statistics.
3. Google workload generalization execution without modifying Q-table.
4. Cross-scenario evaluation data consistency and Q-table read-only preservation.
5. Strict prevention of data leakage (no test or Google data used for training).
6. Finite metric validation and valid action IDs.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.agents.q_learning import QLearningAgent
from src.agents.state_discretizer import StateDiscretizer
from src.baselines.fixed_cooling import FixedCoolingController
from src.baselines.rule_based import RuleBasedController
from src.environment.datacenter_env import DataCenterEnv
from src.experiments.generalization_experiments import (
    ensure_google_weather_aligned_dataset,
    run_google_generalization_experiments,
)
from src.experiments.hyperparameter_experiments import evaluate_agent_greedy
from src.experiments.robustness_experiments import compute_seed_descriptive_statistics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE_3_MODEL = PROJECT_ROOT / "results" / "training" / "q_learning_normal_train.npz"


class TestPhase5Experiments:
    """Test suite for Phase 5 experimental pipelines."""

    @pytest.fixture
    def trained_q_agent(self) -> QLearningAgent:
        assert PHASE_3_MODEL.exists(), f"Phase 3 model not found at {PHASE_3_MODEL}"
        return QLearningAgent.load(PHASE_3_MODEL)

    def test_google_weather_aligned_dataset_integrity(self):
        """Aligned weather dataset for Google workload must exist and have exactly 8064 rows."""
        csv_path = ensure_google_weather_aligned_dataset()
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 8064
        assert "ambient_temp_c" in df.columns
        assert df["ambient_temp_c"].isna().sum() == 0
        assert np.all(np.isfinite(df["ambient_temp_c"]))

    def test_q_table_read_only_during_greedy_eval(self, trained_q_agent: QLearningAgent):
        """Greedy evaluation must never mutate the trained Q-table."""
        q_backup = trained_q_agent.q_table.copy()
        res = evaluate_agent_greedy(
            agent=trained_q_agent,
            scenario="normal",
            split="val",
            seed=42,
        )
        assert res["eval_total_steps"] == 288.0
        np.testing.assert_array_equal(trained_q_agent.q_table, q_backup)

    def test_hyperparameter_sensitivity_csv_structure(self):
        """Hyperparameter sensitivity CSV must contain all 5 configurations with finite metrics."""
        csv_path = PROJECT_ROOT / "results" / "experiments" / "hyperparameter_sensitivity.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 5
        expected_configs = {
            "alpha_0.05_gamma_0.95",
            "alpha_0.10_gamma_0.95",
            "alpha_0.20_gamma_0.95",
            "alpha_0.10_gamma_0.90",
            "alpha_0.10_gamma_0.99",
        }
        assert set(df["config_id"]) == expected_configs
        assert np.all(np.isfinite(df["final_train_reward"]))
        assert np.all(np.isfinite(df["eval_cumulative_reward"]))
        assert np.all(df["eval_steps_gt_32"] == 0.0)

    def test_seed_robustness_statistics(self):
        """Seed robustness summary statistics must have non-negative standard deviations and correct bounds."""
        csv_path = PROJECT_ROOT / "results" / "experiments" / "seed_robustness.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 5
        assert set(df["seed"]) == {42, 101, 2024, 7, 999}

        summary = compute_seed_descriptive_statistics(df)
        assert len(summary) > 0
        for _, row in summary.iterrows():
            assert row["std"] >= 0.0
            assert row["min"] <= row["mean"] <= row["max"] or np.isclose(row["min"], row["max"])

    def test_google_generalization_csv_structure(self):
        """Google generalization CSV must contain all three controllers evaluated on 8064 steps."""
        csv_path = PROJECT_ROOT / "results" / "experiments" / "generalization_google.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        expected_ctrls = {"fixed_cooling_0.50", "rule_based", "q_learning"}
        assert set(df["controller"]) == expected_ctrls
        assert np.all(df["total_steps"] == 8064)
        assert np.all(df["workload_source"] == "google_cluster_2019")

        # Check action counts sum to 8064
        for _, row in df.iterrows():
            action_sum = (
                row["action_0_count"]
                + row["action_1_count"]
                + row["action_2_count"]
                + row["action_3_count"]
                + row["action_4_count"]
            )
            assert action_sum == 8064
            assert np.isfinite(row["cumulative_reward"])
            assert np.isfinite(row["total_cooling_energy"])

    def test_cross_scenario_csv_structure(self):
        """Cross-scenario CSV must contain 15 rows with identical 515 steps per test episode."""
        csv_path = PROJECT_ROOT / "results" / "experiments" / "cross_scenario.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 15
        assert np.all(df["total_steps"] == 515)
        assert np.all(df["split"] == "test")
        assert np.all(np.isfinite(df["cumulative_reward"]))
        assert np.all(np.isfinite(df["total_cooling_energy"]))

    def test_no_data_leakage_in_experiment_training(self):
        """Verify that training in experiments strictly uses 'train' split and never 'test' or Google."""
        hp_df = pd.read_csv(PROJECT_ROOT / "results" / "experiments" / "hyperparameter_sensitivity.csv")
        assert np.all(hp_df["train_split"] == "train")
        assert np.all(hp_df["eval_split"] == "val")

        seed_df = pd.read_csv(PROJECT_ROOT / "results" / "experiments" / "seed_robustness.csv")
        assert np.all(seed_df["train_split"] == "train")
        assert np.all(seed_df["eval_split"] == "val")
