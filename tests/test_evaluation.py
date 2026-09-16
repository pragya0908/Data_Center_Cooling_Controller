"""Integration tests for comparative evaluation pipeline in CoolRL.

Validates:
1. End-to-end execution of Fixed, Rule-based, and Q-learning controllers on test split.
2. Identical trajectory step counts (515 steps on test split).
3. Action distribution consistency (action sums strictly equal total steps).
4. Determinism of repeated evaluation rollouts.
5. Read-only integrity of the pre-trained Q-table during evaluation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.agents.q_learning import QLearningAgent
from src.agents.state_discretizer import StateDiscretizer
from src.baselines.fixed_cooling import FixedCoolingController
from src.baselines.rule_based import RuleBasedController
from src.evaluation.evaluate_controllers import (
    evaluate_single_run,
    run_full_comparative_evaluation,
)

MODEL_PATH = Path("results/training/q_learning_normal_train.npz")


class TestComparativeEvaluationPipeline:
    """Validate comparative evaluation mechanics across controllers."""

    @pytest.fixture
    def trained_q_agent(self) -> QLearningAgent:
        assert MODEL_PATH.exists(), f"Phase 3 model not found at: {MODEL_PATH}"
        return QLearningAgent.load(MODEL_PATH)

    def test_all_controllers_execute_on_test_split(self, trained_q_agent: QLearningAgent):
        """All three controllers must execute exactly 515 steps on the test split."""
        controllers = [
            ("fixed_cooling_0.50", FixedCoolingController(target_cooling=0.50)),
            ("rule_based", RuleBasedController()),
            ("q_learning", trained_q_agent),
        ]
        discretizer = StateDiscretizer()

        for name, ctrl in controllers:
            res = evaluate_single_run(
                controller_name=name,
                controller=ctrl,
                scenario="normal",
                split="test",
                discretizer=discretizer,
                seed=42,
            )

            assert res["controller"] == name
            assert res["scenario"] == "normal"
            assert res["total_steps"] == 515
            assert np.isfinite(res["cumulative_reward"])
            assert np.isfinite(res["total_cooling_energy"])
            assert np.isfinite(res["mean_internal_temperature"])

            # Verify action counts partition the 515 total steps
            total_actions = (
                res["action_0_count"]
                + res["action_1_count"]
                + res["action_2_count"]
                + res["action_3_count"]
                + res["action_4_count"]
            )
            assert total_actions == 515
            assert res["maintain_action_count"] + res["number_of_action_changes"] == 515

    def test_q_table_read_only_integrity_during_evaluation(self, trained_q_agent: QLearningAgent):
        """Greedy evaluation must never mutate Q-table values."""
        q_table_before = trained_q_agent.q_table.copy()

        evaluate_single_run(
            controller_name="q_learning",
            controller=trained_q_agent,
            scenario="combined_stress",
            split="test",
            seed=42,
        )

        np.testing.assert_array_equal(trained_q_agent.q_table, q_table_before)

    def test_evaluation_determinism(self, trained_q_agent: QLearningAgent):
        """Repeated evaluation under identical conditions produces bitwise identical metrics."""
        res1 = evaluate_single_run(
            controller_name="q_learning",
            controller=trained_q_agent,
            scenario="high_workload",
            split="test",
            seed=100,
        )
        res2 = evaluate_single_run(
            controller_name="q_learning",
            controller=trained_q_agent,
            scenario="high_workload",
            split="test",
            seed=100,
        )

        assert res1 == res2

    def test_full_evaluation_pipeline_artifacts(self, trained_q_agent: QLearningAgent):
        """Verify full evaluation saves controller_comparison.csv and summary.csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df_detailed, df_summary = run_full_comparative_evaluation(
                model_path=MODEL_PATH,
                split="test",
                scenarios=["normal", "high_ambient"],
                output_dir=tmpdir,
                seed=42,
                verbose=False,
            )

            # 2 scenarios x 3 controllers = 6 rows
            assert len(df_detailed) == 6
            assert len(df_summary) == 3

            assert (Path(tmpdir) / "controller_comparison.csv").exists()
            assert (Path(tmpdir) / "controller_comparison_summary.csv").exists()
