"""Integration and unit tests for the tabular Q-learning training pipeline.

Validates:
1. Short smoke test training execution (10 episodes).
2. Q-table modifications (learning progression away from zero-initialization).
3. History DataFrame structure, complete columns, and rolling window calculation.
4. Epsilon progression across training episodes.
5. Strict reproducibility under fixed random seeds.
6. Automatic artifact saving (.npz model and .csv history).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.agents.q_learning import QLearningAgent
from src.agents.state_discretizer import StateDiscretizer
from src.environment.datacenter_env import DataCenterEnv
from src.training.train_q_learning import train_q_learning


class TestTrainingPipelineSmoke:
    """Validate end-to-end execution of a 10-episode training run."""

    def test_smoke_training_run(self):
        """10 episodes on train split complete cleanly and update the Q-table."""
        env = DataCenterEnv(scenario="normal", split="train")
        agent, history = train_q_learning(
            env=env,
            num_episodes=10,
            alpha=0.10,
            gamma=0.95,
            epsilon=1.00,
            epsilon_decay=0.90,
            seed=42,
            verbose=False,
        )

        # 1. Verify history DataFrame structure
        expected_cols = [
            "episode",
            "cumulative_reward",
            "mean_step_reward",
            "epsilon",
            "steps",
            "total_cooling_energy",
            "average_internal_temperature",
            "maximum_internal_temperature",
            "overheating_steps_gt_27",
            "severe_overheating_steps_gt_32",
            "overcooling_steps_lt_18",
            "rolling_mean_reward",
        ]
        for col in expected_cols:
            assert col in history.columns, f"Missing column: {col}"

        assert len(history) == 10
        assert list(history["episode"]) == list(range(1, 11))
        assert np.all(np.isfinite(history["cumulative_reward"]))
        assert np.all(np.isfinite(history["rolling_mean_reward"]))

        # 2. Verify all episodes completed exact train split length (1440 steps)
        assert np.all(history["steps"] == 1440)

        # 3. Verify Q-table has been modified from all-zeros
        nonzero_count = np.count_nonzero(agent.q_table)
        assert nonzero_count > 0, "Q-table remained all zeros after 10 episodes of training"
        assert np.all(np.isfinite(agent.q_table))

        # 4. Verify epsilon decayed
        initial_eps = history["epsilon"].iloc[0]
        final_eps = history["epsilon"].iloc[-1]
        assert initial_eps > final_eps
        assert np.isclose(final_eps, 1.00 * (0.90 ** 9), atol=1e-4)


class TestTrainingReproducibility:
    """Verify bit-for-bit determinism of training across independent seeded runs."""

    def test_reproducible_seeded_training(self):
        """Two 5-episode training runs with identical seed must yield identical results."""
        # Run 1
        agent1, hist1 = train_q_learning(
            scenario="normal",
            split="train",
            num_episodes=5,
            seed=777,
            verbose=False,
        )

        # Run 2
        agent2, hist2 = train_q_learning(
            scenario="normal",
            split="train",
            num_episodes=5,
            seed=777,
            verbose=False,
        )

        # Verify Q-table equality
        np.testing.assert_array_equal(agent1.q_table, agent2.q_table)

        # Verify history DataFrame equality
        np.testing.assert_array_almost_equal(
            hist1["cumulative_reward"].values, hist2["cumulative_reward"].values
        )
        np.testing.assert_array_almost_equal(
            hist1["total_cooling_energy"].values, hist2["total_cooling_energy"].values
        )
        np.testing.assert_array_almost_equal(
            hist1["average_internal_temperature"].values,
            hist2["average_internal_temperature"].values,
        )


class TestArtifactSaving:
    """Verify automatic saving of .npz model and .csv history."""

    def test_save_artifacts_on_completion(self):
        """Specifying save_dir writes q_learning_{scenario}_{split}.npz and .csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            agent, history = train_q_learning(
                scenario="normal",
                split="train",
                num_episodes=3,
                seed=999,
                save_dir=save_path,
                verbose=False,
            )

            expected_model_path = save_path / "q_learning_normal_train.npz"
            expected_hist_path = save_path / "q_learning_normal_train_history.csv"

            assert expected_model_path.exists()
            assert expected_hist_path.exists()

            # Verify saved model can be loaded and matches trained agent
            loaded_agent = QLearningAgent.load(expected_model_path)
            np.testing.assert_array_equal(loaded_agent.q_table, agent.q_table)
            np.testing.assert_array_equal(loaded_agent.get_policy(), agent.get_policy())
