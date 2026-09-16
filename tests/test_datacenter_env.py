"""Unit tests for DataCenterEnv simulation environment.

Tests cover API conformance, action space validation, observation shapes,
scenario initialization, continuous state tracking, determinism, split slicing,
and physical behavioral sanity across all five scenarios.
"""

import pytest
import numpy as np

from src.environment import (
    ContinuousObservationSpace,
    DataCenterEnv,
    DiscreteActionSpace,
)
from src.data.scenario_config import ScenarioType


class TestEnvironmentAPIAndActions:
    """Tests verifying the Gymnasium-style interface and action mechanics."""

    def test_discrete_action_space(self):
        """Action space descriptor has 5 actions with correct delta increments."""
        space = DiscreteActionSpace()
        assert space.n == 5
        assert space.actions == (0, 1, 2, 3, 4)
        assert space.deltas == (-0.20, -0.10, 0.00, +0.10, +0.20)
        assert space.contains(0)
        assert space.contains(4)
        assert not space.contains(5)
        assert not space.contains(-1)
        assert not space.contains("invalid")

    def test_observation_space(self):
        """Observation space has 5 continuous dimensions."""
        space = ContinuousObservationSpace()
        assert space.shape == (5,)
        assert len(space.low) == 5
        assert len(space.high) == 5

    def test_invalid_action_raises_value_error(self):
        """Invalid actions must raise a clear ValueError."""
        env = DataCenterEnv(scenario="normal")
        env.reset()

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(-1)

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(5)

        with pytest.raises(ValueError, match="Invalid action"):
            env.step("maintain")

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(2.5)

    def test_cooling_level_clamping(self):
        """Cooling level remains strictly within [0.0, 1.0] under extreme actions."""
        env = DataCenterEnv(scenario="normal")
        env.reset()

        # Repeatedly decrease cooling (action 0 = -0.20)
        for _ in range(10):
            obs, _, _, _, info = env.step(action=0)
            assert 0.0 <= info["cooling_level"] <= 1.0
            assert 0.0 <= obs[3] <= 1.0
        assert np.isclose(env.cooling_level, 0.0)

        # Repeatedly increase cooling (action 4 = +0.20)
        for _ in range(15):
            obs, _, _, _, info = env.step(action=4)
            assert 0.0 <= info["cooling_level"] <= 1.0
            assert 0.0 <= obs[3] <= 1.0
        assert np.isclose(env.cooling_level, 1.0)


class TestResetAndStepMechanics:
    """Tests verifying state reset and step transitions."""

    def test_reset_initial_conditions(self):
        """Reset returns observation of shape (5,) and expected initial values."""
        env = DataCenterEnv(scenario="normal")
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)
        assert obs.dtype == np.float32

        # Check initial values: T_int = 24.0, C = 0.50, delta_T = 0.0
        assert np.isclose(obs[0], 24.0)
        assert np.isclose(obs[3], 0.50)
        assert np.isclose(obs[4], 0.0)

        # Info dict completeness
        required_keys = [
            "workload",
            "ambient_temperature",
            "internal_temperature",
            "cooling_level",
            "ambient_derating_factor",
            "cooling_effect",
            "cop",
            "cooling_power",
            "cooling_energy",
            "temperature_change",
            "temperature_trend",
            "ashrae_zone",
            "step_index",
            "scenario",
        ]
        for key in required_keys:
            assert key in info, f"Missing key '{key}' in info dict"

        assert info["step_index"] == 0
        assert info["temperature_trend"] == "stable"
        assert info["ashrae_zone"] == "recommended"

    def test_step_progression_and_return_signature(self):
        """Step returns (obs, reward, terminated, truncated, info)."""
        env = DataCenterEnv(scenario="normal")
        env.reset()

        obs, reward, terminated, truncated, info = env.step(action=2)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)
        assert isinstance(reward, float)
        assert np.isfinite(reward)
        assert reward == info["reward"]
        assert "reward_breakdown" in info
        assert isinstance(terminated, bool)
        assert not terminated
        assert isinstance(truncated, bool)
        assert not truncated
        assert isinstance(info, dict)
        assert info["step_index"] == 0  # Step that just executed

    def test_step_after_termination_raises_runtime_error(self):
        """Stepping an exhausted environment raises RuntimeError."""
        # Use a small split (validation = 288 steps) to verify exhaustion
        env = DataCenterEnv(scenario="normal", split="val")
        env.reset()

        for _ in range(288):
            _, _, terminated, _, _ = env.step(action=2)

        assert terminated
        with pytest.raises(RuntimeError, match="Cannot step in an exhausted environment"):
            env.step(action=2)


class TestDeterminism:
    """Tests verifying strict bit-for-bit determinism of the baseline environment."""

    def test_deterministic_identical_rollouts(self):
        """Two independent environments running identical action sequences produce identical outputs."""
        env1 = DataCenterEnv(scenario="normal")
        env2 = DataCenterEnv(scenario="normal")

        obs1, info1 = env1.reset()
        obs2, info2 = env2.reset()

        np.testing.assert_array_equal(obs1, obs2)
        assert info1 == info2

        actions = [3, 4, 2, 1, 0, 0, 4, 3, 2, 2, 1, 0, 4, 4, 3, 2, 1, 0, 3, 2]
        for a in actions:
            o1, r1, term1, trunc1, i1 = env1.step(a)
            o2, r2, term2, trunc2, i2 = env2.step(a)

            np.testing.assert_array_equal(o1, o2)
            assert r1 == r2
            assert term1 == term2
            assert trunc1 == trunc2
            assert i1 == i2


class TestAllFiveScenarios:
    """Tests verifying that all five scenarios instantiate and behave properly."""

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "normal",
            "high_workload",
            "workload_spikes",
            "high_ambient",
            "combined_stress",
        ],
    )
    def test_scenario_instantiation_and_rollout(self, scenario_name: str):
        """Instantiate scenario, reset, and step 25 actions without error."""
        env = DataCenterEnv(scenario=scenario_name)
        obs, info = env.reset()

        assert obs is not None
        assert 0.0 <= obs[3] <= 1.0  # Cooling level
        assert info["cop"] >= env.cop_params.cop_min
        assert info["cop"] <= env.cop_params.cop_max

        for _ in range(25):
            obs, r, term, trunc, info = env.step(action=2)
            assert np.all(np.isfinite(obs))
            assert 0.0 <= info["cooling_level"] <= 1.0
            assert info["cooling_energy"] >= 0.0
            assert info["cop"] >= env.cop_params.cop_min
            assert info["cop"] <= env.cop_params.cop_max

    def test_invalid_scenario_raises_value_error(self):
        """Unknown scenario names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            DataCenterEnv(scenario="super_extreme_scenario")


class TestTemporalSplits:
    """Tests verifying temporal dataset partition slicing."""

    def test_split_lengths(self):
        """Verifies that split horizons match the Phase 1C configuration."""
        env_all = DataCenterEnv(split="all")
        assert env_all.trajectory_length == 2243

        env_train = DataCenterEnv(split="train")
        assert env_train.trajectory_length == 1440

        env_val = DataCenterEnv(split="val")
        assert env_val.trajectory_length == 288

        env_test = DataCenterEnv(split="test")
        assert env_test.trajectory_length == 515


class TestPhysicalSanityAcrossScenarios:
    """Tests verifying physical contrasts between normal and stress scenarios."""

    def test_high_workload_elevates_temperatures(self):
        """Scenario 2 produces higher mean workload and internal temperature than Scenario 1."""
        env_norm = DataCenterEnv(scenario="normal", split="val")
        env_high = DataCenterEnv(scenario="high_workload", split="val")

        env_norm.reset()
        env_high.reset()

        temps_norm = []
        temps_high = []

        # Run 50 steps with fixed medium cooling (maintain cooling)
        for _ in range(50):
            o_n, _, _, _, _ = env_norm.step(action=2)
            o_h, _, _, _, _ = env_high.step(action=2)
            temps_norm.append(o_n[0])
            temps_high.append(o_h[0])

        assert np.mean(temps_high) > np.mean(temps_norm)
        assert np.mean(env_high.workload_trajectory) > np.mean(env_norm.workload_trajectory)

    def test_high_ambient_degrades_cop(self):
        """Scenario 4 produces higher ambient temperature and lower COP than Scenario 1."""
        env_norm = DataCenterEnv(scenario="normal", split="val")
        env_amb = DataCenterEnv(scenario="high_ambient", split="val")

        env_norm.reset()
        env_amb.reset()

        cops_norm = []
        cops_amb = []

        for _ in range(50):
            _, _, _, _, i_n = env_norm.step(action=2)
            _, _, _, _, i_a = env_amb.step(action=2)
            cops_norm.append(i_n["cop"])
            cops_amb.append(i_a["cop"])

        assert np.mean(cops_amb) < np.mean(cops_norm)
        assert np.mean(env_amb.ambient_trajectory) > np.mean(env_norm.ambient_trajectory)
