"""Unit tests for Phase 2E RL State/Action Interface and StateDiscretizer.

Validates:
1. Final observation contract (5 features, shape (5,), ordering, finiteness).
2. Final action contract (5 discrete actions, valid IDs 0..4, exact deltas, clipping).
3. State bin boundaries (temperature, workload, ambient, cooling, trend).
4. Full 960-state space enumeration (unique IDs 0..959, no collisions, exact bijection).
5. Input validation (wrong length, non-numeric, NaN, Inf, None).
6. Deterministic encoding and decode reversibility.
7. End-to-end integration with DataCenterEnv across all five scenarios.
8. Clean state/action loop interface.
"""

from __future__ import annotations

import itertools
import numpy as np
import pytest

from src.agents.state_discretizer import (
    DEFAULT_DISCRETIZATION_CONFIG,
    DiscretizationConfig,
    StateDiscretizer,
)
from src.environment.datacenter_env import DataCenterEnv


class TestObservationContract:
    """Verify DataCenterEnv observation contract meets all Phase 2E requirements."""

    def test_observation_dimensions_and_types(self):
        """Observation must have exactly 5 float elements with fixed ordering."""
        env = DataCenterEnv(scenario="normal")
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))

        # Check feature ordering matches specification: [T_int, W, T_amb, C, delta_T]
        assert np.isclose(obs[0], info["internal_temperature"])
        assert np.isclose(obs[1], info["workload"])
        assert np.isclose(obs[2], info["ambient_temperature"])
        assert np.isclose(obs[3], info["cooling_level"])
        assert np.isclose(obs[4], info["temperature_change"])

    def test_observation_consistency_across_reset_and_step(self):
        """Observation shape and dtype must remain invariant across reset and subsequent steps."""
        env = DataCenterEnv(scenario="normal")
        obs_reset, _ = env.reset()

        assert obs_reset.shape == (5,)
        assert obs_reset.dtype == np.float32

        for action in range(5):
            obs_step, _, _, _, _ = env.step(action)
            assert obs_step.shape == (5,)
            assert obs_step.dtype == np.float32
            assert np.all(np.isfinite(obs_step))


class TestActionContract:
    """Verify discrete action space and mapping to physical cooling deltas."""

    def test_action_space_specification(self):
        """Action space must have exactly 5 discrete actions with IDs 0..4."""
        discretizer = StateDiscretizer()
        assert discretizer.NUM_ACTIONS == 5
        assert set(discretizer.ACTION_MAP.keys()) == {0, 1, 2, 3, 4}
        assert discretizer.ACTION_MAP[0] == -0.20
        assert discretizer.ACTION_MAP[1] == -0.10
        assert discretizer.ACTION_MAP[2] == 0.00
        assert discretizer.ACTION_MAP[3] == +0.10
        assert discretizer.ACTION_MAP[4] == +0.20

    def test_invalid_action_rejection(self):
        """DataCenterEnv must reject actions outside {0, 1, 2, 3, 4}."""
        env = DataCenterEnv(scenario="normal")
        env.reset()

        for invalid_act in [-1, 5, 10, 2.5, "0", None]:
            with pytest.raises(ValueError):
                env.step(invalid_act)  # type: ignore

    def test_maintain_action_produces_zero_delta(self):
        """Action 2 (maintain) must leave cooling level unchanged."""
        env = DataCenterEnv(scenario="normal")
        env.reset()
        initial_c = env.cooling_level

        _, _, _, _, info = env.step(action=2)
        assert info["cooling_level"] == initial_c
        assert info["action_churn_penalty"] == 0.0


class TestStateBinBoundaries:
    """Validate boundary thresholds for all 5 state dimensions as specified in Phase 2E."""

    @pytest.fixture
    def discretizer(self) -> StateDiscretizer:
        return StateDiscretizer()

    def test_temperature_boundaries(self, discretizer: StateDiscretizer):
        """Test internal temperature bins: <18 (0), [18, 21) (1), [21, 24) (2), [24, 27) (3), >=27 (4)."""
        cases = [
            (17.99, 0),
            (18.00, 1),
            (20.99, 1),
            (21.00, 2),
            (23.99, 2),
            (24.00, 3),
            (26.99, 3),
            (27.00, 4),
            (32.00, 4),
            (45.00, 4),
        ]
        for val, expected_bin in cases:
            assert discretizer.discretize_temperature(val) == expected_bin, f"Temp {val} failed"

    def test_workload_boundaries(self, discretizer: StateDiscretizer):
        """Test workload bins: <0.33 (0), [0.33, 0.45) (1), [0.45, 0.55) (2), >=0.55 (3)."""
        cases = [
            (0.3299, 0),
            (0.3300, 1),
            (0.4499, 1),
            (0.4500, 2),
            (0.5499, 2),
            (0.5500, 3),
            (0.8500, 3),
            (1.0000, 3),
        ]
        for val, expected_bin in cases:
            assert discretizer.discretize_workload(val) == expected_bin, f"Workload {val} failed"

    def test_ambient_boundaries(self, discretizer: StateDiscretizer):
        """Test ambient temperature bins: <23 (0), [23, 27) (1), [27, 31) (2), >=31 (3)."""
        cases = [
            (22.99, 0),
            (23.00, 1),
            (26.99, 1),
            (27.00, 2),
            (30.99, 2),
            (31.00, 3),
            (35.00, 3),
        ]
        for val, expected_bin in cases:
            assert discretizer.discretize_ambient(val) == expected_bin, f"Ambient {val} failed"

    def test_cooling_boundaries(self, discretizer: StateDiscretizer):
        """Test cooling bins: <0.25 (0), [0.25, 0.50) (1), [0.50, 0.75) (2), >=0.75 (3)."""
        cases = [
            (0.2499, 0),
            (0.2500, 1),
            (0.4999, 1),
            (0.5000, 2),
            (0.7499, 2),
            (0.7500, 3),
            (1.0000, 3),
        ]
        for val, expected_bin in cases:
            assert discretizer.discretize_cooling(val) == expected_bin, f"Cooling {val} failed"

    def test_trend_boundaries(self, discretizer: StateDiscretizer):
        """Test trend bins: <-0.20 (0), [-0.20, +0.20] (1), >+0.20 (2)."""
        cases = [
            (-0.2001, 0),
            (-0.2000, 1),
            (0.0000, 1),
            (0.2000, 1),
            (0.2001, 2),
            (1.5000, 2),
        ]
        for val, expected_bin in cases:
            assert discretizer.discretize_trend(val) == expected_bin, f"Trend {val} failed"


class TestFullStateSpaceValidation:
    """Enumerate all 960 state combinations and verify uniqueness, coverage, and bijection."""

    def test_full_960_state_enumeration_and_bijection(self):
        """Verify exactly 960 combinations, unique IDs [0..959], and decode(encode(c)) == c."""
        discretizer = StateDiscretizer()
        assert discretizer.NUM_STATES == 960

        combinations = list(
            itertools.product(range(5), range(4), range(4), range(4), range(3))
        )
        assert len(combinations) == 960

        state_ids = []
        for c in combinations:
            sid = discretizer.encode_bins(*c)
            state_ids.append(sid)

            # Reversibility / bijection check
            decoded = discretizer.decode(sid)
            assert decoded == c, f"Decoding failed for tuple {c} (got {decoded})"

        # Verify uniqueness
        assert len(set(state_ids)) == 960
        assert min(state_ids) == 0
        assert max(state_ids) == 959
        assert sorted(state_ids) == list(range(960))

    def test_decode_bounds_and_type_checking(self):
        """Decode must reject out-of-range or non-integer state IDs."""
        discretizer = StateDiscretizer()

        with pytest.raises(ValueError):
            discretizer.decode(-1)
        with pytest.raises(ValueError):
            discretizer.decode(960)
        with pytest.raises(TypeError):
            discretizer.decode("42")  # type: ignore

    def test_get_bin_descriptions(self):
        """Verify human-readable semantic explanation of state ID."""
        discretizer = StateDiscretizer()
        desc = discretizer.get_bin_descriptions(0)
        assert desc["state_id"] == 0
        assert desc["bins"] == (0, 0, 0, 0, 0)
        assert "T < 18°C" in desc["internal_temperature"]
        assert "W < 0.33" in desc["workload"]


class TestDataValidation:
    """Verify strict input validation rejecting malformed observations."""

    @pytest.fixture
    def discretizer(self) -> StateDiscretizer:
        return StateDiscretizer()

    def test_reject_wrong_feature_count(self, discretizer: StateDiscretizer):
        """Observations with != 5 elements must raise ValueError."""
        with pytest.raises(ValueError, match="Expected observation of shape"):
            discretizer.encode([24.0, 0.4, 25.0, 0.5])  # 4 features
        with pytest.raises(ValueError, match="Expected observation of shape"):
            discretizer.encode([24.0, 0.4, 25.0, 0.5, 0.0, 1.0])  # 6 features

    def test_reject_none(self, discretizer: StateDiscretizer):
        """None observation must raise ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            discretizer.encode(None)

    def test_reject_non_numeric(self, discretizer: StateDiscretizer):
        """Non-numeric elements must raise TypeError."""
        with pytest.raises(TypeError, match="must be numeric"):
            discretizer.encode(["invalid", 0.4, 25.0, 0.5, 0.0])

    def test_reject_nan_and_inf(self, discretizer: StateDiscretizer):
        """Observations containing NaN or Inf must raise ValueError."""
        with pytest.raises(ValueError, match="NaN or infinite"):
            discretizer.encode([np.nan, 0.4, 25.0, 0.5, 0.0])
        with pytest.raises(ValueError, match="NaN or infinite"):
            discretizer.encode([24.0, np.inf, 25.0, 0.5, 0.0])
        with pytest.raises(ValueError, match="NaN or infinite"):
            discretizer.encode([24.0, 0.4, -np.inf, 0.5, 0.0])


class TestEnvironmentDiscretizerIntegration:
    """Verify integration between DataCenterEnv outputs and StateDiscretizer."""

    @pytest.mark.parametrize(
        "scenario_name",
        ["normal", "high_workload", "workload_spikes", "high_ambient", "combined_stress"],
    )
    def test_all_scenarios_encode_valid_state_ids(self, scenario_name: str):
        """Observations generated across all 5 scenarios must encode to valid state IDs in [0, 959]."""
        discretizer = StateDiscretizer()
        env = DataCenterEnv(scenario=scenario_name, split="val")
        obs, info = env.reset(seed=42)

        # Initial observation check
        state_id = discretizer.encode(obs)
        assert 0 <= state_id < 960

        # Step through representative sequence
        actions = [1, 3, 2, 4, 0, 2, 2, 4, 1, 0] * 10
        for action in actions:
            obs, reward, term, trunc, info = env.step(action)
            sid = discretizer.encode(obs)
            assert 0 <= sid < 960
            assert isinstance(sid, int)

            # Verify decode matches bins
            b0, b1, b2, b3, b4 = discretizer.decode(sid)
            assert b0 == discretizer.discretize_temperature(obs[0])
            assert b1 == discretizer.discretize_workload(obs[1])
            assert b2 == discretizer.discretize_ambient(obs[2])
            assert b3 == discretizer.discretize_cooling(obs[3])
            assert b4 == discretizer.discretize_trend(obs[4])

            if term or trunc:
                break

    def test_canonical_rl_interaction_loop(self):
        """Simulate the exact canonical tabular Q-learning step loop."""
        discretizer = StateDiscretizer()
        env = DataCenterEnv(scenario="normal", split="val")

        obs, info = env.reset()
        state_id = discretizer.encode(obs)

        # Mock agent action selection
        action = 2
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state_id = discretizer.encode(next_obs)

        assert 0 <= state_id < 960
        assert 0 <= next_state_id < 960
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
