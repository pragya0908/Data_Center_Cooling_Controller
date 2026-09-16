"""Unit tests for the CoolRL multi-objective reward function.

Tests verify thermal safety priority, piecewise nonlinear penalties, energy monotonicity,
overcooling avoidance, action churn penalties, numerical stability, determinism,
breakdown consistency, and environment integration.
"""

import pytest
import numpy as np

from src.environment.reward import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    calculate_reward,
    compute_action_churn_penalty,
    compute_energy_penalty,
    compute_overcooling_penalty,
    compute_thermal_safety_penalties,
)
from src.environment.datacenter_env import DataCenterEnv


class TestThermalSafetyPenalties:
    """Tests verifying the temperature safety term across ASHRAE boundaries."""

    def test_safe_temperature_recommended_range(self):
        """1. Temperatures inside recommended range (18°C-27°C) have zero safety penalty."""
        for t in [18.0, 20.0, 24.0, 25.5, 27.0]:
            safety_pen, severe_pen = compute_thermal_safety_penalties(t, DEFAULT_REWARD_CONFIG)
            assert safety_pen == 0.0
            assert severe_pen == 0.0

    def test_above_recommended_range(self):
        """3. Temperatures between 27°C and 32°C incur linear safety penalty but zero severe penalty."""
        safety_pen_28, severe_pen_28 = compute_thermal_safety_penalties(28.0, DEFAULT_REWARD_CONFIG)
        safety_pen_30, severe_pen_30 = compute_thermal_safety_penalties(30.0, DEFAULT_REWARD_CONFIG)
        safety_pen_32, severe_pen_32 = compute_thermal_safety_penalties(32.0, DEFAULT_REWARD_CONFIG)

        # Linear with slope safety_weight = 2.0
        assert np.isclose(safety_pen_28, -2.0 * 1.0)
        assert np.isclose(safety_pen_30, -2.0 * 3.0)
        assert np.isclose(safety_pen_32, -2.0 * 5.0)

        assert severe_pen_28 == 0.0
        assert severe_pen_30 == 0.0
        assert severe_pen_32 == 0.0

    def test_beyond_allowable_temperature(self):
        """4. Temperatures above 32°C incur both linear safety and quadratic severe penalties."""
        safety_pen_33, severe_pen_33 = compute_thermal_safety_penalties(33.0, DEFAULT_REWARD_CONFIG)
        assert np.isclose(safety_pen_33, -2.0 * (33.0 - 27.0))  # -12.0
        assert np.isclose(severe_pen_33, -5.0 * ((33.0 - 32.0) ** 2))  # -5.0

    def test_severe_overheating_exponential_scaling(self):
        """5. Severe overheating at 36°C is heavily penalized."""
        safety_pen_36, severe_pen_36 = compute_thermal_safety_penalties(36.0, DEFAULT_REWARD_CONFIG)
        assert np.isclose(safety_pen_36, -2.0 * (36.0 - 27.0))  # -18.0
        assert np.isclose(severe_pen_36, -5.0 * ((36.0 - 32.0) ** 2))  # -80.0
        assert np.isclose(safety_pen_36 + severe_pen_36, -98.0)


class TestOvercoolingAndActionChurn:
    """Tests verifying overcooling and action churn penalties."""

    def test_undercooling_penalty_below_18c(self):
        """2. Temperatures below 18°C incur overcooling penalty proportional to deficit."""
        overcool_17 = compute_overcooling_penalty(17.0, DEFAULT_REWARD_CONFIG)
        overcool_15 = compute_overcooling_penalty(15.0, DEFAULT_REWARD_CONFIG)
        overcool_18 = compute_overcooling_penalty(18.0, DEFAULT_REWARD_CONFIG)
        overcool_24 = compute_overcooling_penalty(24.0, DEFAULT_REWARD_CONFIG)

        assert np.isclose(overcool_17, -1.0 * (18.0 - 17.0))
        assert np.isclose(overcool_15, -1.0 * (18.0 - 15.0))
        assert overcool_18 == 0.0
        assert overcool_24 == 0.0

    def test_zero_churn_for_maintain_cooling(self):
        """8. Maintaining cooling (action_delta = 0.0) incurs zero churn penalty."""
        churn_pen = compute_action_churn_penalty(0.0, DEFAULT_REWARD_CONFIG)
        assert churn_pen == 0.0

    def test_action_churn_monotonicity(self):
        """7. Larger action changes receive strictly larger churn penalties."""
        churn_00 = compute_action_churn_penalty(0.00, DEFAULT_REWARD_CONFIG)
        churn_10 = compute_action_churn_penalty(0.10, DEFAULT_REWARD_CONFIG)
        churn_neg10 = compute_action_churn_penalty(-0.10, DEFAULT_REWARD_CONFIG)
        churn_20 = compute_action_churn_penalty(0.20, DEFAULT_REWARD_CONFIG)
        churn_neg20 = compute_action_churn_penalty(-0.20, DEFAULT_REWARD_CONFIG)

        assert churn_00 == 0.0
        assert churn_10 == churn_neg10 == -0.05
        assert churn_20 == churn_neg20 == -0.10
        assert churn_20 < churn_10 < churn_00


class TestEnergyMonotonicity:
    """Tests verifying energy penalty behavior."""

    def test_energy_monotonicity(self):
        """6. Energy penalty increases monotonically (becomes more negative) with cooling energy."""
        energies = [0.0, 0.005, 0.015, 0.030, 0.050]
        penalties = [compute_energy_penalty(e, DEFAULT_REWARD_CONFIG) for e in energies]

        for i in range(len(penalties) - 1):
            assert penalties[i] > penalties[i + 1]
            assert penalties[i] <= 0.0


class TestThermalSafetyPriority:
    """Tests verifying that thermal safety strictly dominates energy savings."""

    def test_thermal_safety_priority_over_energy_savings(self):
        """9. Reward at safe temperature + high energy is strictly better than severe overheating + low energy."""
        # Safe state at 24°C with maximum cooling energy (~0.032)
        safe_high_energy, _ = calculate_reward(
            internal_temperature=24.0,
            cooling_energy=0.032,
            action_delta=0.0,
            config=DEFAULT_REWARD_CONFIG,
        )

        # Severe overheating state at 36°C with zero cooling energy (0.0)
        overheat_zero_energy, _ = calculate_reward(
            internal_temperature=36.0,
            cooling_energy=0.000,
            action_delta=0.0,
            config=DEFAULT_REWARD_CONFIG,
        )

        assert safe_high_energy > overheat_zero_energy
        # Safe high energy: ~ -0.32; Overheat zero energy: -98.0
        assert safe_high_energy - overheat_zero_energy > 90.0

    def test_temperature_penalty_monotonicity(self):
        """Verify: 33°C is penalized more than 28°C; 36°C is penalized more than 33°C."""
        r_28, _ = calculate_reward(28.0, 0.015, 0.0, DEFAULT_REWARD_CONFIG)
        r_33, _ = calculate_reward(33.0, 0.015, 0.0, DEFAULT_REWARD_CONFIG)
        r_36, _ = calculate_reward(36.0, 0.015, 0.0, DEFAULT_REWARD_CONFIG)

        assert r_28 > r_33 > r_36


class TestNumericalValidationStates:
    """Verifies all 11 required numerical evaluation states from Phase 2C specification."""

    @pytest.mark.parametrize(
        "t_int, energy, delta_c, exp_safety, exp_severe, exp_overcool, exp_energy, exp_churn",
        [
            (24.0, 0.005, 0.0, 0.00, 0.00, 0.00, -0.050, 0.00),   # 1. 24°C low cool
            (24.0, 0.030, 0.0, 0.00, 0.00, 0.00, -0.300, 0.00),   # 2. 24°C high cool
            (17.0, 0.015, 0.0, 0.00, 0.00, -1.00, -0.150, 0.00),  # 3. 17°C mod cool
            (20.0, 0.015, 0.0, 0.00, 0.00, 0.00, -0.150, 0.00),   # 4. 20°C mod cool
            (26.0, 0.015, 0.0, 0.00, 0.00, 0.00, -0.150, 0.00),   # 5. 26°C mod cool
            (28.0, 0.015, 0.0, -2.00, 0.00, 0.00, -0.150, 0.00),  # 6. 28°C mod cool
            (31.0, 0.015, 0.0, -8.00, 0.00, 0.00, -0.150, 0.00),  # 7. 31°C mod cool
            (32.0, 0.015, 0.0, -10.00, 0.00, 0.00, -0.150, 0.00), # 8. 32°C mod cool
            (33.0, 0.015, 0.0, -12.00, -5.00, 0.00, -0.150, 0.0), # 9. 33°C mod cool
            (36.0, 0.032, 0.0, -18.00, -80.00, 0.00, -0.320, 0.0),# 10. 36°C high cool
            (36.0, 0.005, 0.0, -18.00, -80.00, 0.00, -0.050, 0.0),# 11. 36°C low cool
        ],
    )
    def test_numerical_validation_matrix(
        self, t_int, energy, delta_c, exp_safety, exp_severe, exp_overcool, exp_energy, exp_churn
    ):
        tot, breakdown = calculate_reward(t_int, energy, delta_c, DEFAULT_REWARD_CONFIG)
        exp_total = exp_safety + exp_severe + exp_overcool + exp_energy + exp_churn

        assert np.isclose(breakdown["safety_penalty"], exp_safety, atol=1e-3)
        assert np.isclose(breakdown["severe_overheat_penalty"], exp_severe, atol=1e-3)
        assert np.isclose(breakdown["overcooling_penalty"], exp_overcool, atol=1e-3)
        assert np.isclose(breakdown["energy_penalty"], exp_energy, atol=1e-3)
        assert np.isclose(breakdown["action_churn_penalty"], exp_churn, atol=1e-3)
        assert np.isclose(tot, exp_total, atol=1e-3)


class TestRewardIntegrityAndEnvironmentIntegration:
    """Tests verifying breakdown additivity, determinism, and DataCenterEnv integration."""

    def test_reward_breakdown_additivity(self):
        """13. Total reward strictly equals the sum of its 5 constituent penalties."""
        for t in [15.0, 22.0, 29.0, 34.0]:
            for e in [0.005, 0.025]:
                for da in [-0.20, 0.0, +0.10]:
                    tot, b = calculate_reward(t, e, da, DEFAULT_REWARD_CONFIG)
                    expected_sum = (
                        b["safety_penalty"]
                        + b["severe_overheat_penalty"]
                        + b["overcooling_penalty"]
                        + b["energy_penalty"]
                        + b["action_churn_penalty"]
                    )
                    assert np.isclose(tot, expected_sum)
                    assert np.isfinite(tot)

    def test_reward_determinism(self):
        """11. The reward calculation is 100% deterministic."""
        r1, b1 = calculate_reward(28.5, 0.018, 0.10, DEFAULT_REWARD_CONFIG)
        r2, b2 = calculate_reward(28.5, 0.018, 0.10, DEFAULT_REWARD_CONFIG)
        assert r1 == r2
        assert b1 == b2

    def test_datacenter_env_reward_integration(self):
        """12. DataCenterEnv returns step reward and info breakdown matching calculate_reward."""
        env = DataCenterEnv(scenario="normal")
        env.reset()

        obs, reward, terminated, truncated, info = env.step(action=4)  # +0.20 action

        # Verify reward is a finite float
        assert isinstance(reward, float)
        assert np.isfinite(reward)

        # Verify reward matches calculate_reward on the resulting transition state
        expected_r, expected_b = calculate_reward(
            internal_temperature=info["internal_temperature"],
            cooling_energy=info["cooling_energy"],
            action_delta=+0.20,
            config=env.reward_config,
        )
        assert np.isclose(reward, expected_r)
        assert np.isclose(info["reward"], expected_r)
        assert info["reward_breakdown"] == expected_b
