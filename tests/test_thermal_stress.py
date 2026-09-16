"""Thermal stress testing and environment robustness validation for CoolRL.

Phase 2D validation suite verifying:
1. Five-scenario stress rollouts
2. Fixed-cooling sweep monotonicity (temperature reduction, energy scaling)
3. Action responsiveness, churn penalty tracking, and boundary clipping
4. Workload responsiveness and thermal forcing
5. Ambient responsiveness, COP degradation, and electrical power impact
6. Extreme combined-stress mitigation under maximum cooling
7. Strict environment determinism under stress
8. Numerical validity (absence of NaN/Inf, physical bounds compliance)
"""

from __future__ import annotations

import dataclasses
import numpy as np
import pytest

from src.environment.datacenter_env import DataCenterEnv
from src.environment.reward import DEFAULT_REWARD_CONFIG, calculate_reward
from src.environment.thermal_model_spec import (
    DEFAULT_COP_PARAMS,
    DEFAULT_THERMAL_PARAMS,
    compute_cooling_power_and_energy,
    compute_cop,
    compute_thermal_step,
)


class TestFiveScenarioStressRollouts:
    """Validate full-episode deterministic rollouts across all five calibrated scenarios."""

    @pytest.mark.parametrize(
        "scenario_name",
        ["normal", "high_workload", "workload_spikes", "high_ambient", "combined_stress"],
    )
    def test_full_scenario_rollout_integrity(self, scenario_name: str):
        """Verify episode completes with exactly 2243 steps, finite metrics, and valid breakdowns."""
        env = DataCenterEnv(scenario=scenario_name, split="all")
        obs, info = env.reset()
        assert info["step_index"] == 0

        steps = 0
        cum_reward = 0.0
        temps = []

        done = False
        while not done:
            obs, reward, term, trunc, info = env.step(action=2)  # Maintain cooling
            done = term or trunc
            steps += 1
            cum_reward += reward
            temps.append(info["internal_temperature"])

            assert np.isfinite(reward)
            assert np.isfinite(info["internal_temperature"])
            assert np.isfinite(info["cooling_energy"])
            assert 0.0 <= info["cooling_level"] <= 1.0

            # Verify reward breakdown strictly matches reward
            b = info["reward_breakdown"]
            expected_sum = (
                b["safety_penalty"]
                + b["severe_overheat_penalty"]
                + b["overcooling_penalty"]
                + b["energy_penalty"]
                + b["action_churn_penalty"]
            )
            assert np.isclose(reward, expected_sum, atol=1e-5)

        assert steps == 2243
        assert len(temps) == 2243

    def test_scenario_stress_hierarchy(self):
        """Verify combined_stress produces higher temperatures and severe penalties than normal."""
        env_normal = DataCenterEnv(scenario="normal", split="all")
        env_stress = DataCenterEnv(scenario="combined_stress", split="all")

        env_normal.reset()
        env_stress.reset()

        cum_r_normal = 0.0
        cum_r_stress = 0.0
        max_t_normal = -np.inf
        max_t_stress = -np.inf

        done = False
        while not done:
            _, r_n, term_n, _, info_n = env_normal.step(2)
            _, r_s, term_s, _, info_s = env_stress.step(2)
            cum_r_normal += r_n
            cum_r_stress += r_s
            max_t_normal = max(max_t_normal, info_n["internal_temperature"])
            max_t_stress = max(max_t_stress, info_s["internal_temperature"])
            done = term_n

        assert max_t_stress > max_t_normal
        assert cum_r_normal > cum_r_stress  # Stress receives substantially worse reward


class TestFixedCoolingSweepMonotonicity:
    """Validate physical response across fixed cooling levels [0.00, 0.25, 0.50, 0.75, 1.00]."""

    @pytest.mark.parametrize("scenario_name", ["normal", "combined_stress"])
    def test_temperature_decreases_with_increasing_cooling(self, scenario_name: str):
        """Higher cooling effort must monotonically decrease mean, min, and max temperatures."""
        cooling_levels = [0.00, 0.25, 0.50, 0.75, 1.00]
        mean_temps = []
        max_temps = []
        total_energies = []

        for c_lvl in cooling_levels:
            params = dataclasses.replace(DEFAULT_THERMAL_PARAMS, initial_cooling_level=c_lvl)
            # Use validation split (288 steps / 24 hrs) for rapid test execution
            env = DataCenterEnv(scenario=scenario_name, split="val", thermal_params=params)
            env.reset()

            t_list = []
            e_sum = 0.0
            done = False
            while not done:
                _, _, term, trunc, info = env.step(action=2)
                done = term or trunc
                t_list.append(info["internal_temperature"])
                e_sum += info["cooling_energy"]

            mean_temps.append(float(np.mean(t_list)))
            max_temps.append(float(np.max(t_list)))
            total_energies.append(e_sum)

        # 1. Monotonic decrease in mean temperature
        for i in range(len(mean_temps) - 1):
            assert mean_temps[i] > mean_temps[i + 1], (
                f"Mean temp failed monotonicity: C={cooling_levels[i]} ({mean_temps[i]:.2f}°C) "
                f"<= C={cooling_levels[i+1]} ({mean_temps[i+1]:.2f}°C)"
            )

        # 2. Monotonic decrease in maximum temperature
        for i in range(len(max_temps) - 1):
            assert max_temps[i] > max_temps[i + 1], (
                f"Max temp failed monotonicity: C={cooling_levels[i]} ({max_temps[i]:.2f}°C) "
                f"<= C={cooling_levels[i+1]} ({max_temps[i+1]:.2f}°C)"
            )

        # 3. Monotonic increase in cooling energy
        for i in range(len(total_energies) - 1):
            assert total_energies[i] < total_energies[i + 1], (
                f"Energy failed monotonicity: C={cooling_levels[i]} ({total_energies[i]:.4f}) "
                f">= C={cooling_levels[i+1]} ({total_energies[i+1]:.4f})"
            )


class TestActionResponsivenessAndClipping:
    """Validate discrete control actions, exact deltas, churn penalties, and boundary clipping."""

    def test_action_adjustments_from_midpoint(self):
        """From C=0.50, actions 0..4 produce exact deltas [-0.20, -0.10, 0.00, +0.10, +0.20]."""
        expected_deltas = [-0.20, -0.10, 0.00, +0.10, +0.20]
        expected_coolings = [0.30, 0.40, 0.50, 0.60, 0.70]
        expected_churns = [-0.10, -0.05, 0.00, -0.05, -0.10]

        for action_idx in range(5):
            env = DataCenterEnv(scenario="normal")
            env.reset()

            _, reward, _, _, info = env.step(action=action_idx)

            assert np.isclose(info["cooling_level"], expected_coolings[action_idx])
            assert np.isclose(info["action_churn_penalty"], expected_churns[action_idx])
            assert info["reward_breakdown"]["action_churn_penalty"] == expected_churns[action_idx]

    def test_lower_boundary_clipping(self):
        """Action 0 (-0.20) from near-zero cooling (C=0.05) clips strictly to 0.00."""
        params = dataclasses.replace(DEFAULT_THERMAL_PARAMS, initial_cooling_level=0.05)
        env = DataCenterEnv(scenario="normal", thermal_params=params)
        env.reset()

        _, _, _, _, info = env.step(action=0)  # -0.20 requested
        assert np.isclose(info["cooling_level"], 0.00)
        assert 0.0 <= info["cooling_level"] <= 1.0

        # Further negative actions remain clamped at 0.00
        _, _, _, _, info2 = env.step(action=0)
        assert np.isclose(info2["cooling_level"], 0.00)

    def test_upper_boundary_clipping(self):
        """Action 4 (+0.20) from near-maximum cooling (C=0.95) clips strictly to 1.00."""
        params = dataclasses.replace(DEFAULT_THERMAL_PARAMS, initial_cooling_level=0.95)
        env = DataCenterEnv(scenario="normal", thermal_params=params)
        env.reset()

        _, _, _, _, info = env.step(action=4)  # +0.20 requested
        assert np.isclose(info["cooling_level"], 1.00)
        assert 0.0 <= info["cooling_level"] <= 1.0

        # Further positive actions remain clamped at 1.00
        _, _, _, _, info2 = env.step(action=4)
        assert np.isclose(info2["cooling_level"], 1.00)


class TestWorkloadResponsiveness:
    """Validate thermal response to low, normal, and high computational workloads."""

    def test_workload_thermal_forcing(self):
        """Higher workload under identical state produces higher temperature and heating rate."""
        t_int = 24.0
        t_amb = 28.0
        c_lvl = 0.50

        workloads = [0.10, 0.40, 0.85]
        t_nexts = []
        delta_ts = []

        for w in workloads:
            t_next, dt = compute_thermal_step(
                t_int, w, t_amb, c_lvl, DEFAULT_THERMAL_PARAMS, disturbance=0.0
            )
            t_nexts.append(t_next)
            delta_ts.append(dt)

        # Monotonically increasing post-step temperature and temperature increment
        assert t_nexts[0] < t_nexts[1] < t_nexts[2]
        assert delta_ts[0] < delta_ts[1] < delta_ts[2]

    def test_high_workload_scenario_increases_thermal_penalties(self):
        """High workload scenario generates greater safety penalties than normal scenario."""
        env_norm = DataCenterEnv(scenario="normal", split="val")
        env_high = DataCenterEnv(scenario="high_workload", split="val")

        env_norm.reset()
        env_high.reset()

        pen_norm = 0.0
        pen_high = 0.0

        done = False
        while not done:
            _, _, term_n, _, info_n = env_norm.step(2)
            _, _, term_h, _, info_h = env_high.step(2)
            pen_norm += info_n["safety_penalty"] + info_n["severe_overheat_penalty"]
            pen_high += info_h["safety_penalty"] + info_h["severe_overheat_penalty"]
            done = term_n

        # More negative penalty under high workload
        assert pen_high < pen_norm


class TestAmbientResponsivenessAndCOP:
    """Validate behavior under representative ambient temperatures [20°C, 25°C, 30°C, 35°C]."""

    def test_cop_and_cooling_energy_response(self):
        """COP must decrease and cooling electrical energy must increase with ambient temperature."""
        amb_temps = [20.0, 25.0, 30.0, 35.0]
        cops = []
        energies = []

        for t_amb in amb_temps:
            cop = compute_cop(t_amb, DEFAULT_THERMAL_PARAMS, DEFAULT_COP_PARAMS)
            cops.append(cop)
            _, _, e_cool = compute_cooling_power_and_energy(
                cooling_level=0.50,
                cop=cop,
                cop_params=DEFAULT_COP_PARAMS,
                dt_hours=DEFAULT_THERMAL_PARAMS.timestep_hours,
            )
            energies.append(e_cool)

        # COP monotonically decreases
        assert cops[0] > cops[1] > cops[2] > cops[3]
        # Electrical cooling energy monotonically increases
        assert energies[0] < energies[1] < energies[2] < energies[3]
        # Numerical bounds check
        for cop in cops:
            assert DEFAULT_COP_PARAMS.cop_min <= cop <= DEFAULT_COP_PARAMS.cop_max


class TestExtremeCombinedStressMitigation:
    """Validate combined stress behavior under minimum vs maximum cooling effort."""

    def test_maximum_cooling_materially_mitigates_combined_stress(self):
        """C=1.00 must substantially reduce peak temperatures and eliminate severe overheating."""
        # 1. Zero cooling (C=0.00)
        params_zero = dataclasses.replace(DEFAULT_THERMAL_PARAMS, initial_cooling_level=0.00)
        env_zero = DataCenterEnv(scenario="combined_stress", split="val", thermal_params=params_zero)
        env_zero.reset()

        temps_zero = []
        steps_above_32_zero = 0
        cum_r_zero = 0.0

        done = False
        while not done:
            _, r, term, _, info = env_zero.step(2)
            done = term
            temps_zero.append(info["internal_temperature"])
            if info["internal_temperature"] > 32.0:
                steps_above_32_zero += 1
            cum_r_zero += r

        # 2. Maximum cooling (C=1.00)
        params_max = dataclasses.replace(DEFAULT_THERMAL_PARAMS, initial_cooling_level=1.00)
        env_max = DataCenterEnv(scenario="combined_stress", split="val", thermal_params=params_max)
        env_max.reset()

        temps_max = []
        steps_above_32_max = 0
        cum_r_max = 0.0

        done = False
        while not done:
            _, r, term, _, info = env_max.step(2)
            done = term
            temps_max.append(info["internal_temperature"])
            if info["internal_temperature"] > 32.0:
                steps_above_32_max += 1
            cum_r_max += r

        # Mitigation proofs
        max_t_zero = float(np.max(temps_zero))
        max_t_max = float(np.max(temps_max))

        # Peak temperature is reduced by over 15°C
        assert max_t_zero - max_t_max > 15.0
        # Severe overheating (>32°C) is completely eliminated under maximum cooling
        assert steps_above_32_zero > 200
        assert steps_above_32_max == 0
        # Reward is vastly superior under maximum cooling
        assert cum_r_max > cum_r_zero


class TestStressDeterminism:
    """Verify bit-for-bit repeatability under identical seeds and scenario configuration."""

    def test_identical_rollout_reproducibility(self):
        """Two separate instances running combined_stress must produce bitwise identical states."""
        env1 = DataCenterEnv(scenario="combined_stress", split="val")
        env2 = DataCenterEnv(scenario="combined_stress", split="val")

        obs1, info1 = env1.reset(seed=123)
        obs2, info2 = env2.reset(seed=123)

        np.testing.assert_array_equal(obs1, obs2)
        assert info1 == info2

        actions = [1, 3, 2, 4, 0, 2, 2, 4, 1, 0] * 10
        for action in actions:
            o1, r1, term1, trunc1, i1 = env1.step(action)
            o2, r2, term2, trunc2, i2 = env2.step(action)

            np.testing.assert_array_equal(o1, o2)
            assert r1 == r2
            assert term1 == term2
            assert trunc1 == trunc2
            assert i1 == i2


class TestNumericalSafetyAndIntegrity:
    """Verify absence of NaN, Inf, out-of-bound cooling, negative energy, and malformed rewards."""

    def test_numerical_integrity_across_random_actions(self):
        """100 random actions in combined_stress must produce strictly valid, finite numbers."""
        env = DataCenterEnv(scenario="combined_stress", split="val")
        env.reset(seed=42)

        rng = np.random.default_rng(42)
        for _ in range(100):
            action = int(rng.integers(0, 5))
            obs, reward, terminated, truncated, info = env.step(action)

            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            assert 0.0 <= info["cooling_level"] <= 1.0
            assert info["cooling_energy"] >= 0.0
            assert DEFAULT_COP_PARAMS.cop_min <= info["cop"] <= DEFAULT_COP_PARAMS.cop_max
            assert np.isfinite(info["internal_temperature"])
            assert np.isfinite(info["temperature_change"])
            assert info["ashrae_zone"] in ["below_recommended", "recommended", "above_recommended", "beyond_a1_allowable"]
            assert info["temperature_trend"] in ["cooling", "stable", "warming"]
            if terminated:
                break
