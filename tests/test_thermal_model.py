"""Unit tests for the CoolRL thermal model mathematical specification.

Tests verify first-principles physical properties, monotonicity, boundary behavior,
COP thermodynamic curves, numerical stability, and sanity test cases A through F.
"""

import pytest
import numpy as np

from src.environment.thermal_model_spec import (
    DEFAULT_COP_PARAMS,
    DEFAULT_THERMAL_PARAMS,
    AshraeZone,
    COPParameters,
    TemperatureTrend,
    ThermalModelParameters,
    classify_ashrae_zone,
    classify_temperature_trend,
    compute_ambient_derating_factor,
    compute_cooling_effect,
    compute_cooling_power_and_energy,
    compute_cop,
    compute_thermal_step,
)


class TestThermalPhysicsMonotonicity:
    """Tests verifying monotonic and physical directionality of thermal terms."""

    def test_higher_workload_produces_greater_heating(self):
        """1. Higher workload produces greater heating contribution."""
        t_int = 24.0
        t_amb = 25.0
        c = 0.50
        t_next_low, dt_low = compute_thermal_step(t_int, 0.20, t_amb, c, DEFAULT_THERMAL_PARAMS)
        t_next_high, dt_high = compute_thermal_step(t_int, 0.80, t_amb, c, DEFAULT_THERMAL_PARAMS)

        assert t_next_high > t_next_low
        assert dt_high > dt_low
        # Theoretical delta difference should be alpha * (0.80 - 0.20) = 1.00 * 0.60 = 0.60
        assert np.isclose(dt_high - dt_low, 0.60)

    def test_higher_ambient_increases_ambient_thermal_pressure(self):
        """2. Higher ambient temperature increases ambient thermal pressure."""
        t_int = 24.0
        w = 0.40
        c = 0.50
        t_next_cool, dt_cool = compute_thermal_step(t_int, w, 22.0, c, DEFAULT_THERMAL_PARAMS)
        t_next_hot, dt_hot = compute_thermal_step(t_int, w, 34.0, c, DEFAULT_THERMAL_PARAMS)

        assert t_next_hot > t_next_cool
        assert dt_hot > dt_cool

    def test_increasing_cooling_reduces_internal_temperature(self):
        """3. Increasing cooling reduces internal temperature relative to identical conditions."""
        t_int = 24.0
        w = 0.50
        t_amb = 28.0

        t_next_low_cool, dt_low_cool = compute_thermal_step(t_int, w, t_amb, 0.20, DEFAULT_THERMAL_PARAMS)
        t_next_high_cool, dt_high_cool = compute_thermal_step(t_int, w, t_amb, 0.80, DEFAULT_THERMAL_PARAMS)

        assert t_next_high_cool < t_next_low_cool
        assert dt_high_cool < dt_low_cool

    def test_ambient_derating_increases_with_higher_ambient(self):
        """4. Ambient derating factor increases with higher ambient temperature."""
        f_mild = compute_ambient_derating_factor(25.0, DEFAULT_THERMAL_PARAMS)
        f_warm = compute_ambient_derating_factor(30.0, DEFAULT_THERMAL_PARAMS)
        f_hot = compute_ambient_derating_factor(35.0, DEFAULT_THERMAL_PARAMS)

        assert f_mild < f_warm < f_hot
        assert np.isclose(f_mild, 1.000)
        assert np.isclose(f_hot, 1.250)

    def test_cop_decreases_with_increasing_ambient(self):
        """5. COP decreases with increasing ambient temperature within unclipped bounds."""
        cop_cool = compute_cop(20.0, DEFAULT_THERMAL_PARAMS, DEFAULT_COP_PARAMS)
        cop_ref = compute_cop(25.0, DEFAULT_THERMAL_PARAMS, DEFAULT_COP_PARAMS)
        cop_hot = compute_cop(35.0, DEFAULT_THERMAL_PARAMS, DEFAULT_COP_PARAMS)

        assert cop_cool > cop_ref > cop_hot
        assert np.isclose(cop_cool, 3.90)
        assert np.isclose(cop_ref, 3.50)
        assert np.isclose(cop_hot, 2.70)


class TestEnergyAndBoundaryBehavior:
    """Tests verifying energy calculations, boundary limits, and zero/max extractions."""

    def test_cooling_energy_is_non_negative(self):
        """6. Cooling energy is non-negative across all ambient and cooling levels."""
        for c in [0.0, 0.25, 0.50, 0.75, 1.0]:
            for t_amb in [15.0, 25.0, 35.0, 45.0]:
                cop = compute_cop(t_amb, DEFAULT_THERMAL_PARAMS, DEFAULT_COP_PARAMS)
                q, p, e = compute_cooling_power_and_energy(
                    c, cop, DEFAULT_COP_PARAMS, DEFAULT_THERMAL_PARAMS.timestep_hours
                )
                assert q >= 0.0
                assert p >= 0.0
                assert e >= 0.0

    def test_cooling_energy_increases_with_cooling_level(self):
        """7. Cooling energy increases when cooling level increases under same ambient condition."""
        t_amb = 28.0
        cop = compute_cop(t_amb, DEFAULT_THERMAL_PARAMS, DEFAULT_COP_PARAMS)
        _, _, e_low = compute_cooling_power_and_energy(0.30, cop, DEFAULT_COP_PARAMS, DEFAULT_THERMAL_PARAMS.timestep_hours)
        _, _, e_high = compute_cooling_power_and_energy(0.70, cop, DEFAULT_COP_PARAMS, DEFAULT_THERMAL_PARAMS.timestep_hours)

        assert e_high > e_low

    def test_zero_cooling_produces_no_extraction(self):
        """10. Zero cooling produces zero cooling extraction and zero energy consumption."""
        cool_eff = compute_cooling_effect(0.0, 30.0, DEFAULT_THERMAL_PARAMS)
        cop = compute_cop(30.0, DEFAULT_THERMAL_PARAMS, DEFAULT_COP_PARAMS)
        q, p, e = compute_cooling_power_and_energy(0.0, cop, DEFAULT_COP_PARAMS, DEFAULT_THERMAL_PARAMS.timestep_hours)

        assert cool_eff == 0.0
        assert q == 0.0
        assert p == 0.0
        assert e == 0.0

    def test_maximum_cooling_produces_maximum_normalized_extraction(self):
        """11. Maximum cooling produces maximum normalized extraction."""
        # At T_ref = 25°C, F_amb = 1.0, so cool_eff = gamma * 1.0 / 1.0 = gamma
        cool_eff_max = compute_cooling_effect(1.0, 25.0, DEFAULT_THERMAL_PARAMS)
        assert np.isclose(cool_eff_max, DEFAULT_THERMAL_PARAMS.gamma)


class TestNumericalSanityCasesAThroughF:
    """Verifies the exact on-paper sanity test cases calculated in Phase 2A."""

    def test_case_a_normal_baseline(self):
        """Case A: Normal load (0.40) + moderate ambient (27°C) + medium cooling (0.50)."""
        t_next, dt = compute_thermal_step(24.0, 0.40, 27.0, 0.50, DEFAULT_THERMAL_PARAMS)
        cop = compute_cop(27.0, DEFAULT_THERMAL_PARAMS, DEFAULT_COP_PARAMS)
        trend = classify_temperature_trend(dt, DEFAULT_THERMAL_PARAMS)

        assert np.isclose(t_next, 23.9548, atol=1e-3)
        assert np.isclose(dt, -0.0452, atol=1e-3)
        assert np.isclose(cop, 3.34, atol=1e-2)
        assert trend == TemperatureTrend.STABLE

    def test_case_b_high_workload(self):
        """Case B: High load (0.70) + moderate ambient (27°C) + medium cooling (0.50)."""
        t_next, dt = compute_thermal_step(24.0, 0.70, 27.0, 0.50, DEFAULT_THERMAL_PARAMS)
        trend = classify_temperature_trend(dt, DEFAULT_THERMAL_PARAMS)

        assert np.isclose(t_next, 24.2548, atol=1e-3)
        assert np.isclose(dt, +0.2548, atol=1e-3)
        assert trend == TemperatureTrend.WARMING

    def test_case_c_high_ambient(self):
        """Case C: Normal load (0.40) + high ambient (35°C) + medium cooling (0.50)."""
        t_next, dt = compute_thermal_step(24.0, 0.40, 35.0, 0.50, DEFAULT_THERMAL_PARAMS)
        cop = compute_cop(35.0, DEFAULT_THERMAL_PARAMS, DEFAULT_COP_PARAMS)
        trend = classify_temperature_trend(dt, DEFAULT_THERMAL_PARAMS)

        assert np.isclose(t_next, 24.4500, atol=1e-3)
        assert np.isclose(dt, +0.4500, atol=1e-3)
        assert np.isclose(cop, 2.70, atol=1e-2)
        assert trend == TemperatureTrend.WARMING

    def test_case_d_high_workload_high_ambient_max_cool(self):
        """Case D: High load (0.75) + high ambient (36°C) + maximum cooling (1.00)."""
        t_next, dt = compute_thermal_step(24.0, 0.75, 36.0, 1.00, DEFAULT_THERMAL_PARAMS)
        cop = compute_cop(36.0, DEFAULT_THERMAL_PARAMS, DEFAULT_COP_PARAMS)

        assert np.isclose(t_next, 24.3696, atol=1e-3)
        assert np.isclose(dt, +0.3696, atol=1e-3)
        assert np.isclose(cop, 2.62, atol=1e-2)

    def test_case_e_high_workload_high_ambient_zero_cool(self):
        """Case E: High load (0.75) + high ambient (36°C) + zero cooling (0.00)."""
        t_next, dt = compute_thermal_step(24.0, 0.75, 36.0, 0.00, DEFAULT_THERMAL_PARAMS)
        assert np.isclose(t_next, 25.3500, atol=1e-3)
        assert np.isclose(dt, +1.3500, atol=1e-3)

    def test_case_f_low_workload_cool_ambient_low_cool(self):
        """Case F: Low load (0.20) + cool ambient (20°C) + low cooling (0.20)."""
        t_next, dt = compute_thermal_step(24.0, 0.20, 20.0, 0.20, DEFAULT_THERMAL_PARAMS)
        cop = compute_cop(20.0, DEFAULT_THERMAL_PARAMS, DEFAULT_COP_PARAMS)
        trend = classify_temperature_trend(dt, DEFAULT_THERMAL_PARAMS)

        assert np.isclose(t_next, 23.7143, atol=1e-3)
        assert np.isclose(dt, -0.2857, atol=1e-3)
        assert np.isclose(cop, 3.90, atol=1e-2)
        assert trend == TemperatureTrend.COOLING


class TestStabilityAndClassifications:
    """Verifies mathematical stability criteria and ASHRAE classifications."""

    def test_recurrence_is_strictly_stable(self):
        """Stability analysis: 0 < beta < 2 and discrete root A = 1 - beta in (0, 1)."""
        beta = DEFAULT_THERMAL_PARAMS.beta
        assert 0.0 < beta < 2.0
        A = 1.0 - beta
        assert 0.0 < A < 1.0
        assert np.isclose(A, 0.95)

    def test_ashrae_zone_classifications(self):
        """Verifies boundary mapping to ASHRAE Class A1 zones."""
        assert classify_ashrae_zone(17.5, DEFAULT_THERMAL_PARAMS) == AshraeZone.BELOW_RECOMMENDED
        assert classify_ashrae_zone(18.0, DEFAULT_THERMAL_PARAMS) == AshraeZone.RECOMMENDED
        assert classify_ashrae_zone(24.0, DEFAULT_THERMAL_PARAMS) == AshraeZone.RECOMMENDED
        assert classify_ashrae_zone(27.0, DEFAULT_THERMAL_PARAMS) == AshraeZone.RECOMMENDED
        assert classify_ashrae_zone(27.5, DEFAULT_THERMAL_PARAMS) == AshraeZone.ABOVE_RECOMMENDED
        assert classify_ashrae_zone(32.0, DEFAULT_THERMAL_PARAMS) == AshraeZone.ABOVE_RECOMMENDED
        assert classify_ashrae_zone(32.5, DEFAULT_THERMAL_PARAMS) == AshraeZone.BEYOND_A1_ALLOWABLE
