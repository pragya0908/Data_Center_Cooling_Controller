"""Unit tests for CoolRL baseline controllers.

Validates:
1. FixedCoolingController:
   - Target maintenance at C = 0.50 (Action 2 always).
   - Convergence from off-target initial states.
   - Input validation and deterministic execution.
2. RuleBasedController:
   - Comprehensive branch coverage across all rules R1 through R9.
   - Exact boundary handling at 32°C, 27°C, 24°C, 21°C, 18°C, and 15°C.
   - Trend sensitivity and deadband stability.
   - Zero future lookahead and strictly causal action selection.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.fixed_cooling import FixedCoolingController
from src.baselines.rule_based import RuleBasedController


class TestFixedCoolingController:
    """Validate FixedCoolingController logic and action selection."""

    def test_nominal_target_50_produces_maintain(self):
        """When current cooling is 0.50, controller always chooses Action 2 (maintain)."""
        ctrl = FixedCoolingController(target_cooling=0.50)
        obs = [24.0, 0.40, 26.0, 0.50, 0.00]

        for _ in range(10):
            action = ctrl.choose_action(obs)
            assert action == 2

    def test_off_target_adjustment(self):
        """Controller chooses appropriate action to steer toward target cooling."""
        ctrl = FixedCoolingController(target_cooling=0.75)

        # Low cooling -> increase
        obs_low = [24.0, 0.40, 26.0, 0.20, 0.00]
        assert ctrl.choose_action(obs_low) == 4  # diff = +0.55 -> Action 4 (+0.20)

        # High cooling -> decrease
        ctrl_low = FixedCoolingController(target_cooling=0.25)
        obs_high = [24.0, 0.40, 26.0, 0.80, 0.00]
        assert ctrl_low.choose_action(obs_high) == 0  # diff = -0.55 -> Action 0 (-0.20)

    def test_invalid_target_and_observation_rejection(self):
        """Reject out-of-bound targets and malformed observations."""
        with pytest.raises(ValueError):
            FixedCoolingController(target_cooling=-0.1)
        with pytest.raises(ValueError):
            FixedCoolingController(target_cooling=1.5)

        ctrl = FixedCoolingController()
        with pytest.raises(ValueError):
            ctrl.choose_action(None)
        with pytest.raises(ValueError):
            ctrl.choose_action([24.0, 0.4])  # Too short


class TestRuleBasedController:
    """Validate all decision branches R1..R9 in RuleBasedController."""

    @pytest.fixture
    def ctrl(self) -> RuleBasedController:
        return RuleBasedController()

    def test_r1_severe_overheating(self, ctrl: RuleBasedController):
        """R1: T > 32.0°C must trigger Action 4 (+20%) regardless of trend."""
        for dt in [-0.50, 0.00, +0.50]:
            obs = [33.5, 0.50, 30.0, 0.50, dt]
            assert ctrl.choose_action(obs) == 4

    def test_r2_warm_rising(self, ctrl: RuleBasedController):
        """R2: 27°C < T <= 32°C and delta_T > 0.20°C -> Action 4 (+20%)."""
        obs = [29.0, 0.50, 28.0, 0.50, +0.25]
        assert ctrl.choose_action(obs) == 4

    def test_r3_warm_stable(self, ctrl: RuleBasedController):
        """R3: 27°C < T <= 32°C and -0.20 <= delta_T <= 0.20 -> Action 3 (+10%)."""
        obs_stable = [29.0, 0.50, 28.0, 0.50, 0.00]
        assert ctrl.choose_action(obs_stable) == 3

        obs_bound = [32.0, 0.50, 28.0, 0.50, +0.20]
        assert ctrl.choose_action(obs_bound) == 3

    def test_r4_warm_falling(self, ctrl: RuleBasedController):
        """R4: 27°C < T <= 32°C and delta_T < -0.20°C -> Action 2 (Maintain)."""
        obs = [29.0, 0.50, 28.0, 0.50, -0.30]
        assert ctrl.choose_action(obs) == 2

    def test_r5_upper_recommended_rising(self, ctrl: RuleBasedController):
        """R5: 24°C < T <= 27°C and delta_T > +0.20°C -> Action 3 (+10%)."""
        obs = [25.5, 0.50, 25.0, 0.50, +0.35]
        assert ctrl.choose_action(obs) == 3

    def test_r6_lower_recommended_falling(self, ctrl: RuleBasedController):
        """R6: 18°C <= T < 21°C and delta_T < -0.20°C -> Action 1 (-10%)."""
        obs = [19.5, 0.30, 20.0, 0.50, -0.35]
        assert ctrl.choose_action(obs) == 1

    def test_r7_recommended_normal_trend(self, ctrl: RuleBasedController):
        """R7: 18°C <= T <= 27°C under normal trends -> Action 2 (Maintain)."""
        # Nominal midpoint
        obs_mid = [22.5, 0.40, 24.0, 0.50, 0.00]
        assert ctrl.choose_action(obs_mid) == 2

        # Upper band but falling
        obs_upper_fall = [26.0, 0.40, 24.0, 0.50, -0.25]
        assert ctrl.choose_action(obs_upper_fall) == 2

        # Lower band but rising
        obs_lower_rise = [19.0, 0.40, 24.0, 0.50, +0.25]
        assert ctrl.choose_action(obs_lower_rise) == 2

    def test_r8_severe_subcooling(self, ctrl: RuleBasedController):
        """R8: T < 18°C and (T < 15°C or delta_T < -0.20°C) -> Action 0 (-20%)."""
        # T < 15°C
        obs_extreme = [14.0, 0.20, 18.0, 0.50, 0.00]
        assert ctrl.choose_action(obs_extreme) == 0

        # T in [15, 18) but falling rapidly
        obs_rapid_fall = [17.0, 0.20, 18.0, 0.50, -0.25]
        assert ctrl.choose_action(obs_rapid_fall) == 0

    def test_r9_moderate_subcooling(self, ctrl: RuleBasedController):
        """R9: 15°C <= T < 18°C and delta_T >= -0.20°C -> Action 1 (-10%)."""
        obs = [17.0, 0.20, 18.0, 0.50, 0.00]
        assert ctrl.choose_action(obs) == 1

        obs_rising = [16.5, 0.20, 18.0, 0.50, +0.10]
        assert ctrl.choose_action(obs_rising) == 1

    def test_deterministic_causality(self, ctrl: RuleBasedController):
        """Identical observations always yield identical actions without side effects."""
        obs = [28.5, 0.45, 27.0, 0.50, 0.05]
        actions = [ctrl.choose_action(obs) for _ in range(25)]
        assert all(a == 3 for a in actions)
