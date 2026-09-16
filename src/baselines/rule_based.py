"""Rule-Based Adaptive Controller for CoolRL.

This module implements a transparent, deterministic heuristic controller based
on ASHRAE TC 9.9 thermal envelopes and internal temperature trends.

DECISION LOGIC TABLE:
┌──────┬───────────────────────────────┬──────────────────────────────┬────────┬─────────────────────────────┐
│ Rule │ Temperature Condition (T)     │ Trend Condition (ΔT)         │ Action │ Description / Rationale     │
├──────┼───────────────────────────────┼──────────────────────────────┼────────┼─────────────────────────────┤
│ R1   │ T > 32.0°C                    │ Any                          │ 4      │ +20% (Severe emergency)     │
│ R2   │ 27.0°C < T <= 32.0°C          │ ΔT > +0.20°C                 │ 4      │ +20% (Warm & rising)        │
│ R3   │ 27.0°C < T <= 32.0°C          │ -0.20°C <= ΔT <= +0.20°C     │ 3      │ +10% (Warm & stable)        │
│ R4   │ 27.0°C < T <= 32.0°C          │ ΔT < -0.20°C                 │ 2      │ Maintain (Warm but falling) │
│ R5   │ 24.0°C < T <= 27.0°C          │ ΔT > +0.20°C                 │ 3      │ +10% (Upper rec & rising)   │
│ R6   │ 18.0°C <= T < 21.0°C          │ ΔT < -0.20°C                 │ 1      │ -10% (Lower rec & falling)  │
│ R7   │ 18.0°C <= T <= 27.0°C         │ All other trends             │ 2      │ Maintain (Recommended band) │
│ R8   │ T < 18.0°C                    │ T < 15.0°C or ΔT < -0.20°C   │ 0      │ -20% (Severe subcooling)    │
│ R9   │ T < 18.0°C                    │ T >= 15.0°C and ΔT >= -0.20°C│ 1      │ -10% (Subcooling moderate)  │
└──────┴───────────────────────────────┴──────────────────────────────┴────────┴─────────────────────────────┘
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np


class RuleBasedController:
    """Deterministic, interpretable adaptive baseline controller for HVAC regulation.

    Features Used:
    - observation[0]: Internal temperature T_int (°C)
    - observation[4]: Temperature change rate delta_T (°C/5-min step)

    Guarantees:
    - Zero future lookahead (strictly causal).
    - No learned parameters or reward dependencies.
    - Fully deterministic output matching standard engineering logic.
    """

    def __init__(
        self,
        temp_severe: float = 32.0,
        temp_warn: float = 27.0,
        temp_nominal_high: float = 24.0,
        temp_nominal_low: float = 21.0,
        temp_subcool: float = 18.0,
        temp_subcool_severe: float = 15.0,
        trend_threshold: float = 0.20,
    ) -> None:
        """Initialize rule thresholds adhering to ASHRAE Class A1 standards.

        Args:
            temp_severe: Severe overheating threshold (°C).
            temp_warn: Upper recommended threshold (°C).
            temp_nominal_high: Upper nominal comfort threshold (°C).
            temp_nominal_low: Lower nominal comfort threshold (°C).
            temp_subcool: Lower recommended / overcooling threshold (°C).
            temp_subcool_severe: Severe subcooling / lower allowable threshold (°C).
            trend_threshold: Sensitivity for rapid temperature rise/fall (°C/step).
        """
        self.temp_severe = float(temp_severe)
        self.temp_warn = float(temp_warn)
        self.temp_nominal_high = float(temp_nominal_high)
        self.temp_nominal_low = float(temp_nominal_low)
        self.temp_subcool = float(temp_subcool)
        self.temp_subcool_severe = float(temp_subcool_severe)
        self.trend_threshold = float(trend_threshold)

    def choose_action(self, observation: Any) -> int:
        """Select a discrete cooling adjustment action based on current observation.

        Args:
            observation: 5-element continuous observation [T_int, W, T_amb, C, delta_T].

        Returns:
            Integer action index in {0, 1, 2, 3, 4}.
        """
        if observation is None or len(observation) < 5:
            raise ValueError(f"Invalid observation provided to RuleBasedController: {observation}")

        t_int = float(observation[0])
        delta_t = float(observation[4])

        # R1: Severe Overheating (Emergency response beyond Class A1 allowable envelope)
        if t_int > self.temp_severe:
            return 4  # +0.20

        # R2 - R4: Warm / Above Recommended envelope (27°C < T <= 32°C)
        elif t_int > self.temp_warn:
            if delta_t > self.trend_threshold:
                return 4  # R2: Rising rapidly -> aggressive cooling (+0.20)
            elif delta_t >= -self.trend_threshold:
                return 3  # R3: Stable or slight rise -> moderate cooling (+0.10)
            else:
                return 2  # R4: Falling rapidly -> maintain to avoid overshoot (0.00)

        # R5: Upper Recommended Zone with rapid warming (24°C < T <= 27°C)
        elif t_int > self.temp_nominal_high and delta_t > self.trend_threshold:
            return 3  # R5: Anticipatory cooling increase (+0.10)

        # R8 - R9: Subcooling Risk / Below Recommended (T < 18°C)
        elif t_int < self.temp_subcool:
            if t_int < self.temp_subcool_severe or delta_t < -self.trend_threshold:
                return 0  # R8: Critical subcooling or falling fast -> aggressive throttle (-0.20)
            else:
                return 1  # R9: Moderate subcooling -> moderate throttle (-0.10)

        # R6: Lower Recommended Zone with rapid cooling (18°C <= T < 21°C)
        elif t_int < self.temp_nominal_low and delta_t < -self.trend_threshold:
            return 1  # R6: Anticipatory energy conservation (-0.10)

        # R7: Recommended comfort band under normal trends (18°C <= T <= 27°C)
        else:
            return 2  # R7: Steady state maintenance (0.00)
