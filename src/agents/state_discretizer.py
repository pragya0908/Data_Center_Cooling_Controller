"""State Discretizer and RL State/Action Interface for CoolRL.

This module establishes the formal, deterministic contract between the continuous
simulation environment (DataCenterEnv) and the discrete tabular Q-learning agent:

    Environment Observation (5 continuous features)
        ↓
    StateDiscretizer.discretize(observation) (5 discrete bin indices)
        ↓
    StateDiscretizer.encode(observation) (unique integer state ID in 0..959)
        ↓
    Tabular Q-Learning Action Selection (discrete action ID in 0..4)
        ↓
    DataCenterEnv.step(action) (applied cooling delta in [-0.20, +0.20])

STATE SPACE DIMENSIONALITY (960 Discrete States):
    1. Internal Temperature (5 bins):
       0: T < 18°C (below recommended)
       1: 18°C <= T < 21°C (recommended - lower comfort)
       2: 21°C <= T < 24°C (recommended - nominal)
       3: 24°C <= T < 27°C (recommended - upper comfort)
       4: T >= 27°C (above recommended / allowable breach)

    2. Workload (4 bins):
       0: W < 0.33 (light IT load)
       1: 0.33 <= W < 0.45 (moderate IT load)
       2: 0.45 <= W < 0.55 (high IT load)
       3: W >= 0.55 (peak IT load)

    3. Ambient Temperature (4 bins):
       0: T_amb < 23°C (cool outdoor)
       1: 23°C <= T_amb < 27°C (mild outdoor)
       2: 27°C <= T_amb < 31°C (warm outdoor)
       3: T_amb >= 31°C (heatwave outdoor)

    4. Cooling Level (4 bins):
       0: C < 0.25 (low cooling)
       1: 0.25 <= C < 0.50 (moderate-low cooling)
       2: 0.50 <= C < 0.75 (moderate-high cooling)
       3: C >= 0.75 (high cooling)

    5. Temperature Trend (3 bins):
       0: delta_T < -0.20°C (rapid cooling)
       1: -0.20°C <= delta_T <= +0.20°C (stable temperature)
       2: delta_T > +0.20°C (rapid warming)

    Total Combinations: 5 * 4 * 4 * 4 * 3 = 960 states.

ENCODING METHOD:
    Deterministic mixed-radix positional encoding:
        state_id = (b0 * 192) + (b1 * 48) + (b2 * 12) + (b3 * 3) + b4
    where:
        radix multipliers: m = (192, 48, 12, 3, 1)
        state_id in [0, 959] strictly, bijectively, and reversibly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class DiscretizationConfig:
    """Thresholds and bin boundary specifications for CoolRL state discretization."""

    # Internal temperature thresholds (°C) -> 5 bins: [0, 18), [18, 21), [21, 24), [24, 27), [27, inf)
    temp_thresholds: tuple[float, ...] = (18.0, 21.0, 24.0, 27.0)

    # Workload thresholds [0.0, 1.0] -> 4 bins: [0, 0.33), [0.33, 0.45), [0.45, 0.55), [0.55, inf)
    workload_thresholds: tuple[float, ...] = (0.33, 0.45, 0.55)

    # Ambient temperature thresholds (°C) -> 4 bins: [0, 23), [23, 27), [27, 31), [31, inf)
    ambient_thresholds: tuple[float, ...] = (23.0, 27.0, 31.0)

    # Cooling effort thresholds [0.0, 1.0] -> 4 bins: [0, 0.25), [0.25, 0.50), [0.50, 0.75), [0.75, inf)
    cooling_thresholds: tuple[float, ...] = (0.25, 0.50, 0.75)

    # Temperature rate-of-change thresholds (°C/step) -> 3 bins: (-inf, -0.20), [-0.20, +0.20], (+0.20, +inf)
    trend_threshold_lower: float = -0.20
    trend_threshold_upper: float = 0.20


DEFAULT_DISCRETIZATION_CONFIG = DiscretizationConfig()


class StateDiscretizer:
    """Deterministic state space discretizer mapping continuous observations to discrete state IDs.

    Guarantees:
    - Exactly 960 discrete state IDs [0, 959].
    - Bijective, deterministic mixed-radix encoding and exact inversion (decoding).
    - Strict input validation rejecting malformed, non-numeric, NaN, or infinite observations.
    - Zero Python hash() dependencies for 100% reproducible cross-process execution.
    """

    NUM_STATES: int = 960
    NUM_ACTIONS: int = 5
    DIM_SIZES: tuple[int, ...] = (5, 4, 4, 4, 3)
    RADIX_MULTIPLIERS: tuple[int, ...] = (192, 48, 12, 3, 1)

    FEATURE_NAMES: tuple[str, ...] = (
        "internal_temperature",
        "workload",
        "ambient_temperature",
        "cooling_level",
        "temperature_change",
    )

    ACTION_MAP: dict[int, float] = {
        0: -0.20,
        1: -0.10,
        2: 0.00,
        3: +0.10,
        4: +0.20,
    }

    ACTION_DESCRIPTIONS: dict[int, str] = {
        0: "Decrease cooling by 20% (-0.20)",
        1: "Decrease cooling by 10% (-0.10)",
        2: "Maintain cooling level (0.00)",
        3: "Increase cooling by 10% (+0.10)",
        4: "Increase cooling by 20% (+0.20)",
    }

    def __init__(self, config: DiscretizationConfig = DEFAULT_DISCRETIZATION_CONFIG) -> None:
        """Initialize the state discretizer with the specified bin boundaries."""
        self.config = config

    def validate_observation(self, observation: Any) -> np.ndarray:
        """Strictly validate that observation has exactly 5 numeric, finite values.

        Raises:
            TypeError: If observation or elements are non-numeric.
            ValueError: If length != 5, or contains NaN or infinite values.
        """
        if observation is None:
            raise ValueError("Observation cannot be None.")

        # Convert to numpy array
        try:
            arr = np.asarray(observation, dtype=np.float64)
        except (ValueError, TypeError) as err:
            raise TypeError(f"Observation must be numeric: {err}") from err

        if arr.ndim != 1 or arr.shape[0] != 5:
            raise ValueError(
                f"Expected observation of shape (5,), but got shape {arr.shape} (value: {observation})."
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Observation contains NaN or infinite values: {arr}")

        return arr

    def discretize_temperature(self, t: float) -> int:
        """Discretize internal temperature into 5 bins [0..4].

        0: T < 18.0°C
        1: 18.0°C <= T < 21.0°C
        2: 21.0°C <= T < 24.0°C
        3: 24.0°C <= T < 27.0°C
        4: T >= 27.0°C
        """
        th = self.config.temp_thresholds
        if t < th[0]:
            return 0
        elif t < th[1]:
            return 1
        elif t < th[2]:
            return 2
        elif t < th[3]:
            return 3
        else:
            return 4

    def discretize_workload(self, w: float) -> int:
        """Discretize workload into 4 bins [0..3].

        0: W < 0.33
        1: 0.33 <= W < 0.45
        2: 0.45 <= W < 0.55
        3: W >= 0.55
        """
        th = self.config.workload_thresholds
        if w < th[0]:
            return 0
        elif w < th[1]:
            return 1
        elif w < th[2]:
            return 2
        else:
            return 3

    def discretize_ambient(self, t_amb: float) -> int:
        """Discretize ambient temperature into 4 bins [0..3].

        0: T_amb < 23.0°C
        1: 23.0°C <= T_amb < 27.0°C
        2: 27.0°C <= T_amb < 31.0°C
        3: T_amb >= 31.0°C
        """
        th = self.config.ambient_thresholds
        if t_amb < th[0]:
            return 0
        elif t_amb < th[1]:
            return 1
        elif t_amb < th[2]:
            return 2
        else:
            return 3

    def discretize_cooling(self, c: float) -> int:
        """Discretize cooling effort into 4 bins [0..3].

        0: C < 0.25
        1: 0.25 <= C < 0.50
        2: 0.50 <= C < 0.75
        3: C >= 0.75
        """
        th = self.config.cooling_thresholds
        if c < th[0]:
            return 0
        elif c < th[1]:
            return 1
        elif c < th[2]:
            return 2
        else:
            return 3

    def discretize_trend(self, delta_t: float) -> int:
        """Discretize temperature trend into 3 bins [0..2].

        0: delta_T < -0.20°C
        1: -0.20°C <= delta_T <= +0.20°C
        2: delta_T > +0.20°C
        """
        if delta_t < self.config.trend_threshold_lower:
            return 0
        elif delta_t <= self.config.trend_threshold_upper:
            return 1
        else:
            return 2

    def discretize(self, observation: Any) -> tuple[int, int, int, int, int]:
        """Convert a continuous 5-element observation vector to 5 discrete bin indices.

        Args:
            observation: 5-element array-like [T_int, W, T_amb, C, delta_T].

        Returns:
            tuple of (b_temp, b_workload, b_ambient, b_cooling, b_trend)
        """
        arr = self.validate_observation(observation)
        b0 = self.discretize_temperature(float(arr[0]))
        b1 = self.discretize_workload(float(arr[1]))
        b2 = self.discretize_ambient(float(arr[2]))
        b3 = self.discretize_cooling(float(arr[3]))
        b4 = self.discretize_trend(float(arr[4]))
        return (b0, b1, b2, b3, b4)

    def encode_bins(self, b0: int, b1: int, b2: int, b3: int, b4: int) -> int:
        """Encode 5 discrete bin indices into a unique integer state ID in [0, 959].

        Formula:
            state_id = b0 * 192 + b1 * 48 + b2 * 12 + b3 * 3 + b4
        """
        if not (0 <= b0 < 5):
            raise ValueError(f"Temperature bin b0 must be in [0, 4], got {b0}")
        if not (0 <= b1 < 4):
            raise ValueError(f"Workload bin b1 must be in [0, 3], got {b1}")
        if not (0 <= b2 < 4):
            raise ValueError(f"Ambient bin b2 must be in [0, 3], got {b2}")
        if not (0 <= b3 < 4):
            raise ValueError(f"Cooling bin b3 must be in [0, 3], got {b3}")
        if not (0 <= b4 < 3):
            raise ValueError(f"Trend bin b4 must be in [0, 2], got {b4}")

        return int(b0 * 192 + b1 * 48 + b2 * 12 + b3 * 3 + b4)

    def encode(self, observation: Any) -> int:
        """Discretize and encode a continuous observation directly into an integer state ID.

        Args:
            observation: 5-element continuous observation from DataCenterEnv.

        Returns:
            integer state_id in [0, 959].
        """
        b0, b1, b2, b3, b4 = self.discretize(observation)
        return self.encode_bins(b0, b1, b2, b3, b4)

    def decode(self, state_id: int) -> tuple[int, int, int, int, int]:
        """Decode a discrete state ID in [0, 959] back to its 5 constituent bin indices.

        Args:
            state_id: Integer in range [0, 959].

        Returns:
            tuple of (b_temp, b_workload, b_ambient, b_cooling, b_trend).
        """
        if not isinstance(state_id, (int, np.integer)):
            raise TypeError(f"State ID must be an integer, got {type(state_id).__name__}")
        sid = int(state_id)
        if not (0 <= sid < self.NUM_STATES):
            raise ValueError(f"State ID must be in range [0, {self.NUM_STATES - 1}], got {sid}")

        b4 = sid % 3
        rem3 = sid // 3
        b3 = rem3 % 4
        rem2 = rem3 // 4
        b2 = rem2 % 4
        rem1 = rem2 // 4
        b1 = rem1 % 4
        b0 = rem1 // 4

        return (b0, b1, b2, b3, b4)

    def get_bin_descriptions(self, state_id: int) -> dict[str, Any]:
        """Return a human-readable semantic dictionary explaining the state ID."""
        b0, b1, b2, b3, b4 = self.decode(state_id)

        temp_desc = {
            0: "T < 18°C (Below Recommended / Overcooling Risk)",
            1: "18°C <= T < 21°C (Recommended - Lower)",
            2: "21°C <= T < 24°C (Recommended - Nominal Comfort)",
            3: "24°C <= T < 27°C (Recommended - Upper)",
            4: "T >= 27°C (Above Recommended / Warning)",
        }
        wl_desc = {
            0: "W < 0.33 (Low Workload)",
            1: "0.33 <= W < 0.45 (Moderate Workload)",
            2: "0.45 <= W < 0.55 (High Workload)",
            3: "W >= 0.55 (Peak Workload)",
        }
        amb_desc = {
            0: "T_amb < 23°C (Cool Outdoor)",
            1: "23°C <= T_amb < 27°C (Mild Outdoor)",
            2: "27°C <= T_amb < 31°C (Warm Outdoor)",
            3: "T_amb >= 31°C (Heatwave Outdoor)",
        }
        cool_desc = {
            0: "C < 0.25 (Low Cooling Effort)",
            1: "0.25 <= C < 0.50 (Moderate-Low Cooling)",
            2: "0.50 <= C < 0.75 (Moderate-High Cooling)",
            3: "C >= 0.75 (High Cooling Effort)",
        }
        trend_desc = {
            0: "delta_T < -0.20°C (Rapid Cooling)",
            1: "-0.20°C <= delta_T <= +0.20°C (Stable)",
            2: "delta_T > +0.20°C (Rapid Warming)",
        }

        return {
            "state_id": state_id,
            "bins": (b0, b1, b2, b3, b4),
            "internal_temperature": temp_desc[b0],
            "workload": wl_desc[b1],
            "ambient_temperature": amb_desc[b2],
            "cooling_level": cool_desc[b3],
            "temperature_trend": trend_desc[b4],
        }
