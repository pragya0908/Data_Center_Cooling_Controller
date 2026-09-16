"""Multi-objective reward function for CoolRL data-center cooling optimization.

This module formalizes the mathematical reward formulation balancing:
1. Thermal safety (heavy penalty for T > 27°C, severe nonlinear penalty for T > 32°C).
2. Energy conservation (negative penalty proportional to cooling energy consumption).
3. Overcooling avoidance (penalty for T < 18°C).
4. Actuator stability (penalty on magnitude of cooling action change |ΔC|).

REWARD HIERARCHY:
    Severe Overheating Penalty (T > 32°C)
        >> Ordinary Thermal Deviation (27°C < T <= 32°C)
        >> Overcooling Penalty (T < 18°C)
        >> Energy Penalty (E_cooling)
        >> Action Churn Penalty (|ΔC|)

TIMING CONVENTION:
    The reward is evaluated on the internal temperature RESULTING from the step transition
    T_internal(t+1), reflecting the immediate consequences of action delta ΔC applied at step t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    """Centralized configuration parameters and weights for the CoolRL reward function."""

    # Thermal safety weights
    # Linear slope on temperature excess above 27.0°C
    safety_weight: float = 2.0

    # Quadratic coefficient on severe temperature excess above 32.0°C
    severe_overheat_weight: float = 5.0

    # Energy conservation weight
    # Multiplier on normalized electrical cooling energy
    energy_weight: float = 10.0

    # Overcooling penalty weight
    # Linear slope on temperature deficit below 18.0°C
    overcooling_weight: float = 1.0

    # Actuator churn penalty weight
    # Multiplier on the absolute cooling action increment |ΔC|
    action_churn_weight: float = 0.50

    # ASHRAE TC 9.9 Class A1 temperature thresholds
    temp_recommended_min: float = 18.0   # °C (below which overcooling penalty begins)
    temp_recommended_max: float = 27.0   # °C (above which thermal safety penalty begins)
    temp_allowable_max: float = 32.0     # °C (above which severe quadratic penalty begins)


# Default singleton reward configuration
DEFAULT_REWARD_CONFIG = RewardConfig()


def compute_thermal_safety_penalties(
    internal_temperature: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> tuple[float, float]:
    """Compute ordinary and severe thermal safety penalties.

    Formulation:
        For T <= 27.0°C:
            safety_penalty = 0.0
            severe_penalty = 0.0

        For 27.0°C < T <= 32.0°C:
            safety_penalty = -safety_weight * (T - 27.0)
            severe_penalty = 0.0

        For T > 32.0°C:
            safety_penalty = -safety_weight * (T - 27.0)
            severe_penalty = -severe_overheat_weight * (T - 32.0)^2

    Returns:
        tuple of (safety_penalty, severe_overheat_penalty)
    """
    safety_penalty = 0.0
    severe_overheat_penalty = 0.0

    if internal_temperature > config.temp_recommended_max:
        excess_recommended = internal_temperature - config.temp_recommended_max
        safety_penalty = -config.safety_weight * excess_recommended

        if internal_temperature > config.temp_allowable_max:
            excess_allowable = internal_temperature - config.temp_allowable_max
            severe_overheat_penalty = -config.severe_overheat_weight * (excess_allowable ** 2)

    return float(safety_penalty), float(severe_overheat_penalty)


def compute_overcooling_penalty(
    internal_temperature: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """Compute penalty for operating below the ASHRAE recommended range (T < 18.0°C).

    Formulation:
        For T >= 18.0°C: 0.0
        For T < 18.0°C: -overcooling_weight * (18.0 - T)
    """
    if internal_temperature < config.temp_recommended_min:
        deficit = config.temp_recommended_min - internal_temperature
        return float(-config.overcooling_weight * deficit)
    return 0.0


def compute_energy_penalty(
    cooling_energy: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """Compute penalty for consumed electrical cooling energy.

    Formulation:
        -energy_weight * cooling_energy
    """
    return float(-config.energy_weight * cooling_energy)


def compute_action_churn_penalty(
    action_delta: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """Compute actuator wear / churn penalty for changing the cooling level.

    Formulation:
        -action_churn_weight * abs(action_delta)
    """
    return float(-config.action_churn_weight * abs(action_delta))


def calculate_reward(
    internal_temperature: float,
    cooling_energy: float,
    action_delta: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> tuple[float, dict[str, float]]:
    """Calculate the total additive multi-objective reward and component breakdown.

    Formula:
        R_total = R_safety + R_severe + R_overcooling + R_energy + R_action

    Args:
        internal_temperature: Room intake temperature resulting from the step (°C).
        cooling_energy: Normalized electrical cooling energy consumed during the step.
        action_delta: Applied cooling adjustment ΔC in [-0.20, +0.20].
        config: Reward configuration dataclass.

    Returns:
        tuple of (total_reward, breakdown_dict)
    """
    safety_pen, severe_pen = compute_thermal_safety_penalties(internal_temperature, config)
    overcool_pen = compute_overcooling_penalty(internal_temperature, config)
    energy_pen = compute_energy_penalty(cooling_energy, config)
    churn_pen = compute_action_churn_penalty(action_delta, config)

    total_reward = safety_pen + severe_pen + overcool_pen + energy_pen + churn_pen

    breakdown = {
        "total_reward": float(total_reward),
        "safety_penalty": float(safety_pen),
        "severe_overheat_penalty": float(severe_pen),
        "overcooling_penalty": float(overcool_pen),
        "energy_penalty": float(energy_pen),
        "action_churn_penalty": float(churn_pen),
    }

    return float(total_reward), breakdown
