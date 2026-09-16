"""Pure mathematical specifications and parameters for the CoolRL thermal simulation.

This module defines the mathematical equations, dataclasses, and calibration constants
for the first-order lumped data-center thermal model designed in Phase 2A.

ARCHITECTURAL PRINCIPLES:
1. SPECIFICATION ONLY: This module contains pure mathematical helper functions and
   parameter definitions.
2. NO ENVIRONMENT IMPLEMENTATION: Does NOT contain gym/Gymnasium environments,
   step(), reset(), render(), or action-selection logic.
3. NO REINFORCEMENT LEARNING: Does NOT contain Q-tables, agents, rewards, or policies.
4. DETERMINISTIC BY DEFAULT: Default disturbance is strictly zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class AshraeZone(str, Enum):
    """ASHRAE TC 9.9 thermal envelope classifications for Class A1 environments."""

    BELOW_RECOMMENDED = "below_recommended"      # < 18.0°C (overcooling risk)
    RECOMMENDED = "recommended"                  # 18.0°C to 27.0°C (optimal envelope)
    ABOVE_RECOMMENDED = "above_recommended"      # 27.0°C to 32.0°C (warm / fan escalation)
    BEYOND_A1_ALLOWABLE = "beyond_a1_allowable"  # > 32.0°C (thermal boundary violation)


class TemperatureTrend(str, Enum):
    """Internal temperature rate-of-change classifications over a 5-minute timestep."""

    COOLING = "cooling"  # ΔT < -0.20°C
    STABLE = "stable"    # -0.20°C <= ΔT <= +0.20°C
    WARMING = "warming"  # ΔT > +0.20°C


@dataclass(frozen=True)
class ThermalModelParameters:
    """Calibrated physical and model parameters for the first-order lumped thermal simulation."""

    # Time step
    timestep_seconds: int = 300                 # 5 minutes
    timestep_hours: float = 300.0 / 3600.0      # 1/12 hour (~0.08333 hr)

    # Workload heat conversion coefficient
    # Meaning: Temperature rise in °C per 5-minute step at maximum workload (W = 1.0)
    alpha: float = 1.00                         # °C / step

    # Ambient envelope coupling coefficient
    # Meaning: Fraction of internal-ambient temperature differential conducted per step
    # Stability condition: 0.0 < beta < 2.0 (physically calibrated for building envelope inertia)
    beta: float = 0.05                          # dimensionless fraction / step

    # Cooling capacity coefficient
    # Meaning: Nominal temperature reduction in °C per step at maximum cooling (C = 1.0)
    gamma: float = 1.25                         # °C / step

    # Ambient cooling derating parameters
    t_ref_ambient: float = 25.0                 # °C (reference outdoor temperature)
    t_hot_ambient: float = 35.0                 # °C (severe heatwave benchmark)
    lambda_derate: float = 0.25                 # dimensionless derating slope (25% penalty at 35°C)
    f_ambient_lower_bound: float = 0.80         # minimum derating factor (sub-ambient enhancement cap)
    f_ambient_upper_bound: float = 1.30         # maximum derating factor (extreme heat penalty cap)

    # Simulation initialization defaults
    initial_internal_temperature: float = 24.0  # °C (simulation initialization within ASHRAE recommended range)
    initial_cooling_level: float = 0.50         # dimensionless normalized effort [0.0, 1.0]

    # ASHRAE TC 9.9 thermal safety boundaries (Class A1)
    ashrae_recommended_min: float = 18.0        # °C
    ashrae_recommended_max: float = 27.0        # °C
    ashrae_allowable_min: float = 15.0          # °C
    ashrae_allowable_max: float = 32.0          # °C

    # Temperature trend sensitivity threshold
    trend_threshold_c: float = 0.20             # °C per 5 minutes (equivalent to 2.4°C / hour)


@dataclass(frozen=True)
class COPParameters:
    """Parameters for the ambient-dependent Coefficient of Performance (COP) and power model."""

    # Reference COP at reference ambient temperature (T_ref = 25°C)
    cop_ref: float = 3.50                       # dimensionless

    # Temperature sensitivity of COP
    # Meaning: Linear reduction in COP per 1°C increase in outdoor temperature
    k_cop: float = 0.08                         # °C^-1

    # Safety clipping bounds
    cop_min: float = 1.50                       # minimum physical COP under extreme condensing heat
    cop_max: float = 5.00                       # maximum physical COP under cool economizer operation

    # Thermal capacity normalization
    # By default, Q_max = 1.0 (normalized thermal capacity)
    # Optional physical scaling: e.g., 100.0 kW for a standard modular IT pod
    q_max: float = 1.00                         # normalized thermal units (or kW)


def compute_ambient_derating_factor(
    t_ambient: float,
    params: ThermalModelParameters,
) -> float:
    """Compute the ambient-dependent cooling resistance/derating factor F_ambient(t).

    Formula:
        F_ambient = clip(1 + lambda * (T_ambient - T_ref) / (T_hot - T_ref), lower, upper)

    At high ambient (e.g. 35°C), F_ambient rises to 1.25, derating cooling capacity by 20%.
    At cool ambient (e.g. 20°C), F_ambient drops to ~0.875, enhancing cooling efficiency.
    """
    delta_t_ratio = (t_ambient - params.t_ref_ambient) / (params.t_hot_ambient - params.t_ref_ambient)
    raw_factor = 1.0 + params.lambda_derate * delta_t_ratio
    return float(np.clip(raw_factor, params.f_ambient_lower_bound, params.f_ambient_upper_bound))


def compute_cooling_effect(
    cooling_level: float,
    t_ambient: float,
    params: ThermalModelParameters,
) -> float:
    """Compute the net temperature reduction achieved by active cooling over one step.

    Formula:
        cooling_effect(t) = gamma * C(t) / F_ambient(t)
    """
    f_amb = compute_ambient_derating_factor(t_ambient, params)
    return float((params.gamma * cooling_level) / f_amb)


def compute_thermal_step(
    t_internal: float,
    workload: float,
    t_ambient: float,
    cooling_level: float,
    params: ThermalModelParameters,
    disturbance: float = 0.0,
) -> tuple[float, float]:
    """Calculate the next internal temperature and step delta via the lumped thermal equation.

    Formula:
        T_internal(t+1) = T_internal(t)
                          + alpha * W(t)
                          + beta * (T_ambient(t) - T_internal(t))
                          - (gamma * C(t) / F_ambient(t))
                          + disturbance(t)

    Returns:
        tuple of (t_internal_next, delta_t)
    """
    dt_workload = params.alpha * workload
    dt_ambient = params.beta * (t_ambient - t_internal)
    dt_cooling = compute_cooling_effect(cooling_level, t_ambient, params)

    t_next = t_internal + dt_workload + dt_ambient - dt_cooling + disturbance
    delta_t = t_next - t_internal
    return float(t_next), float(delta_t)


def compute_cop(
    t_ambient: float,
    params: ThermalModelParameters,
    cop_params: COPParameters,
) -> float:
    """Compute the instantaneous Coefficient of Performance (COP) as a function of ambient temperature.

    Formula:
        COP(t) = clip(COP_ref - k_cop * (T_ambient(t) - T_ref), COP_min, COP_max)
    """
    raw_cop = cop_params.cop_ref - cop_params.k_cop * (t_ambient - params.t_ref_ambient)
    return float(np.clip(raw_cop, cop_params.cop_min, cop_params.cop_max))


def compute_cooling_power_and_energy(
    cooling_level: float,
    cop: float,
    cop_params: COPParameters,
    dt_hours: float,
) -> tuple[float, float, float]:
    """Compute thermal extraction, electrical power demand, and electrical energy consumption.

    Formulas:
        Q_cooling(t) = C(t) * Q_max
        P_cooling(t) = Q_cooling(t) / COP(t)
        E_cooling(t) = P_cooling(t) * dt_hours

    Returns:
        tuple of (Q_cooling, P_cooling, E_cooling)
    """
    q_cooling = cooling_level * cop_params.q_max
    p_cooling = q_cooling / cop
    e_cooling = p_cooling * dt_hours
    return float(q_cooling), float(p_cooling), float(e_cooling)


def classify_ashrae_zone(t_internal: float, params: ThermalModelParameters) -> AshraeZone:
    """Classify internal data center temperature into ASHRAE TC 9.9 thermal envelopes."""
    if t_internal < params.ashrae_recommended_min:
        return AshraeZone.BELOW_RECOMMENDED
    elif t_internal <= params.ashrae_recommended_max:
        return AshraeZone.RECOMMENDED
    elif t_internal <= params.ashrae_allowable_max:
        return AshraeZone.ABOVE_RECOMMENDED
    else:
        return AshraeZone.BEYOND_A1_ALLOWABLE


def classify_temperature_trend(delta_t: float, params: ThermalModelParameters) -> TemperatureTrend:
    """Classify temperature rate of change into Cooling, Stable, or Warming categories."""
    if delta_t < -params.trend_threshold_c:
        return TemperatureTrend.COOLING
    elif delta_t <= params.trend_threshold_c:
        return TemperatureTrend.STABLE
    else:
        return TemperatureTrend.WARMING


# Singleton default specifications
DEFAULT_THERMAL_PARAMS = ThermalModelParameters()
DEFAULT_COP_PARAMS = COPParameters()
