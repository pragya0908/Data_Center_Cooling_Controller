"""CoolRL environment module."""

from src.environment.datacenter_env import (
    ContinuousObservationSpace,
    DataCenterEnv,
    DiscreteActionSpace,
)
from src.environment.reward import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    calculate_reward,
    compute_action_churn_penalty,
    compute_energy_penalty,
    compute_overcooling_penalty,
    compute_thermal_safety_penalties,
)
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

__all__ = [
    "DataCenterEnv",
    "DiscreteActionSpace",
    "ContinuousObservationSpace",
    "ThermalModelParameters",
    "COPParameters",
    "AshraeZone",
    "TemperatureTrend",
    "DEFAULT_THERMAL_PARAMS",
    "DEFAULT_COP_PARAMS",
    "RewardConfig",
    "DEFAULT_REWARD_CONFIG",
    "calculate_reward",
    "compute_thermal_safety_penalties",
    "compute_overcooling_penalty",
    "compute_energy_penalty",
    "compute_action_churn_penalty",
    "compute_ambient_derating_factor",
    "compute_cooling_effect",
    "compute_thermal_step",
    "compute_cop",
    "compute_cooling_power_and_energy",
    "classify_ashrae_zone",
    "classify_temperature_trend",
]
