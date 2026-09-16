"""CoolRL Data Center Thermal Simulation Environment.

This module implements the functional DataCenterEnv, providing a lightweight,
Gymnasium-style discrete-control simulation environment without external gym
dependencies.

KEY ARCHITECTURAL FEATURES:
1. TRACE-DRIVEN SIMULATION:
   - Workload driven by real Alibaba Cluster Trace 2018.
   - Ambient temperature driven by real NASA POWER Bengaluru hourly reanalysis.
2. FIVE FORMAL SCENARIOS:
   - normal, high_workload, workload_spikes, high_ambient, combined_stress.
3. PHYSICAL & ENERGY ACCOUNTING:
   - First-order lumped thermal transition from thermal_model_spec.
   - Ambient-dependent cooling derating and thermodynamic COP.
   - Normalized cooling power and energy tracking per 5-minute step.
4. DETERMINISTIC & COMPATIBLE:
   - Standard reset() and step() API matching modern RL conventions.
   - Preserves continuous internal state and exposes discretization-ready observation vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.data.scenario_config import (
    SCENARIOS,
    ScenarioType,
)
from src.environment.reward import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    calculate_reward,
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

# Root directory resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class DiscreteActionSpace:
    """Lightweight discrete action space representation."""

    n: int = 5
    actions: tuple[int, ...] = (0, 1, 2, 3, 4)
    deltas: tuple[float, ...] = (-0.20, -0.10, 0.00, +0.10, +0.20)

    def sample(self, seed: int | None = None) -> int:
        """Sample an action uniformly at random."""
        rng = np.random.default_rng(seed)
        return int(rng.integers(0, self.n))

    def contains(self, x: Any) -> bool:
        """Check if action x is valid."""
        return isinstance(x, (int, np.integer)) and int(x) in self.actions


@dataclass(frozen=True)
class ContinuousObservationSpace:
    """Lightweight 5-dimensional continuous observation space specification."""

    shape: tuple[int, ...] = (5,)
    # Bounds: [T_int, W, T_amb, C, delta_T]
    low: tuple[float, ...] = (-50.0, 0.0, -20.0, 0.0, -50.0)
    high: tuple[float, ...] = (120.0, 1.0, 70.0, 1.0, 50.0)



class DataCenterEnv:
    """Trace-driven data center thermal simulation environment for CoolRL.

    Lightweight Gymnasium-style interface:
        obs, info = env.reset(seed=None, options=None)
        obs, reward, terminated, truncated, info = env.step(action)
    """

    ACTION_DELTAS: tuple[float, ...] = (-0.20, -0.10, 0.00, +0.10, +0.20)

    def __init__(
        self,
        scenario: str | ScenarioType = "normal",
        split: str = "all",
        thermal_params: ThermalModelParameters = DEFAULT_THERMAL_PARAMS,
        cop_params: COPParameters = DEFAULT_COP_PARAMS,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
        workload_csv_path: Path | str | None = None,
        weather_csv_path: Path | str | None = None,
    ) -> None:
        """Initialize the DataCenterEnv simulation environment.

        Args:
            scenario: Name or enum of the experimental scenario:
                      'normal', 'high_workload', 'workload_spikes',
                      'high_ambient', or 'combined_stress'.
            split: Dataset slice to evaluate:
                   'all' (entire trace, 2243 steps),
                   'train' (steps 0 to 1439, 1440 steps),
                   'val' (steps 1440 to 1727, 288 steps),
                   'test' (steps 1728 to 2242, 515 steps).
            thermal_params: Calibrated physical and thermal constants.
            cop_params: Calibrated chiller COP and energy constants.
            workload_csv_path: Optional custom path to alibaba_workload_300s.csv.
            weather_csv_path: Optional custom path to bengaluru_weather_300s_aligned.csv.
        """
        self.thermal_params = thermal_params
        self.cop_params = cop_params
        self.reward_config = reward_config
        self.split_name = str(split).lower()

        # Resolve and normalize scenario
        self.scenario_name = self._resolve_scenario_name(scenario)

        # Action and observation space descriptors
        self.action_space = DiscreteActionSpace()
        self.observation_space = ContinuousObservationSpace()

        # Load and transform real trajectories in memory
        self._load_and_prepare_trajectories(workload_csv_path, weather_csv_path)

        # Continuous state variables
        self.T_internal: float = self.thermal_params.initial_internal_temperature
        self.cooling_level: float = self.thermal_params.initial_cooling_level
        self.previous_internal_temperature: float = self.thermal_params.initial_internal_temperature
        self.temperature_change: float = 0.0
        self.current_step: int = 0

    @staticmethod
    def _resolve_scenario_name(scenario: str | ScenarioType) -> str:
        """Normalize scenario argument to a standardized string."""
        if isinstance(scenario, ScenarioType):
            raw = scenario.value
        else:
            raw = str(scenario).lower().strip()

        alias_map = {
            "normal": "scenario_1_normal",
            "scenario_1_normal": "scenario_1_normal",
            "1": "scenario_1_normal",
            "high_workload": "scenario_2_high_workload",
            "scenario_2_high_workload": "scenario_2_high_workload",
            "2": "scenario_2_high_workload",
            "workload_spikes": "scenario_3_workload_spikes",
            "scenario_3_workload_spikes": "scenario_3_workload_spikes",
            "3": "scenario_3_workload_spikes",
            "high_ambient": "scenario_4_high_ambient",
            "scenario_4_high_ambient": "scenario_4_high_ambient",
            "4": "scenario_4_high_ambient",
            "combined_stress": "scenario_5_combined_stress",
            "scenario_5_combined_stress": "scenario_5_combined_stress",
            "5": "scenario_5_combined_stress",
        }
        if raw not in alias_map:
            raise ValueError(
                f"Unknown scenario '{scenario}'. Supported scenarios: "
                "['normal', 'high_workload', 'workload_spikes', 'high_ambient', 'combined_stress']"
            )
        return alias_map[raw]

    def _load_and_prepare_trajectories(
        self,
        workload_csv_path: Path | str | None,
        weather_csv_path: Path | str | None,
    ) -> None:
        """Load underlying processed CSV datasets and apply scenario transformations in memory."""
        w_path = Path(workload_csv_path) if workload_csv_path else PROCESSED_DATA_DIR / "alibaba_workload_300s.csv"
        amb_path = Path(weather_csv_path) if weather_csv_path else PROCESSED_DATA_DIR / "bengaluru_weather_300s_aligned.csv"

        if not w_path.exists():
            raise FileNotFoundError(f"Workload dataset not found at: {w_path}")
        if not amb_path.exists():
            raise FileNotFoundError(f"Weather dataset not found at: {amb_path}")

        df_w = pd.read_csv(w_path)
        df_amb = pd.read_csv(amb_path)

        base_workload = df_w["workload"].to_numpy(dtype=np.float64)
        base_ambient = df_amb["ambient_temp_c"].to_numpy(dtype=np.float64)

        if len(base_workload) != len(base_ambient):
            raise ValueError(
                f"Length mismatch: workload has {len(base_workload)} steps, "
                f"ambient has {len(base_ambient)} steps."
            )

        # Apply in-memory scenario transformations
        transformed_workload = base_workload.copy()
        transformed_ambient = base_ambient.copy()

        if self.scenario_name == "scenario_1_normal":
            # Baseline: unaltered real trajectories
            pass

        elif self.scenario_name == "scenario_2_high_workload":
            # Controlled amplification: W_high(t) = clip(1.25 * W(t), 0, 1)
            k = SCENARIOS.high_workload.scaling_factor_k
            transformed_workload = np.clip(k * base_workload, 0.0, 1.0)

        elif self.scenario_name == "scenario_3_workload_spikes":
            # Transient burst injection (magnitude +0.25, duration 4 steps)
            transformed_workload = self._inject_workload_spikes(
                base_workload,
                magnitude=SCENARIOS.spikes.spike_magnitude,
                duration=SCENARIOS.spikes.default_spike_duration_steps,
                seed=SCENARIOS.spikes.random_seed,
            )

        elif self.scenario_name == "scenario_4_high_ambient":
            # Additive heatwave thermal anomaly (+3.0°C)
            transformed_ambient = base_ambient + SCENARIOS.ambient.heatwave_anomaly_offset_c

        elif self.scenario_name == "scenario_5_combined_stress":
            # Concurrent compute amplification + spikes + heatwave anomaly
            k = SCENARIOS.combined.workload_scaling_factor_k
            scaled_w = np.clip(k * base_workload, 0.0, 1.0)
            transformed_workload = self._inject_workload_spikes(
                scaled_w,
                magnitude=SCENARIOS.combined.spike_magnitude,
                duration=SCENARIOS.combined.spike_duration_steps,
                seed=SCENARIOS.combined.random_seed,
            )
            transformed_ambient = base_ambient + SCENARIOS.combined.ambient_anomaly_offset_c

        # Apply chronological partition slice if requested
        start_idx, end_idx = self._resolve_split_indices(len(transformed_workload))
        self.workload_trajectory = transformed_workload[start_idx:end_idx]
        self.ambient_trajectory = transformed_ambient[start_idx:end_idx]
        self.trajectory_length = len(self.workload_trajectory)

    def _resolve_split_indices(self, total_len: int) -> tuple[int, int]:
        """Resolve slice range based on configured temporal split."""
        if self.split_name == "all":
            return 0, total_len
        elif self.split_name in ("train", "training"):
            return SCENARIOS.splits.train_start_step, SCENARIOS.splits.train_end_step + 1
        elif self.split_name in ("val", "validation"):
            return SCENARIOS.splits.val_start_step, SCENARIOS.splits.val_end_step + 1
        elif self.split_name in ("test", "testing"):
            return SCENARIOS.splits.test_start_step, min(SCENARIOS.splits.test_end_step + 1, total_len)
        else:
            raise ValueError(
                f"Invalid split '{self.split_name}'. Must be 'all', 'train', 'val', or 'test'."
            )

    @staticmethod
    def _inject_workload_spikes(
        base_w: np.ndarray,
        magnitude: float = 0.25,
        duration: int = 4,
        seed: int = 42,
    ) -> np.ndarray:
        """Inject deterministic transient workload spikes across diurnal cycles."""
        w = base_w.copy()
        rng = np.random.default_rng(seed)
        steps_per_day = 288
        n_days = int(np.ceil(len(w) / steps_per_day))

        for day in range(n_days):
            day_start = day * steps_per_day
            day_end = min((day + 1) * steps_per_day, len(w))
            day_len = day_end - day_start
            if day_len < 40:
                continue

            # Subdivide each day into two halves to place 2 distinct non-overlapping bursts
            mid = day_start + day_len // 2

            # Window 1 (first half of day)
            w1_low = day_start + 15
            w1_high = max(w1_low + 1, mid - duration - 10)
            if w1_high > w1_low:
                s1 = int(rng.integers(w1_low, w1_high))
                w[s1 : s1 + duration] = np.clip(w[s1 : s1 + duration] + magnitude, 0.0, 1.0)

            # Window 2 (second half of day)
            w2_low = mid + 15
            w2_high = max(w2_low + 1, day_end - duration - 10)
            if w2_high > w2_low:
                s2 = int(rng.integers(w2_low, w2_high))
                w[s2 : s2 + duration] = np.clip(w[s2 : s2 + duration] + magnitude, 0.0, 1.0)

        return w

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the simulation environment to initial conditions.

        Returns:
            tuple of (initial_observation, initial_info)
        """
        self.current_step = 0
        self.T_internal = float(self.thermal_params.initial_internal_temperature)
        self.cooling_level = float(self.thermal_params.initial_cooling_level)
        self.previous_internal_temperature = float(self.thermal_params.initial_internal_temperature)
        self.temperature_change = 0.0

        w_0 = float(self.workload_trajectory[0])
        t_amb_0 = float(self.ambient_trajectory[0])

        f_amb = compute_ambient_derating_factor(t_amb_0, self.thermal_params)
        cool_eff = compute_cooling_effect(self.cooling_level, t_amb_0, self.thermal_params)
        cop = compute_cop(t_amb_0, self.thermal_params, self.cop_params)
        q_cool, p_cool, e_cool = compute_cooling_power_and_energy(
            self.cooling_level, cop, self.cop_params, self.thermal_params.timestep_hours
        )
        zone = classify_ashrae_zone(self.T_internal, self.thermal_params)
        trend = classify_temperature_trend(self.temperature_change, self.thermal_params)

        observation = self._build_observation(self.T_internal, w_0, t_amb_0, self.cooling_level, 0.0)

        info: dict[str, Any] = {
            "workload": w_0,
            "ambient_temperature": t_amb_0,
            "internal_temperature": self.T_internal,
            "cooling_level": self.cooling_level,
            "ambient_derating_factor": f_amb,
            "cooling_effect": cool_eff,
            "cop": cop,
            "cooling_power": p_cool,
            "cooling_energy": e_cool,
            "temperature_change": self.temperature_change,
            "temperature_trend": trend.value,
            "ashrae_zone": zone.value,
            "step_index": self.current_step,
            "scenario": self.scenario_name,
            "reward": 0.0,
            "safety_penalty": 0.0,
            "severe_overheat_penalty": 0.0,
            "overcooling_penalty": 0.0,
            "energy_penalty": 0.0,
            "action_churn_penalty": 0.0,
            "reward_breakdown": {
                "total_reward": 0.0,
                "safety_penalty": 0.0,
                "severe_overheat_penalty": 0.0,
                "overcooling_penalty": 0.0,
                "energy_penalty": 0.0,
                "action_churn_penalty": 0.0,
            },
        }
        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one simulation step of duration Delta_t = 300 seconds.

        Args:
            action: Discrete action index in {0, 1, 2, 3, 4},
                    representing cooling adjustments in [-0.20, -0.10, 0.00, +0.10, +0.20].

        Returns:
            tuple of (observation, reward_placeholder, terminated, truncated, info)
        """
        # Strict action validation
        if not isinstance(action, (int, np.integer)) or int(action) not in (0, 1, 2, 3, 4):
            raise ValueError(
                f"Invalid action {action!r}. Action must be an integer in {{0, 1, 2, 3, 4}}, "
                f"corresponding to adjustments {self.ACTION_DELTAS}."
            )

        if self.current_step >= self.trajectory_length:
            raise RuntimeError(
                f"Cannot step in an exhausted environment (step {self.current_step} >= {self.trajectory_length}). "
                "Call env.reset() before stepping again."
            )

        # 1. Update cooling state
        action_delta = self.ACTION_DELTAS[int(action)]
        self.cooling_level = float(np.clip(round(self.cooling_level + action_delta, 4), 0.0, 1.0))

        # 2. Ingest current step external forcing variables
        w_current = float(self.workload_trajectory[self.current_step])
        t_amb_current = float(self.ambient_trajectory[self.current_step])

        # 3. Compute thermodynamic transition
        f_amb = compute_ambient_derating_factor(t_amb_current, self.thermal_params)
        cool_eff = compute_cooling_effect(self.cooling_level, t_amb_current, self.thermal_params)
        t_next, delta_t = compute_thermal_step(
            self.T_internal,
            w_current,
            t_amb_current,
            self.cooling_level,
            self.thermal_params,
            disturbance=0.0,  # Deterministic baseline
        )

        # 4. Compute COP, power, and energy
        cop = compute_cop(t_amb_current, self.thermal_params, self.cop_params)
        q_cool, p_cool, e_cool = compute_cooling_power_and_energy(
            self.cooling_level, cop, self.cop_params, self.thermal_params.timestep_hours
        )

        # 5. Classify thermal health and rate of change
        zone = classify_ashrae_zone(t_next, self.thermal_params)
        trend = classify_temperature_trend(delta_t, self.thermal_params)

        # 6. Update internal continuous state
        self.previous_internal_temperature = self.T_internal
        self.T_internal = t_next
        self.temperature_change = delta_t
        step_index_recorded = self.current_step
        self.current_step += 1

        # 7. Check episode termination
        terminated = self.current_step >= self.trajectory_length
        truncated = False

        # 8. Compute formal multi-objective reward for Phase 2C
        step_reward, reward_breakdown = calculate_reward(
            internal_temperature=t_next,
            cooling_energy=e_cool,
            action_delta=action_delta,
            config=self.reward_config,
        )

        # 9. Next observation
        if not terminated:
            next_w = float(self.workload_trajectory[self.current_step])
            next_amb = float(self.ambient_trajectory[self.current_step])
        else:
            next_w = w_current
            next_amb = t_amb_current

        observation = self._build_observation(
            self.T_internal, next_w, next_amb, self.cooling_level, self.temperature_change
        )

        info: dict[str, Any] = {
            "workload": w_current,
            "ambient_temperature": t_amb_current,
            "internal_temperature": self.T_internal,
            "cooling_level": self.cooling_level,
            "ambient_derating_factor": f_amb,
            "cooling_effect": cool_eff,
            "cop": cop,
            "cooling_power": p_cool,
            "cooling_energy": e_cool,
            "temperature_change": self.temperature_change,
            "temperature_trend": trend.value,
            "ashrae_zone": zone.value,
            "step_index": step_index_recorded,
            "scenario": self.scenario_name,
            "reward": step_reward,
            "safety_penalty": reward_breakdown["safety_penalty"],
            "severe_overheat_penalty": reward_breakdown["severe_overheat_penalty"],
            "overcooling_penalty": reward_breakdown["overcooling_penalty"],
            "energy_penalty": reward_breakdown["energy_penalty"],
            "action_churn_penalty": reward_breakdown["action_churn_penalty"],
            "reward_breakdown": reward_breakdown,
        }

        return observation, step_reward, terminated, truncated, info

    @staticmethod
    def _build_observation(
        t_internal: float,
        workload: float,
        ambient_temperature: float,
        cooling_level: float,
        temperature_change: float,
    ) -> np.ndarray:
        """Construct the 5-dimensional NumPy float32 observation vector."""
        return np.array(
            [
                float(t_internal),
                float(workload),
                float(ambient_temperature),
                float(cooling_level),
                float(temperature_change),
            ],
            dtype=np.float32,
        )
