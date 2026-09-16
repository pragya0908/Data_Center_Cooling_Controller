"""Experimental scenario configurations and reproducible parameters for CoolRL.

This module formalizes the data-driven experimental scenarios, temporal dataset
partitions, workload stress transformations, and recommended state discretization
parameters defined in Phase 1C.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. This module contains ONLY configuration parameters and definitions.
2. It does NOT implement simulation physics, thermal dynamics, or cooling equations.
3. It does NOT implement reinforcement learning, Q-tables, or agent policies.
4. All stress transformations are reproducible and parameterized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ScenarioType(str, Enum):
    """The five formal experimental scenarios evaluated in CoolRL."""

    SCENARIO_1_NORMAL = "scenario_1_normal"
    SCENARIO_2_HIGH_WORKLOAD = "scenario_2_high_workload"
    SCENARIO_3_WORKLOAD_SPIKES = "scenario_3_workload_spikes"
    SCENARIO_4_HIGH_AMBIENT = "scenario_4_high_ambient"
    SCENARIO_5_COMBINED_STRESS = "scenario_5_combined_stress"
    GENERALIZATION_GOOGLE = "generalization_google"


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Chronological, leakage-safe dataset partitioning specification."""

    dataset_name: str = "Alibaba Cluster Trace 2018 (300s Aggregated)"
    total_observations: int = 2243
    step_duration_seconds: int = 300
    total_duration_hours: float = 186.833333

    # Training partition: Days 1 to 5 (5 complete 24-hour diurnal cycles)
    train_start_step: int = 0
    train_end_step: int = 1439  # Inclusive slice: steps 0 to 1439 (1440 steps total)
    train_duration_hours: float = 120.0
    train_ratio: float = 1440 / 2243  # ~64.20%

    # Validation partition: Day 6 (1 complete 24-hour diurnal cycle, contains peak heat)
    val_start_step: int = 1440
    val_end_step: int = 1727  # Inclusive slice: steps 1440 to 1727 (288 steps total)
    val_duration_hours: float = 24.0
    val_ratio: float = 288 / 2243  # ~12.84%

    # Testing partition: Days 7 to 8 (~1.79 days, strictly held-out unseen evaluation)
    test_start_step: int = 1728
    test_end_step: int = 2242  # Inclusive slice: steps 1728 to 2242 (515 steps total)
    test_duration_hours: float = 42.833333
    test_ratio: float = 515 / 2243  # ~22.96%

    notes: str = (
        "Strict chronological partitioning respects temporal causality and avoids data leakage "
        "caused by high lag-1 autocorrelation (r = 0.8066). Partitions align to complete 288-step "
        "diurnal cycles."
    )


@dataclass(frozen=True)
class HighWorkloadConfig:
    """Parameters for Scenario 2: Controlled workload amplification."""

    # Mathematical formulation: W_high(t) = clip(scaling_factor * W(t), 0.0, 1.0)
    scaling_factor_k: float = 1.25

    # Empirical rationale:
    # Original Alibaba max: 0.7907, mean: 0.4018
    # Transformed max: 1.25 * 0.7907 = 0.9884 (< 1.0)
    # Transformed mean: 0.5022 (+25.0% continuous compute demand)
    # Saturation: exactly 0.00% clipped values, preventing artificial flat-top clipping
    clip_min: float = 0.0
    clip_max: float = 1.0
    expected_mean_shift_percent: float = +25.0


@dataclass(frozen=True)
class WorkloadSpikeConfig:
    """Parameters for Scenario 3: Transient workload bursts."""

    # Empirical variability baselines from Step 3:
    # 95th percentile |ΔW| = 0.1222
    # 99th percentile |ΔW| = 0.1849
    # Real trace maximum positive jump = +0.3187
    empirical_p95_step_jump: float = 0.1222
    empirical_p99_step_jump: float = 0.1849

    # Controlled spike injection parameters
    spike_magnitude: float = 0.25  # Injected transient jump (+25% utilization)
    min_spike_duration_steps: int = 3   # 15 minutes
    max_spike_duration_steps: int = 6   # 30 minutes
    default_spike_duration_steps: int = 4  # 20 minutes

    # Deterministic reproducibility
    random_seed: int = 42
    target_spikes_per_day: int = 2  # Approx 12-16 spikes across the 8-day trace
    clip_max: float = 1.0


@dataclass(frozen=True)
class HighAmbientConfig:
    """Parameters for Scenario 4: Extreme ambient thermal stress."""

    # Empirical ambient temperature baselines from NASA POWER May 2018 (Bengaluru):
    # Mean: 26.45°C, Median: 25.64°C, Min: 19.29°C, Max: 36.19°C
    # 75th percentile: 30.09°C
    # 90th percentile: 32.14°C
    # 95th percentile: 33.30°C
    ambient_baseline_mean_c: float = 26.45
    ambient_warm_p75_c: float = 30.09
    ambient_high_p90_c: float = 32.14
    ambient_extreme_max_c: float = 36.19

    # Method 1: Subsequence selection of high-temperature diurnal period
    # Day 6 (steps 1440 to 1728) experienced the highest recorded peak (36.19°C)
    peak_heatwave_day_indices: tuple[int, int] = (1440, 1728)

    # Method 2: Controlled additive thermal anomaly (heatwave stress shift)
    heatwave_anomaly_offset_c: float = 3.0  # Raises mean to ~29.5°C and peak to ~39.2°C


@dataclass(frozen=True)
class CombinedStressConfig:
    """Parameters for Scenario 5: Concurrent thermal and computational stress."""

    workload_scaling_factor_k: float = 1.25
    spike_magnitude: float = 0.20  # Slightly lower magnitude to accommodate scaled baseline
    spike_duration_steps: int = 4
    ambient_anomaly_offset_c: float = 3.0
    random_seed: int = 42


@dataclass(frozen=True)
class GoogleGeneralizationConfig:
    """Parameters for the independent Google cross-cluster evaluation."""

    dataset_name: str = "Google Cluster Workload Traces 2019 (Processed Instance Usage)"
    total_observations: int = 8064
    duration_days: int = 28
    step_duration_seconds: int = 300

    # Empirical statistics: Mean = 0.4728, Std = 0.0399, Min = 0.3203, Max = 0.5924
    mean_workload: float = 0.4728
    std_workload: float = 0.0399

    intended_role: str = (
        "Strictly evaluated as an out-of-distribution / cross-environment zero-shot test. "
        "The Q-learning policy is never trained on Google data; it is deployed to assess "
        "whether control policies generalize to different workload distributions and variances."
    )


@dataclass(frozen=True)
class RecommendedStateDiscretization:
    """Recommended candidate state discretization bins for future Phase 3.

    IMPORTANT NOTE:
    - Workload and Ambient bins are grounded directly in empirical data distributions.
    - Internal Temperature and Cooling Level bins are SIMULATION DESIGN CHOICES,
      as these variables are not present in the historical traces.
    """

    # --- OBSERVED DATA: Workload Discretization (Alibaba) ---
    # Range: [0.1613, 0.7907], Mean: 0.4018, P25: 0.3245, P75: 0.4658, P90: 0.5351
    # 4-bin empirical scheme:
    workload_bins_4: list[float] = field(
        default_factory=lambda: [0.00, 0.33, 0.45, 0.55, 1.00]
    )
    workload_labels_4: list[str] = field(
        default_factory=lambda: ["Low", "Normal", "High", "Peak/Extreme"]
    )

    # 5-bin empirical scheme (alternative higher resolution):
    workload_bins_5: list[float] = field(
        default_factory=lambda: [0.00, 0.30, 0.38, 0.48, 0.60, 1.00]
    )
    workload_labels_5: list[str] = field(
        default_factory=lambda: ["Idle/Low", "Normal-Low", "Normal-High", "High", "Extreme"]
    )

    # --- OBSERVED DATA: Ambient Temperature Discretization (NASA POWER Bengaluru) ---
    # Range: [19.29°C, 36.19°C], Mean: 26.45°C, P25: 22.74°C, P75: 30.09°C, P90: 32.14°C
    ambient_temp_bins: list[float] = field(
        default_factory=lambda: [0.0, 23.0, 27.0, 31.0, 50.0]
    )
    ambient_temp_labels: list[str] = field(
        default_factory=lambda: ["Cool (<23°C)", "Mild (23-27°C)", "Warm (27-31°C)", "Hot (≥31°C)"]
    )

    # --- SIMULATION DESIGN CHOICES (NOT OBSERVED DATA) ---
    # Grounded in ASHRAE Thermal Guidelines for Data Processing Environments (Class A1: 18-27°C)
    internal_temp_bins_design: list[float] = field(
        default_factory=lambda: [0.0, 18.0, 21.0, 24.0, 27.0, 100.0]
    )
    internal_temp_labels_design: list[str] = field(
        default_factory=lambda: [
            "Undercooled (<18°C)",
            "Target Low (18-21°C)",
            "Target High (21-24°C)",
            "Warning Warm (24-27°C)",
            "Critical Hot (≥27°C)",
        ]
    )

    # Cooling Level: simulation control effort in [0.0, 1.0]
    cooling_level_bins_design: list[float] = field(
        default_factory=lambda: [0.00, 0.25, 0.50, 0.75, 1.00]
    )
    cooling_level_labels_design: list[str] = field(
        default_factory=lambda: ["Low (0-25%)", "Medium-Low (25-50%)", "Medium-High (50-75%)", "High (75-100%)"]
    )

    # Temperature Trend: simulated derivative ΔT_int = T(t) - T(t-1)
    temp_trend_bins_design: list[float] = field(
        default_factory=lambda: [-100.0, -0.20, 0.20, 100.0]
    )
    temp_trend_labels_design: list[str] = field(
        default_factory=lambda: ["Cooling (ΔT < -0.2°C)", "Stable (-0.2°C ≤ ΔT ≤ +0.2°C)", "Warming (ΔT > +0.2°C)"]
    )


@dataclass(frozen=True)
class ScenarioMasterConfig:
    """Master configuration container for Phase 1C scenario definitions."""

    splits: TemporalSplitConfig = field(default_factory=TemporalSplitConfig)
    high_workload: HighWorkloadConfig = field(default_factory=HighWorkloadConfig)
    spikes: WorkloadSpikeConfig = field(default_factory=WorkloadSpikeConfig)
    ambient: HighAmbientConfig = field(default_factory=HighAmbientConfig)
    combined: CombinedStressConfig = field(default_factory=CombinedStressConfig)
    google_gen: GoogleGeneralizationConfig = field(default_factory=GoogleGeneralizationConfig)
    discretization: RecommendedStateDiscretization = field(
        default_factory=RecommendedStateDiscretization
    )


# Default singleton instance
SCENARIOS = ScenarioMasterConfig()
