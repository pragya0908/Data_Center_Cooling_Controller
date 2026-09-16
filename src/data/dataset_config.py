"""Dataset metadata and configuration for CoolRL.

This module defines structured configurations, data schemas, and source
metadata for the real-world datasets that ground the CoolRL simulation
environment.

IMPORTANT METHODOLOGICAL PRINCIPLES:
1. Real-world datasets provide external workload and meteorological
   trajectories only.
2. The internal data-center thermal dynamics, cooling physics, energy
   consumption, and rewards are computed via simulation.
3. Datasets are NOT used as supervised learning labels for tabular Q-learning;
   the agent learns strictly through interaction with the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlibabaWorkloadMetadata:
    """Metadata and ingestion specifications for the Alibaba Cluster Trace 2018."""

    dataset_name: str = "Alibaba Cluster Trace 2018 (Processed)"
    original_source_url: str = (
        "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018"
    )
    processed_source_url: str = "https://zenodo.org/records/14564847"
    doi: str = "10.5281/zenodo.14564847"
    license: str = "Creative Commons Attribution 4.0 International (CC-BY 4.0)"

    # Targeted processed file from Zenodo repository
    target_file: str = "machine_usage_days_1_to_8_grouped_300_seconds.csv"
    temporal_resolution: str = "300 seconds (5 minutes)"
    duration: str = "8 days (days 1 to 8)"
    approx_records: int = 2304  # 8 days * 24 hours * 12 intervals/hr

    # Verified columns from Zenodo dataset documentation
    columns: dict[str, str] = field(
        default_factory=lambda: {
            "cpu_util_percent": "Average CPU utilization of the data center [0, 100]",
            "mem_util_percent": "Average memory utilization of the data center [0, 100]",
            "net_in": "Normalized incoming network traffic [0, 100]",
            "net_out": "Normalized outgoing network traffic [0, 100]",
            "disk_io_percent": "Disk I/O utilization [0, 100] (abnormal values: -1, 101)",
        }
    )

    # Primary variable mapped to CoolRL environment workload
    primary_variable: str = "cpu_util_percent"
    normalized_range: tuple[float, float] = (0.0, 1.0)

    # Intended role in CoolRL
    intended_use: str = (
        "Provides realistic, non-stationary compute workload trajectories to "
        "drive heat generation in the data-center simulation."
    )

    # Preprocessing requirements to be applied in Phase 1B/1C
    preprocessing_requirements: list[str] = field(
        default_factory=lambda: [
            "Validate and filter abnormal values (e.g., disk_io_percent in {-1, 101})",
            "Handle any missing timestamps or null readings via forward-fill or linear interpolation",
            "Normalize cpu_util_percent from [0, 100] to [0.0, 1.0]",
            "Resample/align time steps to the CoolRL simulation step interval",
        ]
    )

    limitations: list[str] = field(
        default_factory=lambda: [
            "Represents aggregate cluster-level utilization rather than individual rack/server thermals",
            "Temporal granularity is aggregated to 300-second windows in the processed release",
            "Workload trace does not include physical server thermal or fan telemetry",
        ]
    )


@dataclass(frozen=True)
class GoogleWorkloadMetadata:
    """Metadata and ingestion specifications for the Google Cluster Data."""

    dataset_name: str = "Google Cluster Workload Traces 2019 (Processed Instance Usage)"
    original_source_url: str = "https://github.com/google/cluster-data"
    reference_url: str = (
        "https://research.google/tools/datasets/google-cluster-workload-traces-2019/"
    )
    processed_source_url: str = "https://zenodo.org/records/14564847"
    license: str = "Creative Commons Attribution 4.0 International (CC-BY 4.0)"

    # Targeted processed file from Zenodo repository
    target_file: str = "instance_usage_grouped_300_seconds_month.csv"
    temporal_resolution: str = "300 seconds (5 minutes)"
    duration: str = "30 days (1 month)"

    # Verified columns from Zenodo dataset documentation
    columns: dict[str, str] = field(
        default_factory=lambda: {
            "avg_cpu": "Average CPU utilization across the whole data center [0.0, 1.0]",
            "avg_mem": "Average memory utilization across the whole data center [0.0, 1.0]",
            "avg_assigned_mem": "Average assigned memory ratio [0.0, 1.0]",
            "avg_cycles_per_instruction": "Average cycles per instruction (CPI) [0, _]",
        }
    )

    primary_variable: str = "avg_cpu"
    normalized_range: tuple[float, float] = (0.0, 1.0)

    intended_use: str = (
        "Serves as an independent secondary workload source to evaluate controller "
        "generalization and ensure policies do not overfit exclusively to Alibaba traces."
    )

    limitations: list[str] = field(
        default_factory=lambda: [
            "Already normalized to [0, 1] across heterogeneous machine architectures",
            "Does not contain direct temperature or cooling power measurements",
        ]
    )


@dataclass(frozen=True)
class NasaPowerWeatherMetadata:
    """Metadata and ingestion specifications for NASA POWER meteorological data."""

    dataset_name: str = "NASA POWER Hourly Meteorological Data"
    portal_url: str = "https://power.larc.nasa.gov/"
    open_data_url: str = (
        "https://data.nasa.gov/dataset/prediction-of-worldwide-energy-resources-power"
    )
    api_endpoint: str = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    license: str = "NASA Open Data Policy (Public Domain / Free for academic use)"

    # Primary meteorological parameter
    parameter: str = "T2M"
    parameter_description: str = "Near-Surface Air Temperature at 2 meters above ground level"
    units: str = "degrees Celsius (°C)"
    temporal_resolution: str = "Hourly (60 minutes)"

    # Scenario coordinates (Bengaluru, India)
    location_name: str = "Bengaluru, India"
    latitude: float = 12.9716
    longitude: float = 77.5946

    # API parameters
    api_community: str = "RE"  # Renewable Energy community
    api_format: str = "JSON"

    intended_use: str = (
        "Provides realistic, non-stationary outdoor ambient temperature trajectories "
        "ambient_temp(t) that dictate natural heat dissipation and chiller thermal lift."
    )

    # Academic & scientific attribution distinction
    methodological_note: str = (
        "Bengaluru weather represents an external real-world environmental scenario "
        "used to drive ambient temperature in the simulation. Neither the Alibaba nor "
        "the Google workload traces physically originated in Bengaluru; this is explicitly "
        "described as real-world workload traces combined with real-world meteorological "
        "observations to construct a data-grounded simulation environment."
    )

    limitations: list[str] = field(
        default_factory=lambda: [
            "Hourly resolution requires interpolation when simulation steps run at sub-hour frequencies",
            "Macro-scale weather measurements do not reflect localized data-center exhaust recirculations",
            "Date range alignment with workload traces to be verified during ingestion",
        ]
    )


@dataclass(frozen=True)
class DatasetManifest:
    """Complete dataset manifest and path conventions for CoolRL."""

    project_root: Path = Path(__file__).resolve().parent.parent.parent
    raw_data_dir: Path = project_root / "data" / "raw"
    processed_data_dir: Path = project_root / "data" / "processed"
    external_data_dir: Path = project_root / "data" / "external"

    alibaba: AlibabaWorkloadMetadata = field(default_factory=AlibabaWorkloadMetadata)
    google: GoogleWorkloadMetadata = field(default_factory=GoogleWorkloadMetadata)
    nasa_power: NasaPowerWeatherMetadata = field(default_factory=NasaPowerWeatherMetadata)

    # Explicit pipeline variable boundary
    real_world_inputs: list[str] = field(
        default_factory=lambda: [
            "workload(t) [from Alibaba or Google cluster trace]",
            "ambient_temp(t) [from NASA POWER meteorological observations]",
        ]
    )

    simulation_generated_variables: list[str] = field(
        default_factory=lambda: [
            "internal_temp(t) [simulated data-center temperature]",
            "cooling_action(t) [discrete cooling adjustment in {-20%, -10%, 0%, +10%, +20%}]",
            "cooling_level(t) [effective cooling effort in [0.0, 1.0]]",
            "cooling_energy(t) [energy consumption model based on cooling level & chiller lift]",
            "reward(t) [balance of thermal safety, energy penalty, and boundary violations]",
        ]
    )


# Default singleton configuration
MANIFEST = DatasetManifest()
