# CoolRL Dataset Sources & Methodological Grounding

This document serves as the formal data manifest for the **CoolRL** project. It details the provenance, access methods, schemas, temporal resolutions, licenses, and specific operational roles for all real-world data sources incorporated into the research.

---

## 1. Primary Workload Dataset: Alibaba Cluster Trace 2018 (Processed)

- **Dataset Name**: Alibaba Cluster Trace 2018 (Machine Usage Subset)
- **Originating Organization**: Alibaba Group (Alibaba Cloud Infrastructure)
- **Authoritative Official Repository**: [alibaba/clusterdata (v2018)](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018)
- **Derivative Curated Source**: Zenodo Record [14564847](https://zenodo.org/records/14564847) (DOI: `10.5281/zenodo.14564847`)
- **Derivative Curator**: Fernández-Montes & Fernández Cerero, Universidad de Sevilla
- **Download / Retrieval Method**: Direct programmatic retrieval from Zenodo REST Content API (`https://zenodo.org/api/records/14564847/files/machine_usage_days_1_to_8_grouped_300_seconds.csv/content`)
- **Files Used**:
  - Raw: `data/raw/alibaba/machine_usage_days_1_to_8_grouped_300_seconds.csv` (197.2 KB)
  - Processed: `data/processed/alibaba_workload_300s.csv`
- **Variables Used**:
  - `cpu_util_percent` (Normalized to `workload` $\in [0.0, 1.0]$)
  - `mem_util_percent` (Normalized to `mem_util` $\in [0.0, 1.0]$)
  - `net_in`, `net_out`, `disk_io_percent` (Retained as auxiliary cluster metrics)
- **Units**: Percent (`%`) in raw trace, dimensionless ratio $[0.0, 1.0]$ in processed environment inputs
- **Temporal Resolution**: 300 seconds (5 minutes)
- **Time Span**: 2,243 intervals = 186.92 hours (~7.79 days)
- **License**: [Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Role in CoolRL**: Primary production compute workload trajectory $W(t)$ driving internal IT thermal heat generation inside the simulated server room.
- **Limitations**:
  - Represents whole-cluster aggregated utilization, not individual server rack hot spots.
  - Does not contain hardware fan power or server casing thermistor telemetry.

---

## 2. Secondary Workload Dataset: Google Cluster Workload Traces 2019

- **Dataset Name**: Google Cluster Workload Traces 2019 (Instance Usage Subset)
- **Originating Organization**: Google LLC
- **Authoritative Official Repository**: [google/cluster-data](https://github.com/google/cluster-data) / [Google Research Tools](https://research.google/tools/datasets/google-cluster-workload-traces-2019/)
- **Derivative Curated Source**: Zenodo Record [14564847](https://zenodo.org/records/14564847)
- **Download / Retrieval Method**: Direct programmatic retrieval from Zenodo REST Content API (`https://zenodo.org/api/records/14564847/files/instance_usage_grouped_300_seconds_month.csv/content`)
- **Files Used**:
  - Raw: `data/raw/google/instance_usage_grouped_300_seconds_month.csv` (593.1 KB)
  - Processed: `data/processed/google_workload_300s.csv`
- **Variables Used**:
  - `avg_cpu` (Directly mapped to `workload` $\in [0.0, 1.0]$)
  - `avg_mem`, `avg_assigned_mem`, `avg_cycles_per_instruction`
- **Units**: Normalized fraction $[0.0, 1.0]$ for CPU and memory; cycles per instruction for CPI
- **Temporal Resolution**: 300 seconds (5 minutes)
- **Time Span**: 8,064 intervals = 672 hours (28.0 days / 4 weeks)
- **License**: [Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Role in CoolRL**: Independent evaluation workload trajectory used to assess policy transferability and prevent overfitting to Alibaba-specific diurnal rhythms.
- **Limitations**:
  - Highly aggregated across diverse Borg workload tiers; contains no direct temperature measurements.

---

## 3. Real Meteorological Dataset: NASA POWER Hourly Surface Meteorology

- **Dataset Name**: NASA Prediction of Worldwide Energy Resources (POWER) Hourly Meteorology
- **Originating Organization**: National Aeronautics and Space Administration (NASA) Langley Research Center
- **Authoritative Official Portal**: [NASA POWER Portal](https://power.larc.nasa.gov/) / [NASA Open Data Catalog](https://data.nasa.gov/dataset/prediction-of-worldwide-energy-resources-power)
- **Download / Retrieval Method**: Automated REST API query via `src/data/weather_api.py` targeting `https://power.larc.nasa.gov/api/temporal/hourly/point`
- **Files Used**:
  - Raw Response: `data/raw/weather/nasa_power_hourly_T2M_20180501_20180530.json` (21.7 KB)
  - Query Parameters: `data/raw/weather/api_request_params.json`
  - Processed Hourly: `data/processed/bengaluru_weather_hourly.csv`
  - Aligned 300s: `data/processed/bengaluru_weather_300s_aligned.csv`
- **Variables Used**:
  - `T2M`: Near-surface air temperature at 2 meters above ground level
- **Units**: Degrees Celsius (°C)
- **Temporal Resolution**: 1 hour (3,600 seconds) in raw observations; linearly interpolated to 300-second steps in processed alignment
- **Time Span**: 720 hours (30 continuous days: May 1, 2018 to May 30, 2018)
- **Target Geographic Scenario**: Bengaluru, Karnataka, India (Latitude: `12.9716° N`, Longitude: `77.5946° E`)
- **License**: NASA Open Data Policy (Public Domain / Unrestricted academic use)
- **Role in CoolRL**: Supplies exogenous ambient temperature $T^{\text{amb}}(t)$, establishing realistic outdoor thermal boundary conditions and chiller lift requirements.
- **Limitations**:
  - Regional synoptic weather station observations do not account for microclimatic hot air recirculations around cooling towers.

---

## 4. Fundamental Methodological Boundaries

```
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                                 COOLRL DATA TAXONOMY                                  ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║  REAL OBSERVED EMPIRICAL DATA (Exogenous Drivers)                                      ║
║  ├── Compute Workload Trajectory W(t)       [Source: Alibaba 2018 / Google 2019]       ║
║  └── Ambient Outdoor Temperature T_amb(t)   [Source: NASA POWER - Bengaluru, India]   ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║  SIMULATION-DERIVED DATA (Controlled Physics Model & Policy)                          ║
║  ├── Server-Room Temperature T_room(t)      [Simulated thermal capacitance & exchange]║
║  ├── Cooling Actuation Delta ΔC(t)          [RL Agent Action in {-20%,-10%,0,+10%,+20%}]║
║  ├── Delivered Cooling Effort C(t)          [Simulated actuator level in [0.0, 1.0]]  ║
║  ├── Chiller & Fan Power P_cool(t)          [Thermodynamic COP energy model]          ║
║  ├── Thermal Safety / Overheating Metrics   [Evaluated against ASHRAE envelope (27°C)]║
║  └── Step Reward R(t)                       [Multi-objective optimization penalty]    ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
```

> [!CAUTION]
> **Scientific Attribution & Integrity Notice**:
> Under no circumstances does this project claim that the Alibaba cluster (Hangzhou, China) or Google cluster physically resided in Bengaluru, India. 
> Similarly, internal server-room temperatures ($T^{\text{room}}$), cooling power, and rewards are **never** described as real sensor measurements; they are explicitly derived from our physical thermal simulation driven by real workload and weather inputs.
