# CoolRL Data Dictionary

This document details all data fields across raw and processed datasets in the CoolRL repository, including physical units, valid ranges, schema types, and operational mappings.

---

## 1. Alibaba 2018 Cluster Machine Usage

### Raw Schema (`data/raw/alibaba/machine_usage_days_1_to_8_grouped_300_seconds.csv`)

| Column Name | Data Type | Units | Valid Domain | Missing Strategy | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cpu_util_percent` | `float64` | `%` | `[0.0, 100.0]` | Reject/Interpolate if null | Average cluster-wide CPU utilization over the 300s window |
| `mem_util_percent` | `float64` | `%` | `[0.0, 100.0]` | Reject/Interpolate if null | Average cluster-wide memory utilization over the 300s window |
| `net_in` | `float64` | `%` (norm.) | `[0.0, 100.0]` | Forward fill | Normalized incoming network bandwidth utilization |
| `net_out` | `float64` | `%` (norm.) | `[0.0, 100.0]` | Forward fill | Normalized outgoing network bandwidth utilization |
| `disk_io_percent` | `float64` | `%` | `[0.0, 100.0]` (flag `-1, 101`) | Filter abnormal | Disk I/O activity percentage; abnormal flags (-1, 101) filtered |

### Processed Schema (`data/processed/alibaba_workload_300s.csv`)

| Column Name | Data Type | Units | Valid Domain | Mapping / Derivation |
| :--- | :--- | :--- | :--- | :--- |
| `step` | `int64` | Index | $[0, 2242]$ | Sequential simulation time-step index ($t = 0, 1, 2, \dots$) |
| `elapsed_seconds` | `int64` | Seconds ($s$) | $[0, 672600]$ | Time elapsed since trace start: $\text{step} \times 300$ |
| `elapsed_hours` | `float64` | Hours ($h$) | $[0.0, 186.83]$ | Elapsed time in fractional hours: $\text{elapsed\_seconds} / 3600$ |
| `workload` | `float64` | Ratio | $[0.0, 1.0]$ | **Primary RL input**: $\text{clip}(\text{cpu\_util\_percent} / 100.0, 0, 1)$ |
| `mem_util` | `float64` | Ratio | $[0.0, 1.0]$ | Normalized memory utilization: $\text{clip}(\text{mem\_util\_percent} / 100.0, 0, 1)$ |
| `cpu_util_percent` | `float64` | `%` | $[0.0, 100.0]$ | Retained raw metric for diagnostic reference |
| `mem_util_percent` | `float64` | `%` | $[0.0, 100.0]$ | Retained raw metric for diagnostic reference |
| `net_in` | `float64` | `%` (norm.) | $[0.0, 100.0]$ | Retained raw metric for diagnostic reference |
| `net_out` | `float64` | `%` (norm.) | $[0.0, 100.0]$ | Retained raw metric for diagnostic reference |
| `disk_io_percent` | `float64` | `%` | $[0.0, 100.0]$ | Retained raw metric for diagnostic reference |

---

## 2. Google Cluster Workload Traces 2019

### Raw Schema (`data/raw/google/instance_usage_grouped_300_seconds_month.csv`)

| Column Name | Data Type | Units | Valid Domain | Missing Strategy | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `avg_cpu` | `float64` | Ratio | `[0.0, 1.0]` | Reject if null | Average cluster-wide CPU utilization over the 300s window |
| `avg_mem` | `float64` | Ratio | `[0.0, 1.0]` | Forward fill | Average memory utilization across instances |
| `avg_assigned_mem`| `float64` | Ratio | `[0.0, 1.0]` | Forward fill | Average assigned memory ratio |
| `avg_cycles_per_instruction` | `float64` | CPI | `[0.0, \infty)` | Forward fill | Microarchitectural cycles per instruction metric |

### Processed Schema (`data/processed/google_workload_300s.csv`)

| Column Name | Data Type | Units | Valid Domain | Mapping / Derivation |
| :--- | :--- | :--- | :--- | :--- |
| `step` | `int64` | Index | $[0, 8063]$ | Sequential simulation time-step index ($t = 0, 1, \dots, 8063$) |
| `elapsed_seconds` | `int64` | Seconds ($s$) | $[0, 2418900]$ | Time elapsed since trace start: $\text{step} \times 300$ |
| `elapsed_hours` | `float64` | Hours ($h$) | $[0.0, 671.92]$ | Elapsed time in fractional hours: $\text{elapsed\_seconds} / 3600$ |
| `workload` | `float64` | Ratio | $[0.0, 1.0]$ | **Secondary RL input**: $\text{clip}(\text{avg\_cpu}, 0.0, 1.0)$ |
| `avg_cpu` | `float64` | Ratio | $[0.0, 1.0]$ | Retained raw metric for diagnostic reference |
| `avg_mem` | `float64` | Ratio | $[0.0, 1.0]$ | Retained raw metric for diagnostic reference |
| `avg_assigned_mem`| `float64` | Ratio | $[0.0, 1.0]$ | Retained raw metric for diagnostic reference |
| `avg_cycles_per_instruction` | `float64` | CPI | $[0.0, \infty)$ | Retained raw metric for diagnostic reference |

---

## 3. NASA POWER Meteorological Data (Bengaluru Scenario)

### Raw Schema (`data/raw/weather/nasa_power_hourly_T2M_20180501_20180530.json`)

| Parameter / Key | Data Type | Units | Valid Domain | Missing Strategy | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `T2M` | Key-value (`YYYYMMDDHH`: `float`) | °C | `[-50.0, 60.0]` | Flag `-999.0` (None found) | Hourly near-surface air temperature at 2 meters |

### Processed Hourly Schema (`data/processed/bengaluru_weather_hourly.csv`)

| Column Name | Data Type | Units | Valid Domain | Description |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp_raw` | `string` | Format `YYYYMMDDHH` | `2018050100` to `2018053023` | Original NASA POWER temporal key |
| `timestamp_utc` | `datetime64[ns, UTC]` | ISO 8601 | 2018-05-01 00:00:00+00:00 to 2018-05-30 23:00:00+00:00 | Parsed UTC timestamp |
| `ambient_temp_c`| `float64` | °C | $[19.29, 36.19]$ | Direct 2m ambient air temperature observation |
| `elapsed_hours` | `float64` | Hours ($h$) | $[0.0, 719.0]$ | Cumulative hours elapsed from start of weather sequence |

### Processed Aligned Schema (`data/processed/bengaluru_weather_300s_aligned.csv`)

| Column Name | Data Type | Units | Valid Domain | Description |
| :--- | :--- | :--- | :--- | :--- |
| `step` | `int64` | Index | $[0, 2242]$ | Simulation step index matching Alibaba trace length |
| `elapsed_seconds` | `int64` | Seconds ($s$) | $[0, 672600]$ | Time elapsed since start: $\text{step} \times 300$ |
| `elapsed_hours` | `float64` | Hours ($h$) | $[0.0, 186.83]$ | Time elapsed in fractional hours: $\text{elapsed\_seconds} / 3600$ |
| `ambient_temp_c`| `float64` | °C | $[19.29, 36.19]$ | **Ambient temp input**: Linearly interpolated to 300s step grid |

---

## 4. Summary of Operational Mappings to CoolRL Environment

```
Real-World Data Variables                       CoolRL Environment Inputs
┌──────────────────────────────────────┐       ┌──────────────────────────────────────┐
│ Alibaba / Google: `workload`         │ ───►  │ `env.current_workload`               │
│ Normalized Compute Load in [0.0, 1.0]│       │ Drives thermal dissipation: Q_IT(t)  │
├──────────────────────────────────────┤       ├──────────────────────────────────────┤
│ NASA POWER: `ambient_temp_c`         │ ───►  │ `env.ambient_temp`                   │
│ Outdoor Air Temperature in °C        │       │ Governs external thermal boundary    │
└──────────────────────────────────────┘       └──────────────────────────────────────┘
```
