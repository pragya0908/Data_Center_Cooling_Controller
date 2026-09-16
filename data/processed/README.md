# Processed Data Directory (`data/processed/`)

This directory will store curated, cleaned, and normalized datasets produced by the preprocessing pipeline (`src/data/`).

> [!NOTE]
> No processed files exist yet. Transformations will be executed during the preprocessing phase.

---

## Intended Transformation Pipeline

```
Raw Workload Data (Alibaba / Google)
               +
External Meteorological Data (NASA POWER)
               ↓
    [Data Validation & Cleaning]
  - Remove/impute missing timestamps
  - Filter anomalous readings (e.g. disk I/O values of -1 or 101)
               ↓
    [Normalization & Scaling]
  - Map CPU utilization to [0.0, 1.0]
  - Validate ambient temperatures in Celsius
               ↓
    [Temporal Synchronization]
  - Align 300-second workload intervals with interpolated hourly weather observations
  - Standardize time-step indices for episode generation
               ↓
   `data/processed/` Curated Artifacts
```

---

## Processed Dataset Specifications

### 1. Primary Simulation Input: Alibaba Cleaned Trajectory
- **Expected Artifact**: `alibaba_workload_300s.csv` (or `.parquet`)
- **Schema**:
  - `timestamp_step`: Integer index (`0, 1, 2, ...`)
  - `workload`: Float in `[0.0, 1.0]` representing normalized compute load
  - `mem_util`: Float in `[0.0, 1.0]` representing memory utilization

### 2. Secondary Validation Input: Google Cleaned Trajectory
- **Expected Artifact**: `google_workload_300s.csv` (or `.parquet`)
- **Schema**:
  - `timestamp_step`: Integer index (`0, 1, 2, ...`)
  - `workload`: Float in `[0.0, 1.0]` representing normalized compute load

### 3. Ambient Temperature Scenario
- **Expected Artifact**: `bengaluru_ambient_hourly.csv`
- **Schema**:
  - `timestamp_utc`: ISO-8601 timestamp string
  - `ambient_temp_c`: Float (°C) representing 2-meter ambient temperature `T2M`

---

## Scientific Modeling Distinction

The datasets stored here represent **external drivers** only:
- **Real inputs**: Workload utilization ($W_t \in [0.0, 1.0]$) and ambient temperature ($T^{\text{amb}}_t \in \mathbb{R}$).
- **Simulation variables**: Internal server-room temperatures ($T^{\text{room}}_t$), cooling response ($\Delta C_t$), energy expenditure ($E_t$), and transition rewards ($R_t$) are dynamically evaluated inside the simulation environment during policy interaction. They are **never** statically pre-computed or stored as supervised labels.
