# Raw Data Directory (`data/raw/`)

This directory is designated for original, immutable raw dataset files downloaded from official repositories.

> [!IMPORTANT]
> **No raw files are downloaded in Phase 1A.** Ingestion, checksum verification, and storage will occur strictly during Phase 1B/1C.

---

## 1. Primary Workload Dataset: Alibaba Cluster Trace 2018 (Processed)

- **Source**: Zenodo Record [14564847](https://zenodo.org/records/14564847) (DOI: `10.5281/zenodo.14564847`)
- **Original Upstream Repository**: [alibaba/clusterdata (cluster-trace-v2018)](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018)
- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **Expected Filename**: `machine_usage_days_1_to_8_grouped_300_seconds.csv`
- **Temporal Resolution**: 300 seconds (5 minutes)
- **Time Span**: 8 days (Days 1 to 8)
- **Verified Schema**:
  | Field | Type | Domain | Description |
  | :--- | :--- | :--- | :--- |
  | `cpu_util_percent` | integer/float | `[0, 100]` | Average CPU utilization across whole cluster |
  | `mem_util_percent` | integer/float | `[0, 100]` | Average memory utilization across whole cluster |
  | `net_in` | float | `[0, 100]` | Normalized incoming network traffic |
  | `net_out` | float | `[0, 100]` | Normalized outgoing network traffic |
  | `disk_io_percent` | float | `[0, 100]` | Disk I/O utilization (abnormal values: -1 or 101) |
- **Primary Utilization Metric**: `cpu_util_percent` (to be normalized to `[0.0, 1.0]`).
- **Role in CoolRL**: Provides empirical, non-stationary compute workload trajectories to drive simulated server thermal dissipation.

---

## 2. Secondary Workload Dataset: Google Cluster Workload Traces 2019

- **Source**: Zenodo Record [14564847](https://zenodo.org/records/14564847) (Queried via BigQuery from Google Cluster Traces 2019)
- **Original Upstream Repository**: [google/cluster-data](https://github.com/google/cluster-data)
- **Reference**: [Google Cluster Workload Traces 2019](https://research.google/tools/datasets/google-cluster-workload-traces-2019/)
- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **Expected Filename**: `instance_usage_grouped_300_seconds_month.csv`
- **Temporal Resolution**: 300 seconds (5 minutes)
- **Time Span**: 30 days (1 month)
- **Verified Schema**:
  | Field | Type | Domain | Description |
  | :--- | :--- | :--- | :--- |
  | `avg_cpu` | float | `[0.0, 1.0]` | Average CPU utilization across the whole data center |
  | `avg_mem` | float | `[0.0, 1.0]` | Average memory utilization across the whole data center |
  | `avg_assigned_mem` | float | `[0.0, 1.0]` | Average assigned memory ratio |
  | `avg_cycles_per_instruction` | float | `[0, _]` | Average CPI across workloads |
- **Primary Utilization Metric**: `avg_cpu` (already scaled in `[0.0, 1.0]`).
- **Role in CoolRL**: Independent secondary validation workload to evaluate whether policies generalize beyond the Alibaba trace characteristics.

---

## Data Ingestion Rules
1. Files placed in this directory must remain **read-only** and unmodified.
2. Raw data files are tracked in `.gitignore` to avoid checking large binary/CSV files into Git.
3. All transformations, cleaning, and resampling must output exclusively to `data/processed/`.
