"""Comprehensive inspection and statistical analysis of acquired CoolRL datasets.

Performs rigorous profiling on:
1. Alibaba Cluster Trace 2018 (machine usage)
2. Google Cluster Workload Traces 2019 (instance usage)
3. NASA POWER Hourly Weather (ambient temperature for Bengaluru)

Calculates row/column counts, schema types, missing values, duplicates,
value distributions (min/max/mean/std), sampling intervals, and anomaly flags.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.weather_api import parse_nasa_power_t2m_to_dataframe


def inspect_dataframe(df: pd.DataFrame, dataset_name: str, file_path: Path) -> dict[str, Any]:
    """Perform a comprehensive statistical inspection on a pandas DataFrame."""
    file_size_bytes = file_path.stat().st_size if file_path.exists() else 0
    file_size_kb = file_size_bytes / 1024

    stats_dict: dict[str, Any] = {
        "dataset_name": dataset_name,
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_size_kb": round(file_size_kb, 2),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        summary[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "25%": float(df[col].quantile(0.25)),
            "50% (median)": float(df[col].median()),
            "75%": float(df[col].quantile(0.75)),
        }
    stats_dict["numerical_summary"] = summary

    return stats_dict


def run_full_inspection() -> dict[str, Any]:
    """Inspect all raw datasets and print a structured report."""
    raw_dir = _PROJECT_ROOT / "data" / "raw"
    results: dict[str, Any] = {}

    print("=" * 80)
    print("CoolRL Phase 1B: Raw Dataset Inspection Report")
    print("=" * 80)

    # 1. Alibaba 2018 Trace
    alibaba_file = raw_dir / "alibaba" / "machine_usage_days_1_to_8_grouped_300_seconds.csv"
    if alibaba_file.exists():
        df_ali = pd.read_csv(alibaba_file)
        res_ali = inspect_dataframe(df_ali, "Alibaba Cluster Trace 2018", alibaba_file)
        res_ali["sampling_interval"] = "300 seconds (5 minutes)"
        res_ali["time_span"] = f"{len(df_ali) * 300 / 3600:.2f} hours (~{len(df_ali) * 300 / 86400:.2f} days)"
        # Check for abnormal disk_io values documented in Zenodo (-1 or 101)
        abnormal_disk_io = int(((df_ali["disk_io_percent"] < 0) | (df_ali["disk_io_percent"] > 100)).sum())
        res_ali["anomalies"] = {
            "abnormal_disk_io_count": abnormal_disk_io,
            "cpu_util_bounds": [float(df_ali["cpu_util_percent"].min()), float(df_ali["cpu_util_percent"].max())],
        }
        results["alibaba"] = res_ali
        _print_dataset_report(res_ali)
    else:
        print(f"[MISSING] Alibaba file not found: {alibaba_file}")

    # 2. Google 2019 Trace
    google_file = raw_dir / "google" / "instance_usage_grouped_300_seconds_month.csv"
    if google_file.exists():
        df_goo = pd.read_csv(google_file)
        res_goo = inspect_dataframe(df_goo, "Google Cluster Workload Traces 2019", google_file)
        res_goo["sampling_interval"] = "300 seconds (5 minutes)"
        res_goo["time_span"] = f"{len(df_goo) * 300 / 3600:.2f} hours (~{len(df_goo) * 300 / 86400:.2f} days)"
        res_goo["anomalies"] = {
            "avg_cpu_bounds": [float(df_goo["avg_cpu"].min()), float(df_goo["avg_cpu"].max())],
            "avg_mem_bounds": [float(df_goo["avg_mem"].min()), float(df_goo["avg_mem"].max())],
        }
        results["google"] = res_goo
        _print_dataset_report(res_goo)
    else:
        print(f"[MISSING] Google file not found: {google_file}")

    # 3. NASA POWER Hourly Weather
    weather_json = raw_dir / "weather" / "nasa_power_hourly_T2M_20180501_20180530.json"
    if weather_json.exists():
        with weather_json.open("r", encoding="utf-8") as f:
            weather_payload = json.load(f)
        df_wea = parse_nasa_power_t2m_to_dataframe(weather_payload)
        res_wea = inspect_dataframe(df_wea, "NASA POWER Hourly Weather (Bengaluru)", weather_json)
        res_wea["sampling_interval"] = "1 hour (3600 seconds)"
        res_wea["time_span"] = f"{len(df_wea)} hours ({len(df_wea) / 24:.1f} days)"
        res_wea["start_time_utc"] = str(df_wea["timestamp_utc"].min())
        res_wea["end_time_utc"] = str(df_wea["timestamp_utc"].max())
        res_wea["anomalies"] = {
            "temp_bounds_celsius": [float(df_wea["ambient_temp_c"].min()), float(df_wea["ambient_temp_c"].max())],
        }
        results["weather"] = res_wea
        _print_dataset_report(res_wea)
    else:
        print(f"[MISSING] Weather file not found: {weather_json}")

    return results


def _print_dataset_report(res: dict[str, Any]) -> None:
    """Pretty-print a single dataset inspection report."""
    print("-" * 80)
    print(f"DATASET: {res['dataset_name']}")
    print(f"  File: {res['file_name']} ({res['file_size_kb']} KB)")
    print(f"  Rows: {res['num_rows']:,} | Columns: {res['num_columns']}")
    print(f"  Sampling Interval: {res.get('sampling_interval', 'N/A')}")
    print(f"  Time Span: {res.get('time_span', 'N/A')}")
    print(f"  Missing Values: {res['total_missing']} | Duplicate Rows: {res['duplicate_rows']}")
    print(f"  Columns: {res['columns']}")
    print("  Numerical Distributions:")
    for col, stat in res.get("numerical_summary", {}).items():
        print(
            f"    - {col:<26}: min={stat['min']:>8.3f}, max={stat['max']:>8.3f}, "
            f"mean={stat['mean']:>8.3f}, std={stat['std']:>8.3f}"
        )
    if "anomalies" in res:
        print(f"  Anomalies / Boundary Checks: {res['anomalies']}")


if __name__ == "__main__":
    run_full_inspection()
