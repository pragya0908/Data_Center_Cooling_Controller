"""Preprocessing and Exploratory Data Analysis (EDA) for CoolRL datasets.

Transforms raw workload and meteorological observations into normalized,
temporally-aligned arrays for the CoolRL simulation environment.
Generates diagnostic EDA plots saved in results/data_eda/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.weather_api import parse_nasa_power_t2m_to_dataframe

RAW_DIR = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
EDA_DIR = _PROJECT_ROOT / "results" / "data_eda"


def preprocess_alibaba_workload(
    input_csv: Path | str = RAW_DIR / "alibaba" / "machine_usage_days_1_to_8_grouped_300_seconds.csv",
    output_csv: Path | str = PROCESSED_DIR / "alibaba_workload_300s.csv",
) -> pd.DataFrame:
    """Clean and normalize Alibaba 2018 cluster machine usage data.

    Transformations:
    - Maps `cpu_util_percent` from [0, 100] to `workload` in [0.0, 1.0].
    - Maps `mem_util_percent` to `mem_util` in [0.0, 1.0].
    - Adds integer `step`, `elapsed_seconds`, and `elapsed_hours` indices.
    - Preserves chronological sequence.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_path)
    df = df_raw.copy()

    # Normalize utilization metrics to [0.0, 1.0]
    df["workload"] = np.clip(df["cpu_util_percent"] / 100.0, 0.0, 1.0)
    df["mem_util"] = np.clip(df["mem_util_percent"] / 100.0, 0.0, 1.0)

    # Construct temporal indices (each row represents a 300s interval)
    df.insert(0, "step", range(len(df)))
    df.insert(1, "elapsed_seconds", df["step"] * 300)
    df.insert(2, "elapsed_hours", df["elapsed_seconds"] / 3600.0)

    # Select and order columns
    cols_to_keep = [
        "step",
        "elapsed_seconds",
        "elapsed_hours",
        "workload",
        "mem_util",
        "cpu_util_percent",
        "mem_util_percent",
        "net_in",
        "net_out",
        "disk_io_percent",
    ]
    df = df[cols_to_keep]
    df.to_csv(output_path, index=False)
    print(f"[PREPROCESS] Processed Alibaba workload: {len(df)} rows -> {output_path}")
    return df


def preprocess_google_workload(
    input_csv: Path | str = RAW_DIR / "google" / "instance_usage_grouped_300_seconds_month.csv",
    output_csv: Path | str = PROCESSED_DIR / "google_workload_300s.csv",
) -> pd.DataFrame:
    """Clean and standardize Google 2019 cluster instance usage data."""
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_path)
    df = df_raw.copy()

    # avg_cpu is already in [0.0, 1.0]
    df["workload"] = np.clip(df["avg_cpu"], 0.0, 1.0)

    df.insert(0, "step", range(len(df)))
    df.insert(1, "elapsed_seconds", df["step"] * 300)
    df.insert(2, "elapsed_hours", df["elapsed_seconds"] / 3600.0)

    cols_to_keep = [
        "step",
        "elapsed_seconds",
        "elapsed_hours",
        "workload",
        "avg_cpu",
        "avg_mem",
        "avg_assigned_mem",
        "avg_cycles_per_instruction",
    ]
    df = df[cols_to_keep]
    df.to_csv(output_path, index=False)
    print(f"[PREPROCESS] Processed Google workload: {len(df)} rows -> {output_path}")
    return df


def preprocess_nasa_weather(
    input_json: Path | str = RAW_DIR / "weather" / "nasa_power_hourly_T2M_20180501_20180530.json",
    output_hourly_csv: Path | str = PROCESSED_DIR / "bengaluru_weather_hourly.csv",
    output_aligned_300s_csv: Path | str = PROCESSED_DIR / "bengaluru_weather_300s_aligned.csv",
    align_steps: int = 2243,  # Matching Alibaba 8-day trace length
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse raw NASA POWER JSON, extract hourly T2M, and interpolate to 300s step grid."""
    input_path = Path(input_json)
    output_hourly = Path(output_hourly_csv)
    output_aligned = Path(output_aligned_300s_csv)
    output_hourly.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    df_hourly = parse_nasa_power_t2m_to_dataframe(payload)
    df_hourly["elapsed_hours"] = np.arange(len(df_hourly), dtype=float)
    df_hourly.to_csv(output_hourly, index=False)
    print(f"[PREPROCESS] Processed hourly weather: {len(df_hourly)} rows -> {output_hourly}")

    # Interpolate to 300s (5-minute) resolution matching the simulation step grid
    # Target steps = align_steps (300s intervals)
    target_elapsed_hours = np.arange(align_steps) * (300.0 / 3600.0)
    hourly_hours = df_hourly["elapsed_hours"].values
    hourly_temps = df_hourly["ambient_temp_c"].values

    interpolated_temps = np.interp(target_elapsed_hours, hourly_hours, hourly_temps)

    df_aligned = pd.DataFrame({
        "step": np.arange(align_steps),
        "elapsed_seconds": np.arange(align_steps) * 300,
        "elapsed_hours": target_elapsed_hours,
        "ambient_temp_c": np.round(interpolated_temps, 3),
    })
    df_aligned.to_csv(output_aligned, index=False)
    print(f"[PREPROCESS] Aligned 300s weather: {len(df_aligned)} steps -> {output_aligned}")

    return df_hourly, df_aligned


def generate_eda_plots(
    df_alibaba: pd.DataFrame,
    df_google: pd.DataFrame,
    df_weather_hourly: pd.DataFrame,
    df_weather_aligned: pd.DataFrame,
    output_dir: Path | str = EDA_DIR,
) -> list[Path]:
    """Generate exploratory data analysis visualization plots."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_plots: list[Path] = []

    # 1. Alibaba Workload Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_alibaba["workload"], bins=30, color="#1f77b4", edgecolor="black", alpha=0.7)
    ax.axvline(df_alibaba["workload"].mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {df_alibaba['workload'].mean():.3f}")
    ax.axvline(df_alibaba["workload"].median(), color="orange", linestyle="-.", linewidth=2,
               label=f"Median: {df_alibaba['workload'].median():.3f}")
    ax.set_title("Alibaba 2018 Cluster Workload Distribution (Normalized)", fontsize=13)
    ax.set_xlabel("Normalized Workload [0.0, 1.0]", fontsize=11)
    ax.set_ylabel("Frequency (300s Intervals)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plot1 = out_dir / "alibaba_workload_distribution.png"
    fig.tight_layout()
    fig.savefig(plot1, dpi=150)
    plt.close(fig)
    generated_plots.append(plot1)

    # 2. Alibaba Workload Over Time
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_alibaba["elapsed_hours"], df_alibaba["workload"], color="#1f77b4", linewidth=1.2, alpha=0.85)
    ax.set_title("Alibaba 2018 Cluster Workload Trajectory (8-Day Span)", fontsize=13)
    ax.set_xlabel("Elapsed Time (Hours)", fontsize=11)
    ax.set_ylabel("Normalized Workload [0.0, 1.0]", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    plot2 = out_dir / "alibaba_workload_timeline.png"
    fig.tight_layout()
    fig.savefig(plot2, dpi=150)
    plt.close(fig)
    generated_plots.append(plot2)

    # 3. Alibaba Workload Spike Analysis (Step-to-step delta)
    fig, ax = plt.subplots(figsize=(10, 5))
    workload_deltas = df_alibaba["workload"].diff().dropna()
    ax.plot(df_alibaba["elapsed_hours"].iloc[1:], workload_deltas, color="#d62728", linewidth=0.9, alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Alibaba 2018 Workload Dynamics (Step-to-Step Rate of Change)", fontsize=13)
    ax.set_xlabel("Elapsed Time (Hours)", fontsize=11)
    ax.set_ylabel("Delta Workload per 300s", fontsize=11)
    ax.grid(True, alpha=0.3)
    plot3 = out_dir / "alibaba_workload_spikes.png"
    fig.tight_layout()
    fig.savefig(plot3, dpi=150)
    plt.close(fig)
    generated_plots.append(plot3)

    # 4. Google Workload Over Time
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_google["elapsed_hours"], df_google["workload"], color="#2ca02c", linewidth=1.0, alpha=0.85)
    ax.set_title("Google Cluster Workload Trajectory (28-Day Span)", fontsize=13)
    ax.set_xlabel("Elapsed Time (Hours)", fontsize=11)
    ax.set_ylabel("Normalized Workload [0.0, 1.0]", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    plot4 = out_dir / "google_workload_timeline.png"
    fig.tight_layout()
    fig.savefig(plot4, dpi=150)
    plt.close(fig)
    generated_plots.append(plot4)

    # 5. NASA POWER Ambient Temperature Timeline
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_weather_hourly["elapsed_hours"], df_weather_hourly["ambient_temp_c"], color="#ff7f0e", linewidth=1.5)
    ax.set_title("NASA POWER Hourly Ambient Air Temperature T2M (Bengaluru, India - May 2018)", fontsize=13)
    ax.set_xlabel("Elapsed Time (Hours)", fontsize=11)
    ax.set_ylabel("Ambient Temperature (°C)", fontsize=11)
    ax.grid(True, alpha=0.3)
    plot5 = out_dir / "weather_ambient_temp_timeline.png"
    fig.tight_layout()
    fig.savefig(plot5, dpi=150)
    plt.close(fig)
    generated_plots.append(plot5)

    # 6. NASA POWER Ambient Temperature Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_weather_hourly["ambient_temp_c"], bins=25, color="#ff7f0e", edgecolor="black", alpha=0.7)
    ax.axvline(df_weather_hourly["ambient_temp_c"].mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {df_weather_hourly['ambient_temp_c'].mean():.2f}°C")
    ax.set_title("Ambient Temperature Distribution (Bengaluru Scenario)", fontsize=13)
    ax.set_xlabel("Temperature (°C)", fontsize=11)
    ax.set_ylabel("Hourly Observations Count", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plot6 = out_dir / "weather_ambient_temp_distribution.png"
    fig.tight_layout()
    fig.savefig(plot6, dpi=150)
    plt.close(fig)
    generated_plots.append(plot6)

    # 7. Dual Axis Coupled Scenario: Alibaba Workload + Aligned Ambient Temp
    fig, ax1 = plt.subplots(figsize=(13, 5))
    color_w = "#1f77b4"
    ax1.set_xlabel("Elapsed Time (Hours)", fontsize=11)
    ax1.set_ylabel("Normalized Workload [0, 1]", color=color_w, fontsize=11)
    ax1.plot(df_alibaba["elapsed_hours"], df_alibaba["workload"], color=color_w, linewidth=1.2, label="Workload W(t)")
    ax1.tick_params(axis="y", labelcolor=color_w)
    ax1.set_ylim(0.0, 1.0)

    ax2 = ax1.twinx()
    color_t = "#d95f02"
    ax2.set_ylabel("Ambient Temperature T_amb (°C)", color=color_t, fontsize=11)
    ax2.plot(df_weather_aligned["elapsed_hours"], df_weather_aligned["ambient_temp_c"],
             color=color_t, linestyle="--", linewidth=1.6, label="Ambient Temp T_amb(t)")
    ax2.tick_params(axis="y", labelcolor=color_t)

    plt.title("Coupled Simulation Scenario: Alibaba Workload & Bengaluru Weather", fontsize=13)
    fig.tight_layout()
    plot7 = out_dir / "workload_weather_aligned.png"
    fig.savefig(plot7, dpi=150)
    plt.close(fig)
    generated_plots.append(plot7)

    return generated_plots


def run_full_preprocessing() -> dict[str, Any]:
    """Execute complete cleaning, alignment, and EDA visualization pipeline."""
    print("=" * 80)
    print("CoolRL Phase 1B: Data Preprocessing & EDA Pipeline")
    print("=" * 80)

    df_ali = preprocess_alibaba_workload()
    df_goo = preprocess_google_workload()
    df_wea_hourly, df_wea_aligned = preprocess_nasa_weather(align_steps=len(df_ali))

    print("\nGenerating EDA diagnostic plots...")
    plots = generate_eda_plots(df_ali, df_goo, df_wea_hourly, df_wea_aligned)
    print(f"Generated {len(plots)} EDA plots in {EDA_DIR}:")
    for p in plots:
        print(f"  - {p.name}")

    return {
        "alibaba_rows": len(df_ali),
        "google_rows": len(df_goo),
        "weather_hourly_rows": len(df_wea_hourly),
        "weather_aligned_rows": len(df_wea_aligned),
        "plots": [str(p) for p in plots],
    }


if __name__ == "__main__":
    run_full_preprocessing()
