"""Phase 1C Workload and Scenario Characterization Plot Generator.

This script loads the processed real datasets and generates diagnostic
visualizations for workload distributions, empirical variability, spike thresholds,
high-workload transformations, ambient temperature profiles, and experimental scenario
definitions under results/data_eda/phase_1c/.

IMPORTANT CONSTRAINTS:
- Only reads data from data/processed/.
- Does not modify any existing dataset.
- Does not implement RL, thermal models, or reward functions.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "data_eda" / "phase_1c"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling parameters
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the processed datasets."""
    alibaba = pd.read_csv(PROCESSED_DIR / "alibaba_workload_300s.csv")
    google = pd.read_csv(PROCESSED_DIR / "google_workload_300s.csv")
    weather_h = pd.read_csv(PROCESSED_DIR / "bengaluru_weather_hourly.csv")
    weather_300s = pd.read_csv(PROCESSED_DIR / "bengaluru_weather_300s_aligned.csv")
    return alibaba, google, weather_h, weather_300s


def plot_alibaba_distribution_regions(alibaba: pd.DataFrame) -> None:
    """Plot 1: Alibaba workload distribution with empirical percentile regions."""
    w = alibaba["workload"]
    p25 = w.quantile(0.25)
    p50 = w.quantile(0.50)
    p75 = w.quantile(0.75)
    p95 = w.quantile(0.95)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    n, bins, patches = ax.hist(w, bins=45, color="#2b5c8f", edgecolor="white", alpha=0.85, density=True)

    # Shading regions
    ax.axvspan(w.min(), p25, color="#6baed6", alpha=0.25, label=f"Low Workload (<P25: <{p25:.3f})")
    ax.axvspan(p25, p75, color="#74c476", alpha=0.25, label=f"Normal Workload (P25-P75: {p25:.3f}-{p75:.3f})")
    ax.axvspan(p75, p95, color="#fd8d3c", alpha=0.25, label=f"High Workload (P75-P95: {p75:.3f}-{p95:.3f})")
    ax.axvspan(p95, w.max(), color="#de2d26", alpha=0.25, label=f"Extreme Workload (>P95: >{p95:.3f})")

    # Lines for key statistics
    ax.axvline(w.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean: {w.mean():.4f}")
    ax.axvline(p50, color="#08519c", linestyle="-", linewidth=1.5, label=f"Median: {p50:.4f}")

    ax.set_title("Alibaba Cluster Trace 2018 — Empirical Workload Distribution & Characterized Regions")
    ax.set_xlabel("Normalized Workload (cpu_util_percent / 100)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(0.10, 0.85)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "alibaba_workload_distribution_regions.png")
    plt.close(fig)
    print("Saved: alibaba_workload_distribution_regions.png")


def plot_alibaba_timeline(alibaba: pd.DataFrame) -> None:
    """Plot 2: Alibaba workload timeline with train/val/test chronological partition."""
    w = alibaba["workload"]
    t = alibaba["elapsed_hours"]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(t, w, color="#1f77b4", linewidth=0.9, label="Observed 300s Workload")

    # Train / Val / Test vertical spans
    # Train: 0 to 1440 steps = 0 to 120 hours (Days 1 to 5)
    # Val: 1440 to 1728 steps = 120 to 144 hours (Day 6)
    # Test: 1728 to 2243 steps = 144 to 186.83 hours (Days 7 to 8)
    ax.axvspan(0, 120, color="#2ca02c", alpha=0.15, label="Training Split: Days 1–5 (Steps 0–1439, 64.2%)")
    ax.axvspan(120, 144, color="#ff7f0e", alpha=0.18, label="Validation Split: Day 6 (Steps 1440–1727, 12.8%)")
    ax.axvspan(144, t.max(), color="#d62728", alpha=0.18, label="Testing Split: Days 7–8 (Steps 1728–2242, 23.0%)")

    # Day markers
    for day in range(1, 9):
        ax.axvline(day * 24, color="gray", linestyle=":", alpha=0.5)

    ax.set_title("Alibaba Cluster Trace 2018 — 8-Day Workload Trajectory & Chronological Partitions")
    ax.set_xlabel("Elapsed Time (Hours)")
    ax.set_ylabel("Normalized Workload [0.0, 1.0]")
    ax.set_xlim(0, t.max())
    ax.set_ylim(0.10, 0.85)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "alibaba_workload_timeline.png")
    plt.close(fig)
    print("Saved: alibaba_workload_timeline.png")


def plot_alibaba_absolute_change(alibaba: pd.DataFrame) -> None:
    """Plot 3: Distribution of absolute first differences |W(t) - W(t-1)|."""
    w = alibaba["workload"]
    abs_delta = w.diff().dropna().abs()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    n, bins, _ = ax.hist(abs_delta, bins=50, color="#4a148c", edgecolor="white", alpha=0.85, density=True)

    # Key statistics
    mean_val = abs_delta.mean()
    p50_val = abs_delta.median()
    p90_val = abs_delta.quantile(0.90)
    p95_val = abs_delta.quantile(0.95)
    p99_val = abs_delta.quantile(0.99)

    ax.axvline(mean_val, color="#00bcd4", linestyle="--", linewidth=1.5, label=f"Mean |ΔW|: {mean_val:.4f}")
    ax.axvline(p50_val, color="#ffeb3b", linestyle="-", linewidth=1.5, label=f"Median |ΔW|: {p50_val:.4f}")
    ax.axvline(p95_val, color="#ff9800", linestyle="--", linewidth=1.5, label=f"95th Percentile: {p95_val:.4f}")
    ax.axvline(p99_val, color="#f44336", linestyle="-", linewidth=1.5, label=f"99th Percentile: {p99_val:.4f}")

    ax.set_title("Alibaba Workload Step-to-Step Absolute Variability Distribution |ΔW(t)|")
    ax.set_xlabel("Absolute Step Change: |W(t) - W(t-1)| (300s interval)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(0.0, 0.35)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "alibaba_workload_absolute_change.png")
    plt.close(fig)
    print("Saved: alibaba_workload_absolute_change.png")


def plot_spike_threshold(alibaba: pd.DataFrame) -> None:
    """Plot 4: Spike threshold analysis on first differences with timeline overlay."""
    w = alibaba["workload"]
    diff = w.diff().fillna(0)
    abs_diff = diff.abs()
    p95 = abs_diff.quantile(0.95)  # ~0.1222
    p99 = abs_diff.quantile(0.99)  # ~0.1849
    t = alibaba["elapsed_hours"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, dpi=150)

    # Upper panel: Workload trace with spike markers
    spike_idx = abs_diff >= p95
    ax1.plot(t, w, color="#1f77b4", linewidth=0.9, label="Observed Workload W(t)")
    ax1.scatter(t[spike_idx], w[spike_idx], color="#d62728", s=18, zorder=5, label=f"Rapid Shift (|ΔW| ≥ {p95:.3f}, N={spike_idx.sum()})")
    ax1.set_ylabel("Workload [0.0, 1.0]")
    ax1.set_title("Empirical Workload Spikes and Threshold Identification on Real Data")
    ax1.grid(True)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Lower panel: First difference ΔW(t)
    ax2.plot(t, diff, color="#555555", linewidth=0.7, label="Step Delta ΔW(t)")
    ax2.axhline(p95, color="#ff7f0e", linestyle="--", linewidth=1.2, label=f"+P95 Threshold (+{p95:.3f})")
    ax2.axhline(-p95, color="#ff7f0e", linestyle="--", linewidth=1.2, label=f"-P95 Threshold (-{p95:.3f})")
    ax2.axhline(p99, color="#d62728", linestyle=":", linewidth=1.5, label=f"±P99 Extreme Jump (±{p99:.3f})")
    ax2.axhline(-p99, color="#d62728", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("Elapsed Time (Hours)")
    ax2.set_ylabel("Step Difference ΔW(t)")
    ax2.set_ylim(-0.35, 0.35)
    ax2.grid(True)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "alibaba_spike_threshold.png")
    plt.close(fig)
    print("Saved: alibaba_spike_threshold.png")


def plot_high_workload_comparison(alibaba: pd.DataFrame) -> None:
    """Plot 5: Original vs transformed high workload W_high(t) = clip(1.25 * W(t), 0, 1)."""
    w = alibaba["workload"]
    k = 1.25
    w_high = np.clip(k * w, 0.0, 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Left panel: Histogram comparison
    bins = np.linspace(0.10, 1.05, 45)
    ax1.hist(w, bins=bins, color="#1f77b4", alpha=0.6, density=True, label=f"Original (Mean: {w.mean():.4f}, Max: {w.max():.4f})")
    ax1.hist(w_high, bins=bins, color="#d62728", alpha=0.6, density=True, label=f"High Workload k=1.25 (Mean: {w_high.mean():.4f}, Max: {w_high.max():.4f})")
    ax1.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="Physical Capacity Cap (1.0)")
    ax1.set_title("Workload Density: Original vs. High Workload Stress")
    ax1.set_xlabel("Workload")
    ax1.set_ylabel("Density")
    ax1.grid(True)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Right panel: 48-hour timeline comparison snippet
    sample_hours = 48
    mask = alibaba["elapsed_hours"] <= sample_hours
    t_sub = alibaba.loc[mask, "elapsed_hours"]
    w_sub = w[mask]
    w_high_sub = w_high[mask]

    ax2.plot(t_sub, w_sub, color="#1f77b4", linewidth=1.2, label="Original W(t)")
    ax2.plot(t_sub, w_high_sub, color="#d62728", linewidth=1.2, linestyle="--", label="High Workload W_high(t) [k=1.25]")
    ax2.set_title(f"First {sample_hours}h Trajectory Comparison")
    ax2.set_xlabel("Elapsed Time (Hours)")
    ax2.set_ylabel("Workload")
    ax2.set_ylim(0.15, 1.05)
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "high_workload_comparison.png")
    plt.close(fig)
    print("Saved: high_workload_comparison.png")


def plot_google_distribution(google: pd.DataFrame, alibaba: pd.DataFrame) -> None:
    """Plot 6: Google 28-day workload distribution compared to Alibaba."""
    w_google = google["workload"]
    w_alibaba = alibaba["workload"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bins = np.linspace(0.10, 0.85, 50)
    ax.hist(w_google, bins=bins, color="#2ca02c", edgecolor="white", alpha=0.7, density=True,
            label=f"Google Trace (28 Days, N=8,064, Mean: {w_google.mean():.4f}, Std: {w_google.std():.4f})")
    ax.hist(w_alibaba, bins=bins, color="#1f77b4", edgecolor="white", alpha=0.5, density=True,
            label=f"Alibaba Trace (8 Days, N=2,243, Mean: {w_alibaba.mean():.4f}, Std: {w_alibaba.std():.4f})")

    ax.axvline(w_google.mean(), color="#1b5e20", linestyle="-", linewidth=1.5, label=f"Google Mean: {w_google.mean():.4f}")
    ax.axvline(w_alibaba.mean(), color="#0d47a1", linestyle="--", linewidth=1.5, label=f"Alibaba Mean: {w_alibaba.mean():.4f}")

    ax.set_title("Cross-Dataset Workload Distribution: Google (Evaluation) vs. Alibaba (Training)")
    ax.set_xlabel("Normalized Workload [0.0, 1.0]")
    ax.set_ylabel("Probability Density")
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "google_workload_distribution.png")
    plt.close(fig)
    print("Saved: google_workload_distribution.png")


def plot_ambient_distribution(weather_h: pd.DataFrame, weather_300s: pd.DataFrame) -> None:
    """Plot 7: NASA POWER ambient temperature distribution (hourly vs 300s aligned)."""
    t_h = weather_h["ambient_temp_c"]
    t_300 = weather_300s["ambient_temp_c"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bins = np.linspace(18, 38, 40)
    ax.hist(t_h, bins=bins, color="#ff7f0e", alpha=0.6, density=True, edgecolor="white",
            label=f"Hourly May 2018 (30 Days, N=720, Mean: {t_h.mean():.2f}°C, Max: {t_h.max():.2f}°C)")
    ax.hist(t_300, bins=bins, color="#e377c2", alpha=0.6, density=True, edgecolor="white",
            label=f"300s Aligned Days 1–8 (N=2,243, Mean: {t_300.mean():.2f}°C, Max: {t_300.max():.2f}°C)")

    # Percentiles for hourly
    p25 = t_h.quantile(0.25)
    p75 = t_h.quantile(0.75)
    p90 = t_h.quantile(0.90)
    ax.axvline(p25, color="gray", linestyle=":", label=f"P25: {p25:.2f}°C")
    ax.axvline(p75, color="gray", linestyle="--", label=f"P75: {p75:.2f}°C")
    ax.axvline(p90, color="#d62728", linestyle="-", linewidth=1.5, label=f"P90 High Ambient: {p90:.2f}°C")

    ax.set_title("NASA POWER Bengaluru Meteorological Ambient Temperature Distribution (T2M)")
    ax.set_xlabel("Ambient Temperature (°C)")
    ax.set_ylabel("Probability Density")
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ambient_temperature_distribution.png")
    plt.close(fig)
    print("Saved: ambient_temperature_distribution.png")


def plot_high_ambient_threshold(weather_300s: pd.DataFrame) -> None:
    """Plot 8: High ambient threshold and thermal regime visualization."""
    temp = weather_300s["ambient_temp_c"]
    t = weather_300s["elapsed_hours"]
    p75 = temp.quantile(0.75)
    p90 = temp.quantile(0.90)
    p95 = temp.quantile(0.95)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(t, temp, color="#e65100", linewidth=1.1, label="Aligned Ambient Temperature T_amb(t)")

    # Threshold horizontal lines
    ax.axhline(p75, color="#fb8c00", linestyle="--", linewidth=1.2, label=f"Warm (P75: {p75:.2f}°C)")
    ax.axhline(p90, color="#d84315", linestyle="-", linewidth=1.5, label=f"High Ambient Threshold (P90: {p90:.2f}°C)")
    ax.axhline(p95, color="#b71c1c", linestyle=":", linewidth=1.5, label=f"Extreme Heatwave (P95: {p95:.2f}°C)")

    # Highlight high ambient intervals
    high_mask = temp >= p90
    ax.fill_between(t, p90, temp, where=high_mask, color="#ffab91", alpha=0.6, label="High Ambient Regime (≥P90)")

    ax.set_title("Ambient Temperature Trajectory & High-Ambient Thermal Stress Regime")
    ax.set_xlabel("Elapsed Time (Hours)")
    ax.set_ylabel("Outdoor Temperature (°C)")
    ax.set_xlim(0, t.max())
    ax.set_ylim(18, 38)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "high_ambient_threshold.png")
    plt.close(fig)
    print("Saved: high_ambient_threshold.png")


def plot_scenarios_overview(alibaba: pd.DataFrame, weather_300s: pd.DataFrame) -> None:
    """Plot 9: Synthetic visual overview demonstrating the 5 experimental scenarios."""
    # 48-hour representative slice
    mask = alibaba["elapsed_hours"] <= 48
    t = alibaba.loc[mask, "elapsed_hours"].values
    w_norm = alibaba.loc[mask, "workload"].values
    t_amb = weather_300s.loc[mask, "ambient_temp_c"].values

    # Scenario 2: High Workload (k = 1.25)
    w_high = np.clip(1.25 * w_norm, 0.0, 1.0)

    # Scenario 3: Workload Spikes (+0.25 on steps 72-76, 216-220, 360-365)
    w_spikes = w_norm.copy()
    spike_intervals = [(40, 44), (160, 165), (280, 286), (420, 425)]
    for start, end in spike_intervals:
        if end < len(w_spikes):
            w_spikes[start:end] = np.clip(w_spikes[start:end] + 0.25, 0.0, 1.0)

    # Scenario 4: High Ambient (+3.0°C thermal anomaly)
    t_amb_high = t_amb + 3.0

    # Scenario 5: Combined Stress (w_high + spikes + t_amb_high)
    w_combined = w_high.copy()
    for start, end in spike_intervals:
        if end < len(w_combined):
            w_combined[start:end] = np.clip(w_combined[start:end] + 0.20, 0.0, 1.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, dpi=150)

    # Panel 1: Workload across scenarios
    ax1.plot(t, w_norm, color="#1f77b4", linewidth=1.5, label="Scen 1: Normal Workload W(t)")
    ax1.plot(t, w_high, color="#ff7f0e", linewidth=1.2, linestyle="--", label="Scen 2: High Workload (k=1.25)")
    ax1.plot(t, w_spikes, color="#9467bd", linewidth=1.2, linestyle="-.", label="Scen 3: Workload Spikes (+0.25)")
    ax1.plot(t, w_combined, color="#d62728", linewidth=1.3, label="Scen 5: Combined Workload Stress")
    ax1.set_ylabel("Workload [0.0, 1.0]")
    ax1.set_title("CoolRL Five Experimental Scenarios (48-Hour Illustration)")
    ax1.set_ylim(0.15, 1.05)
    ax1.grid(True)
    ax1.legend(loc="upper right", framealpha=0.9, ncol=2)

    # Panel 2: Ambient Temperature across scenarios
    ax2.plot(t, t_amb, color="#2ca02c", linewidth=1.5, label="Scen 1-3: Real Ambient T_amb(t)")
    ax2.plot(t, t_amb_high, color="#d62728", linewidth=1.5, linestyle="--", label="Scen 4-5: High Ambient Stress (+3.0°C Heatwave)")
    ax2.set_xlabel("Elapsed Time (Hours)")
    ax2.set_ylabel("Ambient Temp (°C)")
    ax2.set_xlim(0, 48)
    ax2.set_ylim(18, 42)
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "scenarios_overview.png")
    plt.close(fig)
    print("Saved: scenarios_overview.png")


def main() -> None:
    print("Generating Phase 1C characterization plots...")
    alibaba, google, weather_h, weather_300s = load_datasets()
    plot_alibaba_distribution_regions(alibaba)
    plot_alibaba_timeline(alibaba)
    plot_alibaba_absolute_change(alibaba)
    plot_spike_threshold(alibaba)
    plot_high_workload_comparison(alibaba)
    plot_google_distribution(google, alibaba)
    plot_ambient_distribution(weather_h, weather_300s)
    plot_high_ambient_threshold(weather_300s)
    plot_scenarios_overview(alibaba, weather_300s)
    print("All 9 Phase 1C plots generated successfully in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
