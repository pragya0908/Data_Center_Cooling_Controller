"""Page 2: Controller Comparison."""

import streamlit as st
import pandas as pd
from pathlib import Path
from src.dashboard.visualization import plot_comparison_metrics

st.set_page_config(page_title="Controller Comparison", page_icon="📊", layout="wide")

st.title("📊 Controller Comparison")
st.markdown("Evaluate the Q-learning agent against Fixed Cooling and Rule-based baselines across the 5 experimental scenarios.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "results" / "evaluation"
RESULTS_FILE = EVAL_DIR / "controller_comparison.csv"
SUMMARY_FILE = EVAL_DIR / "controller_comparison_summary.csv"

if not RESULTS_FILE.exists() or not SUMMARY_FILE.exists():
    st.error("Evaluation results not found. Please ensure Phase 4 evaluation has been run.")
    st.stop()

# Load data
df = pd.read_csv(RESULTS_FILE)
summary_df = pd.read_csv(SUMMARY_FILE)

# Scenario selection
scenario = st.selectbox(
    "Select Scenario to View",
    options=df["scenario"].unique(),
    index=0
)

st.markdown(f"### Results for: **{scenario.replace('_', ' ').title()}**")

# Display metrics
scenario_df = summary_df[summary_df["scenario"] == scenario]

if not scenario_df.empty:
    st.dataframe(
        scenario_df[["controller", "mean_cumulative_reward", "mean_internal_temperature", "percentage_gt_27", "mean_cooling_energy"]],
        use_container_width=True,
        hide_index=True
    )

st.markdown("### Time Series Analysis")
st.markdown("Compare the temperature and energy consumption trajectories for the selected scenario on the test split.")

# Filter test split for the specific scenario
plot_df = df[(df["scenario"] == scenario) & (df["split"] == "test")]

if not plot_df.empty:
    fig = plot_comparison_metrics(plot_df, scenario)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No time series data available for the selected scenario.")
