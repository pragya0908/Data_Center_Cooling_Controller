"""Page 5: Experimental Results."""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Experimental Results", page_icon="📈", layout="wide")

st.title("📈 Phase 5 Experimental Results")
st.markdown("Review the statistical robustness, hyperparameter sensitivity, and out-of-distribution generalization results of the Q-learning agent.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = PROJECT_ROOT / "results" / "experiments"

def load_data(filename):
    path = EXP_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None

hp_df = load_data("hyperparameter_sensitivity.csv")
seed_df = load_data("seed_robustness.csv")
cross_df = load_data("cross_scenario.csv")
google_df = load_data("generalization_google.csv")

tab1, tab2, tab3, tab4 = st.tabs(["Hyperparameter Sensitivity", "Seed Robustness", "Cross-Scenario", "Google Trace Generalization"])

with tab1:
    st.markdown("### Hyperparameter Sensitivity")
    if hp_df is not None:
        st.dataframe(hp_df, use_container_width=True, hide_index=True)
        # Plot Alpha vs Cumulative Reward grouped by Gamma
        fig = px.line(hp_df, x="alpha", y="cumulative_reward", color="gamma", title="Cumulative Reward vs Learning Rate (Alpha) by Discount Factor (Gamma)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data not found.")

with tab2:
    st.markdown("### Seed Robustness")
    if seed_df is not None:
        st.dataframe(seed_df, use_container_width=True, hide_index=True)
        fig = px.box(seed_df, y="cumulative_reward", title="Distribution of Cumulative Reward across Training Seeds")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data not found.")

with tab3:
    st.markdown("### Cross-Scenario Generalization")
    st.markdown("Agent trained on Normal scenario tested across all stress scenarios.")
    if cross_df is not None:
        st.dataframe(cross_df, use_container_width=True, hide_index=True)
        fig = px.bar(cross_df, x="test_scenario", y="cumulative_reward", title="Cumulative Reward by Test Scenario")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data not found.")

with tab4:
    st.markdown("### Google Trace Generalization (Zero-Shot)")
    st.markdown("Agent evaluated on independent, unseen Google cluster trace.")
    if google_df is not None:
        st.dataframe(google_df, use_container_width=True, hide_index=True)
        fig = px.bar(google_df, x="controller", y=["total_cooling_energy", "steps_gt_27"], barmode="group", title="Cooling Energy and Safety Violations on Google Trace")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data not found.")
