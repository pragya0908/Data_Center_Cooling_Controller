"""CoolRL Streamlit Dashboard Main Entrypoint."""

import streamlit as st
import sys
from pathlib import Path

# Ensure src can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="CoolRL Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("❄️ CoolRL: Data-Driven Adaptive Data Center Cooling")
st.markdown("### Reinforcement Learning for Thermal Optimization")

st.markdown("""
Welcome to the interactive CoolRL research dashboard!

This dashboard is the final interactive artifact of **Phase 6**, built upon the fully validated
and frozen research pipeline (Phases 1-5).

### Dashboard Capabilities
Use the sidebar on the left to navigate through the modules:

1. **Live RL Simulation**: Watch the actual trained Q-learning agent control the `DataCenterEnv` in real-time step-by-step.
2. **Controller Comparison**: Compare the RL agent against fixed and rule-based baselines across 5 experimental scenarios.
3. **What-If Simulator**: Create custom anomaly scenarios (heatwaves, workload spikes) and test the agent's generalization on the fly.
4. **Learned Policy Inspector**: Introspect the actual learned Q-table to understand the agent's decision boundaries.
5. **Experimental Results**: View the aggregated Phase 5 statistical results on hyperparameter sensitivity and independent test-set generalization.

### Project Context
- **Environment**: A lumped-capacitance thermal model (`DataCenterEnv`) parameterized by ASHRAE guidelines.
- **Data**: Driven by real Alibaba 2018 cluster traces and NASA POWER Bengaluru weather data.
- **Agent**: Tabular Q-Learning (960 discrete states, 5 discrete actions) trained to prioritize thermal safety above all else.

*Use the sidebar to begin exploring.*
""")

st.info("👈 Select a page from the sidebar to start!")
