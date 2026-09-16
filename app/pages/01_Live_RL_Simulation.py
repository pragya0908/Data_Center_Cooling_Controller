"""Page 1: Live RL Simulation."""

import streamlit as st
from src.dashboard.rl_simulator import LiveSimulator
from src.dashboard.visualization import plot_simulation_metrics

st.set_page_config(page_title="Live RL Simulation", page_icon="▶️", layout="wide")

st.title("▶️ Live RL Simulation")
st.markdown("Watch the trained Q-learning agent make cooling decisions in real time using the actual `DataCenterEnv`.")

# Initialize the simulator in session state if it doesn't exist
if "simulator" not in st.session_state:
    st.session_state.simulator = LiveSimulator(scenario="normal", split="test")

sim = st.session_state.simulator

# Sidebar controls
st.sidebar.header("Simulation Controls")

scenario_choice = st.sidebar.selectbox(
    "Scenario",
    options=["normal", "high_workload", "workload_spikes", "high_ambient", "combined_stress"],
    index=["normal", "high_workload", "workload_spikes", "high_ambient", "combined_stress"].index(sim.scenario)
)

# Handle scenario change
if scenario_choice != sim.scenario:
    st.session_state.simulator = LiveSimulator(scenario=scenario_choice, split="test")
    sim = st.session_state.simulator

col1, col2, col3 = st.sidebar.columns(3)
if col1.button("Step (1x)"):
    sim.step()
if col2.button("Play (10x)"):
    sim.auto_play(10)
if col3.button("Reset"):
    sim.reset()

st.sidebar.markdown(f"**Current Step**: {sim.env.current_step} / {sim.env.trajectory_length}")

# Check termination
if sim.terminated:
    st.warning("Simulation episode has terminated. Please click Reset to start over.")

# Get history and visualize
df = sim.get_history_df()

if not df.empty:
    # Display current state metrics
    latest = df.iloc[-1]
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Internal Temp", f"{latest['internal_temperature']:.2f} °C")
    m2.metric("Workload", f"{latest['workload']:.2f}")
    m3.metric("Cooling Level", f"{latest['cooling_level']:.2f}")
    m4.metric("Ambient Temp", f"{latest['ambient_temperature']:.2f} °C")
    
    # Plot
    fig = plot_simulation_metrics(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show history table (latest 5 steps)
    st.markdown("### Recent Step History")
    display_cols = ["step", "internal_temperature", "workload", "ambient_temperature", "cooling_level", "action", "reward"]
    st.dataframe(df[display_cols].tail(5), use_container_width=True)
else:
    st.info("No data to display. Click Step or Play to begin.")
