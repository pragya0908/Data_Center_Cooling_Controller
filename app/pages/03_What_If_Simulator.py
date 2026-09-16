"""Page 3: What-If Simulator."""

import streamlit as st
import pandas as pd
from src.dashboard.rl_simulator import LiveSimulator
from src.dashboard.visualization import plot_simulation_metrics
from src.data.scenario_config import SCENARIOS

st.set_page_config(page_title="What-If Simulator", page_icon="🧪", layout="wide")

st.title("🧪 What-If Simulator")
st.markdown("""
Test the Q-learning agent's behavior under extreme, synthetic conditions.
Select an anomaly scenario to simulate a full episode and evaluate how the agent maintains thermal safety.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Configuration")
    scenario_choice = st.selectbox(
        "Select Anomaly Scenario",
        options=["high_workload", "workload_spikes", "high_ambient", "combined_stress", "normal"]
    )
    
    st.markdown("#### Scenario Details")
    if scenario_choice == "high_workload":
        st.info(f"Multiplies baseline workload by {SCENARIOS.high_workload.scaling_factor_k}x to simulate sustained high compute demand.")
    elif scenario_choice == "workload_spikes":
        st.info(f"Injects transient workload spikes of +{SCENARIOS.spikes.spike_magnitude} magnitude to simulate sudden traffic bursts.")
    elif scenario_choice == "high_ambient":
        st.info(f"Adds a +{SCENARIOS.ambient.heatwave_anomaly_offset_c}°C heatwave anomaly to the ambient temperature.")
    elif scenario_choice == "combined_stress":
        st.info("Simultaneously applies workload scaling, transient spikes, and a heatwave anomaly.")
    else:
        st.info("Baseline unaltered real traces.")
        
    run_button = st.button("Run Full Simulation", type="primary")

with col2:
    st.markdown("### Simulation Results")
    if run_button:
        with st.spinner(f"Running simulation for '{scenario_choice}' scenario..."):
            sim = LiveSimulator(scenario=scenario_choice, split="test")
            # Auto-play until termination
            sim.auto_play(n_steps=sim.env.trajectory_length)
            df = sim.get_history_df()
            
            if not df.empty:
                st.success("Simulation complete!")
                
                # Show key metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Max Internal Temp", f"{df['internal_temperature'].max():.2f} °C")
                m2.metric("Total Cooling Energy", f"{df['cooling_energy'].sum():.2f} kWh")
                m3.metric("Safety Violations (>27°C)", f"{(df['internal_temperature'] > 27).sum()} steps")
                
                fig = plot_simulation_metrics(df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Simulation failed to produce data.")
    else:
        st.info("Click 'Run Full Simulation' to see results.")
