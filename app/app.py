"""CoolRL Streamlit Dashboard."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for local imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from agent.q_learning import QLearningAgent
from environment.datacenter_env import DataCenterEnv

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoolRL Dashboard",
    page_icon="snowflake",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("CoolRL Dashboard")
    st.markdown(
        "Reinforcement-learning controller for data-center cooling.\n\n"
        "Use the tabs to explore training analytics or run a live simulation."
    )

# ── Tabs ─────────────────────────────────────────────────────────────────
tab_analytics, tab_simulator = st.tabs(["Analytics", "Live Simulator"])

# ── Analytics tab ────────────────────────────────────────────────────────
OUTPUTS_DIR = Path("outputs")

with tab_analytics:
    st.header("Training & Evaluation Analytics")

    # 1 — Training curve
    training_curve = OUTPUTS_DIR / "training_curve.png"
    if training_curve.exists():
        st.subheader("Training Curve")
        st.image(str(training_curve), caption="Episode vs Total Reward (Q-Learning)", width="stretch")
    else:
        st.warning(f"`{training_curve}` not found. Run training first.")

    st.divider()

    # 2 — Temperature comparison
    temp_plot = OUTPUTS_DIR / "temp_comparison.png"
    if temp_plot.exists():
        st.subheader("Temperature Comparison")
        st.image(str(temp_plot), caption="Temperature over time for all controllers", use_container_width=True)
    else:
        st.warning(f"`{temp_plot}` not found. Run evaluation first.")

    st.divider()

    # 3 — Energy comparison
    energy_plot = OUTPUTS_DIR / "energy_comparison.png"
    if energy_plot.exists():
        st.subheader("Energy Comparison")
        st.image(str(energy_plot), caption="Total cooling energy per controller", use_container_width=True)
    else:
        st.warning(f"`{energy_plot}` not found. Run evaluation first.")

# ── Live Simulator tab ───────────────────────────────────────────────────
MODEL_PATH = Path("models/q_table.pkl")
SIM_STEPS = 100

with tab_simulator:
    st.header("Live Simulator")
    st.markdown("Run a real-time simulation with the trained Q-Learning agent.")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        workload_mode = st.selectbox(
            "Workload Mode",
            options=["normal", "spike"],
            index=0,
        )
    with col_cfg2:
        st.metric("Simulation Length", f"{SIM_STEPS} steps")

    run_clicked = st.button("Run Simulation", type="primary")

    if run_clicked:
        if not MODEL_PATH.exists():
            st.error(f"Model not found at `{MODEL_PATH}`. Run training first.")
        else:
            # Load agent
            agent = QLearningAgent(num_actions=DataCenterEnv.NUM_ACTIONS)
            agent.load(MODEL_PATH)
            agent.epsilon = 0.0  # pure exploitation

            env = DataCenterEnv(workload_mode=workload_mode, seed=42)
            state = env._get_discrete_state()

            temps: list[float] = [env.current_temp]
            cooling_levels: list[float] = [env.current_cooling]

            progress_bar = st.progress(0, text="Simulating...")
            status_text = st.empty()

            for step in range(SIM_STEPS):
                action = agent.choose_action(state)
                state, reward, done = env.step(action)

                temps.append(env.current_temp)
                cooling_levels.append(env.current_cooling)

                # Update progress
                pct = (step + 1) / SIM_STEPS
                progress_bar.progress(pct, text=f"Step {step + 1}/{SIM_STEPS}")
                time.sleep(0.05)

                if done:
                    break

            progress_bar.progress(1.0, text="Simulation complete!")

            # ── Results ──────────────────────────────────────────────
            st.divider()
            st.subheader("Results")

            col_m1, col_m2, col_m3 = st.columns(3)
            total_energy = sum(cooling_levels)
            col_m1.metric("Total Energy", f"{total_energy:.1f}")
            col_m2.metric("Max Temp", f"{max(temps):.1f} C")
            col_m3.metric("Min Temp", f"{min(temps):.1f} C")

            st.subheader("Temperature Trajectory")
            st.line_chart({"Temperature (C)": temps})

            st.subheader("Cooling Level")
            st.line_chart({"Cooling Level": cooling_levels})
