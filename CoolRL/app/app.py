import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.datacenter_env import DataCenterEnv
from agent.q_learning import QLearningAgent

st.set_page_config(page_title="CoolRL Dashboard", layout="wide", page_icon="🧊")

st.title("🧊 CoolRL — Data Center Cooling Controller")
st.markdown("> Reinforcement Learning based adaptive cooling controller for energy-efficient data center management.")

tabs = st.tabs(["🖥️ Dashboard Simulator", "📈 Training Results", "⚔️ Controller Comparison"])

# --- TAB 1: Simulator ---
with tabs[0]:
    st.header("Interactive Data Center Simulator")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.subheader("Environment Settings")
        init_temp = st.slider("Initial Temperature (°C)", 15.0, 45.0, 24.0, 0.5)
        workload_input = st.slider("Workload (%)", 0.0, 100.0, 50.0, 5.0)
        ambient_temp = st.slider("Ambient Temperature (°C)", 20.0, 35.0, 25.0, 1.0)
        init_cooling = st.slider("Initial Cooling (%)", 0.0, 100.0, 50.0, 10.0)
        steps_to_run = st.slider("Steps to Simulate", 1, 50, 10)
        
    with col2:
        st.subheader("Load Agent")
        
        agent_loaded = False
        try:
            q_table = joblib.load('models/q_table.pkl')
            st.success("Trained Q-Learning Agent Loaded!")
            agent = QLearningAgent(action_space=[0, 1, 2, 3, 4])
            agent.q_table = q_table
            agent_loaded = True
        except:
            st.error("No trained agent found. Run training script first.")
            
        if st.button("▶️ RUN SIMULATION", use_container_width=True) and agent_loaded:
            st.session_state['run_sim'] = True
        else:
            st.session_state['run_sim'] = False

    with col3:
        st.subheader("Simulation Results")
        if st.session_state.get('run_sim', False):
            env = DataCenterEnv()
            env.temperature = init_temp
            env.workload = workload_input
            env.ambient_temp = ambient_temp
            env.cooling_level = init_cooling
            
            # Run one step to get state
            state = env._get_state()
            
            history = []
            for _ in range(steps_to_run):
                action = agent.choose_action(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                
                action_text = {0: "-20%", 1: "-10%", 2: "Maintain", 3: "+10%", 4: "+20%"}[action]
                
                history.append({
                    "Action": action_text,
                    "Temp (°C)": next_state[0],
                    "Cooling (%)": next_state[3],
                    "Energy (kWh)": info['energy'],
                    "Reward": reward
                })
                state = next_state
                
            df_hist = pd.DataFrame(history)
            st.dataframe(df_hist, use_container_width=True)
            
            st.metric("Final Temperature", f"{state[0]} °C", delta=f"{round(state[0] - init_temp, 2)} °C" if init_temp else None, delta_color="inverse")
        else:
            st.info("Configure parameters and click 'RUN SIMULATION'")

# --- TAB 2: Training Results ---
with tabs[1]:
    st.header("RL Agent Training Progress")
    if os.path.exists('outputs/training_curve.png'):
        st.image('outputs/training_curve.png', caption="Q-Learning Episodic Reward", use_column_width=True)
    else:
        st.info("Training curve not generated yet.")

# --- TAB 3: Evaluation ---
with tabs[2]:
    st.header("Baseline vs RL Comparison")
    
    if os.path.exists('outputs/evaluation_metrics.csv'):
        df_metrics = pd.read_csv('outputs/evaluation_metrics.csv')
        st.dataframe(df_metrics, use_container_width=True)
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if os.path.exists('outputs/temperature_comparison.png'):
                st.image('outputs/temperature_comparison.png', use_column_width=True)
        with col_img2:
            if os.path.exists('outputs/energy_comparison.png'):
                st.image('outputs/energy_comparison.png', use_column_width=True)
    else:
        st.info("Evaluation metrics not generated yet. Run evaluation script.")
