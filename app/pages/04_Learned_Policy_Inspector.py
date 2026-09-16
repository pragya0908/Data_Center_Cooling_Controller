"""Page 4: Learned Policy Inspector."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from src.agents.q_learning import QLearningAgent
from src.agents.state_discretizer import StateDiscretizer

st.set_page_config(page_title="Learned Policy Inspector", page_icon="🧠", layout="wide")

st.title("🧠 Learned Policy Inspector")
st.markdown("Introspect the actual learned Q-table to understand the agent's decision boundaries.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "results" / "training" / "q_learning_normal_train.npz"

if not MODEL_PATH.exists():
    st.error("Model not found. Please ensure Phase 3 training has been run.")
    st.stop()

agent = QLearningAgent.load(MODEL_PATH)
discretizer = StateDiscretizer()
policy = agent.get_policy()

st.markdown("### Q-Value Heatmap Slice")
st.markdown("Select fixed values for 3 of the 5 state dimensions to view the Q-values for the remaining 2 dimensions.")

col1, col2, col3 = st.columns(3)
with col1:
    fixed_workload = st.selectbox("Workload Bin", options=[0, 1, 2, 3], format_func=lambda x: ["< 0.33", "0.33 - 0.45", "0.45 - 0.55", ">= 0.55"][x])
with col2:
    fixed_ambient = st.selectbox("Ambient Temp Bin", options=[0, 1, 2, 3], format_func=lambda x: ["< 23°C", "23 - 27°C", "27 - 31°C", ">= 31°C"][x])
with col3:
    fixed_trend = st.selectbox("Temp Trend Bin", options=[0, 1, 2], format_func=lambda x: ["Cooling", "Stable", "Warming"][x])

action_choice = st.selectbox("Action to visualize", options=[0, 1, 2, 3, 4], format_func=lambda x: discretizer.ACTION_DESCRIPTIONS[x])

# Build heatmap data: Temp vs Cooling
temp_labels = ["< 18°C", "18-21°C", "21-24°C", "24-27°C", ">= 27°C"]
cooling_labels = ["Low (<25%)", "Med-Low", "Med-High", "High (>=75%)"]

heatmap_data = np.zeros((5, 4))
for t in range(5):
    for c in range(4):
        state_id = discretizer.encode_bins(t, fixed_workload, fixed_ambient, c, fixed_trend)
        heatmap_data[t, c] = agent.q_table[state_id, action_choice]

fig = go.Figure(data=go.Heatmap(
    z=heatmap_data,
    x=cooling_labels,
    y=temp_labels,
    colorscale="Viridis",
    colorbar=dict(title="Q-Value")
))

fig.update_layout(
    title="Q-Values: Internal Temperature vs Cooling Level",
    xaxis_title="Cooling Level Bin",
    yaxis_title="Internal Temperature Bin",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Optimal Policy Map")
st.markdown("Shows the best action (highest Q-value) for the slice above.")
optimal_policy = np.zeros((5, 4))
for t in range(5):
    for c in range(4):
        state_id = discretizer.encode_bins(t, fixed_workload, fixed_ambient, c, fixed_trend)
        optimal_policy[t, c] = policy[state_id]

fig2 = go.Figure(data=go.Heatmap(
    z=optimal_policy,
    x=cooling_labels,
    y=temp_labels,
    colorscale="RdBu",
    colorbar=dict(title="Action Index", tickvals=[0,1,2,3,4], ticktext=["-20%", "-10%", "0%", "+10%", "+20%"])
))

fig2.update_layout(
    title="Optimal Action Policy",
    xaxis_title="Cooling Level Bin",
    yaxis_title="Internal Temperature Bin",
    height=500
)

st.plotly_chart(fig2, use_container_width=True)
