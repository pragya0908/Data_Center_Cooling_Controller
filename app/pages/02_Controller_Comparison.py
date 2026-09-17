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
scenario_df = df[(df["scenario"] == scenario) & (df["split"] == "test")]

if not scenario_df.empty:
    st.dataframe(
        scenario_df[["controller", "cumulative_reward", "mean_internal_temperature", "percentage_gt_27", "total_cooling_energy"]],
        use_container_width=True,
        hide_index=True
    )

st.markdown("### Time Series Analysis")
st.markdown("Compare the temperature and energy consumption trajectories for the selected scenario on the test split.")

@st.cache_data(show_spinner="Simulating time series...")
def generate_time_series(scenario_name: str) -> pd.DataFrame:
    from src.baselines.fixed_cooling import FixedCoolingController
    from src.baselines.rule_based import RuleBasedController
    from src.agents.q_learning import QLearningAgent
    from src.agents.state_discretizer import StateDiscretizer
    from src.environment.datacenter_env import DataCenterEnv
    
    fixed_ctrl = FixedCoolingController(target_cooling=0.50)
    rule_ctrl = RuleBasedController()
    model_path = PROJECT_ROOT / "results" / "training" / "q_learning_normal_train.npz"
    
    if not model_path.exists():
        return pd.DataFrame()
        
    q_agent = QLearningAgent.load(model_path)
    discretizer = StateDiscretizer()
    
    controllers = {
        "fixed_cooling_0.50": fixed_ctrl,
        "rule_based": rule_ctrl,
        "q_learning": q_agent
    }
    
    records = []
    for ctrl_name, ctrl in controllers.items():
        env = DataCenterEnv(scenario=scenario_name, split="test")
        obs, info = env.reset(seed=42)
        
        records.append({
            "controller": ctrl_name,
            "step": info.get("step_index", 0),
            "internal_temperature": info["internal_temperature"],
            "cooling_energy": info.get("cooling_energy", 0.0),
        })
        
        done = False
        while not done:
            if ctrl_name == "q_learning":
                state_id = discretizer.encode(obs)
                action = ctrl.choose_action(state_id, training=False)
            else:
                action = ctrl.choose_action(obs)
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            records.append({
                "controller": ctrl_name,
                "step": info.get("step_index", 0),
                "internal_temperature": info["internal_temperature"],
                "cooling_energy": info["cooling_energy"],
            })
            
    return pd.DataFrame(records)

plot_df = generate_time_series(scenario)

if not plot_df.empty:
    fig = plot_comparison_metrics(plot_df, scenario)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No time series data available for the selected scenario.")
