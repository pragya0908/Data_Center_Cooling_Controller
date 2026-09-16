"""Visualization components for the Streamlit Dashboard using Plotly."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.environment.thermal_model_spec import DEFAULT_THERMAL_PARAMS


def plot_simulation_metrics(df: pd.DataFrame) -> go.Figure:
    """Plot the main live simulation metrics: Temperature, Workload, and Cooling.
    
    Args:
        df: DataFrame containing the simulation history.
        
    Returns:
        Plotly Figure object.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Internal Temperature vs Thresholds", "Workload & Cooling Level"),
        row_heights=[0.6, 0.4]
    )

    # Top Plot: Temperatures
    fig.add_trace(
        go.Scatter(
            x=df["step"], y=df["internal_temperature"],
            mode="lines", name="Internal Temp (°C)",
            line=dict(color="red", width=2)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["step"], y=df["ambient_temperature"],
            mode="lines", name="Ambient Temp (°C)",
            line=dict(color="orange", width=1, dash="dot")
        ),
        row=1, col=1
    )
    
    # Safety thresholds
    t_min = DEFAULT_THERMAL_PARAMS.min_safe_temperature
    t_max = DEFAULT_THERMAL_PARAMS.max_safe_temperature
    
    fig.add_hline(y=t_min, line_dash="dash", line_color="blue", annotation_text="Min Safe (18°C)", row=1, col=1)
    fig.add_hline(y=t_max, line_dash="dash", line_color="purple", annotation_text="Max Safe (27°C)", row=1, col=1)
    
    # Bottom Plot: Workload & Cooling
    fig.add_trace(
        go.Scatter(
            x=df["step"], y=df["workload"],
            mode="lines", name="Workload (0-1)",
            line=dict(color="green", width=2)
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["step"], y=df["cooling_level"],
            mode="lines", name="Cooling Level (0-1)",
            line=dict(color="blue", width=2, shape="hv")
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Value", row=2, col=1)
    fig.update_xaxes(title_text="Time Step (5 min)", row=2, col=1)
    
    return fig


def plot_comparison_metrics(df: pd.DataFrame, scenario_name: str) -> go.Figure:
    """Plot comparative metrics across controllers from evaluation results."""
    # Group by controller
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"Internal Temperature - {scenario_name}", "Cooling Energy")
    )
    
    colors = {"fixed": "gray", "rule_based": "blue", "q_learning": "red"}
    
    for controller in df["controller"].unique():
        ctrl_df = df[df["controller"] == controller]
        color = colors.get(controller, "black")
        
        # Temperature
        fig.add_trace(
            go.Scatter(
                x=ctrl_df["step"], y=ctrl_df["internal_temperature"],
                mode="lines", name=f"{controller} Temp",
                line=dict(color=color, width=2)
            ),
            row=1, col=1
        )
        
        # Energy
        fig.add_trace(
            go.Scatter(
                x=ctrl_df["step"], y=ctrl_df["cooling_energy"],
                mode="lines", name=f"{controller} Energy",
                line=dict(color=color, width=1, dash="dot")
            ),
            row=2, col=1
        )
        
    t_min = DEFAULT_THERMAL_PARAMS.min_safe_temperature
    t_max = DEFAULT_THERMAL_PARAMS.max_safe_temperature
    fig.add_hline(y=t_min, line_dash="dash", line_color="blue", row=1, col=1)
    fig.add_hline(y=t_max, line_dash="dash", line_color="purple", row=1, col=1)

    fig.update_layout(height=600, hovermode="x unified")
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Energy (kWh)", row=2, col=1)
    
    return fig
