"""Live Reinforcement Learning Simulator for the Streamlit Dashboard.

This module provides a stateful wrapper around the DataCenterEnv and QLearningAgent
to allow for interactive, step-by-step or automated execution in a Streamlit context.
"""

from typing import Any
import pandas as pd
from pathlib import Path

from src.environment.datacenter_env import DataCenterEnv
from src.agents.q_learning import QLearningAgent
from src.agents.state_discretizer import StateDiscretizer

# Constants for default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "results" / "training" / "q_learning_normal_train.npz"

class LiveSimulator:
    """Manages the state of a live RL simulation run for Streamlit."""
    
    def __init__(
        self,
        scenario: str = "normal",
        model_path: Path | str = DEFAULT_MODEL_PATH,
        split: str = "test",
    ):
        """Initialize the simulator with a given scenario and model."""
        self.scenario = scenario
        self.model_path = Path(model_path)
        
        # Initialize environment
        self.env = DataCenterEnv(scenario=scenario, split=split)
        
        # Initialize discretizer (assuming 960 states configuration from Phase 2E)
        self.discretizer = StateDiscretizer()
        
        # Load agent
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.agent = QLearningAgent.load(self.model_path)
        
        # State tracking
        self.history: list[dict[str, Any]] = []
        self.terminated = False
        
        # Reset to initial state
        self.reset()
        
    def reset(self) -> None:
        """Reset the environment and history."""
        obs, info = self.env.reset()
        self.terminated = False
        self.history = []
        self._record_step(obs, info, action=None, reward=None)
        
    def step(self) -> bool:
        """Advance the simulation by one step using the greedy Q-learning policy.
        
        Returns:
            bool: True if the episode is terminated, False otherwise.
        """
        if self.terminated:
            return True
            
        # Get last observation from history
        last_obs = self.history[-1]["observation"]
        
        # Discretize state
        state_id = self.discretizer.discretize(last_obs)
        
        # Choose greedy action
        action = self.agent.choose_action(state_id, training=False)
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.terminated = terminated
        
        # Record step
        self._record_step(obs, info, action, reward)
        
        return self.terminated
        
    def auto_play(self, n_steps: int = 10) -> bool:
        """Advance the simulation by n_steps.
        
        Returns:
            bool: True if terminated during auto-play.
        """
        for _ in range(n_steps):
            if self.step():
                return True
        return self.terminated
        
    def _record_step(self, obs: Any, info: dict[str, Any], action: int | None, reward: float | None) -> None:
        """Append the current step state to history."""
        record = {
            "step": info.get("step_index", len(self.history)),
            "observation": obs,
            "action": action,
            "reward": reward,
            "internal_temperature": info["internal_temperature"],
            "ambient_temperature": info["ambient_temperature"],
            "workload": info["workload"],
            "cooling_level": info["cooling_level"],
            "cooling_energy": info["cooling_energy"],
            "safety_penalty": info.get("safety_penalty", 0.0),
            "energy_penalty": info.get("energy_penalty", 0.0),
        }
        self.history.append(record)
        
    def get_history_df(self) -> pd.DataFrame:
        """Return the simulation history as a Pandas DataFrame."""
        return pd.DataFrame(self.history)
