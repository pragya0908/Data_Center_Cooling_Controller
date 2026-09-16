"""CoolRL Agents Package."""

from src.agents.q_learning import QLearningAgent
from src.agents.state_discretizer import (
    DEFAULT_DISCRETIZATION_CONFIG,
    DiscretizationConfig,
    StateDiscretizer,
)

__all__ = [
    "DEFAULT_DISCRETIZATION_CONFIG",
    "DiscretizationConfig",
    "QLearningAgent",
    "StateDiscretizer",
]
