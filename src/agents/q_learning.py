"""Tabular Q-Learning Agent for CoolRL.

This module implements the tabular Q-learning algorithm matching Phase 2E specifications:
- Discrete state space: 960 states (indexed 0..959).
- Discrete action space: 5 actions (indexed 0..4).
- Standard temporal-difference Bellman update with terminal masking.
- Epsilon-greedy exploration with episode-level exponential decay.
- Model persistence via NumPy (.npz).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent with epsilon-greedy action selection and persistence.

    Mathematical Formulation:
        Target (non-terminal): y_t = r_t + gamma * max_a' Q(s_{t+1}, a')
        Target (terminal):     y_t = r_t
        TD Error:             delta_t = y_t - Q(s_t, a_t)
        Update:                Q(s_t, a_t) <- Q(s_t, a_t) + alpha * delta_t
        Epsilon Decay:         epsilon <- max(epsilon_min, epsilon * epsilon_decay)
    """

    def __init__(
        self,
        n_states: int = 960,
        n_actions: int = 5,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon: float = 1.00,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        """Initialize the tabular Q-learning agent.

        Args:
            n_states: Number of discrete states (default 960 from Phase 2E).
            n_actions: Number of discrete actions (default 5 from Phase 2E).
            alpha: Learning rate parameter in (0, 1].
            gamma: Discount factor in [0, 1].
            epsilon: Initial exploration probability in [0, 1].
            epsilon_min: Minimum exploration floor in [0, 1].
            epsilon_decay: Multiplicative decay factor per episode in (0, 1].
            seed: Seed for random number generator reproducibility.
        """
        if n_states <= 0:
            raise ValueError(f"n_states must be positive, got {n_states}")
        if n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {n_actions}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if not (0.0 <= epsilon_min <= 1.0):
            raise ValueError(f"epsilon_min must be in [0, 1], got {epsilon_min}")
        if not (0.0 < epsilon_decay <= 1.0):
            raise ValueError(f"epsilon_decay must be in (0, 1], got {epsilon_decay}")

        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # Zero-initialize tabular Q-table of shape (n_states, n_actions)
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

        # Seed-controlled random number generator
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state_id: int, training: bool = True) -> int:
        """Select an action using epsilon-greedy (training) or pure greedy (evaluation).

        Args:
            state_id: Discrete state index in [0, n_states - 1].
            training: If True, explore with probability epsilon; if False, act greedily.

        Returns:
            Integer action index in [0, n_actions - 1].
        """
        if not isinstance(state_id, (int, np.integer)):
            raise TypeError(f"state_id must be an integer, got {type(state_id).__name__}")
        if not (0 <= int(state_id) < self.n_states):
            raise ValueError(f"state_id must be in [0, {self.n_states - 1}], got {state_id}")

        sid = int(state_id)

        # Epsilon-greedy exploration in training mode
        if training and (self.rng.random() < self.epsilon):
            return int(self.rng.integers(0, self.n_actions))

        # Pure greedy exploitation: break ties deterministically using lowest index
        q_values = self.q_table[sid]
        return int(np.argmax(q_values))

    def update(
        self,
        state_id: int,
        action: int,
        reward: float,
        next_state_id: int,
        terminated: bool,
    ) -> float:
        """Execute a tabular Q-learning update step.

        Args:
            state_id: Current discrete state index.
            action: Action taken in state_id.
            reward: Scalar reward received from environment transition.
            next_state_id: Resulting discrete state index.
            terminated: True if transition reached terminal state (no bootstrapping).

        Returns:
            Scalar temporal-difference (TD) error.
        """
        if not (0 <= state_id < self.n_states):
            raise ValueError(f"state_id must be in [0, {self.n_states - 1}], got {state_id}")
        if not (0 <= action < self.n_actions):
            raise ValueError(f"action must be in [0, {self.n_actions - 1}], got {action}")
        if not (0 <= next_state_id < self.n_states):
            raise ValueError(f"next_state_id must be in [0, {self.n_states - 1}], got {next_state_id}")
        if not np.isfinite(reward):
            raise ValueError(f"Reward must be finite, got {reward}")

        sid = int(state_id)
        act = int(action)
        nsid = int(next_state_id)
        r = float(reward)

        current_q = self.q_table[sid, act]

        if terminated:
            target = r
        else:
            target = r + self.gamma * float(np.max(self.q_table[nsid]))

        td_error = target - current_q
        self.q_table[sid, act] += self.alpha * td_error

        return float(td_error)

    def decay_epsilon(self) -> float:
        """Decay exploration parameter epsilon by epsilon_decay down to epsilon_min floor.

        Returns:
            New decayed epsilon value.
        """
        self.epsilon = float(max(self.epsilon_min, self.epsilon * self.epsilon_decay))
        return self.epsilon

    def get_policy(self) -> np.ndarray:
        """Extract deterministic greedy policy from the current Q-table.

        Returns:
            1D integer array of shape (n_states,) containing optimal action per state.
        """
        return np.argmax(self.q_table, axis=1).astype(np.int64)

    def save(self, filepath: str | Path) -> Path:
        """Save the agent Q-table and hyperparameters to a NumPy .npz file.

        Args:
            filepath: Destination path for .npz file.

        Returns:
            Resolved Path of the saved artifact.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            q_table=self.q_table,
            n_states=np.int64(self.n_states),
            n_actions=np.int64(self.n_actions),
            alpha=np.float64(self.alpha),
            gamma=np.float64(self.gamma),
            epsilon=np.float64(self.epsilon),
            epsilon_min=np.float64(self.epsilon_min),
            epsilon_decay=np.float64(self.epsilon_decay),
        )
        return path

    @classmethod
    def load(cls, filepath: str | Path, seed: int | None = None) -> QLearningAgent:
        """Load a QLearningAgent from a saved NumPy .npz file.

        Args:
            filepath: Path to .npz file.
            seed: Optional random seed for the restored agent's RNG.

        Returns:
            Instantiated QLearningAgent with restored Q-table and parameters.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = np.load(path)
        agent = cls(
            n_states=int(data["n_states"]),
            n_actions=int(data["n_actions"]),
            alpha=float(data["alpha"]),
            gamma=float(data["gamma"]),
            epsilon=float(data["epsilon"]),
            epsilon_min=float(data["epsilon_min"]),
            epsilon_decay=float(data["epsilon_decay"]),
            seed=seed,
        )
        agent.q_table = np.array(data["q_table"], dtype=np.float64)
        return agent
