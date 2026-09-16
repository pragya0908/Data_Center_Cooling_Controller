"""Tabular Q-learning agent for CoolRL."""

from __future__ import annotations

import pickle
import random
from pathlib import Path


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning agent.

    Parameters
    ----------
    num_actions : int
        Size of the discrete action space (default ``5``).
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration probability.
    epsilon_min : float
        Floor for epsilon after decay.
    epsilon_decay : float
        Multiplicative decay applied to epsilon after every ``learn`` call.
    seed : int | None
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_actions: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: maps (state_tuple, action) -> float
        self.q_table: dict[tuple, float] = {}

        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state: tuple) -> int:
        """Select an action using an epsilon-greedy policy.

        With probability ``epsilon`` a random action is chosen; otherwise
        the action with the highest Q-value for *state* is returned.

        Parameters
        ----------
        state : tuple
            Discretised environment state.

        Returns
        -------
        int
            Selected action in ``{0, …, num_actions - 1}``.
        """
        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.num_actions - 1)

        # Greedy: pick the action with the max Q-value (0.0 default)
        q_values = [
            self.q_table.get((state, a), 0.0)
            for a in range(self.num_actions)
        ]
        max_q = max(q_values)
        # Break ties randomly
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return self._rng.choice(best_actions)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
    ) -> None:
        """Perform a single tabular Q-learning (Bellman) update.

        .. math::

            Q(s, a) \\leftarrow Q(s, a)
                + \\alpha \\bigl[
                    r + \\gamma \\max_{a'} Q(s', a') - Q(s, a)
                \\bigr]

        After the update, epsilon is decayed toward ``epsilon_min``.
        """
        current_q = self.q_table.get((state, action), 0.0)

        max_next_q = max(
            self.q_table.get((next_state, a), 0.0)
            for a in range(self.num_actions)
        )

        # Bellman update
        td_target = reward + self.gamma * max_next_q
        self.q_table[(state, action)] = (
            current_q + self.alpha * (td_target - current_q)
        )

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the Q-table to *path* using :mod:`pickle`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.q_table, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        """Load a Q-table from *path*, replacing the current one."""
        with Path(path).open("rb") as f:
            self.q_table = pickle.load(f)  # noqa: S301
