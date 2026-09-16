"""Fixed-Cooling Baseline Controller for CoolRL.

This module implements a static, non-adaptive cooling baseline that maintains
a constant target cooling level (default C = 0.50) across simulation episodes.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np


class FixedCoolingController:
    """Baseline controller that maintains a constant target cooling level.

    For C = 0.50 (the nominal data-center operating point), the environment
    initializes at C_0 = 0.50, and the controller repeatedly applies Action 2
    (maintain cooling, delta = 0.00).

    For targets differing from initial state, the controller selects discrete
    adjustments in {-0.20, -0.10, 0.00, +0.10, +0.20} to converge to and maintain
    the specified target level.
    """

    def __init__(self, target_cooling: float = 0.50) -> None:
        """Initialize the fixed cooling controller.

        Args:
            target_cooling: Target cooling effort in [0.0, 1.0] (default 0.50).
        """
        if not (0.0 <= target_cooling <= 1.0):
            raise ValueError(f"target_cooling must be in [0.0, 1.0], got {target_cooling}")
        self.target_cooling = float(target_cooling)

    def choose_action(self, observation: Any) -> int:
        """Select a discrete action in {0, 1, 2, 3, 4} to maintain target cooling.

        Args:
            observation: 5-element continuous observation [T_int, W, T_amb, C, delta_T].

        Returns:
            Integer action index in {0, 1, 2, 3, 4}.
        """
        if observation is None or len(observation) < 4:
            raise ValueError(f"Invalid observation provided to FixedCoolingController: {observation}")

        current_cooling = float(observation[3])

        # If already at target cooling, strictly maintain (Action 2)
        if np.isclose(current_cooling, self.target_cooling, atol=1e-3):
            return 2

        diff = self.target_cooling - current_cooling

        # Discrete delta options: -0.20 (0), -0.10 (1), 0.00 (2), +0.10 (3), +0.20 (4)
        if diff > 0.15:
            return 4  # +0.20
        elif diff > 0.05:
            return 3  # +0.10
        elif diff < -0.15:
            return 0  # -0.20
        elif diff < -0.05:
            return 1  # -0.10
        else:
            return 2  # 0.00
