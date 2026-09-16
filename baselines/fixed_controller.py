"""Fixed-setpoint baseline controller for CoolRL."""

from __future__ import annotations

from environment.datacenter_env import DataCenterEnv


class FixedController:
    """Always drives cooling toward a fixed setpoint (default 0.7).

    Parameters
    ----------
    target_cooling : float
        Desired cooling level in [0.0, 1.0].
    """

    # Map each action index to its cooling delta
    _DELTAS: dict[int, float] = DataCenterEnv.ACTION_DELTAS

    def __init__(self, target_cooling: float = 0.7) -> None:
        self.target_cooling = target_cooling

    def choose_action(self, state: tuple, current_cooling: float) -> int:
        """Return the action that moves *current_cooling* closest to the target.

        Parameters
        ----------
        state : tuple
            Discretised environment state (unused by this controller).
        current_cooling : float
            Current cooling level in [0.0, 1.0].

        Returns
        -------
        int
            Action in ``{0, 1, 2, 3, 4}``.
        """
        best_action = 2  # default: no change
        best_distance = abs(current_cooling - self.target_cooling)

        for action, delta in self._DELTAS.items():
            projected = max(0.0, min(1.0, current_cooling + delta))
            distance = abs(projected - self.target_cooling)
            if distance < best_distance:
                best_distance = distance
                best_action = action

        return best_action
