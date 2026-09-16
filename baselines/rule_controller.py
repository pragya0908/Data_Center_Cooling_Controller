"""Rule-based baseline controller for CoolRL."""

from __future__ import annotations


class RuleController:
    """Simple threshold-based controller that reacts to the temperature bin.

    Decision logic
    --------------
    * Temp bin 5 (> 30 C)  -> action 4 (max cooling increase +20 %)
    * Temp bin 0 (< 22 C)  -> action 0 (max cooling decrease -20 %)
    * Otherwise             -> action 2 (maintain current cooling)
    """

    def choose_action(self, state: tuple, current_cooling: float) -> int:
        """Select an action based on the temperature bin in *state*.

        Parameters
        ----------
        state : tuple
            Discretised state where ``state[0]`` is the temperature bin
            (0-5, see ``DataCenterEnv._TEMP_EDGES``).
        current_cooling : float
            Current cooling level (unused by this controller).

        Returns
        -------
        int
            Action in ``{0, 2, 4}``.
        """
        temp_bin = state[0]

        if temp_bin >= 5:       # > 30 C  — critical overheat
            return 4
        elif temp_bin == 0:     # < 22 C  — over-cooling
            return 0
        else:                   # 22-30 C — acceptable range
            return 2
