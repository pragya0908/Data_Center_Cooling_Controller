"""Data-center cooling environment for CoolRL."""

from __future__ import annotations

import random

from environment.workload_generator import WorkloadGenerator


class DataCenterEnv:
    """Simplified data-center thermal environment.

    State variables
    ---------------
    current_temp : float
        Server-room temperature (°C).  Starts at 24.0.
    ambient_temp : float
        Outside / ambient temperature (°C).  Starts at 22.0.
    current_cooling : float
        Cooling effort in [0.0, 1.0].  Starts at 0.5.
    current_workload : float
        Server workload in [0.0, 1.0].  Driven by ``WorkloadGenerator``.

    Parameters
    ----------
    workload_mode : str
        Mode forwarded to :pymethod:`WorkloadGenerator.get_workload`.
    seed : int | None
        RNG seed passed to the workload generator for reproducibility.
    """

    # -- Action space ------------------------------------------------------
    # 0 → -20 %, 1 → -10 %, 2 → 0 %, 3 → +10 %, 4 → +20 %
    ACTION_DELTAS: dict[int, float] = {
        0: -0.20,
        1: -0.10,
        2:  0.00,
        3: +0.10,
        4: +0.20,
    }
    NUM_ACTIONS: int = len(ACTION_DELTAS)

    # -- Discretisation bin edges ------------------------------------------
    _TEMP_EDGES: list[float] = [22.0, 24.0, 26.0, 28.0, 30.0]       # → 6 bins
    _WORKLOAD_EDGES: list[float] = [0.2, 0.4, 0.6, 0.8]             # → 5 bins
    _AMBIENT_EDGES: list[float] = [20.0, 25.0, 30.0]                # → 4 bins
    _COOLING_EDGES: list[float] = [0.2, 0.4, 0.6, 0.8]             # → 5 bins

    def __init__(
        self,
        workload_mode: str = "normal",
        seed: int | None = None,
    ) -> None:
        # Continuous state
        self.current_temp: float = 24.0
        self.ambient_temp: float = 22.0
        self.current_cooling: float = 0.5
        self.current_workload: float = 0.0

        # Bookkeeping
        self._step_count: int = 0
        self._workload_mode = workload_mode

        # Sub-components
        self._workload_gen = WorkloadGenerator(seed=seed)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # State discretisation
    # ------------------------------------------------------------------

    @staticmethod
    def _digitise(value: float, edges: list[float]) -> int:
        """Return the bin index for *value* given sorted *edges*.

        Bins are numbered ``0 … len(edges)``:
        * bin 0  → value < edges[0]
        * bin k  → edges[k-1] ≤ value < edges[k]
        * bin N  → value ≥ edges[-1]
        """
        for i, edge in enumerate(edges):
            if value < edge:
                return i
        return len(edges)

    def _get_discrete_state(self) -> tuple[int, int, int, int]:
        """Discretise the current continuous state into an integer tuple.

        Returns
        -------
        tuple[int, int, int, int]
            ``(temp_bin, workload_bin, ambient_bin, cooling_bin)``
        """
        return (
            self._digitise(self.current_temp, self._TEMP_EDGES),
            self._digitise(self.current_workload, self._WORKLOAD_EDGES),
            self._digitise(self.ambient_temp, self._AMBIENT_EDGES),
            self._digitise(self.current_cooling, self._COOLING_EDGES),
        )

    # ------------------------------------------------------------------
    # Environment dynamics
    # ------------------------------------------------------------------

    def step(
        self,
        action: int,
    ) -> tuple[tuple[int, int, int, int], float, bool]:
        """Advance the environment by one time-step.

        Parameters
        ----------
        action : int
            An integer in ``{0, 1, 2, 3, 4}`` that adjusts the cooling
            level by the corresponding delta (see ``ACTION_DELTAS``).

        Returns
        -------
        next_state : tuple[int, int, int, int]
            Discretised state after the transition.
        reward : float
            Scalar reward for this transition.
        done : bool
            Always ``False`` (placeholder for episode termination).

        Raises
        ------
        ValueError
            If *action* is not in the valid action space.
        """
        if action not in self.ACTION_DELTAS:
            raise ValueError(
                f"Invalid action {action!r}. "
                f"Expected one of {set(self.ACTION_DELTAS)}."
            )

        # 1. Apply cooling adjustment (clamped to [0, 1])
        self.current_cooling = max(
            0.0, min(1.0, self.current_cooling + self.ACTION_DELTAS[action])
        )

        # 2. Update workload
        self.current_workload = self._workload_gen.get_workload(
            self._step_count, mode=self._workload_mode
        )

        # 3. Thermal update
        #    T(t+1) = T(t)
        #             + 2.0 * workload
        #             + 0.1 * (ambient - T(t))
        #             - 3.0 * cooling
        #             + noise
        noise = self._rng.uniform(-0.3, 0.3)
        self.current_temp = (
            self.current_temp
            + 2.0 * self.current_workload
            + 0.1 * (self.ambient_temp - self.current_temp)
            - 3.0 * self.current_cooling
            + noise
        )

        self._step_count += 1

        # 4. Compute reward & discretised next state
        reward = self._compute_reward()
        next_state = self._get_discrete_state()
        done = False

        return next_state, reward, done

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        """Compute the scalar reward for the current state.

        Temperature component
        ~~~~~~~~~~~~~~~~~~~~~
        * ``> 30 °C`` → −100  (critical overheat)
        * ``< 22 °C`` → −10   (over-cooling)
        * ``22 – 27 °C`` → +10  (ideal range)
        * ``27 – 30 °C`` → −5   (warm, but not critical)

        Energy cost
        ~~~~~~~~~~~
        Subtract ``cooling_level × 5`` to discourage excessive cooling.
        """
        temp = self.current_temp

        if temp > 30.0:
            temp_reward = -100.0
        elif temp < 22.0:
            temp_reward = -10.0
        elif temp <= 27.0:
            temp_reward = 10.0
        else:  # 27 < temp <= 30
            temp_reward = -5.0

        energy_penalty = self.current_cooling * 5.0

        return temp_reward - energy_penalty
