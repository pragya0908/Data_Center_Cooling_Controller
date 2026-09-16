"""Workload generator for the CoolRL environment."""

import math
import random


class WorkloadGenerator:
    """Generates synthetic server workload values in [0.0, 1.0].

    Supports two modes:
        - ``'normal'``: a gentle sine-wave fluctuation between 0.3 and 0.6.
        - ``'spike'``:  a low baseline (~0.1–0.3) that randomly spikes to
          0.9 or 1.0 with configurable probability.

    Parameters
    ----------
    period : int
        Wavelength (in steps) of the sine cycle used in *normal* mode.
        Defaults to ``100``.
    spike_prob : float
        Per-step probability of a spike in *spike* mode.
        Defaults to ``0.05`` (5 %).
    seed : int | None
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        period: int = 100,
        spike_prob: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self._period = period
        self._spike_prob = spike_prob
        self._rng = random.Random(seed)

    def get_workload(self, step: int, mode: str = "normal") -> float:
        """Return a workload value for the given *step*.

        Parameters
        ----------
        step : int
            Current simulation step (used to drive the sine wave).
        mode : str
            ``'normal'`` or ``'spike'``.

        Returns
        -------
        float
            A value clamped to [0.0, 1.0].

        Raises
        ------
        ValueError
            If *mode* is not one of the supported modes.
        """
        match mode:
            case "normal":
                return self._normal(step)
            case "spike":
                return self._spike()
            case _:
                raise ValueError(
                    f"Unknown mode {mode!r}. Expected 'normal' or 'spike'."
                )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normal(self, step: int) -> float:
        """Gentle sine fluctuation mapped to [0.3, 0.6]."""
        amplitude = 0.15  # half-range of the 0.3–0.6 band
        midpoint = 0.45
        value = midpoint + amplitude * math.sin(
            2.0 * math.pi * step / self._period
        )
        return float(max(0.0, min(1.0, value)))

    def _spike(self) -> float:
        """Low baseline with occasional spikes to 0.9–1.0."""
        if self._rng.random() < self._spike_prob:
            return self._rng.uniform(0.9, 1.0)
        return self._rng.uniform(0.1, 0.3)
