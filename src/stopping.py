"""Stopping rule for the steady SIMPLE loop: estimated iteration error and continuity.

A small velocity step between outer iterations is not a small error. While
the iteration contracts geometrically at a rate rho, the error left after a
step s is about s rho / (1 - rho), and rho approaches 1 as the grid is
refined, so a fixed bound on the step leaves more error on every finer grid.
On an open domain part of the error is a drift of the through-flow set by
the per-cell mass imbalance, which the step does not see. Both are measured
in docs/reports/stopping_rule_evidence.md, sections 4 and 5.

ErrorEstimateRule stops when (a) the estimated iteration error over a
physical velocity scale and (b) the worst absolute per-cell mass imbalance
are both below their tolerances. It knows nothing of the solver: it is fed
one outer iteration at a time, so it can be tested on synthetic histories.
"""

import math
from collections import deque
from collections.abc import Callable

import numpy as np

# Trailing window of the rate fit, in outer iterations. While the iteration is
# geometric the length hardly matters: from 1e-5 to 1e-8, halving or doubling
# it moved no estimate-to-error ratio by more than 0.015, except 0.095 on
# VAL-001 80x40 near 1e-5 (evidence report, section 4). Off that regime a long
# window averages across a change of rate, so it must be short against a
# decade. On the fastest case, VAL-001 40x20 at about 71 outer iterations per
# decade, 100 spans 1.4 decades and is full by outer iteration 142, the
# residual's 1e-5, where 200 was not.
RATE_WINDOW = 100


class ErrorEstimateRule:
    """Stop on the estimated iteration error and the worst per-cell mass imbalance.

    Condition (a): ``step * rho_hat / (1 - rho_hat) / velocity_scale`` is below
    ``iteration_error_tol``, with rho_hat the exp of the least-squares slope of
    log(step) over the last RATE_WINDOW steps. Condition (b): the worst
    absolute per-cell mass imbalance is below ``mass_imbalance_tol``. The
    imbalance is asked for only when (a) holds.

    Parameters
    ----------
    velocity_scale : float
        Physical velocity the estimate is divided by, m/s: the largest
        prescribed boundary velocity. Positive and finite.
    iteration_error_tol : float
        Bound on the estimate over velocity_scale, dimensionless. Positive
        and finite.
    mass_imbalance_tol : float
        Bound on the worst absolute per-cell imbalance, kg/s per unit depth
        (PressureCorrector.mass_imbalance). Positive and finite.

    Attributes
    ----------
    estimate_history : list[float]
        The estimate over velocity_scale after each step. ``inf`` where there
        is none: fewer than RATE_WINDOW steps, a step in the window that is
        zero or not finite, or rho_hat not strictly between 0 and 1. A stall
        therefore never reads as convergence.

    Raises
    ------
    ValueError
        If any parameter is not positive and finite.
    """

    def __init__(
        self,
        velocity_scale: float,
        iteration_error_tol: float,
        mass_imbalance_tol: float,
    ) -> None:
        for name, value in (
            ("velocity_scale", velocity_scale),
            ("iteration_error_tol", iteration_error_tol),
            ("mass_imbalance_tol", mass_imbalance_tol),
        ):
            # bool is an int; True would pass as 1.0.
            if isinstance(value, bool) or not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value}")
        self._scale = velocity_scale
        self._error_tol = iteration_error_tol
        self._imbalance_tol = mass_imbalance_tol
        self._steps: deque[float] = deque(maxlen=RATE_WINDOW)
        self.estimate_history: list[float] = []

    def _estimate(self) -> float:
        """The current estimate over velocity_scale, or inf when there is none."""
        if len(self._steps) < RATE_WINDOW:
            return math.inf
        steps = np.array(self._steps)
        if not np.all(np.isfinite(steps) & (steps > 0.0)):
            return math.inf
        x = np.arange(steps.size) - (steps.size - 1) / 2.0
        rho = math.exp(float(x @ np.log(steps)) / float(x @ x))
        if not 0.0 < rho < 1.0:
            return math.inf
        return float(steps[-1]) * rho / (1.0 - rho) / self._scale

    def update(self, step: float, worst_imbalance: Callable[[], float]) -> bool:
        """Record one outer iteration and answer whether the solve has converged.

        Parameters
        ----------
        step : float
            Largest change of the velocity over the outer iteration, m/s.
        worst_imbalance : Callable[[], float]
            Returns the worst absolute per-cell mass imbalance of the current
            field. Called only when condition (a) holds.

        Returns
        -------
        bool
            True when conditions (a) and (b) both hold.
        """
        self._steps.append(float(step))
        estimate = self._estimate()
        self.estimate_history.append(estimate)
        return estimate < self._error_tol and worst_imbalance() < self._imbalance_tol
