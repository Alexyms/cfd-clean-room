"""The measuring devices prompt 50's runs share: the recording rule and the run's trace.

RecordingRule50 is prompt 46's RecordingRule (docs/reports/probe46/coupled46.py)
for rule version 5: it takes the per-step ``nu_scale`` as well, asks for the
imbalance every outer iteration and keeps the readings, the steps and the
scales it was passed, so a record can say from which iteration each of the
five conditions holds. The decision is the committed rule's, on the same
readings. ``_nu_scale`` reads the last scale passed, the name prompt 46's and
47's runners record, so they run unchanged under version 5.

TracedSolver is the committed StaggeredSolver with its callback wrapped to
keep, every outer iteration, the k and eps solves' sweeps and whether both
converged (the model's ``last_sweeps`` and ``solves_converged``, as prompt
46's runner reads them). Nothing it reads changes the solve.
"""

import math
from collections.abc import Callable
from typing import ClassVar

import numpy as np

from src import solver_staggered
from src.solver_staggered import StaggeredSolver
from src.stopping import ErrorEstimateRule, ImbalanceSummary, IterationState


class RecordingRule50(ErrorEstimateRule):
    """The committed rule, asking for the imbalance every outer iteration."""

    instances: ClassVar[list["RecordingRule50"]] = []

    def __init__(self, *args: float, **kwargs: bool) -> None:
        super().__init__(*args, **kwargs)
        self.readings: list[ImbalanceSummary] = []
        self.steps: list[float] = []
        self.viscosity_steps: list[float] = []
        RecordingRule50.instances.append(self)

    @property
    def _nu_scale(self) -> float | None:
        """The last scale (e) was passed; None before the first step or with (e) off."""
        history = self.viscosity_scale_history
        return history[-1] if history else None

    def update(
        self,
        step: float,
        imbalance: Callable[[], ImbalanceSummary],
        viscosity_step: float | None = None,
        nu_scale: float | None = None,
    ) -> bool:
        """Record the readings, then decide as the committed rule does on them."""
        found = imbalance()
        self.readings.append(found)
        self.steps.append(float(step))
        self.viscosity_steps.append(
            math.nan if viscosity_step is None else float(viscosity_step)
        )
        return super().update(
            step, lambda: found, viscosity_step=viscosity_step, nu_scale=nu_scale
        )

    def summary(self, tols: tuple[float, float]) -> dict:
        """Per-iteration readings and the iteration from which each condition holds.

        Parameters
        ----------
        tols : tuple[float, float]
            (iteration_error_tol, mass_imbalance_tol).
        """
        flux = self._flux_scale
        readings = {
            "a": list(self.estimate_history),
            "b": [r.worst for r in self.readings],
            "c": [r.absolute_sum / flux for r in self.readings],
            "d": [abs(r.signed_sum) for r in self.readings],
            "e": list(self.viscosity_estimate_history),
        }
        holds = {
            "a": [x < tols[0] for x in readings["a"]],
            "b": [x < tols[1] for x in readings["b"]],
            "c": [x < tols[0] for x in readings["c"]],
            "d": [x < tols[1] for x in readings["d"]],
            "e": [x < tols[0] for x in readings["e"]],
        }
        return {
            "readings_at_stop": {k: v[-1] for k, v in readings.items() if v},
            "from_when": {k: from_when(v) for k, v in holds.items()},
            "first_all": first_all(holds),
            "velocity_step": self.steps,
            "viscosity_step": self.viscosity_steps,
            "nu_scale": list(self.viscosity_scale_history),
            "estimate_a": readings["a"],
            "estimate_e": readings["e"],
        }


def from_when(holds: list[bool]) -> int | None:
    """The 1-based outer iteration from which a condition holds to the end, or None."""
    if not holds or not holds[-1]:
        return None
    i = len(holds)
    while i > 0 and holds[i - 1]:
        i -= 1
    return i + 1


def first_all(holds: dict[str, list[bool]]) -> int | None:
    """The 1-based outer iteration at which every condition first holds at once."""
    n = len(holds["a"])
    keys = [k for k, v in holds.items() if len(v) == n]
    together = np.all([holds[k] for k in keys], axis=0) if n else np.array([])
    hits = np.nonzero(together)[0]
    return int(hits[0]) + 1 if hits.size else None


def install() -> None:
    """Make every solver built after this call use RecordingRule50."""
    solver_staggered.ErrorEstimateRule = RecordingRule50  # type: ignore[misc]
    RecordingRule50.instances.clear()


class TracedSolver(StaggeredSolver):
    """The committed solver, its k and eps solve counts kept per outer iteration."""

    instances: ClassVar[list["TracedSolver"]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.trace: dict[str, list] = {
            "k_sweeps": [],
            "eps_sweeps": [],
            "scalar_converged": [],
            "residual": [],
        }
        TracedSolver.instances.append(self)

    def solve_steady(
        self,
        on_iteration: Callable[[IterationState], None] | None = None,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The committed solve, with the callback wrapped."""

        def traced(state: IterationState) -> None:
            model = self._model
            if model is not None:
                self.trace["k_sweeps"].append(int(model.last_sweeps[0]))
                self.trace["eps_sweeps"].append(int(model.last_sweeps[1]))
                self.trace["scalar_converged"].append(bool(model.solves_converged))
            self.trace["residual"].append(float(state.residual))
            if on_iteration is not None:
                on_iteration(state)

        return super().solve_steady(on_iteration=traced, **kwargs)  # type: ignore[arg-type]
