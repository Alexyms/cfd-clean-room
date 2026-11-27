"""The march's records and constants, shared by the reference's and the core's runs.

``time_integration`` marches a scenario on the reference or, under
``solver.backend: core``, through ``core_backend``'s march. Both build on the
records here, so neither imports the other's module for them: ``IntervalPlan``
(an interval as the march steps it), ``MarchStep`` (what ``on_step`` is handed),
the recovery challenge's constants and the copy of a budget list. Moved out of
``time_integration`` by prompt 96b, which broke the import cycle between it and
``core_backend``; ``time_integration`` still names them, so that
``from src.time_integration import IntervalPlan`` reads as it did.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from src.monitor import Alert
from src.solver_transport import MassBudget

# ISO 14644-3's recovery challenge: the 100:1 time is the time a reading takes
# to fall to a hundredth of its initial level (ADR-014 C).
HUNDRED_TO_ONE = 100.0
# The 100:1 time is defined where the initial level is at least 100 times the
# threshold. The clean-out sets it to exactly that, and an initial stated as a
# concentration can carry the rounding of the product (35.2 x 100 is
# 3520.0000000000005), so the comparison is read to this relative tolerance.
DEFINED_RTOL = 1.0e-9


@dataclass(frozen=True)
class IntervalPlan:
    """An interval as the march steps it.

    Parameters
    ----------
    index : int
        Its place in ``ScenarioManager.intervals()``.
    t_start, t_end : float
        Seconds. A restart whose checkpoint lies inside an interval it must
        divide again starts that interval's plan at the checkpoint's time.
    fingerprint : str
        The flow state it steps on.
    events : tuple[str, ...]
        The events active over it.
    steps : int
        Equal steps from ``t_start`` to ``t_end``.
    dt : float
        ``(t_end - t_start) / steps``, at most ``stable_dt``.
    stable_dt : float
        The smallest ``stable_dt`` over the classes on its faces.
    """

    index: int
    t_start: float
    t_end: float
    fingerprint: str
    events: tuple[str, ...]
    steps: int
    dt: float
    stable_dt: float

    def time(self, i: int) -> float:
        """The time of step i of this interval, its end exactly at i = steps.

        Parameters
        ----------
        i : int
            The step within the interval, 0 to ``steps``.

        Returns
        -------
        float
            ``t_start + i dt``, and ``t_end`` itself at ``i = steps``, so a
            switch lands on the interval's end whatever the rounding of the
            product.
        """
        if i == self.steps:
            return self.t_end
        return self.t_start + i * self.dt


@dataclass(frozen=True, eq=False)
class MarchStep:
    """What ``on_step`` is handed after each step's writes.

    Parameters
    ----------
    step : int
        Steps taken, the run's count.
    t : float
        Simulated time after the step, seconds.
    dt : float
        The step just taken.
    interval : int
        The interval's index.
    fields : Mapping[int, np.ndarray]
        Class to its field after the step, read-only.
    budgets : tuple[MassBudget, ...]
        Copies of the transport solver's budgets after the step.
    alerts : tuple[Alert, ...]
        The alerts the step opened or closed (``AlertMonitor.update``);
        empty without sensors.
    """

    step: int
    t: float
    dt: float
    interval: int
    fields: Mapping[int, np.ndarray]
    budgets: tuple[MassBudget, ...]
    alerts: tuple[Alert, ...] = ()


def copies_of(budgets: Sequence[MassBudget]) -> tuple[MassBudget, ...]:
    """Copies of budgets that share nothing with them."""
    return tuple(replace(b, deposited=dict(b.deposited)) for b in budgets)


class RecoveryTrackerView(Protocol):
    """What the core's march asks of the reference's recovery tracker.

    ``time_integration.RecoveryTracker`` is one: the core hands its state over
    before a march and sets it from the core's after (ADR-017 B), through the
    tracker's own checks, so ``core_backend`` needs the two methods and not the
    class.
    """

    def state(self) -> dict[str, np.ndarray]:
        """The tracker's state as arrays, for a checkpoint or the core."""
        ...

    def restore(self, state: Mapping[str, np.ndarray]) -> None:
        """Set the tracker from a state, as a checkpoint's restore does."""
        ...
