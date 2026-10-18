"""The observed order in time of the transient solve (ADR-014 step 6a).

The case is VAL-002's lid-driven cavity at Re 100 on a fixed uniform 20x20
grid, spun up from rest to T = 1 s. No exact unsteady solution fits this
solver's boundaries, so the order is temporal self-convergence: the same
transient at dt 0.1, 0.05, 0.025 and 0.0125 s, read from the ratio of
successive differences of every u and v face at T in the largest absolute
value. The grid is fixed, so the differences hold the time error only. The
cylinder cannot show the order: its error mixes grid, boundaries and time.

The order is read from the two finest differences, the asymptotic pair;
docs/reports/probe67/tables67.md gives every difference and order of the
recorded run (results/builder67/order67.json). The per-step criterion is
1e-10 of the lid speed with the pressure solve at its floor of 1e-10, and the
third test shows the iteration error it leaves is far below the smallest
difference. A BDF2 that quietly fell back to implicit Euler fails the first
test and passes the other two.

The planted defects each test was shown to catch are named in its docstring
(results/builder67/plant67.py).
"""

from itertools import pairwise

import numpy as np
import pytest
import yaml

from src.config import SimConfig
from src.solver_staggered import BDF2, IMPLICIT_EULER, FlowState
from src.staggered import allocate_fields
from tests.helpers import staggered_solver
from validation.cases import case_path

GRID = 20
T_END = 1.0
STEPS = (0.1, 0.05, 0.025, 0.0125)
STEP_TOL = 1e-10
REFERENCE_TOL = 1e-12
ORDER_TOLERANCE = 0.2
# The iteration error must sit below this fraction of the smallest difference.
ITERATION_SHARE = 1e-2


def _spin_up(scheme: str, dt: float, step_tol: float) -> FlowState:
    """The cavity from rest to T_END under one scheme and step."""
    raw = yaml.safe_load(case_path("cavity").read_text(encoding="utf-8"))
    raw["domain"]["nx"] = raw["domain"]["ny"] = GRID
    raw["solver"]["pressure_rtol"] = 1e-10
    raw["solver"]["stopping_rule"] = "velocity_step"
    solver = staggered_solver(SimConfig.from_dict(raw))
    u, v, p = allocate_fields(solver._mesh)
    return solver.solve_transient(
        dt,
        T_END,
        initial=FlowState(u=u, v=v, p=p),
        step_tol=step_tol,
        step_cap=1000,
        scheme=scheme,
    )


def _difference(a: FlowState, b: FlowState) -> float:
    """The largest absolute difference over every u and v face, m/s."""
    return float(max(np.max(np.abs(a.u - b.u)), np.max(np.abs(a.v - b.v))))


@pytest.fixture(scope="module")
def ladder() -> dict[str, list[FlowState]]:
    """Each scheme's end state at every step of the ladder."""
    return {
        scheme: [_spin_up(scheme, dt, STEP_TOL) for dt in STEPS]
        for scheme in (BDF2, IMPLICIT_EULER)
    }


def _differences(states: list[FlowState]) -> list[float]:
    return [_difference(a, b) for a, b in pairwise(states)]


def _finest_order(states: list[FlowState]) -> float:
    """log2 of the ratio of the two finest successive differences."""
    diffs = _differences(states)
    return float(np.log2(diffs[-2] / diffs[-1]))


@pytest.mark.validation
class TestOrderInTime:
    """BDF2 within 0.2 of 2, implicit Euler, the control, within 0.2 of 1."""

    def test_bdf2_is_second_order(self, ladder: dict[str, list[FlowState]]) -> None:
        """Catches: the BDF2 weights perturbed, the old levels swapped, the first
        step left BDF2 with a zero old level, BDF2 falling back to implicit Euler.
        """
        order = _finest_order(ladder[BDF2])
        assert abs(order - 2.0) <= ORDER_TOLERANCE, order

    def test_implicit_euler_is_first_order(
        self, ladder: dict[str, list[FlowState]]
    ) -> None:
        order = _finest_order(ladder[IMPLICIT_EULER])
        assert abs(order - 1.0) <= ORDER_TOLERANCE, order

    def test_the_iteration_error_is_far_below_the_smallest_difference(
        self, ladder: dict[str, list[FlowState]]
    ) -> None:
        """BDF2's finest run against itself at a criterion 100 times tighter.

        BDF2's finest difference is the smallest the ladder measures, and its
        finest run takes the most steps, so the iteration error it carries is
        the one that could reach a difference; docs/reports/probe67 records
        the coarsest run's and implicit Euler's as well.
        """
        smallest = min(
            _differences(ladder[BDF2]) + _differences(ladder[IMPLICIT_EULER])
        )
        reference = _spin_up(BDF2, STEPS[-1], REFERENCE_TOL)
        iteration = _difference(ladder[BDF2][-1], reference)
        assert iteration < ITERATION_SHARE * smallest, (iteration, smallest)
