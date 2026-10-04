"""VAL-014: the sealed box loses mass at settling velocity over height (ADR-011 D and H).

A closed 8.0 m by 3.0 m room on 40x30 cells of 0.2 m by 0.1 m, no flow, the
5 um class with its Stokes settling velocity, diffusion off, a uniform initial
concentration. Nothing enters through the ceiling, so a front comes down from
it at v_s while the floor row still holds C_0, and the floor removes
v_s C_0 W per second: the exact line is M(t) = M_0 (1 - v_s t / H) until the
front reaches the floor. The faces are the case's own, the floor's deposition
velocity equal to v_s exactly, so the line is exact (through
ConcentrationBoundary the floor would carry v_s + D / delta and the row would
not be stationary).

The run is 20 steps at cfl_number 0.4: the front has moved 8 of the 30 cells,
and the scheme's leading edge, one cell per step, has reached 20, ten rows
above the floor (deferred D1 of PR 30 asked for this margin to be stated).

The control that guards section D's composition plants the settling increment
on the floor face as well, the double count test 30 B1 found: the floor face
then carries an advective outflow beside the deposition, the floor row falls,
and the line fails at the first step (ADR-011 H as amended 2026-10-03; the
earlier doubled-floor control was dropped, since the deposit over a long run
is set by the settling supply from above and not by the floor's velocity).
"""

from time import perf_counter

import numpy as np
import pytest

from src.solver_transport import TransportSolver
from validation.transport_cases import SEALED_BOX, TransportCase, sealed_box_case

STEPS = 20
LINE_TOLERANCE = 1.0e-10
ROUNDING = 1.0e-14


def _run(
    case: TransportCase, steps: int = STEPS
) -> tuple[np.ndarray, TransportSolver, float]:
    solver = TransportSolver(case.mesh, case.config, case.physics, case.conditions)
    dt = solver.stable_dt(case.faces, 0)
    c = case.initial.copy()
    for _ in range(steps):
        c = solver.solve_timestep(c, case.faces, 0, dt)
    return c, solver, dt


@pytest.mark.validation
def test_sealed_box_decay_val014() -> None:
    """VAL-014: floor deposit equals v_s C_0 W T to 1e-10, the budget closes, bounds hold."""
    case = sealed_box_case()
    v_s = case.physics.settling
    start = perf_counter()
    c, solver, dt = _run(case)
    seconds = perf_counter() - start
    t_total = STEPS * dt
    width, height, c_0 = SEALED_BOX["width"], SEALED_BOX["height"], SEALED_BOX["c_0"]
    expected = v_s * c_0 * width * t_total
    budget = solver.budget[0]
    front_cells = v_s * t_total / case.mesh.dy
    print(
        f"VAL-014: v_s {v_s:.4e} m/s, {STEPS} steps of {dt:.2f} s, T = {t_total:.1f} s "
        f"of H / v_s = {height / v_s:.0f} s, in {seconds:.2f} s; front at "
        f"{front_cells:.1f} of {SEALED_BOX['ny']} cells, leading edge at {STEPS}, "
        f"{SEALED_BOX['ny'] - STEPS} rows above the floor"
    )
    print(
        f"  floor deposit {budget.deposited['floor']:.12e} against v_s C_0 W T "
        f"{expected:.12e}: relative difference "
        f"{budget.deposited['floor'] / expected - 1.0:.2e} (criterion {LINE_TOLERANCE}); "
        f"budget relative residual {budget.relative():.2e}; field in "
        f"[{c.min():.6e}, {c.max():.6e}]"
    )
    assert budget.deposited["floor"] == pytest.approx(expected, rel=LINE_TOLERANCE)
    assert budget.current + sum(budget.deposited.values()) == pytest.approx(
        budget.initial, rel=1e-12
    )
    assert budget.initial == pytest.approx(c_0 * width * height, rel=1e-12)
    assert budget.inflow == 0.0 and budget.outflow == 0.0 and budget.source == 0.0
    assert all(budget.deposited[k] == 0.0 for k in ("ceiling", "wall", "obstacle"))
    assert c.max() <= c_0 * (1.0 + ROUNDING)
    assert c.min() >= 0.0
    # The rows the leading edge has not reached are untouched, bitwise above
    # the floor row; the floor row itself passes through the implicit solve
    # as C_0 (1 + a) / (1 + a) and keeps C_0 to an ulp.
    assert np.all(c[1 : SEALED_BOX["ny"] - STEPS, :] == c_0)
    assert np.allclose(c[0, :], c_0, rtol=1e-14, atol=0.0)
    assert c[-1, :].max() < c_0


@pytest.mark.validation
def test_the_settling_increment_on_the_floor_face_breaks_the_line_val014() -> None:
    """The control that guards ADR-011 D: with the increment planted on the
    floor face too, the floor row carries an advective outflow beside its
    deposition, falls toward C_0 / 2, and the deposit misses the line by
    about half over the run."""
    case = sealed_box_case(settle_floor_face=True)
    c, solver, dt = _run(case)
    v_s = case.physics.settling
    expected = v_s * SEALED_BOX["c_0"] * SEALED_BOX["width"] * STEPS * dt
    budget = solver.budget[0]
    miss = budget.deposited["floor"] / expected - 1.0
    print(
        f"VAL-014 control, increment on the floor face: deposit misses the line by "
        f"{miss:+.3f}; outflow booked through the floor {budget.outflow:.4e}; floor row "
        f"{c[0, 0]:.6e} against C_0 {SEALED_BOX['c_0']:.6e}"
    )
    assert abs(miss) > 1e-2
    assert budget.outflow > 0.0
    assert c[0, 0] < SEALED_BOX["c_0"]
