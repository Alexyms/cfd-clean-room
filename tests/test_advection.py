"""VAL-004: pulse advection in two rows (REQ-T08; REQ-T12's no-negative clause; ADR-011 H).

Row 1 carries a four-cell Gaussian 1.0 m along a uniform field 30 degrees
above the x axis of a 100x60 channel; row 2 turns a four-cell puff through
one revolution of solid-body rotation on 64x64 cells. Both at the product
case's Courant number of 0.1. Thresholds are 1.5 times the error the chosen
scheme measured on the prototype (results/builder30b/pulse_2d.json), which
results/builder32/item0.md reproduces with this solver to four figures. Row 2
is recorded through FieldHistory every output_interval steps and saved under
pytest's tmp_path, so the format has its Phase 3 consumer; the deliverable
under results/builder32/ is written by results/builder32/puff_visual.py, which
runs the case itself. A test writes no deliverable (prompt 32b, decision 3).

The planted control switches the limiter off in the solver by monkeypatch, so
the face value is QUICK's unlimited quadratic on the same fields: the field
goes negative within a hundred steps, which is why the clause needs the clamp.
"""

import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

import src.solver_transport as solver_transport
from src.solver_transport import FieldHistory, TransportSolver
from validation.metrics import centroid, field_minimum, peak_retention, relative_l2
from validation.transport_cases import (
    TransportCase,
    oblique_pulse_case,
    rotating_puff_case,
)

# The plan's gate rows: 1.5 times the measured 82.2% / 11.3% and 75.3% / 15.5%.
ROW_1 = {"peak_min": 0.73, "l2_max": 0.17}
ROW_2 = {"peak_min": 0.62, "l2_max": 0.24}
# The no-negative bound is exact for a bounded scheme; a subtraction of equal
# fluxes may leave rounding (ADR-011 H).
ROUNDING = 1e-14
CENTROID_CELLS_MAX = 1.0


def _run(
    case: TransportCase, history: FieldHistory | None = None, steps: int | None = None
) -> tuple[np.ndarray, TransportSolver, int, float]:
    """Run a case to t_end (or a given number of steps) at its stable step."""
    solver = TransportSolver(case.mesh, case.config, case.physics, case.conditions)
    n_steps = math.ceil(case.t_end / solver.stable_dt(case.faces, 0))
    dt = case.t_end / n_steps
    if steps is not None:
        n_steps = steps
    c = case.initial.copy()
    if history is not None:
        history.record(0, 0.0, {0: c})
    for step in range(1, n_steps + 1):
        c = solver.solve_timestep(c, case.faces, 0, dt)
        if history is not None:
            history.record(step, step * dt, {0: c})
    return c, solver, n_steps, dt


def _metrics(c: np.ndarray, case: TransportCase) -> dict[str, float]:
    mesh = case.mesh
    (xa, ya), (xe, ye) = centroid(c, mesh), centroid(case.exact, mesh)
    return {
        "peak": peak_retention(c, case.exact, mesh),
        "l2": relative_l2(c, case.exact, mesh),
        "minimum_over_peak": field_minimum(c, mesh) / float(case.exact.max()),
        "centroid_cells": math.hypot(xa - xe, ya - ye) / mesh.dx,
    }


def _report(
    label: str,
    m: dict[str, float],
    thresholds: dict[str, float],
    seconds: float,
    steps: int,
) -> None:
    print(
        f"{label}: {steps} steps in {seconds:.1f} s; peak retained {m['peak']:.4f} "
        f"(> {thresholds['peak_min']}); relative L2 {m['l2']:.4f} (< {thresholds['l2_max']}); "
        f"minimum / peak {m['minimum_over_peak']:.2e} (>= -{ROUNDING}); centroid error "
        f"{m['centroid_cells']:.4f} cells (< {CENTROID_CELLS_MAX})"
    )


def _assert_row(m: dict[str, float], thresholds: dict[str, float]) -> None:
    assert m["peak"] > thresholds["peak_min"]
    assert m["l2"] < thresholds["l2_max"]
    assert m["minimum_over_peak"] >= -ROUNDING
    assert m["centroid_cells"] < CENTROID_CELLS_MAX


@pytest.mark.validation
def test_oblique_channel_pulse_val004_row1() -> None:
    """VAL-004 row 1: peak > 73%, L2 < 17%, no cell below zero, centroid within a cell."""
    case = oblique_pulse_case()
    start = perf_counter()
    c, solver, steps, _ = _run(case)
    m = _metrics(c, case)
    _report(
        "VAL-004 row 1 (oblique channel pulse)", m, ROW_1, perf_counter() - start, steps
    )
    assert steps == 684
    _assert_row(m, ROW_1)
    # The pulse leaves nothing but its cut tail: what left is booked as outflow.
    assert abs(solver.budget[0].relative()) < 1e-13


@pytest.mark.validation
def test_rotating_puff_val004_row2(tmp_path: Path) -> None:
    """VAL-004 row 2: after one revolution, peak > 62%, L2 < 24% against the
    initial field, no cell below zero, centroid within a cell; the puff is
    recorded through FieldHistory and saved for Phase 7."""
    case = rotating_puff_case()
    history = FieldHistory(case.config.output_interval)
    start = perf_counter()
    c, solver, steps, _ = _run(case, history)
    m = _metrics(c, case)
    _report("VAL-004 row 2 (rotating puff)", m, ROW_2, perf_counter() - start, steps)
    assert steps == 3959
    _assert_row(m, ROW_2)
    assert abs(solver.budget[0].relative()) < 1e-13

    assert len(history.frames) == steps // case.config.output_interval + 1
    assert history.frames[0][0] == 0 and history.frames[-1][0] <= steps
    path = tmp_path / "puff_frames.npz"
    history.save(path)
    with np.load(path) as data:
        assert data["C_0"].shape == (len(history.frames), 64, 64)
        assert np.array_equal(data["C_0"][0], case.initial)
    print(
        f"  {len(history.frames)} frames every {case.config.output_interval} steps saved to {path}"
    )


@pytest.mark.validation
def test_the_unlimited_quick_face_value_breaks_the_no_negative_clause_val004(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The planted control of ADR-011 H: with the clamp removed the face value
    is QUICK's quadratic, and row 1's pulse goes below zero within 100 steps
    by far more than the rounding allowance, so the clause can fail."""
    case = oblique_pulse_case()
    monkeypatch.setattr(
        solver_transport,
        "limited_face_values",
        lambda c_up, c_c, c_d, quick: quick,
    )
    c, _, _, _ = _run(case, steps=100)
    minimum = field_minimum(c, case.mesh) / float(case.exact.max())
    print(
        f"VAL-004 control, unlimited QUICK after 100 steps: minimum / peak {minimum:.2e}"
    )
    assert minimum < -1e-6
    assert minimum < -ROUNDING
