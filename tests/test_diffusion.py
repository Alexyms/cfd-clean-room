"""VAL-003: pure diffusion of a Gaussian against the heat kernel (REQ-T07, ADR-011 H).

A closed 2.0 m by 1.2 m box on 200x120 cells, a zero face field, a synthetic
D = 1e-3 m^2/s, and the Gaussian of standard deviation five cells centred off
every node, as exact cell averages, run until sigma has doubled. Section H
leaves the time step to the build "so that the backward Euler error is below
the spatial one". The test measures that split itself: two runs at diffusion
numbers 0.25 and 0.125 fit the first-order model error = spatial + time * d,
and the gate runs at 0.1, where the fit puts the time share under half. The
shares are printed (prompt 32b, decision 2).
"""

import math
from time import perf_counter

import numpy as np
import pytest

from src.solver_transport import TransportSolver
from validation.metrics import field_minimum, relative_l2
from validation.transport_cases import DIFFUSION, TransportCase, diffusion_case

# The plan's gate row.
L2_CRITERION = 0.01
# ADR-011 H asks for the budget to close to rounding. The implicit solve stops
# at a cell residual of 1e-13 of the largest right-hand side, so a step's
# budget error is at most that times the number of cells times the largest
# cell content: about 1.5e-11 of the Gaussian's content per step, 2e-9 over the
# 150 steps at 0.25 and 6e-9 over the gate's 375, in the worst case where every
# residual has the same sign (review 32 S10's bound, checked by test 32b). The
# measured values are a few 1e-15, so the criterion is the measured order with
# margin, not the worst case.
BUDGET_CRITERION = 1e-10
# The two runs the split is fitted from, and the gate's step from the fit.
SPLIT_DIFFUSION_NUMBERS = (0.25, 0.125)
GATE_DIFFUSION_NUMBER = 0.1
TIME_SHARE_LIMIT = 0.5


def _run(case: TransportCase, diffusion_number: float) -> dict[str, float]:
    mesh = case.mesh
    dt = diffusion_number * mesh.dx**2 / DIFFUSION["diffusivity"]
    n_steps = round(case.t_end / dt)
    assert n_steps * dt == pytest.approx(case.t_end)
    solver = TransportSolver(mesh, case.config, case.physics, case.conditions)
    c = case.initial.copy()
    sweeps = 0
    start = perf_counter()
    for _ in range(n_steps):
        c = solver.solve_timestep(c, case.faces, 0, dt)
        sweeps += solver.last_diffusion_sweeps
        assert solver.diffusion_converged
    return {
        "dt": dt,
        "steps": n_steps,
        "seconds": perf_counter() - start,
        "sweeps_per_step": sweeps / n_steps,
        "l2": relative_l2(c, case.exact, mesh),
        "minimum": field_minimum(c, mesh),
        "peak_ratio": float(c.max() / case.exact.max()),
        "budget_relative": solver.budget[0].relative(),
    }


def _report(d: float, r: dict[str, float]) -> None:
    print(
        f"VAL-003 diffusion number {d}: {r['steps']} steps of {r['dt']:.4f} s in "
        f"{r['seconds']:.1f} s, {r['sweeps_per_step']:.1f} Jacobi sweeps per step"
    )
    print(
        f"  relative L2 {r['l2']:.4e} (criterion < {L2_CRITERION}); peak ratio "
        f"{r['peak_ratio']:.5f}; minimum {r['minimum']:.2e}; budget relative "
        f"residual {r['budget_relative']:.2e}"
    )


@pytest.mark.validation
def test_pure_diffusion_gaussian_val003() -> None:
    """VAL-003: relative L2 against the heat kernel below 1% at a step whose
    time error is below the spatial one.

    Backward Euler is first order in dt at a fixed mesh and both errors
    under-diffuse, so error(d) = spatial + time_per_unit * d. The two split
    runs give the fit; the gate step is where the fit's time share is under
    half, and the gate run must land on the fit. The budget closes to
    rounding on every run and no cell goes negative (REQ-T12).
    """
    case = diffusion_case()
    split = {d: _run(case, d) for d in SPLIT_DIFFUSION_NUMBERS}
    d_a, d_b = SPLIT_DIFFUSION_NUMBERS
    time_per_unit = (split[d_a]["l2"] - split[d_b]["l2"]) / (d_a - d_b)
    spatial = split[d_a]["l2"] - time_per_unit * d_a
    assert time_per_unit > 0.0 and spatial > 0.0
    for d in SPLIT_DIFFUSION_NUMBERS:
        _report(d, split[d])
        share = time_per_unit * d / split[d]["l2"]
        print(
            f"  fit: time error {time_per_unit * d:.3e} ({share:.0%}), spatial {spatial:.3e}"
        )
    gate_time = time_per_unit * GATE_DIFFUSION_NUMBER
    predicted = spatial + gate_time
    time_share = gate_time / predicted
    print(
        f"VAL-003 gate step: diffusion number {GATE_DIFFUSION_NUMBER}, fit predicts "
        f"{predicted:.3e} with time share {time_share:.0%} (limit {TIME_SHARE_LIMIT:.0%}), "
        f"spatial share {1 - time_share:.0%}"
    )
    assert time_share < TIME_SHARE_LIMIT

    gate = _run(case, GATE_DIFFUSION_NUMBER)
    _report(GATE_DIFFUSION_NUMBER, gate)
    assert gate["l2"] < L2_CRITERION
    assert gate["l2"] == pytest.approx(predicted, rel=0.05)
    for r in (*split.values(), gate):
        assert abs(r["budget_relative"]) < BUDGET_CRITERION
        assert r["minimum"] >= 0.0


@pytest.mark.validation
def test_val003_initial_field_carries_the_analytical_mass() -> None:
    """The discrete initial content is the Gaussian's 2 pi sigma_0^2 to rounding,
    so the budget's initial is the analytical one and not a quadrature of it."""
    case = diffusion_case()
    content = float((case.initial * case.mesh.dx * case.mesh.dy).sum())
    assert content == pytest.approx(
        2.0 * math.pi * DIFFUSION["sigma_0"] ** 2, rel=1e-12
    )
    assert np.all(case.initial >= 0.0)
