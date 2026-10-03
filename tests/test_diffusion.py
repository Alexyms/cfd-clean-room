"""VAL-003: pure diffusion of a Gaussian against the heat kernel (REQ-T07, ADR-011 H).

A closed 2.0 m by 1.2 m box on 200x120 cells, a zero face field, a synthetic
D = 1e-3 m^2/s, and the Gaussian of standard deviation five cells centred off
every node, as exact cell averages, run until sigma has doubled. The time step
is the test's: a diffusion number of 0.25 is the design's starting point, and
0.125 is run beside it so backward Euler's share of the error is visible.
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
# "The budget must close to rounding over the run" (ADR-011 H): the implicit
# solve stops at a residual of 1e-13 of the right-hand side, so the budget's
# relative residual is bounded by that times the number of steps.
BUDGET_CRITERION = 1e-10


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


@pytest.mark.validation
def test_pure_diffusion_gaussian_val003() -> None:
    """VAL-003: relative L2 of the cell field against the heat kernel below 1%.

    Run at diffusion numbers 0.25 and 0.125 over the same 3.75 s; the first
    is the gate, the second shows the time error's share. The budget closes to
    rounding on both, and no cell goes negative (REQ-T12).
    """
    case = diffusion_case()
    results = {d: _run(case, d) for d in (0.25, 0.125)}
    for d, r in results.items():
        print(
            f"VAL-003 diffusion number {d}: {r['steps']} steps of {r['dt']:.4f} s in "
            f"{r['seconds']:.1f} s, {r['sweeps_per_step']:.1f} Jacobi sweeps per step"
        )
        print(
            f"  relative L2 {r['l2']:.4e} (criterion < {L2_CRITERION}); peak ratio "
            f"{r['peak_ratio']:.5f}; minimum {r['minimum']:.2e}; budget relative "
            f"residual {r['budget_relative']:.2e}"
        )
    gate = results[0.25]
    assert gate["l2"] < L2_CRITERION
    assert results[0.125]["l2"] < gate["l2"]
    for r in results.values():
        assert abs(r["budget_relative"]) < BUDGET_CRITERION
        assert r["minimum"] >= 0.0
    # Backward Euler is first order in dt at a fixed mesh, so halving the
    # step must remove a visible share of the error, not a rounding's worth.
    assert gate["l2"] - results[0.125]["l2"] > 0.1 * results[0.125]["l2"]


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
