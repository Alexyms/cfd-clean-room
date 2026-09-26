"""VAL-002: Lid-driven cavity validation tests.

Verifies that the NS solver reproduces the centerline velocity profiles of a
lid-driven cavity at Re=100: the maximum normalized error must be below 2%
for both u and v (REQ-S03). The staggered solver is scored against Marchi,
Suero and Araki (2009) by validation.metrics.cavity_marchi_centerline_errors,
as ECR-001 criteria 3 and 3a have been since the 2026-09-24 amendment, with
Ghia et al. (1982) printed beside it and no threshold. The collocated solver
keeps its xfail and its Ghia metric, cavity_true_centerline_errors. The case
configuration is configs/validation_cavity.yaml, shared with the benchmark
harness so the two cannot drift apart.
"""

import time

import pytest

from src.boundary import BoundaryManager
from src.boundary_staggered import StaggeredBoundary
from src.mesh import Mesh
from src.solver_ns import NavierStokesSolver
from src.solver_staggered import StaggeredSolver
from validation.cases import load_case, with_velocity_step
from validation.metrics import (
    cavity_marchi_centerline_errors,
    cavity_true_centerline_errors,
)


@pytest.mark.validation
@pytest.mark.xfail(
    reason=(
        "The collocated solver fails the 2% criterion in both u and v against the "
        "corrected reference ghia_1982_re100_r2, on the true centerlines as well as "
        "half a cell off them; the staggered solver's VAL-002 is the test below"
    )
)
def test_lid_driven_cavity_val002() -> None:
    """VAL-002: Lid-driven cavity -- centerline profiles within 2% of Ghia et al.

    Solves steady flow in a square cavity at Re=100 with a moving top
    lid on the grid the case file specifies. Compares u-velocity along
    the vertical centerline and v-velocity along the horizontal
    centerline against the Ghia et al. (1982) benchmark data.
    """
    config = with_velocity_step(load_case("cavity"))
    mesh = Mesh(config)
    boundary = BoundaryManager(mesh, config)
    solver = NavierStokesSolver(mesh, config, boundary)

    u, v, _p = solver.solve_steady()

    n_iter = len(solver.residual_history)
    final_residual = solver.compute_residual()
    metric = cavity_true_centerline_errors(config, mesh, u, v)
    max_u_error = metric.components["u"]
    max_v_error = metric.components["v"]

    print("VAL-002 Lid-driven cavity (Re=100):")
    print(f"  Iterations: {n_iter}")
    print(f"  Final residual: {final_residual:.6e}")
    print(f"  Max u-error (normalized): {max_u_error:.6e}")
    print(f"  Max v-error (normalized): {max_v_error:.6e}")

    assert max_u_error < 0.02, (
        f"u-velocity error {max_u_error:.4e} exceeds 2% threshold"
    )
    assert max_v_error < 0.02, (
        f"v-velocity error {max_v_error:.4e} exceeds 2% threshold"
    )


@pytest.mark.validation
def test_lid_driven_cavity_staggered_val002() -> None:
    """VAL-002 on the staggered solver: u and v within 2% of Marchi et al. (2009).

    Runs the case file's 40x40 under its error_estimate rule, which must
    stop the solve rather than the cap. ECR-001 criterion 3 names 80x80,
    which is judged from a harness row, not in CI.
    """
    config = load_case("cavity")
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))

    start = time.perf_counter()
    u, v, _p = solver.solve_steady()
    seconds = time.perf_counter() - start
    metric = cavity_marchi_centerline_errors(config, mesh, u, v).components
    ghia = cavity_true_centerline_errors(config, mesh, u, v).components

    print(f"VAL-002 staggered, {config.nx}x{config.ny}:")
    print(f"  Outer iterations: {len(solver.residual_history)} in {seconds:.1f} s")
    print(f"  Stop: {solver.stop_reason}")
    print(f"  Against Marchi: u {metric['u']:.6e}, v {metric['v']:.6e}")
    print(f"  Against Ghia, no threshold: u {ghia['u']:.6e}, v {ghia['v']:.6e}")

    assert solver.converged is True
    assert solver.stop_reason == "error_estimate_and_continuity"
    for axis in ("u", "v"):
        assert metric[axis] < 0.02, f"{axis} error {metric[axis]:.4e} exceeds 2%"
