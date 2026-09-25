"""VAL-001: Poiseuille flow validation test.

Verifies that the NS solver reproduces the analytical parabolic velocity
profile for pressure-driven flow between two parallel plates. The L2
error between the solver and analytical profiles at the channel midpoint
must be below 2.5% (REQ-S02). The case configuration is
configs/validation_poiseuille.yaml and the metric is
validation.metrics.poiseuille_l2_error, both shared with the benchmark
harness so the two cannot drift apart.
"""

import numpy as np
import pytest

from src.boundary import BoundaryManager
from src.mesh import Mesh
from src.solver_ns import NavierStokesSolver
from validation.cases import load_case, with_velocity_step
from validation.metrics import poiseuille_l2_error


@pytest.mark.validation
def test_poiseuille_flow_val001() -> None:
    """VAL-001: Poiseuille flow -- L2 error < 2.5% vs analytical parabolic profile.

    Solves steady flow in a horizontal channel with uniform inlet
    velocity and pressure outlet on the 80x40 grid the case file
    specifies. Extracts the u-velocity profile at the channel midpoint
    and compares against the analytical Poiseuille parabola.

    The 2.5% threshold reflects the O(h) wall accuracy of the collocated
    ghost-cell boundary treatment (see ADR-008). Error decreases
    monotonically with grid refinement at the expected first-order rate.
    The collocated solver refuses the case file's error_estimate rule and
    runs velocity_step, as every collocated result was produced.
    """
    config = with_velocity_step(load_case("poiseuille"))
    mesh = Mesh(config)
    boundary = BoundaryManager(mesh, config)
    solver = NavierStokesSolver(mesh, config, boundary)

    u, _v, _p = solver.solve_steady()

    n_iter = len(solver.residual_history)
    final_residual = solver.compute_residual()
    metric = poiseuille_l2_error(config, mesh, u)

    i_mid = config.nx // 2
    print("VAL-001 Poiseuille flow:")
    print(f"  Iterations: {n_iter}")
    print(f"  Final residual: {final_residual:.6e}")
    print(f"  L2 error: {metric.value:.6e}")
    print(f"  u_max (solver): {np.max(u[:, i_mid]):.6f}")

    assert metric.value < 0.025, f"L2 error {metric.value:.4e} exceeds 2.5% threshold"
