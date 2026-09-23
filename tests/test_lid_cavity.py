"""VAL-002: Lid-driven cavity validation test.

Verifies that the NS solver reproduces the Ghia et al. (1982) benchmark
centerline velocity profiles for a lid-driven cavity at Re=100. The
maximum normalized error must be below 2% for both u and v profiles
(REQ-S03). The case configuration is configs/validation_cavity.yaml and
the metric is validation.metrics.cavity_centerline_errors, both shared
with the benchmark harness so the two cannot drift apart.
"""

import pytest

from src.boundary import BoundaryManager
from src.mesh import Mesh
from src.solver_ns import NavierStokesSolver
from validation.cases import load_case
from validation.metrics import cavity_centerline_errors


@pytest.mark.validation
@pytest.mark.xfail(
    reason=(
        "The collocated solver fails the 2% criterion in both u and v against the "
        "corrected reference ghia_1982_re100_r2; the staggered solver's VAL-002 "
        "validation is ECR-001 steps 7 and 8"
    )
)
def test_lid_driven_cavity_val002() -> None:
    """VAL-002: Lid-driven cavity -- centerline profiles within 2% of Ghia et al.

    Solves steady flow in a square cavity at Re=100 with a moving top
    lid on the grid the case file specifies. Compares u-velocity along
    the vertical centerline and v-velocity along the horizontal
    centerline against the Ghia et al. (1982) benchmark data.
    """
    config = load_case("cavity")
    mesh = Mesh(config)
    boundary = BoundaryManager(mesh, config)
    solver = NavierStokesSolver(mesh, config, boundary)

    u, v, _p = solver.solve_steady()

    n_iter = len(solver.residual_history)
    final_residual = solver.compute_residual()
    metric = cavity_centerline_errors(config, mesh, u, v)
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
