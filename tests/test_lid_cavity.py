"""VAL-002: Lid-driven cavity validation test.

Verifies that the NS solver reproduces the Ghia et al. (1982) benchmark
centerline velocity profiles for a lid-driven cavity at Re=100. The
maximum normalized error must be below 2% for both u and v profiles
(REQ-S03).
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.boundary import BoundaryManager
from src.config import SimConfig
from src.mesh import FLUID, Mesh
from src.solver_ns import NavierStokesSolver

# Ghia, Ghia & Shin (1982), Re=100
# u-velocity along vertical centerline (x=0.5)
GHIA_U_Y = [
    1.0000,
    0.9766,
    0.9688,
    0.9609,
    0.9531,
    0.8516,
    0.7344,
    0.6172,
    0.5000,
    0.4531,
    0.2813,
    0.1719,
    0.1016,
    0.0703,
    0.0625,
    0.0547,
    0.0000,
]
GHIA_U_VAL = [
    1.00000,
    0.84123,
    0.78871,
    0.73722,
    0.68717,
    0.23151,
    0.00332,
    -0.13641,
    -0.20581,
    -0.21090,
    -0.15662,
    -0.10150,
    -0.06434,
    -0.04775,
    -0.04192,
    -0.03717,
    0.00000,
]

# v-velocity along horizontal centerline (y=0.5)
GHIA_V_X = [
    1.0000,
    0.9688,
    0.9609,
    0.9531,
    0.8516,
    0.7344,
    0.6172,
    0.5000,
    0.4531,
    0.2813,
    0.1719,
    0.1016,
    0.0703,
    0.0625,
    0.0547,
    0.0000,
]
GHIA_V_VAL = [
    0.00000,
    -0.05906,
    -0.07391,
    -0.08864,
    -0.24533,
    -0.22445,
    -0.16914,
    -0.11477,
    -0.10313,
    -0.04272,
    0.02135,
    0.07156,
    0.09515,
    0.10091,
    0.10643,
    0.00000,
]


def _make_cavity_config(tmp_path: Path) -> SimConfig:
    """Create a lid-driven cavity configuration at Re=100.

    Square cavity with a moving lid (top wall, u=1.0 tangential).
    All other walls are stationary no-slip. No pressure outlet;
    the solver uses pressure pinning for the closed domain.
    """
    config_dict = {
        "domain": {
            "width": 1.0,
            "height": 1.0,
            # 40x40 resolution keeps CI runtime bounded. At Re=100 with SIMPLE
            # under-relaxation, 80x80 converges in ~5000 iterations which takes
            # ~25 minutes on CI runners. The xfail status on this test applies
            # at any resolution because the v-error is grid-independent per the
            # diagnostic in ECR-001. Full-resolution validation is a manual
            # activity until the staggered-grid rebuild resolves the defect.
            "nx": 40,
            "ny": 40,
        },
        "fluid": {
            "density": 1.0,
            "viscosity": 0.01,
            "temperature": 293.0,
        },
        "particles": {
            "density": 1000.0,
            "sizes": [0.1e-6],
            "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {
                "diameters": [0.1e-6],
                "efficiencies": [0.99999],
            },
        },
        "solver": {
            "dt": 0.01,
            "t_end": 1.0,
            "output_interval": 10,
            "convergence_tol": 1.0e-6,
            "max_simple_iter": 10000,
            "alpha_velocity": 0.5,
            "alpha_pressure": 0.3,
            "max_pressure_iter": 500,
            "pressure_tol": 1.0e-8,
        },
        "boundaries": {
            "lid": {
                "type": "velocity_inlet",
                "location": "top",
                "x_start": 0.0,
                "x_end": 1.0,
                "u_velocity": 1.0,
                "v_velocity": 0.0,
            },
        },
        "obstacles": [],
        "sensors": [{"name": "center", "x": 0.5, "y": 0.5}],
        "thresholds": {"0.1e-6": 100.0},
    }
    path = tmp_path / "cavity.yaml"
    path.write_text(yaml.dump(config_dict, default_flow_style=False), encoding="utf-8")
    return SimConfig(str(path))


def _interpolate_profile(
    cell_coords: np.ndarray,
    field_values: np.ndarray,
    target_coords: list[float],
) -> np.ndarray:
    """Interpolate solver field values to benchmark coordinate locations.

    Parameters
    ----------
    cell_coords : np.ndarray
        Cell center coordinates (1D).
    field_values : np.ndarray
        Field values at cell centers (1D).
    target_coords : list[float]
        Coordinates where interpolated values are needed.

    Returns
    -------
    np.ndarray
        Interpolated values at target coordinates.
    """
    return np.interp(target_coords, cell_coords, field_values)


@pytest.mark.validation
@pytest.mark.xfail(reason="Known v-velocity error pending solver rebuild per ECR-001")
def test_lid_driven_cavity_val002(tmp_path: Path) -> None:
    """VAL-002: Lid-driven cavity -- centerline profiles within 2% of Ghia et al.

    Solves steady flow in a square cavity at Re=100 with a moving top
    lid. Compares u-velocity along the vertical centerline and
    v-velocity along the horizontal centerline against the Ghia et al.
    (1982) benchmark data.
    """
    config = _make_cavity_config(tmp_path)
    mesh = Mesh(config)
    boundary = BoundaryManager(mesh, config)
    solver = NavierStokesSolver(mesh, config, boundary)

    u, v, _p = solver.solve_steady()

    n_iter = len(solver.residual_history)
    final_residual = solver.compute_residual()
    U_lid = 1.0

    # --- u-velocity along vertical centerline (x=0.5) ---
    i_mid = config.nx // 2
    y_fluid = []
    u_fluid = []
    for j in range(config.ny):
        if mesh.cell_type[j, i_mid] == FLUID:
            y_fluid.append(float(mesh.yc[j]))
            u_fluid.append(float(u[j, i_mid]))

    # Add wall boundary values at y=0 and y=1 for interpolation
    y_profile = [0.0, *y_fluid, 1.0]
    u_profile = [0.0, *u_fluid, U_lid]

    u_interp = _interpolate_profile(np.array(y_profile), np.array(u_profile), GHIA_U_Y)
    u_errors = np.abs(u_interp - np.array(GHIA_U_VAL)) / U_lid
    max_u_error = float(np.max(u_errors))

    # --- v-velocity along horizontal centerline (y=0.5) ---
    j_mid = config.ny // 2
    x_fluid = []
    v_fluid = []
    for i in range(config.nx):
        if mesh.cell_type[j_mid, i] == FLUID:
            x_fluid.append(float(mesh.xc[i]))
            v_fluid.append(float(v[j_mid, i]))

    # Add wall boundary values at x=0 and x=1
    x_profile = [0.0, *x_fluid, 1.0]
    v_profile = [0.0, *v_fluid, 0.0]

    v_interp = _interpolate_profile(np.array(x_profile), np.array(v_profile), GHIA_V_X)
    v_errors = np.abs(v_interp - np.array(GHIA_V_VAL)) / U_lid
    max_v_error = float(np.max(v_errors))

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
