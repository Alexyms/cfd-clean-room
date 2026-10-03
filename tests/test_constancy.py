"""VAL-012: a uniform field stays uniform to the stopping rule's bound (REQ-T11, ADR-011 G).

The committed VAL-001 case at 40x20 is solved under its own error_estimate
rule; its face velocities carry a per-cell mass imbalance b_P below
mass_imbalance_tol. A field of ones, with the inlet carrying one and diffusion,
settling, deposition and sources off, is advected for about 40 s at the stable
step. REQ-T11 bounds the largest relative departure by the largest over cells
of |b_P| T / (rho V_P); the test asserts the measured departure is below that
bound and above a tenth of it, so the drift is bounded and the mechanism is
the one the bound describes. The planted control perturbs one interior face
velocity, whose drift must exceed the bound.
"""

import math
from time import perf_counter

import numpy as np
import pytest
import yaml

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.pressure import PressureCorrector
from src.solver_staggered import StaggeredSolver
from src.solver_transport import TransportSolver
from src.staggered import FaceVelocities, u_shape
from validation.cases import case_path
from validation.transport_cases import (
    CFL_NUMBER,
    FixedConditions,
    ScalarPhysics,
    conditions_with,
)

GRID = (40, 20)
# About 40 s, so the predicted departure (about 2e-7) is six orders above the
# rounding floor and the second-order term six orders below it (ADR-011 G).
SIMULATED_SECONDS = 40.0


def _channel_config() -> SimConfig:
    """VAL-001 at 40x20, as committed, with a transport section added."""
    raw = yaml.safe_load(case_path("poiseuille").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = GRID
    raw["transport"] = {
        "cfl_number": CFL_NUMBER,
        "advection_scheme": "umist",
        "max_diffusion_iter": 100,
        "diffusion_tol": 1.0e-10,
    }
    config = SimConfig.from_dict(raw)
    assert config.stopping_rule == "error_estimate"
    return config


def _solve_faces(config: SimConfig) -> tuple[Mesh, FaceVelocities, np.ndarray, float]:
    """The converged faces, their per-cell imbalance, and the solve's wall time."""
    mesh = Mesh(config)
    boundary = StaggeredBoundary(mesh, config)
    solver = StaggeredSolver(mesh, config, boundary)
    start = perf_counter()
    solver.solve_steady()
    seconds = perf_counter() - start
    assert solver.stop_reason == "error_estimate_and_continuity"
    faces = solver.face_velocities
    imbalance = PressureCorrector(mesh, config, boundary).mass_imbalance(
        faces.u, faces.v
    )
    return mesh, faces, imbalance, seconds


def _drift(
    mesh: Mesh, config: SimConfig, faces: FaceVelocities, steps: int
) -> tuple[float, float]:
    """Largest relative departure of a field of ones after ``steps`` steps, and T."""
    inflow_u = np.zeros(u_shape(mesh))
    inflow_u[:, 0] = 1.0
    conditions = FixedConditions(conditions_with(mesh, inflow_u=inflow_u))
    solver = TransportSolver(
        mesh, config, ScalarPhysics(settling=0.0, diffusion=0.0), conditions
    )
    dt = solver.stable_dt(faces, 0)
    c = np.ones((config.ny, config.nx))
    for _ in range(steps):
        c = solver.solve_timestep(c, faces, 0, dt)
    live = mesh.cell_type != SOLID
    return float(np.abs(c[live] - 1.0).max()), steps * dt


@pytest.mark.validation
def test_uniform_field_drifts_within_the_imbalance_bound_val012() -> None:
    """VAL-012: measured departure in [bound / 10, bound], bound = max |b_P| T / (rho V_P)."""
    config = _channel_config()
    mesh, faces, imbalance, solve_seconds = _solve_faces(config)
    transport = TransportSolver(
        mesh,
        config,
        ScalarPhysics(settling=0.0, diffusion=0.0),
        FixedConditions(conditions_with(mesh)),
    )
    dt = transport.stable_dt(faces, 0)
    steps = math.ceil(SIMULATED_SECONDS / dt)
    start = perf_counter()
    departure, t_total = _drift(mesh, config, faces, steps)
    seconds = perf_counter() - start

    volume = np.outer(mesh.dy_cell, mesh.dx_cell)
    live = mesh.cell_type != SOLID
    rate = np.abs(imbalance) / (config.rho * volume)
    bound = float(rate[live].max()) * t_total
    worst = np.unravel_index(np.argmax(np.where(live, rate, 0.0)), rate.shape)
    print(
        f"VAL-012: VAL-001 {GRID[0]}x{GRID[1]} solved in {solve_seconds:.1f} s; worst "
        f"|b_P| {np.abs(imbalance)[live].max():.3e} kg/s at cell (j, i) = {worst}, "
        f"rate {rate[live].max():.3e} per second"
    )
    print(
        f"  {steps} steps of {dt:.4e} s, T = {t_total:.2f} s, in {seconds:.1f} s; "
        f"largest departure {departure:.3e}; bound {bound:.3e}; departure / bound "
        f"{departure / bound:.3f}"
    )
    assert departure <= bound
    assert departure >= bound / 10.0

    # The planted control: one interior face perturbed by 1e-6 m/s, an
    # imbalance of rho dy 1e-6 in two cells, whose drift 1e-6 T / dx is four
    # orders above the bound.
    u = faces.u.copy()
    u[GRID[1] // 2, GRID[0] // 2] += 1.0e-6
    perturbed = FaceVelocities.copy_of(u, faces.v)
    control, _ = _drift(mesh, config, perturbed, steps)
    print(
        f"  control, one interior face perturbed by 1e-6 m/s: departure {control:.3e}"
    )
    assert control > bound
