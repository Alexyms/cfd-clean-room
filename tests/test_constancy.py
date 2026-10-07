"""VAL-012: a uniform field stays uniform to the stopping rule's bound (REQ-T11, ADR-011 G).

REQ-T11 bounds the largest relative departure of a uniform field by the
largest over cells of |b_P| T / (rho V_P), b_P the cell's mass imbalance in
the advecting face field. VAL-012 shows two things, each on the field that
can still show it (Alex, 2026-10-07).

The requirement, on the solver's faces. The committed VAL-001 case at 40x20
is solved under its own error_estimate rule, a field of ones is advected on
its faces for about 40 s with the inlet carrying one and diffusion, settling,
deposition and sources off, and the departure must be at most the bound.
Since ECR-003 step 1 (conjugate gradients to pressure_rtol 1e-8) those faces
are balanced to rounding, worst |b_P| about 6e-16 kg/s, so the departure and
the bound are both rounding-level quantities and the check holds by a factor
of about ten. The planted control perturbs one interior face velocity, whose
drift must exceed the bound, so the bound is not vacuous on these faces.

The mechanism, on a planted field. Under weighted Jacobi the same faces
carried an imbalance six orders above rounding, and the test also asserted
the departure above a tenth of the bound, so that the drift was shown to be
the mechanism the bound describes. That clause moves to a prescribed face
field with a known imbalance, where its ratio is predicted before it runs.
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
from src.staggered import FaceVelocities, u_shape, v_shape
from validation.cases import case_path
from validation.transport_cases import (
    CFL_NUMBER,
    FixedConditions,
    ScalarPhysics,
    conditions_with,
)

GRID = (40, 20)
# About 40 s on the solver's faces. Under weighted Jacobi this put the
# predicted departure (about 2e-7) six orders above the rounding floor; under
# conjugate gradients the faces' own imbalance is rounding, and the length
# only sets how much of it accumulates.
SIMULATED_SECONDS = 40.0
# The planted field: uniform flow at the VAL-001 inlet speed, one whole face
# column raised by a thousandth of it, an imbalance ten orders above rounding.
PLANTED_SPEED = 0.1
PLANTED_RISE = 1.0e-3 * PLANTED_SPEED
# Ten steps at the stable step, Courant 0.1, make T one residence time of a
# planted cell (ADR-011 G: on an open field a parcel is flushed, so its
# departure is the imbalance rate times the residence time, not T).
PLANTED_STEPS = 10


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


def _bound(
    mesh: Mesh, config: SimConfig, imbalance: np.ndarray, t_total: float
) -> tuple[float, np.ndarray]:
    """REQ-T11's bound, max over non-SOLID cells of |b_P| T / (rho V_P), and the rates."""
    volume = np.outer(mesh.dy_cell, mesh.dx_cell)
    live = mesh.cell_type != SOLID
    rate = np.abs(imbalance) / (config.rho * volume)
    return float(rate[live].max()) * t_total, rate


@pytest.mark.validation
def test_uniform_field_drifts_within_the_imbalance_bound_val012() -> None:
    """VAL-012, the requirement: on the solver's faces the departure is at most the bound.

    The faces are balanced to rounding under conjugate gradients, so the
    lower clause this test used to assert, departure at least a tenth of the
    bound, sat on the 0.1 line (0.100 on Windows, 0.099 on Linux CI) and has
    moved to the planted-field test below. Pressure tolerances from 1e-8 to
    1e-4 leave the worst |b_P| at 6.2e-16 to 6.4e-16 kg/s, so no setting of
    this case restores an imbalance above rounding.
    """
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

    bound, rate = _bound(mesh, config, imbalance, t_total)
    live = mesh.cell_type != SOLID
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

    # The planted control: one interior face perturbed by 1e-6 m/s, an
    # imbalance of rho dy 1e-6 in two cells. A closed cell would drift by
    # 1e-6 T / dx, 1.6e-3; on the open channel the perturbed cells are flushed
    # and the measured departure is 6.7e-6, far above the bound of the
    # balanced faces (test 32; the bound was 1.9e-7 under weighted Jacobi).
    u = faces.u.copy()
    u[GRID[1] // 2, GRID[0] // 2] += 1.0e-6
    perturbed = FaceVelocities.copy_of(u, faces.v)
    control, _ = _drift(mesh, config, perturbed, steps)
    print(
        f"  control, one interior face perturbed by 1e-6 m/s: departure {control:.3e}"
    )
    assert control > bound


@pytest.mark.validation
def test_a_planted_imbalance_drifts_by_the_mechanism_the_bound_describes_val012() -> (
    None
):
    """VAL-012, the mechanism: on a planted field the ratio is the predicted 0.651.

    Every vertical face of the VAL-001 40x20 mesh carries U = 0.1 m/s and
    every horizontal face zero, except one whole face column, i0 = nx / 4,
    raised to U + delta, delta = 1e-4 m/s. Column i0 - 1 then has a net
    outflow rho delta dy, 2.5e-6 kg/s, ten orders above rounding, and drains at
    delta / dx per second; column i0 gains at the same rate; every other cell
    is balanced exactly. Upstream of the draining cell the field stays exactly
    one and the draining cell is a local minimum, so the limiter returns the
    upwind value at both its faces and the cell follows forward Euler,
    C <- C + dt / dx (U - (U + delta) C). After N steps its departure is
    (delta / (U + delta)) (1 - (1 - s dt)^N) with s = (U + delta) / dx, and the
    bound is (delta / dx) N dt, so the ratio is (1 - (1 - s dt)^N) / (s N dt):
    1 - 0.9^10 = 0.651 at the stable step (s dt = 0.1) and N = 10, T one
    residence time of the cell. The gaining column's excess peaks near
    0.37 delta / U, below the draining cell's departure, so the largest
    departure is the draining cell's. Predicted before it ran; measured
    0.6513 on 2026-10-07, within 1e-13 of the closed form.

    Both clauses hold, so the drift is bounded and is the mechanism the bound
    describes; the closed form is asserted too. Defect caught: a transport
    step that is not conservative at one face, which leaves a cell a source
    or sink of the field that no imbalance accounts for.
    """
    config = _channel_config()
    mesh = Mesh(config)
    u = np.full(u_shape(mesh), PLANTED_SPEED)
    i0 = config.nx // 4
    u[:, i0] += PLANTED_RISE
    faces = FaceVelocities.copy_of(u, np.zeros(v_shape(mesh)))
    imbalance = PressureCorrector(
        mesh, config, StaggeredBoundary(mesh, config)
    ).mass_imbalance(faces.u, faces.v)
    unbalanced = np.flatnonzero(np.abs(imbalance).max(axis=0) > 0.0)
    assert list(unbalanced) == [i0 - 1, i0]

    departure, t_total = _drift(mesh, config, faces, PLANTED_STEPS)
    bound, _rate = _bound(mesh, config, imbalance, t_total)
    dt = t_total / PLANTED_STEPS
    s_dt = (PLANTED_SPEED + PLANTED_RISE) / float(mesh.dx_cell[i0 - 1]) * dt
    kept = (1.0 - s_dt) ** PLANTED_STEPS
    predicted = PLANTED_RISE / (PLANTED_SPEED + PLANTED_RISE) * (1.0 - kept)
    print(
        f"VAL-012 planted: {PLANTED_STEPS} steps of {dt:.4e} s (s dt = {s_dt:.4f}), "
        f"departure {departure:.4e}, bound {bound:.4e}, ratio {departure / bound:.4f}, "
        f"predicted {(1.0 - kept) / (PLANTED_STEPS * s_dt):.4f}"
    )
    assert s_dt == pytest.approx(CFL_NUMBER, rel=1e-12)
    assert departure == pytest.approx(predicted, rel=1e-9)
    assert departure <= bound
    assert departure >= bound / 10.0
