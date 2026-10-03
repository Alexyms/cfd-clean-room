"""VAL-007: mass conservation on the solver's own budget (REQ-T05, ADR-011 F and H).

Two runs with every term on: an inlet carrying a concentration, an outlet,
deposition on the walls for the 5 um class with its settling, physical
diffusion and a source in one interior cell. (i) A seeded random face field
that is not divergence-free on 64x36 cells: the budget telescopes on any face
velocities, so what it tests is the flux assembly, that every interior face
appears in two cells with opposite signs and every boundary flux booked is
the one applied. (ii) The VAL-001 40x20 faces converged under error_estimate,
the field the product case will hand the solver. The criterion is the plan's
0.01%; the residual is printed beside it, expected at rounding. The planted
control drops one inlet face from the booking in the test's arithmetic, and
the budget must then miss the criterion.
"""

from time import perf_counter

import numpy as np
import pytest
import yaml

from src.boundary_concentration import ConcentrationBoundary
from src.boundary_registry import BoundaryRegistry
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import Mesh
from src.particles import ParticlePhysics
from src.solver_staggered import StaggeredSolver
from src.solver_transport import MassBudget, TransportSolver
from src.staggered import FaceVelocities
from validation.cases import case_path
from validation.transport_cases import (
    CFL_NUMBER,
    PARTICLE_SIZE,
    random_face_field,
    transport_config,
)

CRITERION = 1.0e-4
# ADR-011 F: the residual is rounding, about the number of steps times the
# machine epsilon; prompt 32 stops at 1e-10 as a flux-assembly defect.
ROUNDING_EXPECTATION = 1.0e-10
STEPS = 500
INLET_CONCENTRATION = 2.0e5
SOURCE_RATE = 5.0e4


def _run(
    config: SimConfig, mesh: Mesh, faces: FaceVelocities, source_cell: tuple[int, int]
) -> tuple[TransportSolver, MassBudget, float, float]:
    """500 steps at the stable step from a seeded field with a source; returns the budget."""
    physics = ParticlePhysics(config)
    boundary = ConcentrationBoundary(mesh, config, physics, BoundaryRegistry(config))
    solver = TransportSolver(mesh, config, physics, boundary)
    rate = np.zeros((config.ny, config.nx))
    rate[source_cell] = SOURCE_RATE
    rng = np.random.default_rng(7)
    c = 1.0e5 * rng.uniform(0.5, 1.5, size=(config.ny, config.nx))
    dt = solver.stable_dt(faces, 0)
    start = perf_counter()
    for _ in range(STEPS):
        c = solver.solve_timestep(c, faces, 0, dt, sources=rate)
    return solver, solver.budget[0], dt, perf_counter() - start


def _report(label: str, budget: MassBudget, dt: float, seconds: float) -> None:
    print(
        f"{label}: {STEPS} steps of {dt:.3e} s in {seconds:.1f} s; relative residual "
        f"{budget.relative():.3e} (criterion < {CRITERION}, expected at rounding)"
    )
    print(
        f"  initial {budget.initial:.6e}, inflow {budget.inflow:.6e}, outflow "
        f"{budget.outflow:.6e}, source {budget.source:.6e}, deposited "
        + ", ".join(f"{k} {v:.4e}" for k, v in budget.deposited.items())
        + f", in domain {budget.current:.6e}"
    )


def _dropped_face_control(
    budget: MassBudget, faces: FaceVelocities, mesh: Mesh, dt: float, row: int
) -> float:
    """The relative residual had one inlet face been left out of the booking."""
    carried = INLET_CONCENTRATION
    through_face = faces.u[row, 0] * mesh.dy_cell[row] * carried * dt * STEPS
    assert through_face > 0.0
    supplied = budget.initial + budget.inflow - through_face + budget.source
    return (budget.residual() - through_face) / supplied


@pytest.mark.validation
def test_mass_conservation_on_a_random_face_field_val007() -> None:
    """VAL-007 (i): the budget closes to rounding on an arbitrary face field."""
    config = transport_config(
        1.6,
        0.9,
        64,
        36,
        boundaries={
            "inlet": {
                "type": "velocity_inlet",
                "location": "left",
                "y_start": 0.2,
                "y_end": 0.7,
                "velocity": 0.3,
                "concentration": [INLET_CONCENTRATION],
            },
            "outlet": {
                "type": "pressure_outlet",
                "location": "right",
                "y_start": 0.1,
                "y_end": 0.8,
            },
        },
        diffusion_tol=1.0e-14,
    )
    mesh = Mesh(config)
    faces = random_face_field(mesh, seed=32, scale=0.3)
    _, budget, dt, seconds = _run(config, mesh, faces, (18, 32))
    _report("VAL-007 (i), random field 64x36", budget, dt, seconds)
    assert abs(budget.relative()) < CRITERION
    assert abs(budget.relative()) < ROUNDING_EXPECTATION
    assert budget.inflow > 0.0 and budget.outflow > 0.0 and budget.source > 0.0
    assert all(budget.deposited[k] > 0.0 for k in ("floor", "ceiling", "wall"))
    assert budget.deposited["obstacle"] == 0.0

    # Planted control: an inlet row whose random face velocity points inward.
    inward = np.flatnonzero(faces.u[:, 0] > 0.0)
    row = int(inward[len(inward) // 2])
    control = _dropped_face_control(budget, faces, mesh, dt, row)
    print(f"  control, inlet face at row {row} dropped from the booking: {control:.3e}")
    assert abs(control) > CRITERION


@pytest.mark.validation
def test_mass_conservation_on_the_val001_faces_val007() -> None:
    """VAL-007 (ii): the same budget on VAL-001 40x20 converged under error_estimate."""
    raw = yaml.safe_load(case_path("poiseuille").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = 40, 20
    raw["particles"]["sizes"] = [PARTICLE_SIZE]
    raw["particles"]["hepa_reference"] = {
        "diameters": [PARTICLE_SIZE],
        "efficiencies": [0.99999],
    }
    raw["boundaries"]["inlet"]["concentration"] = [INLET_CONCENTRATION]
    raw["transport"] = {
        "cfl_number": CFL_NUMBER,
        "advection_scheme": "umist",
        "max_diffusion_iter": 100,
        "diffusion_tol": 1.0e-14,
    }
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    staggered = StaggeredBoundary(mesh, config)
    velocity = StaggeredSolver(mesh, config, staggered)
    start = perf_counter()
    velocity.solve_steady()
    solve_seconds = perf_counter() - start
    assert velocity.stop_reason == "error_estimate_and_continuity"
    faces = velocity.face_velocities

    _, budget, dt, seconds = _run(config, mesh, faces, (10, 20))
    _report(
        f"VAL-007 (ii), VAL-001 40x20 faces (solved in {solve_seconds:.1f} s)",
        budget,
        dt,
        seconds,
    )
    assert abs(budget.relative()) < CRITERION
    assert abs(budget.relative()) < ROUNDING_EXPECTATION
    assert budget.inflow > 0.0 and budget.outflow > 0.0 and budget.source > 0.0
    assert budget.deposited["floor"] > 0.0 and budget.deposited["ceiling"] > 0.0
    control = _dropped_face_control(budget, faces, mesh, dt, 10)
    print(f"  control, inlet face at row 10 dropped from the booking: {control:.3e}")
    assert abs(control) > CRITERION
