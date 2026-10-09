"""VAL-016, plane Couette flow (ECR-002 step 6, item 6; ADR-012 G (ii)).

The criterion is split (Alex, 2026-10-09; ADR-012 G's note of that date):

(a) The implementation. In the developed section, the 2D coupled solve
    equals the one-dimensional solve on the same grid (tests/couette_reference.py
    at REFINE 1, the 2D stencil's own x-invariant limit, which imports nothing
    from src/) in u / U_w and in k / u_tau^2 to within 1e-4, each k normalised
    by its own solve's u_tau.
(b) The model. The refined one-dimensional solve (REFINE 41, the same wall
    cells), the model's answer, keeps the core k (0.2 H to 0.8 H) within 1% of
    u_tau^2 / sqrt(C_mu), u_tau from its mid-gap stress, each variant against
    its own C_mu.
(c) Reported, not scored: the 2D core-k excess and the first node's y+ on
    every grid, in docs/reports/probe45/ with the full matrix (both variants,
    12, 24 and 48 rows, the plane channel against Dean) and its runtimes.

Here, to keep the suite's cost under a minute (46 s on this machine), (a)
runs one grid and one variant: the standard model on 12 rows, a 120 m channel (400 gaps; the
profile stops changing at about 200) in 1 m cells along x. Positivity is
asserted by the solve itself, every outer iteration (REQ-S15). A planted
defect in the coupled path, the wall-cell production scaled by 1.01, fails
(a) (docs/reports/probe45/plant45.py).
"""

import math

import numpy as np
import pytest

from src import turbulence
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import Mesh
from src.solver_staggered import StaggeredSolver
from tests import couette_reference as ref

IMPLEMENTATION_TOL = 1.0e-4
MODEL_TOL = 0.01
LENGTH = 120.0
DX = 1.0
CFL = 0.25
ROWS = 12
REFINE = 41
STATION = 0.8


def _core(yc: list[float], k: list[float], u_tau: float, c_mu: float) -> np.ndarray:
    """k over u_tau^2 / sqrt(C_mu) at the centres from 0.2 to 0.8 of the gap."""
    y, kk = np.array(yc), np.array(k)
    core = (y >= 0.2 * ref.GAP) & (y <= 0.8 * ref.GAP)
    return kk[core] / (u_tau**2 / math.sqrt(c_mu))


@pytest.fixture(scope="module")
def couette_2d() -> dict:
    """The standard model's Couette solve on 12 rows, its developed section."""
    config = SimConfig.from_dict(
        ref.case_raw("couette", "standard", ROWS, LENGTH, DX, CFL)
    )
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    solver.solve_steady()
    faces, state = solver.face_velocities, solver.turbulence_state
    assert faces is not None and state is not None
    i = int(STATION * mesh.xc.size)
    u = np.array(faces.u[:, i])
    k = 0.5 * (state.k[:, i - 1] + state.k[:, i])
    nu_t = 0.5 * (state.nu_t[:, i - 1] + state.nu_t[:, i])
    j = ROWS // 2
    a, b = ref.NU + nu_t[j - 1], ref.NU + nu_t[j]
    # The mid-gap stress through the face rule the momentum uses there.
    stress = 2.0 * a * b / (a + b) * (u[j] - u[j - 1]) / (mesh.yc[j] - mesh.yc[j - 1])
    return {
        "solver": solver,
        "u": u,
        "k": k,
        "u_tau": math.sqrt(stress),
        "k_min": float(state.k[state.k > 0.0].min()),
    }


@pytest.mark.unit
def test_the_references_copied_constants_are_the_modules() -> None:
    """The reference imports nothing from src/, so its copies are checked here."""
    assert (ref.KAPPA, ref.E_WALL) == (turbulence.KAPPA, turbulence.E_WALL)
    assert pytest.approx(turbulence.Y_STAR_FLOOR, rel=1e-15) == ref.Y_FLOOR
    for name, constants in ref.VARIANTS.items():
        built = turbulence.VARIANTS[name]
        for key, value in constants.items():
            assert getattr(built, key) == value, (name, key)


@pytest.mark.validation
def test_couette_converges_under_rule_version_4_val016(couette_2d: dict) -> None:
    """VAL-016: the coupled solve stops by its rule, condition (e) included."""
    solver = couette_2d["solver"]
    assert solver.stop_reason == "error_estimate_and_continuity"
    assert solver.rule_version == 4
    assert couette_2d["k_min"] > 0.0


@pytest.mark.validation
def test_couette_equals_its_same_grid_reference_val016(couette_2d: dict) -> None:
    """VAL-016 (a): the developed 2D profile is the stencil's 1D answer to 1e-4."""
    one_d = ref.reference("standard", ROWS, 1)
    du = np.abs(couette_2d["u"] - np.array(one_d["u"])).max() / ref.U_W
    dk = np.abs(
        couette_2d["k"] / couette_2d["u_tau"] ** 2
        - np.array(one_d["k"]) / one_d["u_tau_mid"] ** 2
    ).max()
    print(f"VAL-016 (a): max |du| / U_w {du:.2e}, max |d(k / u_tau^2)| {dk:.2e}")
    assert du < IMPLEMENTATION_TOL
    assert dk < IMPLEMENTATION_TOL
    assert couette_2d["u_tau"] == pytest.approx(one_d["u_tau_mid"], rel=1e-4)


@pytest.mark.validation
@pytest.mark.parametrize("variant", ["standard", "rng"])
def test_the_models_core_k_is_u_tau2_over_sqrt_c_mu_val016(variant: str) -> None:
    """VAL-016 (b): the refined 1D solve's core k within 1%, against its own C_mu.

    Wherever k is uniform and the stress constant, production equals
    dissipation and k = u_tau^2 / sqrt(C_mu) (ADR-012 G (ii)); the molecular
    viscosity's share keeps it a little below.
    """
    one_d = ref.reference(variant, ROWS, REFINE)
    ratio = _core(
        one_d["yc"], one_d["k"], one_d["u_tau_mid"], ref.VARIANTS[variant]["c_mu"]
    )
    print(f"VAL-016 (b) {variant}: core k ratio {ratio.min():.5f} .. {ratio.max():.5f}")
    assert np.abs(ratio - 1.0).max() < MODEL_TOL
