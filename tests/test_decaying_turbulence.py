"""VAL-015: decaying turbulence in a closed box at rest (ECR-002 criterion 2, ADR-012 G (ii)).

A closed 2.4 m by 1.5 m box, rho 1.2, every face at rest, uniform k0 and eps0
off round values. With no velocity there is no advection and no production,
and a uniform field does not diffuse, so the k-epsilon equations reduce to

    dk/dt = -eps,   deps/dt = -C_2 eps^2 / k,

whose solution is ``k = k0 (1 + (C_2 - 1) eps0 t / k0)^(-1 / (C_2 - 1))``
and ``eps = eps0 (k / k0)^C_2``. The step puts both decays in the implicit
diagonal with eps / k from the previous iterate, backward Euler in a
linearised form, so the error at a fixed end time falls at first order in
the true-time step. The criterion: observed order between 0.9 and 1.1 over
three steps in a ratio of two, for k and for eps, both variants; and the
field uniform to rounding, its relative spread below 2e-15 per step taken. RNG's R is zero at rest (eta = 0), so RNG decays
with its own C_2, 1.68.

The end time is two initial turnover times, 2 k0 / eps0, and the steps are
1/40, 1/80 and 1/160 of it: at 1/20 the coarse pair's order for RNG's k is
0.887, outside the band, the dt^2 term still visible (prompt 35's smoke run,
results/builder35/).
"""

import numpy as np
import pytest

from src.config import SimConfig
from src.mesh import Mesh
from src.staggered import FaceVelocities, u_shape, v_shape
from src.turbulence import KEpsilonModel, TurbulenceConditions
from validation.transport_cases import AIR, PARTICLES, SOLVER_BLOCK

K0 = 0.0137
EPS0 = 0.00291
END_TIME = 2.0 * K0 / EPS0
STEPS = (40, 80, 160)
ROUNDING_PER_STEP = 2.0e-15


def _box(variant: str) -> SimConfig:
    """The closed box, every edge a wall, a mesh stretched on one axis."""
    raw = {
        "domain": {"width": 2.4, "height": 1.5, "nx": 12, "ny": 8},
        "mesh": {"x": {"stretch_ratio": 1.2}},
        "fluid": AIR,
        "particles": PARTICLES,
        "solver": SOLVER_BLOCK,
        "turbulence": {
            "model": "k_epsilon",
            "variant": variant,
            "wall_treatment": "scalable_wall_functions",
            "cfl_number": 0.5,
            "alpha_turbulence": 0.7,
            "max_iter": 2000,
            "tol": 1.0e-15,
        },
        "boundaries": {},
        "obstacles": [],
        "sensors": [{"name": "centre", "x": 1.2, "y": 0.75}],
        "thresholds": {"5e-06": 100.0},
    }
    return SimConfig.from_dict(raw)


def _exact(c_2: float, t: float) -> tuple[float, float]:
    k = K0 * (1.0 + (c_2 - 1.0) * EPS0 * t / K0) ** (-1.0 / (c_2 - 1.0))
    return k, EPS0 * (k / K0) ** c_2


@pytest.mark.validation
@pytest.mark.parametrize("variant", ["standard", "rng"])
def test_decaying_turbulence_val015(variant: str) -> None:
    """VAL-015: first order in dt, observed order in [0.9, 1.1] for k and eps, uniform."""
    config = _box(variant)
    assert config.rho == 1.2
    mesh = Mesh(config)
    model = KEpsilonModel(mesh, config)
    faces = FaceVelocities.copy_of(np.zeros(u_shape(mesh)), np.zeros(v_shape(mesh)))
    conditions = TurbulenceConditions.uniform(mesh, K0, EPS0)
    k_exact, eps_exact = _exact(model.constants.c_2, END_TIME)

    errors = []
    for n in STEPS:
        state = model.initial(K0, EPS0)
        for _ in range(n):
            state = model.step(state, faces, conditions, END_TIME / n)
        assert model.solves_converged
        k_spread = float(np.ptp(state.k) / state.k.mean())
        eps_spread = float(np.ptp(state.eps) / state.eps.mean())
        # Each step's solve stops at 1e-15 of its right-hand side and rounds by
        # a few ulps, so the spread is a sum over the steps: about nine ulps a
        # step bounds it.
        assert k_spread < n * ROUNDING_PER_STEP and eps_spread < n * ROUNDING_PER_STEP
        errors.append(
            (
                abs(float(state.k.mean()) / k_exact - 1.0),
                abs(float(state.eps.mean()) / eps_exact - 1.0),
            )
        )
        print(
            f"VAL-015 {variant}, dt = T/{n}: k error {errors[-1][0]:.4e}, eps error "
            f"{errors[-1][1]:.4e}; spread k {k_spread:.1e}, eps {eps_spread:.1e}"
        )

    for q, name in ((0, "k"), (1, "eps")):
        orders = [
            float(np.log2(errors[i][q] / errors[i + 1][q]))
            for i in range(len(STEPS) - 1)
        ]
        print(
            f"VAL-015 {variant}: observed order of {name} {orders[0]:.4f}, {orders[1]:.4f}"
        )
        assert all(0.9 <= order <= 1.1 for order in orders)


@pytest.mark.validation
def test_the_planted_explicit_decay_breaks_val015_at_a_large_step() -> None:
    """The trap's control: with the decay explicit, ``k - dt eps``, one step of
    dt = 2 k0 / eps0 takes k below zero; the implicit decay keeps it positive."""
    config = _box("standard")
    mesh = Mesh(config)
    model = KEpsilonModel(mesh, config)
    faces = FaceVelocities.copy_of(np.zeros(u_shape(mesh)), np.zeros(v_shape(mesh)))
    conditions = TurbulenceConditions.uniform(mesh, K0, EPS0)
    state = model.initial(K0, EPS0)
    implicit = model.step(state, faces, conditions, END_TIME)
    assert implicit.k.min() > 0.0
    explicit_k = state.k - END_TIME * state.eps
    print(
        f"VAL-015 control: one step of {END_TIME:.3f} s, implicit k "
        f"{implicit.k.mean():.4e}, explicit k {explicit_k.mean():.4e}"
    )
    assert explicit_k.max() < 0.0
