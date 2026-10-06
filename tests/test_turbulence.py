"""Tests for the k-epsilon model on a prescribed face field (src/turbulence.py; ADR-012 A, C).

ECR-002 step 1. VAL-015 is in tests/test_decaying_turbulence.py. Here:

- the constants against their sourced values (results/builder35/constants.md);
- the checks on states, conditions and steps, and the two kinds of step;
- where the gradients live, at walls and obstacle faces;
- RNG's R split by sign;
- the boundary rules: the inflow value where the flux enters, nothing across
  a wall or an obstacle face, eps held where the conditions say;
- the face diffusivity against a dense solve;
- positivity (REQ-S15, ECR-002 criterion 3) on the Smith-Hutton field and a
  random divergence-free field with production on, with the planted
  explicit decay as its control;
- constancy on the VAL-001 faces, production and decay switched off.

Each planted defect of prompt 35's traps fails a test here; the mutation logs
are results/builder35/mutation35.md and mutation35b.md. Meshes are non-square
and, where it matters, stretched, so an index that works only on a square
uniform grid shows. Every rule with an x and a y half is tested on a field
that moves in y, or in both directions with different magnitudes: test 32b
found the transport solver's tests pinned one direction only, and review 35
and test 35 found the same here (prompt 35b).
"""

import dataclasses
import math
from collections.abc import Callable

import numpy as np
import pytest
import yaml

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.pressure import PressureCorrector
from src.solver_staggered import StaggeredSolver
from src.staggered import FaceVelocities, u_shape, v_shape
from src.turbulence import (
    VARIANTS,
    KEpsilonModel,
    PositivityError,
    StepTerms,
    TurbulenceConditions,
    TurbulenceState,
)
from validation.cases import case_path
from validation.transport_cases import (
    AIR,
    PARTICLES,
    SOLVER_BLOCK,
    smith_hutton_face_field,
)

VARIANT_NAMES = ["standard", "rng"]
SMITH_HUTTON_SPEED = 0.45


def _config(
    width: float,
    height: float,
    nx: int,
    ny: int,
    variant: str = "standard",
    *,
    cfl: float = 0.5,
    tol: float = 1.0e-12,
    max_iter: int = 2000,
    obstacles: list[dict] | None = None,
    mesh: dict | None = None,
    viscosity: float = AIR["viscosity"],
) -> SimConfig:
    """A validated configuration with a turbulence section, air at rho 1.2."""
    raw = {
        "domain": {"width": width, "height": height, "nx": nx, "ny": ny},
        "mesh": mesh or {},
        "fluid": {**AIR, "viscosity": viscosity},
        "particles": PARTICLES,
        "solver": SOLVER_BLOCK,
        "turbulence": {
            "model": "k_epsilon",
            "variant": variant,
            "wall_treatment": "scalable_wall_functions",
            "cfl_number": cfl,
            "alpha_turbulence": 0.7,
            "max_iter": max_iter,
            "tol": tol,
        },
        "boundaries": {},
        "obstacles": obstacles or [],
        "sensors": [{"name": "centre", "x": width / 2.0, "y": height / 2.0}],
        "thresholds": {"5e-06": 100.0},
    }
    return SimConfig.from_dict(raw)


def _model(config: SimConfig) -> tuple[Mesh, KEpsilonModel]:
    mesh = Mesh(config)
    return mesh, KEpsilonModel(mesh, config)


def _at_rest(mesh: Mesh) -> FaceVelocities:
    return FaceVelocities.copy_of(np.zeros(u_shape(mesh)), np.zeros(v_shape(mesh)))


def _shear(mesh: Mesh, speed: float) -> tuple[FaceVelocities, TurbulenceConditions]:
    """Plane Couette flow u = U y / H: the top edge moves at U, the bottom rests."""
    height = mesh.y[-1]
    u = np.repeat((speed * mesh.yc / height)[:, None], u_shape(mesh)[1], axis=1)
    conditions = dataclasses.replace(
        TurbulenceConditions.uniform(mesh, 1.0e-3, 1.0e-4),
        tangential_top=np.full(mesh.x.shape, speed),
    )
    return FaceVelocities.copy_of(u, np.zeros(v_shape(mesh))), conditions


def _streamfunction_field(mesh: Mesh, seed: int, scale: float) -> FaceVelocities:
    """A random divergence-free field from a streamfunction at the corners.

    The streamfunction is zero on the domain edges and on every corner of a
    SOLID cell, so no face of a wall or an obstacle carries a normal velocity
    and every cell's net flux is zero to rounding.
    """
    rng = np.random.default_rng(seed)
    ny, nx = mesh.cell_type.shape
    psi = rng.uniform(-scale, scale, size=(ny + 1, nx + 1))
    psi[[0, -1], :] = 0.0
    psi[:, [0, -1]] = 0.0
    solid = mesh.cell_type == SOLID
    for j, i in np.argwhere(solid):
        psi[j : j + 2, i : i + 2] = 0.0
    u = (psi[1:, :] - psi[:-1, :]) / mesh.dy_cell[:, None]
    v = -(psi[:, 1:] - psi[:, :-1]) / mesh.dx_cell[None, :]
    return FaceVelocities.copy_of(u, v)


def _smith_hutton_conditions(mesh: Mesh, k: float, eps: float) -> TurbulenceConditions:
    """The field's own tangential velocity on each edge, so no edge is a false wall."""
    x_prime = mesh.x - 1.0
    y = mesh.y
    speed = SMITH_HUTTON_SPEED
    return dataclasses.replace(
        TurbulenceConditions.uniform(mesh, k, eps),
        tangential_bottom=np.zeros_like(x_prime),
        tangential_top=2.0 * speed * (1.0 - x_prime**2),
        tangential_left=2.0 * speed * (1.0 - y**2),
        tangential_right=-2.0 * speed * (1.0 - y**2),
    )


def _patchy_start(
    model: KEpsilonModel, shape: tuple[int, int], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """A non-uniform start with a patch where eps / k is 100 per second.

    There ``dt eps / k`` exceeds one at a Courant number of 1/2 on these
    fields, so an explicit decay takes k below zero on the first step.
    """
    rng = np.random.default_rng(seed)
    k = rng.uniform(0.5, 2.0, size=shape) * 1.0e-3
    eps = rng.uniform(0.5, 2.0, size=shape) * 1.0e-4
    ny, nx = shape
    eps[ny // 3 : ny // 2, nx // 4 : nx // 2] = 0.1
    k[ny // 3 : ny // 2, nx // 4 : nx // 2] = 1.0e-3
    return k, eps


def _explicit_decay(
    model: KEpsilonModel,
    monkeypatch: pytest.MonkeyPatch,
    *,
    for_k: bool = True,
    for_eps: bool = True,
) -> None:
    """Plant the explicit form of the decay in step 2: ``- dt eps`` for k and
    ``- dt C_2 eps^2 / k`` for eps, moved out of the implicit diagonal."""
    original = model._terms

    def explicit(
        state: TurbulenceState,
        faces: FaceVelocities,
        conditions: TurbulenceConditions,
    ) -> StepTerms:
        t = original(state, faces, conditions)
        if for_k:
            t = dataclasses.replace(
                t,
                growth_k=t.growth_k - t.decay_k * state.k,
                decay_k=np.zeros_like(t.decay_k),
            )
        if for_eps:
            t = dataclasses.replace(
                t,
                growth_eps=t.growth_eps - t.decay_eps * state.eps,
                decay_eps=np.zeros_like(t.decay_eps),
            )
        return t

    monkeypatch.setattr(model, "_terms", explicit)


def _no_sources(model: KEpsilonModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch production and decay off: every growth and decay term zero."""

    def zero(
        state: TurbulenceState,
        faces: FaceVelocities,
        conditions: TurbulenceConditions,
    ) -> StepTerms:
        z = np.zeros_like(state.k)
        return StepTerms(
            production=z, rng_r=z, growth_k=z, growth_eps=z, decay_k=z, decay_eps=z
        )

    monkeypatch.setattr(model, "_terms", zero)


LINEAR = (0.7, 0.3, 0.6, 0.2)


def _linear_field(
    mesh: Mesh, a: float, b: float, c: float, d: float
) -> tuple[FaceVelocities, TurbulenceConditions]:
    """u = a x + b y on the u faces, v = c x + d y on the v faces, each edge
    carrying the field's own tangential value, so the field moves in both
    directions with gradients of four different magnitudes."""
    u = a * mesh.x[None, :] + b * mesh.yc[:, None]
    v = c * mesh.xc[None, :] + d * mesh.y[:, None]
    conditions = dataclasses.replace(
        TurbulenceConditions.uniform(mesh, 1.0e-3, 1.0e-4),
        tangential_bottom=a * mesh.x + b * mesh.y[0],
        tangential_top=a * mesh.x + b * mesh.y[-1],
        tangential_left=c * mesh.x[0] + d * mesh.y,
        tangential_right=c * mesh.x[-1] + d * mesh.y,
    )
    return FaceVelocities.copy_of(u, v), conditions


def _rate_by_loop(mesh: Mesh, faces: FaceVelocities) -> np.ndarray:
    """``max(|u_w|, |u_e|) / dx + max(|v_s|, |v_n|) / dy`` cell by cell, both halves."""
    ny, nx = mesh.cell_type.shape
    rate = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            horizontal = (
                max(abs(faces.u[j, i]), abs(faces.u[j, i + 1])) / mesh.dx_cell[i]
            )
            vertical = max(abs(faces.v[j, i]), abs(faces.v[j + 1, i])) / mesh.dy_cell[j]
            rate[j, i] = horizontal + vertical
    return rate


# ---------------------------------------------------------------------------
# The constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstants:
    def test_the_tables_hold_the_sourced_values(self) -> None:
        """Launder and Spalding (1974) Table 2.1; Yakhot et al. (1992) with C_mu
        0.0845 (Alex, 2026-10-05; results/builder35/constants.md)."""
        std, rng = VARIANTS["standard"], VARIANTS["rng"]
        assert (std.c_mu, std.c_1, std.c_2, std.sigma_k, std.sigma_e) == (
            0.09,
            1.44,
            1.92,
            1.0,
            1.3,
        )
        assert (std.eta_0, std.beta) == (None, None)
        assert (rng.c_mu, rng.c_1, rng.c_2, rng.sigma_k, rng.sigma_e) == (
            0.0845,
            1.42,
            1.68,
            0.7194,
            0.7194,
        )
        assert (rng.eta_0, rng.beta) == (4.38, 0.012)

    def test_rng_eta_0_is_the_fixed_point_its_constants_give(self) -> None:
        """eta_0 = ((C_2 - 1) / (C_mu (C_1 - 1)))^(1/2): 4.377 at C_mu 0.0845,
        which the printed 4.38 rounds; 0.085 would give 4.364."""
        rng = VARIANTS["rng"]

        def fixed_point(c_mu: float) -> float:
            return math.sqrt((rng.c_2 - 1.0) / (c_mu * (rng.c_1 - 1.0)))

        assert abs(fixed_point(rng.c_mu) - rng.eta_0) < 0.005
        assert abs(fixed_point(0.085) - rng.eta_0) > 0.01

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_the_configured_variant_is_the_one_used(self, variant: str) -> None:
        _, model = _model(_config(1.2, 0.8, 6, 4, variant))
        assert model.constants == VARIANTS[variant]


# ---------------------------------------------------------------------------
# States, conditions and steps: the checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChecks:
    def test_a_configuration_without_the_section_is_refused(self) -> None:
        config = _config(1.2, 0.8, 6, 4)
        config.turbulence = None
        with pytest.raises(ValueError, match="needs a turbulence section"):
            KEpsilonModel(Mesh(config), config)

    def test_a_state_carries_the_kinematic_eddy_viscosity(self) -> None:
        obstacle = [
            {"name": "b", "x_start": 0.4, "x_end": 0.6, "y_start": 0.0, "y_end": 0.2}
        ]
        mesh, model = _model(_config(1.2, 0.8, 6, 4, "rng", obstacles=obstacle))
        solid = mesh.cell_type == SOLID
        assert solid.any()
        rng = np.random.default_rng(2)
        k = rng.uniform(0.01, 0.1, solid.shape)
        eps = rng.uniform(0.01, 0.1, solid.shape)
        state = model.state(k, eps)
        live = ~solid
        assert np.array_equal(state.nu_t[live], 0.0845 * k[live] ** 2 / eps[live])
        for array in (state.k, state.eps, state.nu_t):
            assert not array.flags.writeable
            assert np.all(array[solid] == 0.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0e-3, float("nan"), float("inf")])
    def test_a_state_refuses_a_value_that_is_not_positive_and_finite(
        self, bad: float
    ) -> None:
        mesh, model = _model(_config(1.2, 0.8, 6, 4))
        k = np.full(mesh.cell_type.shape, 1.0e-3)
        k[2, 3] = bad
        with pytest.raises(ValueError, match="k must be positive and finite"):
            model.state(k, np.full(mesh.cell_type.shape, 1.0e-4))
        with pytest.raises(ValueError, match="expected eps of shape"):
            model.state(np.full(mesh.cell_type.shape, 1.0e-3), np.ones((3, 3)))

    def test_initial_is_uniform_and_checked(self) -> None:
        _, model = _model(_config(1.2, 0.8, 6, 4))
        state = model.initial(0.0137, 0.00291)
        assert np.all(state.k == 0.0137) and np.all(state.eps == 0.00291)
        with pytest.raises(TypeError, match="k must be a number"):
            model.initial(True, 0.00291)
        with pytest.raises(ValueError, match="eps must be positive"):
            model.initial(0.0137, 0.0)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("inflow_k_u", np.zeros((4, 6)), "inflow_k_u must be float64 of shape"),
            ("tangential_top", np.zeros(6), "tangential_top must be float64 of shape"),
            ("eps_held", np.zeros((4, 6)), "eps_held must be bool"),
            ("production", np.full((4, 6), np.nan), "production must be finite"),
        ],
    )
    def test_conditions_of_the_wrong_shape_dtype_or_value_are_refused(
        self, field: str, value: np.ndarray, match: str
    ) -> None:
        mesh, model = _model(_config(1.2, 0.8, 6, 4))
        conditions = dataclasses.replace(
            TurbulenceConditions.uniform(mesh, 1.0e-3, 1.0e-4), **{field: value}
        )
        with pytest.raises(ValueError, match=match):
            model.step(model.initial(1.0e-3, 1.0e-4), _at_rest(mesh), conditions, 1.0)

    def test_conditions_out_of_range_are_refused(self) -> None:
        obstacle = [
            {"name": "b", "x_start": 0.4, "x_end": 0.6, "y_start": 0.0, "y_end": 0.2}
        ]
        mesh, model = _model(_config(1.2, 0.8, 6, 4, obstacles=obstacle))
        base = TurbulenceConditions.uniform(mesh, 1.0e-3, 1.0e-4)
        state, faces = model.initial(1.0e-3, 1.0e-4), _at_rest(mesh)
        inflow = base.inflow_eps_v.copy()
        inflow[0, 1] = 0.0
        held = np.zeros(mesh.cell_type.shape, dtype=bool)
        held[1, 1] = True
        negative = np.zeros(mesh.cell_type.shape)
        negative[1, 1] = -1.0
        solid_cell = np.argwhere(mesh.cell_type == SOLID)[0]
        held_solid = np.zeros(mesh.cell_type.shape, dtype=bool)
        held_solid[tuple(solid_cell)] = True
        cases = [
            ({"inflow_eps_v": inflow}, "inflow k and eps must be positive"),
            ({"eps_held": held}, "eps_wall must be positive where eps is held"),
            (
                {"production_given": held, "production": negative},
                "production must be non-negative",
            ),
            ({"eps_held": held_solid}, "eps_held names a SOLID cell"),
        ]
        for changes, match in cases:
            with pytest.raises(ValueError, match=match):
                model.step(state, faces, dataclasses.replace(base, **changes), 1.0)

    @pytest.mark.parametrize(
        ("field", "defect", "match"),
        [
            ("k", "shape", "state.k must have shape"),
            ("nu_t", "shape", "state.nu_t must have shape"),
            ("k", -1.0e-3, "state.k must be positive and finite"),
            ("eps", 0.0, "state.eps must be positive and finite"),
            ("eps", float("nan"), "state.eps must be positive and finite"),
            ("nu_t", float("nan"), "state.nu_t must be non-negative and finite"),
            ("nu_t", -1.0e-3, "state.nu_t must be non-negative and finite"),
            ("nu_t", float("inf"), "state.nu_t must be non-negative and finite"),
            ("k", "solid", "state.k must be zero in SOLID cells"),
            ("nu_t", "solid", "state.nu_t must be zero in SOLID cells"),
        ],
    )
    def test_a_state_is_checked_before_any_arithmetic(
        self,
        monkeypatch: pytest.MonkeyPatch,
        field: str,
        defect: object,
        match: str,
    ) -> None:
        """A TurbulenceState built directly, as step 6 will build one with an
        under-relaxed nu_t, is refused by ``step`` before any of the step's
        arithmetic runs: the terms are replaced by a tripwire, and the refusal
        comes first. A negative nu_t would give a negative face conductance."""
        obstacle = [
            {"name": "b", "x_start": 0.4, "x_end": 0.6, "y_start": 0.0, "y_end": 0.2}
        ]
        mesh, model = _model(_config(1.2, 0.8, 6, 4, obstacles=obstacle))
        solid = mesh.cell_type == SOLID
        valid = model.initial(1.0e-3, 1.0e-4)
        arrays = {name: getattr(valid, name).copy() for name in ("k", "eps", "nu_t")}
        if defect == "shape":
            arrays[field] = arrays[field][:, :-1]
        elif defect == "solid":
            arrays[field][tuple(np.argwhere(solid)[0])] = 1.0e-3
        else:
            arrays[field][1, 1] = defect
        broken = TurbulenceState(**arrays)

        def tripwire(*args: object) -> StepTerms:
            raise AssertionError("the step reached its arithmetic")

        monkeypatch.setattr(model, "_terms", tripwire)
        faces, conditions = _shear(mesh, 0.3)
        with pytest.raises(ValueError, match=match):
            model.step(broken, faces, conditions, 0.01)

    def test_faces_are_checked_before_any_arithmetic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A face pair of another mesh's shape, or holding a NaN or an
        infinity, is refused before the step's arithmetic, which a tripwire
        stands in for; FaceVelocities itself checks neither."""
        mesh, model = _model(_config(1.2, 0.8, 6, 4))
        state = model.initial(1.0e-3, 1.0e-4)

        def tripwire(*args: object) -> StepTerms:
            raise AssertionError("the step reached its arithmetic")

        monkeypatch.setattr(model, "_terms", tripwire)
        conditions = TurbulenceConditions.uniform(mesh, 1.0e-3, 1.0e-4)
        other = Mesh(_config(1.2, 0.8, 7, 4))
        with pytest.raises(ValueError, match="faces must have shapes"):
            model.step(state, _at_rest(other), conditions, 0.01)
        u = np.zeros(u_shape(mesh))
        u[1, 2] = np.nan
        v = np.zeros(v_shape(mesh))
        with pytest.raises(ValueError, match="faces must be finite"):
            model.step(state, FaceVelocities.copy_of(u, v), conditions, 0.01)
        v[2, 3] = np.inf
        with pytest.raises(ValueError, match="faces must be finite"):
            model.step(
                state, FaceVelocities.copy_of(np.zeros(u_shape(mesh)), v), conditions
            )

    @pytest.mark.parametrize(
        ("dt", "error", "match"),
        [
            (True, TypeError, "dt must be a number"),
            ("0.1", TypeError, "dt must be a number"),
            (0.0, ValueError, "dt must be positive"),
            (-0.5, ValueError, "dt must be positive"),
            (float("nan"), ValueError, "dt must be finite"),
            (float("inf"), ValueError, "dt must be finite"),
            (10.0, ValueError, "exceeds the stable step"),
        ],
    )
    def test_a_true_time_step_is_checked(
        self, dt: object, error: type[Exception], match: str
    ) -> None:
        mesh, model = _model(_config(1.2, 0.8, 6, 4))
        faces, conditions = _shear(mesh, 0.3)
        with pytest.raises(error, match=match):
            model.step(model.initial(1.0e-3, 1.0e-4), faces, conditions, dt)


# ---------------------------------------------------------------------------
# The two kinds of step
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTwoKindsOfStep:
    def test_the_pseudo_time_step_has_no_value_on_a_field_at_rest(self) -> None:
        """dt None needs a moving cell; at rest the step raises and names the remedy."""
        mesh, model = _model(_config(2.4, 1.5, 12, 8))
        conditions = TurbulenceConditions.uniform(mesh, 1.0e-3, 1.0e-4)
        with pytest.raises(ValueError, match=r"no cell moves.*pass a true-time dt"):
            model.step(model.initial(1.0e-3, 1.0e-4), _at_rest(mesh), conditions)

    def test_the_pseudo_time_step_is_per_cell_and_capped_where_a_cell_rests(
        self,
    ) -> None:
        """Couette flow with its bottom row at rest: each moving cell takes
        cfl over its own rate, and the row at rest takes the largest of them."""
        mesh, model = _model(
            _config(1.2, 0.8, 6, 5, mesh={"y": {"stretch_ratio": 1.2}})
        )
        faces, _ = _shear(mesh, 0.3)
        u = faces.u.copy()
        u[0, :] = 0.0
        faces = FaceVelocities.copy_of(u, faces.v)
        dt = model.pseudo_time_step(faces)
        moving_rate = np.abs(u[1:, 0]) / mesh.dx_cell[0]
        assert np.allclose(dt[1:, 0], 0.5 / moving_rate, rtol=1e-14)
        assert len(np.unique(dt[1:, 0])) == mesh.cell_type.shape[0] - 1
        assert np.all(dt[0, :] == dt[1:, :].max())

    def test_the_cap_keeps_the_growth_of_a_cell_at_rest_finite(self) -> None:
        """The row at rest has production from its top corners' shear; with
        the cap its growth is finite and k there rises, step after step."""
        mesh, model = _model(_config(1.2, 0.8, 6, 5))
        faces, conditions = _shear(mesh, 0.3)
        u = faces.u.copy()
        u[0, :] = 0.0
        faces = FaceVelocities.copy_of(u, faces.v)
        state = model.initial(1.0e-3, 1.0e-4)
        assert np.all(model.terms(state, faces, conditions).production[0, :] > 0.0)
        for _ in range(50):
            state = model.step(state, faces, conditions)
        assert np.all(np.isfinite(state.k)) and np.all(state.k[0, :] > 1.0e-3)

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_a_true_time_step_is_one_value_for_every_cell(self, variant: str) -> None:
        """Couette flow with its bottom row at rest, uniform k and eps, one float
        dt at the stable step: the advection of a uniform field is zero, and
        with the viscosity and nu_t small enough that diffusion between rows
        moves a value by under 1e-8 of it, every cell's step is k1 = (k0 + dt
        P) / (1 + dt eps0 / k0) and eps1 = (eps0 + dt C_1 (eps0 / k0) P) / (1 +
        dt C_2 eps0 / k0) with the one dt, the row at rest as the moving rows.
        P is evaluated by hand from the corner shear. The row at rest has
        production from its top corners, so it does not simply decay."""
        mesh, model = _model(_config(1.2, 0.8, 6, 5, variant, viscosity=1.0e-12))
        const = model.constants
        faces, conditions = _shear(mesh, 0.3)
        u = faces.u.copy()
        u[0, :] = 0.0
        faces = FaceVelocities.copy_of(u, faces.v)
        dt = 0.5 / float(_rate_by_loop(mesh, faces).max())
        k0, eps0 = 1.0e-4, 1.0e-2
        # The inlet carries the field's own values, so the uniform field's
        # advection is zero in every row.
        same = TurbulenceConditions.uniform(mesh, k0, eps0)
        conditions = dataclasses.replace(
            conditions,
            inflow_k_u=same.inflow_k_u,
            inflow_k_v=same.inflow_k_v,
            inflow_eps_u=same.inflow_eps_u,
            inflow_eps_v=same.inflow_eps_v,
        )
        state = model.step(model.initial(k0, eps0), faces, conditions, dt)

        speed, height = 0.3, 0.8
        nu_t = const.c_mu * k0**2 / eps0
        column = u[:, 0]
        nodes = np.concatenate(([0.0], mesh.yc, [height]))
        values = np.concatenate(([0.0], column, [speed]))
        corner = np.diff(values) / np.diff(nodes)
        shear = 0.5 * (corner[:-1] + corner[1:])
        for row in (0, 3):
            p = nu_t * shear[row] ** 2
            k1 = (k0 + dt * p) / (1.0 + dt * eps0 / k0)
            eps1 = (eps0 + dt * const.c_1 * eps0 / k0 * p) / (
                1.0 + dt * const.c_2 * eps0 / k0
            )
            assert np.allclose(state.k[row, :], k1, rtol=1e-8, atol=0.0), row
            assert np.allclose(state.eps[row, :], eps1, rtol=1e-8, atol=0.0), row
        assert shear[0] > 0.0

    def test_the_pseudo_time_step_reads_both_directions(self) -> None:
        """A field moving in x and y with four different gradients on a mesh
        stretched both ways: each cell's step is cfl over the sum of its
        horizontal and vertical rates, evaluated cell by cell. The one-direction
        trap of test 32b: a rate that drops its vertical half passes a field
        that moves along x only."""
        mesh, model = _model(
            _config(
                1.2,
                0.8,
                7,
                5,
                mesh={"x": {"stretch_ratio": 1.2}, "y": {"stretch_ratio": 1.3}},
            )
        )
        faces, _ = _linear_field(mesh, *LINEAR)
        expected = 0.5 / _rate_by_loop(mesh, faces)
        assert np.allclose(model.pseudo_time_step(faces), expected, rtol=1e-14)

    def test_a_true_time_step_a_little_above_the_stable_step_is_refused(
        self,
    ) -> None:
        """The stable step on a field moving both ways, from the loop: 0.999 of
        it is accepted, 1.02 of it refused, so a bound loosened by a few percent,
        or one that reads one direction only, fails here."""
        mesh, model = _model(
            _config(1.2, 0.8, 7, 5, mesh={"y": {"stretch_ratio": 1.3}})
        )
        faces, conditions = _linear_field(mesh, *LINEAR)
        stable = 0.5 / float(_rate_by_loop(mesh, faces).max())
        state = model.initial(1.0e-3, 1.0e-4)
        model.step(state, faces, conditions, 0.999 * stable)
        with pytest.raises(ValueError, match="exceeds the stable step"):
            model.step(state, faces, conditions, 1.02 * stable)


# ---------------------------------------------------------------------------
# Where the gradients live
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStrain:
    def test_plane_couette_flow_has_its_shear_in_every_cell_the_wall_cells_included(
        self,
    ) -> None:
        """u = U y / H on a stretched mesh, the top edge moving at U: the corner
        on each wall reads the wall's own velocity at the wall, so S^2 is
        (U / H)^2 in the wall cells as in the core."""
        mesh, model = _model(
            _config(1.2, 0.8, 6, 7, mesh={"y": {"stretch_ratio": 1.3}})
        )
        faces, conditions = _shear(mesh, 0.3)
        s2 = model.strain_squared(faces, conditions)
        assert np.allclose(s2, (0.3 / 0.8) ** 2, rtol=1e-12, atol=0.0)

    def test_a_pure_strain_has_no_shear(self) -> None:
        """u = a x, v = -a y with the edges carrying the field's own values:
        S^2 = 2 a^2 + 2 a^2, from the face differences alone."""
        mesh, model = _model(
            _config(1.2, 0.8, 6, 5, mesh={"x": {"stretch_ratio": 1.2}})
        )
        a = 0.7
        u = np.repeat((a * mesh.x)[None, :], mesh.yc.shape[0], axis=0)
        v = np.repeat((-a * mesh.y)[:, None], mesh.xc.shape[0], axis=1)
        conditions = dataclasses.replace(
            TurbulenceConditions.uniform(mesh, 1.0e-3, 1.0e-4),
            tangential_bottom=a * mesh.x,
            tangential_top=a * mesh.x,
            tangential_left=-a * mesh.y,
            tangential_right=-a * mesh.y,
        )
        s2 = model.strain_squared(FaceVelocities.copy_of(u, v), conditions)
        assert np.allclose(s2, 4.0 * a**2, rtol=1e-12, atol=0.0)

    def test_an_obstacle_face_gives_the_full_shear_too(self) -> None:
        """A full-width obstacle 0.3 m high under Couette flow from its top to
        the moving lid: the cells resting on it read the obstacle's zero at
        its face, so S^2 is (U / (H - h))^2 there as in the core, and zero in
        the obstacle."""
        obstacle = [
            {
                "name": "floor",
                "x_start": 0.0,
                "x_end": 1.2,
                "y_start": 0.0,
                "y_end": 0.3,
            }
        ]
        mesh, model = _model(_config(1.2, 0.9, 6, 9, obstacles=obstacle))
        solid = mesh.cell_type == SOLID
        assert solid[:3, :].all() and not solid[3:, :].any()
        speed, h = 0.3, 0.3
        profile = np.where(mesh.yc > h, speed * (mesh.yc - h) / (0.9 - h), 0.0)
        u = np.repeat(profile[:, None], u_shape(mesh)[1], axis=1)
        conditions = dataclasses.replace(
            TurbulenceConditions.uniform(mesh, 1.0e-3, 1.0e-4),
            tangential_top=np.full(mesh.x.shape, speed),
        )
        s2 = model.strain_squared(
            FaceVelocities.copy_of(u, np.zeros(v_shape(mesh))), conditions
        )
        assert np.allclose(s2[~solid], (speed / (0.9 - h)) ** 2, rtol=1e-12, atol=0.0)
        assert np.all(s2[solid] == 0.0)

    def test_a_linear_field_in_both_directions_has_each_term_in_its_place(
        self,
    ) -> None:
        """u = a x + b y, v = c x + d y with four different coefficients, the
        edges carrying the field's own values, on a mesh stretched both ways:
        S^2 = 2 a^2 + 2 d^2 + (b + c)^2 in every cell. A strain that drops dv/dx
        or dv/dy, or swaps the left and right edges' velocities, fails: the
        one-direction trap of test 32b, which a field moving along x cannot
        show."""
        mesh, model = _model(
            _config(
                1.2,
                0.8,
                7,
                5,
                mesh={"x": {"stretch_ratio": 1.25}, "y": {"stretch_ratio": 1.2}},
            )
        )
        a, b, c, d = LINEAR
        faces, conditions = _linear_field(mesh, a, b, c, d)
        s2 = model.strain_squared(faces, conditions)
        expected = 2.0 * a**2 + 2.0 * d**2 + (b + c) ** 2
        assert np.allclose(s2, expected, rtol=1e-12, atol=0.0)

    def test_a_shear_that_differs_at_each_corner_is_averaged_over_all_four(
        self,
    ) -> None:
        """u = q y^2 and v = r x^2 on a mesh stretched both ways: every corner
        carries its own shear, du/dy = q (y_hi + y_lo) and dv/dx = r (x_hi + x_lo)
        between the two nodes either side, so a cell's shear is the mean of its
        four corners, ((g_j + g_j+1) / 2 + (h_i + h_i+1) / 2), evaluated here node
        by node. One corner in place of four, or the y half of the trap of test
        32b, fails."""
        mesh, model = _model(
            _config(
                1.2,
                0.8,
                7,
                5,
                mesh={"x": {"stretch_ratio": 1.3}, "y": {"stretch_ratio": 1.25}},
            )
        )
        q, r = 0.9, -0.4
        width, height = 1.2, 0.8
        u = np.repeat((q * mesh.yc**2)[:, None], u_shape(mesh)[1], axis=1)
        v = np.repeat((r * mesh.xc**2)[None, :], v_shape(mesh)[0], axis=0)
        conditions = dataclasses.replace(
            TurbulenceConditions.uniform(mesh, 1.0e-3, 1.0e-4),
            tangential_bottom=np.zeros(mesh.x.shape),
            tangential_top=np.full(mesh.x.shape, q * height**2),
            tangential_left=np.zeros(mesh.y.shape),
            tangential_right=np.full(mesh.y.shape, r * width**2),
        )
        s2 = model.strain_squared(FaceVelocities.copy_of(u, v), conditions)
        y_nodes = np.concatenate(([0.0], mesh.yc, [height]))
        x_nodes = np.concatenate(([0.0], mesh.xc, [width]))
        g = q * (y_nodes[1:] + y_nodes[:-1])
        h = r * (x_nodes[1:] + x_nodes[:-1])
        shear = 0.5 * (g[:-1] + g[1:])[:, None] + 0.5 * (h[:-1] + h[1:])[None, :]
        assert len(np.unique(np.round(shear, 12))) == shear.size
        assert np.allclose(s2, shear**2, rtol=1e-12, atol=0.0)

    def test_a_vertical_obstacle_face_gives_the_full_shear_too(self) -> None:
        """A full-height obstacle 0.3 m wide on the left under vertical Couette
        flow to the moving right edge, v = V (x - w) / (W - w): the cells beside
        its vertical face read the obstacle's zero at the face, so S^2 is (V /
        (W - w))^2 there as in the core. The vertical face is the y half of the
        horizontal-face test above, the one-direction trap of test 32b."""
        obstacle = [
            {"name": "wall", "x_start": 0.0, "x_end": 0.3, "y_start": 0.0, "y_end": 0.8}
        ]
        mesh, model = _model(_config(1.2, 0.8, 8, 5, obstacles=obstacle))
        solid = mesh.cell_type == SOLID
        assert solid[:, :2].all() and not solid[:, 2:].any()
        speed, w, width = 0.3, 0.3, 1.2
        profile = np.where(mesh.xc > w, speed * (mesh.xc - w) / (width - w), 0.0)
        v = np.repeat(profile[None, :], v_shape(mesh)[0], axis=0)
        conditions = dataclasses.replace(
            TurbulenceConditions.uniform(mesh, 1.0e-3, 1.0e-4),
            tangential_right=np.full(mesh.y.shape, speed),
        )
        s2 = model.strain_squared(
            FaceVelocities.copy_of(np.zeros(u_shape(mesh)), v), conditions
        )
        assert np.allclose(s2[~solid], (speed / (width - w)) ** 2, rtol=1e-12, atol=0.0)
        assert np.all(s2[solid] == 0.0)

    def test_a_velocity_held_on_an_obstacle_face_makes_no_strain(self) -> None:
        """Every face beside or inside an obstacle holds a velocity and every
        other face is at rest: those faces carry no air, so the strain and the
        production are zero everywhere, production on, and a step equals the
        step on the field at rest bit for bit (review 35, S3)."""
        obstacle = [
            {"name": "b", "x_start": 0.5, "x_end": 0.8, "y_start": 0.0, "y_end": 0.35}
        ]
        mesh, model = _model(_config(1.5, 0.9, 10, 7, obstacles=obstacle))
        solid = mesh.cell_type == SOLID
        u = np.zeros(u_shape(mesh))
        v = np.zeros(v_shape(mesh))
        u[:, 1:-1] = np.where(solid[:, :-1] | solid[:, 1:], 0.3, 0.0)
        u[:, 0] = np.where(solid[:, 0], 0.3, 0.0)
        v[1:-1, :] = np.where(solid[:-1, :] | solid[1:, :], -0.3, 0.0)
        v[0, :] = np.where(solid[0, :], -0.3, 0.0)
        assert np.abs(u).sum() > 0.0 and np.abs(v).sum() > 0.0
        faces = FaceVelocities.copy_of(u, v)
        conditions = TurbulenceConditions.uniform(mesh, 1.0e-3, 1.0e-4)
        state = model.initial(1.0e-3, 1.0e-4)
        assert np.all(model.strain_squared(faces, conditions) == 0.0)
        assert np.all(model.terms(state, faces, conditions).production == 0.0)
        held = model.step(state, faces, conditions, 0.5)
        rest = model.step(state, _at_rest(mesh), conditions, 0.5)
        assert np.array_equal(held.k, rest.k) and np.array_equal(held.eps, rest.eps)


# ---------------------------------------------------------------------------
# The terms, and RNG's R
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTerms:
    def test_the_standard_terms_and_a_given_production(self) -> None:
        mesh, model = _model(_config(1.2, 0.8, 6, 5))
        faces, conditions = _shear(mesh, 0.3)
        given = np.zeros(mesh.cell_type.shape, dtype=bool)
        given[0, :] = True
        conditions = dataclasses.replace(
            conditions, production_given=given, production=np.full(given.shape, 0.25)
        )
        rng = np.random.default_rng(4)
        state = model.state(
            rng.uniform(1e-3, 1e-2, given.shape), rng.uniform(1e-4, 1e-3, given.shape)
        )
        t = model.terms(state, faces, conditions)
        p = state.nu_t * (0.3 / 0.8) ** 2
        p[0, :] = 0.25
        ratio = state.eps / state.k
        assert np.allclose(t.production, p, rtol=1e-12)
        assert np.all(t.production[0, :] == 0.25)
        assert np.array_equal(t.growth_k, t.production)
        assert np.allclose(t.growth_eps, 1.44 * ratio * p, rtol=1e-12)
        assert np.array_equal(t.decay_k, ratio)
        assert np.allclose(t.decay_eps, 1.92 * ratio, rtol=1e-14)
        assert np.all(t.rng_r == 0.0)

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_one_step_at_rest_with_a_given_production_is_the_linearised_formula(
        self, variant: str
    ) -> None:
        """A closed box at rest, uniform k and eps, production given in every
        cell: no advection, no diffusion of a uniform field, so one step is
        k1 = (k0 + dt P) / (1 + dt eps0 / k0) and eps1 = (eps0 + dt C_1 (eps0
        / k0) P) / (1 + dt C_2 eps0 / k0). Growth explicit and decay implicit,
        both from the previous iterate (ADR-012 C), with C_1 and C_2 in their
        places. R is zero at rest."""
        mesh, model = _model(_config(2.4, 1.5, 6, 4, variant, tol=1e-15))
        const = model.constants
        k0, eps0, p, dt = 0.0137, 0.00291, 0.0042, 1.7
        given = np.ones(mesh.cell_type.shape, dtype=bool)
        conditions = dataclasses.replace(
            TurbulenceConditions.uniform(mesh, k0, eps0),
            production_given=given,
            production=np.full(given.shape, p),
        )
        state = model.step(model.initial(k0, eps0), _at_rest(mesh), conditions, dt)
        k1 = (k0 + dt * p) / (1.0 + dt * eps0 / k0)
        eps1 = (eps0 + dt * const.c_1 * eps0 / k0 * p) / (
            1.0 + dt * const.c_2 * eps0 / k0
        )
        assert np.allclose(state.k, k1, rtol=1e-13, atol=0.0)
        assert np.allclose(state.eps, eps1, rtol=1e-13, atol=0.0)

    def test_rng_at_rest_has_no_r_and_decays_with_its_own_c_2(self) -> None:
        mesh, model = _model(_config(1.2, 0.8, 6, 5, "rng"))
        state = model.initial(0.0137, 0.00291)
        t = model.terms(
            state, _at_rest(mesh), TurbulenceConditions.uniform(mesh, 0.0137, 0.00291)
        )
        assert np.all(t.rng_r == 0.0)
        assert np.allclose(t.decay_eps, 1.68 * 0.00291 / 0.0137, rtol=1e-15)

    def test_rng_r_splits_by_sign_into_the_growth_and_the_diagonal(self) -> None:
        """At a uniform shear S = U / H = 0.375 1/s, so that S and S^2 differ,
        k / eps varied cell by cell puts eta from 1 to 9, either side of eta_0 =
        4.38, so (1 - eta / eta_0) changes sign across the cells: R from its
        formula; negative R in eps's growth, positive R over eps in its
        diagonal, so both stay non-negative."""
        mesh, model = _model(_config(1.2, 0.8, 8, 6, "rng"))
        speed = 0.3
        faces, conditions = _shear(mesh, speed)
        s = speed / 0.8
        eta_target = np.linspace(1.0, 9.0, 48).reshape(6, 8)
        k = np.full((6, 8), 2.0e-3)
        eps = s * k / eta_target
        state = model.state(k, eps)
        t = model.terms(state, faces, conditions)

        assert s != s**2
        eta = s * k / eps
        r = 0.0845 * eta**3 * (1.0 - eta / 4.38) / (1.0 + 0.012 * eta**3) * eps**2 / k
        assert (r > 0.0).any() and (r < 0.0).any()
        assert np.allclose(t.rng_r, r, rtol=1e-11, atol=0.0)
        production = state.nu_t * s**2
        assert np.allclose(
            t.growth_eps,
            1.42 * eps / k * production + np.where(r < 0.0, -r, 0.0),
            rtol=1e-11,
        )
        assert np.allclose(
            t.decay_eps, 1.68 * eps / k + np.where(r > 0.0, r / eps, 0.0), rtol=1e-11
        )
        assert t.growth_eps.min() >= 0.0 and t.decay_eps.min() > 0.0


# ---------------------------------------------------------------------------
# Boundaries and the face rule
# ---------------------------------------------------------------------------


def _channel(mesh: Mesh, speed: float) -> FaceVelocities:
    """Uniform flow to the right; the top and bottom faces carry nothing."""
    return FaceVelocities.copy_of(
        np.full(u_shape(mesh), speed), np.zeros(v_shape(mesh))
    )


def _slip(mesh: Mesh, speed: float, k: float, eps: float) -> TurbulenceConditions:
    """Edges moving with the uniform flow, so it has no shear anywhere."""
    return dataclasses.replace(
        TurbulenceConditions.uniform(mesh, k, eps),
        tangential_bottom=np.full(mesh.x.shape, speed),
        tangential_top=np.full(mesh.x.shape, speed),
    )


@pytest.mark.unit
class TestBoundaries:
    def test_the_inflow_value_is_read_where_the_flux_enters_and_nowhere_else(
        self,
    ) -> None:
        mesh, model = _model(_config(1.2, 0.8, 6, 4))
        faces = _channel(mesh, 0.2)
        state = model.initial(1.0e-3, 1.0e-4)
        base = _slip(mesh, 0.2, 1.0e-3, 1.0e-4)
        reference = model.step(state, faces, base)

        richer = base.inflow_k_u.copy()
        richer[:, 0] = 4.0e-3
        entering = model.step(
            state, faces, dataclasses.replace(base, inflow_k_u=richer)
        )
        assert np.all(entering.k[:, 0] > reference.k[:, 0])

        # The outflow faces and the faces with no flux never read their values.
        ignored_u = base.inflow_k_u.copy()
        ignored_u[:, -1] = 9.0
        ignored_v = base.inflow_eps_v.copy()
        ignored_v[[0, -1], :] = 9.0
        other = model.step(
            state,
            faces,
            dataclasses.replace(base, inflow_k_u=ignored_u, inflow_eps_v=ignored_v),
        )
        assert np.array_equal(other.k, reference.k)
        assert np.array_equal(other.eps, reference.eps)

    def test_k_and_eps_enter_through_a_bottom_face_each_with_its_own_value(
        self,
    ) -> None:
        """Uniform flow upward, edges slipping with it so nothing is sheared:
        k's inflow on the bottom faces raises k in the bottom row and leaves eps
        bit for bit, eps's inflow raises eps and leaves k, and the top faces'
        values, where the air leaves, are never read. The one-direction trap of
        test 32b: the left-edge inflow test cannot see the v faces' arrays."""
        mesh, model = _model(_config(1.2, 0.8, 6, 4))
        speed = 0.2
        faces = FaceVelocities.copy_of(
            np.zeros(u_shape(mesh)), np.full(v_shape(mesh), speed)
        )
        base = dataclasses.replace(
            TurbulenceConditions.uniform(mesh, 1.0e-3, 1.0e-4),
            tangential_left=np.full(mesh.y.shape, speed),
            tangential_right=np.full(mesh.y.shape, speed),
        )
        state = model.initial(1.0e-3, 1.0e-4)
        assert np.all(model.terms(state, faces, base).production == 0.0)
        reference = model.step(state, faces, base)

        richer_k = base.inflow_k_v.copy()
        richer_k[0, :] = 4.0e-3
        k_in = model.step(state, faces, dataclasses.replace(base, inflow_k_v=richer_k))
        assert np.all(k_in.k[0, :] > reference.k[0, :])
        assert np.array_equal(k_in.eps, reference.eps)

        richer_eps = base.inflow_eps_v.copy()
        richer_eps[0, :] = 4.0e-4
        eps_in = model.step(
            state, faces, dataclasses.replace(base, inflow_eps_v=richer_eps)
        )
        assert np.all(eps_in.eps[0, :] > reference.eps[0, :])
        assert np.array_equal(eps_in.k, reference.k)

        top_k = base.inflow_k_v.copy()
        top_k[-1, :] = 9.0
        top_eps = base.inflow_eps_v.copy()
        top_eps[-1, :] = 9.0
        leaving = model.step(
            state,
            faces,
            dataclasses.replace(base, inflow_k_v=top_k, inflow_eps_v=top_eps),
        )
        assert np.array_equal(leaving.k, reference.k)
        assert np.array_equal(leaving.eps, reference.eps)

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_nothing_crosses_a_wall_or_an_obstacle_face(
        self, monkeypatch: pytest.MonkeyPatch, variant: str
    ) -> None:
        """Production and decay off, a closed box with an obstacle, one
        true-time step for every cell: the content sum(k V) is conserved,
        though the faces of the obstacle hold a velocity, and the obstacle
        stays empty. (The pseudo-time step is per cell and conserves nothing,
        ADR-012 C, so the check needs a uniform step.)"""
        obstacle = [
            {"name": "b", "x_start": 0.5, "x_end": 0.8, "y_start": 0.0, "y_end": 0.35}
        ]
        mesh, model = _model(
            _config(
                1.5,
                0.9,
                10,
                7,
                variant,
                obstacles=obstacle,
                mesh={"x": {"stretch_ratio": 1.2}},
            )
        )
        solid = mesh.cell_type == SOLID
        faces = _streamfunction_field(mesh, seed=7, scale=0.05)
        u = faces.u.copy()
        v = faces.v.copy()
        u[:, 1:-1] = np.where(solid[:, :-1] | solid[:, 1:], 0.3, u[:, 1:-1])
        v[1:-1, :] = np.where(solid[:-1, :] | solid[1:, :], -0.3, v[1:-1, :])
        faces = FaceVelocities.copy_of(u, v)
        _no_sources(model, monkeypatch)
        rng = np.random.default_rng(8)
        state = model.state(
            rng.uniform(1e-3, 2e-3, solid.shape), rng.uniform(1e-4, 2e-4, solid.shape)
        )
        volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        before = float(np.sum(state.k * volume))
        conditions = TurbulenceConditions.uniform(mesh, 5.0e-3, 5.0e-4)
        dt = 0.5 / float(model._rate(faces)[~solid].max())
        for _ in range(50):
            state = model.step(state, faces, conditions, dt)
        assert abs(float(np.sum(state.k * volume)) / before - 1.0) < 1e-10
        assert np.all(state.k[solid] == 0.0)

    def test_held_eps_keeps_its_value_and_its_neighbours_read_it(self) -> None:
        mesh, model = _model(_config(1.2, 0.8, 6, 5))
        faces, conditions = _shear(mesh, 0.3)
        state = model.initial(1.0e-3, 1.0e-4)
        free = model.step(state, faces, conditions)
        held = np.zeros(mesh.cell_type.shape, dtype=bool)
        held[0, :] = True
        wall = np.full(held.shape, 3.7e-3)
        fixed = model.step(
            state,
            faces,
            dataclasses.replace(conditions, eps_held=held, eps_wall=wall),
        )
        assert np.all(fixed.eps[0, :] == 3.7e-3)
        assert np.all(fixed.eps[1, :] > free.eps[1, :])

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    @pytest.mark.parametrize("quantity", ["k", "eps"])
    def test_the_face_diffusivity_is_the_harmonic_mean_with_its_own_sigma_adr012_f(
        self, quantity: str, variant: str
    ) -> None:
        """One true-time step at rest on a stretched mesh with a varying eddy
        viscosity: each quantity's implicit solve against the system assembled
        face by face, each face's (nu + nu_t / sigma) the distance-weighted
        harmonic mean of its two cells, with that quantity's sigma written here
        as ADR-012 A tabulates it (1.0 and 1.3 standard, 0.7194 for both in
        RNG), and its decay, eps / k for k and C_2 eps / k for eps, in the
        diagonal. A sigma swapped between k and eps, dropped, or taken from the
        other variant fails."""
        mesh, model = _model(
            _config(
                1.2,
                0.8,
                7,
                5,
                variant,
                tol=1e-15,
                mesh={"x": {"stretch_ratio": 1.3}, "y": {"stretch_ratio": 1.2}},
            )
        )
        sigma = {
            ("standard", "k"): 1.0,
            ("standard", "eps"): 1.3,
            ("rng", "k"): 0.7194,
            ("rng", "eps"): 0.7194,
        }[(variant, quantity)]
        c_2 = {"standard": 1.92, "rng": 1.68}[variant]
        rng = np.random.default_rng(12)
        shape = mesh.cell_type.shape
        state = model.state(
            rng.uniform(0.02, 0.2, shape), rng.uniform(1e-3, 1e-2, shape)
        )
        dt = 0.3
        result = model.step(
            state, _at_rest(mesh), TurbulenceConditions.uniform(mesh, 0.1, 0.01), dt
        )
        got = result.k if quantity == "k" else result.eps
        start = state.k if quantity == "k" else state.eps
        decay = state.eps / state.k * (1.0 if quantity == "k" else c_2)

        nu = AIR["viscosity"] / AIR["density"]
        gamma = nu + state.nu_t / sigma
        ny, nx = shape
        idx = np.arange(ny * nx).reshape(shape)
        volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        a = np.zeros((ny * nx, ny * nx))
        a[idx, idx] = volume / dt + volume * decay

        def couple(p: int, q: int, g: float) -> None:
            a[p, p] += g
            a[q, q] += g
            a[p, q] -= g
            a[q, p] -= g

        for j in range(ny):
            for i in range(1, nx):
                d_w, d_e = mesh.x[i] - mesh.xc[i - 1], mesh.xc[i] - mesh.x[i]
                face = (d_w + d_e) / (d_w / gamma[j, i - 1] + d_e / gamma[j, i])
                couple(idx[j, i - 1], idx[j, i], face * mesh.dy_cell[j] / (d_w + d_e))
        for j in range(1, ny):
            for i in range(nx):
                d_s, d_n = mesh.y[j] - mesh.yc[j - 1], mesh.yc[j] - mesh.y[j]
                face = (d_s + d_n) / (d_s / gamma[j - 1, i] + d_n / gamma[j, i])
                couple(idx[j - 1, i], idx[j, i], face * mesh.dx_cell[i] / (d_s + d_n))
        expected = np.linalg.solve(a, (volume / dt * start).ravel()).reshape(shape)
        assert np.allclose(got, expected, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# Positivity (REQ-S15, ECR-002 criterion 3)
# ---------------------------------------------------------------------------

POSITIVITY_STEPS = 300


KAPPA = 0.41  # the stand-in wall function's von Karman constant (ADR-012 B)


def _smith_hutton_case(
    variant: str,
) -> tuple[Mesh, KEpsilonModel, FaceVelocities, Callable]:
    """The Smith-Hutton field, 40x20, with ten wall cells as step 6 will give them.

    Ten cells of the top row are wall cells. Their conditions are rebuilt
    every step from the state, as step 6 rebuilds them from the wall
    functions: eps held at the log law's ``C_mu^(3/4) k^(3/2) / (kappa
    y_P)`` and the production given equal to it, the local equilibrium a
    wall function assumes. A stand-in for ADR-012 B, not its build. A held
    eps that does not follow k is no wall cell: beside a shear it pins eps
    low while the production raises k, nu_t grows as k^2 / eps, and k runs
    away (prompt 35, results/builder35/runaway35.md).
    """
    mesh, model = _model(_config(2.0, 1.0, 40, 20, variant, tol=1e-10, max_iter=500))
    faces = smith_hutton_face_field(mesh, SMITH_HUTTON_SPEED)
    base = _smith_hutton_conditions(mesh, 1.0e-3, 1.0e-4)
    wall = np.zeros(mesh.cell_type.shape, dtype=bool)
    wall[-1, 25:35] = True
    y_p = mesh.y[-1] - mesh.yc[-1]
    c_mu = model.constants.c_mu

    def conditions_for(state: TurbulenceState) -> TurbulenceConditions:
        eps_wall = np.where(wall, c_mu**0.75 * state.k**1.5 / (KAPPA * y_p), 0.0)
        return dataclasses.replace(
            base,
            eps_held=wall,
            eps_wall=eps_wall,
            production_given=wall,
            production=eps_wall,
        )

    return mesh, model, faces, conditions_for


def _random_case(
    variant: str,
) -> tuple[Mesh, KEpsilonModel, FaceVelocities, Callable]:
    """A random divergence-free field in a closed stretched room with an obstacle."""
    obstacle = [
        {"name": "b", "x_start": 0.9, "x_end": 1.3, "y_start": 0.0, "y_end": 0.5}
    ]
    mesh, model = _model(
        _config(
            2.4,
            1.5,
            24,
            15,
            variant,
            tol=1e-10,
            max_iter=500,
            obstacles=obstacle,
            mesh={"x": {"stretch_ratio": 1.15}, "y": {"stretch_ratio": 1.1}},
        )
    )
    faces = _streamfunction_field(mesh, seed=21, scale=0.06)
    conditions = TurbulenceConditions.uniform(mesh, 1.0e-3, 1.0e-4)
    return mesh, model, faces, lambda state: conditions


CASES = {"smith_hutton": _smith_hutton_case, "random": _random_case}


RANDOM_CONVERGED_STEPS = 40


def _bounded_start(shape: tuple[int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A non-uniform start near k = eps = 1e-4, nu_t about 1e-5 m^2/s.

    On the random field the frozen strain grows k without a steady state (the
    model's homogeneous-shear growth, test 35 check 7); from this start nu_t
    stays small enough over RANDOM_CONVERGED_STEPS steps that every implicit
    solve meets its tolerance within the 500-sweep cap.
    """
    rng = np.random.default_rng(seed)
    k = rng.uniform(0.5, 1.5, size=shape) * 1.0e-4
    eps = rng.uniform(0.5, 1.5, size=shape) * 1.0e-4
    return k, eps


@pytest.mark.unit
class TestPositivity:
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    @pytest.mark.parametrize("case", list(CASES))
    def test_k_and_eps_stay_positive_and_every_solve_converges(
        self, case: str, variant: str
    ) -> None:
        """Production on, from a non-uniform start: the Smith-Hutton field from
        the patchy start for 300 pseudo-time steps, the random field from a
        start whose nu_t stays bounded for 40. At every step both implicit
        solves meet their tolerance, so each state is the step's own, and the
        step's assertion holds; the run produced and moved k."""
        mesh, model, faces, conditions_for = CASES[case](variant)
        live = mesh.cell_type != SOLID
        if case == "smith_hutton":
            k, eps = _patchy_start(model, mesh.cell_type.shape, seed=3)
            steps = POSITIVITY_STEPS
        else:
            k, eps = _bounded_start(mesh.cell_type.shape, seed=5)
            steps = RANDOM_CONVERGED_STEPS
        state = model.state(k, eps)
        produced = model.terms(state, faces, conditions_for(state)).production
        assert produced.max() > 0.0
        least_k = least_eps = math.inf
        most_sweeps = 0
        for n in range(steps):
            state = model.step(state, faces, conditions_for(state))
            assert model.solves_converged, f"step {n}: sweeps {model.last_sweeps}"
            most_sweeps = max(most_sweeps, *model.last_sweeps)
            least_k = min(least_k, float(state.k[live].min()))
            least_eps = min(least_eps, float(state.eps[live].min()))
        print(
            f"positivity {case} {variant}: {steps} steps, most sweeps {most_sweeps}; "
            f"least k {least_k:.3e}, least eps {least_eps:.3e}; end k "
            f"[{state.k[live].min():.3e}, {state.k[live].max():.3e}]"
        )
        assert least_k > 0.0 and least_eps > 0.0
        assert not np.allclose(state.k[live], k[live])

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_a_long_run_stays_positive_while_its_solves_are_truncated(
        self, variant: str
    ) -> None:
        """The random field from the patchy start for 300 pseudo-time steps.
        The frozen strain grows k without a steady state, nu_t with it, and
        the face conductances outgrow V / dt by orders, so the implicit solves
        stop at their 500-sweep cap on most steps (test 35 check 7): the
        states are truncated Jacobi iterates, not the step's solution.
        Positivity holds all the same, because every Jacobi sweep is a
        non-negative combination of a non-negative right-hand side and
        non-negative neighbours (scalar_scheme.implicit_step), so a truncated
        iterate is positive as the converged one is. Whether the coupled solve
        should iterate further is a question for steps 5 and 6, not this
        test."""
        mesh, model, faces, conditions_for = CASES["random"](variant)
        live = mesh.cell_type != SOLID
        state = model.state(*_patchy_start(model, mesh.cell_type.shape, seed=3))
        truncated = 0
        least_k = least_eps = math.inf
        for _ in range(POSITIVITY_STEPS):
            state = model.step(state, faces, conditions_for(state))
            truncated += not model.solves_converged
            least_k = min(least_k, float(state.k[live].min()))
            least_eps = min(least_eps, float(state.eps[live].min()))
        print(
            f"long run {variant}: {truncated} of {POSITIVITY_STEPS} steps truncated; "
            f"least k {least_k:.3e}, least eps {least_eps:.3e}"
        )
        # The run exists to exercise truncated solves; if they all converge,
        # its docstring no longer describes it.
        assert truncated > 0
        assert least_k > 0.0 and least_eps > 0.0
        assert np.all(np.isfinite(state.k)) and np.all(np.isfinite(state.eps))

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    @pytest.mark.parametrize("case", list(CASES))
    def test_the_planted_explicit_decay_goes_negative(
        self, monkeypatch: pytest.MonkeyPatch, case: str, variant: str
    ) -> None:
        """The control: the same start with the decay moved to step 2's explicit
        growth, ``k - dt eps``, takes k below zero within a few steps at a
        Courant number of 1/2, and the step's assertion names it."""
        mesh, model, faces, conditions_for = CASES[case](variant)
        _explicit_decay(model, monkeypatch)
        state = model.state(*_patchy_start(model, mesh.cell_type.shape, seed=3))
        with pytest.raises(PositivityError) as raised:
            for _ in range(5):
                state = model.step(state, faces, conditions_for(state))
        assert raised.value.minimum < 0.0

    def test_the_assertion_names_eps_when_only_eps_goes_negative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """eps's own branch of the assertion: with only eps's decay made
        explicit, k stays positive and the step raises naming eps, with a
        negative least value."""
        mesh, model, faces, conditions_for = CASES["smith_hutton"]("standard")
        _explicit_decay(model, monkeypatch, for_k=False, for_eps=True)
        state = model.state(*_patchy_start(model, mesh.cell_type.shape, seed=3))
        with pytest.raises(PositivityError, match="eps is not positive") as raised:
            model.step(state, faces, conditions_for(state))
        assert raised.value.minimum < 0.0

    def test_the_assertion_refuses_a_value_that_is_not_finite(self) -> None:
        mesh, model = _model(_config(1.2, 0.8, 6, 4))
        q = np.full(mesh.cell_type.shape, 1.0)
        q[1, 2] = np.inf
        with pytest.raises(PositivityError, match="k is not positive and finite"):
            model._check_positive(q, "k")


# ---------------------------------------------------------------------------
# Constancy on the VAL-001 faces
# ---------------------------------------------------------------------------

CONSTANCY_SECONDS = 40.0


@pytest.fixture(scope="module")
def val001_faces() -> tuple[Mesh, FaceVelocities, np.ndarray, SimConfig]:
    """VAL-001 at 40x20 solved under its own rule, its faces and their imbalance."""
    raw = yaml.safe_load(case_path("poiseuille").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = 40, 20
    raw["turbulence"] = {
        "model": "k_epsilon",
        "wall_treatment": "scalable_wall_functions",
        "cfl_number": 0.5,
        "alpha_turbulence": 0.7,
        "max_iter": 500,
        "tol": 1.0e-12,
    }
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    boundary = StaggeredBoundary(mesh, config)
    solver = StaggeredSolver(mesh, config, boundary)
    solver.solve_steady()
    faces = solver.face_velocities
    imbalance = PressureCorrector(mesh, config, boundary).mass_imbalance(
        faces.u, faces.v
    )
    return mesh, faces, imbalance, config


def _drift(
    mesh: Mesh,
    config: SimConfig,
    faces: FaceVelocities,
    seconds: float,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[float, float, float]:
    """Largest relative departure of uniform k and eps, sources off, and T."""
    model = KEpsilonModel(mesh, config)
    _no_sources(model, monkeypatch)
    k0, eps0 = 2.3e-3, 4.1e-4
    conditions = TurbulenceConditions.uniform(mesh, k0, eps0)
    dt = 0.5 / float(model._rate(faces)[mesh.cell_type != SOLID].max())
    steps = math.ceil(seconds / dt)
    state = model.initial(k0, eps0)
    for _ in range(steps):
        state = model.step(state, faces, conditions, dt)
    live = mesh.cell_type != SOLID
    return (
        float(np.abs(state.k[live] / k0 - 1.0).max()),
        float(np.abs(state.eps[live] / eps0 - 1.0).max()),
        steps * dt,
    )


@pytest.mark.integration
def test_uniform_k_and_eps_stay_uniform_on_the_val001_faces(
    val001_faces: tuple[Mesh, FaceVelocities, np.ndarray, SimConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production and decay off, uniform k and eps with the inlet carrying the
    same: the departure stays within the transport scheme's constancy bound,
    max |b_P| T / (rho V_P) (REQ-T11, VAL-012); one interior face perturbed
    by 1e-6 m/s breaks it, so the bound can fail."""
    mesh, faces, imbalance, config = val001_faces
    k_drift, eps_drift, t_total = _drift(
        mesh, config, faces, CONSTANCY_SECONDS, monkeypatch
    )
    volume = np.outer(mesh.dy_cell, mesh.dx_cell)
    live = mesh.cell_type != SOLID
    bound = float((np.abs(imbalance) / (config.rho * volume))[live].max()) * t_total
    u = faces.u.copy()
    u[10, 20] += 1.0e-6
    control, _, _ = _drift(
        mesh, config, FaceVelocities.copy_of(u, faces.v), CONSTANCY_SECONDS, monkeypatch
    )
    print(
        f"constancy: T {t_total:.2f} s; departure k {k_drift:.3e}, eps "
        f"{eps_drift:.3e}; bound {bound:.3e}; perturbed face {control:.3e}"
    )
    assert k_drift <= bound and eps_drift <= bound
    assert control > bound
