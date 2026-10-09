"""Tests for the wall functions and the coupled solve's boundary data (ECR-002 step 6).

ADR-012 B and C, src/turbulence.py: the log-law floor and the wall viscosity
against their formulas, which faces and cells take a wall function, the wall
cells' held eps and given production (one wall, and the corner rule under
reflection), the inflow values and the initial state, and
StaggeredBoundary.wall_faces. Each planted defect of prompt 45's list for
commit A fails a test here (docs/reports/probe45/plant45.py). Meshes are
non-square, so an index that works only on square cells shows.
"""

import math

import numpy as np
import pytest

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.momentum import MomentumPredictor
from src.staggered import FaceVelocities, u_shape, v_shape
from src.turbulence import (
    E_WALL,
    KAPPA,
    VARIANTS,
    Y_STAR_FLOOR,
    KEpsilonModel,
    TurbulenceBoundary,
    inflow_values,
    log_law_floor,
    wall_viscosity,
)
from validation.transport_cases import AIR, PARTICLES, SOLVER_BLOCK

NU = AIR["viscosity"] / AIR["density"]
INLET_KEYS = {"turbulence_intensity": 0.05, "dissipation_length": 0.2}


def _config(
    width: float,
    height: float,
    nx: int,
    ny: int,
    boundaries: dict,
    obstacles: list[dict] | None = None,
    variant: str = "standard",
) -> SimConfig:
    """A validated configuration with the model on, air at rho 1.2."""
    raw = {
        "domain": {"width": width, "height": height, "nx": nx, "ny": ny},
        "fluid": AIR,
        "particles": PARTICLES,
        "solver": {**SOLVER_BLOCK, "stopping_rule": "error_estimate"},
        "turbulence": {
            "model": "k_epsilon",
            "variant": variant,
            "wall_treatment": "scalable_wall_functions",
            "cfl_number": 0.5,
            "alpha_turbulence": 0.7,
            "max_iter": 500,
            "tol": 1.0e-12,
        },
        "boundaries": boundaries,
        "obstacles": obstacles or [],
        "sensors": [{"name": "centre", "x": width / 2.0, "y": height / 2.0}],
        "thresholds": {"5e-06": 100.0},
    }
    return SimConfig.from_dict(raw)


def _built(config: SimConfig) -> tuple[Mesh, StaggeredBoundary, TurbulenceBoundary]:
    mesh = Mesh(config)
    boundary = StaggeredBoundary(mesh, config)
    return mesh, boundary, TurbulenceBoundary(mesh, config, boundary)


def _mixed_room() -> SimConfig:
    """Six by four cells of 1.0 by 0.5 m: every kind of edge face, and an obstacle.

    Top: an inlet over x 2 to 4 (cells 2 and 3, corners 2, 3 and 4), walls
    elsewhere. Left: a moving wall, a velocity inlet with zero normal
    velocity. Right: a pressure outlet over y 0 to 1 (cells 0 and 1, corners
    0, 1 and 2), wall above. Bottom: a fixed-flow outlet over x 4 to 6
    (cells 4 and 5, corners 4, 5 and 6), wall elsewhere. Cell (1, 2) SOLID.
    """
    return _config(
        6.0,
        2.0,
        6,
        4,
        {
            "supply": {
                "type": "velocity_inlet",
                "location": "top",
                "x_start": 2.0,
                "x_end": 4.0,
                "velocity": 1.0,
                **INLET_KEYS,
            },
            "belt": {
                "type": "velocity_inlet",
                "location": "left",
                "y_start": 0.0,
                "y_end": 2.0,
                "u_velocity": 0.0,
                "v_velocity": 0.5,
            },
            "vent": {
                "type": "pressure_outlet",
                "location": "right",
                "y_start": 0.0,
                "y_end": 1.0,
            },
            "fan": {
                "type": "fixed_flow_outlet",
                "location": "bottom",
                "x_start": 4.0,
                "x_end": 6.0,
                "velocity": 0.2,
            },
        },
        obstacles=[
            {"name": "box", "x_start": 2.0, "x_end": 3.0, "y_start": 0.5, "y_end": 1.0}
        ],
    )


def _mask(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for j, i in cells:
        out[j, i] = True
    return out


# ---------------------------------------------------------------------------
# The log-law floor and the wall viscosity
# ---------------------------------------------------------------------------


def _root_by_bisection(kappa: float, e: float) -> float:
    """The upper root of kappa y = ln(E y), found here without the module."""
    lo, hi = 1.0 / kappa, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if kappa * mid - math.log(e * mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@pytest.mark.unit
class TestLogLawFloor:
    """y*_0 from the coded constants, never typed (ADR-012 B, premise review S1)."""

    def test_constants_are_launder_and_spalding(self) -> None:
        assert (KAPPA, E_WALL) == (0.41, 9.793)

    def test_floor_is_the_root_of_the_two_laws(self) -> None:
        root = _root_by_bisection(KAPPA, E_WALL)
        assert pytest.approx(root, rel=1e-13) == Y_STAR_FLOOR
        assert round(Y_STAR_FLOOR, 2) == 11.53
        assert (
            pytest.approx(math.log(E_WALL * Y_STAR_FLOOR), rel=1e-15)
            == KAPPA * Y_STAR_FLOOR
        )

    def test_other_constants_give_their_own_floor(self) -> None:
        """kappa 0.41 with E 8.43 (B = 5.2) is where 11.06 came from."""
        assert log_law_floor(0.41, 8.43) == pytest.approx(
            _root_by_bisection(0.41, 8.43), rel=1e-13
        )
        assert round(log_law_floor(0.41, 8.43), 1) == 11.1

    def test_laws_that_do_not_meet_are_refused(self) -> None:
        with pytest.raises(ValueError, match="do not meet"):
            log_law_floor(0.41, 1.0)


@pytest.mark.unit
class TestWallViscosity:
    """mu_w against its formula; equal to mu where y* is the floor."""

    @pytest.mark.parametrize("variant", ["standard", "rng"])
    def test_equals_mu_at_the_floor(self, variant: str) -> None:
        """At y* = y*_0 the wall function is the molecular viscosity, to rounding.

        Taken at the module's floor and at the floor found here, so a typed
        11.06 in either place fails.
        """
        c_mu = VARIANTS[variant].c_mu
        y_p = np.array([1.0e-3, 0.0125, 0.02, 0.15])
        mu = AIR["viscosity"]
        for floor in (Y_STAR_FLOOR, _root_by_bisection(KAPPA, E_WALL)):
            k_p = (floor * NU / (c_mu**0.25 * y_p)) ** 2
            mu_w = wall_viscosity(k_p, y_p, c_mu, AIR["density"], NU)
            np.testing.assert_allclose(mu_w, mu, rtol=4e-15, atol=0.0)

    def test_matches_the_formula_above_and_below_the_floor(self) -> None:
        c_mu = VARIANTS["standard"].c_mu
        rho, floor = AIR["density"], _root_by_bisection(KAPPA, E_WALL)
        y_p = np.full(4, 0.02)
        y_star = np.array([2.0, 11.0, 30.0, 300.0])
        k_p = (y_star * NU / (c_mu**0.25 * y_p)) ** 2
        mu_w = wall_viscosity(k_p, y_p, c_mu, rho, NU)
        u_k = c_mu**0.25 * np.sqrt(k_p)
        expected = rho * u_k * 0.41 * y_p / np.log(9.793 * np.maximum(y_star, floor))
        np.testing.assert_allclose(mu_w, expected, rtol=1e-14)
        mu = rho * NU
        # Below the floor: mu y* / y*_0, less than mu; above it, more.
        np.testing.assert_allclose(mu_w[:2], mu * y_star[:2] / floor, rtol=1e-13)
        assert np.all(mu_w[:2] < mu) and np.all(mu_w[2:] > mu)


# ---------------------------------------------------------------------------
# Which faces and cells take a wall function
# ---------------------------------------------------------------------------

U_WALLS = {(0, 1), (0, 2), (0, 3), (4, 1), (4, 5), (1, 2), (1, 3), (2, 2), (2, 3)}
V_WALLS = {(1, 0), (2, 0), (3, 0), (3, 6), (1, 2), (2, 2), (1, 3), (2, 3)}
WALL_CELLS = (
    {(0, i) for i in range(4)}
    | {(2, 2), (0, 2), (1, 3), (1, 1)}
    | {(3, 0), (3, 1), (3, 4), (3, 5)}
    | {(j, 0) for j in range(4)}
    | {(2, 5), (3, 5)}
)


@pytest.mark.unit
class TestWallFaceSelection:
    """Walls at rest or moving and obstacle faces; never an inlet or an outlet."""

    def test_edge_walls_of_the_boundary_layer(self) -> None:
        """StaggeredBoundary.wall_faces, per edge cell and per corner."""
        config = _mixed_room()
        mesh = Mesh(config)
        assert mesh.cell_type[1, 2] == SOLID
        walls = StaggeredBoundary(mesh, config).wall_faces()
        assert walls["top"].cells.tolist() == [True, True, False, False, True, True]
        assert walls["top"].corners.tolist() == [
            True,
            True,
            False,
            False,
            False,
            True,
            True,
        ]
        assert walls["bottom"].cells.tolist() == [True] * 4 + [False] * 2
        assert walls["bottom"].corners.tolist() == [True] * 4 + [False] * 3
        assert walls["left"].cells.all() and walls["left"].corners.all()
        assert walls["right"].cells.tolist() == [False, False, True, True]
        assert walls["right"].corners.tolist() == [False, False, False, True, True]
        for edge in walls.values():
            assert not edge.cells.flags.writeable
            assert not edge.corners.flags.writeable

    def test_a_solid_edge_cell_has_no_wall_cell_but_bounds_a_wall_corner(
        self,
    ) -> None:
        config = _config(
            3.0,
            1.0,
            6,
            4,
            {},
            obstacles=[
                {
                    "name": "b",
                    "x_start": 1.0,
                    "x_end": 1.5,
                    "y_start": 0.0,
                    "y_end": 0.25,
                }
            ],
        )
        mesh = Mesh(config)
        assert mesh.cell_type[0, 2] == SOLID
        bottom = StaggeredBoundary(mesh, config).wall_faces()["bottom"]
        assert bottom.cells.tolist() == [True, True, False, True, True, True]
        assert bottom.corners.all()

    def test_velocity_faces_given_a_wall_function(self) -> None:
        mesh, _, walls = _built(_mixed_room())
        shape = (mesh.y.size, mesh.x.size)
        np.testing.assert_array_equal(walls.wall_faces["u"], _mask(shape, U_WALLS))
        np.testing.assert_array_equal(walls.wall_faces["v"], _mask(shape, V_WALLS))

    def test_the_stencil_reads_exactly_the_wall_faces_and_the_openings(self) -> None:
        """Against the predictor: what it reads beyond the walls is inlet and outlet faces."""
        config = _mixed_room()
        mesh, boundary, walls = _built(config)
        read = MomentumPredictor(mesh, config, boundary)._wall_read
        shape = (mesh.y.size, mesh.x.size)
        openings = {
            "u": _mask(shape, {(0, 4), (0, 5), (4, 2), (4, 3), (4, 4)}),
            "v": np.zeros(shape, dtype=bool),
        }
        for key in ("u", "v"):
            np.testing.assert_array_equal(
                read[key] & ~walls.wall_faces[key], openings[key]
            )
            assert not (walls.wall_faces[key] & ~read[key]).any()

    def test_wall_cells(self) -> None:
        mesh, _, walls = _built(_mixed_room())
        np.testing.assert_array_equal(
            walls.wall_cells, _mask(mesh.cell_type.shape, WALL_CELLS)
        )
        assert not walls.wall_cells.flags.writeable

    def test_the_conditions_hold_eps_and_give_production_in_the_wall_cells(
        self,
    ) -> None:
        config = _mixed_room()
        mesh, _, walls = _built(config)
        state = KEpsilonModel(mesh, config).initial(0.02, 0.01)
        c = walls.conditions(state, _random_faces(mesh, 3))
        expected = _mask(mesh.cell_type.shape, WALL_CELLS)
        np.testing.assert_array_equal(c.eps_held, expected)
        np.testing.assert_array_equal(c.production_given, expected)
        assert (c.eps_wall[expected] > 0.0).all()
        assert (c.eps_wall[~expected] == 0.0).all()


def _random_faces(mesh: Mesh, seed: int) -> FaceVelocities:
    """Faces of both signs, different in x and y; not divergence-free (not needed)."""
    rng = np.random.default_rng(seed)
    return FaceVelocities.copy_of(
        rng.uniform(-1.0, 2.0, size=u_shape(mesh)),
        rng.uniform(-0.3, 0.6, size=v_shape(mesh)),
    )


# ---------------------------------------------------------------------------
# The wall cells' values: one wall, the log layer, and the corner rule
# ---------------------------------------------------------------------------


def _one_wall(k: float, y_p: float, slip: float, c_mu: float) -> tuple[float, float]:
    """eps held and production for one wall, written out from the docstring."""
    u_k = c_mu**0.25 * math.sqrt(k)
    y_star = u_k * y_p / NU
    log_term = math.log(9.793 * max(y_star, Y_STAR_FLOOR))
    tau = u_k * 0.41 * slip / log_term
    return c_mu**0.75 * k**1.5 / (0.41 * y_p), tau * u_k / (0.41 * y_p)


def _box(variant: str = "standard") -> SimConfig:
    """Five by four cells of 0.5 by 0.3 m, every edge a wall at rest."""
    return _config(2.5, 1.2, 5, 4, {}, variant=variant)


@pytest.mark.unit
class TestWallCellValues:
    """ADR-012 B's eps and production in a wall cell; the corner rule (Notes)."""

    @pytest.mark.parametrize("variant", ["standard", "rng"])
    def test_one_wall_follows_the_formulas(self, variant: str) -> None:
        config = _box(variant)
        mesh, _, walls = _built(config)
        rng = np.random.default_rng(7)
        model = KEpsilonModel(mesh, config)
        state = model.state(
            rng.uniform(0.01, 0.2, size=(4, 5)), rng.uniform(0.01, 0.5, size=(4, 5))
        )
        faces = _random_faces(mesh, 11)
        c = walls.conditions(state, faces)
        c_mu = VARIANTS[variant].c_mu
        u_c = 0.5 * (faces.u[:, :-1] + faces.u[:, 1:])
        v_c = 0.5 * (faces.v[:-1, :] + faces.v[1:, :])
        # (0, 2): the bottom wall alone, 0.15 m below the centre.
        eps, production = _one_wall(state.k[0, 2], 0.15, abs(u_c[0, 2]), c_mu)
        assert c.eps_wall[0, 2] == pytest.approx(eps, rel=1e-14)
        assert c.production[0, 2] == pytest.approx(production, rel=1e-14)
        # (2, 4): the right wall alone, 0.25 m beside the centre.
        eps, production = _one_wall(state.k[2, 4], 0.25, abs(v_c[2, 4]), c_mu)
        assert c.eps_wall[2, 4] == pytest.approx(eps, rel=1e-14)
        assert c.production[2, 4] == pytest.approx(production, rel=1e-14)
        # (0, 0): the corner, the mean of its two walls.
        south = _one_wall(state.k[0, 0], 0.15, abs(u_c[0, 0]), c_mu)
        west = _one_wall(state.k[0, 0], 0.25, abs(v_c[0, 0]), c_mu)
        assert c.eps_wall[0, 0] == pytest.approx(0.5 * (south[0] + west[0]), rel=1e-14)
        assert c.production[0, 0] == pytest.approx(
            0.5 * (south[1] + west[1]), rel=1e-14
        )
        # (1, 2): no wall.
        assert not c.eps_held[1, 2] and c.production[1, 2] == 0.0

    def test_in_the_log_layer_production_equals_the_held_eps(self) -> None:
        """Where the slip is the log law's, P = tau_w du/dy = eps (local equilibrium)."""
        config = _box()
        mesh, _, walls = _built(config)
        c_mu = VARIANTS["standard"].c_mu
        k = np.full((4, 5), 0.05)
        u_k = c_mu**0.25 * math.sqrt(0.05)
        slip = u_k / KAPPA * math.log(E_WALL * u_k * 0.15 / NU)
        u = np.zeros(u_shape(mesh))
        u[0, :] = slip
        state = KEpsilonModel(mesh, config).state(k, np.full((4, 5), 0.02))
        c = walls.conditions(state, FaceVelocities.copy_of(u, np.zeros(v_shape(mesh))))
        assert u_k * 0.15 / NU > Y_STAR_FLOOR
        np.testing.assert_allclose(c.production[0, 1:4], c.eps_wall[0, 1:4], rtol=1e-14)

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_the_corner_rule_is_symmetric_under_reflection(self, axis: str) -> None:
        """A reflected room gives the reflected eps and production in every cell.

        The corner cells see two walls at different distances (0.25 and
        0.15 m) and different slips, so a rule that drops or favours one wall
        breaks the symmetry.
        """
        config = _box()
        mesh, _, walls = _built(config)
        rng = np.random.default_rng(5)
        model = KEpsilonModel(mesh, config)
        k = rng.uniform(0.01, 0.2, size=(4, 5))
        eps = rng.uniform(0.01, 0.5, size=(4, 5))
        u = rng.uniform(-1.0, 2.0, size=u_shape(mesh))
        v = rng.uniform(-0.3, 0.6, size=v_shape(mesh))
        if axis == "x":
            flip = (slice(None), slice(None, None, -1))
            mirrored = (-u[flip], v[flip])
        else:
            flip = (slice(None, None, -1), slice(None))
            mirrored = (u[flip], -v[flip])
        c = walls.conditions(model.state(k, eps), FaceVelocities.copy_of(u, v))
        c_r = walls.conditions(
            model.state(k[flip], eps[flip]), FaceVelocities.copy_of(*mirrored)
        )
        assert c.eps_held[0, 0] and c.eps_held[-1, -1]
        np.testing.assert_allclose(c_r.eps_wall, c.eps_wall[flip], rtol=1e-14)
        np.testing.assert_allclose(c_r.production, c.production[flip], rtol=1e-14)

    def test_a_moving_wall_counts_the_slip_against_its_own_speed(self) -> None:
        lid = {
            "lid": {
                "type": "velocity_inlet",
                "location": "top",
                "x_start": 0.0,
                "x_end": 2.5,
                "u_velocity": 1.5,
                "v_velocity": 0.0,
            }
        }
        config = _config(2.5, 1.2, 5, 4, lid)
        mesh, _, walls = _built(config)
        u = np.full(u_shape(mesh), 1.5)
        state = KEpsilonModel(mesh, config).initial(0.05, 0.02)
        c = walls.conditions(state, FaceVelocities.copy_of(u, np.zeros(v_shape(mesh))))
        # Moving with the lid: no slip, no production from the top wall in
        # the middle of the top row.
        assert c.production[-1, 2] == 0.0
        assert c.production[0, 2] > 0.0
        np.testing.assert_array_equal(c.tangential_top, np.full(6, 1.5))


# ---------------------------------------------------------------------------
# The inflow values and the initial state
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInflow:
    """k = 1.5 (I |u_n|)^2 and eps = k^(3/2) / l_e, no C_mu^(3/4) (ADR-012 C)."""

    def test_the_formulas(self) -> None:
        k, eps = inflow_values(0.05, -0.45, 0.2)
        assert k == pytest.approx(1.5 * (0.05 * 0.45) ** 2, rel=1e-15)
        assert eps == pytest.approx(k**1.5 / 0.2, rel=1e-15)

    def test_the_conditions_carry_them_on_the_inlet_faces(self) -> None:
        config = _mixed_room()
        mesh, _, walls = _built(config)
        model = KEpsilonModel(mesh, config)
        rng = np.random.default_rng(2)
        k = rng.uniform(0.01, 0.2, size=(4, 6))
        eps = rng.uniform(0.01, 0.5, size=(4, 6))
        state = model.state(k, eps)
        c = walls.conditions(state, _random_faces(mesh, 4))
        k_in = 1.5 * (0.05 * 1.0) ** 2
        eps_in = k_in**1.5 / 0.2
        np.testing.assert_allclose(c.inflow_k_v[-1, 2:4], k_in, rtol=1e-15)
        np.testing.assert_allclose(c.inflow_eps_v[-1, 2:4], eps_in, rtol=1e-15)
        # Every other domain face: the adjacent cell's value, so a pressure
        # outlet the flux turns inward through brings the cell's own.
        np.testing.assert_array_equal(c.inflow_k_u[:, -1], state.k[:, -1])
        np.testing.assert_array_equal(c.inflow_eps_u[:, 0], state.eps[:, 0])
        np.testing.assert_array_equal(c.inflow_k_v[0, :], state.k[0, :])
        top = np.r_[state.k[-1, :2], state.k[-1, 4:]]
        np.testing.assert_array_equal(
            np.r_[c.inflow_k_v[-1, :2], c.inflow_k_v[-1, 4:]], top
        )

    def test_the_initial_state_is_the_inflow_weighted_mean(self) -> None:
        """Two inlets that differ: k and eps weighted by each face's inflow."""
        boundaries = {
            "supply": {
                "type": "velocity_inlet",
                "location": "top",
                "x_start": 0.0,
                "x_end": 1.5,
                "velocity": 0.45,
                "turbulence_intensity": 0.05,
                "dissipation_length": 0.2,
            },
            "jet": {
                "type": "velocity_inlet",
                "location": "left",
                "y_start": 0.0,
                "y_end": 0.6,
                "velocity": 0.3,
                "turbulence_intensity": 0.1,
                "dissipation_length": 0.05,
            },
            "out": {
                "type": "pressure_outlet",
                "location": "right",
                "y_start": 0.0,
                "y_end": 1.2,
            },
        }
        _, _, walls = _built(_config(2.5, 1.2, 5, 4, boundaries))
        k1, e1 = 1.5 * (0.05 * 0.45) ** 2, (1.5 * (0.05 * 0.45) ** 2) ** 1.5 / 0.2
        k2, e2 = 1.5 * (0.1 * 0.3) ** 2, (1.5 * (0.1 * 0.3) ** 2) ** 1.5 / 0.05
        q1, q2 = 0.45 * 1.5, 0.3 * 0.6
        k, eps = walls.initial_values()
        assert k == pytest.approx((k1 * q1 + k2 * q2) / (q1 + q2), rel=1e-14)
        assert eps == pytest.approx((e1 * q1 + e2 * q2) / (q1 + q2), rel=1e-14)
        c_mu = VARIANTS["standard"].c_mu
        assert walls.largest_inlet_eddy_viscosity() == pytest.approx(
            max(c_mu * k1**2 / e1, c_mu * k2**2 / e2), rel=1e-14
        )

    def test_one_inlet_starts_at_its_own_values(self) -> None:
        _, _, walls = _built(_mixed_room())
        k_in = 1.5 * 0.05**2
        assert walls.initial_values() == pytest.approx((k_in, k_in**1.5 / 0.2))

    def test_no_inlet_that_admits_air_is_refused(self) -> None:
        _, _, walls = _built(_box())
        with pytest.raises(ValueError, match="no velocity inlet admits air"):
            walls.initial_values()
        with pytest.raises(ValueError, match="no velocity inlet admits air"):
            walls.largest_inlet_eddy_viscosity()

    def test_a_configuration_without_the_section_is_refused(self) -> None:
        raw_config = _box()
        mesh = Mesh(raw_config)
        object.__setattr__(raw_config, "turbulence", None)
        with pytest.raises(ValueError, match="turbulence section"):
            TurbulenceBoundary(mesh, raw_config, StaggeredBoundary(mesh, raw_config))
