"""Tests for the staggered momentum predictor with QUICK advection (REQ-S07, REQ-S09).

There is no analytical solution for a predictor alone, so the tests check
properties that hold exactly. Where floating point forbids ``==``, the
tolerance is stated with its reason: Lagrange weights sum to one only to
rounding, so a reproduced quadratic or an advected constant differs from the
exact value by a few ulps of the values involved.
"""

import numpy as np
import pytest
import yaml

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.momentum import (
    MomentumCoefficients,
    MomentumPredictor,
    _Orientation,
    quick_face_values,
)
from src.solver_staggered import StaggeredSolver
from src.staggered import allocate_fields
from tests.frozen34_reference import load_frozen_predictor
from validation.cases import CONFIG_DIR

STRETCHED = {"x": {"stretch_ratio": 1.15}, "y": {"stretch_ratio": 1.25}}
ROUNDING = 1e-12


def _inlet(location: str, u_velocity: float, v_velocity: float) -> dict:
    span = {"x_start": 0.0, "x_end": 2.0} if location in ("top", "bottom") else {}
    span = span or {"y_start": 0.0, "y_end": 1.0}
    return {
        "type": "velocity_inlet",
        "location": location,
        "u_velocity": u_velocity,
        "v_velocity": v_velocity,
        **span,
    }


CHANNEL = {
    "inlet": {
        "type": "velocity_inlet",
        "location": "left",
        "y_start": 0.0,
        "y_end": 1.0,
        "velocity": 0.3,
    },
    "outlet": {
        "type": "pressure_outlet",
        "location": "right",
        "y_start": 0.0,
        "y_end": 1.0,
    },
}
CAVITY = {"lid": _inlet("top", 1.0, 0.0)}


def _config(
    boundaries: dict,
    nx: int = 8,
    ny: int = 6,
    width: float = 2.0,
    height: float = 1.0,
    obstacles: list[dict] | None = None,
    mesh: dict | None = None,
    mu: float = 0.05,
    sweeps: int | None = None,
    solver: dict | None = None,
) -> SimConfig:
    raw = {
        "domain": {"width": width, "height": height, "nx": nx, "ny": ny},
        "fluid": {"density": 1.2, "viscosity": mu, "temperature": 293.0},
        "particles": {
            "density": 1000.0,
            "sizes": [0.1e-6],
            "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {"diameters": [0.1e-6], "efficiencies": [0.99999]},
        },
        "solver": {
            "dt": 0.01,
            "t_end": 1.0,
            "output_interval": 10,
            "convergence_tol": 1.0e-6,
            "max_simple_iter": 100,
            "alpha_velocity": 0.7,
            "alpha_pressure": 0.3,
            "max_pressure_iter": 5000,
            "pressure_rtol": 1.0e-8,
        },
        "boundaries": boundaries,
        "obstacles": obstacles or [],
        "sensors": [{"name": "center", "x": width / 2, "y": height / 2}],
        "thresholds": {"0.1e-6": 100.0},
    }
    if mesh is not None:
        raw["mesh"] = mesh
    if sweeps is not None:
        raw["solver"]["momentum_sweeps"] = sweeps
    raw["solver"].update(solver or {})
    return SimConfig.from_dict(raw)


def _build(config: SimConfig) -> tuple[Mesh, StaggeredBoundary, MomentumPredictor]:
    mesh = Mesh(config)
    boundary = StaggeredBoundary(mesh, config)
    return mesh, boundary, MomentumPredictor(mesh, config, boundary)


def _uniform_config(u0: float, v0: float, mesh: dict | None = None) -> SimConfig:
    """Every edge prescribes the same (u0, v0), so a uniform field is consistent."""
    edges = {e: _inlet(e, u0, v0) for e in ("bottom", "top", "left", "right")}
    return _config(edges, mesh=mesh)


def _stream_function_field(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """A discretely divergence-free field from a stream function, on any mesh."""

    def psi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x / 2.0) * np.sin(np.pi * y) + 0.3 * x * y

    x_f, y_f = np.meshgrid(mesh.x, mesh.y)
    u = (psi(x_f[1:, :], y_f[1:, :]) - psi(x_f[:-1, :], y_f[:-1, :])) / mesh.dy_cell[
        :, None
    ]
    v = (
        -(psi(x_f[:, 1:], y_f[:, 1:]) - psi(x_f[:, :-1], y_f[:, :-1]))
        / mesh.dx_cell[None, :]
    )
    return u, v


# ---------------------------------------------------------------------------
# QUICK face reconstruction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQuickFaceValues:
    """The quadratic interpolant and its boundary forms."""

    def test_uniform_weights_are_leonards(self) -> None:
        """6/8 C + 3/8 D - 1/8 U, both flow directions, on unit spacing."""
        nodes = np.arange(5.0)
        phi = np.array([[1.0, 10.0, 100.0, 1000.0, 10000.0]])
        left = np.array([1, 2])
        faces = np.array([1.5, 2.5])
        pos = quick_face_values(phi, nodes, left, faces, np.array([[True, True]]))
        neg = quick_face_values(phi, nodes, left, faces, np.array([[False, False]]))
        assert pos[0, 0] == 0.75 * 10.0 + 0.375 * 100.0 - 0.125 * 1.0
        assert pos[0, 1] == 0.75 * 100.0 + 0.375 * 1000.0 - 0.125 * 10.0
        assert neg[0, 0] == 0.75 * 100.0 + 0.375 * 10.0 - 0.125 * 1000.0
        assert neg[0, 1] == 0.75 * 1000.0 + 0.375 * 100.0 - 0.125 * 10000.0

    @pytest.mark.parametrize("positive", [True, False])
    def test_quadratic_is_reproduced_on_stretched_nodes(self, positive: bool) -> None:
        """A quadratic interpolant returns the quadratic's face value."""
        nodes = np.array([0.0, 0.7, 1.9, 2.4, 4.0, 4.3, 6.1])
        faces = np.array([0.3, 1.1, 2.2, 3.0, 4.2, 5.5])
        left = np.arange(6)

        def q(s: np.ndarray) -> np.ndarray:
            return 1.5 * s * s - 2.0 * s + 0.25

        phi = q(nodes)[None, :]
        got = quick_face_values(phi, nodes, left, faces, np.full((1, 6), positive))
        assert got == pytest.approx(q(faces)[None, :], rel=ROUNDING, abs=ROUNDING)

    def test_wall_form_uses_the_wall_value_at_its_location(self) -> None:
        """phi_f = phi_C + (phi_D - phi_wall) / 3 with the wall half a cell from C."""
        nodes = np.array([0.0, 0.5, 1.5, 2.5])
        phi = np.array([[7.0, 2.0, 5.0, 11.0]])
        got = quick_face_values(
            phi, nodes, np.array([1]), np.array([1.0]), np.array([[True]])
        )
        assert got[0, 0] == pytest.approx(2.0 + (5.0 - 7.0) / 3.0, rel=ROUNDING)

    def test_inflow_form_uses_the_boundary_and_two_interior_nodes(self) -> None:
        """3/8 phi_B + 6/8 phi_1 - 1/8 phi_2 at the face beside an inflow node."""
        nodes = np.array([0.0, 1.0, 2.0, 3.0])
        phi = np.array([[3.0, 5.0, 9.0, 40.0]])
        got = quick_face_values(
            phi, nodes, np.array([0]), np.array([0.5]), np.array([[True]])
        )
        assert got[0, 0] == 0.375 * 3.0 + 0.75 * 5.0 - 0.125 * 9.0

    def test_fewer_than_three_nodes_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="three nodes"):
            quick_face_values(
                np.ones((1, 2)),
                np.array([0.0, 1.0]),
                np.array([0]),
                np.array([0.5]),
                np.array([[True]]),
            )


# ---------------------------------------------------------------------------
# Assembled coefficients
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCoefficients:
    """Positivity, dominance and the two explicit sources."""

    @pytest.mark.parametrize(
        "mesh_spec", [None, STRETCHED], ids=["uniform", "stretched"]
    )
    def test_neighbour_coefficients_are_non_negative_and_diagonal_dominates(
        self, mesh_spec: dict | None
    ) -> None:
        """Upwind assembly keeps every a_nb >= 0; on a divergence-free field
        a_P >= sum(a_nb), the property deferred correction exists to keep."""
        mesh, _bc, mp = _build(_config(CAVITY, mesh=mesh_spec))
        u, v = _stream_function_field(mesh)
        for c in mp.momentum_coefficients(u, v):
            unknown = c.a_p > 0.0
            for a in (c.a_s_plus, c.a_s_minus, c.a_t_plus, c.a_t_minus):
                assert np.all(a >= 0.0)
            neighbours = c.a_s_plus + c.a_s_minus + c.a_t_plus + c.a_t_minus
            assert np.all(c.a_p[unknown] >= neighbours[unknown] * (1.0 - ROUNDING))

    @pytest.mark.parametrize(
        "mesh_spec", [None, STRETCHED], ids=["uniform", "stretched"]
    )
    def test_deferred_correction_vanishes_for_uniform_flow(
        self, mesh_spec: dict | None
    ) -> None:
        """QUICK and upwind agree on a constant, so the source is rounding only."""
        mesh, bc, mp = _build(_uniform_config(0.4, -0.2, mesh=mesh_spec))
        u, v, _p = allocate_fields(mesh)
        u.fill(0.4)
        v.fill(-0.2)
        bc.apply_normal_velocity(u, v)
        c_u, c_v = mp.momentum_coefficients(u, v)
        assert np.abs(c_u.b_deferred).max() < ROUNDING
        assert np.abs(c_v.b_deferred).max() < ROUNDING

    def test_deferred_correction_is_nonzero_where_the_profile_curves(self) -> None:
        """The test above would pass a scheme that never computed QUICK."""
        mesh, _bc, mp = _build(_config(CAVITY))
        u, v = _stream_function_field(mesh)
        c_u, _c_v = mp.momentum_coefficients(u, v)
        assert np.abs(c_u.b_deferred).max() > 1e-3

    @pytest.mark.parametrize(
        "mesh_spec", [None, STRETCHED], ids=["uniform", "stretched"]
    )
    def test_linear_profile_has_zero_diffusive_residual(
        self, mesh_spec: dict | None
    ) -> None:
        """With v = 0 and u linear in y, every u equation balances exactly.

        Advection contributes nothing (u uniform along x, no transverse
        flux) so the residual is the diffusion operator on a linear
        profile, including the one-sided wall gradient, which is exact
        only if the wall distance is the mesh's.
        """
        slope, offset = 0.8, 0.1
        top = offset + slope * 1.0
        edges = {
            "bottom": _inlet("bottom", offset, 0.0),
            "top": _inlet("top", top, 0.0),
        }
        mesh, bc, mp = _build(_config(edges, mesh=mesh_spec))
        u, v, _p = allocate_fields(mesh)
        profile = offset + slope * mesh.yc
        u[:] = profile[:, None]
        bc.apply_normal_velocity(u, v)
        # The side edges are walls whose normal value is zero; the predictor
        # reads boundary columns as given, so hold the profile there.
        u[:, 0] = profile
        u[:, -1] = profile
        c, _ = mp.momentum_coefficients(u, v)
        padded = np.pad(u, ((1, 1), (1, 1)))
        residual = (
            c.a_s_plus * padded[1:-1, 2:]
            + c.a_s_minus * padded[1:-1, :-2]
            + c.a_t_plus * padded[2:, 1:-1]
            + c.a_t_minus * padded[:-2, 1:-1]
            + c.b_boundary
            + c.b_deferred
            - c.a_p * u
        )
        scale = np.abs(c.a_p * u).max()
        assert np.abs(residual[c.a_p > 0]).max() < ROUNDING * scale

    def test_wall_contribution_uses_the_reported_wall_distance(self) -> None:
        """On a stretched mesh the top-row diagonal carries mu * dx_face / dy_face[ny]."""
        mesh, _bc, mp = _build(_config(CAVITY, mesh=STRETCHED))
        u, v, _p = allocate_fields(mesh)
        c, _ = mp.momentum_coefficients(u, v)
        i = 3
        expected = mp._mu * (
            mesh.dy_cell[-1] / mesh.dx_cell[i]
            + mesh.dy_cell[-1] / mesh.dx_cell[i - 1]
            + mesh.dx_face[i] / mesh.dy_face[-2]
            + mesh.dx_face[i] / mesh.dy_face[-1]
        )
        assert c.a_p[-1, i] == pytest.approx(expected, rel=ROUNDING)
        assert c.a_t_plus[-1, i] == 0.0

    def test_outlet_edge_carries_no_transverse_diffusion(self) -> None:
        """A zero-gradient top edge adds nothing to the diagonal of the top row."""
        top_outlet = {
            "out": {
                "type": "pressure_outlet",
                "location": "top",
                "x_start": 0.0,
                "x_end": 2.0,
            }
        }
        mesh, _bc, mp = _build(_config(top_outlet))
        u, v, _p = allocate_fields(mesh)
        c, _ = mp.momentum_coefficients(u, v)
        i = 3
        expected = mp._mu * (
            2.0 * mesh.dy_cell[-1] / mesh.dx_cell[i]
            + mesh.dx_face[i] / mesh.dy_face[-2]
        )
        assert c.a_p[-1, i] == pytest.approx(expected, rel=ROUNDING)
        assert c.b_boundary[-1, i] == 0.0


# ---------------------------------------------------------------------------
# The prediction and its contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPrediction:
    """u*, v* and the a_P contract with step 5."""

    @pytest.mark.parametrize(
        "mesh_spec", [None, STRETCHED], ids=["uniform", "stretched"]
    )
    def test_uniform_flow_is_a_fixed_point(self, mesh_spec: dict | None) -> None:
        """Consistent uniform velocity and zero pressure predict themselves."""
        mesh, bc, mp = _build(_uniform_config(0.4, -0.2, mesh=mesh_spec))
        u, v, p = allocate_fields(mesh)
        u.fill(0.4)
        v.fill(-0.2)
        bc.apply_normal_velocity(u, v)
        out = mp.predict(u, v, p)
        assert out.u_star == pytest.approx(u, rel=ROUNDING)
        assert out.v_star == pytest.approx(v, rel=ROUNDING)

    def test_shapes_masks_and_untouched_boundary_entries(self) -> None:
        config = _config(CHANNEL)
        mesh, bc, mp = _build(config)
        u, v, p = allocate_fields(mesh)
        bc.apply_normal_velocity(u, v)
        u[:, -1] = 0.3
        out = mp.predict(u, v, p)
        nx, ny = config.nx, config.ny
        assert out.u_star.shape == (ny, nx + 1) and out.a_p_u.shape == (ny, nx + 1)
        assert out.v_star.shape == (ny + 1, nx) and out.a_p_v.shape == (ny + 1, nx)
        for arr in (out.u_star, out.v_star, out.a_p_u, out.a_p_v):
            assert arr.dtype == np.float64 and arr.flags["C_CONTIGUOUS"]
        assert np.array_equal(out.u_star[:, 0], u[:, 0])
        assert np.array_equal(out.u_star[:, -1], u[:, -1])
        assert np.array_equal(out.v_star[0, :], v[0, :])
        assert np.array_equal(out.v_star[-1, :], v[-1, :])
        expected_u = np.zeros((ny, nx + 1), dtype=bool)
        expected_u[:, 1:-1] = True
        expected_v = np.zeros((ny + 1, nx), dtype=bool)
        expected_v[1:-1, :] = True
        assert np.array_equal(out.a_p_u > 0.0, expected_u)
        assert np.array_equal(out.a_p_v > 0.0, expected_v)
        assert np.all(out.a_p_u[expected_u] > 0.0)

    def test_pressure_drop_drives_flow_from_rest(self) -> None:
        mesh, bc, mp = _build(_config(CAVITY))
        u, v, p = allocate_fields(mesh)
        bc.apply_normal_velocity(u, v)
        p[:] = -mesh.xc[None, :]
        out = mp.predict(u, v, p)
        assert np.all(out.u_star[:, 1:-1] > 0.0)
        assert np.all(out.v_star == 0.0)

    def test_solid_faces_are_zero_and_not_unknowns(self) -> None:
        block = {
            "name": "block",
            "x_start": 0.7,
            "x_end": 1.3,
            "y_start": 0.3,
            "y_end": 0.7,
        }
        mesh, bc, mp = _build(_config(CAVITY, obstacles=[block]))
        solid = mesh.cell_type == SOLID
        assert solid.any()
        u, v, p = allocate_fields(mesh)
        u.fill(0.5)
        v.fill(0.5)
        bc.apply_normal_velocity(u, v)
        p[:] = -mesh.xc[None, :]
        out = mp.predict(u, v, p)
        u_solid = np.zeros_like(u, dtype=bool)
        u_solid[:, :-1] |= solid
        u_solid[:, 1:] |= solid
        v_solid = np.zeros_like(v, dtype=bool)
        v_solid[:-1, :] |= solid
        v_solid[1:, :] |= solid
        assert np.all(out.u_star[u_solid] == 0.0)
        assert np.all(out.v_star[v_solid] == 0.0)
        assert np.all(out.a_p_u[u_solid] == 0.0)
        assert np.all(out.a_p_v[v_solid] == 0.0)

    def test_v_equation_is_the_u_equation_rotated(self) -> None:
        """A channel flowing in +y predicts v* equal to the +x channel's u* transposed."""
        along_x = _config(CHANNEL, nx=8, ny=6, width=2.0, height=1.0, mesh=STRETCHED)
        rotated = {
            "inlet": {
                "type": "velocity_inlet",
                "location": "bottom",
                "x_start": 0.0,
                "x_end": 1.0,
                "velocity": 0.3,
            },
            "outlet": {
                "type": "pressure_outlet",
                "location": "top",
                "x_start": 0.0,
                "x_end": 1.0,
            },
        }
        along_y = _config(
            rotated,
            nx=6,
            ny=8,
            width=1.0,
            height=2.0,
            mesh={"x": STRETCHED["y"], "y": STRETCHED["x"]},
        )
        mesh_x, bc_x, mp_x = _build(along_x)
        mesh_y, bc_y, mp_y = _build(along_y)
        rng = np.random.default_rng(7)
        u, v, p = allocate_fields(mesh_x)
        u[:] = rng.random(u.shape)
        v[:] = rng.random(v.shape)
        p[:] = rng.random(p.shape)
        bc_x.apply_normal_velocity(u, v)
        ur, vr, pr = allocate_fields(mesh_y)
        vr[:] = u.T
        ur[:] = v.T
        pr[:] = p.T
        bc_y.apply_normal_velocity(ur, vr)
        out_x = mp_x.predict(u, v, p)
        out_y = mp_y.predict(ur, vr, pr)
        assert out_y.v_star == pytest.approx(out_x.u_star.T, rel=ROUNDING)
        assert out_y.u_star == pytest.approx(out_x.v_star.T, rel=ROUNDING)
        assert out_y.a_p_v == pytest.approx(out_x.a_p_u.T, rel=ROUNDING)

    def test_wrong_shapes_are_rejected(self) -> None:
        config = _config(CHANNEL)
        mesh, _bc, mp = _build(config)
        u, v, p = allocate_fields(mesh)
        with pytest.raises(ValueError, match="staggered shapes"):
            mp.predict(v, u, p)
        with pytest.raises(ValueError, match="expected p"):
            mp.predict(u, v, p.T.copy())

    def test_too_small_a_mesh_is_rejected(self) -> None:
        config = _config(CAVITY, nx=1, ny=4)
        mesh = Mesh(config)
        with pytest.raises(ValueError, match="2 cells"):
            MomentumPredictor(mesh, config, StaggeredBoundary(mesh, config))


# ---------------------------------------------------------------------------
# The viscosity field, the wall viscosity and the sweep count (ECR-002 step 4)
# ---------------------------------------------------------------------------

# A block standing on the floor of the 2 m by 1 m box, so the face rule meets
# SOLID cells across row boundaries and across columns.
FLOOR_BLOCK = {
    "name": "block",
    "x_start": 0.75,
    "x_end": 1.25,
    "y_start": 0.0,
    "y_end": 0.5,
}


def _random_field(mesh: Mesh, rng: np.random.Generator, mu: float) -> np.ndarray:
    """A positive viscosity per cell spread over four decades above mu."""
    return mu * (1.0 + 10.0 ** rng.uniform(-2.0, 2.0, mesh.cell_type.shape))


def _random_state(
    mesh: Mesh, bc: StaggeredBoundary, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v, p = allocate_fields(mesh)
    u[:] = rng.normal(0.0, 0.5, u.shape)
    v[:] = rng.normal(0.0, 0.5, v.shape)
    p[:] = rng.normal(0.0, 0.1, p.shape)
    bc.apply_normal_velocity(u, v)
    return u, v, p


def _hand_corner(mu: np.ndarray, o: _Orientation, j: int, i: int) -> float:
    """ADR-012 D's rule at one transverse face, written out from the coordinates."""
    nt = mu.shape[0]
    columns = []
    for c, width in (
        (i - 1, o.s_faces[i] - o.s_centers[i - 1]),
        (i, o.s_centers[i] - o.s_faces[i]),
    ):
        if j == 0:
            value = mu[0, c]
        elif j == nt:
            value = mu[nt - 1, c]
        else:
            low, high = mu[j - 1, c], mu[j, c]
            if o.solid[j - 1, c] and not o.solid[j, c]:
                low = high
            if o.solid[j, c] and not o.solid[j - 1, c]:
                high = low
            d_low = o.t_faces[j] - o.t_centers[j - 1]
            d_high = o.t_centers[j] - o.t_faces[j]
            value = (d_low + d_high) / (d_low / low + d_high / high)
        columns.append((width, value))
    (w_0, m_0), (w_1, m_1) = columns
    return float((w_0 * m_0 + w_1 * m_1) / (w_0 + w_1))


def _beyond_quick_reach(solid: np.ndarray) -> np.ndarray:
    """Faces of one component's frame [nt, ns+1] two nodes or more from a SOLID face.

    QUICK's stencil reaches two nodes along either axis, so the boundary
    form at an obstacle changes the deferred correction only nearer than
    that.
    """
    nt, ns = solid.shape
    near = np.zeros((nt, ns + 1), dtype=bool)
    near[:, :-1] |= solid
    near[:, 1:] |= solid
    grown = near.copy()
    for shift in (1, 2):
        grown[shift:, :] |= near[:-shift, :]
        grown[:-shift, :] |= near[shift:, :]
        grown[:, shift:] |= near[:, :-shift]
        grown[:, :-shift] |= near[:, shift:]
    far = ~grown
    far[:, 0] = far[:, -1] = False
    return far


def _product_room(sweeps: int, obstacles: bool = True) -> SimConfig:
    """The committed product configuration on the 40x15 grid of the ladder."""
    raw = yaml.safe_load(
        (CONFIG_DIR / "clean_room_default.yaml").read_text(encoding="utf-8")
    )
    raw["domain"]["nx"], raw["domain"]["ny"] = 40, 15
    if not obstacles:
        raw["obstacles"] = []
    raw["solver"]["alpha_velocity"] = 0.5
    raw["solver"]["momentum_sweeps"] = sweeps
    return SimConfig.from_dict(raw)


@pytest.mark.unit
class TestViscosityFaceRule:
    """ADR-012 D's face rule: harmonic across the row boundary, width mean across columns."""

    def test_a_uniform_mesh_face_by_hand(self) -> None:
        """Cells 1 and 3 in series give 1.5; beside cells of 2, the width mean is 1.75."""
        mesh, _bc, mp = _build(_config(CAVITY))
        mu = np.full(mesh.cell_type.shape, 2.0)
        mu[2, 3], mu[3, 3] = 1.0, 3.0
        corner = mp._corner_viscosity(mu, mp._for_u)
        assert corner[3, 4] == pytest.approx(1.75, rel=ROUNDING)
        assert corner[3, 3] == pytest.approx(1.75, rel=ROUNDING)
        assert corner[3, 5] == 2.0

    @pytest.mark.parametrize("component", ["u", "v"])
    def test_every_face_on_a_stretched_mesh_with_solid_cells(
        self, component: str
    ) -> None:
        """Both axes, every interior column, edge rows and SOLID neighbours included."""
        mesh, _bc, mp = _build(_config(CAVITY, obstacles=[FLOOR_BLOCK], mesh=STRETCHED))
        assert (mesh.cell_type == SOLID).any()
        mu = _random_field(mesh, np.random.default_rng(431), 0.05)
        o = mp._for_u if component == "u" else mp._for_v
        cells = mu if component == "u" else np.ascontiguousarray(mu.T)
        corner = mp._corner_viscosity(cells, o)
        nt, ns = cells.shape
        beside_solid = 0
        for j in range(nt + 1):
            for i in range(1, ns):
                expected = _hand_corner(cells, o, j, i)
                assert corner[j, i] == pytest.approx(expected, rel=ROUNDING), (j, i)
                if 0 < j < nt and (o.solid[j - 1, i] != o.solid[j, i]):
                    beside_solid += 1
        assert beside_solid > 0

    def test_a_solid_cell_contributes_its_neighbours_value(self) -> None:
        """Across an obstacle top the face carries the wall cell's value, no SOLID value."""
        mesh, _bc, mp = _build(_config(CAVITY, obstacles=[FLOOR_BLOCK]))
        solid = mesh.cell_type == SOLID
        mu = np.full(mesh.cell_type.shape, 0.05)
        mu[solid] = 1.0e6
        top = int(np.flatnonzero(solid.any(axis=1)).max())
        columns = np.flatnonzero(solid[top])
        mu[top + 1, columns] = 0.2
        corner = mp._corner_viscosity(mu, mp._for_u)
        inner = columns[1:]
        assert np.all(corner[top + 1, inner] == 0.2)

    def test_a_uniform_field_returns_itself_exactly(self) -> None:
        """Equal inputs return the input to the bit at every face, both means."""
        mesh, _bc, mp = _build(_config(CAVITY, obstacles=[FLOOR_BLOCK], mesh=STRETCHED))
        value = 0.0371
        for o, shape in (
            (mp._for_u, mesh.cell_type.shape),
            (mp._for_v, mesh.cell_type.shape[::-1]),
        ):
            cells = np.full(shape, value)
            assert np.all(mp._corner_viscosity(cells, o) == value)
            assert np.all(mp._face_viscosity(cells, o) == value)

    def test_the_assembly_reads_the_rule(self) -> None:
        """At rest the transverse coefficient is the corner value times width over distance."""
        mesh, _bc, mp = _build(_config(CAVITY, obstacles=[FLOOR_BLOCK], mesh=STRETCHED))
        mu = mp._check_mu_eff(_random_field(mesh, np.random.default_rng(432), 0.05))
        u, v, _p = allocate_fields(mesh)
        o = mp._for_u
        c = mp._assemble(u, v, o, mu_cells=mu)
        corner = mp._corner_viscosity(mu, o)
        unknown = c.a_p > 0.0
        nt = mu.shape[0]
        checked = 0
        for j in range(nt - 1):
            for i in range(1, mu.shape[1]):
                if unknown[j, i] and unknown[j + 1, i]:
                    expected = corner[j + 1, i] * o.ds_face[i] / o.dt_face[j + 1]
                    assert c.a_t_plus[j, i] == pytest.approx(expected, rel=ROUNDING)
                    checked += 1
        assert checked > 0


@pytest.mark.unit
class TestStressSource:
    """Form b of the stress terms a varying viscosity adds (ADR-012 D)."""

    def test_zero_for_any_uniform_field(self) -> None:
        """A uniform viscosity on a random, divergent velocity field gives exact zeros."""
        mesh, bc, mp = _build(_config(CAVITY, obstacles=[FLOOR_BLOCK], mesh=STRETCHED))
        u, v, _p = _random_state(mesh, bc, np.random.default_rng(433))
        for value in (0.05, 3.7e-4, 12.0):
            for comp, o in (("u", mp._for_u), ("v", mp._for_v)):
                cells = np.full(o.solid.shape, value)
                corner = mp._corner_viscosity(cells, o)
                a, b = (u, v) if comp == "u" else (v.T, u.T)
                assert np.all(mp._stress_source(a, b, o, cells, corner) == 0.0)

    def test_equal_to_the_probes_on_a_varying_field(self) -> None:
        """Bitwise frozen34.py's form b, both components, a SOLID-adjacent spread."""
        frozen = load_frozen_predictor()
        config = _product_room(1)
        mesh, bc, mp = _build(config)
        rng = np.random.default_rng(434)
        u, v, _p = _random_state(mesh, bc, rng)
        mu_t = _random_field(mesh, rng, config.mu) - config.mu
        probe = frozen(mesh, config, bc, mu_t)
        cells = config.mu + mu_t
        for comp, o in (("u", mp._for_u), ("v", mp._for_v)):
            field = cells if comp == "u" else np.ascontiguousarray(cells.T)
            a, b = (u, v) if comp == "u" else (v.T, u.T)
            corner = mp._corner_viscosity(field, o)
            ours = mp._stress_source(a, b, o, field, corner)
            theirs, _pair = probe.stress_source(a, b, o, field, corner)
            assert np.array_equal(ours, theirs), comp
            assert np.any(ours != 0.0)

    def test_linear_viscosity_on_a_linear_velocity_by_hand(self) -> None:
        """mu = mu0 + rho (1 + x), u = x, v = 0: d/dx(mu du/dx) = rho, so rho per unit volume."""
        config = _config(CAVITY, mesh=STRETCHED)
        mesh, _bc, mp = _build(config)
        mu = config.mu + config.rho * (1.0 + np.tile(mesh.xc, (mesh.yc.size, 1)))
        u = np.tile(mesh.x, (mesh.yc.size, 1))
        v = np.zeros((mesh.y.size, mesh.xc.size))
        o = mp._for_u
        source = mp._stress_source(u, v, o, mu, mp._corner_viscosity(mu, o))
        volume = o.dt_cell[:, None] * o.ds_face[None, 1:-1]
        assert source == pytest.approx(config.rho * volume, rel=1e-12)


@pytest.mark.integration
class TestFieldPathAgainstTheProbe:
    """The field path is frozen34.py's FrozenPredictor (form b, stress on), to the bit.

    The probe kept the obstacle stencil the scalar path had then (a SOLID
    face a whole cell away, and QUICK's far node at the SOLID face's stored
    zero). Since the obstacle stencil was built, the two agree to the bit in
    a room without obstacles, and in the product room everywhere but at the
    unknowns beside an obstacle face, where the implicit coefficients differ
    exactly by the stencil, and within QUICK's reach of a SOLID face, where
    the deferred correction takes the boundary form (TestObstacleWallStencil).
    """

    @pytest.mark.parametrize("sweeps", [1, 10])
    def test_three_random_states_of_the_room_without_obstacles(
        self, sweeps: int
    ) -> None:
        """u*, v* and both diagonals equal at every face, one sweep and ten."""
        frozen = load_frozen_predictor()
        config = _product_room(sweeps, obstacles=False)
        mesh, bc, mp = _build(config)
        assert config.momentum_sweeps == sweeps
        assert not (mesh.cell_type == SOLID).any()
        rng = np.random.default_rng(435 + sweeps)
        for _state in range(3):
            u, v, p = _random_state(mesh, bc, rng)
            mu_t = _random_field(mesh, rng, config.mu) - config.mu
            ours = mp.predict(u, v, p, mu_eff=config.mu + mu_t)
            theirs = frozen(mesh, config, bc, mu_t, sweeps=sweeps).predict(u, v, p)
            for name in ("u_star", "v_star", "a_p_u", "a_p_v"):
                assert np.array_equal(getattr(ours, name), getattr(theirs, name)), name

    def test_product_room_coefficients_differ_only_beside_obstacle_faces(
        self,
    ) -> None:
        """Three random states, both components: the stencil is the only difference."""
        frozen = load_frozen_predictor()
        config = _product_room(1)
        mesh, bc, mp = _build(config)
        rng = np.random.default_rng(438)
        for _state in range(3):
            u, v, _p = _random_state(mesh, bc, rng)
            mu_t = _random_field(mesh, rng, config.mu) - config.mu
            probe = frozen(mesh, config, bc, mu_t)
            cells = config.mu + mu_t
            pairs = (("u", mp._for_u, probe._for_u), ("v", mp._for_v, probe._for_v))
            for comp, o, o_probe in pairs:
                field = cells if comp == "u" else np.ascontiguousarray(cells.T)
                a, b = (u, v) if comp == "u" else (v.T, u.T)
                ours = mp._assemble(a, b, o, mu_cells=field)
                theirs = probe._assemble_field(a, b, o_probe)
                beside = np.zeros_like(ours.a_p, dtype=bool)
                beside[:, 1:-1] = o.walls.north | o.walls.south
                assert beside.any()
                for name in ("a_s_plus", "a_s_minus", "b_boundary"):
                    assert np.array_equal(getattr(ours, name), getattr(theirs, name))
                far = _beyond_quick_reach(o.solid)
                assert far.sum() >= 100
                assert np.array_equal(ours.b_deferred[far], theirs.b_deferred[far])
                assert np.array_equal(ours.a_p[~beside], theirs.a_p[~beside])
                assert np.all(ours.a_p[beside] > theirs.a_p[beside])
                north = np.zeros_like(beside)
                north[:, 1:-1] = o.walls.north
                south = np.zeros_like(beside)
                south[:, 1:-1] = o.walls.south
                assert np.array_equal(ours.a_t_plus[~north], theirs.a_t_plus[~north])
                assert np.array_equal(ours.a_t_minus[~south], theirs.a_t_minus[~south])
                assert np.all(ours.a_t_plus[north] == 0.0)
                assert np.all(ours.a_t_minus[south] == 0.0)

    def test_ten_sweeps_on_the_same_coefficients(self) -> None:
        """_sweep_n is the probe's on the product room's own coefficients."""
        frozen = load_frozen_predictor()
        config = _product_room(10)
        mesh, bc, mp = _build(config)
        u, v, p = _random_state(mesh, bc, np.random.default_rng(439))
        mu_t = _random_field(mesh, np.random.default_rng(440), config.mu) - config.mu
        probe = frozen(mesh, config, bc, mu_t, sweeps=10)
        c = mp._assemble(u, v, mp._for_u, mu_cells=config.mu + mu_t)
        b_p = mp._pressure_source(p, mp._for_u)
        assert np.array_equal(mp._sweep_n(u, c, b_p), probe._sweep_n(u, c, b_p))


@pytest.mark.unit
class TestFieldPathLimits:
    """The field path's two exact limits, and that SOLID cells are not read."""

    def test_a_uniform_field_equal_to_air_is_the_scalar_path_to_the_byte(self) -> None:
        """Coefficients and the prediction, on a stretched mesh with an obstacle."""
        config = _config(CAVITY, obstacles=[FLOOR_BLOCK], mesh=STRETCHED)
        mesh, bc, mp = _build(config)
        u, v, p = _random_state(mesh, bc, np.random.default_rng(436))
        air = np.full(mesh.cell_type.shape, config.mu)
        scalar = mp._assemble(u, v, mp._for_u)
        field = mp._assemble(u, v, mp._for_u, mu_cells=mp._check_mu_eff(air))
        for name in ("a_p", "a_s_plus", "a_s_minus", "a_t_plus", "a_t_minus"):
            assert getattr(scalar, name).tobytes() == getattr(field, name).tobytes()
        assert np.array_equal(scalar.b_deferred, field.b_deferred)
        a, b = mp.predict(u, v, p), mp.predict(u, v, p, mu_eff=air)
        assert a.u_star.tobytes() == b.u_star.tobytes()
        assert a.v_star.tobytes() == b.v_star.tobytes()
        assert a.a_p_u.tobytes() == b.a_p_u.tobytes()
        assert a.a_p_v.tobytes() == b.a_p_v.tobytes()

    def test_solid_cells_are_not_read(self) -> None:
        """NaN in every SOLID cell predicts what any finite value there does."""
        config = _config(CAVITY, obstacles=[FLOOR_BLOCK], mesh=STRETCHED)
        mesh, bc, mp = _build(config)
        rng = np.random.default_rng(437)
        u, v, p = _random_state(mesh, bc, rng)
        mu = _random_field(mesh, rng, config.mu)
        solid = mesh.cell_type == SOLID
        holes = mu.copy()
        holes[solid] = np.nan
        a, b = mp.predict(u, v, p, mu_eff=mu), mp.predict(u, v, p, mu_eff=holes)
        assert np.array_equal(a.u_star, b.u_star)
        assert np.array_equal(a.v_star, b.v_star)

    @pytest.mark.parametrize(
        ("bad", "error", "match"),
        [
            ("list", TypeError, "float64 ndarray"),
            ("int", TypeError, "float64 ndarray"),
            ("shape", ValueError, "shape"),
            ("nan", ValueError, "finite"),
            ("inf", ValueError, "finite"),
            ("zero", ValueError, "positive"),
            ("negative", ValueError, "positive"),
        ],
    )
    def test_a_malformed_field_is_refused(
        self, bad: str, error: type[Exception], match: str
    ) -> None:
        """Type, shape, and a non-finite or non-positive value in a non-SOLID cell."""
        config = _config(CAVITY, obstacles=[FLOOR_BLOCK])
        mesh, _bc, mp = _build(config)
        u, v, p = allocate_fields(mesh)
        good = np.full(mesh.cell_type.shape, config.mu)
        field: object = {
            "list": good.tolist(),
            "int": good.astype(np.int64),
            "shape": good[:, :-1],
            "nan": np.where(
                np.arange(good.size).reshape(good.shape) == 0, np.nan, good
            ),
            "inf": np.where(
                np.arange(good.size).reshape(good.shape) == 0, np.inf, good
            ),
            "zero": np.where(np.arange(good.size).reshape(good.shape) == 0, 0.0, good),
            "negative": -good,
        }[bad]
        with pytest.raises(error, match=match):
            mp.predict(u, v, p, mu_eff=field)  # type: ignore[arg-type]


@pytest.mark.unit
class TestWallViscosity:
    """The wall viscosity hook of ADR-012 B: one value per wall face, None today."""

    @staticmethod
    def _walls(mesh: Mesh, value: float) -> dict[str, np.ndarray]:
        shape = (mesh.yc.size + 1, mesh.xc.size + 1)
        return {"u": np.full(shape, value), "v": np.full(shape, value)}

    def test_air_at_every_wall_face_is_the_path_without_it(self) -> None:
        """With and without a field, wall_mu equal to the stencil's own value changes no bit."""
        config = _config(CAVITY, mesh=STRETCHED)
        mesh, bc, mp = _build(config)
        u, v, p = _random_state(mesh, bc, np.random.default_rng(438))
        a = mp.predict(u, v, p)
        b = mp.predict(u, v, p, wall_mu=self._walls(mesh, config.mu))
        assert a.u_star.tobytes() == b.u_star.tobytes()
        assert a.v_star.tobytes() == b.v_star.tobytes()
        air = np.full(mesh.cell_type.shape, config.mu)
        c = mp.predict(u, v, p, mu_eff=air, wall_mu=self._walls(mesh, config.mu))
        assert a.u_star.tobytes() == c.u_star.tobytes()

    def test_it_replaces_the_wall_rows_conductance(self) -> None:
        """At rest, the floor row's diagonal moves by (mu_w - mu) dx_face / wall distance."""
        config = _config(CAVITY, mesh=STRETCHED)
        mesh, _bc, mp = _build(config)
        u, v, p = allocate_fields(mesh)
        walls = self._walls(mesh, config.mu)
        walls["u"][0, :] = 7.0 * config.mu
        walls["v"][:, -1] = 3.0 * config.mu
        a = mp.predict(u, v, p)
        b = mp.predict(u, v, p, wall_mu=walls)
        i = 3
        expected = 6.0 * config.mu * mesh.dx_face[i] / mesh.dy_face[0]
        assert b.a_p_u[0, i] - a.a_p_u[0, i] == pytest.approx(expected, rel=1e-12)
        assert np.array_equal(a.a_p_u[1:, :], b.a_p_u[1:, :])
        j = 2
        expected_v = 2.0 * config.mu * mesh.dy_face[j] / mesh.dx_face[-1]
        assert b.a_p_v[j, -1] - a.a_p_v[j, -1] == pytest.approx(expected_v, rel=1e-12)

    def test_it_replaces_the_fields_edge_value(self) -> None:
        """With a field, the wall row reads wall_mu in place of the wall cells' mean."""
        config = _config(CAVITY, mesh=STRETCHED)
        mesh, _bc, mp = _build(config)
        u, v, p = allocate_fields(mesh)
        mu = _random_field(mesh, np.random.default_rng(439), config.mu)
        walls = self._walls(mesh, config.mu)
        a = mp.predict(u, v, p, mu_eff=mu)
        b = mp.predict(u, v, p, mu_eff=mu, wall_mu=walls)
        corner = mp._corner_viscosity(mu, mp._for_u)
        i = 3
        expected = (config.mu - corner[0, i]) * mesh.dx_face[i] / mesh.dy_face[0]
        assert b.a_p_u[0, i] - a.a_p_u[0, i] == pytest.approx(expected, rel=1e-9)

    def test_faces_that_are_not_wall_faces_are_not_read(self) -> None:
        """NaN everywhere but at the faces the wall stencil crosses is accepted."""
        config = _config(CAVITY)
        mesh, bc, mp = _build(config)
        u, v, p = _random_state(mesh, bc, np.random.default_rng(440))
        walls = self._walls(mesh, np.nan)
        walls["u"][0, 1:-1] = walls["u"][-1, 1:-1] = config.mu
        walls["v"][1:-1, 0] = walls["v"][1:-1, -1] = config.mu
        a = mp.predict(u, v, p)
        b = mp.predict(u, v, p, wall_mu=walls)
        assert a.u_star.tobytes() == b.u_star.tobytes()
        assert a.v_star.tobytes() == b.v_star.tobytes()

    @pytest.mark.parametrize(
        ("bad", "error", "match"),
        [
            ("tuple", TypeError, "dict"),
            ("keys", ValueError, "keys 'u' and 'v'"),
            ("extra", ValueError, "keys 'u' and 'v'"),
            ("list", TypeError, "float64 ndarray"),
            ("int", TypeError, "float64 ndarray"),
            ("shape", ValueError, "shape"),
            ("nan_floor", ValueError, "finite"),
            ("zero_floor", ValueError, "positive"),
            ("negative_side", ValueError, "positive"),
        ],
    )
    def test_a_malformed_hook_is_refused(
        self, bad: str, error: type[Exception], match: str
    ) -> None:
        """Container, keys, dtype, shape, and a bad value at a wall face are each refused."""
        config = _config(CAVITY)
        mesh, _bc, mp = _build(config)
        u, v, p = allocate_fields(mesh)
        walls = self._walls(mesh, config.mu)
        hook: object = walls
        if bad == "tuple":
            hook = (walls["u"], walls["v"])
        elif bad == "keys":
            hook = {"u": walls["u"]}
        elif bad == "extra":
            hook = {**walls, "w": walls["u"]}
        elif bad == "list":
            walls["u"] = walls["u"].tolist()  # type: ignore[assignment]
        elif bad == "int":
            walls["v"] = walls["v"].astype(np.int64)
        elif bad == "shape":
            walls["v"] = walls["v"][:-1, :]
        elif bad == "nan_floor":
            walls["u"][0, 3] = np.nan
        elif bad == "zero_floor":
            walls["u"][0, 3] = 0.0
        else:
            walls["v"][2, -1] = -config.mu
        with pytest.raises(error, match=match):
            mp.predict(u, v, p, wall_mu=hook)  # type: ignore[arg-type]


@pytest.mark.unit
class TestMomentumSweeps:
    """solver.momentum_sweeps Jacobi sweeps per outer iteration on one assembly."""

    def test_one_sweep_is_the_committed_sweep_to_the_byte(self) -> None:
        """_sweep_n at one sweep returns _sweep's array, both components."""
        config = _config(CAVITY, obstacles=[FLOOR_BLOCK], mesh=STRETCHED)
        mesh, bc, mp = _build(config)
        assert config.momentum_sweeps == 1
        u, v, p = _random_state(mesh, bc, np.random.default_rng(441))
        c = mp._assemble(u, v, mp._for_u)
        b_p = mp._pressure_source(p, mp._for_u)
        assert mp._sweep_n(u, c, b_p).tobytes() == mp._sweep(u, c, b_p).tobytes()

    def test_three_sweeps_iterate_the_neighbours_with_the_sources_held(self) -> None:
        """Three sweeps equal three Jacobi steps written out face by face."""
        raw_sweeps = 3
        base = _config(
            CAVITY, obstacles=[FLOOR_BLOCK], mesh=STRETCHED, sweeps=raw_sweeps
        )
        mesh, bc, mp = _build(base)
        u, v, p = _random_state(mesh, bc, np.random.default_rng(442))
        c = mp._assemble(u, v, mp._for_u)
        b_p = mp._pressure_source(p, mp._for_u)
        alpha = base.alpha_velocity
        b = c.b_boundary + c.b_deferred + b_p + (1.0 - alpha) / alpha * c.a_p * u
        current = u.copy()
        nt, ns1 = u.shape
        for _ in range(raw_sweeps):
            following = u.copy()
            for j in range(nt):
                for i in range(1, ns1 - 1):
                    if c.a_p[j, i] <= 0.0:
                        following[j, i] = 0.0
                        continue
                    total = b[j, i]
                    total += c.a_s_plus[j, i] * current[j, i + 1]
                    total += c.a_s_minus[j, i] * current[j, i - 1]
                    if j + 1 < nt:
                        total += c.a_t_plus[j, i] * current[j + 1, i]
                    if j > 0:
                        total += c.a_t_minus[j, i] * current[j - 1, i]
                    following[j, i] = total / (c.a_p[j, i] / alpha)
            current = following
        swept = mp._sweep_n(u, c, b_p)
        assert swept == pytest.approx(current, rel=1e-12, abs=1e-14)
        assert not np.allclose(swept, mp._sweep(u, c, b_p))

    def test_the_predictor_reads_the_count_from_the_configuration(self) -> None:
        """predict at three sweeps differs from one and equals three sweeps of _sweep_n."""
        base = _config(CAVITY, mesh=STRETCHED)
        mesh, bc, one = _build(base)
        three = MomentumPredictor(mesh, _config(CAVITY, mesh=STRETCHED, sweeps=3), bc)
        u, v, p = _random_state(mesh, bc, np.random.default_rng(443))
        a, b = one.predict(u, v, p), three.predict(u, v, p)
        assert not np.array_equal(a.u_star, b.u_star)
        assert np.array_equal(a.a_p_u, b.a_p_u)
        c = three._assemble(u, v, three._for_u)
        expected = three._sweep_n(u, c, three._pressure_source(p, three._for_u))
        assert np.array_equal(b.u_star, expected)


# ---------------------------------------------------------------------------
# The obstacle wall stencil (ADR-012 B, ECR-002 step 4)
# ---------------------------------------------------------------------------


def _floor_channel(
    solid_floor: bool, mesh_spec: dict | None = None, solver: dict | None = None
) -> SimConfig:
    """A 2 by 1 channel of six fluid rows; its floor the domain edge or a SOLID row.

    With a SOLID floor the domain is one row taller and the inlet and outlet
    cover the fluid rows only, so the fluid region is the same channel.
    """
    ny, height = 6, 1.0
    base = height / ny if solid_floor else 0.0
    span = {"y_start": base, "y_end": height + base}
    boundaries = {
        "inlet": {**CHANNEL["inlet"], **span},
        "outlet": {**CHANNEL["outlet"], **span},
    }
    floor = {
        "name": "floor",
        "x_start": 0.0,
        "x_end": 2.0,
        "y_start": 0.0,
        "y_end": base,
    }
    return _config(
        boundaries,
        ny=ny + (1 if solid_floor else 0),
        height=height + base,
        obstacles=[floor] if solid_floor else None,
        mesh=mesh_spec,
        solver=solver,
    )


def _floor_states(rng: np.random.Generator) -> tuple[tuple, tuple, tuple]:
    """The two channels' predictors and one random state shared by their fluid rows."""
    edge_mesh, edge_bc, edge = _build(_floor_channel(False))
    solid_mesh, solid_bc, solid = _build(_floor_channel(True))
    assert np.all(solid_mesh.cell_type[0, :] == SOLID)
    u, v, _p = _random_state(edge_mesh, edge_bc, rng)
    u_s = np.vstack([np.zeros((1, u.shape[1])), u])
    v_s = np.vstack([np.zeros((1, v.shape[1])), v])
    solid_bc.apply_normal_velocity(u_s, v_s)
    assert np.array_equal(u_s[1:, :], u) and np.array_equal(v_s[1:, :], v)
    # A viscosity varying across the channel only, so both see the same field.
    profile = 0.05 * (1.0 + 4.0 * edge_mesh.yc**2)
    cells = np.tile(profile[:, None], (1, edge_mesh.xc.size))
    cells_s = np.vstack([np.full((1, cells.shape[1]), np.nan), cells])
    return (edge, u, v, cells), (solid, u_s, v_s, cells_s), (edge_mesh, solid_mesh)


@pytest.mark.unit
class TestObstacleWallStencil:
    """A tangential unknown beside a SOLID face takes the domain edge's wall stencil."""

    @staticmethod
    def _both(
        mp: MomentumPredictor, u: np.ndarray, v: np.ndarray, **kw: np.ndarray
    ) -> tuple[MomentumCoefficients, MomentumCoefficients]:
        """The u equation and the v equation in v's own frame (transposed)."""
        c_u = mp._assemble(u, v, mp._for_u, **kw)
        field = kw.get("mu_cells")
        kw_v = dict(kw)
        if field is not None:
            kw_v["mu_cells"] = np.ascontiguousarray(field.T)
        c_v = mp._assemble(v.T, u.T, mp._for_v, **kw_v)
        return c_u, c_v

    @pytest.mark.parametrize("with_field", [False, True])
    def test_a_solid_floor_row_assembles_as_the_domain_floor(
        self, with_field: bool
    ) -> None:
        """Every implicit coefficient and the boundary source of the fluid rows agree."""
        (edge, u, v, cells), (solid, u_s, v_s, cells_s), _ = _floor_states(
            np.random.default_rng(450)
        )
        kw_e = {"mu_cells": edge._check_mu_eff(cells)} if with_field else {}
        kw_s = {"mu_cells": solid._check_mu_eff(cells_s)} if with_field else {}
        e_u, e_vt = self._both(edge, u, v, **kw_e)
        s_u, s_vt = self._both(solid, u_s, v_s, **kw_s)
        names = ("a_p", "a_s_plus", "a_s_minus", "a_t_plus", "a_t_minus", "b_boundary")
        for name in names:
            # Rows of u above the SOLID row; columns of v (transposed rows)
            # above the wall face, which is no unknown.
            assert getattr(s_u, name)[1:, :] == pytest.approx(
                getattr(e_u, name), rel=ROUNDING, abs=1e-15
            ), name
            assert getattr(s_vt, name)[:, 1:] == pytest.approx(
                getattr(e_vt, name), rel=ROUNDING, abs=1e-15
            ), name
        assert np.all(s_u.a_t_minus[1, :] == 0.0)

    def test_the_wall_is_half_the_unknowns_cell_away_by_hand(self) -> None:
        """At rest, above a block's top and at its corner: four conductances, one to the face."""
        config = _config(CAVITY, obstacles=[FLOOR_BLOCK], mesh=STRETCHED)
        mesh, _bc, mp = _build(config)
        solid = mesh.cell_type == SOLID
        top = int(np.flatnonzero(solid.any(axis=1)).max())
        columns = np.flatnonzero(solid[top])
        j = top + 1
        u, v, _p = allocate_fields(mesh)
        c, _ = mp.momentum_coefficients(u, v)
        mu = config.mu
        # columns[0] is the face at the block's west side: its south
        # neighbour bounds a SOLID cell on the east only, the corner.
        for i in (columns[0], columns[1], columns[-1] + 1):
            expected = mu * (
                mesh.dy_cell[j] / mesh.dx_cell[i]
                + mesh.dy_cell[j] / mesh.dx_cell[i - 1]
                + mesh.dx_face[i] / mesh.dy_face[j + 1]
                + mesh.dx_face[i] / (mesh.yc[j] - mesh.y[j])
            )
            assert c.a_p[j, i] == pytest.approx(expected, rel=ROUNDING), i
            assert c.a_t_minus[j, i] == 0.0
            assert c.b_boundary[j, i] == 0.0

    @pytest.mark.parametrize("with_field", [False, True])
    def test_the_quick_correction_takes_the_boundary_form_at_the_floor_row(
        self, with_field: bool
    ) -> None:
        """The deferred correction of the fluid rows equals the domain floor's too."""
        (edge, u, v, cells), (solid, u_s, v_s, cells_s), _ = _floor_states(
            np.random.default_rng(451)
        )
        kw_e = {"mu_cells": edge._check_mu_eff(cells)} if with_field else {}
        kw_s = {"mu_cells": solid._check_mu_eff(cells_s)} if with_field else {}
        e_u, e_vt = self._both(edge, u, v, **kw_e)
        s_u, s_vt = self._both(solid, u_s, v_s, **kw_s)
        assert np.any(e_u.b_deferred != 0.0) and np.any(e_vt.b_deferred != 0.0)
        assert s_u.b_deferred[1:, :] == pytest.approx(
            e_u.b_deferred, rel=ROUNDING, abs=1e-15
        )
        assert s_vt.b_deferred[:, 1:] == pytest.approx(
            e_vt.b_deferred, rel=ROUNDING, abs=1e-15
        )

    def test_wall_mu_reaches_an_obstacle_face(self) -> None:
        """The hook replaces the obstacle face's viscosity, and refuses NaN there."""
        config = _config(CAVITY, obstacles=[FLOOR_BLOCK])
        mesh, _bc, mp = _build(config)
        solid = mesh.cell_type == SOLID
        top = int(np.flatnonzero(solid.any(axis=1)).max())
        i = int(np.flatnonzero(solid[top])[1])
        j = top + 1
        u, v, p = allocate_fields(mesh)
        shape = (mesh.yc.size + 1, mesh.xc.size + 1)
        walls = {"u": np.full(shape, config.mu), "v": np.full(shape, config.mu)}
        a = mp.predict(u, v, p, wall_mu=walls)
        walls["u"][j, i] = 5.0 * config.mu
        b = mp.predict(u, v, p, wall_mu=walls)
        expected = 4.0 * config.mu * mesh.dx_face[i] / (mesh.yc[j] - mesh.y[j])
        assert b.a_p_u[j, i] - a.a_p_u[j, i] == pytest.approx(expected, rel=1e-12)
        walls["u"][j, i] = np.nan
        with pytest.raises(ValueError, match="finite"):
            mp.predict(u, v, p, wall_mu=walls)

    def test_a_room_without_obstacles_has_no_obstacle_face(self) -> None:
        """The stencil acts only where a SOLID cell is, so VAL-001 and VAL-002 are untouched."""
        _mesh, _bc, mp = _build(_config(CHANNEL, mesh=STRETCHED))
        for o in (mp._for_u, mp._for_v):
            assert not o.walls.face.any()
            assert not o.walls.north.any() and not o.walls.south.any()


@pytest.mark.integration
def test_a_solid_floor_channel_solves_as_the_domain_floor_channel() -> None:
    """The obstacle stencil's test: the two channels agree to rounding, same stop.

    Before the stencil the SOLID floor's first row differs by a first-order
    amount (the wall a whole cell away). With the diffusive stencil and not
    QUICK's boundary form it differs by about 1.3e-4 m/s, 1.3e-3 of the
    0.1 m/s inflow, near the inlet, where the flow is developing
    (docs/reports/probe43/field43.py floor). With both, the fluid faces agree to rounding.
    """
    solver_keys = {"max_simple_iter": 5000, "convergence_tol": 1e-10}
    edge_cfg = _floor_channel(False, solver=solver_keys)
    solid_cfg = _floor_channel(True, solver=solver_keys)
    edge_mesh = Mesh(edge_cfg)
    solid_mesh = Mesh(solid_cfg)
    edge = StaggeredSolver(edge_mesh, edge_cfg, StaggeredBoundary(edge_mesh, edge_cfg))
    solid = StaggeredSolver(
        solid_mesh, solid_cfg, StaggeredBoundary(solid_mesh, solid_cfg)
    )
    edge.solve_steady()
    solid.solve_steady()
    assert edge.converged and solid.converged
    assert len(edge.residual_history) == len(solid.residual_history)
    fe, fs = edge.face_velocities, solid.face_velocities
    assert fe is not None and fs is not None
    assert np.abs(fs.u[1:, :] - fe.u).max() < 1e-12
    assert np.abs(fs.v[1:, :] - fe.v).max() < 1e-12
    assert np.abs(fe.v).max() > 1e-3


# ---------------------------------------------------------------------------
# The obstacle stencil on every side and at a corner (review 43 B2, test 43 B1)
# ---------------------------------------------------------------------------

# Stretched along the walls only: across them the two channels must share
# their cells, and a stretched axis spreads its spacing over the whole domain.
ALONG_X = {"x": {"stretch_ratio": 1.15}, "y": {"stretch_ratio": 1.0}}
ALONG_Y = {"x": {"stretch_ratio": 1.0}, "y": {"stretch_ratio": 1.15}}
SOLVE_KEYS = {"max_simple_iter": 5000, "convergence_tol": 1e-10}


def _x_channel(floor: bool, ceiling: bool, solver: dict | None = None) -> SimConfig:
    """CHANNEL's 2 by 1 flow along x over six fluid rows; SOLID rows below and above as asked."""
    ny, height = 6, 1.0
    dy = height / ny
    base = dy if floor else 0.0
    span = {"y_start": base, "y_end": base + height}
    # One value for the domain's extent and the last obstacle's end, so no
    # rounding puts the obstacle outside the domain.
    total = base + height + (dy if ceiling else 0.0)
    obstacles = []
    if floor:
        obstacles.append(
            {"name": "floor", "x_start": 0.0, "x_end": 2.0, "y_start": 0.0, "y_end": dy}
        )
    if ceiling:
        top = base + height
        obstacles.append(
            {
                "name": "ceiling",
                "x_start": 0.0,
                "x_end": 2.0,
                "y_start": top,
                "y_end": total,
            }
        )
    return _config(
        {
            "inlet": {**CHANNEL["inlet"], **span},
            "outlet": {**CHANNEL["outlet"], **span},
        },
        ny=ny + int(floor) + int(ceiling),
        height=total,
        obstacles=obstacles or None,
        mesh=ALONG_X,
        solver=solver,
    )


def _y_channel(sides: bool, solver: dict | None = None) -> SimConfig:
    """A 1 by 2 channel driven along y over six fluid columns; SOLID columns at both sides."""
    nx, width, height = 6, 1.0, 2.0
    dx = width / nx
    left = dx if sides else 0.0
    span = {"x_start": left, "x_end": left + width}
    total = width + (2 * dx if sides else 0.0)
    boundaries = {
        "inlet": {
            "type": "velocity_inlet",
            "location": "bottom",
            "velocity": 0.3,
            **span,
        },
        "outlet": {"type": "pressure_outlet", "location": "top", **span},
    }
    obstacles = None
    if sides:
        right = left + width
        obstacles = [
            {
                "name": "west",
                "x_start": 0.0,
                "x_end": dx,
                "y_start": 0.0,
                "y_end": height,
            },
            {
                "name": "east",
                "x_start": right,
                "x_end": total,
                "y_start": 0.0,
                "y_end": height,
            },
        ]
    return _config(
        boundaries,
        nx=nx + (2 if sides else 0),
        ny=8,
        width=total,
        height=height,
        obstacles=obstacles,
        mesh=ALONG_Y,
        solver=solver,
    )


def _pad(a: np.ndarray, axis: int, low: bool, high: bool) -> np.ndarray:
    """``a`` with a row (axis 0) or column (axis 1) of zeros added on the sides asked."""
    pieces = []
    shape = list(a.shape)
    shape[axis] = 1
    if low:
        pieces.append(np.zeros(shape))
    pieces.append(a)
    if high:
        pieces.append(np.zeros(shape))
    return np.concatenate(pieces, axis=axis)


def _coefficients(
    mp: MomentumPredictor, u: np.ndarray, v: np.ndarray, cells: np.ndarray | None
) -> tuple[MomentumCoefficients, MomentumCoefficients]:
    """Both equations, v in its own frame (transposed), with or without a field."""
    if cells is None:
        return mp._assemble(u, v, mp._for_u), mp._assemble(v.T, u.T, mp._for_v)
    field = mp._check_mu_eff(cells)
    return (
        mp._assemble(u, v, mp._for_u, mu_cells=field),
        mp._assemble(v.T, u.T, mp._for_v, mu_cells=np.ascontiguousarray(field.T)),
    )


COEFFICIENT_NAMES = (
    "a_p",
    "a_s_plus",
    "a_s_minus",
    "a_t_plus",
    "a_t_minus",
    "b_boundary",
    "b_deferred",
)


def _part(c: MomentumCoefficients, index: tuple[slice, slice]) -> MomentumCoefficients:
    """The same equation restricted to one block of its faces."""
    return MomentumCoefficients(**{n: getattr(c, n)[index] for n in COEFFICIENT_NAMES})


def _assert_same(ours: MomentumCoefficients, theirs: MomentumCoefficients) -> None:
    """Every coefficient and source of the two equations, to rounding."""
    for name in COEFFICIENT_NAMES:
        assert getattr(ours, name) == pytest.approx(
            getattr(theirs, name), rel=ROUNDING, abs=1e-15
        ), name


def _solve(config: SimConfig) -> StaggeredSolver:
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    solver.solve_steady()
    assert solver.converged
    return solver


@pytest.mark.unit
class TestObstacleStencilOnEverySide:
    """SOLID rows and columns on the high side and on both sides equal the domain edges.

    The floor tests reach only the low side in each frame. A SOLID ceiling
    reaches walls.north, wall_neg and blocked_neg (u and v frames); SOLID
    side columns reach the v frame's transverse paths and the u frame's
    streamwise ones, low and high.
    """

    @pytest.mark.parametrize(
        ("floor", "ceiling"), [(False, True), (True, True)], ids=["ceiling", "both"]
    )
    @pytest.mark.parametrize("with_field", [False, True], ids=["air", "field"])
    def test_solid_rows_assemble_as_the_domain_edges(
        self, floor: bool, ceiling: bool, with_field: bool
    ) -> None:
        """Every coefficient and source of the fluid rows, the deferred correction included."""
        edge_mesh, edge_bc, edge = _build(_x_channel(False, False))
        solid_mesh, solid_bc, solid = _build(_x_channel(floor, ceiling))
        rows = slice(int(floor), solid_mesh.yc.size - int(ceiling))
        assert np.all(solid_mesh.cell_type[rows, :] != SOLID)
        assert np.all(solid_mesh.cell_type[-1, :] == SOLID)
        u, v, _p = _random_state(edge_mesh, edge_bc, np.random.default_rng(460))
        u_s, v_s = _pad(u, 0, floor, ceiling), _pad(v, 0, floor, ceiling)
        solid_bc.apply_normal_velocity(u_s, v_s)
        assert np.array_equal(u_s[rows, :], u)
        cells = cells_s = None
        if with_field:
            profile = 0.05 * (1.0 + 4.0 * edge_mesh.yc**2)
            cells = np.tile(profile[:, None], (1, edge_mesh.xc.size))
            cells_s = np.full(solid_mesh.cell_type.shape, np.nan)
            cells_s[rows, :] = cells
        e_u, e_vt = _coefficients(edge, u, v, cells)
        s_u, s_vt = _coefficients(solid, u_s, v_s, cells_s)
        _assert_same(_part(s_u, np.s_[rows, :]), e_u)
        faces = slice(int(floor), int(floor) + v.shape[0])
        _assert_same(_part(s_vt, np.s_[:, faces]), e_vt)
        # The path this test exists for is taken.
        assert solid._for_u.walls.north.any()

    @pytest.mark.parametrize("with_field", [False, True], ids=["air", "field"])
    def test_solid_columns_assemble_as_the_domain_sides(self, with_field: bool) -> None:
        """A channel along y: every coefficient and source beside both SOLID columns."""
        edge_mesh, edge_bc, edge = _build(_y_channel(False))
        solid_mesh, solid_bc, solid = _build(_y_channel(True))
        columns = slice(1, solid_mesh.xc.size - 1)
        assert np.all(solid_mesh.cell_type[:, 0] == SOLID)
        assert np.all(solid_mesh.cell_type[:, -1] == SOLID)
        assert np.all(solid_mesh.cell_type[:, columns] != SOLID)
        u, v, _p = _random_state(edge_mesh, edge_bc, np.random.default_rng(461))
        u_s, v_s = _pad(u, 1, True, True), _pad(v, 1, True, True)
        solid_bc.apply_normal_velocity(u_s, v_s)
        assert np.array_equal(v_s[:, columns], v)
        cells = cells_s = None
        if with_field:
            profile = 0.05 * (1.0 + 4.0 * edge_mesh.xc**2)
            cells = np.tile(profile[None, :], (edge_mesh.yc.size, 1))
            cells_s = np.full(solid_mesh.cell_type.shape, np.nan)
            cells_s[:, columns] = cells
        e_u, e_vt = _coefficients(edge, u, v, cells)
        s_u, s_vt = _coefficients(solid, u_s, v_s, cells_s)
        faces = slice(1, 1 + u.shape[1])
        _assert_same(_part(s_u, np.s_[:, faces]), e_u)
        _assert_same(_part(s_vt, np.s_[columns, :]), e_vt)
        walls = solid._for_v.walls
        assert walls.north.any() and walls.south.any()


@pytest.mark.integration
@pytest.mark.parametrize("case", ["ceiling", "floor_and_ceiling", "sides"])
def test_solid_walls_solve_as_the_domain_edges(case: str) -> None:
    """Each channel solves to the faces of its domain-edge twin within 1e-12, same stop."""
    if case == "sides":
        edge, solid = (
            _solve(_y_channel(False, SOLVE_KEYS)),
            _solve(_y_channel(True, SOLVE_KEYS)),
        )
        fe, fs = edge.face_velocities, solid.face_velocities
        assert fe is not None and fs is not None
        pairs = ((fs.u[:, 1:-1], fe.u), (fs.v[:, 1:-1], fe.v))
        tangential = np.abs(fe.u).max()
    else:
        floor = case == "floor_and_ceiling"
        edge = _solve(_x_channel(False, False, SOLVE_KEYS))
        solid = _solve(_x_channel(floor, True, SOLVE_KEYS))
        fe, fs = edge.face_velocities, solid.face_velocities
        assert fe is not None and fs is not None
        start = int(floor)
        pairs = (
            (fs.u[start : start + fe.u.shape[0], :], fe.u),
            (fs.v[start : start + fe.v.shape[0], :], fe.v),
        )
        tangential = np.abs(fe.v).max()
    assert len(edge.residual_history) == len(solid.residual_history)
    for ours, theirs in pairs:
        assert np.abs(ours - theirs).max() < 1e-12
    # The flow develops, so the QUICK boundary form is exercised, not idle.
    assert tangential > 1e-3


@pytest.mark.unit
@pytest.mark.parametrize("component", ["u", "v"])
def test_an_obstacle_corner_face_adds_nothing_to_the_deferred_source(
    component: str,
) -> None:
    """The corner rule's QUICK half, by a construction where it must hold.

    At an obstacle's corner the neighbour face bounds a SOLID cell on one
    side only, so mass crosses the face through its open half. The corner
    rule makes the face a wall over its whole span: its face value is the
    wall's under QUICK and under upwind alike, so its correction is zero
    whatever that mass flux. The unknown beside it reads the open half's
    normal velocity through that face alone, so its deferred source must not
    change when that velocity does. A face away from every obstacle is the
    control: there the same change moves the source.
    """
    config = _config(CAVITY, obstacles=[FLOOR_BLOCK], mesh=STRETCHED)
    mesh, bc, mp = _build(config)
    u, v, _p = _random_state(mesh, bc, np.random.default_rng(462))
    o = mp._for_u if component == "u" else mp._for_v
    own, other = (u, v) if component == "u" else (v.T.copy(), u.T.copy())
    solid = o.solid
    nt = solid.shape[0]
    cases = []
    for side, mask in (("south", o.walls.south), ("north", o.walls.north)):
        for j, k in zip(*np.nonzero(mask), strict=True):
            i = k + 1  # the unknown's index along its own axis
            r = j - 1 if side == "south" else j + 1  # the neighbour face's row
            face = j if side == "south" else j + 1  # the transverse face's row
            if solid[r, i - 1] != solid[r, i]:
                open_column = i if solid[r, i - 1] else i - 1
                cases.append((j, i, face, open_column))
    assert cases, "the block has corners in this frame"

    def deferred(other_field: np.ndarray) -> np.ndarray:
        return mp._assemble(own, other_field, o).b_deferred

    base = deferred(other)
    for j, i, face, open_column in cases:
        nudged = other.copy()
        nudged[face, open_column] += 0.37
        assert deferred(nudged)[j, i] == base[j, i], (j, i)
    # Control: an unknown with no obstacle in reach answers the same nudge.
    far = _beyond_quick_reach(solid)
    j, i = (int(n) for n in np.argwhere(far[1 : nt - 1, :])[0])
    j += 1
    nudged = other.copy()
    nudged[j, i] += 0.37
    assert deferred(nudged)[j, i] != base[j, i]
