"""Tests for the staggered momentum predictor with QUICK advection (REQ-S07, REQ-S09).

There is no analytical solution for a predictor alone, so the tests check
properties that hold exactly. Where floating point forbids ``==``, the
tolerance is stated with its reason: Lagrange weights sum to one only to
rounding, so a reproduced quadratic or an advected constant differs from the
exact value by a few ulps of the values involved.
"""

import numpy as np
import pytest

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.momentum import MomentumPredictor, quick_face_values
from src.staggered import allocate_fields

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
            "max_pressure_iter": 200,
            "pressure_tol": 1.0e-6,
        },
        "boundaries": boundaries,
        "obstacles": obstacles or [],
        "sensors": [{"name": "center", "x": width / 2, "y": height / 2}],
        "thresholds": {"0.1e-6": 100.0},
    }
    if mesh is not None:
        raw["mesh"] = mesh
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
