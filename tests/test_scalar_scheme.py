"""Unit tests for the shared scalar scheme (src/scalar_scheme.py; ADR-011 B, C; ADR-012 C).

The transport gate tests guard the scheme through the transport solver, and
the hash comparison of prompt 35 (results/builder35/) shows the move kept
their bits. These tests hold the module to its own contract, including what
the transport solver never asks of it: per-face conductances that differ
face by face, a sink in the diagonal of every cell, and a step per cell.

Meshes are small, non-square and stretched, with an obstacle, so an index
that works only on a square uniform grid shows.
"""

import numpy as np
import pytest

from src import scalar_scheme, solver_transport
from src.mesh import SOLID, Mesh
from src.scalar_scheme import (
    CONJUGATE_GRADIENT,
    JACOBI,
    ImplicitResult,
    advective_flux,
    implicit_step,
    mesh_axes,
)
from src.solver_transport import TransportSolver
from src.staggered import u_shape, v_shape
from validation.transport_cases import (
    FixedConditions,
    ScalarPhysics,
    random_face_field,
    transport_config,
    uniform_face_field,
    zero_conditions,
)

OBSTACLE = [
    {"name": "block", "x_start": 0.6, "x_end": 0.9, "y_start": 0.0, "y_end": 0.4}
]


def _mesh(obstacles: list[dict] | None = OBSTACLE) -> Mesh:
    """A 1.5 m by 0.9 m mesh of 11 by 7 cells, stretched on both axes."""
    config = transport_config(
        1.5,
        0.9,
        11,
        7,
        obstacles=obstacles,
        mesh={"x": {"stretch_ratio": 1.25}, "y": {"stretch_ratio": 1.15}},
    )
    mesh = Mesh(config)
    assert not mesh.is_uniform
    return mesh


def _conductances(
    mesh: Mesh, rng: np.random.Generator, high: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Random positive conductances on the faces between two non-SOLID cells, zero elsewhere."""
    solid = mesh.cell_type == SOLID
    g_u = np.zeros(u_shape(mesh))
    g_v = np.zeros(v_shape(mesh))
    inner_u = ~solid[:, :-1] & ~solid[:, 1:]
    inner_v = ~solid[:-1, :] & ~solid[1:, :]
    g_u[:, 1:-1] = np.where(inner_u, rng.uniform(0.0, high, inner_u.shape), 0.0)
    g_v[1:-1, :] = np.where(inner_v, rng.uniform(0.0, high, inner_v.shape), 0.0)
    return g_u, g_v


def _dense(
    mesh: Mesh,
    c_star: np.ndarray,
    dt: float | np.ndarray,
    g_u: np.ndarray,
    g_v: np.ndarray,
    diagonal: np.ndarray,
    held: np.ndarray | None = None,
) -> np.ndarray:
    """The implicit system assembled densely, face by face, and solved directly.

    SOLID rows read ``C_P = 0`` and held rows ``C_P = C*_P``.
    """
    ny, nx = c_star.shape
    n = ny * nx
    idx = np.arange(n).reshape(ny, nx)
    volume = np.outer(mesh.dy_cell, mesh.dx_cell)
    over_dt = volume / np.broadcast_to(dt, (ny, nx))
    a = np.zeros((n, n))
    a[idx, idx] = over_dt + diagonal
    for j in range(ny):
        for i in range(1, nx):
            g = g_u[j, i]
            p, q = idx[j, i - 1], idx[j, i]
            a[p, p] += g
            a[q, q] += g
            a[p, q] -= g
            a[q, p] -= g
    for j in range(1, ny):
        for i in range(nx):
            g = g_v[j, i]
            p, q = idx[j - 1, i], idx[j, i]
            a[p, p] += g
            a[q, q] += g
            a[p, q] -= g
            a[q, p] -= g
    solid = (mesh.cell_type == SOLID).ravel()
    a[solid, :] = 0.0
    a[solid, solid] = 1.0
    rhs = (over_dt * c_star).ravel()
    rhs[solid] = 0.0
    if held is not None:
        fixed = held.ravel()
        a[fixed, :] = 0.0
        a[fixed, fixed] = 1.0
        rhs[fixed] = c_star.ravel()[fixed]
    return np.linalg.solve(a, rhs).reshape(ny, nx)


@pytest.mark.unit
class TestUniformField:
    @pytest.mark.parametrize("upwind", [False, True])
    def test_a_uniform_field_takes_its_own_value_on_every_face(
        self, upwind: bool
    ) -> None:
        """Every face of a uniform field carries the field's value exactly,
        whatever the flux, on both axes and beside the obstacle, so the
        advective update of a divergence-free field is exact."""
        mesh = _mesh()
        axis_x, axis_y = mesh_axes(mesh)
        faces = random_face_field(mesh, seed=3, scale=0.7)
        flux_u = faces.u * mesh.dy_cell[:, None]
        flux_v = faces.v * mesh.dx_cell[None, :]
        c0 = 0.371
        c = np.full(mesh.cell_type.shape, c0)
        got_u = advective_flux(c, flux_u, np.full(flux_u.shape, c0), axis_x, upwind)
        got_v = advective_flux(
            c.T, flux_v.T, np.full(flux_v.T.shape, c0), axis_y, upwind
        ).T
        assert np.array_equal(got_u, flux_u * c0)
        assert np.array_equal(got_v, flux_v * c0)

    def test_the_implicit_solve_keeps_a_uniform_field(self) -> None:
        """No sink, differing conductances, a step per cell: a uniform field
        is the solution, so its residual is rounding alone (about 1e-11 of
        the right-hand side here, where the conductances outweigh V / dt by
        up to 1e5) and the solve returns it."""
        mesh = _mesh()
        rng = np.random.default_rng(11)
        solid = mesh.cell_type == SOLID
        g_u, g_v = _conductances(mesh, rng)
        c0 = 2.3
        c_star = np.where(solid, 0.0, c0)
        dt = rng.uniform(1e-3, 1e2, size=c_star.shape)
        result = implicit_step(
            c_star,
            np.outer(mesh.dy_cell, mesh.dx_cell),
            dt,
            g_u,
            g_v,
            np.zeros_like(c_star),
            solid,
            1e-9,
            5000,
        )
        assert result.converged
        assert np.max(np.abs(result.field[~solid] / c0 - 1.0)) < 1e-13
        assert np.all(result.field[solid] == 0.0)


@pytest.mark.unit
class TestPositivity:
    @pytest.mark.parametrize("dt", [1e-9, 1e-3, 1.0, 1e3, 1e9])
    @pytest.mark.parametrize("max_sweeps", [1, 7, 2000])
    def test_non_negative_input_gives_non_negative_output_at_any_step(
        self, dt: float, max_sweeps: int
    ) -> None:
        """The matrix is an M-matrix, so a non-negative right-hand side
        gives a non-negative solution; and each Jacobi sweep is a
        non-negative combination, so an unconverged iterate is too."""
        mesh = _mesh()
        rng = np.random.default_rng(int(np.log10(dt)) + 100 * max_sweeps)
        solid = mesh.cell_type == SOLID
        g_u, g_v = _conductances(mesh, rng, high=1e3)
        c_star = rng.uniform(0.0, 1.0, size=solid.shape)
        c_star[rng.uniform(size=solid.shape) < 0.4] = 0.0
        c_star[solid] = 0.0
        diagonal = np.where(solid, 0.0, rng.uniform(0.0, 50.0, size=solid.shape))
        result = implicit_step(
            c_star,
            np.outer(mesh.dy_cell, mesh.dx_cell),
            dt,
            g_u,
            g_v,
            diagonal,
            solid,
            1e-14,
            max_sweeps,
        )
        assert result.field.min() >= 0.0

    def test_a_per_cell_step_over_eighteen_decades_stays_non_negative(self) -> None:
        mesh = _mesh()
        rng = np.random.default_rng(5)
        solid = mesh.cell_type == SOLID
        g_u, g_v = _conductances(mesh, rng, high=1e2)
        c_star = np.where(solid, 0.0, rng.uniform(0.0, 1.0, size=solid.shape))
        dt = 10.0 ** rng.uniform(-9.0, 9.0, size=solid.shape)
        diagonal = np.where(solid, 0.0, rng.uniform(0.0, 1e4, size=solid.shape))
        result = implicit_step(
            c_star,
            np.outer(mesh.dy_cell, mesh.dx_cell),
            dt,
            g_u,
            g_v,
            diagonal,
            solid,
            1e-14,
            20000,
        )
        assert result.converged
        assert result.field.min() >= 0.0


@pytest.mark.unit
class TestPerCellStep:
    def test_a_per_cell_step_and_a_sink_match_a_dense_solve(self) -> None:
        """Each cell's own V / dt_P and sink sit in its own row: the Jacobi
        result equals the system assembled face by face and solved directly."""
        mesh = _mesh()
        rng = np.random.default_rng(23)
        solid = mesh.cell_type == SOLID
        g_u, g_v = _conductances(mesh, rng, high=0.5)
        c_star = np.where(solid, 0.0, rng.uniform(0.5, 1.5, size=solid.shape))
        dt = rng.uniform(0.05, 5.0, size=solid.shape)
        diagonal = np.where(solid, 0.0, rng.uniform(0.0, 0.3, size=solid.shape))
        result = implicit_step(
            c_star,
            np.outer(mesh.dy_cell, mesh.dx_cell),
            dt,
            g_u,
            g_v,
            diagonal,
            solid,
            1e-15,
            20000,
        )
        expected = _dense(mesh, c_star, dt, g_u, g_v, diagonal)
        assert result.converged
        assert np.allclose(result.field, expected, rtol=1e-12, atol=1e-15)

    def test_a_per_cell_step_differs_from_a_uniform_one(self) -> None:
        """The control on the test above: the dense solve with one mean step
        in every row does not match, so the per-cell rows were read."""
        mesh = _mesh()
        rng = np.random.default_rng(23)
        solid = mesh.cell_type == SOLID
        g_u, g_v = _conductances(mesh, rng, high=0.5)
        c_star = np.where(solid, 0.0, rng.uniform(0.5, 1.5, size=solid.shape))
        dt = rng.uniform(0.05, 5.0, size=solid.shape)
        diagonal = np.where(solid, 0.0, rng.uniform(0.0, 0.3, size=solid.shape))
        result = implicit_step(
            c_star,
            np.outer(mesh.dy_cell, mesh.dx_cell),
            dt,
            g_u,
            g_v,
            diagonal,
            solid,
            1e-15,
            20000,
        )
        wrong = _dense(mesh, c_star, float(dt.mean()), g_u, g_v, diagonal)
        assert not np.allclose(result.field, wrong, rtol=1e-6)

    def test_an_array_of_one_step_is_bitwise_the_scalar_step(self) -> None:
        mesh = _mesh()
        rng = np.random.default_rng(29)
        solid = mesh.cell_type == SOLID
        g_u, g_v = _conductances(mesh, rng)
        c_star = np.where(solid, 0.0, rng.uniform(0.0, 1.0, size=solid.shape))
        diagonal = np.where(solid, 0.0, rng.uniform(0.0, 1.0, size=solid.shape))
        volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        args = (g_u, g_v, diagonal, solid, 1e-14, 3000)
        scalar = implicit_step(c_star, volume, 0.37, *args)
        array = implicit_step(c_star, volume, np.full(solid.shape, 0.37), *args)
        assert np.array_equal(scalar.field, array.field)
        assert scalar.iterations == array.iterations

    def test_nothing_to_solve_returns_the_explicit_field(self) -> None:
        mesh = _mesh()
        solid = mesh.cell_type == SOLID
        c_star = np.where(solid, 0.0, 1.0)
        zero_u, zero_v = np.zeros(u_shape(mesh)), np.zeros(v_shape(mesh))
        result = implicit_step(
            c_star,
            np.outer(mesh.dy_cell, mesh.dx_cell),
            1.0,
            zero_u,
            zero_v,
            np.zeros_like(c_star),
            solid,
            1e-14,
            100,
        )
        assert isinstance(result, ImplicitResult)
        assert result.field is c_star
        assert (result.iterations, result.converged) == (0, True)

    def test_the_sweep_cap_is_reported(self) -> None:
        mesh = _mesh()
        rng = np.random.default_rng(31)
        solid = mesh.cell_type == SOLID
        g_u, g_v = _conductances(mesh, rng, high=1e3)
        c_star = np.where(solid, 0.0, rng.uniform(0.0, 1.0, size=solid.shape))
        result = implicit_step(
            c_star,
            np.outer(mesh.dy_cell, mesh.dx_cell),
            1.0,
            g_u,
            g_v,
            np.zeros_like(c_star),
            solid,
            1e-14,
            3,
        )
        assert result.iterations == 3
        assert not result.converged


@pytest.mark.unit
class TestHeldCells:
    def test_a_held_cell_keeps_its_value_and_its_neighbours_read_it(self) -> None:
        """Held cells are Dirichlet rows: exactly their C*, and the solve
        around them equals the dense system with those rows replaced by
        C_P = C*_P (ADR-012 C's held wall eps, the limit of a dominant
        diagonal)."""
        mesh = _mesh()
        rng = np.random.default_rng(37)
        solid = mesh.cell_type == SOLID
        g_u, g_v = _conductances(mesh, rng, high=0.5)
        c_star = np.where(solid, 0.0, rng.uniform(0.5, 1.5, size=solid.shape))
        held = np.zeros_like(solid)
        held[-1, :] = True
        held[2, 1] = True
        c_star[held] = 4.0
        dt = rng.uniform(0.05, 5.0, size=solid.shape)
        diagonal = np.where(solid, 0.0, rng.uniform(0.0, 0.3, size=solid.shape))
        result = implicit_step(
            c_star,
            np.outer(mesh.dy_cell, mesh.dx_cell),
            dt,
            g_u,
            g_v,
            diagonal,
            solid,
            1e-13,
            20000,
            held=held,
        )
        assert result.converged
        assert np.all(result.field[held] == 4.0)

        ny, nx = solid.shape
        idx = np.arange(ny * nx).reshape(ny, nx)
        volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        a = np.zeros((ny * nx, ny * nx))
        a[idx, idx] = volume / dt + diagonal
        for j in range(ny):
            for i in range(1, nx):
                p, q = idx[j, i - 1], idx[j, i]
                a[[p, q], [p, q]] += g_u[j, i]
                a[p, q] -= g_u[j, i]
                a[q, p] -= g_u[j, i]
        for j in range(1, ny):
            for i in range(nx):
                p, q = idx[j - 1, i], idx[j, i]
                a[[p, q], [p, q]] += g_v[j, i]
                a[p, q] -= g_v[j, i]
                a[q, p] -= g_v[j, i]
        rhs = (volume / dt * c_star).ravel()
        fixed = (solid | held).ravel()
        a[fixed, :] = 0.0
        a[fixed, fixed] = 1.0
        rhs[fixed] = c_star.ravel()[fixed]
        expected = np.linalg.solve(a, rhs).reshape(ny, nx)
        assert np.allclose(result.field, expected, rtol=1e-10, atol=1e-14)

        free = implicit_step(
            c_star,
            volume,
            dt,
            g_u,
            g_v,
            diagonal,
            solid,
            1e-13,
            20000,
        )
        assert np.all(free.field[-1, ~solid[-1, :]] < 4.0)


@pytest.mark.unit
def test_the_axes_carry_the_mesh_nodes_and_the_solid_mask() -> None:
    mesh = _mesh()
    axis_x, axis_y = mesh_axes(mesh)
    ny, nx = mesh.cell_type.shape
    solid = mesh.cell_type == SOLID
    assert np.array_equal(axis_x.nodes[1:-1], mesh.xc)
    assert (axis_x.nodes[0], axis_x.nodes[-1]) == (mesh.x[0], mesh.x[-1])
    assert np.array_equal(axis_y.nodes[1:-1], mesh.yc)
    assert np.array_equal(axis_x.faces, mesh.x[1:-1])
    assert np.array_equal(axis_y.left, np.arange(1, ny))
    assert np.array_equal(axis_x.solid_ext[:, 1:-1], solid)
    assert axis_y.solid_ext.shape == (nx, ny + 2)
    assert np.array_equal(axis_y.solid_ext[:, 1:-1], solid.T)
    assert not axis_x.solid_ext[:, [0, -1]].any()


BROWNIAN = 1.0e-4
SCHMIDT = 0.7


def _harmonic(nu_p: float, nu_e: float, d_p: float, d_e: float) -> float:
    """The distance-weighted harmonic mean of two values, written from the formula."""
    if nu_p == 0.0 or nu_e == 0.0:
        return 0.0
    return (d_p + d_e) / (d_p / nu_p + d_e / nu_e)


def _face_conductances(
    mesh: Mesh, nu_t: np.ndarray, d_brownian: float, schmidt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(D_B + nu_f / Sc_t) A_f / d_f`` on every interior face, by plain loops.

    Returns the two conductance arrays and the diffusivity on every interior
    face between non-SOLID cells, flattened, for the control's mean.
    """
    solid = mesh.cell_type == SOLID
    ny, nx = solid.shape
    g_u = np.zeros(u_shape(mesh))
    g_v = np.zeros(v_shape(mesh))
    diffusivities = []
    for j in range(ny):
        for i in range(1, nx):
            if solid[j, i - 1] or solid[j, i]:
                continue
            nu_f = _harmonic(
                nu_t[j, i - 1],
                nu_t[j, i],
                mesh.x[i] - mesh.xc[i - 1],
                mesh.xc[i] - mesh.x[i],
            )
            d = d_brownian + nu_f / schmidt
            diffusivities.append(d)
            g_u[j, i] = d * mesh.dy_cell[j] / mesh.dx_face[i]
    for j in range(1, ny):
        for i in range(nx):
            if solid[j - 1, i] or solid[j, i]:
                continue
            nu_f = _harmonic(
                nu_t[j - 1, i],
                nu_t[j, i],
                mesh.y[j] - mesh.yc[j - 1],
                mesh.yc[j] - mesh.y[j],
            )
            d = d_brownian + nu_f / schmidt
            diffusivities.append(d)
            g_v[j, i] = d * mesh.dx_cell[i] / mesh.dy_face[j]
    return g_u, g_v, np.array(diffusivities)


@pytest.mark.unit
class TestPiecewiseDiffusivity:
    """REQ-T13: ADR-012 F asks the dense-solve test for a per-face coefficient."""

    def _room(self) -> tuple[Mesh, TransportSolver]:
        """The stretched 11 by 7 room with one SOLID cell, Brownian 1e-4 m^2/s."""
        stretch = {"x": {"stretch_ratio": 1.25}, "y": {"stretch_ratio": 1.15}}
        plain = Mesh(transport_config(1.5, 0.9, 11, 7, mesh=stretch))
        obstacle = {
            "name": "b",
            "x_start": float(plain.x[4]),
            "x_end": float(plain.x[5]),
            "y_start": float(plain.y[3]),
            "y_end": float(plain.y[4]),
        }
        config = transport_config(
            1.5,
            0.9,
            11,
            7,
            mesh=stretch,
            obstacles=[obstacle],
            diffusion_tol=1e-14,
            max_diffusion_iter=20000,
            turbulent_schmidt=SCHMIDT,
        )
        mesh = Mesh(config)
        assert (mesh.cell_type == SOLID).sum() == 1
        solver = TransportSolver(
            mesh,
            config,
            ScalarPhysics(settling=0.0, diffusion=BROWNIAN),
            FixedConditions(zero_conditions(mesh)),
        )
        return mesh, solver

    def _field(self, mesh: Mesh) -> np.ndarray:
        """nu_t from 5e-5 to 1.5e-3 m^2/s, a column of zeros, and 9 m^2/s in the SOLID cell."""
        ny, nx = mesh.cell_type.shape
        i, j = np.meshgrid(np.arange(nx), np.arange(ny))
        field = 5.0e-5 * 30.0 ** ((i + 0.5 * j) / (nx - 1 + 0.5 * (ny - 1)))
        field[:, 7] = 0.0
        field[mesh.cell_type == SOLID] = 9.0
        return field

    def test_the_step_with_a_field_matches_a_dense_solve_with_the_same_diffusivity(
        self,
    ) -> None:
        """The implicit step with a per-face diffusivity agrees with a dense solve."""
        mesh, solver = self._room()
        nu_t = self._field(mesh)
        solid = mesh.cell_type == SOLID
        rng = np.random.default_rng(31)
        c = np.where(solid, 0.0, rng.uniform(0.5, 1.5, size=solid.shape))
        dt = 5.0
        got = solver.solve_timestep(
            c, uniform_face_field(mesh, 0.0, 0.0), 0, dt, eddy_viscosity=nu_t
        )
        assert solver.diffusion_converged and solver.last_diffusion_sweeps > 0

        g_u, g_v, diffusivities = _face_conductances(mesh, nu_t, BROWNIAN, SCHMIDT)
        expected = _dense(mesh, c, dt, g_u, g_v, np.zeros(solid.shape))
        assert np.allclose(got, expected, rtol=1e-12, atol=1e-14)
        # The faces in the zero column carry the Brownian coefficient alone, so
        # the test covers the zero branch as well as the spread of the rest.
        assert diffusivities.min() == pytest.approx(BROWNIAN, rel=1e-15)
        assert diffusivities.max() > 10.0 * BROWNIAN

        # The control: the same system with the field's mean diffusivity on
        # every face is a different system, far outside the tolerance.
        mean = diffusivities.mean()
        uniform_u = np.where(
            g_u > 0.0, mean * mesh.dy_cell[:, None] / mesh.dx_face[None, :], 0.0
        )
        uniform_v = np.where(
            g_v > 0.0, mean * mesh.dx_cell[None, :] / mesh.dy_face[:, None], 0.0
        )
        wrong = _dense(mesh, c, dt, uniform_u, uniform_v, np.zeros(solid.shape))
        assert np.max(np.abs(wrong - expected) / expected.max()) > 1e-4


def _cg_system(
    mesh: Mesh, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Diffusion, a sink in every cell and a per-cell step on the obstacle mesh.

    Returns c_star, the step, g_u, g_v, the sink and a held mask of the top
    row and one interior cell beside the obstacle, at values of their own.
    """
    rng = np.random.default_rng(seed)
    solid = mesh.cell_type == SOLID
    g_u, g_v = _conductances(mesh, rng, high=0.5)
    c_star = np.where(solid, 0.0, rng.uniform(0.5, 1.5, size=solid.shape))
    held = np.zeros_like(solid)
    held[-1, :] = True
    held[2, 1] = True
    c_star[held] = 3.0
    dt = rng.uniform(0.05, 5.0, size=solid.shape)
    diagonal = np.where(solid, 0.0, rng.uniform(0.0, 0.3, size=solid.shape))
    return c_star, dt, g_u, g_v, diagonal, held


def _residual(
    c: np.ndarray,
    c_star: np.ndarray,
    volume: np.ndarray,
    dt: float | np.ndarray,
    g_u: np.ndarray,
    g_v: np.ndarray,
    diagonal: np.ndarray,
    fixed: np.ndarray,
) -> np.ndarray:
    """``b + sum G C_N - a_P C`` at the unknowns, zero at the fixed cells, face by face."""
    over_dt = volume / dt
    r = over_dt * c_star - (over_dt + diagonal) * c
    r[:, 1:] += g_u[:, 1:-1] * (c[:, :-1] - c[:, 1:])
    r[:, :-1] += g_u[:, 1:-1] * (c[:, 1:] - c[:, :-1])
    r[1:, :] += g_v[1:-1, :] * (c[:-1, :] - c[1:, :])
    r[:-1, :] += g_v[1:-1, :] * (c[1:, :] - c[:-1, :])
    return np.where(fixed, 0.0, r)


# The sign trap's seed and tolerance. Most seeds of the construction leave a
# converged CG iterate negative somewhere; seed 0 does after 9 iterations, at
# -1.6e-10 (results/builder51/seed_trap.py).
SIGN_TRAP_SEED = 0
SIGN_TRAP_TOL = 1.5e-5


def _sign_trap(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A system whose converged CG iterate has a negative value (prompt 51).

    A background of 1e-9 with three cells near 1, a sink up to 11 times
    V / dt and weak diffusion: the solution is positive, but at a tolerance
    of 1.5e-5 CG stops with an error larger than the background in some
    cells. Returns c_star, g_u, g_v and the sink; the step is 1 s.
    """
    rng = np.random.default_rng(SIGN_TRAP_SEED)
    solid = mesh.cell_type == SOLID
    g_u, g_v = _conductances(mesh, rng, high=0.7)
    c_star = np.where(solid, 0.0, 1e-9 * rng.uniform(0.5, 1.5, size=solid.shape))
    c_star.flat[rng.integers(0, solid.size, size=3)] = rng.uniform(0.5, 1.5, size=3)
    c_star[solid] = 0.0
    diagonal = np.where(solid, 0.0, 11.0 * rng.uniform(0.0, 1.0, size=solid.shape))
    return c_star, g_u, g_v, diagonal


@pytest.mark.unit
class TestConjugateGradient:
    """The CG path of implicit_step (prompt 51): the system, the stop, the
    held cells, the cap and the Jacobi fallback."""

    def test_cg_and_jacobi_agree_and_both_match_a_dense_solve(self) -> None:
        """Diffusion, a sink, a step per cell, held cells and a SOLID block:
        both methods meet the tolerance and land on the dense solution, and CG
        takes fewer iterations than Jacobi sweeps."""
        mesh = _mesh()
        solid = mesh.cell_type == SOLID
        c_star, dt, g_u, g_v, diagonal, held = _cg_system(mesh, seed=41)
        volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        args = (c_star, volume, dt, g_u, g_v, diagonal, solid, 1e-13, 20000)
        cg = implicit_step(*args, held=held, method=CONJUGATE_GRADIENT)
        jacobi = implicit_step(*args, held=held, method=JACOBI)
        expected = _dense(mesh, c_star, dt, g_u, g_v, diagonal, held=held)
        assert cg.converged and jacobi.converged
        assert (cg.method, cg.fallbacks, jacobi.fallbacks) == (CONJUGATE_GRADIENT, 0, 0)
        assert np.allclose(cg.field, expected, rtol=1e-10, atol=1e-14)
        assert np.allclose(jacobi.field, expected, rtol=1e-10, atol=1e-14)
        assert 0 < cg.iterations < jacobi.iterations

    def test_the_stop_is_the_largest_residual_over_the_largest_rhs(self) -> None:
        """The returned field's largest residual over the unknowns is at most
        tol times the largest |b|. On this 40 by 30 field the 2-norm stop
        CG's own loop in pressure.py uses, ``||r|| <= tol ||b||``, stops
        earlier with the largest residual several times over that; the plant
        of that stop fails here (prompt 51)."""
        config = transport_config(2.0, 1.5, 40, 30)
        mesh = Mesh(config)
        rng = np.random.default_rng(43)
        solid = mesh.cell_type == SOLID
        g_u, g_v = _conductances(mesh, rng, high=0.05)
        c_star = rng.uniform(0.5, 1.5, size=solid.shape)
        volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        diagonal = np.zeros_like(c_star)
        tol = 1e-6
        result = implicit_step(
            c_star,
            volume,
            1.0,
            g_u,
            g_v,
            diagonal,
            solid,
            tol,
            500,
            method=CONJUGATE_GRADIENT,
        )
        assert result.converged and result.fallbacks == 0
        r = _residual(result.field, c_star, volume, 1.0, g_u, g_v, diagonal, solid)
        assert np.abs(r).max() <= tol * np.abs(volume * c_star).max()

    @pytest.mark.parametrize("seed", [207, 214])
    def test_converged_means_the_true_residual_meets_the_stop(self, seed: int) -> None:
        """On a stiff system whose stop rounding cannot reach, the residual
        recursion drifts below the stop while the true residual stays above
        it. The solve may only report converged if the true residual meets
        the stop; otherwise it falls back. Defect caught: the confirmation on
        the true residual removed, under which both seeds report converged
        after 374 and 361 iterations with no fallback (review 51 B1, test 51;
        seeds from results/tester51/survivors_t51.py's construction)."""
        rng0 = np.random.default_rng(seed)
        high = 10.0 ** rng0.uniform(2.0, 9.0)
        mesh = Mesh(transport_config(1.3, 0.9, 40, 30))
        solid = mesh.cell_type == SOLID
        rng = np.random.default_rng(seed)
        g_u = np.zeros(u_shape(mesh))
        g_v = np.zeros(v_shape(mesh))
        g_u[:, 1:-1] = rng.uniform(0.1 * high, high, (30, 39))
        g_v[1:-1, :] = rng.uniform(0.1 * high, high, (29, 40))
        volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        c_star = rng.uniform(0.5, 1.5, solid.shape)
        dt = 10.0 ** rng.uniform(0.0, 4.0)
        diagonal = np.zeros(solid.shape)
        tol = 10.0 ** -rng.uniform(8.0, 12.0)
        result = implicit_step(
            c_star,
            volume,
            dt,
            g_u,
            g_v,
            diagonal,
            solid,
            tol,
            500,
            method=CONJUGATE_GRADIENT,
        )
        r = _residual(result.field, c_star, volume, dt, g_u, g_v, diagonal, solid)
        stop = tol * np.abs(volume / dt * c_star).max()
        if result.converged and result.fallbacks == 0:
            assert np.abs(r).max() <= stop
        else:
            assert result.fallbacks == 1

    @pytest.mark.parametrize("method", [CONJUGATE_GRADIENT, JACOBI])
    def test_the_stop_scale_counts_the_held_cells(self, method: str) -> None:
        """The scale is the largest |b| over every cell, held ones included,
        as Jacobi has always taken it: in the model a held wall eps holds the
        largest |b| in most eps solves. Here one isolated held cell at 1e7
        sets the scale, so the start already meets the stop and the solve
        returns C* untouched. Defect caught: the scale over the unknowns
        only, under which the stop is about 1e-9 of this one and the solve
        iterates (test 51)."""
        mesh = Mesh(transport_config(1.0, 0.5, 20, 10))
        solid = mesh.cell_type == SOLID
        rng = np.random.default_rng(53)
        g_u, g_v = _conductances(mesh, rng, high=0.5)
        held = np.zeros_like(solid)
        held[5, 10] = True
        g_u[5, 10] = g_u[5, 11] = 0.0
        g_v[5, 10] = g_v[6, 10] = 0.0
        c_star = rng.uniform(0.5, 1.5, size=solid.shape)
        c_star[held] = 1.0e7
        volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        diagonal = np.zeros_like(c_star)
        result = implicit_step(
            c_star,
            volume,
            1.0,
            g_u,
            g_v,
            diagonal,
            solid,
            1e-3,
            500,
            held=held,
            method=method,
        )
        assert result.converged and result.iterations == 0 and result.fallbacks == 0
        assert np.array_equal(result.field, c_star)

    def test_a_held_cell_keeps_its_value_exactly(self) -> None:
        """Held cells are not unknowns under CG: their values are C*'s to the
        bit, with no fallback to supply them; leaving them as unknowns moves
        them (the plant)."""
        mesh = _mesh()
        solid = mesh.cell_type == SOLID
        c_star, dt, g_u, g_v, diagonal, held = _cg_system(mesh, seed=47)
        result = implicit_step(
            c_star,
            np.outer(mesh.dy_cell, mesh.dx_cell),
            dt,
            g_u,
            g_v,
            diagonal,
            solid,
            1e-13,
            20000,
            held=held,
            method=CONJUGATE_GRADIENT,
        )
        assert result.converged and result.fallbacks == 0
        assert np.array_equal(result.field[held], c_star[held])
        assert np.all(result.field[solid] == 0.0)

    def test_the_cap_is_honoured_and_falls_back(self) -> None:
        """Three CG iterations on a system that needs more: the count is the
        cap, the solve falls back to Jacobi from C*, which caps at three
        sweeps too, and the result says it did not converge."""
        mesh = _mesh()
        rng = np.random.default_rng(31)
        solid = mesh.cell_type == SOLID
        g_u, g_v = _conductances(mesh, rng, high=1e3)
        c_star = np.where(solid, 0.0, rng.uniform(0.5, 1.0, size=solid.shape))
        args = (
            c_star,
            np.outer(mesh.dy_cell, mesh.dx_cell),
            1.0,
            g_u,
            g_v,
            np.zeros_like(c_star),
            solid,
            1e-14,
            3,
        )
        result = implicit_step(*args, method=CONJUGATE_GRADIENT)
        jacobi = implicit_step(*args, method=JACOBI)
        assert (result.iterations, result.fallbacks, result.fallback_sweeps) == (
            3,
            1,
            3,
        )
        assert not result.converged
        assert np.array_equal(result.field, jacobi.field)

    def test_a_negative_converged_cg_value_falls_back_to_jacobi_from_c_star(
        self,
    ) -> None:
        """CG meets the stop with a negative value in a cell it solves for;
        the solve falls back, and the result is bitwise the Jacobi path's
        from C*, non-negative. Planted, a fallback on the cap alone returns
        the negative CG iterate, and a fallback restarted from the CG iterate
        returns a different field (prompt 51)."""
        mesh = _mesh()
        solid = mesh.cell_type == SOLID
        c_star, g_u, g_v, diagonal = _sign_trap(mesh)
        volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        args = (c_star, volume, 1.0, g_u, g_v, diagonal, solid, SIGN_TRAP_TOL, 500)
        # The premise: CG's own iterate converges with a negative value.
        a_p = volume + g_u[:, :-1] + g_u[:, 1:] + g_v[:-1, :] + g_v[1:, :] + diagonal
        stop = SIGN_TRAP_TOL * float(np.abs(volume * c_star).max())
        own, iterations, converged = scalar_scheme._conjugate_gradient(
            c_star.copy(), a_p, volume * c_star, g_u, g_v, solid, stop, 500
        )
        assert converged and iterations < 500
        assert own[~solid].min() < 0.0

        result = implicit_step(*args, method=CONJUGATE_GRADIENT)
        jacobi = implicit_step(*args, method=JACOBI)
        assert (result.iterations, result.fallbacks) == (iterations, 1)
        assert result.fallback_sweeps == jacobi.iterations
        assert result.converged
        assert np.array_equal(result.field, jacobi.field)
        assert result.field.min() >= 0.0

    def test_a_zero_converged_cg_value_falls_back_too(self) -> None:
        """The fallback's test is a value at or below zero: two sources in a
        field of zeros and weak diffusion, so CG meets the stop in two
        iterations with the cells its Krylov space has not reached still at
        zero. Planted, a test of ``>= 0`` keeps the CG iterate."""
        mesh = _mesh()
        rng = np.random.default_rng(0)
        solid = mesh.cell_type == SOLID
        g_u, g_v = _conductances(mesh, rng, high=1e-3)
        c_star = np.zeros(solid.shape)
        c_star[0, 0] = 1.0
        c_star[-1, -1] = 0.5
        args = (
            c_star,
            np.outer(mesh.dy_cell, mesh.dx_cell),
            1.0,
            g_u,
            g_v,
            np.zeros_like(c_star),
            solid,
            1e-3,
            500,
        )
        result = implicit_step(*args, method=CONJUGATE_GRADIENT)
        jacobi = implicit_step(*args, method=JACOBI)
        assert result.iterations < 500 and result.fallbacks == 1
        assert result.converged
        assert np.array_equal(result.field, jacobi.field)

    def test_an_unknown_method_is_refused(self) -> None:
        mesh = _mesh()
        solid = mesh.cell_type == SOLID
        c_star = np.where(solid, 0.0, 1.0)
        with pytest.raises(ValueError, match="method must be one of"):
            implicit_step(
                c_star,
                np.outer(mesh.dy_cell, mesh.dx_cell),
                1.0,
                np.zeros(u_shape(mesh)),
                np.zeros(v_shape(mesh)),
                np.zeros_like(c_star),
                solid,
                1e-10,
                10,
                method="gauss-seidel",
            )

    def test_the_transport_solver_asks_for_jacobi(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The transport solver's implicit step names JACOBI: its iterates
        keep a concentration's sign down to zero (prompt 51). The plant, the
        transport on CG, fails here."""
        mesh, solver = TestPiecewiseDiffusivity()._room()
        methods: list[str] = []

        def spy(*args: object, **kwargs: object) -> ImplicitResult:
            methods.append(str(kwargs.get("method", JACOBI)))
            return implicit_step(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(solver_transport, "implicit_step", spy)
        solid = mesh.cell_type == SOLID
        c = np.where(solid, 0.0, 1.0)
        solver.solve_timestep(c, uniform_face_field(mesh, 0.0, 0.0), 0, 5.0)
        assert methods == [JACOBI]


@pytest.mark.unit
@pytest.mark.parametrize("upwind", [True, False], ids=["upwind", "umist"])
def test_a_vent_face_carries_the_inflow_in_and_the_room_cell_out(upwind: bool) -> None:
    """ADR-014 J: a face of a SOLID cell that the caller opens.

    One row of six cells, cell 2 SOLID. Face 2 (between cells 1 and 2) and
    face 3 (between cells 2 and 3) are vents. Into the room from the obstacle
    the face carries the inflow value; out of the room, the room cell's
    value, with no stencil whatever the scheme.
    """
    nodes = np.array([0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.0])
    solid = np.zeros((1, 6), dtype=bool)
    solid[0, 2] = True
    axis = scalar_scheme.Axis(
        nodes=nodes,
        faces=np.arange(1.0, 6.0),
        left=np.arange(1, 6),
        solid_ext=np.pad(solid, ((0, 0), (1, 1))),
    )
    c = np.array([[1.0, 2.0, 0.0, 4.0, 5.0, 6.0]])
    inflow = np.zeros((1, 7))
    inflow[0, 2] = inflow[0, 3] = 7.0
    vents = np.zeros((1, 7), dtype=bool)
    vents[0, 2] = vents[0, 3] = True
    # Face 2 flows -x out of the obstacle into cell 1; face 3 flows -x from
    # cell 3 into the obstacle.
    flux = np.zeros((1, 7))
    flux[0, 2] = flux[0, 3] = -0.5
    out = advective_flux(c, flux, inflow, axis, upwind, vents)
    assert out[0, 2] == -0.5 * 7.0
    assert out[0, 3] == -0.5 * 4.0
    # +x: face 2 draws from cell 1 into the obstacle, face 3 discharges.
    out = advective_flux(c, -flux, inflow, axis, upwind, vents)
    assert out[0, 2] == 0.5 * 2.0
    assert out[0, 3] == 0.5 * 7.0
    # Without the argument the faces are what they always were.
    assert np.array_equal(
        advective_flux(c, flux, inflow, axis, upwind),
        advective_flux(c, flux, inflow, axis, upwind, None),
    )
