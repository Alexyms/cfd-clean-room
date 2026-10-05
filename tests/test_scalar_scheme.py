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

from src.mesh import SOLID, Mesh
from src.scalar_scheme import (
    ImplicitResult,
    advective_flux,
    implicit_step,
    mesh_axes,
)
from src.staggered import u_shape, v_shape
from validation.transport_cases import random_face_field, transport_config

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
) -> np.ndarray:
    """The implicit system assembled densely, face by face, and solved directly."""
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
        assert scalar.sweeps == array.sweeps

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
        assert (result.sweeps, result.converged) == (0, True)

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
        assert result.sweeps == 3
        assert not result.converged


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
