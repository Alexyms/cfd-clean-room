"""Tests for the staggered field layout: shapes, allocation, face-to-center averaging."""

import math
from itertools import pairwise

import numpy as np
import pytest

from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.staggered import (
    FaceVelocities,
    allocate_fields,
    cell_center_coordinates,
    check_staggered_pair,
    edge_cell_inputs,
    edge_cells,
    p_shape,
    to_cell_centers,
    u_face_coordinates,
    u_shape,
    v_face_coordinates,
    v_shape,
)


def _config(
    nx: int, ny: int, mesh: dict | None = None, obstacles: list[dict] | None = None
) -> SimConfig:
    raw = {
        "domain": {"width": 2.0, "height": 1.0, "nx": nx, "ny": ny},
        "fluid": {"density": 1.2, "viscosity": 1.81e-5, "temperature": 293.0},
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
        "boundaries": {
            "top": {
                "type": "velocity_inlet",
                "location": "top",
                "x_start": 0.0,
                "x_end": 2.0,
                "velocity": 0.1,
            },
        },
        "obstacles": obstacles or [],
        "sensors": [{"name": "center", "x": 1.0, "y": 0.5}],
        "thresholds": {"0.1e-6": 100.0},
    }
    if mesh is not None:
        raw["mesh"] = mesh
    return SimConfig.from_dict(raw)


STRETCHED = {"x": {"stretch_ratio": 1.08}, "y": {"stretch_ratio": 1.15}}


@pytest.mark.unit
class TestShapesAndAllocation:
    def test_shapes_follow_the_layout(self) -> None:
        mesh = Mesh(_config(8, 5))
        assert u_shape(mesh) == (5, 9)
        assert v_shape(mesh) == (6, 8)
        assert p_shape(mesh) == (5, 8)

    def test_allocation_is_zero_float64_contiguous(self) -> None:
        mesh = Mesh(_config(8, 5))
        u, v, p = allocate_fields(mesh)
        for arr, shape in ((u, (5, 9)), (v, (6, 8)), (p, (5, 8))):
            assert arr.shape == shape
            assert arr.dtype == np.float64
            assert arr.flags["C_CONTIGUOUS"]
            assert not arr.any()

    def test_storage_coordinates_sit_on_faces_and_centers(self) -> None:
        mesh = Mesh(_config(8, 5, STRETCHED))
        xu, yu = u_face_coordinates(mesh)
        assert xu.shape == u_shape(mesh)
        assert np.array_equal(xu[0], mesh.x)
        assert np.array_equal(yu[:, 0], mesh.yc)
        xv, yv = v_face_coordinates(mesh)
        assert xv.shape == v_shape(mesh)
        assert np.array_equal(xv[0], mesh.xc)
        assert np.array_equal(yv[:, 0], mesh.y)
        xp, yp = cell_center_coordinates(mesh)
        assert xp.shape == p_shape(mesh)
        assert np.array_equal(xp[0], mesh.xc)
        assert np.array_equal(yp[:, 0], mesh.yc)

    def test_inconsistent_shapes_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="inconsistent"):
            to_cell_centers(np.zeros((5, 9)), np.zeros((7, 8)))


@pytest.mark.unit
class TestFaceToCenterAveraging:
    """Centers are face midpoints, so a linear field must survive exactly."""

    @pytest.mark.parametrize("stretch", [None, STRETCHED], ids=["uniform", "stretched"])
    def test_linear_field_is_reproduced_to_machine_precision(self, stretch) -> None:
        mesh = Mesh(_config(16, 9, stretch))
        xu, yu = u_face_coordinates(mesh)
        xv, yv = v_face_coordinates(mesh)
        u = 0.3 + 1.7 * xu - 0.4 * yu
        v = -1.1 + 0.6 * xv + 2.3 * yv
        u_c, v_c = to_cell_centers(u, v)
        xp, yp = cell_center_coordinates(mesh)
        np.testing.assert_allclose(u_c, 0.3 + 1.7 * xp - 0.4 * yp, rtol=0, atol=1e-14)
        np.testing.assert_allclose(v_c, -1.1 + 0.6 * xp + 2.3 * yp, rtol=0, atol=1e-14)

    def test_a_constant_field_is_exact_bit_for_bit(self) -> None:
        mesh = Mesh(_config(16, 9, STRETCHED))
        u = np.full(u_shape(mesh), 0.7)
        v = np.full(v_shape(mesh), -2.5)
        u_c, v_c = to_cell_centers(u, v)
        assert np.array_equal(u_c, np.full(p_shape(mesh), 0.7))
        assert np.array_equal(v_c, np.full(p_shape(mesh), -2.5))

    @pytest.mark.parametrize("stretched", [False, True], ids=["uniform", "stretched"])
    def test_quadratic_field_error_falls_at_second_order(self, stretched) -> None:
        """The control for the linear test: a curved field must not be exact,
        and its error must halve twice per refinement.

        A refinement family on a stretched mesh keeps the mapping fixed: the
        total growth of cell width from wall to center stays constant, so the
        per-cell ratio is that growth to the power 1 / (cells per half).
        Refining at a fixed per-cell ratio is not a refinement family; the
        center cells then grow by r to the n and the error cannot converge.
        """
        errors = []
        for n in (16, 32, 64):
            stretch = None
            if stretched:
                stretch = {
                    "x": {"stretch_ratio": 2.0 ** (1.0 / n)},
                    "y": {"stretch_ratio": 3.0 ** (2.0 / n)},
                }
            mesh = Mesh(_config(2 * n, n, stretch))
            xu, yu = u_face_coordinates(mesh)
            xv, yv = v_face_coordinates(mesh)
            u = xu**2 + 0.5 * yu**2
            v = 3.0 * xv**2 - yv**2
            u_c, v_c = to_cell_centers(u, v)
            xp, yp = cell_center_coordinates(mesh)
            err_u = np.max(np.abs(u_c - (xp**2 + 0.5 * yp**2)))
            err_v = np.max(np.abs(v_c - (3.0 * xp**2 - yp**2)))
            errors.append(max(err_u, err_v))
        assert errors[0] > 1e-6, "quadratic field was reproduced exactly; not a test"
        orders = [math.log2(a / b) for a, b in pairwise(errors)]
        assert all(order > 1.9 for order in orders), orders


@pytest.mark.unit
class TestFaceVelocities:
    """The read-only face pair the velocity solver exposes (REQ-S13, ADR-011 A)."""

    def test_copy_of_gives_read_only_float64_contiguous_copies(self) -> None:
        """Writes to the copies raise, and writes to the sources do not reach them."""
        u = np.arange(5 * 9, dtype=np.float32).reshape(5, 9)
        v = np.asfortranarray(np.arange(6 * 8, dtype=np.float64).reshape(6, 8))
        faces = FaceVelocities.copy_of(u, v)
        for arr, source in ((faces.u, u), (faces.v, v)):
            assert arr.dtype == np.float64
            assert arr.flags["C_CONTIGUOUS"]
            assert not arr.flags.writeable
            assert not np.shares_memory(arr, source)
            assert np.array_equal(arr, source)
            with pytest.raises(ValueError, match="read-only"):
                arr[0, 0] = 1.0
        u[0, 0] = -7.0
        v[0, 0] = -7.0
        assert faces.u[0, 0] == 0.0
        assert faces.v[0, 0] == 0.0
        assert u.flags.writeable and v.flags.writeable

    def test_the_pair_averages_like_any_other(self) -> None:
        """to_cell_centers accepts the read-only pair and gives the same means."""
        mesh = Mesh(_config(8, 5))
        u, v, _p = allocate_fields(mesh)
        u[:] = np.random.default_rng(3).standard_normal(u.shape)
        v[:] = np.random.default_rng(4).standard_normal(v.shape)
        faces = FaceVelocities.copy_of(u, v)
        expected = to_cell_centers(u, v)
        got = to_cell_centers(faces.u, faces.v)
        assert np.array_equal(got[0], expected[0])
        assert np.array_equal(got[1], expected[1])

    def test_inconsistent_shapes_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="inconsistent"):
            FaceVelocities.copy_of(np.zeros((5, 9)), np.zeros((7, 8)))
        with pytest.raises(ValueError, match="2D"):
            check_staggered_pair(np.zeros(9), np.zeros((6, 8)))

    def test_constructor_refuses_arrays_that_break_the_contract(self) -> None:
        """A writeable, non-float64 or non-contiguous array cannot become a member."""
        u = np.zeros((5, 9))
        v = np.zeros((6, 8))
        with pytest.raises(ValueError, match="read-only"):
            FaceVelocities(u, v)
        u_ro = np.zeros((5, 9), dtype=np.float32)
        u_ro.flags.writeable = False
        v_ro = np.zeros((6, 8))
        v_ro.flags.writeable = False
        with pytest.raises(ValueError, match="float64"):
            FaceVelocities(u_ro, v_ro)
        u_f = np.asfortranarray(np.zeros((5, 9)))
        u_f.flags.writeable = False
        with pytest.raises(ValueError, match="C-contiguous"):
            FaceVelocities(u_f, v_ro)
        good = FaceVelocities.copy_of(u, v)
        again = FaceVelocities(good.u, good.v)
        assert again.u is good.u and again.v is good.v

    def test_a_read_only_view_of_a_writeable_array_is_refused(self) -> None:
        """Review 31 S4: the owner of the viewed array could still change the instance."""
        u = np.zeros((5, 9))
        v = np.zeros((6, 8))
        u_view, v_view = u.view(), v.view()
        u_view.flags.writeable = False
        v_view.flags.writeable = False
        with pytest.raises(ValueError, match="must own its data"):
            FaceVelocities(u_view, v_view)
        faces = FaceVelocities.copy_of(u_view, v_view)
        u[0, 0] = 42.0
        assert faces.u[0, 0] == 0.0

    def test_instances_are_not_compared_or_hashed_by_value(self) -> None:
        """eq=False: identity semantics, so == and hash never hit the ndarray truth value."""
        a = FaceVelocities.copy_of(np.zeros((5, 9)), np.zeros((6, 8)))
        b = FaceVelocities.copy_of(np.zeros((5, 9)), np.zeros((6, 8)))
        assert a != b and a == a
        assert len({a, b}) == 2


@pytest.mark.unit
class TestEdgeCells:
    """The cells behind each edge's faces, and the inputs both boundary layers share."""

    def test_each_edge_is_the_right_row_or_column_in_face_order(self) -> None:
        """An asymmetric array, so a swapped or reversed edge shows."""
        cell_type = np.arange(5 * 8).reshape(5, 8)
        expected = {
            "bottom": [0, 1, 2, 3, 4, 5, 6, 7],
            "top": [32, 33, 34, 35, 36, 37, 38, 39],
            "left": [0, 8, 16, 24, 32],
            "right": [7, 15, 23, 31, 39],
        }
        for edge, cells in expected.items():
            got = edge_cells(cell_type, edge)
            assert got.tolist() == cells, edge
            assert np.shares_memory(got, cell_type)

    def test_unknown_edge_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown edge 'front'"):
            edge_cells(np.zeros((2, 2)), "front")
        with pytest.raises(ValueError, match="unknown edge"):
            edge_cell_inputs(Mesh(_config(4, 3)), "north")

    def test_edge_cell_inputs_are_the_centers_and_the_solid_mask(self) -> None:
        """Against a mask built from the obstacle's own extents."""
        raw_obstacles = [
            {
                "name": "floor",
                "x_start": 0.62,
                "x_end": 1.08,
                "y_start": 0.0,
                "y_end": 0.28,
            },
            {
                "name": "right",
                "x_start": 1.72,
                "x_end": 2.0,
                "y_start": 0.42,
                "y_end": 0.78,
            },
        ]
        mesh = Mesh(_config(20, 10, obstacles=raw_obstacles))
        assert (mesh.cell_type == SOLID).sum() == 5 * 3 + 3 * 4
        expected_solid = {
            "bottom": [6 <= i <= 10 for i in range(20)],
            "top": [False] * 20,
            "left": [False] * 10,
            "right": [4 <= j <= 7 for j in range(10)],
        }
        for edge, solid in expected_solid.items():
            coordinates, mask = edge_cell_inputs(mesh, edge)
            assert mask.tolist() == solid, edge
            centers = mesh.xc if edge in ("bottom", "top") else mesh.yc
            assert np.array_equal(coordinates, centers)
            assert mask.shape == coordinates.shape
