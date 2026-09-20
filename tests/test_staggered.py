"""Tests for the staggered field layout: shapes, allocation, face-to-center averaging."""

import math
from itertools import pairwise

import numpy as np
import pytest

from src.config import SimConfig
from src.mesh import Mesh
from src.staggered import (
    allocate_fields,
    cell_center_coordinates,
    p_shape,
    to_cell_centers,
    u_face_coordinates,
    u_shape,
    v_face_coordinates,
    v_shape,
)


def _config(nx: int, ny: int, mesh: dict | None = None) -> SimConfig:
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
            "max_pressure_iter": 200,
            "pressure_tol": 1.0e-6,
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
        "obstacles": [],
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

    @pytest.mark.parametrize("stretch", [None, STRETCHED], ids=["uniform", "stretched"])
    def test_quadratic_field_error_falls_at_second_order(self, stretch) -> None:
        """The control for the linear test: a curved field must not be exact,
        and its error must halve twice per refinement."""
        errors = []
        for n in (8, 16, 32):
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
