"""Tests for the structured mesh module.

Unit tests verify grid geometry, cell classification, neighbor lookup,
and obstacle handling. Integration tests verify that the mesh loads
correctly from the default clean room configuration.
"""

import numpy as np
import pytest
import yaml

from src.config import SimConfig
from src.mesh import BOUNDARY, FLUID, SOLID, Mesh


def _make_config(tmp_path, overrides: dict | None = None) -> SimConfig:
    """Write a YAML config and return a loaded SimConfig.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest tmp_path fixture directory.
    overrides : dict or None
        Keys to replace in the base config dict before writing.

    Returns
    -------
    SimConfig
        Loaded and validated configuration.
    """
    base = {
        "domain": {"width": 1.0, "height": 1.0, "nx": 10, "ny": 10},
        "fluid": {"density": 1.2, "viscosity": 1.81e-5, "temperature": 293.0},
        "particles": {
            "density": 1000.0,
            "sizes": [0.1e-6],
            "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {
                "diameters": [0.1e-6],
                "efficiencies": [0.99999],
            },
        },
        "solver": {
            "dt": 0.01,
            "t_end": 1.0,
            "output_interval": 10,
            "convergence_tol": 1.0e-6,
            "max_simple_iter": 100,
        },
        "boundaries": {
            "top": {
                "type": "velocity_inlet",
                "location": "top",
                "x_start": 0.0,
                "x_end": 1.0,
                "velocity": 0.1,
            },
        },
        "obstacles": [],
        "sensors": [{"name": "center", "x": 0.5, "y": 0.5}],
        "thresholds": {"0.1e-6": 100.0},
    }
    if overrides:
        for key, val in overrides.items():
            base[key] = val
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(base, default_flow_style=False), encoding="utf-8")
    return SimConfig(str(path))


# ---------------------------------------------------------------------------
# Unit tests -- Grid geometry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMeshGeometry:
    """Verify face coordinates, cell centers, and grid spacing."""

    def test_dx_dy_correct(self, tmp_path) -> None:
        """dx and dy match room dimensions divided by cell counts."""
        config = _make_config(
            tmp_path,
            overrides={"domain": {"width": 2.0, "height": 1.0, "nx": 20, "ny": 10}},
        )
        mesh = Mesh(config)
        assert mesh.dx == pytest.approx(0.1)
        assert mesh.dy == pytest.approx(0.1)

    def test_x_face_coordinates(self, tmp_path) -> None:
        """x face array has shape (nx+1,) with correct endpoints."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        assert mesh.x.shape == (11,)
        assert mesh.x[0] == pytest.approx(0.0)
        assert mesh.x[-1] == pytest.approx(1.0)

    def test_y_face_coordinates(self, tmp_path) -> None:
        """y face array has shape (ny+1,) with correct endpoints."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        assert mesh.y.shape == (11,)
        assert mesh.y[0] == pytest.approx(0.0)
        assert mesh.y[-1] == pytest.approx(1.0)

    def test_xc_cell_centers(self, tmp_path) -> None:
        """xc has shape (nx,) with values at cell centers."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        assert mesh.xc.shape == (10,)
        assert mesh.xc[0] == pytest.approx(0.05)
        assert mesh.xc[-1] == pytest.approx(0.95)

    def test_yc_cell_centers(self, tmp_path) -> None:
        """yc has shape (ny,) with values at cell centers."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        assert mesh.yc.shape == (10,)
        assert mesh.yc[0] == pytest.approx(0.05)
        assert mesh.yc[-1] == pytest.approx(0.95)

    def test_cell_type_shape_and_dtype(self, tmp_path) -> None:
        """cell_type has shape (ny, nx) and dtype int32."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        assert mesh.cell_type.shape == (10, 10)
        assert mesh.cell_type.dtype == np.int32

    def test_all_arrays_contiguous(self, tmp_path) -> None:
        """All arrays are C-contiguous."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        assert mesh.x.flags["C_CONTIGUOUS"]
        assert mesh.y.flags["C_CONTIGUOUS"]
        assert mesh.xc.flags["C_CONTIGUOUS"]
        assert mesh.yc.flags["C_CONTIGUOUS"]
        assert mesh.cell_type.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# Unit tests -- Cell classification
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCellClassification:
    """Verify FLUID, SOLID, and BOUNDARY classification logic."""

    def test_total_count_equals_nx_times_ny(self, tmp_path) -> None:
        """Total cells equals nx * ny."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        assert mesh.cell_type.size == 100

    def test_counts_sum_to_total(self, tmp_path) -> None:
        """FLUID + SOLID + BOUNDARY counts sum to total cell count."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        n_fluid = np.sum(mesh.cell_type == FLUID)
        n_solid = np.sum(mesh.cell_type == SOLID)
        n_boundary = np.sum(mesh.cell_type == BOUNDARY)
        assert n_fluid + n_solid + n_boundary == 100

    def test_no_obstacles_boundary_and_fluid(self, tmp_path) -> None:
        """With no obstacles, edge cells are BOUNDARY, interior are FLUID."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        # No SOLID cells
        assert np.sum(mesh.cell_type == SOLID) == 0
        # Edge cells: 4 sides minus 4 corners counted once = 4*10 - 4 = 36
        assert np.sum(mesh.cell_type == BOUNDARY) == 36
        # Interior: 8*8 = 64
        assert np.sum(mesh.cell_type == FLUID) == 64

    def test_obstacle_cell_is_solid(self, tmp_path) -> None:
        """A cell whose center falls inside an obstacle is SOLID."""
        # 10x10 grid on 1m x 1m domain, dx=dy=0.1
        # Cell centers at 0.05, 0.15, ..., 0.95
        # Obstacle from (0.2, 0.2) to (0.6, 0.6) should capture
        # centers at 0.25, 0.35, 0.45, 0.55 in both x and y
        config = _make_config(
            tmp_path,
            overrides={
                "obstacles": [
                    {
                        "name": "block",
                        "x_start": 0.2,
                        "x_end": 0.6,
                        "y_start": 0.2,
                        "y_end": 0.6,
                    }
                ]
            },
        )
        mesh = Mesh(config)
        # Cell (i=2, j=2) has center (0.25, 0.25), inside obstacle
        assert mesh.cell_type[2, 2] == SOLID
        # Cell (i=5, j=5) has center (0.55, 0.55), inside obstacle
        assert mesh.cell_type[5, 5] == SOLID

    def test_cell_outside_obstacle_is_fluid(self, tmp_path) -> None:
        """A cell whose center is outside all obstacles is FLUID."""
        config = _make_config(
            tmp_path,
            overrides={
                "obstacles": [
                    {
                        "name": "block",
                        "x_start": 0.2,
                        "x_end": 0.6,
                        "y_start": 0.2,
                        "y_end": 0.6,
                    }
                ]
            },
        )
        mesh = Mesh(config)
        # Cell (i=8, j=8) has center (0.85, 0.85), well outside
        assert mesh.cell_type[8, 8] == FLUID

    def test_cell_on_obstacle_edge_is_solid(self, tmp_path) -> None:
        """A cell whose center is exactly on the obstacle edge is SOLID.

        Obstacle uses closed interval [x_start, x_end].
        """
        # Obstacle x_end=0.55 matches center of cell i=5 (0.55)
        config = _make_config(
            tmp_path,
            overrides={
                "obstacles": [
                    {
                        "name": "block",
                        "x_start": 0.2,
                        "x_end": 0.55,
                        "y_start": 0.2,
                        "y_end": 0.55,
                    }
                ]
            },
        )
        mesh = Mesh(config)
        assert mesh.cell_type[5, 5] == SOLID

    def test_boundary_cell_on_edge_not_obstacle(self, tmp_path) -> None:
        """Edge cells not inside obstacles are BOUNDARY."""
        config = _make_config(
            tmp_path,
            overrides={
                "obstacles": [
                    {
                        "name": "block",
                        "x_start": 0.3,
                        "x_end": 0.7,
                        "y_start": 0.3,
                        "y_end": 0.7,
                    }
                ]
            },
        )
        mesh = Mesh(config)
        # Bottom-left corner (i=0, j=0) is a domain edge, not in obstacle
        assert mesh.cell_type[0, 0] == BOUNDARY

    def test_obstacle_on_domain_edge_is_solid(self, tmp_path) -> None:
        """An obstacle touching the domain edge produces SOLID, not BOUNDARY."""
        # Obstacle at bottom-left corner, covering cells with centers
        # at (0.05, 0.05), (0.15, 0.05), (0.05, 0.15), (0.15, 0.15)
        config = _make_config(
            tmp_path,
            overrides={
                "obstacles": [
                    {
                        "name": "corner",
                        "x_start": 0.0,
                        "x_end": 0.2,
                        "y_start": 0.0,
                        "y_end": 0.2,
                    }
                ]
            },
        )
        mesh = Mesh(config)
        # Cell (0, 0) center is (0.05, 0.05), inside obstacle
        assert mesh.cell_type[0, 0] == SOLID
        assert mesh.cell_type[1, 1] == SOLID

    def test_multiple_obstacles(self, tmp_path) -> None:
        """Multiple obstacles each produce SOLID cells independently."""
        config = _make_config(
            tmp_path,
            overrides={
                "obstacles": [
                    {
                        "name": "left",
                        "x_start": 0.0,
                        "x_end": 0.15,
                        "y_start": 0.0,
                        "y_end": 0.15,
                    },
                    {
                        "name": "right",
                        "x_start": 0.85,
                        "x_end": 1.0,
                        "y_start": 0.85,
                        "y_end": 1.0,
                    },
                ]
            },
        )
        mesh = Mesh(config)
        # Bottom-left obstacle: cell (0,0) center (0.05, 0.05) is inside
        assert mesh.cell_type[0, 0] == SOLID
        # Top-right obstacle: cell (9,9) center (0.95, 0.95) is inside
        assert mesh.cell_type[9, 9] == SOLID
        # Middle cell (5,5) center (0.55, 0.55) is FLUID
        assert mesh.cell_type[5, 5] == FLUID


# ---------------------------------------------------------------------------
# Unit tests -- is_fluid
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsFluid:
    """Verify is_fluid returns correct values and raises on out-of-range."""

    def test_fluid_cell_returns_true(self, tmp_path) -> None:
        """is_fluid returns True for a FLUID cell."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        # Interior cell (5, 5) with no obstacles is FLUID
        assert mesh.is_fluid(5, 5) is True

    def test_solid_cell_returns_false(self, tmp_path) -> None:
        """is_fluid returns False for a SOLID cell."""
        config = _make_config(
            tmp_path,
            overrides={
                "obstacles": [
                    {
                        "name": "block",
                        "x_start": 0.0,
                        "x_end": 1.0,
                        "y_start": 0.0,
                        "y_end": 1.0,
                    }
                ]
            },
        )
        mesh = Mesh(config)
        assert mesh.is_fluid(5, 5) is False

    def test_boundary_cell_returns_false(self, tmp_path) -> None:
        """is_fluid returns False for a BOUNDARY cell."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        # Edge cell (0, 0) is BOUNDARY
        assert mesh.is_fluid(0, 0) is False

    def test_negative_index_raises(self, tmp_path) -> None:
        """Negative index raises IndexError."""
        mesh = Mesh(_make_config(tmp_path))
        with pytest.raises(IndexError):
            mesh.is_fluid(-1, 0)

    def test_i_too_large_raises(self, tmp_path) -> None:
        """i >= nx raises IndexError."""
        mesh = Mesh(_make_config(tmp_path))
        with pytest.raises(IndexError):
            mesh.is_fluid(10, 0)

    def test_j_too_large_raises(self, tmp_path) -> None:
        """j >= ny raises IndexError."""
        mesh = Mesh(_make_config(tmp_path))
        with pytest.raises(IndexError):
            mesh.is_fluid(0, 10)


# ---------------------------------------------------------------------------
# Unit tests -- get_neighbors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetNeighbors:
    """Verify neighbor lookup returns correct cells and raises on errors."""

    def test_interior_cell_has_four_neighbors(self, tmp_path) -> None:
        """An interior cell returns exactly 4 neighbors."""
        mesh = Mesh(_make_config(tmp_path))
        neighbors = mesh.get_neighbors(5, 5)
        assert len(neighbors) == 4

    def test_corner_cell_has_two_neighbors(self, tmp_path) -> None:
        """A corner cell returns exactly 2 neighbors."""
        mesh = Mesh(_make_config(tmp_path))
        neighbors = mesh.get_neighbors(0, 0)
        assert len(neighbors) == 2

    def test_edge_cell_has_three_neighbors(self, tmp_path) -> None:
        """A non-corner edge cell returns exactly 3 neighbors."""
        mesh = Mesh(_make_config(tmp_path))
        neighbors = mesh.get_neighbors(5, 0)
        assert len(neighbors) == 3

    def test_neighbors_within_bounds(self, tmp_path) -> None:
        """All returned neighbors are within grid bounds."""
        mesh = Mesh(_make_config(tmp_path))
        for i in range(10):
            for j in range(10):
                for ni, nj in mesh.get_neighbors(i, j):
                    assert 0 <= ni < 10
                    assert 0 <= nj < 10

    def test_out_of_range_raises(self, tmp_path) -> None:
        """Out-of-range input raises IndexError."""
        mesh = Mesh(_make_config(tmp_path))
        with pytest.raises(IndexError):
            mesh.get_neighbors(10, 5)
        with pytest.raises(IndexError):
            mesh.get_neighbors(5, -1)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMeshIntegration:
    """Verify mesh works with the full default clean room config."""

    def test_default_config_loads(self) -> None:
        """Mesh initializes from clean_room_default.yaml without error."""
        config = SimConfig("configs/clean_room_default.yaml")
        mesh = Mesh(config)
        assert mesh.cell_type.shape == (75, 200)

    def test_default_config_shapes_and_dtypes(self) -> None:
        """Arrays from default config have correct shapes and dtypes."""
        config = SimConfig("configs/clean_room_default.yaml")
        mesh = Mesh(config)
        assert mesh.x.shape == (201,)
        assert mesh.y.shape == (76,)
        assert mesh.xc.shape == (200,)
        assert mesh.yc.shape == (75,)
        assert mesh.cell_type.dtype == np.int32

    def test_default_config_has_solid_cells(self) -> None:
        """Default config with 4 obstacles produces a reasonable SOLID count.

        The four obstacles occupy a known area of the domain. SOLID
        cell count should be in a plausible range based on the
        obstacle dimensions relative to the 200x75 grid.
        """
        config = SimConfig("configs/clean_room_default.yaml")
        mesh = Mesh(config)
        n_solid = int(np.sum(mesh.cell_type == SOLID))
        # Obstacles span a meaningful fraction of the domain
        assert n_solid > 100
        # But not the entire domain
        assert n_solid < 200 * 75 // 2

    def test_default_config_counts_sum(self) -> None:
        """FLUID + SOLID + BOUNDARY equals total cells for default config."""
        config = SimConfig("configs/clean_room_default.yaml")
        mesh = Mesh(config)
        total = 200 * 75
        n_fluid = int(np.sum(mesh.cell_type == FLUID))
        n_solid = int(np.sum(mesh.cell_type == SOLID))
        n_boundary = int(np.sum(mesh.cell_type == BOUNDARY))
        assert n_fluid + n_solid + n_boundary == total
