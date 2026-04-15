"""Tests for the boundary condition module.

Unit tests verify BC mapping, ghost cell interpolation formulas, and
flux calculations. Integration tests verify correct initialization
from the default clean room configuration.
"""

import numpy as np
import pytest
import yaml

from src.boundary import BoundaryManager
from src.config import SimConfig
from src.mesh import BOUNDARY, Mesh


def _make_config(tmp_path, overrides: dict | None = None) -> SimConfig:
    """Write a YAML config and return a loaded SimConfig."""
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
            "alpha_velocity": 0.7,
            "alpha_pressure": 0.3,
            "max_pressure_iter": 200,
            "pressure_tol": 1.0e-6,
        },
        "boundaries": {
            "top_inlet": {
                "type": "velocity_inlet",
                "location": "top",
                "x_start": 0.0,
                "x_end": 1.0,
                "velocity": 0.45,
            },
            "bottom_outlet": {
                "type": "pressure_outlet",
                "location": "bottom",
                "x_start": 0.0,
                "x_end": 1.0,
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
# Unit tests -- BC mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBCMapping:
    """Verify that boundary cells are mapped to the correct BC types."""

    def test_top_inlet_mapped(self, tmp_path) -> None:
        """Top-edge cells in the inlet range are mapped as velocity_inlet."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        top_entries = [e for e in bm._entries if e.edge == "top"]
        assert len(top_entries) > 0
        for e in top_entries:
            assert e.bc_type == "velocity_inlet"

    def test_bottom_outlet_mapped(self, tmp_path) -> None:
        """Bottom-edge cells in the outlet range are mapped as pressure_outlet."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        bottom_entries = [e for e in bm._entries if e.edge == "bottom"]
        assert len(bottom_entries) > 0
        for e in bottom_entries:
            assert e.bc_type == "pressure_outlet"

    def test_unmatched_edges_default_to_wall(self, tmp_path) -> None:
        """Edge cells not covered by any named boundary default to wall."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        side_entries = [e for e in bm._entries if e.edge in ("left", "right")]
        assert len(side_entries) > 0
        for e in side_entries:
            assert e.bc_type == "wall"

    def test_partial_boundary_coverage(self, tmp_path) -> None:
        """Cells outside a named boundary's spatial range default to wall."""
        config = _make_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "partial_top": {
                        "type": "velocity_inlet",
                        "location": "top",
                        "x_start": 0.3,
                        "x_end": 0.7,
                        "velocity": 0.1,
                    },
                }
            },
        )
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        top_entries = [e for e in bm._entries if e.edge == "top"]
        inlet_count = sum(1 for e in top_entries if e.bc_type == "velocity_inlet")
        wall_count = sum(1 for e in top_entries if e.bc_type == "wall")
        assert inlet_count > 0
        assert wall_count > 0

    def test_corner_cell_handled(self, tmp_path) -> None:
        """Corner cells get a BC assignment without error."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        # Bottom-left corner (0,0) should be in the entries
        corner = [e for e in bm._entries if e.i == 0 and e.j == 0]
        assert len(corner) == 1

    def test_all_boundary_cells_mapped(self, tmp_path) -> None:
        """Every BOUNDARY cell in the mesh has a corresponding entry."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        n_boundary = int(np.sum(mesh.cell_type == BOUNDARY))
        assert len(bm._entries) == n_boundary


# ---------------------------------------------------------------------------
# Unit tests -- Ghost cell velocity application
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVelocityBC:
    """Verify ghost cell velocity formulas for wall, inlet, and outlet."""

    def test_wall_noslip_formula(self, tmp_path) -> None:
        """Wall BC: u_bnd = u_interior / 3 (Dirichlet V=0 at face).

        NOT u_bnd = 0 (sets BC at cell center, first-order error).
        NOT u_bnd = -u_interior (equidistant ghost cell formula).
        """
        config = _make_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "left_wall": {
                        "type": "wall",
                        "location": "left",
                        "y_start": 0.0,
                        "y_end": 1.0,
                    },
                    "right_wall": {
                        "type": "wall",
                        "location": "right",
                        "y_start": 0.0,
                        "y_end": 1.0,
                    },
                    "top_wall": {
                        "type": "wall",
                        "location": "top",
                        "x_start": 0.0,
                        "x_end": 1.0,
                    },
                    "bottom_wall": {
                        "type": "wall",
                        "location": "bottom",
                        "x_start": 0.0,
                        "x_end": 1.0,
                    },
                }
            },
        )
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)

        u = np.ones((10, 10), dtype=np.float64) * 3.0
        v = np.ones((10, 10), dtype=np.float64) * 6.0
        bm.apply_velocity_bc(u, v)

        # Check non-corner wall entries whose interior neighbor is FLUID.
        # Corner cells may have boundary neighbors that were already
        # modified, so their values depend on processing order.
        from src.mesh import FLUID as _FLUID

        for e in bm._entries:
            if mesh.cell_type[e.nj, e.ni] != _FLUID:
                continue
            assert u[e.j, e.i] == pytest.approx(1.0), (
                f"Wall u at ({e.i},{e.j}): expected 1.0, got {u[e.j, e.i]}"
            )
            assert v[e.j, e.i] == pytest.approx(2.0), (
                f"Wall v at ({e.i},{e.j}): expected 2.0, got {v[e.j, e.i]}"
            )

    def test_velocity_inlet_top_formula(self, tmp_path) -> None:
        """Velocity inlet on top: v_bnd = (2*(-V) + v_interior) / 3."""
        config = _make_config(tmp_path)  # top inlet V=0.45
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)

        u = np.zeros((10, 10), dtype=np.float64)
        v = np.zeros((10, 10), dtype=np.float64)
        bm.apply_velocity_bc(u, v)

        top_inlets = [
            e for e in bm._entries if e.edge == "top" and e.bc_type == "velocity_inlet"
        ]
        for e in top_inlets:
            # v_int = 0, v_face = -0.45
            # v_bnd = (2*(-0.45) + 0) / 3 = -0.30
            assert v[e.j, e.i] == pytest.approx(-0.30), (
                f"Inlet v at ({e.i},{e.j}): expected -0.30, got {v[e.j, e.i]}"
            )
            # u_face = 0 (no horizontal component)
            # u_bnd = (2*0 + 0) / 3 = 0
            assert u[e.j, e.i] == pytest.approx(0.0)

    def test_velocity_inlet_with_nonzero_interior(self, tmp_path) -> None:
        """Inlet ghost cell formula includes the interior neighbor value.

        Only checks entries whose interior neighbor is FLUID, since
        corner cells may have boundary neighbors modified by earlier
        BC application.
        """
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)

        from src.mesh import FLUID as _FLUID

        u = np.zeros((10, 10), dtype=np.float64)
        v = np.ones((10, 10), dtype=np.float64) * 0.3
        bm.apply_velocity_bc(u, v)

        top_inlets = [
            e
            for e in bm._entries
            if e.edge == "top"
            and e.bc_type == "velocity_inlet"
            and mesh.cell_type[e.nj, e.ni] == _FLUID
        ]
        assert len(top_inlets) > 0
        for e in top_inlets:
            # v_int = 0.3, v_face = -0.45
            # v_bnd = (2*(-0.45) + 0.3) / 3 = (-0.9 + 0.3) / 3 = -0.2
            assert v[e.j, e.i] == pytest.approx(-0.2), (
                f"Inlet v at ({e.i},{e.j}): expected -0.2, got {v[e.j, e.i]}"
            )

    def test_pressure_outlet_zero_gradient(self, tmp_path) -> None:
        """Pressure outlet: u_bnd = u_interior (zero gradient).

        Only checks entries whose interior neighbor is FLUID, since
        corner cells may have boundary neighbors modified by earlier
        BC application.
        """
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)

        from src.mesh import FLUID as _FLUID

        u = np.ones((10, 10), dtype=np.float64) * 2.5
        v = np.ones((10, 10), dtype=np.float64) * 1.5
        bm.apply_velocity_bc(u, v)

        outlet_entries = [
            e
            for e in bm._entries
            if e.bc_type == "pressure_outlet" and mesh.cell_type[e.nj, e.ni] == _FLUID
        ]
        assert len(outlet_entries) > 0
        for e in outlet_entries:
            # Interior FLUID cells are unchanged at 2.5/1.5
            assert u[e.j, e.i] == pytest.approx(2.5)
            assert v[e.j, e.i] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Unit tests -- Ghost cell pressure application
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPressureBC:
    """Verify ghost cell pressure formulas."""

    def test_wall_zero_gradient(self, tmp_path) -> None:
        """Wall pressure: zero gradient, p_bnd = p_interior."""
        config = _make_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "all_walls": {
                        "type": "wall",
                        "location": "top",
                        "x_start": 0.0,
                        "x_end": 1.0,
                    },
                }
            },
        )
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)

        p = np.ones((10, 10), dtype=np.float64) * 100.0
        bm.apply_pressure_bc(p)

        wall_entries = [e for e in bm._entries if e.bc_type == "wall"]
        for e in wall_entries:
            assert p[e.j, e.i] == pytest.approx(p[e.nj, e.ni])

    def test_inlet_zero_gradient(self, tmp_path) -> None:
        """Velocity inlet pressure: zero gradient, p_bnd = p_interior."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)

        p = np.ones((10, 10), dtype=np.float64) * 50.0
        bm.apply_pressure_bc(p)

        inlet_entries = [e for e in bm._entries if e.bc_type == "velocity_inlet"]
        for e in inlet_entries:
            assert p[e.j, e.i] == pytest.approx(p[e.nj, e.ni])

    def test_outlet_dirichlet_zero(self, tmp_path) -> None:
        """Pressure outlet: Dirichlet p=0, p_bnd = p_interior / 3."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)

        p = np.ones((10, 10), dtype=np.float64) * 90.0
        bm.apply_pressure_bc(p)

        outlet_entries = [e for e in bm._entries if e.bc_type == "pressure_outlet"]
        for e in outlet_entries:
            assert p[e.j, e.i] == pytest.approx(90.0 / 3.0)


# ---------------------------------------------------------------------------
# Unit tests -- Inlet flux
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInletFlux:
    """Verify volumetric flux calculation."""

    def test_full_top_inlet_flux(self, tmp_path) -> None:
        """Flux through a full-width top inlet matches V * width."""
        config = _make_config(tmp_path)  # V=0.45, width=1.0
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        flux = bm.get_inlet_flux("top_inlet")
        # 10 cells, each dx=0.1 wide, velocity=0.45
        assert flux == pytest.approx(0.45 * 1.0, rel=0.01)

    def test_nonexistent_boundary_raises(self, tmp_path) -> None:
        """get_inlet_flux raises KeyError for unknown boundary name."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        with pytest.raises(KeyError, match="not_real"):
            bm.get_inlet_flux("not_real")

    def test_outlet_flux_is_zero(self, tmp_path) -> None:
        """Flux calculation for a pressure_outlet returns zero."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        flux = bm.get_inlet_flux("bottom_outlet")
        assert flux == 0.0


# ---------------------------------------------------------------------------
# Unit tests -- Symmetry and misc
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBoundaryMisc:
    """Miscellaneous boundary tests."""

    def test_concentration_bc_raises(self, tmp_path) -> None:
        """apply_concentration_bc raises NotImplementedError."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        c = np.zeros((10, 10), dtype=np.float64)
        with pytest.raises(NotImplementedError):
            bm.apply_concentration_bc(c, 0)

    def test_symmetry_uniform_initial(self, tmp_path) -> None:
        """Symmetric domain with uniform IC produces symmetric BCs.

        Left and right wall cells should get the same ghost cell
        values when the interior field is uniform.
        """
        config = _make_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "left_wall": {
                        "type": "wall",
                        "location": "left",
                        "y_start": 0.0,
                        "y_end": 1.0,
                    },
                    "right_wall": {
                        "type": "wall",
                        "location": "right",
                        "y_start": 0.0,
                        "y_end": 1.0,
                    },
                    "top_wall": {
                        "type": "wall",
                        "location": "top",
                        "x_start": 0.0,
                        "x_end": 1.0,
                    },
                    "bot_wall": {
                        "type": "wall",
                        "location": "bottom",
                        "x_start": 0.0,
                        "x_end": 1.0,
                    },
                }
            },
        )
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)

        u = np.ones((10, 10), dtype=np.float64) * 5.0
        v = np.ones((10, 10), dtype=np.float64) * 5.0
        bm.apply_velocity_bc(u, v)

        # Left edge and right edge should have the same values
        for j in range(1, 9):
            assert u[j, 0] == pytest.approx(u[j, 9])
            assert v[j, 0] == pytest.approx(v[j, 9])

    def test_field_shapes_preserved(self, tmp_path) -> None:
        """Apply methods do not change array shape, dtype, or contiguity."""
        config = _make_config(tmp_path)
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)

        u = np.zeros((10, 10), dtype=np.float64)
        v = np.zeros((10, 10), dtype=np.float64)
        p = np.zeros((10, 10), dtype=np.float64)

        bm.apply_velocity_bc(u, v)
        bm.apply_pressure_bc(p)

        assert u.shape == (10, 10)
        assert u.dtype == np.float64
        assert u.flags["C_CONTIGUOUS"]
        assert p.shape == (10, 10)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBoundaryIntegration:
    """Verify BoundaryManager with the full default clean room config."""

    def test_default_config_initializes(self) -> None:
        """BoundaryManager initializes from default config without error."""
        config = SimConfig("configs/clean_room_default.yaml")
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        assert len(bm._entries) > 0

    def test_hepa_cells_are_inlet(self) -> None:
        """HEPA supply boundary cells are mapped as velocity_inlet."""
        config = SimConfig("configs/clean_room_default.yaml")
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        hepa_entries = [
            e for e in bm._entries if e.edge == "top" and e.bc_type == "velocity_inlet"
        ]
        assert len(hepa_entries) > 0

    def test_floor_returns_are_outlet(self) -> None:
        """Floor return vent cells are mapped as pressure_outlet."""
        config = SimConfig("configs/clean_room_default.yaml")
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        outlet_entries = [
            e
            for e in bm._entries
            if e.edge == "bottom" and e.bc_type == "pressure_outlet"
        ]
        assert len(outlet_entries) > 0

    def test_door_cells_are_wall(self) -> None:
        """Door boundary cells are mapped as wall."""
        config = SimConfig("configs/clean_room_default.yaml")
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)
        # Door is on left wall, y=[0, 2.1]
        door_entries = [
            e for e in bm._entries if e.edge == "left" and e.bc_type == "wall"
        ]
        assert len(door_entries) > 0

    def test_hepa_ghost_cell_on_zero_field(self) -> None:
        """HEPA inlet ghost cells on zero field have expected v value.

        With v_interior = 0 and V = 0.45:
        v_bnd = (2*(-0.45) + 0) / 3 = -0.30
        """
        config = SimConfig("configs/clean_room_default.yaml")
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)

        u = np.zeros((config.ny, config.nx), dtype=np.float64)
        v = np.zeros((config.ny, config.nx), dtype=np.float64)
        bm.apply_velocity_bc(u, v)

        hepa_entries = [
            e for e in bm._entries if e.edge == "top" and e.bc_type == "velocity_inlet"
        ]
        for e in hepa_entries:
            assert v[e.j, e.i] == pytest.approx(-0.30, abs=1e-10)

    def test_field_integrity_after_bc(self) -> None:
        """Fields retain correct shape, dtype, contiguity after BC apply."""
        config = SimConfig("configs/clean_room_default.yaml")
        mesh = Mesh(config)
        bm = BoundaryManager(mesh, config)

        u = np.zeros((config.ny, config.nx), dtype=np.float64)
        v = np.zeros((config.ny, config.nx), dtype=np.float64)
        p = np.zeros((config.ny, config.nx), dtype=np.float64)

        bm.apply_velocity_bc(u, v)
        bm.apply_pressure_bc(p)

        assert u.shape == (config.ny, config.nx)
        assert u.dtype == np.float64
        assert u.flags["C_CONTIGUOUS"]
