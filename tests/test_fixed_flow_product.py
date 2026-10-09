"""Tests for the fixed-flow outlets of the shipped product configuration (REQ-S18).

The product room has four floor returns that state no velocity and a hood that
states 0.5 m/s, all fixed-flow outlets (ADR-012 D, amended 2026-10-08). The
checks run the shipped configuration on three grids and a stretched mesh, since
the balance holds only because the shares are formed from the face widths the
mesh really has. Expected values are derived here from the configured segment
ranges and the cell centers, not from the registry's coverage, so a layer that
decided coverage on its own, or balanced the configured lengths instead of the
discrete ones, would fail.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.boundary_concentration import (
    SURFACE_NONE,
    ConcentrationBoundary,
    ConcentrationFaces,
)
from src.boundary_registry import FIXED_FLOW_OUTLET, BoundaryRegistry
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import FLUID, SOLID, Mesh
from src.particles import ParticlePhysics
from src.pressure import PressureCorrector
from src.staggered import allocate_fields

PRODUCT_CONFIG = Path(__file__).resolve().parents[1] / "configs/clean_room_default.yaml"
RETURNS = ("floor_return_1", "floor_return_2", "floor_return_3", "floor_return_4")
HOOD = "hood_exhaust"


def _product_raw(nx: int, ny: int, mesh: dict | None = None) -> dict:
    raw = yaml.safe_load(PRODUCT_CONFIG.read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    if mesh is not None:
        raw["mesh"] = mesh
    return raw


def _build(config: SimConfig) -> tuple[Mesh, StaggeredBoundary]:
    mesh = Mesh(config)
    return mesh, StaggeredBoundary(mesh, config)


def _product(
    nx: int, ny: int, mesh: dict | None = None
) -> tuple[SimConfig, Mesh, StaggeredBoundary]:
    config = SimConfig.from_dict(_product_raw(nx, ny, mesh))
    return (config, *_build(config))


PRODUCT_ROOMS = [
    pytest.param(40, 15, None, id="40x15"),
    pytest.param(80, 30, None, id="80x30"),
    pytest.param(200, 75, None, id="200x75"),
    pytest.param(
        80,
        30,
        {"x": {"stretch_ratio": 1.1}, "y": {"stretch_ratio": 1.2}},
        id="80x30-stretched",
    ),
]


def _segment_faces(
    config: SimConfig, mesh: Mesh, name: str
) -> tuple[str, np.ndarray, np.ndarray]:
    """Edge, boolean face mask along it, and the face widths, from the segment's range.

    Derived from the configured range and the cell centers directly. A cell
    behind a SOLID cell is a wall, so it is excluded, as the registry's rule
    says.
    """
    spec = config.boundaries[name]
    if spec.location == "bottom":
        centers, widths = mesh.xc, mesh.dx_cell
        solid = mesh.cell_type[0, :] == SOLID
        lo, hi = spec.x_start, spec.x_end
    elif spec.location == "right":
        centers, widths = mesh.yc, mesh.dy_cell
        solid = mesh.cell_type[:, -1] == SOLID
        lo, hi = spec.y_start, spec.y_end
    else:
        raise ValueError(f"the product room has no outlet on {spec.location}")
    return spec.location, (centers >= lo) & (centers <= hi) & ~solid, widths


def _supply_faces(config: SimConfig, mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Face mask along the top edge under the HEPA supply, and the face widths."""
    spec = config.boundaries["hepa_supply"]
    mask = (mesh.xc >= spec.x_start) & (mesh.xc <= spec.x_end)
    return mask & (mesh.cell_type[-1, :] != SOLID), mesh.dx_cell


def _outflow(config: SimConfig, mesh: Mesh, u: np.ndarray, v: np.ndarray) -> dict:
    """Volumetric outflow of each fixed-flow segment read off the written faces."""
    out = {}
    for name, spec in config.boundaries.items():
        if spec.type != FIXED_FLOW_OUTLET:
            continue
        edge, mask, widths = _segment_faces(config, mesh, name)
        faces = -v[0, :] if edge == "bottom" else u[:, -1]
        out[name] = float(np.sum(faces[mask] * widths[mask]))
    return out


@pytest.mark.unit
class TestProductConfiguration:
    """What the shipped file says about its outlets."""

    def test_the_shipped_product_configuration_uses_fixed_flow_outlets_only(
        self,
    ) -> None:
        config = SimConfig.from_dict(_product_raw(40, 15))
        outlets = {
            name: spec
            for name, spec in config.boundaries.items()
            if "outlet" in spec.type
        }
        assert list(outlets) == [*RETURNS, HOOD]
        assert all(spec.type == FIXED_FLOW_OUTLET for spec in outlets.values())
        assert [outlets[name].velocity for name in RETURNS] == [None] * 4
        assert outlets[HOOD].velocity == 0.5


@pytest.mark.integration
@pytest.mark.parametrize(("nx", "ny", "mesh_spec"), PRODUCT_ROOMS)
class TestProductRoomBalance:
    """The shipped configuration: four unstated returns and the hood at 0.5 m/s."""

    def _faces(
        self, nx: int, ny: int, mesh_spec: dict | None
    ) -> tuple[SimConfig, Mesh, StaggeredBoundary, np.ndarray, np.ndarray]:
        config, mesh, bc = _product(nx, ny, mesh_spec)
        u, v, _ = allocate_fields(mesh)
        u.fill(3.0)
        v.fill(3.0)
        bc.apply_normal_velocity(u, v)
        return config, mesh, bc, u, v

    def test_outflow_over_every_fixed_flow_face_equals_the_inflow(
        self, nx: int, ny: int, mesh_spec: dict | None
    ) -> None:
        config, mesh, _, u, v = self._faces(nx, ny, mesh_spec)
        supply_mask, widths = _supply_faces(config, mesh)
        inflow = float(np.sum(-v[-1, supply_mask] * widths[supply_mask]))
        out = _outflow(config, mesh, u, v)
        assert inflow == pytest.approx(0.45 * widths[supply_mask].sum(), rel=1e-14)
        assert sum(out.values()) == pytest.approx(inflow, rel=1e-14)

    def test_the_resolved_velocities_are_listed_in_configuration_order(
        self, nx: int, ny: int, mesh_spec: dict | None
    ) -> None:
        _, _, bc, _, _ = self._faces(nx, ny, mesh_spec)
        assert list(bc.fixed_flow_velocities()) == [*RETURNS, HOOD]

    def test_the_stated_hood_carries_exactly_its_velocity(
        self, nx: int, ny: int, mesh_spec: dict | None
    ) -> None:
        config, mesh, bc, u, _ = self._faces(nx, ny, mesh_spec)
        _, mask, _ = _segment_faces(config, mesh, HOOD)
        assert mask.sum() > 0
        assert np.all(u[mask, -1] == 0.5)
        assert bc.fixed_flow_velocities()[HOOD] == 0.5

    def test_the_returns_share_one_velocity_over_their_discrete_length(
        self, nx: int, ny: int, mesh_spec: dict | None
    ) -> None:
        config, mesh, bc, _, v = self._faces(nx, ny, mesh_spec)
        supply_mask, supply_widths = _supply_faces(config, mesh)
        inflow = float(np.sum(0.45 * supply_widths[supply_mask]))
        _, hood_mask, hood_widths = _segment_faces(config, mesh, HOOD)
        hood_outflow = 0.5 * float(hood_widths[hood_mask].sum())
        return_length = 0.0
        for name in RETURNS:
            _, mask, widths = _segment_faces(config, mesh, name)
            assert mask.sum() > 0
            return_length += float(widths[mask].sum())
        expected = (inflow - hood_outflow) / return_length
        for name in RETURNS:
            _, mask, _ = _segment_faces(config, mesh, name)
            assert np.all(-v[0, mask] == pytest.approx(expected, rel=1e-14))
            assert bc.fixed_flow_velocities()[name] == pytest.approx(
                expected, rel=1e-14
            )
        values = [bc.fixed_flow_velocities()[name] for name in RETURNS]
        assert len(set(values)) == 1

    def test_no_other_bottom_or_right_face_is_written(
        self, nx: int, ny: int, mesh_spec: dict | None
    ) -> None:
        config, mesh, _, u, v = self._faces(nx, ny, mesh_spec)
        covered_bottom = np.zeros(mesh.xc.shape[0], dtype=bool)
        for name in RETURNS:
            covered_bottom |= _segment_faces(config, mesh, name)[1]
        assert np.all(v[0, ~covered_bottom] == 0.0)
        _, hood_mask, _ = _segment_faces(config, mesh, HOOD)
        assert np.all(u[~hood_mask, -1] == 0.0)

    def test_every_tangential_location_on_the_outlet_edges_is_dirichlet_zero(
        self, nx: int, ny: int, mesh_spec: dict | None
    ) -> None:
        _, _, bc, _, _ = self._faces(nx, ny, mesh_spec)
        for edge in ("bottom", "right"):
            condition = bc.tangential_conditions()[edge]
            assert np.all(condition.is_dirichlet)
            assert np.all(condition.value == 0.0)

    def test_the_outlets_are_not_inlet_flux_and_no_pressure_outlet_remains(
        self, nx: int, ny: int, mesh_spec: dict | None
    ) -> None:
        _, _, bc, _, _ = self._faces(nx, ny, mesh_spec)
        assert bc.get_total_inlet_flux() == bc.get_inlet_flux("hepa_supply")
        assert bc.get_inlet_flux(HOOD) == 0.0
        assert not bc.has_pressure_outlet()

    def test_the_corrector_pins_and_takes_the_inflow_branch(
        self, nx: int, ny: int, mesh_spec: dict | None
    ) -> None:
        config, mesh, bc, _, _ = self._faces(nx, ny, mesh_spec)
        corrector = PressureCorrector(mesh, config, bc)
        assert corrector.needs_pin
        assert corrector.pin_cell == (1, 1)
        assert mesh.cell_type[1, 1] == FLUID
        assert corrector.flux_scale == pytest.approx(
            config.rho * bc.get_total_inlet_flux(), rel=1e-15
        )


@pytest.mark.integration
class TestProductConcentrationFaces:
    """A fixed-flow outlet carries nothing in and deposits nothing."""

    def _faces(
        self, nx: int, ny: int, size_class: int = 0
    ) -> tuple[SimConfig, Mesh, ConcentrationFaces]:
        config = SimConfig.from_dict(_product_raw(nx, ny))
        mesh = Mesh(config)
        boundary = ConcentrationBoundary(
            mesh, config, ParticlePhysics(config), BoundaryRegistry(config)
        )
        return config, mesh, boundary.faces_for(size_class)

    @pytest.mark.parametrize("size_class", [0, 4])
    def test_every_fixed_flow_face_carries_nothing_in_and_deposits_nothing(
        self, size_class: int
    ) -> None:
        config, mesh, faces = self._faces(80, 30, size_class)
        for name in (*RETURNS, HOOD):
            edge, mask, _ = _segment_faces(config, mesh, name)
            if edge == "bottom":
                index = (0, mask)
                inflow, deposition = faces.inflow_v, faces.deposition_v
                surface = faces.surface_v
            else:
                index = (mask, -1)
                inflow, deposition = faces.inflow_u, faces.deposition_u
                surface = faces.surface_u
            assert mask.sum() > 0
            assert np.all(inflow[index] == 0.0), name
            assert np.all(deposition[index] == 0.0), name
            assert np.all(surface[index] == SURFACE_NONE), name

    def test_the_floor_beside_the_returns_still_deposits(self) -> None:
        """The control: the zero above is the outlet's, not an all-zero array."""
        config, mesh, faces = self._faces(80, 30)
        covered = np.zeros(mesh.xc.shape[0], dtype=bool)
        for name in RETURNS:
            covered |= _segment_faces(config, mesh, name)[1]
        walls = ~covered & (mesh.cell_type[0, :] != SOLID)
        assert walls.any()
        assert np.all(faces.deposition_v[0, walls] > 0.0)
