"""Tests for direct boundary imposition on the staggered grid (REQ-S12).

The exactness claims are compared with ``==``, not a tolerance: a wall's
normal component is zero and an inlet's is the prescribed value, and a test
that allowed an approximation could not tell an assignment from an
interpolation. The tangential data is checked against the distance the mesh
holds, on a stretched mesh as well as a uniform one, so an implementation
that assumed dy/2 would fail.
"""

import numpy as np
import pytest

from src.boundary import BoundaryManager
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.staggered import allocate_fields
from validation.cases import load_case

STRETCHED = {"x": {"stretch_ratio": 1.1}, "y": {"stretch_ratio": 1.2}}

LEFT_INLET = {
    "type": "velocity_inlet",
    "location": "left",
    "y_start": 0.0,
    "y_end": 1.0,
    "velocity": 0.1,
}
RIGHT_OUTLET = {
    "type": "pressure_outlet",
    "location": "right",
    "y_start": 0.0,
    "y_end": 1.0,
}
TOP_INLET = {
    "type": "velocity_inlet",
    "location": "top",
    "x_start": 0.0,
    "x_end": 2.0,
    "velocity": 0.45,
}
LID = {
    "type": "velocity_inlet",
    "location": "top",
    "x_start": 0.0,
    "x_end": 2.0,
    "u_velocity": 1.0,
    "v_velocity": 0.0,
}

CHANNEL = {"inlet": LEFT_INLET, "outlet": RIGHT_OUTLET}
CAVITY = {"lid": LID}


def _config(
    boundaries: dict,
    nx: int = 8,
    ny: int = 5,
    obstacles: list[dict] | None = None,
    mesh: dict | None = None,
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
            "max_pressure_iter": 200,
            "pressure_tol": 1.0e-6,
        },
        "boundaries": boundaries,
        "obstacles": obstacles or [],
        "sensors": [{"name": "center", "x": 1.0, "y": 0.5}],
        "thresholds": {"0.1e-6": 100.0},
    }
    if mesh is not None:
        raw["mesh"] = mesh
    return SimConfig.from_dict(raw)


def _build(config: SimConfig) -> tuple[Mesh, StaggeredBoundary]:
    mesh = Mesh(config)
    return mesh, StaggeredBoundary(mesh, config)


def _filled(mesh: Mesh, fill: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Staggered u and v holding a nonzero constant, so an untouched entry is visible."""
    u, v, _p = allocate_fields(mesh)
    u.fill(fill)
    v.fill(fill)
    return u, v


# ---------------------------------------------------------------------------
# Deliverable 1: exact imposition of the normal components
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalImposition:
    """The normal component at a domain face is written, exactly."""

    def test_wall_normal_component_is_exactly_zero(self) -> None:
        mesh, bc = _build(_config(CHANNEL))
        u, v = _filled(mesh)
        bc.apply_normal_velocity(u, v)
        assert np.all(v[0, :] == 0.0)
        assert np.all(v[-1, :] == 0.0)

    def test_inlet_normal_component_equals_the_prescribed_value_exactly(self) -> None:
        mesh, bc = _build(_config({"inlet": LEFT_INLET, "top": TOP_INLET}))
        u, v = _filled(mesh)
        bc.apply_normal_velocity(u, v)
        assert np.all(u[:, 0] == 0.1)
        assert np.all(v[-1, :] == -0.45)

    def test_tangential_lid_has_zero_normal_and_an_untouched_storage_row(self) -> None:
        """A moving lid writes v = 0 on the top face and nothing into u.

        The top row of u sits below the lid, not on it. A ghost-cell scheme
        would write something there; this layer must not.
        """
        mesh, bc = _build(_config(CAVITY))
        u, v = _filled(mesh)
        bc.apply_normal_velocity(u, v)
        assert np.all(v[-1, :] == 0.0)
        assert np.all(u[-1, 1:-1] == 3.0)

    def test_outlet_normal_faces_are_not_written(self) -> None:
        mesh, bc = _build(_config(CHANNEL))
        u, v = _filled(mesh)
        bc.apply_normal_velocity(u, v)
        assert np.all(u[:, -1] == 3.0)

    def test_no_interior_entry_is_written(self) -> None:
        mesh, bc = _build(_config({"inlet": LEFT_INLET, "top": TOP_INLET}))
        u, v = _filled(mesh)
        bc.apply_normal_velocity(u, v)
        assert np.all(u[:, 1:-1] == 3.0)
        assert np.all(v[1:-1, :] == 3.0)

    def test_side_walls_win_at_the_lid_corners(self) -> None:
        """u at the lid's two ends is a side-wall normal component, so it is zero.

        The tangential data still reports the lid value at those two
        positions; the imposer does not read it, and the corner is the
        classic cavity singularity resolved in favour of the impermeable
        wall.
        """
        mesh, bc = _build(_config(CAVITY))
        u, v = _filled(mesh)
        bc.apply_normal_velocity(u, v)
        assert u[-1, 0] == 0.0
        assert u[-1, -1] == 0.0
        top = bc.tangential_conditions()["top"]
        assert top.value[0] == 1.0
        assert top.value[-1] == 1.0

    def test_solid_edge_cells_are_walls(self) -> None:
        """An obstacle on the bottom edge blocks the inlet segment it sits in."""
        bottom_inlet = {
            "type": "velocity_inlet",
            "location": "bottom",
            "x_start": 0.0,
            "x_end": 2.0,
            "velocity": 0.2,
        }
        block = {
            "name": "block",
            "x_start": 0.6,
            "x_end": 1.1,
            "y_start": 0.0,
            "y_end": 0.3,
        }
        mesh, bc = _build(_config({"inlet": bottom_inlet}, obstacles=[block]))
        solid = mesh.cell_type[0, :] == SOLID
        assert solid.any() and not solid.all()
        u, v = _filled(mesh)
        bc.apply_normal_velocity(u, v)
        assert np.all(v[0, solid] == 0.0)
        assert np.all(v[0, ~solid] == 0.2)

    def test_collocated_shapes_are_rejected(self) -> None:
        mesh, bc = _build(_config(CHANNEL))
        ny, nx = mesh.cell_type.shape
        u = np.zeros((ny, nx))
        v = np.zeros((ny, nx))
        with pytest.raises(ValueError, match="staggered shapes"):
            bc.apply_normal_velocity(u, v)

    def test_imposition_is_idempotent(self) -> None:
        mesh, bc = _build(_config({"inlet": LEFT_INLET, "top": TOP_INLET}))
        u, v = _filled(mesh)
        bc.apply_normal_velocity(u, v)
        u_once, v_once = u.copy(), v.copy()
        bc.apply_normal_velocity(u, v)
        assert np.array_equal(u, u_once)
        assert np.array_equal(v, v_once)


# ---------------------------------------------------------------------------
# Deliverable 2: tangential and pressure conditions as data
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTangentialData:
    """Value and wall distance for the component that has no storage on the wall."""

    @pytest.mark.parametrize(
        "mesh_spec", [None, STRETCHED], ids=["uniform", "stretched"]
    )
    def test_wall_distance_is_the_distance_the_mesh_holds(
        self, mesh_spec: dict | None
    ) -> None:
        mesh, bc = _build(_config(CHANNEL, mesh=mesh_spec))
        data = bc.tangential_conditions()
        assert data["bottom"].wall_distance == mesh.dy_face[0]
        assert data["top"].wall_distance == mesh.dy_face[-1]
        assert data["left"].wall_distance == mesh.dx_face[0]
        assert data["right"].wall_distance == mesh.dx_face[-1]

    def test_stretched_wall_distance_is_not_half_the_mean_spacing(self) -> None:
        """On a clustered mesh the first center is closer to the wall than dy/2."""
        mesh, bc = _build(_config(CHANNEL, mesh=STRETCHED))
        data = bc.tangential_conditions()
        assert data["bottom"].wall_distance < 0.5 * mesh.dy
        assert data["left"].wall_distance < 0.5 * mesh.dx

    def test_shapes_index_like_the_tangential_storage(self) -> None:
        mesh, bc = _build(_config(CHANNEL))
        ny, nx = mesh.cell_type.shape
        data = bc.tangential_conditions()
        for edge in ("bottom", "top"):
            assert data[edge].component == "u"
            assert data[edge].value.shape == (nx + 1,)
            assert data[edge].is_dirichlet.shape == (nx + 1,)
        for edge in ("left", "right"):
            assert data[edge].component == "v"
            assert data[edge].value.shape == (ny + 1,)
            assert data[edge].is_dirichlet.shape == (ny + 1,)

    def test_lid_prescribes_its_speed_along_the_top(self) -> None:
        _mesh, bc = _build(_config(CAVITY))
        data = bc.tangential_conditions()
        assert np.all(data["top"].is_dirichlet)
        assert np.all(data["top"].value == 1.0)
        assert np.all(data["bottom"].is_dirichlet)
        assert np.all(data["bottom"].value == 0.0)

    def test_outlet_edge_has_no_tangential_dirichlet_value(self) -> None:
        _mesh, bc = _build(_config(CHANNEL))
        right = bc.tangential_conditions()["right"]
        assert not right.is_dirichlet.any()
        assert np.all(right.value == 0.0)

    def test_inlet_edge_prescribes_zero_tangential_velocity(self) -> None:
        _mesh, bc = _build(_config(CHANNEL))
        left = bc.tangential_conditions()["left"]
        assert np.all(left.is_dirichlet)
        assert np.all(left.value == 0.0)

    def test_faces_of_solid_edge_cells_are_walls(self) -> None:
        lid = dict(LID, location="bottom")
        block = {
            "name": "block",
            "x_start": 0.6,
            "x_end": 1.1,
            "y_start": 0.0,
            "y_end": 0.3,
        }
        mesh, bc = _build(_config({"lid": lid}, obstacles=[block]))
        solid = mesh.cell_type[0, :] == SOLID
        touches_solid = np.zeros(solid.shape[0] + 1, dtype=bool)
        touches_solid[:-1] |= solid
        touches_solid[1:] |= solid
        bottom = bc.tangential_conditions()["bottom"]
        assert np.all(bottom.is_dirichlet)
        assert np.all(bottom.value[touches_solid] == 0.0)
        assert np.all(bottom.value[~touches_solid] == 1.0)

    def test_data_arrays_are_read_only(self) -> None:
        _mesh, bc = _build(_config(CHANNEL))
        top = bc.tangential_conditions()["top"]
        with pytest.raises(ValueError):
            top.value[0] = 5.0
        with pytest.raises(ValueError):
            top.is_dirichlet[0] = False


@pytest.mark.unit
class TestPressureOutletData:
    """Which edge cells are outlets; nothing is applied to a pressure array."""

    def test_outlet_edge_is_marked_and_the_others_are_not(self) -> None:
        mesh, bc = _build(_config(CHANNEL))
        ny, nx = mesh.cell_type.shape
        outlets = bc.pressure_outlets()
        assert outlets["right"].is_outlet.shape == (ny,)
        assert np.all(outlets["right"].is_outlet)
        assert outlets["right"].pressure == 0.0
        for edge in ("left", "top", "bottom"):
            assert not outlets[edge].is_outlet.any()
        assert outlets["top"].is_outlet.shape == (nx,)
        assert bc.has_pressure_outlet()

    def test_closed_domain_has_no_outlet(self) -> None:
        _mesh, bc = _build(_config(CAVITY))
        assert not bc.has_pressure_outlet()
        assert not any(o.is_outlet.any() for o in bc.pressure_outlets().values())

    def test_outlet_mask_is_read_only(self) -> None:
        _mesh, bc = _build(_config(CHANNEL))
        with pytest.raises(ValueError):
            bc.pressure_outlets()["right"].is_outlet[0] = False


# ---------------------------------------------------------------------------
# Fluxes and scales
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFluxAndScales:
    """Inlet flux from face values and face widths, with no interpolation."""

    @pytest.mark.parametrize(
        "mesh_spec", [None, STRETCHED], ids=["uniform", "stretched"]
    )
    def test_inlet_flux_is_speed_times_extent(self, mesh_spec: dict | None) -> None:
        _mesh, bc = _build(_config(CHANNEL, mesh=mesh_spec))
        assert bc.get_inlet_flux("inlet") == pytest.approx(0.1 * 1.0, rel=1e-14)

    def test_partial_inlet_counts_only_the_faces_it_covers(self) -> None:
        partial = dict(TOP_INLET, x_start=0.6, x_end=1.4)
        mesh, bc = _build(_config({"inlet": partial}))
        covered = (mesh.xc >= 0.6) & (mesh.xc <= 1.4)
        expected = 0.45 * float(np.sum(mesh.dx_cell[covered]))
        assert covered.sum() == 4
        assert bc.get_inlet_flux("inlet") == pytest.approx(expected, rel=1e-14)

    def test_tangential_lid_carries_no_flux(self) -> None:
        _mesh, bc = _build(_config(CAVITY))
        assert bc.get_inlet_flux("lid") == 0.0
        assert bc.get_total_inlet_flux() == 0.0

    def test_outlet_is_not_an_inlet(self) -> None:
        _mesh, bc = _build(_config(CHANNEL))
        assert bc.get_inlet_flux("outlet") == 0.0

    def test_solid_edge_cells_carry_no_flux(self) -> None:
        bottom_inlet = {
            "type": "velocity_inlet",
            "location": "bottom",
            "x_start": 0.0,
            "x_end": 2.0,
            "velocity": 0.2,
        }
        block = {
            "name": "block",
            "x_start": 0.6,
            "x_end": 1.1,
            "y_start": 0.0,
            "y_end": 0.3,
        }
        mesh, bc = _build(_config({"inlet": bottom_inlet}, obstacles=[block]))
        open_faces = mesh.cell_type[0, :] != SOLID
        expected = 0.2 * float(np.sum(mesh.dx_cell[open_faces]))
        assert bc.get_inlet_flux("inlet") == pytest.approx(expected, rel=1e-14)

    def test_total_flux_sums_every_inlet(self) -> None:
        _mesh, bc = _build(_config({"inlet": LEFT_INLET, "top": TOP_INLET}))
        assert bc.get_total_inlet_flux() == pytest.approx(
            0.1 * 1.0 + 0.45 * 2.0, rel=1e-14
        )

    def test_unknown_boundary_name_raises(self) -> None:
        _mesh, bc = _build(_config(CHANNEL))
        with pytest.raises(KeyError, match="not_real"):
            bc.get_inlet_flux("not_real")

    def test_max_boundary_velocity_includes_the_tangential_lid(self) -> None:
        _mesh, bc = _build(_config(CAVITY))
        assert bc.get_max_boundary_velocity() == 1.0

    def test_max_boundary_velocity_is_zero_for_walls_only(self) -> None:
        _mesh, bc = _build(_config({}))
        assert bc.get_max_boundary_velocity() == 0.0
        assert not bc.has_pressure_outlet()


# ---------------------------------------------------------------------------
# The seeded validation cases: staggered against collocated
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSeededCases:
    """Inlet flux on the harness cases, both layers from the same configuration."""

    def test_val001_staggered_flux_is_exact_and_collocated_omits_two_corner_cells(
        self,
    ) -> None:
        """The collocated map assigns corner ring cells to the bottom and top
        edges, so its left inlet is two cells short. The staggered inlet spans
        the full edge because every left face is a storage location."""
        config = load_case("poiseuille", grid=(80, 40))
        mesh = Mesh(config)
        staggered = StaggeredBoundary(mesh, config).get_inlet_flux("inlet")
        collocated = BoundaryManager(mesh, config).get_inlet_flux("inlet")
        assert staggered == pytest.approx(0.1 * 0.5, rel=1e-14)
        assert staggered - collocated == pytest.approx(2 * 0.1 * mesh.dy, rel=1e-12)

    def test_val002_both_layers_report_no_inlet_flux(self) -> None:
        config = load_case("cavity", grid=(20, 20))
        mesh = Mesh(config)
        assert StaggeredBoundary(mesh, config).get_total_inlet_flux() == 0.0
        assert BoundaryManager(mesh, config).get_total_inlet_flux() == 0.0
