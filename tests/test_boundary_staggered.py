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

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.staggered import allocate_fields, obstacle_face_inputs
from tests.vent_cases import (
    BOX,
    DISCHARGE,
    EXHAUST,
    INTAKE,
    SHELF,
    SHELF_VENTS,
    vent_config,
    vent_raw,
)
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
            "max_pressure_iter": 5000,
            "pressure_rtol": 1.0e-8,
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
            "y_end": 0.2,
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

    def test_conditions_are_compared_and_hashed_by_identity(self) -> None:
        """eq=False, so == and hash never hit the ndarray truth value (issue 51 D2).

        Two boundaries built from the same configuration hold equal arrays in
        different objects. With the dataclass default ``==`` would compare the
        arrays and raise on their truth value; by identity it is False.
        """
        config = _config(CHANNEL)
        a = _build(config)[1].tangential_conditions()["bottom"]
        b = _build(config)[1].tangential_conditions()["bottom"]
        assert a == a
        assert a != b
        assert len({a, b}) == 2

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
            "y_end": 0.2,
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

    def test_adjacent_inlets_each_count_their_own_faces_once(self) -> None:
        """Review 31 S1 and test 31b B1: the flux is attributed by the shared coverage.

        Two inlets meeting at 0.875 on 8 cells of 0.25, a cell center that
        both ranges cover. The shared face is the first segment's (first
        match in configuration order) and carries its velocity; the second
        owns the three centers above it. A method that re-derived coverage
        with ``covers`` would count the shared face for both segments, at
        the first's velocity, and the total would exceed the face sum.
        """
        first = dict(TOP_INLET, x_start=0.3, x_end=0.875, velocity=0.2)
        second = dict(TOP_INLET, x_start=0.875, x_end=1.8, velocity=0.6)
        mesh, bc = _build(_config({"first": first, "second": second}))
        assert mesh.xc[3] == 0.875
        u, v = _filled(mesh)
        bc.apply_normal_velocity(u, v)
        assert v[-1, 3] == -0.2
        face_sum = float(np.sum(-v[-1, :] * mesh.dx_cell * (v[-1, :] != 3.0)))
        flux_first = bc.get_inlet_flux("first")
        flux_second = bc.get_inlet_flux("second")
        assert flux_first == pytest.approx(0.2 * 0.25 * 3, rel=1e-14)
        assert flux_second == pytest.approx(0.6 * 0.25 * 3, rel=1e-14)
        assert flux_first + flux_second == pytest.approx(face_sum, rel=1e-14)
        assert bc.get_total_inlet_flux() == pytest.approx(face_sum, rel=1e-14)

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
            "y_end": 0.2,
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
# The seeded validation cases
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSeededCases:
    """Inlet flux on the harness cases, from the committed configurations."""

    def test_val001_inlet_flux_is_the_exact_face_sum(self) -> None:
        """The inlet spans the full left edge because every left face is a
        storage location, so the flux is the prescribed velocity times the
        height. The retired collocated layer was two corner cells short of it
        (docs/reports/inlet_flux_comparison.md)."""
        config = load_case("poiseuille", grid=(80, 40))
        mesh = Mesh(config)
        staggered = StaggeredBoundary(mesh, config).get_inlet_flux("inlet")
        assert staggered == pytest.approx(0.1 * 0.5, rel=1e-14)

    def test_val002_reports_no_inlet_flux(self) -> None:
        config = load_case("cavity", grid=(20, 20))
        mesh = Mesh(config)
        assert StaggeredBoundary(mesh, config).get_total_inlet_flux() == 0.0


@pytest.mark.unit
class TestVents:
    """The velocity layer of the equipment vents (ADR-014 J, step 0b).

    tests/vent_cases.py places the box over cells 6 to 9 by 0 to 3 of a 16
    by 8 mesh of 0.125 m cells, so every face index below is read off by hand.
    """

    def _boundary(self, **vents: dict) -> tuple[Mesh, StaggeredBoundary]:
        config = vent_config(vents)
        mesh = Mesh(config)
        return mesh, StaggeredBoundary(mesh, config)

    def test_the_box_faces_are_the_faces_between_its_cells_and_the_room(
        self,
    ) -> None:
        mesh = Mesh(vent_config())
        box = vent_config().obstacles[0]
        top = obstacle_face_inputs(mesh, box, "top")
        assert top.component == "v"
        assert top.rows.tolist() == [4] * 4 and top.cols.tolist() == [6, 7, 8, 9]
        assert top.coordinates.tolist() == [0.8125, 0.9375, 1.0625, 1.1875]
        # The tangential stencil faces strictly over the top, not its corners.
        assert top.corner_rows.tolist() == [4] * 3
        assert top.corner_cols.tolist() == [7, 8, 9]
        left = obstacle_face_inputs(mesh, box, "left")
        assert left.component == "u"
        assert left.rows.tolist() == [0, 1, 2, 3] and left.cols.tolist() == [6] * 4
        assert left.corner_rows.tolist() == [1, 2, 3]
        right = obstacle_face_inputs(mesh, box, "right")
        assert right.cols.tolist() == [10] * 4
        # The box stands on the floor: its bottom borders no room.
        assert obstacle_face_inputs(mesh, box, "bottom").rows.size == 0
        assert BOX["y_start"] == 0.0

    def test_the_normal_velocity_is_written_on_exactly_the_covered_faces(
        self,
    ) -> None:
        mesh, plain = self._boundary()
        # The discharge returns what the intake draws, so the returns' share,
        # and every entry the plain room writes, is the same in both rooms.
        _, vented = self._boundary(discharge=DISCHARGE, intake=INTAKE)
        u0, v0, _ = allocate_fields(mesh)
        u0[:] = np.nan
        v0[:] = np.nan
        u1, v1 = u0.copy(), v0.copy()
        plain.apply_normal_velocity(u0, v0)
        vented.apply_normal_velocity(u1, v1)
        # Up out of the top, into the box through both sides.
        assert v1[4, 7] == v1[4, 8] == 0.5
        assert u1[1, 6] == u1[2, 6] == 0.5
        written_u = ~np.isnan(u1) & np.isnan(u0)
        written_v = ~np.isnan(v1) & np.isnan(v0)
        assert np.argwhere(written_u).tolist() == [[1, 6], [2, 6]]
        assert np.argwhere(written_v).tolist() == [[4, 7], [4, 8]]
        # Every entry the plain room writes, the vented room writes alike.
        same_u = ~np.isnan(u0)
        same_v = ~np.isnan(v0)
        assert np.array_equal(u0[same_u], u1[same_u])
        assert np.array_equal(v0[same_v], v1[same_v])
        # An exhaust on the right side draws into the box, toward -x.
        _, exhausted = self._boundary(exhaust=EXHAUST)
        u2, v2 = u0.copy(), v0.copy()
        exhausted.apply_normal_velocity(u2, v2)
        assert u2[1, 10] == u2[2, 10] == -0.4

    def test_the_flux_through_a_vent_is_its_velocity_times_its_span(self) -> None:
        _, boundary = self._boundary(discharge=DISCHARGE, exhaust=EXHAUST)
        assert boundary.get_inlet_flux("discharge") == 0.5 * 0.25
        assert boundary.get_inlet_flux("exhaust") == 0.0
        assert boundary.get_total_inlet_flux() == pytest.approx(0.425, rel=1e-15)
        run = {r.name: r for r in boundary.vent_runs()}
        assert run["exhaust"].widths.sum() * 0.4 == 0.1
        assert run["discharge"].is_inlet and not run["exhaust"].is_inlet

    def test_the_returns_share_the_supply_less_the_stated_outlets(self) -> None:
        """Inflow 0.3 + 0.125; stated 0.125 + 0.1; the 0.2 left over 1 m of returns."""
        _, boundary = self._boundary(
            discharge=DISCHARGE, intake=INTAKE, exhaust=EXHAUST
        )
        velocities = boundary.fixed_flow_velocities()
        assert velocities["intake"] == 0.5
        assert velocities["exhaust"] == 0.4
        assert velocities["return_left"] == pytest.approx(0.2, rel=1e-15)
        assert velocities["return_right"] == velocities["return_left"]
        # Without the vents the returns take the whole supply.
        _, plain = self._boundary()
        assert plain.fixed_flow_velocities()["return_left"] == pytest.approx(0.3)

    def test_a_discharge_sets_the_velocity_scale_and_an_exhaust_does_not(
        self,
    ) -> None:
        _, boundary = self._boundary(discharge=DISCHARGE)
        assert boundary.get_max_boundary_velocity() == 0.5
        _, exhausted = self._boundary(exhaust={**EXHAUST, "velocity": 0.9})
        assert exhausted.get_max_boundary_velocity() == 0.2

    def test_the_vent_masks_mark_the_normal_and_tangential_faces(self) -> None:
        _, boundary = self._boundary(discharge=DISCHARGE, intake=INTAKE)
        faces = boundary.vent_faces()
        assert faces.any
        assert np.argwhere(faces.normal_v).tolist() == [[4, 7], [4, 8]]
        assert np.argwhere(faces.normal_u).tolist() == [[1, 6], [2, 6]]
        # The u stencil faces at x 0.875, 1.0 and 1.125 over the top vent, the
        # v stencil faces at y 0.125, 0.25 and 0.375 beside the intake.
        assert np.argwhere(faces.tangential_u).tolist() == [[4, 7], [4, 8], [4, 9]]
        assert np.argwhere(faces.tangential_v).tolist() == [[1, 6], [2, 6], [3, 6]]
        for array in (faces.normal_u, faces.tangential_v):
            assert not array.flags.writeable
        _, plain = self._boundary()
        assert plain.vent_runs() == [] and not plain.vent_faces().any

    def test_a_vent_that_covers_no_face_of_the_mesh_is_refused(self) -> None:
        """x 0.76 to 0.8 holds no cell centre of the top (the first is 0.8125)."""
        narrow = {**DISCHARGE, "x_start": 0.76, "x_end": 0.8}
        config = vent_config({"discharge": narrow})
        with pytest.raises(ValueError, match=r"discharge: the vent's extent covers no"):
            StaggeredBoundary(Mesh(config), config)

    def test_a_closed_box_must_balance_on_the_mesh_faces(self) -> None:
        """Balanced on the configured extents, 0.35 m of intake covers 0.25 m of faces."""
        closed = {"discharge": DISCHARGE, "intake": INTAKE}
        config = vent_config(closed, supply=False, returns=False)
        StaggeredBoundary(Mesh(config), config)
        wide = {**INTAKE, "y_start": 0.1, "y_end": 0.45, "velocity": 0.125 / 0.35}
        config = vent_config(
            {"discharge": DISCHARGE, "intake": wide}, supply=False, returns=False
        )
        with pytest.raises(
            ValueError, match=r"on this mesh's faces the stated outflow"
        ):
            StaggeredBoundary(Mesh(config), config)

    def test_the_first_vent_in_configuration_order_takes_a_shared_point(
        self,
    ) -> None:
        """Two vents meeting at x 1.0: the face at 1.0 is a stencil corner, the first's."""
        first = {**DISCHARGE, "x_end": 1.0}
        second = {**DISCHARGE, "x_start": 1.0, "velocity": 0.25}
        raw = vent_raw({"first": first, "second": second})
        config = vent_config({"first": first, "second": second})
        assert list(raw["boundaries"])[-2:] == ["first", "second"]
        runs = {r.name: r for r in StaggeredBoundary(Mesh(config), config).vent_runs()}
        assert runs["first"].cols.tolist() == [7]
        assert runs["second"].cols.tolist() == [8]
        assert runs["first"].corner_cols.tolist() == [7, 8]
        assert runs["second"].corner_cols.tolist() == [9]


@pytest.mark.unit
class TestVentOrientations:
    """The vent kinds on the faces the box leaves out (prompt 61b, review 61 B1).

    tests/vent_cases.py's shelf stands clear of the floor over cells 11 to 13
    by 3 to 5, with a discharge and an exhaust under it, a discharge on each
    side and an exhaust on top.
    """

    def _boundary(self) -> tuple[Mesh, StaggeredBoundary]:
        config = vent_config(SHELF_VENTS, shelf=True)
        mesh = Mesh(config)
        return mesh, StaggeredBoundary(mesh, config)

    def test_a_floating_obstacles_bottom_face_borders_the_room_below(self) -> None:
        config = vent_config(shelf=True)
        mesh = Mesh(config)
        bottom = obstacle_face_inputs(mesh, config.obstacles[1], "bottom")
        assert config.obstacles[1].name == SHELF["name"]
        assert bottom.component == "v"
        assert bottom.rows.tolist() == [3, 3, 3]
        assert bottom.cols.tolist() == [11, 12, 13]
        assert bottom.corner_cols.tolist() == [12, 13]
        right = obstacle_face_inputs(mesh, config.obstacles[1], "right")
        assert right.cols.tolist() == [14, 14, 14] and right.rows.tolist() == [3, 4, 5]

    def test_each_vent_writes_its_velocity_with_the_sign_its_face_gives(self) -> None:
        mesh, boundary = self._boundary()
        u, v, _ = allocate_fields(mesh)
        boundary.apply_normal_velocity(u, v)
        # Down out of the bottom, up into it; out of each side; down into the top.
        assert v[3, 11] == -0.2
        assert v[3, 13] == 0.2
        assert u[4, 11] == -0.2
        assert u[4, 14] == 0.2
        assert v[6, 12] == v[6, 13] == -0.2
        assert v[3, 12] == 0.0 and v[6, 11] == 0.0 and u[3, 11] == 0.0
        # Each discharge brings 0.2 m/s over 0.125 m into the room.
        for name in ("under_discharge", "left_discharge", "right_discharge"):
            assert boundary.get_inlet_flux(name) == pytest.approx(0.025, rel=1e-15)
        velocities = boundary.fixed_flow_velocities()
        assert velocities["under_exhaust"] == velocities["top_exhaust"] == 0.2

    def test_the_vent_masks_mark_every_orientation(self) -> None:
        _, boundary = self._boundary()
        faces = boundary.vent_faces()
        assert np.argwhere(faces.normal_v).tolist() == [
            [3, 11],
            [3, 13],
            [6, 12],
            [6, 13],
        ]
        assert np.argwhere(faces.normal_u).tolist() == [[4, 11], [4, 14]]
        assert np.argwhere(faces.tangential_u).tolist() == [
            [3, 12],
            [3, 13],
            [6, 12],
            [6, 13],
        ]
        assert np.argwhere(faces.tangential_v).tolist() == [
            [4, 11],
            [4, 14],
            [5, 11],
            [5, 14],
        ]

    def test_two_obstacles_with_vents_on_one_face_name_keep_their_own_faces(
        self,
    ) -> None:
        """The box's top discharge and the shelf's top exhaust: rows 4 and 6."""
        config = vent_config({"discharge": DISCHARGE, **SHELF_VENTS}, shelf=True)
        runs = {r.name: r for r in StaggeredBoundary(Mesh(config), config).vent_runs()}
        assert runs["discharge"].rows.tolist() == [4, 4]
        assert runs["discharge"].cols.tolist() == [7, 8]
        assert runs["top_exhaust"].rows.tolist() == [6, 6]
        assert runs["top_exhaust"].cols.tolist() == [12, 13]
