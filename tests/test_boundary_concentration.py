"""Tests for the concentration boundary layer (ADR-011 E; REQ-S12.1's second reader).

The synthetic configuration has non-unit density, a non-square domain
(2.0 m by 1.2 m on 20 by 12 cells of 0.1 m), one obstacle away from the
edges, an inlet whose ends sit off the mesh nodes, and a wall segment with
deposition switched off. Expected arrays are built face by face in the
tests from the obstacle's cell extents and the segment ranges, not from the
module's own classification, and compared whole, so a face booked where
nothing should be is as visible as one missing.
"""

import numpy as np
import pytest
import yaml

from src.boundary_concentration import (
    SURFACE_CEILING,
    SURFACE_FLOOR,
    SURFACE_NONE,
    SURFACE_WALL,
    ConcentrationBoundary,
    ConcentrationFaces,
)
from src.boundary_registry import BoundaryRegistry
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.particles import ParticlePhysics
from src.staggered import allocate_fields
from validation.cases import CONFIG_DIR, case_path

SIZES = [0.1e-6, 0.3e-6, 0.5e-6, 1.0e-6, 5.0e-6]
SURFACES = ("floor", "ceiling", "wall")

# The obstacle covers cell rows 3 to 7 and columns 6 to 10 (centers at
# 0.05 + 0.1 i; the bounds sit between centers so no center is on one).
OBSTACLE = {
    "name": "bench",
    "x_start": 0.62,
    "x_end": 1.08,
    "y_start": 0.32,
    "y_end": 0.78,
}
ROWS = slice(3, 8)
COLS = slice(6, 11)

# Off-node inlet: covers cell centers 0.45 to 1.55, columns 4 to 15.
TOP_INLET = {
    "type": "velocity_inlet",
    "location": "top",
    "x_start": 0.37,
    "x_end": 1.63,
    "velocity": 0.3,
}
# Wall with deposition off: covers centers 0.05 to 0.45, rows 0 to 4.
DEAD_WALL = {
    "type": "wall",
    "location": "left",
    "y_start": 0.0,
    "y_end": 0.5,
    "deposition_surface": "none",
}
# Outlet on the floor: columns 0 to 4.
FLOOR_OUTLET = {
    "type": "pressure_outlet",
    "location": "bottom",
    "x_start": 0.0,
    "x_end": 0.5,
}
INLET_COLS = slice(4, 16)
DEAD_ROWS = slice(0, 5)
OUTLET_COLS = slice(0, 5)


def _raw(
    boundaries: dict,
    obstacles: list[dict] | None = None,
    nx: int = 20,
    ny: int = 12,
) -> dict:
    return {
        "domain": {"width": 2.0, "height": 1.2, "nx": nx, "ny": ny},
        "fluid": {"density": 1.2, "viscosity": 1.81e-5, "temperature": 293.0},
        "particles": {
            "density": 1000.0,
            "sizes": SIZES,
            "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {
                "diameters": SIZES,
                "efficiencies": [0.99999, 0.99970, 0.99990, 0.99999, 0.99999],
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
        "boundaries": boundaries,
        "obstacles": obstacles or [],
        "sensors": [{"name": "center", "x": 1.0, "y": 0.6}],
        "thresholds": {"0.1e-6": 100.0},
    }


def _synthetic() -> SimConfig:
    return SimConfig.from_dict(
        _raw(
            {"supply": TOP_INLET, "dead": DEAD_WALL, "drain": FLOOR_OUTLET},
            obstacles=[OBSTACLE],
        )
    )


def _build(
    config: SimConfig,
) -> tuple[Mesh, ParticlePhysics, ConcentrationBoundary]:
    mesh = Mesh(config)
    physics = ParticlePhysics(config)
    return (
        mesh,
        physics,
        ConcentrationBoundary(mesh, config, physics, BoundaryRegistry(config)),
    )


def _expected_surfaces(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Surface codes built from the segment ranges and the obstacle extents."""
    ny, nx = mesh.cell_type.shape
    surface_u = np.zeros((ny, nx + 1), dtype=np.int32)
    surface_v = np.zeros((ny + 1, nx), dtype=np.int32)
    surface_v[0, :] = SURFACE_FLOOR
    surface_v[0, OUTLET_COLS] = SURFACE_NONE
    surface_v[ny, :] = SURFACE_CEILING
    surface_v[ny, INLET_COLS] = SURFACE_NONE
    surface_u[:, 0] = SURFACE_WALL
    surface_u[DEAD_ROWS, 0] = SURFACE_NONE
    surface_u[:, nx] = SURFACE_WALL
    surface_v[ROWS.stop, COLS] = SURFACE_FLOOR  # above the obstacle
    surface_v[ROWS.start, COLS] = SURFACE_CEILING  # below it
    surface_u[ROWS, COLS.start] = SURFACE_WALL  # its left side
    surface_u[ROWS, COLS.stop] = SURFACE_WALL  # its right side
    return surface_u, surface_v


@pytest.mark.unit
class TestSynthetic:
    """One obstacle, an off-node inlet, a dead wall and a floor outlet."""

    def test_the_mesh_is_what_the_expectations_assume(self) -> None:
        mesh, _physics, _bc = _build(_synthetic())
        solid = mesh.cell_type == SOLID
        expected = np.zeros_like(solid)
        expected[ROWS, COLS] = True
        assert np.array_equal(solid, expected)
        assert mesh.cell_type.shape == (12, 20)

    def test_every_array_has_the_documented_shape_dtype_and_is_read_only(self) -> None:
        _mesh, _physics, bc = _build(_synthetic())
        faces = bc.faces_for(2)
        assert isinstance(faces, ConcentrationFaces)
        u_shape, v_shape = (12, 21), (13, 20)
        expected = {
            "inflow_u": (u_shape, np.float64),
            "inflow_v": (v_shape, np.float64),
            "deposition_u": (u_shape, np.float64),
            "deposition_v": (v_shape, np.float64),
            "surface_u": (u_shape, np.int32),
            "surface_v": (v_shape, np.int32),
            "settling_v": (v_shape, np.bool_),
        }
        for name, (shape, dtype) in expected.items():
            arr = getattr(faces, name)
            assert arr.shape == shape, name
            assert arr.dtype == dtype, name
            assert arr.flags["C_CONTIGUOUS"], name
            assert not arr.flags.writeable, name
            with pytest.raises(ValueError, match="read-only"):
                arr[0, 0] = 1

    def test_obstacle_and_edge_faces_take_the_surface_their_normal_gives(self) -> None:
        """Floor above the obstacle, ceiling below, walls beside; the edges by default; the dead wall off; nothing else."""
        mesh, physics, bc = _build(_synthetic())
        surface_u, surface_v = _expected_surfaces(mesh)
        for k in range(len(SIZES)):
            faces = bc.faces_for(k)
            assert np.array_equal(faces.surface_u, surface_u)
            assert np.array_equal(faces.surface_v, surface_v)
            velocity = {
                SURFACE_NONE: 0.0,
                SURFACE_FLOOR: physics.deposition_velocity(k, "floor"),
                SURFACE_CEILING: physics.deposition_velocity(k, "ceiling"),
                SURFACE_WALL: physics.deposition_velocity(k, "wall"),
            }
            lookup = np.vectorize(velocity.get)
            assert np.array_equal(faces.deposition_u, lookup(surface_u))
            assert np.array_equal(faces.deposition_v, lookup(surface_v))
        # The three orientations are distinct for the 5 um class, so a swap shows.
        floor, ceiling, wall = (physics.deposition_velocity(4, s) for s in SURFACES)
        assert floor > ceiling == wall > 0.0
        faces = bc.faces_for(4)
        assert np.all(faces.deposition_v[ROWS.stop, COLS] == floor)
        assert np.all(faces.deposition_v[ROWS.start, COLS] == ceiling)
        assert np.all(faces.deposition_u[ROWS, COLS.start] == wall)
        assert np.all(faces.deposition_u[ROWS, COLS.stop] == wall)
        assert np.all(faces.deposition_u[DEAD_ROWS, 0] == 0.0)
        assert np.all(faces.deposition_u[DEAD_ROWS.stop :, 0] == wall)

    def test_settling_mask_excludes_the_domain_top_and_bottom_and_the_obstacle(
        self,
    ) -> None:
        """Count: interior horizontal faces minus the obstacle's columns times its rows plus one."""
        mesh, _physics, bc = _build(_synthetic())
        ny, nx = mesh.cell_type.shape
        faces = bc.faces_for(0)
        expected = np.zeros((ny + 1, nx), dtype=bool)
        expected[1:ny, :] = True
        expected[ROWS.start : ROWS.stop + 1, COLS] = False
        assert np.array_equal(faces.settling_v, expected)
        n_cols = COLS.stop - COLS.start
        n_rows = ROWS.stop - ROWS.start
        assert int(faces.settling_v.sum()) == (ny - 1) * nx - n_cols * (n_rows + 1)
        assert int(faces.settling_v.sum()) == 190
        assert not faces.settling_v[0, :].any()
        assert not faces.settling_v[ny, :].any()

    def test_inflow_is_zero_everywhere_with_no_concentration_key(self) -> None:
        _mesh, _physics, bc = _build(_synthetic())
        for k in range(len(SIZES)):
            faces = bc.faces_for(k)
            assert not faces.inflow_u.any()
            assert not faces.inflow_v.any()

    def test_classes_differ_only_where_deposition_depends_on_the_class(self) -> None:
        _mesh, physics, bc = _build(_synthetic())
        first, last = bc.faces_for(0), bc.faces_for(4)
        for name in ("inflow_u", "inflow_v", "surface_u", "surface_v", "settling_v"):
            assert np.array_equal(getattr(first, name), getattr(last, name)), name
        depositing = first.surface_v != SURFACE_NONE
        assert np.array_equal(first.deposition_v == 0.0, ~depositing)
        assert np.array_equal(last.deposition_v == 0.0, ~depositing)
        assert not np.array_equal(first.deposition_v, last.deposition_v)
        ratio = physics.deposition_velocity(4, "floor") / physics.deposition_velocity(
            0, "floor"
        )
        floor = first.surface_v == SURFACE_FLOOR
        assert np.allclose(last.deposition_v[floor] / first.deposition_v[floor], ratio)

    def test_a_class_out_of_range_raises(self) -> None:
        _mesh, _physics, bc = _build(_synthetic())
        with pytest.raises(IndexError, match="size_class"):
            bc.faces_for(5)
        with pytest.raises(IndexError, match="size_class"):
            bc.faces_for(-1)

    def test_physics_built_for_another_class_count_is_refused(self) -> None:
        config = _synthetic()
        other = SimConfig.from_dict(
            _raw({"supply": TOP_INLET})
            | {
                "particles": {
                    **_raw({})["particles"],
                    "sizes": SIZES[:2],
                    "hepa_reference": {
                        "diameters": SIZES[:2],
                        "efficiencies": [0.99999, 0.99970],
                    },
                }
            }
        )
        with pytest.raises(ValueError, match="classes"):
            ConcentrationBoundary(
                Mesh(config), config, ParticlePhysics(other), BoundaryRegistry(config)
            )


def _product_raw(**supply_keys: object) -> dict:
    raw = yaml.safe_load(
        (CONFIG_DIR / "clean_room_default.yaml").read_text(encoding="utf-8")
    )
    raw["domain"]["nx"], raw["domain"]["ny"] = 100, 30
    raw["boundaries"]["hepa_supply"].update(supply_keys)
    return raw


CONCENTRATION = [1234.5, 67.89, 4.321, 0.0987, 55.55]


@pytest.mark.integration
class TestProduct:
    """The product configuration on a coarser grid (the segments are unchanged)."""

    def test_clean_supply_carries_nothing_in(self) -> None:
        """hepa_supply has hepa_filtered and no concentration: zero inflow for every class."""
        config = SimConfig.from_dict(_product_raw())
        assert config.boundaries["hepa_supply"].hepa_filtered is True
        _mesh, _physics, bc = _build(config)
        for k in range(len(SIZES)):
            faces = bc.faces_for(k)
            assert not faces.inflow_u.any()
            assert not faces.inflow_v.any()

    def test_a_filtered_supply_carries_the_value_times_one_minus_its_efficiency(
        self,
    ) -> None:
        """Five distinct non-round values, so a dropped efficiency or a class off by one shows."""
        config = SimConfig.from_dict(_product_raw(concentration=CONCENTRATION))
        mesh, physics, bc = _build(config)
        ny, nx = mesh.cell_type.shape
        supply = (mesh.xc >= 0.5) & (mesh.xc <= 7.5)
        assert 0 < supply.sum() < nx
        for k, value in enumerate(CONCENTRATION):
            faces = bc.faces_for(k)
            expected = value * (1.0 - physics.hepa_efficiency(k))
            assert expected != value
            assert np.all(faces.inflow_v[ny, supply] == expected)
            assert not faces.inflow_v[ny, ~supply].any()
            assert not faces.inflow_v[:ny, :].any()
            assert not faces.inflow_u.any()
        unfiltered = SimConfig.from_dict(
            _product_raw(concentration=CONCENTRATION, hepa_filtered=False)
        )
        _mesh, _physics, plain = _build(unfiltered)
        for k, value in enumerate(CONCENTRATION):
            assert np.all(plain.faces_for(k).inflow_v[ny, supply] == value)

    def test_floor_returns_carry_nothing_in_and_do_not_settle(self) -> None:
        config = SimConfig.from_dict(_product_raw(concentration=CONCENTRATION))
        mesh, _physics, bc = _build(config)
        faces = bc.faces_for(4)
        returns = np.zeros(mesh.xc.shape[0], dtype=bool)
        for name, spec in config.boundaries.items():
            if name.startswith("floor_return"):
                returns |= (mesh.xc >= spec.x_start) & (mesh.xc <= spec.x_end)
        assert returns.any()
        assert not faces.inflow_v[0, returns].any()
        assert not faces.settling_v[0, :].any()
        assert np.all(faces.deposition_v[0, returns] == 0.0)
        assert np.all(faces.surface_v[0, returns] == SURFACE_NONE)


COMMITTED = {
    "product": CONFIG_DIR / "clean_room_default.yaml",
    "val001": case_path("poiseuille"),
    "val002": case_path("cavity"),
}


def _marked_inlets(path: object) -> SimConfig:
    """The committed file with every velocity_inlet carrying a unit concentration."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    n = len(raw["particles"]["sizes"])
    for spec in raw["boundaries"].values():
        if spec["type"] == "velocity_inlet":
            spec["concentration"] = [1.0] * n
    return SimConfig.from_dict(raw)


@pytest.mark.integration
@pytest.mark.parametrize("name", list(COMMITTED))
def test_both_layers_agree_on_which_faces_are_inlets(name: str) -> None:
    """REQ-S12.1, the trap of prompt 31.

    S: faces the staggered layer writes a nonzero normal velocity to. T:
    faces the concentration layer marks as inlets, made visible through
    the contract by giving every velocity_inlet segment a concentration.
    Z: faces the staggered layer writes exactly zero to (walls, SOLID edge
    cells, and inlets with no normal component such as the cavity lid).
    S must lie inside T, so no face drives air in without a concentration
    condition, and T inside S or Z, so no scalar inlet is an outlet or an
    unwritten face. The faces in T but not S are the zero-normal inlets,
    reported; on the product case and the channel there are none and the
    two sets are equal.
    """
    config = _marked_inlets(COMMITTED[name])
    mesh = Mesh(config)
    staggered = StaggeredBoundary(mesh, config)
    u, v, _p = allocate_fields(mesh)
    u[:] = np.nan
    v[:] = np.nan
    staggered.apply_normal_velocity(u, v)
    written_u, written_v = ~np.isnan(u), ~np.isnan(v)
    s_u, s_v = written_u & (u != 0.0), written_v & (v != 0.0)
    z_u, z_v = written_u & (u == 0.0), written_v & (v == 0.0)

    faces = ConcentrationBoundary(
        mesh, config, ParticlePhysics(config), BoundaryRegistry(config)
    ).faces_for(0)
    t_u, t_v = faces.inflow_u != 0.0, faces.inflow_v != 0.0

    driven_without_condition = int((s_u & ~t_u).sum() + (s_v & ~t_v).sum())
    condition_without_dirichlet = int(
        (t_u & ~(s_u | z_u)).sum() + (t_v & ~(s_v | z_v)).sum()
    )
    zero_normal_inlets = int((t_u & z_u).sum() + (t_v & z_v).sum())
    print(
        f"{name}: staggered nonzero {int(s_u.sum() + s_v.sum())}, scalar inlet "
        f"{int(t_u.sum() + t_v.sum())}, zero-normal inlet faces {zero_normal_inlets}, "
        f"in dispute {driven_without_condition + condition_without_dirichlet}"
    )
    assert driven_without_condition == 0
    assert condition_without_dirichlet == 0
    assert np.array_equal(s_u, t_u & ~z_u)
    assert np.array_equal(s_v, t_v & ~z_v)
    if name != "val002":
        assert zero_normal_inlets == 0
        assert np.array_equal(s_u, t_u) and np.array_equal(s_v, t_v)
        assert int(s_u.sum() + s_v.sum()) > 0
