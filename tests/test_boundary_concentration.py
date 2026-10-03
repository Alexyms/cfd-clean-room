"""Tests for the concentration boundary layer (ADR-011 E; REQ-S12.1's second reader).

The synthetic configuration has non-unit density, a non-square domain
(2.0 m by 1.2 m on 20 by 12 cells of 0.1 m), one obstacle away from the
edges, an inlet whose ends sit off the mesh nodes, and a wall segment with
deposition switched off. Expected arrays are built face by face in the
tests from the obstacle's cell extents and the segment ranges, not from the
module's own classification, and compared whole, so a face booked where
nothing should be is as visible as one missing. Two further configurations
put an obstacle on a domain edge under an inlet (test 31's cases), so the
SOLID rule is exercised and the agreement test can see a mask that differs
between the layers.
"""

import copy

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
from validation.cases import CONFIG_DIR, case_path, load_case

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


# Test 31's cases: an obstacle standing on the bottom edge under a bottom
# inlet (SOLID columns 6 to 10 of the inlet's 4 to 15), and one against the
# left edge behind a full-height left inlet (SOLID rows 3 to 7).
FLOOR_OBSTACLE_UNDER_INLET = {
    "supply": {
        "type": "velocity_inlet",
        "location": "bottom",
        "x_start": 0.37,
        "x_end": 1.63,
        "velocity": 0.3,
        "concentration": [10.0, 20.0, 30.0, 40.0, 50.0],
    },
    "exhaust": {
        "type": "pressure_outlet",
        "location": "top",
        "x_start": 0.0,
        "x_end": 2.0,
    },
}
FLOOR_BENCH = {
    "name": "bench",
    "x_start": 0.62,
    "x_end": 1.08,
    "y_start": 0.0,
    "y_end": 0.48,
}
LEFT_OBSTACLE_BEHIND_INLET = {
    "supply": {
        "type": "velocity_inlet",
        "location": "left",
        "y_start": 0.0,
        "y_end": 1.2,
        "velocity": 0.3,
        "concentration": [10.0, 20.0, 30.0, 40.0, 50.0],
    },
    "exhaust": {
        "type": "pressure_outlet",
        "location": "right",
        "y_start": 0.0,
        "y_end": 1.2,
    },
}
LEFT_CABINET = {
    "name": "cabinet",
    "x_start": 0.0,
    "x_end": 0.48,
    "y_start": 0.32,
    "y_end": 0.78,
}


def _floor_obstacle_case() -> SimConfig:
    return SimConfig.from_dict(
        _raw(FLOOR_OBSTACLE_UNDER_INLET, obstacles=[FLOOR_BENCH])
    )


def _left_obstacle_case() -> SimConfig:
    return SimConfig.from_dict(
        _raw(LEFT_OBSTACLE_BEHIND_INLET, obstacles=[LEFT_CABINET])
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

    def test_a_class_out_of_range_or_of_the_wrong_type_raises(self) -> None:
        _mesh, _physics, bc = _build(_synthetic())
        with pytest.raises(IndexError, match="size_class"):
            bc.faces_for(5)
        with pytest.raises(IndexError, match="size_class"):
            bc.faces_for(-1)
        with pytest.raises(TypeError, match="size_class must be an int"):
            bc.faces_for(True)
        with pytest.raises(TypeError, match="size_class must be an int"):
            bc.faces_for(1.0)

    def test_faces_are_not_compared_or_hashed_by_value(self) -> None:
        """eq=False, so == and hash never hit the ndarray truth value."""
        _mesh, _physics, bc = _build(_synthetic())
        a, b = bc.faces_for(0), bc.faces_for(0)
        assert a != b and a == a
        assert len({a, b}) == 2

    @pytest.mark.parametrize(
        ("edge", "surface"),
        [("right", "floor"), ("bottom", "ceiling"), ("bottom", "wall")],
    )
    def test_deposition_surface_override_reaches_exactly_the_overridden_faces(
        self, edge: str, surface: str
    ) -> None:
        """Review 31 B1 (3), test 31b B3: a wall segment booked as another surface.

        The named surface is never the edge's own default (right is wall,
        bottom is floor), so an override that is ignored shows. The segment
        covers rows 2 to 6 of the right edge or columns 7 to 11 of the
        bottom edge; those faces take the named surface and its deposition
        velocity, every other face is as in the base case.
        """
        base = _synthetic()
        override = {"type": "wall", "location": edge, "deposition_surface": surface}
        if edge == "right":
            override |= {"y_start": 0.22, "y_end": 0.68}
        else:
            override |= {"x_start": 0.72, "x_end": 1.18}
        config = SimConfig.from_dict(
            _raw(
                {
                    "supply": TOP_INLET,
                    "dead": DEAD_WALL,
                    "drain": FLOOR_OUTLET,
                    "ov": override,
                },
                obstacles=[OBSTACLE],
            )
        )
        mesh, physics, bc = _build(config)
        _mesh, _physics, plain = _build(base)
        code = {
            "floor": SURFACE_FLOOR,
            "ceiling": SURFACE_CEILING,
            "wall": SURFACE_WALL,
        }[surface]
        nx = mesh.cell_type.shape[1]
        index = (slice(2, 7), nx) if edge == "right" else (0, slice(7, 12))
        for k in (0, 4):
            got, expected = bc.faces_for(k), plain.faces_for(k)
            surfaces = {
                "u": (got.surface_u, expected.surface_u),
                "v": (got.surface_v, expected.surface_v),
            }
            depositions = {
                "u": (got.deposition_u, expected.deposition_u),
                "v": (got.deposition_v, expected.deposition_v),
            }
            own = "u" if edge == "right" else "v"
            other = "v" if own == "u" else "u"
            got_surface, plain_surface = surfaces[own]
            got_deposition, plain_deposition = depositions[own]
            assert np.all(got_surface[index] == code)
            assert np.all(plain_surface[index] != code)
            assert np.all(
                got_deposition[index] == physics.deposition_velocity(k, surface)
            )
            mask = np.zeros_like(got_surface, dtype=bool)
            mask[index] = True
            assert np.array_equal(got_surface[~mask], plain_surface[~mask])
            assert np.array_equal(got_deposition[~mask], plain_deposition[~mask])
            assert np.array_equal(*surfaces[other])
            assert np.array_equal(*depositions[other])
            assert np.array_equal(got.inflow_u, expected.inflow_u)
            assert np.array_equal(got.inflow_v, expected.inflow_v)

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


@pytest.mark.unit
class TestSolidEdgeCells:
    """Review 31 B1 (1): a SOLID cell on a domain edge has no scalar face."""

    @pytest.mark.parametrize(
        ("build", "edge_axis", "solid_run"),
        [
            (_floor_obstacle_case, "v", slice(6, 11)),
            (_left_obstacle_case, "u", slice(3, 8)),
        ],
        ids=["floor-obstacle-under-bottom-inlet", "left-obstacle-behind-left-inlet"],
    )
    def test_faces_behind_solid_edge_cells_carry_nothing(
        self, build: object, edge_axis: str, solid_run: slice
    ) -> None:
        config = build()
        mesh, _physics, bc = _build(config)
        solid_edge = mesh.cell_type[0, :] if edge_axis == "v" else mesh.cell_type[:, 0]
        expected = np.zeros(solid_edge.shape, dtype=bool)
        expected[solid_run] = True
        assert np.array_equal(solid_edge == SOLID, expected)
        for k in range(len(SIZES)):
            faces = bc.faces_for(k)
            if edge_axis == "v":
                behind = (
                    faces.surface_v[0, solid_run],
                    faces.deposition_v[0, solid_run],
                    faces.inflow_v[0, solid_run],
                )
                beside = faces.inflow_v[0, INLET_COLS]
            else:
                behind = (
                    faces.surface_u[solid_run, 0],
                    faces.deposition_u[solid_run, 0],
                    faces.inflow_u[solid_run, 0],
                )
                beside = faces.inflow_u[:, 0]
            for arr in behind:
                assert not arr.any()
            # The inlet still carries its concentration on the faces that are not behind the obstacle.
            carried = beside[beside != 0.0]
            assert carried.size == beside.size - (solid_run.stop - solid_run.start)
            assert np.all(carried == [10.0, 20.0, 30.0, 40.0, 50.0][k])

    def test_product_floor_faces_under_the_obstacles_take_no_condition(self) -> None:
        """Test 31: 98 floor faces on the product's own 200x75 grid sit under obstacles."""
        config = SimConfig(CONFIG_DIR / "clean_room_default.yaml")
        mesh, _physics, bc = _build(config)
        under = mesh.cell_type[0, :] == SOLID
        assert int(under.sum()) == 98
        faces = bc.faces_for(4)
        assert not faces.surface_v[0, under].any()
        assert not faces.deposition_v[0, under].any()
        assert not faces.inflow_v[0, under].any()
        assert np.all(
            faces.surface_v[0, ~under & (faces.surface_v[0, :] != SURFACE_NONE)]
            == SURFACE_FLOOR
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


def _normal_is_zero(spec: dict) -> bool:
    """A velocity_inlet written with components whose normal one is absent or zero."""
    if "velocity" in spec:
        return False
    normal = "v_velocity" if spec["location"] in ("bottom", "top") else "u_velocity"
    return not spec.get(normal)


def _marked_inlets(raw: dict) -> SimConfig:
    """A configuration with every air-admitting velocity_inlet carrying a unit concentration.

    ``hepa_filtered`` is cleared so the set does not depend on reference data
    (review 31 S14). A zero-normal inlet may not carry a concentration and is
    left as it is.
    """
    n = len(raw["particles"]["sizes"])
    for spec in raw["boundaries"].values():
        if spec["type"] == "velocity_inlet" and not _normal_is_zero(spec):
            spec["concentration"] = [1.0] * n
            spec["hepa_filtered"] = False
    return SimConfig.from_dict(raw)


def _agreement_configs() -> dict[str, SimConfig]:
    configs = {}
    for name, path in COMMITTED.items():
        with open(path, encoding="utf-8") as handle:
            configs[name] = _marked_inlets(yaml.safe_load(handle))
    # Deep copies: _marked_inlets edits the segment dicts it is handed.
    configs["floor-obstacle-under-inlet"] = _marked_inlets(
        copy.deepcopy(_raw(FLOOR_OBSTACLE_UNDER_INLET, obstacles=[FLOOR_BENCH]))
    )
    configs["left-obstacle-behind-inlet"] = _marked_inlets(
        copy.deepcopy(_raw(LEFT_OBSTACLE_BEHIND_INLET, obstacles=[LEFT_CABINET]))
    )
    return configs


AGREEMENT = _agreement_configs()


@pytest.mark.integration
@pytest.mark.parametrize("name", list(AGREEMENT))
def test_both_layers_agree_on_which_faces_are_inlets(name: str) -> None:
    """REQ-S12.1, the trap of prompt 31, exact on every configuration.

    S: faces the staggered layer writes a nonzero normal velocity to. T:
    faces the concentration layer marks as inlets, made visible through
    the contract by giving every air-admitting velocity_inlet segment a
    concentration. The two sets are equal: no face drives air in without a
    concentration condition, and no scalar inlet is a wall, an outlet, a
    face behind a SOLID cell or a zero-normal inlet (which is a wall to the
    scalar layer, ADR-011 E as amended). The three committed configurations
    and test 31's two SOLID-edge cases.
    """
    config = AGREEMENT[name]
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

    dispute = int((s_u ^ t_u).sum() + (s_v ^ t_v).sum())
    zero_normal = int((t_u & z_u).sum() + (t_v & z_v).sum())
    print(
        f"{name}: staggered nonzero {int(s_u.sum() + s_v.sum())}, scalar inlet "
        f"{int(t_u.sum() + t_v.sum())}, in dispute {dispute}"
    )
    assert dispute == 0
    assert zero_normal == 0
    assert np.array_equal(s_u, t_u) and np.array_equal(s_v, t_v)
    if name != "val002":
        assert int(s_u.sum() + s_v.sum()) > 0


@pytest.mark.integration
def test_the_cavity_lid_is_a_ceiling_to_the_scalar_layer() -> None:
    """Decision 2 of 2026-10-03: a zero-normal inlet deposits and admits nothing."""
    config = load_case("cavity", grid=(10, 10))
    mesh, physics, bc = _build(config)
    ny = mesh.cell_type.shape[0]
    for k in range(len(config.particle_sizes)):
        faces = bc.faces_for(k)
        assert np.all(faces.surface_v[ny, :] == SURFACE_CEILING)
        assert np.all(
            faces.deposition_v[ny, :] == physics.deposition_velocity(k, "ceiling")
        )
        assert not faces.inflow_v.any() and not faces.inflow_u.any()
        assert not faces.settling_v[ny, :].any()
