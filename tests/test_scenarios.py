"""Tests for the scenario manager (ADR-014 step 2, src/scenarios.py).

The manager turns a loaded scenario into its intervals, its distinct flow
states, the source rates at a time, an interval's concentration boundary and
the initial fields. Nothing here solves a flow: the fingerprint is a property
of the configuration, and the one check that reads the mesh's faces
(``test_the_joined_supply_is_the_base_velocity_layer_to_rounding``) writes the
boundary's normal velocities and compares them. Every expected number is
derived from the configured values and the mesh's own cell sizes, not from the
manager.
"""

import dataclasses
import logging
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.boundary_registry import UNCOVERED, BoundaryRegistry
from src.boundary_staggered import StaggeredBoundary
from src.config import MESH_BALANCE_TOLERANCE, BoundarySpec, SimConfig
from src.mesh import SOLID, Mesh
from src.scenarios import (
    FLOW_KEYS,
    TRANSPORT_KEYS,
    ScenarioManager,
    flow_fingerprint,
    flow_segments,
)
from src.staggered import allocate_fields, obstacle_face_inputs

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
DEFAULT = CONFIG_DIR / "clean_room_default.yaml"
VENT_FAILURES = Path(__file__).resolve().parent / "scenario_vent_failures.yaml"
# The reference scenarios and their stated flow state counts (ADR-014 B).
REFERENCE_STATES = {
    "scenario_door_leak.yaml": 2,
    "scenario_filter_breach.yaml": 1,
    "scenario_equipment_dust.yaml": 1,
    "scenario_clean_out.yaml": 1,
}
# The coarse product grid: every stated position is a multiple of 0.05 m, so
# 160x60 puts each on a cell face (ECR-002 section 8).
COARSE = (160, 60)
EMISSION = 1.0e4


def _door_open(name: str = "door_open", start: float = 10.0) -> dict:
    """The door's two-way exchange (configs/scenario_door_leak.yaml) as an event."""
    path = CONFIG_DIR / "scenario_door_leak.yaml"
    door = yaml.safe_load(path.read_text(encoding="utf-8"))
    event = door["scenario"]["events"][0]
    return {**event, "name": name, "start": start}


def _scenario(
    tmp_path: Path,
    events: list,
    grid: tuple[int, int] | None = COARSE,
    base: Path = DEFAULT,
) -> SimConfig:
    """A scenario over the default room (or ``base``), its grid overridden."""
    raw: dict = {
        "scenario": {"base": str(base), "events": events},
        "solver": {"t_end": 60.0, "output_interval": 100},
        "transport": {"cfl_number": 0.4},
    }
    if grid is not None:
        raw["domain"] = {"nx": grid[0], "ny": grid[1]}
    return SimConfig.from_dict(raw, base_dir=tmp_path)


def _manager(config: SimConfig) -> ScenarioManager:
    return ScenarioManager(config, Mesh(config))


def _source(start: float, duration: float, name: str = "dust", **box: float) -> dict:
    body = {"x_start": 3.80, "x_end": 3.90, "y_start": 2.00, "y_end": 2.10}
    body.update(box)
    return {
        "name": name,
        "start": start,
        "duration": duration,
        "source": {**body, "emission": [EMISSION] * 5},
    }


def _set(name: str, start: float, segment: str, **keys: object) -> dict:
    return {
        "name": name,
        "start": start,
        "duration": 10.0,
        "set": {"segment": segment, "keys": keys},
    }


def _all_keys_segment() -> BoundarySpec:
    """A segment with every field set, for the per-key fingerprint checks."""
    return BoundarySpec(
        type="velocity_inlet",
        location="top",
        x_start=0.5,
        x_end=1.0,
        y_start=0.25,
        y_end=0.75,
        velocity=0.45,
        u_velocity=0.1,
        v_velocity=-0.45,
        concentration=(1.0, 2.0),
        hepa_filtered=True,
        deposition_surface="ceiling",
        turbulence_intensity=0.05,
        dissipation_length=0.1,
        obstacle="rack",
        face="top",
        recirculate_from="rack_intake",
    )


def _another(value: object) -> object:
    """A different value of the same kind."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, float):
        return value * 1.5 + 0.01
    if isinstance(value, tuple):
        return tuple(v + 1.0 for v in value)
    return f"{value}_other"


# The join's equality clause compares every flow key but the extents.
JOIN_KEYS: tuple[str, ...] = tuple(
    k for k in FLOW_KEYS if k not in ("x_start", "x_end", "y_start", "y_end")
)
# Values that keep a piece's extent on the x axis, so a changed piece still
# meets its neighbour end to start.
_JOIN_OTHER: dict[str, str] = {
    "type": "fixed_flow_outlet",
    "location": "bottom",
    "obstacle": "tool",
    "face": "bottom",
}


def _join_other(key: str, value: object) -> object:
    """Another value of ``key`` that leaves the extent's axis unchanged."""
    return _JOIN_OTHER.get(key, _another(value))


def _vent(obstacle: str, face: str, x_start: float, x_end: float) -> BoundarySpec:
    return BoundarySpec(
        type="velocity_inlet",
        location="obstacle",
        obstacle=obstacle,
        face=face,
        x_start=x_start,
        x_end=x_end,
        velocity=0.5,
        turbulence_intensity=0.1,
        dissipation_length=0.06,
    )


def _supply(x_start: float, x_end: float, velocity: float = 0.45) -> BoundarySpec:
    return BoundarySpec(
        type="velocity_inlet",
        location="top",
        x_start=x_start,
        x_end=x_end,
        velocity=velocity,
        turbulence_intensity=0.05,
        dissipation_length=0.1,
    )


def _stand_in(*segments: BoundarySpec) -> SimpleNamespace:
    """What flow_fingerprint reads of a configuration: its boundaries, in order."""
    return SimpleNamespace(boundaries={f"s{i}": s for i, s in enumerate(segments)})


def _fingerprint_of(*segments: BoundarySpec) -> str:
    return flow_fingerprint(_stand_in(*segments))  # type: ignore[arg-type]


@pytest.mark.unit
class TestIntervals:
    """An event is active from its start and has expired at its end (ADR-014 A)."""

    def test_activation_at_the_start_and_expiry_at_the_end(
        self, tmp_path: Path
    ) -> None:
        manager = _manager(_scenario(tmp_path, [_door_open()]))
        spans = [(i.t_start, i.t_end, i.events) for i in manager.intervals()]
        assert spans == [
            (0.0, 10.0, ()),
            (10.0, 15.0, ("door_open",)),
            (15.0, 60.0, ()),
        ]
        closed, opened, closed_again = manager.intervals()
        assert "door" in closed.mapping["boundaries"]
        assert "door_in" not in closed.mapping["boundaries"]
        assert {"door_in", "door_out"} <= set(opened.mapping["boundaries"])
        assert "door" not in opened.mapping["boundaries"]
        assert closed_again.mapping == closed.mapping
        assert opened.config.boundaries["door_in"].velocity == 0.25

    def test_a_source_is_active_from_its_start_to_its_end(self, tmp_path: Path) -> None:
        manager = _manager(_scenario(tmp_path, [_source(10.0, 30.0)]))
        for t, on in [(0.0, False), (9.999, False), (10.0, True), (39.999, True)]:
            assert bool(manager.sources_at(t)[4].any()) is on, t
        assert not manager.sources_at(40.0)[4].any()
        assert [(i.t_start, i.t_end) for i in manager.intervals()] == [
            (0.0, 10.0),
            (10.0, 40.0),
            (40.0, 60.0),
        ]

    def test_the_intervals_cover_the_run_without_gaps(self, tmp_path: Path) -> None:
        events = [_door_open(), _source(12.0, 30.0), _door_open("again", 30.0)]
        intervals = _manager(_scenario(tmp_path, events)).intervals()
        assert intervals[0].t_start == 0.0
        assert intervals[-1].t_end == 60.0
        for a, b in pairwise(intervals):
            assert a.t_end == b.t_start
        assert [i.t_start for i in intervals] == [
            0.0,
            10.0,
            12.0,
            15.0,
            30.0,
            35.0,
            42.0,
        ]

    def test_overlapping_events_on_one_segment_are_refused(
        self, tmp_path: Path
    ) -> None:
        """The manager's input never holds them: the loader refuses at load."""
        with pytest.raises(ValueError, match=r"both change segment 'door' and overlap"):
            _scenario(tmp_path, [_door_open(), _door_open("again", 12.0)])

    def test_intervals_are_new_objects_on_every_call(self, tmp_path: Path) -> None:
        manager = _manager(_scenario(tmp_path, [_door_open()]))
        manager.intervals()[1].mapping["boundaries"].clear()
        assert "door_in" in manager.intervals()[1].mapping["boundaries"]

    def test_a_configuration_without_a_scenario_is_refused(self) -> None:
        config = SimConfig(DEFAULT)
        with pytest.raises(ValueError, match=r"needs a configuration with a scenario"):
            ScenarioManager(config, Mesh(config))

    def test_a_mesh_of_another_grid_is_refused(self, tmp_path: Path) -> None:
        config = _scenario(tmp_path, [_door_open()])
        other = SimConfig(DEFAULT)
        with pytest.raises(ValueError, match=r"the mesh is 320x120 but .* 160x60"):
            ScenarioManager(config, Mesh(other))


@pytest.mark.unit
class TestFlowStates:
    """The fingerprint: the velocity layer's keys, in configuration order."""

    def test_two_openings_give_two_states_not_four(self, tmp_path: Path) -> None:
        events = [_door_open(), _door_open("door_again", 30.0)]
        manager = _manager(_scenario(tmp_path, events))
        prints = [i.fingerprint for i in manager.intervals()]
        assert len(prints) == 5
        closed, opened = prints[0], prints[1]
        assert closed != opened
        assert prints == [closed, opened, closed, opened, closed]
        assert manager.n_flow_states == 2
        states = manager.flow_states()
        assert list(states) == [closed, opened]
        assert "door_in" in states[opened]["boundaries"]
        assert flow_fingerprint(SimConfig.from_dict(states[opened])) == opened

    def test_the_count_is_reported_when_the_manager_is_built(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = _scenario(tmp_path, [_door_open(), _door_open("again", 30.0)])
        with caplog.at_level(logging.INFO, logger="src.scenarios"):
            _manager(config)
        assert "5 intervals, 2 distinct flow states" in caplog.text
        # A mapping has no file of its own; the line names its base (review 62 S4).
        assert "scenario on clean_room_default.yaml: 5 intervals" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="src.scenarios"):
            _manager(SimConfig(CONFIG_DIR / "scenario_door_leak.yaml"))
        assert "scenario scenario_door_leak.yaml: 3 intervals" in caplog.text

    def test_a_breach_of_transport_keys_keeps_the_bases_fingerprint(
        self, tmp_path: Path
    ) -> None:
        breach = _set(
            "breach",
            10.0,
            "hepa_supply",
            hepa_filtered=False,
            concentration=[1.0e3] * 5,
        )
        manager = _manager(_scenario(tmp_path, [breach]))
        assert len({i.fingerprint for i in manager.intervals()}) == 1
        assert manager.n_flow_states == 1
        assert manager.intervals()[0].fingerprint == flow_fingerprint(
            SimConfig(DEFAULT)
        )

    def test_a_change_of_the_supplys_turbulence_intensity_makes_a_new_state(
        self, tmp_path: Path
    ) -> None:
        event = _set("gusty", 10.0, "hepa_supply", turbulence_intensity=0.1)
        assert _manager(_scenario(tmp_path, [event])).n_flow_states == 2

    def test_reordering_two_segments_makes_a_new_state(self) -> None:
        raw = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
        same = SimConfig.from_dict(raw)
        boundaries = list(raw["boundaries"].items())
        i, j = (
            [name for name, _ in boundaries].index(n)
            for n in ("floor_return_1", "floor_return_2")
        )
        boundaries[i], boundaries[j] = boundaries[j], boundaries[i]
        raw["boundaries"] = dict(boundaries)
        swapped = SimConfig.from_dict(raw)
        assert set(swapped.boundaries) == set(same.boundaries)
        assert flow_fingerprint(same) == flow_fingerprint(SimConfig(DEFAULT))
        assert flow_fingerprint(swapped) != flow_fingerprint(same)

    def test_every_segment_key_is_a_flow_key_or_a_transport_key(self) -> None:
        """A key added to the segments must be classified here before it loads."""
        assert set(FLOW_KEYS).isdisjoint(TRANSPORT_KEYS)
        fields = {f.name for f in dataclasses.fields(BoundarySpec)}
        assert set(FLOW_KEYS) | set(TRANSPORT_KEYS) == fields

    @pytest.mark.parametrize("key", FLOW_KEYS)
    def test_a_change_of_each_flow_key_makes_a_new_state(self, key: str) -> None:
        spec = _all_keys_segment()
        changed = dataclasses.replace(spec, **{key: _another(getattr(spec, key))})
        assert _fingerprint_of(spec) != _fingerprint_of(changed)

    @pytest.mark.parametrize("key", TRANSPORT_KEYS)
    def test_a_change_of_each_transport_key_keeps_the_state(self, key: str) -> None:
        spec = _all_keys_segment()
        changed = dataclasses.replace(spec, **{key: _another(getattr(spec, key))})
        assert _fingerprint_of(spec) == _fingerprint_of(changed)

    def test_contiguous_pieces_of_one_segment_join(self) -> None:
        whole = _supply(0.5, 7.5)
        pieces = [_supply(0.5, 3.5), _supply(3.5, 4.1), _supply(4.1, 7.5)]
        assert _fingerprint_of(*pieces) == _fingerprint_of(whole)
        assert flow_segments(_stand_in(*pieces)) == flow_segments(_stand_in(whole))

    @pytest.mark.parametrize(
        "pieces",
        [
            pytest.param(
                [_supply(0.5, 3.5), _supply(3.5, 4.1, velocity=0.5), _supply(4.1, 7.5)],
                id="another-velocity",
            ),
            pytest.param(
                [_supply(0.5, 3.5), _supply(3.55, 7.5)],
                id="a-gap",
            ),
            pytest.param(
                [_supply(3.5, 7.5), _supply(0.5, 3.5)],
                id="out-of-order",
            ),
            pytest.param(
                [
                    _supply(0.5, 3.5),
                    BoundarySpec(type="wall", location="left", y_start=0.0, y_end=2.1),
                    _supply(3.5, 7.5),
                ],
                id="not-adjacent-in-order",
            ),
        ],
    )
    def test_pieces_that_are_not_one_segment_stay_apart(
        self, pieces: list[BoundarySpec]
    ) -> None:
        whole = [_supply(0.5, 7.5)] + [p for p in pieces if p.type == "wall"]
        assert _fingerprint_of(*pieces) != _fingerprint_of(*whole)

    @pytest.mark.parametrize("key", JOIN_KEYS)
    def test_pieces_that_differ_in_one_flow_key_stay_apart(self, key: str) -> None:
        """Review and test 62 B1: each key of the join's equality clause, alone.

        The two pieces meet end to start on one axis and differ in ``key`` only,
        so that key's comparison is the one thing that keeps them apart.
        """
        first = dataclasses.replace(_all_keys_segment(), x_start=0.5, x_end=1.0)
        second = dataclasses.replace(
            first,
            x_start=1.0,
            x_end=1.5,
            **{key: _join_other(key, getattr(first, key))},
        )
        joined = dataclasses.replace(first, x_end=1.5)
        assert len(flow_segments(_stand_in(first, second))) == 2
        assert _fingerprint_of(first, second) != _fingerprint_of(joined)
        # Control: equal in that key too, the same two pieces join.
        same = dataclasses.replace(second, **{key: getattr(first, key)})
        assert _fingerprint_of(first, same) == _fingerprint_of(joined)

    @pytest.mark.parametrize(
        ("first", "second", "joined"),
        [
            pytest.param(
                _supply(0.5, 3.5),
                dataclasses.replace(_supply(3.5, 7.5), location="bottom"),
                _supply(0.5, 7.5),
                id="two-edges",
            ),
            pytest.param(
                _vent("rack", "top", 1.0, 1.5),
                _vent("rack", "bottom", 1.5, 2.0),
                _vent("rack", "top", 1.0, 2.0),
                id="one-obstacle-two-faces",
            ),
            pytest.param(
                _vent("rack", "top", 1.0, 1.5),
                _vent("tool", "top", 1.5, 2.0),
                _vent("rack", "top", 1.0, 2.0),
                id="one-face-two-obstacles",
            ),
        ],
    )
    def test_pieces_on_two_edges_or_faces_stay_apart(
        self, first: BoundarySpec, second: BoundarySpec, joined: BoundarySpec
    ) -> None:
        assert len(flow_segments(_stand_in(first, second))) == 2
        assert _fingerprint_of(first, second) != _fingerprint_of(joined)

    @pytest.mark.parametrize("name", sorted(REFERENCE_STATES))
    def test_each_reference_scenario_gives_its_stated_number_of_states(
        self, name: str
    ) -> None:
        config = SimConfig(CONFIG_DIR / name)
        assert _manager(config).n_flow_states == REFERENCE_STATES[name]


@pytest.mark.integration
def test_the_joined_supply_is_the_base_velocity_layer_to_rounding() -> None:
    """The premise of the joining rule, on the breach's own split (160x60).

    Every face condition is the base's; the one difference is the rounding of
    the inflow summed over three segments, which the returns' shared velocity
    carries.
    """
    spec = SimConfig(CONFIG_DIR / "scenario_filter_breach.yaml").scenario
    layers = []
    for events in ((), spec.active(10.0)):
        raw = spec.mapping(events)
        raw["domain"]["nx"], raw["domain"]["ny"] = COARSE
        config = SimConfig.from_dict(raw)
        mesh = Mesh(config)
        boundary = StaggeredBoundary(mesh, config)
        u, v, _ = allocate_fields(mesh)
        boundary.apply_normal_velocity(u, v)
        layers.append((u, v, boundary))
    (u0, v0, b0), (u1, v1, b1) = layers
    np.testing.assert_array_equal(u1, u0)
    np.testing.assert_allclose(v1, v0, rtol=MESH_BALANCE_TOLERANCE, atol=0.0)
    # The faces that differ at all are on the floor, the returns' row.
    assert set(np.nonzero(v1 != v0)[0].tolist()) <= {0}
    assert b1.get_total_inlet_flux() == pytest.approx(
        b0.get_total_inlet_flux(), rel=MESH_BALANCE_TOLERANCE
    )
    for edge, walls in b0.wall_faces().items():
        assert np.array_equal(b1.wall_faces()[edge].cells, walls.cells)
        assert np.array_equal(b1.wall_faces()[edge].corners, walls.corners)
    for edge, cond in b0.tangential_conditions().items():
        assert np.array_equal(b1.tangential_conditions()[edge].value, cond.value)
    # The hood states its velocity; the returns share the rounded remainder.
    shares = b0.fixed_flow_velocities()
    assert b1.fixed_flow_velocities()["hood_exhaust"] == shares["hood_exhaust"]
    for name in (f"floor_return_{n}" for n in range(1, 5)):
        assert b1.fixed_flow_velocities()[name] == pytest.approx(
            shares[name], rel=MESH_BALANCE_TOLERANCE
        )


@pytest.mark.unit
class TestSources:
    """A source's rate over its box's non-SOLID cells (ADR-014 B)."""

    @pytest.mark.parametrize("grid", [COARSE, (320, 120)], ids=["160x60", "320x120"])
    def test_a_sources_rate_summed_over_its_box_equals_its_emission(
        self, tmp_path: Path, grid: tuple[int, int]
    ) -> None:
        config = _scenario(tmp_path, [_source(10.0, 30.0)], grid)
        mesh = Mesh(config)
        rates = ScenarioManager(config, mesh).sources_at(10.0)
        volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        for k in range(5):
            assert float((rates[k] * volume).sum()) == pytest.approx(
                EMISSION, rel=1e-12
            )

    def test_the_rate_is_uniform_over_the_boxs_cells_and_zero_elsewhere(
        self, tmp_path: Path
    ) -> None:
        config = _scenario(tmp_path, [_source(10.0, 30.0)], (320, 120))
        mesh = Mesh(config)
        rate = ScenarioManager(config, mesh).sources_at(20.0)[2]
        inside = rate > 0
        # The 0.1 m square on 2.5 cm cells: sixteen cells.
        assert inside.sum() == 16
        assert np.all(rate[inside] == rate[inside][0])
        assert rate[inside][0] == pytest.approx(EMISSION / 0.01, rel=1e-12)
        cols = np.nonzero(inside.any(axis=0))[0]
        rows = np.nonzero(inside.any(axis=1))[0]
        assert np.all((mesh.xc[cols] > 3.80) & (mesh.xc[cols] < 3.90))
        assert np.all((mesh.yc[rows] > 2.00) & (mesh.yc[rows] < 2.10))
        assert rate.dtype == np.float64 and rate.flags.c_contiguous

    def test_a_box_across_an_obstacle_uses_its_open_cells_only(
        self, tmp_path: Path
    ) -> None:
        """Half the box lies inside the litho tool: the emission is the open half's."""
        config = _scenario(tmp_path, [_source(10.0, 30.0, y_start=1.90)], (320, 120))
        mesh = Mesh(config)
        rates = ScenarioManager(config, mesh).sources_at(10.0)
        volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        solid = mesh.cell_type == SOLID
        assert not rates[0][solid].any()
        assert (rates[0] > 0).sum() == 16
        assert float((rates[0] * volume).sum()) == pytest.approx(EMISSION, rel=1e-12)

    @pytest.mark.parametrize("form", ["emission", "rate"])
    def test_each_class_carries_its_own_value(self, tmp_path: Path, form: str) -> None:
        """Review 62 B2: five different values, so a class can only match its own."""
        values = [1.0e3, 2.0e3, 3.0e3, 5.0e3, 8.0e3]
        event = _source(10.0, 30.0)
        del event["source"]["emission"]
        event["source"][form] = values
        config = _scenario(tmp_path, [event], (320, 120))
        mesh = Mesh(config)
        rates = ScenarioManager(config, mesh).sources_at(10.0)
        volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        inside = rates[0] > 0
        # The 0.1 m square on 2.5 cm cells: sixteen cells of 6.25e-4 m^2.
        assert inside.sum() == 16
        for k, value in enumerate(values):
            assert np.array_equal(rates[k] > 0, inside), k
            if form == "emission":
                assert float((rates[k] * volume).sum()) == pytest.approx(
                    value, rel=1e-12
                )
                assert np.all(
                    rates[k][inside] == pytest.approx(value / 0.01, rel=1e-12)
                )
            else:
                assert np.all(rates[k][inside] == value), k
                assert float((rates[k] * volume).sum()) == pytest.approx(
                    value * 0.01, rel=1e-12
                )

    def test_a_rate_is_applied_as_stated(self, tmp_path: Path) -> None:
        event = _source(10.0, 30.0)
        event["source"]["rate"] = [7.0] * 5
        del event["source"]["emission"]
        rate = _manager(_scenario(tmp_path, [event])).sources_at(10.0)[1]
        assert set(np.unique(rate)) == {0.0, 7.0}

    def test_two_active_sources_sum(self, tmp_path: Path) -> None:
        events = [_source(10.0, 30.0), _source(20.0, 30.0, "more")]
        manager = _manager(_scenario(tmp_path, events))
        one = manager.sources_at(15.0)[3]
        both = manager.sources_at(25.0)[3]
        np.testing.assert_array_equal(both, 2.0 * one)
        np.testing.assert_array_equal(manager.sources_at(45.0)[3], one)

    def test_every_class_has_an_array_when_nothing_is_active(
        self, tmp_path: Path
    ) -> None:
        sources = _manager(_scenario(tmp_path, [_source(10.0, 30.0)])).sources_at(5.0)
        assert sorted(sources) == [0, 1, 2, 3, 4]
        assert all(a.shape == (60, 160) and not a.any() for a in sources.values())

    def test_a_box_with_no_open_cell_centre_is_refused(self, tmp_path: Path) -> None:
        """Inside the litho tool (x 3.2 to 4.5, y 0 to 2): the rule needs the mesh."""
        config = _scenario(tmp_path, [_source(10.0, 30.0, y_start=1.0, y_end=1.5)])
        with pytest.raises(ValueError, match=r"'dust': the source's box .* holds no"):
            _manager(config)


@pytest.mark.integration
class TestFieldsAndConditions:
    """The initial fields and an interval's concentration boundary."""

    def test_the_clean_outs_initial_field(self) -> None:
        config = SimConfig(CONFIG_DIR / "scenario_clean_out.yaml")
        mesh = Mesh(config)
        fields = ScenarioManager(config, mesh).initial_fields()
        solid = mesh.cell_type == SOLID
        for k, threshold in enumerate(config.class_thresholds()):
            assert fields[k].shape == solid.shape
            assert np.all(fields[k][~solid] == 100.0 * threshold)
            assert not fields[k][solid].any()

    def test_without_an_initial_event_the_fields_are_zero(self, tmp_path: Path) -> None:
        fields = _manager(_scenario(tmp_path, [_door_open()])).initial_fields()
        assert sorted(fields) == [0, 1, 2, 3, 4]
        assert not any(f.any() for f in fields.values())

    def test_the_breachs_interval_carries_its_panels_concentration(self) -> None:
        config = SimConfig(CONFIG_DIR / "scenario_filter_breach.yaml")
        mesh = Mesh(config)
        manager = ScenarioManager(config, mesh)
        before, during, _ = manager.intervals()
        panel = config.scenario.events[0].change.segments[1][1]
        for k in range(5):
            top = manager.conditions_for(during).faces_for(k).inflow_v[-1, :]
            covered = (mesh.xc > 3.5) & (mesh.xc < 4.1)
            np.testing.assert_array_equal(top[covered], panel["concentration"][k])
            assert not top[~covered].any()
            assert not manager.conditions_for(before).faces_for(k).inflow_v.any()


@pytest.mark.integration
class TestVentFailures:
    """The vented room's fan stop and exhaust failure (tests/scenario_vent_failures.yaml)."""

    def test_the_fixture_loads_and_gives_three_flow_states(self) -> None:
        config = SimConfig(VENT_FAILURES)
        manager = _manager(config)
        spans = [(i.t_start, i.t_end, i.events) for i in manager.intervals()]
        assert spans == [
            (0.0, 10.0, ()),
            (10.0, 20.0, ("fan_stops_discharge", "fan_stops_intake")),
            (20.0, 30.0, ()),
            (30.0, 40.0, ("exhaust_fails",)),
            (40.0, 60.0, ()),
        ]
        prints = [i.fingerprint for i in manager.intervals()]
        assert prints[0] == prints[2] == prints[4]
        assert len(set(prints)) == manager.n_flow_states == 3
        assert prints[0] == flow_fingerprint(
            SimConfig(CONFIG_DIR / "clean_room_vents.yaml")
        )

    def test_the_stopped_vents_faces_are_walls_in_their_interval(self) -> None:
        config = SimConfig(VENT_FAILURES)
        mesh = Mesh(config)
        configured, fan_stopped, _, exhaust_failed, _ = ScenarioManager(
            config, mesh
        ).intervals()
        obstacles = {o.name: o for o in config.obstacles}
        vents = {
            "rack_discharge": ("server_rack", "top"),
            "rack_intake": ("server_rack", "left"),
            "etch_exhaust": ("etch_chamber", "right"),
        }
        stopped_in = {
            "rack_discharge": fan_stopped,
            "rack_intake": fan_stopped,
            "etch_exhaust": exhaust_failed,
        }
        for name, (obstacle, face) in vents.items():
            coords = obstacle_face_inputs(mesh, obstacles[obstacle], face).coordinates
            before = BoundaryRegistry(configured.config).coverage_on(
                obstacle, face, coords
            )
            assert any(point.name == name for point in before), name
            after = BoundaryRegistry(stopped_in[name].config).coverage_on(
                obstacle, face, coords
            )
            assert all(point == UNCOVERED for point in after), name
            assert all(point.condition.bc_type == "wall" for point in after), name

    def test_the_fan_stopped_by_halves_is_refused(self, tmp_path: Path) -> None:
        """The discharge stopped while its intake draws is refused at load."""
        raw = yaml.safe_load(VENT_FAILURES.read_text(encoding="utf-8"))
        raw["scenario"]["base"] = str(CONFIG_DIR / "clean_room_vents.yaml")
        raw["scenario"]["events"] = raw["scenario"]["events"][:1]
        with pytest.raises(
            ValueError,
            match=r"the intake 'rack_intake' still draws while no discharge returns it",
        ):
            SimConfig.from_dict(raw)
        # The intake stopped alone leaves the discharge naming a missing segment.
        raw["scenario"]["events"] = [
            yaml.safe_load(VENT_FAILURES.read_text(encoding="utf-8"))["scenario"][
                "events"
            ][1]
        ]
        with pytest.raises(ValueError, match=r"names 'rack_intake', which is not a"):
            SimConfig.from_dict(raw)

    def test_a_vent_replacement_that_covers_no_face_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Review 62 S6: a 1 mm exhaust holds no face centre on 160x60 (5 cm cells)."""
        narrow = {
            "name": "exhaust_throttled",
            "start": 30.0,
            "duration": 10.0,
            "replace": {
                "segment": "etch_exhaust",
                "with": [
                    {
                        "name": "etch_slot",
                        "type": "fixed_flow_outlet",
                        "location": "obstacle",
                        "obstacle": "etch_chamber",
                        "face": "right",
                        "y_start": 1.200,
                        "y_end": 1.201,
                        "velocity": 0.25,
                    }
                ],
            },
        }
        config = _scenario(
            tmp_path, [narrow], base=CONFIG_DIR / "clean_room_vents.yaml"
        )
        with pytest.raises(
            ValueError,
            match=r"from 30\.0 s to 40\.0 s with \['exhaust_throttled'\] active: "
            r"boundaries\.etch_slot: the vent's extent covers no face",
        ):
            _manager(config)
        # One face wide, the same replacement is built.
        narrow["replace"]["with"][0]["y_end"] = 1.25
        config = _scenario(
            tmp_path, [narrow], base=CONFIG_DIR / "clean_room_vents.yaml"
        )
        assert _manager(config).n_flow_states == 2

    def test_a_discharge_slowed_without_its_intake_fails_the_balance_check(
        self, tmp_path: Path
    ) -> None:
        """The merged validation runs the recirculating pair's balance check."""
        event = _set("fan_slows", 10.0, "rack_discharge", velocity=0.25)
        with pytest.raises(
            ValueError,
            match=r"10\.0 s to 20\.0 s .* the discharge's flow 0\.15 m\^2/s differs "
            r"from its intake 'rack_intake''s 0\.3 m\^2/s",
        ):
            _scenario(tmp_path, [event], base=CONFIG_DIR / "clean_room_vents.yaml")
