"""Sensor placement by arrival maps (src/placement.py, ADR-015 D; steps 3 and 4).

What is pinned here, unit tier, each test naming in its docstring the
planted defect it was shown to catch:

- ``arrival_from_frames`` on planted frames: the first frame time at or
  after each onset at or above the threshold (a frame at the onset counts),
  inf where none is, NaN in SOLID cells, a second event's map from its own
  onset, each class against its own threshold; its refusals.
- ``arrival_from_run``'s refusals on files it cannot read: a recorded
  configuration with no scenario section, and a file with neither maps nor
  frames.
- ``latency`` and ``detection_units``: arrival minus onset, and the
  (scenario, event, class) units, every event of a two-event scenario kept,
  each capped at its own onset's remaining time.
- ``score`` on planted maps with known answers: the least over the layout's
  cells, the worst case, the capped mean, the undetected count, coverage.
- ``search`` against brute force on random small maps (counts 1 to 3, 31
  candidates, latencies with ties and infinities): exact for the worst-case
  rule at every count, exact for the mean rule at one and two, and the
  greedy's answer at three recorded against the optimum; the set cover at
  two equal to the exhaustive pairs; the uncoverable units removed and
  reported; the ties broken by the other rule; the distinct masks counted.
- ``candidates`` on a 10x6 room: each surface, the standoff, the inlet,
  outlet and vent exclusion, a scenario's open door, the free set.
- ``rank``: the worst case, then the mean, and one set of units.

The march's maps against the frames' on a coarse run, a Phase 4 file read
through ``arrival_from_run`` and flagged with its reason, are integration
tests in tests/test_time_integration.py; REQ-A05's scenarios on 80x30 are
tests/test_sensor_evaluation.py.
"""

from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import MEAN, MOUNT_FREE, WORST_CASE, SimConfig
from src.io_manager import RunFile, TransportFrames
from src.mesh import SOLID, Mesh
from src.placement import (
    ARGMIN,
    EXHAUSTIVE,
    FRAME_RESOLVED,
    GIVEN,
    GREEDY,
    MAX_SET_COVER_UNITS,
    SET_COVER,
    STEP_RESOLVED,
    ClearUnits,
    Layout,
    RunArrival,
    Score,
    Units,
    _maximal,
    _worst_case_cover,
    add_all_clear,
    arrival_from_frames,
    arrival_from_run,
    candidates,
    clear_score,
    clear_times,
    clear_units,
    detection_units,
    exclude_units,
    given_layout,
    latency,
    rank,
    room_clear,
    scenario_end,
    score,
    search,
    search_roles,
)
from src.solver_transport import MassBudget

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
# A 2x3 grid with one SOLID cell and two classes, frames at 0, 0.5, 1 and 2 s.
LIVE = np.array([[True, True, False], [True, True, True]])
TIMES = np.array([0.0, 0.5, 1.0, 2.0])
THRESHOLDS = np.array([1.0, 10.0])
PROVENANCE = {
    "commit": "test",
    "started": "2026-11-01T00:00:00",
    "machine": "test-machine",
    "concurrent": 0,
}
INF = np.inf


def frames_of(cells: dict[tuple[int, int], list[float]]) -> TransportFrames:
    """Four frames of two classes: CELLS' class-0 series, ten times it in class 1."""
    stack = np.zeros((TIMES.size, *LIVE.shape))
    for (j, i), series in cells.items():
        stack[:, j, i] = series
    budgets = tuple((MassBudget(), MassBudget()) for _ in TIMES)
    return TransportFrames(
        steps=np.array([0, 50, 100, 200]),
        times=TIMES,
        fields={0: stack, 1: 10.0 * stack},
        budgets=budgets,
    )


def units_of(rows: list[list[float]], caps: float | list[float] = 10.0) -> Units:
    """Units on a 1 x n grid: ROWS[u][c] is unit u's latency at cell (0, c)."""
    latencies = np.array(rows, dtype=np.float64)[:, None, :]
    caps = np.broadcast_to(np.asarray(caps, dtype=np.float64), (len(rows),)).copy()
    labels = tuple(("s", "e", u) for u in range(len(rows)))
    return Units(latencies, caps, labels, (False,) * len(rows))


def row_cells(*columns: int) -> list[tuple[int, int]]:
    """Cells (0, c) of the 1 x n grid of ``units_of``."""
    return [(0, c) for c in columns]


@pytest.mark.unit
class TestArrivalFromFrames:
    def test_the_first_frame_at_or_after_each_onset_at_or_above(self) -> None:
        """Onsets 0 and 0.75: per cell the first such frame, inf where none, NaN SOLID.

        Cell (0, 0) is above from 0.5 s: event 0 reads 0.5, event 1 the
        first frame after its onset, 1.0. Cell (0, 1) is exactly at the
        threshold at 0.5 only: event 0 reads 0.5 and event 1 never. Cell
        (1, 2) is above at 2.0: both read 2.0. Defects caught: ">" for "at
        or above" ((0, 1) inf); the second event's map started at t = 0
        ((0, 0) 0.5 for event 1); the last frame kept ((0, 0) 2.0); the
        SOLID cell given inf.
        """
        frames = frames_of(
            {
                (0, 0): [0.0, 2.0, 2.0, 2.0],
                (0, 1): [0.0, 1.0, 0.0, 0.0],
                (1, 2): [0.0, 0.0, 0.0, 5.0],
                (0, 2): [9.0, 9.0, 9.0, 9.0],
            }
        )
        maps = arrival_from_frames(frames, THRESHOLDS, LIVE, np.array([0.0, 0.75]))
        assert maps.shape == (2, 2, 2, 3)
        for k in (0, 1):
            assert maps[0, k].tolist()[0][:2] == [0.5, 0.5]
            assert maps[1, k].tolist()[0][:2] == [1.0, np.inf]
            assert maps[0, k, 1, 2] == maps[1, k, 1, 2] == 2.0
            assert np.isinf(maps[:, k, 1, :2]).all()
        assert np.isnan(maps[:, :, 0, 2]).all()

    def test_a_frame_at_an_onset_counts_for_that_event(self) -> None:
        """A frame whose time is an onset is the first frame at or after it.

        Onsets 0 and 1 s, both frame times. Cell (0, 0) is above in the
        frame at 0 only: event 0 reads 0.0 and event 1 never. Cell (0, 1) is
        above in the frame at 1.0 only: both events read 1.0. Every clean-out
        has this tie (its onset and its first frame are both t = 0). Defect
        caught: ">" for ">=" in the comparison with the onset ((0, 0) inf
        for event 0, (0, 1) inf for event 1).
        """
        frames = frames_of({(0, 0): [2.0, 0.0, 0.0, 0.0], (0, 1): [0.0, 0.0, 2.0, 0.0]})
        maps = arrival_from_frames(frames, THRESHOLDS, LIVE, np.array([0.0, 1.0]))
        for k in (0, 1):
            assert maps[0, k, 0, 0] == 0.0 and np.isinf(maps[1, k, 0, 0])
            assert maps[0, k, 0, 1] == maps[1, k, 0, 1] == 1.0

    def test_a_class_with_its_own_threshold(self) -> None:
        """Each class against its own threshold, not class 0's and not the last.

        Class 1's field is ten times class 0's and its threshold twenty
        times, so the cell reaches class 0's threshold at 1.0 s and never
        reaches class 1's. Defects caught: class 0's threshold used for every
        class (class 1 reads 0.5); the last class's used for every class
        (class 0 never).
        """
        frames = frames_of({(1, 0): [0.0, 0.5, 1.0, 1.0]})
        maps = arrival_from_frames(frames, np.array([1.0, 20.0]), LIVE, np.array([0.0]))
        assert maps[0, 0, 1, 0] == 1.0 and np.isinf(maps[0, 1, 1, 0])

    @pytest.mark.parametrize(
        ("thresholds", "live", "onsets", "match"),
        [
            (np.array([1.0]), LIVE, np.array([0.0]), r"thresholds of shape \(1,\)"),
            (
                THRESHOLDS[None],
                LIVE,
                np.array([0.0]),
                r"thresholds must be \[classes\]",
            ),
            (THRESHOLDS, LIVE[:, :2], np.array([0.0]), "frames are"),
            (THRESHOLDS, LIVE, np.array([[0.0]]), r"onsets must be \[events\]"),
        ],
    )
    def test_the_inputs_are_refused_unless_they_fit_the_frames(
        self, thresholds: np.ndarray, live: np.ndarray, onsets: np.ndarray, match: str
    ) -> None:
        """One threshold per class of the frames, live of their shape, onsets [events]."""
        with pytest.raises(ValueError, match=match):
            arrival_from_frames(frames_of({}), thresholds, live, onsets)

    @pytest.mark.parametrize(
        ("thresholds", "onsets", "error", "match"),
        [
            (np.array([1.0, np.nan]), [0.0], ValueError, "thresholds must be finite"),
            (np.array([1.0, np.inf]), [0.0], ValueError, "thresholds must be finite"),
            (np.array([1.0, -1.0]), [0.0], ValueError, "finite and non-negative"),
            (np.array([True, False]), [0.0], TypeError, "a threshold must be a real"),
            (THRESHOLDS, [np.nan], ValueError, "the onsets must be finite"),
            (THRESHOLDS, [0.0, np.inf], ValueError, "the onsets must be finite"),
            (THRESHOLDS, [-0.5], ValueError, "the onsets must be non-negative"),
            (THRESHOLDS, [1.0, 0.5], ValueError, "the onsets must be in order"),
            (THRESHOLDS, np.array([True]), TypeError, "an onset must be a real"),
        ],
    )
    def test_thresholds_and_onsets_the_tracker_would_refuse_are_refused(
        self, thresholds: np.ndarray, onsets: list, error: type, match: str
    ) -> None:
        """The tracker's input guards, at the frames reader too.

        A NaN threshold or onset compares False everywhere, so without the
        guards every cell would silently read "never". Defects caught: any of
        the checks missing from ``arrival_from_frames`` (the frames read
        with the bad value).
        """
        with pytest.raises(error, match=match):
            arrival_from_frames(frames_of({}), thresholds, LIVE, np.array(onsets))

    def test_frames_whose_classes_are_not_0_to_n_minus_1_are_refused(self) -> None:
        """A frames object holding classes 0 and 2 has no class 1 to read."""
        frames = frames_of({})
        frames.fields[2] = frames.fields.pop(1)
        with pytest.raises(ValueError, match=r"frames' classes \[0, 2\]"):
            arrival_from_frames(frames, THRESHOLDS, LIVE, np.array([0.0]))


@pytest.mark.unit
class TestArrivalFromRun:
    def test_a_file_recording_no_scenario_is_refused(self, tmp_path: Path) -> None:
        """A tier-2 or bare configuration has no plan, so no events to map."""
        raw = yaml.safe_load(
            (CONFIGS / "clean_room_default.yaml").read_text(encoding="utf-8")
        )
        raw["domain"].update({"nx": 80, "ny": 30})
        run = RunFile(tmp_path / "run.h5")
        run.write_config(raw, **PROVENANCE)
        with pytest.raises(ValueError, match="records no scenario section"):
            arrival_from_run(run.path)

    def test_a_file_with_neither_maps_nor_frames_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A run killed before its first frame: KeyError from the frames' reader."""
        raw = yaml.safe_load(
            (CONFIGS / "scenario_door_leak.yaml").read_text(encoding="utf-8")
        )
        raw["scenario"]["base"] = str(CONFIGS / "clean_room_default.yaml")
        raw["domain"] = {"nx": 80, "ny": 30}
        run = RunFile(tmp_path / "run.h5")
        run.write_config(raw, **PROVENANCE)
        with pytest.raises(KeyError, match="holds no /transport/ frames"):
            arrival_from_run(run)


def arrival(
    maps: np.ndarray,
    onsets: list[float],
    t_final: float,
    names: tuple[str, ...],
    resolution: str = STEP_RESOLVED,
) -> RunArrival:
    """A RunArrival from planted maps, step-resolved unless RESOLUTION says otherwise."""
    return RunArrival(maps, np.array(onsets), resolution, names, t_final)


@pytest.mark.unit
class TestLatencyAndUnits:
    def test_latency_is_arrival_minus_the_events_onset(self) -> None:
        """Each event's map less its own onset; inf and NaN kept.

        Defect caught: every event's map less the first onset (event 1's
        cell reads 3.0, not 1.0).
        """
        maps = np.array([[[[2.0, INF, np.nan]]], [[[3.0, 6.0, np.nan]]]])
        out = latency(maps, np.array([1.0, 2.0]))
        assert out[0, 0, 0, :2].tolist() == [1.0, INF]
        assert out[1, 0, 0, :2].tolist() == [1.0, 4.0]
        assert np.isnan(out[:, 0, 0, 2]).all()

    @pytest.mark.parametrize(
        ("maps", "onsets", "match"),
        [
            (np.zeros((2, 1, 1, 1)), [0.0], r"maps of shape \(2, 1, 1, 1\) for 1"),
            (np.zeros((1, 1, 1)), [0.0], "they must be"),
            (np.full((1, 1, 1, 1), 0.5), [1.0], "before its event's onset"),
        ],
    )
    def test_latency_refuses_maps_that_are_not_the_onsets(
        self, maps: np.ndarray, onsets: list[float], match: str
    ) -> None:
        """One map per onset, 4-D, and no arrival before its onset."""
        with pytest.raises(ValueError, match=match):
            latency(maps, np.array(onsets))

    def test_every_event_of_every_scenario_is_a_unit_per_class(self) -> None:
        """A two-event door and a one-event breach, two classes: six units.

        The door's second opening is detected at cell 1 only, so the layout
        of cell 1 detects every door unit and the layout of cell 0 the first
        opening's. Each unit is capped at its own onset's remaining time.
        Defect caught: one event of a two-event scenario dropped (four
        units, and cell 0's layout reads every door unit detected).
        """
        door = np.full((2, 2, 1, 2), INF)
        door[0, :, 0, 0] = [11.0, 12.0]
        door[0, :, 0, 1] = [13.0, 14.0]
        door[1, :, 0, 1] = [31.0, 32.0]
        breach = np.full((1, 2, 1, 2), INF)
        breach[0, :, 0, 0] = [12.0, 15.0]
        units = detection_units(
            {
                "door": arrival(door, [10.0, 30.0], 60.0, ("open", "again")),
                "breach": arrival(breach, [10.0], 120.0, ("breach",)),
            }
        )
        assert units.labels == (
            ("door", "open", 0),
            ("door", "open", 1),
            ("door", "again", 0),
            ("door", "again", 1),
            ("breach", "breach", 0),
            ("breach", "breach", 1),
        )
        assert units.caps.tolist() == [50.0, 50.0, 30.0, 30.0, 110.0, 110.0]
        assert units.latencies[:, 0, 1].tolist() == [3.0, 4.0, 1.0, 2.0, INF, INF]
        assert score(units, [(0, 0)]).undetected == 2
        assert score(units, [(0, 1)]).undetected == 2
        both = score(units, [(0, 0), (0, 1)])
        assert both.undetected == 0
        assert both.worst == 5.0

    @pytest.mark.parametrize(
        ("arrivals", "error", "match"),
        [
            ({}, ValueError, "no run to score"),
            (
                {
                    "a": arrival(np.zeros((1, 1, 1, 2)), [0.0], 1.0, ("e",)),
                    "b": arrival(np.zeros((1, 1, 2, 1)), [0.0], 1.0, ("e",)),
                },
                ValueError,
                "must share a mesh",
            ),
            (
                {"a": arrival(np.zeros((1, 1, 1, 2)), [0.0], 1.0, ("e", "f"))},
                ValueError,
                "names 2 events for 1 maps",
            ),
            (
                {"a": arrival(np.zeros((1, 1, 1, 2)), [2.0], 1.0, ("e",))},
                ValueError,
                "before its event's onset",
            ),
            (
                {"a": arrival(np.full((1, 1, 1, 2), 3.0), [2.0], 1.0, ("e",))},
                ValueError,
                "at or after every onset",
            ),
            (
                {"a": arrival(np.zeros((1, 1, 1, 2)), [0.0], INF, ("e",))},
                ValueError,
                "must be finite",
            ),
            (
                {"a": arrival(np.zeros((1, 1, 1, 2)), [0.0], True, ("e",))},
                TypeError,
                "t_final must be a real number",
            ),
            (
                {3: arrival(np.zeros((1, 1, 1, 2)), [0.0], 1.0, ("e",))},
                TypeError,
                "a scenario name must be a str",
            ),
        ],
    )
    def test_runs_that_do_not_make_units_are_refused(
        self, arrivals: dict, error: type, match: str
    ) -> None:
        """At least one run, one mesh, a name per event, a run ending after its onsets."""
        with pytest.raises(error, match=match):
            detection_units(arrivals)

    def test_the_frame_resolved_flag_reaches_the_scores_and_layouts(self) -> None:
        """Units of a frame-built run are flagged, and so is whatever scores them.

        A step-resolved run and a frame-resolved one, one class each: the
        units carry their run's flag, ``exclude_units`` keeps the flags with
        their units, a score and a given layout over a flagged unit are
        flagged, and a search whose flagged unit no candidate reaches
        (reported uncovered, so not scored) is not. Defects caught: the
        flag dropped (``arrival_from_run``'s fallback unmarked downstream);
        flagged for every unit once any is; the flag kept for a unit the
        search removed.
        """
        step = np.full((1, 1, 1, 2), INF)
        step[0, 0, 0, 1] = 2.0
        frame = np.full((1, 1, 1, 2), INF)
        frame[0, 0, 0, 0] = 3.0
        units = detection_units(
            {
                "step": arrival(step, [0.0], 10.0, ("e",)),
                "frame": arrival(frame, [0.0], 10.0, ("e",), FRAME_RESOLVED),
            }
        )
        assert units.frame_resolved == (False, True)
        only_step = exclude_units(units, [("frame", "e", 0)])
        assert only_step.frame_resolved == (False,)
        only_frame = exclude_units(units, [("step", "e", 0)])
        assert only_frame.frame_resolved == (True,)
        assert score(only_frame, row_cells(0)).frame_resolved
        assert score(units, row_cells(0, 1)).frame_resolved
        assert not score(only_step, row_cells(0, 1)).frame_resolved
        assert given_layout(units, row_cells(1)).frame_resolved
        both = search(units, row_cells(0, 1), 1, MEAN)
        assert both.frame_resolved and both.score.frame_resolved
        flat = search(units, row_cells(1), 1, MEAN)
        assert flat.uncovered == (("frame", "e", 0),) and not flat.frame_resolved

    def test_a_resolution_that_is_neither_is_refused(self) -> None:
        """A run's resolution is ``STEP_RESOLVED`` or ``FRAME_RESOLVED``, nothing else."""
        bad = arrival(np.zeros((1, 1, 1, 2)), [0.0], 1.0, ("e",), "coarse")
        with pytest.raises(ValueError, match="resolution is 'coarse'"):
            detection_units({"a": bad})

    def test_exclude_units_leaves_the_named_out_and_refuses_the_unknown(self) -> None:
        """The rest in order; an unknown label, or none left, is refused."""
        units = units_of([[1.0], [2.0], [3.0]])
        rest = exclude_units(units, [("s", "e", 1)])
        assert rest.labels == (("s", "e", 0), ("s", "e", 2))
        assert rest.latencies[:, 0, 0].tolist() == [1.0, 3.0]
        with pytest.raises(ValueError, match="no such units"):
            exclude_units(units, [("s", "e", 7)])
        with pytest.raises(ValueError, match="leaves no unit"):
            exclude_units(units, units.labels)


@pytest.mark.unit
class TestScore:
    def test_the_three_rules_on_planted_maps(self) -> None:
        """Three units, two cells: the least per unit, then the rules over the units.

        Unit 0 reads 4 and 1, unit 1 reads 2 and 6, unit 2 never and 3. The
        layout of both reads 1, 2 and 3: worst 3, mean 2, coverage within
        2 s two of three. The layout of cell 0 alone leaves unit 2
        undetected: worst inf, its mean capped at 10. Defect caught: the
        least over the layout's cells taken as the most (4, 6, 3: worst 6).
        """
        units = units_of([[4.0, 1.0], [2.0, 6.0], [INF, 3.0]])
        both = score(units, row_cells(0, 1), within=2.0)
        assert both == Score(worst=3.0, mean=2.0, undetected=0, coverage=2 / 3)
        alone = score(units, row_cells(0))
        assert alone == Score(worst=INF, mean=16.0 / 3, undetected=1, coverage=None)

    def test_the_mean_caps_each_unit_at_its_own_remaining_time(self) -> None:
        """An undetected unit counts its cap, not inf and not another unit's cap."""
        units = units_of([[1.0], [INF]], caps=[5.0, 20.0])
        assert score(units, row_cells(0)).mean == 10.5

    @pytest.mark.parametrize(
        ("cells", "within", "error", "match"),
        [
            ([], None, ValueError, "cells is empty"),
            (np.array([[0.0, 0.0]]), None, TypeError, "integer"),
            (np.array([[True, False]]), None, TypeError, "integer"),
            (np.array([0, 1]), None, ValueError, r"must be \[n, 2\]"),
            ([(0, 3)], None, ValueError, "off the 1x3 grid"),
            ([(-1, 0)], None, ValueError, "off the 1x3 grid"),
            ([(0, 1), (0, 1)], None, ValueError, "names a cell twice"),
            ([(0, 2)], None, ValueError, "SOLID cell"),
            ([(0, 0)], 0.0, ValueError, "within must be positive"),
            ([(0, 0)], np.nan, ValueError, "within must be positive"),
            ([(0, 0)], True, TypeError, "within must be a real number"),
        ],
    )
    def test_a_layout_that_is_not_cells_of_the_grid_is_refused(
        self, cells: object, within: object, error: type, match: str
    ) -> None:
        """Integer (j, i) on the grid, none twice, none SOLID; a positive finite within."""
        units = units_of([[1.0, 2.0, np.nan]])
        with pytest.raises(error, match=match):
            score(units, cells, within=within)

    def test_a_given_layout_keeps_its_cells_and_scores_by_lookup(self) -> None:
        """The file's sensors as a Layout: GIVEN, no rule, not exact, nothing uncovered."""
        units = units_of([[4.0, 1.0], [2.0, 6.0]])
        layout = given_layout(units, row_cells(1, 0))
        assert layout.cells == ((0, 1), (0, 0))
        assert layout.score == score(units, row_cells(0, 1))
        assert (layout.method, layout.rule, layout.exact) == (GIVEN, None, False)
        assert layout.units == units.labels and layout.uncovered == ()


def random_units(seed: int) -> tuple[Units, list[tuple[int, int]]]:
    """Six units on a 4x8 grid, integer latencies 0 to 5, half inf, one SOLID cell.

    Integer latencies and caps make every mean exact whatever the order of
    the sum, so two layouts tie exactly when their latencies tie.
    """
    rng = np.random.default_rng(seed)
    latencies = rng.integers(0, 6, size=(6, 4, 8)).astype(np.float64)
    latencies[rng.random(latencies.shape) < 0.5] = INF
    latencies[:, 0, 0] = np.nan
    labels = tuple(("s", "e", u) for u in range(6))
    units = Units(latencies, np.full(6, 10.0), labels, (False,) * 6)
    cells = [(j, i) for j in range(4) for i in range(8) if (j, i) != (0, 0)]
    return units, cells


def brute_force(
    units: Units, cells: list[tuple[int, int]], count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Every layout of COUNT of CELLS: (worst, mean) per combination, by lookup."""
    table = np.stack([units.latencies[:, j, i] for j, i in cells])
    combos = np.array(list(combinations(range(len(cells)), count)))
    least = table[combos].min(axis=1)
    return least.max(axis=1), np.minimum(least, units.caps).mean(axis=1)


SEEDS = range(12)


# Six units, seven cells, each detecting what it covers at the latency
# given (inf elsewhere): cells 0 to 2 cover the units in pairs (0, 1),
# (2, 3), (4, 5) at 1 s; cells 3 to 5 cover (0, 2), (1, 4), (3, 5), the
# first two of them at 0 s; cell 6 covers unit 0 alone, inside cell 0's
# and cell 3's masks. The only covers of three are cells 0 to 2 and
# cells 3 to 5, both with worst case 1; the second has mean 1/3.
PAIRED = [
    [1.0, INF, INF, 0.0, INF, INF, 1.0],
    [1.0, INF, INF, INF, 0.0, INF, INF],
    [INF, 1.0, INF, 0.0, INF, INF, INF],
    [INF, 1.0, INF, INF, INF, 1.0, INF],
    [INF, INF, 1.0, INF, 0.0, INF, INF],
    [INF, INF, 1.0, INF, INF, 1.0, INF],
]


@pytest.mark.unit
class TestSearch:
    @pytest.mark.parametrize("count", [1, 2, 3])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_worst_case_rule_is_exact_at_every_count(
        self, seed: int, count: int
    ) -> None:
        """Against every layout of the count: the least worst case, exactly.

        At one and two sensors the tie-break by the mean is exact too; at
        three it is no better than the best mean among the worst-case
        optimal layouts. The units no candidate reaches are removed first
        (``uncovered``), and brute force is scored over the rest. Defects
        caught: the threshold sweep stopping one distinct latency early or
        late; the set cover's union of k masks not built from k - 1.
        """
        units, cells = random_units(seed)
        layout = search(units, cells, count, WORST_CASE)
        kept = exclude_units(units, layout.uncovered)
        worst, mean = brute_force(kept, cells, count)
        assert layout.score.worst == worst.min()
        assert layout.exact
        best_mean = mean[worst == worst.min()].min()
        if np.isinf(worst.min()):
            assert layout.rule == MEAN and layout.score.mean == mean.min()
        elif count <= 2:
            assert layout.rule == WORST_CASE and layout.score.mean == best_mean
        else:
            assert layout.method == SET_COVER and layout.score.mean >= best_mean

    @pytest.mark.parametrize("count", [1, 2])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_mean_rule_is_exact_at_one_and_two(self, seed: int, count: int) -> None:
        """The argmin and every pair: the least mean, ties by the worst case."""
        units, cells = random_units(seed)
        layout = search(units, cells, count, MEAN)
        worst, mean = brute_force(exclude_units(units, layout.uncovered), cells, count)
        assert layout.score.mean == mean.min()
        assert layout.score.worst == worst[mean == mean.min()].min()
        assert (layout.method, layout.rule, layout.exact) == (
            ARGMIN if count == 1 else EXHAUSTIVE,
            MEAN,
            True,
        )

    def test_the_greedy_at_three_against_the_optimum_is_recorded(self) -> None:
        """The mean rule at three sensors, greedy with exchanges, on twelve random maps.

        Measured (prompt 75): the greedy's mean equals the optimum's on 10
        of the 12 maps; on the other two it is 0.67 and 0.83 s against the
        optimum's 0.5 (the means run from 0 to 0.83 s); it is never below
        the optimum. Pinned as recorded, so a change to
        the greedy that loses ground on these maps is seen.
        """
        gaps = []
        for seed in SEEDS:
            units, cells = random_units(seed)
            layout = search(units, cells, 3, MEAN)
            assert (layout.method, layout.exact) == (GREEDY, False)
            _, mean = brute_force(exclude_units(units, layout.uncovered), cells, 3)
            gaps.append(layout.score.mean - mean.min())
        assert min(gaps) >= 0.0
        assert sum(gap == 0.0 for gap in gaps) == 10
        assert max(gaps) == pytest.approx(1 / 3)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_set_cover_at_two_equals_every_pair(self, seed: int) -> None:
        """The set cover asked for two sensors finds the exhaustive pairs' worst case."""
        units, cells = random_units(seed)
        layout = search(units, cells, 2, WORST_CASE)
        kept = exclude_units(units, layout.uncovered)
        table = np.ascontiguousarray(
            np.stack([kept.latencies[:, j, i] for j, i in cells])
        )
        cover = _worst_case_cover(table, kept.caps, 2)
        if np.isinf(layout.score.worst):
            assert cover.columns == ()
        else:
            assert table[list(cover.columns)].min(axis=0).max() == layout.score.worst

    @pytest.mark.parametrize("chunk", [1, 3, 7, 256])
    def test_the_maximal_masks_are_those_no_other_contains_at_any_chunk(
        self, chunk: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Against a plain containment test, with the chunk from one row to more than all.

        Defects caught: a chunk's diagonal cleared at the wrong offset (a
        mask in the second chunk judged against itself, kept as contained,
        or a container's twin dropped); the last short chunk left out.
        """
        monkeypatch.setattr("src.placement._MAXIMAL_CHUNK", chunk)
        rng = np.random.default_rng(chunk)
        masks = np.unique(rng.integers(0, 1 << 9, size=40)).astype(np.int64)
        expected = [
            int(m)
            for m in masks
            if m and not any(o != m and (o & m) == m for o in masks)
        ]
        assert _maximal(masks).tolist() == expected
        assert _maximal(np.array([0, 5], dtype=np.int64)).tolist() == [5]

    def test_units_no_candidate_reaches_are_removed_and_reported(self) -> None:
        """Unit 1 is inf at every candidate: reported uncovered, the rest searched.

        The worst case is over units 0 and 2, finite, and the layout's units
        say so. Defect caught: the worst case taken over the coverable units
        without reporting the rest (``uncovered`` empty, ``units`` all three).
        """
        units = units_of([[1.0, 4.0], [INF, INF], [5.0, 2.0]])
        layout = search(units, row_cells(0, 1), 1, WORST_CASE)
        assert layout.uncovered == (("s", "e", 1),)
        assert layout.units == (("s", "e", 0), ("s", "e", 2))
        assert layout.score.worst == 4.0 and layout.cells == ((0, 1),)

    def test_a_tie_in_the_worst_case_is_broken_by_the_mean(self) -> None:
        """Cells 0 and 1 share the worst case 5; cell 1's mean is lower.

        Defect caught: the tie broken by candidate order (cell 0).
        """
        units = units_of([[5.0, 5.0], [4.0, 1.0]])
        assert search(units, row_cells(0, 1), 1, WORST_CASE).cells == ((0, 1),)
        units = units_of([[3.0, 2.0], [3.0, 4.0]])
        assert search(units, row_cells(0, 1), 1, MEAN).cells == ((0, 0),)

    def test_the_set_cover_on_planted_masks(self) -> None:
        """Count 3: worst case 1 by the cover of cells 3 to 5, the better mean.

        The masks are counted at the optimum: seven distinct, six maximal
        (cell 6's is inside two others). Defects caught: the covers' tie
        broken by order (cells 0 to 2, mean 1); the set cover's programme
        missing the empty set's base case (no cover found, so the mean
        rule's greedy); a dominated mask kept as maximal (seven).
        """
        units = units_of(PAIRED)
        layout = search(units, row_cells(*range(7)), 3, WORST_CASE)
        assert (layout.method, layout.rule, layout.exact) == (
            SET_COVER,
            WORST_CASE,
            True,
        )
        assert layout.cells == ((0, 3), (0, 4), (0, 5))
        assert layout.score.worst == 1.0 and layout.score.mean == pytest.approx(1 / 3)
        assert (layout.distinct_masks, layout.maximal_masks) == (7, 6)

    def test_no_cover_of_the_count_falls_to_the_mean_rule_and_says_so(self) -> None:
        """Two sensors cannot cover six units in pairs; three can, with four units more.

        Count 2: every pair leaves a unit undetected, the worst case is inf
        for all, and the layout is the mean's (rule MEAN, exact over pairs).
        Count 3 on ten units the cells cover two at a time: no cover, the
        greedy, rule MEAN, the masks still counted.
        """
        units = units_of(PAIRED)
        two = search(units, row_cells(*range(7)), 2, WORST_CASE)
        assert (two.method, two.rule, two.exact) == (EXHAUSTIVE, MEAN, True)
        assert np.isinf(two.score.worst)
        rows = [[1.0 if c == u // 2 else INF for c in range(5)] for u in range(10)]
        three = search(units_of(rows), row_cells(*range(5)), 3, WORST_CASE)
        assert (three.method, three.rule, three.exact) == (GREEDY, MEAN, False)
        assert three.distinct_masks == 5 and three.score.undetected == 4

    @pytest.mark.parametrize(("count", "left_out"), [(3, (0, 1)), (4, (0,))])
    def test_the_fallback_layout_at_three_or_more_is_the_mean_rules_search(
        self, count: int, left_out: tuple[int, ...]
    ) -> None:
        """No cover of ten units: the layout is the mean rule's, not the first candidates'.

        Ten units, unit u reached only by candidate u // 2 at latency 5 - c
        (the later candidates are the faster), caps 10. Three sensors cover
        three of the five pairs and four cover four, so no layout detects
        every unit and the worst case is inf for all. The optimum leaves out
        the slowest candidates: cells 2 to 4 at three sensors (mean 5.2) and
        1 to 4 at four (mean 4.0), which the mean rule's search and a brute
        force over every layout both find, and the first ``count``
        candidates are not (mean 6.4 and 5.8). Defect caught: the arm
        returning the first ``count`` candidates, with the right rule,
        method and mask counts, which the symmetric fixture above scores
        alike for every layout.
        """
        latencies = np.full((10, 1, 5), INF)
        for u in range(10):
            latencies[u, 0, u // 2] = 5.0 - u // 2
        labels = tuple(("s", "e", u) for u in range(10))
        units = Units(latencies, np.full(10, 10.0), labels, (False,) * 10)
        cells = row_cells(*range(5))
        found = search(units, cells, count, WORST_CASE)
        by_mean = search(units, cells, count, MEAN)
        optimum = tuple((0, c) for c in range(5) if c not in left_out)
        _, means = brute_force(units, cells, count)
        assert (found.method, found.rule, found.exact) == (GREEDY, MEAN, False)
        assert (found.distinct_masks, found.maximal_masks) == (5, 5)
        assert found.cells == by_mean.cells == optimum
        assert found.score == by_mean.score
        assert found.score.mean == pytest.approx(means.min())
        assert found.score.mean < score(units, cells[:count]).mean
        assert np.isinf(found.score.worst)

    def test_a_searched_layouts_cells_are_in_row_major_order(self) -> None:
        """Candidates given last row first: the layout's cells come back sorted by (j, i).

        Two rows of two cells, each unit reached by one cell only, so all
        four are chosen at four sensors. Defect caught: the cells in the
        order of the candidates given (the documented row-major order
        holding only for candidates as ``candidates()`` returns them).
        """
        latencies = np.full((4, 2, 2), INF)
        for u, (j, i) in enumerate([(1, 1), (0, 0), (1, 0), (0, 1)]):
            latencies[u, j, i] = 1.0 + u
        labels = tuple(("s", "e", u) for u in range(4))
        units = Units(latencies, np.full(4, 10.0), labels, (False,) * 4)
        backwards = [(1, 1), (1, 0), (0, 1), (0, 0)]
        for rule in (WORST_CASE, MEAN):
            found = search(units, backwards, 4, rule)
            assert found.cells == ((0, 0), (0, 1), (1, 0), (1, 1))

    @pytest.mark.parametrize(
        ("count", "rule", "cells", "error", "match"),
        [
            (1, "coverage", [(0, 0)], ValueError, "rule must be one of"),
            (True, WORST_CASE, [(0, 0)], TypeError, "count must be an int"),
            (1.0, WORST_CASE, [(0, 0)], TypeError, "count must be an int"),
            (0, WORST_CASE, [(0, 0)], ValueError, "count must be positive"),
            (3, WORST_CASE, [(0, 0), (0, 1)], ValueError, "at most the 2 candidates"),
            (1, WORST_CASE, [(0, 2)], ValueError, "SOLID cell"),
            (1, WORST_CASE, [(0, 1)], ValueError, "no candidate reaches any unit"),
        ],
    )
    def test_a_search_that_cannot_run_is_refused(
        self, count: object, rule: str, cells: list, error: type, match: str
    ) -> None:
        """A known rule, a positive int count within the candidates, live cells, a reach."""
        units = units_of([[1.0, INF, np.nan]])
        with pytest.raises(error, match=match):
            search(units, cells, count, rule)

    def test_more_units_than_the_set_cover_holds_are_refused(self) -> None:
        """2^units subsets: above MAX_SET_COVER_UNITS the cover is refused, not run."""
        units = units_of([[1.0, 1.0, 1.0]] * (MAX_SET_COVER_UNITS + 1))
        with pytest.raises(ValueError, match="at most MAX_SET_COVER_UNITS"):
            search(units, row_cells(0, 1, 2), 3, WORST_CASE)


@pytest.mark.unit
class TestRank:
    def test_by_the_worst_case_then_the_mean(self) -> None:
        """Worst 3 before worst 4; at worst 3 the lower mean first; equal keep order.

        Defect caught: ranked by the mean alone (the worst-4 layout, mean 1,
        first).
        """
        units = units_of([[3.0, 4.0, 3.0, 3.0], [3.0, 0.0, 1.0, 1.0]])
        layouts = [given_layout(units, row_cells(c)) for c in range(4)]
        ranked = rank(layouts)
        assert [layout.cells[0][1] for layout in ranked] == [2, 3, 0, 1]

    def test_layouts_over_different_units_or_none_are_refused(self) -> None:
        """A search's units lack the uncovered ones: score the file's over the same."""
        units = units_of([[1.0, 4.0], [INF, INF]])
        found = search(units, row_cells(0, 1), 1, WORST_CASE)
        mine = given_layout(units, row_cells(1))
        with pytest.raises(ValueError, match="different units"):
            rank([found, mine])
        same = given_layout(exclude_units(units, found.uncovered), row_cells(1))
        assert [layout.method for layout in rank([same, found])] == [ARGMIN, GIVEN]
        with pytest.raises(ValueError, match="no layout to rank"):
            rank([])

    def test_a_layout_is_a_named_tuple_of_its_parts(self) -> None:
        """The fields the latency report reads, by name."""
        assert Layout._fields == (
            "cells",
            "score",
            "rule",
            "method",
            "exact",
            "units",
            "uncovered",
            "distinct_masks",
            "maximal_masks",
        )


# The four fixed surfaces of the product file's mount, without its stand.
SURFACES = ("walls", "ceiling", "obstacle_tops", "obstacle_sides")


def small_room(**changes: object) -> dict:
    """A 1.0 x 0.6 m room on 10x6 cells of 0.1 m, from the product file's sections.

    A supply on the ceiling over x 0.3 to 0.5, a return on the floor over
    0.7 to 0.9, a wall door on the left over y 0 to 0.3, and one box
    x 0.4 to 0.6, y 0 to 0.2 (cells i 4 and 5, j 0 and 1).
    """
    raw = yaml.safe_load(
        (CONFIGS / "clean_room_default.yaml").read_text(encoding="utf-8")
    )
    raw["domain"] = {"width": 1.0, "height": 0.6, "nx": 10, "ny": 6}
    # The product's stand band, 0.3 to 1.2 m, reaches above this room's 0.6 m.
    raw["placement"] = dict(raw["placement"], stand_height=[0.1, 0.3])
    supply = dict(raw["boundaries"]["hepa_supply"], x_start=0.3, x_end=0.5)
    raw["boundaries"] = {
        "hepa_supply": supply,
        "door": {"type": "wall", "location": "left", "y_start": 0.0, "y_end": 0.3},
        "floor_return": {
            "type": "fixed_flow_outlet",
            "location": "bottom",
            "x_start": 0.7,
            "x_end": 0.9,
        },
    }
    raw["obstacles"] = [
        {"name": "box", "x_start": 0.4, "x_end": 0.6, "y_start": 0.0, "y_end": 0.2}
    ]
    raw["sensors"] = [{"name": "s", "x": 0.15, "y": 0.45}]
    raw.update(changes)
    return raw


def cells_of(found: np.ndarray) -> set[tuple[int, int]]:
    """Candidate rows as a set of (j, i)."""
    return {(int(j), int(i)) for j, i in found}


@pytest.mark.unit
class TestCandidates:
    STANDOFF = 0.1

    def setup_method(self) -> None:
        """The small room's configuration and mesh."""
        self.config = SimConfig.from_dict(small_room())
        self.mesh = Mesh(self.config)

    def found(self, *mount: str, standoff: float = STANDOFF) -> set[tuple[int, int]]:
        """The candidates for MOUNT at STANDOFF."""
        return cells_of(candidates(self.mesh, self.config, mount, standoff))

    def test_each_surface_at_a_one_cell_standoff(self) -> None:
        """Centres within 0.1 m of each surface's faces, the ends' diagonals included.

        The walls are columns 0 and 9 but for (0, 9), 0.071 m from the return's
        end; the ceiling row 5 but for the two cells under the supply and the
        two 0.071 m from its ends; the box's top row 2 from i 3 to 6, and
        (1, 3) and (1, 6) beside its sides, 0.071 m from its top corners as
        (2, 3) and (2, 6) are; its sides columns 3 and 6 from j 0 to 2 but for
        (0, 6), 0.071 m from the return's end, and (2, 4) and (2, 5) above the
        top, 0.071 m from the sides' upper ends. Defect caught: the standoff
        measured from the box's bottom face (on the floor, so no face and no
        top candidate).
        """
        rows = range(6)
        walls = {(j, 0) for j in rows} | {(j, 9) for j in rows[1:]}
        assert self.found("walls") == walls
        assert self.found("ceiling") == {(5, i) for i in (0, 1, 6, 7, 8, 9)}
        tops = {(2, 3), (2, 4), (2, 5), (2, 6), (1, 3), (1, 6)}
        assert self.found("obstacle_tops") == tops
        sides = {(j, 3) for j in range(3)} | {(1, 6), (2, 6), (2, 4), (2, 5)}
        assert self.found("obstacle_sides") == sides
        union = self.found("walls", "ceiling", "obstacle_tops", "obstacle_sides")
        assert union == (self.found("walls") | self.found("ceiling") | tops | sides)

    def test_a_wall_or_ceiling_face_behind_a_solid_cell_is_no_surface(self) -> None:
        """Boxes in the top-left and on the right wall: their edge faces bound no room.

        At 0.15 m, (5, 1) and (3, 8) are 0.15 m from the wall faces of the
        SOLID cells (5, 0) and (3, 9) and 0.158 m from any face of a live
        cell, and (4, 0) is 0.15 m from (5, 0)'s ceiling face, so none is a
        candidate. Defect caught: a domain-edge face behind a SOLID cell
        counted as a wall or the ceiling (all three candidates).
        """
        raw = small_room()
        raw["obstacles"] += [
            {
                "name": "corner",
                "x_start": 0.0,
                "x_end": 0.1,
                "y_start": 0.5,
                "y_end": 0.6,
            },
            {
                "name": "shelf",
                "x_start": 0.9,
                "x_end": 1.0,
                "y_start": 0.3,
                "y_end": 0.4,
            },
        ]
        config = SimConfig.from_dict(raw)
        mesh = Mesh(config)
        assert mesh.cell_type[5, 0] == SOLID and mesh.cell_type[3, 9] == SOLID
        walls = cells_of(candidates(mesh, config, ["walls"], 0.15))
        assert (5, 1) not in walls and (3, 8) not in walls
        assert {(4, 1), (2, 8)} <= walls
        ceiling = cells_of(candidates(mesh, config, ["ceiling"], 0.15))
        assert (4, 0) not in ceiling and (4, 1) in ceiling

    def test_a_larger_standoff_reaches_the_next_cells(self) -> None:
        """At 0.15 m the walls reach columns 1 and 8 as well (centres 0.15 m away)."""
        walls = self.found("walls", standoff=0.15)
        assert {i for _, i in walls} == {0, 1, 8, 9}

    def test_cells_beside_or_within_the_standoff_of_an_opening_are_excluded(
        self,
    ) -> None:
        """The supply's two cells, the return's two and their neighbours, free or not.

        Beside: (5, 3), (5, 4), (0, 7), (0, 8). Within 0.1 m of a face's end:
        (5, 2), (5, 5), (0, 6), (0, 9), each 0.071 m away. At a standoff under
        the half cell only the cells beside the faces go, so the free set
        still excludes them. Defects caught: the inlet's cells kept on the
        ceiling; the free set not excluding them; the exclusion read as
        beside the face alone.
        """
        free = self.found("free")
        live = Mesh(self.config).cell_type != SOLID
        every = {(j, i) for j, i in zip(*np.nonzero(live), strict=True)}
        beside = {(5, 3), (5, 4), (0, 7), (0, 8)}
        near = {(5, 2), (5, 5), (0, 6), (0, 9)}
        assert free == every - beside - near
        assert not (beside | near) & self.found("walls", "ceiling", "obstacle_tops")
        assert self.found("free", standoff=0.01) == every - beside

    def test_a_vent_on_an_obstacle_face_excludes_the_room_cell_beside_it(self) -> None:
        """An exhaust over the box's top from x 0.4 to 0.5: its room cell and neighbours.

        Cell (2, 4) is beside the vent; (2, 3), (2, 5) and (1, 3) are 0.071 m
        from its ends. Defect caught: the vent's faces left out of the
        openings, or their room cell taken on the obstacle's side.
        """
        raw = small_room()
        raw["boundaries"]["box_exhaust"] = {
            "type": "fixed_flow_outlet",
            "location": "obstacle",
            "obstacle": "box",
            "face": "top",
            "x_start": 0.4,
            "x_end": 0.5,
            "velocity": 0.1,
        }
        config = SimConfig.from_dict(raw)
        found = cells_of(candidates(Mesh(config), config, ["obstacle_tops"], 0.1))
        assert found == {(2, 6), (1, 6)}

    @pytest.mark.parametrize(
        ("face", "lost"),
        [("right", {(1, 6)}), ("left", {(0, 3), (1, 3)})],
    )
    def test_a_vent_on_an_obstacle_side_excludes_the_cells_near_its_faces(
        self, face: str, lost: set[tuple[int, int]]
    ) -> None:
        """An exhaust over the box's side from y 0 to 0.1: the room cell and the next above.

        The vent's face is vertical, at the side's x, so the cell beside it
        and the one above, 0.071 m from the face's top, go. On the right the
        cell beside it, (0, 6), is out already for the return. Defect caught:
        a side vent's face stored as a horizontal one (the cell above it, 0.15
        m from the wrong segment, kept).
        """
        raw = small_room()
        raw["boundaries"]["box_exhaust"] = {
            "type": "fixed_flow_outlet",
            "location": "obstacle",
            "obstacle": "box",
            "face": face,
            "y_start": 0.0,
            "y_end": 0.1,
            "velocity": 0.1,
        }
        config = SimConfig.from_dict(raw)
        found = cells_of(candidates(Mesh(config), config, ["obstacle_sides"], 0.1))
        assert found == self.found("obstacle_sides") - lost

    def test_a_vent_on_an_obstacle_bottom_excludes_the_room_cell_below_it(self) -> None:
        """A shelf at x 0.1 to 0.2, y 0.3 to 0.4 (cell (3, 1)), its underside venting.

        Under a standoff below the half cell only the cell beside the face
        goes: (2, 1), below the shelf, while (4, 1) above it stays. Defect
        caught: the room's cell taken above the face, inside the shelf.
        """
        raw = small_room()
        raw["obstacles"].append(
            {
                "name": "shelf",
                "x_start": 0.1,
                "x_end": 0.2,
                "y_start": 0.3,
                "y_end": 0.4,
            }
        )
        raw["boundaries"]["shelf_exhaust"] = {
            "type": "fixed_flow_outlet",
            "location": "obstacle",
            "obstacle": "shelf",
            "face": "bottom",
            "x_start": 0.1,
            "x_end": 0.2,
            "velocity": 0.1,
        }
        config = SimConfig.from_dict(raw)
        mesh = Mesh(config)
        assert mesh.cell_type[3, 1] == SOLID
        free = cells_of(candidates(mesh, config, ["free"], 0.01))
        assert (2, 1) not in free and (4, 1) in free

    def test_a_scenario_excludes_the_cells_an_event_opens(self, tmp_path: Path) -> None:
        """The door replaced by an inlet for one second: its wall cells are out.

        The base's door is a wall, so the cells beside it are wall
        candidates there; the scenario's interval with the door open makes
        them an inlet's, and (3, 0), 0.071 m from the door's top, goes with
        them. Defect caught: only the base's segments read.
        """
        base = tmp_path / "room.yaml"
        base.write_text(yaml.safe_dump(small_room()), encoding="utf-8")
        door_in = {
            "name": "door_in",
            "type": "velocity_inlet",
            "location": "left",
            "y_start": 0.0,
            "y_end": 0.3,
            "velocity": 0.1,
            "turbulence_intensity": 0.05,
            "dissipation_length": 0.1,
            "concentration": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
        raw = {
            "scenario": {
                "base": str(base),
                "events": [
                    {
                        "name": "door_open",
                        "start": 1.0,
                        "duration": 1.0,
                        "replace": {"segment": "door", "with": [door_in]},
                    }
                ],
            },
            "solver": {"t_end": 4.0, "output_interval": 10},
            "transport": {"cfl_number": 0.4},
        }
        scenario = SimConfig.from_dict(raw)
        found = cells_of(candidates(Mesh(scenario), scenario, ["walls"], 0.1))
        assert found == self.found("walls") - {(0, 0), (1, 0), (2, 0), (3, 0)}

    def test_a_sensor_cannot_hang_in_the_open_doorway_on_the_product_grid(self) -> None:
        """320x120, the door scenario: 0.05 m holds two columns, and neither is mountable.

        The cells are 0.025 m, so columns 0 and 1 of the doorway's rows (y to
        2.1 m, rows 0 to 83) lie within the standoff of the opening; column 1
        is not beside its faces. The wall above the doorway is mountable from
        the rows the door's top no longer reaches: (86, 0) and (85, 1), and
        (85, 0) and (84, 1) are 0.040 m from it and are not. The second row
        under the supply and the second column of the hood's rows go for the
        same reason, while the ceiling beyond the supply stays. Defect caught:
        the edge counted whole, the exclusion read as beside the face alone
        (84 doorway cells, 282 under the supply and 38 at the hood mounted
        with nothing to mount on).
        """
        config = SimConfig(CONFIGS / "scenario_door_leak.yaml")
        mesh = Mesh(config)
        assert (config.ny, config.nx) == (120, 320)
        spec = config.placement
        mounted = cells_of(candidates(mesh, config, SURFACES, spec.standoff))
        doorway = np.flatnonzero(mesh.yc <= 2.1)
        assert doorway.tolist() == list(range(84))
        for j in doorway:
            assert (int(j), 0) not in mounted and (int(j), 1) not in mounted
        assert {(85, 0), (84, 1), (84, 0)}.isdisjoint(mounted)
        assert {(j, 0) for j in range(86, 120)} <= mounted
        assert {(j, 1) for j in range(85, 120)} <= mounted
        under_supply = np.flatnonzero((mesh.xc >= 0.5) & (mesh.xc <= 7.5))
        assert not {(118, int(i)) for i in under_supply} & mounted
        assert {(118, 0), (119, 0), (118, 319), (119, 319)} <= mounted
        hood = np.flatnonzero((mesh.yc >= 0.9) & (mesh.yc <= 1.8))
        assert not {(int(j), 318) for j in hood} & mounted
        free = cells_of(candidates(mesh, config, ("free",), spec.standoff))
        assert mounted <= free
        assert not {(int(j), 1) for j in doorway} & free

    @pytest.mark.parametrize(
        ("mount", "standoff", "error", "match"),
        [
            ([], 0.1, ValueError, "mount must name at least one"),
            (["walls", 1], 0.1, TypeError, r"mount\[1\] must be a str"),
            (["floor"], 0.1, ValueError, r"mount\[0\] must be one of"),
            (["walls", "walls"], 0.1, ValueError, "names 'walls' twice"),
            (["walls"], 0.0, ValueError, "standoff must be positive"),
            (["walls"], np.inf, ValueError, "standoff must be positive"),
            (["walls"], True, TypeError, "standoff must be a real number"),
            (["walls"], 0.01, ValueError, "no cell is a candidate"),
        ],
    )
    def test_a_mount_or_standoff_that_names_no_cell_is_refused(
        self, mount: list, standoff: object, error: type, match: str
    ) -> None:
        """Known surfaces, none twice, a positive finite standoff, some cell within it."""
        with pytest.raises(error, match=match):
            candidates(self.mesh, self.config, mount, standoff)

    def test_a_mesh_of_another_configuration_is_refused(self) -> None:
        """The candidates read the configuration's segments on its own mesh."""
        other = small_room()
        other["domain"] = dict(other["domain"], nx=20)
        with pytest.raises(
            ValueError, match="candidates need the configurations' mesh"
        ):
            candidates(Mesh(SimConfig.from_dict(other)), self.config, ["walls"], 0.1)


# ---------------------------------------------------------------------------
# Prompt 76b: every scored scenario, the source boxes, the stand, the metric
# ---------------------------------------------------------------------------


def product_config(name: str) -> SimConfig:
    """A reference scenario on the product grid, 320x120."""
    config = SimConfig(CONFIGS / name)
    assert (config.ny, config.nx) == (120, 320)
    return config


@pytest.mark.unit
class TestCandidatesOverScenarios:
    def test_every_scored_scenarios_openings_exclude(self) -> None:
        """The breach's configuration alone gives the doorway back; with the door's, not.

        The breach leaves the doorway a wall, so its cells within 0.05 m of the
        left wall (columns 0 and 1, rows 0 to 83) are wall-mounted candidates.
        The door's intervals open it, and a cell excluded under any scenario
        scored is out. The breach is given first. Defect caught: the union over
        the configurations replaced by the first configuration (the doorway's
        168 cells kept).
        """
        breach = product_config("scenario_filter_breach.yaml")
        door = product_config("scenario_door_leak.yaml")
        mesh = Mesh(breach)
        doorway = {(j, i) for j in range(84) for i in (0, 1)}
        alone = cells_of(candidates(mesh, breach, SURFACES, 0.05))
        assert doorway <= alone
        both = cells_of(candidates(mesh, [breach, door], SURFACES, 0.05))
        assert doorway.isdisjoint(both)
        assert both == cells_of(candidates(mesh, [door, breach], SURFACES, 0.05))
        assert both == cells_of(candidates(mesh, door, SURFACES, 0.05))
        assert cells_of(candidates(mesh, (breach,), SURFACES, 0.05)) == alone

    def test_a_source_box_and_its_ring_are_no_candidate_in_any_mount(self) -> None:
        """The dust's box, x 3.80 to 3.90 and y 2.00 to 2.10, and 0.05 m around it.

        The box holds rows 80 to 83 and columns 152 to 155. A cell one row
        above it, (84, 153) at y 2.1125, and the next, (85, 153) at 2.1375,
        read the source as its own cells do and are out; (86, 153) at 2.1625
        is 0.0625 m away and is a free candidate. On the tool's top, (80, 150)
        at x 3.7625 is out and (80, 149) at 3.7375 is in. The same cells go
        from the free set. Defects caught: the ring dropped ((84, 153) and
        (80, 150) kept); the box read for the mountable set only (the free set
        keeps the box).
        """
        dust = product_config("scenario_equipment_dust.yaml")
        door = product_config("scenario_door_leak.yaml")
        mesh = Mesh(dust)
        box = {(j, i) for j in range(80, 84) for i in range(152, 156)}
        ring = {(84, 153), (85, 153), (80, 150), (80, 151), (81, 156), (81, 157)}
        tops = cells_of(candidates(mesh, [door, dust], ("obstacle_tops",), 0.05))
        free = cells_of(candidates(mesh, [door, dust], (MOUNT_FREE,), 0.05))
        for found in (tops, free):
            assert (box | ring).isdisjoint(found)
        assert {(80, 149), (81, 158)} <= tops
        assert (86, 153) in free
        # Without the dust's configuration the box is a candidate like any cell.
        assert box <= cells_of(candidates(mesh, door, (MOUNT_FREE,), 0.05))

    def test_a_stand_stands_on_the_floor_within_its_band(self) -> None:
        """Stand cells: centre height in [0.3, 1.2] m and no SOLID cell below.

        Rows 12 (y 0.3125) to 47 (y 1.1875) are in the band; rows 11 and 48
        are not. Over the hood's bench (y to 0.9 m, columns 256 to 303) rows
        36 to 47 are air but stand on the bench, not the floor, and are out.
        The doorway's two columns are out under the door's opening, whatever
        the band, and the standoff does not apply to a stand. Defects caught:
        the column check dropped ((40, 280) a stand cell over the bench); the
        band read as the standoff from the floor; the opening's exclusion not
        applied to stands ((20, 0) kept).
        """
        door = product_config("scenario_door_leak.yaml")
        mesh = Mesh(door)
        stands = cells_of(
            candidates(mesh, door, ("stand",), 0.05, stand_height=(0.3, 1.2))
        )
        assert {(12, 100), (47, 100), (40, 250), (20, 2), (30, 310)} <= stands
        assert {(11, 100), (48, 100), (40, 280), (36, 256), (20, 0)}.isdisjoint(stands)
        assert {j for j, _ in stands} == set(range(12, 48))
        assert all(mesh.cell_type[j, i] != SOLID for j, i in stands)
        with_walls = cells_of(
            candidates(mesh, door, ("walls", "stand"), 0.05, stand_height=(0.3, 1.2))
        )
        assert with_walls == stands | cells_of(candidates(mesh, door, ("walls",), 0.05))
        free = cells_of(candidates(mesh, door, (MOUNT_FREE,), 0.05))
        assert stands <= free

    @pytest.mark.parametrize(
        ("mount", "height", "error", "match"),
        [
            (("stand",), None, ValueError, "give its stand_height"),
            (("walls",), (0.1, 0.3), ValueError, "mount does not name 'stand'"),
            (("stand",), (0.1, 0.2, 0.3), ValueError, r"must be \(low, high\)"),
            (("stand",), (0.3, 0.1), ValueError, "0 <= low < high"),
            (("stand",), (-0.1, 0.3), ValueError, "0 <= low < high"),
            (("stand",), (0.1, np.nan), ValueError, "0 <= low < high"),
            (("stand",), (True, 0.3), TypeError, r"stand_height\[0\] must be a real"),
            (("stand",), (0.41, 0.44), ValueError, "no cell is a candidate"),
        ],
    )
    def test_a_stand_height_that_does_not_fit_is_refused(
        self, mount: tuple, height: object, error: type, match: str
    ) -> None:
        """Given exactly with a stand, two finite numbers from the floor up, some cell."""
        config = SimConfig.from_dict(small_room())
        with pytest.raises(error, match=match):
            candidates(Mesh(config), config, mount, 0.1, stand_height=height)

    def test_the_configurations_given_are_checked(self) -> None:
        """At least one, each a SimConfig, each on the mesh's grid."""
        config = SimConfig.from_dict(small_room())
        mesh = Mesh(config)
        with pytest.raises(ValueError, match="give at least one configuration"):
            candidates(mesh, [], ["walls"], 0.1)
        with pytest.raises(TypeError, match=r"configs\[1\] must be a SimConfig"):
            candidates(mesh, [config, "door.yaml"], ["walls"], 0.1)
        other = small_room()
        other["domain"] = dict(other["domain"], nx=20)
        with pytest.raises(ValueError, match="configuration 1's"):
            candidates(mesh, [config, SimConfig.from_dict(other)], ["walls"], 0.1)

    def test_the_standoff_is_euclidean(self) -> None:
        """A cell diagonal to the box's corner: 0.0707 m away, inside a 0.08 m standoff.

        Cell (2, 3), centre (0.35, 0.25), lies 0.05 m left of and above the
        box's top-left corner (0.4, 0.2): 0.0707 m by Euclidean distance and
        0.1 m by Manhattan; (1, 3) lies as far left of and below it, and (2, 6)
        and (1, 6) mirror both at the top-right corner. Defect caught:
        ``gap_x + gap_y`` for the hypotenuse (the four cells out).
        """
        config = SimConfig.from_dict(small_room())
        found = cells_of(candidates(Mesh(config), config, ["obstacle_tops"], 0.08))
        assert {(2, 3), (2, 6), (1, 3), (1, 6)} <= found
        assert found == {(1, 3), (2, 3), (2, 4), (2, 5), (2, 6), (1, 6)}


# ---------------------------------------------------------------------------
# Prompt 76b: the all-clear role
# ---------------------------------------------------------------------------

AREAS = np.ones(LIVE.shape)


def clear_of(cells: dict[tuple[int, int], list[float]], end: float = 0.0) -> ClearUnits:
    """The two classes' all-clear units of frames_of(CELLS), t_e END."""
    clear, times, fractions = clear_times(
        frames_of(cells), THRESHOLDS, LIVE, AREAS, end
    )
    return ClearUnits(
        clear=clear,
        ends=np.full(2, end),
        times=(times, times),
        fractions=tuple(fractions),
        spacing=np.full(2, float(np.diff(times).max())),
        labels=(("s", 0), ("s", 1)),
    )


# Room (1, 0) clears last, at 2.0; (0, 0) at 1.0; (0, 1) at 0.5; the rest are
# never above. The SOLID cell (0, 2) holds 9, above every threshold, throughout.
CLEARING = {
    (0, 0): [2.0, 2.0, 0.0, 0.0],
    (0, 1): [2.0, 0.0, 0.0, 0.0],
    (1, 0): [2.0, 2.0, 2.0, 0.0],
    (0, 2): [9.0, 9.0, 9.0, 9.0],
}


@pytest.mark.unit
class TestClearTimes:
    def test_a_cell_clears_after_its_last_frame_at_or_above(self) -> None:
        """The frame after the last at or above, t_e when none, not clear when the last is.

        (0, 0) dips below at 0.5 s and rises at 1.0 s: it clears at 2.0, not at
        the dip. (0, 1) is above at 0 only: 0.5. (1, 0) is never above: t_e.
        (1, 1) is above at the last frame: not clear (inf), not the run's end.
        (1, 2) is exactly at the threshold at 1.0: 2.0. Defects caught: the
        dip read as clearing ((0, 0) at 0.5); "not clear" read as the run's
        end ((1, 1) at 2.0); ">" for "at or above" ((1, 2) at t_e); SOLID cells
        given a time.
        """
        clear, times, fractions = clear_times(
            frames_of(
                {
                    (0, 0): [2.0, 0.0, 2.0, 0.0],
                    (0, 1): [2.0, 0.0, 0.0, 0.0],
                    (1, 1): [0.0, 0.0, 0.0, 5.0],
                    (1, 2): [0.0, 0.0, 1.0, 0.0],
                    (0, 2): [9.0, 9.0, 9.0, 9.0],
                }
            ),
            THRESHOLDS,
            LIVE,
            AREAS,
            0.0,
        )
        assert times.tolist() == TIMES.tolist()
        row0, row1 = clear[0].tolist()
        assert row0[:2] == [2.0, 0.5] and np.isnan(row0[2])
        assert row1 == [0.0, np.inf, 2.0]
        # Class 1 is ten times the field against ten times the threshold.
        assert np.array_equal(clear[1], clear[0], equal_nan=True)
        # Live area at or above, five live cells: (0, 0) and (0, 1) at 0; none at
        # 0.5; (0, 0) and (1, 2) at 1.0; (1, 1) at 2.0.
        assert fractions[0].tolist() == [0.4, 0.0, 0.4, 0.2]

    def test_frames_before_t_e_are_not_read(self) -> None:
        """From t_e = 0.75 s: the frames at 1.0 and 2.0 only.

        (0, 1) is above at 0 only and clears at t_e itself; (0, 0), above at
        1.0, at 2.0. Defect caught: every frame read (a cell above before t_e
        given that frame's successor, (0, 1) at 0.5).
        """
        clear, times, fractions = clear_times(
            frames_of({(0, 0): [2.0, 0.0, 2.0, 0.0], (0, 1): [2.0, 0.0, 0.0, 0.0]}),
            THRESHOLDS,
            LIVE,
            AREAS,
            0.75,
        )
        assert times.tolist() == [1.0, 2.0]
        assert clear[0, 0, 0] == 2.0 and clear[0, 0, 1] == 0.75
        assert clear[0, 1].tolist() == [0.75, 0.75, 0.75]
        assert fractions.shape == (2, 2)

    def test_the_area_fraction_weighs_cells_by_their_area(self) -> None:
        """A cell three times the others' area is three sevenths of the live area."""
        areas = np.ones(LIVE.shape)
        areas[0, 0] = 3.0
        _, _, fractions = clear_times(
            frames_of({(0, 0): [2.0, 0.0, 0.0, 0.0]}), THRESHOLDS, LIVE, areas, 0.0
        )
        assert fractions[0, 0] == pytest.approx(3.0 / 7.0)

    @pytest.mark.parametrize(
        ("thresholds", "areas", "end", "error", "match"),
        [
            (np.array([1.0]), AREAS, 0.0, ValueError, r"thresholds of shape \(1,\)"),
            (THRESHOLDS, AREAS[:, :2], 0.0, ValueError, "areas are"),
            (THRESHOLDS, AREAS, -1.0, ValueError, "end must be finite"),
            (THRESHOLDS, AREAS, np.nan, ValueError, "end must be finite"),
            (THRESHOLDS, AREAS, True, TypeError, "end must be a real number"),
            (THRESHOLDS, AREAS, 2.5, ValueError, "no stored frame lies at or after"),
        ],
    )
    def test_inputs_that_do_not_fit_the_frames_are_refused(
        self,
        thresholds: np.ndarray,
        areas: np.ndarray,
        end: object,
        error: type,
        match: str,
    ) -> None:
        """One threshold per class, areas of the grid, t_e finite, from the floor, framed."""
        with pytest.raises(error, match=match):
            clear_times(frames_of({}), thresholds, LIVE, areas, end)

    def test_frames_of_another_grid_are_refused(self) -> None:
        """A class whose frames are not live's [ny, nx]."""
        with pytest.raises(ValueError, match="class 0's frames are"):
            clear_times(frames_of({}), THRESHOLDS, LIVE[:, :2], AREAS[:, :2], 0.0)


@pytest.mark.unit
class TestScenarioEnd:
    def test_t_e_is_the_end_of_the_last_event_to_end(self, tmp_path: Path) -> None:
        """Three sources ending at 15, 50 and 35 s: t_e 50, the middle one in the file.

        Defects caught: t_e from the first event's end (15); from the last
        event's in the file (35).
        """
        events = [
            {"name": "a", "start": 10.0, "duration": 5.0, "source": DUST_BOX},
            {"name": "b", "start": 20.0, "duration": 30.0, "source": DUST_BOX},
            {"name": "c", "start": 30.0, "duration": 5.0, "source": DUST_BOX},
        ]
        assert scenario_end(scenario_config(tmp_path, events, 60.0)) == 50.0

    def test_an_initial_ends_at_zero_and_an_event_at_the_run_end_is_none(
        self, tmp_path: Path
    ) -> None:
        """The clean-out's t_e is 0; a scenario whose last event reaches t_end has none."""
        assert scenario_end(SimConfig(CONFIGS / "scenario_clean_out.yaml")) == 0.0
        assert scenario_end(SimConfig(CONFIGS / "scenario_seal_leak.yaml")) == 40.0
        assert scenario_end(SimConfig(CONFIGS / "scenario_door_leak.yaml")) == 15.0
        late = [{"name": "a", "start": 10.0, "duration": 50.0, "source": DUST_BOX}]
        assert scenario_end(scenario_config(tmp_path, late, 60.0)) is None
        assert scenario_end(SimConfig.from_dict(small_room())) is None


DUST_BOX = {
    "x_start": 3.80,
    "x_end": 3.90,
    "y_start": 2.00,
    "y_end": 2.10,
    "emission": [1.0e4] * 5,
}


def scenario_config(tmp_path: Path, events: list, t_end: float) -> SimConfig:
    """A scenario of EVENTS over the default room on 80x30, run to T_END."""
    return SimConfig.from_dict(
        {
            "scenario": {
                "base": str(CONFIGS / "clean_room_default.yaml"),
                "events": events,
            },
            "domain": {"nx": 80, "ny": 30},
            "solver": {"t_end": t_end, "output_interval": 100},
            "transport": {"cfl_number": 0.4},
        },
        base_dir=tmp_path,
    )


@pytest.mark.unit
class TestClearScore:
    def test_the_layout_clears_when_its_last_cell_does(self) -> None:
        """Clear times 1.0 and 0.5: the layout's is 1.0, the room's 2.0 at (1, 0).

        The interval is the room's less the layout's, 1.0; the live area at or
        above at the layout's clear time is (1, 0) alone, a fifth. The SOLID
        cell, above throughout, sets nothing. Defects caught: the layout's
        clear time taken as the earliest of its cells (0.5, interval 1.5);
        the interval's sign flipped (-1.0); SOLID cells counted in the room's
        clear time (inf, or the SOLID cell named).
        """
        result = clear_score(clear_of(CLEARING), [(0, 0), (0, 1)])
        assert result.layout.tolist() == [1.0, 1.0]
        assert result.room.tolist() == [2.0, 2.0]
        assert result.room_cells == ((1, 0), (1, 0))
        assert result.interval.tolist() == [1.0, 1.0]
        assert result.fraction.tolist() == [0.2, 0.2]
        assert (result.worst, result.mean, result.worst_fraction) == (1.0, 1.0, 0.2)
        alone = clear_score(clear_of(CLEARING), [(0, 1)])
        assert alone.interval.tolist() == [1.5, 1.5]
        assert alone.fraction.tolist() == [0.4, 0.4]
        last = clear_score(clear_of(CLEARING), [(1, 0), (0, 1)])
        assert last.interval.tolist() == [0.0, 0.0] and last.fraction.tolist() == [0, 0]

    def test_a_not_clear_never_enters_arithmetic(self) -> None:
        """(1, 1) above at the last frame: the room is not clear.

        A layout that clears is reported with no interval and ranks worst
        (inf); a layout holding (1, 1) never declares the room clean, so it
        declares nothing early: no interval, no fraction, rank zero. Defect
        caught: "not clear" read as the run's end (an interval of 2.0 less the
        layout's, a finite number).
        """
        units = clear_of({**CLEARING, (1, 1): [0.0, 0.0, 0.0, 5.0]})
        early = clear_score(units, [(0, 0)])
        assert np.isinf(early.room).all() and early.room_cells[0] == (1, 1)
        assert np.isnan(early.interval).all() and early.layout.tolist() == [1.0, 1.0]
        assert np.isinf(early.worst) and np.isinf(early.mean)
        holding = clear_score(units, [(0, 0), (1, 1)])
        assert np.isinf(holding.layout).all() and np.isnan(holding.interval).all()
        assert np.isnan(holding.fraction).all()
        assert (holding.worst, holding.mean, holding.worst_fraction) == (0.0, 0.0, 0.0)

    def test_the_room_ties_go_to_the_first_cell_in_row_major_order(self) -> None:
        """Two cells clear last at 2.0: the room names the first, (0, 0)."""
        units = clear_of({(0, 0): [2.0, 2.0, 2.0, 0.0], (1, 0): [2.0, 2.0, 2.0, 0.0]})
        assert room_clear(units)[1] == ((0, 0), (0, 0))

    def test_a_layout_off_the_grid_or_solid_is_refused(self) -> None:
        """The cell checks the detection scoring makes."""
        with pytest.raises(ValueError, match="SOLID"):
            clear_score(clear_of(CLEARING), [(0, 2)])
        with pytest.raises(ValueError, match="is empty"):
            clear_score(clear_of(CLEARING), [])


def line_units(
    rows: Sequence[Sequence[float]], fractions: Sequence[Sequence[float]]
) -> ClearUnits:
    """All-clear units on a 1 x n grid: ROWS[u][c] is unit u's clear time at cell c.

    Each unit's frames are at 0, 1, 2, ... s, one per entry of FRACTIONS[u], the
    live area above at each.
    """
    times = np.arange(float(len(fractions[0])))
    return ClearUnits(
        clear=np.array(rows, dtype=np.float64)[:, None, :],
        ends=np.zeros(len(rows)),
        times=tuple(times for _ in rows),
        fractions=tuple(np.array(f, dtype=np.float64) for f in fractions),
        spacing=np.ones(len(rows)),
        labels=tuple(("s", u) for u in range(len(rows))),
    )


@pytest.mark.unit
class TestAddAllClear:
    # Two units, the room clearing at 4 in both. Cell 0 is the detection
    # sensor (clear at 1). Added to it, cell 1 gives intervals (2, 2); cell 2
    # (0, 2), read at the area fractions of t = 4 and 2, at most 0.1; cell 3
    # (1, 1); cell 4 (2, 0), read at t = 2 and 4, at most 0.3; cell 5 (0, 0).
    ROWS = ((1.0, 2.0, 4.0, 3.0, 2.0, 4.0), (1.0, 2.0, 2.0, 3.0, 4.0, 4.0))
    FRACTIONS = ((0.5, 0.4, 0.3, 0.2, 0.0), (0.5, 0.4, 0.1, 0.6, 0.0))

    def test_each_is_the_least_worst_interval_ties_by_the_mean_then_the_area(
        self,
    ) -> None:
        """The least worst interval, then the least mean, then the least area fraction.

        Cell 5 (worst 0) comes first; after it every cell leaves the layout's
        clear times at 4, and the tie goes to candidate order (cell 1). Of
        cells 1 to 4, cell 3 (worst 1). Of cells 1, 4 and 2, all worst 2:
        cell 1's mean is 2, cells 4 and 2 tie at 1, and cell 2's largest
        fraction, 0.1, is below cell 4's 0.3. Defects caught: the fraction
        tie-break dropped (cell 4); the worst case dropped (cell 2 or 4 before
        cell 3 would need a lower mean, which the next test plants).
        """
        units = line_units(self.ROWS, self.FRACTIONS)
        pool = [(0, c) for c in range(1, 6)]
        assert add_all_clear(units, pool, [(0, 0)], 2) == ((0, 5), (0, 1))
        assert add_all_clear(units, pool[:4], [(0, 0)], 1) == ((0, 3),)
        ties = [(0, 1), (0, 4), (0, 2)]
        assert add_all_clear(units, ties, [(0, 0)], 1) == ((0, 2),)

    def test_the_worst_comes_before_the_mean_and_the_mean_before_the_area(
        self,
    ) -> None:
        """Three units, rooms clearing at 6; cells 1 to 3 against the detection cell 0.

        Cell 1 clears at (4, 4, 4): intervals (2, 2, 2), mean 2, largest
        fraction 0.1. Cell 2 at (4, 5, 6): (2, 1, 0), mean 1, but unit 1's
        field rises again at 5 s, so its fraction is 0.3. Cell 3 at
        (6, 6, 3.5): (0, 0, 2.5), worst 2.5, mean 0.83. Cell 2 is the answer.
        Defects caught: the worst and the mean swapped (cell 3, the least
        mean); the mean tie-break dropped (cell 1, the least fraction among the
        worst-2 cells).
        """
        units = line_units(
            [
                [1.0, 4.0, 4.0, 6.0, 6.0],
                [1.0, 4.0, 5.0, 6.0, 6.0],
                [1.0, 4.0, 6.0, 3.5, 6.0],
            ],
            [
                [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0],
                [0.5, 0.4, 0.3, 0.2, 0.05, 0.3, 0.0],
                [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0],
            ],
        )
        assert room_clear(units)[0].tolist() == [6.0, 6.0, 6.0]
        pool = [(0, 1), (0, 2), (0, 3)]
        assert add_all_clear(units, pool, [(0, 0)], 1) == ((0, 2),)
        scores = [clear_score(units, [(0, 0), cell]) for cell in pool]
        assert [s.worst for s in scores] == [2.0, 2.0, 2.5]
        assert [s.mean for s in scores] == pytest.approx([2.0, 1.0, 2.5 / 3.0])
        assert [s.worst_fraction for s in scores] == [0.1, 0.3, 0.1]

    def test_count_zero_adds_nothing_and_a_count_past_the_pool_is_refused(
        self,
    ) -> None:
        """From 0 to the candidates not already in the layout."""
        units = line_units(self.ROWS, self.FRACTIONS)
        pool = [(0, c) for c in range(6)]
        assert add_all_clear(units, pool, [(0, 0)], 0) == ()
        with pytest.raises(ValueError, match="count must be from 0 to the 5"):
            add_all_clear(units, pool, [(0, 0)], 6)
        with pytest.raises(TypeError, match="count must be an int"):
            add_all_clear(units, pool, [(0, 0)], True)
        assert len(add_all_clear(units, pool, [], 6)) == 6

    def test_the_whole_search_detects_then_clears(self) -> None:
        """search_roles: the detection search at count - all_clear, then the additions."""
        units = line_units(self.ROWS, self.FRACTIONS)
        detection = units_of(
            [[0.5, 1.0, 9.0, 9.0, 9.0, 9.0], [0.5, 2.0, 9.0, 9.0, 9.0, 1.0]]
        )
        pool = [(0, c) for c in range(6)]
        roles = search_roles(detection, units, pool, 2, 1, WORST_CASE)
        assert roles.detection.cells == ((0, 0),)
        assert roles.all_clear == ((0, 5),)
        assert roles.layout.cells == ((0, 0), (0, 5))
        assert roles.clear.worst == 0.0
        assert roles.layout.units == detection.labels
        with pytest.raises(ValueError, match="all_clear must be from 0 to count - 1"):
            search_roles(detection, units, pool, 2, 2, WORST_CASE)
        with pytest.raises(TypeError, match="all_clear must be an int"):
            search_roles(detection, units, pool, 2, 1.0, WORST_CASE)
        other = line_units([[1.0, 2.0]], [[0.0] * 5])
        with pytest.raises(ValueError, match="they must share a mesh"):
            search_roles(detection, other, pool, 2, 1, WORST_CASE)
        none = search_roles(detection, units, pool, 1, 0, WORST_CASE)
        assert none.all_clear == () and none.layout.cells == none.detection.cells


@pytest.mark.unit
class TestClearUnitsFromRuns:
    def _run(
        self,
        tmp_path: Path,
        name: str,
        events: list,
        t_end: float,
        grid: tuple[int, int] = (80, 30),
    ) -> RunFile:
        """A run file of EVENTS: four frames at 0, 10, 20 and 30 s, above until 25 s."""
        mapping = {
            "scenario": {
                "base": str(CONFIGS / "clean_room_default.yaml"),
                "events": events,
            },
            "domain": {"nx": grid[0], "ny": grid[1]},
            "solver": {"t_end": t_end, "output_interval": 100},
            "transport": {"cfl_number": 0.4},
        }
        mesh = Mesh(SimConfig.from_dict(mapping))
        run = RunFile(tmp_path / f"{name}.h5")
        run.write_config(mapping, **PROVENANCE)
        live = mesh.cell_type != SOLID
        for step, t in enumerate((0.0, 10.0, 20.0, 30.0)):
            fields = {k: np.where(live & (t < 25.0), 1.0e9, 0.0) for k in range(5)}
            run.write_frame(step, t, fields, [MassBudget()] * 5)
        return run

    def test_the_units_of_runs_that_end_in_time(self, tmp_path: Path) -> None:
        """A run whose source ends at 15 s gives five units; one ending at t_end none."""
        early = [{"name": "a", "start": 5.0, "duration": 10.0, "source": DUST_BOX}]
        late = [{"name": "b", "start": 5.0, "duration": 25.0, "source": DUST_BOX}]
        runs = {
            "early": self._run(tmp_path, "early", early, 30.0),
            "late": self._run(tmp_path, "late", late, 30.0),
        }
        units = clear_units(runs)
        assert units.labels == tuple(("early", k) for k in range(5))
        assert units.ends.tolist() == [15.0] * 5
        assert units.spacing.tolist() == [10.0] * 5
        assert all(times.tolist() == [20.0, 30.0] for times in units.times)
        live = ~np.isnan(units.clear[0])
        assert (units.clear[:, live] == 30.0).all()
        assert units.fractions[0] == pytest.approx([1.0, 0.0], abs=1e-12)
        with pytest.raises(ValueError, match="there is no all-clear unit"):
            clear_units({"late": runs["late"].path})
        with pytest.raises(ValueError, match="no run to read"):
            clear_units({})
        with pytest.raises(TypeError, match="a scenario name must be a str"):
            clear_units({1: runs["early"]})
        coarse = self._run(tmp_path, "coarse", early, 30.0, grid=(40, 15))
        with pytest.raises(ValueError, match="the runs must share a mesh"):
            clear_units({"early": runs["early"], "coarse": coarse})
