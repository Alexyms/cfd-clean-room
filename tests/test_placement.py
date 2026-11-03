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

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import MEAN, WORST_CASE, SimConfig
from src.io_manager import RunFile, TransportFrames
from src.mesh import SOLID, Mesh
from src.placement import (
    ARGMIN,
    EXHAUSTIVE,
    GIVEN,
    GREEDY,
    MAX_SET_COVER_UNITS,
    SET_COVER,
    Layout,
    RunArrival,
    Score,
    Units,
    _worst_case_cover,
    arrival_from_frames,
    arrival_from_run,
    candidates,
    detection_units,
    exclude_units,
    given_layout,
    latency,
    rank,
    score,
    search,
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
    return Units(latencies, caps, labels)


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
    maps: np.ndarray, onsets: list[float], t_final: float, names: tuple[str, ...]
) -> RunArrival:
    """A RunArrival from planted maps, step-resolved."""
    return RunArrival(maps, np.array(onsets), "step", names, t_final)


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
    units = Units(latencies, np.full(6, 10.0), tuple(("s", "e", u) for u in range(6)))
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

        The walls are columns 0 and 9; the ceiling row 5 but for the two
        cells under the supply; the box's top row 2 from i 3 to 6, and (1, 3)
        and (1, 6) beside its sides, 0.071 m from its top corners as (2, 3)
        and (2, 6) are; its sides columns 3 and 6 from j 0 to 2, and (2, 4)
        and (2, 5) above the top, 0.071 m from the sides' upper ends. Defect
        caught: the standoff measured from the box's bottom face (on the
        floor, so no face and no top candidate).
        """
        rows = range(6)
        assert self.found("walls") == {(j, 0) for j in rows} | {(j, 9) for j in rows}
        assert self.found("ceiling") == {(5, i) for i in range(10)} - {(5, 3), (5, 4)}
        tops = {(2, 3), (2, 4), (2, 5), (2, 6), (1, 3), (1, 6)}
        assert self.found("obstacle_tops") == tops
        sides = {(j, 3) for j in range(3)} | {(j, 6) for j in range(3)}
        sides |= {(2, 4), (2, 5)}
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

    def test_cells_beside_an_inlet_or_outlet_face_are_excluded(self) -> None:
        """The supply's two cells and the return's two are never candidates, free or not.

        Defects caught: the inlet's cells kept on the ceiling; the free set
        not excluding them.
        """
        free = self.found("free")
        live = Mesh(self.config).cell_type != SOLID
        openings = {(5, 3), (5, 4), (0, 7), (0, 8)}
        assert (
            free == {(j, i) for j, i in zip(*np.nonzero(live), strict=True)} - openings
        )
        assert not openings & self.found("walls", "ceiling", "obstacle_tops")

    def test_a_vent_on_an_obstacle_face_excludes_the_room_cell_beside_it(self) -> None:
        """An exhaust over the box's top half from x 0.4 to 0.5: cell (2, 4) is out."""
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
        assert found == {(2, 3), (2, 5), (2, 6), (1, 3), (1, 6)}

    def test_a_scenario_excludes_the_cells_an_event_opens(self, tmp_path: Path) -> None:
        """The door replaced by an inlet for one second: its three wall cells are out.

        The base's door is a wall, so the cells beside it are wall
        candidates there; the scenario's interval with the door open makes
        them an inlet's. Defect caught: only the base's segments read.
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
        assert found == self.found("walls") - {(0, 0), (1, 0), (2, 0)}

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
            ValueError, match="candidates need the configuration's mesh"
        ):
            candidates(Mesh(SimConfig.from_dict(other)), self.config, ["walls"], 0.1)
