"""Tests of the comparator of REQ-N03 level 2 (``scripts/compare_runs.py``, ADR-017 H).

The measures first, on planted arrays: a difference read the same whichever its
sign, a time reached in one run only read as a move of infinity (a move that
straddles a threshold), a move across an interval's end read in the step of the
later time, and a field in each of the three tiers. Test 85 S1 found these
unpinned in prompt 85's comparators, from which these come.

Then whole run files: one short dust release on 80x30 run on the reference, and
copies of its file with one record planted, each of which the comparison must
read as the failing verdict of its kind (a decision, a field, a stop, a budget
verdict and the closure), and the unchanged copy as a pass, bit for bit.
"""

import math
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_runs  # noqa: E402 -- scripts/ is not a package; path set above
from compare_runs import (  # noqa: E402 -- follows sys.path.insert
    BAND,
    BETWEEN,
    BEYOND,
    BUDGET,
    CLOSED,
    DECISION,
    FIELD,
    INSIDE,
    STOP,
    Comparison,
    Item,
    Moves,
    compare,
    grid_index,
    index_moves,
    moved,
    pointwise,
    relative,
    tier,
)

from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.io_manager import RunFile, config_text  # noqa: E402 -- follows sys.path.insert
from src.placement import (  # noqa: E402 -- follows sys.path.insert
    arrival_from_run,
    clear_units,
)
from src.time_integration import Provenance, run  # noqa: E402
from validation.transport_cases import conservation_keys  # noqa: E402

CONFIGS = ROOT / "configs"
PROVENANCE = Provenance("test", "2026-11-26T00:00:00", "test-machine", 0)


# ---- The measures --------------------------------------------------------------


@pytest.mark.unit
def test_relative_reads_a_difference_of_either_sign_the_same() -> None:
    """A cell moved up or down by d reads as d over the largest magnitude.

    Defect caught: the difference taken signed, which reads a field the port
    lowered as no difference (test 85 S1, P14).
    """
    a = np.linspace(1.0, 10.0, 12).reshape(3, 4)
    up, down = a.copy(), a.copy()
    up[1, 2] += 1e-9
    down[1, 2] -= 1e-9
    assert relative(a, a.copy()) == 0.0
    assert relative(a, up) == pytest.approx(1e-10, rel=1e-6)
    assert relative(a, down) == pytest.approx(1e-10, rel=1e-6)
    # A negative field: the scale is its largest magnitude, not its largest value.
    assert relative(-a, -up) == pytest.approx(1e-10, rel=1e-6)


@pytest.mark.unit
def test_relative_reads_a_change_of_kind_as_infinite() -> None:
    """A value gone non-finite, an infinity changed, a shape changed: infinity.

    NaN where both are NaN and an infinity where both hold it compare equal.
    Defect caught: prompt 85's measure, which compared the finite entries of
    both and so read a cell the port made NaN as no difference.
    """
    a = np.array([[1.0, 2.0], [np.nan, np.inf]])
    assert relative(a, a.copy()) == 0.0
    gone = a.copy()
    gone[0, 0] = np.nan
    assert relative(a, gone) == math.inf
    flipped = a.copy()
    flipped[1, 1] = -np.inf
    assert relative(a, flipped) == math.inf
    assert relative(a, np.ones((2, 3))) == math.inf
    assert relative(np.zeros(3), np.array([0.0, 1e-300, 0.0])) == math.inf


@pytest.mark.unit
def test_pointwise_reads_either_sign_above_the_floor_only() -> None:
    """The largest |b - a| / |a| where |a| is at least the floor, either sign."""
    a = np.array([1.0, 100.0, 0.5])
    b = np.array([1.0 - 1e-6, 100.0 + 2e-6, 0.5 + 1.0])
    assert pointwise(a, b, 0.9) == pytest.approx(1e-6, rel=1e-6)
    assert pointwise(a, b, 0.4) == pytest.approx(2.0, rel=1e-12)
    assert pointwise(a, b, 1000.0) == 0.0
    assert pointwise(a, np.array([np.nan, 100.0, 0.5]), 0.9) == math.inf


@pytest.mark.unit
@pytest.mark.parametrize("name", list(BAND))
def test_tier_reads_each_of_the_three(name: str) -> None:
    """At or below the spread inside; to the limit between; above it beyond.

    Defect caught: the two bounds swapped, which reads a difference ten times
    the spread as inside it.
    """
    spread, limit = BAND[name]
    assert spread < limit
    assert tier(0.0, name) == INSIDE
    assert tier(spread, name) == INSIDE
    assert tier(spread * 1.01, name) == BETWEEN
    assert tier(limit, name) == BETWEEN
    assert tier(limit * 1.01, name) == BEYOND
    assert tier(math.inf, name) == BEYOND


# Two intervals: ten steps of 0.1 s to 1.0 s, then five of 0.2 s to 2.0 s.
TIMES = np.concatenate([np.linspace(0.0, 1.0, 11), np.linspace(1.2, 2.0, 5)])


@pytest.mark.unit
def test_grid_index_is_the_nearest_time_and_keeps_never_and_none() -> None:
    """A time falls on the grid time nearest it; inf and NaN stay as given.

    A time that was added to and subtracted from (latency plus onset) is a
    rounding off its step and still falls on that step. Defect caught: the first
    time at or after it, which puts a time a rounding above its step on the
    next one, and so reads a decision that did not move as moved by one.
    """
    wobble = 1.0 + 4.0e-16
    times = np.array([0.0, 0.1 * wobble, 1.0, 1.0 / wobble, 1.2, 2.0, np.inf, np.nan])
    index = grid_index(TIMES, times)
    np.testing.assert_array_equal(index[:6], [0, 1, 10, 10, 11, 15])
    assert index[6] == math.inf and math.isnan(index[7])
    # A time before the first grid time, or past the last, falls on the end one.
    np.testing.assert_array_equal(grid_index(TIMES, np.array([-1.0, 5.0])), [0, 15])
    assert grid_index(TIMES, np.array([0.45]))[0] in (4, 5)
    with pytest.raises(ValueError, match="at least one"):
        grid_index(np.array([]), np.array([1.0]))


@pytest.mark.unit
def test_index_moves_count_the_steps_between_the_indexes_across_a_switch() -> None:
    """Across the switch a time moved from 1.0 s to 1.2 s is one step, either way.

    The two are steps 10 and 11 of the run, whatever their lengths. Defect
    caught: the move read in a step's length (two steps from the earlier time's
    0.1 s, or half a step), or as signed (minus one).
    """
    ref = np.array([1.0, 1.2, 0.5, 0.5])
    other = np.array([1.2, 1.0, 0.7, 0.3])
    np.testing.assert_array_equal(index_moves(ref, other, TIMES), [1.0, 1.0, 2.0, 2.0])


@pytest.mark.unit
def test_index_moves_read_each_run_on_its_own_grid() -> None:
    """Two runs whose steps differ: a time is a step of its own run.

    The other run's steps are twice as long, so its step 5 is at 1.0 s where
    the reference's step 10 is: the index differs by five, not zero.
    """
    other_grid = np.linspace(0.0, 2.0, 11)
    moves = index_moves(np.array([1.0]), np.array([1.0]), TIMES, other_grid)
    assert moves[0] == 5.0
    assert index_moves(np.array([1.0]), np.array([1.0]), TIMES, TIMES)[0] == 0.0


@pytest.mark.unit
def test_a_threshold_crossed_in_one_run_only_is_a_move_of_infinity() -> None:
    """A time in one run and never in the other moves by infinity, either way.

    Both never is no move; both NaN (a SOLID cell) is not compared. Defect
    caught: a straddling move read as no move (test 85 S1, P17).
    """
    ref = np.array([0.5, np.inf, np.inf, np.nan, 0.5])
    other = np.array([np.inf, 0.5, np.inf, np.nan, 0.5])
    out = moved(ref, other, 0.1)
    assert out[0] == math.inf and out[1] == math.inf
    assert out[2] == 0.0 and math.isnan(out[3]) and out[4] == 0.0
    counts = Moves.of(out)
    assert (counts.compared, counts.one, counts.more, counts.only) == (4, 0, 0, 2)
    assert not counts.equal


@pytest.mark.unit
def test_moves_count_one_step_more_and_the_largest() -> None:
    """One is any move up to a step, more is above it; a fraction is not no move.

    Defect caught: a move rounded to the nearest step, which reads 0.4 of a
    step as none (test 96 B1).
    """
    counts = Moves.of(np.array([0.0, 0.4, 1.0, 2.0, 3.0, np.nan]))
    assert (counts.compared, counts.one, counts.more, counts.only) == (5, 2, 2, 0)
    assert counts.largest == 3.0
    assert Moves.of(np.zeros(4)).equal
    assert not Moves.of(np.array([0.0, 0.4])).equal
    assert not Moves.of(np.array([0.0, 2.0])).equal


def _stop(apart: int) -> Item:
    return Item("stop", "s", apart <= 1, {"kind": STOP, "apart": apart})


@pytest.mark.unit
def test_a_middle_tier_field_passes_only_when_an_outer_stop_is_one_apart() -> None:
    """Explained by a stop one apart; not by equal stops, nor by stops two apart.

    Beyond the limit fails whatever the stops say.
    """
    assert compare_runs._explained([_stop(0), _stop(1)])
    assert not compare_runs._explained([_stop(0), _stop(0)])
    assert not compare_runs._explained([_stop(1), _stop(2)])
    assert not compare_runs._explained([])
    spread, limit = BAND["u"]
    inside = compare_runs._field("u", spread, "u", False, "")
    between = compare_runs._field("u", spread * 2, "u", False, "")
    explained = compare_runs._field("u", spread * 2, "u", True, "")
    beyond = compare_runs._field("u", limit * 2, "u", True, "")
    assert inside.passed and inside.detail["tier"] == INSIDE
    assert not between.passed and between.detail["tier"] == BETWEEN
    assert explained.passed
    assert not beyond.passed and beyond.detail["tier"] == BEYOND


@pytest.mark.unit
def test_the_verdict_names_each_failing_kind_once() -> None:
    """PASS with no failing item; FAIL naming each kind once, in the items' order."""
    good = Item("decision", "a", True, {"kind": DECISION})
    assert Comparison([good], {}).verdict == "PASS"
    items = [
        Item("stop", "s", False, {"kind": STOP}),
        Item("field", "f", False, {"kind": FIELD, "tier": BEYOND}),
        Item("field", "g", False, {"kind": FIELD, "tier": BEYOND}),
        good,
    ]
    comparison = Comparison(items, {})
    assert comparison.failures == [STOP, FIELD]
    assert comparison.verdict == f"FAIL: {STOP}; {FIELD}"
    assert not comparison.first_tier


# ---- Whole run files -----------------------------------------------------------


@pytest.fixture(scope="module")
def dust(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A short dust release on 80x30 from rest, on the reference.

    The release from 2 s to 6 s and the run to 10.6 s, so that its maps and the
    clear times have entries and its last stored frame lies close to the one
    before it (the last gap is 0.02 of the spacing, as in every product run,
    whose last step ends between two frames); the grid and the tolerance are
    the suite's coarse ones (``tests/test_time_integration.py``).
    """
    raw = yaml.safe_load(
        (CONFIGS / "scenario_equipment_dust.yaml").read_text(encoding="utf-8")
    )
    raw["scenario"]["base"] = str(CONFIGS / "clean_room_default.yaml")
    raw["domain"] = {"nx": 80, "ny": 30}
    raw["solver"]["mass_imbalance_tol"] = (
        conservation_keys()["conservation_limit"] * 1.2 * (0.1 * 0.1) / 60.0
    )
    raw["solver"]["t_end"] = 10.6
    raw["scenario"]["events"][0]["start"] = 2.0
    raw["scenario"]["events"][0]["duration"] = 4.0
    path = tmp_path_factory.mktemp("dust") / "dust.h5"
    run(SimConfig.from_dict(raw), path, provenance=PROVENANCE)
    return path


def _copy(source: Path, tmp_path: Path, name: str) -> Path:
    out = tmp_path / f"{name}.h5"
    shutil.copy(source, out)
    return out


def _flow_group(handle: h5py.File) -> h5py.Group:
    (key,) = list(handle["flow"])
    return handle["flow"][key]


@pytest.mark.integration
def test_a_copy_passes_bit_for_bit(dust: Path, tmp_path: Path) -> None:
    """The run against its own copy: PASS, every field inside, every record bitwise."""
    comparison = compare(dust, _copy(dust, tmp_path, "copy"))
    assert comparison.verdict == "PASS"
    assert comparison.first_tier
    assert comparison.bitwise and all(comparison.bitwise.values())
    names = [item.name for item in comparison.items]
    # The run reaches what the items read: finite arrival times and clear times.
    assert any(name.startswith("arrival times") for name in names)
    assert any(name.startswith("cells' clear times") for name in names)
    arrival = next(i for i in comparison.items if i.name.startswith("arrival times"))
    assert arrival.detail["compared"] > 0


@pytest.mark.integration
@pytest.mark.parametrize(
    ("factor", "level"),
    [(1.0e-8, INSIDE), (3.0e-7, BETWEEN), (1.0e-5, BEYOND)],
)
def test_a_class_field_moved_reads_in_its_tier(
    dust: Path, tmp_path: Path, factor: float, level: str
) -> None:
    """Class 2's frames scaled by 1 + f: inside passes, between and beyond fail.

    Between fails here because every stop is equal, so none explains it.
    """
    other = _copy(dust, tmp_path, "class")
    with h5py.File(other, "r+") as handle:
        data = handle["transport/C_2"]
        data[...] = data[...] * (1.0 + factor)
    comparison = compare(dust, other)
    item = next(
        i for i in comparison.items if i.name.startswith("concentration, every")
    )
    assert item.detail["tier"] == level
    assert item.detail["difference"] == pytest.approx(factor, rel=1e-3)
    if level == INSIDE:
        assert comparison.verdict == "PASS"
        assert not comparison.bitwise["frames"]
    else:
        assert FIELD in comparison.failures


@pytest.mark.integration
@pytest.mark.parametrize(("outer", "passes"), [(0, False), (1, True), (2, False)])
def test_a_flow_field_in_the_middle_tier_needs_an_outer_stop_one_apart(
    dust: Path, tmp_path: Path, outer: int, passes: bool
) -> None:
    """u scaled by 1 + 2e-9, the outer count moved by 0, 1 or 2.

    One apart explains the field and passes; equal counts leave it unexplained;
    two apart fails the stop and explains nothing.
    """
    other = _copy(dust, tmp_path, "flow")
    with h5py.File(other, "r+") as handle:
        group = _flow_group(handle)
        group["u_faces"][...] = group["u_faces"][...] * (1.0 + 2.0e-9)
        group.attrs["outer"] = int(group.attrs["outer"]) + outer
    comparison = compare(dust, other)
    item = next(i for i in comparison.items if i.name.startswith("u, state"))
    assert item.detail["tier"] == BETWEEN
    assert (comparison.verdict == "PASS") is passes
    if outer == 2:
        assert comparison.failures[0] == STOP
    if outer == 0:
        assert comparison.failures == [FIELD]


def _scaled_class_frames(handle: h5py.File, factor: float) -> None:
    """Every class's frames scaled by 1 + factor: the concentration's two items move."""
    for name in handle["transport"]:
        if name.startswith("C_"):
            data = handle["transport"][name]
            data[...] = data[...] * (1.0 + factor)


def _scaled_sensor_channels(handle: h5py.File, factor: float) -> None:
    """The sensor channels scaled by 1 + factor."""
    data = handle["transport/sensors/channels"]
    data[...] = data[...] * (1.0 + factor)


def _moved_leftover(handle: h5py.File, factor: float) -> None:
    """Class 0's leftover raised by factor of its supply, the content lowered by as much.

    The budget still closes (the residual is unchanged), so only the leftover's
    change and, past REQ-T05's limit, the verdict move.
    """
    budgets = handle["transport/budgets"]
    values = budgets[...]
    supplied = values[-1, 0, 0] + values[-1, 0, 1] + values[-1, 0, 3]
    values[-1, 0, -1] += factor * supplied
    values[-1, 0, -2] -= factor * supplied
    budgets[...] = values


@pytest.mark.integration
@pytest.mark.parametrize(
    ("plant", "prefix"),
    [
        (_scaled_class_frames, "concentration, every"),
        (_scaled_class_frames, "concentration, pointwise"),
        (_scaled_sensor_channels, "sensor channels"),
        (_moved_leftover, "leftover / supplied"),
    ],
    ids=["frames", "pointwise", "sensors", "leftover"],
)
@pytest.mark.parametrize("outer", [0, 1])
def test_a_class_field_in_the_middle_tier_is_not_explained_by_the_outer_count(
    dust: Path,
    tmp_path: Path,
    plant: Callable[[h5py.File, float], None],
    prefix: str,
    outer: int,
) -> None:
    """A class field between its spread and its limit fails, one outer count apart or not.

    ADR-017 H lets the stop counts explain a middle-tier field only when they
    are of the kind that moved it that far in the probe: an outer count one
    apart moved the class fields by 2.7e-11 at most, inside their first tier,
    and a Jacobi sweep, which a run file does not record, moved them further.
    So the comparator reads the field as beyond the limit and says why, where
    the same count explains a flow field. The leftover moves by 3e-9 of the
    supply, the others by 3e-7.

    Defect caught: the flag that explains a flow field handed to a class field
    (test 96 B1), at each of the four items that read it.
    """
    other = _copy(dust, tmp_path, "middle")
    factor = 3.0e-9 if plant is _moved_leftover else 3.0e-7
    with h5py.File(other, "r+") as handle:
        plant(handle, factor)
        group = _flow_group(handle)
        group.attrs["outer"] = int(group.attrs["outer"]) + outer
    comparison = compare(dust, other)
    item = next(i for i in comparison.items if i.name.startswith(prefix))
    assert item.detail["tier"] == BETWEEN
    assert not item.passed
    assert item.detail["explained by the stops"] is False
    assert item.detail["note"] == compare_runs.MIDDLE_TIER_NOTE
    assert FIELD in comparison.failures
    # The stop is within one: it is the field that fails, not the stop.
    assert STOP not in comparison.failures


@pytest.mark.integration
def test_a_class_field_inside_its_spread_passes_with_the_outer_count_one_apart(
    dust: Path, tmp_path: Path
) -> None:
    """The control: the same plant at a hundredth of the size is inside and passes."""
    other = _copy(dust, tmp_path, "inside")
    with h5py.File(other, "r+") as handle:
        _scaled_class_frames(handle, 3.0e-9)
        group = _flow_group(handle)
        group.attrs["outer"] = int(group.attrs["outer"]) + 1
    assert compare(dust, other).verdict == "PASS"


@pytest.mark.integration
def test_a_stop_for_another_reason_fails(dust: Path, tmp_path: Path) -> None:
    """The same outer count, another reason: a stop that differs."""
    other = _copy(dust, tmp_path, "reason")
    with h5py.File(other, "r+") as handle:
        _flow_group(handle).attrs["stop_reason"] = "velocity_step_below_tol"
    assert compare(dust, other).failures == [STOP]


def _arrival_planted(dust: Path, tmp_path: Path, name: str, change: str) -> Comparison:
    other = _copy(dust, tmp_path, name)
    with h5py.File(other, "r+") as handle:
        maps = handle["arrival/first_above"]
        values = maps[...]
        times = handle["transport/sensors/times"][...]
        index = tuple(np.argwhere(np.isfinite(values) & (values > times[1]))[0])
        if change == "later":
            at = int(np.searchsorted(times, values[index]))
            values[index] = times[at + 1]
        else:
            values[index] = np.inf
        maps[...] = values
    return compare(dust, other)


@pytest.mark.integration
@pytest.mark.parametrize("change", ["later", "never"])
def test_an_arrival_moved_one_step_or_never_reached_fails(
    dust: Path, tmp_path: Path, change: str
) -> None:
    """One cell's arrival a step later, or never: a decision that differs.

    Defects caught: a move of one step read as none; a threshold crossed in one
    run only read as no move.
    """
    comparison = _arrival_planted(dust, tmp_path, change, change)
    assert comparison.failures == [DECISION]
    item = next(i for i in comparison.items if i.name.startswith("arrival times"))
    if change == "later":
        assert (item.detail["moved one"], item.detail["one only"]) == (1, 0)
    else:
        assert (item.detail["moved one"], item.detail["one only"]) == (0, 1)


@pytest.mark.integration
def test_an_alert_the_reference_did_not_open_fails(dust: Path, tmp_path: Path) -> None:
    """An alert added to the other run's table: a decision that differs."""
    other = _copy(dust, tmp_path, "alert")
    with h5py.File(other, "r+") as handle:
        group = handle["transport/alerts"]
        times = handle["transport/sensors/times"][...]
        row = {
            "step_on": (np.int64, 5),
            "step_off": (np.int64, 7),
            "size_class": (np.int64, 0),
            "sensor": (np.int64, 0),
            "t_on": (np.float64, times[5]),
            "t_off": (np.float64, times[7]),
            "reading_on": (np.float64, 1.0e9),
            "peak": (np.float64, 1.0e9),
            "peak_t": (np.float64, times[5]),
        }
        for column, (dtype, value) in row.items():
            old = group[column][...]
            del group[column]
            group.create_dataset(column, data=np.append(old, value).astype(dtype))
    comparison = compare(dust, other)
    assert comparison.failures == [DECISION]
    item = next(i for i in comparison.items if i.name.startswith("alerts'"))
    assert item.detail["one only"] == 2


@pytest.mark.integration
def test_a_leftover_beyond_req_t05_fails_the_budget_verdict(
    dust: Path, tmp_path: Path
) -> None:
    """Class 0's leftover at the end set to a thousandth of its supply.

    Its verdict against REQ-T05 differs (and so do the closure and the
    leftover's change, which move with it).
    """
    other = _copy(dust, tmp_path, "leftover")
    with h5py.File(other, "r+") as handle:
        budgets = handle["transport/budgets"]
        values = budgets[...]
        supplied = values[-1, 0, 0] + values[-1, 0, 1] + values[-1, 0, 3]
        values[-1, 0, -1] = 1.0e-3 * supplied
        budgets[...] = values
    comparison = compare(dust, other)
    assert BUDGET in comparison.failures
    item = next(
        i for i in comparison.items if i.name.startswith("every class's budget")
    )
    assert item.detail["ref"][0] is True and item.detail["other"][0] is False


@pytest.mark.integration
def test_a_budget_that_does_not_close_fails_the_closure(
    dust: Path, tmp_path: Path
) -> None:
    """Class 0's content at the end moved by a billionth of its supply: no closure.

    Nothing else moves: the verdict's one failing kind is the closure.
    """
    other = _copy(dust, tmp_path, "closure")
    with h5py.File(other, "r+") as handle:
        budgets = handle["transport/budgets"]
        values = budgets[...]
        supplied = values[-1, 0, 0] + values[-1, 0, 1] + values[-1, 0, 3]
        values[-1, 0, -2] += 1.0e-9 * supplied
        budgets[...] = values
    comparison = compare(dust, other)
    assert comparison.failures == [CLOSED]


def _quiet_cell(path: Path) -> tuple[int, int, int, float, np.ndarray]:
    """A cell of the last channel that the run never reads above after the release.

    Returns its class (the channel's), its (j, i), the class's threshold and the
    stored frames' times.
    """
    units = clear_units({"run": path})
    with h5py.File(path, "r") as handle:
        thresholds = handle["arrival/thresholds"][()]
        times = handle["transport/times"][()]
    k = len(thresholds) - 1
    quiet = np.argwhere(units.clear[k] == units.ends[k])
    j, i = (int(v) for v in quiet[len(quiet) // 2])
    return k, j, i, float(thresholds[k]), times


@pytest.mark.integration
def test_a_clear_time_that_moves_onto_the_last_frame_is_a_move_of_one(
    dust: Path, tmp_path: Path
) -> None:
    """A cell's clear time from the second-to-last frame to the last fails the item.

    One cell is put at twice its threshold on the third-to-last frame in both
    files and on the second-to-last in one, so its clear time is the
    second-to-last frame's in the first and the last frame's in the second. The
    last stored frame is the run's last step, 0.02 of the spacing after the one
    before it, so the move is a fiftieth of the unit the old reading used (the
    largest gap) and rounded to none (test 96 B1, ``own_clear2.py``). A decision
    is equal only when it falls at the same frame (ADR-017 H, item 1).

    Defect caught: a clear time's move counted in the largest gap between
    frames.
    """
    k, j, i, threshold, times = _quiet_cell(dust)
    gap = float(times[-1] - times[-2])
    assert gap < 0.1 * float(np.diff(times).max()), "the premise: a short last gap"
    first = _copy(dust, tmp_path, "clear_a")
    second = _copy(dust, tmp_path, "clear_b")
    for path, frames in ((first, (-3,)), (second, (-3, -2))):
        with h5py.File(path, "r+") as handle:
            for frame in frames:
                handle[f"transport/C_{k}"][frame, j, i] = 2.0 * threshold
    comparison = compare(first, second)
    cells = next(x for x in comparison.items if x.name.startswith("cells' clear"))
    assert (cells.detail["moved one"], cells.detail["moved more"]) == (1, 0)
    assert cells.detail["one only"] == 0
    assert not cells.passed
    assert DECISION in comparison.failures


def _named_core(path: Path) -> None:
    """Rewrite the run file's stored configuration to name ``solver.backend: core``."""
    mapping = RunFile(path).read_config().mapping
    mapping["solver"]["backend"] = "core"
    with h5py.File(path, "r+") as handle:
        del handle["config/yaml"]
        handle["config"].create_dataset("yaml", data=config_text(mapping))


@pytest.mark.integration
def test_a_core_run_file_is_read_where_the_core_is_not_built(
    dust: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file whose configuration names the core is compared and analysed without it.

    The comparator, the arrival maps and the clear times each rebuild a
    ``SimConfig`` from the stored mapping. Where ``_cfdcore`` cannot be imported
    (Windows) they still read it; only building a core solver or march refuses
    (test 96 S4, review 96 S5). The module is hidden, so this holds where the
    core is built too.

    Defect caught: the module looked for when a configuration loads.
    """
    other = _copy(dust, tmp_path, "core_named")
    _named_core(other)
    monkeypatch.setitem(sys.modules, "_cfdcore", None)
    assert RunFile(other).read_config().mapping["solver"]["backend"] == "core"
    assert compare_runs.Run(other).config.backend == "core"
    assert arrival_from_run(other).maps.shape[0] == 1
    assert clear_units({"run": other}).clear.shape[0] == 5
    comparison = compare(dust, other)
    assert comparison.verdict == "PASS"
