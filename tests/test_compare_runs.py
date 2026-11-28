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

import json
import math
import os
import shutil
import sys
from collections.abc import Callable, Sequence
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
    POINTWISE,
    STOP,
    Comparison,
    Item,
    Moves,
    check_band,
    compare,
    grid_index,
    index_moves,
    moved,
    pointwise,
    relative,
    tier,
)

from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.io_manager import (  # noqa: E402 -- follows sys.path.insert
    ArrivalRecord,
    RunFile,
    config_text,
)
from src.mesh import Mesh  # noqa: E402 -- follows sys.path.insert
from src.monitor import sensor_cells  # noqa: E402 -- follows sys.path.insert
from src.placement import (  # noqa: E402 -- follows sys.path.insert
    ClearUnits,
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


@pytest.mark.unit
def test_check_band_accepts_the_band_and_a_spread_at_its_limit() -> None:
    """The comparator's own band passes, and so does a spread equal to its limit."""
    check_band()
    check_band({name: (limit, limit) for name, (_, limit) in BAND.items()})


@pytest.mark.unit
@pytest.mark.parametrize("name", list(BAND))
@pytest.mark.parametrize(
    "spread", [1.01, math.nan, -1.0], ids=["above", "nan", "negative"]
)
def test_check_band_refuses_a_spread_above_its_limit(name: str, spread: float) -> None:
    """A spread above its limit (or not a spread) on any field is refused by name.

    Defect caught: prompt 97's probe set the 320x120 pointwise concentration's
    first tier to 4.8e-6 and the leftover's to 7.6e-8, above their limits of
    1e-6 and 1e-8, so a field beyond its limit read as inside its spread.
    """
    band = dict(BAND)
    limit = band[name][1]
    band[name] = (spread * limit, limit)
    with pytest.raises(ValueError, match="spread exceeds its limit") as refused:
        check_band(band)
    message = str(refused.value)
    assert f"{name} (spread" in message
    assert message.count("(spread") == 1


@pytest.mark.unit
def test_check_band_refuses_a_limit_that_is_not_finite() -> None:
    """An infinite limit bounds nothing and is refused."""
    band = dict(BAND)
    band[POINTWISE] = (band[POINTWISE][0], math.inf)
    with pytest.raises(ValueError, match="spread exceeds its limit"):
        check_band(band)


@pytest.mark.unit
def test_compare_refuses_a_band_beyond_its_limit_before_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``compare`` checks the module's band before it opens either file.

    Defect caught: the refusal left out of ``compare``, so a caller that patches
    ``BAND`` with a grid's spreads, as prompt 97's probe did, is not refused.
    """
    monkeypatch.setitem(compare_runs.BAND, POINTWISE, (4.8e-6, 1.0e-6))
    absent = tmp_path / "absent.h5"
    with pytest.raises(ValueError, match="concentration, pointwise"):
        compare(absent, absent)


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


def _prebuilt(name: str) -> Path | None:
    """A run file built once for many runs of these tests, or None.

    ``CFD_COMPARE_RUNS`` names a folder that holds ``dust.h5`` and ``clean.h5``,
    made by this module's fixtures on the same commit (the planted-defect runner
    of ``docs/reports/probe96/variants96.py`` builds them once for its
    comparator plants, none of which changes how a run is made). The tests only
    copy from them.
    """
    folder = os.environ.get("CFD_COMPARE_RUNS")
    path = None if folder is None else Path(folder) / name
    return path if path is not None and path.exists() else None


def build_dust(path: Path) -> None:
    """A short dust release on 80x30 from rest, on the reference, written to PATH.

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
    run(SimConfig.from_dict(raw), path, provenance=PROVENANCE)


def build_clean(path: Path, dust: Path) -> None:
    """A clean-out on 80x30 to 30 s on the dust release's flow state, written to PATH.

    The flow state is the dust release's (``flows_from``: a source and an
    initial condition change no boundary), so only the march is run. Its
    sensors open and close alerts and read a 100:1 recovery before 30 s.
    """
    raw = yaml.safe_load(
        (CONFIGS / "scenario_clean_out.yaml").read_text(encoding="utf-8")
    )
    raw["scenario"]["base"] = str(CONFIGS / "clean_room_default.yaml")
    raw["domain"] = {"nx": 80, "ny": 30}
    raw["solver"]["mass_imbalance_tol"] = (
        conservation_keys()["conservation_limit"] * 1.2 * (0.1 * 0.1) / 60.0
    )
    raw["solver"]["t_end"] = 30.0
    run(SimConfig.from_dict(raw), path, flows_from=dust, provenance=PROVENANCE)


def build_runs(folder: Path) -> None:
    """Both fixtures' run files in FOLDER, for ``CFD_COMPARE_RUNS`` (``_prebuilt``)."""
    folder.mkdir(parents=True, exist_ok=True)
    build_dust(folder / "dust.h5")
    build_clean(folder / "clean.h5", folder / "dust.h5")


@pytest.fixture(scope="module")
def dust(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The short dust release (``build_dust``)."""
    if (built := _prebuilt("dust.h5")) is not None:
        return built
    path = tmp_path_factory.mktemp("dust") / "dust.h5"
    build_dust(path)
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
    # What "equal" covers: the cells the release reached, the same on both sides.
    assert arrival.detail["reached, ref"] == arrival.detail["reached, other"] > 0
    assert arrival.detail["reached, ref"] < arrival.detail["compared"]


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
        assert item.detail["reached, ref"] == item.detail["reached, other"]
    else:
        assert (item.detail["moved one"], item.detail["one only"]) == (0, 1)
        # A cell the reference reaches and the other never does.
        assert item.detail["reached, ref"] == item.detail["reached, other"] + 1


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


# ---- Every item pinned (test 96 B2, review 96 B2 and B3) ----------------------
#
# One planted run-file test per item the comparator reports, each of which fails
# under the defect that reads the item wrongly or not at all. They run on the
# dust release above and on a short clean-out on the same flow state, which has
# the alerts, the recovery times and the clean-out's clear times the dust release
# lacks.


@pytest.fixture(scope="module")
def clean(dust: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The clean-out on the dust release's flow state (``build_clean``)."""
    if (built := _prebuilt("clean.h5")) is not None:
        return built
    path = tmp_path_factory.mktemp("clean") / "clean.h5"
    build_clean(path, dust)
    return path


def _planted(source: Path, tmp_path: Path, name: str, change: Callable) -> Path:
    """A copy of SOURCE with CHANGE applied to its open file."""
    other = _copy(source, tmp_path, name)
    with h5py.File(other, "r+") as handle:
        change(handle)
    return other


def _item(comparison: Comparison, prefix: str) -> Item:
    """The one item whose name starts with PREFIX."""
    (item,) = [i for i in comparison.items if i.name.startswith(prefix)]
    return item


def _supplied(values: np.ndarray, frame: int, k: int = 0) -> float:
    """Class K's particles supplied by FRAME: initial, inflow and source."""
    return float(values[frame, k, 0] + values[frame, k, 1] + values[frame, k, 3])


def _scale_dataset(name: str, factor: float) -> Callable[[h5py.File], None]:
    def change(handle: h5py.File) -> None:
        data = handle[name]
        data[...] = data[...] * (1.0 + factor)

    return change


# -- The decisions --


@pytest.mark.integration
def test_an_arrival_moved_by_more_than_one_step_fails(
    dust: Path, tmp_path: Path
) -> None:
    """One cell's arrival three steps later: a move of more than one, counted so.

    Defect caught: a move of more than one step read as equal.
    """

    def change(handle: h5py.File) -> None:
        maps = handle["arrival/first_above"]
        values = maps[...]
        times = handle["transport/sensors/times"][...]
        index = tuple(np.argwhere(np.isfinite(values) & (values > times[1]))[0])
        at = int(np.searchsorted(times, values[index]))
        values[index] = times[at + 3]
        maps[...] = values

    comparison = compare(dust, _planted(dust, tmp_path, "three", change))
    item = _item(comparison, "arrival times")
    assert (item.detail["moved one"], item.detail["moved more"]) == (0, 1)
    assert item.detail["largest"] == pytest.approx(3.0)
    assert not item.passed and comparison.failures == [DECISION]


def _alert_row(handle: h5py.File) -> int:
    """The alert that opens and closes with room between, alone of its sensor and class."""
    group = handle["transport/alerts"]
    on, off = group["step_on"][()], group["step_off"][()]
    key = list(
        zip(group["sensor"][()].tolist(), group["size_class"][()].tolist(), strict=True)
    )
    for row in np.argsort(on):
        if key.count(key[row]) == 1 and off[row] - on[row] > 3:
            return int(row)
    raise AssertionError("no alert of the clean-out opens and closes on its own")


def _move_alert(handle: h5py.File, column: str) -> None:
    """The alert row's opening (``step_on``) or closing (``step_off``) a step later."""
    group = handle["transport/alerts"]
    times = handle["transport/sensors/times"][()]
    row = _alert_row(handle)
    steps, stamps = group[column][()], group["t_" + column[5:]][()]
    steps[row] += 1
    stamps[row] = times[steps[row]]
    group[column][...] = steps
    group["t_" + column[5:]][...] = stamps


@pytest.mark.integration
def test_an_alert_opened_a_step_later_moves_the_alert_and_its_sensors_latency(
    clean: Path, tmp_path: Path
) -> None:
    """One alert opened a step later: the alert item and the latency item each read one.

    Defects caught: an alert's opening step not compared, and the latencies
    compared with themselves.
    """
    comparison = compare(clean, _planted(clean, tmp_path, "open", _opening))
    for prefix in ("alerts'", "the file's sensors' latencies"):
        item = _item(comparison, prefix)
        assert (item.detail["moved one"], item.detail["moved more"]) == (1, 0), prefix
        assert item.detail["one only"] == 0 and not item.passed
    assert comparison.failures == [DECISION]


def _opening(handle: h5py.File) -> None:
    _move_alert(handle, "step_on")


def _closing(handle: h5py.File) -> None:
    _move_alert(handle, "step_off")


@pytest.mark.integration
def test_an_alert_closed_a_step_later_moves_the_alert_alone(
    clean: Path, tmp_path: Path
) -> None:
    """One alert closed a step later: the alert item reads one, the latencies none.

    A latency is read from the opening, so a closing moves no latency.

    Defect caught: an alert's closing step not compared.
    """
    comparison = compare(clean, _planted(clean, tmp_path, "close", _closing))
    alerts = _item(comparison, "alerts'")
    assert (alerts.detail["moved one"], alerts.detail["moved more"]) == (1, 0)
    assert not alerts.passed
    assert _item(comparison, "the file's sensors' latencies").passed
    assert comparison.failures == [DECISION]


@pytest.mark.integration
def test_a_recovery_time_a_step_later_fails(clean: Path, tmp_path: Path) -> None:
    """The first channel's 100:1 recovery time one step later: a move of one.

    Defect caught: the clean-out's recovery times compared with themselves.
    """

    def change(handle: h5py.File) -> None:
        times = handle["transport/sensors/times"][()]
        data = handle["recovery/recovery_time"]
        values = data[()]
        assert np.isfinite(values[0]), "the premise: the clean-out recovers by 30 s"
        values[0] += times[1] - times[0]
        data[...] = values

    comparison = compare(clean, _planted(clean, tmp_path, "recovery", change))
    item = _item(comparison, "recovery times")
    assert (item.detail["moved one"], item.detail["moved more"]) == (1, 0)
    assert not item.passed and comparison.failures == [DECISION]


@pytest.mark.integration
def test_a_hold_claimed_by_one_run_only_fails(clean: Path, tmp_path: Path) -> None:
    """The recovery record's hold set in one run: the run's end differs."""

    def change(handle: h5py.File) -> None:
        handle["recovery"].attrs["held"] = np.True_

    comparison = compare(clean, _planted(clean, tmp_path, "hold", change))
    item = _item(comparison, "the hold")
    assert (item.detail["ref"], item.detail["other"]) == (False, True)
    assert not item.passed and DECISION in comparison.failures


@pytest.mark.integration
@pytest.mark.parametrize(
    ("name", "dataset", "item_name", "bump"),
    [
        ("times", "transport/sensors/times", "the step times", 1.0e-9),
        ("steps", "transport/steps", "the run's last step", 1),
        ("final_time", "transport/times", "the run's last time", 1.0e-6),
    ],
)
def test_the_runs_end_and_its_step_times_are_compared(
    dust: Path, tmp_path: Path, name: str, dataset: str, item_name: str, bump: float
) -> None:
    """One step time, the last step or the last time moved: that item fails.

    A step time 1e-9 s later, the last frame's step one later, its time 1e-6 s
    later.

    Defects caught: each of the three always passing, or comparing the
    reference with itself.
    """

    def change(handle: h5py.File) -> None:
        data = handle[dataset]
        values = data[()]
        values[100 if name == "times" else -1] += bump
        data[...] = values

    comparison = compare(dust, _planted(dust, tmp_path, name, change))
    item = _item(comparison, item_name)
    assert not item.passed and DECISION in comparison.failures


def _cleared_from(frame: int) -> Callable[[h5py.File], None]:
    """Every class's frames from FRAME on set to zero: the room is clear from there."""

    def change(handle: h5py.File) -> None:
        for name in handle["transport"]:
            if name.startswith("C_"):
                handle["transport"][name][frame:] = 0.0

    return change


@pytest.mark.integration
def test_the_rooms_clear_time_moved_by_a_cell_above_its_threshold_fails(
    dust: Path, tmp_path: Path
) -> None:
    """A cleared room against the same room with one cell above on a late frame.

    Every frame from the sixth on is zero in the reference, so its room is
    clear from the first frame it reads. One cell is put at twice its
    threshold on the third-to-last frame of the other, so the room, and
    that cell, are clear a frame or more later.

    Defects caught: the room's clear times compared with themselves, and the
    cells' too.
    """
    k, j, i, threshold, _ = _quiet_cell(dust)
    reference = _planted(dust, tmp_path, "room_ref", _cleared_from(5))

    def change(handle: h5py.File) -> None:
        _cleared_from(5)(handle)
        handle[f"transport/C_{k}"][-3, j, i] = 2.0 * threshold

    comparison = compare(reference, _planted(dust, tmp_path, "room_late", change))
    for prefix in ("room clear times", "cells' clear times"):
        item = _item(comparison, prefix)
        assert item.detail["moved one"] + item.detail["moved more"] >= 1, prefix
        assert not item.passed and item.detail["one only"] == 0, prefix
    assert DECISION in comparison.failures


# -- The fields --


@pytest.mark.integration
@pytest.mark.parametrize(
    ("name", "dataset"),
    [
        ("u", "u_faces"),
        ("v", "v_faces"),
        ("p", "p"),
        ("k", "k"),
        ("eps", "eps"),
        ("nu_t", "nu_t"),
    ],
)
def test_each_of_the_six_flow_fields_is_read(
    dust: Path, tmp_path: Path, name: str, dataset: str
) -> None:
    """Each flow field scaled by 1 + 1e-6 fails its own item, and no other's.

    Defect caught: only some of the six fields compared (test 96 B2: of u, v, p,
    k, eps and nu_t only u).
    """

    def change(handle: h5py.File) -> None:
        group = _flow_group(handle)
        data = group[dataset]
        data[...] = data[...] * (1.0 + 1.0e-6)

    comparison = compare(dust, _planted(dust, tmp_path, name, change))
    failing = [
        i.name for i in comparison.items if i.criterion == "field" and not i.passed
    ]
    assert [n.split(",")[0] for n in failing] == [name]
    item = next(i for i in comparison.items if i.name.startswith(f"{name}, state"))
    assert item.detail["tier"] == BEYOND and comparison.failures == [FIELD]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("plant", "prefix", "factor"),
    [
        (_scaled_class_frames, "concentration, every", 1.0e-4),
        (_scaled_class_frames, "concentration, pointwise", 1.0e-4),
        (_scaled_sensor_channels, "sensor channels", 1.0e-4),
        (_moved_leftover, "leftover / supplied", 1.0e-6),
    ],
    ids=["frames", "pointwise", "sensors", "leftover"],
)
def test_each_class_field_reads_beyond_its_limit(
    dust: Path, tmp_path: Path, plant: Callable, prefix: str, factor: float
) -> None:
    """Each class field moved past its limit fails, whatever the stops say.

    Defect caught: the concentration, the pointwise measure, the sensor channels
    or the leftover's change compared with the reference itself.
    """
    other = _copy(dust, tmp_path, "beyond")
    with h5py.File(other, "r+") as handle:
        plant(handle, factor)
        group = _flow_group(handle)
        group.attrs["outer"] = int(group.attrs["outer"]) + 1
    item = _item(compare(dust, other), prefix)
    assert item.detail["tier"] == BEYOND and not item.passed
    assert "note" not in item.detail


def _lowest_read_cell(handle: h5py.File, floor: float) -> tuple[int, tuple, float]:
    """The class, the (frame, j, i) and the value of the smallest entry at or above FLOOR
    of its class's threshold."""
    thresholds = handle["arrival/thresholds"][()]
    best = None
    for k, threshold in enumerate(thresholds):
        frames = handle[f"transport/C_{k}"][()]
        reads = np.where(frames >= floor * threshold, frames, np.inf)
        at = np.unravel_index(np.argmin(reads), reads.shape)
        if np.isfinite(reads[at]) and (best is None or reads[at] / threshold < best[2]):
            best = (k, tuple(int(v) for v in at), float(reads[at] / threshold))
    assert best is not None
    return best


@pytest.mark.integration
def test_the_pointwise_measure_reads_cells_from_a_hundredth_of_the_threshold(
    dust: Path, tmp_path: Path
) -> None:
    """A cell just above a hundredth of its threshold, moved by 1e-4 of itself.

    The frame measure, over the class's largest value, stays inside its spread;
    the pointwise measure reads the cell and fails. A cell just below the
    hundredth moved the same is not read: both pass.

    Defects caught: the pointwise measure compared with itself, its floor at
    the threshold, and its floor at nothing.
    """
    with h5py.File(dust, "r") as handle:
        k, at, _ = _lowest_read_cell(handle, 0.01)
        threshold = float(handle["arrival/thresholds"][k])
        values = handle[f"transport/C_{k}"][()]
    below = np.where(values < 0.01 * threshold, values, 0.0)
    cell = np.unravel_index(np.argmax(below), below.shape)
    assert 0.0 < values[cell] < 0.01 * threshold, "the premise: a cell below the floor"

    def moved(position: tuple) -> Callable[[h5py.File], None]:
        def change(handle: h5py.File) -> None:
            data = handle[f"transport/C_{k}"]
            data[position] = data[position] * (1.0 + 1.0e-4)

        return change

    read = compare(dust, _planted(dust, tmp_path, "read", moved(at)))
    assert _item(read, "concentration, every").passed
    pointwise_item = _item(read, "concentration, pointwise")
    assert pointwise_item.detail["tier"] == BEYOND and not pointwise_item.passed
    skipped = compare(dust, _planted(dust, tmp_path, "skipped", moved(cell)))
    assert _item(skipped, "concentration, pointwise").passed
    assert skipped.verdict == "PASS"


# -- The budgets, their signs and the stops --


@pytest.mark.integration
def test_a_leftover_below_the_references_fails_the_verdict_and_the_change(
    dust: Path, tmp_path: Path
) -> None:
    """Class 0's leftover lowered by a thousandth of its supply, the content raised.

    The residual is unchanged, so the closure holds; the leftover's magnitude is
    beyond REQ-T05's limit (the verdict differs) and its change beyond its limit.
    The other sign of ``test_a_leftover_beyond_req_t05_fails_the_budget_verdict``.

    Defects caught: the absolute value of the leftover's change, and of the
    leftover in the verdict, removed or the sign reversed (review 96 B3).
    """
    other = _planted(
        dust, tmp_path, "lowered", lambda handle: _moved_leftover(handle, -1.0e-3)
    )
    comparison = compare(dust, other)
    verdict = _item(comparison, "every class's budget")
    assert verdict.detail["ref"][0] is True and verdict.detail["other"][0] is False
    change = _item(comparison, "leftover / supplied")
    assert change.detail["tier"] == BEYOND and not change.passed
    assert change.detail["difference"] == pytest.approx(1.0e-3, rel=1e-3)
    assert _item(comparison, "the budget's closure").passed


@pytest.mark.integration
def test_a_content_lowered_by_a_billionth_fails_the_closure(
    dust: Path, tmp_path: Path
) -> None:
    """Class 0's content at the end lowered by a billionth of its supply: no closure.

    The residual is positive; ``test_a_budget_that_does_not_close_fails_the_closure``
    raises it, which makes the residual negative.

    Defect caught: the closure's residual read signed, or its sign reversed
    (review 96 B3).
    """

    def change(handle: h5py.File) -> None:
        budgets = handle["transport/budgets"]
        values = budgets[...]
        values[-1, 0, -2] -= 1.0e-9 * _supplied(values, -1)
        budgets[...] = values

    comparison = compare(dust, _planted(dust, tmp_path, "lowered_content", change))
    assert comparison.failures == [CLOSED]
    closure = _item(comparison, "the budget's closure")
    assert closure.detail["other"] == pytest.approx(1.0e-9, rel=1e-3)


@pytest.mark.integration
def test_the_budgets_residual_and_its_leftover_are_each_read_against_req_t05(
    dust: Path, tmp_path: Path
) -> None:
    """A class's verdict reads both ratios, each by itself.

    The content lowered by a thousandth of the supply moves the residual alone;
    the leftover raised by as much with the content lowered moves the leftover
    alone. Each leaves the class's verdict False.

    Defects caught: a verdict that reads one of its two ratios (review 96 B3,
    test 96 B2).
    """

    def residual_only(handle: h5py.File) -> None:
        budgets = handle["transport/budgets"]
        values = budgets[...]
        values[-1, 0, -2] -= 1.0e-3 * _supplied(values, -1)
        budgets[...] = values

    for name, change in (
        ("residual", residual_only),
        ("leftover", lambda handle: _moved_leftover(handle, 1.0e-3)),
    ):
        comparison = compare(dust, _planted(dust, tmp_path, name, change))
        verdict = _item(comparison, "every class's budget")
        assert verdict.detail["ref"][0] is True, name
        assert verdict.detail["other"][0] is False, name
        assert not verdict.passed, name


@pytest.mark.integration
@pytest.mark.parametrize("side", ["other", "reference"])
def test_the_closure_is_read_on_every_stored_frame_of_both_runs(
    dust: Path, tmp_path: Path, side: str
) -> None:
    """A budget that does not close on the fourth frame, in either run, fails.

    Nothing else moves: the frame is not the last, and the verdict and the
    leftover read the end.

    Defects caught: the closure read on the last frame only, or on one run's
    frames only.
    """

    def change(handle: h5py.File) -> None:
        budgets = handle["transport/budgets"]
        values = budgets[...]
        values[3, 0, -2] += 1.0e-9 * _supplied(values, 3)
        budgets[...] = values

    planted = _planted(dust, tmp_path, "frame", change)
    comparison = compare(dust, planted) if side == "other" else compare(planted, dust)
    assert comparison.failures == [CLOSED]
    closure = _item(comparison, "the budget's closure")
    assert closure.detail["ref" if side == "reference" else "other"] > 1.0e-12


@pytest.mark.integration
def test_a_budget_that_is_not_a_number_does_not_close(
    dust: Path, tmp_path: Path
) -> None:
    """A NaN residual on a stored frame fails the closure.

    ``max`` skips a NaN, so the closure had read an open budget as closed
    (review 96 S1). The outflow is the term made NaN: a NaN initial or current
    content reads as None (no step yet), which the closure leaves out.
    """

    def change(handle: h5py.File) -> None:
        budgets = handle["transport/budgets"]
        values = budgets[...]
        values[3, 0, 2] = np.nan  # the outflow: a NaN content reads as None
        budgets[...] = values

    comparison = compare(dust, _planted(dust, tmp_path, "nan", change))
    assert comparison.failures == [CLOSED]
    assert _item(comparison, "the budget's closure").detail["other"] == math.inf


@pytest.mark.integration
def test_a_flow_state_held_by_one_run_only_fails_the_stop(
    dust: Path, tmp_path: Path
) -> None:
    """A run whose flow state is under another fingerprint: both runs' states fail.

    Defect caught: a flow state in one run only passing.
    """

    def change(handle: h5py.File) -> None:
        (key,) = list(handle["flow"])
        handle["flow"].move(key, "0" * 64)

    comparison = compare(dust, _planted(dust, tmp_path, "moved", change))
    stops = [i for i in comparison.items if i.criterion == "stop"]
    assert len(stops) == 2 and not any(i.passed for i in stops)
    assert sorted((i.detail["ref"], i.detail["other"]) for i in stops) == [
        (False, True),
        (True, False),
    ]
    assert STOP in comparison.failures and not comparison.bitwise["flow states"]


@pytest.mark.integration
def test_a_middle_tier_flow_field_leaves_the_run_out_of_its_first_tier(
    dust: Path, tmp_path: Path
) -> None:
    """u at 2e-9 with the outer count one apart passes, and is not inside its spread.

    The verdict is PASS and ``first_tier`` False, in the object and in its JSON.

    Defect caught: ``first_tier`` true with a field between its spread and its
    limit.
    """
    other = _copy(dust, tmp_path, "tiers")
    with h5py.File(other, "r+") as handle:
        group = _flow_group(handle)
        group["u_faces"][...] = group["u_faces"][...] * (1.0 + 2.0e-9)
        group.attrs["outer"] = int(group.attrs["outer"]) + 1
    comparison = compare(dust, other)
    assert comparison.verdict == "PASS"
    assert not comparison.first_tier
    assert comparison.as_dict()["first_tier"] is False
    assert compare(dust, _copy(dust, tmp_path, "same")).first_tier


# -- The command line --


@pytest.mark.integration
def test_the_command_line_exits_one_on_a_fail_and_zero_on_a_pass(
    dust: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` prints the verdict, writes the JSON and returns 0 on PASS, 1 on FAIL.

    Defect caught: an exit status of 0 whatever the verdict.
    """
    same = _copy(dust, tmp_path, "cli_same")
    out = tmp_path / "pass.json"
    assert compare_runs.main([str(dust), str(same), "--json", str(out)]) == 0
    assert "verdict: PASS" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "PASS"
    failing = _planted(
        dust,
        tmp_path,
        "cli_fail",
        lambda handle: _scale_dataset("transport/C_2", 1.0e-4)(handle),
    )
    out = tmp_path / "fail.json"
    assert compare_runs.main([str(dust), str(failing), "--json", str(out)]) == 1
    assert "verdict: FAIL" in capsys.readouterr().out
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["verdict"].startswith("FAIL") and written["first_tier"] is False


@pytest.mark.unit
def test_clear_times_of_runs_with_another_number_of_units_are_a_move_of_infinity() -> (
    None
):
    """Two runs whose clear units differ in number cannot be read against each other."""
    stub = ClearUnits(
        np.zeros((2, 3, 3)),
        np.zeros(2),
        (np.array([0.0, 1.0]),) * 2,
        (np.zeros(2),) * 2,
        np.ones(2),
        (("a", 0), ("a", 1)),
        "cumulative",
    )
    short = stub._replace(clear=stub.clear[:1], times=stub.times[:1])
    moves = compare_runs._frame_moves(stub, short, stub.clear, short.clear)
    assert moves.tolist() == [math.inf]
    same = compare_runs._frame_moves(stub, stub, stub.clear, stub.clear)
    assert same.shape == (18,) and not same.any()


# -- The placement of several scenarios (review 96 B2) --


def _file_sensors(path: Path) -> tuple[tuple[int, int], ...]:
    """The cells of the run file's four sensors."""
    config = compare_runs.Run(path).config
    return tuple(sensor_cells(config, Mesh(config)))


def _arrivals_at(
    cells: Sequence[tuple[int, int]], steps: int
) -> Callable[[h5py.File], None]:
    """The first event's arrival at each of CELLS, every channel, STEPS after the onset."""

    def change(handle: h5py.File) -> None:
        maps = handle["arrival/first_above"]
        values = maps[...]
        times = handle["transport/sensors/times"][()]
        base = int(np.searchsorted(times, float(handle["arrival/onsets"][0])))
        for j, i in cells:
            values[0, :, j, i] = times[base + steps]
        maps[...] = values

    return change


def _placement_items(ref: Path, other: Path) -> dict[str, Item]:
    """``compare_placement`` of one scenario's pair, its items by name."""
    items = compare_runs.compare_placement({"dust": (ref, other)})
    return {item.name: item for item in items}


@pytest.mark.integration
def test_the_placement_of_a_run_against_its_copy_is_equal_in_every_item(
    dust: Path, tmp_path: Path
) -> None:
    """The detection units, then per rule four layouts' cells, their order and latencies.

    Four layouts (the file's four and the three searched pools) under two rules,
    and the 20 unit latencies of each rule (four layouts, five units).
    """
    items = _placement_items(dust, _copy(dust, tmp_path, "same"))
    assert len(items) == 1 + 2 * (4 + 2)
    assert all(item.passed for item in items.values())
    latencies = items["worst_case: the layouts' unit latencies"]
    assert latencies.detail["compared"] == 20 and latencies.detail["largest"] == 0.0


@pytest.mark.integration
def test_a_layout_whose_cells_and_order_differ_fails_the_placement(
    dust: Path, tmp_path: Path
) -> None:
    """The file's four sensors reached a step after the onset in the other run only.

    The searched layouts then pick a sensor's cell in the other run (the cells
    differ), the file's four moves from last to first (the order differs) and a
    unit read at a cell reached in one run only moves by infinity. The file's
    own cells and the units are the same, and pass.

    Defects caught: the placement compared on one side's runs, a layout's cells
    not compared, the layouts' order not compared, and a latency reached in one
    run only read as no move (review 96 B2).
    """
    reference = _copy(dust, tmp_path, "placement_ref")
    other = _planted(
        dust, tmp_path, "placement_early", _arrivals_at(_file_sensors(dust), 1)
    )
    items = _placement_items(reference, other)
    assert items["the detection units"].passed
    for rule in ("worst_case", "mean"):
        assert items[f"{rule}: the file's four layout's cells"].passed
        moved_cells = [
            items[f"{rule}: the {pool} layout's cells"] for pool in ("product", "free")
        ]
        assert [not item.passed for item in moved_cells] == [True, True]
        for item in moved_cells:
            assert item.detail["ref"] != item.detail["other"]
        order = items[f"{rule}: the layouts' order"]
        assert not order.passed and order.detail["ref"] != order.detail["other"]
        assert order.detail["ref"][-1] == "file's four"
        assert order.detail["other"][0] == "file's four"
        latencies = items[f"{rule}: the layouts' unit latencies"]
        assert not latencies.passed and latencies.detail["one only"] > 0


@pytest.mark.integration
def test_a_layouts_latency_a_step_later_fails_the_placement_by_one(
    dust: Path, tmp_path: Path
) -> None:
    """One sensor cell reached five steps after the onset in one run, six in the other.

    The layouts are the same, so every cells item and the order pass; the
    layouts that read that cell (the file's four, and a pool that picks it, five
    units each) move by one step and by no more.

    Defects caught: the layouts' latencies not compared, and a latency's move
    counted in the wrong step.
    """
    cell = _file_sensors(dust)[:1]
    first = _planted(dust, tmp_path, "latency_5", _arrivals_at(cell, 5))
    second = _planted(dust, tmp_path, "latency_6", _arrivals_at(cell, 6))
    items = _placement_items(first, second)
    for name, item in items.items():
        if "unit latencies" in name:
            assert item.detail["moved one"] >= 5, "the file's four, five units"
            assert item.detail["moved more"] == 0
            assert item.detail["one only"] == 0 and not item.passed
        else:
            assert item.passed, name


@pytest.mark.integration
def test_a_placement_pair_on_other_step_times_moves_the_latencies(
    dust: Path, tmp_path: Path
) -> None:
    """The other run's step times halved: the same arrival times fall on other steps.

    Each run's latencies are read on its own steps, so times that are equal fall
    on steps far apart where the runs' steps differ.

    Defect caught: the other run's times placed on the reference's steps.
    """

    def change(handle: h5py.File) -> None:
        data = handle["transport/sensors/times"]
        data[...] = data[...] * 0.5

    items = _placement_items(dust, _planted(dust, tmp_path, "halved", change))
    for rule in ("worst_case", "mean"):
        latencies = items[f"{rule}: the layouts' unit latencies"]
        assert latencies.detail["moved more"] > 0 and not latencies.passed


SECOND_ONSET = 6.0


def _with_second_event(
    source: Path, tmp_path: Path, name: str, cell: tuple[int, int], steps: int
) -> Path:
    """A copy of SOURCE with a second contaminating event in longer steps.

    The dust release's file has one event, at 2 s, and one length of step. The
    copy's arrival record gains a second event at 6 s that reaches one cell,
    CELL, in every channel ``steps`` of its own steps later, and every step
    from 6 s on is made twice as long (the step times stretched about the
    second onset), as the flow of a state at half the velocity would have it.
    So the second event's steps are not the first's, as a scenario whose events
    lie in intervals of different flows has them.
    """
    other = _copy(source, tmp_path, name)
    with h5py.File(other, "r+") as handle:
        times = handle["transport/sensors/times"]
        values = times[()]
        start = int(np.searchsorted(values, SECOND_ONSET))
        values[start:] = values[start] + 2.0 * (values[start:] - values[start])
        times[...] = values
    file = RunFile(other)
    first = file.read_arrival()
    reached = np.full_like(first.first_above[:1], np.inf)
    reached[0, :, cell[0], cell[1]] = values[start + steps]
    file.write_arrival(
        ArrivalRecord(
            thresholds=first.thresholds,
            event_names=(*first.event_names, "second"),
            onsets=np.array([*first.onsets, SECOND_ONSET]),
            first_above=np.concatenate([first.first_above, reached]),
            t_final=first.t_final,
            counting=first.counting,
        )
    )
    return other


@pytest.mark.integration
def test_a_second_events_latency_is_read_from_its_own_onset_and_steps(
    dust: Path, tmp_path: Path
) -> None:
    """A unit of the second event, whose steps are twice the first's, moved by one step.

    One cell is reached three steps after the second onset in one run and four in
    the other, on steps twice as long as the first event's. Read from the second
    event's onset the move is one step; read from the first's (2 s, in the short
    steps) it is two. Every layout that holds the cell moves by one and by no more.

    Defect caught: every unit's latency added to the first event's onset (review
    96 B2), which reads a later event's moves in another interval's steps.
    """
    cell = _file_sensors(dust)[0]
    first = _with_second_event(dust, tmp_path, "second_3", cell, 3)
    second = _with_second_event(dust, tmp_path, "second_4", cell, 4)
    items = _placement_items(first, second)
    assert items["the detection units"].detail["ref"][-1][1] == "second"
    for rule in ("worst_case", "mean"):
        latencies = items[f"{rule}: the layouts' unit latencies"]
        assert latencies.detail["compared"] == 40
        assert latencies.detail["moved one"] >= 5
        assert latencies.detail["moved more"] == 0 and latencies.detail["one only"] == 0
        assert not latencies.passed


@pytest.mark.integration
def test_every_decision_reports_how_many_each_run_reached(clean: Path) -> None:
    """The clean-out's decisions say how many of their entries each run reached.

    A decision never reached in either run compares as no move, so the counts say
    what "every decision equal" covers (review 96 S7): 20 alerts, the finite
    arrival times, latencies, clear times and recovery times, the same on both
    sides of a copy.
    """
    comparison = compare(clean, clean)
    for prefix, minimum in (
        ("alerts'", 20),
        ("arrival times", 1),
        ("the file's sensors' latencies", 1),
        ("cells' clear times", 1),
        ("room clear times", 0),
        ("recovery times", 1),
    ):
        detail = _item(comparison, prefix).detail
        assert detail["reached, ref"] == detail["reached, other"] >= minimum, prefix
    assert _item(comparison, "alerts'").detail["reached, ref"] == 20
