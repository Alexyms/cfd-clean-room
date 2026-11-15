"""ISO 14644-1 cumulative counting (ADR-015 C, amended 2026-11-14; prompt 81).

Every reader of a threshold in ``src/`` compares channel k, the sum of class
k and every larger class, with class k's threshold, as a real counter's
channel reads ISO 14644-1's cumulative limits. What is pinned here, each
test naming in its docstring the planted defect it was shown to catch
(``docs/reports/probe81/plant81.py`` plants each and records the outcome):

- The channels: ``channel_classes`` names the class and every larger one,
  largest first, on the product's sizes and on sizes given out of order;
  the largest class's channel is that class alone; ``channel_sums`` adds them
  from zero in that order, so a one-class channel is its field bitwise.
- The monitor: a channel's alert opens at a reading exactly at its limit
  and not one step below it, on a planted series where no class alone
  reaches its own threshold and the sum does; the channel readings are kept
  beside each class's own.
- The arrival maps and the recovery: the same planted series reaches the
  maps and puts the room out of spec, and only by the sum.
- The records: ``/arrival/``, the alert table and ``/recovery/`` record their
  counting (read back by value, with the older file's missing mark, in
  ``tests/test_io_manager.py``'s ``TestCountingMarks``); an older file (no
  mark) reads as per-class; no record has a default counting;
  ``placement`` refuses to score per-class maps beside cumulative ones
  (``detection_units``), to search detection and all-clear units of two
  countings (``search_roles``) and to rank layouts of two (``rank``); a
  checkpoint of the per-class code refuses the restart; the hold's start
  check reads the channels.

The run-level check (the march's maps never later than the per-class maps
of the same run's fields) is ``tests/test_alert_latency.py``'s
``TestCumulativeMaps``, on that file's release run. REQ-A07's clear times
from a run file, through ``clear_units``, on the sum and not the class alone:
``tests/test_placement.py``'s ``TestClearUnitsFromRuns``.
"""

import dataclasses
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from src.config import WORST_CASE, SimConfig, channel_classes
from src.io_manager import (
    COUNTING_ATTR,
    CUMULATIVE,
    PER_CLASS,
    AlertTable,
    ArrivalRecord,
    RecoveryRecord,
    RunFile,
)
from src.mesh import SOLID, Mesh
from src.monitor import (
    COUNTING_STATE,
    AlertMonitor,
    ArrivalTracker,
    channel_sums,
    checked_channels,
    sensor_cells,
)
from src.placement import (
    STEP_RESOLVED,
    ClearUnits,
    Layout,
    RunArrival,
    Units,
    arrival_from_frames,
    arrival_from_run,
    detection_units,
    exclude_units,
    given_layout,
    search,
)
from src.time_integration import RecoveryTracker, alert_table

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
COARSE_GRID = {"nx": 80, "ny": 30}
# The product's classes, smallest first, and their channels: each the class
# and every larger one, largest first.
SIZES = (1.0e-7, 3.0e-7, 5.0e-7, 1.0e-6, 5.0e-6)
CHANNELS = ((4, 3, 2, 1, 0), (4, 3, 2, 1), (4, 3, 2), (4, 3), (4,))
# A planted reading at channel 3 (>= 1 um, threshold 832): class 4 at 29,
# below its own 29.3, and class 3 at 803, below its own 832, sum to 832
# exactly in floating point (0 + 29 + 803); 802 makes 831.
BELOW_OWN_5UM = 29.0
AT_SUM_1UM = 803.0
BELOW_SUM_1UM = 802.0


def product() -> tuple[SimConfig, Mesh]:
    """The product file on 80x30 and its mesh."""
    raw = yaml.safe_load((CONFIGS / "clean_room_default.yaml").read_text("utf-8"))
    raw["domain"].update(COARSE_GRID)
    config = SimConfig.from_dict(raw)
    return config, Mesh(config)


def planted(mesh: Mesh, cell: tuple[int, int], one_um: float) -> dict[int, np.ndarray]:
    """Zero fields but CELL: class 4 at 29 and class 3 at one_um."""
    fields = {k: np.zeros(mesh.cell_type.shape) for k in range(5)}
    fields[4][cell] = BELOW_OWN_5UM
    fields[3][cell] = one_um
    return fields


@pytest.mark.unit
class TestChannels:
    def test_each_channel_sums_its_class_and_every_larger_one(self) -> None:
        """The product's sizes: the class and every larger one, largest first.

        Defects caught: the sum over the class and the smaller classes; the
        class alone.
        """
        assert channel_classes(SIZES) == CHANNELS
        config, _ = product()
        assert config.channel_classes() == CHANNELS

    def test_the_channels_follow_the_sizes_not_the_class_order(self) -> None:
        """Sizes given out of order: still every class at or above, largest first.

        Defect caught: the class index taken for the size order.
        """
        sizes = (5.0e-6, 1.0e-7, 1.0e-6)
        assert channel_classes(sizes) == ((0,), (0, 2, 1), (0, 2))

    def test_the_largest_class_is_its_own_channel_bitwise(self) -> None:
        """The 5 um channel is the 5 um field; every channel is its terms summed from zero.

        Defects caught: the sum over the smaller classes (the 5 um channel then
        holds them all); the class alone (channel 0 then misses the others).
        """
        rng = np.random.default_rng(81)
        fields = {k: rng.random((3, 4)) * 10.0**k for k in range(5)}
        sums = channel_sums(fields, channel_classes(SIZES))
        assert sums[4].tobytes() == fields[4].tobytes()
        for k, members in enumerate(CHANNELS):
            expected = np.zeros((3, 4))
            for j in members:
                expected += fields[j]
            assert sums[k].tobytes() == expected.tobytes()
        assert sums[0].tobytes() != fields[0].tobytes()

    @pytest.mark.parametrize(
        ("channels", "error", "match"),
        [
            (CHANNELS[:4], ValueError, r"4 channels for 5 classes"),
            (((4, 3, 2, 1), *CHANNELS[1:]), ValueError, r"channel 0 sums"),
            (((0, 0), *CHANNELS[1:]), ValueError, r"channel 0 sums"),
            ((*CHANNELS[:4], (4, 5)), ValueError, r"channel 4 sums"),
            ((*CHANNELS[:4], (True,)), TypeError, r"must be an int"),
        ],
    )
    def test_malformed_channels_are_refused(
        self, channels: tuple, error: type, match: str
    ) -> None:
        with pytest.raises(error, match=match):
            checked_channels(channels, 5)


@pytest.mark.unit
class TestMonitorOnTheSum:
    def test_an_alert_opens_on_the_sum_exactly_at_the_limit_and_not_below(
        self,
    ) -> None:
        """Channel 3 opens at 832 exactly, from classes each below their own limits.

        Neither class 3 (803 against 832) nor class 4 (29 against 29.3)
        reaches its own threshold; their sum does, at step 2 and not at step
        1 (831). Defects caught: the class alone (no alert); the sum over the
        smaller classes (803, no alert); ">" for "at or above" (no alert at
        step 2).
        """
        config, mesh = product()
        monitor = AlertMonitor(config, mesh)
        cell = sensor_cells(config, mesh)[1]
        zero = {k: np.zeros(mesh.cell_type.shape) for k in range(5)}
        assert monitor.update(0, 0.0, zero) == ()
        assert monitor.update(1, 0.5, planted(mesh, cell, BELOW_SUM_1UM)) == ()
        (alert,) = monitor.update(2, 1.0, planted(mesh, cell, AT_SUM_1UM))
        assert (alert.size_class, alert.sensor, alert.step_on) == (3, "above_gap_1", 2)
        assert alert.reading_on == 832.0 == float(monitor.thresholds[3])
        assert monitor.readings[3, 1] == AT_SUM_1UM
        assert monitor.readings[4, 1] == BELOW_OWN_5UM
        assert monitor.channel_readings[3, 1] == 832.0
        assert monitor.channel_readings[4, 1] == BELOW_OWN_5UM

    def test_the_alert_table_records_its_counting(self, tmp_path: Path) -> None:
        """The table the march builds says cumulative, and so does the file it is written to.

        The table takes its counting without a default and an unmarked group
        reads as per-class, so the file read back says cumulative only if the
        table was built with the mark and the writer wrote it. Defects caught:
        the counting left off the table (TypeError); the table marked
        per-class; ``write_alerts`` dropping the mark (the file reads
        per-class).
        """
        config, mesh = product()
        monitor = AlertMonitor(config, mesh)
        table = alert_table(monitor)
        assert table.counting == CUMULATIVE
        run = RunFile(tmp_path / "run.h5")
        run.write_alerts(table)
        assert run.read_alerts().counting == CUMULATIVE

    def test_a_restored_monitor_sums_its_channels_again(self) -> None:
        """restore() rebuilds the channel readings from the carried per-class ones."""
        config, mesh = product()
        monitor = AlertMonitor(config, mesh)
        cell = sensor_cells(config, mesh)[1]
        monitor.update(0, 0.0, planted(mesh, cell, AT_SUM_1UM))
        again = AlertMonitor(config, mesh)
        again.restore(monitor.state())
        assert again.channel_readings.tobytes() == monitor.channel_readings.tobytes()


@pytest.mark.unit
class TestMapsAndRecoveryOnTheSum:
    def test_the_arrival_map_reaches_a_cell_by_the_sum_only(self) -> None:
        """Channel 3's map reads the step at 832, not the one at 831; class 4's never.

        Defects caught: the class alone; the sum over the smaller classes.
        """
        config, mesh = product()
        live = mesh.cell_type != SOLID
        cell = sensor_cells(config, mesh)[1]
        tracker = ArrivalTracker(
            config.class_thresholds(), live, [0.0], channels=config.channel_classes()
        )
        tracker.update(0.0, {k: np.zeros(live.shape) for k in range(5)})
        tracker.update(0.5, planted(mesh, cell, BELOW_SUM_1UM))
        tracker.update(1.0, planted(mesh, cell, AT_SUM_1UM))
        first = tracker.first_above
        assert first[0, 3][cell] == 1.0
        assert np.isnan(first[0, 4][cell])
        assert np.isnan(first[0, 2][cell])
        record = tracker.record(["e"], 1.0)
        assert record.counting == CUMULATIVE
        assert tracker.state()[COUNTING_STATE] == 1.0

    def test_the_room_is_out_of_spec_on_the_sum(self) -> None:
        """Within spec is every channel below: the planted sum keeps the room out.

        The local recovery field reads the same channel. Defects caught: the
        class alone; the sum over the smaller classes.
        """
        config, mesh = product()
        live = mesh.cell_type != SOLID
        cells = sensor_cells(config, mesh)
        thresholds = config.class_thresholds()
        tracker = RecoveryTracker(
            thresholds,
            [100.0 * t for t in thresholds],
            cells,
            live,
            None,
            channels=config.channel_classes(),
        )
        tracker.update(0.0, planted(mesh, cells[1], BELOW_SUM_1UM), [0.0] * 5)
        assert tracker.room_since == 0.0
        tracker.update(1.0, planted(mesh, cells[1], AT_SUM_1UM), [0.0] * 5)
        assert np.isnan(tracker.room_since)
        assert np.isnan(tracker.class_since[3])
        assert tracker.class_since[4] == 0.0
        assert tracker.last_above[3][cells[1]] == 1.0
        assert np.isnan(tracker.last_above[4][cells[1]])

    def test_the_recovery_record_says_cumulative_in_the_file(
        self, tmp_path: Path
    ) -> None:
        """The record's counting is cumulative, and reads back so from /recovery/.

        Defects caught: the record marked per-class; ``write_recovery``
        dropping the mark (the file reads per-class).
        """
        config, mesh = product()
        live = mesh.cell_type != SOLID
        cells = sensor_cells(config, mesh)
        thresholds = config.class_thresholds()
        tracker = RecoveryTracker(
            thresholds,
            [100.0 * t for t in thresholds],
            cells,
            live,
            None,
            channels=config.channel_classes(),
        )
        tracker.update(0.0, planted(mesh, cells[1], BELOW_SUM_1UM), [0.0] * 5)
        record = tracker.record([f"s{i}" for i in range(len(cells))], 0.18, False)
        assert record.counting == CUMULATIVE
        run = RunFile(tmp_path / "run.h5")
        run.write_recovery(record)
        assert run.read_recovery().counting == CUMULATIVE


def _arrival_of(counting: str) -> RunArrival:
    """A one-event run's maps of COUNTING, planted zeros on a 2x2 grid."""
    return RunArrival(
        np.zeros((1, 5, 2, 2)),
        np.zeros(1),
        STEP_RESOLVED,
        ("e",),
        3.0,
        counting=counting,
    )


def _record(counting: str) -> ArrivalRecord:
    """A one-event, one-cell record of COUNTING."""
    return ArrivalRecord(
        thresholds=np.ones(5),
        event_names=("e",),
        onsets=np.zeros(1),
        first_above=np.full((1, 5, 1, 1), 1.0),
        t_final=2.0,
        counting=counting,
    )


@pytest.mark.unit
class TestRecords:
    def test_an_older_arrival_group_reads_as_per_class(self, tmp_path: Path) -> None:
        """/arrival/ without the mark (a file before prompt 81) reads PER_CLASS.

        Defects caught: the missing mark read as cumulative; the mark not
        written (a new file would then read per-class).
        """
        run = RunFile(tmp_path / "run.h5")
        run.write_arrival(_record(CUMULATIVE))
        assert run.read_arrival().counting == CUMULATIVE
        with h5py.File(run.path, "a") as handle:
            del handle["arrival"].attrs[COUNTING_ATTR]
        assert run.read_arrival().counting == PER_CLASS

    def test_an_older_run_file_reads_with_its_maps_flagged_per_class(
        self, tmp_path: Path
    ) -> None:
        """arrival_from_run hands an unmarked /arrival/ back flagged PER_CLASS.

        Defect caught: the counting dropped by the reader (the maps then
        pass as cumulative).
        """
        run = RunFile(tmp_path / "run.h5")
        run.write_arrival(_record(CUMULATIVE))
        with h5py.File(run.path, "a") as handle:
            del handle["arrival"].attrs[COUNTING_ATTR]
        arrival = arrival_from_run(run)
        assert (arrival.resolution, arrival.counting) == (STEP_RESOLVED, PER_CLASS)

    def test_placement_refuses_per_class_maps_beside_cumulative_ones(self) -> None:
        """detection_units refuses a mix and keeps the one counting it is given.

        Defect caught: the check removed (the units then stack unlike maps).
        """

        with pytest.raises(ValueError, match=r"a layout is scored over maps of one"):
            detection_units({"a": _arrival_of(CUMULATIVE), "b": _arrival_of(PER_CLASS)})
        assert detection_units({"a": _arrival_of(PER_CLASS)}).counting == PER_CLASS
        assert detection_units({"a": _arrival_of(CUMULATIVE)}).counting == CUMULATIVE

    def test_the_units_keep_their_counting_through_every_step(self) -> None:
        """exclude_units, search and given_layout hand back the counting they were given.

        Per-class units are the case that tells: every step leaves them
        per-class. Defects caught: ``exclude_units`` marking its units
        cumulative whatever they were; ``search`` or ``given_layout`` marking
        the layout cumulative.
        """
        units = detection_units({"a": _arrival_of(PER_CLASS)})
        assert exclude_units(units, [units.labels[0]]).counting == PER_CLASS
        cells = [(0, 0), (0, 1)]
        assert given_layout(units, cells[:1]).counting == PER_CLASS
        assert search(units, cells, 1, WORST_CASE).counting == PER_CLASS

    @pytest.mark.parametrize(
        "record",
        [
            RunArrival,
            Units,
            ClearUnits,
            Layout,
            ArrivalRecord,
            AlertTable,
            RecoveryRecord,
        ],
    )
    def test_no_record_has_a_default_counting(self, record: type) -> None:
        """A caller says which counting it holds; leaving it out is a TypeError.

        A default would let per-class maps (an older file's, a probe's) pass
        as cumulative, past ``detection_units``' refusal of a mix. Defect
        caught: a default restored on any of the seven.
        """
        if dataclasses.is_dataclass(record):
            (field,) = [f for f in dataclasses.fields(record) if f.name == "counting"]
            assert field.default is dataclasses.MISSING
            assert field.default_factory is dataclasses.MISSING
        else:
            assert "counting" in record._fields
            assert "counting" not in record._field_defaults

    def test_the_counting_is_one_of_two(self) -> None:
        with pytest.raises(ValueError, match=r"counting must be one of"):
            _record("cumulativ")
        with pytest.raises(TypeError, match=r"counting must be a str"):
            _record(None)  # type: ignore[arg-type]

    def test_the_sensor_readings_hold_the_channels_beside_the_classes(
        self, tmp_path: Path
    ) -> None:
        """Both series are stored; a write that gives one and not the other is refused."""
        run = RunFile(tmp_path / "run.h5")
        readings = np.arange(10.0).reshape(1, 5, 2)
        channels = readings + 1.0
        run.write_sensor_readings([0], [0.0], readings, ["a", "b"], channels=channels)
        held = run.read_sensor_readings()
        assert held.readings.tobytes() == readings.tobytes()
        assert held.channels.tobytes() == channels.tobytes()
        with pytest.raises(ValueError, match=r"hold channel readings and this write"):
            run.write_sensor_readings([1], [0.5], readings, ["a", "b"])
        other = RunFile(tmp_path / "other.h5")
        other.write_sensor_readings([0], [0.0], readings, ["a", "b"])
        assert other.read_sensor_readings().channels is None
        with pytest.raises(ValueError, match=r"hold no channel readings"):
            other.write_sensor_readings(
                [1], [0.5], readings, ["a", "b"], channels=channels
            )

    def test_channels_of_another_shape_than_the_readings_are_refused(
        self, tmp_path: Path
    ) -> None:
        """One channel per class: a series of 4 channels beside 5 classes is refused.

        Defect caught: the shape check removed from ``write_sensor_readings``
        (the series is stored, and the file then holds readings and channels
        that do not line up).
        """
        run = RunFile(tmp_path / "run.h5")
        readings = np.arange(10.0).reshape(1, 5, 2)
        with pytest.raises(ValueError, match=r"one channel per class"):
            run.write_sensor_readings(
                [0], [0.0], readings, ["a", "b"], channels=readings[:, :4]
            )
        assert not run.path.exists()

    def test_a_per_class_checkpoint_refuses_the_tracker(self) -> None:
        """ArrivalTracker.restore refuses maps carried without the channels' mark."""
        config, mesh = product()
        live = mesh.cell_type != SOLID
        tracker = ArrivalTracker(
            config.class_thresholds(), live, [0.0], channels=config.channel_classes()
        )
        state = tracker.state()
        del state[COUNTING_STATE]
        with pytest.raises(ValueError, match=r"made per class, before the channels"):
            tracker.restore(state)


@pytest.mark.unit
class TestFramesOnTheSum:
    def test_the_frames_map_and_the_clear_times_read_the_channels(self) -> None:
        """arrival_from_frames reaches by the sum; a per-class plant would not.

        Defects caught: the class alone; the sum over the smaller classes.
        """
        from src.io_manager import TransportFrames
        from src.placement import clear_times

        config, mesh = product()
        live = mesh.cell_type != SOLID
        cell = sensor_cells(config, mesh)[1]
        frames_fields = [
            planted(mesh, cell, BELOW_SUM_1UM),
            planted(mesh, cell, AT_SUM_1UM),
            planted(mesh, cell, BELOW_SUM_1UM),
        ]
        frames = TransportFrames(
            np.array([0, 1, 2]),
            np.array([0.0, 1.0, 2.0]),
            {k: np.stack([f[k] for f in frames_fields]) for k in range(5)},
            ((),) * 3,
        )
        thresholds = np.array(config.class_thresholds())
        maps = arrival_from_frames(
            frames, thresholds, live, np.zeros(1), channels=config.channel_classes()
        )
        assert maps[0, 3][cell] == 1.0
        assert np.isinf(maps[0, 4][cell])
        areas = np.outer(mesh.dy_cell, mesh.dx_cell)
        clear, _, _ = clear_times(
            frames, thresholds, live, areas, 0.0, channels=config.channel_classes()
        )
        assert clear[3][cell] == 2.0
        assert clear[4][cell] == 0.0


@pytest.mark.unit
def test_the_hold_reads_the_start_on_the_channels(tmp_path: Path) -> None:
    """A start with every class below its own threshold but a channel at it is accepted.

    The start is the planted series' 803 and 29 per cubic metre in classes 3
    and 4, zero elsewhere: channel 3 is at its limit, so the room is not
    within spec at t = 0 and the hold may stand. With 802 every channel is
    below and the hold is refused. Defect caught: the class alone (both
    refused).
    """
    raw = yaml.safe_load((CONFIGS / "scenario_clean_out.yaml").read_text("utf-8"))
    raw["scenario"]["base"] = str(CONFIGS / "clean_room_default.yaml")
    raw["scenario"]["hold_time"] = 5.0
    event = raw["scenario"]["events"][0]
    event["initial"] = {"concentration": [0.0, 0.0, 0.0, AT_SUM_1UM, BELOW_OWN_5UM]}
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert SimConfig(path).scenario.hold_time == 5.0
    event["initial"] = {"concentration": [0.0, 0.0, 0.0, BELOW_SUM_1UM, BELOW_OWN_5UM]}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=r"starts every channel"):
        SimConfig(path)
