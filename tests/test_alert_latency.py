"""VAL-009: the alert's step is the crossing step (ADR-015 C; REQ-A01, REQ-A06).

VAL-009 as written ("detection delay 0 or 1 timestep") holds by construction
when the monitor reads every step. What is tested, as ADR-015 C lists it,
each test naming in its docstring the planted defect it was shown to catch:

- On planted step series handed to ``update``: the alert's step is the step
  of the first reading at or above threshold, exactly; a reading equal to
  the threshold opens one and a reading one ulp below does not; the alert
  closes at the first step below; a second crossing opens a second alert.
- On a scenario run (80x30, a source in a sensor's cell): the run file's
  alert table is the table the stored sensor series implies, rebuilt here
  from the series alone, and the series has every step (the march called
  the monitor after every step); the stored readings are the frames' cell
  values at the frames' steps.
- REQ-A06: the fields the march hands the monitor are not writeable, and
  their bytes are unchanged across every ``update`` of the run.
- A restart across an open alert writes the uninterrupted run's alert table
  and readings, bitwise.

Criterion row: the alert step equals the crossing step, exact; the latency's
resolution is the step. Nothing here asserts a value recorded on another
machine: every comparison is between two computations of this process.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import yaml

from src import time_integration
from src.config import SimConfig
from src.io_manager import OPEN_STEP, AlertTable, RunFile
from src.mesh import Mesh
from src.monitor import Alert, AlertMonitor, sensor_cells
from src.time_integration import MarchStep, Provenance, run

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
# The coarse grid and its imbalance tolerance, ADR-011 G's formula at the
# base's t_end, as tests/test_time_integration.py runs the scenarios.
COARSE_GRID = {"nx": 80, "ny": 30}
COARSE_MASS_IMBALANCE_TOL = 1.0e-4 * 1.2 * (0.1 * 0.1) / 60.0
PROVENANCE = Provenance("test", "2026-10-31T00:00:00", "test-machine", 0)
# The dust release's box on the litho tool's top is one cell of 80x30, the
# cell this sensor reads. Released from 0.5 s for 1 s, the run to 7 s sees
# every class open in that cell and close after the release ends (the 5 um
# class last, near 6.5 s).
SOURCE_SENSOR = {"name": "in_source", "x": 3.85, "y": 2.05}
RELEASE = (0.5, 1.0)
T_END = 7.0
CHECKPOINT_INTERVAL = 7


class _KilledError(RuntimeError):
    """The run killed at a chosen step, as on_step raises it."""


def source_raw(**solver: object) -> dict:
    """The dust release on 80x30, rescheduled, with a sensor in the source's cell."""
    raw = yaml.safe_load(
        (CONFIGS / "scenario_equipment_dust.yaml").read_text(encoding="utf-8")
    )
    raw["scenario"]["base"] = str(CONFIGS / "clean_room_default.yaml")
    raw["domain"] = dict(COARSE_GRID)
    raw["solver"]["mass_imbalance_tol"] = COARSE_MASS_IMBALANCE_TOL
    raw["solver"]["t_end"] = T_END
    raw["solver"]["output_interval"] = 10
    raw["solver"].update(solver)
    (event,) = raw["scenario"]["events"]
    event["start"], event["duration"] = RELEASE
    base = yaml.safe_load(
        (CONFIGS / "clean_room_default.yaml").read_text(encoding="utf-8")
    )
    raw["sensors"] = [*base["sensors"], dict(SOURCE_SENSOR)]
    return raw


def expected_table(
    readings: np.ndarray,
    steps: np.ndarray,
    times: np.ndarray,
    thresholds: np.ndarray,
    names: tuple[str, ...],
) -> AlertTable:
    """The alert table a stored series implies, by the rule alone.

    For each class and sensor in turn: an alert opens at each step whose
    reading is at or above the threshold after a step below it (or at the
    first step), and closes at the next step below. The rows are ordered
    as the monitor orders them: closed alerts by (step_on, class, sensor's
    place), then open ones the same way.
    """
    rows = []
    for k in range(readings.shape[1]):
        for s in range(len(names)):
            series = readings[:, k, s]
            above = series >= thresholds[k]
            n = 0
            while n < series.size:
                if not above[n]:
                    n += 1
                    continue
                on = n
                while n < series.size and above[n]:
                    n += 1
                inside = series[on:n]
                peak = int(np.argmax(inside))
                off = n if n < series.size else None
                rows.append(
                    (
                        off is None,
                        int(steps[on]),
                        k,
                        s,
                        {
                            "step_on": steps[on],
                            "t_on": times[on],
                            "step_off": OPEN_STEP if off is None else steps[off],
                            "t_off": np.nan if off is None else times[off],
                            "size_class": k,
                            "sensor": s,
                            "reading_on": series[on],
                            "peak": inside[peak],
                            "peak_t": times[on + peak],
                        },
                    )
                )
    rows.sort(key=lambda r: r[:4])
    columns = {
        key: [r[4][key] for r in rows]
        for key in (
            "step_on",
            "step_off",
            "size_class",
            "sensor",
            "t_on",
            "t_off",
            "reading_on",
            "peak",
            "peak_t",
        )
    }
    return AlertTable(names=names, **columns)


def same_table(a: AlertTable, b: AlertTable) -> None:
    """Two alert tables, column by column, bitwise."""
    assert a.names == b.names
    for name in (
        "step_on",
        "step_off",
        "size_class",
        "sensor",
        "t_on",
        "t_off",
        "reading_on",
        "peak",
        "peak_t",
    ):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), name


class _Watched(AlertMonitor):
    """The monitor, checking at every update what REQ-A06 asks of the march."""

    seen: ClassVar[list[int]] = []
    unwritable: ClassVar[list[bool]] = []
    unchanged: ClassVar[list[bool]] = []

    def update(
        self, step: int, t: float, fields: Mapping[int, np.ndarray]
    ) -> tuple[Alert, ...]:
        """Record the flags and compare the fields' bytes across the update."""
        before = {k: fields[k].tobytes() for k in fields}
        self.unwritable.append(all(not fields[k].flags.writeable for k in fields))
        out = super().update(step, t, fields)
        self.unchanged.append(all(fields[k].tobytes() == before[k] for k in fields))
        self.seen.append(step)
        return out


@pytest.fixture(scope="module")
def released(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """The release solved from rest and marched, its monitor watched every step."""
    folder = tmp_path_factory.mktemp("released")
    config = SimConfig.from_dict(source_raw(checkpoint_interval=CHECKPOINT_INTERVAL))
    _Watched.seen, _Watched.unwritable, _Watched.unchanged = [], [], []
    steps: list[MarchStep] = []
    patch = pytest.MonkeyPatch()
    patch.setattr(time_integration, "AlertMonitor", _Watched)
    try:
        record = run(
            config, folder / "run.h5", on_step=steps.append, provenance=PROVENANCE
        )
    finally:
        patch.undo()
    return SimpleNamespace(
        config=config,
        record=record,
        run=RunFile(record.path),
        steps=steps,
        seen=list(_Watched.seen),
        unwritable=list(_Watched.unwritable),
        unchanged=list(_Watched.unchanged),
    )


def _series(
    monitor: AlertMonitor,
    mesh: Mesh,
    config: SimConfig,
    levels: list[float],
    k: int = 0,
    s: int = 0,
) -> list:
    """Hand LEVELS (multiples of class K's threshold) at sensor S, zero elsewhere."""
    cells = sensor_cells(config, mesh)
    j, i = cells[s]
    out = []
    for n, level in enumerate(levels):
        fields = {c: np.zeros(mesh.cell_type.shape) for c in range(5)}
        fields[k][j, i] = level
        out.append(monitor.update(n, 0.25 * n, fields))
    return out


@pytest.fixture(scope="module")
def coarse() -> tuple[SimConfig, Mesh]:
    """The product file on 80x30, for the planted series."""
    raw = yaml.safe_load(
        (CONFIGS / "clean_room_default.yaml").read_text(encoding="utf-8")
    )
    raw["domain"].update(COARSE_GRID)
    config = SimConfig.from_dict(raw)
    return config, Mesh(config)


@pytest.mark.validation
class TestPlantedSeries:
    def test_a_reading_equal_to_the_threshold_opens_and_one_below_does_not(
        self, coarse: tuple
    ) -> None:
        """VAL-009: "at or above" opens; one ulp below the threshold opens nothing.

        Defect caught: ">" for "at or above" (no alert at step 2).
        """
        config, mesh = coarse
        monitor = AlertMonitor(config, mesh)
        threshold = float(monitor.thresholds[0])
        below = float(np.nextafter(threshold, 0.0))
        returned = _series(monitor, mesh, config, [0.0, below, threshold, threshold])
        assert returned[1] == ()
        (alert,) = returned[2]
        assert (alert.step_on, alert.t_on, alert.reading_on) == (2, 0.5, threshold)
        assert monitor.alerts == (alert,)

    def test_the_alert_opens_at_the_first_reading_at_or_above_exactly(
        self, coarse: tuple
    ) -> None:
        """The crossing step itself, not the step after, for every class.

        Defects caught: the alert raised at the step after the crossing; the
        thresholds of one class applied to another.
        """
        config, mesh = coarse
        for k in range(5):
            monitor = AlertMonitor(config, mesh)
            threshold = float(monitor.thresholds[k])
            levels = [0.0, 0.5 * threshold, 0.99 * threshold, 1.5 * threshold]
            _series(monitor, mesh, config, levels, k=k, s=2)
            (alert,) = monitor.alerts
            assert (alert.size_class, alert.sensor, alert.step_on) == (
                k,
                "above_gap_2",
                3,
            )

    def test_the_alert_closes_at_the_first_step_below(self, coarse: tuple) -> None:
        """Open at 1, above to 3, below at 4: step_off 4 and t_off its time.

        Defect caught: the alert closed at the step after the first below
        (step_off 5).
        """
        config, mesh = coarse
        monitor = AlertMonitor(config, mesh)
        threshold = float(monitor.thresholds[0])
        levels = [0.0, 2.0, 3.0, 1.0, 0.5, 0.1, 0.0]
        returned = _series(monitor, mesh, config, [x * threshold for x in levels])
        (closed,) = returned[4]
        assert (closed.step_on, closed.step_off, closed.t_off) == (1, 4, 1.0)
        assert (closed.peak, closed.peak_t) == (3.0 * threshold, 0.5)
        assert all(r == () for n, r in enumerate(returned) if n not in (1, 4))

    def test_a_second_crossing_opens_a_second_alert(self, coarse: tuple) -> None:
        """Above, below, above again: two alerts, the second still open.

        Defects caught: a closed alert reopened in place of a new one; the
        second crossing ignored once a first alert exists.
        """
        config, mesh = coarse
        monitor = AlertMonitor(config, mesh)
        threshold = float(monitor.thresholds[0])
        levels = [1.0, 1.0, 0.5, 0.5, 2.0, 1.0]
        _series(monitor, mesh, config, [x * threshold for x in levels])
        first, second = monitor.alerts
        assert (first.step_on, first.step_off) == (0, 2)
        assert (second.step_on, second.step_off, second.t_off) == (4, None, None)
        assert second.is_open and second.peak == 2.0 * threshold

    def test_a_writeable_field_is_left_bitwise_as_it_was(self, coarse: tuple) -> None:
        """REQ-A06 on update alone: every field's bytes before and after are equal.

        The fields are handed writeable here, so a write would land. Defect
        caught: a field written in update (the reading's cell zeroed).
        """
        config, mesh = coarse
        monitor = AlertMonitor(config, mesh)
        rng = np.random.default_rng(73)
        fields = {k: rng.random(mesh.cell_type.shape) * 1e6 for k in range(5)}
        before = {k: fields[k].tobytes() for k in fields}
        monitor.update(0, 0.0, fields)
        assert all(fields[k].tobytes() == before[k] for k in fields)


@pytest.mark.validation
class TestScenarioRun:
    def test_every_class_alerts_in_the_source_cell(
        self, released: SimpleNamespace
    ) -> None:
        """The run this class reads has what it needs: each class opens and closes.

        Without this the comparisons below could pass on an empty table.
        """
        table = released.run.read_alerts()
        source = released.run.read_sensor_readings().names.index("in_source")
        for k in range(5):
            mine = (table.sensor == source) & (table.size_class == k)
            assert mine.any() and (table.step_off[mine] != OPEN_STEP).any(), k
        assert len(released.record.alerts) == len(table)

    def test_the_alert_table_is_the_one_the_sensor_series_implies(
        self, released: SimpleNamespace
    ) -> None:
        """VAL-009's criterion on a run: every alert's step is its crossing step, exact.

        The table is rebuilt from the stored readings alone by the rule and
        compared bitwise, every column. Defects caught: ">" for "at or
        above" would not show on a run (no reading lands on a threshold), so
        the planted series carry it; the alert closed a step late; the alert
        opened a step late; the peak not updated.
        """
        series = released.run.read_sensor_readings()
        thresholds = np.array(released.config.class_thresholds())
        expected = expected_table(
            series.readings, series.steps, series.times, thresholds, series.names
        )
        same_table(released.run.read_alerts(), expected)

    def test_the_series_has_every_step_and_the_monitor_saw_each(
        self, released: SimpleNamespace
    ) -> None:
        """Steps 0 to the last, no gap: the march called the monitor after every step.

        Defect caught: the monitor called every other step (the stored
        series then skips the steps it was not called on).
        """
        series = released.run.read_sensor_readings()
        last = released.record.steps
        assert series.steps.tolist() == list(range(last + 1))
        assert released.seen == list(range(last + 1))
        assert [m.step for m in released.steps] == list(range(1, last + 1))

    def test_the_stored_readings_are_the_frames_cell_values(
        self, released: SimpleNamespace
    ) -> None:
        """At each frame's step the readings are the field at each sensor's cell, bitwise."""
        frames = released.run.read_frames()
        series = released.run.read_sensor_readings()
        mesh = Mesh(released.config)
        cells = sensor_cells(released.config, mesh)
        rows = np.array([j for j, _ in cells])
        cols = np.array([i for _, i in cells])
        for n, step in enumerate(frames.steps):
            stored = series.readings[int(step)]
            at = np.array([frames.fields[k][n][rows, cols] for k in range(5)])
            assert stored.tobytes() == at.tobytes(), int(step)

    def test_on_step_and_the_record_report_the_alerts(
        self, released: SimpleNamespace
    ) -> None:
        """Each MarchStep carries the alerts its step opened or closed; the record all.

        Defect caught: on_step handed every alert so far rather than the
        step's; the record's alerts left empty.
        """
        table = released.run.read_alerts()
        opened, closed = [], []
        for info in released.steps:
            for alert in info.alerts:
                if alert.is_open:
                    assert alert.step_on == info.step
                    opened.append((alert.step_on, alert.size_class, alert.sensor))
                else:
                    assert alert.step_off == info.step
                    closed.append(alert)
        names = released.run.read_sensor_readings().names
        rows = [
            (int(on), int(k), names[s])
            for on, k, s in zip(
                table.step_on, table.size_class, table.sensor, strict=True
            )
        ]
        assert sorted(opened) == sorted(rows)
        place = {name: s for s, name in enumerate(names)}
        assert sorted(
            closed, key=lambda a: (a.step_on, a.size_class, place[a.sensor])
        ) == [a for a in released.record.alerts if not a.is_open]
        assert [
            (a.step_on, a.size_class, a.sensor) for a in released.record.alerts
        ] == (rows)

    def test_the_fields_are_read_only_and_unchanged_across_every_update(
        self, released: SimpleNamespace
    ) -> None:
        """REQ-A06 in the march: at every update the fields are not writeable, and
        their bytes are the same after it as before.

        Defects caught: the march handing the monitor writeable fields (step
        0's initial fields were writeable before step 2); a field written in
        update.
        """
        assert released.unwritable and all(released.unwritable)
        assert released.unchanged and all(released.unchanged)


def _killer(at: int) -> Callable[[MarchStep], None]:
    """An on_step that kills the run after step AT's writes."""

    def kill(info: MarchStep) -> None:
        if info.step == at:
            raise _KilledError

    return kill


@pytest.mark.validation
class TestRestart:
    def test_a_restart_across_an_open_alert_writes_the_uninterrupted_table(
        self, released: SimpleNamespace, tmp_path: Path
    ) -> None:
        """Killed after a checkpoint inside an open alert, restarted: the same table.

        The checkpoint chosen is the first multiple of the cadence with an
        alert of the uninterrupted run open across it, another opened after
        it, and a frame before the next checkpoint. The run is killed at that
        frame, so the killed file holds readings and an alert table past the
        checkpoint (the table the readings imply, written at the frame's
        flush), which the restart discards and replaces. Defects caught: the
        open alert dropped from the checkpoint (it opens again at the step
        after the checkpoint); the table written only at the end of the march
        (the killed file holds none).
        """
        table = released.run.read_alerts()
        last = released.record.steps
        off = np.where(table.step_off == OPEN_STEP, last + 1, table.step_off)
        frame = released.config.output_interval

        def kill_at(c: int) -> int | None:
            """The frame step after checkpoint C and before the next, if any."""
            f = (c // frame + 1) * frame
            return f if f < c + CHECKPOINT_INTERVAL else None

        checkpoint = next(
            c
            for c in range(CHECKPOINT_INTERVAL, last, CHECKPOINT_INTERVAL)
            if (f := kill_at(c)) is not None
            and np.any((table.step_on < c) & (off > f))
            and np.any((c < table.step_on) & (table.step_on <= f))
        )
        kill = kill_at(checkpoint)
        assert kill is not None
        config = released.config
        path = tmp_path / "run.h5"
        with pytest.raises(_KilledError):
            run(
                config,
                path,
                flows_from=released.record.path,
                on_step=_killer(kill),
                provenance=PROVENANCE,
            )
        killed = RunFile(path)
        assert killed.read_checkpoint().step == checkpoint
        series = killed.read_sensor_readings()
        assert series.steps[-1] == kill
        thresholds = np.array(config.class_thresholds())
        written = killed.read_alerts()
        same_table(
            written,
            expected_table(
                series.readings, series.steps, series.times, thresholds, series.names
            ),
        )
        assert np.any(written.step_on > checkpoint)
        record = run(config, restart=path)
        assert record.restarted_from == checkpoint
        same_table(RunFile(path).read_alerts(), table)
        assert record.alerts == released.record.alerts
        a, b = RunFile(path).read_sensor_readings(), released.run.read_sensor_readings()
        assert a.steps.tolist() == b.steps.tolist()
        assert a.readings.tobytes() == b.readings.tobytes()
