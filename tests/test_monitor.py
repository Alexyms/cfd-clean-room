"""The alert monitor, the arrival maps and the thresholds (src/monitor.py, src/config.py).

ADR-015 H's steps 2 (section C) and 3 (section D), unit tier. What is pinned, each test naming in its
docstring the planted defect it was shown to catch:

- The construction: no sensor, a class without a threshold and a sensor in a
  SOLID cell are refused; the thresholds are the size mapping's, in class
  order; each sensor reads the cell of ``sensor_cells``, which moved here
  from ``time_integration`` and is re-exported there unchanged.
- ``update``'s refusals (the step and time order, the classes, the shapes,
  a non-finite reading) and its return: the alerts the step opened or
  closed, the peak carried while open, the closed alerts before the open.
- ``latencies`` on planted alerts with two onsets, by section D's rule (an
  alert open at an onset is its detection; no cut at the next onset), and
  its refusals.
- ``state`` and ``restore``: a restored monitor continues as the one it was
  taken from, and a malformed state is refused.
- REQ-A03's second form, ``thresholds: {iso_class: N}``: the product file's
  five Class 5 values reproduced bitwise, the rounding to three figures,
  and the rejection paths.
- Step 3: ``contaminating_events`` on the reference scenarios and on the
  set event section D's list left out; ``ArrivalTracker`` on planted
  series (a cell at or above at the onset reads the onset, a second
  event's map starts at its onset, at or above, the first time kept),
  ``state`` and ``restore`` bitwise and their refusals, the record's inf
  and NaN, and the map against the monitor at the sensors to the step.

VAL-009's own tests (the alert step against the crossing step, the march's
every step, the fields' bytes, the restart) are tests/test_alert_latency.py.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src import time_integration
from src.config import ISO_CLASS_RANGE, SimConfig
from src.mesh import SOLID, Mesh
from src.monitor import (
    ALERT_COLUMNS,
    ARRIVAL_STATE_NAMES,
    RUN_START,
    SENSOR_FACE_RTOL,
    STATE_NAMES,
    Alert,
    AlertMonitor,
    ArrivalTracker,
    carries_contamination,
    contaminating_events,
    sensor_cells,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
# The product file's Class 5 limits by size, as configs/clean_room_default.yaml
# states them (ISO 14644-1:2015 table 1, 5 um by the formula).
CLASS_5 = (1.0e5, 1.02e4, 3.52e3, 8.32e2, 2.93e1)
COARSE_GRID = {"nx": 80, "ny": 30}


def room_raw() -> dict:
    """The product file on the coarse grid, as a mapping."""
    raw = yaml.safe_load((CONFIGS / "clean_room_default.yaml").read_text("utf-8"))
    raw["domain"].update(COARSE_GRID)
    return raw


def room(raw: dict | None = None) -> tuple[SimConfig, Mesh]:
    """A configuration and its mesh, the product file's by default."""
    config = SimConfig.from_dict(room_raw() if raw is None else raw)
    return config, Mesh(config)


def fields_of(mesh: Mesh, values: np.ndarray, cells: tuple) -> dict[int, np.ndarray]:
    """Five zero fields with VALUES [classes, sensors] at the sensors' CELLS."""
    out = {}
    for k in range(values.shape[0]):
        field = np.zeros(mesh.cell_type.shape)
        for s, (j, i) in enumerate(cells):
            field[j, i] = values[k, s]
        out[k] = field
    return out


@pytest.fixture(scope="module")
def product() -> tuple[SimConfig, Mesh]:
    """The product file on 80x30."""
    return room()


def feed(
    monitor: AlertMonitor, room: tuple, series: list[np.ndarray], dt: float = 0.1
) -> list[tuple[Alert, ...]]:
    """Hand SERIES (one [classes, sensors] array a step) to update from step 0."""
    config, mesh = room
    cells = sensor_cells(config, mesh)
    return [
        monitor.update(n, n * dt, fields_of(mesh, values, cells))
        for n, values in enumerate(series)
    ]


@pytest.mark.unit
class TestConstruction:
    def test_no_sensor_is_refused(self) -> None:
        """An alert is a sensor's reading; a monitor with none is refused."""
        raw = room_raw()
        raw["sensors"] = []
        config, mesh = room(raw)
        with pytest.raises(ValueError, match="at least one sensor"):
            AlertMonitor(config, mesh)

    def test_a_class_without_a_threshold_is_refused(self) -> None:
        """Every class needs its threshold; the message names the monitor."""
        raw = room_raw()
        del raw["thresholds"]["0.3e-6"]
        config, mesh = room(raw)
        with pytest.raises(ValueError, match=r"the alert monitor: the class of diam"):
            AlertMonitor(config, mesh)

    def test_a_sensor_in_a_solid_cell_is_refused(self, product: tuple) -> None:
        """A sensor inside an obstacle reads nothing: refused, naming it."""
        _, mesh = product
        j, i = np.argwhere(mesh.cell_type == SOLID)[0]
        raw = room_raw()
        raw["sensors"] = [
            {"name": "inside", "x": float(mesh.xc[i]), "y": float(mesh.yc[j])}
        ]
        bad, bad_mesh = room(raw)
        with pytest.raises(ValueError, match=r"sensor inside reads a SOLID cell"):
            AlertMonitor(bad, bad_mesh)

    def test_the_thresholds_are_the_size_mapping_in_class_order(
        self, product: tuple
    ) -> None:
        """thresholds is class_thresholds, [classes], and a copy.

        Defect caught: the mapping read in file order rather than by the
        class's diameter (the file's keys reordered here).
        """
        raw = room_raw()
        raw["thresholds"] = dict(reversed(list(raw["thresholds"].items())))
        config, mesh = room(raw)
        monitor = AlertMonitor(config, mesh)
        assert monitor.thresholds.tolist() == list(CLASS_5)
        monitor.thresholds[0] = 0.0
        assert monitor.thresholds[0] == CLASS_5[0]

    def test_each_sensor_reads_the_cell_of_sensor_cells(self, product: tuple) -> None:
        """A distinct value in every sensor's cell comes back as its reading.

        Defects caught: rows and columns swapped; the readings of one class
        given for every class; the sensors read out of the file's order.
        """
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        cells = sensor_cells(config, mesh)
        values = np.arange(1.0, 21.0).reshape(5, 4)
        monitor.update(0, 0.0, fields_of(mesh, values, cells))
        assert monitor.readings.tobytes() == values.tobytes()
        assert monitor.names == tuple(s.name for s in config.sensors)

    def test_sensor_cells_moved_and_re_exported_unchanged(self) -> None:
        """time_integration's names are the monitor's objects, so nothing else changes."""
        assert time_integration.sensor_cells is sensor_cells
        assert time_integration.SENSOR_FACE_RTOL is SENSOR_FACE_RTOL

    def test_the_readings_before_any_update_are_nan(self, product: tuple) -> None:
        """No reading has been taken: NaN, [classes, sensors], and no alerts."""
        monitor = AlertMonitor(*product)
        assert monitor.readings.shape == (5, 4)
        assert np.isnan(monitor.readings).all()
        assert monitor.alerts == ()


@pytest.mark.unit
class TestUpdate:
    def test_the_return_is_the_alerts_opened_and_closed_at_the_step(
        self, product: tuple
    ) -> None:
        """Opening steps return the open alert, closing steps the closed one.

        Defects caught: update returning every alert so far; a closed alert
        returned without its step_off; the peak not carried while open.
        """
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        threshold = monitor.thresholds[4]
        series = [np.zeros((5, 4)) for _ in range(5)]
        for n, level in ((1, 2.0), (2, 5.0), (3, 3.0)):
            series[n][4, 1] = level * threshold
        returned = feed(monitor, product, series)
        assert returned[0] == () and returned[2] == () and returned[3] == ()
        (opened,) = returned[1]
        assert (opened.sensor, opened.size_class, opened.step_on) == (
            "above_gap_1",
            4,
            1,
        )
        assert opened.is_open and opened.t_off is None
        (closed,) = returned[4]
        assert (closed.step_on, closed.step_off, closed.t_off) == (1, 4, 0.4)
        assert (closed.peak, closed.peak_t) == (5.0 * threshold, 0.2)
        assert closed.reading_on == 2.0 * threshold and not closed.is_open
        assert monitor.alerts == (closed,)

    def test_an_open_alert_carries_its_peak_to_the_last_update(
        self, product: tuple
    ) -> None:
        """The alerts property rebuilds an open alert with the peak so far.

        Defect caught: the peak taken at the last reading rather than the
        largest; a later equal reading moving peak_t.
        """
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        threshold = monitor.thresholds[0]
        levels = [1.0, 3.0, 2.0, 3.0]
        series = []
        for level in levels:
            values = np.zeros((5, 4))
            values[0, 0] = level * threshold
            series.append(values)
        feed(monitor, product, series)
        (alert,) = monitor.alerts
        assert alert.is_open and (alert.peak, alert.peak_t) == (3 * threshold, 0.1)

    def test_closed_alerts_come_before_open_ones_each_by_opening_step(
        self, product: tuple
    ) -> None:
        """The order is the contract's: closed first, then open; each by step_on.

        Defect caught: the alerts listed in the order they were opened, an
        open alert before a closed one.
        """
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        above = 2.0 * monitor.thresholds[:, None] * np.ones((5, 4))
        series = [np.zeros((5, 4)) for _ in range(4)]
        series[0][1, 0] = above[1, 0]  # opens at 0, stays open
        series[1][1, 0] = above[1, 0]
        series[2][1, 0] = above[1, 0]
        series[3][1, 0] = above[1, 0]
        series[1][3, 2] = above[3, 2]  # opens at 1, closes at 2
        feed(monitor, product, series)
        alerts = monitor.alerts
        assert [(a.size_class, a.is_open) for a in alerts] == [(3, False), (1, True)]

    @pytest.mark.parametrize(
        ("step", "t", "error", "match"),
        [
            (True, 1.0, TypeError, "step must be an int"),
            (1.0, 1.0, TypeError, "step must be an int"),
            (1, "1", TypeError, "t must be a real number"),
            (1, np.True_, TypeError, "t must be a real number"),
            (-1, 1.0, ValueError, "step must be non-negative"),
            (1, float("nan"), ValueError, "t must be finite"),
            (1, float("inf"), ValueError, "t must be finite"),
            (0, 1.0, ValueError, r"step 0 is not above the last update's 0"),
            (1, 0.0, ValueError, r"t = 0\.0 s is not above the last update's 0\.0 s"),
        ],
    )
    def test_the_step_and_time_are_refused_out_of_order_or_mistyped(
        self, product: tuple, step: object, t: object, error: type, match: str
    ) -> None:
        """Each refusal of the step and the time, after one update at (0, 0.0)."""
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        zeros = {k: np.zeros(mesh.cell_type.shape) for k in range(5)}
        monitor.update(0, 0.0, zeros)
        with pytest.raises(error, match=match):
            monitor.update(step, t, zeros)  # type: ignore[arg-type]

    def test_the_fields_must_be_every_class_on_the_mesh(self, product: tuple) -> None:
        """A missing class, an extra one and a field of another shape are refused."""
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        shape = mesh.cell_type.shape
        four = {k: np.zeros(shape) for k in range(4)}
        with pytest.raises(ValueError, match=r"fields for classes \[0, 1, 2, 3\]"):
            monitor.update(0, 0.0, four)
        six = {k: np.zeros(shape) for k in range(6)}
        with pytest.raises(ValueError, match="the monitor reads classes 0 to 4"):
            monitor.update(0, 0.0, six)
        wrong = {k: np.zeros(shape) for k in range(5)}
        wrong[2] = np.zeros((shape[0], shape[1] + 1))
        with pytest.raises(ValueError, match=r"class 2's field has shape \(30, 81\)"):
            monitor.update(0, 0.0, wrong)

    def test_a_reading_that_is_not_finite_is_refused(self, product: tuple) -> None:
        """NaN compares false both ways, so it would neither open nor close."""
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        values = np.zeros((5, 4))
        values[2, 3] = np.nan
        with pytest.raises(ValueError, match="a reading at step 0 is not finite"):
            feed(monitor, product, [values])
        assert monitor.alerts == () and np.isnan(monitor.readings).all()

    def test_a_refused_update_leaves_the_monitor_as_it_was(
        self, product: tuple
    ) -> None:
        """The order check runs before anything changes: the next good step works."""
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        feed(monitor, product, [np.zeros((5, 4))])
        zeros = {k: np.zeros(mesh.cell_type.shape) for k in range(5)}
        with pytest.raises(ValueError):
            monitor.update(0, 0.5, zeros)
        assert monitor.update(1, 0.1, zeros) == ()


def planted_state(monitor: AlertMonitor, alerts: list[tuple[int, int, float]]) -> dict:
    """A state of closed alerts (sensor index, class, t_on), each one step long."""
    rows = [[0, t_on, 1, t_on + 1.0, k, s, 1.0, 1.0, t_on] for s, k, t_on in alerts]
    return {
        "alert_rows": np.array(rows, dtype=np.float64),
        "alert_step": np.array(1.0),
        "alert_t": np.array(10.0),
        "alert_readings": np.zeros(monitor.readings.shape),
    }


@pytest.mark.unit
class TestLatencies:
    def test_each_onset_takes_the_first_reading_at_or_after_it(
        self, product: tuple
    ) -> None:
        """[events, classes, sensors]: section D's rule on planted alerts, onsets 1 and 3.

        Each planted alert lasts 1 s. above_gap_1's alerts at 1.5, 2.5 and
        4.0 give 0.5 for the first event and zero for the second, whose
        onset falls inside the one from 2.5; hood_entry's at 3.0 gives 2.0
        and 0.0 (an event's latency is not cut at the next onset, and an
        alert opened at the onset exactly is the onset's); near_door's open
        from 0.5 to 1.5 is open at the first onset and gives zero, and the
        one at 3.5 gives 0.5; above_gap_2's closed at the first onset is not
        open at it, so that event is never detected there. Defects caught:
        the window closed at the next onset (step 2's rule: hood_entry's
        first event inf); an alert open at the onset not counted (near_door
        2.5); the latest alert taken rather than the first; the onset's own
        alert excluded by a strict comparison; an alert closed at the onset
        counted as open (above_gap_2 zero).
        """
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        # Sensors 0 to 3 are near_door, above_gap_1, above_gap_2, hood_entry.
        planted = [(1, 0, 1.5), (1, 0, 2.5), (1, 0, 4.0), (3, 2, 3.0)]
        planted += [(0, 4, 0.5), (0, 4, 3.5), (2, 1, 0.0)]
        monitor.restore(planted_state(monitor, planted))
        latency = monitor.latencies([1.0, 3.0])
        assert latency.shape == (2, 5, 4)
        assert latency[0, 0, 1] == 0.5 and latency[1, 0, 1] == 0.0
        assert latency[0, 2, 3] == 2.0 and latency[1, 2, 3] == 0.0
        assert latency[0, 4, 0] == 0.0 and latency[1, 4, 0] == 0.5
        assert latency[0, 1, 2] == np.inf and latency[1, 1, 2] == np.inf
        others = np.ones((2, 5, 4), dtype=bool)
        others[:, 0, 1] = others[:, 2, 3] = others[:, 4, 0] = False
        assert np.isinf(latency[others]).all()

    def test_an_alert_open_at_an_onset_reads_zero_until_the_last_update(
        self, product: tuple
    ) -> None:
        """An alert open from before the onset is the onset's detection; none after the end.

        near_door is at or above from step 0 to the last update at 0.2 s:
        onsets 0, 0.1 and 0.2 read zero, and an onset after the last update
        has no reading at or after it, so inf. Defects caught: the open
        alert excluded (step 2's rule, inf at 0.1); an onset after the end
        given zero by the open alert.
        """
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        values = np.zeros((5, 4))
        values[0, 0] = 2.0 * monitor.thresholds[0]
        feed(monitor, product, [values, values, values])
        latency = monitor.latencies([0.0, 0.1, 0.2, 0.3])
        assert latency[:, 0, 0].tolist() == [0.0, 0.0, 0.0, np.inf]

    def test_two_events_may_share_an_onset(self, product: tuple) -> None:
        """Equal onsets are two rows of one value: two events starting together."""
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        monitor.restore(planted_state(monitor, [(1, 0, 1.5)]))
        latency = monitor.latencies([1.0, 1.0])
        assert latency[0].tobytes() == latency[1].tobytes()
        assert latency[0, 0, 1] == 0.5

    def test_no_onsets_give_no_events(self, product: tuple) -> None:
        """An empty sequence is [0, classes, sensors]."""
        assert AlertMonitor(*product).latencies([]).shape == (0, 5, 4)

    @pytest.mark.parametrize(
        ("onsets", "error", "match"),
        [
            ([1.0, True], TypeError, "an onset must be a real number"),
            (["1.0"], TypeError, "an onset must be a real number"),
            ([2.0, 1.0], ValueError, "the onsets must be in order"),
            ([float("nan")], ValueError, "the onsets must be finite"),
            ([-1.0, 2.0], ValueError, "the onsets must be non-negative"),
        ],
    )
    def test_the_onsets_are_refused_unless_finite_and_in_order(
        self, product: tuple, onsets: list, error: type, match: str
    ) -> None:
        """Each refusal of the onsets."""
        with pytest.raises(error, match=match):
            AlertMonitor(*product).latencies(onsets)

    def test_a_two_dimensional_array_of_onsets_is_refused(self, product: tuple) -> None:
        """An array's rows are not onsets: refused by shape."""
        with pytest.raises(ValueError, match=r"onsets must be \[events\]"):
            AlertMonitor(*product).latencies(np.array([[1.0, 2.0]]))  # type: ignore[arg-type]


@pytest.mark.unit
class TestState:
    def _history(self, monitor: AlertMonitor) -> list[np.ndarray]:
        """Six steps: one alert closed, one open across step 3, one opened after."""
        above = 2.0 * monitor.thresholds
        series = [np.zeros((5, 4)) for _ in range(6)]
        series[1][0, 0] = above[0]
        series[2][0, 0] = above[0] * 2  # closes at 3
        for n in (2, 3, 4, 5):
            series[n][4, 3] = above[4] * n  # open from 2 to the end
        series[5][1, 2] = above[1]
        return series

    def test_a_restored_monitor_continues_as_the_one_it_was_taken_from(
        self, product: tuple
    ) -> None:
        """state() after step 3 restored into a new monitor: steps 4 and 5 agree, bitwise.

        Defects caught: the open alert dropped from the state (it reopens at
        step 4); its peak dropped (the restored peak restarts at step 4's
        reading); the closed alerts dropped; the last step not carried (step
        3 accepted again).
        """
        config, mesh = product
        whole = AlertMonitor(config, mesh)
        series = self._history(whole)
        feed(whole, product, series)
        first = AlertMonitor(config, mesh)
        feed(first, product, series[:4])
        state = first.state()
        assert set(state) == set(STATE_NAMES)
        assert state["alert_rows"].shape == (2, len(ALERT_COLUMNS))
        resumed = AlertMonitor(config, mesh)
        resumed.restore(state)
        assert resumed.alerts == first.alerts
        assert resumed.readings.tobytes() == first.readings.tobytes()
        cells = sensor_cells(config, mesh)
        zeros = {k: np.zeros(mesh.cell_type.shape) for k in range(5)}
        with pytest.raises(ValueError, match="step 3 is not above"):
            resumed.update(3, 0.35, zeros)
        with pytest.raises(ValueError, match=r"t = 0\.3 s is not above the last"):
            resumed.update(4, 0.3, zeros)
        for n in (4, 5):
            resumed.update(n, n * 0.1, fields_of(mesh, series[n], cells))
        assert resumed.alerts == whole.alerts
        assert resumed.state()["alert_rows"].tobytes() == (
            whole.state()["alert_rows"].tobytes()
        )

    def test_a_fresh_state_restores_a_fresh_monitor(self, product: tuple) -> None:
        """Before any update: no rows, NaN step and time; step 0 is accepted after."""
        config, mesh = product
        state = AlertMonitor(config, mesh).state()
        assert state["alert_rows"].shape == (0, len(ALERT_COLUMNS))
        assert np.isnan(state["alert_step"]) and np.isnan(state["alert_t"])
        monitor = AlertMonitor(config, mesh)
        monitor.restore(state)
        zeros = {k: np.zeros(mesh.cell_type.shape) for k in range(5)}
        assert monitor.update(0, 0.0, zeros) == ()

    def test_the_other_carried_names_are_ignored(self, product: tuple) -> None:
        """The checkpoint holds the tracker's arrays beside the monitor's."""
        config, mesh = product
        state = dict(AlertMonitor(config, mesh).state())
        state["last_above"] = np.zeros(3)
        AlertMonitor(config, mesh).restore(state)

    @pytest.mark.parametrize("missing", STATE_NAMES)
    def test_a_missing_array_is_refused(self, product: tuple, missing: str) -> None:
        """Each of the four names is required."""
        config, mesh = product
        state = AlertMonitor(config, mesh).state()
        del state[missing]
        with pytest.raises(KeyError, match=missing):
            AlertMonitor(config, mesh).restore(state)

    def _rows(self, *rows: list[float]) -> dict[str, np.ndarray]:
        """A state holding ROWS and a last update at step 9."""
        return {
            "alert_rows": np.array(rows, dtype=np.float64).reshape(len(rows), 9),
            "alert_step": np.array(9.0),
            "alert_t": np.array(0.9),
            "alert_readings": np.zeros((5, 4)),
        }

    @pytest.mark.parametrize(
        ("state", "match"),
        [
            ({"alert_rows": np.zeros((2, 8))}, r"alert_rows has shape \(2, 8\)"),
            ({"alert_rows": np.zeros(9)}, r"alert_rows has shape \(9,\)"),
            ({"alert_step": np.zeros(2)}, "alert_step and alert_t must be scalars"),
            ({"alert_t": np.zeros((1,))}, "alert_step and alert_t must be scalars"),
            (
                {"alert_readings": np.zeros((4, 5))},
                r"alert_readings has shape \(4, 5\)",
            ),
        ],
    )
    def test_a_malformed_array_is_refused(
        self, product: tuple, state: dict, match: str
    ) -> None:
        """Each array's shape is checked against this monitor's."""
        config, mesh = product
        given = {**self._rows(), **state}
        with pytest.raises(ValueError, match=match):
            AlertMonitor(config, mesh).restore(given)

    @pytest.mark.parametrize(
        ("row", "match"),
        [
            ([1, 0.1, 2, 0.2, 5, 0, 1, 1, 0.1], "names class 5.0 and sensor 0.0"),
            ([1, 0.1, 2, 0.2, 0, 4, 1, 1, 0.1], "names class 0.0 and sensor 4.0"),
            ([1, 0.1, 2, 0.2, -1, 0, 1, 1, 0.1], r"names class -1\.0"),
            ([1, 0.1, 2, 0.2, 0.5, 0, 1, 1, 0.1], "names class 0.5"),
            ([1, 0.1, np.nan, 0.2, 0, 0, 1, 1, 0.1], "one of step_off and t_off"),
            ([1, 0.1, 2, np.nan, 0, 0, 1, 1, 0.1], "one of step_off and t_off"),
            ([1.5, 0.1, 2, 0.2, 0, 0, 1, 1, 0.1], "step_on 1.5 is not a non-negative"),
            ([-1, 0.1, 2, 0.2, 0, 0, 1, 1, 0.1], r"step_on -1\.0 is not"),
            ([1, np.inf, 2, 0.2, 0, 0, 1, 1, 0.1], "or peak_t is not finite"),
            ([1, 0.1, 2, 0.2, 0, 0, 1, np.nan, 0.1], "or peak_t is not finite"),
            (
                [3, 0.3, 3, 0.4, 0, 0, 1, 1, 0.3],
                r"closes at step 3\.0, t = 0\.4 s, not",
            ),
            ([1, 0.3, 2, 0.3, 0, 0, 1, 1, 0.3], "not after it opened"),
            ([1, 0.1, 2.5, 0.2, 0, 0, 1, 1, 0.1], "not after it opened"),
            ([1, 0.1, 2, np.inf, 0, 0, 1, 1, 0.1], "not after it opened"),
            ([1, 0.1, 12, 1.2, 0, 0, 1, 1, 0.1], r"last update \(step 9\.0"),
        ],
    )
    def test_a_row_naming_no_class_or_sensor_or_half_open_is_refused(
        self, product: tuple, row: list[float], match: str
    ) -> None:
        """A row's class and sensor are this monitor's, and open is both NaN or neither."""
        config, mesh = product
        with pytest.raises(ValueError, match=match):
            AlertMonitor(config, mesh).restore(self._rows(row))

    @pytest.mark.parametrize(
        ("step", "t", "match"),
        [
            (np.nan, 0.9, "is not both NaN or a whole step"),
            (9.0, np.nan, "is not both NaN or a whole step"),
            (9.5, 0.9, "is not both NaN or a whole step"),
            (9.0, np.inf, "is not both NaN or a whole step"),
            (np.nan, np.nan, "holds alerts but no last update"),
        ],
    )
    def test_the_last_update_is_refused_unless_whole_finite_and_after_the_rows(
        self, product: tuple, step: float, t: float, match: str
    ) -> None:
        """The restored step and time guard every later update, so they are checked."""
        config, mesh = product
        state = self._rows([1, 0.1, 2, 0.2, 0, 0, 1, 1, 0.1])
        state["alert_step"], state["alert_t"] = np.array(step), np.array(t)
        with pytest.raises(ValueError, match=match):
            AlertMonitor(config, mesh).restore(state)

    def test_a_checkpoint_without_the_monitors_arrays_names_why(
        self, product: tuple
    ) -> None:
        """A checkpoint from before the monitor (or a run without sensors) says so."""
        config, mesh = product
        with pytest.raises(KeyError, match="written before the alert monitor"):
            AlertMonitor(config, mesh).restore({"last_above": np.zeros(1)})

    def test_two_open_alerts_of_one_sensor_and_class_are_refused(
        self, product: tuple
    ) -> None:
        """A sensor's class has one reading a step, so at most one open alert."""
        config, mesh = product
        row = [1, 0.1, np.nan, np.nan, 0, 0, 1, 1, 0.1]
        with pytest.raises(ValueError, match="two open alerts of sensor near_door"):
            AlertMonitor(config, mesh).restore(self._rows(row, row))


def iso_raw(value: object) -> dict:
    """The product file with ``thresholds: {iso_class: VALUE}``."""
    raw = room_raw()
    raw["thresholds"] = {"iso_class": value}
    return raw


@pytest.mark.unit
class TestIsoClass:
    def test_class_5_reproduces_the_product_files_five_values_bitwise(self) -> None:
        """REQ-A03's test: N = 5 gives 100,000; 10,200; 3,520; 832; 29.3.

        Defects caught: the exponent 2.0 (11,100 at 0.3 um); the reference
        diameter 1 um; the limit unrounded (10,173 at 0.3 um); rounded to two
        figures (10,000).
        """
        by_size = SimConfig.from_dict(room_raw())
        by_class = SimConfig.from_dict(iso_raw(5))
        assert by_class.class_thresholds() == by_size.class_thresholds() == CLASS_5
        assert by_class.iso_class == 5.0 and by_size.iso_class is None
        mesh = Mesh(by_class)
        assert AlertMonitor(by_class, mesh).thresholds.tolist() == list(CLASS_5)

    @pytest.mark.parametrize(
        ("number", "expected"),
        [
            (1, (10.0, 1.02, 0.352, 0.0832, 0.00293)),
            (8, (1.0e8, 1.02e7, 3.52e6, 8.32e5, 2.93e4)),
            (9, (1.0e9, 1.02e8, 3.52e7, 8.32e6, 2.93e5)),
            (6.5, (3.16e6, 3.22e5, 1.11e5, 2.63e4, 925.0)),
        ],
    )
    def test_other_classes_follow_the_formula_to_three_figures(
        self, number: float, expected: tuple[float, ...]
    ) -> None:
        """The range's ends and an intermediate class, each to three figures.

        10^6.5 (0.1 / 0.3)^2.08 is 321,801, so three figures give 322,000;
        at 5 um it is 925.001, so 925.
        """
        assert SimConfig.from_dict(iso_raw(number)).class_thresholds() == expected

    def test_the_keys_are_the_class_diameters(self) -> None:
        """The loader keys each limit by its class's diameter, so the lookup finds it."""
        config = SimConfig.from_dict(iso_raw(5))
        assert list(config.thresholds) == ["1e-07", "3e-07", "5e-07", "1e-06", "5e-06"]

    @pytest.mark.parametrize(
        ("value", "error", "match"),
        [
            (True, TypeError, r"thresholds\.iso_class must be a number, got bool"),
            ("5", TypeError, r"thresholds\.iso_class must be a number, got str"),
            (None, TypeError, r"thresholds\.iso_class must be a number"),
            (float("nan"), ValueError, r"thresholds\.iso_class must be finite"),
            (float("inf"), ValueError, r"thresholds\.iso_class must be finite"),
            (ISO_CLASS_RANGE[0] - 0.5, ValueError, "must be from 1 to 9"),
            (ISO_CLASS_RANGE[1] + 0.5, ValueError, "must be from 1 to 9"),
            (0, ValueError, r"must be from 1 to 9 \(ISO 14644-1's classes\), got 0"),
        ],
    )
    def test_the_class_number_is_refused_unless_a_number_from_1_to_9(
        self, value: object, error: type, match: str
    ) -> None:
        """Each refusal of N."""
        with pytest.raises(error, match=match):
            SimConfig.from_dict(iso_raw(value))

    def test_iso_class_beside_a_size_key_is_refused(self) -> None:
        """The two forms are exclusive; the message names the other keys."""
        raw = iso_raw(5)
        raw["thresholds"]["0.5e-6"] = 3520.0
        with pytest.raises(ValueError, match=r"iso_class and \['0\.5e-6'\]: give"):
            SimConfig.from_dict(raw)

    def test_a_scenario_cannot_switch_forms_over_its_base(self, tmp_path: Path) -> None:
        """A scenario's thresholds merge over the base's key by key, so mixing is refused.

        The base states the size mapping; a scenario stating iso_class alone
        merges to both, which the loader refuses with the reason.
        """
        raw = yaml.safe_load(
            (CONFIGS / "scenario_equipment_dust.yaml").read_text("utf-8")
        )
        raw["scenario"]["base"] = str(CONFIGS / "clean_room_default.yaml")
        raw["thresholds"] = {"iso_class": 5}
        with pytest.raises(ValueError, match="a scenario states the base's form"):
            SimConfig.from_dict(raw)


# ---------------------------------------------------------------------------
# ADR-015 step 3: the contaminating events and the arrival maps
# ---------------------------------------------------------------------------


def scenario_raw(name: str) -> dict:
    """A reference scenario file on 80x30, its base an absolute path."""
    raw = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
    raw["scenario"]["base"] = str(CONFIGS / "clean_room_default.yaml")
    raw["domain"] = dict(COARSE_GRID)
    return raw


# The whole-supply breach of ADR-014 B: the supply's filtration lost, its air
# carrying the thresholds times 100 (the reference breach's values).
SUPPLY_BREACH = {
    "name": "supply_breach",
    "start": 20.0,
    "duration": 5.0,
    "set": {
        "segment": "hepa_supply",
        "keys": {
            "hepa_filtered": False,
            "concentration": [1.0e7, 1.02e6, 3.52e5, 8.32e4, 2.93e3],
        },
    },
}


def events_of(raw: dict) -> tuple[tuple[str, float], ...]:
    """``contaminating_events`` of a scenario mapping."""
    spec = SimConfig.from_dict(raw).scenario
    assert spec is not None
    return contaminating_events(spec)


@pytest.mark.unit
class TestContaminatingEvents:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("scenario_clean_out.yaml", (("contaminated_start", 0.0),)),
            ("scenario_door_leak.yaml", (("door_open", 10.0),)),
            ("scenario_filter_breach.yaml", (("panel_breach", 10.0),)),
            ("scenario_equipment_dust.yaml", (("litho_dust", 10.0),)),
        ],
    )
    def test_each_reference_scenarios_one_event(
        self, name: str, expected: tuple
    ) -> None:
        """An initial, a replace with a concentration and a source: one event each.

        Defects caught: the initial read at its start key rather than t = 0
        (the same here, see the ordering test); a kind left out.
        """
        raw = scenario_raw(name)
        assert events_of(raw) == expected
        assert all(carries_contamination(e) for e in raw_events(raw))

    def test_a_set_giving_a_segment_a_concentration_is_counted(self) -> None:
        """The whole-supply breach, a set, is an event; section D's list left it out.

        Defects caught: the set event not counted (the run would have the
        one event RUN_START at t = 0 and a latency from it).
        """
        raw = scenario_raw("scenario_door_leak.yaml")
        raw["scenario"]["events"].append(dict(SUPPLY_BREACH))
        assert events_of(raw) == (("door_open", 10.0), ("supply_breach", 20.0))

    def test_changes_without_a_concentration_are_not_events(self) -> None:
        """A set of other keys and a replace whose segments state none carry nothing.

        With only those, the plan has the one event RUN_START at t = 0.
        Defects caught: every replace or set counted; the fallback event
        missing (no map at all).
        """
        raw = scenario_raw("scenario_door_leak.yaml")
        (door,) = raw["scenario"]["events"]
        for segment in door["replace"]["with"]:
            segment.pop("concentration", None)
        raw["scenario"]["events"].append(
            {
                "name": "slow_supply",
                "start": 20.0,
                "duration": 5.0,
                "set": {"segment": "hepa_supply", "keys": {"velocity": 0.4}},
            }
        )
        spec = SimConfig.from_dict(raw).scenario
        assert spec is not None
        assert not any(carries_contamination(e) for e in spec.events)
        assert contaminating_events(spec) == ((RUN_START, 0.0),)

    def test_events_are_ordered_by_onset_then_by_the_file(self) -> None:
        """Onsets in order whatever the file's order; two at one onset keep theirs.

        Defects caught: the file's order kept (the onsets out of order, which
        the tracker refuses); a sort that is not stable.
        """
        raw = scenario_raw("scenario_equipment_dust.yaml")
        (dust,) = raw["scenario"]["events"]
        dust["start"] = 30.0
        early = yaml.safe_load(yaml.safe_dump(dust))
        early["name"], early["start"] = "early_dust", 5.0
        twin = yaml.safe_load(yaml.safe_dump(dust))
        twin["name"] = "twin_dust"
        raw["scenario"]["events"] = [dust, twin, early]
        assert events_of(raw) == (
            ("early_dust", 5.0),
            ("litho_dust", 30.0),
            ("twin_dust", 30.0),
        )

    def test_the_onset_is_the_start_of_the_interval_the_event_is_active_over(
        self,
    ) -> None:
        """A start within the time tolerance of a boundary takes the boundary's time.

        The march has a step at each interval boundary, not at the stated
        start, so the onset is the boundary. Defects caught: the event's
        stated start used (no step at it; the map would start a step late).
        """
        raw = scenario_raw("scenario_equipment_dust.yaml")
        (dust,) = raw["scenario"]["events"]
        spec = SimConfig.from_dict(raw).scenario
        assert spec is not None
        nudge = 0.5 * spec.time_tolerance
        early = yaml.safe_load(yaml.safe_dump(dust))
        early["name"], early["start"], early["duration"] = "early", 5.0, 5.0
        dust["start"] = 10.0 + nudge
        raw["scenario"]["events"] = [early, dust]
        assert events_of(raw) == (("early", 5.0), ("litho_dust", 10.0))

    def test_an_event_active_over_no_interval_is_left_out(self) -> None:
        """A release inside the time tolerance of t_end starts no interval: no event.

        The loader accepts it (it ends by t_end), and ``times`` merges its
        start and end into t_end, so it is active over no interval and
        brings nothing in. Defect caught: such an event given its stated
        start as an onset, past the march's last step.
        """
        raw = scenario_raw("scenario_equipment_dust.yaml")
        spec = SimConfig.from_dict(raw).scenario
        assert spec is not None
        (dust,) = raw["scenario"]["events"]
        dust["start"] = spec.t_end - 0.5 * spec.time_tolerance
        dust["duration"] = 0.25 * spec.time_tolerance
        assert events_of(raw) == ((RUN_START, 0.0),)


def raw_events(raw: dict) -> tuple:
    """The loaded events of a scenario mapping."""
    spec = SimConfig.from_dict(raw).scenario
    assert spec is not None
    return spec.events


# A 3x4 grid with one SOLID cell, two classes. The planted fields hold one
# value per cell and class; each update's time is a multiple of 0.1 s.
LIVE = np.ones((3, 4), dtype=bool)
LIVE[1, 2] = False
THRESHOLDS = (1.0, 10.0)
TIMES = [0.1 * n for n in range(8)]


def planted(value: float, cells: dict[tuple[int, int], float] | None = None) -> dict:
    """Two class fields: VALUE everywhere, CELLS' values in class 0 and 10x in class 1."""
    out = {}
    for k, scale in enumerate((1.0, 10.0)):
        field = np.full(LIVE.shape, value * scale)
        for (j, i), v in (cells or {}).items():
            field[j, i] = v * scale
        out[k] = field
    return out


def run_series(
    tracker: ArrivalTracker, series: list[dict], times: list[float] | None = None
) -> None:
    """Hand SERIES to update at TIMES (TIMES by default)."""
    for t, fields in zip(times or TIMES, series, strict=False):
        tracker.update(t, fields)


@pytest.mark.unit
class TestArrivalTracker:
    def test_a_cell_at_or_above_at_the_onset_reads_the_onset(self) -> None:
        """The first update at or after the onset, the onset itself where it is one.

        Onset 0.2 s. Cell (0, 0) is above from t = 0: it reads 0.2. Cell
        (0, 1) is exactly at its threshold from 0.3: it reads 0.3. Cell
        (2, 3) is above at 0.1 only, before the onset, and again from 0.5:
        it reads 0.5. Cell (2, 0) is above from 0.4 to the end: 0.4, not
        the last time. Defects caught: ">" for "at or above" ((0, 1) NaN);
        the map started at t = 0 ((0, 0) 0.0, (2, 3) 0.1); the last time
        kept ((2, 0) 0.7).
        """
        tracker = ArrivalTracker(THRESHOLDS, LIVE, [0.2])
        series = []
        for n, t in enumerate(TIMES):
            cells = {(0, 0): 5.0}
            if n >= 3:
                cells[(0, 1)] = 1.0
            if n == 1 or n >= 5:
                cells[(2, 3)] = 2.0
            if n >= 4:
                cells[(2, 0)] = 3.0
            series.append(planted(0.0, cells))
            assert t == TIMES[n]
        run_series(tracker, series)
        first = tracker.first_above
        assert first.shape == (1, 2, 3, 4)
        for k in (0, 1):
            assert first[0, k, 0, 0] == TIMES[2]
            assert first[0, k, 0, 1] == TIMES[3]
            assert first[0, k, 2, 3] == TIMES[5]
            assert first[0, k, 2, 0] == TIMES[4]
        reached = np.zeros((3, 4), dtype=bool)
        reached[0, 0] = reached[0, 1] = reached[2, 3] = reached[2, 0] = True
        assert np.isnan(first[0][:, ~reached]).all()

    def test_the_maps_and_the_state_are_copies(self) -> None:
        """A write to what first_above and state() return leaves the tracker as it was.

        Reached cells and a set time, so there is something to corrupt.
        Defects caught: ``first_above`` returning the live array; ``state()``
        returning the live maps, onsets or time (the checkpoint's write would
        then alias a later update).
        """
        tracker = ArrivalTracker(THRESHOLDS, LIVE, [0.0])
        tracker.update(0.0, planted(0.0, {(0, 0): 2.0}))
        held = {**tracker.state(), "onsets": tracker.onsets}
        before = {name: array.copy() for name, array in held.items()}
        assert before["first_above"][0, 0, 0, 0] == 0.0
        tracker.first_above[...] = -1.0
        for array in tracker.state().values():
            array[...] = -1.0
        tracker.onsets[...] = -1.0
        after = {**tracker.state(), "onsets": tracker.onsets}
        assert after.keys() == before.keys()
        for name, array in before.items():
            assert after[name].tobytes() == array.tobytes(), name
        assert tracker.first_above.tobytes() == before["first_above"].tobytes()

    def test_update_leaves_a_writeable_field_bitwise_as_it_was(self) -> None:
        """REQ-A06 on the tracker's update alone: every field's bytes are equal after.

        The fields are handed writeable and with cells above and below the
        thresholds, so a write would land. Defect caught: a field written in
        update (the reached cells zeroed).
        """
        tracker = ArrivalTracker(THRESHOLDS, LIVE, [0.0])
        fields = planted(0.5, {(0, 0): 2.0, (2, 3): 1.0})
        before = {k: fields[k].tobytes() for k in fields}
        tracker.update(0.0, fields)
        assert np.isfinite(tracker.first_above).any()
        assert all(fields[k].tobytes() == before[k] for k in fields)

    def test_a_second_events_map_starts_at_its_onset(self) -> None:
        """Onsets 0 and 0.25: a cell reached for event 0 counts for event 1 only from 0.3.

        Cell (0, 0) is above at 0.1 only: event 0 reads 0.1 and event 1
        never. Cell (1, 1) is above from 0.1 to the end: event 0 reads 0.1,
        event 1 the first update at or after 0.25, 0.3. Defects caught: a
        later event's map started at t = 0 (event 1 reading 0.1 at both);
        a cell reached for event 0 not counted again for event 1 ((1, 1)
        NaN for event 1).
        """
        tracker = ArrivalTracker(THRESHOLDS, LIVE, [0.0, 0.25])
        series = []
        for n in range(len(TIMES)):
            cells = {}
            if n == 1:
                cells[(0, 0)] = 2.0
            if n >= 1:
                cells[(1, 1)] = 2.0
            series.append(planted(0.0, cells))
        run_series(tracker, series)
        first = tracker.first_above
        assert first[0, 0, 0, 0] == TIMES[1] and np.isnan(first[1, 0, 0, 0])
        assert first[0, 0, 1, 1] == TIMES[1] and first[1, 0, 1, 1] == TIMES[3]

    def test_solid_cells_are_never_reached_and_the_record_marks_the_rest(
        self,
    ) -> None:
        """The record: inf where a non-SOLID cell was not reached, NaN in SOLID ones.

        Every cell, the SOLID one too, is planted above from 0.1. Defects
        caught: a SOLID cell given a time; the record's unreached cells left
        NaN, which a lookup cannot tell from SOLID.
        """
        tracker = ArrivalTracker(THRESHOLDS, LIVE, [0.0, 0.5])
        run_series(tracker, [planted(0.0)] + [planted(2.0)] * 3)
        record = tracker.record(["a", "b"], TIMES[3])
        onsets = tracker.onsets
        onsets[0] = 9.0
        assert tracker.onsets.tolist() == [0.0, 0.5]
        assert record.event_names == ("a", "b")
        assert record.onsets.tolist() == [0.0, 0.5]
        assert record.thresholds.tolist() == list(THRESHOLDS)
        assert record.t_final == TIMES[3]
        assert np.isnan(record.first_above[:, :, 1, 2]).all()
        live = np.broadcast_to(LIVE, (2, 3, 4))
        assert (record.first_above[0][:, live[0]] == TIMES[1]).all()
        assert np.isposinf(record.first_above[1][:, live[0]]).all()
        assert np.isnan(tracker.first_above[1]).all()

    def test_a_restored_tracker_continues_bitwise(self) -> None:
        """state() after four updates restored into a new tracker: the end agrees bitwise.

        Defects caught: the maps dropped from the state (cells reached
        before the checkpoint read later times); the last time not carried
        (an update at it accepted again).
        """
        series = [planted(0.0, {(j, j): 0.5 * n for j in range(3)}) for n in range(8)]
        whole = ArrivalTracker(THRESHOLDS, LIVE, [0.0, 0.25])
        run_series(whole, series)
        first = ArrivalTracker(THRESHOLDS, LIVE, [0.0, 0.25])
        run_series(first, series[:4])
        state = first.state()
        assert set(state) == set(ARRIVAL_STATE_NAMES)
        assert np.isfinite(state["first_above"]).any()
        resumed = ArrivalTracker(THRESHOLDS, LIVE, [0.0, 0.25])
        resumed.restore({**state, "last_above": np.zeros(2)})
        with pytest.raises(ValueError, match="is not above the last update's"):
            resumed.update(TIMES[3], series[3])
        run_series(resumed, series[4:], TIMES[4:])
        assert resumed.first_above.tobytes() == whole.first_above.tobytes()
        assert resumed.state()["arrival_t"] == whole.state()["arrival_t"]

    def test_a_fresh_state_restores_a_fresh_tracker(self) -> None:
        """Before any update the maps are NaN and the time NaN; t = 0 is accepted after."""
        state = ArrivalTracker(THRESHOLDS, LIVE, [0.0]).state()
        assert np.isnan(state["first_above"]).all() and np.isnan(state["arrival_t"])
        tracker = ArrivalTracker(THRESHOLDS, LIVE, [0.0])
        tracker.restore(state)
        tracker.update(0.0, planted(2.0))
        assert tracker.first_above[0, 0, 0, 0] == 0.0

    @pytest.mark.parametrize("missing", ARRIVAL_STATE_NAMES)
    def test_a_checkpoint_without_the_maps_names_why(self, missing: str) -> None:
        """A checkpoint from before step 3 carries none of the three names."""
        state = ArrivalTracker(THRESHOLDS, LIVE, [0.0]).state()
        del state[missing]
        with pytest.raises(
            KeyError, match=r"before the arrival maps \(ADR-015 step 3\)"
        ):
            ArrivalTracker(THRESHOLDS, LIVE, [0.0]).restore(state)

    def _state(self, first: np.ndarray, t: float) -> dict[str, np.ndarray]:
        """A state of FIRST with onsets [0, 0.25] and the last update at T."""
        return {
            "first_above": first,
            "arrival_onsets": np.array([0.0, 0.25]),
            "arrival_t": np.array(t),
        }

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ("shape", "first_above has shape"),
            ("onsets", r"onsets \[0\.0, 0\.3\] are not this run's"),
            ("t_shape", "arrival_t must be a NaN or finite scalar"),
            ("t_inf", "arrival_t must be a NaN or finite scalar"),
            ("no_update", "hold times but no last update"),
            ("solid", "a time in a SOLID cell"),
            ("before_onset", "before its event's onset or after the last update"),
            ("after_last", "before its event's onset or after the last update"),
        ],
    )
    def test_a_malformed_state_is_refused(self, change: str, match: str) -> None:
        """Each refusal of restore, from a well-formed state with one array changed."""
        first = np.full((2, 2, 3, 4), np.nan)
        first[0, 0, 0, 0] = 0.1
        state = self._state(first, 0.5)
        if change == "shape":
            state["first_above"] = first[:, :, :2]
        elif change == "onsets":
            state["arrival_onsets"] = np.array([0.0, 0.3])
        elif change == "t_shape":
            state["arrival_t"] = np.array([0.5])
        elif change == "t_inf":
            state["arrival_t"] = np.array(np.inf)
        elif change == "no_update":
            state["arrival_t"] = np.array(np.nan)
        elif change == "solid":
            first[0, 0, 1, 2] = 0.1
        elif change == "before_onset":
            first[1, 0, 0, 0] = 0.2
        else:
            first[0, 1, 0, 0] = 0.6
        tracker = ArrivalTracker(THRESHOLDS, LIVE, [0.0, 0.25])
        with pytest.raises(ValueError, match=match):
            tracker.restore(state)
        assert np.isnan(tracker.first_above).all()

    @pytest.mark.parametrize(
        ("args", "error", "match"),
        [
            (((), LIVE, [0.0]), ValueError, r"thresholds must be \[classes\]"),
            ((((1.0,),), LIVE, [0.0]), ValueError, r"thresholds must be \[classes\]"),
            (((1.0, True), LIVE, [0.0]), TypeError, "a threshold must be a real"),
            (((1.0, -1.0), LIVE, [0.0]), ValueError, "finite and non-negative"),
            (((1.0, np.inf), LIVE, [0.0]), ValueError, "finite and non-negative"),
            ((THRESHOLDS, LIVE.astype(int), [0.0]), TypeError, "live must be a bool"),
            ((THRESHOLDS, LIVE[0], [0.0]), ValueError, r"live must be \[ny, nx\]"),
            ((THRESHOLDS, LIVE, []), ValueError, "at least one event"),
            ((THRESHOLDS, LIVE, [-1.0]), ValueError, "onsets must be non-negative"),
            ((THRESHOLDS, LIVE, [1.0, 0.5]), ValueError, "onsets must be in order"),
            ((THRESHOLDS, LIVE, [np.nan]), ValueError, "onsets must be finite"),
            ((THRESHOLDS, LIVE, [False]), TypeError, "an onset must be a real"),
        ],
    )
    def test_the_construction_is_refused_unless_well_formed(
        self, args: tuple, error: type, match: str
    ) -> None:
        """Each refusal of the constructor."""
        with pytest.raises(error, match=match):
            ArrivalTracker(*args)

    def test_update_refuses_the_time_and_the_fields(self) -> None:
        """A mistyped, non-finite or repeated time and fields of the wrong classes or shape.

        A refused update leaves the maps as they were.
        """
        tracker = ArrivalTracker(THRESHOLDS, LIVE, [0.0])
        tracker.update(0.0, planted(0.0))
        cases = [
            (True, planted(2.0), TypeError, "t must be a real number"),
            (np.inf, planted(2.0), ValueError, "t must be finite"),
            (0.0, planted(2.0), ValueError, "is not above the last update's"),
            (0.1, {0: planted(2.0)[0]}, ValueError, r"fields for classes \[0\]"),
            (0.1, {0: np.ones((2, 2)), 1: np.ones((2, 2))}, ValueError, "shape"),
        ]
        for t, fields, error, match in cases:
            with pytest.raises(error, match=match):
                tracker.update(t, fields)
        assert np.isnan(tracker.first_above).all()

    def test_record_refuses_the_names_and_the_end(self) -> None:
        """One name per event, and an end that is finite and not before the last update."""
        tracker = ArrivalTracker(THRESHOLDS, LIVE, [0.0, 0.25])
        tracker.update(0.5, planted(0.0))
        with pytest.raises(ValueError, match="1 event names for 2 events"):
            tracker.record(["a"], 0.5)
        with pytest.raises(TypeError, match="t_final must be a real number"):
            tracker.record(["a", "b"], "0.5")  # type: ignore[arg-type]
        for end in (0.4, np.nan):
            with pytest.raises(ValueError, match="not finite and at or after"):
                tracker.record(["a", "b"], end)
        assert tracker.record(["a", "b"], 0.6).t_final == 0.6

    def test_the_maps_and_the_monitor_agree_at_the_sensors_to_the_step(
        self, product: tuple
    ) -> None:
        """A random series at the sensors: the map's latency there is the monitor's, exactly.

        Sixty updates of readings drawn up to 3% over each class's threshold
        and a few exactly at it (each class's own), hood_entry's 5 um class
        never reached; four onsets at update times, two of them equal.
        Defects caught: any difference between the two rules (the window,
        the open alert at an onset, the strict comparison) shows as a sensor
        that disagrees.
        """
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        cells = sensor_cells(config, mesh)
        live = mesh.cell_type != SOLID
        times = [0.05 * n for n in range(60)]
        onsets = [times[0], times[17], times[17], times[41]]
        tracker = ArrivalTracker(monitor.thresholds.tolist(), live, onsets)
        rng = np.random.default_rng(74)
        for n, t in enumerate(times):
            scale = rng.uniform(0.0, 1.03, size=(5, 4))
            values = scale * monitor.thresholds[:, None]
            at_threshold = rng.uniform(size=(5, 4)) < 0.02
            values = np.where(at_threshold, monitor.thresholds[:, None], values)
            values[4, 3] = 0.5 * monitor.thresholds[4]
            fields = fields_of(mesh, values, cells)
            monitor.update(n, t, fields)
            tracker.update(t, fields)
        rows = [j for j, _ in cells]
        cols = [i for _, i in cells]
        maps = tracker.record(list("abcd"), times[-1]).first_above
        mapped = maps[:, :, rows, cols] - np.array(onsets)[:, None, None]
        latency = monitor.latencies(onsets)
        assert np.isfinite(latency).any() and np.isinf(latency).any()
        assert (latency == 0.0).any() and (latency > 0.0).any()
        assert mapped.tobytes() == latency.tobytes()
