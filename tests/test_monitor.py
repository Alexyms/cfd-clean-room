"""The alert monitor and the thresholds it reads (src/monitor.py, src/config.py; ADR-015 C).

ADR-015 H's step 2, unit tier. What is pinned, each test naming in its
docstring the planted defect it was shown to catch:

- The construction: no sensor, a class without a threshold and a sensor in a
  SOLID cell are refused; the thresholds are the size mapping's, in class
  order; each sensor reads the cell of ``sensor_cells``, which moved here
  from ``time_integration`` and is re-exported there unchanged.
- ``update``'s refusals (the step and time order, the classes, the shapes,
  a non-finite reading) and its return: the alerts the step opened or
  closed, the peak carried while open, the closed alerts before the open.
- ``latencies`` on planted alerts with two onsets, and its refusals.
- ``state`` and ``restore``: a restored monitor continues as the one it was
  taken from, and a malformed state is refused.
- REQ-A03's second form, ``thresholds: {iso_class: N}``: the product file's
  five Class 5 values reproduced bitwise, the rounding to three figures,
  and the rejection paths.

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
    SENSOR_FACE_RTOL,
    STATE_NAMES,
    Alert,
    AlertMonitor,
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


def planted_alert(sensor: str, size_class: int, t_on: float) -> Alert:
    """A closed alert opened at T_ON, for latencies."""
    return Alert(sensor, size_class, 0, t_on, 1.0, 1, t_on + 1.0, 1.0, t_on)


@pytest.mark.unit
class TestLatencies:
    def test_two_onsets_each_take_the_first_alert_in_their_window(
        self, product: tuple
    ) -> None:
        """[events, classes, sensors]: the first t_on at or after the onset, before the next.

        Planted on the monitor's alerts: a sensor alerting at 1.5 and 4.0
        with onsets 1.0 and 3.0 gives 0.5 and 1.0; an alert opened at an
        onset exactly gives zero; an alert before the first onset belongs to
        no event; a sensor with no alert in a window gives inf. Defects
        caught: the window not closed at the next onset (event 0 taking
        4.0's alert in place of none, see near_door); the latest alert taken
        rather than the first; the onset's own alert excluded by a strict
        comparison.
        """
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        monitor._closed = [  # planted, as update would leave them
            planted_alert("above_gap_1", 0, 1.5),
            planted_alert("above_gap_1", 0, 2.5),
            planted_alert("above_gap_1", 0, 4.0),
            planted_alert("hood_entry", 2, 3.0),
            planted_alert("near_door", 4, 0.5),
            planted_alert("near_door", 4, 3.5),
        ]
        latency = monitor.latencies([1.0, 3.0])
        assert latency.shape == (2, 5, 4)
        assert latency[0, 0, 1] == 0.5 and latency[1, 0, 1] == 1.0
        assert latency[0, 2, 3] == np.inf and latency[1, 2, 3] == 0.0
        assert latency[0, 4, 0] == np.inf and latency[1, 4, 0] == 0.5
        others = np.ones((2, 5, 4), dtype=bool)
        others[:, 0, 1] = others[:, 2, 3] = others[:, 4, 0] = False
        assert np.isinf(latency[others]).all()

    def test_an_alert_open_at_an_onset_is_not_that_events_detection(
        self, product: tuple
    ) -> None:
        """An alert still open from before the onset does not count; inf until a new one."""
        config, mesh = product
        monitor = AlertMonitor(config, mesh)
        values = np.zeros((5, 4))
        values[0, 0] = 2.0 * monitor.thresholds[0]
        feed(monitor, product, [values, values, values])
        assert monitor.latencies([0.1])[0, 0, 0] == np.inf
        assert monitor.latencies([0.0])[0, 0, 0] == 0.0

    def test_no_onsets_give_no_events(self, product: tuple) -> None:
        """An empty sequence is [0, classes, sensors]."""
        assert AlertMonitor(*product).latencies([]).shape == (0, 5, 4)

    @pytest.mark.parametrize(
        ("onsets", "error", "match"),
        [
            ([1.0, True], TypeError, "an onset must be a real number"),
            (["1.0"], TypeError, "an onset must be a real number"),
            ([2.0, 1.0], ValueError, "the onsets must be increasing"),
            ([1.0, 1.0], ValueError, "the onsets must be increasing"),
            ([float("nan")], ValueError, "the onsets must be finite"),
        ],
    )
    def test_the_onsets_are_refused_unless_finite_and_increasing(
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
    def _history(self, mesh: Mesh, monitor: AlertMonitor) -> list[np.ndarray]:
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
        series = self._history(mesh, whole)
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
        ],
    )
    def test_a_row_naming_no_class_or_sensor_or_half_open_is_refused(
        self, product: tuple, row: list[float], match: str
    ) -> None:
        """A row's class and sensor are this monitor's, and open is both NaN or neither."""
        config, mesh = product
        with pytest.raises(ValueError, match=match):
            AlertMonitor(config, mesh).restore(self._rows(row))

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
