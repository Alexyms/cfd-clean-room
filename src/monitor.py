"""The alert monitor: each sensor's reading every step and the alerts it raises (ADR-015 C).

A sensor reads the cell containing it, the one below or to the left of it
when it sits on a face (``sensor_cells``, ADR-014 C's rule, moved here from
``time_integration``). The reading is instantaneous: the cell's value after
the step, the reading the clean-out's record already uses, so Phase 4's
recovery times and Phase 5's latencies come from one instrument.

An alert opens at the first step whose reading is at or above the class's
threshold and closes at the first step after it whose reading is below; it
records both steps and times, the opening reading and the peak between
them. The march calls ``update`` after every transport step with fields it
has already made read-only, so the monitor adds no delay of its own to a
detection (REQ-A01, VAL-009) and reads the fields without writing them
(REQ-A06). A counter's sample time is not modelled: it is an instrument
delay stated beside the latency (ADR-015 E, OPEN).

The monitor's whole state is ``state()``, every alert so far and the last
update's step, time and readings, so a march restarted from its checkpoint
continues an open alert and writes the alert table the uninterrupted run
writes, bitwise. The closed alerts are carried as well as the open ones
because the run file's alert table is replaced whole at each write, and a
restart would otherwise lose the rows before the checkpoint.

The arrival maps (ADR-015 D, step 3). Per contaminating event, class and
non-SOLID cell, ``ArrivalTracker`` keeps the first step time at or after the
event's onset at which the cell is at or above its class's threshold: the
forward counterpart of the clean-out's local recovery field, updated every
step beside it from the same fields. The events and their onsets come from
the scenario's plan (``contaminating_events``), and the monitor's
``latencies`` follow the same rule on the sensors' cells, so the two
instruments agree to the step at a sensor.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.config import (
    EventSpec,
    Initial,
    Replace,
    ScenarioSpec,
    Set,
    SimConfig,
    Source,
)
from src.io_manager import (
    ArrivalRecord,
    checked_onsets,
    checked_thresholds,
    real_number,
)
from src.mesh import SOLID, Mesh

# A sensor this fraction of the room's extent from a face is on it. The faces
# carry the rounding of the mesh's coordinates (about 1e-16 of the extent),
# and the room's sensors are stated on multiples of 0.05 m that fall on faces.
SENSOR_FACE_RTOL = 1.0e-9
# The columns of ``state()["alert_rows"]``, one row per alert. Steps and the
# class and sensor indices are stored as float64, exact far beyond any run's
# step count; step_off and t_off are NaN while the alert is open.
ALERT_COLUMNS: tuple[str, ...] = (
    "step_on",
    "t_on",
    "step_off",
    "t_off",
    "size_class",
    "sensor",
    "reading_on",
    "peak",
    "peak_t",
)
# The names ``state()`` carries; distinct from the recovery tracker's, since
# the march's checkpoint holds both in one mapping.
STATE_NAMES: tuple[str, ...] = (
    "alert_rows",
    "alert_step",
    "alert_t",
    "alert_readings",
)
# The names ``ArrivalTracker.state()`` carries, distinct from the recovery
# tracker's and the monitor's in the one checkpoint mapping.
ARRIVAL_STATE_NAMES: tuple[str, ...] = ("first_above", "arrival_onsets", "arrival_t")
# The one event of a run whose plan has no contaminating event (ADR-015 D).
RUN_START = "run_start"
# The segment key that gives a boundary segment a concentration of its own.
CONCENTRATION_KEY = "concentration"


@dataclass(frozen=True)
class Alert:
    """One alert: a sensor's reading of one class at or above its threshold.

    Parameters
    ----------
    sensor : str
        The sensor's name.
    size_class : int
        The class index.
    step_on, t_on : int, float
        The step that opened it and its time, seconds: the first step of
        the stretch whose reading is at or above the threshold.
    reading_on : float
        That step's reading, particles per cubic metre.
    step_off, t_off : int or None, float or None
        The first step after it whose reading is below the threshold, and
        its time; None while the alert is open.
    peak, peak_t : float
        The largest reading from ``step_on`` to the step before
        ``step_off`` (to the last update while open) and the first time it
        was read.
    """

    sensor: str
    size_class: int
    step_on: int
    t_on: float
    reading_on: float
    step_off: int | None
    t_off: float | None
    peak: float
    peak_t: float

    @property
    def is_open(self) -> bool:
        """Whether the reading was still at or above the threshold at the last update."""
        return self.step_off is None


def _cell_holding(faces: np.ndarray, position: float) -> int:
    """The index of the cell whose faces hold POSITION, the lower one on a face."""
    tolerance = SENSOR_FACE_RTOL * float(faces[-1] - faces[0])
    index = int(np.searchsorted(faces, position - tolerance, side="left")) - 1
    return min(max(index, 0), faces.size - 2)


def sensor_cells(config: SimConfig, mesh: Mesh) -> tuple[tuple[int, int], ...]:
    """The (j, i) of the cell each sensor reads, in the configuration's order.

    The cell containing the sensor (ADR-014 C). The room's sensors sit on
    cell faces on every grid that puts its positions on faces, so a sensor
    within ``SENSOR_FACE_RTOL`` of the room's extent of a face reads the
    cell below it or to its left, on each axis. On 160x60 that is the cell
    the probe of ADR-014 T7 read for every sensor, by the nearest centre;
    the nearest centre itself is not a rule on a face, since which of the
    two equidistant centres is nearer is decided by rounding (on 80x30 it
    puts above_gap_1 to the right of its face and the other three to the
    left).

    Parameters
    ----------
    config : SimConfig
        Supplies the sensors.
    mesh : Mesh
        The run's mesh.

    Returns
    -------
    tuple[tuple[int, int], ...]
        One (j, i) per sensor.

    Raises
    ------
    ValueError
        If a sensor's cell is SOLID.
    """
    cells = []
    for sensor in config.sensors:
        i = _cell_holding(mesh.x, sensor.x)
        j = _cell_holding(mesh.y, sensor.y)
        if mesh.cell_type[j, i] == SOLID:
            raise ValueError(f"sensor {sensor.name} reads a SOLID cell ({j}, {i})")
        cells.append((j, i))
    return tuple(cells)


def carries_contamination(event: EventSpec) -> bool:
    """Whether an event brings particles into the room (ADR-015 D).

    An initial condition, a source, a replacement one of whose segments
    states a concentration, or a set that gives its segment one (the
    whole-supply breach of ADR-014 B, a clarification of section D's list,
    which names the first three). A concentration key counts whatever its
    values: the plan's events are read from the file, not from the fields.

    Parameters
    ----------
    event : EventSpec
        One of the scenario's events.

    Returns
    -------
    bool
    """
    change = event.change
    if isinstance(change, (Initial, Source)):
        return True
    if isinstance(change, Replace):
        return any(CONCENTRATION_KEY in keys for _, keys in change.segments)
    assert isinstance(change, Set)
    return CONCENTRATION_KEY in change.keys


def contaminating_events(scenario: ScenarioSpec) -> tuple[tuple[str, float], ...]:
    """The plan's contaminating events and their onsets, in onset order (ADR-015 D).

    Parameters
    ----------
    scenario : ScenarioSpec
        The configuration's scenario section.

    Returns
    -------
    tuple[tuple[str, float], ...]
        (name, onset) per event that ``carries_contamination``, ordered by
        onset and then by the file's order. An initial's onset is t = 0;
        any other event's is the start of the first interval it is active
        over (``ScenarioSpec.times`` and ``is_active``, the manager's
        rule), so it is a step time of the march. An event active over no
        interval contaminates nothing and is left out. A plan with none has
        the one event ``RUN_START`` at t = 0.
    """
    starts = scenario.times()[:-1]
    out = []
    for event in scenario.events:
        if not carries_contamination(event):
            continue
        if isinstance(event.change, Initial):
            out.append((event.name, 0.0))
            continue
        onset = next((t for t in starts if scenario.is_active(event, t)), None)
        if onset is not None:
            out.append((event.name, onset))
    if not out:
        return ((RUN_START, 0.0),)
    return tuple(sorted(out, key=lambda pair: pair[1]))


def _check_fields(
    fields: Mapping[int, np.ndarray], n_classes: int, shape: tuple[int, ...]
) -> None:
    """Raise unless FIELDS holds classes 0 to n - 1, each of SHAPE."""
    if sorted(fields) != list(range(n_classes)):
        raise ValueError(
            f"fields for classes {sorted(fields)}; the monitor reads classes 0 "
            f"to {n_classes - 1}"
        )
    for k in range(n_classes):
        given = np.shape(fields[k])
        if given != shape:
            raise ValueError(
                f"class {k}'s field has shape {given}; the mesh's is {shape}"
            )


def _whole(value: float, below: float = math.inf) -> bool:
    """Whether VALUE is a whole number from 0 to below BELOW (finite by then)."""
    return bool(np.isfinite(value) and value == int(value) and 0 <= value < below)


@dataclass
class _Open:
    """An open alert's running record."""

    step_on: int
    t_on: float
    reading_on: float
    peak: float
    peak_t: float


class AlertMonitor:
    """Each sensor's reading of each class every step, and the alerts (ADR-015 C).

    Parameters
    ----------
    config : SimConfig
        Supplies the sensors and each class's threshold
        (``SimConfig.class_thresholds``).
    mesh : Mesh
        The run's mesh, for the cells the sensors read.

    Raises
    ------
    ValueError
        With no sensor, when a class has no threshold, or when a sensor
        reads a SOLID cell.

    Notes
    -----
    "At or above" opens an alert and "below" closes it, matching the
    recovery field's "not below": a reading exactly at its threshold is an
    exceedance. The fields handed to ``update`` are read by index and never
    written (REQ-A06).
    """

    def __init__(self, config: SimConfig, mesh: Mesh) -> None:
        if not config.sensors:
            raise ValueError(
                "the alert monitor needs at least one sensor: an alert is a sensor's "
                "reading at or above its class's threshold (ADR-015 C)"
            )
        self._thresholds = np.array(
            config.class_thresholds("the alert monitor"), dtype=np.float64
        )
        cells = sensor_cells(config, mesh)
        self._rows = np.array([j for j, _ in cells])
        self._cols = np.array([i for _, i in cells])
        self._shape = mesh.cell_type.shape
        self.names: tuple[str, ...] = tuple(sensor.name for sensor in config.sensors)
        n_classes, n_sensors = self._thresholds.shape[0], len(cells)
        self._readings = np.full((n_classes, n_sensors), np.nan)
        self._closed: list[Alert] = []
        self._open: dict[tuple[int, int], _Open] = {}
        self._step: int | None = None
        self._t: float | None = None

    @property
    def thresholds(self) -> np.ndarray:
        """[classes], each class's threshold, particles per cubic metre (a copy)."""
        return self._thresholds.copy()

    @property
    def readings(self) -> np.ndarray:
        """[classes, sensors], the last update's readings (a copy); NaN before any."""
        return self._readings.copy()

    @property
    def alerts(self) -> tuple[Alert, ...]:
        """Every alert so far: the closed ones, then the open ones last.

        Each group is ordered by opening step, then class, then the
        sensor's place in the configuration. An open alert carries its peak
        to the last update.
        """
        index = {name: s for s, name in enumerate(self.names)}

        def order(alert: Alert) -> tuple[int, int, int]:
            return (alert.step_on, alert.size_class, index[alert.sensor])

        closed = sorted(self._closed, key=order)
        opened = sorted(
            (self._alert(key, record) for key, record in self._open.items()),
            key=order,
        )
        return (*closed, *opened)

    def _alert(
        self,
        key: tuple[int, int],
        record: _Open,
        step_off: int | None = None,
        t_off: float | None = None,
    ) -> Alert:
        """The Alert of an open record, closed at (step_off, t_off) when given."""
        k, s = key
        return Alert(
            sensor=self.names[s],
            size_class=k,
            step_on=record.step_on,
            t_on=record.t_on,
            reading_on=record.reading_on,
            step_off=step_off,
            t_off=t_off,
            peak=record.peak,
            peak_t=record.peak_t,
        )

    def update(
        self, step: int, t: float, fields: Mapping[int, np.ndarray]
    ) -> tuple[Alert, ...]:
        """Read every sensor after a step (or the initial fields at step 0).

        Parameters
        ----------
        step : int
            The run's step count, non-negative and above the last update's.
        t : float
            The step's time, seconds, finite and above the last update's.
        fields : Mapping[int, np.ndarray]
            Class to its field, [ny, nx], for every class 0 to n - 1. Read,
            never written.

        Returns
        -------
        tuple[Alert, ...]
            The alerts this step opened (``step_on == step``) and closed
            (``step_off == step``), by class then sensor. A sensor's class
            opens or closes at most one per step.

        Raises
        ------
        TypeError
            If ``step`` is not an int (a bool is not one here) or ``t`` not
            a real number.
        ValueError
            If ``step`` or ``t`` is not above the last update's, ``t`` is
            not finite, the classes are not 0 to n - 1, a field is not
            [ny, nx], or a reading is not finite.
        """
        if isinstance(step, (bool, np.bool_)) or not isinstance(
            step, (int, np.integer)
        ):
            raise TypeError(f"step must be an int, got {type(step).__name__}")
        if isinstance(t, (bool, np.bool_)) or not isinstance(
            t, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"t must be a real number, got {type(t).__name__}")
        step, t = int(step), float(t)
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        if not math.isfinite(t):
            raise ValueError(f"t must be finite, got {t}")
        if self._step is not None and step <= self._step:
            raise ValueError(f"step {step} is not above the last update's {self._step}")
        if self._t is not None and t <= self._t:
            raise ValueError(
                f"t = {t!r} s is not above the last update's {self._t!r} s"
            )
        n_classes = self._thresholds.shape[0]
        _check_fields(fields, n_classes, self._shape)
        readings = np.array(
            [np.asarray(fields[k])[self._rows, self._cols] for k in range(n_classes)],
            dtype=np.float64,
        )
        if not np.isfinite(readings).all():
            raise ValueError(f"a reading at step {step} is not finite")
        changed = []
        above = readings >= self._thresholds[:, None]
        for k in range(n_classes):
            for s in range(len(self.names)):
                key, reading = (k, s), float(readings[k, s])
                record = self._open.get(key)
                if record is None and above[k, s]:
                    record = _Open(step, t, reading, reading, t)
                    self._open[key] = record
                    changed.append(self._alert(key, record))
                elif record is not None and not above[k, s]:
                    closed = self._alert(key, self._open.pop(key), step, t)
                    self._closed.append(closed)
                    changed.append(closed)
                elif record is not None and reading > record.peak:
                    record.peak, record.peak_t = reading, t
        self._readings = readings
        self._step, self._t = step, t
        return tuple(changed)

    def latencies(self, onsets: Sequence[float]) -> np.ndarray:
        """Each sensor's detection latency per event and class (REQ-A04, ADR-015 D).

        Parameters
        ----------
        onsets : Sequence[float]
            The events' onsets, seconds (``contaminating_events``), finite,
            non-negative and in order; two events may share one.

        Returns
        -------
        np.ndarray
            [events, classes, sensors]: the time of the first reading at or
            above the threshold at or after the event's onset, minus the
            onset; inf when there is none by the last update. An alert open
            at the onset counts as the onset itself, so a sensor already at
            or above its threshold when the event starts reads zero.

        Raises
        ------
        TypeError
            If an onset is not a real number.
        ValueError
            If the onsets are not one-dimensional, finite, non-negative and
            in order.

        Notes
        -----
        This is the arrival map's rule on the sensors' cells
        (``ArrivalTracker``), so the two agree to the step: every event
        starts on an interval boundary, and the march puts a step on each
        boundary, so an onset is an update's time. An alert closed at the
        onset's own update is not open at it. An event's latency is not cut
        off at the next event's onset, as the map's is not.
        """
        times = checked_onsets(onsets)
        n_classes, n_sensors = self._readings.shape
        out = np.full((times.size, n_classes, n_sensors), np.inf)
        last = -math.inf if self._t is None else self._t
        sensor_index = {name: s for s, name in enumerate(self.names)}
        for alert in self.alerts:
            s, k = sensor_index[alert.sensor], alert.size_class
            off = math.inf if alert.t_off is None else alert.t_off
            for e, onset in enumerate(times):
                if alert.t_on >= onset:
                    delay = alert.t_on - onset
                elif onset < off and onset <= last:
                    delay = 0.0
                else:
                    continue
                out[e, k, s] = min(out[e, k, s], delay)
        return out

    def state(self) -> dict[str, np.ndarray]:
        """The monitor's state, for the march's checkpoint.

        Returns
        -------
        dict[str, np.ndarray]
            ``alert_rows`` [alerts, len(ALERT_COLUMNS)], every alert in
            ``alerts``' order with NaN for an open alert's step_off and
            t_off; ``alert_step`` and ``alert_t``, the last update's step
            and time (NaN before any); ``alert_readings`` [classes,
            sensors]. All float64 and new arrays.
        """
        sensor_index = {name: s for s, name in enumerate(self.names)}
        rows = [
            [
                alert.step_on,
                alert.t_on,
                np.nan if alert.step_off is None else alert.step_off,
                np.nan if alert.t_off is None else alert.t_off,
                alert.size_class,
                sensor_index[alert.sensor],
                alert.reading_on,
                alert.peak,
                alert.peak_t,
            ]
            for alert in self.alerts
        ]
        table = np.array(rows, dtype=np.float64).reshape(len(rows), len(ALERT_COLUMNS))
        return {
            "alert_rows": table,
            "alert_step": np.array(np.nan if self._step is None else float(self._step)),
            "alert_t": np.array(np.nan if self._t is None else self._t),
            "alert_readings": self._readings.copy(),
        }

    def restore(self, state: Mapping[str, np.ndarray]) -> None:
        """Continue from ``state()`` as a checkpoint carried it.

        Parameters
        ----------
        state : Mapping[str, np.ndarray]
            The checkpoint's carried arrays; the names of ``STATE_NAMES``
            are read and any others ignored (the recovery tracker's).

        Raises
        ------
        KeyError
            If one of ``STATE_NAMES`` is missing: a checkpoint written
            before the monitor existed, or by a run without sensors.
        ValueError
            If an array's shape is not this monitor's; a row's class or
            sensor is not one of this monitor's; a step is not a
            non-negative whole number or a time not finite; an open row's
            step_off and t_off are not both NaN, or a closed row closes at
            or before it opened; two alerts of one sensor and class are
            open; or the last update's step and time are not both NaN or
            both finite, at or after every row's steps.
        """
        missing = [name for name in STATE_NAMES if name not in state]
        if missing:
            raise KeyError(
                f"the checkpoint carries no {missing}: it was written before the "
                "alert monitor (ADR-015 step 2) or by a run without sensors"
            )
        rows = np.array(state["alert_rows"], dtype=np.float64)
        step = np.array(state["alert_step"], dtype=np.float64)
        t = np.array(state["alert_t"], dtype=np.float64)
        readings = np.array(state["alert_readings"], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(ALERT_COLUMNS):
            raise ValueError(
                f"the checkpoint's alert_rows has shape {rows.shape}; expected "
                f"[alerts, {len(ALERT_COLUMNS)}]"
            )
        if step.shape != () or t.shape != ():
            raise ValueError("the checkpoint's alert_step and alert_t must be scalars")
        if readings.shape != self._readings.shape:
            raise ValueError(
                f"the checkpoint's alert_readings has shape {readings.shape}; this "
                f"monitor's is {self._readings.shape}"
            )
        n_classes, n_sensors = self._readings.shape
        closed: list[Alert] = []
        opened: dict[tuple[int, int], _Open] = {}
        latest = -1.0
        for row in rows:
            if not (_whole(row[4], n_classes) and _whole(row[5], n_sensors)):
                raise ValueError(
                    f"an alert row names class {row[4]} and sensor {row[5]}; this "
                    f"monitor has {n_classes} classes and {n_sensors} sensors"
                )
            k, s = int(row[4]), int(row[5])
            if not _whole(row[0]) or not np.isfinite(row[[1, 6, 7, 8]]).all():
                raise ValueError(
                    f"an alert row's step_on {row[0]} is not a non-negative whole "
                    "number, or its t_on, reading_on, peak or peak_t is not finite"
                )
            record = _Open(int(row[0]), float(row[1]), *map(float, row[6:9]))
            if np.isnan(row[2]) != np.isnan(row[3]):
                raise ValueError(
                    "an alert row has one of step_off and t_off and not the other"
                )
            if not np.isnan(row[2]) and not (
                _whole(row[2])
                and row[2] > row[0]
                and np.isfinite(row[3])
                and row[3] > row[1]
            ):
                raise ValueError(
                    f"an alert row closes at step {row[2]}, t = {row[3]} s, not "
                    f"after it opened at step {row[0]}, t = {row[1]} s"
                )
            latest = max(latest, row[0], -1.0 if np.isnan(row[2]) else row[2])
            if np.isnan(row[2]):
                if (k, s) in opened:
                    raise ValueError(
                        f"two open alerts of sensor {self.names[s]} and class {k}"
                    )
                opened[(k, s)] = record
            else:
                closed.append(self._alert((k, s), record, int(row[2]), float(row[3])))
        if np.isnan(step) != np.isnan(t) or not (
            np.isnan(step) or (_whole(step) and np.isfinite(t) and step >= latest)
        ):
            raise ValueError(
                f"the checkpoint's last update (step {step}, t = {t} s) is not both "
                "NaN or a whole step and a finite time at or after every alert's"
            )
        if np.isnan(step) and rows.shape[0]:
            raise ValueError("the checkpoint holds alerts but no last update")
        self._closed = closed
        self._open = opened
        self._step = None if np.isnan(step) else int(step)
        self._t = None if np.isnan(t) else float(t)
        self._readings = readings


class ArrivalTracker:
    """Per event, class and cell, the first time at or above threshold (ADR-015 D).

    Parameters
    ----------
    thresholds : Sequence[float]
        Per class, particles per cubic metre, finite and non-negative.
    live : np.ndarray
        [ny, nx] bool, the non-SOLID cells.
    onsets : Sequence[float]
        The contaminating events' onsets, seconds, finite, non-negative
        and in order (``contaminating_events``); at least one.

    Raises
    ------
    TypeError
        If a threshold or an onset is not a real number, or ``live`` is
        not a bool array.
    ValueError
        With no class or no onset, a threshold not finite and
        non-negative, ``live`` not 2-D, or the onsets not finite,
        non-negative and in order.

    Notes
    -----
    ``first_above[e, k, j, i]`` is the first update's time at or after
    onset e at which class k's field in cell (j, i) is at or above its
    threshold, NaN until then. Each event's map starts at its own onset, so
    a cell already at or above its threshold then reads the onset itself,
    and a cell reached for one event before a later event's onset is not
    counted for the later one until it is at or above at or after that
    onset. "At or above" is the monitor's opening rule and the recovery
    field's "not below". SOLID cells stay NaN. The arrays are the whole
    state (``state`` and ``restore``), so a restart continues the maps
    bitwise.
    """

    def __init__(
        self,
        thresholds: Sequence[float],
        live: np.ndarray,
        onsets: Sequence[float],
    ) -> None:
        self._thresholds = checked_thresholds(thresholds)
        if self._thresholds.size == 0:
            raise ValueError("thresholds must be [classes], at least one, got none")
        if not isinstance(live, np.ndarray) or live.dtype != np.bool_:
            raise TypeError("live must be a bool array, the non-SOLID cells")
        if live.ndim != 2:
            raise ValueError(f"live must be [ny, nx], got shape {live.shape}")
        self._live = live.copy()
        self._onsets = checked_onsets(onsets)
        if self._onsets.size == 0:
            raise ValueError(
                "the arrival maps need at least one event; a plan with no "
                f"contaminating event has {RUN_START} at t = 0"
            )
        shape = (self._onsets.size, self._thresholds.size, *live.shape)
        self._first = np.full(shape, np.nan)
        self._t: float | None = None

    @property
    def onsets(self) -> np.ndarray:
        """[events], the events' onsets, seconds (a copy)."""
        return self._onsets.copy()

    @property
    def first_above(self) -> np.ndarray:
        """[events, classes, ny, nx], the first time at or above, NaN until then (a copy)."""
        return self._first.copy()

    def update(self, t: float, fields: Mapping[int, np.ndarray]) -> None:
        """Read the fields after a step (or the initial fields at t = 0).

        Parameters
        ----------
        t : float
            The step's time, seconds, finite and above the last update's.
        fields : Mapping[int, np.ndarray]
            Class to its field, [ny, nx], for every class 0 to n - 1. Read,
            never written.

        Raises
        ------
        TypeError
            If ``t`` is not a real number.
        ValueError
            If ``t`` is not finite or not above the last update's, or the
            classes are not 0 to n - 1 on the mesh's shape.
        """
        t = real_number("t", t)
        if not math.isfinite(t):
            raise ValueError(f"t must be finite, got {t}")
        if self._t is not None and t <= self._t:
            raise ValueError(
                f"t = {t!r} s is not above the last update's {self._t!r} s"
            )
        _check_fields(fields, self._thresholds.size, self._live.shape)
        started = np.flatnonzero(self._onsets <= t)
        if started.size:
            for k, threshold in enumerate(self._thresholds):
                above = (np.asarray(fields[k]) >= threshold) & self._live
                for e in started:
                    first = self._first[e, k]
                    first[above & np.isnan(first)] = t
        self._t = t

    def state(self) -> dict[str, np.ndarray]:
        """The tracker's state, for the march's checkpoint.

        Returns
        -------
        dict[str, np.ndarray]
            ``ARRIVAL_STATE_NAMES``: ``first_above`` [events, classes, ny,
            nx], ``arrival_onsets`` [events] and ``arrival_t``, the last
            update's time (NaN before any). All float64 and new arrays.
        """
        return {
            "first_above": self._first.copy(),
            "arrival_onsets": self._onsets.copy(),
            "arrival_t": np.array(np.nan if self._t is None else self._t),
        }

    def restore(self, state: Mapping[str, np.ndarray]) -> None:
        """Continue from ``state()`` as a checkpoint carried it.

        Parameters
        ----------
        state : Mapping[str, np.ndarray]
            The checkpoint's carried arrays; the names of
            ``ARRIVAL_STATE_NAMES`` are read and any others ignored.

        Raises
        ------
        KeyError
            If one of ``ARRIVAL_STATE_NAMES`` is missing: a checkpoint
            written before the arrival maps (ADR-015 step 3).
        ValueError
            If ``first_above`` is not this tracker's shape, the onsets are
            not this tracker's, ``arrival_t`` is not a scalar that is NaN or
            finite, or a time in the maps is set before any update, in a
            SOLID cell, before its event's onset or after ``arrival_t``.
        """
        missing = [name for name in ARRIVAL_STATE_NAMES if name not in state]
        if missing:
            raise KeyError(
                f"the checkpoint carries no {missing}: it was written before the "
                "arrival maps (ADR-015 step 3)"
            )
        first = np.array(state["first_above"], dtype=np.float64)
        onsets = np.array(state["arrival_onsets"], dtype=np.float64)
        t = np.array(state["arrival_t"], dtype=np.float64)
        if first.shape != self._first.shape:
            raise ValueError(
                f"the checkpoint's first_above has shape {first.shape}; this run's "
                f"is {self._first.shape}"
            )
        if onsets.shape != self._onsets.shape or (
            onsets.tobytes() != self._onsets.tobytes()
        ):
            raise ValueError(
                f"the checkpoint's onsets {onsets.tolist()} are not this run's "
                f"{self._onsets.tolist()}"
            )
        if t.shape != () or np.isinf(t):
            raise ValueError(
                "the checkpoint's arrival_t must be a NaN or finite scalar"
            )
        reached = ~np.isnan(first)
        if np.isnan(t) and reached.any():
            raise ValueError("the checkpoint's maps hold times but no last update")
        if (reached & ~self._live).any():
            raise ValueError("the checkpoint's maps hold a time in a SOLID cell")
        lower = np.broadcast_to(self._onsets[:, None, None, None], first.shape)
        if not ((first[reached] >= lower[reached]) & (first[reached] <= t)).all():
            raise ValueError(
                "the checkpoint's maps hold a time before its event's onset or after "
                "the last update"
            )
        self._first = first
        self._t = None if np.isnan(t) else float(t)

    def record(self, event_names: Sequence[str], t_final: float) -> ArrivalRecord:
        """The maps as the run file's ``/arrival/`` holds them.

        Parameters
        ----------
        event_names : Sequence[str]
            One per event, in the onsets' order.
        t_final : float
            The run's last step time, seconds, at or after the last
            update's.

        Returns
        -------
        ArrivalRecord
            ``first_above`` with inf in a non-SOLID cell not reached by
            the last update, NaN in SOLID cells, as ``local_recovery`` is
            written; new arrays.

        Raises
        ------
        TypeError
            If ``t_final`` is not a real number.
        ValueError
            If the names are not one per event, or ``t_final`` is not
            finite or is before the last update.
        """
        names = tuple(event_names)
        if len(names) != self._onsets.size:
            raise ValueError(f"{len(names)} event names for {self._onsets.size} events")
        t_final = real_number("t_final", t_final)
        if not math.isfinite(t_final) or (self._t is not None and t_final < self._t):
            raise ValueError(
                f"t_final = {t_final!r} s is not finite and at or after the last "
                f"update's {self._t!r} s"
            )
        maps = self._first.copy()
        live = np.broadcast_to(self._live, maps.shape)
        maps[live & np.isnan(maps)] = np.inf
        return ArrivalRecord(
            thresholds=self._thresholds.copy(),
            event_names=names,
            onsets=self._onsets.copy(),
            first_above=maps,
            t_final=t_final,
        )
