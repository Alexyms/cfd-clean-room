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
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.config import SimConfig
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
        if sorted(fields) != list(range(n_classes)):
            raise ValueError(
                f"fields for classes {sorted(fields)}; the monitor reads classes 0 "
                f"to {n_classes - 1}"
            )
        for k in range(n_classes):
            shape = np.shape(fields[k])
            if shape != self._shape:
                raise ValueError(
                    f"class {k}'s field has shape {shape}; the mesh's is {self._shape}"
                )
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
        """Each sensor's detection latency per event and class (REQ-A04).

        Parameters
        ----------
        onsets : Sequence[float]
            The events' onsets, seconds, finite and increasing.

        Returns
        -------
        np.ndarray
            [events, classes, sensors]: the ``t_on`` of the first alert
            opened at or after the event's onset and before the next one's
            (or at any time after the last onset) minus the onset; inf when
            no alert opened in that window. An alert already open at an
            onset is not that event's detection.

        Raises
        ------
        TypeError
            If an onset is not a real number.
        ValueError
            If the onsets are not one-dimensional, finite and increasing.
        """
        if np.ndim(onsets) != 1:
            raise ValueError(f"onsets must be [events], got shape {np.shape(onsets)}")
        for onset in onsets:
            if isinstance(onset, (bool, np.bool_)) or not isinstance(
                onset, (int, float, np.integer, np.floating)
            ):
                raise TypeError(
                    f"an onset must be a real number, got {type(onset).__name__}"
                )
        times = np.asarray(onsets, dtype=np.float64)
        if not np.isfinite(times).all():
            raise ValueError("the onsets must be finite")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("the onsets must be increasing")
        n_classes, n_sensors = self._readings.shape
        out = np.full((times.size, n_classes, n_sensors), np.inf)
        ends = np.append(times[1:], np.inf)
        sensor_index = {name: s for s, name in enumerate(self.names)}
        for alert in self.alerts:
            s = sensor_index[alert.sensor]
            inside = (times <= alert.t_on) & (alert.t_on < ends)
            for e in np.flatnonzero(inside):
                delay = alert.t_on - times[e]
                if delay < out[e, alert.size_class, s]:
                    out[e, alert.size_class, s] = delay
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
            "alert_step": np.array(np.nan if self._step is None else self._step),
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
            If one of ``STATE_NAMES`` is missing.
        ValueError
            If an array's shape is not this monitor's, a row's class or
            sensor is not one of this monitor's, an open row's step_off and
            t_off are not both NaN, or two alerts of one sensor and class
            are open.
        """
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
        for row in rows:
            k, s = int(row[4]), int(row[5])
            if (
                row[4] != k
                or row[5] != s
                or not (0 <= k < n_classes and 0 <= s < n_sensors)
            ):
                raise ValueError(
                    f"an alert row names class {row[4]} and sensor {row[5]}; this "
                    f"monitor has {n_classes} classes and {n_sensors} sensors"
                )
            record = _Open(int(row[0]), float(row[1]), *map(float, row[6:9]))
            if np.isnan(row[2]) != np.isnan(row[3]):
                raise ValueError(
                    "an alert row has one of step_off and t_off and not the other"
                )
            if np.isnan(row[2]):
                if (k, s) in opened:
                    raise ValueError(
                        f"two open alerts of sensor {self.names[s]} and class {k}"
                    )
                opened[(k, s)] = record
            else:
                closed.append(self._alert((k, s), record, int(row[2]), float(row[3])))
        self._closed = closed
        self._open = opened
        self._step = None if np.isnan(step) else int(step)
        self._t = None if np.isnan(t) else float(t)
        self._readings = readings
