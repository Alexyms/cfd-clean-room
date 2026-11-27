"""The alert monitor: each sensor's reading every step and the alerts it raises (ADR-015 C).

A sensor reads the cell containing it, the one below or to the left of it
when it sits on a face (``sensor_cells``, ADR-014 C's rule, moved here from
``time_integration``). The reading is instantaneous: the cell's value after
the step, the reading the clean-out's record already uses, so Phase 4's
recovery times and Phase 5's latencies come from one instrument.

The channels (ADR-015 C, amended 2026-11-14 by Alex's decision on prompt
80's measurement). ISO 14644-1's limits are cumulative and a real counter's
channel at a size counts every particle at or above it, so the monitor reads
what a counter reads: channel k is the sum of class k and every larger class
(``SimConfig.channel_classes``, summed by ``channel_sums``), compared with
class k's threshold. Each of the model's classes stands for the particles
nearest its size. There is no per-class comparison beside it: every reader
of a threshold in ``src/`` reads the channels.

An alert opens at the first step whose channel reading is at or above the
channel's threshold and closes at the first step after it whose reading is
below; it records both steps and times, the opening reading and the peak
between them. Each class's own reading is kept beside the channels', for
the run file. The march calls ``update`` after every transport step with fields it
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

The arrival maps (ADR-015 D, step 3). Per contaminating event, channel and
non-SOLID cell, ``ArrivalTracker`` keeps the first step time at or after the
event's onset at which the cell's channel reading is at or above the
channel's threshold: the
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
    CUMULATIVE,
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
# The checkpoint's mark that its maps, alerts and recovery readings were made
# on the channels (prompt 81). A checkpoint without it was written by the
# per-class code, and continuing it would join two countings in one run.
COUNTING_STATE = "cumulative_channels"
# The names ``ArrivalTracker.state()`` carries, distinct from the recovery
# tracker's and the monitor's in the one checkpoint mapping.
ARRIVAL_STATE_NAMES: tuple[str, ...] = (
    "first_above",
    "arrival_onsets",
    "arrival_t",
    COUNTING_STATE,
)
# The one event of a run whose plan has no contaminating event (ADR-015 D).
RUN_START = "run_start"
# The segment key that gives a boundary segment a concentration of its own.
CONCENTRATION_KEY = "concentration"


@dataclass(frozen=True)
class Alert:
    """One alert: a sensor's reading of one channel at or above its threshold.

    Parameters
    ----------
    sensor : str
        The sensor's name.
    size_class : int
        The channel, named by the class of its lower size edge: channel k
        sums class k and every larger class (ADR-015 C).
    step_on, t_on : int, float
        The step that opened it and its time, seconds: the first step of
        the stretch whose channel reading is at or above the threshold.
    reading_on : float
        That step's channel reading, particles per cubic metre.
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


def checked_channels(
    channels: Sequence[Sequence[int]], n_classes: int
) -> tuple[tuple[int, ...], ...]:
    """The channels' classes (``SimConfig.channel_classes``), or a refusal.

    Parameters
    ----------
    channels : Sequence[Sequence[int]]
        Per channel k, the classes it sums in the order they are added.
    n_classes : int
        The classes the fields hold.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        The channels as tuples of ints.

    Raises
    ------
    TypeError
        If a member is not an int (a bool is not one here).
    ValueError
        If there is not one channel per class, or a channel is empty, names
        a class outside 0 to n - 1 or twice, or leaves out its own class.
    """
    out = tuple(tuple(members) for members in channels)
    if len(out) != n_classes:
        raise ValueError(f"{len(out)} channels for {n_classes} classes")
    for k, members in enumerate(out):
        for j in members:
            if isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer)):
                raise TypeError(
                    f"channel {k} names {j!r}; a class index must be an int"
                )
        if (
            k not in members
            or len(set(members)) != len(members)
            or not all(0 <= j < n_classes for j in members)
        ):
            raise ValueError(
                f"channel {k} sums {list(members)}: it must name its own class and "
                f"others of 0 to {n_classes - 1}, each once"
            )
    return tuple(tuple(int(j) for j in members) for members in out)


def channel_sums(
    values: Mapping[int, np.ndarray], channels: Sequence[Sequence[int]]
) -> dict[int, np.ndarray]:
    """Each channel's reading: the sum of the classes it names (ADR-015 C).

    Parameters
    ----------
    values : Mapping[int, np.ndarray]
        Class to its concentration (a field [ny, nx], or the sensors'
        readings), particles per cubic metre, all of one shape. Read, never
        written.
    channels : Sequence[Sequence[int]]
        ``SimConfig.channel_classes``: per channel, the classes it sums in
        the order they are added.

    Returns
    -------
    dict[int, np.ndarray]
        Channel k to a new float64 array of the values' shape: zero, then
        each named class added in turn. A one-class channel is that class's
        values bitwise, and two readers given the same channels add the
        same terms in the same order, so they agree bit for bit (the sum at
        a sensor's cell equals the field sum read at that cell).
    """
    return {k: channel_sum(values, members) for k, members in enumerate(channels)}


def channel_sum(values: Mapping[int, np.ndarray], members: Sequence[int]) -> np.ndarray:
    """One channel's reading: zero, then each class of MEMBERS added in turn.

    Parameters
    ----------
    values : Mapping[int, np.ndarray]
        Class to its concentration, all of one shape; read, never written.
    members : Sequence[int]
        One channel of ``SimConfig.channel_classes``, at least one class.

    Returns
    -------
    np.ndarray
        A new float64 array of the values' shape (``channel_sums``' entry).
    """
    total = np.zeros(np.shape(values[members[0]]), dtype=np.float64)
    for j in members:
        total += values[j]
    return total


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
    """Each sensor's reading of each channel every step, and the alerts (ADR-015 C).

    Parameters
    ----------
    config : SimConfig
        Supplies the sensors, each class's threshold
        (``SimConfig.class_thresholds``) and the classes each channel sums
        (``SimConfig.channel_classes``).
    mesh : Mesh
        The run's mesh, for the cells the sensors read.

    Raises
    ------
    ValueError
        With no sensor, when a class has no threshold, or when a sensor
        reads a SOLID cell.

    Notes
    -----
    Channel k's reading is the sum of class k and every larger class at the
    sensor's cell, compared with class k's threshold: a counter's
    cumulative count against ISO 14644-1's cumulative limit. "At or above"
    opens an alert and "below" closes it, matching the recovery field's
    "not below": a reading exactly at its threshold is an exceedance. The
    fields handed to ``update`` are read by index and never written
    (REQ-A06).
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
        self._channels = checked_channels(
            config.channel_classes(), self._thresholds.shape[0]
        )
        cells = sensor_cells(config, mesh)
        self._rows = np.array([j for j, _ in cells])
        self._cols = np.array([i for _, i in cells])
        self._shape = mesh.cell_type.shape
        self.names: tuple[str, ...] = tuple(sensor.name for sensor in config.sensors)
        n_classes, n_sensors = self._thresholds.shape[0], len(cells)
        self._readings = np.full((n_classes, n_sensors), np.nan)
        self._channel_readings = np.full((n_classes, n_sensors), np.nan)
        self._closed: list[Alert] = []
        self._open: dict[tuple[int, int], _Open] = {}
        self._step: int | None = None
        self._t: float | None = None

    @property
    def thresholds(self) -> np.ndarray:
        """[channels], channel k's threshold, class k's, per cubic metre (a copy)."""
        return self._thresholds.copy()

    @property
    def channels(self) -> tuple[tuple[int, ...], ...]:
        """Per channel, the classes it sums (``SimConfig.channel_classes``)."""
        return self._channels

    @property
    def readings(self) -> np.ndarray:
        """[classes, sensors], each class's own reading at the last update (a copy).

        NaN before any update. These are the concentrations; the alerts
        read ``channel_readings``.
        """
        return self._readings.copy()

    @property
    def channel_readings(self) -> np.ndarray:
        """[channels, sensors], each channel's reading at the last update (a copy).

        The sum ``channel_sums`` forms from ``readings``; NaN before any
        update.
        """
        return self._channel_readings.copy()

    def _sums(self, readings: np.ndarray) -> np.ndarray:
        """[channels, sensors], the channels of per-class READINGS [classes, sensors]."""
        sums = channel_sums(
            {k: readings[k] for k in range(readings.shape[0])}, self._channels
        )
        return np.array([sums[k] for k in range(readings.shape[0])], dtype=np.float64)

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
            (``step_off == step``), by channel then sensor. A sensor's
            channel opens or closes at most one per step.

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
        step, t = self._checked_time(step, t)
        n_classes = self._thresholds.shape[0]
        _check_fields(fields, n_classes, self._shape)
        readings = np.array(
            [np.asarray(fields[k])[self._rows, self._cols] for k in range(n_classes)],
            dtype=np.float64,
        )
        if not np.isfinite(readings).all():
            raise ValueError(f"a reading at step {step} is not finite")
        return self._advance(step, t, readings, self._sums(readings))

    def update_from_readings(
        self, step: int, t: float, readings: np.ndarray, channels: np.ndarray
    ) -> tuple[Alert, ...]:
        """Take a step's readings as the compiled core's march returns them (ADR-017 E).

        The update from fields reads each class at the sensors' cells and sums
        the channels itself; this one is handed both, read from the fields by
        the core and summed in ``channel_sums``' order (ADR-017 F), so the march
        does not stop each step for the monitor. Given the readings and channels
        the update from fields would form, it opens and closes the same alerts
        and leaves the same state, bit for bit.

        Parameters
        ----------
        step : int
            The run's step count, non-negative and above the last update's.
        t : float
            The step's time, seconds, finite and above the last update's.
        readings : np.ndarray
            [classes, sensors], each class's concentration at each sensor's
            cell, particles per cubic metre. Copied, never written.
        channels : np.ndarray
            [channels, sensors], each channel's reading at each sensor's cell,
            the sum of its classes (``channel_sums``). Copied, never written.

        Returns
        -------
        tuple[Alert, ...]
            The alerts this step opened and closed, as ``update`` returns them.

        Raises
        ------
        TypeError
            If ``step`` is not an int (a bool is not one here) or ``t`` not a
            real number.
        ValueError
            If ``step`` or ``t`` is not above the last update's, ``t`` is not
            finite, an array is not [classes, sensors] or holds a value that is
            not finite.
        """
        step, t = self._checked_time(step, t)
        expected = self._readings.shape
        arrays = {
            "readings": np.array(readings, dtype=np.float64),
            "channels": np.array(channels, dtype=np.float64),
        }
        for name, array in arrays.items():
            if array.shape != expected:
                raise ValueError(
                    f"the {name} at step {step} have shape {array.shape}; the "
                    f"monitor's are {expected} (classes, sensors)"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"a {name[:-1]} at step {step} is not finite")
        return self._advance(step, t, arrays["readings"], arrays["channels"])

    def _checked_time(self, step: int, t: float) -> tuple[int, float]:
        """An update's step and time as int and float, or the update's refusal."""
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
        return step, t

    def _advance(
        self, step: int, t: float, readings: np.ndarray, sums: np.ndarray
    ) -> tuple[Alert, ...]:
        """Open, close and carry the alerts on a step's readings and channel SUMS."""
        n_classes = self._thresholds.shape[0]
        changed = []
        above = sums >= self._thresholds[:, None]
        for k in range(n_classes):
            for s in range(len(self.names)):
                key, reading = (k, s), float(sums[k, s])
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
        self._channel_readings = sums
        self._step, self._t = step, t
        return tuple(changed)

    def latencies(self, onsets: Sequence[float]) -> np.ndarray:
        """Each sensor's detection latency per event and channel (REQ-A04, ADR-015 D).

        Parameters
        ----------
        onsets : Sequence[float]
            The events' onsets, seconds (``contaminating_events``), finite,
            non-negative and in order; two events may share one.

        Returns
        -------
        np.ndarray
            [events, channels, sensors]: the time of the first channel
            reading at or above the threshold at or after the event's onset,
            minus the
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
            sensors], each class's own reading, from which the channels'
            are summed again on restore. All float64 and new arrays.
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
        self._channel_readings = self._sums(readings)


class ArrivalTracker:
    """Per event, channel and cell, the first time at or above threshold (ADR-015 D).

    Parameters
    ----------
    thresholds : Sequence[float]
        Per class, particles per cubic metre, finite and non-negative:
        channel k's threshold is class k's.
    live : np.ndarray
        [ny, nx] bool, the non-SOLID cells.
    onsets : Sequence[float]
        The contaminating events' onsets, seconds, finite, non-negative
        and in order (``contaminating_events``); at least one.
    channels : Sequence[Sequence[int]]
        ``SimConfig.channel_classes``: per channel, the classes it sums.

    Raises
    ------
    TypeError
        If a threshold or an onset is not a real number, ``live`` is not a
        bool array, or a channel's member not an int.
    ValueError
        With no class or no onset, a threshold not finite and
        non-negative, ``live`` not 2-D, the onsets not finite, non-negative
        and in order, or the channels not ``checked_channels``'.

    Notes
    -----
    ``first_above[e, k, j, i]`` is the first update's time at or after
    onset e at which channel k's reading in cell (j, i), the sum of class
    k and every larger class (``channel_sums``), is at or above class k's
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
        *,
        channels: Sequence[Sequence[int]],
    ) -> None:
        self._thresholds = checked_thresholds(thresholds)
        if self._thresholds.size == 0:
            raise ValueError("thresholds must be [classes], at least one, got none")
        self._channels = checked_channels(channels, self._thresholds.size)
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
        """[events, channels, ny, nx], the first time at or above, NaN until then (a copy)."""
        return self._first.copy()

    @property
    def channels(self) -> tuple[tuple[int, ...], ...]:
        """Per channel, the classes it sums."""
        return self._channels

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
            sums = channel_sums(fields, self._channels)
            for k, threshold in enumerate(self._thresholds):
                above = (sums[k] >= threshold) & self._live
                for e in started:
                    first = self._first[e, k]
                    first[above & np.isnan(first)] = t
        self._t = t

    def state(self) -> dict[str, np.ndarray]:
        """The tracker's state, for the march's checkpoint.

        Returns
        -------
        dict[str, np.ndarray]
            ``ARRIVAL_STATE_NAMES``: ``first_above`` [events, channels, ny,
            nx], ``arrival_onsets`` [events], ``arrival_t``, the last
            update's time (NaN before any), and ``COUNTING_STATE``, 1.0,
            the mark that the checkpoint was made on the channels. All
            float64 and new arrays.
        """
        return {
            "first_above": self._first.copy(),
            "arrival_onsets": self._onsets.copy(),
            "arrival_t": np.array(np.nan if self._t is None else self._t),
            COUNTING_STATE: np.array(1.0),
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
            If one of the maps' names is missing: a checkpoint written
            before the arrival maps (ADR-015 step 3).
        ValueError
            If the maps are there without ``COUNTING_STATE`` (a checkpoint
            of the per-class code, before prompt 81), ``first_above`` is not
            this tracker's shape, the onsets are not this tracker's,
            ``arrival_t`` is not a scalar that is NaN or finite, or a time
            in the maps is set before any update, in a SOLID cell, before
            its event's onset or after ``arrival_t``.
        """
        missing = [
            name
            for name in ARRIVAL_STATE_NAMES
            if name != COUNTING_STATE and name not in state
        ]
        if missing:
            raise KeyError(
                f"the checkpoint carries no {missing}: it was written before the "
                "arrival maps (ADR-015 step 3)"
            )
        if COUNTING_STATE not in state:
            raise ValueError(
                "the checkpoint's arrival maps were made per class, before the "
                "channels of prompt 81 (ADR-015 C): continuing them would join two "
                "countings in one map"
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
            written; new arrays; counting ``CUMULATIVE``.

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
            counting=CUMULATIVE,
        )
