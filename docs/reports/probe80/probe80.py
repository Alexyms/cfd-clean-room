"""Prompt 80: arrival maps under ISO 14644-1's cumulative counting, beside the per-class ones.

The model's five classes are single sizes. Each is read as standing for the particles
nearest its size, so a counter's channel k (the count at or above class k's size) is the
sum of class k and every larger class, and it is compared with class k's threshold. The
per-class reading, the monitor's and the arrival maps' today, is class k alone against the
same threshold. One code path makes both: ``channel`` sums the classes ``members`` names,
which for the per-class reading is class k alone.

Two uses:

    python docs/reports/probe80/probe80.py run SCENARIO --out RUN.h5 [run_scenario's options]

runs ``scripts/run_scenario.py``'s ``main`` in this process after one change: the march's
``ArrivalTracker.update`` is wrapped so that, after the tracker's own update, two
``ChannelTracker`` instances (per-class and cumulative) read the same step's fields. The
wrapper reads the fields and writes nothing the march reads, so the run is the runner's;
the calibration in tables80.py compares the run file with prompt 79's, bit for bit. At exit
the maps go to ``RUN.channels.npz`` beside the run file, with the tracker's own map, so the
per-class reading can be checked against it in the process that made both.

``frame_maps`` builds both readings' maps from a run file's stored frames, reading one
frame of one class at a time, never a run's whole field history.

The planted defect of the structural check (section 3) is the environment variable
``PROBE80_PLANT=smaller``: the cumulative channel sums class k and the smaller classes
instead of class k and the larger ones.
"""

import atexit
import math
import os
import runpy
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.io_manager import checked_onsets, checked_thresholds  # noqa: E402
from src.monitor import ArrivalTracker  # noqa: E402

PER_CLASS = "per_class"
CUMULATIVE = "cumulative"
COUNTINGS = (PER_CLASS, CUMULATIVE)
PLANT_ENV = "PROBE80_PLANT"
PLANT_SMALLER = "smaller"


def members(k: int, n: int, counting: str) -> tuple[int, ...]:
    """The classes channel k sums, largest first.

    Parameters
    ----------
    k : int
        The channel, the class index of its lower size edge (0 the smallest).
    n : int
        The number of classes.
    counting : str
        ``PER_CLASS`` (class k alone) or ``CUMULATIVE`` (class k and every
        larger class).

    Returns
    -------
    tuple[int, ...]
        Class indices in the order they are added.
    """
    if not 0 <= k < n:
        raise ValueError(f"channel {k} is not one of {n} classes")
    if counting == PER_CLASS:
        return (k,)
    if counting != CUMULATIVE:
        raise ValueError(f"counting must be one of {COUNTINGS}, got {counting!r}")
    if os.environ.get(PLANT_ENV) == PLANT_SMALLER:
        return tuple(range(k, -1, -1))
    return tuple(range(n - 1, k - 1, -1))


def channel(fields: Mapping[int, np.ndarray], k: int, counting: str) -> np.ndarray:
    """Channel k's reading of every cell: the sum of the classes ``members`` names.

    Parameters
    ----------
    fields : Mapping[int, np.ndarray]
        Class to its concentration, particles per cubic metre, classes 0 to
        n - 1, all one shape.
    k : int
        The channel.
    counting : str
        ``PER_CLASS`` or ``CUMULATIVE``.

    Returns
    -------
    np.ndarray
        A new float64 array of the fields' shape. Adding from zero keeps the
        per-class reading bitwise the class's own field.
    """
    out = np.zeros(np.shape(fields[0]), dtype=np.float64)
    for j in members(k, len(fields), counting):
        out += fields[j]
    return out


class ChannelTracker:
    """Per event, channel and cell, the first update at or after onset at or above threshold.

    ``monitor.ArrivalTracker``'s rule with the class's field replaced by the
    channel's reading (``channel``): NaN until reached, an event's map
    starting at its own onset, "at or above" counting.

    Parameters
    ----------
    thresholds : Sequence[float]
        Per channel, particles per cubic metre: class k's threshold.
    live : np.ndarray
        Bool, the cells read (the non-SOLID cells, or a [1, sensors] row).
    onsets : Sequence[float]
        The contaminating events' onsets, seconds, in order.
    counting : str
        ``PER_CLASS`` or ``CUMULATIVE``.
    """

    def __init__(
        self,
        thresholds: Sequence[float] | np.ndarray,
        live: np.ndarray,
        onsets: Sequence[float] | np.ndarray,
        counting: str,
    ) -> None:
        self.thresholds = checked_thresholds(thresholds)
        self.live = np.asarray(live, dtype=bool).copy()
        self.onsets = checked_onsets(onsets)
        self.counting = counting
        members(0, self.thresholds.size, counting)
        self.first = np.full(
            (self.onsets.size, self.thresholds.size, *self.live.shape), np.nan
        )
        self.t: float | None = None
        self.low = math.inf

    def update(self, t: float, fields: Mapping[int, np.ndarray]) -> None:
        """Read one step's (or one frame's) fields at time t, above the last."""
        if self.t is not None and not t > self.t:
            raise ValueError(f"t = {t!r} is not above the last update's {self.t!r}")
        started = np.flatnonzero(self.onsets <= t)
        if started.size:
            for k, threshold in enumerate(self.thresholds):
                reading = channel(fields, k, self.counting)
                above = (reading >= threshold) & self.live
                for e in started:
                    first = self.first[e, k]
                    first[above & np.isnan(first)] = t
        for field in fields.values():
            self.low = min(self.low, float(np.min(np.asarray(field)[self.live])))
        self.t = float(t)

    def maps(self) -> np.ndarray:
        """[events, channels, *shape]: inf where never reached, NaN off ``live``."""
        out = np.where(np.isnan(self.first), np.inf, self.first)
        out[:, :, ~self.live] = np.nan
        return out


def frame_maps(
    path: Path,
    thresholds: Sequence[float] | np.ndarray,
    live: np.ndarray,
    onsets: Sequence[float] | np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Both readings' arrival maps from a run file's stored frames.

    Parameters
    ----------
    path : Path
        A run file with ``/transport/`` frames.
    thresholds, live, onsets
        As ``ChannelTracker`` takes them; ``live`` [ny, nx].

    Returns
    -------
    dict[str, np.ndarray]
        Counting to its maps [events, channels, ny, nx]: the first frame time
        at or after the onset at or above threshold, inf never, NaN SOLID.
    np.ndarray
        The frame times, seconds.
    float
        The least concentration of any class in any live cell at any frame.

    Notes
    -----
    One frame of one class is read at a time (the datasets are chunked by
    frame); one frame's classes are held together for the sums.
    """
    trackers = {c: ChannelTracker(thresholds, live, onsets, c) for c in COUNTINGS}
    with h5py.File(path, "r") as run_file:
        group = run_file["transport"]
        times = group["times"][()]
        names = sorted(
            (int(name[2:]) for name in group if name.startswith("C_")),
        )
        if names != list(range(len(names))):
            raise ValueError(f"{path}: classes {names}")
        for f, t in enumerate(times):
            fields = {k: group[f"C_{k}"][f] for k in names}
            for tracker in trackers.values():
                tracker.update(float(t), fields)
    low = min(tracker.low for tracker in trackers.values())
    return {c: tracker.maps() for c, tracker in trackers.items()}, times, low


def reading_latencies(
    readings: np.ndarray,
    times: np.ndarray,
    thresholds: Sequence[float] | np.ndarray,
    onsets: Sequence[float] | np.ndarray,
    counting: str,
) -> np.ndarray:
    """Per event, channel and sensor, the latency of every-step sensor readings.

    Parameters
    ----------
    readings : np.ndarray
        [steps, classes, sensors], ``/transport/sensors/readings``.
    times : np.ndarray
        [steps], seconds.
    thresholds, onsets
        As ``ChannelTracker`` takes them.
    counting : str
        ``PER_CLASS`` or ``CUMULATIVE``.

    Returns
    -------
    np.ndarray
        [events, channels, sensors]: first time at or above minus the onset,
        inf for never.
    """
    sensors = readings.shape[2]
    tracker = ChannelTracker(thresholds, np.ones((1, sensors), bool), onsets, counting)
    for s, t in enumerate(times):
        fields = {k: readings[s, k][None, :] for k in range(readings.shape[1])}
        tracker.update(float(t), fields)
    return tracker.maps()[:, :, 0, :] - tracker.onsets[:, None, None]


# -- The step probe ------------------------------------------------------------

PROBES: dict[str, ChannelTracker] = {}
OWN: list[ArrivalTracker] = []


def install() -> None:
    """Wrap ArrivalTracker.update so both readings follow the march's every step."""
    original = ArrivalTracker.update

    def update(
        self: ArrivalTracker, t: float, fields: Mapping[int, np.ndarray]
    ) -> None:
        original(self, t, fields)
        if not OWN:
            OWN.append(self)
            for counting in COUNTINGS:
                PROBES[counting] = ChannelTracker(
                    self._thresholds, self._live, self._onsets, counting
                )
        if OWN[0] is not self:
            raise RuntimeError("probe80 follows one arrival tracker per process")
        for tracker in PROBES.values():
            tracker.update(t, fields)

    ArrivalTracker.update = update


def save(path: Path) -> None:
    """Write both readings' step maps and the tracker's own beside the run file."""
    if not OWN:
        return
    own = OWN[0]
    np.savez(
        path.with_name(f"{path.stem}.channels.npz"),
        own=own.first_above,
        per_class=PROBES[PER_CLASS].first,
        cumulative=PROBES[CUMULATIVE].first,
        thresholds=PROBES[PER_CLASS].thresholds,
        onsets=PROBES[PER_CLASS].onsets,
        t_final=np.array(PROBES[PER_CLASS].t),
        low=np.array(min(p.low for p in PROBES.values())),
        plant=np.array(os.environ.get(PLANT_ENV, "")),
    )


def main() -> None:
    """``run SCENARIO --out RUN.h5 ...``: run_scenario.py's main under the probe."""
    if len(sys.argv) < 2 or sys.argv[1] != "run":
        raise SystemExit(__doc__)
    argv = sys.argv[2:]
    if "--out" not in argv:
        raise SystemExit("probe80: give --out (a new run; restarts are not probed)")
    path = Path(argv[argv.index("--out") + 1])
    install()
    atexit.register(save, path)
    sys.argv = [str(ROOT / "scripts" / "run_scenario.py"), *argv]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
