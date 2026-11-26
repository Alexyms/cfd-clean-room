#!/usr/bin/env python
"""Compare a scenario run with the reference's run of it: ADR-017 H's verdict (REQ-N03 level 2).

Run from the repository root with the project's interpreter, where the run
files' scenario bases resolve (the machine the runs were made on):

    python scripts/compare_runs.py REFERENCE.h5 OTHER.h5
    python scripts/compare_runs.py REFERENCE.h5 OTHER.h5 --json out.json

Two run files of one scenario in; ADR-017 H's verdict out, item by item. The
items are its three criteria:

1. *every decision equal*: each arrival time of the maps (per event, channel
   and cell), each alert's opening and closing step at the file's sensors and
   so each sensor's latency (``AlertMonitor.latencies`` on the table), each
   clear time of the all-clear units at its frame (``placement.clear_units``),
   the clean-out's recovery times, the run's end (steps, time, the hold), and
   every class's verdict against REQ-T05; a time moves by its difference over
   the step of the reference's run at the later of the two, a clear time by
   its difference over the unit's frame spacing, and a time reached in one run
   only is a move of infinity;
2. *every field inside its limit*, read in the three tiers: each flow field of
   each flow state, the class fields over every stored frame (and pointwise
   where the reference reads at least a hundredth of the class's threshold),
   the sensor channels at every step, and the change in the accumulated
   leftover over the supply at the end, each as its largest difference over
   the reference's largest magnitude; and the budget closing to 1e-12 of the
   supply on every stored frame, not tiered;
3. *every stop within one*: each flow state's outer count and stop reason.

A field between its measured spread and its limit passes only when the stop
counts explain it (ADR-017 H): here, when some flow state's outer count differs
by one and none by more, the kind of change that moved the probe's fields that
far. The CG and Jacobi counts are not in a run file, so no class field's middle
tier is explained by them; such a field is read as beyond the limit, and goes to
the step's report.

``compare_placement`` gives the decisions that read several scenarios at once:
the searched layouts' cells under each rule and pool, their order, and every
unit's latency of the file's four sensors and of each searched layout.

The measures and their tests come from prompt 85's probe
(``docs/reports/probe85/tables85.py``), with what test 85 S1 found unpinned
pinned in ``tests/test_compare_runs.py``: the signs, a move that straddles a
threshold, a field in each tier and a failing verdict of each kind. Prompt 96
built it; step 6b scores 320x120 with it.
"""

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402 -- follows sys.path.insert
    MEAN,
    MOUNT_FREE,
    MOUNT_STAND,
    WORST_CASE,
    SimConfig,
)
from src.io_manager import OPEN_STEP, RunFile  # noqa: E402 -- follows sys.path.insert
from src.mesh import Mesh  # noqa: E402 -- follows sys.path.insert
from src.monitor import ALERT_COLUMNS, AlertMonitor  # noqa: E402
from src.placement import (  # noqa: E402 -- follows sys.path.insert
    RunArrival,
    arrival_from_run,
    candidates,
    clear_units,
    detection_units,
    given_layout,
    rank,
    room_clear,
    scenario_end,
    search,
)
from src.solver_staggered import FlowState  # noqa: E402 -- follows sys.path.insert

# ADR-017 H: (the measured spread, the limit) per field. The spreads are the
# largest prompt 85's probe measured (its P8; the sensor channels P2's, as
# ADR-017 H takes them); the limits ten times them rounded up to a decade.
FLOW_FIELDS: tuple[str, ...] = ("u", "v", "p", "k", "eps", "nu_t")
FLOW_SPREAD, FLOW_LIMIT = 7.8e-10, 1.0e-8
CONCENTRATION = "concentration"
POINTWISE = "concentration, pointwise"
SENSORS = "sensor channels"
LEFTOVER = "leftover / supplied, change"
BAND: dict[str, tuple[float, float]] = {
    **{name: (FLOW_SPREAD, FLOW_LIMIT) for name in FLOW_FIELDS},
    CONCENTRATION: (2.7e-8, 1.0e-6),
    POINTWISE: (5.8e-8, 1.0e-6),
    SENSORS: (4.7e-8, 1.0e-6),
    LEFTOVER: (3.6e-10, 1.0e-8),
}
# The budget's closure, the reference's own check, not tiered.
CLOSURE = 1.0e-12
# REQ-T05: the budget's residual and the accumulated leftover each below this
# fraction of the particles supplied.
REQ_T05 = 1.0e-4
# The pointwise measure reads cells at or above this fraction of the threshold,
# the decades the alerts and the maps read.
POINTWISE_FLOOR = 0.01
INSIDE, BETWEEN, BEYOND = (
    "inside the spread",
    "between the spread and the limit",
    "beyond the limit",
)
# The kinds of a failing verdict.
DECISION, FIELD, STOP, CLOSED, BUDGET = (
    "a decision differs",
    "a field outside its limit",
    "a stop more than one apart",
    "a budget that does not close",
    "a budget verdict differs",
)
RULES = (WORST_CASE, MEAN)
FILE_LAYOUT, SURFACE, STAND, FREE = "the file's four", "surface", "product", "free"


# ---------------------------------------------------------------------------
# The measures


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """The largest |b - a| over the largest |a|; infinity where the two disagree in kind.

    Parameters
    ----------
    a, b : np.ndarray
        The reference's values and the other run's, of one shape.

    Returns
    -------
    float
        0 when equal (NaN where both are NaN, an infinity where both are the
        same one); infinity when the shapes differ, one is finite where the
        other is not, the two infinities differ, or a is zero everywhere and b
        is not; otherwise max |b - a| over max |a| on the finite entries.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return math.inf
    finite = np.isfinite(a)
    if not np.array_equal(finite, np.isfinite(b)):
        return math.inf
    odd = ~finite
    if not np.array_equal(a[odd], b[odd], equal_nan=True):
        return math.inf
    if not finite.any():
        return 0.0
    diff = float(np.abs(b[finite] - a[finite]).max())
    if diff == 0.0:
        return 0.0
    scale = float(np.abs(a[finite]).max())
    return diff / scale if scale > 0.0 else math.inf


def pointwise(a: np.ndarray, b: np.ndarray, floor: float) -> float:
    """The largest |b - a| / |a| over the entries where |a| is at least FLOOR.

    Infinity when the shapes differ or b is not finite where a is read.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return math.inf
    keep = np.isfinite(a) & (np.abs(a) >= floor)
    if not keep.any():
        return 0.0
    if not np.isfinite(b[keep]).all():
        return math.inf
    return float((np.abs(b[keep] - a[keep]) / np.abs(a[keep])).max())


def tier(value: float, name: str) -> str:
    """Which of ADR-017 H's three tiers VALUE falls in for the field NAME."""
    spread, limit = BAND[name]
    if value <= spread:
        return INSIDE
    if value <= limit:
        return BETWEEN
    return BEYOND


def step_of(times: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The step that ends at each time T: its length in the run whose step TIMES are given.

    Parameters
    ----------
    times : np.ndarray
        [n] the run's step times, increasing, its first the time before the
        first step (0, or a restart's checkpoint).
    t : np.ndarray
        Times to place, any shape; each in (times[0], times[-1]] reads the
        step from the time before it to the first step time at or after it, a
        time at or before times[0] the first step's.

    Returns
    -------
    np.ndarray
        The steps' lengths, of t's shape.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.size < 2:
        raise ValueError("a run's step times need at least two entries")
    lengths = np.diff(times)
    index = np.searchsorted(times, np.asarray(t, dtype=np.float64), side="left") - 1
    return lengths[np.clip(index, 0, lengths.size - 1)]


def moved(
    ref: np.ndarray, other: np.ndarray, spacing: np.ndarray | float
) -> np.ndarray:
    """How far each time moved, in units of SPACING.

    Parameters
    ----------
    ref, other : np.ndarray
        Times of one shape: finite, inf for never, NaN where there is none
        (a SOLID cell).
    spacing : np.ndarray or float
        Each time's unit, broadcast to the shape.

    Returns
    -------
    np.ndarray
        |other - ref| / spacing where both are finite; 0 where both are never;
        infinity where one is never and the other a time (a move across the
        threshold that one run reaches and the other does not); NaN where both
        are NaN; infinity where one is NaN and the other is not.
    """
    ref = np.asarray(ref, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    if ref.shape != other.shape:
        raise ValueError(f"times of shapes {ref.shape} and {other.shape}")
    unit = np.broadcast_to(np.asarray(spacing, dtype=np.float64), ref.shape)
    out = np.full(ref.shape, math.inf)
    both = np.isfinite(ref) & np.isfinite(other)
    out[both] = np.abs(other[both] - ref[both]) / unit[both]
    out[np.isinf(ref) & np.isinf(other)] = 0.0
    out[np.isnan(ref) & np.isnan(other)] = math.nan
    return out


def step_moves(ref: np.ndarray, other: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Times moved in steps of the reference's run at the later of the two (``moved``)."""
    ref = np.asarray(ref, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    later = np.where(np.isfinite(other) & (other > ref), other, ref)
    later = np.where(np.isfinite(later), later, times[-1])
    return moved(ref, other, step_of(times, later))


@dataclass(frozen=True)
class Moves:
    """How a set of decisions moved.

    Parameters
    ----------
    compared : int
        The decisions compared (NaN entries left out).
    one, more : int
        Moved by one unit, and by more, rounded to the nearest.
    only : int
        Reached in one run only.
    largest : float
        The largest finite move.
    """

    compared: int
    one: int
    more: int
    only: int
    largest: float

    @property
    def equal(self) -> bool:
        """Whether no decision moved."""
        return self.one == 0 and self.more == 0 and self.only == 0

    @classmethod
    def of(cls, steps: np.ndarray) -> "Moves":
        """The counts of an array of moves (``moved``'s units)."""
        s = np.asarray(steps, dtype=np.float64).ravel()
        s = s[~np.isnan(s)]
        finite = s[np.isfinite(s)]
        whole = np.rint(finite)
        return cls(
            compared=int(s.size),
            one=int((whole == 1).sum()),
            more=int((whole > 1).sum()),
            only=int(np.isinf(s).sum()),
            largest=float(finite.max()) if finite.size else 0.0,
        )


def same_bytes(a: object, b: object) -> bool:
    """Equal shapes, dtypes and bytes, or both None."""
    if a is None or b is None:
        return a is None and b is None
    x, y = np.asarray(a), np.asarray(b)
    return x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()


# ---------------------------------------------------------------------------
# A run file, read


BUDGET_TERMS = ("initial", "inflow", "outflow", "source", "current", "leftover")


@dataclass
class Run:
    """A tier-1 run file's records, as the comparison reads them.

    Parameters
    ----------
    path : Path
        The run file.
    """

    path: Path
    file: RunFile = field(init=False)
    config: SimConfig = field(init=False)
    flows: dict[str, tuple[FlowState, bool]] = field(init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.file = RunFile(self.path)
        self.config = SimConfig.from_dict(self.file.read_config().mapping)
        self.flows = {}
        for key in self.file.flow_fingerprints():
            self.flows[key] = (self.file.read_flow_state(key).state, False)
        try:
            key, steady = self.file.read_steady_checkpoint()
        except KeyError:
            steady = None
        if steady is not None and key not in self.flows:
            self.flows[key] = (steady.state, True)
        self.marched = True
        try:
            self.frames = self.file.read_frames()
        except KeyError:
            self.marched = False
        self.sensors = self.file.read_sensor_readings() if self.marched else None
        self.alerts = self.file.read_alerts() if self.marched else None
        self.arrival: RunArrival | None = (
            arrival_from_run(self.file) if self.marched else None
        )
        self.recovery = None
        if self.marched:
            try:
                self.recovery = self.file.read_recovery()
            except KeyError:
                self.recovery = None

    @property
    def step_times(self) -> np.ndarray:
        """Every step's time from the first update on, the sensor table's."""
        assert self.sensors is not None
        return np.asarray(self.sensors.times, dtype=np.float64)

    def budget_rows(self) -> np.ndarray:
        """[frames, classes, terms] of every stored budget, None read as NaN."""
        rows = [
            [
                [
                    *(
                        math.nan if getattr(b, t) is None else float(getattr(b, t))
                        for t in BUDGET_TERMS
                    ),
                    *(float(b.deposited[s]) for s in b.deposited),
                ]
                for b in budgets
            ]
            for budgets in self.frames.budgets
        ]
        return np.array(rows, dtype=np.float64)

    def latencies(self) -> np.ndarray:
        """[events, channels, sensors]: the monitor's latencies, from the alert table.

        ``AlertMonitor.latencies`` on a monitor restored from the table and the
        last update, so the rule is the monitor's own.
        """
        assert self.alerts is not None and self.sensors is not None
        assert self.arrival is not None
        table = self.alerts
        rows = np.array(
            [
                [
                    table.step_on[i],
                    table.t_on[i],
                    math.nan if table.t_off[i] != table.t_off[i] else table.step_off[i],
                    table.t_off[i],
                    table.size_class[i],
                    table.sensor[i],
                    table.reading_on[i],
                    table.peak[i],
                    table.peak_t[i],
                ]
                for i in range(len(table.step_on))
            ],
            dtype=np.float64,
        ).reshape(len(table.step_on), len(ALERT_COLUMNS))
        monitor = AlertMonitor(self.config, Mesh(self.config))
        monitor.restore(
            {
                "alert_rows": rows,
                "alert_step": np.array(float(self.sensors.steps[-1])),
                "alert_t": np.array(float(self.sensors.times[-1])),
                "alert_readings": np.array(self.sensors.readings[-1]),
            }
        )
        return monitor.latencies(np.asarray(self.arrival.onsets).tolist())


# ---------------------------------------------------------------------------
# One pair


@dataclass
class Item:
    """One item of the verdict.

    Parameters
    ----------
    criterion : str
        ``decision``, ``field``, ``stop`` or ``closure``.
    name : str
        What is compared.
    passed : bool
        Whether it meets ADR-017 H.
    detail : dict
        Its numbers.
    """

    criterion: str
    name: str
    passed: bool
    detail: dict


@dataclass
class Comparison:
    """ADR-017 H's verdict on a pair of runs, item by item.

    Parameters
    ----------
    items : list[Item]
        Every decision, field, stop and closure compared.
    bitwise : dict[str, bool]
        Per record (flow states, frames, budgets, sensors, alerts, arrival
        maps, recovery), whether the two are equal bit for bit.
    """

    items: list[Item]
    bitwise: dict[str, bool]

    @property
    def failures(self) -> list[str]:
        """The kinds of failing verdict, each once, in the criteria's order."""
        kinds = []
        for item in self.items:
            if item.passed:
                continue
            kind = item.detail.get("kind")
            if kind not in kinds:
                kinds.append(kind)
        return kinds

    @property
    def verdict(self) -> str:
        """PASS, or FAIL with the kinds that failed."""
        return "PASS" if not self.failures else "FAIL: " + "; ".join(self.failures)

    @property
    def first_tier(self) -> bool:
        """Whether every field is inside its measured spread."""
        return all(
            item.detail["tier"] == INSIDE
            for item in self.items
            if item.criterion == "field"
        )

    def as_dict(self) -> dict:
        """The comparison as JSON can hold it."""
        return {
            "verdict": self.verdict,
            "first_tier": self.first_tier,
            "bitwise": self.bitwise,
            "items": [
                {
                    "criterion": i.criterion,
                    "name": i.name,
                    "passed": i.passed,
                    "detail": _plain(i.detail),
                }
                for i in self.items
            ],
        }


def _plain(value: object) -> object:
    """VALUE with numpy scalars and arrays as Python's, infinities and NaN as text."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _decision(name: str, moves: Moves) -> Item:
    """A decision item from its moves."""
    return Item(
        "decision",
        name,
        moves.equal,
        {
            "kind": DECISION,
            "compared": moves.compared,
            "moved one": moves.one,
            "moved more": moves.more,
            "one only": moves.only,
            "largest": moves.largest,
        },
    )


def _same(name: str, ref: object, other: object) -> Item:
    """A decision item that is one value on each side."""
    return Item(
        "decision", name, ref == other, {"kind": DECISION, "ref": ref, "other": other}
    )


def compare_stops(ref: Run, other: Run) -> list[Item]:
    """Each flow state's stop reason and outer count (ADR-017 H, item 3)."""
    items = []
    for key in sorted(set(ref.flows) | set(other.flows)):
        a = ref.flows.get(key)
        b = other.flows.get(key)
        if a is None or b is None:
            items.append(
                Item(
                    "stop",
                    f"flow state {key[:12]}",
                    False,
                    {"kind": STOP, "ref": a is not None, "other": b is not None},
                )
            )
            continue
        (sa, capped_a), (sb, capped_b) = a, b
        apart = abs(int(sb.outer) - int(sa.outer))
        items.append(
            Item(
                "stop",
                f"flow state {key[:12]}",
                apart <= 1
                and sa.stop_reason == sb.stop_reason
                and capped_a == capped_b,
                {
                    "kind": STOP,
                    "ref outer": int(sa.outer),
                    "other outer": int(sb.outer),
                    "apart": apart,
                    "ref reason": sa.stop_reason,
                    "other reason": sb.stop_reason,
                    "ref capped": capped_a,
                    "other capped": capped_b,
                },
            )
        )
    return items


def _explained(stops: Sequence[Item]) -> bool:
    """Whether the stops explain a field between the spread and the limit.

    Some flow state's outer count is one apart and none more: the kind and
    number of differing stops that moved the probe's fields that far
    (ADR-017 H). The CG and Jacobi counts are not in a run file.
    """
    aparts = [item.detail.get("apart", math.inf) for item in stops]
    return bool(aparts) and max(aparts) == 1


def _field(name: str, value: float, band: str, explained: bool, where: str) -> Item:
    """A field item read in its tiers."""
    level = tier(value, band)
    passed = level == INSIDE or (level == BETWEEN and explained)
    spread, limit = BAND[band]
    return Item(
        "field",
        f"{name}{where}",
        passed,
        {
            "kind": FIELD,
            "difference": value,
            "spread": spread,
            "limit": limit,
            "tier": level,
            "explained by the stops": explained if level == BETWEEN else None,
        },
    )


def compare_flows(ref: Run, other: Run, explained: bool) -> tuple[list[Item], bool]:
    """Each flow field of each flow state both runs hold, and whether all are bitwise."""
    items, bitwise = [], set(ref.flows) == set(other.flows)
    for key in sorted(set(ref.flows) & set(other.flows)):
        a, b = ref.flows[key][0], other.flows[key][0]
        for name in FLOW_FIELDS:
            x, y = getattr(a, name), getattr(b, name)
            if x is None and y is None:
                continue
            if x is None or y is None:
                value = math.inf
            else:
                value = relative(x, y)
                bitwise = bitwise and same_bytes(x, y)
            items.append(_field(name, value, name, explained, f", state {key[:12]}"))
    return items, bitwise


def compare_march(
    ref: Run, other: Run, explained: bool
) -> tuple[list[Item], dict[str, bool]]:
    """The march's decisions, fields and closure; the records' bitwise flags."""
    items: list[Item] = []
    flags: dict[str, bool] = {}
    if not (ref.marched and other.marched):
        items.append(_same("the march ran", ref.marched, other.marched))
        return items, flags
    times = ref.step_times
    flags["step times"] = same_bytes(times, other.step_times)
    items.append(
        Item(
            "decision",
            "the step times",
            flags["step times"],
            {
                "kind": DECISION,
                "ref steps": int(times.size),
                "other steps": int(other.step_times.size),
            },
        )
    )
    fr, fo = ref.frames, other.frames
    items.append(_same("the run's last step", int(fr.steps[-1]), int(fo.steps[-1])))
    items.append(_same("the run's last time", float(fr.times[-1]), float(fo.times[-1])))
    rr, ro = ref.recovery, other.recovery
    if rr is not None or ro is not None:
        items.append(
            _same(
                "the hold",
                None if rr is None else bool(rr.held),
                None if ro is None else bool(ro.held),
            )
        )
    # Arrival maps and the file's sensors.
    ar, ao = ref.arrival, other.arrival
    assert ar is not None and ao is not None
    flags["arrival maps"] = same_bytes(ar.maps, ao.maps)
    items.append(
        _decision(
            "arrival times (events x channels x cells)",
            Moves.of(step_moves(ar.maps, ao.maps, times)),
        )
    )
    alerts_r, alerts_o = ref.alerts, other.alerts
    assert alerts_r is not None and alerts_o is not None
    flags["alerts"] = all(
        same_bytes(getattr(alerts_r, c), getattr(alerts_o, c))
        for c in ("step_on", "step_off", "size_class", "sensor", "t_on", "t_off")
    ) and all(
        same_bytes(getattr(alerts_r, c), getattr(alerts_o, c))
        for c in ("reading_on", "peak", "peak_t")
    )
    items.append(_alerts_item(alerts_r, alerts_o))
    lr, lo = ref.latencies(), other.latencies()
    onsets = np.asarray(ar.onsets, dtype=np.float64)[:, None, None]
    flags["latencies"] = same_bytes(lr, lo)
    items.append(
        _decision(
            "the file's sensors' latencies (events x channels x sensors)",
            Moves.of(step_moves(lr + onsets, lo + onsets, times)),
        )
    )
    # Clear times, and the clean-out's recovery times.
    if scenario_end(ref.config) is not None:
        cr = clear_units({"run": ref.path})
        co = clear_units({"run": other.path})
        spacing = cr.spacing[:, None, None]
        items.append(
            _decision(
                "cells' clear times (channels x cells)",
                Moves.of(moved(cr.clear, co.clear, spacing)),
            )
        )
        room_r, _ = room_clear(cr)
        room_o, _ = room_clear(co)
        items.append(
            _decision(
                "room clear times (channels)",
                Moves.of(moved(room_r, room_o, cr.spacing)),
            )
        )
    if rr is not None and ro is not None:
        recovery_r = _recovery_times(rr)
        recovery_o = _recovery_times(ro)
        items.append(
            _decision(
                "recovery times (room, channels, sensors, means)",
                Moves.of(step_moves(recovery_r, recovery_o, times)),
            )
        )
        flags["recovery"] = same_bytes(recovery_r, recovery_o) and same_bytes(
            np.asarray(rr.local_recovery), np.asarray(ro.local_recovery)
        )
    # The budgets: the verdict against REQ-T05, the closure, the leftover.
    end_r = [_budget_end(b) for b in fr.budgets[-1]]
    end_o = [_budget_end(b) for b in fo.budgets[-1]]
    verdict_r = [_verdict(r, lo_) for r, lo_ in end_r]
    verdict_o = [_verdict(r, lo_) for r, lo_ in end_o]
    items.append(
        Item(
            "decision",
            "every class's budget verdict against REQ-T05",
            verdict_r == verdict_o,
            {
                "kind": BUDGET,
                "ref": verdict_r,
                "other": verdict_o,
                "ref leftover / supplied": [lo_ for _, lo_ in end_r],
                "other leftover / supplied": [lo_ for _, lo_ in end_o],
            },
        )
    )
    closure = max(_closure(ref), _closure(other))
    items.append(
        Item(
            "closure",
            "the budget's closure, every stored frame",
            closure <= CLOSURE,
            {
                "kind": CLOSED,
                "ref": _closure(ref),
                "other": _closure(other),
                "limit": CLOSURE,
            },
        )
    )
    br, bo = ref.budget_rows(), other.budget_rows()
    flags["budgets"] = same_bytes(br, bo)
    change = _leftover_change(end_r, end_o)
    items.append(
        _field(
            "leftover / supplied, change at the end", change, LEFTOVER, explained, ""
        )
    )
    # The class fields and the sensor channels.
    aligned = same_bytes(fr.steps, fo.steps) and same_bytes(fr.times, fo.times)
    flags["frames"] = aligned
    thresholds = np.array(
        ref.config.class_thresholds("the comparison"), dtype=np.float64
    )
    conc, point = [], []
    for k in sorted(fr.fields):
        a, b = fr.fields[k], fo.fields.get(k)
        if b is None or not aligned:
            conc.append(math.inf)
            point.append(math.inf)
            flags["frames"] = False
            continue
        conc.append(relative(a, b))
        point.append(pointwise(a, b, POINTWISE_FLOOR * thresholds[k]))
        flags["frames"] = flags["frames"] and same_bytes(a, b)
    items.append(
        _field(
            "concentration, every stored frame", max(conc), CONCENTRATION, explained, ""
        )
    )
    items.append(
        _field("concentration, pointwise", max(point), POINTWISE, explained, "")
    )
    sr, so = ref.sensors, other.sensors
    assert sr is not None and so is not None
    flags["sensors"] = same_bytes(sr.readings, so.readings) and same_bytes(
        sr.channels, so.channels
    )
    items.append(
        _field(
            "sensor channels, every step",
            relative(sr.channels, so.channels),
            SENSORS,
            explained,
            "",
        )
    )
    return items, flags


def _alerts_item(ref: object, other: object) -> Item:
    """Every alert's opening and closing step, matched by sensor and channel in order."""

    def keyed(table: object) -> dict[tuple[int, int], list[tuple[float, float]]]:
        out: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for i in range(len(table.step_on)):
            key = (int(table.sensor[i]), int(table.size_class[i]))
            off = table.step_off[i]
            closing = math.inf if int(off) == OPEN_STEP else float(off)
            out.setdefault(key, []).append((float(table.step_on[i]), closing))
        for value in out.values():
            value.sort()
        return out

    a, b = keyed(ref), keyed(other)
    moves: list[float] = []
    for key in sorted(set(a) | set(b)):
        xs, ys = a.get(key, []), b.get(key, [])
        for x, y in zip(xs, ys, strict=False):
            # A step's move; an alert open at the end in one run and closed in
            # the other is a move across the threshold, of infinity.
            moves.extend(moved(np.array(x), np.array(y), 1.0).tolist())
        moves.extend([math.inf] * (2 * abs(len(xs) - len(ys))))
    return _decision("alerts' opening and closing steps", Moves.of(np.array(moves)))


def _recovery_times(record: object) -> np.ndarray:
    """The clean-out's recovery times, one flat array: room, channels, sensors, means."""
    return np.concatenate(
        [
            [float(record.room_recovery_time)],
            np.asarray(record.recovery_time, dtype=np.float64).ravel(),
            np.asarray(record.sensor_time, dtype=np.float64).ravel(),
            np.asarray(record.mean_time, dtype=np.float64).ravel(),
        ]
    )


def _budget_end(b: object) -> tuple[float, float]:
    """(residual over supply, leftover over supply) of a class's budget; NaN if nothing.

    A class nothing supplied (a release of other sizes) has no ratio to read.
    """
    try:
        return float(b.relative()), float(b.leftover_relative())
    except ValueError:
        return math.nan, math.nan


def _verdict(residual: float, leftover: float) -> bool | None:
    """A class's verdict against REQ-T05: both ratios below it; None if undefined."""
    if math.isnan(residual) or math.isnan(leftover):
        return None
    return abs(residual) < REQ_T05 and abs(leftover) < REQ_T05


def _leftover_change(
    ref: Sequence[tuple[float, float]], other: Sequence[tuple[float, float]]
) -> float:
    """The largest change of a class's leftover over its supply; infinity on a mismatch.

    Classes undefined on both sides are left out; one defined on one side only
    is a change of infinity.
    """
    largest = 0.0
    for (_, a), (_, b) in zip(ref, other, strict=True):
        if math.isnan(a) and math.isnan(b):
            continue
        if math.isnan(a) or math.isnan(b):
            return math.inf
        largest = max(largest, abs(b - a))
    return largest


def _closure(run: Run) -> float:
    """The largest |residual| over the supply, every stored frame with a supply."""
    largest = 0.0
    for budgets in run.frames.budgets:
        for b in budgets:
            try:
                largest = max(largest, abs(float(b.relative())))
            except ValueError:
                # Nothing supplied yet: nothing to close against.
                continue
    return largest


def compare(ref_path: str | Path, other_path: str | Path) -> Comparison:
    """ADR-017 H's verdict on OTHER against REF, two runs of one scenario.

    Parameters
    ----------
    ref_path, other_path : str or Path
        The reference's run file and the run compared with it.

    Returns
    -------
    Comparison
        Every stop, decision, field and closure, and the bitwise flags.
    """
    ref, other = Run(Path(ref_path)), Run(Path(other_path))
    stops = compare_stops(ref, other)
    explained = _explained(stops)
    flows, flow_bitwise = compare_flows(ref, other, explained)
    march, flags = compare_march(ref, other, explained)
    return Comparison(
        items=[*stops, *flows, *march], bitwise={"flow states": flow_bitwise, **flags}
    )


# ---------------------------------------------------------------------------
# Several scenarios: the placement


def compare_placement(pairs: Mapping[str, tuple[str | Path, str | Path]]) -> list[Item]:
    """The searched layouts, their order and their units' latencies, both sides.

    Parameters
    ----------
    pairs : Mapping[str, tuple]
        Scenario name to its (reference, other) run files, all on one mesh,
        the first's configuration holding the placement section.

    Returns
    -------
    list[Item]
        Per rule: the cells of the file's four and of each searched pool, the
        layouts' order, and every unit's latency of each layout in steps.
    """
    sides = []
    for side in (0, 1):
        runs = {name: Run(Path(paths[side])) for name, paths in pairs.items()}
        sides.append(runs)
    first = next(iter(sides[0].values()))
    config, mesh = first.config, Mesh(first.config)
    spec = config.placement
    if spec is None:
        raise ValueError("the first run's configuration has no placement section")
    configs = [r.config for r in sides[0].values()]
    surfaces = tuple(m for m in spec.mount if m != MOUNT_STAND)
    pools = {
        SURFACE: candidates(mesh, configs, surfaces, spec.standoff),
        STAND: candidates(
            mesh, configs, spec.mount, spec.standoff, stand_height=spec.stand_height
        ),
        FREE: candidates(mesh, configs, (MOUNT_FREE,), spec.standoff),
    }
    from src.monitor import sensor_cells

    sensors = tuple(sensor_cells(config, mesh))
    results = []
    for runs in sides:
        units = detection_units({name: r.arrival for name, r in runs.items()})
        per_rule = {}
        for rule in RULES:
            cells = {FILE_LAYOUT: sensors}
            for name, pool in pools.items():
                cells[name] = search(units, pool, spec.count, rule).cells
            scored = {n: given_layout(units, c) for n, c in cells.items()}
            name_of = {id(layout): n for n, layout in scored.items()}
            per_rule[rule] = {
                "cells": {n: tuple(sorted(c)) for n, c in cells.items()},
                "order": [name_of[id(layout)] for layout in rank(scored.values())],
                "per_unit": {
                    n: np.array(
                        [
                            min(units.latencies[u, j, i] for j, i in c)
                            for u in range(len(units.labels))
                        ]
                    )
                    for n, c in cells.items()
                },
            }
        results.append((units, per_rule))
    (units_r, ref_rules), (units_o, other_rules) = results
    items: list[Item] = [_same("the detection units", units_r.labels, units_o.labels)]
    for rule in RULES:
        r, o = ref_rules[rule], other_rules[rule]
        for name in r["cells"]:
            items.append(
                _same(
                    f"{rule}: the {name} layout's cells",
                    r["cells"][name],
                    o["cells"][name],
                )
            )
        items.append(_same(f"{rule}: the layouts' order", r["order"], o["order"]))
        moves = []
        for name in r["cells"]:
            for u, (scenario, _event, _k) in enumerate(units_r.labels):
                run = sides[0][scenario]
                onset = float(run.arrival.onsets[0])
                a = np.array([r["per_unit"][name][u] + onset])
                b = np.array([o["per_unit"][name][u] + onset])
                moves.append(step_moves(a, b, run.step_times)[0])
        items.append(
            _decision(f"{rule}: the layouts' unit latencies", Moves.of(np.array(moves)))
        )
    return items


# ---------------------------------------------------------------------------
# The command line


def report(comparison: Comparison) -> list[str]:
    """The comparison's lines: the verdict, then one line per item."""
    lines = [f"verdict: {comparison.verdict}"]
    for item in comparison.items:
        mark = "pass" if item.passed else "FAIL"
        detail = {k: v for k, v in item.detail.items() if k != "kind"}
        lines.append(f"  [{mark}] {item.criterion}: {item.name}: {_plain(detail)}")
    lines.append(
        "bitwise: "
        + ", ".join(
            f"{name} {'yes' if same else 'no'}"
            for name, same in comparison.bitwise.items()
        )
    )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Compare two run files and print the verdict; 0 on PASS, 1 on FAIL.

    Parameters
    ----------
    argv : Sequence[str], optional
        The arguments; ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        0 when every item passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path, help="the reference's run file")
    parser.add_argument("other", type=Path, help="the run compared with it")
    parser.add_argument("--json", type=Path, help="write the comparison here as JSON")
    args = parser.parse_args(argv)
    comparison = compare(args.reference, args.other)
    print("\n".join(report(comparison)))
    if args.json is not None:
        args.json.write_text(
            json.dumps(comparison.as_dict(), indent=1) + "\n", encoding="utf-8"
        )
    return 0 if comparison.verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
