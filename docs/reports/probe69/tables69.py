"""The tables and figures of ADR-014 step 7, generated from the run files (prompt 69).

    python docs/reports/probe69/tables69.py             # writes tables69.md
    python docs/reports/probe69/tables69.py --figures   # and the figures under docs/reports/
    python docs/reports/probe69/tables69.py --splice    # and the report's marked copies
    python docs/reports/probe69/tables69.py --check     # regenerates and compares

Reads the four run files under results/builder69/ (written by run69.sh through
scripts/run_scenario.py), the runner's summaries beside them (the origin of each
flow state and the run's wall time, which the run file does not hold), and the
T7 table of docs/reports/probe59/tables59.md for the time error at Courant 0.4
that ADR-014 decision 2 asks to be stated beside every recovery and detection
time. Every number the gate report states about the scenarios is in a table
here; the report copies the tables between markers and ``--check`` compares
both the written file and the report's copies with what the records give. Do
not edit tables69.md by hand.

A run that is missing prints as "not run". ``--coarse`` points the script at the
step 4 coarse runs (results/builder64/coarse/), the same file format, for
developing the tables before the product runs finish.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.io_manager import (  # noqa: E402 -- follows sys.path.insert
    RecoveryRecord,
    RunFile,
)
from src.mesh import SOLID, Mesh  # noqa: E402 -- follows sys.path.insert

RESULTS = ROOT / "results" / "builder69"
TABLES = HERE / "tables69.md"
REPORT = ROOT / "docs" / "reports" / "phase4_scenarios_report.md"
FIGURES = ROOT / "docs" / "reports"
T7 = ROOT / "docs" / "reports" / "probe59" / "tables59.md"
# The state of record for the closed room (ECR-002 step 8's L0 on 320x120, whose face
# hash prompt 68 reproduced at 6fe887b) and T1's two solves of the open-door state.
L0_RECORD = ROOT / "results" / "builder56" / "L0_320x120.json"
# hash62.py's records: the rows solved at this commit (_69) against prompt 68's (_68).
HASHES = ROOT / "results" / "builder62"
HASH_CASES = (
    "val001_80x40",
    "val001_80x40_stretched",
    "val002_20x20",
    "val002_40x40",
    "val002_80x80",
    "product_80x30",
)
T1_ARMS = {
    "rest": ROOT / "results" / "builder59" / "rest.npz",
    "warm": ROOT / "results" / "builder59" / "warm.npz",
}
NAMES = ("door_leak", "filter_breach", "equipment_dust", "clean_out")
TITLES = {
    "door_leak": "Door leak",
    "filter_breach": "Filter breach",
    "equipment_dust": "Equipment dust",
    "clean_out": "Clean-out",
}
# The snapshot of each scenario at its most telling time: the end of the door's
# opening, the end of the breach, the end of the release; the clean-out at the
# free-area mean's 100:1 time of the class shown, where what lingers is visible.
SNAPSHOT_TIMES = {"door_leak": 15.0, "filter_breach": 70.0, "equipment_dust": 40.0}
SNAPSHOT_CLASS = 2  # 0.5 um
RECOVERY_MAP_CLASSES = (0, 4)  # 0.1 um and 5 um
HISTORY_CLASSES = (0, 4)
MARKERS = (
    "reload",
    "flows",
    "hashes",
    "rows",
    "march",
    "budgets",
    "accrual",
    "sensors",
    "door",
    "recovery",
    "lingering",
    "t7",
    "grid",
    "predictions",
)

# Prompt 69's predictions (the orchestrator, 2026-10-28, committed before the runs).
PREDICTION_BUDGET = 1.0e-6
# REQ-T05's bound on the same ratio (SYSTEM.md section 2: 0.01% across all timesteps).
REQ_T05_RELATIVE = 1.0e-4
# The step 4 coarse runs (prompt 64, 80x30), read for the grid comparison in G1.
COARSE = ROOT / "results" / "builder64" / "coarse"
COARSE_STEMS = {name: name for name in NAMES} | {"clean_out": "clean_out_nohold"}
PREDICTION_ROOM_RECOVERY = (25.0, 35.0)
PREDICTION_MEAN_RATIO = (2.0, 4.0)
PREDICTION_WARM_RATIO = (0.8, 1.0)


@dataclass(frozen=True)
class Summary:
    """What scripts/run_scenario.py printed for a run, read back from its summary file.

    Parameters
    ----------
    flows : dict[str, tuple[str, str | None]]
        Fingerprint prefix to (origin, the state it was warm-started from or None).
    intervals : list[str]
        The interval lines as printed.
    wall_seconds : float
        The run's wall time, the flow solves included.
    """

    flows: dict[str, tuple[str, str | None]]
    intervals: list[str]
    wall_seconds: float


class Run:
    """One scenario's records: its run file, its summary and its configuration."""

    def __init__(self, root: Path, name: str) -> None:
        self.name = name
        self.file = RunFile(root / f"{name}.h5")
        self.summary = read_summary(root / f"{name}.txt")
        self.config_record = self.file.read_config()
        self.config = SimConfig.from_dict(self.config_record.mapping)
        self.mesh = Mesh(self.config)
        self.live = self.mesh.cell_type != SOLID
        self.free_area = float(
            np.outer(self.mesh.dy_cell, self.mesh.dx_cell)[self.live].sum()
        )
        self.frames = self.file.read_frames()
        self.sensors = self.file.read_sensor_readings()
        self.thresholds = np.asarray(
            self.config.class_thresholds("the step 7 tables"), dtype=float
        )
        self.recovery: RecoveryRecord | None
        try:
            self.recovery = self.file.read_recovery()
        except KeyError:
            self.recovery = None

    @property
    def sizes_um(self) -> list[float]:
        """The class diameters in micrometres."""
        return [size * 1.0e6 for size in self.config.particle_sizes]

    @property
    def event_start(self) -> float | None:
        """The first event's start, or None for a scenario of an initial event alone."""
        starts = [
            event.start
            for event in self.config.scenario.events
            if event.duration is not None
        ]
        return min(starts) if starts else None

    def flow_rows(self) -> list[tuple[str, str, str | None, int, str, float]]:
        """Per flow state: prefix, origin, warm_from, outer, stop reason, wall seconds."""
        rows = []
        for fingerprint in self.file.flow_fingerprints():
            record = self.file.read_flow_state(fingerprint)
            origin, warm_from = self.summary.flows.get(fingerprint[:12], ("?", None))
            rows.append(
                (
                    fingerprint[:12],
                    origin,
                    warm_from,
                    record.state.outer,
                    str(record.state.stop_reason),
                    record.wall_seconds,
                )
            )
        return rows


def read_summary(path: Path) -> Summary:
    """Parse the runner's summary (its flow-state, interval and wall-time lines)."""
    flows: dict[str, tuple[str, str | None]] = {}
    intervals: list[str] = []
    wall = math.nan
    flow_line = re.compile(
        r"^\s+([0-9a-f]{12}): (rest|warm|resumed|run file|kept)(?: from ([0-9a-f]{12}))?, "
        r"(\d+) outer"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        found = flow_line.match(line)
        if found:
            flows[found.group(1)] = (found.group(2), found.group(3))
        elif line.startswith("  [") and " steps of " in line:
            intervals.append(line.strip())
        elif line.startswith("wall time: "):
            wall = float(line.split()[2])
    return Summary(flows, intervals, wall)


def fmt(value: float, spec: str = ".4f") -> str:
    """A number, or the words for NaN (not reached) and inf (not recovered)."""
    if math.isnan(value):
        return "not reached"
    if math.isinf(value):
        return "not recovered"
    return format(value, spec)


def pct(value: float) -> str:
    """A relative difference as a signed percentage to two decimals."""
    return f"{100.0 * value:+.2f}%"


def t7_errors() -> tuple[dict[str, float], float]:
    """T7's time error at Courant 0.4 against 0.1, per sensor and for the free-area mean.

    Returns
    -------
    tuple[dict[str, float], float]
        Per sensor the signed relative difference of largest magnitude over
        T7's classes and levels, and the largest relative difference of the
        free-area mean's columns over T7's rows after t = 0.
    """
    text = T7.read_text(encoding="utf-8")
    start = text.index("### T7.")
    end = text.find("\n### ", start + 1)
    section = text[start : end if end > 0 else None]
    worst: dict[str, float] = {}
    row = re.compile(
        r"^\| (?:0\.5|5) \| (\S+) \| (?:0\.01|0\.001) \| .* \| (-?[\d.]+)% \|$"
    )
    mean_row = re.compile(
        r"^\| \d+ \| [\d.]+ \| [\d.e+-]+ \| ([\d.e+-]+) \| ([\d.e+-]+) \| ([\d.e+-]+) \| ([\d.e+-]+) \|$"
    )
    mean_worst = 0.0
    for line in section.splitlines():
        found = row.match(line)
        if found:
            value = float(found.group(2)) / 100.0
            if abs(value) > abs(worst.get(found.group(1), 0.0)):
                worst[found.group(1)] = value
            continue
        found = mean_row.match(line)
        if found:
            a1, a4, b1, b4 = (float(found.group(i)) for i in range(1, 5))
            mean_worst = max(mean_worst, abs(a4 / a1 - 1.0), abs(b4 / b1 - 1.0))
    if not worst:
        raise ValueError(f"no T7 sensor rows found in {T7}")
    return worst, mean_worst


def t7_text(errors: dict[str, float], sensor: str) -> str:
    """The T7 column entry for a sensor."""
    if sensor in errors:
        return pct(errors[sensor])
    return "not in T7"


def reload_table(runs: dict[str, Run | None]) -> list[str]:
    """R1: each run file read back through RunFile, group by group."""
    lines = [
        "### R1. The run files reloaded (RunFile, every group read back)",
        "",
        "| Run file | Commit | Started | Concurrent | Config SHA-256 (12) | Flow states | Frames (n, ny, nx) | Sensor readings (steps, classes, sensors) | /recovery/ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        if run is None:
            lines.append(f"| {name}.h5 | not run | | | | | | | |")
            continue
        record = run.config_record
        frames = run.frames.fields[0].shape
        readings = run.sensors.readings.shape
        recovery = "read" if run.recovery is not None else "none (no initial event)"
        lines.append(
            f"| {name}.h5 | {record.commit} | {record.started} | {record.concurrent} | "
            f"{record.sha256[:12]} | {len(run.file.flow_fingerprints())} | "
            f"{frames} | {readings} | {recovery} |"
        )
    return lines


def flows_table(runs: dict[str, Run | None]) -> list[str]:
    """F1: every flow state solved or read, with its outer count and wall time."""
    lines = [
        "### F1. The flow states (320x120, from rest or warm-started)",
        "",
        "| Run | State | Origin | Outer | Stop | Wall (s) | s per outer |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        if run is None:
            lines.append(f"| {name} | not run | | | | | |")
            continue
        for prefix, origin, warm_from, outer, stop, wall in run.flow_rows():
            start = origin if warm_from is None else f"{origin} from {warm_from}"
            lines.append(
                f"| {name} | {prefix} | {start} | {outer:,} | {stop} | {wall:.0f} | "
                f"{wall / outer:.3f} |"
            )
    return lines


def face_hash(u: np.ndarray, v: np.ndarray) -> str:
    """SHA-256 over u's bytes then v's (docs/reports/probe62/hash62.py's hash)."""
    return hashlib.sha256(
        np.ascontiguousarray(u, dtype="<f8").tobytes()
        + np.ascontiguousarray(v, dtype="<f8").tobytes()
    ).hexdigest()


def hash_table(runs: dict[str, Run | None]) -> list[str]:
    """H1: each run's flow states against the states of record.

    The closed state is compared by face hash with ECR-002 step 8's L0 record,
    the hash of record prompt 68 reproduced; the open state, warm-started, is
    compared with T1's rest and warm arms by the largest absolute face
    difference, since a warm start is not bitwise a solve from rest.
    """
    lines = [
        "### H1. The flow states against the states of record",
        "",
        "| Run | State | Origin | Face hash (first 16) | Against the record | Result |",
        "|---|---|---|---|---|---|",
    ]
    record = (
        json.loads(L0_RECORD.read_text(encoding="utf-8"))
        if L0_RECORD.exists()
        else None
    )
    arms = {name: np.load(path) for name, path in T1_ARMS.items() if path.exists()}
    for name, run in runs.items():
        if run is None:
            lines.append(f"| {name} | not run | | | | |")
            continue
        for fingerprint in run.file.flow_fingerprints():
            state = run.file.read_flow_state(fingerprint).state
            origin = run.summary.flows.get(fingerprint[:12], ("?", None))[0]
            digest = face_hash(state.u, state.v)
            if origin in ("rest", "kept", "run file"):
                if record is None:
                    against, result = "L0_320x120 record missing", "not compared"
                elif state.u.shape != (
                    record["raw"]["domain"]["ny"],
                    record["raw"]["domain"]["nx"] + 1,
                ):
                    against, result = "L0_320x120 record", "grid differs"
                else:
                    against = f"L0_320x120 face hash {record['face_hash'][:16]}"
                    result = "equal" if digest == record["face_hash"] else "DIFFERS"
            elif arms:
                parts = []
                for arm, fields in arms.items():
                    if fields["u_faces"].shape != state.u.shape:
                        parts.append(f"T1 {arm}: grid differs")
                        continue
                    du = float(np.abs(fields["u_faces"] - state.u).max())
                    dv = float(np.abs(fields["v_faces"] - state.v).max())
                    parts.append(f"T1 {arm}: {max(du, dv):.2e} m/s")
                against, result = "largest face difference", "; ".join(parts)
            else:
                against, result = "T1 arms missing", "not compared"
            lines.append(
                f"| {name} | {fingerprint[:12]} | {origin} | {digest[:16]} | {against} | "
                f"{result} |"
            )
    return lines


def rows_table() -> list[str]:
    """H2: the laminar rows and the coarse product row solved at this commit, against prompt 68's."""
    lines = [
        "### H2. The hashes of record at the gate's source (hash62.py, this commit, against prompt 68's _68 records)",
        "",
        "| Case | Commit | Stop | Outer | Hash (first 16) | Equal to _68 | Wall (s) | Processes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in HASH_CASES:
        path = HASHES / f"hash_{case}_69.json"
        if not path.exists():
            lines.append(f"| {case} | | not run | | | | | |")
            continue
        new = json.loads(path.read_text(encoding="utf-8"))
        old_path = HASHES / f"hash_{case}_68.json"
        if old_path.exists():
            old = json.loads(old_path.read_text(encoding="utf-8"))
            same = "yes" if old["face_hash"] == new["face_hash"] else "NO"
        else:
            same = "no record"
        lines.append(
            f"| {case} | {new['commit']} | {new['stop']} | {new['outer']:,} | "
            f"{new['face_hash'][:16]} | {same} | {new['wall_seconds']:.0f} | "
            f"{new['concurrent_processes']} |"
        )
    return lines


def march_table(runs: dict[str, Run | None]) -> list[str]:
    """M1: the intervals, the steps and the march's wall time per run."""
    lines = [
        "### M1. The marches (Courant 0.4, five classes at one step)",
        "",
        "| Run | Intervals (as planned) | Steps | t at the end (s) | Run wall (s) | Flow solves' wall (s) | March wall (s) | March wall s per simulated s | Frames |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        if run is None:
            lines.append(f"| {name} | not run | | | | | | | |")
            continue
        steps = int(run.frames.steps[-1])
        t_end = float(run.frames.times[-1])
        solves = sum(row[5] for row in run.flow_rows())
        march = run.summary.wall_seconds - solves
        intervals = "; ".join(run.summary.intervals)
        lines.append(
            f"| {name} | {intervals} | {steps:,} | {t_end:.4f} | "
            f"{run.summary.wall_seconds:.0f} | {solves:.0f} | {march:.0f} | "
            f"{march / t_end:.2f} | {len(run.frames.steps)} |"
        )
    return lines


def budget_table(runs: dict[str, Run | None]) -> list[str]:
    """B1: each class's budget closure at the last frame."""
    lines = [
        "### B1. Budget closure at the end of each run (residual relative to the particles supplied)",
        "",
        "| Run | Class (um) | Supplied (initial + inflow + source) | Current | Outflow | Deposited | Residual | Relative | Below 1e-6 (prediction (a)) | Below REQ-T05's 1e-4 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        if run is None:
            lines.append(f"| {name} | not run | | | | | | | | |")
            continue
        for k, budget in enumerate(run.frames.budgets[-1]):
            supplied = (budget.initial or 0.0) + budget.inflow + budget.source
            deposited = sum(budget.deposited.values())
            relative = budget.relative()
            lines.append(
                f"| {name} | {run.sizes_um[k]:g} | {supplied:.6e} | {budget.current:.6e} | "
                f"{budget.outflow:.6e} | {deposited:.6e} | {budget.residual():.6e} | "
                f"{relative:.3e} | {'yes' if abs(relative) < PREDICTION_BUDGET else 'NO'} | "
                f"{'yes' if abs(relative) < REQ_T05_RELATIVE else 'NO'} |"
            )
    return lines


def residual_history(run: Run, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame times, the class's budget residual and its content, NaN before the first step."""
    times = np.asarray(run.frames.times, dtype=float)
    residual = np.full(len(times), math.nan)
    content = np.full(len(times), math.nan)
    for i, budgets in enumerate(run.frames.budgets):
        b = budgets[k]
        if b.initial is None or b.current is None:
            continue
        residual[i] = b.residual()
        content[i] = b.current
    return times, residual, content


def nearest(times: np.ndarray, values: np.ndarray, t: float) -> float:
    """The value at the frame nearest ``t``."""
    return float(values[int(np.argmin(np.abs(times - t)))])


def accrual_table(runs: dict[str, Run | None]) -> list[str]:
    """B2: when the 0.1 um residual accrued, against the particles present (the records only)."""
    lines = [
        "### B2. The 0.1 um budget residual as it accrued (frames nearest each interval's end), against the particles present",
        "",
        "| Run | Residual at t = 10 s | Residual at the event's end (t, s) | Residual at the end | Largest content (particles per m depth) | Content's time integral (particle s) | Residual over the integral (1/s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        if run is None:
            lines.append(f"| {name} | not run | | | | | |")
            continue
        times, residual, content = residual_history(run, 0)
        ok = np.isfinite(residual)
        integral = float(np.trapezoid(content[ok], times[ok]))
        ends = [
            event.start + event.duration
            for event in run.config.scenario.events
            if event.duration is not None
        ]
        event_end = (
            f"{nearest(times, residual, max(ends)):.4e} ({max(ends):g})"
            if ends
            else "no timed event"
        )
        lines.append(
            f"| {name} | {nearest(times, residual, 10.0):.4e} | {event_end} | {residual[ok][-1]:.4e} | "
            f"{np.nanmax(content):.4e} | {integral:.4e} | {residual[ok][-1] / integral:.3e} |"
        )
    return lines


def grid_table(
    runs: dict[str, Run | None], coarse: dict[str, Run | None] | None
) -> list[str]:
    """G1: the step 4 coarse runs (80x30) against the product runs (320x120), 0.1 um."""
    lines = [
        "### G1. The 80x30 runs of step 4 (prompt 64) against the 320x120 runs, 0.1 um class",
        "",
    ]
    if coarse is None:
        lines.append("not applicable in --coarse mode")
        return lines
    lines.extend(
        [
            "| Run | Grid | Supplied | Residual | Relative | Relative over the classes | Room recovery time (s) | Free-area mean 100:1 (s) | Ratio to ln(100)/N |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for name in runs:
        for label, run in (("80x30", coarse.get(name)), ("320x120", runs.get(name))):
            if run is None:
                lines.append(f"| {name} | {label} | not run | | | | | | |")
                continue
            budgets = run.frames.budgets[-1]
            b = budgets[0]
            supplied = (b.initial or 0.0) + b.inflow + b.source
            relatives = [x.relative() for x in budgets]
            if run.recovery is None:
                room = mean = ratio = "none"
            else:
                r = run.recovery
                room = fmt(r.room_recovery_time)
                mean = fmt(float(r.mean_time[0]))
                ratio = fmt(float(r.ratio[0]))
            lines.append(
                f"| {name} | {label} | {supplied:.4e} | {b.residual():.4e} | {b.relative():.3e} | "
                f"{min(relatives):.3e} to {max(relatives):.3e} | {room} | {mean} | {ratio} |"
            )
    return lines


def sensor_table(runs: dict[str, Run | None], errors: dict[str, float]) -> list[str]:
    """S1: per run, class and sensor, the peak, the first time at threshold and the time there.

    A reading counts as above when it is not below the threshold (ADR-014 C's
    within-spec rule). The time above is the sum of the steps whose end
    reading is above; the first time above is that step's time, and the
    latency is measured from the first event's start.
    """
    lines = [
        "### S1. The sensors: peak reading, first time at or above threshold, time above (T7's time error beside)",
        "",
        "| Run | Class (um) | Sensor | Threshold | Peak (per m^3) | Peak over threshold | Peak time (s) | First time at or above (s) | Latency from event start (s) | Time above (s) | T7 time error at Courant 0.4 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        if run is None:
            lines.append(f"| {name} | not run | | | | | | | | | |")
            continue
        times = run.sensors.times
        dt = np.diff(times)
        start = run.event_start
        for k in range(len(run.sizes_um)):
            threshold = run.thresholds[k]
            for s, sensor in enumerate(run.sensors.names):
                series = run.sensors.readings[:, k, s]
                peak = int(np.argmax(series))
                above = series >= threshold
                first = np.flatnonzero(above)
                first_t = float(times[first[0]]) if first.size else math.nan
                latency = (
                    "none"
                    if start is None
                    else ("never" if not first.size else f"{first_t - start:.4f}")
                )
                time_above = float(dt[above[1:]].sum())
                lines.append(
                    f"| {name} | {run.sizes_um[k]:g} | {sensor} | {threshold:.3g} | "
                    f"{series[peak]:.4e} | {series[peak] / threshold:.3e} | "
                    f"{times[peak]:.4f} | {fmt(first_t) if first.size else 'never'} | "
                    f"{latency} | {time_above:.4f} | {t7_text(errors, sensor)} |"
                )
    return lines


def door_table(runs: dict[str, Run | None]) -> list[str]:
    """D1: the door's two states, the warm start against the closed state from rest."""
    lines = [
        "### D1. The door: the open state warm-started against the closed state from rest",
        "",
        "| State | Origin | Outer | Wall (s) | Worst imbalance at the stop (kg/s per m) | Residual at the stop |",
        "|---|---|---|---|---|---|",
    ]
    run = runs.get("door_leak")
    if run is None:
        lines.append("| not run | | | | | |")
        return lines
    rest = warm = None
    for prefix, origin, warm_from, outer, _stop, wall in run.flow_rows():
        readings = run.file.read_flow_state(
            next(f for f in run.file.flow_fingerprints() if f.startswith(prefix))
        ).rule_readings
        start = origin if warm_from is None else f"{origin} from {warm_from}"
        label = "door closed" if origin == "rest" else "door open"
        lines.append(
            f"| {label} ({prefix}) | {start} | {outer:,} | {wall:.0f} | "
            f"{readings.get('worst_imbalance', math.nan):.2e} | "
            f"{readings.get('residual', math.nan):.2e} |"
        )
        if origin == "rest":
            rest = (outer, wall)
        elif origin == "warm":
            warm = (outer, wall)
    lines.append("")
    lines.append("| Ratio, open warm over closed from rest | Outer | Wall |")
    lines.append("|---|---|---|")
    if rest and warm:
        lo, hi = PREDICTION_WARM_RATIO
        ratio = warm[0] / rest[0]
        verdict = "holds" if lo <= ratio <= hi else "fails"
        lines.append(
            f"| prediction (d): {lo} to {hi} on outer | {ratio:.3f} ({verdict}) | "
            f"{warm[1] / rest[1]:.3f} |"
        )
    else:
        lines.append("| prediction (d) | not measured | |")
    return lines


def recovery_table(
    runs: dict[str, Run | None], errors: dict[str, float], mean_error: float
) -> list[str]:
    """C1: the clean-out's outputs of ADR-014 C per class, T7's time error beside."""
    lines = [
        "### C1. The clean-out: recovery (ADR-014 C), T7's time error at Courant 0.4 beside each time",
        "",
    ]
    run = runs.get("clean_out")
    if run is None or run.recovery is None:
        lines.append("not run")
        return lines
    record = run.recovery
    lines.append(
        f"Air change rate N = {record.air_change_rate:.6g} 1/s (1/N = {1.0 / record.air_change_rate:.4f} s); "
        f"well-mixed 100:1 time ln(100)/N = {record.well_mixed_time:.4f} s; the run went to "
        f"t = {record.t_final:.4f} s. T7's time error at Courant 0.4 against 0.1 (160x60): "
        + ", ".join(f"{s} {pct(e)}" for s, e in errors.items())
        + f"; free-area mean within {100.0 * mean_error:.2f}% at every tabulated air change."
    )
    lines.append("")
    lines.append(
        "| Class (um) | Recovery time (every sensor below and staying below), s | "
        "Free-area mean 100:1, s | Ratio to ln(100)/N (prediction (c): 2 to 4) | "
        + " | ".join(
            f"{s} 100:1, s (T7 {pct(errors.get(s, math.nan))})"
            for s in record.sensor_names
        )
        + " | Local recovery field maximum, s |"
    )
    lines.append("|" + "---|" * (4 + len(record.sensor_names) + 1))
    for k, size in enumerate(run.sizes_um):
        ratio = float(record.ratio[k])
        lo, hi = PREDICTION_MEAN_RATIO
        ratio_text = (
            f"above {record.t_final / record.well_mixed_time:.4f} (fails)"
            if math.isnan(ratio)
            else f"{ratio:.4f} ({'holds' if lo <= ratio <= hi else 'fails'})"
        )
        sensors = " | ".join(fmt(float(t)) for t in record.sensor_time[k])
        lines.append(
            f"| {size:g} | {fmt(float(record.recovery_time[k]))} | "
            f"{fmt(float(record.mean_time[k]))} | {ratio_text} | {sensors} | "
            f"{fmt(float(record.local_recovery_max[k]))} |"
        )
    lo, hi = PREDICTION_ROOM_RECOVERY
    room = record.room_recovery_time
    verdict = "holds" if lo <= room <= hi else "fails"
    lines.append("")
    lines.append(
        f"Room recovery time (every sensor of every class within spec and staying so): "
        f"{fmt(room)} s; prediction (b), {lo:g} to {hi:g} s, {verdict}. "
        f"T7's error at the last sensor, near_door, is {pct(errors.get('near_door', math.nan))}."
    )
    return lines


def clusters(mask: np.ndarray) -> list[np.ndarray]:
    """The 4-connected components of a boolean mask, largest first, as [n, 2] (j, i) arrays."""
    remaining = mask.copy()
    found: list[np.ndarray] = []
    while remaining.any():
        seed = np.zeros_like(remaining)
        j, i = np.argwhere(remaining)[0]
        seed[j, i] = True
        while True:
            grown = seed.copy()
            grown[1:, :] |= seed[:-1, :]
            grown[:-1, :] |= seed[1:, :]
            grown[:, 1:] |= seed[:, :-1]
            grown[:, :-1] |= seed[:, 1:]
            grown &= remaining
            if (grown == seed).all():
                break
            seed = grown
        found.append(np.argwhere(seed))
        remaining &= ~seed
    found.sort(key=len, reverse=True)
    return found


def lingering_table(runs: dict[str, Run | None]) -> list[str]:
    """L1: the cells not recovered by the end, and where the field clears last, per class."""
    lines = [
        "### L1. The clean-out: where the room clears last (the local recovery field)",
        "",
        "| Class (um) | Free cells | Not recovered at the end | Share | Where (x range, y range, m; the three largest clusters, cells) | Cell clearing last (x, y, m) and its time, s | Cells clearing after the room recovery time | Share |",
        "|---|---|---|---|---|---|---|---|",
    ]
    run = runs.get("clean_out")
    if run is None or run.recovery is None:
        lines.append("| not run | | | | | | | |")
        return lines
    record = run.recovery
    mesh = run.mesh
    free = int(run.live.sum())
    for k, size in enumerate(run.sizes_um):
        field = record.local_recovery[k]
        not_recovered = np.isinf(field) & run.live
        count = int(not_recovered.sum())
        where = "none"
        if count:
            parts = []
            for cells in clusters(not_recovered)[:3]:
                xs = mesh.xc[cells[:, 1]]
                ys = mesh.yc[cells[:, 0]]
                parts.append(
                    f"x {xs.min():.3f} to {xs.max():.3f}, y {ys.min():.3f} to {ys.max():.3f} "
                    f"({len(cells)})"
                )
            where = "; ".join(parts)
        finite = np.where(np.isfinite(field) & run.live, field, -np.inf)
        j, i = np.unravel_index(int(np.argmax(finite)), finite.shape)
        last = (
            f"({mesh.xc[i]:.3f}, {mesh.yc[j]:.3f}) at {finite[j, i]:.4f}"
            if count == 0
            else f"({mesh.xc[i]:.3f}, {mesh.yc[j]:.3f}) at {finite[j, i]:.4f} among the recovered"
        )
        room = record.room_recovery_time
        late = int(((field > room) & run.live).sum()) if not math.isnan(room) else 0
        lines.append(
            f"| {size:g} | {free:,} | {count:,} | {100.0 * count / free:.2f}% | {where} | "
            f"{last} | {late:,} | {100.0 * late / free:.2f}% |"
        )
    return lines


def t7_table(errors: dict[str, float], mean_error: float) -> list[str]:
    """E1: T7's time error at Courant 0.4, the column the tables above carry."""
    lines = [
        "### E1. The time error at Courant 0.4 against 0.1 (T7 of tables59.md, 160x60, read here)",
        "",
        "| Reading | Largest relative difference over T7's classes and levels |",
        "|---|---|",
    ]
    for sensor, value in errors.items():
        lines.append(f"| {sensor} | {pct(value)} |")
    lines.append(
        f"| free-area mean (at the tabulated air changes) | within {100.0 * mean_error:.2f}% |"
    )
    return lines


def predictions_table(runs: dict[str, Run | None]) -> list[str]:
    """P1: prompt 69's predictions against the measurements."""
    lines = [
        "### P1. Prompt 69's predictions against the measurements",
        "",
        "| Prediction | Measured | Verdict |",
        "|---|---|---|",
    ]
    worst = -math.inf
    missing = False
    for run in runs.values():
        if run is None:
            missing = True
            continue
        worst = max(worst, max(abs(b.relative()) for b in run.frames.budgets[-1]))
    if missing:
        lines.append(
            "| (a) every budget closes below 1e-6 relative | not every run finished | not scored |"
        )
    else:
        lines.append(
            f"| (a) every budget closes below 1e-6 relative | largest magnitude {worst:.3e}, "
            f"inside REQ-T05's {REQ_T05_RELATIVE:.0e} by {REQ_T05_RELATIVE / worst:.0f} times | "
            f"{'holds' if worst < PREDICTION_BUDGET else 'fails'} |"
        )
    clean = runs.get("clean_out")
    if clean is None or clean.recovery is None:
        lines.append("| (b) room recovery time 25 to 35 s | not run | not scored |")
        lines.append(
            "| (c) free-area mean 100:1 at 2 to 4 times ln(100)/N | not run | not scored |"
        )
    else:
        record = clean.recovery
        room = record.room_recovery_time
        lo, hi = PREDICTION_ROOM_RECOVERY
        lines.append(
            f"| (b) room recovery time 25 to 35 s | {fmt(room)} s | "
            f"{'holds' if lo <= room <= hi else 'fails'} |"
        )
        ratios = [float(r) for r in record.ratio]
        lo, hi = PREDICTION_MEAN_RATIO
        if any(math.isnan(r) for r in ratios):
            measured = f"not reached by {record.t_final:.1f} s in some class (ratio above {record.t_final / record.well_mixed_time:.2f})"
            verdict = "fails"
        else:
            measured = f"{min(ratios):.4f} to {max(ratios):.4f} over the classes"
            verdict = "holds" if all(lo <= r <= hi for r in ratios) else "fails"
        lines.append(
            f"| (c) free-area mean 100:1 at 2 to 4 times ln(100)/N | {measured} | {verdict} |"
        )
    door = runs.get("door_leak")
    if door is None:
        lines.append(
            "| (d) open state warm-started at 0.8 to 1.0 of the closed state's outer count | not run | not scored |"
        )
    else:
        rest = warm = None
        for _, origin, _, outer, _, _ in door.flow_rows():
            if origin == "rest":
                rest = outer
            elif origin == "warm":
                warm = outer
        if rest and warm:
            ratio = warm / rest
            lo, hi = PREDICTION_WARM_RATIO
            lines.append(
                f"| (d) open state warm-started at 0.8 to 1.0 of the closed state's outer count | "
                f"{warm:,} over {rest:,} = {ratio:.3f} | {'holds' if lo <= ratio <= hi else 'fails'} |"
            )
        else:
            lines.append(
                "| (d) open state warm-started at 0.8 to 1.0 of the closed state's outer count | states missing | not scored |"
            )
    return lines


def load_runs(root: Path, stems: dict[str, str]) -> dict[str, Run | None]:
    """The runs that have a run file and a summary; None for one that has not finished.

    Parameters
    ----------
    root : Path
        The directory of the run files.
    stems : dict[str, str]
        Scenario name to the stem of its run file and summary under ``root``.
    """
    runs: dict[str, Run | None] = {}
    for name, stem in stems.items():
        path = root / f"{stem}.h5"
        summary = root / f"{stem}.txt"
        if not path.exists() or not summary.exists() or summary.stat().st_size == 0:
            runs[name] = None
            continue
        runs[name] = Run(root, stem)
    return runs


def render(
    runs: dict[str, Run | None], coarse: dict[str, Run | None] | None
) -> tuple[str, dict[str, str]]:
    """The tables file's text and each table's text keyed by its marker name."""
    errors, mean_error = t7_errors()
    tables = {
        "reload": reload_table(runs),
        "flows": flows_table(runs),
        "hashes": hash_table(runs),
        "rows": rows_table(),
        "march": march_table(runs),
        "budgets": budget_table(runs),
        "accrual": accrual_table(runs),
        "sensors": sensor_table(runs, errors),
        "door": door_table(runs),
        "recovery": recovery_table(runs, errors, mean_error),
        "lingering": lingering_table(runs),
        "t7": t7_table(errors, mean_error),
        "grid": grid_table(runs, coarse),
        "predictions": predictions_table(runs),
    }
    head = [
        "# Step 7's tables: the four reference scenarios on the product grid",
        "",
        "Generated by `docs/reports/probe69/tables69.py` from the run files and summaries under "
        "`results/builder69/` and T7 of `docs/reports/probe59/tables59.md`; `--check` regenerates "
        "and compares, the copies in `docs/reports/phase4_scenarios_report.md` included. Do not "
        "edit by hand. Wall times were taken with four processes sharing the machine.",
        "",
    ]
    body: list[str] = []
    for name in MARKERS:
        body.extend(tables[name])
        body.append("")
    text = "\n".join(head + body)
    return text, {name: "\n".join(lines) for name, lines in tables.items()}


def splice(text: str, tables: dict[str, str]) -> str:
    """The report's text with each marked region replaced by its table."""
    for name, table in tables.items():
        begin = f"<!-- tables69 {name} begin -->"
        end = f"<!-- tables69 {name} end -->"
        if begin not in text or end not in text:
            continue
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{begin}" + chr(10) + table + chr(10) + f"{end}{tail}"
    return text


def report_copies(text: str) -> dict[str, str]:
    """The report's copies of the tables, keyed by marker name."""
    copies = {}
    for name in MARKERS:
        begin = f"<!-- tables69 {name} begin -->"
        end = f"<!-- tables69 {name} end -->"
        if begin in text and end in text:
            copies[name] = text[
                text.index(begin) + len(begin) : text.index(end)
            ].strip()
    return copies


def nearest_frame(run: Run, t: float) -> int:
    """The index of the frame nearest a time."""
    return int(np.argmin(np.abs(run.frames.times - t)))


def draw_room(ax: object, run: Run) -> None:
    """Obstacles and sensors on an axes."""
    import matplotlib.patches as patches
    import matplotlib.patheffects as patheffects

    for obstacle in run.config.obstacles:
        ax.add_patch(
            patches.Rectangle(
                (obstacle.x_start, obstacle.y_start),
                obstacle.x_end - obstacle.x_start,
                obstacle.y_end - obstacle.y_start,
                facecolor="0.75",
                edgecolor="0.3",
                linewidth=0.8,
                zorder=3,
            )
        )
    for sensor in run.config.sensors:
        ax.plot(
            sensor.x,
            sensor.y,
            marker="o",
            color="white",
            markeredgecolor="black",
            markersize=6,
            zorder=4,
        )
        ax.annotate(
            sensor.name,
            (sensor.x, sensor.y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="white",
            path_effects=[patheffects.withStroke(linewidth=2, foreground="black")],
            zorder=4,
        )
    ax.set_xlim(run.mesh.x[0], run.mesh.x[-1])
    ax.set_ylim(run.mesh.y[0], run.mesh.y[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def figures(runs: dict[str, Run | None], out: Path) -> list[Path]:
    """The step 7 figures (recovery maps, snapshots, sensor histories) under ``out``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    written: list[Path] = []
    clean = runs.get("clean_out")
    if clean is not None and clean.recovery is not None:
        record = clean.recovery
        for k in RECOVERY_MAP_CLASSES:
            field = record.local_recovery[k].copy()
            not_recovered = np.isinf(field)
            shown = np.where(np.isfinite(field), field, np.nan)
            fig, ax = plt.subplots(figsize=(10, 4.2))
            mesh = clean.mesh
            image = ax.pcolormesh(
                mesh.x,
                mesh.y,
                np.ma.masked_invalid(shown),
                cmap="viridis",
                shading="flat",
                vmin=0.0,
            )
            if not_recovered.any():
                ax.pcolormesh(
                    mesh.x,
                    mesh.y,
                    np.ma.masked_where(~not_recovered, np.ones_like(field)),
                    cmap="autumn",
                    shading="flat",
                    vmin=0,
                    vmax=1,
                )
            draw_room(ax, clean)
            fig.colorbar(image, ax=ax, label="last time at or above threshold (s)")
            size = clean.sizes_um[k]
            title = f"Clean-out, {size:g} um: local recovery time (ADR-014 C)"
            if not_recovered.any():
                title += f"; orange: not recovered by {record.t_final:.0f} s"
            ax.set_title(title, fontsize=10)
            path = out / f"phase4_cleanout_recovery_{size:g}um.png"
            fig.tight_layout()
            fig.savefig(path, dpi=130)
            plt.close(fig)
            written.append(path)
        fig, axes = plt.subplots(
            1, len(HISTORY_CLASSES), figsize=(11, 4.2), sharey=True
        )
        rate = record.air_change_rate
        for ax, k in zip(axes, HISTORY_CLASSES, strict=True):
            c0 = float(record.initial[k])
            t = clean.sensors.times
            for s, sensor in enumerate(clean.sensors.names):
                ax.semilogy(
                    t, clean.sensors.readings[:, k, s] / c0, linewidth=0.9, label=sensor
                )
            means = (
                np.array(
                    [
                        math.nan if b[k].current is None else b[k].current
                        for b in clean.frames.budgets
                    ]
                )
                / clean.free_area
                / c0
            )
            ax.semilogy(
                clean.frames.times,
                means,
                color="black",
                linewidth=1.4,
                label="free-area mean",
            )
            ax.semilogy(
                t,
                np.exp(-rate * t),
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="well-mixed exp(-N t)",
            )
            ax.axhline(
                1.0 / 100.0,
                color="0.5",
                linestyle=":",
                linewidth=1.0,
                label="100:1 (the threshold)",
            )
            ax.set_ylim(1e-6, 2.0)
            ax.set_xlim(0.0, float(t[-1]))
            ax.set_xlabel("t (s)")
            ax.set_title(f"{clean.sizes_um[k]:g} um", fontsize=10)
            ax.grid(True, which="both", linewidth=0.3)
        axes[0].set_ylabel("C / C0")
        axes[-1].legend(fontsize=7, loc="upper right")
        fig.suptitle(
            "Clean-out: the sensors, the free-area mean and the well-mixed line",
            fontsize=10,
        )
        path = out / "phase4_cleanout_sensors.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    for name, run in runs.items():
        if run is None:
            continue
        k = SNAPSHOT_CLASS
        if name == "clean_out":
            assert run.recovery is not None
            t_show = float(run.recovery.mean_time[k])
            if math.isnan(t_show):
                t_show = float(run.frames.times[-1])
        else:
            t_show = SNAPSHOT_TIMES[name]
        frame = nearest_frame(run, t_show)
        ratio = run.frames.fields[k][frame] / run.thresholds[k]
        shown = np.where(run.live, np.maximum(ratio, 1e-3), np.nan)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        image = ax.pcolormesh(
            run.mesh.x,
            run.mesh.y,
            np.ma.masked_invalid(shown),
            cmap="magma",
            shading="flat",
            norm=LogNorm(vmin=1e-3, vmax=max(1e2, float(np.nanmax(shown)))),
        )
        ax.contour(
            run.mesh.xc,
            run.mesh.yc,
            np.where(run.live, ratio, 0.0),
            levels=[1.0],
            colors="cyan",
            linewidths=0.8,
        )
        draw_room(ax, run)
        fig.colorbar(
            image, ax=ax, label="C over the class threshold (cyan: the threshold)"
        )
        ax.set_title(
            f"{TITLES[name]}, {run.sizes_um[k]:g} um at t = {run.frames.times[frame]:.2f} s "
            f"(step {int(run.frames.steps[frame])})",
            fontsize=10,
        )
        path = out / f"phase4_{name}_snapshot.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    return written


def main() -> int:
    """Write the tables (and figures), or with --check compare them with the committed files."""
    parser = argparse.ArgumentParser(description="ADR-014 step 7's tables")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument(
        "--splice",
        action="store_true",
        help="also write the tables into the report's marked regions",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="read the step 4 coarse runs (results/builder64/coarse/) instead",
    )
    parser.add_argument("--out", type=Path, default=TABLES)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    root, stems = RESULTS, {name: name for name in NAMES}
    if args.coarse:
        root, stems = COARSE, dict(COARSE_STEMS)
    runs = load_runs(root, stems)
    coarse = None if args.coarse else load_runs(COARSE, COARSE_STEMS)
    text, tables = render(runs, coarse)
    if args.check:
        failures = []
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != text:
            failures.append(f"{args.out} differs from the records")
        if REPORT.exists():
            copies = report_copies(REPORT.read_text(encoding="utf-8"))
            for name, copy in copies.items():
                if copy != tables[name]:
                    failures.append(
                        f"{REPORT}: the copy of table '{name}' differs from the records"
                    )
            for name in MARKERS:
                if name not in copies:
                    failures.append(f"{REPORT}: no copy of table '{name}'")
        for failure in failures:
            print(failure)
        print("tables69: " + ("differs" if failures else "matches"))
        return 1 if failures else 0
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    if args.splice and REPORT.exists():
        REPORT.write_text(
            splice(REPORT.read_text(encoding="utf-8"), tables), encoding="utf-8"
        )
        print(f"spliced {REPORT}")
    if args.figures:
        for path in figures(runs, args.figures_dir):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
