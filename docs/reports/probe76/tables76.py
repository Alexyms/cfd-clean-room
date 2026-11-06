"""Prompt 76: the latency report's tables and figures, from the run files (ADR-015 E).

Reads the four reference scenarios on 320x120 (results/builder76/), step 4's 80x30
runs (results/builder75/runs/) and step 1's arm runs (results/builder72/), all untracked
and on this machine, and writes tables76.md, the copies in
docs/reports/phase5_alert_report.md and the arrival-map figures.

The layouts are read as ADR-015 D and E state them: the units are (scenario, event,
class) over the three event scenarios; the candidates come from the door scenario's
configuration, whose intervals open every opening the three scenarios have; the mount,
the standoff, the count and the rule are the product file's placement section. A layout's
latency comes from the arrival maps by lookup. A file sensor's peak and time above come
from its every-step readings, a searched cell's from the stored frames, and the tables say
which. The coverage rule's time is this report's, WITHIN: the product file states none.

    python docs/reports/probe76/tables76.py --splice --figures    # write everything
    python docs/reports/probe76/tables76.py --check               # regenerate and compare
"""

import argparse
import json
import math
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "probe69"))

import tables69  # noqa: E402 -- follows sys.path.insert

from src.config import MEAN, MOUNT_FREE, WORST_CASE, SimConfig  # noqa: E402
from src.io_manager import OPEN_STEP, RunFile  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.monitor import sensor_cells  # noqa: E402
from src.placement import (  # noqa: E402
    Layout,
    RunArrival,
    Score,
    Units,
    arrival_from_run,
    candidates,
    detection_units,
    exclude_units,
    given_layout,
    latency,
    rank,
    score,
    search,
)

RESULTS = ROOT / "results" / "builder76"
COARSE = ROOT / "results" / "builder75" / "runs"
COARSE_STEMS = {
    "door_leak": "door",
    "filter_breach": "breach",
    "equipment_dust": "dust",
}
SWEEP = ROOT / "results" / "builder72"
GATE = ROOT / "results" / "builder69"
TABLES = HERE / "tables76.md"
REPORT = ROOT / "docs" / "reports" / "phase5_alert_report.md"
GATE_REPORT = ROOT / "docs" / "reports" / "phase5_alert_system_report.md"
HASHES = ROOT / "results" / "builder62"
# The six rows the Phase 4 gate reproduced, its table H2.
HASH_CASES = (
    "val001_80x40",
    "val001_80x40_stretched",
    "val002_20x20",
    "val002_40x40",
    "val002_80x80",
    "product_80x30",
)
# Phase 5 step 0's coverage, the baseline the gate is measured from (prompt 71).
COVERAGE_BASE = ROOT / "results" / "builder71" / "after" / "coverage.json"
COVERAGE_HEAD = RESULTS / "coverage.json"
FIGURES = ROOT / "docs" / "reports"
PRODUCT = ROOT / "configs" / "clean_room_default.yaml"

EVENT_NAMES = ("door_leak", "filter_breach", "equipment_dust")
NAMES = (*EVENT_NAMES, "clean_out")
TITLES = {
    "door_leak": "the door",
    "filter_breach": "the breach",
    "equipment_dust": "the dust",
}
REPORT_CLASSES = (0, 4)  # 0.1 um and 5 um
FREE_COUNTS = (1, 2, 3, 4)
ARMS = ("control", "i10", "max_nut", "min_nut", "sc02", "widest")
# The report's time for the coverage rule, seconds: the REQ-A05 test's. The product
# file states none, so this is the report's reading, not a requirement.
WITHIN = 10.0
RESOLVED_FLOOR = 1.0e-2
FILE = "the file's four"
MOUNTED = "searched, mountable"
FREE = "free optimum"
MARKERS = (
    "reload",
    "flows",
    "states",
    "march",
    "budgets",
    "alerts",
    "layouts",
    "curve",
    "headline",
    "units",
    "recovery",
    "stability",
    "arms",
    "predictions",
)
GATE_MARKERS = ("hashes", "bitwise", "coverage")


class Run(tables69.Run):
    """A scenario's records, as the gate's, with its alert table."""

    def __init__(self, root: Path, name: str) -> None:
        super().__init__(root, name)
        self.alerts = self.file.read_alerts()


def where(mesh: Mesh, cell: tuple[int, int]) -> tuple[float, float]:
    """A cell's centre (x, y), metres."""
    j, i = cell
    return float(mesh.xc[i]), float(mesh.yc[j])


def cell_at(mesh: Mesh, x: float, y: float) -> tuple[int, int]:
    """The cell containing a point, the one below or left of it on a face (ADR-014 C)."""
    i = int(np.searchsorted(mesh.x, x, side="left")) - 1
    j = int(np.searchsorted(mesh.y, y, side="left")) - 1
    return max(j, 0), max(i, 0)


def mapped(
    source: Mesh, target: Mesh, cells: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    """The cells of TARGET holding the centres of CELLS of SOURCE; ValueError on SOLID.

    Two cells of a finer grid in one cell of a coarser are one reading there, so a
    layout read on a coarser grid can hold fewer cells than it has sensors.
    """
    out: list[tuple[int, int]] = []
    for cell in cells:
        x, y = where(source, cell)
        found = cell_at(target, x, y)
        if target.cell_type[found] == SOLID:
            raise ValueError(f"({x}, {y}) is SOLID on the target grid")
        if found not in out:
            out.append(found)
    return tuple(out)


def places(mesh: Mesh, cells: tuple[tuple[int, int], ...]) -> str:
    """Cell centres as text."""
    return ", ".join("({:.3f}, {:.3f})".format(*where(mesh, cell)) for cell in cells)


def text_time(value: float) -> str:
    """A time, or "inf" for none."""
    return "inf" if math.isinf(value) else f"{value:.3f}"


def detected_by_scenario(
    units: Units, cells: tuple[tuple[int, int], ...]
) -> dict[str, tuple[int, int]]:
    """Per scenario, (units a layout detects, units there are)."""
    out: dict[str, list[int]] = {}
    for u, (name, _, _) in enumerate(units.labels):
        row = out.setdefault(name, [0, 0])
        row[1] += 1
        row[0] += any(np.isfinite(units.latencies[u, j, i]) for j, i in cells)
    return {name: (row[0], row[1]) for name, row in out.items()}


def working_cells(units: Units, cells: tuple[tuple[int, int], ...]) -> int:
    """How many of a layout's cells set some unit's latency, the least over the layout.

    A search adds spare sensors when a cover needs fewer than the count, and a
    spare can detect nothing the others do not. A tie counts every cell in it.
    """
    table = np.stack([units.latencies[:, j, i] for j, i in cells], axis=1)
    least = table.min(axis=1, keepdims=True)
    return int(((table == least) & np.isfinite(table)).any(axis=0).sum())


def detected_text(counts: dict[str, tuple[int, int]]) -> str:
    """Detected counts as text, door first, the scenarios present only."""
    return ", ".join(
        f"{TITLES[name].removeprefix('the ')} {counts[name][0]} of {counts[name][1]}"
        for name in EVENT_NAMES
        if name in counts
    )


def order(entries: dict[str, Score], rule: str) -> str:
    """The layouts by one rule alone, best first, with the ties said."""
    if rule == WORST_CASE:
        values = {name: s.worst for name, s in entries.items()}
    elif rule == MEAN:
        values = {name: s.mean for name, s in entries.items()}
    else:
        values = {name: -(s.coverage or 0.0) for name, s in entries.items()}
    ranked = sorted(values, key=lambda name: values[name])
    out = ranked[0]
    for before, name in pairwise(ranked):
        out += (" = " if values[before] == values[name] else " < ") + name
    return out


class Analysis:
    """The units, the candidates and the layouts on the product grid's runs."""

    def __init__(self, runs: dict[str, Run]) -> None:
        door = runs["door_leak"]
        self.runs = runs
        self.config = door.config
        self.mesh = door.mesh
        self.spec = SimConfig(PRODUCT).placement
        assert self.spec is not None
        self.arrivals: dict[str, RunArrival] = {
            name: arrival_from_run(runs[name].file) for name in EVENT_NAMES
        }
        self.units = detection_units(self.arrivals)
        self.sensors = sensor_cells(self.config, self.mesh)
        self.sensor_names = tuple(sensor.name for sensor in self.config.sensors)
        self.mounted = candidates(
            self.mesh, self.config, self.spec.mount, self.spec.standoff
        )
        self.free = candidates(
            self.mesh, self.config, (MOUNT_FREE,), self.spec.standoff
        )
        self.found = search(
            self.units, self.mounted, self.spec.count, self.spec.rule, within=WITHIN
        )
        self.by_mean = search(
            self.units, self.mounted, self.spec.count, MEAN, within=WITHIN
        )
        self.kept = exclude_units(self.units, self.found.uncovered)
        self.file_all = given_layout(self.units, self.sensors, within=WITHIN)
        self.file_kept = given_layout(self.kept, self.sensors, within=WITHIN)
        self.curve = {
            n: search(self.units, self.free, n, WORST_CASE, within=WITHIN)
            for n in FREE_COUNTS
        }
        self.free_mean = search(
            self.units, self.free, self.spec.count, MEAN, within=WITHIN
        )
        self.free_on_kept = search(
            self.kept, self.free, self.spec.count, WORST_CASE, within=WITHIN
        )
        self.t7, self.t7_mean, self.t7_air = tables69.t7_errors()
        self.t7_largest = max(abs(value) for value in self.t7.values())
        self._widths: dict[str, list[np.ndarray]] = {}

    def widths(self, scenario: str, k: int, y: float) -> str:
        """The sweep's width at the threshold on a height's row, over step 1's 160x60 arms."""
        if scenario not in self._widths:
            rows = []
            for arm in ARMS:
                path = SWEEP / f"{scenario}_{arm}_160x60.h5"
                summary = path.with_suffix(".txt")
                if not (
                    summary.exists()
                    and "wall time:" in summary.read_text(encoding="utf-8")
                ):
                    continue
                run = RunFile(path)
                config = SimConfig.from_dict(run.read_config().mapping)
                mesh = Mesh(config)
                live = mesh.cell_type != SOLID
                frames = run.read_frames()
                thresholds = config.class_thresholds("the sweep's width")
                table = np.stack(
                    [
                        ((frames.fields[c] >= thresholds[c]) & live)
                        .sum(axis=2)
                        .max(axis=0)
                        * float(mesh.dx_cell[0])
                        for c in range(len(thresholds))
                    ]
                )
                rows.append(table)
            self._widths[scenario] = rows
            self._y = np.asarray(mesh.y)
        j = int(np.searchsorted(self._y, y, side="left")) - 1
        values = [float(table[k, max(j, 0)]) for table in self._widths[scenario]]
        return f"{min(values):.2f} to {max(values):.2f}"


def reload_table(runs: dict[str, Run]) -> list[str]:
    """R1: each run file read back, its alert table and its arrival maps."""
    lines = [
        "### R1. The run files reloaded (RunFile, every group read back)",
        "",
        "| Run file | Commit | Started | Concurrent | Config SHA-256 (12) | Flow states | Frames (n, ny, nx) | Sensor readings (steps, classes, sensors) | Alerts | /arrival/ (events, classes, ny, nx), t_final (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in NAMES:
        run = runs[name]
        record = run.config_record
        shape = run.frames.fields[0].shape
        readings = run.sensors.readings.shape
        arrival = arrival_from_run(run.file)
        lines.append(
            f"| {name}.h5 | {record.commit} | {record.started} | {record.concurrent} | "
            f"{record.sha256[:12]} | {len(run.file.flow_fingerprints())} | {shape} | "
            f"{readings} | {run.alerts.step_on.size} | "
            f"{arrival.resolution}: {arrival.maps.shape}, {arrival.t_final:.4f} |"
        )
    return lines


def states_table(runs: dict[str, Run]) -> list[str]:
    """F2: each flow state against the gate's run of the same fingerprint."""
    lines = [
        "### F2. The flow states against the gate's (prompt 69, the same files before the placement section)",
        "",
        "| Run | State | Face hash (first 16) | The gate's | Largest face difference (m/s) |",
        "|---|---|---|---|---|",
    ]
    for name in NAMES:
        run = runs[name]
        gate_path = GATE / f"{name}.h5"
        gate = RunFile(gate_path) if gate_path.exists() else None
        for fingerprint in run.file.flow_fingerprints():
            state = run.file.read_flow_state(fingerprint).state
            ours = tables69.face_hash(state.u, state.v)[:16]
            theirs, gap = "no such state", "-"
            if gate is not None and fingerprint in gate.flow_fingerprints():
                other = gate.read_flow_state(fingerprint).state
                if other.u.shape != state.u.shape:
                    lines.append(
                        f"| {name} | {fingerprint[:12]} | {ours} | another grid | - |"
                    )
                    continue
                theirs = tables69.face_hash(other.u, other.v)[:16]
                gap = f"{max(np.abs(state.u - other.u).max(), np.abs(state.v - other.v).max()):.2e}"
            lines.append(f"| {name} | {fingerprint[:12]} | {ours} | {theirs} | {gap} |")
    return lines


def bitwise_table(runs: dict[str, Run]) -> list[str]:
    """X1: each run's stored records against the gate's run of the same scenario."""
    lines = [
        "### X1. The runs against the gate's (prompt 69), the same scenario files before Phase 5's modules: stored frames, budgets and sensor readings",
        "",
        "| Run | Frames here, the gate's | Fields, every class, bitwise | Budgets, bitwise | Sensor readings, bitwise |",
        "|---|---|---|---|---|",
    ]
    for name in NAMES:
        run = runs[name]
        path = GATE / f"{name}.h5"
        if not path.exists():
            lines.append(f"| {name} | no such run | | | |")
            continue
        gate = RunFile(path)
        frames = gate.read_frames()
        readings = gate.read_sensor_readings()
        same_fields = sorted(frames.fields) == sorted(run.frames.fields) and all(
            frames.fields[k].tobytes() == run.frames.fields[k].tobytes()
            for k in frames.fields
        )
        same_budgets = frames.budgets == run.frames.budgets
        same_readings = readings.readings.tobytes() == run.sensors.readings.tobytes()
        lines.append(
            f"| {name} | {len(run.frames.steps)}, {len(frames.steps)} | "
            f"{'equal' if same_fields else 'DIFFER'} | "
            f"{'equal' if same_budgets else 'DIFFER'} | "
            f"{'equal' if same_readings else 'DIFFER'} |"
        )
    return lines


def hashes_table() -> list[str]:
    """H1: the six rows the Phase 4 gate reproduced, solved at this commit."""
    lines = [
        "### H1. The hashes of record at the gate's source (hash62.py, against the Phase 4 gate's _69 records)",
        "",
        "| Case | Commit | Stop | Outer | Hash (first 16) | Equal to _69 | Wall (s) | Processes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in HASH_CASES:
        path = HASHES / f"hash_{case}_76.json"
        if not path.exists():
            lines.append(f"| {case} | | not run | | | | | |")
            continue
        new = json.loads(path.read_text(encoding="utf-8"))
        old = json.loads((HASHES / f"hash_{case}_69.json").read_text(encoding="utf-8"))
        lines.append(
            f"| {case} | {new['commit']} | {new['stop']} | {new['outer']:,} | "
            f"{new['face_hash'][:16]} | {'yes' if old['face_hash'] == new['face_hash'] else 'NO'} | "
            f"{new['wall_seconds']:.0f} | {new['concurrent_processes']} |"
        )
    return lines


def coverage_table() -> list[str]:
    """Cv1: statements and branches apart, at the gate against Phase 5 step 0."""
    lines = [
        "### Cv1. Coverage, statements and branches apart, the gate against Phase 4's after step 0",
        "",
    ]
    if not COVERAGE_HEAD.exists():
        return [*lines, "not run"]
    head = json.loads(COVERAGE_HEAD.read_text(encoding="utf-8"))
    base = json.loads(COVERAGE_BASE.read_text(encoding="utf-8"))
    h, b = head["totals"], base["totals"]
    missing = [
        f"`{name.replace(chr(92), '/')}` {arc[0]}->{arc[1]}"
        for name, entry in head["files"].items()
        for arc in entry.get("missing_branches", [])
    ]
    uncovered = [
        name.replace(chr(92), "/")
        for name, entry in head["files"].items()
        if entry["missing_lines"]
    ]
    lines += [
        "| Measure | The gate | After step 0 (prompt 71) | Covered count, change | Percentage, change (points) |",
        "|---|---|---|---|---|",
        f"| Statements covered | {h['covered_lines']:,} of {h['num_statements']:,} "
        f"({100 * h['covered_lines'] / h['num_statements']:.2f}%) | {b['covered_lines']:,} of "
        f"{b['num_statements']:,} ({100 * b['covered_lines'] / b['num_statements']:.2f}%) | "
        f"+{h['covered_lines'] - b['covered_lines']:,} | "
        f"{100 * (h['covered_lines'] / h['num_statements'] - b['covered_lines'] / b['num_statements']):+.2f} |",
        f"| Branches covered | {h['covered_branches']:,} of {h['num_branches']:,} "
        f"({100 * h['covered_branches'] / h['num_branches']:.2f}%) | {b['covered_branches']:,} of "
        f"{b['num_branches']:,} ({100 * b['covered_branches'] / b['num_branches']:.2f}%) | "
        f"+{h['covered_branches'] - b['covered_branches']:,} | "
        f"{100 * (h['covered_branches'] / h['num_branches'] - b['covered_branches'] / b['num_branches']):+.2f} |",
        "",
        f"Files with an uncovered statement: {', '.join(uncovered) or 'none'}. Uncovered branch "
        f"arms: {', '.join(missing) or 'none'}.",
    ]
    return lines


def budget_table(runs: dict[str, Run]) -> list[str]:
    """B1: per run and class, the relative residual at the end and its worst stored frame."""
    lines = [
        "### B1. Budget closure (residual over the particles supplied), at the end and at the worst stored frame",
        "",
        "| Run | Class (um) | Relative at the end | Worst relative over the stored frames | at t (s) | Inside REQ-T05's 1e-4 at every frame |",
        "|---|---|---|---|---|---|",
    ]
    worst_all = 0.0
    for name in NAMES:
        run = runs[name]
        for k, size in enumerate(run.sizes_um):
            _, relative = tables69.frame_relatives(run, k)
            worst, at = tables69.worst_frame(run, k)
            worst_all = max(worst_all, abs(worst))
            lines.append(
                f"| {name} | {size:g} | {relative[-1]:.3e} | {worst:.3e} | {at:.2f} | "
                f"{'yes' if abs(worst) < 1.0e-4 else 'NO'} |"
            )
    lines += [
        "",
        f"The largest magnitude over every class of every run at every stored frame is "
        f"{worst_all:.3e}, inside REQ-T05's 1e-4 by {1.0e-4 / worst_all:.0f} times.",
    ]
    return lines


def alerts_table(analysis: Analysis) -> list[str]:
    """A1: each run's alert table and the monitor against the map at the file's sensors."""
    lines = [
        "### A1. The alert tables, and the monitor's latency against the map's at the file's sensors",
        "",
        "| Run | Alerts | Still open at the end | Sensor, class and event triples | The two agree on |",
        "|---|---|---|---|---|",
    ]
    for name in NAMES:
        run = analysis.runs[name]
        table = run.alerts
        arrival = arrival_from_run(run.file)
        total = agree = 0
        for e, onset in enumerate(arrival.onsets):
            for k in range(arrival.maps.shape[1]):
                for s, (j, i) in enumerate(analysis.sensors):
                    rows = (table.size_class == k) & (table.sensor == s)
                    open_at = (table.t_on <= onset) & (
                        (table.step_off == OPEN_STEP) | (table.t_off > onset)
                    )
                    later = table.t_on[rows & (table.t_on >= onset)]
                    if (rows & open_at).any():
                        monitor = float(onset)
                    elif later.size:
                        monitor = float(later.min())
                    else:
                        monitor = math.inf
                    total += 1
                    agree += monitor == float(arrival.maps[e, k, j, i])
        lines.append(
            f"| {name} | {table.step_on.size} | {(table.step_off == OPEN_STEP).sum()} | "
            f"{total} | {agree} |"
        )
    return lines


def score_row(name: str, layout: Layout, mesh: Mesh, units: Units) -> str:
    """One row of the layouts' tables."""
    s = layout.score
    masks = (
        "-"
        if layout.distinct_masks is None
        else f"{layout.distinct_masks} / {layout.maximal_masks}"
    )
    return (
        f"| {name} | {places(mesh, layout.cells)} | {layout.rule or '-'} | {layout.method} | "
        f"{'yes' if layout.exact else 'no'} | {len(layout.units)} | {len(layout.uncovered)} | "
        f"{text_time(s.worst)} | {s.mean:.3f} | {s.undetected} | {s.coverage:.3f} | {masks} | "
        f"{detected_text(detected_by_scenario(units, layout.cells))} | "
        f"{working_cells(units, layout.cells)} of {len(layout.cells)} |"
    )


LAYOUT_HEAD = (
    "| Layout | Cell centres (x, y), m | Rule | Method | Exact | Units scored | Uncovered | "
    "Worst case (s) | Mean (s) | Undetected | Coverage within 10 s | Distinct / maximal "
    "masks | Detected, of all fifteen | Cells setting a latency |"
)
LAYOUT_RULE = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def layouts_table(analysis: Analysis) -> list[str]:
    """L1: the configurations, over the units each is scored on."""
    a = analysis
    lines = [
        f"### L1. The layouts on 320x120: mount {list(a.spec.mount)} at {a.spec.standoff} m, "
        f"{len(a.mounted):,} mountable and {len(a.free):,} free candidates; the means are capped "
        "at each unit's remaining run time",
        "",
        LAYOUT_HEAD,
        LAYOUT_RULE,
        score_row(f"{FILE}, all fifteen units", a.file_all, a.mesh, a.units),
        score_row(f"{FILE}, the mountable set's units", a.file_kept, a.mesh, a.units),
        score_row(f"{MOUNTED}, worst-case rule", a.found, a.mesh, a.units),
        score_row(f"{MOUNTED}, mean rule", a.by_mean, a.mesh, a.units),
        score_row(f"{FREE}, worst-case rule", a.curve[a.spec.count], a.mesh, a.units),
        score_row(f"{FREE}, mean rule", a.free_mean, a.mesh, a.units),
        score_row(
            f"{FREE}, worst-case rule, the mountable set's units",
            a.free_on_kept,
            a.mesh,
            a.units,
        ),
        "",
        "Uncovered by the mountable set: "
        + (", ".join(f"{s} {e} class {k}" for s, e, k in a.found.uncovered) or "none")
        + ".",
    ]
    reach = a.units.latencies[:, a.mounted[:, 0], a.mounted[:, 1]]
    door = [u for u, label in enumerate(a.units.labels) if label[0] == "door_leak"]
    lines.append(
        f"Mountable candidates that detect a door unit: "
        f"{int(np.isfinite(reach[door]).any(axis=0).sum())} of {len(a.mounted):,}."
    )
    named = {
        FILE: a.file_kept,
        f"{MOUNTED} (worst case)": a.found,
        f"{MOUNTED} (mean)": a.by_mean,
        f"{FREE} (worst case)": a.free_on_kept,
    }
    scores = {name: layout.score for name, layout in named.items()}
    names = {id(layout): name for name, layout in named.items()}
    ranked = rank(list(named.values()))
    lines += [
        "",
        f"Over the {len(a.kept.labels)} units the mountable set covers, the order per rule, "
        "best first:",
        "",
        f"- worst case: {order(scores, WORST_CASE)};",
        f"- mean: {order(scores, MEAN)};",
        f"- coverage within {WITHIN:g} s: {order(scores, 'coverage')};",
        "- section D's ranking, the worst case with ties by the mean: "
        + ", ".join(names[id(layout)] for layout in ranked)
        + ".",
    ]
    return lines


def curve_table(analysis: Analysis) -> list[str]:
    """L2: the free optimum for one to four sensors, the bound."""
    a = analysis
    lines = [
        "### L2. The free optimum for one to four sensors, the worst-case rule (the bound)",
        "",
        LAYOUT_HEAD,
        LAYOUT_RULE,
    ]
    for n in FREE_COUNTS:
        lines.append(score_row(f"{n} sensors", a.curve[n], a.mesh, a.units))
    return lines


def headline_table(analysis: Analysis) -> list[str]:
    """U2: per unit, each configuration's latency, the least over its cells."""
    a = analysis
    layouts = (
        (FILE, a.file_all),
        (MOUNTED, a.found),
        (FREE, a.curve[a.spec.count]),
    )
    lines = [
        "### U2. Per unit, the layout's latency, the least over its cells (s; never with the run's end)",
        "",
        "| Scenario | Event | Class (um) | " + " | ".join(n for n, _ in layouts) + " |",
        "|---|---|---|" + "---|" * len(layouts),
    ]
    sizes = [size * 1.0e6 for size in a.config.particle_sizes]
    for u, (name, event, k) in enumerate(a.units.labels):
        end = a.arrivals[name].t_final
        cells = []
        for _, layout in layouts:
            values = [a.units.latencies[u, j, i] for j, i in layout.cells]
            best = float(min(values))
            cells.append(f"never (to {end:g})" if math.isinf(best) else f"{best:.3f}")
        lines.append(f"| {name} | {event} | {sizes[k]:g} | " + " | ".join(cells) + " |")
    return lines


def read_series(
    a: Analysis, name: str, k: int, cell: tuple[int, int], s: int | None
) -> tuple[float, float, str]:
    """A cell's peak over threshold and its time above, from the steps or the frames."""
    run = a.runs[name]
    threshold = float(run.thresholds[k])
    if s is not None:
        times, series, source = (
            run.sensors.times,
            run.sensors.readings[:, k, s],
            "steps",
        )
    else:
        j, i = cell
        times, series, source = (
            run.frames.times,
            run.frames.fields[k][:, j, i],
            "frames",
        )
    above = series >= threshold
    return (
        float(series.max() / threshold),
        float(np.diff(times)[above[1:]].sum()),
        source,
    )


def peak_text(ratio: float) -> str:
    """A peak over threshold, or the floor's words (ADR-015 B)."""
    if ratio < RESOLVED_FLOOR:
        return f"below the resolved plume (< {RESOLVED_FLOOR:g})"
    return f"{ratio:.3g}"


def units_table(a: Analysis, classes: tuple[int, ...] | None) -> list[str]:
    """U1: per scenario, event, class and sensor, for each configuration."""
    configurations = (
        (FILE, a.sensors, a.sensor_names),
        (
            MOUNTED,
            a.found.cells,
            tuple(f"S{n + 1}" for n in range(len(a.found.cells))),
        ),
        (
            FREE,
            a.curve[a.spec.count].cells,
            tuple(f"F{n + 1}" for n in range(len(a.curve[a.spec.count].cells))),
        ),
    )
    which = (
        "every class"
        if classes is None
        else "0.1 um and 5 um; every class in tables76.md"
    )
    lines = [
        f"### U1. Per scenario, event, class and sensor, for each configuration ({which}). A file "
        "sensor's peak and time above are from its every-step readings, a searched cell's from "
        "the stored frames (every 100 steps, about 0.36 s). The time error is T7's (160x60, one "
        "direction), measured at the file's sensors only. The width is the plume's largest at "
        "the threshold on the cell's row over step 1's 160x60 arms, read at frame resolution.",
        "",
        "| Scenario | Event | Class (um) | Configuration | Sensor | Latency (s) | Peak over threshold | Time above (s) | Read from | T7 time error | Sweep's width on the row (m) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    sizes = [size * 1.0e6 for size in a.config.particle_sizes]
    table_by_name = {
        name: latency(a.arrivals[name].maps, a.arrivals[name].onsets)
        for name in EVENT_NAMES
    }
    for name in EVENT_NAMES:
        arrival = a.arrivals[name]
        for e, event in enumerate(arrival.event_names):
            for k, size in enumerate(sizes):
                if classes is not None and k not in classes:
                    continue
                for title, cells, labels in configurations:
                    for position, (cell, label) in enumerate(
                        zip(cells, labels, strict=True)
                    ):
                        j, i = cell
                        value = float(table_by_name[name][e, k, j, i])
                        text = (
                            f"never (to {arrival.t_final:g})"
                            if math.isinf(value)
                            else f"{value:.3f}"
                        )
                        s = position if title == FILE else None
                        ratio, above, source = read_series(a, name, k, cell, s)
                        t7 = (
                            f"{a.t7[label] * 100:.2f}%"
                            if label in a.t7
                            else f"up to {a.t7_largest * 100:.2f}% (T7's largest)"
                        )
                        _, y = where(a.mesh, cell)
                        lines.append(
                            f"| {name} | {event} | {size:g} | {title} | {label} "
                            f"{where_text(a, cell)} | {text} | {peak_text(ratio)} | "
                            f"{above:.2f} | {source} | {t7} | {a.widths(name, k, y)} |"
                        )
    return lines


def where_text(a: Analysis, cell: tuple[int, int]) -> str:
    """A cell's centre in brackets."""
    return "({:.3f}, {:.3f})".format(*where(a.mesh, cell))


def recovery_table(a: Analysis) -> list[str]:
    """C1: each layout's clean-out recovery times, the clearing companion (ADR-014 C)."""
    record = a.runs["clean_out"].recovery
    assert record is not None
    configurations: list[tuple[str, tuple[tuple[int, int], ...]]] = [
        (FILE, a.sensors),
        (MOUNTED, a.found.cells),
    ]
    configurations += [(f"{FREE}, {n}", a.curve[n].cells) for n in FREE_COUNTS]
    sizes = [size * 1.0e6 for size in a.config.particle_sizes]
    lines = [
        f"### C1. The clean-out: each layout's recovery, the last time each cell was at or above "
        f"its threshold (the run went to {record.t_final:g} s; inf is still above at the end). "
        f"The well-mixed 100:1 time is {record.well_mixed_time:.4f} s, the file's room recovery "
        f"time, the first step below and staying below, {record.room_recovery_time:.4f} s. "
        f"T7's time error at the file's sensors is up to "
        f"{a.t7_largest * 100:.2f}%, near_door {abs(a.t7['near_door']) * 100:.2f}%.",
        "",
        "| Layout | Class (um) | Cells' recovery (s) | The layout's recovery, its last cell (s) |",
        "|---|---|---|---|",
    ]
    for title, cells in configurations:
        for k in REPORT_CLASSES:
            values = [float(record.local_recovery[k, j, i]) for j, i in cells]
            lines.append(
                f"| {title} | {sizes[k]:g} | "
                + ", ".join(text_time(v) for v in values)
                + f" | {text_time(max(values))} |"
            )
    return lines


def stability_table(a: Analysis) -> list[str]:
    """K1: the 80x30 and 320x120 searches, same rule and mount, and each on the other's maps."""
    source = {name: COARSE / f"{stem}.h5" for name, stem in COARSE_STEMS.items()}
    door = RunFile(source["door_leak"])
    config = SimConfig.from_dict(door.read_config().mapping)
    mesh = Mesh(config)
    arrivals = {name: arrival_from_run(path) for name, path in source.items()}
    units = detection_units(arrivals)
    mounted = candidates(mesh, config, a.spec.mount, a.spec.standoff)
    free = candidates(mesh, config, (MOUNT_FREE,), a.spec.standoff)
    found = search(units, mounted, a.spec.count, a.spec.rule, within=WITHIN)
    kept = exclude_units(units, found.uncovered)
    grid_sensors = sensor_cells(config, mesh)
    free_found = search(kept, free, a.spec.count, WORST_CASE, within=WITHIN)
    lines = [
        "### K1. The searches on 80x30 and 320x120, the same rule and mount, each layout scored on both grids over the units the grid's mountable search covers; a layout from the other grid is read at the cells holding its centres",
        "",
        "| Grid | Layout | Cell centres (x, y), m | Mountable on this grid | Worst case (s) | Mean (s) | Undetected | Coverage within 10 s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    fine_on_coarse = mapped(a.mesh, mesh, a.found.cells)
    coarse_on_fine = mapped(mesh, a.mesh, found.cells)
    entries = {
        "80x30": (
            mesh,
            kept,
            mounted,
            {
                "the file's four": grid_sensors,
                "searched here": found.cells,
                "searched on 320x120": fine_on_coarse,
                "free optimum here": free_found.cells,
            },
        ),
        "320x120": (
            a.mesh,
            a.kept,
            a.mounted,
            {
                "the file's four": a.sensors,
                "searched here": a.found.cells,
                "searched on 80x30": coarse_on_fine,
                "free optimum here": a.free_on_kept.cells,
            },
        ),
    }
    orders = []
    for grid, (grid_mesh, grid_units, pool, layouts) in entries.items():
        scores = {}
        for name, cells in layouts.items():
            s = score(grid_units, cells, within=WITHIN)
            scores[name] = s
            lines.append(
                f"| {grid} | {name} | {places(grid_mesh, cells)} | "
                f"{mountable_text(cells, pool)} | {text_time(s.worst)} | "
                f"{s.mean:.3f} | {s.undetected} of {len(grid_units.labels)} | {s.coverage:.3f} |"
            )
        orders.append(
            f"- {grid}: worst case {order(scores, WORST_CASE)}; mean {order(scores, MEAN)}; "
            f"coverage within {WITHIN:g} s {order(scores, 'coverage')}."
        )
    lines += [
        "",
        f"Uncovered by the mountable set on 80x30: {len(found.uncovered)} units "
        f"({', '.join(sorted({label[0] for label in found.uncovered})) or 'none'}); on 320x120: "
        f"{len(a.found.uncovered)} ({', '.join(sorted({label[0] for label in a.found.uncovered})) or 'none'}).",
        "",
        "The order per rule, best first:",
        "",
        *orders,
    ]
    return lines


def arms_table(a: Analysis) -> list[str]:
    """K2: the searched 320x120 layout on step 1's arms, frame-resolved and flagged."""
    lines = [
        "### K2. The searched 320x120 layout on step 1's arm runs (arrival_from_run: frame-resolved, flagged), read at the cells holding its centres, beside the file's four and the arm's own mountable search, over the units the arm's own search covers",
        "",
        "| Arm | Grid | Scenarios run | Frame-resolved | Layout | Mountable on this grid | Worst case (s) | Mean (s) | Undetected | Detected, of the arm's units | Own search's rule, method |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    orders = []
    rows = [(arm, "160x60") for arm in ARMS] + [("min_nut", "320x120")]
    for arm, grid in rows:
        arrivals = {}
        config_door = None
        for name in EVENT_NAMES:
            path = SWEEP / f"{name}_{arm}_{grid}.h5"
            summary = path.with_suffix(".txt")
            if summary.exists() and "wall time:" in summary.read_text(encoding="utf-8"):
                arrivals[name] = arrival_from_run(path)
                run = RunFile(path)
                if config_door is None or name == "door_leak":
                    config_door = SimConfig.from_dict(run.read_config().mapping)
        if not arrivals:
            lines.append(f"| {arm} | {grid} | none | | | | | | | | |")
            continue
        assert config_door is not None
        mesh = Mesh(config_door)
        if "door_leak" not in arrivals:
            # The door's run is capped at this arm, but its openings are the room's;
            # the door's configuration of the arm gives the same candidates.
            config_door = SimConfig(
                SWEEP / "configs" / f"scenario_door_leak_{arm}_{grid}.yaml"
            )
        units = detection_units(arrivals)
        mounted = candidates(mesh, config_door, a.spec.mount, a.spec.standoff)
        own = search(units, mounted, a.spec.count, a.spec.rule)
        kept = exclude_units(units, own.uncovered)
        layouts = {
            "searched on 320x120": mapped(a.mesh, mesh, a.found.cells),
            "the file's four": sensor_cells(config_door, mesh),
            "the arm's own": own.cells,
        }
        scores = {}
        for title, cells in layouts.items():
            s = score(kept, cells, within=WITHIN)
            scores[title] = s
            lines.append(
                f"| {arm} | {grid} | {detected_scenarios(arrivals)} | "
                f"{'yes' if s.frame_resolved else 'no'} | {title} | "
                f"{mountable_text(cells, mounted)} | {text_time(s.worst)} | "
                f"{s.mean:.3f} | {s.undetected} of {len(kept.labels)} | "
                f"{detected_text(detected_by_scenario(units, cells))} | "
                f"{own.rule}, {own.method} |"
            )
        orders.append(
            f"- {arm} ({grid}): worst case {order(scores, WORST_CASE)}; mean "
            f"{order(scores, MEAN)}."
        )
    lines += ["", "The order per rule, best first:", "", *orders]
    return lines


def mountable_text(cells: tuple[tuple[int, int], ...], pool: np.ndarray) -> str:
    """How many of a layout's cells are candidates on the grid it is read on."""
    allowed = {(int(j), int(i)) for j, i in pool}
    return f"{sum(cell in allowed for cell in cells)} of {len(cells)}"


def detected_scenarios(arrivals: dict[str, RunArrival]) -> str:
    """The scenarios an arm ran."""
    return ", ".join(TITLES[name].removeprefix("the ") for name in arrivals)


def predictions_table(a: Analysis) -> list[str]:
    """P1: prompt 76's predictions against the measurements."""
    door = {label for label in a.units.labels if label[0] == "door_leak"}
    rest = [label for label in a.units.labels if label[0] != "door_leak"]
    reach = a.units.latencies[:, a.mounted[:, 0], a.mounted[:, 1]]
    door_rows = [u for u, label in enumerate(a.units.labels) if label[0] == "door_leak"]
    door_cells = int(np.isfinite(reach[door_rows]).any(axis=0).sum())
    file_detected = len(a.units.labels) - a.file_all.score.undetected
    counts = detected_by_scenario(a.units, a.found.cells)
    k_holds = (
        all(label in a.found.units for label in rest)
        and a.found.score.undetected == 0
        and a.found.score.worst < 3.0
    )
    return [
        "### P1. Prompt 76's predictions against the measurements",
        "",
        "| Prediction | Measured | Verdict |",
        "|---|---|---|",
        f"| (i) the file's four detect none of the event scenarios' units on 320x120 | "
        f"{file_detected} of {len(a.units.labels)} units detected | "
        f"{'holds' if file_detected == 0 else 'fails'} |",
        f"| (j) no mountable candidate detects a door unit on 320x120 | {door_cells} of "
        f"{len(a.mounted):,} mountable candidates detect one; door units uncovered: "
        f"{len(door & set(a.found.uncovered))} of {len(door)} | "
        f"{'holds' if door_cells == 0 else 'fails'} |",
        f"| (k) the searched mountable four detect every breach and dust unit, worst case "
        f"under 3 s | {detected_text(counts)}; worst case over the units scored "
        f"{text_time(a.found.score.worst)} s, {len(a.found.units)} units scored | "
        f"{'holds' if k_holds else 'fails'} |",
    ]


def render(a: Analysis) -> tuple[str, dict[str, str]]:
    """The tables file's text and each marked table's text."""
    runs = a.runs
    ordered = {name: runs[name] for name in NAMES}
    tables = {
        "reload": reload_table(runs),
        "flows": tables69.flows_table(ordered),
        "states": states_table(runs),
        "march": tables69.march_table(ordered),
        "budgets": budget_table(runs),
        "alerts": alerts_table(a),
        "layouts": layouts_table(a),
        "curve": curve_table(a),
        "headline": headline_table(a),
        "units": units_table(a, REPORT_CLASSES),
        "recovery": recovery_table(a),
        "stability": stability_table(a),
        "arms": arms_table(a),
        "predictions": predictions_table(a),
        "hashes": hashes_table(),
        "bitwise": bitwise_table(runs),
        "coverage": coverage_table(),
    }
    head = [
        "# Step 5's tables: the latency report (ADR-015 E)",
        "",
        "Generated by `docs/reports/probe76/tables76.py` from the run files under "
        "`results/builder76/`, `results/builder75/runs/` and `results/builder72/`; `--check` "
        "regenerates and compares, the copies in `docs/reports/phase5_alert_report.md` included. "
        "Do not edit by hand.",
        "",
    ]
    body: list[str] = []
    for name in (*MARKERS, *GATE_MARKERS):
        body.extend(tables[name])
        body.append("")
        if name == "units":
            body.extend(units_table(a, None))
            body.append("")
    return "\n".join(head + body), {
        name: "\n".join(lines) for name, lines in tables.items()
    }


def splice(text: str, tables: dict[str, str]) -> str:
    """The report's text with each marked region replaced by its table."""
    for name, table in tables.items():
        begin = f"<!-- tables76 {name} begin -->"
        end = f"<!-- tables76 {name} end -->"
        if begin not in text or end not in text:
            continue
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{table}\n{end}{tail}"
    return text


def report_copies(text: str, markers: tuple[str, ...]) -> dict[str, str]:
    """A report's copies of the tables, keyed by marker name."""
    copies = {}
    for name in markers:
        begin = f"<!-- tables76 {name} begin -->"
        end = f"<!-- tables76 {name} end -->"
        if begin in text and end in text:
            copies[name] = text[
                text.index(begin) + len(begin) : text.index(end)
            ].strip()
    return copies


def figures(a: Analysis, out: Path) -> list[Path]:
    """Per event scenario, the 0.1 um and 5 um arrival maps with the sensors drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    paths = []
    sizes = [size * 1.0e6 for size in a.config.particle_sizes]
    cmap = plt.get_cmap("Blues_r").copy()
    cmap.set_bad("0.93")
    for name in EVENT_NAMES:
        arrival = a.arrivals[name]
        table = latency(arrival.maps, arrival.onsets)
        fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), layout="constrained")
        for ax, k in zip(axes, REPORT_CLASSES, strict=True):
            field = np.where(np.isfinite(table[0, k]), table[0, k], np.nan)
            image = ax.pcolormesh(
                a.mesh.x,
                a.mesh.y,
                np.ma.masked_invalid(field),
                cmap=cmap,
                norm=colors.LogNorm(vmin=0.1, vmax=arrival.t_final - arrival.onsets[0]),
                shading="flat",
            )
            for obstacle in a.config.obstacles:
                ax.add_patch(
                    patches.Rectangle(
                        (obstacle.x_start, obstacle.y_start),
                        obstacle.x_end - obstacle.x_start,
                        obstacle.y_end - obstacle.y_start,
                        facecolor="0.55",
                        edgecolor="0.3",
                        linewidth=0.8,
                        zorder=3,
                    )
                )
            for cells, marker, label in (
                (a.sensors, "o", "the file's four"),
                (a.found.cells, "s", "searched, mountable"),
                (a.curve[a.spec.count].cells, "^", "free optimum"),
            ):
                xy = np.array([where(a.mesh, cell) for cell in cells])
                ax.plot(
                    xy[:, 0],
                    xy[:, 1],
                    linestyle="none",
                    marker=marker,
                    markersize=8,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    label=label,
                    zorder=4,
                )
            ax.set_xlim(a.mesh.x[0], a.mesh.x[-1])
            ax.set_ylim(a.mesh.y[0], a.mesh.y[-1])
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(
                f"{TITLES[name].capitalize()}, {sizes[k]:g} um: first time at or above "
                f"threshold after the onset (light grey: never by {arrival.t_final:g} s)",
                fontsize=10,
            )
            fig.colorbar(image, ax=ax, label="s after the onset", shrink=0.9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
        path = out / f"phase5_arrival_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> int:
    """Write the tables and figures, or with --check compare the tables with the files."""
    parser = argparse.ArgumentParser(description="ADR-015 step 5's tables")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--splice", action="store_true")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=TABLES)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    runs = {name: Run(args.results, name) for name in NAMES}
    analysis = Analysis(runs)
    text, tables = render(analysis)
    if args.check:
        failures = []
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != text:
            failures.append(f"{args.out} differs from the records")
        for path, markers in ((REPORT, MARKERS), (GATE_REPORT, GATE_MARKERS)):
            if not path.exists():
                failures.append(f"{path} is missing")
                continue
            copies = report_copies(path.read_text(encoding="utf-8"), markers)
            for name in markers:
                if name not in copies:
                    failures.append(f"{path}: no copy of table '{name}'")
                elif copies[name] != tables[name]:
                    failures.append(
                        f"{path}: the copy of '{name}' differs from the records"
                    )
        for failure in failures:
            print(failure)
        print("tables76: " + ("differs" if failures else "matches"))
        return 1 if failures else 0
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    if args.splice:
        for path in (REPORT, GATE_REPORT):
            if path.exists():
                path.write_text(
                    splice(path.read_text(encoding="utf-8"), tables),
                    encoding="utf-8",
                    newline="\n",
                )
                print(f"spliced {path}")
    if args.figures:
        for path in figures(analysis, args.figures_dir):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
