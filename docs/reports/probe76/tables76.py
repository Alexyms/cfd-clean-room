"""Prompts 76 to 76c: the latency report's tables and figures, from the run files (ADR-015 E).

Reads the five reference scenarios on 320x120 (results/builder76/: step 5's four and
prompt 76b's seal leak), step 4's 80x30 runs (results/builder75/runs/), step 1's 80x30
clean-out and arm runs (results/builder72/), all untracked and on this machine, and writes
tables76.md, the copies in docs/reports/phase5_alert_report.md and the figures.

The layouts are read as ADR-015 D and E state them, with prompt 76b's decisions: the
detection units are (scenario, event, class) over the four event scenarios, the all-clear
units (scenario, class) over the five; the candidates come from every scored scenario's
configuration, the source boxes and their standoff excluded; the mount, the standoff, the
stand's height band, the count, the all-clear count and the rule are the product file's
placement section (prompt 76c: the band from 0.1 m, no all-clear sensor). The 0.3 m band's
search and the three-plus-one are kept as one row each. A latency read after an event's end
changed the flow state is marked (``after_flow_switch``). A layout's latency comes from the
arrival maps by lookup, its clear time
from the stored frames. A file sensor's peak and time above come from its every-step
readings, a searched cell's from the stored frames, and the tables say which. The coverage
rule's time is this report's, WITHIN: the product file states none.

    python docs/reports/probe76/tables76.py --splice --figures    # write everything
    python docs/reports/probe76/tables76.py --check               # regenerate and compare
"""

import argparse
import json
import math
import sys
from itertools import pairwise, product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "probe69"))

import tables69  # noqa: E402 -- follows sys.path.insert

from src.config import (  # noqa: E402
    MEAN,
    MOUNT_FREE,
    MOUNT_STAND,
    WORST_CASE,
    SimConfig,
)
from src.io_manager import OPEN_STEP, RunFile  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.monitor import sensor_cells  # noqa: E402
from src.placement import (  # noqa: E402
    ClearScore,
    ClearUnits,
    Layout,
    RunArrival,
    Score,
    Units,
    after_flow_switch,
    arrival_from_run,
    candidates,
    clear_score,
    clear_units,
    detection_units,
    exclude_units,
    flow_switches,
    given_layout,
    latency,
    rank,
    room_clear,
    score,
    search,
    search_roles,
)

RESULTS = ROOT / "results" / "builder76"
COARSE = ROOT / "results" / "builder75" / "runs"
COARSE_STEMS = {
    "door_leak": "door",
    "filter_breach": "breach",
    "equipment_dust": "dust",
}
SWEEP = ROOT / "results" / "builder72"
# Step 1's third clean-out grid (Phase 4's condition 2), the 80x30 all-clear reading.
COARSE_CLEAN_OUT = SWEEP / "clean_out_80x30.h5"
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

STEP5_EVENTS = ("door_leak", "filter_breach", "equipment_dust")
EVENT_NAMES = (*STEP5_EVENTS, "seal_leak")
NAMES = (*EVENT_NAMES, "clean_out")
TITLES = {
    "door_leak": "the door",
    "filter_breach": "the breach",
    "equipment_dust": "the dust",
    "seal_leak": "the seal leak",
    "clean_out": "the clean-out",
}
REPORT_CLASSES = (0, 4)  # 0.1 um and 5 um
FREE_COUNTS = (1, 2, 3, 4)
ARMS = ("control", "i10", "max_nut", "min_nut", "sc02", "widest")
# The report's time for the coverage rule, seconds: the REQ-A05 test's. The product
# file states none, so this is the report's reading, not a requirement.
WITHIN = 10.0
RESOLVED_FLOOR = 1.0e-2
# The standoffs the door is read at beside the product file's (prompt 76's (j)).
DOOR_STANDOFFS = (0.05, 0.1, 0.2, 0.5, 1.0)
FILE = "the file's four"
SURF = "searched four, mountable without stands"
EARLIER = "searched four with stands, the earlier 0.3 m band"
STAND = "the product search, four with stands"
ROLES = "the three-plus-one, fourth cell set by the clean-out's run end"
FREE = "free optimum"
# Prompt 76b's band, kept as one row so the cost of the band's floor is shown.
EARLIER_BAND = (0.3, 1.2)
# Prompt 76b's all-clear count, kept as one row (the three-plus-one); the
# product file carries the all-clear position open (prompt 76c).
ROLES_ALL_CLEAR = 1
AFTER_SWITCH = " (after the flow switch)"
MARKERS = (
    "reload",
    "flows",
    "states",
    "march",
    "budgets",
    "alerts",
    "candidates",
    "layouts",
    "curve",
    "headline",
    "units",
    "rooms",
    "clearing",
    "recovery",
    "stability",
    "arms",
    "predictions",
    "predictions76b",
    "predictions76c",
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


def cells_around(mesh: Mesh, x: float, y: float) -> list[tuple[int, int]]:
    """The cells sharing a point: one inside a cell, two on a face, four on a corner.

    The first is ``cell_at``'s, the one below or left of it (the lower-left).
    """
    rows = [int(np.searchsorted(mesh.y, y, side="left")) - 1]
    columns = [int(np.searchsorted(mesh.x, x, side="left")) - 1]
    if np.isclose(mesh.y[rows[0] + 1], y):
        rows.append(rows[0] + 1)
    if np.isclose(mesh.x[columns[0] + 1], x):
        columns.append(columns[0] + 1)
    return [(j, i) for j in rows for i in columns]


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


def clear_text(value: float) -> str:
    """A clear time, or "not clear"."""
    return "not clear" if math.isinf(value) else f"{value:.2f}"


def key_text(value: float) -> str:
    """A premature interval's ranking key: inf only where the room is not clear and the layout is."""
    if math.isinf(value):
        return "unbounded (the room not clear where the layout reads clean)"
    return f"{value:.2f} s"


def seconds(value: float) -> str:
    """A clear time with its unit, or "not clear"."""
    return "not clear" if math.isinf(value) else f"{value:.2f} s"


def short(name: str) -> str:
    """A scenario's short title."""
    return TITLES[name].removeprefix("the ")


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
    """Detected counts as text, the scenarios present only, in the report's order."""
    return ", ".join(
        f"{short(name)} {counts[name][0]} of {counts[name][1]}"
        for name in EVENT_NAMES
        if name in counts
    )


def uncovered_text(labels: tuple[tuple[str, str, int], ...]) -> str:
    """Uncoverable units by scenario, or none."""
    if not labels:
        return "none"
    counts: dict[str, int] = {}
    for name, _, _ in labels:
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{short(name)} {n}" for name, n in counts.items())


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
    """The units, the candidates and the layouts on the product grid's runs (prompt 76b)."""

    def __init__(self, runs: dict[str, Run]) -> None:
        door = runs["door_leak"]
        self.runs = runs
        self.config = door.config
        self.mesh = door.mesh
        self.spec = SimConfig(PRODUCT).placement
        assert self.spec is not None
        self.surfaces = tuple(m for m in self.spec.mount if m != MOUNT_STAND)
        # Every scored scenario's configuration (review 76 S5): the detection
        # scenarios' and the clean-out's, whose units the all-clear role scores.
        self.configs = [runs[name].config for name in NAMES]
        self.arrivals: dict[str, RunArrival] = {
            name: arrival_from_run(runs[name].file) for name in EVENT_NAMES
        }
        self.units = detection_units(self.arrivals)
        self.clear = clear_units({name: runs[name].file for name in NAMES})
        self.sensors = sensor_cells(self.config, self.mesh)
        self.sensor_names = tuple(sensor.name for sensor in self.config.sensors)
        standoff = self.spec.standoff
        self.mountable = candidates(self.mesh, self.configs, self.surfaces, standoff)
        self.product = candidates(
            self.mesh,
            self.configs,
            self.spec.mount,
            standoff,
            stand_height=self.spec.stand_height,
        )
        self.earlier_pool = candidates(
            self.mesh,
            self.configs,
            self.spec.mount,
            standoff,
            stand_height=EARLIER_BAND,
        )
        self.free = candidates(self.mesh, self.configs, (MOUNT_FREE,), standoff)
        count, rule = self.spec.count, self.spec.rule
        # The product search is the file's: four detection sensors (prompt 76c).
        assert self.spec.all_clear == 0, "the product file states no all-clear sensor"
        self.surf = search(self.units, self.mountable, count, rule, within=WITHIN)
        self.stand = search(self.units, self.product, count, rule, within=WITHIN)
        self.earlier = search(self.units, self.earlier_pool, count, rule, within=WITHIN)
        self.stand_mean = search(self.units, self.product, count, MEAN, within=WITHIN)
        self.roles = search_roles(
            self.units,
            self.clear,
            self.product,
            count,
            ROLES_ALL_CLEAR,
            rule,
            within=WITHIN,
        )
        # Prompt 76b's configurations as it scored them, for its predictions (P2).
        self.earlier_roles = search_roles(
            self.units,
            self.clear,
            self.earlier_pool,
            count,
            ROLES_ALL_CLEAR,
            rule,
            within=WITHIN,
        )
        self.switches = {name: flow_switches(runs[name].config) for name in EVENT_NAMES}
        self.onsets = {
            (name, event): float(onset)
            for name in EVENT_NAMES
            for event, onset in zip(
                self.arrivals[name].event_names,
                self.arrivals[name].onsets,
                strict=True,
            )
        }
        self.curve = {
            n: search(self.units, self.free, n, WORST_CASE, within=WITHIN)
            for n in FREE_COUNTS
        }
        self.kept = exclude_units(self.units, self.stand.uncovered)
        self.file_all = given_layout(self.units, self.sensors, within=WITHIN)
        self.file_kept = given_layout(self.kept, self.sensors, within=WITHIN)
        self.free_on_kept = search(
            self.kept, self.free, count, WORST_CASE, within=WITHIN
        )
        self.layouts: dict[str, tuple[tuple[int, int], ...]] = {
            FILE: self.sensors,
            SURF: self.surf.cells,
            EARLIER: self.earlier.cells,
            STAND: self.stand.cells,
            ROLES: self.roles.layout.cells,
            FREE: self.curve[count].cells,
        }
        self.clearing = {
            name: clear_score(self.clear, cells) for name, cells in self.layouts.items()
        }
        # Step 5's search as first scored: the door's configuration alone, the
        # surfaces, the three event scenarios (prediction (k)'s basis).
        self.step5_units = detection_units(
            {name: self.arrivals[name] for name in STEP5_EVENTS}
        )
        self.step5_mounted = candidates(self.mesh, door.config, self.surfaces, standoff)
        self.step5_free = candidates(self.mesh, door.config, (MOUNT_FREE,), standoff)
        self.step5_found = search(
            self.step5_units, self.step5_mounted, count, rule, within=WITHIN
        )
        self.t7, self.t7_mean, self.t7_air = tables69.t7_errors()
        self.t7_largest = max(abs(value) for value in self.t7.values())
        self._widths: dict[str, list[np.ndarray]] = {}
        self._y = np.asarray(self.mesh.y)

    def switched(self, label: tuple[str, str, int], value: float) -> bool:
        """Whether a unit's latency falls after a flow switch of its scenario."""
        name, event, _ = label
        return bool(
            after_flow_switch(value, self.onsets[(name, event)], self.switches[name])
        )

    def latency_text(self, label: tuple[str, str, int], value: float) -> str:
        """A latency with its marker, or never with the run's end."""
        if math.isinf(value):
            return f"never (to {self.arrivals[label[0]].t_final:g})"
        return f"{value:.3f}" + (AFTER_SWITCH if self.switched(label, value) else "")

    def reach(self, scenario: str) -> np.ndarray:
        """[ny, nx] bool: the cells some unit of SCENARIO reaches."""
        rows = [u for u, label in enumerate(self.units.labels) if label[0] == scenario]
        return np.isfinite(self.units.latencies[rows]).any(axis=0)

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
                self._sweep_y = np.asarray(mesh.y)
            self._widths[scenario] = rows
        if not self._widths[scenario]:
            return "no arm"
        j = int(np.searchsorted(self._sweep_y, y, side="left")) - 1
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
    """F2: each flow state against the gate's run of the same fingerprint.

    The seal leak has no gate run; its closed state is the door's rest state, so it
    is read against the gate's door run.
    """
    lines = [
        "### F2. The flow states against the gate's (prompt 69, the same files before the placement section; the seal leak against the gate's door run)",
        "",
        "| Run | State | Face hash (first 16) | The gate's | Largest face difference (m/s) |",
        "|---|---|---|---|---|",
    ]
    for name in NAMES:
        run = runs[name]
        gate_path = GATE / f"{name}.h5"
        if not gate_path.exists():
            gate_path = GATE / "door_leak.h5"
        gate = RunFile(gate_path) if gate_path.exists() else None
        for fingerprint in run.file.flow_fingerprints():
            state = run.file.read_flow_state(fingerprint).state
            ours = tables69.face_hash(state.u, state.v)[:16]
            theirs, gap = "no such state", "-"
            if gate is not None and fingerprint in gate.flow_fingerprints():
                other = gate.read_flow_state(fingerprint).state
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
            lines.append(f"| {name} | no gate run (a scenario of prompt 76b) | | | |")
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


def candidates_table(a: Analysis) -> list[str]:
    """N1: the candidate counts on 320x120, step 5's rule and each of prompt 76b's."""
    door = a.runs["door_leak"].config
    no_source = [a.runs[name].config for name in NAMES if name != "equipment_dust"]
    standoff = a.spec.standoff
    band = a.spec.stand_height
    rows = (
        (
            "step 5's: the door's configuration, the surfaces",
            candidates(a.mesh, door, a.surfaces, standoff),
            candidates(a.mesh, door, (MOUNT_FREE,), standoff),
        ),
        (
            "every scenario's openings, no source box",
            candidates(a.mesh, no_source, a.surfaces, standoff),
            candidates(a.mesh, no_source, (MOUNT_FREE,), standoff),
        ),
        ("and the dust's box and its standoff", a.mountable, a.free),
        (
            f"and the stands at the earlier band, {EARLIER_BAND[0]:g} to {EARLIER_BAND[1]:g} m",
            a.earlier_pool,
            a.free,
        ),
        (
            f"and the stands at {band[0]:g} to {band[1]:g} m (the product mount)",
            a.product,
            a.free,
        ),
    )
    lines = [
        f"### N1. The candidates on 320x120 (standoff {standoff:g} m; the stand's inlet {band[0]:g} to {band[1]:g} m above the floor, the earlier band {EARLIER_BAND[0]:g} to {EARLIER_BAND[1]:g} m)",
        "",
        "| Rule | Mountable | Free |",
        "|---|---|---|",
    ]
    for title, mounted, free in rows:
        lines.append(f"| {title} | {len(mounted):,} | {len(free):,} |")
    mounted = {(int(j), int(i)) for j, i in a.mountable}
    counts = []
    for low_high in (band, EARLIER_BAND):
        stands = candidates(
            a.mesh, a.configs, (MOUNT_STAND,), standoff, stand_height=low_high
        )
        extra = sum((int(j), int(i)) not in mounted for j, i in stands)
        counts.append(
            f"{len(stands):,} at {low_high[0]:g} to {low_high[1]:g} m, of which "
            f"{extra:,} are not already mountable on a surface"
        )
    lines += ["", "Stand cells: " + "; ".join(counts) + "."]
    return lines


def switch_text(a: Analysis, layout: Layout) -> str:
    """Whether a unit setting the worst case is read after a flow switch, and how many are."""
    labels = set(layout.units)
    picked = [c[0] for c in layout.cells], [c[1] for c in layout.cells]
    worst_marked, marked = False, 0
    for u, label in enumerate(a.units.labels):
        if label not in labels:
            continue
        best = float(a.units.latencies[u][picked].min())
        if math.isinf(best) or not a.switched(label, best):
            continue
        marked += 1
        worst_marked |= best == layout.score.worst
    return f"{'yes' if worst_marked else 'no'}; {marked} of {len(labels)}"


def score_row(
    name: str,
    layout: Layout,
    a: Analysis,
    uncovered: tuple[tuple[str, str, int], ...] | None = None,
) -> str:
    """One row of the layouts' tables."""
    mesh, units = a.mesh, a.units
    s = layout.score
    masks = (
        "-"
        if layout.distinct_masks is None
        else f"{layout.distinct_masks} / {layout.maximal_masks}"
    )
    gaps = layout.uncovered if uncovered is None else uncovered
    return (
        f"| {name} | {places(mesh, layout.cells)} | {layout.rule or '-'} | {layout.method} | "
        f"{'yes' if layout.exact else 'no'} | {len(layout.units)} | "
        f"{'-' if layout.method == 'given' and uncovered is None else uncovered_text(gaps)} | "
        f"{text_time(s.worst)} | {switch_text(a, layout)} | {s.mean:.3f} | {s.undetected} | "
        f"{s.coverage:.3f} | {masks} | "
        f"{detected_text(detected_by_scenario(units, layout.cells))} | "
        f"{working_cells(units, layout.cells)} of {len(layout.cells)} |"
    )


LAYOUT_HEAD = (
    "| Layout | Cell centres (x, y), m | Rule | Method | Exact | Units scored | Uncoverable "
    "by its candidates | Worst case (s) | Worst case after the flow switch; units so "
    "marked | Mean (s) | Undetected | Coverage within 10 s | "
    "Distinct / maximal masks | Detected, of all twenty | Cells setting a latency |"
)
LAYOUT_RULE = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def layouts_table(analysis: Analysis) -> list[str]:
    """L1: the configurations, over the units each is scored on."""
    a = analysis
    roles = a.roles
    lines = [
        f"### L1. The layouts on 320x120: mount {list(a.spec.mount)} at {a.spec.standoff} m "
        f"(stand {a.spec.stand_height[0]:g} to {a.spec.stand_height[1]:g} m), "
        f"{len(a.mountable):,} mountable without stands, {len(a.product):,} with them "
        f"({len(a.earlier_pool):,} at the earlier band, {EARLIER_BAND[0]:g} to "
        f"{EARLIER_BAND[1]:g} m) and {len(a.free):,} free candidates; the means are capped at "
        "each unit's remaining run time. A latency after the flow switch is one read after "
        "the end of an event whose end changes the flow state (the door's close at 15 s, the "
        "seal leak's end at 40 s)",
        "",
        LAYOUT_HEAD,
        LAYOUT_RULE,
        score_row(f"{FILE}, all twenty units", a.file_all, a),
        score_row(f"{FILE}, the product search's units", a.file_kept, a),
        score_row(f"{SURF}, worst-case rule", a.surf, a),
        score_row(f"{EARLIER}, worst-case rule", a.earlier, a),
        score_row(f"{STAND}, worst-case rule", a.stand, a),
        score_row(f"{STAND}, mean rule", a.stand_mean, a),
        score_row(f"{ROLES}: its three detection sensors", roles.detection, a),
        score_row(f"{ROLES}: the four", roles.layout, a, roles.detection.uncovered),
        score_row(f"{FREE}, worst-case rule", a.curve[a.spec.count], a),
        score_row(
            f"{FREE}, worst-case rule, the product search's units", a.free_on_kept, a
        ),
        "",
        f"The three-plus-one's all-clear sensor: {places(a.mesh, roles.all_clear)}.",
    ]
    for scenario in ("door_leak", "seal_leak"):
        reached = a.reach(scenario)
        rows, columns = np.nonzero(reached)
        if rows.size == 0:
            lines.append(f"{TITLES[scenario].capitalize()}'s plume reaches no cell.")
            continue
        counts = []
        for title, pool in (
            ("mountable without stands", a.mountable),
            ("with stands at the earlier band", a.earlier_pool),
            ("with stands", a.product),
            ("free", a.free),
        ):
            counts.append(
                f"{int(reached[pool[:, 0], pool[:, 1]].sum()):,} of {len(pool):,} {title}"
            )
        lines.append(
            f"{TITLES[scenario].capitalize()}'s plume reaches {int(reached.sum()):,} cells, x up "
            f"to {a.mesh.x[columns.max() + 1]:.3f} m and y up to {a.mesh.y[rows.max() + 1]:.3f} m; "
            f"candidates reaching a unit of it: " + "; ".join(counts) + "."
        )
    seal = a.reach("seal_leak")
    free_seal = [where(a.mesh, (int(j), int(i))) for j, i in a.free if seal[j, i]]
    if free_seal:
        lines.append(
            f"The free cells that detect a seal-leak unit lie from y "
            f"{min(y for _, y in free_seal):.4f} to {max(y for _, y in free_seal):.4f} m and "
            f"x {min(x for x, _ in free_seal):.4f} to {max(x for x, _ in free_seal):.4f} m."
        )
    door = a.reach("door_leak")
    counts = []
    for standoff in DOOR_STANDOFFS:
        pool = candidates(a.mesh, a.configs, a.surfaces, standoff)
        counts.append(
            f"{standoff:g} m: {int(door[pool[:, 0], pool[:, 1]].sum())} of {len(pool):,}"
        )
    lines.append(
        "Candidates on the surfaces that detect a door unit at other standoffs, the openings' "
        "exclusion widening with the standoff: " + "; ".join(counts) + "."
    )
    named = {
        FILE: a.file_kept,
        f"{SURF}": given_layout(a.kept, a.surf.cells, within=WITHIN),
        f"{EARLIER}": given_layout(a.kept, a.earlier.cells, within=WITHIN),
        f"{STAND}": a.stand,
        ROLES: given_layout(a.kept, a.roles.layout.cells, within=WITHIN),
        f"{FREE}": a.free_on_kept,
    }
    scores = {name: layout.score for name, layout in named.items()}
    names = {id(layout): name for name, layout in named.items()}
    ranked = rank(list(named.values()))
    lines += [
        "",
        f"Over the {len(a.kept.labels)} units the product search covers, the order per rule, "
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
        lines.append(score_row(f"{n} sensors", a.curve[n], a))
    return lines


def headline_table(analysis: Analysis) -> list[str]:
    """U2: per unit, each configuration's latency, the least over its cells."""
    a = analysis
    lines = [
        "### U2. Per unit, the layout's latency, the least over its cells (s; never with the run's end; marked when read after the end of an event whose end changes the flow state)",
        "",
        "| Scenario | Event | Class (um) | " + " | ".join(a.layouts) + " |",
        "|---|---|---|" + "---|" * len(a.layouts),
    ]
    sizes = [size * 1.0e6 for size in a.config.particle_sizes]
    for u, (name, event, k) in enumerate(a.units.labels):
        cells = []
        for layout in a.layouts.values():
            best = float(min(a.units.latencies[u, j, i] for j, i in layout))
            cells.append(a.latency_text((name, event, k), best))
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
    roles = a.roles.layout.cells
    configurations = (
        (FILE, a.sensors, a.sensor_names),
        (
            STAND,
            a.stand.cells,
            tuple(f"P{n + 1}" for n in range(len(a.stand.cells))),
        ),
        (
            ROLES,
            roles,
            tuple(f"D{n + 1}" for n in range(len(a.roles.detection.cells)))
            + tuple(f"A{n + 1}" for n in range(len(a.roles.all_clear))),
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
        f"### U1. Per scenario, event, class and sensor, for the file's four, the product search (P), the three-plus-one (D detection, A all-clear) and the free optimum ({which}); a latency read after the flow switch is marked. A file "
        "sensor's peak and time above are from its every-step readings, a searched cell's from "
        "the stored frames (every 100 steps, about 0.36 s). The time error is T7's (160x60, one "
        "direction), measured at the file's sensors only. The width is the plume's largest at "
        "the threshold on the cell's row over step 1's 160x60 arms, read at frame resolution; "
        "the seal leak ran on no arm.",
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
                        text = a.latency_text((name, event, k), value)
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


def rooms_table(a: Analysis) -> list[str]:
    """C0: per all-clear unit, t_e, the frames' resolution and the room's clear time."""
    times, cells = room_clear(a.clear)
    sizes = [size * 1.0e6 for size in a.config.particle_sizes]
    lines = [
        "### C0. The all-clear units: per scenario and class, t_e (the end of its last event), the stored frames from t_e on and their largest spacing (the clear times' resolution), and the room's clear time, the latest over the live cells, with the cell that sets it",
        "",
        "| Scenario | Class (um) | t_e (s) | Frames from t_e | Largest spacing (s) | The room's clear time (s) | Set by the cell at (x, y), m |",
        "|---|---|---|---|---|---|---|",
    ]
    for u, (name, k) in enumerate(a.clear.labels):
        lines.append(
            f"| {name} | {sizes[k]:g} | {a.clear.ends[u]:g} | {a.clear.times[u].size} | "
            f"{a.clear.spacing[u]:.3f} | {clear_text(times[u])} | {where_text(a, cells[u])} |"
        )
    return lines


def clearing_table(a: Analysis, classes: tuple[int, ...] | None) -> list[str]:
    """C2: per configuration and all-clear unit, the layout's clear time against the room's."""
    which = (
        "every class"
        if classes is None
        else "0.1 um and 5 um; every class in tables76.md"
    )
    sizes = [size * 1.0e6 for size in a.config.particle_sizes]
    lines = [
        f"### C2. Each configuration's all-clear reading ({which}): the layout's clear time (the latest of its cells'), the room's, the premature interval (the room's less the layout's) and the live area's fraction at or above threshold at the first stored frame at or after the layout's clear time. Frame-resolved (C0's spacing); a \"not clear\" enters no arithmetic",
        "",
    ]
    for title, result in a.clearing.items():
        lines += [
            f"**{title}**, {places(a.mesh, a.layouts[title])}",
            "",
            "| Scenario | Class (um) | The layout's clear time (s) | The room's (s) | Premature interval (s) | Area at or above at the layout's time |",
            "|---|---|---|---|---|---|",
        ]
        for u, (name, k) in enumerate(a.clear.labels):
            if classes is not None and k not in classes:
                continue
            lines.append(
                f"| {name} | {sizes[k]:g} | {clear_text(result.layout[u])} | "
                f"{clear_text(result.room[u])} | {interval_text(result, u)} | "
                f"{fraction_text(result, u)} |"
            )
        lines += ["", clear_summary(result), ""]
    return lines[:-1]


def interval_text(result: ClearScore, u: int) -> str:
    """A unit's premature interval, or which side is not clear."""
    if math.isinf(result.layout[u]):
        return "layout not clear"
    if math.isinf(result.room[u]):
        return "room not clear"
    return f"{result.interval[u]:.2f}"


def fraction_text(result: ClearScore, u: int) -> str:
    """A unit's area fraction at the layout's clear time, or none."""
    return "-" if np.isnan(result.fraction[u]) else f"{100 * result.fraction[u]:.2f}%"


def clear_summary(result: ClearScore) -> str:
    """A layout's all-clear reading in one line."""
    premature = int(np.sum(np.nan_to_num(result.interval, nan=0.0) > 0.0))

    return (
        f"Worst premature interval {key_text(result.worst)}, mean {key_text(result.mean)}, "
        f"largest area fraction {100 * result.worst_fraction:.2f}%; premature in {premature} "
        f"of {result.layout.size} units, the layout not clear in "
        f"{int(np.isinf(result.layout).sum())}, the room not clear in "
        f"{int(np.isinf(result.room).sum())}."
    )


def recovery_table(a: Analysis) -> list[str]:
    """C1: each layout's clean-out recovery times from the steps (ADR-014 C)."""
    record = a.runs["clean_out"].recovery
    assert record is not None
    configurations = [(title, cells) for title, cells in a.layouts.items()]
    configurations += [(f"{FREE}, {n}", a.curve[n].cells) for n in FREE_COUNTS[:-1]]
    sizes = [size * 1.0e6 for size in a.config.particle_sizes]
    lines = [
        f"### C1. The clean-out at the steps: each layout's recovery, the last step time each cell was at or above "
        f"its threshold (the run went to {record.t_final:g} s; inf is still above at the end), beside C2's frame reading. "
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


class Grid:
    """A grid's detection and all-clear units and its product search (K1)."""

    def __init__(
        self,
        mesh: Mesh,
        configs: list[SimConfig],
        units: Units,
        clear: ClearUnits,
        spec: object,
    ) -> None:
        self.mesh = mesh
        self.units = units
        self.clear = clear
        self.product = candidates(
            mesh, configs, spec.mount, spec.standoff, stand_height=spec.stand_height
        )
        self.free = candidates(mesh, configs, (MOUNT_FREE,), spec.standoff)
        self.roles = search_roles(
            units, clear, self.product, spec.count, spec.all_clear, spec.rule
        )
        self.kept = exclude_units(units, self.roles.detection.uncovered)
        self.free_found = search(self.kept, self.free, spec.count, WORST_CASE)


def stability_table(a: Analysis) -> list[str]:
    """K1: the product search on 80x30 and 320x120, and each on the other's maps.

    Both grids are read over the door, the breach and the dust (the seal leak ran on
    320x120 alone) and the all-clear units of those three and the clean-out.
    """
    source = {name: COARSE / f"{stem}.h5" for name, stem in COARSE_STEMS.items()}
    coarse_configs = [
        SimConfig.from_dict(RunFile(path).read_config().mapping)
        for path in (*source.values(), COARSE_CLEAN_OUT)
    ]
    coarse_mesh = Mesh(coarse_configs[0])
    coarse = Grid(
        coarse_mesh,
        coarse_configs,
        detection_units(
            {name: arrival_from_run(path) for name, path in source.items()}
        ),
        clear_units({**source, "clean_out": COARSE_CLEAN_OUT}),
        a.spec,
    )
    fine = Grid(
        a.mesh,
        [a.runs[name].config for name in (*STEP5_EVENTS, "clean_out")],
        a.step5_units,
        clear_units({name: a.runs[name].file for name in (*STEP5_EVENTS, "clean_out")}),
        a.spec,
    )
    fine_on_coarse = mapped(a.mesh, coarse_mesh, fine.roles.layout.cells)
    coarse_on_fine = mapped(coarse_mesh, a.mesh, coarse.roles.layout.cells)
    spec = a.spec
    roles_text = (
        f"{spec.count - spec.all_clear} detection sensors and {spec.all_clear} all-clear"
        if spec.all_clear
        else f"{spec.count} detection sensors, no all-clear"
    )
    lines = [
        f"### K1. The product search ({roles_text}, with stands from {spec.stand_height[0]:g} m) on 80x30 and 320x120, each layout scored on both grids over the door, the breach and the dust units the grid's search covers and over the all-clear units of those three and the clean-out (the seal leak has one grid). A layout from the other grid is read at the cell holding each centre, the one below or left of it on a face or corner (the lower-left); the line under the table gives the range over every reading",
        "",
        "| Grid | Layout | Cell centres (x, y), m | Candidates on this grid | Worst case (s) | Mean (s) | Undetected | Coverage within 10 s | Worst premature interval | Mean interval | Largest area fraction |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    entries = {
        "80x30": (
            coarse,
            {
                FILE: sensor_cells(coarse_configs[0], coarse_mesh),
                "the product searched here": coarse.roles.layout.cells,
                "the product searched on 320x120": fine_on_coarse,
                "free optimum here": coarse.free_found.cells,
            },
        ),
        "320x120": (
            fine,
            {
                FILE: a.sensors,
                "the product searched here": fine.roles.layout.cells,
                "the product searched on 80x30": coarse_on_fine,
                "free optimum here": fine.free_found.cells,
            },
        ),
    }
    orders = []
    for grid, (g, layouts) in entries.items():
        scores = {}
        for name, cells in layouts.items():
            s = score(g.kept, cells, within=WITHIN)
            c = clear_score(g.clear, cells)
            scores[name] = s
            lines.append(
                f"| {grid} | {name} | {places(g.mesh, cells)} | "
                f"{mountable_text(cells, g.product)} | {text_time(s.worst)} | "
                f"{s.mean:.3f} | {s.undetected} of {len(g.kept.labels)} | {s.coverage:.3f} | "
                f"{key_text(c.worst)} | {key_text(c.mean)} | {100 * c.worst_fraction:.2f}% |"
            )
        orders.append(
            f"- {grid}: worst case {order(scores, WORST_CASE)}; mean {order(scores, MEAN)}; "
            f"coverage within {WITHIN:g} s {order(scores, 'coverage')}."
        )
    worsts, means, count = [], [], 0
    choices = [
        cells_around(a.mesh, *where(coarse_mesh, cell))
        for cell in coarse.roles.layout.cells
    ]
    for reading in product(*choices):
        if any(a.mesh.cell_type[c] == SOLID for c in reading) or len(
            set(reading)
        ) < len(reading):
            continue
        s = score(fine.kept, reading, within=WITHIN)
        worsts.append(s.worst)
        means.append(s.mean)
        count += 1
    lines += [
        "",
        f"Read at every choice of the 320x120 cells sharing each 80x30 centre ({' x '.join(str(len(c)) for c in choices)} = "
        f"{math.prod(len(c) for c in choices)} readings, {count} with no SOLID or repeated cell), "
        f"the product searched on 80x30 has a worst case from {text_time(min(worsts))} to "
        f"{text_time(max(worsts))} s and a mean from {min(means):.3f} to {max(means):.3f} s on "
        "320x120; the table's row is the lower-left reading.",
        f"Uncovered by the product's candidates on 80x30: {uncovered_text(coarse.roles.detection.uncovered)}; "
        f"on 320x120: {uncovered_text(fine.roles.detection.uncovered)}. "
        + (
            f"The all-clear sensor: {places(coarse_mesh, coarse.roles.all_clear)} on 80x30, "
            f"{places(a.mesh, fine.roles.all_clear)} on 320x120."
            if spec.all_clear
            else "No all-clear sensor (the product file's all_clear is 0)."
        ),
        "",
        "The order per rule, best first:",
        "",
        *orders,
    ]
    return lines


def arms_table(a: Analysis) -> list[str]:
    """K2: the product layout of 320x120 on step 1's arms, frame-resolved and flagged."""
    lines = [
        "### K2. The product layout of 320x120 (the product search over the door, the breach and the dust, as K1's) on step 1's arm runs (arrival_from_run: frame-resolved, flagged), read at the cells holding its centres, beside the file's four, the arm's own product search and the arm's free optimum, over the units the arm's own search covers",
        "",
        "| Arm | Grid | Scenarios run | Frame-resolved | Layout | Candidates on this grid | Worst case (s) | Mean (s) | Undetected | Detected, of the arm's units | Own search's rule, method |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    fine = Grid(
        a.mesh,
        [a.runs[name].config for name in (*STEP5_EVENTS, "clean_out")],
        a.step5_units,
        clear_units({name: a.runs[name].file for name in (*STEP5_EVENTS, "clean_out")}),
        a.spec,
    )
    orders = []
    readings: dict[str, tuple] = {}
    rows = [(arm, "160x60") for arm in ARMS] + [("min_nut", "320x120")]
    for arm, grid in rows:
        paths = {}
        for name in STEP5_EVENTS:
            path = SWEEP / f"{name}_{arm}_{grid}.h5"
            summary = path.with_suffix(".txt")
            if summary.exists() and "wall time:" in summary.read_text(encoding="utf-8"):
                paths[name] = path
        if not paths:
            lines.append(f"| {arm} | {grid} | none | | | | | | | | |")
            continue
        arrivals = {name: arrival_from_run(path) for name, path in paths.items()}
        configs = [
            SimConfig.from_dict(RunFile(path).read_config().mapping)
            for path in paths.values()
        ]
        if "door_leak" not in paths:
            # The door's run is capped at this arm, but its openings are the room's;
            # the door's configuration of the arm gives the same candidates.
            configs.append(
                SimConfig(SWEEP / "configs" / f"scenario_door_leak_{arm}_{grid}.yaml")
            )
        mesh = Mesh(configs[0])
        own = Grid(mesh, configs, detection_units(arrivals), clear_units(paths), a.spec)
        layouts = {
            "the product of 320x120": mapped(a.mesh, mesh, fine.roles.layout.cells),
            FILE: sensor_cells(configs[0], mesh),
            "the arm's own product search": own.roles.layout.cells,
            "the arm's free optimum": own.free_found.cells,
        }
        scores = {}
        for title, cells in layouts.items():
            s = score(own.kept, cells, within=WITHIN)
            scores[title] = s
            lines.append(
                f"| {arm} | {grid} | {', '.join(short(n) for n in arrivals)} | "
                f"{'yes' if s.frame_resolved else 'no'} | {title} | "
                f"{mountable_text(cells, own.product)} | {text_time(s.worst)} | "
                f"{s.mean:.3f} | {s.undetected} of {len(own.kept.labels)} | "
                f"{detected_text(detected_by_scenario(own.units, cells))} | "
                f"{own.roles.detection.rule}, {own.roles.detection.method} |"
            )
            if title == "the product of 320x120" and grid == "160x60":
                table = own.kept.latencies[
                    :, [c[0] for c in cells], [c[1] for c in cells]
                ]
                readings[arm] = (own.kept.labels, tuple(np.round(table.min(axis=1), 9)))
        orders.append(
            f"- {arm} ({grid}): worst case {order(scores, WORST_CASE)}; mean "
            f"{order(scores, MEAN)}."
        )
    groups: list[list[str]] = []
    for arm, key in readings.items():
        for group in groups:
            if readings[group[0]] == key:
                group.append(arm)
                break
        else:
            groups.append([arm])
    lines += [
        "",
        f"The product layout's readings on the {len(readings)} 160x60 arms that ran: "
        f"{len(groups)} distinct (the same units and the same latency in each): "
        + "; ".join(" = ".join(group) for group in groups)
        + ". With min_nut on 320x120 that is "
        + f"{len(groups) + 1} distinct readings.",
        "",
        "The order per rule, best first:",
        "",
        *orders,
    ]
    return lines


def mountable_text(cells: tuple[tuple[int, int], ...], pool: np.ndarray) -> str:
    """How many of a layout's cells are candidates on the grid it is read on."""
    allowed = {(int(j), int(i)) for j, i in pool}
    return f"{sum(cell in allowed for cell in cells)} of {len(cells)}"


def predictions_table(a: Analysis) -> list[str]:
    """P1: prompt 76's predictions as step 5 first scored them.

    The door's configuration alone, the surfaces, the three event scenarios: the
    candidates and units step 5 had, before prompt 76b's rules.
    """
    units = a.step5_units
    found = a.step5_found
    door = {label for label in units.labels if label[0] == "door_leak"}
    rest = [label for label in units.labels if label[0] != "door_leak"]
    reach = units.latencies[:, a.step5_mounted[:, 0], a.step5_mounted[:, 1]]
    door_rows = [u for u, label in enumerate(units.labels) if label[0] == "door_leak"]
    door_cells = int(np.isfinite(reach[door_rows]).any(axis=0).sum())
    file_detected = len(units.labels) - given_layout(units, a.sensors).score.undetected
    counts = detected_by_scenario(units, found.cells)
    k_holds = (
        all(label in found.units for label in rest)
        and found.score.undetected == 0
        and found.score.worst < 3.0
    )
    return [
        "### P1. Prompt 76's predictions, as step 5 scored them (the door's configuration alone, the surfaces, the three event scenarios)",
        "",
        "| Prediction | Measured | Verdict |",
        "|---|---|---|",
        f"| (i) the file's four detect none of the event scenarios' units on 320x120 | "
        f"{file_detected} of {len(units.labels)} units detected | "
        f"{'holds' if file_detected == 0 else 'fails'} |",
        f"| (j) no mountable candidate detects a door unit on 320x120 | {door_cells} of "
        f"{len(a.step5_mounted):,} mountable candidates detect one; door units uncovered: "
        f"{len(door & set(found.uncovered))} of {len(door)} | "
        f"{'holds' if door_cells == 0 else 'fails'} |",
        f"| (k) the searched mountable four detect every breach and dust unit, worst case "
        f"under 3 s | {detected_text(counts)}; worst case over the units scored "
        f"{text_time(found.score.worst)} s, {len(found.units)} units scored; cells "
        f"{places(a.mesh, found.cells)} | {'holds' if k_holds else 'fails'} |",
    ]


def predictions76b_table(a: Analysis) -> list[str]:
    """P2: prompt 76b's predictions (l) to (p), on its configurations.

    The 0.3 m band and the three-plus-one on it, as prompt 76b scored them.
    """
    surf = detected_by_scenario(a.units, a.surf.cells)
    stand = detected_by_scenario(a.units, a.earlier.cells)
    l_holds = (
        surf["filter_breach"][0] == surf["filter_breach"][1]
        and surf["equipment_dust"][0] == surf["equipment_dust"][1]
        and surf["door_leak"][0] == 0
        and surf["seal_leak"][0] == 0
    )
    m_holds = all(
        stand[name][0] == stand[name][1]
        for name in ("door_leak", "filter_breach", "equipment_dust")
    )
    seal = a.reach("seal_leak")
    mounts = {
        "surfaces": a.mountable,
        "stands": candidates(
            a.mesh,
            a.configs,
            (MOUNT_STAND,),
            a.spec.standoff,
            stand_height=EARLIER_BAND,
        ),
    }
    seal_counts = {
        name: int(seal[p[:, 0], p[:, 1]].sum()) for name, p in mounts.items()
    }
    free_heights = [float(a.mesh.yc[j]) for j, i in a.free if seal[j, i]]
    n_holds = all(v == 0 for v in seal_counts.values()) and (
        not free_heights or max(free_heights) < 0.3
    )
    clean = [u for u, (name, _) in enumerate(a.clear.labels) if name == "clean_out"]
    stand_clear = clear_score(a.clear, a.earlier.cells)
    roles_clear = a.earlier_roles.clear
    remaining = [float(stand_clear.fraction[u]) for u in clean]
    o_holds = all(np.isfinite(v) and v > 0.0 for v in remaining)
    three = [
        u
        for u, label in enumerate(a.units.labels)
        if label[0] in ("door_leak", "filter_breach", "equipment_dust")
    ]

    def worst_over(cells: tuple[tuple[int, int], ...]) -> float:
        table = a.units.latencies[three][
            :, [c[0] for c in cells], [c[1] for c in cells]
        ]
        return float(table.min(axis=1).max())

    roles_worst = worst_over(a.earlier_roles.layout.cells)
    stand_worst = worst_over(a.earlier.cells)
    p_holds = roles_worst == stand_worst and roles_clear.worst < stand_clear.worst
    events = [u for u, (name, _) in enumerate(a.clear.labels) if name != "clean_out"]

    def events_worst(result: ClearScore) -> str:
        values = result.interval[events]
        return (
            key_text(float(values.max()))
            if np.isfinite(values).all()
            else "a not clear"
        )

    return [
        f"### P2. Prompt 76b's predictions against the measurements, on its configurations (the stand band {EARLIER_BAND[0]:g} to {EARLIER_BAND[1]:g} m; the three-plus-one on it)",
        "",
        "| Prediction | Measured | Verdict |",
        "|---|---|---|",
        f"| (l) without stands, the source excluded, the searched four detect every breach and dust unit and no door or seal-leak unit | {detected_text(surf)} | {'holds' if l_holds else 'fails'} |",
        f"| (m) with stands the searched four detect every door, breach and dust unit | {detected_text(stand)} | {'holds' if m_holds else 'fails'} |",
        f"| (n) no candidate of any mount detects a seal-leak unit; the free cells that do lie below 0.3 m | surface candidates {seal_counts['surfaces']}, stand candidates {seal_counts['stands']}; free cells {len(free_heights)}, the highest centre at {max(free_heights, default=math.nan):.4f} m | {'holds' if n_holds else 'fails'} |",
        f"| (o) in the clean-out the detection-only four with stands declare clean while live area at or above threshold remains, at every class | area at or above at its clear time, per class: {', '.join(f'{100 * v:.2f}%' for v in remaining)} | {'holds' if o_holds else 'fails'} |",
        f"| (p) the three-plus-one's worst latency over the door, breach and dust units equals the detection-only four's, and its worst premature interval is smaller | worst latency {text_time(roles_worst)} s against {text_time(stand_worst)} s; worst premature interval {key_text(roles_clear.worst)} against {key_text(stand_clear.worst)}; over the four event scenarios alone {events_worst(roles_clear)} against {events_worst(stand_clear)} | {'holds' if p_holds else 'fails'} |",
    ]


def predictions76c_table(a: Analysis) -> list[str]:
    """P3: prompt 76c's prediction (q) against the re-score."""
    found = a.stand
    counts = detected_by_scenario(a.units, found.cells)
    picked = [c[0] for c in found.cells], [c[1] for c in found.cells]
    best = {
        label: float(a.units.latencies[u][picked].min())
        for u, label in enumerate(a.units.labels)
    }
    seal = [value for label, value in best.items() if label[0] == "seal_leak"]
    worst_labels = [
        label for label, value in best.items() if value == found.score.worst
    ]
    worst_marked = [label for label in worst_labels if a.switched(label, best[label])]
    marked = [
        label
        for label, value in best.items()
        if np.isfinite(value) and a.switched(label, value)
    ]
    holds = (
        found.score.undetected == 0
        and not found.uncovered
        and len(found.units) == len(a.units.labels)
        and found.score.worst < 3.0
        and max(seal) < 1.0
        and not worst_marked
    )
    return [
        "### P3. Prompt 76c's prediction against the re-score",
        "",
        "| Prediction | Measured | Verdict |",
        "|---|---|---|",
        f"| (q) with the band at {a.spec.stand_height[0]:g} m the product search detects all "
        f"{len(a.units.labels)} units with a worst case under 3 s, the seal leak within 1 s, "
        "and no worst-case latency is after the flow switch | "
        f"{detected_text(counts)}; {len(found.units)} units scored, uncovered "
        f"{uncovered_text(found.uncovered)}; worst case {text_time(found.score.worst)} s, set "
        f"by {', '.join(f'{n} {e} class {k}' for n, e, k in worst_labels)}; the seal leak's "
        f"latencies {min(seal):.3f} to {max(seal):.3f} s; units after the flow switch "
        f"{len(marked)}, of them setting the worst case {len(worst_marked)}; cells "
        f"{places(a.mesh, found.cells)} | {'holds' if holds else 'fails'} |",
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
        "candidates": candidates_table(a),
        "layouts": layouts_table(a),
        "curve": curve_table(a),
        "headline": headline_table(a),
        "units": units_table(a, REPORT_CLASSES),
        "rooms": rooms_table(a),
        "clearing": clearing_table(a, REPORT_CLASSES),
        "recovery": recovery_table(a),
        "stability": stability_table(a),
        "arms": arms_table(a),
        "predictions": predictions_table(a),
        "predictions76b": predictions76b_table(a),
        "predictions76c": predictions76c_table(a),
        "hashes": hashes_table(),
        "bitwise": bitwise_table(runs),
        "coverage": coverage_table(),
    }
    head = [
        "# Step 5's tables: the latency report (ADR-015 E, prompts 76 to 76c)",
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
        if name == "clearing":
            body.extend(clearing_table(a, None))
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


def draw_room(ax: object, a: Analysis) -> None:
    """The obstacles and the axes of a room panel."""
    import matplotlib.patches as patches

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
    ax.set_xlim(a.mesh.x[0], a.mesh.x[-1])
    ax.set_ylim(a.mesh.y[0], a.mesh.y[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def draw_cells(ax: object, a: Analysis, sets: list[tuple[tuple, str, str]]) -> None:
    """Each layout's cells as markers."""
    for cells, marker, label in sets:
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


def figures(a: Analysis, out: Path) -> list[Path]:
    """The arrival maps per event scenario, and the clean-out's clear-time map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    paths = []
    sizes = [size * 1.0e6 for size in a.config.particle_sizes]
    cmap = plt.get_cmap("Blues_r").copy()
    cmap.set_bad("0.93")
    sets = [
        (a.sensors, "o", FILE),
        (a.stand.cells, "s", STAND),
        (
            a.roles.all_clear,
            "*",
            "the three-plus-one's fourth cell (set by the run's end)",
        ),
        (a.curve[a.spec.count].cells, "^", FREE),
    ]
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
            draw_room(ax, a)
            draw_cells(ax, a, sets)
            ax.set_title(
                f"{TITLES[name].capitalize()}, {sizes[k]:g} um: first time at or above "
                f"threshold after the onset (light grey: never by {arrival.t_final:g} s)",
                fontsize=10,
            )
            fig.colorbar(image, ax=ax, label="s after the onset", shrink=0.9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
        path = out / f"phase5_arrival_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    clean = {k: u for u, (name, k) in enumerate(a.clear.labels) if name == "clean_out"}
    times, cells = room_clear(a.clear)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), layout="constrained")
    warm = plt.get_cmap("viridis").copy()
    warm.set_bad("0.55")
    warm.set_over("crimson")
    for ax, k in zip(axes, REPORT_CLASSES, strict=True):
        u = clean[k]
        field = a.clear.clear[u]
        finite = field[np.isfinite(field)]
        top = float(finite.max()) if finite.size else 1.0
        shown = np.where(np.isinf(field), top * 2.0, field)
        image = ax.pcolormesh(
            a.mesh.x,
            a.mesh.y,
            np.ma.masked_invalid(shown),
            cmap=warm,
            vmin=0.0,
            vmax=top,
            shading="flat",
        )
        draw_room(ax, a)
        draw_cells(ax, a, sets[1:3])
        x, y = where(a.mesh, cells[u])
        ax.plot(
            [x],
            [y],
            marker="x",
            color="crimson",
            markersize=9,
            zorder=5,
            linestyle="none",
            label="the cell that clears last",
        )
        result = a.clearing[STAND]
        ax.set_title(
            f"The clean-out, {sizes[k]:g} um: each cell's clear time (frames; red: not clear "
            f"by {a.clear.times[u][-1]:g} s)\nthe room {seconds(times[u])}, the product search "
            f"{seconds(result.layout[u])}",
            fontsize=10,
        )
        fig.colorbar(image, ax=ax, label="clear time (s)", shrink=0.9, extend="max")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    path = out / "phase5_clear_clean_out.png"
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
