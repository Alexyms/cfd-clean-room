"""Generate the step 1 sweep's tables from its run files (prompt 72).

    python docs/reports/probe72/tables72.py            # write the tables
    python docs/reports/probe72/tables72.py --check    # regenerate and compare

Writes docs/reports/probe72/tables72.md (the per-class tables) and replaces
the marked blocks of docs/reports/phase5_sensitivity.md (the summaries, class
0.1 um unless a column says otherwise). Every number is read from the run
files under results/builder72/ (configs72.py's runs, made by run72.py), the
gate's 320x120 clean-out and breach under results/builder69/ (prompt 69) and
the step 4 80x30 clean-out under results/builder64/coarse/ (prompt 64). The
readings, with the probe tables of ADR-015 they repeat:

- R1: each run's inputs, the supply's inlet eddy viscosity C_mu sqrt(k) l_e
  with k = 1.5 (I U)^2 (ADR-012 C), its flow states (outer count, stop, wall,
  from the runner's summary) and its march (steps, wall).
- D1 (T6's reading): per run, class and sensor, the distance from the
  sensor's cell to the nearest non-SOLID cell ever at or above the class's
  threshold in any stored frame, centre to centre, in metres and in cells,
  and whether the sensor's own cell is reached. The threshold is the class's
  in every scenario: the breach's panel and the door's corridor air carry a
  hundred times it, the dust source has no inlet concentration.
- W1 (T4's reading): per run, class and sensor row (the row of cells holding
  the sensor height, the cell below a face, ADR-014 C), the number of
  non-SOLID cells at or above the threshold and at or above ten times it,
  times dx, at the stored frame nearest the event's end and the largest over
  the frames, and the x extent of the threshold cells at that frame.
- P1 (T3's reading): per run, class and sensor, the peak reading over the
  threshold, stated as a number only when at or above RESOLVED_FLOOR of it
  (ADR-015 B: below that the value is the grid's, not the inputs'), and the
  first time at or above.
- C1: the clean-out on three grids: the room recovery time, the free-area
  mean's 100:1 time and its ratio to the well-mixed ln(100)/N, and the cells
  not recovered at the end, with the observed order of convergence where the
  three values allow one.
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import configs72  # noqa: E402

from src.config import SimConfig  # noqa: E402
from src.io_manager import RunFile  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.time_integration import sensor_cells  # noqa: E402

RESULTS = ROOT / "results" / "builder72"
LOGS = RESULTS / "logs"
TABLES = HERE / "tables72.md"
REPORT = ROOT / "docs" / "reports" / "phase5_sensitivity.md"
GATE = ROOT / "results" / "builder69"
STEP4 = ROOT / "results" / "builder64" / "coarse"
C_MU = 0.09
RESOLVED_FLOOR = 1e-2
EVENT_START = 10.0
EVENT_END = {"filter_breach": 70.0, "door_leak": 15.0, "equipment_dust": 40.0}
SCENARIO_TITLES = {
    "filter_breach": "The filter breach",
    "door_leak": "The door leak",
    "equipment_dust": "The equipment dust release",
}
# The sensor rows (y of the sensors) and the first row over the tool tops.
ROWS_M = (2.5, 2.05, 1.5, 1.2)
SUMMARY_CLASS = 0  # 0.1 um
DUST_EXTRA_CLASS = 4  # 5 um, whose dust contour is the widest (the prediction)
MARKERS = ("runs", "breach", "door", "dust", "grids", "cleanout")
SCENARIO_MARKER = {
    "filter_breach": "breach",
    "door_leak": "door",
    "equipment_dust": "dust",
}
FLOW_LINE = re.compile(
    r"^\s+([0-9a-f]{12}): (rest|warm|resumed|run file|kept)(?: from ([0-9a-f]{12}))?, "
    r"(\d+) outer, ([a-z_]+), ([0-9.]+) s"
)


def parse_name(name: str) -> tuple[str, str, str]:
    """(scenario, arm, grid) from a queue name such as filter_breach_max_nut_160x60."""
    scenario = next(s for s in configs72.SCENARIOS if name.startswith(s + "_"))
    rest, grid = name[len(scenario) + 1 :].rsplit("_", 1)
    return scenario, rest, grid


@dataclass(frozen=True)
class Summary:
    """The runner's summary, read back: flow states, steps and wall time."""

    flows: list[tuple[str, str, int, str, float]]  # prefix, origin, outer, stop, wall
    steps: int
    t_end: float
    wall: float


def read_summary(path: Path) -> Summary:
    flows = []
    steps, t_end, wall = 0, math.nan, math.nan
    for line in path.read_text(encoding="utf-8").splitlines():
        found = FLOW_LINE.match(line)
        if found:
            origin = found.group(2)
            if found.group(3):
                origin = f"{origin} from {found.group(3)}"
            flows.append(
                (
                    found.group(1),
                    origin,
                    int(found.group(4)),
                    found.group(5),
                    float(found.group(6)),
                )
            )
        elif line.startswith("steps: "):
            parts = line.split()
            steps, t_end = int(parts[1]), float(parts[5])
        elif line.startswith("wall time: "):
            wall = float(line.split()[2])
    return Summary(flows, steps, t_end, wall)


class Run:
    """One run's records: configuration, mesh, frames, sensors and summary."""

    def __init__(self, name: str, scenario_path: Path, run_path: Path) -> None:
        self.name = name
        self.config = SimConfig(scenario_path)
        self.mesh = Mesh(self.config)
        mapping = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
        base_path = Path(mapping["scenario"]["base"])
        if not base_path.is_absolute():
            base_path = scenario_path.parent / base_path
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        supply = base["boundaries"]["hepa_supply"]
        self.intensity = float(supply["turbulence_intensity"])
        self.length = float(supply["dissipation_length"])
        self.schmidt = float(base["transport"]["turbulent_schmidt"])
        self.u_in = float(supply["velocity"])
        self.file = RunFile(run_path)
        self.summary = read_summary(run_path.with_suffix(".txt"))
        self.frames = self.file.read_frames()
        self.sensors = self.file.read_sensor_readings()
        self.thresholds = self.config.class_thresholds()
        self.sizes_um = [s * 1e6 for s in self.config.particle_sizes]
        self.cells = sensor_cells(self.config, self.mesh)
        self.live = self.mesh.cell_type != SOLID
        self.dx = float(self.mesh.dx_cell[0])
        self.xc, self.yc = np.asarray(self.mesh.xc), np.asarray(self.mesh.yc)
        self._ever: dict[int, np.ndarray] = {}

    @property
    def grid(self) -> str:
        return f"{self.config.nx}x{self.config.ny}"

    @property
    def nu_t_in(self) -> float:
        k_in = 1.5 * (self.intensity * self.u_in) ** 2
        return C_MU * math.sqrt(k_in) * self.length

    def ever_reached(self, k: int) -> np.ndarray:
        """[ny, nx] bool: non-SOLID cells at or above the threshold in any frame."""
        if k not in self._ever:
            field = self.frames.fields[k]
            self._ever[k] = (field >= self.thresholds[k]).any(axis=0) & self.live
        return self._ever[k]

    def distances(self, k: int) -> list[tuple[str, float, bool]]:
        """Per sensor: name, distance to the contour (m), the sensor cell reached."""
        ever = self.ever_reached(k)
        out = []
        if not ever.any():
            return [(name, math.inf, False) for name in self.sensors.names]
        jj, ii = np.nonzero(ever)
        for s, (j, i) in enumerate(self.cells):
            d2 = (self.xc[ii] - self.xc[i]) ** 2 + (self.yc[jj] - self.yc[j]) ** 2
            out.append(
                (self.sensors.names[s], float(np.sqrt(d2.min())), bool(ever[j, i]))
            )
        return out

    def row_index(self, y: float) -> int:
        """The cell holding the height, the one below it when it sits on a face."""
        faces = np.asarray(self.mesh.y)
        return int(np.searchsorted(faces, y - 1e-9 * faces[-1], side="right") - 1)

    def widths(self, k: int, y: float) -> tuple[float, float, float, float, str]:
        """Width at the threshold and at ten times it, at the event's end and largest."""
        scenario = parse_name(self.name)[0]
        at_end = int(np.argmin(np.abs(self.frames.times - EVENT_END[scenario])))
        j = self.row_index(y)
        row = self.frames.fields[k][:, j, :] * self.live[j][None, :]
        threshold = self.thresholds[k]
        n1 = (row >= threshold).sum(axis=1)
        n10 = (row >= 10 * threshold).sum(axis=1)
        hit = np.flatnonzero(row[at_end] >= threshold)
        extent = (
            f"{self.xc[hit[0]] - self.dx / 2:.2f} to {self.xc[hit[-1]] + self.dx / 2:.2f}"
            if hit.size
            else "none"
        )
        dx = self.dx
        return n1[at_end] * dx, n1.max() * dx, n10[at_end] * dx, n10.max() * dx, extent

    def peaks(self, k: int) -> list[tuple[str, str, str]]:
        """Per sensor: name, peak over threshold (or the floor's words), first time."""
        out = []
        for s, name in enumerate(self.sensors.names):
            series = self.sensors.readings[:, k, s]
            above = np.flatnonzero(series >= self.thresholds[k])
            first = f"{self.sensors.times[above[0]]:.3f}" if above.size else "never"
            margin = series.max() / self.thresholds[k]
            peak = (
                f"{margin:.3e}"
                if margin >= RESOLVED_FLOOR
                else f"below the resolved plume (< {RESOLVED_FLOOR:g})"
            )
            out.append((name, peak, first))
        return out


def failure(name: str) -> str:
    """Why a run has no summary: the runner's last error line, an unconverged
    source of its flow state, or neither (running or interrupted)."""
    scenario, arm, grid = parse_name(name)
    depends = configs72.FLOWS_FROM.get(arm)
    if depends is not None and grid == configs72.grid_tag(configs72.COARSE):
        source = RESULTS / f"{scenario}_{depends}_{grid}.txt"
        if not (source.exists() and "wall time:" in source.read_text(encoding="utf-8")):
            return f"not run: its flow state is {depends}'s, which did not converge"
    log = LOGS / f"{name}.log"
    if not log.exists():
        return "not run"
    errors = [
        line.removeprefix("error: ").strip()
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith("error: ")
    ]
    if errors:
        return "not run: " + errors[-1].split(" (", 1)[0]
    return "no summary: running or interrupted"


def load_runs() -> dict[str, Run | None]:
    """Every sweep run in queue order (None where it did not complete), then the
    gate's three event runs at the file's inputs on 320x120 (prompt 69), which
    stand as the control arm on the product grid."""
    runs: dict[str, Run | None] = {}
    for name, scenario, _arm, _grid in configs72.run_names():
        if scenario == "clean_out":
            continue
        run_path = RESULTS / f"{name}.h5"
        summary = RESULTS / f"{name}.txt"
        if (
            run_path.exists()
            and summary.exists()
            and "wall time:" in summary.read_text(encoding="utf-8")
        ):
            runs[name] = Run(
                name, RESULTS / "configs" / f"scenario_{name}.yaml", run_path
            )
        else:
            runs[name] = None
    for scenario in configs72.SCENARIOS:
        gate_run = GATE / f"{scenario}.h5"
        if gate_run.exists():
            runs[f"{scenario}_control_320x120 (gate, prompt 69)"] = Run(
                f"{scenario}_control_320x120",
                GATE / "configs" / f"scenario_{scenario}.yaml",
                gate_run,
            )
    return runs


def label(run: Run, name: str) -> tuple[str, str]:
    """(scenario, arm) from the run's queue name; the gate's row says so."""
    scenario, arm, _grid = parse_name(run.name)
    if "gate" in name:
        arm = "control (gate, prompt 69)"
    return scenario, arm


def fmt_d(distance: float) -> str:
    return "no cell reached" if math.isinf(distance) else f"{distance:.3f}"


# ---------------------------------------------------------------------------
# The full tables (tables72.md)
# ---------------------------------------------------------------------------


def r1(runs: dict[str, Run | None]) -> list[str]:
    nu = None
    lines = [
        "### R1. The runs: inputs, the supply's inlet eddy viscosity, the flow states and "
        "the march",
        "",
        "| Run | Grid | Intensity I | Dissipation length l_e (m) | Sc_t | Supply nu_t "
        "(m^2/s) | nu_t / nu | Flow states: origin, outer, stop, wall (s) | March steps | "
        "t at the end (s) | Run wall (s) | Frames |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        if run is None:
            lines.append(f"| {name} | {failure(name)} | | | | | | | | | | |")
            continue
        nu = run.config.mu / run.config.rho
        flows = "; ".join(
            f"{prefix} {origin}, {outer:,}, {stop}, {wall:.0f}"
            for prefix, origin, outer, stop, wall in run.summary.flows
        )
        lines.append(
            f"| {name} | {run.grid} | {run.intensity:.2f} | {run.length:.2f} | "
            f"{run.schmidt:.1f} | {run.nu_t_in:.2e} | {run.nu_t_in / nu:.0f} | {flows} | "
            f"{run.summary.steps:,} | {run.summary.t_end:.1f} | {run.summary.wall:.0f} | "
            f"{len(run.frames.times)} |"
        )
    return lines


def d1(runs: dict[str, Run | None]) -> list[str]:
    lines = [
        "### D1. The plume's edge against the sensors: the distance from each sensor's cell "
        "to the nearest non-SOLID cell ever at or above the class's threshold (centre to "
        "centre), in metres and in cells, per run and class",
        "",
        "| Scenario | Arm | Grid | Class (um) | Sensor | Distance to the threshold contour "
        "(m) | In cells | Sensor cell reached |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        if run is None:
            continue
        scenario, arm = label(run, name)
        for k, size in enumerate(run.sizes_um):
            for sensor, distance, reached in run.distances(k):
                cells = "" if math.isinf(distance) else f"{distance / run.dx:.1f}"
                lines.append(
                    f"| {scenario} | {arm} | {run.grid} | {size:g} | {sensor} | "
                    f"{fmt_d(distance)} | {cells} | {'yes' if reached else 'no'} |"
                )
    return lines


def w1(runs: dict[str, Run | None]) -> list[str]:
    lines = [
        "### W1. The plume's width on the sensor rows and the first row over the tool tops: "
        "cells at or above the threshold and at ten times it, times dx; at the frame nearest "
        "the event's end (70 s breach, 15 s door, 40 s dust) and the largest over the frames",
        "",
        "| Scenario | Arm | Grid | Class (um) | Row y (m) | Width at the threshold at the "
        "end (m) | Largest at the threshold (m) | Width at 10x at the end (m) | Largest at "
        "10x (m) | Extent of the threshold cells at the end (m) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        if run is None:
            continue
        scenario, arm = label(run, name)
        for k, size in enumerate(run.sizes_um):
            for y in ROWS_M:
                w1_end, w1_max, w10_end, w10_max, extent = run.widths(k, y)
                lines.append(
                    f"| {scenario} | {arm} | {run.grid} | {size:g} | {y:.2f} | {w1_end:.2f} | "
                    f"{w1_max:.2f} | {w10_end:.2f} | {w10_max:.2f} | {extent} |"
                )
    return lines


def p1(runs: dict[str, Run | None]) -> list[str]:
    lines = [
        "### P1. The sensors: peak reading over threshold and first time at or above, per "
        f"run and class; a peak below {RESOLVED_FLOOR:g} of threshold is the front's "
        "numerical tail and is reported as below the resolved plume",
        "",
        "| Scenario | Arm | Grid | Class (um) | Sensor | Peak over threshold | First time at "
        "or above (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        if run is None:
            continue
        scenario, arm = label(run, name)
        for k, size in enumerate(run.sizes_um):
            for sensor, peak, first in run.peaks(k):
                lines.append(
                    f"| {scenario} | {arm} | {run.grid} | {size:g} | {sensor} | {peak} | "
                    f"{first} |"
                )
    return lines


# ---------------------------------------------------------------------------
# The clean-out
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanOut:
    label: str
    grid: str
    t_final: float
    room: float
    mean: float
    ratio: float
    well_mixed: float
    free: int
    unrecovered: dict[int, int]  # class index to cells still above at the end
    sizes_um: list[float]


def read_clean_out(
    label_text: str, scenario_path: Path, run_path: Path
) -> CleanOut | None:
    summary = run_path.with_suffix(".txt")
    if not run_path.exists() or not summary.exists():
        return None
    if "wall time:" not in summary.read_text(encoding="utf-8"):  # still running
        return None
    config = SimConfig(scenario_path)
    mesh = Mesh(config)
    live = mesh.cell_type != SOLID
    record = RunFile(run_path).read_recovery()
    unrecovered = {
        k: int((np.isinf(record.local_recovery[k]) & live).sum())
        for k in range(len(config.particle_sizes))
    }
    return CleanOut(
        label_text,
        f"{config.nx}x{config.ny}",
        float(record.t_final),
        float(record.room_recovery_time),
        float(record.mean_time[SUMMARY_CLASS]),
        float(record.ratio[SUMMARY_CLASS]),
        float(record.well_mixed_time),
        int(live.sum()),
        unrecovered,
        [s * 1e6 for s in config.particle_sizes],
    )


def clean_outs() -> list[CleanOut | None]:
    return [
        read_clean_out(
            "80x30, step 4 (prompt 64; G1's row, run to 120 s)",
            STEP4 / "scenario_clean_out_nohold_80x30.yaml",
            STEP4 / "clean_out_nohold.h5",
        ),
        read_clean_out(
            "80x30, this pass (to 240 s)",
            RESULTS / "configs" / "scenario_clean_out_80x30.yaml",
            RESULTS / "clean_out_80x30.h5",
        ),
        read_clean_out(
            "160x60, this pass",
            RESULTS / "configs" / "scenario_clean_out_160x60.yaml",
            RESULTS / "clean_out_160x60.h5",
        ),
        read_clean_out(
            "320x120, the gate (prompt 69; G1's row)",
            GATE / "configs" / "scenario_clean_out.yaml",
            GATE / "clean_out.h5",
        ),
    ]


def observed_order(coarse: float, medium: float, fine: float) -> str:
    """Richardson's order from three values at a refinement ratio of two."""
    if any(math.isnan(v) for v in (coarse, medium, fine)):
        return "not defined (a value is not reached)"
    a, b = coarse - medium, medium - fine
    if a == 0.0 or b == 0.0:
        return "not defined (two values equal)"
    if a * b < 0.0:
        return f"none: the series is not monotone (differences {a:+.4g} then {b:+.4g})"
    return f"{math.log2(a / b):.2f} (differences {a:+.4g} then {b:+.4g})"


def c1(runs_clean: list[CleanOut | None]) -> list[str]:
    lines = [
        "### C1. The clean-out on three grids: section C's outputs (ADR-014 C) at 0.1 um, "
        "and the cells not recovered at the end of the run",
        "",
        "| Run | Grid | Run to t (s) | Well-mixed 100:1 time ln(100)/N (s) | Room recovery "
        "time (s) | Free-area mean 100:1 (s) | Ratio to the well-mixed | Free cells | Not "
        "recovered at the end, 0.1 um (cells, share) | 5 um |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in runs_clean:
        if c is None:
            lines.append("| not run | | | | | | | | | |")
            continue
        u0, u4 = c.unrecovered[0], c.unrecovered[4]
        lines.append(
            f"| {c.label} | {c.grid} | {c.t_final:.1f} | {c.well_mixed:.4f} | {c.room:.4f} | "
            f"{c.mean:.4f} | {c.ratio:.4f} | {c.free:,} | {u0:,} ({100 * u0 / c.free:.2f}%) | "
            f"{u4:,} ({100 * u4 / c.free:.2f}%) |"
        )
    series = [c for c in runs_clean[1:] if c is not None]
    if len(series) == 3:
        coarse, medium, fine = series
        lines += [
            "",
            "Observed order of convergence over 80x30 (to 240 s), 160x60 and 320x120 "
            "(refinement ratio 2; p = log2 of the ratio of successive differences, defined "
            "when the three values change monotonically):",
            "",
            "| Quantity | 80x30 | 160x60 | 320x120 | Observed order |",
            "|---|---|---|---|---|",
            f"| Room recovery time (s) | {coarse.room:.4f} | {medium.room:.4f} | "
            f"{fine.room:.4f} | {observed_order(coarse.room, medium.room, fine.room)} |",
            f"| Free-area mean 100:1 (s) | {coarse.mean:.4f} | {medium.mean:.4f} | "
            f"{fine.mean:.4f} | {observed_order(coarse.mean, medium.mean, fine.mean)} |",
            f"| Ratio to the well-mixed | {coarse.ratio:.4f} | {medium.ratio:.4f} | "
            f"{fine.ratio:.4f} | {observed_order(coarse.ratio, medium.ratio, fine.ratio)} |",
            "| Not recovered at 240 s, 0.1 um (share of the free cells) | "
            + " | ".join(
                f"{100 * c.unrecovered[0] / c.free:.3f}%"
                for c in (coarse, medium, fine)
            )
            + " | "
            + observed_order(
                *(100 * c.unrecovered[0] / c.free for c in (coarse, medium, fine))
            )
            + " |",
        ]
    return lines


# ---------------------------------------------------------------------------
# The report's summary blocks
# ---------------------------------------------------------------------------


def runs_block(runs: dict[str, Run | None]) -> list[str]:
    lines = [
        "| Run | Grid | Flow states: origin, outer, stop, wall (s) | March steps | Run wall "
        "(s) |",
        "|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        if "gate" in name:
            continue
        if run is None:
            lines.append(f"| {name} | {failure(name)} | | | |")
            continue
        flows = "; ".join(
            f"{origin}, {outer:,}, {stop}, {wall:.0f}"
            for _prefix, origin, outer, stop, wall in run.summary.flows
        )
        lines.append(
            f"| {name} | {run.grid} | {flows} | {run.summary.steps:,} | "
            f"{run.summary.wall:.0f} |"
        )
    record_path = RESULTS / "runs72.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        starts = sorted(r["started"] for r in record.values())
        ends = sorted(r["ended"] for r in record.values() if "ended" in r)
        if starts and ends:
            from datetime import datetime

            span = datetime.fromisoformat(ends[-1]) - datetime.fromisoformat(starts[0])
            lines += [
                "",
                f"The queue ran from {starts[0]} to {ends[-1]} (machine local time), "
                f"{span.total_seconds() / 3600:.2f} hours of wall clock, at most five runs at "
                f"a time, each on one BLAS thread (`runs72.json`).",
            ]
    return lines


def scenario_runs(runs: dict[str, Run | None], scenario: str) -> list[tuple[str, Run]]:
    rows = []
    for name, run in runs.items():
        if run is None or not run.name.startswith(scenario):
            continue
        rows.append((label(run, name)[1], run))
    return rows


def scenario_block(runs: dict[str, Run | None], scenario: str) -> list[str]:
    rows = scenario_runs(runs, scenario)
    if not rows:
        return ["(no run of this scenario is complete)"]
    sensors = rows[0][1].sensors.names
    size0 = rows[0][1].sizes_um[SUMMARY_CLASS]
    lines = [
        f"Distance from each sensor's cell to the threshold contour, {size0:g} um, metres "
        "(cells); a sensor whose own cell the contour reaches is marked REACHED:",
        "",
        "| Arm | Grid | " + " | ".join(sensors) + " |",
        "|---|---|" + "---|" * len(sensors),
    ]
    reached_any = []
    for arm, run in rows:
        cells = []
        for sensor, distance, reached in run.distances(SUMMARY_CLASS):
            if reached:
                cells.append("REACHED")
                reached_any.append((arm, run.grid, sensor))
            else:
                cells.append(
                    "no cell reached"
                    if math.isinf(distance)
                    else f"{distance:.2f} ({distance / run.dx:.1f})"
                )
        lines.append(f"| {arm} | {run.grid} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The same over every class, the shortest to the longest distance (m); REACHED names "
        "the classes whose contour reaches the sensor's cell:",
        "",
        "| Arm | Grid | " + " | ".join(sensors) + " |",
        "|---|---|" + "---|" * len(sensors),
    ]
    for arm, run in rows:
        cells = []
        for s in range(len(sensors)):
            per_class = [run.distances(k)[s] for k in range(len(run.sizes_um))]
            hits = [
                f"{run.sizes_um[k]:g}" for k, (_n, _d, r) in enumerate(per_class) if r
            ]
            finite = [d for _n, d, _r in per_class if not math.isinf(d)]
            if hits:
                cells.append("REACHED at " + ", ".join(hits) + " um")
            elif not finite:
                cells.append("no cell reached")
            else:
                cells.append(f"{min(finite):.2f} to {max(finite):.2f}")
                if len(finite) < len(per_class):
                    cells[-1] += " (no cell reached in some classes)"
        lines.append(f"| {arm} | {run.grid} | " + " | ".join(cells) + " |")
    classes = [SUMMARY_CLASS]
    if scenario == "equipment_dust":
        classes.append(DUST_EXTRA_CLASS)
    for k in classes:
        size = rows[0][1].sizes_um[k]
        lines += [
            "",
            f"The plume's width on the sensor rows, {size:g} um: the largest over the frames "
            "of the cells at or above the threshold, times dx, and at ten times it, in metres "
            "(threshold / 10x):",
            "",
            "| Arm | Grid | " + " | ".join(f"y = {y:.2f} m" for y in ROWS_M) + " |",
            "|---|---|" + "---|" * len(ROWS_M),
        ]
        for arm, run in rows:
            cells = []
            for y in ROWS_M:
                _w1e, w1m, _w10e, w10m, _ext = run.widths(k, y)
                cells.append(f"{w1m:.2f} / {w10m:.2f}")
            lines.append(f"| {arm} | {run.grid} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Sensor peaks (P1 of `tables72.md`): "
        + (
            "every sensor of every class at every arm is below the resolved plume (under a "
            "hundredth of its threshold)."
            if not any_peak_resolved(rows)
            else "see P1; at least one sensor reads within a hundredth of its threshold."
        ),
    ]
    if reached_any:
        lines += [
            "",
            "Sensor cells reached at 0.1 um: "
            + "; ".join(f"{s} at {a} on {g}" for a, g, s in reached_any)
            + ".",
        ]
    return lines


def any_peak_resolved(rows: list[tuple[str, Run]]) -> bool:
    for _arm, run in rows:
        for k in range(len(run.sizes_um)):
            for s in range(len(run.sensors.names)):
                series = run.sensors.readings[:, k, s]
                if series.max() / run.thresholds[k] >= RESOLVED_FLOOR:
                    return True
    return False


def grids_block(runs: dict[str, Run | None]) -> list[str]:
    lines = [
        "The two extreme arms on both grids, 0.1 um: the distance from each sensor's cell to "
        "the contour (m), 160x60 against 320x120, and the plume's largest width at the "
        "threshold on the gap sensors' row (y = 2.50 m) and the first row over the tool tops "
        "(y = 2.05 m). A positive difference means the fine grid's contour is farther from "
        "the sensor.",
        "",
        "| Scenario | Arm | Sensor | 160x60 (m) | 320x120 (m) | Fine minus coarse (m) | "
        "Width at y 2.50, 160x60 / 320x120 (m) | Width at y 2.05, 160x60 / 320x120 (m) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scenario in configs72.SCENARIOS:
        for arm in configs72.FINE_ARMS:
            coarse = runs.get(f"{scenario}_{arm}_160x60")
            fine = runs.get(f"{scenario}_{arm}_320x120")
            if coarse is None or fine is None:
                missing = "160x60" if coarse is None else "320x120"
                why = failure(f"{scenario}_{arm}_{missing}")
                lines.append(f"| {scenario} | {arm} | {missing} {why} | | | | | |")
                continue
            wc = {y: coarse.widths(SUMMARY_CLASS, y)[1] for y in (2.5, 2.05)}
            wf = {y: fine.widths(SUMMARY_CLASS, y)[1] for y in (2.5, 2.05)}
            for (sensor, dc, _rc), (_s, df, _rf) in zip(
                coarse.distances(SUMMARY_CLASS),
                fine.distances(SUMMARY_CLASS),
                strict=True,
            ):
                diff = "" if math.isinf(dc) or math.isinf(df) else f"{df - dc:+.3f}"
                lines.append(
                    f"| {scenario} | {arm} | {sensor} | {fmt_d(dc)} | {fmt_d(df)} | {diff} | "
                    f"{wc[2.5]:.2f} / {wf[2.5]:.2f} | {wc[2.05]:.2f} / {wf[2.05]:.2f} |"
                )
    return lines


def cleanout_block(runs_clean: list[CleanOut | None]) -> list[str]:
    return c1(runs_clean)[2:]


# ---------------------------------------------------------------------------
# Writing and checking
# ---------------------------------------------------------------------------


def replace_block(text: str, marker: str, body: list[str]) -> str:
    begin = f"<!-- tables72 {marker} begin -->"
    end = f"<!-- tables72 {marker} end -->"
    start = text.index(begin) + len(begin)
    stop = text.index(end)
    return text[:start] + "\n" + "\n".join(body) + "\n" + text[stop:]


def generate() -> tuple[str, str]:
    runs = load_runs()
    clean = clean_outs()
    tables = [
        "# Prompt 72 sweep tables",
        "",
        "Generated by `tables72.py` from `results/builder72/`, `results/builder69/` and "
        "`results/builder64/coarse/` (untracked, this machine); `--check` regenerates and "
        "compares. The readings are in the script's docstring.",
        "",
        *r1(runs),
        "",
        *d1(runs),
        "",
        *w1(runs),
        "",
        *p1(runs),
        "",
        *c1(clean),
    ]
    report = REPORT.read_text(encoding="utf-8")
    blocks = {
        "runs": runs_block(runs),
        "grids": grids_block(runs),
        "cleanout": cleanout_block(clean),
    }
    for scenario, marker in SCENARIO_MARKER.items():
        blocks[marker] = scenario_block(runs, scenario)
    for marker in MARKERS:
        report = replace_block(report, marker, blocks[marker])
    return "\n".join(tables) + "\n", report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tables, report = generate()
    if args.check:
        same = TABLES.read_text(encoding="utf-8") == tables
        same_report = REPORT.read_text(encoding="utf-8") == report
        if not same:
            print("tables72.md differs from the records")
        if not same_report:
            print("phase5_sensitivity.md's generated blocks differ from the records")
        if same and same_report:
            print("tables72.md and the report's blocks match the records")
            return 0
        return 1
    TABLES.write_text(tables, encoding="utf-8")
    REPORT.write_text(report, encoding="utf-8")
    print(f"wrote {TABLES} and the blocks of {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
