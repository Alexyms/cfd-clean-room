"""The tables of the conservation budget pass, from its records (prompt 78).

    python docs/reports/probe78/tables78.py            # writes tables78.md
    python docs/reports/probe78/tables78.py --splice   # and the report's marked copies
    python docs/reports/probe78/tables78.py --check    # regenerates and compares

Reads results/builder78/ (untracked, this machine): per run and side the run
file RUN.h5, its summary RUN.txt and RUN.meta.json from ``probe78.py``, and
for the probe side RUN.steps.npz; plants.json from the plant runner
(``results/builder78/plant78.py``); harness_flow.jsonl, the flow rows retaken;
and benchmarks/results.jsonl, the rows of record. Prompt 77's records
(results/builder77/, its R1 to R4) are read for the comparison the predictions
name. Every number the report states about the runs is in a table here; the
report copies the tables between markers and ``--check`` compares both the
written file and the report's copies with what the records give. Do not edit
tables78.md by hand.

A step or frame is scored when its class's supply is above zero, as prompt
77's tables score them.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import configs78  # noqa: E402 -- follows sys.path.insert
import probe78  # noqa: E402 -- follows sys.path.insert

from src.boundary_concentration import (  # noqa: E402 -- follows sys.path.insert
    ConcentrationBoundary,
)
from src.boundary_registry import (  # noqa: E402 -- follows sys.path.insert
    BoundaryRegistry,
)
from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.io_manager import RunFile  # noqa: E402 -- follows sys.path.insert
from src.mesh import SOLID, Mesh  # noqa: E402 -- follows sys.path.insert
from src.monitor import sensor_cells  # noqa: E402 -- follows sys.path.insert
from src.particles import ParticlePhysics  # noqa: E402 -- follows sys.path.insert
from src.solver_transport import (  # noqa: E402 -- follows sys.path.insert
    TransportSolver,
)
from src.staggered import FaceVelocities  # noqa: E402 -- follows sys.path.insert
from src.time_integration import map_latencies  # noqa: E402 -- follows sys.path.insert
from validation.transport_cases import (  # noqa: E402 -- follows sys.path.insert
    conservation_keys,
)

RESULTS = configs78.RESULTS
B77 = ROOT / "results" / "builder77"
TABLES = HERE / "tables78.md"
REPORT = ROOT / "docs" / "reports" / "conservation_budget.md"
RECORDS = ROOT / "benchmarks" / "results.jsonl"
FLOW_ROWS = RESULTS / "harness_flow.jsonl"
MAIN_ROWS = RESULTS / "harness_main.jsonl"
SIZES_UM = ("0.1", "0.3", "0.5", "1", "5")
SCENARIOS = {
    "R1": "dust release",
    "R2": "clean-out",
    "R3": "door leak",
    "R4": "filter breach",
    "SL": "seal leak",
}
CLOSURE = 1.0e-12  # prediction (a)
# Two leftovers within this of each other are equal to the rounding of their sums.
EQUAL_RTOL = 1.0e-9
WALL_RISE = 0.15  # prediction (c)
UNIT_ROUNDOFF = 2.0**-53
# The 640x240 step count G2's extrapolation reads: twice the 320x120 run's
# (prompt 77's R10, 33,119 steps), the step halving with the cell at a fixed
# Courant number.
R10_STEPS = 33_119
PRODUCT_320 = ROOT / "results" / "builder69" / "equipment_dust.h5"
MARKERS = (
    "predictions",
    "calibration",
    "branch_main",
    "closure",
    "bound",
    "sweeps",
    "wall",
    "latency",
    "latency_changes",
    "flow_states",
    "harness_flow",
    "harness_transport",
    "plants",
    "floor",
)


def fmt(x: float, digits: int = 4) -> str:
    """A number in the e format, or a dash for None or NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{digits}e}"


def yes(flag: bool) -> str:
    return "yes" if flag else "NO"


class Steps:
    """A probe run's per-step rows, one array per class, by probe78.COLUMNS."""

    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.columns = [str(c) for c in data["columns"]]
        self.rows = {
            int(name.split("_")[1]): data[name]
            for name in data
            if name.startswith("class_")
        }

    def col(self, k: int, name: str) -> np.ndarray:
        return self.rows[k][:, self.columns.index(name)]


def meta(name: str, side: str) -> dict:
    return json.loads(
        (RESULTS / side / f"{name}.meta.json").read_text(encoding="utf-8")
    )


def summary(name: str, side: str) -> str:
    return (RESULTS / side / f"{name}.txt").read_text(encoding="utf-8")


def flow_seconds(text: str) -> float:
    """The wall seconds of the flow states a run solved itself, from its summary."""
    total = 0.0
    for line in text.splitlines():
        found = re.match(
            r"\s+[0-9a-f]{12}: (rest|warm|resumed)\b.*, ([0-9.]+) s$", line
        )
        if found:
            total += float(found.group(2))
    return total


def march_seconds(name: str, side: str) -> float:
    """The run's wall time less the flow solves it made."""
    return meta(name, side)["wall_seconds"] - flow_seconds(summary(name, side))


def prompt77_steps(name: str) -> np.ndarray | None:
    """Prompt 77's per-step series of a run, or None for a run it did not make."""
    path = B77 / f"{name}.steps.npz"
    return np.load(path) if path.exists() else None


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------


def calibration_table() -> list[str]:
    lines = [
        "### C1. The recorder's calibration: the probe run against the branch run, every stored frame",
        "",
        "| Run | Frames | Fields bitwise at every frame | Budgets bitwise at every frame | Sensor readings bitwise | Arrival maps bitwise |",
        "|---|---|---|---|---|---|",
    ]
    for run in configs78.RUNS:
        a = RunFile(configs78.run_path(run.name, "branch"))
        b = RunFile(configs78.run_path(run.name, "probe"))
        fa, fb = a.read_frames(), b.read_frames()
        fields = fa.steps.tolist() == fb.steps.tolist() and all(
            fa.fields[k].tobytes() == fb.fields[k].tobytes() for k in fa.fields
        )
        budgets = all(
            _budget_bytes(x) == _budget_bytes(y)
            for xs, ys in zip(fa.budgets, fb.budgets, strict=True)
            for x, y in zip(xs, ys, strict=True)
        )
        ra, rb = a.read_sensor_readings(), b.read_sensor_readings()
        readings = ra.readings.tobytes() == rb.readings.tobytes()
        aa, ab = a.read_arrival(), b.read_arrival()
        arrival = aa.first_above.tobytes() == ab.first_above.tobytes()
        lines.append(
            f"| {run.name} ({SCENARIOS[run.name]}) | {len(fa.steps):,} | {yes(fields)} | "
            f"{yes(budgets)} | {yes(readings)} | {yes(arrival)} |"
        )
    return lines


def _budget_bytes(b: object, leftover: bool = True) -> bytes:
    values = [
        b.initial,
        b.inflow,
        b.outflow,
        b.source,
        *b.deposited.values(),
        b.current,
    ]
    if leftover:
        values.append(b.leftover)
    return np.array([np.nan if v is None else v for v in values]).tobytes()


def branch_main_table() -> list[str]:
    """The branch's run against main's at every stored frame: where the share never
    binds the two are the same run."""
    lines = [
        "### M1. The branch run against main's, every stored frame",
        "",
        "The budgets compared without the leftover, which main's files do not hold.",
        "",
        "| Run | Frames | Frames whose fields are bitwise main's | First frame that differs (step) | Budgets bitwise at every frame |",
        "|---|---|---|---|---|",
    ]
    for run in configs78.RUNS:
        fa = RunFile(configs78.run_path(run.name, "branch")).read_frames()
        fb = RunFile(configs78.run_path(run.name, "main")).read_frames()
        assert fa.steps.tolist() == fb.steps.tolist()
        same = [
            all(
                fa.fields[k][i].tobytes() == fb.fields[k][i].tobytes()
                for k in fa.fields
            )
            for i in range(len(fa.steps))
        ]
        first = next((int(fa.steps[i]) for i, s in enumerate(same) if not s), None)
        budgets = all(
            _budget_bytes(x, leftover=False) == _budget_bytes(y, leftover=False)
            for xs, ys in zip(fa.budgets, fb.budgets, strict=True)
            for x, y in zip(xs, ys, strict=True)
        )
        lines.append(
            f"| {run.name} ({SCENARIOS[run.name]}) | {len(same)} | {sum(same)} | "
            f"{'none' if first is None else f'{first:,}'} | {yes(budgets)} |"
        )
    return lines


def closure_table() -> list[str]:
    lines = [
        "### A1. Closure with the leftover booked: the largest |residual| over the supply (prediction (a))",
        "",
        "| Run | Stored frames scored | Largest at a stored frame, over the classes | Steps scored (probe) | Largest at any step | Within 1e-12 at every frame and step |",
        "|---|---|---|---|---|---|",
    ]
    for run in configs78.RUNS:
        frames = RunFile(configs78.run_path(run.name, "branch")).read_frames()
        worst_frame, scored = 0.0, 0
        for budgets in frames.budgets:
            for b in budgets:
                if b.current is None or b.supplied() == 0.0:
                    continue
                scored += 1
                worst_frame = max(worst_frame, abs(b.residual()) / b.supplied())
        steps = Steps(RESULTS / "probe" / f"{run.name}.steps.npz")
        worst_step, scored_steps = 0.0, 0
        for k in steps.rows:
            s = steps.col(k, "supplied")
            r = steps.col(k, "residual")
            live = s > 0.0
            scored_steps += int(live.sum())
            if live.any():
                worst_step = max(worst_step, float(np.max(np.abs(r[live]) / s[live])))
        ok = worst_frame <= CLOSURE and worst_step <= CLOSURE
        lines.append(
            f"| {run.name} ({SCENARIOS[run.name]}) | {scored:,} | {fmt(worst_frame, 3)} | "
            f"{scored_steps:,} | {fmt(worst_step, 3)} | {yes(ok)} |"
        )
    return lines


def _before_end(run: configs78.Run, k: int) -> tuple[float, str]:
    """The leftover the run left before this change at its end: prompt 77's T5, or main's residual."""
    old = prompt77_steps(run.name)
    if old is not None:
        columns = [str(c) for c in old["columns"]]
        return float(old[f"class_{k}"][:, columns.index("T5")].sum()), "prompt 77's T5"
    budget = RunFile(configs78.run_path(run.name, "main")).read_frames().budgets[-1][k]
    return budget.residual(), "main's residual"


def bound_table(config: SimConfig) -> list[str]:
    f_l = config.transport.leftover_share * config.transport.conservation_limit
    lines = [
        "### B1. The accumulated leftover against its bound, every step (prediction (b))",
        "",
        f"f L = {f_l:g}. The step's share is f L S_n dt_n / T and the bound f L S(t) t / T, "
        "below f L S(t) for t <= T.",
        "",
        "| Run | Class (um) | Largest abs(step leftover) over its share | Largest abs(accumulated) over f L S(t) t / T | Largest abs(accumulated) over f L S(t) | Over L S(t) | Accumulated at the end | Before this change at the end | Read from | Smaller in magnitude |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in configs78.RUNS:
        steps = Steps(RESULTS / "probe" / f"{run.name}.steps.npz")
        horizon = SimConfig(configs78.scenario_path(run.name, "branch")).t_end
        for k in sorted(steps.rows):
            s = steps.col(k, "supplied")
            dt = steps.col(k, "dt")
            t = np.cumsum(dt)
            acc = steps.col(k, "leftover")
            step_left = steps.col(k, "step_leftover")
            share = steps.col(k, "share")
            live = s > 0.0
            if not live.any():
                lines.append(
                    f"| {run.name} | {SIZES_UM[k]} | nothing supplied | - | - | - | "
                    f"{fmt(float(acc[-1]))} | - | - | - |"
                )
                continue
            per_step = (
                float(np.max(np.abs(step_left[live]) / share[live]))
                if live.any()
                else 0.0
            )
            over_t = float(
                np.max(np.abs(acc[live]) / (f_l * s[live] * t[live] / horizon))
            )
            over_fl = float(np.max(np.abs(acc[live]) / (f_l * s[live])))
            end = float(acc[-1])
            before, source = _before_end(run, k)
            # Prompt 77 summed the residual in another order, so a run the share
            # never changed differs from its T5 by rounding only: equal, not smaller.
            if abs(abs(end) - abs(before)) <= EQUAL_RTOL * abs(before):
                smaller = "equal"
            else:
                smaller = yes(abs(end) < abs(before))
            lines.append(
                f"| {run.name} | {SIZES_UM[k]} | {per_step:.4f} | {over_t:.4f} | {over_fl:.4f} | "
                f"{over_fl * f_l / config.transport.conservation_limit:.4f} | {fmt(end)} | "
                f"{fmt(before)} | {source} | {smaller} |"
            )
    return lines


def sweeps_table() -> list[str]:
    lines = [
        "### K1. Sweeps per solve: the branch, the max-norm stop alone on the same systems, and prompt 77 (prediction (c))",
        "",
        "| Run | Steps | Solves | Mean sweeps, branch | Mean, the max-norm stop alone | Mean, prompt 77 | Rise over prompt 77 | Under one | Solves the share made sweep on | Steps with one such solve or more | First such step, t (s) | Last, t (s) | Largest extra sweeps | Solves at the cap |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in configs78.RUNS:
        steps = Steps(RESULTS / "probe" / f"{run.name}.steps.npz")
        times = np.cumsum(steps.col(0, "dt"))
        sweeps = np.stack([steps.col(k, "sweeps") for k in sorted(steps.rows)])
        free = np.stack([steps.col(k, "sweeps_free") for k in sorted(steps.rows)])
        conv = np.stack([steps.col(k, "converged") for k in sorted(steps.rows)])
        old = prompt77_steps(run.name)
        if old is not None:
            columns = [str(c) for c in old["columns"]]
            old_mean = float(
                np.mean(
                    [old[f"class_{k}"][:, columns.index("sweeps")] for k in range(5)]
                )
            )
            rise = float(sweeps.mean()) - old_mean
            old_text, rise_text, under = (
                f"{old_mean:.2f}",
                f"{rise:+.2f}",
                yes(rise < 1.0),
            )
        else:
            old_text, rise_text, under = "-", "-", "-"
        extra = sweeps - free
        binding = np.nonzero((extra > 0).any(axis=0))[0]
        first = f"{times[binding[0]]:.2f}" if binding.size else "-"
        last = f"{times[binding[-1]]:.2f}" if binding.size else "-"
        lines.append(
            f"| {run.name} ({SCENARIOS[run.name]}) | {sweeps.shape[1]:,} | {sweeps.size:,} | "
            f"{sweeps.mean():.2f} | {free.mean():.2f} | {old_text} | {rise_text} | {under} | "
            f"{int((extra > 0).sum()):,} | {int((extra > 0).any(axis=0).sum()):,} | "
            f"{first} | {last} | "
            f"{int(extra.max())} | {int((conv == 0).sum())} |"
        )
    return lines


def wall_table() -> list[str]:
    lines = [
        "### W1. March wall time: the branch, main, and prompt 77 (prediction (c))",
        "",
        "The march's wall time is the run's less the flow solves it made itself (SL solves its "
        "leaking state on both sides). The branch and main runs of a scenario ran side by side "
        "in the same queue. Prompt 77's runs carried its recorder, which wrapped every step.",
        "",
        "| Run | Branch (s) | Main (s) | Prompt 77 (s) | Branch over main | Branch over prompt 77 | Rise under 15% against prompt 77 | Against main |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in configs78.RUNS:
        branch = march_seconds(run.name, "branch")
        main = march_seconds(run.name, "main")
        path = B77 / f"{run.name}.probe.json"
        old = (
            json.loads(path.read_text(encoding="utf-8"))["wall_seconds"]
            if path.exists()
            else None
        )
        old_text = f"{old:.0f}" if old is not None else "-"
        ratio77 = f"{branch / old:.3f}" if old is not None else "-"
        under77 = yes(branch / old - 1.0 < WALL_RISE) if old is not None else "-"
        lines.append(
            f"| {run.name} ({SCENARIOS[run.name]}) | {branch:.0f} | {main:.0f} | {old_text} | "
            f"{branch / main:.3f} | {ratio77} | {under77} | {yes(branch / main - 1.0 < WALL_RISE)} |"
        )
    return lines


def _latencies(path: Path) -> tuple[np.ndarray, list[str], list[str], float]:
    """The arrival map's latency at each sensor's cell, [events, classes, sensors]."""
    run = RunFile(path)
    mapping = run.read_config().mapping
    # A run file written before ADR-016 has no conservation keys; the sensors,
    # the mesh and the arrival maps read none of them.
    mapping["transport"] = conservation_keys() | mapping["transport"]
    config = SimConfig.from_dict(mapping)
    arrival = run.read_arrival()
    cells = sensor_cells(config, Mesh(config))
    names = [s.name for s in config.sensors]
    return (
        map_latencies(arrival, cells),
        list(arrival.event_names),
        names,
        arrival.t_final,
    )


def latency_tables() -> tuple[list[str], list[str]]:
    lines = [
        "### L1. Detection latencies on 80x30 against the run before this change (prediction (e))",
        "",
        "The arrival map's latency at each sensor's cell, which the monitor's equals to the step "
        "(ADR-015 D); every event, class and sensor. One step is the run's largest step.",
        "",
        "| Run | Before | Entries | Reached in both | Never in both | Equal | Differ | Largest difference (s) | In steps | Within one step |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    changes = [
        "### L2. The latencies that differ",
        "",
        "| Run | Event | Class (um) | Sensor | Before (s) | After (s) | Difference (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    any_change = False
    for run in configs78.RUNS:
        before_path = run.before or configs78.run_path(run.name, "main")
        before, events, names, _ = _latencies(before_path)
        after, events_b, names_b, _ = _latencies(configs78.run_path(run.name, "branch"))
        assert events == events_b and names == names_b
        steps = Steps(RESULTS / "probe" / f"{run.name}.steps.npz")
        dt = float(steps.col(0, "dt").max())
        both = np.isfinite(before) & np.isfinite(after)
        never = np.isinf(before) & np.isinf(after)
        equal = (before == after) | never
        diff = np.zeros_like(before)
        diff[both] = np.abs(after[both] - before[both])
        mismatch = np.isfinite(before) != np.isfinite(after)
        largest = float(diff.max()) if both.any() else 0.0
        within = not mismatch.any() and largest <= dt * (1.0 + 1e-9)
        lines.append(
            f"| {run.name} ({SCENARIOS[run.name]}) | {before_path.relative_to(ROOT).as_posix()} | "
            f"{before.size} | {int(both.sum())} | {int(never.sum())} | {int(equal.sum())} | "
            f"{int((~equal).sum())} | {largest:.6f} | {largest / dt:.2f} | {yes(within)} |"
        )
        for e, k, s in zip(*np.nonzero(~equal), strict=True):
            any_change = True
            changes.append(
                f"| {run.name} | {events[e]} | {SIZES_UM[k]} | {names[s]} | "
                f"{before[e, k, s]:.6f} | {after[e, k, s]:.6f} | "
                f"{after[e, k, s] - before[e, k, s]:+.6f} |"
            )
    if not any_change:
        changes.append("| none | | | | | | |")
    return lines, changes


def flow_state_table() -> list[str]:
    lines = [
        "### F1. The flow states (prediction (d))",
        "",
        "| Run | State | How the branch run came by it | Against | u, v, p, k, eps, nu_t bitwise | Outer iterations |",
        "|---|---|---|---|---|---|",
    ]
    fields = ("u", "v", "p", "k", "eps", "nu_t")
    for run in configs78.RUNS:
        branch = RunFile(configs78.run_path(run.name, "branch"))
        main = RunFile(configs78.run_path(run.name, "main"))
        source = RunFile(run.flows_from)
        solved = set(
            re.findall(r"\s+([0-9a-f]{12}): (?:rest|warm)", summary(run.name, "branch"))
        )
        for fp in branch.flow_fingerprints():
            mine = branch.read_flow_state(fp).state
            if fp[:12] in solved:
                theirs, against, how = (
                    main.read_flow_state(fp).state,
                    "main's solve",
                    "solved",
                )
            else:
                theirs = source.read_flow_state(fp).state
                against, how = run.flows_from.relative_to(ROOT).as_posix(), "kept"
            same = all(
                np.ascontiguousarray(getattr(mine, f)).tobytes()
                == np.ascontiguousarray(getattr(theirs, f)).tobytes()
                for f in fields
            )
            lines.append(
                f"| {run.name} | {fp[:12]} | {how} | {against} | {yes(same)} | "
                f"{mine.outer} against {theirs.outer} |"
            )
    return lines


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


# A trajectory sample's keys that a rerun cannot repeat or that are not the
# solve's: the wall clock, and the accuracy value the harness copies into the
# last sample (scripts/benchmark.py), compared with the accuracy instead.
UNCOMPUTED_SAMPLE_KEYS = ("elapsed_seconds", "error")


def _samples(trajectory: list[dict]) -> dict[int, str]:
    """A trajectory's samples by outer iteration, without the keys above."""
    return {
        s["outer_iteration"]: json.dumps(
            {k: v for k, v in s.items() if k not in UNCOMPUTED_SAMPLE_KEYS},
            sort_keys=True,
        )
        for s in trajectory
    }


def _compared(a: dict, b: dict) -> tuple[bool, int]:
    """Whether two rows' computed parts are equal, and the trajectory samples compared.

    The work and outcome whole; the trajectory at the outer iterations both
    rows sampled, since a row may be taken at another sample interval.
    """
    same = all(
        json.dumps(a[k], sort_keys=True) == json.dumps(b[k], sort_keys=True)
        for k in ("work", "outcome")
    )
    sa, sb = _samples(a["trajectory"]), _samples(b["trajectory"])
    common = sorted(set(sa) & set(sb))
    return same and bool(common) and all(sa[i] == sb[i] for i in common), len(common)


def _same_row(a: dict, b: dict) -> bool:
    """The computed parts of two rows: work, outcome and the common samples."""
    return _compared(a, b)[0]


def _accuracy_differences(a: object, b: object, path: str = "") -> list[str]:
    """The accuracy keys whose recorded values differ, with both values."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            out += _accuracy_differences(a.get(key), b.get(key), f"{path}{key}.")
        return out
    if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
        return [f"{path.rstrip('.')} {a} to {b}"]
    return []


def harness_tables() -> tuple[list[str], list[str]]:
    records = _rows(RECORDS)
    new_flow = _rows(FLOW_ROWS) if FLOW_ROWS.exists() else []
    lines = [
        "### H1. The flow harness rows retaken (prediction (d))",
        "",
        "Each against the last row of record of the case under the method of record. Bitwise "
        "means the row's work and outcome, and its trajectory at every outer iteration both "
        "rows sampled (residuals, outer counts, cell updates; the wall clock and the accuracy "
        "copy in the last sample left out), are equal as recorded, every float to the last "
        "bit. The accuracy record's keys that differ are named with both values. A row whose "
        "accuracy differs from its record is also retaken on origin/main "
        "(results/builder78/harness_main.jsonl) and compared whole, accuracy included.",
        "",
        "| Case | Row of record (commit) | Outer, record | Outer, retaken | Samples compared | Wall (s), record | Retaken | Bitwise | Accuracy keys that differ from the record | Main retaken, every key equal to the branch's |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    main_rows = {r["case"]: r for r in _rows(MAIN_ROWS)} if MAIN_ROWS.exists() else {}
    for row in new_flow:
        old = [
            r for r in records
            if r["case"] == row["case"] and r["method"] == row["method"]
            and r["recorded_at"] < row["recorded_at"]
        ][-1]  # fmt: skip
        same, samples = _compared(old, row)
        changed = _accuracy_differences(old["accuracy"], row["accuracy"])
        main = main_rows.get(row["case"])
        if main is None:
            against_main = "-"
        else:
            against_main = yes(
                _compared(main, row)[0]
                and not _accuracy_differences(main["accuracy"], row["accuracy"])
            )
        shown = "; ".join(changed[:3]) + (
            f"; {len(changed) - 3} more" if len(changed) > 3 else ""
        )
        lines.append(
            f"| {row['case']} | {old['run_id'][:8]} ({old['git_commit'][:7]}) | "
            f"{old['work']['outer_iterations']:,} | {row['work']['outer_iterations']:,} | "
            f"{samples} | {old['time']['wall_seconds']:.0f} | {row['time']['wall_seconds']:.0f} | "
            f"{yes(same)} | {shown or 'none'} | {against_main} |"
        )
    transport = [
        "### H2. The harness rows whose transport output changes: VAL-024 retaken",
        "",
        "The flow part (work, outcome, trajectory) is compared bitwise; the scalar march and "
        "the scored widths are the rows' old and new values, to the last digit a double holds.",
        "",
        "| Case | Old row (commit) | New row (commit) | Flow part bitwise | Largest scored departure, old | New | Decay exponent, old | New | Width ratio 20 M, old | New | 50 M, old | New | March steps, old | New | March wall (s), old | New |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in ("val024_280x50", "val024_560x100"):
        rows = [r for r in records if r["case"] == case]
        new = rows[-1]
        old = [r for r in rows if r["git_commit"] != new["git_commit"]][-1]
        flow = _same_row(old, new)
        a, b = old["accuracy"], new["accuracy"]
        ca, cb = a["components"], b["components"]
        ma, mb = (
            a["report"]["march"] if "march" in a["report"] else a["march"],
            b["march"],
        )
        transport.append(
            f"| {case} | {old['run_id'][:8]} ({old['git_commit'][:7]}) | "
            f"{new['run_id'][:8]} ({new['git_commit'][:7]}) | {yes(flow)} | "
            f"{a['value']:.17g} | {b['value']:.17g} | {ca['decay_exponent']:.17g} | "
            f"{cb['decay_exponent']:.17g} | {ca['width_ratio_20M']:.17g} | "
            f"{cb['width_ratio_20M']:.17g} | {ca['width_ratio_50M']:.17g} | "
            f"{cb['width_ratio_50M']:.17g} | {ma['steps']:,} | {mb['steps']:,} | "
            f"{ma['wall_seconds']:.1f} | {mb['wall_seconds']:.1f} |"
        )
    return lines, transport


def plants_table() -> list[str]:
    plants = json.loads((RESULTS / "plants.json").read_text(encoding="utf-8"))
    lines = [
        "### P2. The planted defects (results/builder78/plant78.py)",
        "",
        "Each applied to a copy of the tree at the code commit and the tests that claim it run "
        "there; `none` runs every one of those tests on the unchanged copy.",
        "",
        "| Plant | File | Tests run | Outcome | Caught |",
        "|---|---|---|---|---|",
    ]
    for name, r in plants.items():
        caught = "baseline" if name == "none" else yes(bool(r["caught"]))
        tests = len(r["tests"])
        lines.append(
            f"| {name} | {r['file'] or '-'} | {tests} | {r['summary']} | {caught} |"
        )
    return lines


def _kappa(path: Path) -> tuple[float, float, int]:
    """The largest dt sum_f G_f / V over the cells of a run file's first state, its dt, and the cells."""
    run = RunFile(path)
    config = SimConfig.from_dict(run.read_config().mapping)
    mesh = Mesh(config)
    physics = ParticlePhysics(config)
    solver = TransportSolver(
        mesh,
        config,
        physics,
        ConcentrationBoundary(mesh, config, physics, BoundaryRegistry(config)),
    )
    state = run.read_flow_state(run.flow_fingerprints()[0]).state
    faces = FaceVelocities.copy_of(state.u, state.v)
    nu_t = np.where(mesh.cell_type == SOLID, 0.0, state.nu_t)
    dt = min(solver.stable_dt(faces, k) for k in range(len(config.particle_sizes)))
    nu_u, nu_v = solver._face_eddy_viscosity(nu_t)
    schmidt = config.transport.turbulent_schmidt
    d_b = max(physics.diffusion_coeff(k) for k in range(len(config.particle_sizes)))
    g_u = (d_b + nu_u / schmidt) * solver._conductance_u
    g_v = (d_b + nu_v / schmidt) * solver._conductance_v
    total = g_u[:, :-1] + g_u[:, 1:] + g_v[:-1, :] + g_v[1:, :]
    volume = np.outer(mesh.dy_cell, mesh.dx_cell)
    live = mesh.cell_type != SOLID
    return float(np.max((dt * total / volume)[live])), dt, int(live.sum())


def floor_table(config: SimConfig) -> list[str]:
    f_l = config.transport.leftover_share * config.transport.conservation_limit
    lines = [
        "### R1. The share against the leftover's rounding floor",
        "",
        "The floor of a step's leftover is about c u (1 + kappa) Q (ADR-016), u the unit "
        "roundoff, Q the content and kappa the largest dt sum_f G_f / V. The measured column "
        "is the smallest share over u Q at any scored step of the run; dividing by c (1 + "
        "kappa), c about ten, gives the margin.",
        "",
        "| Run or grid | Steps | Largest kappa | Smallest share over u Q, measured | Margin at c = 10 |",
        "|---|---|---|---|---|",
    ]
    for run in configs78.RUNS:
        steps = Steps(RESULTS / "probe" / f"{run.name}.steps.npz")
        kappa, _, _ = _kappa(configs78.run_path(run.name, "branch"))
        smallest = math.inf
        for k in steps.rows:
            share = steps.col(k, "share")
            content = steps.col(k, "current")
            live = (share > 0.0) & (content > 0.0)
            if live.any():
                smallest = min(
                    smallest,
                    float(np.min(share[live] / (UNIT_ROUNDOFF * content[live]))),
                )
        if math.isinf(smallest):
            measured = margin = "nothing supplied"
        else:
            measured = fmt(smallest, 3)
            margin = fmt(smallest / (10.0 * (1.0 + kappa)), 3)
        lines.append(
            f"| {run.name} ({SCENARIOS[run.name]}, 80x30) | {len(steps.col(0, 'dt')):,} | "
            f"{kappa:.4f} | {measured} | {margin} |"
        )
    kappa, _, _ = _kappa(PRODUCT_320)
    floor_320 = f_l / (R10_STEPS * UNIT_ROUNDOFF)
    lines.append(
        f"| 320x120 dust release (results/builder69, prompt 77's R10 steps) | {R10_STEPS:,} | "
        f"{kappa:.4f} | {fmt(floor_320, 3)} (f L / (n u), Q = S) | "
        f"{fmt(floor_320 / (10.0 * (1.0 + kappa)), 3)} |"
    )
    steps_640 = 2 * R10_STEPS
    floor_640 = f_l / (steps_640 * UNIT_ROUNDOFF)
    lines.append(
        f"| 640x240, extrapolated (twice the steps and twice kappa of 320x120) | {steps_640:,} | "
        f"{2 * kappa:.4f} | {fmt(floor_640, 3)} (f L / (n u), Q = S) | "
        f"{fmt(floor_640 / (10.0 * (1.0 + 2 * kappa)), 3)} |"
    )
    return lines


def predictions_table(scores: dict[str, tuple[str, str]]) -> list[str]:
    stated = {
        "a": "every 80x30 run, stored frame and class: abs(residual) within 1e-12 of the supply",
        "b": "accumulated leftover within f L S(t) at every step; ends smaller than prompt 77's T5",
        "c": "mean sweeps per solve rise by less than one, march wall by less than 15%, against prompt 77",
        "d": "every flow harness row and flow state reproduces bitwise",
        "e": "every 80x30 detection latency within one time step of the run before",
    }
    lines = [
        "### P1. Prompt 78's predictions against the measurements",
        "",
        "| Prediction | As stated (abridged) | Measured | Outcome |",
        "|---|---|---|---|",
    ]
    for key in "abcde":
        measured, outcome = scores[key]
        lines.append(f"| ({key}) | {stated[key]} | {measured} | {outcome} |")
    return lines


def score(tables: dict[str, list[str]]) -> dict[str, tuple[str, str]]:
    """Each prediction scored from its table's last column."""

    def column(name: str, index: int) -> list[str]:
        rows = [
            r
            for r in tables[name]
            if r.startswith("| ") and not r.startswith("| Run |")
        ]
        rows = [
            r
            for r in rows
            if not r.startswith("| Case |") and not r.startswith("| Prediction")
        ]
        return [r.strip("|").split("|")[index].strip() for r in rows]

    a = column("closure", 5)
    # The seal leak supplies nothing on 80x30 and scores no row ("-").
    b_bound = [x for x in column("bound", 4) if x != "-"]
    b_small = [x for x in column("bound", 9) if x != "-"]
    c_sweeps = [x for x in column("sweeps", 7) if x != "-"]
    c_wall = [x for x in column("wall", 6) if x != "-"]
    d_flow = (
        column("flow_states", 4)
        + column("harness_flow", 7)
        + [x for x in column("harness_flow", 9) if x != "-"]
        + column("harness_transport", 3)
    )
    e = column("latency", 9)
    worst_b = max(float(x) for x in b_bound)
    runs_b = [
        x
        for x, s in zip(column("bound", 0), column("bound", 9), strict=True)
        if s != "-"
    ]
    failing_b = sorted({r for r, s in zip(runs_b, b_small, strict=True) if s != "yes"})

    def verdict(flags: list[str]) -> str:
        return "holds" if all(f == "yes" for f in flags) else "FAILS"

    return {
        "a": (f"{a.count('yes')} of {len(a)} runs within it; table A1", verdict(a)),
        "b": (
            f"largest abs(accumulated) over f L S(t) {worst_b:.4f}; smaller at the end in "
            f"{b_small.count('yes')} of {len(b_small)} run-classes; table B1",
            ("first clause holds" if worst_b <= 1.0 else "first clause FAILS")
            + (
                "; second holds"
                if all(f == "yes" for f in b_small)
                else f"; second FAILS in: {', '.join(failing_b)}"
            ),
        ),
        "c": (
            f"sweeps under one in {c_sweeps.count('yes')} of {len(c_sweeps)} runs, wall under "
            f"15% in {c_wall.count('yes')} of {len(c_wall)}; tables K1, W1",
            verdict(c_sweeps + c_wall),
        ),
        "d": (
            f"{d_flow.count('yes')} of {len(d_flow)} states and rows bitwise; tables F1, H1, H2",
            verdict(d_flow),
        ),
        "e": (
            f"{e.count('yes')} of {len(e)} runs within one step; tables L1, L2",
            verdict(e),
        ),
    }


def build() -> tuple[str, dict[str, str]]:
    config = SimConfig(configs78.scenario_path("R1", "branch"))
    tables: dict[str, list[str]] = {}
    tables["calibration"] = calibration_table()
    tables["branch_main"] = branch_main_table()
    tables["closure"] = closure_table()
    tables["bound"] = bound_table(config)
    tables["sweeps"] = sweeps_table()
    tables["wall"] = wall_table()
    tables["latency"], tables["latency_changes"] = latency_tables()
    tables["flow_states"] = flow_state_table()
    tables["harness_flow"], tables["harness_transport"] = harness_tables()
    tables["plants"] = plants_table()
    tables["floor"] = floor_table(config)
    tables["predictions"] = predictions_table(score(tables))
    head = [
        "# The tables of prompt 78 (generated by tables78.py; do not edit)",
        "",
        f"Columns of the probe's per-step rows: {', '.join(probe78.COLUMNS)}.",
        "",
    ]
    body = []
    for name in MARKERS:
        body.extend(tables[name])
        body.append("")
    return "\n".join(head + body), {
        name: "\n".join(lines) for name, lines in tables.items()
    }


def splice(text: str, tables: dict[str, str]) -> str:
    """The report's text with each marked region replaced by its table."""
    for name, body in tables.items():
        begin = f"<!-- tables78 {name} begin -->"
        end = f"<!-- tables78 {name} end -->"
        if begin not in text or end not in text:
            continue
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{body}\n{end}{tail}"
    return text


def report_copies(text: str) -> dict[str, str]:
    copies = {}
    for name in MARKERS:
        begin = f"<!-- tables78 {name} begin -->"
        end = f"<!-- tables78 {name} end -->"
        if begin in text and end in text:
            copies[name] = text[
                text.index(begin) + len(begin) : text.index(end)
            ].strip()
    return copies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splice", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text, tables = build()
    if args.check:
        ok = TABLES.read_text(encoding="utf-8") == text + "\n"
        copies = report_copies(REPORT.read_text(encoding="utf-8"))
        stale = [n for n, body in copies.items() if body != tables[n].strip()]
        missing = [n for n in MARKERS if n not in copies]
        print("tables78.md", "matches" if ok else "DIFFERS")
        print("report copies:", "match" if not stale else f"DIFFER: {stale}")
        if missing:
            print("report has no copy of:", missing)
        return 0 if ok and not stale else 1
    TABLES.write_text(text + "\n", encoding="utf-8")
    if args.splice:
        REPORT.write_text(
            splice(REPORT.read_text(encoding="utf-8"), tables), encoding="utf-8"
        )
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
