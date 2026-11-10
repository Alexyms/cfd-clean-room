"""The tables and the figure of the budget residual probe, from its records (prompt 77).

    python docs/reports/probe77/tables77.py             # writes tables77.md
    python docs/reports/probe77/tables77.py --figures   # and the figure under docs/reports/
    python docs/reports/probe77/tables77.py --splice    # and the report's marked copies
    python docs/reports/probe77/tables77.py --check     # regenerates and compares

Reads results/builder77/ (untracked, this machine): calibration.json from
``probe77.py --calibrate``; per run, the per-step series RUN.steps.npz and the
record RUN.probe.json from ``probe77.py RUN``, and the run file RUN.h5 that
scripts/run_scenario.py wrote in the same process, whose stored frames hold the
budgets the terms are checked against; and each run's source run file
(configs77.FLOWS) for the comparison of budgets at the stored frames. Every
number the report states about the runs is in a table here; the report copies
the tables between markers and ``--check`` compares both the written file and
the report's copies with what the records give. Do not edit tables77.md by
hand.

A frame is scored when the particles supplied (initial + inflow + source) are
above zero, as table B1 of the Phase 4 report scores them, and the worst frame
is the one of largest relative residual among them.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import configs77  # noqa: E402 -- follows sys.path.insert

from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.io_manager import (  # noqa: E402 -- follows sys.path.insert
    RunFile,
    TransportFrames,
)
from src.mesh import SOLID, Mesh  # noqa: E402 -- follows sys.path.insert
from src.solver_transport import MassBudget  # noqa: E402 -- follows sys.path.insert

RESULTS = ROOT / "results" / "builder77"
TABLES = HERE / "tables77.md"
REPORT = ROOT / "docs" / "reports" / "budget_residual_mechanism.md"
FIGURE = ROOT / "docs" / "reports" / "budget_residual_r10.png"
SPLIT = RESULTS / "split"
SIZES_UM = ("0.1", "0.3", "0.5", "1", "5")
TERMS = ("T1", "T2", "T3", "T4", "T5", "T6")
OTHERS = ("T1", "T2", "T3", "T4", "T6")
SCENARIO_NAMES = {
    "equipment_dust": "dust release",
    "clean_out": "clean-out",
    "door_leak": "door leak",
    "filter_breach": "filter breach",
}
# The predictions' run sets (prompt 77).
AT_FILE_TOL = ("R1", "R2", "R3", "R4", "R9", "R10")
REFERENCE = ("R1", "R2", "R3", "R4")
DOSE = {"equipment_dust": ("R5", "R1", "R7"), "clean_out": ("R6", "R2", "R8")}
GRIDS = ("R1", "R9", "R10")
# Prediction (c)'s frame intervals for R10's 0.1 um class (table B2 of the Phase 4 report).
SIGN_WINDOWS = ((10.5, 10.9), (25.0, 25.4))
# R10's per-step window (prompt 77, section 1).
STEP_WINDOW = (9.0, 27.0)
MARKERS = (
    "calibration",
    "runs",
    "identity",
    "end",
    "worst",
    "closure",
    "predictions",
    "signs",
    "dose",
    "grid",
    "split",
)


@dataclass
class Run:
    """One run's records."""

    name: str
    spec: configs77.Run
    meta: dict
    columns: list[str]
    series: dict[int, np.ndarray]
    frames: TransportFrames
    source: TransportFrames | None
    cells: int

    def col(self, k: int, name: str) -> np.ndarray:
        """Class k's per-step column."""
        return self.series[k][:, self.columns.index(name)]

    def cumulative(self, k: int, name: str) -> np.ndarray:
        """Class k's column summed from the first step, per step."""
        return np.cumsum(self.col(k, name))


def load(name: str) -> Run | None:
    """A run's records, or None when it has not run."""
    probe = RESULTS / f"{name}.probe.json"
    if not probe.exists():
        return None
    meta = json.loads(probe.read_text(encoding="utf-8"))
    data = np.load(RESULTS / f"{name}.steps.npz")
    columns = [str(c) for c in data["columns"]]
    series = {
        int(key.split("_")[1]): data[key] for key in data.files if key != "columns"
    }
    spec = configs77.BY_NAME[name]
    frames = RunFile(RESULTS / f"{name}.h5").read_frames()
    source = None
    if spec.flows_from.exists():
        try:
            source = RunFile(spec.flows_from).read_frames()
        except KeyError:
            source = None
    config = SimConfig(configs77.OUT / f"scenario_{name}.yaml")
    mesh = Mesh(config)
    cells = int(np.count_nonzero(mesh.cell_type != SOLID))
    return Run(name, spec, meta, columns, series, frames, source, cells)


def label(run: Run) -> str:
    """The run's name, scenario, grid and tolerance."""
    return (
        f"{run.name} ({SCENARIO_NAMES[run.spec.scenario]}, "
        f"{configs77.grid_tag(run.spec.grid)}, {run.spec.tolerance:g})"
    )


def supplied(budget: MassBudget) -> float:
    """initial + inflow + source, zero before the first step."""
    if budget.initial is None:
        return 0.0
    return budget.initial + budget.inflow + budget.source


def scored(run: Run, k: int) -> list[int]:
    """Indices of the stored frames that are scored for class k."""
    return [
        i
        for i, frame in enumerate(run.frames.budgets)
        if frame[k].current is not None and supplied(frame[k]) > 0.0
    ]


def worst_frame(run: Run, k: int) -> int:
    """The scored frame of largest relative residual."""
    rows = scored(run, k)
    return max(rows, key=lambda i: abs(run.frames.budgets[i][k].relative()))


def at_frame(run: Run, k: int, i: int) -> dict[str, float]:
    """The terms accumulated to stored frame i, and the frame's residual."""
    step = int(run.frames.steps[i])
    out = {name: float(np.sum(run.col(k, name)[:step])) for name in (*TERMS, "gap")}
    out["R"] = run.frames.budgets[i][k].residual()
    out["t"] = float(run.frames.times[i])
    out["step"] = step
    return out


def e(value: float) -> str:
    """Four significant figures, scientific."""
    return f"{value:.4e}"


def pct(value: float) -> str:
    """A ratio as a percentage, three significant figures."""
    return f"{100.0 * value:.3g}%"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A markdown table."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out


def calibration_table() -> list[str]:
    """C1: the in-process calibration."""
    path = RESULTS / "calibration.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        [
            "R1's configuration, steps 1 to 2,000 (the release starts at step 569)",
            f"{record['steps_plain']:,}",
            f"{record['steps_wrapped']:,}",
            str(record["classes"]),
            f"{record['steps_differing']:,}",
            "yes" if record["bitwise"] else "NO",
            "yes" if record["rebuilt_every_step"] else "NO",
        ]
    ]
    return [
        "### C1. The in-process calibration (probe77.py --calibrate)",
        "",
        *table(
            [
                "Window",
                "Steps unwrapped",
                "Steps wrapped",
                "Classes",
                "Steps whose fields or budgets differ",
                "Bitwise at every step",
                "Advection rebuilt bit for bit at every step",
            ],
            rows,
        ),
    ]


def source_comparison(run: Run) -> str:
    """The budgets at the stored frames against the source run's, bitwise or not."""
    if run.source is None:
        return "no source frames"
    ours = dict(zip(run.frames.steps.tolist(), run.frames.budgets, strict=True))
    theirs = dict(zip(run.source.steps.tolist(), run.source.budgets, strict=True))
    common = sorted(set(ours) & set(theirs))
    differing = 0
    largest = 0.0
    for step in common:
        if ours[step] != theirs[step]:
            differing += 1
            for a, b in zip(ours[step], theirs[step], strict=True):
                if a.current is not None and b.current is not None:
                    largest = max(largest, abs(a.residual() - b.residual()))
    last = common[-1] if common else 0
    if differing == 0:
        return f"bitwise at all {len(common)} common frames (to step {last:,})"
    return (
        f"{differing} of {len(common)} common frames differ (to step {last:,}); "
        f"largest residual difference {e(largest)}"
    )


def runs_table(runs: list[Run]) -> list[str]:
    """M1: the runs, their sweeps, T5 against its bound, and the source comparison."""
    rows = []
    for run in runs:
        sweeps = np.concatenate([run.col(k, "sweeps") for k in run.series])
        capped = int(
            sum(np.count_nonzero(run.col(k, "converged") == 0.0) for k in run.series)
        )
        ratios = []
        for k in run.series:
            bound = run.col(k, "bound")
            live = bound > 0.0
            ratios.append(run.col(k, "T5")[live] / bound[live])
        ratio = np.concatenate(ratios)
        rebuilt = all(bool(np.all(run.col(k, "rebuilt") == 1.0)) for k in run.series)
        steps = run.series[0].shape[0]
        first = run.col(0, "sweeps")
        equal = int(
            np.count_nonzero(
                np.all([run.col(k, "sweeps") == first for k in run.series], axis=0)
            )
        )
        rows.append(
            [
                label(run),
                f"{run.spec.sweep_cap}",
                f"{run.meta['flows_from']}",
                f"{steps:,}",
                f"{run.frames.times[-1]:.4f}",
                f"{run.meta['wall_seconds']:.0f}",
                str(run.meta["exit"]),
                f"{sweeps.mean():.2f}",
                f"{int(sweeps.max())}",
                f"{capped}",
                f"{equal:,} of {steps:,}",
                f"{np.mean(ratio):.3e}",
                f"{np.max(np.abs(ratio)):.3e}",
                "yes" if rebuilt else "NO",
                source_comparison(run),
            ]
        )
    return [
        "### M1. The runs (five classes at one step, Courant 0.4)",
        "",
        *table(
            [
                "Run (scenario, grid, diffusion_tol)",
                "Sweep cap",
                "Flow states from",
                "Steps",
                "t at the end (s)",
                "Wall (s)",
                "Exit",
                "Sweeps per solve, mean",
                "largest",
                "Solves at the cap",
                "Steps at which the five classes took the same sweeps",
                "T5 per step over its bound, mean",
                "largest magnitude",
                "Advection rebuilt bit for bit",
                "Budgets at the stored frames against the source run",
            ],
            rows,
        ),
    ]


def identity_table(runs: list[Run]) -> list[str]:
    """I1: the terms the derivation puts at zero, largest over the classes, against R."""
    rows = []
    for run in runs:
        largest_r = max(
            abs(run.frames.budgets[i][k].residual())
            for k in run.series
            for i in scored(run, k)
        )
        sums = {
            name: max(abs(float(np.sum(run.col(k, name)))) for k in run.series)
            for name in TERMS
        }
        per_step = {
            name: max(float(np.max(np.abs(run.col(k, name)))) for k in run.series)
            for name in ("T1", "T3")
        }
        removed = max(
            float(np.max(np.abs(run.col(k, "solid_removed")))) for k in run.series
        )
        rows.append(
            [
                label(run),
                e(largest_r),
                e(per_step["T1"]),
                e(sums["T2"]),
                e(per_step["T3"]),
                e(removed),
                e(sums["T4"]),
                e(sums["T6"]),
                e(sums["T5"]),
            ]
        )
    return [
        "### I1. The identity against the code as built: the terms it puts at zero, largest over the five classes",
        "",
        *table(
            [
                "Run",
                "Largest residual magnitude over the scored frames",
                "T1, largest per step",
                "T2 over the run",
                "T3, largest per step",
                "Particles the SOLID zeroing removed, largest per step",
                "T4 over the run",
                "T6 over the run",
                "T5 over the run (for scale)",
            ],
            rows,
        ),
    ]


def terms_table(runs: list[Run], which: str) -> list[str]:
    """T1 to T6 and the gap per run and class, at the end or at the worst frame."""
    rows = []
    for run in runs:
        for k in sorted(run.series):
            i = len(run.frames.steps) - 1 if which == "end" else worst_frame(run, k)
            row = at_frame(run, k, i)
            r = row["R"]
            others = sum(row[n] for n in OTHERS)
            rows.append(
                [
                    run.name,
                    SIZES_UM[k],
                    f"{row['t']:.2f}",
                    e(r),
                    *(e(row[n]) for n in TERMS),
                    e(row["gap"]),
                    pct((row["T5"] - r) / r) if r != 0.0 else "R is zero",
                    pct(others / r) if r != 0.0 else "R is zero",
                ]
            )
    title = (
        "### E1. The terms at the end of each run, accumulated over the run"
        if which == "end"
        else "### W1. The terms at each run's worst stored frame (largest relative residual)"
    )
    return [
        title,
        "",
        *table(
            [
                "Run",
                "Class (um)",
                "t (s)",
                "Residual R",
                *TERMS,
                "Gap",
                "(T5 - R) / R",
                "(T1 + T2 + T3 + T4 + T6) / R",
            ],
            rows,
        ),
    ]


def closure_rows(run: Run) -> list[float]:
    """Per class: the largest |sum of T1 to T6 - R| over every frame after step 0."""
    out = []
    for k in sorted(run.series):
        total = np.cumsum(sum(run.col(k, n) for n in TERMS))
        worst = 0.0
        for i in range(1, len(run.frames.steps)):
            step = int(run.frames.steps[i])
            worst = max(
                worst, abs(total[step - 1] - run.frames.budgets[i][k].residual())
            )
        out.append(worst)
    return out


def largest_residual(run: Run) -> float:
    """The largest residual magnitude over the run's scored frames and classes."""
    return max(
        abs(run.frames.budgets[i][k].residual())
        for k in run.series
        for i in scored(run, k)
    )


def closure_table(runs: list[Run]) -> list[str]:
    """A1: prediction (a), per run."""
    rows = []
    for run in runs:
        gaps = closure_rows(run)
        scale = largest_residual(run)
        worst = max(gaps)
        rows.append(
            [
                label(run),
                e(scale),
                e(worst),
                f"{worst / scale:.2e}",
                "yes" if worst <= 1e-6 * scale else "NO",
            ]
        )
    return [
        "### A1. Closure: T1 to T6 summed from the first step against the budget's residual at every stored frame after step 0",
        "",
        *table(
            [
                "Run",
                "Largest residual magnitude",
                "Largest difference over the frames and classes",
                "Over the largest residual",
                "Within 1e-6 of it (prediction (a))",
            ],
            rows,
        ),
    ]


def sign_changes(times: np.ndarray, values: np.ndarray) -> list[tuple[float, float]]:
    """The (t before, t after) of every strict sign change between consecutive points."""
    signs = np.sign(values)
    out = []
    last = None
    for i, s in enumerate(signs):
        if s == 0.0:
            continue
        if last is not None and s != signs[last]:
            out.append((float(times[last]), float(times[i])))
        last = i
    return out


def fmt_changes(changes: list[tuple[float, float]], digits: int = 2) -> str:
    """Sign changes as "a to b; c to d", or none."""
    if not changes:
        return "none"
    return "; ".join(f"{a:.{digits}f} to {b:.{digits}f}" for a, b in changes)


def frame_series(run: Run, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scored frames' times, residuals and cumulative T5."""
    rows = scored(run, k)
    t5 = run.cumulative(k, "T5")
    times = np.array([run.frames.times[i] for i in rows])
    residual = np.array([run.frames.budgets[i][k].residual() for i in rows])
    cum = np.array([t5[int(run.frames.steps[i]) - 1] for i in rows])
    return times, residual, cum


def signs_table(runs: dict[str, Run]) -> list[str]:
    """S1: the residual's and cumulative T5's sign changes, frames and steps."""
    rows = []
    for name in (*REFERENCE, "R9", "R10"):
        run = runs.get(name)
        if run is None:
            continue
        times, residual, cum = frame_series(run, 0)
        rows.append(
            [
                label(run),
                "stored frames",
                fmt_changes(sign_changes(times, residual)),
                fmt_changes(sign_changes(times, cum)),
            ]
        )
    run = runs.get("R10")
    if run is not None:
        t = run.col(0, "t")
        inside = (t >= STEP_WINDOW[0]) & (t <= STEP_WINDOW[1])
        residual = run.col(0, "R")[inside]
        cum = run.cumulative(0, "T5")[inside]
        rows.append(
            [
                label(run),
                f"every step, {STEP_WINDOW[0]:g} to {STEP_WINDOW[1]:g} s",
                fmt_changes(sign_changes(t[inside], residual), 4),
                fmt_changes(sign_changes(t[inside], cum), 4),
            ]
        )
    return [
        "### S1. The 0.1 um class's sign changes: the residual and the cumulative T5 (between the times given, s)",
        "",
        *table(["Run", "Read at", "Residual", "Cumulative T5"], rows),
    ]


def dose_rows(runs: dict[str, Run]) -> list[dict]:
    """Per scenario and class: the end residual and its parts at the three tolerances."""
    out = []
    for scenario, names in DOSE.items():
        trio = [runs.get(n) for n in names]
        if any(r is None for r in trio):
            continue
        for k in range(len(SIZES_UM)):
            ends = [at_frame(r, k, len(r.frames.steps) - 1) for r in trio]
            out.append(
                {
                    "scenario": scenario,
                    "k": k,
                    "names": names,
                    "R": [row["R"] for row in ends],
                    "T5": [row["T5"] for row in ends],
                    "other": [row["R"] - row["T5"] for row in ends],
                    "sweeps": [float(np.mean(r.col(k, "sweeps"))) for r in trio],
                    "worst": [
                        r.frames.budgets[worst_frame(r, k)][k].relative() for r in trio
                    ],
                }
            )
    return out


def per_decade(r: list[float]) -> tuple[float, float]:
    """The end residual's factor per decade between the arms 1e-6, 1e-8 and 1e-10.

    The arms are two decades apart, so a ratio between neighbouring arms is the
    factor over two decades; its square root is the factor per decade.
    """
    return math.sqrt(r[0] / r[1]), math.sqrt(r[1] / r[2])


def decade_ok(f1: float, f2: float) -> bool:
    """Prediction (d)'s first clause: each in 3 to 30, smaller as the tolerance tightens."""
    return 3 <= f1 <= 30 and 3 <= f2 <= 30 and f2 < f1


def dose_table(runs: dict[str, Run]) -> list[str]:
    """D1: prediction (d), the end residual at 1e-6, 1e-8 and 1e-10."""
    rows = []
    for row in dose_rows(runs):
        r = row["R"]
        f1, f2 = per_decade(r)
        sweeps = row["sweeps"]
        other = row["other"]
        spread = (
            (max(other) - min(other)) / max(abs(x) for x in other)
            if any(other)
            else 0.0
        )
        rows.append(
            [
                SCENARIO_NAMES[row["scenario"]],
                SIZES_UM[row["k"]],
                *(e(x) for x in r),
                *(f"{s:.2f}" for s in row["sweeps"]),
                f"{r[0] / r[1]:.3g}",
                f"{r[1] / r[2]:.3g}",
                f"{f1:.3g}",
                f"{f2:.3g}",
                f"{(sweeps[1] - sweeps[0]) / 2:.2f}",
                f"{(sweeps[2] - sweeps[1]) / 2:.2f}",
                "yes" if decade_ok(f1, f2) else "NO",
                *(e(x) for x in other),
                f"{spread:.3g}",
                *(f"{x:.3e}" for x in row["worst"]),
                "yes" if abs(row["worst"][0]) < 1e-4 else "NO",
            ]
        )
    return [
        "### D1. The dose response on 80x30: the end residual at three tolerances two decades apart (R5/R1/R7 and R6/R2/R8)",
        "",
        *table(
            [
                "Scenario",
                "Class (um)",
                "R at 1e-6",
                "R at 1e-8",
                "R at 1e-10",
                "Sweeps per solve, 1e-6",
                "1e-8",
                "1e-10",
                "R(1e-6) / R(1e-8), two decades",
                "R(1e-8) / R(1e-10), two decades",
                "Per decade, 1e-6 to 1e-8",
                "Per decade, 1e-8 to 1e-10",
                "Sweeps per decade, 1e-6 to 1e-8",
                "1e-8 to 1e-10",
                "Per decade both in 3 to 30, the second smaller",
                "R - T5 at 1e-6",
                "at 1e-8",
                "at 1e-10",
                "Spread of R - T5 over its largest magnitude",
                "Worst relative residual over the scored frames, 1e-6",
                "1e-8",
                "1e-10",
                "At 1e-6, inside REQ-T05's 1e-4",
            ],
            rows,
        ),
    ]


def grid_rows(runs: dict[str, Run]) -> list[dict]:
    """The dust release's 0.1 um class on the three grids."""
    out = []
    for name in GRIDS:
        run = runs.get(name)
        if run is None:
            continue
        k = 0
        end = at_frame(run, k, len(run.frames.steps) - 1)
        steps = run.series[k].shape[0]
        content = run.col(k, "content")
        dt = run.col(k, "dt")
        bound = run.col(k, "bound")
        live = bound > 0.0
        out.append(
            {
                "run": run,
                "cells": run.cells,
                "steps": steps,
                "sweeps": float(np.mean(run.col(k, "sweeps"))),
                "R": end["R"],
                "T5": end["T5"],
                "per_step": end["T5"] / steps,
                "ratio": float(np.mean(run.col(k, "T5")[live] / bound[live])),
                "integral": float(np.sum(content * dt)),
                "worst": run.frames.budgets[worst_frame(run, k)][k].relative(),
                "end": run.frames.budgets[-1][k].relative(),
            }
        )
    return out


def grid_table(runs: dict[str, Run]) -> list[str]:
    """G1: how T5 scales with the grid, and the extrapolation to 640x240."""
    rows_in = grid_rows(runs)
    rows = []
    for row in rows_in:
        rows.append(
            [
                label(row["run"]),
                f"{row['cells']:,}",
                f"{row['steps']:,}",
                f"{row['sweeps']:.2f}",
                e(row["R"]),
                e(row["T5"]),
                e(row["per_step"]),
                f"{row['ratio']:.3e}",
                e(row["integral"]),
                e(row["T5"] / row["integral"]),
                f"{row['end']:.3e}",
                f"{row['worst']:.3e}",
            ]
        )
    lines = [
        "### G1. The dust release's 0.1 um class on three grids at diffusion_tol 1e-8: T5 and what it scales with",
        "",
        *table(
            [
                "Run",
                "Non-SOLID cells",
                "Steps",
                "Sweeps per solve, mean",
                "Residual at the end",
                "T5 over the run",
                "T5 per step, mean",
                "T5 per step over its bound, mean",
                "Content's time integral (particle s)",
                "T5 over the integral (1/s)",
                "Relative residual at the end",
                "Worst relative over the scored frames",
            ],
            rows,
        ),
    ]
    if len(rows_in) == 3:
        r = [row["R"] for row in rows_in]
        ratio_1 = r[1] / r[0]
        ratio_2 = r[2] / r[1]
        order = math.log2(abs(ratio_2))
        extrapolated = r[2] * ratio_2
        supplied_fine = supplied(rows_in[2]["run"].frames.budgets[-1][0])
        lines += [
            "",
            "### G2. The extrapolation to 640x240 (an extrapolation, not a measurement)",
            "",
            *table(
                [
                    "R(160x60) / R(80x30)",
                    "R(320x120) / R(160x60)",
                    "Growth per halving of the cell, log2 of the last ratio",
                    "R at 640x240 at the last ratio",
                    "Relative to the particles supplied",
                    "REQ-T05's 1e-4 over it",
                ],
                [
                    [
                        f"{ratio_1:.3g}",
                        f"{ratio_2:.3g}",
                        f"{order:.2f}",
                        e(extrapolated),
                        f"{extrapolated / supplied_fine:.3e}",
                        f"{0.0001 / abs(extrapolated / supplied_fine):.3g}",
                    ]
                ],
            ),
        ]
    return lines


def split_table(runs: dict[str, Run]) -> list[str]:
    """K1: split77.py's diagnostic, T5 carried as its diffusion and sink parts."""
    rows = []
    for spec in configs77.RUNS:
        path = SPLIT / f"{spec.name}.split.npz"
        run = runs.get(spec.name)
        if not path.exists() or run is None:
            continue
        data = np.load(path)
        columns = [str(c) for c in data["columns"]]
        for k in range(len(SIZES_UM)):
            rows_k = data[f"class_{k}"]
            col = {name: rows_k[:, columns.index(name)] for name in columns}
            same_sweeps = bool(np.array_equal(col["sweeps"], run.col(k, "sweeps")))
            same_t5 = bool(np.array_equal(col["T5"], run.col(k, "T5")))
            t5 = float(np.sum(col["T5"]))
            diffusion = float(np.sum(col["T5_diffusion"]))
            sink = float(np.sum(col["T5_sink"]))
            rows.append(
                [
                    label(run),
                    SIZES_UM[k],
                    "yes" if same_sweeps and same_t5 else "NO",
                    e(t5),
                    e(diffusion),
                    e(sink),
                    pct(sink / t5),
                    f"{(diffusion + sink - t5) / t5:.1e}",
                    e(float(np.sum(col["sink_at_cstar"]))),
                    f"{int(np.count_nonzero(col['T5_sink'] > 0.0))}",
                    f"{int(np.count_nonzero(col['T5_diffusion'] > 0.0))} of {len(col['T5'])}",
                ]
            )
    return [
        "### K1. T5 split into the leftover of the diffusion part and of the sink part of the first residual (split77.py, a diagnostic)",
        "",
        *table(
            [
                "Run",
                "Class (um)",
                "Sweeps and T5 equal to probe77's at every step",
                "T5 over the run",
                "Diffusion part",
                "Sink part",
                "Sink part over T5",
                "(diffusion + sink - T5) / T5",
                "Sink at C* over the run (dt sum D_P C*_P)",
                "Steps with a positive sink part",
                "Steps with a positive diffusion part",
            ],
            rows,
        ),
    ]


def predictions_table(runs: dict[str, Run]) -> list[str]:
    """P1: the predictions against the measurements."""
    rows = []
    # (a)
    worst = []
    for run in runs.values():
        gaps = closure_rows(run)
        worst.append((max(gaps) / largest_residual(run), run.name))
    a_ok = all(x <= 1e-6 for x, _ in worst)
    top = max(worst)
    rows.append(
        [
            "(a) closure",
            "every run and class, every stored frame: within 1e-6 of the largest residual",
            f"largest {top[0]:.2e} of it ({top[1]}); table A1",
            "holds" if a_ok else "FAILS",
        ]
    )
    # (b)
    failures = []
    worst_t5, worst_other = 0.0, 0.0
    for name in AT_FILE_TOL:
        run = runs.get(name)
        if run is None:
            failures.append(f"{name} not run")
            continue
        for k in run.series:
            for i in (len(run.frames.steps) - 1, worst_frame(run, k)):
                row = at_frame(run, k, i)
                r = row["R"]
                d5 = abs(row["T5"] - r) / abs(r)
                do = abs(sum(row[n] for n in OTHERS)) / abs(r)
                worst_t5, worst_other = max(worst_t5, d5), max(worst_other, do)
                if d5 > 0.01 or do >= 0.01:
                    failures.append(f"{name} class {SIZES_UM[k]} at {row['t']:.2f} s")
    rows.append(
        [
            "(b) T5 carries it",
            "R1 to R4, R9, R10, every class, end and worst frame: T5 within 1% of R; the rest under 1%",
            f"largest abs(T5 - R) over abs(R) {worst_t5:.2e}; largest abs(rest) over abs(R) {worst_other:.2e}; tables E1, W1",
            "holds" if not failures else "FAILS: " + ", ".join(failures),
        ]
    )
    # (c)
    signs = []
    c_ok = True
    for name in REFERENCE:
        run = runs.get(name)
        if run is None:
            c_ok = False
            signs.append(f"{name} not run")
            continue
        for k in run.series:
            row = at_frame(run, k, len(run.frames.steps) - 1)
            if np.sign(row["T5"]) != np.sign(row["R"]):
                c_ok = False
        row0 = at_frame(run, 0, len(run.frames.steps) - 1)
        signs.append(
            f"{SCENARIO_NAMES[run.spec.scenario]} {'+' if row0['R'] > 0 else '-'}"
        )
    expected = {
        "door_leak": 1.0,
        "equipment_dust": -1.0,
        "clean_out": -1.0,
        "filter_breach": -1.0,
    }
    for name in REFERENCE:
        run = runs.get(name)
        if run is not None:
            for k in run.series:
                row = at_frame(run, k, len(run.frames.steps) - 1)
                if np.sign(row["R"]) != expected[run.spec.scenario]:
                    c_ok = False
    r10 = runs.get("R10")
    r10_text = "R10 not run"
    if r10 is None:
        c_ok = False
    else:
        times, residual, cum = frame_series(r10, 0)
        ch_r = sign_changes(times, residual)
        ch_5 = sign_changes(times, cum)
        within = all(
            any(lo <= a and b <= hi for a, b in ch_5) for lo, hi in SIGN_WINDOWS
        ) and len(ch_5) == len(SIGN_WINDOWS)
        same = ch_r == ch_5
        c_ok = c_ok and within and same
        r10_text = (
            f"R10 0.1 um: residual changes sign {fmt_changes(ch_r)} s, cumulative T5 "
            f"{fmt_changes(ch_5)} s"
        )
    rows.append(
        [
            "(c) both signs, one mechanism",
            "T5's sign is R's at the end of R1 to R4 (door +, others -); R10 0.1 um: the same frame intervals, 10.5 to 10.9 and 25.0 to 25.4 s",
            "end signs (0.1 um): " + ", ".join(signs) + f"; {r10_text}; table S1",
            "holds" if c_ok else "FAILS",
        ]
    )
    # (d)
    d_rows = dose_rows(runs)
    if d_rows:
        factors = [per_decade(row["R"]) for row in d_rows]
        d_ok = all(decade_ok(f1, f2) for f1, f2 in factors)
        low = min(min(f) for f in factors)
        high = max(max(f) for f in factors)
        failing = sorted(
            {
                SCENARIO_NAMES[row["scenario"]]
                for row, (f1, f2) in zip(d_rows, factors, strict=True)
                if not decade_ok(f1, f2)
            }
        )
        rising = sorted(
            {
                SCENARIO_NAMES[row["scenario"]]
                for row, (f1, f2) in zip(d_rows, factors, strict=True)
                if f2 >= f1
            }
        )
        rest = max(
            abs(x) / abs(r)
            for row in d_rows
            for x, r in zip(row["other"], row["R"], strict=True)
        )
        rows.append(
            [
                "(d) dose response",
                "80x30 dust and clean-out, every class: each decade moves R by 3 to 30, less as the tolerance tightens",
                f"per decade {low:.3g} to {high:.3g}; the factor grows as the tolerance "
                f"tightens in: {', '.join(rising) if rising else 'none'}; second clause, "
                f"the part T5 does not carry: abs(R - T5) at most {rest:.1e} of R at each "
                "tolerance; table D1",
                "holds"
                if d_ok
                else "FAILS in: " + ", ".join(failing) + "; holds in the others",
            ]
        )
    return [
        "### P1. Prompt 77's predictions against the measurements",
        "",
        *table(["Prediction", "As stated (abridged)", "Measured", "Outcome"], rows),
    ]


def render(runs: dict[str, Run]) -> tuple[str, dict[str, str]]:
    """The tables file's text and each marked table."""
    ordered = [
        runs[r.name] for r in configs77.RUNS + configs77.RERUNS if r.name in runs
    ]
    tables = {
        "calibration": calibration_table(),
        "runs": runs_table(ordered),
        "identity": identity_table(ordered),
        "end": terms_table(ordered, "end"),
        "worst": terms_table(ordered, "worst"),
        "closure": closure_table(ordered),
        "predictions": predictions_table(runs),
        "signs": signs_table(runs),
        "dose": dose_table(runs),
        "grid": grid_table(runs),
        "split": split_table(runs),
    }
    head = [
        "# Prompt 77: the budget residual's terms",
        "",
        "Generated by `docs/reports/probe77/tables77.py` from `results/builder77/` (untracked, "
        "this machine) and the source run files of `configs77.FLOWS`; `--check` regenerates and "
        "compares. Do not edit by hand.",
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
    for name, body in tables.items():
        begin = f"<!-- tables77 {name} begin -->"
        end = f"<!-- tables77 {name} end -->"
        if begin not in text or end not in text:
            continue
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{body}\n{end}{tail}"
    return text


def report_copies(text: str) -> dict[str, str]:
    """The report's copies of the tables, keyed by marker name."""
    copies = {}
    for name in MARKERS:
        begin = f"<!-- tables77 {name} begin -->"
        end = f"<!-- tables77 {name} end -->"
        if begin in text and end in text:
            copies[name] = text[
                text.index(begin) + len(begin) : text.index(end)
            ].strip()
    return copies


def figure(runs: dict[str, Run], path: Path) -> Path | None:
    """R10's 0.1 um cumulative residual against time, each term overlaid."""
    run = runs.get("R10")
    if run is None:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = 0
    t = run.col(k, "t")
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    top.plot(t, run.col(k, "R"), color="black", lw=2.5, label="budget residual R")
    top.plot(
        t,
        run.cumulative(k, "T5"),
        color="tab:orange",
        lw=1.2,
        ls="--",
        label="T5, implicit leftover",
    )
    for name, colour in zip(
        OTHERS,
        ("tab:blue", "tab:green", "tab:purple", "tab:red", "tab:brown"),
        strict=True,
    ):
        top.plot(t, run.cumulative(k, name), color=colour, lw=1.0, label=name)
    top.axhline(0.0, color="grey", lw=0.5)
    top.set_ylabel("particles per m depth, summed from t = 0")
    top.set_title("R10, dust release on 320x120, 0.1 um: the residual and its terms")
    top.legend(loc="lower left", fontsize=8, ncol=2)
    for name, colour in zip(
        (*OTHERS, "gap"),
        ("tab:blue", "tab:green", "tab:purple", "tab:red", "tab:brown", "grey"),
        strict=True,
    ):
        bottom.plot(t, run.cumulative(k, name), color=colour, lw=1.0, label=name)
    bottom.axhline(0.0, color="grey", lw=0.5)
    bottom.set_ylabel("the other terms, own scale")
    bottom.set_xlabel("t (s)")
    bottom.legend(loc="upper left", fontsize=8, ncol=3)
    for ax in (top, bottom):
        ax.axvspan(10.0, 40.0, color="tab:orange", alpha=0.06)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> int:
    """Write the tables (and figure), or with --check compare them with the committed files."""
    parser = argparse.ArgumentParser(description="prompt 77's tables")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--splice", action="store_true")
    args = parser.parse_args()
    runs = {}
    for spec in configs77.RUNS + configs77.RERUNS:
        loaded = load(spec.name)
        if loaded is not None:
            runs[spec.name] = loaded
    text, tables = render(runs)
    if args.check:
        failures = []
        if not TABLES.exists() or TABLES.read_text(encoding="utf-8") != text:
            failures.append(f"{TABLES} differs from the records")
        if REPORT.exists():
            copies = report_copies(REPORT.read_text(encoding="utf-8"))
            for name in MARKERS:
                if name not in copies:
                    failures.append(f"{REPORT}: no copy of table '{name}'")
                elif copies[name] != tables[name]:
                    failures.append(
                        f"{REPORT}: the copy of table '{name}' differs from the records"
                    )
        for failure in failures:
            print(failure)
        print("tables77: " + ("differs" if failures else "matches"))
        return 1 if failures else 0
    TABLES.write_text(text, encoding="utf-8")
    print(f"wrote {TABLES}")
    if args.splice and REPORT.exists():
        REPORT.write_text(
            splice(REPORT.read_text(encoding="utf-8"), tables), encoding="utf-8"
        )
        print(f"spliced {REPORT}")
    if args.figures:
        written = figure(runs, FIGURE)
        if written is not None:
            print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
