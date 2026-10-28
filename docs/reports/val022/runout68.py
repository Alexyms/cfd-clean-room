"""Prompt 68c: the Re 40 run out to t = 600 (Alex, 2026-10-28).

Prompt 68b's floor probe resumed the decay at a step tolerance of 1e-6 U but at
0.3 of the earlier rate, so it could not say whether the lift reaches the quiet
threshold. Alex kept the row's measure unchanged and asked for a march long
enough to judge it. This script continues the probe's march to T_END at the
probe's settings (floor68.py: step tolerance 1e-6 U, cap 500, the row's dt)
from the row's final checkpoint (step 4000, t = 200, both time levels).

The probe ended at t = 240 without a checkpoint, so the march from 200 to 240 is
taken again. It is deterministic (one BLAS thread, the same commit of src), so
the first PROBE_STEPS steps are compared with floor68.json to the bit before
the record is trusted; the comparison is in the record. Both time levels are
kept in a checkpoint file (``runout68.checkpoint.h5``) every CHECKPOINT_STEPS
steps and at the end, so an interrupted run resumes with ``--resume``.

The readings, defined before the run (prompt 68c, prediction (g)):

- Peaks and the envelope's rate: floor68.peaks (the largest |C_L| between
  successive sign changes) and floor68.rate (the slope of ln(peak) in time).
  The run-out's rate is read over its peaks from t = RATE_FROM (the probe's
  settle of ten units past t = 240) that come before the first peak below
  QUIET_LIFT, or to the end when none does.
- Rate of (g): within RATE_TOLERANCE of the probe's rate, read from
  floor68.json (-0.0154 per unit).
- Quiet time of (g): the time of the first peak below QUIET_LIFT, "near
  t = 500" read as inside QUIET_TIME_WINDOW.
- The measure of (g): ``validation.cylinder.decays``, unchanged, on the whole
  series from t = 0: the row's 4000 steps at 1e-4 U, then this run's steps at
  1e-6 U. The row's own measure at t = 200 is kept beside it (its FAIL).
- The budget: BUDGET seconds. After BUDGET_UNITS time units past t = 240 the
  wall time of those steps gives the estimate for the rest; if the total is
  over the budget the march stops there and the record says so.

    python runout68.py [--out DIR] [--checkpoint PATH] [--resume] [--wall 7200]

Writes DIR/runout68.json: every step's t, outer count, C_L and C_D, the peaks,
the rates, the readings of (g), the measure's verdict on the whole series and
on the row alone, the wall time and whether the budget stopped it. One BLAS
thread.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from floor68 import peaks, rate  # noqa: E402

from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.forces import ForceGauge  # noqa: E402
from src.io_manager import RunFile  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.solver_staggered import FlowState, StaggeredSolver, TransientStep  # noqa: E402
from validation import cylinder  # noqa: E402
from validation.cases import load_preset  # noqa: E402

OUT = ROOT / "results" / "builder68"
CHECKPOINT = (
    ROOT
    / "results"
    / "builder68"
    / "R"
    / "results"
    / "val022"
    / "val022_re40_78fae737.h5"
)
ROW_RECORD = HERE / "val022_re40.json"
PROBE_RECORD = HERE / "floor68.json"
STEP_TOL = 1e-6
STEP_CAP = 500
T_END = 600.0
PROBE_STEPS = 800
PROBE_END = 240.0
CHECKPOINT_STEPS = 500
BUDGET = 7200.0
BUDGET_UNITS = 20.0
RATE_FROM = 250.0
RATE_TOLERANCE = 0.3
QUIET_LIFT = cylinder.QUIET_LIFT
QUIET_TIME_WINDOW = (450.0, 550.0)
WINDOW = 20.0


class _BudgetError(Exception):
    """The estimate is over the budget: the march stops and what it took is kept."""


def commit() -> str:
    """The short commit of the tree the run-out runs in."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.strip()


def main() -> None:
    """March the run-out and write its record."""
    parser = argparse.ArgumentParser(description="Prompt 68c, the Re 40 run-out")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wall", type=float, default=BUDGET)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mine = RunFile(args.out / "runout68.h5")
    source = mine if args.resume else RunFile(args.checkpoint)
    t0_checkpoint, resume = source.read_transient_checkpoint()
    raw = copy.deepcopy(load_preset("val022_re40").merged_mapping())
    raw["transient"]["step_tol"] = STEP_TOL
    raw["transient"]["step_cap"] = STEP_CAP
    dt = raw["solver"]["dt"]
    raw["solver"]["t_end"] = round(T_END / dt) * dt
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    spec = config.transient
    gauge = ForceGauge(
        mesh, config, spec.forces.obstacle, spec.forces.reference_velocity
    )
    last_step = round(T_END / dt)
    rows: list[tuple[int, float, int, float, float]] = []
    if args.resume:
        kept = np.load(args.out / "runout68.rows.npy")
        rows = [
            (int(r[0]), float(r[1]), int(r[2]), float(r[3]), float(r[4]))
            for r in kept
            if r[0] <= resume.step
        ]
    # The checkpoint's previous level is the state before the step it writes,
    # as the tier-2 route holds it (time_integration.run).
    held: dict[str, FlowState | None] = {"previous": resume.state}
    clock = perf_counter()
    marks: dict[str, float] = {}

    def on_step(record: TransientStep, state: FlowState) -> None:
        c_d, c_l = gauge.coefficients(gauge.forces(state.u, state.v, state.p))
        rows.append((record.step, record.t, record.outer, c_d, c_l))
        if record.step % CHECKPOINT_STEPS == 0 or record.step == last_step:
            mine.write_transient_checkpoint(
                record.step, record.t, state, held["previous"]
            )
            np.save(args.out / "runout68.rows.npy", np.array(rows))
        held["previous"] = state
        if record.step % 100 == 0:
            print(
                f"step {record.step} t {record.t:g}: {record.outer} outer, "
                f"C_L {c_l:+.3e}, C_D {c_d:.6f}, wall {perf_counter() - clock:.0f}",
                flush=True,
            )
        if abs(record.t - PROBE_END) < 1e-9 and not args.resume:
            marks["probe_end"] = perf_counter() - clock
        if "probe_end" in marks and abs(record.t - (PROBE_END + BUDGET_UNITS)) < 1e-9:
            marks["estimate_at"] = perf_counter() - clock
            spent = marks["estimate_at"] - marks["probe_end"]
            remaining = (T_END - record.t) / dt
            per_step = spent / round(BUDGET_UNITS / dt)
            marks["estimate_total"] = marks["estimate_at"] + per_step * remaining
            print(
                f"estimate from {BUDGET_UNITS:g} units: {per_step:.3f} s/step, "
                f"total {marks['estimate_total']:.0f} s of {args.wall:.0f}",
                flush=True,
            )
            if marks["estimate_total"] > args.wall:
                raise _BudgetError

    started = datetime.now().isoformat(timespec="seconds")
    stopped_by_budget = False
    try:
        solver.solve_transient(
            config.dt,
            config.t_end,
            resume=resume,
            step_tol=spec.step_tol,
            step_cap=spec.step_cap,
            scheme=spec.scheme,
            on_step=on_step,
        )
    except _BudgetError:
        stopped_by_budget = True
    wall = perf_counter() - clock

    t = np.array([r[1] for r in rows])
    outer = np.array([r[2] for r in rows])
    c_d = np.array([r[3] for r in rows])
    c_l = np.array([r[4] for r in rows])
    row = json.loads(ROW_RECORD.read_text(encoding="utf-8"))
    probe = json.loads(PROBE_RECORD.read_text(encoding="utf-8"))

    same = None
    if not args.resume and len(rows) >= PROBE_STEPS:
        same = {
            "steps": PROBE_STEPS,
            "t": bool(np.array_equal(t[:PROBE_STEPS], probe["series"]["t"])),
            "outer": bool(
                np.array_equal(outer[:PROBE_STEPS], probe["series"]["outer"])
            ),
            "c_l": bool(np.array_equal(c_l[:PROBE_STEPS], probe["series"]["c_l"])),
            "c_d": bool(np.array_equal(c_d[:PROBE_STEPS], probe["series"]["c_d"])),
        }

    peak_t, peak_v = peaks(t, c_l)
    quiet = np.nonzero(peak_v < QUIET_LIFT)[0]
    first_quiet = int(quiet[0]) if quiet.size else None
    quiet_time = None if first_quiet is None else float(peak_t[first_quiet])
    usable = peak_t >= RATE_FROM
    if first_quiet is not None:
        usable &= np.arange(peak_t.size) < first_quiet
    run_rate = rate(peak_t[usable], peak_v[usable])
    reference = probe["probe_rate"]
    rate_holds = (
        run_rate is not None and abs(run_rate / reference - 1.0) <= RATE_TOLERANCE
    )
    quiet_holds = (
        quiet_time is not None
        and QUIET_TIME_WINDOW[0] <= quiet_time <= QUIET_TIME_WINDOW[1]
    )

    t_whole = np.concatenate([row["series"]["t"], t])
    c_l_whole = np.concatenate([row["series"]["c_l"], c_l])
    measure_whole = cylinder.decays(t_whole, c_l_whole)
    measure_row = cylinder.decays(
        np.array(row["series"]["t"]), np.array(row["series"]["c_l"])
    )
    record = {
        "kind": "runout",
        "commit": commit(),
        "started": started,
        "checkpoint": args.checkpoint.name,
        "from_step": resume.step,
        "from_t": t0_checkpoint,
        "resumed_run": args.resume,
        "step_tol": STEP_TOL,
        "step_cap": STEP_CAP,
        "dt": dt,
        "steps": len(rows),
        "t_end": float(t[-1]),
        "stopped_by_budget": stopped_by_budget,
        "wall_seconds": wall,
        "probe_end_seconds": marks.get("probe_end"),
        "estimate_total_seconds": marks.get("estimate_total"),
        "same_as_probe": same,
        "probe_rate": reference,
        "rate_from": RATE_FROM,
        "run_rate": run_rate,
        "rate_ratio": None if run_rate is None else run_rate / reference,
        "rate_tolerance": RATE_TOLERANCE,
        "quiet_lift": QUIET_LIFT,
        "quiet_time": quiet_time,
        "quiet_time_window": list(QUIET_TIME_WINDOW),
        "last_peak": float(peak_v[-1]) if peak_v.size else None,
        "peaks": {"t": peak_t.tolist(), "peak": peak_v.tolist()},
        "prediction_g_rate": rate_holds,
        "prediction_g_quiet_time": quiet_holds,
        "measure_whole_series": measure_whole,
        "measure_row_alone": measure_row,
        "prediction_g_measure": bool(measure_whole["passed"]),
        "prediction_g": bool(rate_holds and quiet_holds and measure_whole["passed"]),
        "series": {
            "t": t.tolist(),
            "outer": outer.tolist(),
            "c_l": c_l.tolist(),
            "c_d": c_d.tolist(),
        },
    }
    (args.out / "runout68.json").write_text(
        json.dumps(record, indent=1), encoding="utf-8"
    )
    summary = {k: v for k, v in record.items() if k not in ("series", "peaks")}
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
