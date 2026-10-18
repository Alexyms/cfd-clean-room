"""Prompt 67: the observed order in time of solve_transient (ADR-014 step 6a).

The case: VAL-002's lid-driven cavity at Re 100 (lid 1 m/s, nu 0.01 m^2/s,
unit square) on a fixed uniform 20x20 grid, spun up from rest to T = 1 s. No
exact unsteady solution fits this solver's boundaries (the Taylor-Green vortex
needs periodic edges, the impulsively started plate a periodic channel), so the
order is temporal self-convergence: the same transient at dt 0.1, 0.05,
0.025 and 0.0125 s, the order read from the ratio of successive differences of
the stated field, every u and v face at T, in the largest absolute value. The
grid is fixed, so the differences hold the time error only.

Each scheme ("bdf2", with its implicit Euler first step, and
"implicit_euler", the control) runs the ladder at the per-step criterion
STEP_TOL of the lid speed and the pressure solve at its floor of 1e-10. The
iteration error is measured, not assumed: each scheme's finest and coarsest
runs are repeated at REFERENCE_TOL, and the largest face difference between
the two is the iteration error the ladder's states carry.

    python order67.py [--grid 20] [--t-end 1.0]

Writes results/builder67/order67.json: per scheme and step, the face
difference to the next step, the observed orders, the outer iterations per
step (mean and largest), the wall time, and the iteration errors. One BLAS
thread.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.solver_staggered import FlowState, StaggeredSolver  # noqa: E402
from src.staggered import allocate_fields  # noqa: E402
from validation.cases import case_path  # noqa: E402

OUT = ROOT / "results" / "builder67"
STEPS = (0.1, 0.05, 0.025, 0.0125)
SCHEMES = ("bdf2", "implicit_euler")
STEP_TOL = 1e-10
REFERENCE_TOL = 1e-12
PRESSURE_RTOL = 1e-10
STEP_CAP = 1000


def commit() -> str:
    """The short commit of the tree the probe runs in."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.strip()


def cavity_raw(grid: int) -> dict:
    """VAL-002's cavity on grid x grid, the pressure solve at its floor."""
    raw = yaml.safe_load(case_path("cavity").read_text(encoding="utf-8"))
    raw["domain"]["nx"] = raw["domain"]["ny"] = grid
    raw["solver"]["pressure_rtol"] = PRESSURE_RTOL
    raw["solver"]["stopping_rule"] = "velocity_step"
    return raw


def spin_up(
    raw: dict, dt: float, t_end: float, scheme: str, step_tol: float
) -> tuple[FlowState, dict]:
    """The cavity from rest to t_end; the end state and the run's counts."""
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    u, v, p = allocate_fields(mesh)
    t0 = perf_counter()
    state = solver.solve_transient(
        dt,
        t_end,
        initial=FlowState(u=u, v=v, p=p),
        step_tol=step_tol,
        step_cap=STEP_CAP,
        scheme=scheme,
    )
    outer = np.array([r.outer for r in solver.transient_steps])
    return state, {
        "dt": dt,
        "scheme": scheme,
        "step_tol": step_tol,
        "steps": len(outer),
        "outer_mean": float(outer.mean()),
        "outer_max": int(outer.max()),
        "worst_imbalance_max": max(r.worst_imbalance for r in solver.transient_steps),
        "wall_seconds": perf_counter() - t0,
    }


def difference(a: FlowState, b: FlowState) -> float:
    """The largest absolute difference over every u and v face, m/s."""
    return float(max(np.max(np.abs(a.u - b.u)), np.max(np.abs(a.v - b.v))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 67, the order in time")
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--t-end", type=float, default=1.0)
    args = parser.parse_args()
    raw = cavity_raw(args.grid)
    record: dict = {
        "kind": "order",
        "commit": commit(),
        "started": datetime.now().isoformat(timespec="seconds"),
        "grid": args.grid,
        "t_end": args.t_end,
        "steps": list(STEPS),
        "step_tol": STEP_TOL,
        "reference_tol": REFERENCE_TOL,
        "pressure_rtol": PRESSURE_RTOL,
        "field": "every u and v face at t_end, largest absolute difference",
        "schemes": {},
    }
    for scheme in SCHEMES:
        states, runs = [], []
        for dt in STEPS:
            state, run = spin_up(raw, dt, args.t_end, scheme, STEP_TOL)
            states.append(state)
            runs.append(run)
            print(
                f"{scheme} dt {dt}: {run['steps']} steps, outer mean "
                f"{run['outer_mean']:.1f} max {run['outer_max']}, "
                f"{run['wall_seconds']:.1f} s",
                flush=True,
            )
        diffs = [difference(a, b) for a, b in pairwise(states)]
        orders = [float(np.log2(d0 / d1)) for d0, d1 in pairwise(diffs)]
        iteration = {}
        for index in (0, len(STEPS) - 1):
            reference, run = spin_up(
                raw, STEPS[index], args.t_end, scheme, REFERENCE_TOL
            )
            iteration[f"{STEPS[index]:g}"] = {
                "error": difference(states[index], reference),
                "reference_outer_mean": run["outer_mean"],
                "reference_wall_seconds": run["wall_seconds"],
            }
        record["schemes"][scheme] = {
            "runs": runs,
            "differences": diffs,
            "orders": orders,
            "iteration_error": iteration,
        }
        print(f"{scheme}: differences {diffs}, orders {orders}", flush=True)
        print(f"{scheme}: iteration error {iteration}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "order67.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
