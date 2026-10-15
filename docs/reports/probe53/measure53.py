"""Item 6 of prompt 53: one correction on each captured system, Jacobi-PCG against MG-PCG.

Usage:
    python measure53.py [--repeats N] [--name NAME]

Loads the ten measurement systems of capture53.py (five cases, early and
late) and solves each to its case's ``pressure_rtol``, with the floor and cap
the corrector uses, under ``jacobi`` and ``multigrid`` in this one process,
the two interleaved, N times each (default 5). Per system and preconditioner
it keeps the iteration count, whether the cap was reached, the true relative
residual at exit, and the median setup, solve and total seconds over the N
repeats; for multigrid, each level's shape, cells with an equation,
eigenvalue estimate and weight. It also records how far the two solutions
lie apart, the largest difference of p' over the largest |p'| of the Jacobi
solution, read on the range part (the mean over the cells with an equation
removed on the closed cavity). Writes results/builder53/NAME.json of the
main checkout (default measure53; a second pass, measure53_pass2, sizes how
far this machine's timings drift between sessions). One BLAS thread; run it
alone on the machine.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common53 import (  # noqa: E402 -- follows sys.path.insert
    hierarchy_summary,
    load_system,
    solve,
)

from src import multigrid  # noqa: E402 -- follows sys.path.insert

MAIN = Path(
    subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
).parent
SYSTEMS = MAIN / "results" / "builder53" / "systems"
CASES = ("val001", "cavity", "product160", "product320", "annex", "step")
PRECONDITIONERS = ("jacobi", "multigrid")


def range_part(x: np.ndarray, active: np.ndarray, closed: bool) -> np.ndarray:
    """x on the cells with an equation, its mean removed there on a closed domain."""
    out = np.where(active, x, 0.0)
    if closed:
        out[active] -= out[active].mean()
    return out


def main() -> None:
    """Every system under both preconditioners; the record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--name", default="measure53")
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rows = []
    for case in CASES:
        for tag in ("early", "late"):
            system = load_system(SYSTEMS / f"{case}_{tag}.npz")
            runs: dict[str, list[dict]] = {name: [] for name in PRECONDITIONERS}
            for _ in range(args.repeats):
                for name in PRECONDITIONERS:
                    runs[name].append(solve(system, name))
            row = {
                "case": case,
                "tag": tag,
                "outer": system.meta["outer"],
                "captured_as": system.meta["tag"],
                "grid": system.meta["grid"],
                "closed": system.needs_pin,
                "cells_with_equation": int(system.active.sum()),
                "rtol": system.rtol,
                "floor": system.floor,
                "f_norm": float(np.linalg.norm(system.f)),
                "floor_over_rtol_f": system.floor
                / (system.rtol * float(np.linalg.norm(system.f))),
            }
            for name in PRECONDITIONERS:
                first = runs[name][0]
                assert all(r["iterations"] == first["iterations"] for r in runs[name])
                row[name] = {
                    "iterations": first["iterations"],
                    "reached_cap": first["reached_cap"],
                    "relative_residual": first["relative_residual"],
                    "setup_seconds": statistics.median(
                        r["setup_seconds"] for r in runs[name]
                    ),
                    "solve_seconds": statistics.median(
                        r["solve_seconds"] for r in runs[name]
                    ),
                    "total_seconds": statistics.median(
                        r["total_seconds"] for r in runs[name]
                    ),
                }
            jacobi_x = range_part(
                runs["jacobi"][0]["x"], system.active, system.needs_pin
            )
            mg_x = range_part(
                runs["multigrid"][0]["x"], system.active, system.needs_pin
            )
            row["solution_difference"] = float(
                np.abs(mg_x - jacobi_x).max() / np.abs(jacobi_x).max()
            )
            row["levels"] = hierarchy_summary(system)
            row["setup_share"] = (
                row["multigrid"]["setup_seconds"] / row["multigrid"]["total_seconds"]
            )
            row["speedup"] = (
                row["jacobi"]["total_seconds"] / row["multigrid"]["total_seconds"]
            )
            rows.append(row)
            print(
                f"{case:10s} {tag:5s} outer {row['outer']:5d}  jacobi {row['jacobi']['iterations']:5d} its "
                f"{1e3 * row['jacobi']['total_seconds']:8.1f} ms  multigrid "
                f"{row['multigrid']['iterations']:3d} its setup {1e3 * row['multigrid']['setup_seconds']:6.1f} "
                f"solve {1e3 * row['multigrid']['solve_seconds']:7.1f} ms  x{row['speedup']:5.1f}  "
                f"diff {row['solution_difference']:.1e}",
                flush=True,
            )
    record = {
        "commit": commit,
        "measured": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repeats": args.repeats,
        "machine": platform.processor(),
        "smoothing_sweeps": multigrid.SMOOTHING_SWEEPS,
        "power_iterations": multigrid.POWER_ITERATIONS,
        "smoother_factor": multigrid.SMOOTHER_FACTOR,
        "coarsest_cells": multigrid.COARSEST_CELLS,
        "rows": rows,
    }
    out = MAIN / "results" / "builder53" / f"{args.name}.json"
    out.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
