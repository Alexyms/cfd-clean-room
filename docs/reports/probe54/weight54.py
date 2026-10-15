"""Item 6 of prompt 54: the smoother weight change alone, on prompt 53's twelve captured systems.

Usage:
    python weight54.py [--repeats N] [--name NAME]

Loads the twelve systems of prompt 53 (six cases, early and late;
``results/builder53/systems/`` of the main checkout, captured by
``probe53/capture53.py``) and solves each to its case's ``pressure_rtol``
with the floor and cap the corrector uses, twice per repeat in this one
process: once with the multigrid module of main at 14166aa (the weight
``(4/3) / lambda`` from three power iterations per level, read with ``git
show`` and loaded as a separate module) and once with the committed module
(the fixed weight 2/3). The two are interleaved, N repeats each (default
5); per system and variant the record keeps the iteration count, whether the
cap was reached, the true relative residual at exit, the median setup, solve
and total seconds, each level's weight, and for the old module each level's
eigenvalue estimate. It also records how far the two solutions lie apart
(the largest difference of p' over the largest |p'| of the old solution, on
the range part). The old module is swapped in by rebinding ``Multigrid`` in
``src.pressure``, which ``multigrid_preconditioner`` reads, so both variants
run the committed conjugate gradient loop and the committed projection.

Writes results/builder54/NAME.json of the main checkout (default weight54).
One BLAS thread; run it alone on the machine.
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
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "docs" / "reports" / "probe53"))

from common53 import (  # noqa: E402 -- follows sys.path.insert
    load_system,
    solve,
)

from src import multigrid as new_module  # noqa: E402 -- follows sys.path.insert
from src import pressure  # noqa: E402 -- follows sys.path.insert
from src.pressure import apply_operator  # noqa: E402 -- follows sys.path.insert

OLD_COMMIT = "14166aa"
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
OUT = MAIN / "results" / "builder54"
CASES = ("val001", "cavity", "product160", "product320", "annex", "step")


def load_old_module() -> types.ModuleType:
    """src/multigrid.py as main had it at OLD_COMMIT, loaded as ``multigrid_old``."""
    text = subprocess.run(
        ["git", "show", f"{OLD_COMMIT}:src/multigrid.py"],
        cwd=HERE,
        capture_output=True,
        check=True,
    ).stdout.decode("ascii")
    module = types.ModuleType("multigrid_old")
    sys.modules["multigrid_old"] = module
    exec(compile(text, "multigrid_old.py", "exec"), module.__dict__)
    return module


def range_part(x: np.ndarray, active: np.ndarray, closed: bool) -> np.ndarray:
    """x on the cells with an equation, its mean removed there on a closed domain."""
    out = np.where(active, x, 0.0)
    if closed:
        out[active] -= out[active].mean()
    return out


def main() -> None:
    """Every system under both weights; the record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--name", default="weight54")
    args = parser.parse_args()
    old_module = load_old_module()
    old_class, new_class = old_module.Multigrid, new_module.Multigrid
    assert pressure.Multigrid is new_class
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    variants = {"old": old_class, "new": new_class}
    rows = []
    for case in CASES:
        for tag in ("early", "late"):
            system = load_system(SYSTEMS / f"{case}_{tag}.npz")
            runs: dict[str, list[dict]] = {name: [] for name in variants}
            for _ in range(args.repeats):
                for name, cls in variants.items():
                    pressure.Multigrid = cls
                    runs[name].append(solve(system, "multigrid"))
            pressure.Multigrid = new_class
            row = {
                "case": case,
                "tag": tag,
                "outer": system.meta["outer"],
                "captured_as": system.meta["tag"],
                "grid": system.meta["grid"],
                "closed": system.needs_pin,
                "rtol": system.rtol,
                "floor": system.floor,
            }
            for name in variants:
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
            old_x = range_part(runs["old"][0]["x"], system.active, system.needs_pin)
            new_x = range_part(runs["new"][0]["x"], system.active, system.needs_pin)
            row["solution_difference"] = float(
                np.abs(new_x - old_x).max() / np.abs(old_x).max()
            )
            old_levels = old_class(system.c, apply_operator).levels
            new_levels = new_class(system.c, apply_operator).levels
            row["levels"] = [
                {
                    "shape": list(a.coefficients.a_p.shape),
                    "old_eigenvalue": a.eigenvalue,
                    "old_weight": a.weight,
                    "new_weight": b.weight,
                }
                for a, b in zip(old_levels, new_levels, strict=True)
            ]
            for name in variants:
                row[name]["setup_share"] = (
                    row[name]["setup_seconds"] / row[name]["total_seconds"]
                )
            row["setup_ratio"] = (
                row["new"]["setup_seconds"] / row["old"]["setup_seconds"]
            )
            row["total_ratio"] = (
                row["new"]["total_seconds"] / row["old"]["total_seconds"]
            )
            rows.append(row)
            print(
                f"{case:10s} {tag:5s} outer {row['outer']:5d}  old "
                f"{row['old']['iterations']:3d} its setup {1e3 * row['old']['setup_seconds']:6.2f} "
                f"solve {1e3 * row['old']['solve_seconds']:7.1f} ms  new "
                f"{row['new']['iterations']:3d} its setup {1e3 * row['new']['setup_seconds']:6.2f} "
                f"solve {1e3 * row['new']['solve_seconds']:7.1f} ms  diff "
                f"{row['solution_difference']:.1e}",
                flush=True,
            )
    record = {
        "commit": commit,
        "old_commit": OLD_COMMIT,
        "measured": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repeats": args.repeats,
        "machine": platform.processor(),
        "smoothing_sweeps": new_module.SMOOTHING_SWEEPS,
        "smoother_factor": new_module.SMOOTHER_FACTOR,
        "old_power_iterations": old_module.POWER_ITERATIONS,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.name}.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
