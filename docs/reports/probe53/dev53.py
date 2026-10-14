"""Prompt 53: choosing nu and the power-iteration count, on systems outside the measurement set.

Usage:
    python dev53.py

Solves the development systems of capture53.py (the product room on 80x30
and the backward-facing step's coarse grid 410x40, at outer 50 and outer
300) with Jacobi-PCG and with MG-PCG under each V(nu, nu) for nu 1 to 3 and
each power-iteration count of 2, 3 and 6, set on src/multigrid.py's module
constants in this process. Each solve is timed three times and the fastest
kept. Writes results/builder53/dev53.json of the main checkout and prints a
table. One BLAS thread.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common53 import load_system, solve  # noqa: E402 -- follows sys.path.insert

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
NAMES = ("product80_early", "product80_late", "step410_early", "step410_late")
REPEATS = 3


def fastest(system: object, preconditioner: str) -> dict:
    """The fastest of REPEATS solves; the iteration count is the same each time."""
    runs = [solve(system, preconditioner) for _ in range(REPEATS)]
    best = min(runs, key=lambda r: r["total_seconds"])
    best.pop("x")
    return best


def main() -> None:
    """Every setting on every development system."""
    rows = []
    for name in NAMES:
        system = load_system(SYSTEMS / f"{name}.npz")
        row = {"system": name, "rtol": system.rtol, "jacobi": fastest(system, "jacobi")}
        for nu in (1, 2, 3):
            for power in (2, 3, 6):
                multigrid.SMOOTHING_SWEEPS = nu
                multigrid.POWER_ITERATIONS = power
                row[f"nu{nu}_power{power}"] = fastest(system, "multigrid")
        rows.append(row)
        print(name, f"rtol {system.rtol:g}", flush=True)
        for key, value in row.items():
            if isinstance(value, dict):
                print(
                    f"  {key:16s} its {value['iterations']:5d}  setup "
                    f"{1e3 * value['setup_seconds']:7.1f} ms  solve "
                    f"{1e3 * value['solve_seconds']:8.1f} ms  total "
                    f"{1e3 * value['total_seconds']:8.1f} ms  cap {value['reached_cap']}",
                    flush=True,
                )
    out = MAIN / "results" / "builder53" / "dev53.json"
    out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
