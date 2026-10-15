"""Prompt 53: where one MG-PCG iteration's time goes, level by level.

Usage:
    python anatomy53.py

On the late Annex 20 and step systems of capture53.py, times (the median of
REPEATS runs of LOOPS calls each, one BLAS thread, alone): one product with
the operator on the finest level; one Jacobi-PCG iteration (the solve's
seconds over its iterations); one V-cycle; and the cycle from each level
down (``Multigrid._cycle(index, r)`` on that level's restricted residual),
whose differences are each level's own share. It also counts the operator
products one cycle forms on each level. Writes results/builder53/anatomy53.json
of the main checkout and prints a table.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common53 import load_system, solve  # noqa: E402 -- follows sys.path.insert

from src.multigrid import Multigrid, restrict  # noqa: E402 -- follows sys.path.insert
from src.pressure import apply_operator  # noqa: E402 -- follows sys.path.insert

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
NAMES = ("annex_late", "step_late")
REPEATS = 7
LOOPS = 50


def timed(call: Callable[[], object]) -> float:
    """Median seconds per call over REPEATS batches of LOOPS calls."""
    batches = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        for _ in range(LOOPS):
            call()
        batches.append((time.perf_counter() - start) / LOOPS)
    return statistics.median(batches)


def anatomy(name: str) -> dict:
    """One system's anatomy."""
    system = load_system(SYSTEMS / f"{name}.npz")
    c = system.c
    hierarchy = Multigrid(c, apply_operator)
    r = system.f.copy()
    residuals = [r]
    for _ in hierarchy.levels[1:]:
        residuals.append(restrict(residuals[-1]))
    products = [0] * len(hierarchy.levels)
    level_now = [0]

    def counted(coefficients: object, x: np.ndarray) -> np.ndarray:
        products[level_now[0]] += 1
        return apply_operator(coefficients, x)

    counter = Multigrid(c, counted)
    products[:] = [0] * len(products)
    original = counter._cycle

    def tracking(index: int, residual: np.ndarray) -> np.ndarray:
        level_now[0] = index
        out = original(index, residual)
        level_now[0] = max(index - 1, 0)
        return out

    counter._cycle = tracking  # type: ignore[method-assign]
    counter.cycle(r)
    down = [
        timed(lambda i=i: hierarchy._cycle(i, residuals[i]))
        for i in range(len(hierarchy.levels))
    ]
    own = [
        down[i] - (down[i + 1] if i + 1 < len(down) else 0.0) for i in range(len(down))
    ]
    jacobi = min(
        (solve(system, "jacobi") for _ in range(3)), key=lambda s: s["solve_seconds"]
    )
    return {
        "system": name,
        "cells": int(c.a_p.size),
        "product_seconds": timed(lambda: apply_operator(c, r)),
        "jacobi_iteration_seconds": jacobi["solve_seconds"] / jacobi["iterations"],
        "cycle_seconds": timed(lambda: hierarchy.cycle(r)),
        "levels": [
            {
                "shape": list(level.coefficients.a_p.shape),
                "own_seconds": own[i],
                "products": products[i],
            }
            for i, level in enumerate(hierarchy.levels)
        ],
    }


def main() -> None:
    """Each system's anatomy; the record."""
    records = []
    for name in NAMES:
        record = anatomy(name)
        records.append(record)
        print(
            f"{name}: product {1e3 * record['product_seconds']:.3f} ms, Jacobi-PCG iteration "
            f"{1e3 * record['jacobi_iteration_seconds']:.3f} ms, cycle "
            f"{1e3 * record['cycle_seconds']:.3f} ms",
            flush=True,
        )
        for level in record["levels"]:
            print(
                f"  {level['shape'][1]}x{level['shape'][0]}: "
                f"{1e3 * level['own_seconds']:.3f} ms, {level['products']} products",
                flush=True,
            )
    (MAIN / "results" / "builder53" / "anatomy53.json").write_text(
        json.dumps(records, indent=1)
    )


if __name__ == "__main__":
    main()
