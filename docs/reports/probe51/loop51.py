"""The cost of one iteration of each implicit loop on the Annex 20 mesh, by part.

Usage:
    python loop51.py

On a 216 by 72 field (the size of ``val017_216x72``, 15,552 cells, no SOLID
cell) with random conductances, a sink and a step large enough that neither
loop stops, this times: a Jacobi sweep (``scalar_scheme._jacobi`` capped at
N sweeps with a zero stop, over N); a CG iteration (``_conjugate_gradient``
the same way, inside the one-thread BLAS limit, so it includes the closing
true residual of the cap, under one part in N); and the parts of a CG
iteration as the loop forms them: the product (the padded copy, the
neighbour sum, the masked zeroing), the two dot products, the largest
absolute residual, the two updates of c and r, the preconditioner and the
direction update. Each time is the least over five repeats of N = 200
iterations. The record is results/builder51/loop.json of the main checkout.
One BLAS thread; run alone.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import subprocess
import sys
import timeit
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
MAIN = Path(
    subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
).parent
OUT = MAIN / "results" / "builder51"

from src import scalar_scheme  # noqa: E402

NY, NX = 72, 216
N = 200
REPEATS = 5


def best(fn: object, number: int) -> float:
    """The least time per call over REPEATS repeats of NUMBER calls, in ms."""
    return 1e3 * min(timeit.repeat(fn, number=number, repeat=REPEATS)) / number


def main() -> None:
    """Time the loops and their parts, and write the record."""
    rng = np.random.default_rng(51)
    g_u = np.zeros((NY, NX + 1))
    g_v = np.zeros((NY + 1, NX))
    g_u[:, 1:-1] = rng.uniform(0.5, 1.5, size=(NY, NX - 1))
    g_v[1:-1, :] = rng.uniform(0.5, 1.5, size=(NY - 1, NX))
    fixed = np.zeros((NY, NX), dtype=bool)
    over_dt = np.full((NY, NX), 1e-3)
    a_p = over_dt + g_u[:, :-1] + g_u[:, 1:] + g_v[:-1, :] + g_v[1:, :] + 1e-3
    c_star = rng.uniform(0.5, 1.5, size=(NY, NX))
    b = over_dt * c_star

    def jacobi() -> None:
        scalar_scheme._jacobi(c_star, a_p, b, g_u, g_v, fixed, None, 0.0, N)

    def cg() -> None:
        with scalar_scheme._BLAS_CONTROLLER.limit(
            limits=scalar_scheme.SCALAR_BLAS_THREADS, user_api="blas"
        ):
            scalar_scheme._conjugate_gradient(
                c_star.copy(), a_p, b, g_u, g_v, fixed, 0.0, N
            )

    padded = np.pad(c_star, 1)
    p = rng.uniform(size=(NY, NX))
    q = rng.uniform(size=(NY, NX))
    r = rng.uniform(size=(NY, NX))
    c = rng.uniform(size=(NY, NX))
    inverse = 1.0 / a_p

    def product() -> None:
        padded[1:-1, 1:-1] = p
        out = a_p * p - scalar_scheme._neighbour_sum(padded, g_u, g_v)
        out[fixed] = 0.0

    def mask_only() -> None:
        q[fixed] = 0.0

    def dots() -> None:
        float(np.vdot(p, q))
        float(np.vdot(r, p))

    def max_abs() -> None:
        float(np.abs(r).max())

    def updates() -> None:
        c.__iadd__(1e-9 * p)
        r.__isub__(1e-9 * q)

    def direction() -> None:
        z = inverse * r
        p.__imul__(1.0)
        p.__iadd__(z)

    record = {
        "cells": NY * NX,
        "iterations_per_repeat": N,
        "repeats": REPEATS,
        "jacobi_sweep_ms": best(jacobi, 1) / N,
        "cg_iteration_ms": best(cg, 1) / N,
        "parts_ms": {
            "product": best(product, N),
            "of_which_masked_zeroing": best(mask_only, N),
            "two_dot_products": best(dots, N),
            "largest_absolute_residual": best(max_abs, N),
            "updates_of_c_and_r": best(updates, N),
            "preconditioner_and_direction": best(direction, N),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "loop.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
