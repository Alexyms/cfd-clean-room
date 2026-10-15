"""Where the fixed-weight setup goes: a hierarchy's build, split by part, on the captured systems.

Usage:
    python setup54.py [--repeats N]

For each of the twelve captured systems of prompt 53 (``weight54.py``'s set)
it times, as the median of N repeats (default 200) in one process, the parts
of ``Multigrid.__init__`` as the committed module runs them: the dominance
check on the finest system, the coarsening of every level (``coarsen``), the
per-level smoother arrays, and the dense pseudo-inverse of the coarsest level;
then the whole constructor. Writes results/builder54/setup54.json of the main
checkout. This explains why the fixed weight cuts less of the setup than half:
the power iterations were a part of the setup, not most of it. One BLAS
thread; run alone.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "docs" / "reports" / "probe53"))

from common53 import load_system  # noqa: E402 -- follows sys.path.insert

from src import multigrid  # noqa: E402 -- follows sys.path.insert
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
OUT = MAIN / "results" / "builder54"
CASES = ("val001", "cavity", "product160", "product320", "annex", "step")


def median_seconds(call: object, repeats: int) -> float:
    """The median wall time of ``call()`` over ``repeats`` calls."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def time_parts(system: object, repeats: int) -> dict:
    """The median seconds of each part of the constructor on one system."""
    c = system.c
    levels = [(c, c.a_p > 0.0)]
    while True:
        lc, active = levels[-1]
        if lc.a_p.size <= multigrid.COARSEST_CELLS or min(lc.a_p.shape) == 1:
            break
        levels.append(multigrid.coarsen(lc, active))

    def dominance() -> None:
        couplings = (c.a_e + c.a_w) + (c.a_n + c.a_s)
        np.any(c.a_p * (1.0 + multigrid.DOMINANCE_TOLERANCE) < couplings)

    def coarsening() -> None:
        for lc, active in levels[:-1]:
            multigrid.coarsen(lc, active)

    def smoothers() -> None:
        for lc, _active in levels[:-1]:
            has = lc.a_p > 0.0
            inverse = np.where(has, 1.0 / np.where(has, lc.a_p, 1.0), 0.0)
            multigrid.SMOOTHER_FACTOR / 2.0 * inverse

    def pseudo_inverse() -> None:
        multigrid._dense_pseudo_inverse(levels[-1][0])

    def whole() -> None:
        multigrid.Multigrid(c, apply_operator)

    return {
        "levels": [list(lc.a_p.shape) for lc, _ in levels],
        "dominance": median_seconds(dominance, repeats),
        "coarsening": median_seconds(coarsening, repeats),
        "smoothers": median_seconds(smoothers, repeats),
        "pseudo_inverse": median_seconds(pseudo_inverse, repeats),
        "whole": median_seconds(whole, repeats),
    }


def main() -> None:
    """Time each part of the constructor on every system; the record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    rows = []
    for case in CASES:
        for tag in ("early", "late"):
            system = load_system(SYSTEMS / f"{case}_{tag}.npz")
            rows.append(
                {
                    "case": case,
                    "tag": tag,
                    "outer": system.meta["outer"],
                    **time_parts(system, args.repeats),
                }
            )
            r = rows[-1]
            print(
                f"{case:10s} {tag:5s} whole {1e3 * r['whole']:6.2f} ms: dominance "
                f"{1e3 * r['dominance']:5.2f} coarsen {1e3 * r['coarsening']:5.2f} "
                f"smoothers {1e3 * r['smoothers']:5.2f} pinv {1e3 * r['pseudo_inverse']:5.2f}",
                flush=True,
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "setup54.json").write_text(
        json.dumps({"repeats": args.repeats, "rows": rows})
    )


if __name__ == "__main__":
    main()
