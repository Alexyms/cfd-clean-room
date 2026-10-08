"""Time ``np.vdot`` against the BLAS thread count, to find where threads pay.

The pressure solve's conjugate gradient loop makes three ``vdot`` calls per
iteration over the whole grid. OpenBLAS splits each call across its thread
pool above a vector length of about 10,000, and the coordination costs far
more than the arithmetic at the lengths the product mesh produces. This
script measures that cost per length under three conditions, each in its own
process so that one condition cannot warm or disturb another:

``default``
    Whatever thread count the process starts with.
``env1``
    ``OPENBLAS_NUM_THREADS=1`` in the environment before NumPy loads.
``limit``
    The default process with a ``threadpoolctl`` limit of one BLAS thread,
    the mechanism ``src.pressure`` uses. This is the control that the limit
    does what the environment variable does.

The whole sequence runs twice, and the spread between the two runs is printed
beside each figure. Run it with nothing else using the machine. The results
and the setting built on them are recorded in ``docs/reports/blas_threads.md``.

Usage
-----
``python scripts/blas_thread_probe.py`` runs both runs and prints the table.
``python scripts/blas_thread_probe.py --worker CONDITION`` times one condition
and prints a JSON line; the driver calls this form.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

LENGTHS: tuple[int, ...] = (
    3_200,
    6_400,
    9_600,
    12_800,
    15_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
)
CONDITIONS: tuple[str, ...] = ("default", "env1", "limit")

# A few seconds per length keeps the timer error far below the differences
# the report draws conclusions from, and the whole table inside minutes.
TARGET_SECONDS = 3.0
PILOT_CALLS = 50
RUNS = 2


def time_vdot(n: int, target_seconds: float = TARGET_SECONDS) -> float:
    """Time ``np.vdot`` on two vectors of ``n`` float64 elements.

    Parameters
    ----------
    n : int
        Vector length.
    target_seconds : float
        Approximate total time to spend, used to scale the repeat count.

    Returns
    -------
    float
        Mean time per call in seconds.
    """
    a = np.random.default_rng(1).random(n)
    b = a.copy()
    t0 = time.perf_counter()
    for _ in range(PILOT_CALLS):
        np.vdot(a, b)
    pilot = (time.perf_counter() - t0) / PILOT_CALLS
    repeats = max(PILOT_CALLS, int(target_seconds / pilot))
    t0 = time.perf_counter()
    for _ in range(repeats):
        np.vdot(a, b)
    return (time.perf_counter() - t0) / repeats


def run_worker(condition: str, lengths: tuple[int, ...] = LENGTHS) -> dict[str, float]:
    """Time every length under one condition, in this process.

    Parameters
    ----------
    condition : str
        One of ``CONDITIONS``. ``env1`` expects the caller to have set
        ``OPENBLAS_NUM_THREADS`` before this process started.
    lengths : tuple[int, ...]
        Vector lengths to time.

    Returns
    -------
    dict[str, float]
        Microseconds per call, keyed by the vector length as a string.
    """
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}; expected one of {CONDITIONS}"
        )
    if condition == "limit":
        with threadpool_limits(limits=1, user_api="blas"):
            return {str(n): 1e6 * time_vdot(n) for n in lengths}
    return {str(n): 1e6 * time_vdot(n) for n in lengths}


def spawn_worker(
    condition: str, lengths: tuple[int, ...] = LENGTHS
) -> dict[str, float]:
    """Run ``run_worker`` for one condition in a fresh process.

    Parameters
    ----------
    condition : str
        One of ``CONDITIONS``.
    lengths : tuple[int, ...]
        Vector lengths to time.

    Returns
    -------
    dict[str, float]
        The worker's microseconds per call, keyed by length.
    """
    env = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.pop(name, None)
    if condition == "env1":
        env["OPENBLAS_NUM_THREADS"] = "1"
    out = subprocess.run(
        [
            sys.executable,
            __file__,
            "--worker",
            condition,
            "--lengths",
            *map(str, lengths),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def format_table(
    runs: list[dict[str, dict[str, float]]], lengths: tuple[int, ...] = LENGTHS
) -> str:
    """Format the per-length figures with the spread between runs.

    Parameters
    ----------
    runs : list[dict[str, dict[str, float]]]
        One entry per run, each mapping condition to the worker's figures.
    lengths : tuple[int, ...]
        Vector lengths, the rows of the table.

    Returns
    -------
    str
        A Markdown table: mean microseconds per call for each condition, the
        spread between runs (max minus min, as a percentage of the mean) in
        brackets, and the ratio of default threads to one thread.
    """
    lines = [
        "| elements | default (us) | env1 (us) | limit (us) | default / env1 | limit / env1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n in lengths:
        cells: dict[str, float] = {}
        text: list[str] = []
        for condition in CONDITIONS:
            values = [run[condition][str(n)] for run in runs]
            mean = sum(values) / len(values)
            spread = 100.0 * (max(values) - min(values)) / mean
            cells[condition] = mean
            text.append(f"{mean:.2f} ({spread:.0f}%)")
        ratios = (cells["default"] / cells["env1"], cells["limit"] / cells["env1"])
        lines.append(
            f"| {n:,} | " + " | ".join(text) + f" | {ratios[0]:.2f} | {ratios[1]:.2f} |"
        )
    return "\n".join(lines)


def describe_machine() -> str:
    """Describe the processor and the BLAS NumPy loaded.

    Returns
    -------
    str
        Processor string, logical core count and the ``threadpool_info`` of
        the BLAS libraries in this process. Cache sizes are not available
        portably and are recorded by hand in the report.
    """
    return (
        f"processor: {platform.processor()}\n"
        f"logical cores: {os.cpu_count()}\n"
        f"numpy: {np.__version__}\n"
        f"threadpool_info: {json.dumps(threadpool_info())}"
    )


def main() -> None:
    """Run the probe: one worker (``--worker``) or the full two-run table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--worker", choices=CONDITIONS, help="time one condition, print JSON"
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=list(LENGTHS),
        help="vector lengths to time",
    )
    args = parser.parse_args()
    lengths = tuple(args.lengths)
    if args.worker:
        print(json.dumps(run_worker(args.worker, lengths)))
        return
    print(describe_machine())
    runs: list[dict[str, dict[str, float]]] = []
    for run in range(RUNS):
        runs.append(
            {condition: spawn_worker(condition, lengths) for condition in CONDITIONS}
        )
        print(f"run {run + 1} done", file=sys.stderr, flush=True)
    print(json.dumps(runs))
    print(format_table(runs, lengths))


if __name__ == "__main__":
    main()
