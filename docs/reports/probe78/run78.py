"""Run prompt 78's jobs, at most five at a time, one BLAS thread each.

    python docs/reports/probe78/run78.py              # every job not yet done
    python docs/reports/probe78/run78.py --only R1:probe val017_216x72
    python docs/reports/probe78/run78.py --status

Two kinds of job. A scenario job ``RUN:SIDE`` is ``probe78.py RUN SIDE``
(configs78's five runs on main, the branch, and the branch recorded). A
harness job is ``scripts/benchmark.py --cases CASE --repeats 1`` at the
working tree's commit: the rows whose transport output this change moves
(VAL-024's two, whose scalar march runs the transport solver) append to
``benchmarks/results.jsonl``, the rows of record; every other row, a flow row
retaken only to see whether it reproduces, goes to
``results/builder78/harness_flow.jsonl``. The longest jobs start first. Each
job's start, end and exit go to ``results/builder78/runs78.json``; a job with
an end there is done and skipped. Logs go to ``results/builder78/logs/``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import configs78  # noqa: E402 -- follows sys.path.insert

RESULTS = configs78.RESULTS
LOGS = RESULTS / "logs"
RECORD = RESULTS / "runs78.json"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
MAX_CONCURRENT = 5
POLL_SECONDS = 10
ENV = os.environ | {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
# The harness rows retaken, with the wall seconds of their last row of record
# (the order they start in). VAL-024's rows carry the transport march.
TRANSPORT_ROWS: tuple[str, ...] = ("val024_560x100", "val024_280x50")
FLOW_ROWS: dict[str, float] = {
    "val020_272x90": 8328,
    "val019_656x60": 2852,
    "val018_320x120": 2144,
    "val017_216x72": 942,
    "val002_80x80": 281,
    "val002_40x40": 41,
    "val001_80x40": 23,
    "val001_80x40_stretched": 18,
    "val002_20x20": 7,
}
FLOW_RESULTS = RESULTS / "harness_flow.jsonl"
TRANSPORT_RESULTS = ROOT / "benchmarks" / "results.jsonl"


def jobs() -> list[str]:
    """Every job in launch order."""
    long_flow = [c for c, s in FLOW_ROWS.items() if s >= 900]
    short_flow = [c for c, s in FLOW_ROWS.items() if s < 900]
    # Each run's main and branch jobs side by side, so their wall times are
    # taken under the same load.
    scenario = [
        f"{run.name}:{side}" for run in configs78.RUNS for side in configs78.SIDES
    ]
    return [*long_flow, *TRANSPORT_ROWS, *scenario, *short_flow]


def command(job: str) -> list[str]:
    """The job's command line."""
    if ":" in job:
        name, side = job.split(":")
        return [
            str(PYTHON),
            str(HERE / "probe78.py"),
            name,
            side,
            "--concurrent",
            str(MAX_CONCURRENT),
        ]
    results = TRANSPORT_RESULTS if job in TRANSPORT_ROWS else FLOW_RESULTS
    return [
        str(PYTHON),
        str(ROOT / "scripts" / "benchmark.py"),
        "--cases",
        job,
        "--repeats",
        "1",
        "--concurrent",
        str(MAX_CONCURRENT),
        "--results",
        str(results),
    ]


def load() -> dict:
    """The record, job to its start, end and exit."""
    if RECORD.exists():
        return json.loads(RECORD.read_text(encoding="utf-8"))
    return {}


def save(record: dict, mine: set[str]) -> None:
    """Write this queue's entries over the file's, keeping every other queue's."""
    current = load()
    current.update({job: record[job] for job in mine})
    RECORD.write_text(json.dumps(current, indent=2), encoding="utf-8")


def status() -> None:
    """Print each job's state."""
    record = load()
    now = time.time()
    for job in jobs():
        entry = record.get(job)
        if entry is None:
            print(f"{job}: queued")
        elif entry.get("end") is None:
            print(f"{job}: running for {now - entry['start_epoch']:.0f} s")
        else:
            seconds = entry["end_epoch"] - entry["start_epoch"]
            print(f"{job}: exit {entry['exit']} after {seconds:.0f} s")


def main() -> int:
    """Run the queue to its end."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--slots",
        type=int,
        default=MAX_CONCURRENT,
        help="processes this queue runs at once; with another queue running, "
        "their sum is kept at MAX_CONCURRENT",
    )
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    LOGS.mkdir(parents=True, exist_ok=True)
    record = load()
    # A job with a start in the record is done or running in another queue.
    queue = [j for j in (args.only or jobs()) if args.only or j not in record]
    mine: set[str] = set()
    running: dict[str, subprocess.Popen] = {}
    handles = {}
    while queue or running:
        while queue and len(running) < args.slots:
            job = queue.pop(0)
            log = (LOGS / f"{job.replace(':', '_')}.log").open("w", encoding="utf-8")
            handles[job] = log
            running[job] = subprocess.Popen(
                command(job), cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT
            )
            record[job] = {
                "start": datetime.now().isoformat(timespec="seconds"),
                "start_epoch": time.time(),
                "end": None,
                "command": command(job),
            }
            mine.add(job)
            save(record, mine)
        time.sleep(POLL_SECONDS)
        for job, process in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handles.pop(job).close()
            record[job].update(
                end=datetime.now().isoformat(timespec="seconds"),
                end_epoch=time.time(),
                exit=code,
            )
            save(record, mine)
            del running[job]
    return 0


if __name__ == "__main__":
    sys.exit(main())
