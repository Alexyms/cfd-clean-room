"""Work prompt 85's runs, at most three processes at once, one BLAS thread each.

    python docs/reports/probe85/run85.py          # every run not yet done
    python docs/reports/probe85/run85.py --list   # the runs and their state

A run is done when its summary (``.json`` beside the run file) exists; a run
that exits without one is logged as failed and not started again by this
process. One line per start and finish goes to ``results/builder85/run85.log``,
and each run's output to ``results/builder85/logs/``. The 160x60 runs start
first, so the longest are not left to the end.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import probe85  # noqa: E402 -- follows sys.path.insert

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = probe85.RESULTS / "run85.log"
LOGS = probe85.RESULTS / "logs"
MAX_PROCESSES = 3  # the prompt's limit
POLL_SECONDS = 5.0
THREADS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def log(text: str) -> None:
    """One dated line to the queue's log and to stdout."""
    line = f"{datetime.now().isoformat(timespec='seconds')} {text}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def main() -> int:
    """Start runs until none is left, never more than MAX_PROCESSES at once."""
    if "--list" in sys.argv:
        for job in probe85.jobs():
            print(*job, "done" if probe85.done(*job) else "")
        return 0
    LOGS.mkdir(parents=True, exist_ok=True)
    waiting = [job for job in probe85.jobs() if not probe85.done(*job)]
    running: dict[tuple[str, str, str], tuple[subprocess.Popen, float]] = {}
    failed = []
    env = {**os.environ, **THREADS}
    log(f"queue: {len(waiting)} runs to do")
    while waiting or running:
        while waiting and len(running) < MAX_PROCESSES:
            job = waiting.pop(0)
            name = "_".join(job)
            with (
                (LOGS / f"{name}.out").open("w") as out,
                (LOGS / f"{name}.err").open("w") as err,
            ):
                proc = subprocess.Popen(
                    [str(PYTHON), str(probe85.Path(probe85.__file__)), "run", *job],
                    cwd=ROOT,
                    env=env,
                    stdout=out,
                    stderr=err,
                )
            running[job] = (proc, time.perf_counter())
            log(f"start {name} pid {proc.pid}")
        time.sleep(POLL_SECONDS)
        for job, (proc, started) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[job]
            name = "_".join(job)
            minutes = (time.perf_counter() - started) / 60.0
            ok = code == 0 and probe85.done(*job)
            if not ok:
                failed.append(name)
            log(f"{'done' if ok else 'failed'} {name} exit {code} {minutes:.1f} min")
    log(f"queue empty; failed: {failed if failed else 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
