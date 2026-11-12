"""Run the residual probe's configurations, at most five at a time (prompt 77).

    python docs/reports/probe77/run77.py                 # every run not yet done
    python docs/reports/probe77/run77.py --only R7b R8b  # named runs
    python docs/reports/probe77/run77.py --status        # progress and time left

Each job is ``probe77.py RUN`` in its own process with one BLAS thread, the
process count beside it passed as --concurrent for the run file's record, as
every product measurement since prompt 46. The 320x120 run starts first, the
long one; the others fill the remaining places. Summaries (the runner's
stdout) go to results/builder77/RUN.txt, logs to results/builder77/logs/, and
each job's start and end to results/builder77/runs77.json, which --status
reads with each running job's checkpoint (written every 1000 steps) for the
march's rate and the time left, the estimate the 3-hour budget is checked
against. A job whose probe record (RUN.probe.json) exists is complete and is
skipped; one whose run file exists without it is reported and left alone.
R7b and R8b, the reruns at 2000 sweeps, run only when named.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import configs77  # noqa: E402 -- follows sys.path.insert

RESULTS = ROOT / "results" / "builder77"
LOGS = RESULTS / "logs"
RECORD = RESULTS / "runs77.json"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PROBE = HERE / "probe77.py"
MAX_CONCURRENT = 5
POLL_SECONDS = 15
ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
PLANNED = re.compile(r"scenario: \d+ intervals, (\d+) steps planned")


def queue(names: list[str] | None) -> list[str]:
    """The jobs in launch order: the finest grid first."""
    if names is None:
        names = [run.name for run in configs77.RUNS]
    return sorted(names, key=lambda n: -configs77.BY_NAME[n].grid[0])


def load_record() -> dict:
    """The queue record, job name to its start, end, pid and exit code."""
    if RECORD.exists():
        return json.loads(RECORD.read_text(encoding="utf-8"))
    return {}


def save_record(record: dict) -> None:
    """Write the queue record."""
    RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")


def launch(name: str, running: int) -> subprocess.Popen:
    """Start one probe process; ``running`` is the count already running."""
    command = [str(PYTHON), str(PROBE), name, "--concurrent", str(running)]
    # Handed to Popen, which holds them for the process lifetime; closed with it.
    out = open(RESULTS / f"{name}.txt", "w", encoding="utf-8")  # noqa: SIM115 -- see above
    err = open(LOGS / f"{name}.log", "w", encoding="utf-8")  # noqa: SIM115 -- see above
    return subprocess.Popen(
        command, stdout=out, stderr=err, cwd=ROOT, env={**os.environ, **ENV}
    )


def run(names: list[str]) -> int:
    """Run the named jobs, at most MAX_CONCURRENT at once."""
    LOGS.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(PYTHON), str(HERE / "configs77.py")], check=True, cwd=ROOT)
    record = load_record()
    pending = []
    for name in names:
        if (RESULTS / f"{name}.probe.json").exists():
            print(f"{name}: complete, skipped")
        elif (RESULTS / f"{name}.h5").exists():
            print(f"{name}: run file exists without a probe record, left alone")
        else:
            pending.append(name)
    active: dict[str, subprocess.Popen] = {}
    failed = []
    while pending or active:
        for name, process in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            del active[name]
            record[name]["ended"] = datetime.now().isoformat(timespec="seconds")
            record[name]["returncode"] = code
            save_record(record)
            print(f"{name}: exit {code} at {record[name]['ended']}", flush=True)
            if code != 0:
                failed.append(name)
        while pending and len(active) < MAX_CONCURRENT:
            name = pending.pop(0)
            active[name] = launch(name, len(active))
            record[name] = {
                "started": datetime.now().isoformat(timespec="seconds"),
                "pid": active[name].pid,
                "concurrent_at_launch": len(active) - 1,
            }
            save_record(record)
            print(f"{name}: started at {record[name]['started']}", flush=True)
        if pending or active:
            time.sleep(POLL_SECONDS)
    print(f"all done; failed: {failed if failed else 'none'}")
    return 1 if failed else 0


def status() -> int:
    """Each running job's march rate from its checkpoint and the time left."""
    from src.io_manager import RunFile

    now = datetime.now()
    for name, row in load_record().items():
        if "ended" in row:
            continue
        elapsed = (now - datetime.fromisoformat(row["started"])).total_seconds()
        log = LOGS / f"{name}.log"
        found = PLANNED.search(log.read_text(encoding="utf-8", errors="replace"))
        planned = int(found.group(1)) if found else None
        try:
            checkpoint = RunFile(RESULTS / f"{name}.h5").read_checkpoint()
            step, t = checkpoint.step, checkpoint.t
        except (KeyError, OSError):
            step, t = None, None
        if step is None or planned is None:
            print(f"{name}: {elapsed / 60:.1f} min elapsed, no checkpoint yet")
            continue
        # Steps over the time since the start: a lower bound on the march's rate,
        # so an upper bound on the time left.
        rate = step / elapsed
        left = (planned - step) / rate if rate > 0 else float("inf")
        print(
            f"{name}: step {step} of {planned} (t = {t:.1f} s) after "
            f"{elapsed / 60:.1f} min; {rate:.2f} steps/s; about {left / 60:.0f} min left"
        )
    return 0


def main() -> int:
    """Run the queue or report the running jobs' progress."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--only", nargs="+", choices=sorted(configs77.BY_NAME))
    args = parser.parse_args()
    if args.status:
        return status()
    return run(queue(args.only))


if __name__ == "__main__":
    sys.exit(main())
