"""Run the sweep's configurations through scripts/run_scenario.py (prompt 72).

    python docs/reports/probe72/run72.py            # run every job not yet run
    python docs/reports/probe72/run72.py --dry-run  # print the queue
    python docs/reports/probe72/run72.py --status   # the running jobs' progress
    python docs/reports/probe72/run72.py --only NAME [NAME ...]

A job is one configuration of configs72.py. At most MAX_CONCURRENT run at
once, each from rest with one BLAS thread, as every product measurement since
prompt 46; an arm whose flow state is another arm's (configs72.FLOWS_FROM)
waits for that arm's run file and passes it as --flows-from. The queue is
ordered so the product-grid runs, the long ones, start first and the coarse
runs fill the remaining places. Run files, summaries (the runner's stdout)
and logs go to results/builder72/; the start and end of every job to
results/builder72/runs72.json, which --status reads with each running job's
checkpoint (step and time, written every 1000 steps) to estimate the march's
rate and the time left, the estimate the pass's budget is checked against.

A job whose summary already reports a wall time is complete and is skipped,
so the script can be run again after an interruption; a job whose run file
exists without a summary is reported and left alone (the runner refuses to
overwrite a run file; remove it or restart it by hand).
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

import configs72  # noqa: E402 -- follows sys.path.insert

RESULTS = ROOT / "results" / "builder72"
LOGS = RESULTS / "logs"
RECORD = RESULTS / "runs72.json"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER = ROOT / "scripts" / "run_scenario.py"
MAX_CONCURRENT = 5
POLL_SECONDS = 15
ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
PLANNED = re.compile(r"scenario: \d+ intervals, (\d+) steps planned")


def jobs() -> list[dict]:
    """Every job in queue order, with its dependency."""
    rows = []
    for name, scenario, arm, grid in configs72.run_names():
        source = configs72.FLOWS_FROM.get(arm)
        depends = (
            f"{scenario}_{source}_{configs72.grid_tag(grid)}"
            if source is not None and grid == configs72.COARSE
            else None
        )
        rows.append(
            {
                "name": name,
                "scenario": scenario,
                "arm": arm,
                "grid": configs72.grid_tag(grid),
                "depends": depends,
                "fine": grid == configs72.FINE,
            }
        )
    # The long runs first: the six product-grid runs lead, the rest keep their order.
    return sorted(rows, key=lambda r: 0 if r["fine"] else 1)


def summary_complete(name: str) -> bool:
    path = RESULTS / f"{name}.txt"
    return path.exists() and "wall time:" in path.read_text(encoding="utf-8")


def load_record() -> dict:
    if RECORD.exists():
        return json.loads(RECORD.read_text(encoding="utf-8"))
    return {}


def save_record(record: dict) -> None:
    RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")


def launch(job: dict, running: int) -> subprocess.Popen:
    name = job["name"]
    command = [
        str(PYTHON),
        str(RUNNER),
        str(RESULTS / "configs" / f"scenario_{name}.yaml"),
        "--out",
        str(RESULTS / f"{name}.h5"),
        "--concurrent",
        str(running),
    ]
    if job["depends"] is not None:
        command += ["--flows-from", str(RESULTS / f"{job['depends']}.h5")]
    # Handed to Popen, which holds them for the process lifetime; closed with it.
    out = open(RESULTS / f"{name}.txt", "w", encoding="utf-8")  # noqa: SIM115 -- see above
    err = open(LOGS / f"{name}.log", "w", encoding="utf-8")  # noqa: SIM115 -- see above
    return subprocess.Popen(
        command, stdout=out, stderr=err, cwd=ROOT, env={**os.environ, **ENV}
    )


def run(selected: list[dict]) -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(PYTHON), str(HERE / "configs72.py")], check=True, cwd=ROOT)
    record = load_record()
    pending = []
    for job in selected:
        if summary_complete(job["name"]):
            print(f"{job['name']}: complete, skipped")
            continue
        if (RESULTS / f"{job['name']}.h5").exists():
            print(f"{job['name']}: run file exists without a summary, left alone")
            continue
        pending.append(job)
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
            print(f"{name}: exit {code} at {record[name]['ended']}")
            if code != 0:
                failed.append(name)
        for job in list(pending):
            if len(active) >= MAX_CONCURRENT:
                break
            depends = job["depends"]
            if depends is not None and not summary_complete(depends):
                if depends in failed:
                    pending.remove(job)
                    failed.append(job["name"])
                    print(f"{job['name']}: not run, {depends} failed")
                continue
            pending.remove(job)
            process = launch(job, len(active))
            active[job["name"]] = process
            record[job["name"]] = {
                "started": datetime.now().isoformat(timespec="seconds"),
                "pid": process.pid,
                "concurrent_at_launch": len(active) - 1,
            }
            save_record(record)
            print(f"{job['name']}: started at {record[job['name']]['started']}")
        if pending or active:
            time.sleep(POLL_SECONDS)
    print(f"all done; failed: {failed if failed else 'none'}")
    return 1 if failed else 0


def status() -> int:
    """Each running job's march rate from its checkpoint and the time left."""
    from src.io_manager import RunFile

    record = load_record()
    now = datetime.now()
    for name, row in record.items():
        if "ended" in row:
            continue
        started = datetime.fromisoformat(row["started"])
        elapsed = (now - started).total_seconds()
        log = LOGS / f"{name}.log"
        planned = None
        if log.exists():
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
        # The flow solve's wall is in the log once it is done; until the first
        # checkpoint the rate cannot be separated from it, so the rate here is
        # steps over the time since the run started, a lower bound on the
        # march's own rate and so an upper bound on the time left.
        rate = step / elapsed
        left = (planned - step) / rate if rate > 0 else float("inf")
        print(
            f"{name}: step {step} of {planned} (t = {t:.1f} s) after "
            f"{elapsed / 60:.1f} min; {rate:.2f} steps/s over the run; "
            f"about {left / 60:.0f} min left at that rate"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="NAME")
    args = parser.parse_args()
    if args.status:
        return status()
    selected = jobs()
    if args.only:
        unknown = set(args.only) - {j["name"] for j in selected}
        if unknown:
            parser.error(f"unknown job(s): {sorted(unknown)}")
        selected = [j for j in selected if j["name"] in args.only]
    if args.dry_run:
        for job in selected:
            after = f" after {job['depends']}" if job["depends"] else ""
            print(f"{job['name']}{after}")
        return 0
    return run(selected)


if __name__ == "__main__":
    sys.exit(main())
