"""Run prompt 84's measurement processes one at a time.

    python docs/reports/probe84/run84.py            # every job not yet recorded
    python docs/reports/probe84/run84.py --list     # the jobs and their state

Each job is one ``probe84.py`` process with one BLAS thread. Before it starts,
the processes using the CPU are sampled (``Get-Counter`` on ``% Processor
Time``, two seconds, the top twelve, the counter Task Manager's CPU column
reads) and written to ``results/builder84/procs/<job>.json``. After each
calibration the prompt's stop rule is applied: a run whose fields, budgets or
flow arrays differ from the first unwrapped run's stops the queue, and so
does a window's (a march calibration's) wrapped wall over 5% above its
unwrapped wall, the prompt's rule. A flow calibration's difference is logged
and not a stop: the prompt's rule names a window, and on this machine the
spread of identical unwrapped flow runs (up to 14%) is wider than the
difference (``tables84.py`` sets it beside the event-count estimate from
``overhead84.py``).
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
RESULTS = ROOT / "results" / "builder84"
PROCS = RESULTS / "procs"
LOG = RESULTS / "run84.log"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PROBE = HERE / "probe84.py"
ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
# The prompt's stop: the wrappers' overhead over 5% of a window's wall time.
OVERHEAD_LIMIT = 0.05

# (job, record it writes, probe arguments), in the order they run: the
# calibrations first, then the coarse grids' whole runs and their windows,
# then the product grid's 300 outer iterations and its window, then cProfile.
# The coarse grids' window calibrations run to 40 s and 20 s so that each run
# lasts about as long as the 320x120 one: an 80x30 window to 10.5 s was 85
# steps under a second, whose identical unwrapped runs spread by 28%
# (results/builder84/calibrate_march_80x30_short.json, kept for the report).
# The first instrument timed the march's inline code by line regions, which
# cost the 80x30 window 7% to 10% (overhead84.py ablate); its records are kept
# as *_regions.json, and the march is now timed without them ("lean",
# probe84.MARCH_REGIONS), each window's inline split taken from a run of it
# with them on (split_window_<grid>).
JOBS = [
    ("calibrate_flow_320x120", ["calibrate-flow", "320x120", "40"]),
    ("calibrate_march_320x120", ["calibrate-march", "320x120", "10.5"]),
    ("calibrate_flow_80x30", ["calibrate-flow", "80x30", "300"]),
    ("calibrate_flow_160x60", ["calibrate-flow", "160x60", "150"]),
    ("run_80x30", ["run", "80x30"]),
    ("calibrate_march_80x30", ["calibrate-march", "80x30", "40"]),
    ("window_80x30", ["window", "80x30", "9", "14"]),
    ("split_window_80x30", ["window", "80x30", "9", "14", "--regions"]),
    ("run_160x60", ["run", "160x60"]),
    ("calibrate_march_160x60", ["calibrate-march", "160x60", "20"]),
    ("window_160x60", ["window", "160x60", "9", "14"]),
    ("split_window_160x60", ["window", "160x60", "9", "14", "--regions"]),
    ("flow_320x120", ["flow", "320x120", "300"]),
    ("window_320x120", ["window", "320x120", "9", "14"]),
    ("split_window_320x120", ["window", "320x120", "9", "14", "--regions"]),
    ("cprofile_80x30", ["cprofile", "80x30"]),
    ("overhead", ["overhead"]),
    ("construct_80x30", ["construct", "80x30"]),
    ("construct_160x60", ["construct", "160x60"]),
    ("construct_320x120", ["construct", "320x120"]),
    (
        "ablate_80x30_parts",
        ["ablate", "80x30", "40", "none,full,none,lean,none,regions,none", "parts"],
    ),
]
OVERHEAD = HERE / "overhead84.py"

SAMPLE = (
    "$s = Get-Counter '\\Process(*)\\% Processor Time' -SampleInterval 2 "
    "-MaxSamples 1 -ErrorAction SilentlyContinue; "
    "$s.CounterSamples | Where-Object { $_.InstanceName -notin @('_total','idle') } "
    "| Sort-Object CookedValue -Descending | Select-Object -First 12 "
    "@{n='process';e={$_.InstanceName}}, @{n='cpu_percent';e={[math]::Round($_.CookedValue,1)}} "
    "| ConvertTo-Json"
)


def log(line: str) -> None:
    """Append LINE to the queue's log with a timestamp, and print it."""
    stamp = datetime.now().isoformat(timespec="seconds")
    with LOG.open("a") as f:
        f.write(f"{stamp} {line}\n")
    print(stamp, line, flush=True)


def sample(job: str) -> None:
    """The top processes by CPU now, to results/builder84/procs/JOB.json."""
    PROCS.mkdir(parents=True, exist_ok=True)
    found = subprocess.run(
        ["powershell", "-NoProfile", "-Command", SAMPLE],
        capture_output=True,
        text=True,
    )
    try:
        rows = json.loads(found.stdout)
    except json.JSONDecodeError:
        rows = {"error": found.stdout[-500:] + found.stderr[-500:]}
    record = {"sampled": datetime.now().isoformat(timespec="seconds"), "top": rows}
    (PROCS / f"{job}.json").write_text(json.dumps(record, indent=1))


def calibration_verdict(job: str) -> str | None:
    """None when JOB's calibration passes the stop rule, else the reason."""
    record = json.loads((RESULTS / f"{job}.json").read_text())
    if any(record["differ"]):
        return f"not bitwise: {record['differ']}"
    walls = record["wall_ns"]
    wrapped = [w for w, o in zip(walls, record["order"], strict=True) if o == "wrapped"]
    plain = [w for w, o in zip(walls, record["order"], strict=True) if o == "unwrapped"]
    overhead = sum(wrapped) / len(wrapped) / (sum(plain) / len(plain)) - 1.0
    log(f"{job}: overhead {overhead:+.4f} of the unwrapped mean")
    if job.startswith("calibrate_march") and overhead > OVERHEAD_LIMIT:
        return f"overhead {overhead:.4f} over {OVERHEAD_LIMIT}"
    return None


def main() -> int:
    """Run each job not yet recorded, one at a time; stop on a failure or the stop rule."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.list:
        for job, argv in JOBS:
            done = (RESULTS / f"{job}.json").exists()
            print(f"{'done' if done else 'todo'}  {job}  {' '.join(argv)}")
        return 0
    env = {**os.environ, **ENV}
    for job, argv in JOBS:
        if (RESULTS / f"{job}.json").exists():
            continue
        sample(job)
        log(f"start {job}: {' '.join(argv)}")
        started = time.perf_counter()
        if argv[0] in ("overhead", "ablate"):
            command = [str(OVERHEAD), *argv[1:]]
            if argv[0] == "ablate":
                command = [str(OVERHEAD), *argv]
        else:
            command = [str(PROBE), *argv]
        with (RESULTS / f"{job}.out").open("w") as out:
            done = subprocess.run(
                [str(PYTHON), *command],
                cwd=ROOT,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
        log(f"end {job}: exit {done.returncode}, {time.perf_counter() - started:.1f} s")
        if done.returncode != 0:
            log("stopped: a job failed")
            return 1
        if job.startswith("calibrate"):
            reason = calibration_verdict(job)
            if reason is not None:
                log(f"stopped by the prompt's rule: {job} {reason}")
                return 2
    log("queue done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
