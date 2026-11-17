"""Work prompt 86's runs under WSL2, one BLAS thread each.

    python docs/reports/probe86/run86.py references   # the three, at once
    python docs/reports/probe86/run86.py locations    # mnt_c then linux_fs, alone
    python docs/reports/probe86/run86.py repeats      # the same pair again
    python docs/reports/probe86/run86.py counted      # the three, counted, here

Run with the WSL2 environment's interpreter, which each run inherits. The three
references start together, as prompt 85's Windows references ran three at once;
the two runs of the run file's placement run one after the other with no other
solver process, so that their wall times differ only by where the file is.
One line per start and finish goes to ``results/builder86/run86.log`` and each
run's output to ``results/builder86/logs/``.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe86  # noqa: E402 -- follows sys.path.insert

LOG = probe86.RESULTS / "run86.log"
LOGS = probe86.RESULTS / "logs"
THREADS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
POLL_SECONDS = 5.0


def log(text: str) -> None:
    """One dated line to the queue's log and to stdout."""
    line = f"{datetime.now().isoformat(timespec='seconds')} {text}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def start(location: str, scenario: str) -> subprocess.Popen:
    """One run's process, its output to LOGS."""
    LOGS.mkdir(parents=True, exist_ok=True)
    name = f"{location}_{scenario}"
    with (
        (LOGS / f"{name}.out").open("w") as out,
        (LOGS / f"{name}.err").open("w") as err,
    ):
        proc = subprocess.Popen(
            [sys.executable, str(HERE / "probe86.py"), "run", location, scenario],
            cwd=probe86.ROOT,
            env={**os.environ, **THREADS},
            stdout=out,
            stderr=err,
        )
    log(f"start {name} pid {proc.pid}")
    return proc


def wait(procs: dict[str, subprocess.Popen]) -> list[str]:
    """Wait for every process; the names of those that failed."""
    started = time.perf_counter()
    failed = []
    while procs:
        time.sleep(POLL_SECONDS)
        for name, proc in list(procs.items()):
            code = proc.poll()
            if code is None:
                continue
            del procs[name]
            minutes = (time.perf_counter() - started) / 60.0
            if code != 0:
                failed.append(name)
            log(
                f"{'done' if code == 0 else 'failed'} {name} exit {code} {minutes:.1f} min"
            )
    return failed


def main() -> int:
    """The queue the argument names."""
    which = sys.argv[1:2]
    if which == ["references"]:
        procs = {f"reference_{s}": start("reference", s) for s in probe86.SCENARIOS}
        failed = wait(procs)
    elif which == ["counted"]:
        location = "windows_counted" if sys.platform == "win32" else "counted"
        procs = {f"{location}_{s}": start(location, s) for s in probe86.SCENARIOS}
        failed = wait(procs)
    elif which in (["locations"], ["repeats"]):
        suffix = "" if which == ["locations"] else "_repeat"
        failed = []
        for location in (f"mnt_c{suffix}", f"linux_fs{suffix}"):
            failed += wait({f"{location}_dust": start(location, "dust")})
    else:
        print(__doc__)
        return 2
    log(f"queue {which[0]} empty; failed: {failed if failed else 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
