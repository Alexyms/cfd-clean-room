"""Work prompt 79's queue of runs through scripts/run_scenario.py, resumably.

    python docs/reports/probe79/run79.py             # work the queue to its end
    python docs/reports/probe79/run79.py --dry-run   # print the queue and estimates
    python docs/reports/probe79/run79.py --status    # progress of the running jobs
    python docs/reports/probe79/run79.py --set max_fine=1

A job is one configuration of ``configs79.jobs()``, run by
``scripts/run_scenario.py`` inside ``record79.py``, which records each
transport solve's sweeps beside the run file. The safety rules are
Alex's (prompt 79): at most ``max_processes`` (3) at once, at most
``max_fine`` (2) of them on 320x120, one BLAS thread each, and nothing started
while the machine's available physical memory is below ``memory_floor`` (25%)
of its total, read from ``GlobalMemoryStatusEx`` before every start and logged
with it. Starts are at least LAUNCH_SPACING apart so a new process has
allocated before the next reading. The caps live in the queue file's
``settings`` and are read again at every poll, so ``--set`` changes a running
queue.

The queue's state is ``results/builder79/queue.json``. A job is done when its
process exited 0 and its summary reports a wall time; it is failed when the
runner ended it with its own ``error:`` line (exit 1, a flow state that did not
converge or a refused breach; exit 2, a refused input), which is a result and
is not retried. Anything else (a crash, a kill, a reboot) leaves it
interrupted: started again, a job whose run file holds a march checkpoint
continues with ``--restart``, and one without has its partial files moved to
``results/builder79/aborted/`` and starts again, at most ``max_attempts``
times. A job left running by an earlier runner is adopted (its pid and
creation time are checked), so running this script again after its own crash
starts nothing twice.

Order: the 320x120 jobs first, longest first by the wall time of the run each
replaces (less the flow solves it made, which this pass copies in), then the
rest, longest first, filling the slot the 320x120 cap leaves.

The stop rule: a job whose predecessor finished (all its flow states
converged and are copied in) must solve none. A ``flow state ...: solving``
line in such a job's log means a state did not reproduce by its fingerprint;
the queue then starts nothing more and says why in the log.

Each job's summary (the runner's stdout) and log (its stderr) go beside its
run file under ``results/builder79/<group>/``; one line per start and finish
to ``results/builder79/queue.log``.
"""

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import configs79  # noqa: E402 -- follows sys.path.insert

RESULTS = configs79.RESULTS
QUEUE = RESULTS / "queue.json"
LOG = RESULTS / "queue.log"
ABORTED = RESULTS / "aborted"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
# run_scenario.py under the sweeps recorder (record79.py), which passes every
# argument through and changes nothing the run computes.
RUNNER = HERE / "record79.py"
ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
POLL_SECONDS = 15
LAUNCH_SPACING = 60.0
MEMORY_LOG_SECONDS = 300.0
DEFAULT_SETTINGS: dict[str, object] = {
    "max_processes": 3,
    "max_fine": 2,
    "memory_floor": 0.25,
    "max_attempts": 3,
    "stopped": None,
}
PLANNED = re.compile(r"(\d+) steps planned")
SOLVING = re.compile(r"flow state (\w+): solving")
OLD_WALL = re.compile(r"^wall time: ([\d.]+) s", re.MULTILINE)
OLD_SOLVED = re.compile(
    r"^  \w+: (?:rest|warm from \w+|resumed), \d+ outer, \w+, ([\d.]+) s$", re.MULTILINE
)
PLACEMENT_WALLS = ROOT / "results" / "builder75" / "run75.json"
SWEEP_QUEUE = ROOT / "results" / "builder72" / "runs72.json"

# -- Windows process and memory queries (ctypes, no new dependency) -----------

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_VM_READ = 0x0010
SYNCHRONIZE = 0x00100000
STILL_ACTIVE = 259
CREATE_NO_WINDOW = 0x08000000
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


class MemoryStatus(ctypes.Structure):
    """MEMORYSTATUSEX."""

    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class ProcessMemoryCounters(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS."""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
KERNEL32.OpenProcess.restype = wintypes.HANDLE
KERNEL32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
KERNEL32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
KERNEL32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
    ctypes.POINTER(wintypes.FILETIME)
] * 4
KERNEL32.K32GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ProcessMemoryCounters),
    wintypes.DWORD,
]
KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]


def memory() -> tuple[int, int]:
    """The machine's (available, total) physical memory in bytes."""
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not KERNEL32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return status.ullAvailPhys, status.ullTotalPhys


def memory_text() -> str:
    """The available memory as the log states it."""
    avail, total = memory()
    return f"avail {avail / 2**30:.1f} of {total / 2**30:.1f} GB ({avail / total:.0%})"


def open_process(pid: int) -> int | None:
    """A handle on a process for its exit code, times and memory, or None."""
    handle = KERNEL32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ | SYNCHRONIZE, False, pid
    )
    return handle or None


def exit_code(handle: int) -> int | None:
    """The process's exit code, None while it runs."""
    code = wintypes.DWORD()
    if not KERNEL32.GetExitCodeProcess(handle, ctypes.byref(code)):
        return None
    return None if code.value == STILL_ACTIVE else int(code.value)


def creation_time(handle: int) -> int:
    """The process's creation time as a FILETIME integer (100 ns since 1601)."""
    times = [wintypes.FILETIME() for _ in range(4)]
    KERNEL32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times))
    return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime


class ProcessEntry(ctypes.Structure):
    """PROCESSENTRY32W."""

    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


TH32CS_SNAPPROCESS = 0x00000002
KERNEL32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
KERNEL32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
KERNEL32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
KERNEL32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]


def child_pid(pid: int) -> int | None:
    """The first python.exe child of a process, or None.

    The venv's ``python.exe`` on Windows is a launcher that starts the base
    interpreter as its child and exits with its code, so the run's memory is
    the child's while the exit code is the launcher's.
    """
    snapshot = KERNEL32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snapshot or snapshot == wintypes.HANDLE(-1).value:
        return None
    entry = ProcessEntry()
    entry.dwSize = ctypes.sizeof(ProcessEntry)
    found = None
    try:
        more = KERNEL32.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            # The console host a windowless process gets is a child as well.
            if (
                entry.th32ParentProcessID == pid
                and entry.th32ProcessID != pid
                and entry.szExeFile.lower() == "python.exe"
            ):
                found = int(entry.th32ProcessID)
                break
            more = KERNEL32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        KERNEL32.CloseHandle(snapshot)
    return found


def peak_working_set(handle: int) -> int | None:
    """The process's peak working set in bytes (readable after it exits too)."""
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    if not KERNEL32.K32GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    ):
        return None
    return int(counters.PeakWorkingSetSize)


# -- The queue file ------------------------------------------------------------


def now() -> str:
    """The local time, as every record states it."""
    return datetime.now().isoformat(timespec="seconds")


def log(line: str) -> None:
    """Append one line to the queue log and echo it."""
    text = f"{now()} {line}"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text, flush=True)


def load_queue() -> dict:
    """The queue file, created with the default settings when absent."""
    if QUEUE.exists():
        queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    else:
        queue = {"settings": dict(DEFAULT_SETTINGS), "jobs": {}}
    for key, value in DEFAULT_SETTINGS.items():
        queue["settings"].setdefault(key, value)
    return queue


def save_queue(queue: dict) -> None:
    """Write the queue file atomically, keeping settings changed on disk meanwhile."""
    if QUEUE.exists():
        on_disk = json.loads(QUEUE.read_text(encoding="utf-8"))
        queue["settings"] = on_disk.get("settings", queue["settings"])
    tmp = QUEUE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    os.replace(tmp, QUEUE)


def set_setting(assignment: str) -> int:
    """Change one setting of the queue file: ``key=value``, the value as JSON."""
    key, _, value = assignment.partition("=")
    if key not in DEFAULT_SETTINGS:
        print(f"unknown setting {key!r}; one of {sorted(DEFAULT_SETTINGS)}")
        return 2
    queue = load_queue()
    queue["settings"][key] = json.loads(value)
    tmp = QUEUE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    os.replace(tmp, QUEUE)
    log(f"setting {key} = {queue['settings'][key]!r} (by --set)")
    return 0


# -- Jobs ----------------------------------------------------------------------


def summary_path(job: dict) -> Path:
    """The job's summary, the runner's stdout."""
    return job["out"].with_suffix(".txt")


def log_path(job: dict) -> Path:
    """The job's log, the runner's stderr."""
    return job["out"].with_suffix(".log")


def summary_complete(path: Path) -> bool:
    """Whether a summary reports a wall time, which the runner prints last."""
    return path.exists() and "wall time:" in path.read_text(
        encoding="utf-8", errors="replace"
    )


def predecessor_finished(job: dict) -> bool:
    """Whether the run a job replaces finished, so every flow state it used is copied in."""
    if job["before"] is None:
        return False
    if job["group"] == "L":
        return job["before"].exists()
    return summary_complete(job["before"].with_suffix(".txt"))


def old_estimate(job: dict) -> float:
    """Seconds the job should take: the replaced run's wall less the flow solves it made."""
    if job["group"] == "L":
        walls = json.loads(PLACEMENT_WALLS.read_text(encoding="utf-8"))["timing"]
        return float(walls[job["name"]])
    if job["before"] is None:
        return 0.0
    old = job["before"].with_suffix(".txt")
    text = old.read_text(encoding="utf-8", errors="replace") if old.exists() else ""
    wall = OLD_WALL.search(text)
    if wall is not None:
        solved = sum(float(s) for s in OLD_SOLVED.findall(text))
        return float(wall.group(1)) - solved
    record = json.loads(SWEEP_QUEUE.read_text(encoding="utf-8")).get(job["name"])
    if record is not None and "ended" in record:
        started = datetime.fromisoformat(record["started"])
        return (datetime.fromisoformat(record["ended"]) - started).total_seconds()
    return 0.0


def ordered_jobs() -> list[dict]:
    """configs79's jobs, the 320x120 ones first, each part longest first."""
    rows = configs79.jobs()
    for row in rows:
        row["estimate"] = old_estimate(row)
        row["fine"] = row["grid"] == configs79.FINE
    return sorted(rows, key=lambda r: (0 if r["fine"] else 1, -r["estimate"]))


def last_error_line(job: dict) -> str | None:
    """The log's last line when it is the runner's own ``error:`` line."""
    path = log_path(job)
    if not path.exists():
        return None
    lines = [
        ln
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip()
    ]
    if lines and lines[-1].startswith("error:"):
        return lines[-1]
    return None


def classify(job: dict, code: int | None) -> tuple[str, str]:
    """A finished process's outcome: (done | failed | interrupted, reason)."""
    if code in (0, None) and summary_complete(summary_path(job)):
        return "done", "summary written"
    error = last_error_line(job)
    if code in (1, 2, None) and error is not None:
        return "failed", error
    return "interrupted", f"exit {code}, no summary and no error line"


def march_checkpoint(job: dict) -> bool:
    """Whether the job's run file holds a march checkpoint to restart from."""
    from src.io_manager import RunFile

    if not job["out"].exists():
        return False
    try:
        RunFile(job["out"]).read_checkpoint()
    except (KeyError, OSError, ValueError):
        return False
    return True


def move_aborted(job: dict) -> list[str]:
    """Move a job's partial files to the aborted directory; return what moved."""
    target = ABORTED / f"{job['name']}.{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    moved = []
    base = job["out"]
    for path in (
        base,
        Path(str(base.with_suffix("")) + ".checkpoint.h5"),
        Path(str(base.with_suffix("")) + ".checkpoint.h5.tmp"),
        summary_path(job),
        log_path(job),
    ):
        if path.exists():
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target / path.name))
            moved.append(path.name)
    return moved


class Running:
    """A job's process, launched here or adopted from an earlier runner."""

    def __init__(self, job: dict, pid: int, handle: int, started: float) -> None:
        self.job = job
        self.pid = pid
        self.handle = handle
        self.started = started
        self.peak = 0
        self.peak_first_10_min: int | None = None
        self.memory_handle: int | None = None

    def sample(self) -> int:
        """Read the interpreter's peak working set (the launcher's child); return it."""
        if self.memory_handle is None:
            child = child_pid(self.pid)
            if child is not None:
                self.memory_handle = open_process(child)
        if self.memory_handle:
            peak = peak_working_set(self.memory_handle)
            if peak is not None:
                self.peak = max(self.peak, peak)
        return self.peak

    def close(self) -> None:
        """Close the handles."""
        for handle in (self.handle, self.memory_handle):
            if handle:
                KERNEL32.CloseHandle(handle)


def launch(job: dict, mode: str, beside: int) -> tuple[int, int]:
    """Start a job's run_scenario.py process; return its pid and creation time."""
    command = [str(PYTHON), str(RUNNER), str(job["config"])]
    if mode == "restart":
        command += ["--restart", str(job["out"])]
    else:
        command += ["--out", str(job["out"])]
        if job["flows_from"] is not None:
            command += ["--flows-from", str(job["flows_from"])]
    command += ["--concurrent", str(beside)]
    job["out"].parent.mkdir(parents=True, exist_ok=True)
    # Handed to Popen, which holds them for the process lifetime; closed after.
    out = open(summary_path(job), "w", encoding="utf-8")  # noqa: SIM115 -- see above
    err = open(log_path(job), "a", encoding="utf-8")  # noqa: SIM115 -- see above
    err.write(f"--- run79 {mode} at {now()}: {' '.join(command)}\n")
    err.flush()
    env = {**os.environ, **ENV}
    flags = CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        process = subprocess.Popen(
            command,
            stdout=out,
            stderr=err,
            cwd=ROOT,
            env=env,
            creationflags=flags | CREATE_BREAKAWAY_FROM_JOB,
        )
    except OSError:
        # The job this runner sits in may not allow breakaway; run inside it.
        process = subprocess.Popen(
            command, stdout=out, stderr=err, cwd=ROOT, env=env, creationflags=flags
        )
    out.close()
    err.close()
    handle = open_process(process.pid)
    created = creation_time(handle) if handle else 0
    if handle:
        KERNEL32.CloseHandle(handle)
    return process.pid, created


def adopt(job: dict, record: dict) -> Running | None:
    """The process an earlier runner left running for this job, if it still runs."""
    attempt = record["attempts"][-1]
    handle = open_process(attempt["pid"])
    if not handle:
        return None
    if exit_code(handle) is not None or creation_time(handle) != attempt["created"]:
        KERNEL32.CloseHandle(handle)
        return None
    started = datetime.fromisoformat(attempt["started"]).timestamp()
    return Running(job, attempt["pid"], handle, started)


def stop_reason(running: Running) -> str | None:
    """Why the queue must stop: a job solving a state its predecessor converged."""
    job = running.job
    if not predecessor_finished(job):
        return None
    path = log_path(job)
    if not path.exists():
        return None
    found = SOLVING.search(path.read_text(encoding="utf-8", errors="replace"))
    if found is None:
        return None
    return (
        f"{job['name']} is solving flow state {found.group(1)}, which its predecessor "
        f"{job['before']} converged and this run should have copied in: the state did "
        "not reproduce by its fingerprint (prompt 79's stop rule)"
    )


def finish(queue: dict, running: Running, code: int | None) -> None:
    """Record a process's end and classify it."""
    job = running.job
    record = queue["jobs"][job["name"]]
    attempt = record["attempts"][-1]
    running.sample()
    running.close()
    status, reason = classify(job, code)
    attempt.update(
        {
            "ended": now(),
            "exit": code,
            "wall_seconds": round(time.time() - running.started, 1),
            "peak_working_set": running.peak or None,
            "peak_working_set_first_10_min": running.peak_first_10_min,
            "outcome": status,
            "reason": reason,
        }
    )
    record["status"] = status
    if (
        status == "interrupted"
        and len(record["attempts"]) >= queue["settings"]["max_attempts"]
    ):
        record["status"] = "failed"
        reason = f"gave up after {len(record['attempts'])} attempts; last: {reason}"
        attempt["reason"] = reason
    peak_text = f"{running.peak / 2**30:.2f} GB" if running.peak else "unknown"
    log(
        f"finish {job['name']} ({job['group']} {job['grid']}) exit {code} "
        f"{record['status']}: {reason}; wall {attempt['wall_seconds']:.0f} s; "
        f"peak working set {peak_text}; {memory_text()}"
    )


def work(selected: list[dict]) -> int:
    """Work the queue to its end; return 0 when nothing failed."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(PYTHON), str(HERE / "configs79.py")],
        check=True,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
    )
    queue = load_queue()
    active: dict[str, Running] = {}
    for job in selected:
        record = queue["jobs"].setdefault(
            job["name"],
            {"status": "pending", "attempts": [], "estimate": job["estimate"]},
        )
        if record["status"] == "running":
            adopted = adopt(job, record)
            if adopted is not None:
                active[job["name"]] = adopted
                log(
                    f"adopt {job['name']} pid {adopted.pid} (left running by an earlier runner)"
                )
            else:
                status, reason = classify(job, None)
                record["status"] = status
                record["attempts"][-1].update(
                    {"ended": None, "outcome": status, "reason": reason}
                )
                log(
                    f"found {job['name']} ended while no runner watched: {status}, {reason}"
                )
    save_queue(queue)
    log(f"runner start, pid {os.getpid()}, {len(selected)} jobs; {memory_text()}")
    last_launch = 0.0
    last_memory_log = 0.0
    while True:
        settings = load_queue()["settings"]
        queue["settings"] = settings
        for name, running in list(active.items()):
            peak = running.sample()
            if (
                peak
                and running.peak_first_10_min is None
                and time.time() - running.started >= 600
            ):
                running.peak_first_10_min = peak
                queue["jobs"][name]["attempts"][-1]["peak_working_set_first_10_min"] = (
                    peak
                )
            queue["jobs"][name]["attempts"][-1]["peak_working_set"] = peak or None
            code = exit_code(running.handle)
            if code is not None:
                del active[name]
                finish(queue, running, code)
                continue
            if settings["stopped"] is None:
                reason = stop_reason(running)
                if reason is not None:
                    settings["stopped"] = reason
                    queue["settings"] = settings
                    tmp = QUEUE.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(queue, indent=2), encoding="utf-8")
                    os.replace(tmp, QUEUE)
                    log(f"STOP: {reason}")
        launchable = []
        for job in selected:
            record = queue["jobs"][job["name"]]
            if (
                record["status"] not in ("pending", "interrupted")
                or job["name"] in active
            ):
                continue
            depends = job["depends"]
            if depends is not None:
                state = queue["jobs"].get(depends, {}).get("status")
                if state == "failed":
                    record["status"] = "failed"
                    record["reason"] = (
                        f"not run: {depends} failed, and its flow state is {depends}'s"
                    )
                    log(f"skip {job['name']}: {record['reason']}")
                    continue
                if state != "done":
                    continue
            launchable.append(job)
        if (
            settings["stopped"] is None
            and launchable
            and time.time() - last_launch >= LAUNCH_SPACING
        ):
            fine_running = sum(1 for r in active.values() if r.job["fine"])
            avail, total = memory()
            for job in launchable:
                if len(active) >= settings["max_processes"]:
                    break
                if job["fine"] and fine_running >= settings["max_fine"]:
                    continue
                if avail < settings["memory_floor"] * total:
                    if time.time() - last_memory_log >= MEMORY_LOG_SECONDS:
                        log(
                            f"wait: {memory_text()}, below the floor of {settings['memory_floor']:.0%}"
                        )
                        last_memory_log = time.time()
                    break
                record = queue["jobs"][job["name"]]
                mode = "new"
                if record["status"] == "interrupted" or job["out"].exists():
                    if march_checkpoint(job):
                        mode = "restart"
                    else:
                        moved = move_aborted(job)
                        if moved:
                            log(
                                f"aborted {job['name']}: moved {', '.join(moved)} to {ABORTED.name}/"
                            )
                pid, created = launch(job, mode, len(active))
                started = time.time()
                record["status"] = "running"
                record["attempts"].append(
                    {
                        "mode": mode,
                        "started": now(),
                        "pid": pid,
                        "created": created,
                        "concurrent": len(active),
                        "avail_bytes": avail,
                        "total_bytes": total,
                    }
                )
                handle = open_process(pid)
                active[job["name"]] = Running(job, pid, handle or 0, started)
                log(
                    f"start {job['name']} ({job['group']} {job['grid']}) {mode}, pid {pid}, "
                    f"{len(active) - 1} beside it; {memory_text()}"
                )
                last_launch = started
                break
        save_queue(queue)
        if not active and (settings["stopped"] is not None or not launchable):
            break
        time.sleep(POLL_SECONDS)
    counts: dict[str, int] = {}
    for job in selected:
        status = queue["jobs"][job["name"]]["status"]
        counts[status] = counts.get(status, 0) + 1
    log(f"runner end: {counts}; stopped: {queue['settings']['stopped']}")
    return (
        0
        if counts.get("failed", 0) == 0 and queue["settings"]["stopped"] is None
        else 1
    )


def status() -> int:
    """Each running job's progress from its checkpoint, and the queue's estimate."""
    from src.io_manager import RunFile

    queue = load_queue()
    jobs = {job["name"]: job for job in ordered_jobs()}
    remaining = {"fine": 0.0, "coarse": 0.0}
    for name, job in jobs.items():
        record = queue["jobs"].get(name, {"status": "pending", "attempts": []})
        kind = "fine" if job["fine"] else "coarse"
        if record["status"] == "pending":
            remaining[kind] += job["estimate"]
        if record["status"] != "running":
            continue
        attempt = record["attempts"][-1]
        elapsed = (
            datetime.now() - datetime.fromisoformat(attempt["started"])
        ).total_seconds()
        text = (
            log_path(job).read_text(encoding="utf-8", errors="replace")
            if log_path(job).exists()
            else ""
        )
        planned = PLANNED.findall(text)
        try:
            checkpoint = RunFile(job["out"]).read_checkpoint()
            step, t = checkpoint.step, checkpoint.t
        except (KeyError, OSError, ValueError):
            step, t = None, None
        peak = attempt.get("peak_working_set")
        peak_text = f", peak {peak / 2**30:.2f} GB" if peak else ""
        if step is None or not planned:
            print(f"{name}: {elapsed / 60:.1f} min, no march checkpoint yet{peak_text}")
            remaining[kind] += max(job["estimate"] - elapsed, 0.0)
            continue
        total = int(planned[-1])
        rate = step / elapsed
        left = (total - step) / rate if rate > 0 else float("inf")
        remaining[kind] += left
        print(
            f"{name}: step {step} of {total} (t = {t:.1f} s) after {elapsed / 60:.1f} min, "
            f"{rate:.2f} steps/s, about {left / 60:.0f} min left{peak_text}"
        )
    counts: dict[str, int] = {}
    for record in queue["jobs"].values():
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    fine_slots = queue["settings"]["max_fine"]
    hours = (
        max(
            remaining["fine"] / fine_slots,
            (remaining["fine"] + remaining["coarse"]) / 3,
        )
        / 3600
    )
    print(f"queue: {counts}; settings {queue['settings']}")
    print(
        f"left by the old walls and the running rates: 320x120 {remaining['fine'] / 3600:.1f} "
        f"process-hours, the rest {remaining['coarse'] / 3600:.1f}; about {hours:.1f} h of wall clock"
    )
    print(memory_text())
    return 0


def main() -> int:
    """Work the queue, print it, report progress, or change a setting."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--set", metavar="KEY=VALUE")
    parser.add_argument("--only", nargs="+", metavar="NAME")
    args = parser.parse_args()
    if args.set:
        return set_setting(args.set)
    if args.status:
        return status()
    selected = ordered_jobs()
    if args.only:
        unknown = set(args.only) - {j["name"] for j in selected}
        if unknown:
            parser.error(f"unknown job(s): {sorted(unknown)}")
        selected = [j for j in selected if j["name"] in args.only]
    if args.dry_run:
        for job in selected:
            after = f" after {job['depends']}" if job["depends"] else ""
            print(
                f"{job['group']} {job['grid']:>8} {job['estimate']:8.0f} s  {job['name']}{after}"
            )
        fine = sum(j["estimate"] for j in selected if j["fine"])
        rest = sum(j["estimate"] for j in selected if not j["fine"])
        print(f"320x120 {fine / 3600:.1f} process-hours, the rest {rest / 3600:.1f}")
        return 0
    return work(selected)


if __name__ == "__main__":
    sys.exit(main())
