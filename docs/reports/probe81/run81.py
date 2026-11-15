"""Work prompt 81's queue of runs through scripts/run_scenario.py, resumably.

    python docs/reports/probe81/run81.py             # work the queue to its end
    python docs/reports/probe81/run81.py --dry-run   # print the queue and estimates
    python docs/reports/probe81/run81.py --status    # progress of the running jobs
    python docs/reports/probe81/run81.py --set max_fine=1

Prompt 81 re-runs prompt 79's five 320x120 scenarios and its three 80x30
placement runs on the code that counts cumulatively (ADR-015 C, amended
2026-11-14), each from prompt 79's own configuration file and with its flow
states copied in from prompt 79's run file (``--flows-from``), so only the
march is new. This is prompt 79's runner (``docs/reports/probe79/run79.py``)
with three changes:

- The caps are refused, not trusted (review 79 S7). ``max_processes`` may not
  exceed MAX_PROCESSES (3), ``max_fine`` may not exceed MAX_FINE (2) or
  ``max_processes``, ``memory_floor`` may not fall below MEMORY_FLOOR (25%)
  or reach 1, ``max_attempts`` is a whole number of at least one, and no
  setting may be a bool. ``--set`` refuses a value outside those limits and
  writes nothing; a queue file edited by hand past them stops the runner at
  start, and while it runs keeps the last valid settings and logs the refusal.
- The runner holds the machine awake itself with
  ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED |
  ES_AWAYMODE_REQUIRED)`` from its start to its end, where prompt 79 used a
  separate PowerShell process. Modern Standby can still suspend the machine
  (prompt 79's did); ``standby81.ps1`` reads those intervals from the System
  event log for the report, which takes them out of the walls.
- The jobs are this prompt's (``jobs``), run by ``scripts/run_scenario.py``
  itself: prompt 79's sweeps recorder is not needed. Every flow state of
  every job is converged in the run it is copied from, so a ``flow state
  ...: solving`` line in any job's log means a state did not reproduce by its
  fingerprint, and the queue then starts nothing more (prompt 79's stop rule).

The other rules are prompt 79's, which are Alex's: one BLAS thread each;
nothing started while the machine's available physical memory is below the
floor of its total, read from ``GlobalMemoryStatusEx`` before every start and
logged with it; starts at least LAUNCH_SPACING apart; the 320x120 jobs first,
longest first by prompt 79's march walls. A job is done when its process
exited 0 and its summary reports a wall time; failed when the runner ended it
with its own ``error:`` line; otherwise interrupted, and started again, with
``--restart`` when its run file holds a march checkpoint and from the start
(its partial files moved to ``aborted/``) when not, at most ``max_attempts``
times. A job left running by an earlier runner is adopted.

The queue's state is ``results/builder81/queue.json``; one line per start and
finish goes to ``results/builder81/queue.log``; each job's summary (stdout)
and log (stderr) go beside its run file.
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

RESULTS = ROOT / "results" / "builder81"
BEFORE = ROOT / "results" / "builder79"
QUEUE = RESULTS / "queue.json"
LOG = RESULTS / "queue.log"
ABORTED = RESULTS / "aborted"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER = ROOT / "scripts" / "run_scenario.py"
ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
POLL_SECONDS = 15
LAUNCH_SPACING = 60.0
MEMORY_LOG_SECONDS = 300.0
# Alex's limits (prompts 79 and 81): never above these, whatever the queue says.
MAX_PROCESSES = 3
MAX_FINE = 2
MEMORY_FLOOR = 0.25
DEFAULT_SETTINGS: dict[str, object] = {
    "max_processes": MAX_PROCESSES,
    "max_fine": MAX_FINE,
    "memory_floor": MEMORY_FLOOR,
    "max_attempts": 3,
    "stopped": None,
}
PRODUCT = ("clean_out", "door_leak", "equipment_dust", "filter_breach", "seal_leak")
PLACEMENT = ("door", "breach", "dust")
FINE = "320x120"
COARSE = "80x30"
PLANNED = re.compile(r"(\d+) steps planned")
SOLVING = re.compile(r"flow state (\w+): solving")
OLD_WALL = re.compile(r"^wall time: ([\d.]+) s", re.MULTILINE)
OLD_SOLVED = re.compile(
    r"^  \w+: (?:rest|warm from \w+|resumed), \d+ outer, \w+, ([\d.]+) s$", re.MULTILINE
)

# -- Windows process, memory and power calls (ctypes, no new dependency) -------

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_VM_READ = 0x0010
SYNCHRONIZE = 0x00100000
STILL_ACTIVE = 259
CREATE_NO_WINDOW = 0x08000000
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


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
KERNEL32.SetThreadExecutionState.restype = wintypes.DWORD
KERNEL32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
TH32CS_SNAPPROCESS = 0x00000002
KERNEL32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
KERNEL32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
KERNEL32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
KERNEL32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]


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


def keep_awake(on: bool) -> int:
    """Hold the system awake (or release it) from this thread; return the previous state."""
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED if on else 0)
    return int(KERNEL32.SetThreadExecutionState(flags))


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


# -- The settings, refused outside Alex's limits --------------------------------


def _whole(value: object) -> bool:
    """Whether VALUE is an int and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def settings_error(settings: dict) -> str | None:
    """Why a settings mapping breaks Alex's limits, or None when it keeps them."""
    procs, fine = settings.get("max_processes"), settings.get("max_fine")
    floor, attempts = settings.get("memory_floor"), settings.get("max_attempts")
    if not (_whole(procs) and 1 <= procs <= MAX_PROCESSES):
        return (
            f"max_processes must be a whole number 1 to {MAX_PROCESSES}, got {procs!r}"
        )
    if not (_whole(fine) and 0 <= fine <= min(MAX_FINE, procs)):
        return (
            f"max_fine must be a whole number 0 to {MAX_FINE} and at most "
            f"max_processes ({procs}), got {fine!r}"
        )
    if (
        isinstance(floor, bool)
        or not isinstance(floor, (int, float))
        or not MEMORY_FLOOR <= floor < 1.0
    ):
        return f"memory_floor must be a number from {MEMORY_FLOOR} to below 1, got {floor!r}"
    if not (_whole(attempts) and attempts >= 1):
        return f"max_attempts must be a whole number of at least 1, got {attempts!r}"
    stopped = settings.get("stopped")
    if stopped is not None and not isinstance(stopped, str):
        return f"stopped must be null or a reason, got {stopped!r}"
    return None


# -- The queue file ------------------------------------------------------------


def now() -> str:
    """The local time, as every record states it."""
    return datetime.now().isoformat(timespec="seconds")


def log(line: str) -> None:
    """Append one line to the queue log and echo it."""
    text = f"{now()} {line}"
    RESULTS.mkdir(parents=True, exist_ok=True)
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


def write_queue(queue: dict) -> None:
    """Write the queue file atomically."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    os.replace(tmp, QUEUE)


def save_queue(queue: dict) -> None:
    """Write the queue file, keeping valid settings changed on disk meanwhile."""
    if QUEUE.exists():
        on_disk = json.loads(QUEUE.read_text(encoding="utf-8")).get("settings")
        if isinstance(on_disk, dict) and settings_error(on_disk) is None:
            queue["settings"] = on_disk
    write_queue(queue)


def set_setting(assignment: str) -> int:
    """Change one setting of the queue file: ``key=value``, the value as JSON."""
    key, _, value = assignment.partition("=")
    if key not in DEFAULT_SETTINGS:
        print(f"unknown setting {key!r}; one of {sorted(DEFAULT_SETTINGS)}")
        return 2
    queue = load_queue()
    try:
        queue["settings"][key] = json.loads(value)
    except json.JSONDecodeError as error:
        print(f"refused: {value!r} is not JSON ({error})")
        return 2
    error = settings_error(queue["settings"])
    if error is not None:
        print(f"refused: {error}; the queue file is unchanged")
        return 2
    write_queue(queue)
    log(f"setting {key} = {queue['settings'][key]!r} (by --set)")
    return 0


# -- Jobs ----------------------------------------------------------------------


def jobs() -> list[dict]:
    """Every job: name, group, grid, configuration, run file, flows-from, estimate.

    The configuration is prompt 79's own file, copied to ``configs/<group>/``
    before the first start; the flows come from prompt 79's run file; the
    estimate is prompt 79's wall less its flow solves (none for these runs,
    which copied their flows in as well).
    """
    rows = []
    for group, names, grid in (
        ("product", PRODUCT, FINE),
        ("placement", PLACEMENT, COARSE),
    ):
        for name in names:
            before = BEFORE / group / f"{name}.h5"
            rows.append(
                {
                    "name": name,
                    "group": group,
                    "grid": grid,
                    "fine": grid == FINE,
                    "source_config": BEFORE
                    / "configs"
                    / group
                    / f"scenario_{name}.yaml",
                    "config": RESULTS / "configs" / group / f"scenario_{name}.yaml",
                    "before": before,
                    "flows_from": before,
                    "out": RESULTS / group / f"{name}.h5",
                    "estimate": old_estimate(before),
                }
            )
    return sorted(rows, key=lambda r: (0 if r["fine"] else 1, -r["estimate"]))


def old_estimate(before: Path) -> float:
    """Seconds a job should take: prompt 79's wall less the flow solves it made."""
    old = before.with_suffix(".txt")
    text = old.read_text(encoding="utf-8", errors="replace") if old.exists() else ""
    wall = OLD_WALL.search(text)
    if wall is None:
        return 0.0
    return float(wall.group(1)) - sum(float(s) for s in OLD_SOLVED.findall(text))


def copy_configs(selected: list[dict]) -> None:
    """Copy prompt 79's configuration files in, byte for byte, refusing a changed copy."""
    for job in selected:
        source, target = job["source_config"], job["config"]
        data = source.read_bytes()
        if target.exists():
            if target.read_bytes() != data:
                raise SystemExit(f"{target} differs from {source}; not overwritten")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


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
        command += ["--out", str(job["out"]), "--flows-from", str(job["flows_from"])]
    command += ["--concurrent", str(beside)]
    job["out"].parent.mkdir(parents=True, exist_ok=True)
    # Handed to Popen, which holds them for the process lifetime; closed after.
    out = open(summary_path(job), "w", encoding="utf-8")  # noqa: SIM115 -- see above
    err = open(log_path(job), "a", encoding="utf-8")  # noqa: SIM115 -- see above
    err.write(f"--- run81 {mode} at {now()}: {' '.join(command)}\n")
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
    """Why the queue must stop: a job solving a state its source run converged."""
    path = log_path(running.job)
    if not path.exists():
        return None
    found = SOLVING.search(path.read_text(encoding="utf-8", errors="replace"))
    if found is None:
        return None
    return (
        f"{running.job['name']} is solving flow state {found.group(1)}, which "
        f"{running.job['flows_from']} holds converged and this run should have copied "
        "in: the state did not reproduce by its fingerprint (prompt 79's stop rule)"
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


def current_settings(queue: dict, refused: list[str]) -> dict:
    """The settings on disk when they keep Alex's limits, else the last valid ones.

    REFUSED holds the last refusal logged, so one bad edit is logged once.
    """
    on_disk = load_queue()["settings"]
    error = settings_error(on_disk)
    if error is None:
        refused.clear()
        return on_disk
    if refused != [error]:
        log(f"refused the queue file's settings ({error}); keeping {queue['settings']}")
        refused[:] = [error]
    return queue["settings"]


def work(selected: list[dict]) -> int:
    """Work the queue to its end; return 0 when nothing failed."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    queue = load_queue()
    error = settings_error(queue["settings"])
    if error is not None:
        log(f"runner refused to start: {error}")
        return 2
    copy_configs(selected)
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
    previous = keep_awake(True)
    log(
        f"runner start, pid {os.getpid()}, {len(selected)} jobs; holding the system "
        f"awake (previous state {previous:#x}); {memory_text()}"
    )
    last_launch = 0.0
    last_memory_log = 0.0
    refused: list[str] = []
    try:
        while True:
            settings = current_settings(queue, refused)
            queue["settings"] = settings
            for name, running in list(active.items()):
                peak = running.sample()
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
                        write_queue(queue)
                        log(f"STOP: {reason}")
            launchable = [
                job
                for job in selected
                if queue["jobs"][job["name"]]["status"] in ("pending", "interrupted")
                and job["name"] not in active
            ]
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
                                f"wait: {memory_text()}, below the floor of "
                                f"{settings['memory_floor']:.0%}"
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
                                    f"aborted {job['name']}: moved {', '.join(moved)} "
                                    f"to {ABORTED.name}/"
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
                        f"start {job['name']} ({job['group']} {job['grid']}) {mode}, "
                        f"pid {pid}, {len(active) - 1} beside it; {memory_text()}"
                    )
                    last_launch = started
                    break
            save_queue(queue)
            if not active and (settings["stopped"] is not None or not launchable):
                break
            time.sleep(POLL_SECONDS)
    finally:
        keep_awake(False)
    counts: dict[str, int] = {}
    for job in selected:
        status = queue["jobs"][job["name"]]["status"]
        counts[status] = counts.get(status, 0) + 1
    log(
        f"runner end: {counts}; stopped: {queue['settings']['stopped']}; released the system"
    )
    return (
        0
        if counts.get("failed", 0) == 0 and queue["settings"]["stopped"] is None
        else 1
    )


def status() -> int:
    """Each running job's progress from its checkpoint, and the queue's estimate."""
    from src.io_manager import RunFile

    queue = load_queue()
    for job in jobs():
        record = queue["jobs"].get(job["name"], {"status": "pending", "attempts": []})
        if record["status"] != "running":
            print(f"{job['name']}: {record['status']}")
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
        if step is None or not planned:
            print(f"{job['name']}: {elapsed / 60:.1f} min, no march checkpoint yet")
            continue
        total = int(planned[-1])
        rate = step / elapsed
        left = (total - step) / rate if rate > 0 else float("inf")
        print(
            f"{job['name']}: step {step} of {total} (t = {t:.1f} s) after "
            f"{elapsed / 60:.1f} min, about {left / 60:.0f} min left"
        )
    print(f"settings {queue['settings']}; {memory_text()}")
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
    selected = jobs()
    if args.only:
        unknown = set(args.only) - {j["name"] for j in selected}
        if unknown:
            parser.error(f"unknown job(s): {sorted(unknown)}")
        selected = [j for j in selected if j["name"] in args.only]
    if args.dry_run:
        for job in selected:
            print(
                f"{job['group']:>9} {job['grid']:>8} {job['estimate']:8.0f} s  {job['name']}"
            )
        fine = sum(j["estimate"] for j in selected if j["fine"])
        rest = sum(j["estimate"] for j in selected if not j["fine"])
        print(f"320x120 {fine / 3600:.2f} process-hours, the rest {rest / 3600:.2f}")
        return 0
    return work(selected)


if __name__ == "__main__":
    sys.exit(main())
