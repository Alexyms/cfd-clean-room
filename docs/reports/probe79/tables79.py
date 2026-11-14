"""The tables of prompt 79, the product grid again on the budgeted solve, from its records.

    python docs/reports/probe79/tables79.py            # write the tables, splice the report
    python docs/reports/probe79/tables79.py --check    # regenerate and compare

Reads ``results/builder79/`` (untracked, this machine): the queue file
``queue.json`` and its log, each job's run file, summary, log and
``RUN.sweeps.<pid>.npz`` (record79.py), under ``product/``, ``sweep/`` and
``placement/``; and the runs each job replaces, ``results/builder76/``,
``results/builder69/``, ``results/builder72/`` and
``results/builder75/runs/``.

It writes four files beside it:

- ``rescored69.md``, ``rescored72.md``, ``rescored76.md``: the tables of
  ``tables69.py``, ``tables72.py`` and ``tables76.py``, made by their own
  code with their run directories pointed at this pass's runs (the
  re-score with main's code and no change to any rule). Their committed
  outputs are not touched. Their prose still names the directories of the
  runs they were written for; the header of each file says what it read.
- ``changes79.md``: every table of each committed output against its
  re-score, cell by cell, the old value and the new side by side, and every
  line of text between the tables that changed.
- ``tables79.md``: this pass's own tables, each also copied into
  ``docs/reports/rerun79.md`` between ``<!-- tables79 NAME begin/end -->``.

``--check`` regenerates all of them and compares, the report's copies too.

The re-scores read the runs where each script expects them, so before they
run the configurations of each group are copied to ``<group>/configs/`` and
the logs to ``<group>/logs/``, and the sweep's queue record is written as
``sweep/runs72.json`` from ``queue.json`` (``prepare``).
"""

import argparse
import importlib.util
import json
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import configs79  # noqa: E402 -- follows sys.path.insert

from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.io_manager import RunFile  # noqa: E402 -- follows sys.path.insert
from src.mesh import SOLID, Mesh  # noqa: E402 -- follows sys.path.insert
from src.monitor import (  # noqa: E402 -- follows sys.path.insert
    contaminating_events,
    sensor_cells,
)
from src.placement import (  # noqa: E402 -- follows sys.path.insert
    FRAME_RESOLVED,
    RunArrival,
    arrival_from_frames,
    arrival_from_run,
    candidates,
    detection_units,
    exclude_units,
    given_layout,
    rank,
    search,
)

RESULTS = configs79.RESULTS
QUEUE = RESULTS / "queue.json"
QUEUE_LOG = RESULTS / "queue.log"
# The machine's Modern Standby intervals (standby79.ps1, from the System event log);
# a run's walls are given less its overlap with them.
STANDBY = RESULTS / "standby.json"
TABLES = HERE / "tables79.md"
CHANGES = HERE / "changes79.md"
RESCORED = {
    "69": HERE / "rescored69.md",
    "72": HERE / "rescored72.md",
    "76": HERE / "rescored76.md",
}
COMMITTED = {
    "69": ROOT / "docs" / "reports" / "probe69" / "tables69.md",
    "72": ROOT / "docs" / "reports" / "probe72" / "tables72.md",
    "76": ROOT / "docs" / "reports" / "probe76" / "tables76.md",
}
REPORT = ROOT / "docs" / "reports" / "rerun79.md"
GATE_69 = ROOT / "results" / "builder69"
CLOSURE = 1.0e-12
SIZES_UM = ("0.1", "0.3", "0.5", "1", "5")
FIELDS = ("u", "v", "p", "k", "eps", "nu_t")
SOLVED = re.compile(
    r"^  ([0-9a-f]{12}): (rest|warm from \w+|resumed), ([\d,]+) outer, (\w+), ([\d.]+) s$",
    re.MULTILINE,
)
KEPT = re.compile(
    r"^  ([0-9a-f]{12}): (kept|run file), ([\d,]+) outer, (\w+), ([\d.]+) s$",
    re.MULTILINE,
)
WALL = re.compile(r"^wall time: ([\d.]+) s", re.MULTILINE)
CAPPED = re.compile(r"flow state (\w+) stopped by (\w+) after (\d+) outer iterations")
SIMPLE = re.compile(r"^SIMPLE iter\s+(\d+): residual = (\S+)$", re.MULTILINE)
GROUP_NAMES = {"P": "product grid", "S": "sweep", "L": "placement, 80x30"}
MARKERS = (
    "queue",
    "flow_states",
    "capped",
    "closure",
    "leftover",
    "dust",
    "sweeps",
    "wall",
    "memory",
    "queue_log",
    "latency",
    "latency_changes",
    "cleanout",
    "placement80",
    "contour",
    "rescored",
    "gates",
    "predictions",
)


def fmt(x: float, digits: int = 4) -> str:
    """A float in scientific notation."""
    return f"{x:.{digits}e}"


def yes(flag: bool) -> str:
    """yes or NO."""
    return "yes" if flag else "NO"


def rel(path: Path) -> str:
    """A path relative to the repository, in forward slashes."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# The records
# ---------------------------------------------------------------------------


def load_queue() -> dict:
    """The queue file."""
    return json.loads(QUEUE.read_text(encoding="utf-8"))


def jobs() -> list[dict]:
    """configs79's jobs, each with its queue record and its summary text."""
    queue = load_queue()["jobs"]
    rows = configs79.jobs()
    for job in rows:
        job["record"] = queue.get(job["name"], {"status": "not queued", "attempts": []})
        job["summary"] = read_text(job["out"].with_suffix(".txt"))
        job["log"] = read_text(job["out"].with_suffix(".log"))
        job["fine"] = job["grid"] == configs79.FINE
        unwatched_end(job)
    return rows


def unwatched_end(job: dict) -> None:
    """Date an attempt that ended while no runner watched by its summary's file time.

    The runner died at about 08:13 on 2026-11-13 with three jobs still running; they
    finished and the restarted runner found them done but could not know when. The
    summary is the run's last write, so its modification time stands in for the end.
    """
    attempts = job["record"]["attempts"]
    summary = job["out"].with_suffix(".txt")
    if not attempts or attempts[-1].get("ended") or not summary.exists():
        return
    if job["record"]["status"] != "done":
        return
    last = attempts[-1]
    ended = datetime.fromtimestamp(summary.stat().st_mtime).replace(microsecond=0)
    last["ended"] = ended.isoformat()
    last["wall_seconds"] = (
        ended - datetime.fromisoformat(last["started"])
    ).total_seconds()
    last["exit"] = "-"
    last["reason"] = (
        f"{last.get('reason', 'summary written')}; ended while no runner watched, end taken from the summary's file time"
    )


def read_text(path: Path | None) -> str:
    """A text file's contents, empty when absent."""
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def old_summary(job: dict) -> str:
    """The summary of the run a job replaces."""
    return "" if job["before"] is None else read_text(job["before"].with_suffix(".txt"))


def finished(job: dict) -> bool:
    """Whether a job finished with its summary written."""
    return job["record"]["status"] == "done" and WALL.search(job["summary"]) is not None


def march_seconds(summary: str) -> float | None:
    """A summary's wall time less the flow solves the run made itself."""
    wall = WALL.search(summary)
    if wall is None:
        return None
    return float(wall.group(1)) - sum(float(m[4]) for m in SOLVED.findall(summary))


def config_of(path: Path) -> SimConfig:
    """A run file's configuration, the conservation keys supplied for a file older than them."""
    mapping = RunFile(path).read_config().mapping
    transport = mapping.get("transport", {})
    if "conservation_limit" not in transport:
        base = SimConfig(configs79.CONFIGS / "product" / "scenario_clean_out.yaml")
        mapping["transport"] = {
            "conservation_limit": base.transport.conservation_limit,
            "leftover_share": base.transport.leftover_share,
        } | transport
    return SimConfig.from_dict(mapping)


def last_error(job: dict) -> str:
    """The job's last ``error:`` line, empty when it has none."""
    errors = [ln for ln in job["log"].splitlines() if ln.startswith("error:")]
    return errors[-1] if errors else ""


def standby_intervals() -> list[tuple[datetime, datetime]]:
    """The standby intervals standby79.ps1 recorded."""
    if not STANDBY.exists():
        return []
    rows = json.loads(STANDBY.read_text(encoding="utf-8-sig"))
    return [
        (datetime.fromisoformat(r["entered"]), datetime.fromisoformat(r["exited"]))
        for r in rows
    ]


def standby_seconds(job: dict) -> float:
    """Seconds of standby inside the job's attempts."""
    total = 0.0
    for attempt in job["record"]["attempts"]:
        if not attempt.get("ended"):
            continue
        start = datetime.fromisoformat(attempt["started"])
        end = datetime.fromisoformat(attempt["ended"])
        for entered, exited in standby_intervals():
            overlap = (min(end, exited) - max(start, entered)).total_seconds()
            total += max(overlap, 0.0)
    return total


def max_step(summary: str) -> float:
    """The largest step a summary's intervals name."""
    steps = [float(s) for s in re.findall(r"steps of ([\d.]+) s", summary)]
    return max(steps) if steps else math.nan


# ---------------------------------------------------------------------------
# The queue, the flow states, the cost
# ---------------------------------------------------------------------------


def queue_table(rows: list[dict]) -> list[str]:
    """Every job's outcome and its attempts."""
    lines = [
        "### Q1. The queue: every job, its outcome and its attempts (run79.py, queue.json)",
        "",
        "Times are the machine's local time. A restart continues from the run file's last march checkpoint; an aborted attempt's partial files are under results/builder79/aborted/. The machine was in Modern Standby for part of the night (results/builder79/standby.json); the process wall is given with that time and without it.",
        "",
        "| Job | Group | Grid | Outcome | Attempts | Started | Ended | Process wall (s) | Less standby (s) | Exit | Reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for job in rows:
        record = job["record"]
        attempts = record["attempts"]
        if not attempts:
            lines.append(
                f"| {job['name']} | {job['group']} | {job['grid']} | {record['status']} | 0 | - | - | - | - | - | "
                f"{record.get('reason', '-')} |"
            )
            continue
        modes = ", ".join(a["mode"] for a in attempts)
        wall = sum(a.get("wall_seconds") or 0.0 for a in attempts)
        last = attempts[-1]
        lines.append(
            f"| {job['name']} | {job['group']} | {job['grid']} | {record['status']} | {len(attempts)} ({modes}) | "
            f"{attempts[0]['started']} | {last.get('ended') or '-'} | {wall:.0f} | {wall - standby_seconds(job):.0f} | {last.get('exit', '-')} | "
            f"{last.get('reason', '-')} |"
        )
    starts = [a["started"] for j in rows for a in j["record"]["attempts"]]
    ends = [a["ended"] for j in rows for a in j["record"]["attempts"] if a.get("ended")]
    if starts and ends:
        first, last = (
            datetime.fromisoformat(min(starts)),
            datetime.fromisoformat(max(ends)),
        )
        span = (last - first).total_seconds()
        asleep = sum(
            max((min(last, b) - max(first, a)).total_seconds(), 0.0)
            for a, b in standby_intervals()
        )
        lines += [
            "",
            f"The queue ran from {min(starts)} to {max(ends)}, {span / 3600:.2f} hours of wall clock, "
            f"of which the machine spent {asleep / 3600:.2f} in Modern Standby with every run suspended: "
            f"{(span - asleep) / 3600:.2f} hours awake.",
        ]
    return lines


def queue_log_table() -> list[str]:
    """The runner's log, verbatim."""
    log = read_text(QUEUE_LOG).rstrip().splitlines()
    return [
        "### Q2. The runner's log (results/builder79/queue.log, verbatim)",
        "",
        "One line per start, finish, skip and wait, with the time, the run, the exit status, the available memory and the wall. The 'found' lines are the restarted runner's: the three runs it names ended while no runner watched (section 10).",
        "",
        "```text",
        *log,
        "```",
    ]


def same_state(a: object, b: object) -> bool:
    """Whether two flow states are equal bit for bit in every field."""
    return all(
        np.ascontiguousarray(getattr(a, f)).tobytes()
        == np.ascontiguousarray(getattr(b, f)).tobytes()
        for f in FIELDS
    )


def flow_state_table(rows: list[dict]) -> tuple[list[str], dict[str, bool]]:
    """Every converged state a finished job holds, against the state it replaces."""
    lines = [
        "### F1. The flow states of the finished runs against the runs they replace (prediction (b))",
        "",
        "Every state a finished run holds, read back from its run file and compared field by field with the replaced run's state under the same fingerprint (u, v, p, k, eps, nu_t, every byte). A state the run solved itself is marked; its counterpart is the state the replaced run solved.",
        "",
        "| Run | Grid | State | How this run came by it | Against | Bitwise | Outer iterations |",
        "|---|---|---|---|---|---|---|",
    ]
    verdicts: dict[str, bool] = {}
    for job in rows:
        if not finished(job) or job["before"] is None:
            continue
        mine = RunFile(job["out"])
        theirs = RunFile(job["before"])
        held = set(theirs.flow_fingerprints())
        solved = {m[0] for m in SOLVED.findall(job["summary"])}
        for fp in mine.flow_fingerprints():
            how = "solved" if fp[:12] in solved else "kept"
            if fp not in held:
                lines.append(
                    f"| {job['name']} | {job['grid']} | {fp[:12]} | {how} | not held by {rel(job['before'])} | NO | - |"
                )
                verdicts[f"{job['name']}:{fp[:12]}"] = False
                continue
            a = mine.read_flow_state(fp).state
            b = theirs.read_flow_state(fp).state
            same = same_state(a, b)
            verdicts[f"{job['name']}:{fp[:12]}"] = same
            lines.append(
                f"| {job['name']} | {job['grid']} | {fp[:12]} | {how} | {rel(job['before'])} | {yes(same)} | "
                f"{a.outer:,} against {b.outer:,} |"
            )
    return lines, verdicts


def capped_table(rows: list[dict]) -> tuple[list[str], list[bool]]:
    """The states that did not converge before, against prompt 72's."""
    lines = [
        "### F2. The two flow states that did not converge in prompt 72, solved again (prediction (b))",
        "",
        "Each capped solve leaves its steady checkpoint in the run file (time_integration). The checkpoint of this pass's solve is compared with prompt 72's: the state's fields, the residual history to the last outer iteration, and the stop the runner reported.",
        "",
        "| Run | Grid | State | Stop, this pass | Stop, prompt 72 | Outer, this pass | Prompt 72 | Fields bitwise | Residual history bitwise | Last residual |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    verdicts = []
    for job in rows:
        if job["before"] is None or job["record"]["status"] != "failed":
            continue
        new = CAPPED.search(job["log"])
        old = CAPPED.search(
            read_text(job["before"].parent / "logs" / f"{job['name']}.log")
        )
        if new is None:
            continue
        try:
            fp_a, a = RunFile(job["out"]).read_steady_checkpoint()
            fp_b, b = RunFile(job["before"]).read_steady_checkpoint()
        except (KeyError, OSError) as err:
            lines.append(
                f"| {job['name']} | {job['grid']} | {new.group(1)[:12]} | {new.group(2)} | - | {new.group(3)} | - | - | - | {err} |"
            )
            verdicts.append(False)
            continue
        fields = fp_a == fp_b and same_state(a.state, b.state)
        history = (
            np.array(a.residual_history).tobytes()
            == np.array(b.residual_history).tobytes()
        )
        old_stop = f"{old.group(2)} at {old.group(3)}" if old else "-"
        verdicts.append(
            fields and history and old is not None and old.groups() == new.groups()
        )
        lines.append(
            f"| {job['name']} | {job['grid']} | {fp_a[:12]} | {new.group(2)} | {old_stop} | {a.state.outer:,} | "
            f"{b.state.outer:,} | {yes(fields)} | {yes(history)} | {a.residual_history[-1]:.6e} |"
        )
    not_run = [
        j["name"]
        for j in rows
        if j["record"]["status"] == "failed" and j["before"] is None
    ]
    if not_run:
        lines += [
            "",
            "Not run, as in prompt 72 (their flow state is a capped one's): "
            + ", ".join(not_run)
            + ".",
        ]
    return lines, verdicts


def closure_and_leftover(
    rows: list[dict],
) -> tuple[list[str], list[str], dict[str, dict]]:
    """Closure at every stored frame, and the accumulated leftover against its bound."""
    closure = [
        "### A1. Closure with the leftover booked, every stored frame and class (prediction (a))",
        "",
        "| Run | Grid | Frames | Frame-classes scored | Largest abs(residual) over the supply | Within 1e-12 |",
        "|---|---|---|---|---|---|",
    ]
    leftover = [
        "### B1. The accumulated leftover against its bound, every stored frame (prediction (a))",
        "",
        "f L is transport.leftover_share times transport.conservation_limit (1e-5). The bound at t <= T is f L S(t) t / T, below f L S(t) (ADR-016); a run is refused when abs(accumulated) reaches L S(t). Read at the stored frames, which the run file holds; the solve enforces each step's share at the step. Largest over the classes.",
        "",
        "| Run | Grid | Frames | Largest abs(accumulated) over f L S(t) t / T | Over f L S(t) | Over L S(t) | Accumulated at the end over the supply, largest class | Within f L S(t) at every frame |",
        "|---|---|---|---|---|---|---|---|",
    ]
    per_run: dict[str, dict] = {}
    for job in rows:
        if not finished(job):
            continue
        config = config_of(job["out"])
        f_l = config.transport.leftover_share * config.transport.conservation_limit
        horizon = config.t_end
        frames = RunFile(job["out"]).read_frames()
        worst, scored, over_t, over_fl, end_rel = 0.0, 0, 0.0, 0.0, 0.0
        ends = []
        for t, budgets in zip(frames.times, frames.budgets, strict=True):
            for b in budgets:
                if b.current is None:
                    continue
                s = b.supplied()
                if s == 0.0:
                    continue
                scored += 1
                worst = max(worst, abs(b.residual()) / s)
                acc = abs(b.leftover)
                over_fl = max(over_fl, acc / (f_l * s))
                if t > 0.0:
                    over_t = max(over_t, acc / (f_l * s * t / horizon))
        for b in frames.budgets[-1]:
            s = b.supplied() if b.current is not None else 0.0
            ends.append(b.leftover / s if s > 0.0 else 0.0)
        end_rel = max(ends, key=abs) if ends else 0.0
        per_run[job["name"]] = {
            "closure": worst,
            "over_fl": over_fl,
            "ends": ends,
            "scored": scored,
        }
        closure.append(
            f"| {job['name']} | {job['grid']} | {len(frames.steps):,} | {scored:,} | {fmt(worst, 3)} | {yes(worst <= CLOSURE)} |"
        )
        if scored == 0:
            leftover.append(
                f"| {job['name']} | {job['grid']} | {len(frames.steps):,} | nothing supplied | - | - | 0 | yes |"
            )
            continue
        leftover.append(
            f"| {job['name']} | {job['grid']} | {len(frames.steps):,} | {over_t:.4f} | {over_fl:.4f} | "
            f"{over_fl * f_l / config.transport.conservation_limit:.5f} | {end_rel:+.3e} | {yes(over_fl <= 1.0)} |"
        )
    return closure, leftover, per_run


def dust_table(rows: list[dict]) -> tuple[list[str], float | None]:
    """The 320x120 dust release's leftover at the end, every class, against the residual before (prediction (e))."""
    lines = [
        "### E1. The 320x120 dust release at the end: the accumulated leftover over the supply, against the residual before the budget (prediction (e))",
        "",
        "Before the budget the residual held the leftover (prompt 77); its value at the end of results/builder76/equipment_dust.h5 (bitwise the Phase 4 gate's results/builder69/equipment_dust.h5) is the 'before' column.",
        "",
        "| Class (um) | Accumulated leftover, this pass | Over the supply | Residual before, over the supply | Residual, this pass, over the supply |",
        "|---|---|---|---|---|",
    ]
    job = next(j for j in rows if j["name"] == "equipment_dust")
    if not finished(job):
        return [*lines, "| not finished | | | | |"], None
    new = RunFile(job["out"]).read_frames().budgets[-1]
    old = RunFile(job["before"]).read_frames().budgets[-1]
    largest = 0.0
    for k, (a, b) in enumerate(zip(new, old, strict=True)):
        ratio = a.leftover / a.supplied()
        largest = max(largest, abs(ratio))
        lines.append(
            f"| {SIZES_UM[k]} | {a.leftover:.6g} | {ratio:+.3e} | {b.residual() / b.supplied():+.3e} | "
            f"{a.residual() / a.supplied():+.3e} |"
        )
    return lines, largest


def sweeps_table(rows: list[dict]) -> list[str]:
    """Sweeps per transport solve, from record79.py's files."""
    lines = [
        "### K1. Sweeps per transport solve (record79.py), every finished run",
        "",
        "One solve per class per step. The cap is transport.max_diffusion_iter. A run restarted from a checkpoint has one file per attempt; an attempt killed before it exited left none, and its solves are not counted.",
        "",
        "| Run | Grid | Solves recorded | Steps planned x classes | Mean sweeps | Largest | Solves at the cap | Cap |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for job in rows:
        if not finished(job):
            continue
        files = sorted(job["out"].parent.glob(f"{job['out'].stem}.sweeps.*.npz"))
        sweeps = (
            np.concatenate([np.load(f)["sweeps"] for f in files])
            if files
            else np.array([], dtype=int)
        )
        config = config_of(job["out"])
        cap = config.transport.max_diffusion_iter
        steps = re.search(r"^steps: (\d+)", job["summary"], re.MULTILINE)
        planned = int(steps.group(1)) * len(config.particle_sizes) if steps else 0
        if sweeps.size == 0:
            lines.append(
                f"| {job['name']} | {job['grid']} | 0 | {planned:,} | - | - | - | {cap} |"
            )
            continue
        lines.append(
            f"| {job['name']} | {job['grid']} | {sweeps.size:,} | {planned:,} | {sweeps.mean():.2f} | "
            f"{sweeps.max()} | {int((sweeps >= cap).sum())} | {cap} |"
        )
    return lines


def wall_table(rows: list[dict]) -> list[str]:
    """The march's wall time against the run it replaces."""
    lines = [
        "### W1. March wall time against the run replaced",
        "",
        "The march's wall is the run's summary wall less the flow solves the run made itself (this pass copies them in), and less the run's overlap with the machine's standby, which the summary's clock ran through. The old runs ran five at a time (prompts 72 and 75 one process, prompt 76 four, 76b one); this pass three, at most two on 320x120. Placement runs: prompt 75's wall from run75.json, which ran them through time_integration.run in one process.",
        "",
        "| Run | Grid | This pass (s) | Standby taken out (s) | Replaced run (s) | Ratio | Processes beside it at its start, this pass |",
        "|---|---|---|---|---|---|---|",
    ]
    placement = json.loads(
        (ROOT / "results" / "builder75" / "run75.json").read_text(encoding="utf-8")
    )["timing"]
    for job in rows:
        if not finished(job):
            continue
        asleep = standby_seconds(job)
        new = march_seconds(job["summary"]) - asleep
        if job["group"] == "L":
            old = float(placement[job["name"]])
        else:
            old = march_seconds(old_summary(job)) if job["before"] is not None else None
        beside = job["record"]["attempts"][-1].get("concurrent", "-")
        ratio = f"{new / old:.3f}" if new is not None and old else "-"
        old_text = f"{old:.0f}" if old is not None else "-"
        lines.append(
            f"| {job['name']} | {job['grid']} | {new:.0f} | {asleep:.0f} | {old_text} | {ratio} | {beside} |"
        )
    return lines


def memory_table(rows: list[dict]) -> list[str]:
    """Peak working set per grid, and the queue log's memory readings."""
    lines = [
        "### M1. Peak working set per grid (run79.py, the interpreter process of each job)",
        "",
        "| Grid | Jobs measured | Largest peak in a job's first 10 minutes (GB) | Largest peak over a job (GB) | Smallest available memory at a start (GB of total) |",
        "|---|---|---|---|---|",
    ]
    grids: dict[str, list[dict]] = {}
    for job in rows:
        for attempt in job["record"]["attempts"]:
            grids.setdefault(job["grid"], []).append(attempt)
    for grid in ("320x120", "160x60", "80x30"):
        attempts = grids.get(grid, [])
        peaks = [a["peak_working_set"] for a in attempts if a.get("peak_working_set")]
        early = [
            a["peak_working_set_first_10_min"]
            for a in attempts
            if a.get("peak_working_set_first_10_min")
        ]
        avail = [a["avail_bytes"] for a in attempts if a.get("avail_bytes")]
        total = attempts[0]["total_bytes"] if attempts else 0
        lines.append(
            f"| {grid} | {len(peaks)} | {max(early) / 2**30:.3f} | {max(peaks) / 2**30:.3f} | "
            f"{min(avail) / 2**30:.1f} of {total / 2**30:.1f} |"
            if peaks and early and avail
            else f"| {grid} | {len(peaks)} | {f'{max(early) / 2**30:.3f}' if early else '- (every run ended inside 10 minutes)'} | "
            f"{f'{max(peaks) / 2**30:.3f}' if peaks else '-'} | "
            f"{f'{min(avail) / 2**30:.1f} of {total / 2**30:.1f}' if avail else '-'} |"
        )
    log = read_text(QUEUE_LOG).splitlines()
    waits = [ln for ln in log if " wait: " in ln]
    lines += [
        "",
        f"The queue log (results/builder79/queue.log) has {sum(' start ' in ln for ln in log)} starts, "
        f"{sum(' finish ' in ln for ln in log)} finishes and {len(waits)} waits for memory; it is reproduced as Q2.",
    ]
    return lines


# ---------------------------------------------------------------------------
# Latencies, the clean-out, the placement
# ---------------------------------------------------------------------------


def latency_tables(rows: list[dict]) -> tuple[list[str], list[str], dict[str, bool]]:
    """Every arrival map cell against the run replaced (predictions (c), (d))."""
    lines = [
        "### L1. The arrival maps against the runs replaced: every event, class and non-SOLID cell (predictions (c), (d))",
        "",
        "A cell's latency is the first step time at or after the event's onset at which it is at or above the class's threshold (ADR-015 D); every detection latency of a layout is the least over its cells, so the maps hold every one. Where the replaced run holds its arrival maps (prompts 75 and 76) both sides are step-resolved and the tolerance is the run's largest step as its summary prints it (six figures), widened by 1e-5 for that rounding; prompt 72's runs predate the maps, so both sides are rebuilt from the stored frames (placement.arrival_from_frames) and the tolerance is the largest frame spacing. 'Reached in one only' counts cells reached in one run and not the other.",
        "",
        "| Run | Grid | Resolution | Entries | Reached in both | Never in both | Reached in one only | Equal | Largest difference (s) | Tolerance (s) | Within it everywhere |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    changes = [
        "### L2. The sensor cells whose latency moved, at the file's four sensors",
        "",
        "| Run | Event | Class (um) | Sensor | Before (s) | After (s) | Difference (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    verdicts: dict[str, bool] = {}
    any_change = False
    for job in rows:
        if not finished(job) or job["before"] is None:
            continue
        config = config_of(job["out"])
        mesh = Mesh(config)
        old, event_names, resolution = arrival_maps(job["before"])
        if resolution == "frames":
            new = arrival_maps(job["out"], from_frames=True)[0]
            frames = RunFile(job["out"]).read_frames()
            tolerance = float(np.diff(frames.times).max())
        else:
            new = arrival_maps(job["out"])[0]
            # The summary prints the step to six significant figures, so the
            # one-step tolerance carries that rounding (5e-6 relative), not 1e-9.
            tolerance = max_step(job["summary"]) * (1.0 + 1e-5)
        live = ~np.isnan(new)
        both = live & np.isfinite(new) & np.isfinite(old)
        never = live & np.isinf(new) & np.isinf(old)
        one = live & (np.isfinite(new) != np.isfinite(old))
        with np.errstate(invalid="ignore"):
            diff = np.where(both, np.abs(new - old), 0.0)
        equal = int(((diff == 0.0) & both).sum() + never.sum())
        largest = float(diff.max()) if both.any() else 0.0
        within = not one.any() and largest <= tolerance * (1.0 + 1e-12)
        verdicts[job["name"]] = within
        lines.append(
            f"| {job['name']} | {job['grid']} | {resolution} | {int(live.sum()):,} | {int(both.sum()):,} | "
            f"{int(never.sum()):,} | {int(one.sum()):,} | {equal:,} | {largest:.6f} | {tolerance:.6f} | {yes(within)} |"
        )
        cells = sensor_cells(config, mesh)
        names = [s.name for s in config.sensors]
        for e in range(new.shape[0]):
            for k in range(new.shape[1]):
                for s, (j, i) in enumerate(cells):
                    x, y = old[e, k, j, i], new[e, k, j, i]
                    if x == y or (np.isinf(x) and np.isinf(y)):
                        continue
                    any_change = True
                    changes.append(
                        f"| {job['name']} | {event_names[e]} | {SIZES_UM[k]} | {names[s]} | {x:.6f} | {y:.6f} | {y - x:+.6f} |"
                    )
    if not any_change:
        changes.append("| none | | | | | | |")
    return lines, changes, verdicts


def arrival_maps(
    path: Path, from_frames: bool = False
) -> tuple[np.ndarray, tuple[str, ...], str]:
    """A run file's arrival maps, its /arrival/ or rebuilt from its frames.

    As ``placement.arrival_from_run``, but on the configuration ``config_of``
    reads, so a run file written before the conservation keys loads. Returns
    the maps [events, classes, ny, nx], the events' names and "steps" or
    "frames".
    """
    run = RunFile(path)
    if not from_frames:
        try:
            record = run.read_arrival()
        except KeyError:
            pass
        else:
            return record.first_above, tuple(record.event_names), "steps"
    config = config_of(path)
    events = contaminating_events(config.scenario)
    maps = arrival_from_frames(
        run.read_frames(),
        np.array(config.class_thresholds("the arrival maps")),
        Mesh(config).cell_type != SOLID,
        np.array([onset for _, onset in events]),
    )
    return maps, tuple(name for name, _ in events), "frames"


def cleanout_table(rows: list[dict]) -> tuple[list[str], list[bool]]:
    """The clean-out's recovery on three grids against the runs replaced (prediction (f))."""
    lines = [
        "### C1. The clean-out's recovery on three grids against the runs replaced (prediction (f))",
        "",
        "From each run file's recovery record (ADR-014 C). The 320x120 run replaced is prompt 76's (bitwise the Phase 4 gate's); the 160x60 and 80x30 runs are prompt 72's, both to 240 s.",
        "",
        "| Grid | Quantity | Class (um) | Before (s) | This pass (s) | Relative change | Within 0.1% |",
        "|---|---|---|---|---|---|---|",
    ]
    verdicts = []
    for name in ("clean_out_80x30", "clean_out_160x60", "clean_out"):
        job = next(j for j in rows if j["name"] == name)
        if not finished(job):
            lines.append(f"| {job['grid']} | not finished | | | | | |")
            verdicts.append(False)
            continue
        a = RunFile(job["out"]).read_recovery()
        b = RunFile(job["before"]).read_recovery()
        quantities = [
            ("room recovery time", "all", b.room_recovery_time, a.room_recovery_time)
        ]
        quantities += [
            (
                "free-area mean 100:1",
                SIZES_UM[k],
                float(b.mean_time[k]),
                float(a.mean_time[k]),
            )
            for k in range(len(a.mean_time))
        ]
        for label, size, old, new in quantities:
            change = (new - old) / old
            ok = abs(change) <= 1.0e-3
            verdicts.append(ok)
            lines.append(
                f"| {job['grid']} | {label} | {size} | {old:.4f} | {new:.4f} | {change:+.2e} | {yes(ok)} |"
            )
    return lines, verdicts


def placement80_table() -> tuple[list[str], list[bool]]:
    """Prompt 75's 80x30 search, under main's code and configuration, on the old runs and the new."""
    lines = [
        "### Z1. Prompt 75's 80x30 placement search under main's code and configuration, on prompt 75's runs and on this pass's",
        "",
        "results/builder75/run75.py's procedure: the door's, the breach's and the dust's arrival maps, the detection units, the configuration's placement spec (main's: mount, standoff, stand band, count, rule), the mountable and the free pools, the search, the units the search cannot reach excluded, the file's four scored on the rest, and the ranking. Both sides run the same code; only the run files differ.",
        "",
        "| Pool | Side | Candidates | Units | Cells (j, i) | Worst (s) | Mean (s) | Undetected | Uncovered units | File's four: worst, mean | Ranked first |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    sides = {
        "prompt 75": configs79.PLACEMENT_BEFORE,
        "this pass": configs79.group_dir("L"),
    }
    results: dict[str, dict] = {}
    for side, root in sides.items():
        paths = {name: root / f"{name}.h5" for name in configs79.PLACEMENT_RUNS}
        if not all(p.exists() for p in paths.values()):
            lines.append(f"| - | {side} | runs missing | | | | | | | | |")
            continue
        configs = [
            config_of(configs79.group_dir("L") / f"{name}.h5")
            for name in configs79.PLACEMENT_RUNS
        ]
        config = configs[0]
        mesh = Mesh(config)
        spec = config.placement
        units = detection_units(
            {name: arrival_from_run(p) for name, p in paths.items()}
        )
        pools = {
            "mounted": candidates(
                mesh, configs, spec.mount, spec.standoff, stand_height=spec.stand_height
            ),
            "free": candidates(mesh, configs, ("free",), spec.standoff),
        }
        for pool, cells in pools.items():
            found = search(units, cells, spec.count, spec.rule)
            kept = exclude_units(units, found.uncovered)
            file_layout = given_layout(kept, sensor_cells(config, mesh))
            ranked = rank([file_layout, found])
            results[f"{pool}|{side}"] = {
                "cells": [tuple(int(v) for v in c) for c in found.cells],
                "worst": found.score.worst,
                "mean": found.score.mean,
                "first": ranked[0].method,
            }
            lines.append(
                f"| {pool} | {side} | {len(cells):,} | {len(units.labels)} | "
                f"{', '.join(f'({j}, {i})' for j, i in found.cells)} | {found.score.worst:.3f} | "
                f"{found.score.mean:.3f} | {found.score.undetected} | {len(found.uncovered)} | "
                f"{file_layout.score.worst:.3f}, {file_layout.score.mean:.3f} | {ranked[0].method} |"
            )
    verdicts = []
    for pool in ("mounted", "free"):
        a, b = results.get(f"{pool}|prompt 75"), results.get(f"{pool}|this pass")
        if a is None or b is None:
            verdicts.append(False)
            continue
        verdicts.append(a["first"] == b["first"])
        lines.append("")
        lines.append(
            f"{pool}: cells {'the same' if a['cells'] == b['cells'] else 'DIFFER'}; worst {a['worst']:.3f} to "
            f"{b['worst']:.3f} s, mean {a['mean']:.3f} to {b['mean']:.3f} s; ranked first {a['first']} and "
            f"{b['first']}."
        )
    return lines, verdicts


# ---------------------------------------------------------------------------
# The re-scores and the side-by-side changes
# ---------------------------------------------------------------------------


def load_module(path: Path, name: str) -> ModuleType:
    """Import a probe script by path under a name of its own."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare() -> None:
    """Copy configurations and logs where the table scripts read them; write runs72.json."""
    for group, folder in configs79.GROUP_DIRS.items():
        target = configs79.group_dir(group)
        (target / "configs").mkdir(parents=True, exist_ok=True)
        for path in (configs79.CONFIGS / folder).glob("*.yaml"):
            shutil.copy2(path, target / "configs" / path.name)
        (target / "logs").mkdir(parents=True, exist_ok=True)
        for path in target.glob("*.log"):
            shutil.copy2(path, target / "logs" / path.name)
    queue = load_queue()["jobs"]
    record = {}
    for job in configs79.jobs():
        if job["group"] != "S":
            continue
        attempts = queue.get(job["name"], {}).get("attempts", [])
        if not attempts:
            continue
        row = {"started": attempts[0]["started"]}
        if attempts[-1].get("ended"):
            row["ended"] = attempts[-1]["ended"]
            row["returncode"] = attempts[-1].get("exit")
        record[job["name"]] = row
    (configs79.group_dir("S") / "runs72.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def accept_older_records() -> None:
    """Let SimConfig load a transport section written before ADR-016's two keys.

    Main's loader requires ``transport.conservation_limit`` and
    ``transport.leftover_share`` (prompt 78), which no run file or YAML
    written before prompt 78 holds, so the table scripts cannot load the
    earlier runs they compare with (the Phase 4 gate's, prompt 64's). In
    this process only, a transport section without them is given the
    product file's values. The two keys steer the implicit solve's stop and
    the refusal (ADR-016); no table script solves anything, so no table
    reads them.
    """
    if getattr(SimConfig, "_tables79_shim", False):
        return
    product = SimConfig(configs79.CONFIGS / "product" / "scenario_clean_out.yaml")
    keys = {
        "conservation_limit": product.transport.conservation_limit,
        "leftover_share": product.transport.leftover_share,
    }
    original = SimConfig._parse_transport.__func__

    def parse(cls: type, section: object) -> object:
        if isinstance(section, dict) and "conservation_limit" not in section:
            section = keys | section
        return original(cls, section)

    SimConfig._parse_transport = classmethod(parse)
    SimConfig._tables79_shim = True


def rescore() -> dict[str, str]:
    """The three scripts' tables on this pass's runs, each by its own code."""
    accept_older_records()
    probe = ROOT / "docs" / "reports"
    product, sweep, placement = (configs79.group_dir(g) for g in ("P", "S", "L"))
    out = {}

    t69 = load_module(probe / "probe69" / "tables69.py", "tables69")
    runs = t69.load_runs(product, {name: name for name in t69.NAMES})
    coarse = t69.load_runs(t69.COARSE, t69.COARSE_STEMS)
    out["69"] = t69.render(runs, coarse)[0]

    t72 = load_module(probe / "probe72" / "tables72.py", "tables72")
    t72.RESULTS, t72.LOGS, t72.GATE = sweep, sweep / "logs", product
    out["72"], out["72 report"] = t72.generate()

    t76 = load_module(probe / "probe76" / "tables76.py", "tables76")
    t76.COARSE, t76.SWEEP = placement, sweep
    t76.COARSE_CLEAN_OUT = sweep / "clean_out_80x30.h5"
    runs76 = {name: t76.Run(product, name) for name in t76.NAMES}
    analysis = t76.Analysis(runs76)
    out["76"] = t76.render(analysis)[0]
    out["76 K2 frames"] = "\n".join(arms_from_frames(t76, analysis, sweep))
    return out


def arms_from_frames(t76: object, analysis: object, sweep: Path) -> list[str]:
    """tables76's K2 with the sweep's arm runs read from their frames.

    Prompt 72's arm runs predate the arrival maps, so the committed K2 read them
    from their frames (frame-resolved, about 0.8 s apart on 160x60). This pass's
    runs hold step-resolved maps, which main's ``arrival_from_run`` prefers, so
    the plain re-score changes the resolution as well as the runs. Reading the
    new runs from their frames as well isolates what the budget changed.
    """
    original = t76.arrival_from_run

    def from_frames(run_file: object) -> RunArrival:
        path = Path(run_file.path if isinstance(run_file, RunFile) else run_file)
        if path.parent.resolve() != sweep.resolve():
            return original(run_file)
        step = original(run_file)
        maps = arrival_maps(path, from_frames=True)[0]
        return RunArrival(
            maps,
            step.onsets,
            FRAME_RESOLVED,
            step.event_names,
            float(RunFile(path).read_frames().times[-1]),
            "prompt 79: read from the frames, as the committed K2 read prompt 72's runs",
        )

    t76.arrival_from_run = from_frames
    try:
        return t76.arms_table(analysis)
    finally:
        t76.arrival_from_run = original


def section(text: str, prefix: str) -> str:
    """One ``### `` section of a tables file, its heading to the next heading."""
    lines, inside = [], False
    for line in text.splitlines():
        if line.startswith("### "):
            inside = line.startswith(prefix)
        if inside:
            lines.append(line)
    return "\n".join(lines)


def header(source: str) -> str:
    """The first lines of a re-scored file."""
    reads = {
        "69": "results/builder79/product/ (the four scenarios the Phase 4 gate ran) and, for G1's 80x30 side, results/builder64/coarse/ as before",
        "72": "results/builder79/sweep/ for the sweep's runs and results/builder79/product/ for its gate rows; C1's step 4 80x30 row reads results/builder64/coarse/ as before",
        "76": "results/builder79/product/, results/builder79/placement/ and results/builder79/sweep/; the comparisons with the Phase 4 gate (F2, X1) and the hashes, coverage and plants read their records as before",
    }[source]
    return (
        f"<!-- Re-scored by docs/reports/probe79/tables79.py with tables{source}.py's own code; "
        f"it read {reads}. Its prose names the directories of the runs it was written for. -->\n\n"
    )


def split_tables(text: str) -> list[tuple[str, list[list[str]], list[str]]]:
    """The text's tables in order: (heading, rows of cells, the text lines that follow)."""
    out: list[tuple[str, list[list[str]], list[str]]] = []
    heading, rows, prose = "(before the first table)", [], []
    for line in text.splitlines():
        if line.startswith("### "):
            out.append((heading, rows, prose))
            heading, rows, prose = line, [], []
        elif line.startswith("|"):
            if re.fullmatch(r"\|(\s*-+\s*\|)+", line.replace(":", "-")):
                continue
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif line.strip() and not line.startswith("<!--"):
            prose.append(line)
    out.append((heading, rows, prose))
    return out


def number(cell: str) -> float | None:
    """A cell's number, when it is one."""
    try:
        return float(cell.replace(",", "").replace("%", ""))
    except ValueError:
        return None


def compare(old: str, new: str, source: str) -> tuple[list[str], list[str]]:
    """The summary rows and the side-by-side detail of one committed output against its re-score."""
    a, b = split_tables(old), split_tables(new)
    summary, detail = [], []
    for n in range(max(len(a), len(b))):
        if n >= len(a) or n >= len(b):
            heading = (a[n] if n < len(a) else b[n])[0]
            summary.append(
                f"| tables{source} | {heading[4:60]} | table only on one side | | | | |"
            )
            continue
        (ha, ra, pa), (hb, rb, pb) = a[n], b[n]
        name = ha[4:].split(" ", 1)[0] if ha.startswith("### ") else ha
        cells, changed, largest = 0, [], 0.0
        if ha != hb:
            changed.append(("heading", "-", ha[4:], hb[4:]))
        header_row = ra[0] if ra else []
        for r in range(max(len(ra), len(rb))):
            if r >= len(ra) or r >= len(rb):
                row = ra[r] if r < len(ra) else rb[r]
                changed.append(
                    (
                        " / ".join(row[:3]),
                        "whole row",
                        "absent" if r >= len(ra) else " | ".join(row),
                        "absent" if r >= len(rb) else " | ".join(row),
                    )
                )
                continue
            for c in range(max(len(ra[r]), len(rb[r]))):
                x = ra[r][c] if c < len(ra[r]) else ""
                y = rb[r][c] if c < len(rb[r]) else ""
                cells += 1
                if x == y:
                    continue
                label = " / ".join(ra[r][:3]) if r else "header"
                column = header_row[c] if c < len(header_row) else str(c)
                changed.append((label, column, x, y))
                u, v = number(x), number(y)
                if u is not None and v is not None and u != 0.0:
                    largest = max(largest, abs(v - u) / abs(u))
        prose_changed = [(o, p) for o, p in zip(pa, pb, strict=False) if o != p]
        if len(pa) != len(pb):
            prose_changed.append((f"{len(pa)} lines", f"{len(pb)} lines"))
        summary.append(
            f"| tables{source} | {name} | {max(len(ra), len(rb)) - 1 if ra or rb else 0} | {cells:,} | "
            f"{len(changed):,} | {largest:.3e} | {len(prose_changed)} |"
        )
        if changed or prose_changed:
            detail += [f"#### tables{source} {ha[4:]}", ""]
            if changed:
                detail += [
                    "| Row | Column | Committed | Re-scored |",
                    "|---|---|---|---|",
                ]
                detail += [f"| {r} | {c} | {x} | {y} |" for r, c, x, y in changed]
                detail.append("")
            for o, p in prose_changed:
                detail += [f"- committed: {o}", f"- re-scored: {p}"]
            if prose_changed:
                detail.append("")
    return summary, detail


def report_blocks72(text: str) -> dict[str, str]:
    """The generated blocks of phase5_sensitivity.md, by marker."""
    blocks = {}
    for found in re.finditer(
        r"<!-- tables72 (\w+) begin -->\n(.*?)<!-- tables72 \1 end -->", text, re.DOTALL
    ):
        blocks[found.group(1)] = found.group(2)
    return blocks


def contour_table(old: str, new: str) -> tuple[list[str], bool, bool]:
    """D1 of tables72: the distances to the contour, committed against re-scored (prediction (d))."""
    a = next(rows for h, rows, _ in split_tables(old) if h.startswith("### D1."))
    b = next(rows for h, rows, _ in split_tables(new) if h.startswith("### D1."))
    key = lambda row: tuple(row[:5])  # noqa: E731 -- a local key for two lookups
    old_rows = {key(r): r for r in a[1:]}
    reached_new = [r for r in b[1:] if r[7] != "no"]
    worst, worst_row, missing, compared = 0.0, None, 0, 0
    for row in b[1:]:
        match = old_rows.get(key(row))
        if match is None:
            missing += 1
            continue
        u, v = number(match[6]), number(row[6])
        if u is None or v is None:
            if match[6] != row[6]:
                worst, worst_row = math.inf, row
            continue
        compared += 1
        if abs(v - u) > worst:
            worst, worst_row = abs(v - u), (match, row)
    lines = [
        "### D1. The sweep's contour distances, re-scored against prompt 72's (tables72 D1; prediction (d))",
        "",
        "| Rows re-scored | Compared with a committed row | Without one | Largest change in cells | At | Sensor cells reached, re-scored |",
        "|---|---|---|---|---|---|",
    ]
    at = "-"
    if isinstance(worst_row, tuple):
        at = f"{' / '.join(worst_row[1][:5])}: {worst_row[0][6]} to {worst_row[1][6]} cells"
    lines.append(
        f"| {len(b) - 1} | {compared} | {missing} | {worst:.1f} | {at} | {len(reached_new)} |"
    )
    return lines, worst <= 1.0 + 1e-9, not reached_new


# ---------------------------------------------------------------------------
# The gates and the predictions
# ---------------------------------------------------------------------------


def ranking_lines(text: str) -> list[str]:
    """The ranking bullets of tables76's L1 and K1."""
    return [
        ln
        for ln in text.splitlines()
        if ln.startswith(
            ("- worst case:", "- mean:", "- coverage within", "- section D's ranking")
        )
    ]


def table_rows(text: str, prefix: str) -> list[list[str]]:
    """The rows of the first table whose heading starts with the prefix."""
    for heading, rows, _ in split_tables(text):
        if heading.startswith(prefix):
            return rows
    return []


def build() -> tuple[str, dict[str, str], dict[str, str], str]:
    """Every table, the re-scores and the changes file."""
    rows = jobs()
    tables: dict[str, list[str]] = {}
    tables["queue"] = queue_table(rows)
    tables["flow_states"], flow_ok = flow_state_table(rows)
    tables["capped"], capped_ok = capped_table(rows)
    tables["closure"], tables["leftover"], per_run = closure_and_leftover(rows)
    tables["dust"], dust_end = dust_table(rows)
    tables["sweeps"] = sweeps_table(rows)
    tables["wall"] = wall_table(rows)
    tables["memory"] = memory_table(rows)
    tables["queue_log"] = queue_log_table()
    tables["latency"], tables["latency_changes"], latency_ok = latency_tables(rows)
    tables["cleanout"], cleanout_ok = cleanout_table(rows)
    tables["placement80"], placement_ok = placement80_table()

    prepare()
    rescored = rescore()
    files = {key: header(key) + rescored[key] for key in RESCORED}
    summary = [
        "### X1. The committed tables against their re-scores, table by table (the detail in changes79.md)",
        "",
        "Tables are aligned by their order in the file; a cell is compared as text. The largest relative change is over the cells that are numbers on both sides.",
        "",
        "| Source | Table | Rows | Cells | Cells that differ | Largest relative change | Text lines that differ |",
        "|---|---|---|---|---|---|---|",
    ]
    detail = [
        "# The committed tables of prompts 69, 72 and 76 against their re-scores on prompt 79's runs",
        "",
        "Generated by tables79.py; do not edit. Each table whose cells differ, the committed value and the re-scored side by side, then the text lines between tables that differ.",
        "",
    ]
    for key, path in COMMITTED.items():
        s, d = compare(path.read_text(encoding="utf-8"), rescored[key], key)
        summary += s
        detail += d
    old72_report = report_blocks72(
        (ROOT / "docs" / "reports" / "phase5_sensitivity.md").read_text(
            encoding="utf-8"
        )
    )
    new72_report = report_blocks72(rescored["72 report"])
    for marker in old72_report:
        s, d = compare(
            old72_report[marker],
            new72_report.get(marker, ""),
            f"72 report block {marker}",
        )
        summary += s
        detail += d
    s, d = compare(
        section(COMMITTED["76"].read_text(encoding="utf-8"), "### K2."),
        rescored["76 K2 frames"],
        "76 K2, arm runs from frames",
    )
    summary += s
    detail += d
    tables["rescored"] = summary
    tables["contour"], contour_ok, unreached = contour_table(
        COMMITTED["72"].read_text(encoding="utf-8"), rescored["72"]
    )

    old76, new76 = COMMITTED["76"].read_text(encoding="utf-8"), rescored["76"]
    ranking_same = ranking_lines(old76) == ranking_lines(new76)
    curve_same = table_rows(old76, "### L2.") == table_rows(new76, "### L2.")
    k1_same = table_rows(old76, "### K1.") == table_rows(new76, "### K1.")
    k2_same = table_rows(old76, "### K2.") == table_rows(
        rescored["76 K2 frames"], "### K2."
    )
    k2_steps_same = table_rows(old76, "### K2.") == table_rows(new76, "### K2.")
    tables["gates"] = gates_table(
        rows,
        flow_ok,
        contour_ok,
        unreached,
        ranking_same,
        curve_same,
        k1_same,
        k2_same,
        k2_steps_same,
        latency_ok,
        cleanout_ok,
    )
    tables["predictions"] = predictions_table(
        rows,
        per_run,
        flow_ok,
        capped_ok,
        latency_ok,
        ranking_same,
        curve_same,
        k1_same,
        placement_ok,
        contour_ok,
        unreached,
        dust_end,
        cleanout_ok,
    )
    head = ["# The tables of prompt 79 (generated by tables79.py; do not edit)", ""]
    body = []
    for name in MARKERS:
        body += tables[name] + [""]
    joined = {name: "\n".join(lines) for name, lines in tables.items()}
    return "\n".join(head + body), joined, files, "\n".join(detail) + "\n"


def gates_table(
    rows: list[dict],
    flow_ok: dict[str, bool],
    contour_ok: bool,
    unreached: bool,
    ranking_same: bool,
    curve_same: bool,
    k1_same: bool,
    k2_same: bool,
    k2_steps_same: bool,
    latency_ok: dict[str, bool],
    cleanout_ok: list[bool],
) -> list[str]:
    """Whether each gate's conclusion holds on the re-run."""
    p_runs = [j["name"] for j in rows if j["group"] == "P"]
    p_latency = all(latency_ok.get(n, False) for n in p_runs)
    return [
        "### G1. The gates' conclusions on the re-run",
        "",
        "| Gate | Conclusion | What it rests on in this pass | Holds |",
        "|---|---|---|---|",
        f"| Phase 4, condition 2 | a third grid of the clean-out, run (prompt 72): the room recovery time is not monotone across the grids and the free-area mean is not converged | C1, and rescored72's C1 (changes79.md) | {yes(all(cleanout_ok))} (every C1 quantity within 0.1%) |",
        f"| Phase 4, condition 3 | the spreading check: no sensor's cell is reached by any plume at any arm that converged | D1 and L1 | {yes(unreached and contour_ok)} |",
        f"| Phase 5, latency claim | the detection latencies of the layouts (U2, L1 of tables76) | L1: every arrival map cell of the five 320x120 runs within one step | {yes(p_latency)} |",
        f"| Phase 5, ranking | the layouts' order per rule on 320x120 (L1's bullets), the free optimum's curve (L2), and the order on 80x30 and the arms (K1, K2) | rescored76 against tables76 | L1 order {yes(ranking_same)}; L2 rows {yes(curve_same)}; K1 rows {yes(k1_same)}; K2 rows, the arm runs read from frames as committed, {yes(k2_same)} (read step-resolved from this pass's maps, as main's arrival_from_run does: {'the same' if k2_steps_same else 'sharper, in rescored76.md'}) |",
        f"| Sweep | no sensor reached at any arm that converges | D1 | {yes(unreached)} |",
    ]


def predictions_table(
    rows: list[dict],
    per_run: dict[str, dict],
    flow_ok: dict[str, bool],
    capped_ok: list[bool],
    latency_ok: dict[str, bool],
    ranking_same: bool,
    curve_same: bool,
    k1_same: bool,
    placement_ok: list[bool],
    contour_ok: bool,
    unreached: bool,
    dust_end: float | None,
    cleanout_ok: list[bool],
) -> list[str]:
    """Prompt 79's predictions against the measurements."""
    done = [j for j in rows if finished(j)]
    refused = [j["name"] for j in rows if "leftover" in last_error(j)]
    closes = sum(per_run[j["name"]]["closure"] <= CLOSURE for j in done)
    bounded = sum(per_run[j["name"]]["over_fl"] <= 1.0 for j in done)
    a_ok = closes == len(done) and bounded == len(done) and not refused
    b_ok = all(flow_ok.values()) and len(capped_ok) > 0 and all(capped_ok)
    fine = [j["name"] for j in done if j["group"] == "P"]
    c_lat = all(latency_ok.get(n, False) for n in fine)
    c_ok = c_lat and ranking_same and curve_same and k1_same and all(placement_ok)
    d_ok = unreached and contour_ok
    e_ok = dust_end is not None and dust_end < 1.0e-6
    f_ok = len(cleanout_ok) > 0 and all(cleanout_ok)

    def outcome(flag: bool) -> str:
        return "holds" if flag else "FAILS"

    return [
        "### P1. Prompt 79's predictions against the measurements",
        "",
        "| Prediction | As stated (abridged) | Measured | Outcome |",
        "|---|---|---|---|",
        f"| (a) | every run closes within 1e-12 at every stored frame and class; accumulated leftover within f L S(t); none refused | closes in {closes} of {len(done)} finished runs; within f L S(t) at every frame in {bounded} of {len(done)}; refused: {', '.join(refused) or 'none'}; tables A1, B1 | {outcome(a_ok)} |",
        f"| (b) | every converged state reproduces by its fingerprint; the two that did not end as in prompt 72 | {sum(flow_ok.values())} of {len(flow_ok)} states bitwise; {sum(capped_ok)} of {len(capped_ok)} capped solves end as before; tables F1, F2 | {outcome(b_ok)} |",
        f"| (c) | on 320x120 every latency and all-clear reading within one step of the Phase 5 gate's; ranking unchanged at every count, 320x120 and 80x30 | arrival maps within one step in {sum(latency_ok.get(n, False) for n in fine)} of {len(fine)} runs (L1); L1 order {yes(ranking_same)}, L2 {yes(curve_same)}, K1 {yes(k1_same)}; 80x30 search ranked first the same in {sum(placement_ok)} of {len(placement_ok)} pools (Z1); all-clear readings in X1 (C2) | {outcome(c_ok)} |",
        f"| (d) | no sensor's cell reached at any converging arm; every contour distance within one cell of prompt 72's | table D1 | {outcome(d_ok)} |",
        f"| (e) | the 320x120 dust release's accumulated leftover at the end below 1e-6 of its supply | largest class {dust_end if dust_end is None else f'{dust_end:.3e}'}; table E1 | {outcome(e_ok)} |",
        f"| (f) | clean-out room recovery and free-area 100:1 within 0.1% on each grid | {sum(cleanout_ok)} of {len(cleanout_ok)} quantities within; table C1 | {outcome(f_ok)} |",
    ]


def splice(text: str, tables: dict[str, str]) -> str:
    """The report's text with each marked region replaced by its table."""
    for name, body in tables.items():
        begin, end = f"<!-- tables79 {name} begin -->", f"<!-- tables79 {name} end -->"
        if begin not in text or end not in text:
            continue
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{body}\n{end}{tail}"
    return text


def report_copies(text: str) -> dict[str, str]:
    """The report's copies of the tables, by marker."""
    copies = {}
    for name in MARKERS:
        begin, end = f"<!-- tables79 {name} begin -->", f"<!-- tables79 {name} end -->"
        if begin in text and end in text:
            copies[name] = text[
                text.index(begin) + len(begin) : text.index(end)
            ].strip()
    return copies


def main() -> int:
    """Write the tables and splice the report, or with --check compare them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text, tables, files, changes = build()
    outputs = {TABLES: text + "\n", CHANGES: changes}
    outputs |= {RESCORED[key]: body for key, body in files.items()}
    if args.check:
        failures = [
            rel(p)
            for p, body in outputs.items()
            if not p.exists() or p.read_text(encoding="utf-8") != body
        ]
        copies = report_copies(REPORT.read_text(encoding="utf-8"))
        failures += [
            f"report copy of {n}"
            for n, body in copies.items()
            if body != tables[n].strip()
        ]
        print("tables79: " + ("matches" if not failures else f"DIFFERS: {failures}"))
        return 1 if failures else 0
    for path, body in outputs.items():
        path.write_text(body, encoding="utf-8", newline="\n")
        print(f"wrote {rel(path)}")
    REPORT.write_text(
        splice(REPORT.read_text(encoding="utf-8"), tables),
        encoding="utf-8",
        newline="\n",
    )
    print(f"spliced {rel(REPORT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
