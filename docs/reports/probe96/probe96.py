"""Prompt 96's probe: the five product scenarios on 80x30 end to end on each backend.

    python docs/reports/probe96/probe96.py configs
    python docs/reports/probe96/probe96.py run <side> <name>
    python docs/reports/probe96/probe96.py fields <name>
    python docs/reports/probe96/probe96.py prompt91 <name>
    python docs/reports/probe96/probe96.py restart-march
    python docs/reports/probe96/probe96.py restart-flow
    python docs/reports/probe96/probe96.py queue
    python docs/reports/probe96/probe96.py compare
    python docs/reports/probe96/probe96.py timing --session N

Run under WSL2 with the Linux environment from the checkout's root, one BLAS
thread, through ``results/builder96/py.sh`` (NumPy's AVX-512 loops off unless
``DISPATCH=default``). Every run file is written on the Linux file system
(``~/scratch96/runs/``) and copied into ``results/builder96/runs/<side>/`` when it
ends (ADR-017 A), with the script's output and a record of the run beside it.

- ``configs``: each scenario's file under ``results/builder96/configs/<side>/``:
  main's ``configs/scenario_*.yaml`` on 80x30 with ``mass_imbalance_tol`` the base
  file's rule for that grid (prompt 85's ``mass_imbalance_tol``), as prompt 91's
  probe wrote its clean-out; ``solver.backend: core`` on the core's side. Also
  the restart's files: the dust release on the core checkpointing every 1,000
  steps, and capped at 600 outer iterations.
- ``run``: one scenario from rest through ``scripts/run_scenario.py`` in its own
  process, on side ``reference`` or ``core``.
- ``fields``: one scenario on the core with the monitor on the fields
  (``CoreRun.monitor_reads_fields``), through ``run_scenario.main`` in this
  process, since the script has no switch for it.
- ``prompt91``: one of prompt 91's three 80x30 reference runs made again through
  ``scripts/run_scenario.py`` from its own scenario file and flow states, the key
  absent, in the environment its record states (no ``NPY_DISABLE_CPU_FEATURES``:
  run with ``DISPATCH=default``).
- ``restart-march``: the dust release on the core with a checkpoint every 1,000
  steps, uninterrupted through the script; then again in this process, stopped
  by an exception from ``on_step`` after step 1,500, and restarted through the
  script.
- ``restart-flow``: the dust release on the core capped at 600 outer iterations
  through the script (it exits 1 with its steady checkpoint), restarted through
  the script under the file's cap.
- ``queue``: every run not yet made, at most three processes at once.
- ``compare``: every pair by ``scripts/compare_runs.py`` and the placement over
  the scenarios with contaminating events, into ``results/builder96/compare.json``.
- ``timing``: the dust release from rest, through ``run_scenario.main`` in one
  process per run, each backend once, in the session's order, the flow solves,
  the march and the rest timed apart, the machine's state recorded.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402 -- the thread variables are set first
import yaml  # noqa: E402 -- the thread variables are set first

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import compare_runs  # noqa: E402 -- follows sys.path.insert

from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.io_manager import RunFile  # noqa: E402 -- follows sys.path.insert

RESULTS = ROOT / "results" / "builder96"
CONFIGS = RESULTS / "configs"
RUNS = RESULTS / "runs"
LINUX_RUNS = Path.home() / "scratch96" / "runs"
B91 = ROOT / "results" / "builder91"
B86 = ROOT / "results" / "builder86"
PYTHON = Path.home() / ".venvs" / "cfd_clean_room" / "bin" / "python"
SCRIPT = ROOT / "scripts" / "run_scenario.py"
SIDES = ("reference", "core")
# The five product scenarios: short name to main's file.
SCENARIOS = {
    "clean80": "scenario_clean_out.yaml",
    "door80": "scenario_door_leak.yaml",
    "dust80": "scenario_equipment_dust.yaml",
    "breach80": "scenario_filter_breach.yaml",
    "seal80": "scenario_seal_leak.yaml",
}
# The scenarios the placement reads (each has a contaminating event).
PLACEMENT = ("door80", "dust80", "breach80", "seal80")
GRID = (80, 30)
ROOM = (8.0, 3.0)
# The base file's rule for mass_imbalance_tol: conservation_limit x rho x the
# smallest cell's area over 60 s (prompt 85's probe85.mass_imbalance_tol).
RULE_LIMIT, RULE_RHO, RULE_SECONDS = 1.0e-4, 1.2, 60.0
# Prompt 91's three Linux reference runs and the flow states they read.
PROMPT91 = {
    "dust80": B86 / "runs" / "reference" / "dust_80x30.h5",
    "door80": B86 / "runs" / "reference" / "door_80x30.h5",
    "breach80": B86 / "runs" / "reference" / "breach_80x30.h5",
}
# The restarts: the march's checkpoint cadence and the step after which it is
# stopped; the flow solve's cap.
MARCH_CADENCE, MARCH_STOP = 1000, 1500
FLOW_CAP = 600
MAX_PROCESSES = 3
POLL_SECONDS = 5.0


def mass_imbalance_tol() -> float:
    """The base file's rule on 80x30, in prompt 85's order of operations."""
    cell = (ROOM[0] / GRID[0]) * (ROOM[1] / GRID[1])
    return RULE_LIMIT * RULE_RHO * cell / RULE_SECONDS


def config_path(side: str, name: str) -> Path:
    """A run's scenario file."""
    return CONFIGS / side / f"{name}.yaml"


def mapping_of(name: str, side: str) -> dict:
    """Main's scenario file on 80x30 with the grid's tolerance, on SIDE's backend."""
    source = ROOT / "configs" / SCENARIOS[name]
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["scenario"]["base"] = str((source.parent / raw["scenario"]["base"]).resolve())
    raw["domain"] = {"nx": GRID[0], "ny": GRID[1]}
    raw.setdefault("solver", {})["mass_imbalance_tol"] = mass_imbalance_tol()
    if side == "core":
        raw["solver"]["backend"] = "core"
    return raw


def write_configs() -> None:
    """Every run's scenario file, each loaded once to refuse an invalid one now."""
    files: dict[tuple[str, str], dict] = {}
    for side in SIDES:
        for name in SCENARIOS:
            files[(side, name)] = mapping_of(name, side)
    cadence = mapping_of("dust80", "core")
    cadence["solver"]["checkpoint_interval"] = MARCH_CADENCE
    files[("core", "dust80_c1000")] = cadence
    capped = mapping_of("dust80", "core")
    capped["solver"]["max_simple_iter"] = FLOW_CAP
    files[("core", "dust80_cap600")] = capped
    for (side, name), raw in files.items():
        path = config_path(side, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        SimConfig(path)
        print("wrote", path)


# ---------------------------------------------------------------------------
# The runs


def commit() -> str:
    """HEAD's hash read from the checkout's files (git runs on Windows only)."""
    head = (ROOT / ".git" / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = ROOT / ".git" / head[5:]
    if ref.exists():
        return ref.read_text().strip()
    for line in (ROOT / ".git" / "packed-refs").read_text().splitlines():
        if line.endswith(head[5:]):
            return line.split()[0]
    return "unknown"


def paths(side: str, name: str) -> tuple[Path, Path]:
    """The run file on the Linux file system and its copy's directory."""
    return LINUX_RUNS / side / f"{name}.h5", RUNS / side


def fresh(path: Path) -> None:
    """Remove a run file and its checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for p in (path, path.with_name(path.stem + ".checkpoint.h5")):
        if p.exists():
            p.unlink()


def environment() -> dict:
    """What the run's record says of its environment."""
    from threadpoolctl import threadpool_info

    return {
        "commit": commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
        "blas": [
            {k: d.get(k) for k in ("internal_api", "num_threads", "version")}
            for d in threadpool_info()
        ],
    }


def finish(path: Path, copy_to: Path, name: str, record: dict) -> None:
    """Copy the run file and its checkpoint into the checkout, with the run's record."""
    copy_to.mkdir(parents=True, exist_ok=True)
    for p in (path, path.with_name(path.stem + ".checkpoint.h5")):
        if p.exists():
            shutil.copy2(p, copy_to / p.name)
    (copy_to / f"{name}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )


def script(args: list[str], log: Path) -> tuple[int, float]:
    """``scripts/run_scenario.py`` in its own process, its output to LOG."""
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as out:
        done = subprocess.run(
            [str(PYTHON), "-u", str(SCRIPT), *args],
            cwd=ROOT,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    return done.returncode, time.perf_counter() - started


def run_side(side: str, name: str) -> None:
    """One scenario from rest through the script, on one side."""
    path, copy_to = paths(side, name)
    fresh(path)
    log = copy_to / f"{name}.log"
    code, wall = script([str(config_path(side, name)), "--out", str(path)], log)
    record = {"name": name, "side": side, "exit": code, "wall_seconds": wall}
    finish(path, copy_to, name, {**record, **environment()})
    print(name, side, "exit", code, f"{wall:.1f} s")


def in_process(args: list[str], log: Path) -> tuple[int, float]:
    """``run_scenario.main`` in this process, its output to LOG."""
    import contextlib
    import logging

    import run_scenario

    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as out, contextlib.redirect_stdout(out):
        handler = logging.StreamHandler(out)
        logging.getLogger().addHandler(handler)
        try:
            code = run_scenario.main([*args, "--quiet"])
        finally:
            logging.getLogger().removeHandler(handler)
    return code, time.perf_counter() - started


def run_fields(name: str) -> None:
    """One scenario on the core with the monitor on the fields."""
    from src.core_backend import CoreRun

    path, copy_to = paths("fields", name)
    fresh(path)
    CoreRun.monitor_reads_fields = True
    code, wall = in_process(
        [str(config_path("core", name)), "--out", str(path)], copy_to / f"{name}.log"
    )
    record = {"name": name, "side": "fields", "exit": code, "wall_seconds": wall}
    finish(path, copy_to, name, {**record, **environment()})
    print(name, "fields", "exit", code, f"{wall:.1f} s")


def run_prompt91(name: str) -> None:
    """One of prompt 91's reference runs again, from its file and its flow states."""
    path, copy_to = paths("prompt91", name)
    fresh(path)
    args = [
        str(B91 / "configs" / f"{name}.yaml"),
        "--out",
        str(path),
        "--flows-from",
        str(PROMPT91[name]),
    ]
    code, wall = script(args, copy_to / f"{name}.log")
    record = {"name": name, "side": "prompt91", "exit": code, "wall_seconds": wall}
    finish(path, copy_to, name, {**record, **environment()})
    print(name, "prompt91", "exit", code, f"{wall:.1f} s")


class StopError(RuntimeError):
    """The crash the march's restart continues from."""


def restart_march() -> None:
    """The cadenced dust release uninterrupted, then stopped and restarted (prediction d)."""
    from src.time_integration import Provenance, run

    name = "dust80_c1000"
    config = config_path("core", name)
    whole, copy_to = paths("restart", name)
    fresh(whole)
    code, wall = script([str(config), "--out", str(whole)], copy_to / f"{name}.log")
    finish(whole, copy_to, name, {"exit": code, "wall_seconds": wall, **environment()})
    cut, _ = paths("restart", f"{name}_restart")
    fresh(cut)

    def stop(step: object) -> None:
        if step.step == MARCH_STOP:
            raise StopError(f"stopped after step {MARCH_STOP}")

    try:
        run(SimConfig(config), cut, on_step=stop, provenance=Provenance.here(commit()))
    except StopError as stopped:
        print(stopped)
    else:
        raise SystemExit("the run was not stopped")
    checkpoint = int(RunFile(cut).read_checkpoint().step)
    log = copy_to / f"{name}_restart.log"
    code, wall = script([str(config), "--restart", str(cut)], log)
    record = {
        "exit": code,
        "wall_seconds": wall,
        "stopped_after": MARCH_STOP,
        "checkpoint_step": checkpoint,
        **environment(),
    }
    finish(cut, copy_to, f"{name}_restart", record)
    print(name, "restart exit", code, "from step", checkpoint)


def restart_flow() -> None:
    """The dust release capped in its flow solve, restarted under the file's cap."""
    capped, copy_to = paths("restart", "dust80_cap600")
    fresh(capped)
    first, wall = script(
        [str(config_path("core", "dust80_cap600")), "--out", str(capped)],
        copy_to / "dust80_cap600.log",
    )
    key, steady = RunFile(capped).read_steady_checkpoint()
    held = {"fingerprint": key, "outer": int(steady.state.outer)}
    shutil.copy2(capped, copy_to / "dust80_cap600_capped.h5")
    code, again = script(
        [str(config_path("core", "dust80")), "--restart", str(capped)],
        copy_to / "dust80_cap600_restart.log",
    )
    record = {
        "first_exit": first,
        "first_wall_seconds": wall,
        "steady_checkpoint": held,
        "exit": code,
        "wall_seconds": again,
        **environment(),
    }
    finish(capped, copy_to, "dust80_cap600", record)
    print("dust80_cap600 first exit", first, "at", held, "restart exit", code)


# ---------------------------------------------------------------------------
# The queue


def jobs() -> list[tuple[list[str], dict[str, str], Path]]:
    """Every run as (probe arguments, environment, record that marks it done)."""
    out = []
    for name in SCENARIOS:
        for side in SIDES:
            out.append((["run", side, name], {}, RUNS / side / f"{name}.json"))
        out.append((["fields", name], {}, RUNS / "fields" / f"{name}.json"))
    for name in PROMPT91:
        out.append(
            (
                ["prompt91", name],
                {"NPY_DISABLE_CPU_FEATURES": ""},
                RUNS / "prompt91" / f"{name}.json",
            )
        )
    out.append((["restart-march"], {}, RUNS / "restart" / "dust80_c1000_restart.json"))
    out.append((["restart-flow"], {}, RUNS / "restart" / "dust80_cap600.json"))
    return out


def log_line(text: str) -> None:
    """One dated line to the queue's log and to stdout."""
    line = f"{datetime.now().isoformat(timespec='seconds')} {text}"
    with (RESULTS / "queue.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def queue() -> int:
    """Start every run not done, never more than MAX_PROCESSES at once."""
    waiting = [job for job in jobs() if not job[2].exists()]
    running: dict[str, subprocess.Popen] = {}
    failed = []
    log_line(f"queue: {len(waiting)} runs to do")
    while waiting or running:
        while waiting and len(running) < MAX_PROCESSES:
            args, env, _ = waiting.pop(0)
            name = "_".join(args)
            environ = os.environ.copy()
            for key, value in env.items():
                if value:
                    environ[key] = value
                else:
                    environ.pop(key, None)
            logs = RESULTS / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / f"{name}.out").open("w") as out:
                running[name] = subprocess.Popen(
                    [str(PYTHON), "-u", str(Path(__file__)), *args],
                    cwd=ROOT,
                    env=environ,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                )
            log_line(f"start {name} pid {running[name].pid}")
        time.sleep(POLL_SECONDS)
        for name, proc in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[name]
            if code != 0:
                failed.append(name)
            log_line(f"{'done' if code == 0 else 'failed'} {name} exit {code}")
    log_line(f"queue empty; failed: {failed if failed else 'none'}")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# The comparison


def run_file(side: str, name: str) -> Path:
    """A run's copy in the checkout."""
    return RUNS / side / f"{name}.h5"


def pairs() -> list[tuple[str, str, Path, Path]]:
    """Every pair compared: (label, kind, the reference's file, the other's)."""
    out = []
    for name in SCENARIOS:
        out.append(
            (name, "backends", run_file("reference", name), run_file("core", name))
        )
        out.append((name, "monitor", run_file("fields", name), run_file("core", name)))
    for name in PROMPT91:
        record = B91 / "runs" / "reference" / f"{name}.h5"
        out.append((name, "prompt91", record, run_file("prompt91", name)))
    whole = run_file("restart", "dust80_c1000")
    out.append(
        (
            "dust80_c1000",
            "restart march",
            whole,
            run_file("restart", "dust80_c1000_restart"),
        )
    )
    out.append(
        (
            "dust80_cap600",
            "restart flow",
            run_file("core", "dust80"),
            run_file("restart", "dust80_cap600"),
        )
    )
    out.append(("dust80_c1000", "checkpoints", run_file("core", "dust80"), whole))
    return out


def compare_all() -> None:
    """Every pair and the placement, into compare.json."""
    out: dict = {"commit": commit(), "pairs": [], "placement": None}
    for label, kind, a, b in pairs():
        if not (a.exists() and b.exists()):
            print("missing", label, kind)
            continue
        comparison = compare_runs.compare(a, b)
        record = comparison.as_dict()
        record.update(
            {
                "label": label,
                "kind": kind,
                "ref": str(a.relative_to(ROOT)),
                "other": str(b.relative_to(ROOT)),
            }
        )
        out["pairs"].append(record)
        print(label, kind, comparison.verdict)
    placement = {
        name: (run_file("reference", name), run_file("core", name))
        for name in PLACEMENT
    }
    if all(a.exists() and b.exists() for a, b in placement.values()):
        items = compare_runs.compare_placement(placement)
        out["placement"] = [
            {
                "name": i.name,
                "passed": i.passed,
                "detail": compare_runs._plain(i.detail),
            }
            for i in items
        ]
        print("placement", all(i.passed for i in items))
    (RESULTS / "compare.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# The timing


def windows_load() -> list[str]:
    """Windows' busiest processes, the power state and the processor's performance.

    Prompt 90's probe (``probe90.windows_load``), sampled over two seconds through
    ``powershell.exe``; empty when Windows cannot be reached.
    """
    command = (
        "$s = Get-Counter '\\Process(*)\\% Processor Time' -SampleInterval 2 -MaxSamples 1"
        " -ErrorAction SilentlyContinue;"
        " $s.CounterSamples | Where-Object { $_.InstanceName -notin '_total','idle' }"
        " | Sort-Object CookedValue -Descending | Select-Object -First 8"
        " | ForEach-Object { '{0} {1:N0}' -f $_.InstanceName, $_.CookedValue };"
        " 'on mains ' + ((Get-CimInstance Win32_Battery).BatteryStatus -eq 2);"
        " 'processor performance ' + [int](Get-Counter"
        " '\\Processor Information(_Total)\\% Processor Performance')"
        ".CounterSamples[0].CookedValue"
    )
    try:
        done = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def timing_one(side: str, out_path: Path) -> None:
    """One timed run of the dust release on SIDE in this process: its parts apart."""
    import src.time_integration as time_integration
    from src.core_backend import CoreRun

    seconds = {"flow": 0.0, "march": 0.0}

    def timed(kind: str, method: object) -> object:
        def wrapper(*args: object, **kwargs: object) -> object:
            started = time.perf_counter()
            try:
                return method(*args, **kwargs)
            finally:
                seconds[kind] += time.perf_counter() - started

        return wrapper

    time_integration._Run._solve = timed("flow", time_integration._Run._solve)
    time_integration._Run.march = timed("march", time_integration._Run.march)
    CoreRun.march = timed("march", CoreRun.march)
    path = LINUX_RUNS / "timing" / f"{side}.h5"
    fresh(path)
    started = time.perf_counter()
    code, _ = in_process(
        [str(config_path(side, "dust80")), "--out", str(path)],
        out_path.with_suffix(".log"),
    )
    total = time.perf_counter() - started
    record = {
        "side": side,
        "exit": code,
        "total": total,
        "flow": seconds["flow"],
        "march": seconds["march"],
        "rest": total - seconds["flow"] - seconds["march"],
        **environment(),
    }
    out_path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")


def timing(session: int) -> None:
    """One session: each backend once, alone, in alternating order, the state recorded."""
    order = SIDES if session % 2 == 1 else tuple(reversed(SIDES))
    folder = RESULTS / "timing" / f"session_{session}"
    folder.mkdir(parents=True, exist_ok=True)
    state = {
        "before": windows_load(),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    for side in order:
        subprocess.run(
            [
                str(PYTHON),
                "-u",
                str(Path(__file__)),
                "timing-one",
                side,
                str(folder / f"{side}.json"),
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            check=True,
        )
    state["after"] = windows_load()
    state["order"] = list(order)
    (folder / "state.json").write_text(
        json.dumps(state, indent=1) + "\n", encoding="utf-8"
    )
    for side in order:
        record = json.loads((folder / f"{side}.json").read_text())
        print(
            session,
            side,
            f"{record['total']:.1f} s",
            f"flow {record['flow']:.1f}",
            f"march {record['march']:.1f}",
        )


def main(argv: list[str] | None = None) -> int:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command")
    parser.add_argument("args", nargs="*")
    parser.add_argument("--session", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "configs":
        write_configs()
    elif args.command == "run":
        run_side(args.args[0], args.args[1])
    elif args.command == "fields":
        run_fields(args.args[0])
    elif args.command == "prompt91":
        run_prompt91(args.args[0])
    elif args.command == "restart-march":
        restart_march()
    elif args.command == "restart-flow":
        restart_flow()
    elif args.command == "queue":
        return queue()
    elif args.command == "compare":
        compare_all()
    elif args.command == "timing":
        timing(args.session)
    elif args.command == "timing-one":
        timing_one(args.args[0], Path(args.args[1]))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
