"""Prompt 97's probe: the band on 320x120 and the product end to end on each backend.

    python docs/reports/probe97/probe97.py configs
    python docs/reports/probe97/probe97.py run <side> <name>
    python docs/reports/probe97/probe97.py row <case> <method>
    python docs/reports/probe97/probe97.py compare <label>
    python docs/reports/probe97/probe97.py band
    python docs/reports/probe97/probe97.py placement
    python docs/reports/probe97/probe97.py timing <grid> <session>
    python docs/reports/probe97/probe97.py queue
    python docs/reports/probe97/probe97.py status

Run under WSL2 with the Linux environment from the checkout's root, one BLAS
thread and ``NPY_DISABLE_CPU_FEATURES=X86_V4``, through
``docs/reports/probe97/run97.sh`` (which sets both). Every run file is written on
the Linux file system (``~/scratch97/runs/``) and copied into
``results/builder97/runs/<side>/`` when it ends (ADR-017 A), with the run's log
and a record of it beside it.

- ``configs``: each run's scenario file under ``results/builder97/configs/<side>/``.
  On 320x120 main's ``configs/scenario_*.yaml`` as they are (the base is the
  320x120 room), the base made absolute; on 160x60 and 80x30 the grid and
  ``mass_imbalance_tol`` the base file's rule for it (``probe85.mass_imbalance_tol``).
  ``solver.backend: core`` on the core's side; ``transport.diffusion_tol`` times
  0.99 on the side ``diffusion_tol_0.99``.
- ``run``: one scenario through ``run_scenario.main`` in this process. On the side
  ``outer_one_later`` the probe's patch (``probe85.OuterOneLater``) is installed
  first. A run named with a source in ``RUNS`` takes its flow states from that
  run's file (``--flows-from``).
- ``row``: one harness row of record through ``benchmark.main`` on the core, into
  ``results/builder97/rows/``.
- ``compare``: one pair by ``scripts/compare_runs.py`` into
  ``results/builder97/compare/<label>.json``; a 320x120 pair of the backends is
  read with each field's spread set to the 320x120 spread (``band``), its limit
  unchanged.
- ``band``: the 320x120 spread of each field, the largest difference over the
  four variant pairs, into ``results/builder97/compare/band320.json``.
- ``placement``: ``compare_runs.compare_placement`` over the four contaminating
  scenarios on 320x120, both backends.
- ``timing``: the dust release from rest on one grid, each backend once in its own
  process, in alternating order by session, the flow solves, the march and the
  rest timed apart, the machine's state recorded before and after.
- ``queue``: the detached runner. It starts every job whose record is absent and
  whose inputs exist, at most three solver processes at once and at most two on
  320x120, one comparison at a time beside them, and the timing sessions last,
  each alone. After each job it reads the job's record against the prompt's
  stop rules; on a stop it writes ``results/builder97/HALT``, starts nothing
  more and lets the jobs in progress finish. It refuses to start while HALT
  exists. One dated line per start and finish goes to ``queue.log``.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe85"))

import compare_runs  # noqa: E402 -- follows sys.path.insert
import probe85  # noqa: E402 -- follows sys.path.insert

from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert

RESULTS = ROOT / "results" / "builder97"
CONFIGS = RESULTS / "configs"
RUNS = RESULTS / "runs"
COMPARE = RESULTS / "compare"
ROWS = RESULTS / "rows"
TIMING = RESULTS / "timing"
LOGS = RESULTS / "logs"
HALT = RESULTS / "HALT"
LINUX_RUNS = Path.home() / "scratch97" / "runs"
PYTHON = Path.home() / ".venvs" / "cfd_clean_room" / "bin" / "python"
HARNESS = ROOT / "benchmarks" / "results.jsonl"

REF, CORE = "reference", "core"
DIFFUSION, OUTER = "diffusion_tol_0.99", "outer_one_later"
DIFFUSION_FACTOR = 0.99
PRODUCT, MIDDLE, COARSE = "320x120", "160x60", "80x30"
# Short name to main's scenario file.
SCENARIOS = {
    "clean": "scenario_clean_out.yaml",
    "door": "scenario_door_leak.yaml",
    "dust": "scenario_equipment_dust.yaml",
    "breach": "scenario_filter_breach.yaml",
    "seal": "scenario_seal_leak.yaml",
}
# The band's runs (ADR-017 H, "On 320x120") and the placement's scenarios.
BAND_SCENARIOS = ("dust", "door")
BAND_VARIANTS = (DIFFUSION, OUTER)
PLACEMENT = ("door", "dust", "breach", "seal")
# The probe's 160x60 runs.
MIDDLE_SCENARIOS = ("dust", "breach")
# The timing sessions: grid and session numbers, run last, each alone.
TIMING_SESSIONS = (
    *((COARSE, s) for s in (1, 2, 3)),
    *((MIDDLE, s) for s in (1, 2, 3)),
    (PRODUCT, 1),
)
MAX_SOLVERS, MAX_PRODUCT = 3, 2
POLL_SECONDS = 10.0
# Tier-2 cases stay on the reference (ADR-017 F, option 1).
TRANSIENT_PREFIX = "val022_"


def run_name(scenario: str, grid: str) -> str:
    """A run's name: the scenario and the grid."""
    return f"{scenario}_{grid}"


def plan() -> dict[tuple[str, str], tuple[str, str] | None]:
    """Every run as (side, name) to the (side, name) whose flow states it takes, or None.

    On each side the flow states are solved once and shared, as prompt 69 did:
    the dust release and the door leak solve theirs from rest and the other
    scenarios take the dust release's closed room. The variant ``diffusion_tol``
    changes nothing the flow solve reads (``time_integration.FLOW_FREE_SECTIONS``
    holds ``transport``), so it takes the reference's; ``outer_one_later`` changes
    the flow solve and solves its own.
    """
    out: dict[tuple[str, str], tuple[str, str] | None] = {}
    for side in (REF, CORE):
        dust = run_name("dust", PRODUCT)
        out[(side, dust)] = None
        out[(side, run_name("door", PRODUCT))] = None
        for scenario in ("clean", "breach", "seal"):
            out[(side, run_name(scenario, PRODUCT))] = (side, dust)
    for scenario in BAND_SCENARIOS:
        name = run_name(scenario, PRODUCT)
        out[(DIFFUSION, name)] = (REF, name)
        out[(OUTER, name)] = None
    for side in (REF, CORE):
        dust = run_name("dust", MIDDLE)
        out[(side, dust)] = None
        out[(side, run_name("breach", MIDDLE))] = (side, dust)
    return out


# ---------------------------------------------------------------------------
# The scenario files


def mapping_of(scenario: str, grid: str, side: str) -> tuple[dict, list]:
    """Main's scenario file on GRID for SIDE, and what was changed for the record."""
    source = ROOT / "configs" / SCENARIOS[scenario]
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    base_path = (source.parent / raw["scenario"]["base"]).resolve()
    raw["scenario"]["base"] = str(base_path)
    changed: list = []
    if grid != PRODUCT:
        nx, ny = probe85.GRIDS[grid]
        raw["domain"] = {"nx": nx, "ny": ny}
        raw.setdefault("solver", {})["mass_imbalance_tol"] = probe85.mass_imbalance_tol(
            grid
        )
    if side == CORE:
        raw.setdefault("solver", {})["backend"] = "core"
    if side == DIFFUSION:
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        old = float(
            raw.get("transport", {}).get(
                "diffusion_tol", base["transport"]["diffusion_tol"]
            )
        )
        raw.setdefault("transport", {})["diffusion_tol"] = old * DIFFUSION_FACTOR
        changed.append(("transport.diffusion_tol", old, old * DIFFUSION_FACTOR))
    return raw, changed


def config_path(side: str, name: str) -> Path:
    """A run's scenario file."""
    return CONFIGS / side / f"{name}.yaml"


def write_configs() -> None:
    """Every run's scenario file and the timing's, each loaded once to refuse an invalid one."""
    names = set(plan())
    for side in (REF, CORE):
        for grid in (PRODUCT, MIDDLE, COARSE):
            names.add((side, run_name("dust", grid)))
    for side, name in sorted(names):
        scenario, grid = name.split("_", 1)
        raw, changed = mapping_of(scenario, grid, side)
        path = config_path(side, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        path.with_suffix(".changed.json").write_text(json.dumps(changed) + "\n")
        SimConfig(path)
        print("wrote", path)


# ---------------------------------------------------------------------------
# The runs


def commit() -> str:
    """HEAD's hash read from the checkout's files (git is not needed)."""
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


def environment() -> dict:
    """What a record says of its environment."""
    from threadpoolctl import threadpool_info

    return {
        "commit": commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "blas": [
            {k: d.get(k) for k in ("internal_api", "num_threads", "version")}
            for d in threadpool_info()
        ],
    }


def linux_path(side: str, name: str) -> Path:
    """A run's file on the Linux file system."""
    return LINUX_RUNS / side / f"{name}.h5"


def run_file(side: str, name: str) -> Path:
    """A run's copy in the checkout."""
    return RUNS / side / f"{name}.h5"


def run_record(side: str, name: str) -> Path:
    """A run's record, which marks it done."""
    return RUNS / side / f"{name}.json"


def fresh(path: Path) -> None:
    """Remove a run file and its checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for p in (path, path.with_name(path.stem + ".checkpoint.h5")):
        if p.exists():
            p.unlink()


def finish(path: Path, copy_to: Path, name: str, record: dict) -> None:
    """Copy the run file and its checkpoint into the checkout, then the record."""
    copy_to.mkdir(parents=True, exist_ok=True)
    for p in (path, path.with_name(path.stem + ".checkpoint.h5")):
        if p.exists():
            shutil.copy2(p, copy_to / p.name)
    (copy_to / f"{name}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )


def concurrent() -> int:
    """The solver processes beside this one, as the runner counted them."""
    return int(os.environ.get("PROBE97_CONCURRENT", "0"))


def run_one(side: str, name: str) -> int:
    """One scenario through ``run_scenario.main`` in this process; its record."""
    import run_scenario

    source = plan()[(side, name)]
    path = linux_path(side, name)
    fresh(path)
    args = [str(config_path(side, name)), "--out", str(path)]
    args += ["--concurrent", str(concurrent())]
    if source is not None:
        args += ["--flows-from", str(linux_path(*source))]
    later = None
    if side == OUTER:
        later = probe85.OuterOneLater()
        later.install()
    started = time.perf_counter()
    code = run_scenario.main(args)
    wall = time.perf_counter() - started
    record = {
        "side": side,
        "name": name,
        "exit": code,
        "wall_seconds": wall,
        "flows_from": None if source is None else list(source),
        "changed": json.loads(
            config_path(side, name).with_suffix(".changed.json").read_text()
        ),
        "outer_later": None if later is None else later.record(),
        "concurrent": concurrent(),
        "finished": datetime.now().isoformat(timespec="seconds"),
        **environment(),
    }
    finish(path, RUNS / side, name, record)
    print(side, name, "exit", code, f"{wall:.1f} s", flush=True)
    return code


# ---------------------------------------------------------------------------
# The harness rows


def rows_of_record() -> list[tuple[str, str, dict]]:
    """(case, method, row): the latest reference row of each steady case and method.

    The rows of record are read from ``benchmarks/results.jsonl``: per case and
    method the last row the reference made (``benchmark.row_backend``), the
    retired methods (refused by ``run_case``) and the tier-2 cases left out.
    """
    import benchmark

    latest: dict[tuple[str, str], dict] = {}
    for line in HARNESS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if benchmark.row_backend(row) != REF:
            continue
        if row["method"] not in benchmark.METHODS:
            continue
        if row["case"].startswith(TRANSIENT_PREFIX):
            continue
        latest[(row["case"], row["method"])] = row
    return [(case, method, row) for (case, method), row in latest.items()]


def row_path(case: str, method: str) -> Path:
    """A core row's file."""
    return ROWS / f"{case}__{method}.jsonl"


def row_one(case: str, method: str) -> int:
    """One harness row on the core through ``benchmark.main``, and its record."""
    import benchmark

    out = row_path(case, method)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    started = time.perf_counter()
    code = benchmark.main(
        [
            "--cases",
            case,
            "--method",
            method,
            "--backend",
            CORE,
            "--repeats",
            "1",
            "--concurrent",
            str(concurrent() + 1),
            "--results",
            str(out),
        ]
    )
    wall = time.perf_counter() - started
    stored = {(c, m): r for c, m, r in rows_of_record()}[(case, method)]
    core = json.loads(out.read_text(encoding="utf-8").splitlines()[-1])
    record = {
        "case": case,
        "method": method,
        "exit": code,
        "wall_seconds": wall,
        "stored_run_id": stored["run_id"],
        "stored": summary_of_row(stored),
        "core": summary_of_row(core),
        "finished": datetime.now().isoformat(timespec="seconds"),
        **environment(),
    }
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    print(case, method, "exit", code, f"{wall:.1f} s", flush=True)
    return code


def summary_of_row(row: dict) -> dict:
    """What the report sets beside each other of a harness row."""
    return {
        "backend": row.get("params", {}).get("backend", REF),
        "commit": row["git_commit"],
        "dirty": row["git_dirty"],
        "recorded_at": row["recorded_at"],
        "passed": row["accuracy"].get("passed"),
        "value": row["accuracy"].get("value"),
        "metric": row["accuracy"].get("metric"),
        "stop_reason": row["outcome"].get("stop_reason"),
        "converged": row["outcome"].get("converged"),
        "outer": row["work"].get("outer_iterations"),
        "inner": row["work"].get("inner_iterations"),
        "wall_seconds": row["time"]["wall_seconds"],
        "concurrent": row["environment"].get("concurrent_processes"),
        "os": row["environment"].get("os"),
    }


# ---------------------------------------------------------------------------
# The comparisons


@dataclass(frozen=True)
class Pair:
    """One comparison: its label, its kind and the two runs (side, name)."""

    label: str
    kind: str
    ref: tuple[str, str]
    other: tuple[str, str]


def pairs() -> list[Pair]:
    """Every pair compared, the band's first."""
    out = []
    for scenario in BAND_SCENARIOS:
        name = run_name(scenario, PRODUCT)
        for variant in BAND_VARIANTS:
            out.append(
                Pair(f"band_{name}_{variant}", "band", (REF, name), (variant, name))
            )
    for scenario in SCENARIOS:
        name = run_name(scenario, PRODUCT)
        out.append(
            Pair(f"backends_{name}", "backends 320x120", (REF, name), (CORE, name))
        )
    for scenario in MIDDLE_SCENARIOS:
        name = run_name(scenario, MIDDLE)
        out.append(
            Pair(f"backends_{name}", "backends 160x60", (REF, name), (CORE, name))
        )
    return out


def compare_path(label: str) -> Path:
    """A comparison's record."""
    return COMPARE / f"{label}.json"


BAND_RECORD = COMPARE / "band320.json"
PLACEMENT_RECORD = COMPARE / "placement.json"


def band_key(item_name: str) -> str:
    """The key of ``compare_runs.BAND`` a field item is read against."""
    head = item_name.split(",")[0]
    if head in compare_runs.FLOW_FIELDS:
        return head
    if item_name.startswith("concentration, every"):
        return compare_runs.CONCENTRATION
    if item_name.startswith("concentration, pointwise"):
        return compare_runs.POINTWISE
    if item_name.startswith("sensor channels"):
        return compare_runs.SENSORS
    if item_name.startswith("leftover"):
        return compare_runs.LEFTOVER
    raise ValueError(f"no band for the item {item_name!r}")


def compare_one(label: str) -> int:
    """One pair by the comparator; a 320x120 pair of the backends on the 320x120 spread."""
    pair = {p.label: p for p in pairs()}[label]
    saved = dict(compare_runs.BAND)
    band = None
    if pair.kind == "backends 320x120":
        band = json.loads(BAND_RECORD.read_text())["band"]
        for key, (_, limit) in saved.items():
            compare_runs.BAND[key] = (float(band[key]), limit)
    try:
        started = time.perf_counter()
        comparison = compare_runs.compare(run_file(*pair.ref), run_file(*pair.other))
        seconds = time.perf_counter() - started
    finally:
        compare_runs.BAND.clear()
        compare_runs.BAND.update(saved)
    record = comparison.as_dict()
    record.update(
        {
            "label": label,
            "kind": pair.kind,
            "ref": str(run_file(*pair.ref).relative_to(ROOT)),
            "other": str(run_file(*pair.other).relative_to(ROOT)),
            "band": band,
            "seconds": seconds,
            "finished": datetime.now().isoformat(timespec="seconds"),
            **environment(),
        }
    )
    COMPARE.mkdir(parents=True, exist_ok=True)
    compare_path(label).write_text(
        json.dumps(compare_runs._plain(record), indent=1) + "\n"
    )
    print(label, comparison.verdict, flush=True)
    return 0


def band() -> int:
    """The 320x120 spread of each field: the largest difference over the variant pairs.

    The flow fields share one spread, the largest over the six and every flow
    state, as P8 took the probe's; each class field has its own.
    """
    keys = list(compare_runs.BAND)
    spreads = dict.fromkeys(keys, 0.0)
    per_pair: dict[str, dict[str, float]] = {}
    for pair in pairs():
        if pair.kind != "band":
            continue
        record = json.loads(compare_path(pair.label).read_text())
        mine = dict.fromkeys(keys, 0.0)
        for item in record["items"]:
            if item["criterion"] != "field":
                continue
            key = band_key(item["name"])
            mine[key] = max(mine[key], float(item["detail"]["difference"]))
        per_pair[pair.label] = mine
        for key in keys:
            spreads[key] = max(spreads[key], mine[key])
    flow = max(spreads[name] for name in compare_runs.FLOW_FIELDS)
    for name in compare_runs.FLOW_FIELDS:
        spreads[name] = flow
    out = {
        "band": spreads,
        "per_pair": per_pair,
        "limits": {k: v[1] for k, v in compare_runs.BAND.items()},
        "probe": {k: v[0] for k, v in compare_runs.BAND.items()},
        "finished": datetime.now().isoformat(timespec="seconds"),
        **environment(),
    }
    BAND_RECORD.write_text(json.dumps(compare_runs._plain(out), indent=1) + "\n")
    print("band", spreads, flush=True)
    return 0


def placement() -> int:
    """``compare_placement`` over the four contaminating scenarios on 320x120."""
    runs = {
        name: (
            run_file(REF, run_name(name, PRODUCT)),
            run_file(CORE, run_name(name, PRODUCT)),
        )
        for name in PLACEMENT
    }
    started = time.perf_counter()
    items = compare_runs.compare_placement(runs)
    record = {
        "items": [
            {
                "name": i.name,
                "passed": i.passed,
                "detail": compare_runs._plain(i.detail),
            }
            for i in items
        ],
        "seconds": time.perf_counter() - started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        **environment(),
    }
    PLACEMENT_RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print("placement", all(i.passed for i in items), flush=True)
    return 0


# ---------------------------------------------------------------------------
# The timing


def windows_load() -> list[str]:
    """Windows' busiest processes, the power state and the processor's performance.

    Prompt 96's probe (``probe96.windows_load``); empty when Windows cannot be
    reached.
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
        ".CounterSamples[0].CookedValue;"
        " 'model servers ' + @(Get-Process -ErrorAction SilentlyContinue"
        " | Where-Object { $_.Name -match 'ollama|llama' }).Count"
    )
    try:
        done = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def timing_path(grid: str, session: int) -> Path:
    """A session's folder."""
    return TIMING / f"{grid}_session_{session}"


def timing_one(side: str, grid: str, out_path: Path) -> int:
    """One timed run of the dust release on GRID and SIDE in this process: its parts apart."""
    import run_scenario

    import src.time_integration as time_integration
    from src.time_integration import CoreRun

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
    name = run_name("dust", grid)
    path = LINUX_RUNS / "timing" / f"{grid}_{side}.h5"
    fresh(path)
    started = time.perf_counter()
    code = run_scenario.main(
        [str(config_path(side, name)), "--out", str(path), "--quiet"]
    )
    total = time.perf_counter() - started
    record = {
        "side": side,
        "grid": grid,
        "exit": code,
        "total": total,
        "flow": seconds["flow"],
        "march": seconds["march"],
        "rest": total - seconds["flow"] - seconds["march"],
        "finished": datetime.now().isoformat(timespec="seconds"),
        **environment(),
    }
    out_path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return code


def timing(grid: str, session: int) -> int:
    """One session: each backend once, alone, in alternating order, the state recorded."""
    order = (REF, CORE) if session % 2 == 1 else (CORE, REF)
    folder = timing_path(grid, session)
    folder.mkdir(parents=True, exist_ok=True)
    state = {
        "before": windows_load(),
        "started": datetime.now().isoformat(timespec="seconds"),
        "order": list(order),
    }
    for side in order:
        subprocess.run(
            [
                str(PYTHON),
                "-u",
                str(Path(__file__)),
                "timing-one",
                side,
                grid,
                str(folder / f"{side}.json"),
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=(folder / f"{side}.log").open("w"),
            stderr=subprocess.STDOUT,
            check=True,
        )
    state["after"] = windows_load()
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    (folder / "state.json").write_text(json.dumps(state, indent=1) + "\n")
    for side in order:
        record = json.loads((folder / f"{side}.json").read_text())
        print(grid, session, side, f"{record['total']:.1f} s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# The stop rules


def run_halt(side: str, name: str) -> str | None:
    """A run that did not finish: every run of the step is expected to."""
    record = json.loads(run_record(side, name).read_text())
    if record["exit"] != 0:
        return f"run {side}/{name} exited {record['exit']}"
    return None


def compare_halt(label: str) -> str | None:
    """The prompt's stops read from a comparison's record.

    A band pair stops on a spread beyond its limit; a pair of the backends on
    any failing item (a decision, a field beyond its limit or in the middle tier
    unexplained, a stop more than one apart or at the cap on one side only, a
    budget that does not close or whose verdict differs), which are the
    comparator's failing kinds.
    """
    record = json.loads(compare_path(label).read_text())
    if record["kind"] == "band":
        beyond = [
            item["name"]
            for item in record["items"]
            if item["criterion"] == "field"
            and item["detail"]["tier"] == compare_runs.BEYOND
        ]
        return f"{label}: beyond the limit: {beyond}" if beyond else None
    failed = [item["name"] for item in record["items"] if not item["passed"]]
    return f"{label}: {record['verdict']}: {failed}" if failed else None


def band_halt() -> str | None:
    """A 320x120 spread beyond its limit."""
    record = json.loads(BAND_RECORD.read_text())
    beyond = [k for k, v in record["band"].items() if v > record["limits"][k]]
    return f"band320: beyond the limit: {beyond}" if beyond else None


def placement_halt() -> str | None:
    """A placement decision that differs."""
    record = json.loads(PLACEMENT_RECORD.read_text())
    failed = [item["name"] for item in record["items"] if not item["passed"]]
    return f"placement: {failed}" if failed else None


def row_halt(case: str, method: str) -> str | None:
    """A validation row whose verdict changes, or that did not run."""
    path = row_path(case, method).with_suffix(".json")
    record = json.loads(path.read_text())
    stored, core = record["stored"], record["core"]
    if record["exit"] != 0:
        return f"row {case} {method} exited {record['exit']}"
    if (
        stored["passed"] != core["passed"]
        or stored["stop_reason"] != core["stop_reason"]
    ):
        return f"row {case} {method}: stored {stored['passed']}/{stored['stop_reason']}, core {core['passed']}/{core['stop_reason']}"
    return None


# ---------------------------------------------------------------------------
# The runner


@dataclass
class Job:
    """One job: its arguments, the record that marks it done, what it needs."""

    name: str
    args: list[str]
    done: Path
    needs: list[Path] = field(default_factory=list)
    kind: str = "solver"
    product: bool = False
    halt: object = None


def jobs() -> list[Job]:
    """Every job, in the order the runner prefers them."""
    out: list[Job] = []
    the_plan = plan()

    def run_job(side: str, name: str) -> Job:
        source = the_plan[(side, name)]
        return Job(
            f"run_{side}_{name}",
            ["run", side, name],
            run_record(side, name),
            [] if source is None else [run_record(*source)],
            product=name.endswith(PRODUCT),
            halt=lambda s=side, n=name: run_halt(s, n),
        )

    dust, door = run_name("dust", PRODUCT), run_name("door", PRODUCT)
    order = [
        (REF, dust),
        (REF, door),
        (OUTER, dust),
        (OUTER, door),
        (DIFFUSION, door),
        (DIFFUSION, dust),
        (CORE, dust),
        (CORE, door),
        (REF, run_name("clean", PRODUCT)),
        (REF, run_name("breach", PRODUCT)),
        (REF, run_name("seal", PRODUCT)),
        (CORE, run_name("clean", PRODUCT)),
        (CORE, run_name("breach", PRODUCT)),
        (CORE, run_name("seal", PRODUCT)),
    ]
    assert set(order) == {k for k in the_plan if k[1].endswith(PRODUCT)}
    out.extend(run_job(side, name) for side, name in order)
    # The third slot: the harness rows on the core, longest first, then 160x60.
    rows = sorted(rows_of_record(), key=lambda r: -float(r[2]["time"]["wall_seconds"]))
    for case, method, row in rows:
        out.append(
            Job(
                f"row_{case}_{method}",
                ["row", case, method],
                row_path(case, method).with_suffix(".json"),
                product=(row["grid"]["nx"], row["grid"]["ny"]) == (320, 120),
                halt=lambda c=case, m=method: row_halt(c, m),
            )
        )
    for scenario in MIDDLE_SCENARIOS:
        for side in (REF, CORE):
            out.append(run_job(side, run_name(scenario, MIDDLE)))
    for pair in pairs():
        needs = [run_record(*pair.ref), run_record(*pair.other)]
        if pair.kind == "backends 320x120":
            needs.append(BAND_RECORD)
        out.append(
            Job(
                f"compare_{pair.label}",
                ["compare", pair.label],
                compare_path(pair.label),
                needs,
                kind="compare",
                halt=lambda label=pair.label: compare_halt(label),
            )
        )
    out.append(
        Job(
            "band",
            ["band"],
            BAND_RECORD,
            [compare_path(p.label) for p in pairs() if p.kind == "band"],
            kind="compare",
            halt=band_halt,
        )
    )
    out.append(
        Job(
            "placement",
            ["placement"],
            PLACEMENT_RECORD,
            [
                run_record(side, run_name(name, PRODUCT))
                for name in PLACEMENT
                for side in (REF, CORE)
            ],
            kind="compare",
            halt=placement_halt,
        )
    )
    for grid, session in TIMING_SESSIONS:
        out.append(
            Job(
                f"timing_{grid}_{session}",
                ["timing", grid, str(session)],
                timing_path(grid, session) / "state.json",
                kind="timing",
            )
        )
    return out


def log_line(text: str) -> None:
    """One dated line to the queue's log and to stdout."""
    line = f"{datetime.now().isoformat(timespec='seconds')} {text}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "queue.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def queue() -> int:
    """Start every job not done, within the limits; stop on the prompt's stop rules."""
    if HALT.exists():
        log_line(f"refused: {HALT} exists: {HALT.read_text().strip()}")
        return 1
    every = jobs()
    waiting = [job for job in every if not job.done.exists()]
    running: dict[str, tuple[Job, subprocess.Popen]] = {}
    failed: list[str] = []
    halted: str | None = None
    LOGS.mkdir(parents=True, exist_ok=True)
    log_line(f"queue at {commit()[:12]}: {len(waiting)} of {len(every)} jobs to do")

    def count(kind: str, product: bool | None = None) -> int:
        return sum(
            1
            for job, _ in running.values()
            if job.kind == kind and (product is None or job.product == product)
        )

    while waiting or running:
        started = 0
        if halted is None:
            for job in list(waiting):
                if not all(p.exists() for p in job.needs):
                    continue
                if job.kind == "timing":
                    # Last, and alone: every other job finished.
                    if running or any(j.kind != "timing" for j in waiting):
                        continue
                elif any(j.kind == "timing" for j, _ in running.values()):
                    continue
                if job.kind == "solver":
                    if count("solver") >= MAX_SOLVERS:
                        continue
                    if job.product and count("solver", True) >= MAX_PRODUCT:
                        continue
                if job.kind == "compare" and count("compare") >= 1:
                    continue
                environ = os.environ.copy()
                environ["PROBE97_CONCURRENT"] = str(count("solver"))
                with (LOGS / f"{job.name}.log").open("w") as out:
                    proc = subprocess.Popen(
                        [str(PYTHON), "-u", str(Path(__file__)), *job.args],
                        cwd=ROOT,
                        env=environ,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                    )
                running[job.name] = (job, proc)
                waiting.remove(job)
                started += 1
                log_line(f"start {job.name} pid {proc.pid}")
                if job.kind == "timing":
                    break
            if not started and not running and waiting:
                log_line(f"blocked: {[j.name for j in waiting]}")
                break
        elif not running:
            break
        time.sleep(POLL_SECONDS)
        for name, (job, proc) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[name]
            reason = None
            if not job.done.exists():
                reason = f"{name} exited {code} without its record"
            elif job.halt is not None:
                try:
                    reason = job.halt()
                except Exception as err:
                    # A stop rule that cannot be read is a stop.
                    reason = f"{name}: the stop rules could not be read: {err!r}"
            if code != 0 or reason:
                failed.append(name)
            log_line(f"{'done' if code == 0 else 'failed'} {name} exit {code}")
            if reason and halted is None:
                halted = reason
                HALT.write_text(reason + "\n")
                log_line(f"HALT: {reason}; starting nothing more")
    log_line(
        f"queue ended; failed: {failed if failed else 'none'}; "
        f"halt: {halted if halted else 'none'}; left: {[j.name for j in waiting]}"
    )
    return 1 if failed or halted else 0


def status() -> int:
    """Each job's state."""
    for job in jobs():
        state = (
            "done"
            if job.done.exists()
            else ("ready" if all(p.exists() for p in job.needs) else "waiting")
        )
        print(f"{state:8s} {job.kind:8s} {job.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command")
    parser.add_argument("args", nargs="*")
    args = parser.parse_args(argv)
    a = args.args
    if args.command == "configs":
        write_configs()
        return 0
    if args.command == "run":
        return run_one(a[0], a[1])
    if args.command == "row":
        return row_one(a[0], a[1])
    if args.command == "compare":
        return compare_one(a[0])
    if args.command == "band":
        return band()
    if args.command == "placement":
        return placement()
    if args.command == "timing":
        return timing(a[0], int(a[1]))
    if args.command == "timing-one":
        return timing_one(a[0], a[1], Path(a[2]))
    if args.command == "queue":
        return queue()
    if args.command == "status":
        return status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
