"""The tables of prompt 81, ISO cumulative counting built in, from its records.

    python docs/reports/probe81/tables81.py            # write the tables, splice the report
    python docs/reports/probe81/tables81.py --check    # regenerate and compare

Reads ``results/builder81/`` (untracked, this machine): the queue file and its
log, the standby intervals (``standby81.ps1``), each job's run file, summary
and log under ``product/`` and ``placement/``, the plants' record
(``plants.json``, ``plant81.py``) and the suite's (``suite.xml``); prompt 79's
runs (``results/builder79/product/``, ``placement/`` and ``sweep/``, the
sweep not run again) and queue; and prompt 80's probe maps
(``results/builder80/placement/<run>.channels.npz``).

The code that reads every threshold is this branch's ``src/``: the runs were
made by it, and every map, clear time and score below is read through it,
with two exceptions written out here so that a reading of the code is not
checked against itself: the per-class reading of prompt 79's runs (its
``/arrival/`` maps, flagged per-class, and each class alone at the frames,
``probe80.frame_clear``), and prompt 80's brackets, which ``tables80.py``'s
own code rebuilds from prompt 79's records.

It writes four files beside it:

- ``rescored69.md`` and ``rescored76.md``: the tables of ``tables69.py`` and
  ``tables76.py`` made by their own code on this pass's runs, as prompt 79's
  ``tables79.py`` made them (``rescore``); the sweep's arm runs (prompt 79's)
  are read from their frames on the channels for tables76's K2, as the prompt
  asks. Their committed outputs and prompt 79's re-scores are not touched.
- ``changes81.md``: every table of prompt 79's two re-scores against this
  pass's, cell by cell (``tables79.compare``).
- ``tables81.md``: this pass's own tables, each also copied into
  ``docs/reports/cumulative_build.md`` between ``<!-- tables81 NAME begin/end
  -->``.

``--check`` regenerates all of them and compares, the report's copies too.
"""

import argparse
import importlib.util
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPORTS = ROOT / "docs" / "reports"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPORTS / "probe79"))
sys.path.insert(0, str(REPORTS / "probe80"))

import run81  # noqa: E402 -- follows sys.path.insert

from src.config import WORST_CASE, SimConfig  # noqa: E402 -- follows sys.path.insert
from src.io_manager import (  # noqa: E402 -- follows sys.path.insert
    CUMULATIVE,
    PER_CLASS,
    RunFile,
)
from src.mesh import SOLID, Mesh  # noqa: E402 -- follows sys.path.insert
from src.monitor import (  # noqa: E402 -- follows sys.path.insert
    channel_sum,
    sensor_cells,
)
from src.placement import (  # noqa: E402 -- follows sys.path.insert
    FRAME_RESOLVED,
    STEP_RESOLVED,
    RunArrival,
    arrival_from_frames,
    arrival_from_run,
    clear_score,
    clear_units,
)

RESULTS = run81.RESULTS
BEFORE = run81.BEFORE
PROBE80 = ROOT / "results" / "builder80" / "placement"
SWEEP = BEFORE / "sweep"
QUEUE = run81.QUEUE
STANDBY = RESULTS / "standby.json"
PLANTS = RESULTS / "plants.json"
SUITE = RESULTS / "suite.xml"
TABLES = HERE / "tables81.md"
CHANGES = HERE / "changes81.md"
RESCORED = {"69": HERE / "rescored69.md", "76": HERE / "rescored76.md"}
RESCORED_79 = {
    "69": REPORTS / "probe79" / "rescored69.md",
    "76": REPORTS / "probe79" / "rescored76.md",
}
REPORT = REPORTS / "cumulative_build.md"
SIZES_UM = ("0.1", "0.3", "0.5", "1", "5")
WALL = re.compile(r"^wall time: ([\d.]+) s", re.MULTILINE)
SOLVING = re.compile(r"flow state (\w+): solving")
# Prediction (c)'s window on the product layout's worst case, seconds.
SUBTRACTION = 1e-9
WORST_WINDOW = (2.5616, 2.565)
EVENT_NAMES = ("door_leak", "filter_breach", "equipment_dust", "seal_leak")
TITLES = {
    "door_leak": "door",
    "filter_breach": "breach",
    "equipment_dust": "dust",
    "seal_leak": "seal leak",
    "clean_out": "clean-out",
    "door": "door",
    "breach": "breach",
    "dust": "dust",
}
# The datasets and attributes this pass is expected to change: the readings of
# a threshold (prompt 81's build) and the run's provenance. Everything else in
# a run file must equal prompt 79's.
EXPECTED_TO_DIFFER = (
    "/arrival/first_above",
    "/arrival@counting",
    "/transport/alerts/",
    "/transport/alerts@",
    "/transport/sensors/channels",
    "/recovery/recovery_time",
    "/recovery/local_recovery",
    "/recovery/local_recovery_max",
    "/recovery@room_recovery_time",
    "/recovery@counting",
    "/config@commit",
    "/config@started",
    "/config@machine",
    "/config@concurrent",
)


def load_module(path: Path, name: str) -> ModuleType:
    """Import a probe script by path under a name of its own."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tables79 = load_module(REPORTS / "probe79" / "tables79.py", "tables79")
tables80 = load_module(REPORTS / "probe80" / "tables80.py", "tables80")
probe80 = sys.modules["probe80"]


def t3(value: float) -> str:
    """Seconds to three decimals, or inf."""
    return "inf" if math.isinf(value) else f"{value:.3f}"


def t4(value: float) -> str:
    """Seconds to four decimals, inf, or not clear."""
    if math.isnan(value):
        return "-"
    return "inf" if math.isinf(value) else f"{value:.4f}"


def yes(flag: bool) -> str:
    """yes or no."""
    return "yes" if flag else "no"


def rel(path: Path) -> str:
    """A path relative to the repository, with forward slashes."""
    return path.resolve().relative_to(ROOT).as_posix()


def run_paths() -> dict[str, tuple[Path, Path, str]]:
    """Job name to (this pass's run file, prompt 79's, grid)."""
    return {
        job["name"]: (job["out"], job["before"], job["grid"]) for job in run81.jobs()
    }


def label_of(sizes: list[float], label: tuple) -> str:
    """A unit as text."""
    name, _, k = label
    return f"{TITLES[name]}, {sizes[k] * 1e6:g} um"


# ---------------------------------------------------------------------------
# The queue and the cost
# ---------------------------------------------------------------------------


def standby_intervals() -> list[tuple[datetime, datetime]]:
    """The standby intervals standby81.ps1 recorded."""
    if not STANDBY.exists():
        return []
    rows = json.loads(STANDBY.read_text(encoding="utf-8-sig"))
    if isinstance(rows, dict):
        rows = [rows]
    return [
        (datetime.fromisoformat(r["entered"]), datetime.fromisoformat(r["exited"]))
        for r in rows
    ]


def standby_seconds(attempts: list[dict]) -> float:
    """Seconds of standby inside the attempts."""
    total = 0.0
    for attempt in attempts:
        if not attempt.get("ended"):
            continue
        start = datetime.fromisoformat(attempt["started"])
        end = datetime.fromisoformat(attempt["ended"])
        for entered, exited in standby_intervals():
            total += max((min(end, exited) - max(start, entered)).total_seconds(), 0.0)
    return total


def old_march_seconds(name: str) -> tuple[float | None, float]:
    """Prompt 79's march wall for a run, less its standby, and the standby taken out."""
    for job in tables79.jobs():
        if job["name"] == name and job["group"] in ("P", "L"):
            march = tables79.march_seconds(job["summary"])
            standby = tables79.standby_seconds(job)
            return (None if march is None else march - standby), standby
    return None, 0.0


def queue_table() -> list[str]:
    """Q1: every job's outcome, attempts, wall and memory, beside prompt 79's wall."""
    queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    lines = [
        "### Q1. The queue: each job's outcome, its attempts, its march wall less the machine's standby, beside prompt 79's run of it, and its peak working set",
        "",
        "Every flow state was copied in from prompt 79's run file (`--flows-from`), so a job's wall is its march's. Standby is the machine's Modern Standby inside the job's attempts (standby81.ps1, the System event log), which the summary's clock ran through. The runner held at most three processes, at most two on 320x120, one BLAS thread each.",
        "",
        "| Run | Grid | Outcome | Attempts | Started | Ended | Wall, summary (s) | Standby taken out (s) | March wall (s) | Prompt 79's march wall (s) | Ratio | Processes beside it at its start | Peak working set (GB) | Flow states solved |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for job in run81.jobs():
        record = queue["jobs"].get(
            job["name"], {"status": "not queued", "attempts": []}
        )
        attempts = record["attempts"]
        summary = tables79.read_text(job["out"].with_suffix(".txt"))
        log = tables79.read_text(job["out"].with_suffix(".log"))
        wall = WALL.search(summary)
        standby = standby_seconds(attempts)
        march = None if wall is None else float(wall.group(1)) - standby
        old, _ = old_march_seconds(job["name"])
        ratio = "-" if march is None or not old else f"{march / old:.3f}"
        peak = (
            max((a.get("peak_working_set") or 0) for a in attempts) if attempts else 0
        )
        first, last = (attempts[0], attempts[-1]) if attempts else ({}, {})
        lines.append(
            f"| {job['name']} | {job['grid']} | {record['status']} | {len(attempts)} "
            f"({', '.join(a['mode'] for a in attempts)}) | {first.get('started', '-')} | "
            f"{last.get('ended') or '-'} | {'-' if wall is None else wall.group(1)} | "
            f"{standby:.0f} | {'-' if march is None else f'{march:.0f}'} | "
            f"{'-' if old is None else f'{old:.0f}'} | {ratio} | "
            f"{first.get('concurrent', '-')} | {peak / 2**30:.2f} | "
            f"{len(SOLVING.findall(log))} |"
        )
    starts = [
        datetime.fromisoformat(a["started"])
        for r in queue["jobs"].values()
        for a in r["attempts"]
    ]
    ends = [
        datetime.fromisoformat(a["ended"])
        for r in queue["jobs"].values()
        for a in r["attempts"]
        if a.get("ended")
    ]
    if starts and ends:
        span = (max(ends) - min(starts)).total_seconds()
        stand = sum(
            max((min(max(ends), b) - max(min(starts), a)).total_seconds(), 0.0)
            for a, b in standby_intervals()
        )
        lines += [
            "",
            f"From the first start to the last end: {span / 3600:.2f} h of wall clock, "
            f"{stand / 3600:.2f} h of it in standby, {(span - stand) / 3600:.2f} h awake. "
            f"The lowest available memory at a start: {min_memory(queue):.0%} of the total "
            "(the floor 25%).",
        ]
    return lines


def min_memory(queue: dict) -> float:
    """The least available fraction of physical memory at any start."""
    out = 1.0
    for record in queue["jobs"].values():
        for a in record["attempts"]:
            if a.get("total_bytes"):
                out = min(out, a["avail_bytes"] / a["total_bytes"])
    return out


# ---------------------------------------------------------------------------
# Prediction (a): every run file against prompt 79's
# ---------------------------------------------------------------------------


def _items(handle: h5py.File) -> dict[str, object]:
    """Every dataset (by path) and attribute (path@name) of a file."""
    out: dict[str, object] = {}

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            out["/" + name] = obj
        for key, value in obj.attrs.items():
            out[f"/{name}@{key}"] = value

    for key, value in handle.attrs.items():
        out[f"/@{key}"] = value
    handle.visititems(visit)
    return out


def _same(a: object, b: object) -> bool:
    """Two datasets (read a row at a time) or two attributes, bit for bit."""
    if isinstance(a, h5py.Dataset) and isinstance(b, h5py.Dataset):
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.ndim == 0 or a.shape[0] == 0:
            return np.asarray(a[()]).tobytes() == np.asarray(b[()]).tobytes()
        return all(
            np.asarray(a[n]).tobytes() == np.asarray(b[n]).tobytes()
            for n in range(a.shape[0])
        )
    if isinstance(a, h5py.Dataset) or isinstance(b, h5py.Dataset):
        return False
    x, y = np.asarray(a), np.asarray(b)
    if x.dtype.kind in "OUS" or y.dtype.kind in "OUS":
        # Strings come back as object arrays, whose bytes are pointers.
        return x.shape == y.shape and x.tolist() == y.tolist()
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def _expected(key: str) -> bool:
    """Whether a dataset or attribute is one this pass is expected to change."""
    return any(key.startswith(p) or key == p for p in EXPECTED_TO_DIFFER)


def file_comparison(new: Path, old: Path) -> dict[str, list[str]]:
    """Every dataset and attribute of two run files: equal, differing, on one side only."""
    out: dict[str, list[str]] = {"equal": [], "differ": [], "new": [], "old": []}
    with h5py.File(new, "r") as a, h5py.File(old, "r") as b:
        ia, ib = _items(a), _items(b)
        for key in sorted(set(ia) | set(ib)):
            if key not in ib:
                out["new"].append(key)
            elif key not in ia:
                out["old"].append(key)
            elif _same(ia[key], ib[key]):
                out["equal"].append(key)
            else:
                out["differ"].append(key)
    return out


def _part(keys: list[str], prefix: str) -> list[str]:
    return [k for k in keys if k.startswith(prefix)]


def bitwise_table(paths: dict) -> tuple[list[str], dict[str, bool], dict]:
    """F1: per run, the fields, budgets, flow states and readings against prompt 79's."""
    lines = [
        "### F1. Every run file against prompt 79's, dataset by dataset and attribute by attribute, bit for bit (prediction (a))",
        "",
        "Fields are /transport/C_k, steps and times; budgets /transport/budgets; flow states every dataset and attribute under /flow/; readings each class's own, /transport/sensors/readings, steps and times. The datasets this pass is expected to change are the readings of a threshold (/arrival/, /transport/alerts/, the clean-out's within-spec times and local field in /recovery/), the new channel series /transport/sensors/channels, and the run's provenance in /config.",
        "",
        "| Run | Grid | Fields | Budgets | Flow states | Per-class readings | Datasets and attributes compared | Equal | Differ, expected | Differ, not expected | Only in this pass's file | Only in prompt 79's |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    verdict: dict[str, bool] = {}
    detail: dict[str, dict] = {}
    for name, (new, old, grid) in paths.items():
        if not new.exists() or not tables79.WALL.search(
            tables79.read_text(new.with_suffix(".txt"))
        ):
            lines.append(f"| {name} | {grid} | not run | | | | | | | | | |")
            verdict[name] = False
            continue
        c = file_comparison(new, old)
        detail[name] = c
        bad = c["differ"] + c["new"] + c["old"]

        def clean(prefixes: tuple[str, ...], bad: list[str] = bad) -> bool:
            return not any(k.startswith(prefixes) for k in bad)

        fields = clean(("/transport/C_", "/transport/steps", "/transport/times"))
        budgets = clean(("/transport/budgets",))
        flows = clean(("/flow",))
        readings = clean(
            (
                "/transport/sensors/readings",
                "/transport/sensors/steps",
                "/transport/sensors/times",
            )
        )
        verdict[name] = fields and budgets and flows
        expected = [k for k in c["differ"] + c["new"] if _expected(k)]
        unexpected = [k for k in c["differ"] + c["new"] + c["old"] if not _expected(k)]
        total = sum(len(v) for v in c.values())
        lines.append(
            f"| {name} | {grid} | {yes(fields)} | {yes(budgets)} | {yes(flows)} | "
            f"{yes(readings)} | {total} | {len(c['equal'])} | {len(expected)} | "
            f"{len(unexpected)}{': ' + ', '.join(unexpected) if unexpected else ''} | "
            f"{len(c['new'])} | {len(c['old'])} |"
        )
    names = sorted({k for c in detail.values() for k in c["differ"] + c["new"]})
    lines += [
        "",
        "What differs, in any run: " + ", ".join(f"`{k}`" for k in names) + ".",
    ]
    return lines, verdict, detail


# ---------------------------------------------------------------------------
# Prediction (b) and the structure: the maps
# ---------------------------------------------------------------------------


def probe_table(paths: dict) -> tuple[list[str], bool]:
    """B1: the 80x30 maps against prompt 80's probe maps, bit for bit."""
    lines = [
        "### B1. The 80x30 cumulative maps of this pass's runs (/arrival/, made by the march) against prompt 80's probe maps (results/builder80/placement/RUN.channels.npz, made by its wrapper on prompt 79's march), bit for bit (prediction (b))",
        "",
        "| Run | /arrival/ counting | Maps equal, bit for bit | Probe's per-class maps = prompt 79's /arrival/ | t_final equal |",
        "|---|---|---|---|---|",
    ]
    ok = True
    for name in run81.PLACEMENT:
        new, old, _ = paths[name]
        if not new.exists():
            lines.append(f"| {name} | not run | | | |")
            ok = False
            continue
        record = RunFile(new).read_arrival()
        saved = np.load(PROBE80 / f"{name}.channels.npz")
        live = Mesh(tables79.config_of(new)).cell_type != SOLID
        maps = {}
        for c in (CUMULATIVE, PER_CLASS):
            m = np.where(np.isnan(saved[c]), np.inf, saved[c])
            m[:, :, ~live] = np.nan
            maps[c] = m
        same = maps[CUMULATIVE].tobytes() == record.first_above.tobytes()
        old_maps = RunFile(old).read_arrival().first_above
        pc = maps[PER_CLASS].tobytes() == old_maps.tobytes()
        tf = float(saved["t_final"]) == record.t_final
        ok = ok and same and tf
        lines.append(
            f"| {name} | {record.counting} | {yes(same)} | {yes(pc)} | {yes(tf)} |"
        )
    return lines, ok


def structure_table(paths: dict) -> tuple[list[str], bool]:
    """S1: this pass's maps against prompt 79's per-class maps, every entry."""
    lines = [
        "### S1. Every run's cumulative step maps (this pass's /arrival/) against its per-class step maps (prompt 79's /arrival/, read flagged per-class), every event, channel and non-SOLID cell (the stop condition: a cumulative arrival later than the per-class one)",
        "",
        "| Run | Grid | Prompt 79's maps read as | Entries | Cumulative earlier | of which reached only cumulatively | Equal | Cumulative later | Reached per-class, not cumulatively | 5 um channel identical |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    ok = True
    for name, (new, old, grid) in paths.items():
        if not new.exists():
            lines.append(f"| {name} | {grid} | not run | | | | | | | |")
            ok = False
            continue
        a, b = arrival_from_run(new), arrival_from_run(old)
        assert a.counting == CUMULATIVE and a.resolution == STEP_RESOLVED
        live = ~np.isnan(a.maps)
        x, y = a.maps[live], b.maps[live]
        earlier = x < y
        only = np.isfinite(x) & np.isinf(y)
        later = x > y
        lost = np.isinf(x) & np.isfinite(y)
        largest = a.maps[:, -1].tobytes() == b.maps[:, -1].tobytes()
        ok = ok and not later.any() and not lost.any() and largest
        lines.append(
            f"| {name} | {grid} | {b.counting} | {x.size:,} | {int(earlier.sum()):,} | "
            f"{int(only.sum()):,} | {int((x == y).sum()):,} | {int(later.sum()):,} | "
            f"{int(lost.sum()):,} | {yes(largest)} |"
        )
    return lines, ok


# ---------------------------------------------------------------------------
# Prediction (c): the brackets, the worst case, the order
# ---------------------------------------------------------------------------


class Step:
    """One grid's per-class (prompt 79) and cumulative (this pass) step maps and searches."""

    def __init__(self, grid: str, paths: dict, extra: list[SimConfig]) -> None:
        names = [n for n, (_, _, g) in paths.items() if g == grid and n != "clean_out"]
        # The 320x120 units in tables80's order, so C1 rows line up with prompt 80's brackets.
        order = tables80.EVENT_NAMES
        names.sort(key=lambda n: order.index(n) if n in order else 0)
        self.names = names
        configs = [tables79.config_of(paths[n][0]) for n in names]
        record = {
            PER_CLASS: {n: arrival_from_run(paths[n][1]) for n in names},
            CUMULATIVE: {n: arrival_from_run(paths[n][0]) for n in names},
        }
        for n in names:
            assert record[PER_CLASS][n].counting == PER_CLASS
            assert record[CUMULATIVE][n].counting == CUMULATIVE
        self.grid = tables80.Grid(
            grid, {n: paths[n][0] for n in names}, configs + extra, record, None
        )


def bracket_table(fine: object, steps: Step) -> tuple[list[str], dict]:
    """C1: per layout of record and unit, this pass's latency against prompt 80's bracket."""
    br = tables80.Brackets(fine)
    g = steps.grid
    lines = [
        "### C1. On 320x120, per layout of record (prompt 80's: searched on prompt 79's per-class step maps, the worst-case rule) and unit, this pass's cumulative step latency (s after the onset) against prompt 80's bracket for it, rebuilt by tables80.py's own code from prompt 79's records (prediction (c))",
        "",
        "| Layout | Unit | Per-class, steps (prompt 79) | Prompt 80's bracket | Cumulative, steps (this pass) | Inside the bracket | Earlier than per-class |",
        "|---|---|---|---|---|---|---|",
    ]
    outside, result = [], {}
    assert g.labels == fine.grid.labels
    for name, cells in br.layouts.items():
        worst = -math.inf
        for u, row in enumerate(br.units[name]):
            label, p, _, _, b, _, exact = row
            t = float(min(g.units[CUMULATIVE].latencies[u, j, i] for j, i in cells))
            if exact is not None:
                inside = t == exact
            elif math.isinf(b.high):
                inside = math.isinf(t) or (t > b.low if not b.closed else t >= b.low)
            else:
                inside = (t >= b.low if b.closed else t > b.low) and t <= b.high
            if not inside:
                outside.append((name, label))
            worst = max(worst, t)
            text = (
                f"exact {t3(exact)}" if exact is not None else tables80.interval_text(b)
            )
            lines.append(
                f"| {name} | {label_of(g.sizes, label)} | {t3(p)} | {text} | {t3(t)} | "
                f"{yes(inside)} | {yes(t < p)} |"
            )
        result[name] = worst
    lines += [
        "",
        f"Outside its bracket: {len(outside)} of {sum(len(v) for v in br.units.values())} "
        "units"
        + (
            ": " + "; ".join(f"{n}, {label_of(g.sizes, la)}" for n, la in outside)
            if outside
            else ""
        )
        + ". Worst case over the units, cumulative at steps: "
        + "; ".join(f"{n} {t4(w)} s" for n, w in result.items())
        + ".",
    ]
    return lines, {"outside": outside, "worst": result, "layouts": br.layouts, "br": br}


def score_table(step: Step, title: str) -> tuple[list[str], dict]:
    """L1/L2: the four layouts under both readings, each searched on its own maps."""
    g = step.grid
    lines = [
        f"### {title}",
        "",
        "| Layout | Search rule | Searched on | Cell centres (x, y), m | Worst (s) | Mean (s) | Undetected | Coverage within 10 s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    out = {}
    for rule in tables80.RULES:
        for counting in (PER_CLASS, CUMULATIVE):
            layouts = g.layouts(counting, rule)
            scored = {n: g.scored(counting, c) for n, c in layouts.items()}
            out[(rule, counting)] = scored
            for n, layout in scored.items():
                s = layout.score
                lines.append(
                    f"| {n} | {rule if n != tables80.FILE else '-'} | "
                    f"{tables80.READING[counting]} | {tables80.places(g.mesh, layout.cells)} | "
                    f"{t3(s.worst)} | {s.mean:.3f} | {s.undetected} | {s.coverage:.3f} |"
                )
    return lines, out


def order_table(grids: dict[str, Step]) -> tuple[list[str], bool]:
    """O1: the order of the four layouts under each reading, both grids."""
    lines = [
        "### O1. The order of the four layouts, each searched and scored on one reading's step maps (per-class prompt 79's, cumulative this pass's), over every unit (prediction (c))",
        "",
        "| Grid | Search rule | Per-class, by the worst case | Cumulative, by the worst case | Same order | Per-class, section D's ranking | Cumulative, section D's ranking | Same ranking |",
        "|---|---|---|---|---|---|---|---|",
    ]
    same = True
    for title, step in grids.items():
        for rule, (a, b, c, d) in tables80.orders(
            step.grid, PER_CLASS, CUMULATIVE
        ).items():
            same = same and a == b
            lines.append(
                f"| {title} | {rule} | {a} | {b} | {yes(a == b)} | {c} | {d} | {yes(c == d)} |"
            )
    return lines, same


# ---------------------------------------------------------------------------
# Prediction (d): the clear times, the premature intervals, the recovery
# ---------------------------------------------------------------------------


def clear_table(fine: object, paths: dict, layouts: dict) -> tuple[list[str], dict]:
    """A1: per all-clear unit, the room's and each layout's clear time, both readings."""
    # tables80's order, so the units line up with prompt 80's.
    new = clear_units({n: paths[n][0] for n in tables80.FINE_NAMES})
    old = fine.clear[PER_CLASS]
    cum80 = fine.clear[CUMULATIVE]
    assert new.labels == old.labels
    same80 = new.clear.tobytes() == cum80.clear.tobytes()
    lines = [
        "### A1. On 320x120, per all-clear unit (scenario, channel): the room's clear time and each layout of record's, per-class (prompt 79's frames, each class alone) and cumulative (this pass's runs through placement.clear_units), and the premature interval REQ-A07 reports, frame-resolved (s; not clear is still at or above at the run's end; prediction (d))",
        "",
        "| Unit | Layout | Room, per-class | Room, cumulative | Layout, per-class | Layout, cumulative | Interval, per-class | Interval, cumulative | Clear times later or equal | Interval equal or longer |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    earlier, shorter, rows = [], [], 0
    sizes = fine.grid.sizes
    for name, cells in layouts.items():
        a, b = clear_score(old, cells), clear_score(new, cells)
        for u, (scenario, k) in enumerate(new.labels):
            rows += 1
            later = (b.room[u] >= a.room[u]) and (b.layout[u] >= a.layout[u])
            ia, ib = a.interval[u], b.interval[u]
            # An interval is a difference of two frame times; the same two frames moved
            # together can differ in the last bits, so shorter means by more than SUBTRACTION.
            longer = not (np.isfinite(ia) and np.isfinite(ib)) or ib >= ia - SUBTRACTION
            if not later:
                earlier.append((name, scenario, k))
            if not longer:
                shorter.append((name, scenario, k))
            lines.append(
                f"| {TITLES[scenario]}, {sizes[k] * 1e6:g} um | {name} | "
                f"{t4(a.room[u]) if np.isfinite(a.room[u]) else 'not clear'} | "
                f"{t4(b.room[u]) if np.isfinite(b.room[u]) else 'not clear'} | "
                f"{t4(a.layout[u]) if np.isfinite(a.layout[u]) else 'not clear'} | "
                f"{t4(b.layout[u]) if np.isfinite(b.layout[u]) else 'not clear'} | "
                f"{t4(ia)} | {t4(ib)} | {yes(later)} | {yes(longer)} |"
            )
    lines += [
        "",
        f"The cumulative clear times through placement.clear_units equal prompt 80's streamed cumulative ones (probe80.frame_clear) on prompt 79's frames, bit for bit: {yes(same80)}. "
        f"Units with a clear time earlier than prompt 79's: {len(earlier)} of {rows}; with an interval shorter by more than {SUBTRACTION:g} s (the last bits of a difference of two frame times; the frames are about 0.36 s apart): {len(shorter)} of {rows}"
        + (
            ": "
            + "; ".join(
                f"{n}, {TITLES[s]}, {sizes[k] * 1e6:g} um" for n, s, k in shorter
            )
            if shorter
            else ""
        )
        + ".",
    ]
    return lines, {"earlier": earlier, "shorter": shorter, "same80": same80}


def recovery_table(paths: dict) -> tuple[list[str], bool]:
    """R1: the clean-out's recovery outputs against prompt 79's."""
    new = RunFile(paths["clean_out"][0]).read_recovery()
    old = RunFile(paths["clean_out"][1]).read_recovery()
    lines = [
        "### R1. The 320x120 clean-out's recovery outputs (ADR-014 C): prompt 79's (per-class) against this pass's (the channels for the within-spec times and the local field; the 100:1 times per class under both)",
        "",
        "| Quantity | Prompt 79 | This pass | Later or equal | Bit for bit |",
        "|---|---|---|---|---|",
        f"| counting | {old.counting} | {new.counting} | | |",
    ]
    ok = True

    def row(title: str, a: float, b: float, compare: bool = True) -> None:
        nonlocal ok
        later = (math.isnan(a) and math.isnan(b)) or b >= a or math.isnan(b)
        if compare:
            ok = ok and later
        lines.append(
            f"| {title} | {t4(a)} | {t4(b)} | {yes(later) if compare else '-'} | "
            f"{yes(np.float64(a).tobytes() == np.float64(b).tobytes())} |"
        )

    row("room recovery time (s)", old.room_recovery_time, new.room_recovery_time)
    for k, size in enumerate(SIZES_UM):
        row(
            f"recovery time, channel {size} um (s)",
            old.recovery_time[k],
            new.recovery_time[k],
        )
    for k, size in enumerate(SIZES_UM):
        row(
            f"local recovery field maximum, channel {size} um (s)",
            old.local_recovery_max[k],
            new.local_recovery_max[k],
        )
    for name in ("sensor_time", "mean_time", "ratio"):
        same = getattr(old, name).tobytes() == getattr(new, name).tobytes()
        lines.append(f"| {name} (per class, every entry) | | | - | {yes(same)} |")
    local_later = bool(
        np.all(
            (new.local_recovery >= old.local_recovery) | np.isnan(old.local_recovery)
        )
    )
    lines.append(
        f"| local recovery field, every non-SOLID cell and channel | | | {yes(local_later)} | "
        f"{yes(new.local_recovery.tobytes() == old.local_recovery.tobytes())} |"
    )
    return lines, ok and local_later


# ---------------------------------------------------------------------------
# Prediction (e): the sweep's arms from their frames
# ---------------------------------------------------------------------------


def sweep_runs() -> list[Path]:
    """The sweep's finished event runs (prompt 79's), not run again."""
    out = []
    for path in sorted(SWEEP.glob("*.h5")):
        if path.name.endswith(".checkpoint.h5") or path.stem.startswith("clean_out"):
            continue
        if tables79.WALL.search(tables79.read_text(path.with_suffix(".txt"))):
            out.append(path)
    return out


def sweep_table() -> tuple[list[str], list[str], bool]:
    """E1 and E2: the sweep's arms at the room's sensors and the contour, on the channels."""
    e1 = [
        "### E1. The sensitivity sweep's arm runs (prompt 79's, results/builder79/sweep/, not run again) at the room's sensors (the layout the sweep scored, the file's four): each channel's cumulative count over its limit, its largest over the stored frames and over every step's readings, beside each class alone (prediction (e)). Frame-resolved readings can miss a crossing between frames; the every-step readings cannot",
        "",
        "| Run | Grid | Frames | Largest channel count over its limit, at the frames | Largest, every step | Channel, sensor of the largest | Largest class alone over its limit, every step | At or above, at the frames | At or above, every step |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    e2 = [
        "### E2. The sweep's arm runs on their frames: the distance from each of the room's sensors' cells to the nearest non-SOLID cell ever at or above in any stored frame, per channel, cumulative against each class alone (the sweep's D1 reading on the channels; centre to centre, metres). Frame-resolved, as D1 always was: a cell at or above only between two frames is not seen",
        "",
        "| Run | Grid | Channel (um) | Sensor | Each class alone (m) | Cumulative (m) | Nearer by (m) | Sensor's cell reached, cumulative |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ok = True
    for path in sweep_runs():
        config = tables79.config_of(path)
        mesh = Mesh(config)
        live = mesh.cell_type != SOLID
        thresholds = np.array(config.class_thresholds())
        channels = config.channel_classes()
        cells = sensor_cells(config, mesh)
        names = [s.name for s in config.sensors]
        rows = np.array([j for j, _ in cells])
        cols = np.array([i for _, i in cells])
        grid = f"{mesh.cell_type.shape[1]}x{mesh.cell_type.shape[0]}"
        ever = {
            c: np.zeros((thresholds.size, *live.shape), bool)
            for c in (PER_CLASS, CUMULATIVE)
        }
        frame_ratio = np.zeros((thresholds.size, len(cells)))
        with h5py.File(path, "r") as handle:
            group = handle["transport"]
            n_frames = group["times"].shape[0]
            for n in range(n_frames):
                fields = {k: group[f"C_{k}"][n] for k in range(thresholds.size)}
                for k in range(thresholds.size):
                    total = channel_sum(fields, channels[k])
                    ever[CUMULATIVE][k] |= (total >= thresholds[k]) & live
                    ever[PER_CLASS][k] |= (fields[k] >= thresholds[k]) & live
                    frame_ratio[k] = np.maximum(
                        frame_ratio[k], total[rows, cols] / thresholds[k]
                    )
        readings = RunFile(path).read_sensor_readings()
        per = readings.readings
        steps = np.stack(
            [
                channel_sum({j: per[:, j] for j in range(per.shape[1])}, channels[k])
                for k in range(per.shape[1])
            ],
            axis=1,
        )
        step_ratio = (steps / thresholds[None, :, None]).max(axis=0)
        class_ratio = (per / thresholds[None, :, None]).max()
        k, s = np.unravel_index(int(np.argmax(step_ratio)), step_ratio.shape)
        at_frames = int((frame_ratio >= 1.0).sum())
        at_steps = int((step_ratio >= 1.0).sum())
        ok = ok and at_frames == 0
        e1.append(
            f"| {path.stem} | {grid} | {n_frames} | {frame_ratio.max():.4f} | "
            f"{step_ratio.max():.4f} | {SIZES_UM[k]} um, {names[s]} | {class_ratio:.4f} | "
            f"{at_frames} | {at_steps} |"
        )
        xc, yc = np.meshgrid(mesh.xc, mesh.yc)
        for k in range(thresholds.size):
            for s, (j, i) in enumerate(cells):
                d = {}
                for c in (PER_CLASS, CUMULATIVE):
                    hit = ever[c][k]
                    d[c] = (
                        math.inf
                        if not hit.any()
                        else float(
                            np.sqrt(
                                ((xc - xc[j, i]) ** 2 + (yc - yc[j, i]) ** 2)[hit].min()
                            )
                        )
                    )
                if math.isinf(d[PER_CLASS]) and math.isinf(d[CUMULATIVE]):
                    continue
                nearer = d[PER_CLASS] - d[CUMULATIVE]
                e2.append(
                    f"| {path.stem} | {grid} | {SIZES_UM[k]} | {names[s]} | "
                    f"{'never' if math.isinf(d[PER_CLASS]) else f'{d[PER_CLASS]:.3f}'} | "
                    f"{'never' if math.isinf(d[CUMULATIVE]) else f'{d[CUMULATIVE]:.3f}'} | "
                    f"{'-' if math.isinf(nearer) or math.isnan(nearer) else f'{nearer:.3f}'} | "
                    f"{yes(bool(ever[CUMULATIVE][k][j, i]))} |"
                )
    return e1, e2, ok


# ---------------------------------------------------------------------------
# The plants and the suite
# ---------------------------------------------------------------------------


def plants_table() -> tuple[list[str], bool]:
    """T1: each planted defect and whether its tests failed."""
    lines = [
        "### T1. The planted defects (docs/reports/probe81/plant81.py, in a worktree of the code commit), the tests run under each, and whether they failed (results/builder81/plants.json)",
        "",
    ]
    if not PLANTS.exists():
        return [*lines, "Not run."], False
    record = json.loads(PLANTS.read_text(encoding="utf-8"))
    lines += [
        f"Commit {record['commit'][:7]}; the named tests without a plant: exit {record['clean']['exit']}, {record['clean']['summary']}.",
        "",
        "| Plant | Defect | File | Tests | Exit | pytest's summary | Caught |",
        "|---|---|---|---|---|---|---|",
    ]
    caught = True
    for p in record["plants"]:
        caught = caught and p["caught"]
        tests = ", ".join(t.split("::", 1)[1] if "::" in t else t for t in p["tests"])
        lines.append(
            f"| {p['name']} | {p['defect']} | {p['file']} | {tests} | {p['exit']} | "
            f"{p['summary']} | {yes(p['caught'])} |"
        )
    n = len(record["plants"])
    lines += ["", f"{sum(p['caught'] for p in record['plants'])} of {n} plants caught."]
    return lines, caught and record["clean"]["exit"] == 0


def suite_table() -> list[str]:
    """U1: the whole suite at the code commit, as pytest recorded it."""
    lines = [
        "### U1. The whole suite at the code commit (pytest tests/, results/builder81/suite.xml)",
        "",
    ]
    if not SUITE.exists():
        return [*lines, "Not run."]
    root = ET.parse(SUITE).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    assert suite is not None
    get = {
        k: int(suite.get(k, "0")) for k in ("tests", "failures", "errors", "skipped")
    }
    lines += [
        "| Tests | Failures | Errors | Skipped | Time (s) |",
        "|---|---|---|---|---|",
        f"| {get['tests']:,} | {get['failures']} | {get['errors']} | {get['skipped']} | {float(suite.get('time', '0')):.0f} |",
    ]
    return lines


# ---------------------------------------------------------------------------
# The re-scores
# ---------------------------------------------------------------------------


def prepare() -> None:
    """Copy configurations and logs where tables69 and tables76 read them."""
    for group in ("product", "placement"):
        target = RESULTS / group
        (target / "configs").mkdir(parents=True, exist_ok=True)
        for path in (RESULTS / "configs" / group).glob("*.yaml"):
            shutil.copy2(path, target / "configs" / path.name)
        (target / "logs").mkdir(parents=True, exist_ok=True)
        for path in target.glob("*.log"):
            shutil.copy2(path, target / "logs" / path.name)


def rescore() -> dict[str, str]:
    """tables69's and tables76's tables on this pass's runs, each by its own code."""
    tables79.accept_older_records()
    product, placement = RESULTS / "product", RESULTS / "placement"
    out = {}
    t69 = load_module(REPORTS / "probe69" / "tables69.py", "tables69")
    runs = t69.load_runs(product, {name: name for name in t69.NAMES})
    coarse = t69.load_runs(t69.COARSE, t69.COARSE_STEMS)
    out["69"] = t69.render(runs, coarse)[0]
    t76 = load_module(REPORTS / "probe76" / "tables76.py", "tables76")
    t76.COARSE, t76.SWEEP = placement, SWEEP
    t76.COARSE_CLEAN_OUT = SWEEP / "clean_out_80x30.h5"
    original = t76.arrival_from_run

    def from_frames(run_file: object) -> RunArrival:
        """The sweep's arm runs on the channels at their frames; every other run as given."""
        path = Path(run_file.path if isinstance(run_file, RunFile) else run_file)
        if path.parent.resolve() != SWEEP.resolve():
            return original(run_file)
        step = original(run_file)
        config = tables79.config_of(path)
        frames = RunFile(path).read_frames()
        maps = arrival_from_frames(
            frames,
            np.array(config.class_thresholds()),
            Mesh(config).cell_type != SOLID,
            step.onsets,
            channels=config.channel_classes(),
        )
        return RunArrival(
            maps,
            step.onsets,
            FRAME_RESOLVED,
            step.event_names,
            float(frames.times[-1]),
            "prompt 81: read from the frames on the channels, as the prompt asks",
            CUMULATIVE,
        )

    t76.arrival_from_run = from_frames
    try:
        analysis = t76.Analysis({name: t76.Run(product, name) for name in t76.NAMES})
        out["76"] = t76.render(analysis)[0]
    finally:
        t76.arrival_from_run = original
    return out


def rescore_header(source: str) -> str:
    """The first lines of a re-scored file."""
    reads = {
        "69": "results/builder81/product/ (the four scenarios the Phase 4 gate ran) and, for G1's 80x30 side, results/builder64/coarse/ as before",
        "76": "results/builder81/product/ and results/builder81/placement/, and prompt 79's results/builder79/sweep/ for K2, its arm runs read from their frames on the channels; the comparisons with the Phase 4 gate (F2, X1) and the hashes, coverage and plants read their records as before",
    }[source]
    return (
        f"<!-- Re-scored by docs/reports/probe81/tables81.py with tables{source}.py's own code "
        f"and this branch's src/ (cumulative counting); it read {reads}. Its prose names the "
        "directories of the runs it was written for. -->\n\n"
    )


def changes(rescored: dict[str, str]) -> tuple[str, list[str]]:
    """changes81.md and its summary rows: prompt 79's re-scores against this pass's."""
    summary = [
        "| Source | Table | Rows | Cells compared | Cells changed | Largest relative change | Prose lines changed |",
        "|---|---|---|---|---|---|---|",
    ]
    detail: list[str] = []
    for source in ("69", "76"):
        old = RESCORED_79[source].read_text(encoding="utf-8")
        s, d = tables79.compare(old, rescored[source], source)
        summary += s
        detail += d
    text = (
        "# Prompt 79's re-scores against prompt 81's (per-class against cumulative counting)\n\n"
        "Generated by `docs/reports/probe81/tables81.py`. Each table of "
        "`docs/reports/probe79/rescored69.md` and `rescored76.md` (main's code before prompt "
        "81 on prompt 79's runs) against the same table of `docs/reports/probe81/rescored69.md` "
        "and `rescored76.md` (this branch's code on this pass's runs), cell by cell. The "
        '"Committed" column is prompt 79\'s re-score.\n\n'
        + "\n".join(summary)
        + "\n\n"
        + "\n".join(detail)
        + "\n"
    )
    return text, summary


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def predictions_table(v: dict) -> list[str]:
    """P1: the predictions against the measurements."""
    rows = [
        (
            "(a)",
            "Every 320x120 and 80x30 run's fields, budgets and flow states equal prompt 79's bit for bit",
            v["a_text"],
            v["a"],
        ),
        (
            "(b)",
            "The 80x30 cumulative maps equal prompt 80's probe maps bit for bit",
            "B1: " + ("all three equal" if v["b"] else "not all equal"),
            v["b"],
        ),
        (
            "(c)",
            "On 320x120 every cumulative latency of a scored layout lies inside prompt 80's bracket for it; the product layout's worst case lies in [2.5616, 2.565] s; the four layouts' order is unchanged on both grids; the file's four detect nothing",
            v["c_text"],
            v["c"],
        ),
        (
            "(d)",
            "Every clear time is later than or equal to prompt 79's, and every premature interval of REQ-A07 equal or longer",
            v["d_text"],
            v["d"],
        ),
        (
            "(e)",
            "Re-scored from the frames, no sweep arm brings any channel's cumulative count to its limit at any sensor cell of the layouts the sweep scored",
            v["e_text"],
            v["e"],
        ),
    ]
    lines = [
        "### P1. The predictions against the measurements",
        "",
        "| Prediction | As committed | Measured | Verdict |",
        "|---|---|---|---|",
    ]
    for key, said, measured, held in rows:
        lines.append(
            f"| {key} | {said} | {measured} | {'holds' if held else 'FAILS'} |"
        )
    return lines


def build() -> tuple[str, dict[str, str], dict[str, str], str]:
    """Every table: tables81.md's text, its blocks, the re-scores, changes81.md."""
    prepare()
    paths = run_paths()
    blocks: dict[str, list[str]] = {}
    v: dict = {}
    blocks["queue"] = queue_table()
    f1, bitwise, _ = bitwise_table(paths)
    blocks["bitwise"] = f1
    v["a"] = all(bitwise.values())
    v["a_text"] = (
        f"F1: {sum(bitwise.values())} of {len(bitwise)} runs equal in fields, budgets and flow states"
    )
    blocks["probe"], v["b"] = probe_table(paths)
    blocks["structure"], structure_ok = structure_table(paths)
    fine = tables80.Fine()
    extra80 = [tables79.config_of(SWEEP / "clean_out_80x30.h5")]
    grids = {
        "320x120": Step("320x120", paths, [tables79.config_of(paths["clean_out"][0])]),
        "80x30": Step("80x30", paths, extra80),
    }
    blocks["brackets"], br = bracket_table(fine, grids["320x120"])
    blocks["fine_scores"], fine_scores = score_table(
        grids["320x120"],
        "L1. On 320x120, the four layouts searched and scored on each reading's step maps (per-class prompt 79's /arrival/, cumulative this pass's) over the twenty units, under each search rule",
    )
    blocks["coarse_scores"], coarse_scores = score_table(
        grids["80x30"],
        "L2. On 80x30, the same over the fifteen units (prompt 79's placement runs per-class, this pass's cumulative)",
    )
    blocks["orders"], same_order = order_table(grids)
    stand = br["worst"][tables80.STAND]
    file_none = all(
        scores[tables80.FILE].score.undetected == len(grids[g].grid.labels)
        for g, table in (("320x120", fine_scores), ("80x30", coarse_scores))
        for scores in [table[(WORST_CASE, CUMULATIVE)]]
    )
    in_window = WORST_WINDOW[0] <= stand <= WORST_WINDOW[1]
    # The per-class step latency of the breach's largest class at the product layout.
    breach5 = next(
        row[1]
        for row in br["br"].units[tables80.STAND]
        if row[0][0] == "filter_breach" and row[0][2] == len(fine.grid.sizes) - 1
    )
    v["c"] = not br["outside"] and in_window and same_order and file_none
    v["c_text"] = (
        f"C1: {len(br['outside'])} units outside their brackets; the product layout's worst case "
        f"{stand:.7f} s ({'inside' if in_window else 'outside'} the window as committed; it is "
        f"{'' if stand == breach5 else 'not '}equal to the breach's 5 um latency, which no "
        f"counting changes and which the window's lower end {WORST_WINDOW[0]} rounded up); "
        "O1: the order "
        f"{'unchanged' if same_order else 'changed'} on both grids; the file's four detect "
        f"{'nothing' if file_none else 'something'} (L1, L2)"
    )
    blocks["clear"], clear = clear_table(fine, paths, br["layouts"])
    blocks["recovery"], recovery_ok = recovery_table(paths)
    v["d"] = not clear["earlier"] and not clear["shorter"] and recovery_ok
    v["d_text"] = (
        f"A1: {len(clear['earlier'])} clear times earlier, {len(clear['shorter'])} intervals "
        f"shorter; R1: the clean-out's recovery times and local field "
        f"{'later or equal' if recovery_ok else 'earlier somewhere'}"
    )
    e1, e2, v["e"] = sweep_table()
    blocks["sweep"], blocks["sweep_contour"] = e1, e2
    v["e_text"] = "E1: " + (
        "no arm run reaches a channel's limit at the room's sensors at any frame"
        if v["e"]
        else "some arm run reaches a channel's limit at a sensor at a frame"
    )
    blocks["plants"], plants_ok = plants_table()
    blocks["suite"] = suite_table()
    blocks["predictions"] = predictions_table(v)
    blocks["stops"] = [
        "### X1. The prompt's stop conditions",
        "",
        "| Condition | Met |",
        "|---|---|",
        f"| A field, budget or flow state differs from prompt 79's | {yes(not v['a'])} |",
        f"| A cumulative arrival is later than the per-class one anywhere (S1) | {yes(not structure_ok)} |",
        f"| A planted defect does not fail its test (T1) | {yes(not plants_ok)} |",
    ]
    rescored = rescore()
    changed, summary = changes(rescored)
    blocks["changes"] = [
        "### X2. Prompt 79's re-scores of tables69 and tables76 (per-class) against this pass's (cumulative), table by table (the detail in changes81.md)",
        "",
        *summary,
    ]
    text = "# Prompt 81's tables\n\nGenerated by `docs/reports/probe81/tables81.py` from the records; `--check` regenerates and compares.\n\n"
    joined = {name: "\n".join(lines) for name, lines in blocks.items()}
    text += "\n\n".join(joined.values()) + "\n"
    files = {k: rescore_header(k) + rescored[k] for k in RESCORED}
    return text, joined, files, changed


def splice(text: str, blocks: dict[str, str]) -> str:
    """The report with each marked block replaced by its table."""
    for name, body in blocks.items():
        pattern = re.compile(
            rf"(<!-- tables81 {name} begin -->\n).*?(<!-- tables81 {name} end -->)",
            re.DOTALL,
        )
        text = pattern.sub(lambda m, b=body: m.group(1) + b + "\n" + m.group(2), text)
    return text


def main() -> int:
    """Write or check the tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text, blocks, files, changed = build()
    outputs = {
        TABLES: text,
        CHANGES: changed,
        **{RESCORED[k]: v for k, v in files.items()},
    }
    report = REPORT.read_text(encoding="utf-8")
    spliced = splice(report, blocks)
    if args.check:
        bad = [
            rel(p)
            for p, t in outputs.items()
            if not p.exists() or p.read_text(encoding="utf-8") != t
        ]
        if spliced != report:
            bad.append(rel(REPORT))
        print("matches" if not bad else "DIFFERS: " + ", ".join(bad))
        return 1 if bad else 0
    for path, body in outputs.items():
        path.write_text(body, encoding="utf-8", newline="\n")
    REPORT.write_text(spliced, encoding="utf-8", newline="\n")
    print("wrote " + ", ".join(rel(p) for p in outputs) + " and spliced " + rel(REPORT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
