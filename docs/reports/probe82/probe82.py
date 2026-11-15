"""Prompt 82's probe: how the placement search's cost grows with the units.

    python docs/reports/probe82/probe82.py build                    # the unit tables
    python docs/reports/probe82/probe82.py case METHOD TABLE COUNT  # one case, this process
    python docs/reports/probe82/probe82.py queue                    # every case, one at a time

``build`` writes the latency tables (candidates by units) the searches run on
to ``results/builder82/tables/``, built with main's ``src/`` on cumulative
maps: the four event scenarios on 320x120 (prompt 81's runs, their step maps)
and, as stand-ins for further release positions, the sweep's finished 160x60
arm runs (prompt 79's, read from their stored frames on the channels). The
stand-ins are other turbulence arms of the door, breach and dust scenarios,
not other release positions: what is measured is the search's cost and its
answer's quality on tables of realistic structure, not any layout.

``case`` runs one method on one table at one count and writes its record to
``results/builder82/cases/``: wall and CPU time, the process's private bytes
before the method and at its peak (Windows' process memory counters), and the
method's answer. ``queue`` runs every case in its own process, one at a time,
skips a case whose estimated peak is above ``MEMORY_LIMIT`` (recording the
estimate) and stops a case at ``TIME_LIMIT``, recording it as over.

The methods:

- ``cover``: ``placement._worst_case_cover`` as built, with
  ``MAX_SET_COVER_UNITS`` raised in this process only (an assignment to the
  module's attribute, not an edit).
- ``enum``: the exact enumeration this probe adds (``enumerate_cover``): the
  same bisection over the sorted distinct latencies, feasibility decided over
  combinations of at most ``count`` maximal masks (``Feasibility``), then the
  built search's own cover enumeration (``placement._covers``) and tie-break
  by the mean, with the subset table replaced by the same feasibility test.
- ``greedy``: the mean rule's greedy additions and exchanges
  (``placement._greedy``), scored under both rules.
- ``exhaustive``: at one and two sensors, ``placement._best_one`` and
  ``_best_two`` under the worst-case rule, the exact answers ``search`` itself
  gives there, as a third check of the worst case.
"""

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import time
from ctypes import wintypes
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src import placement  # noqa: E402 -- follows sys.path.insert
from src.config import WORST_CASE, SimConfig  # noqa: E402 -- follows sys.path.insert
from src.io_manager import (  # noqa: E402 -- follows sys.path.insert
    CUMULATIVE,
    RunFile,
)
from src.mesh import SOLID, Mesh  # noqa: E402 -- follows sys.path.insert
from src.monitor import contaminating_events  # noqa: E402 -- follows sys.path.insert
from src.placement import (  # noqa: E402 -- follows sys.path.insert
    FRAME_RESOLVED,
    STEP_RESOLVED,
    RunArrival,
    arrival_from_frames,
    arrival_from_run,
    candidates,
    detection_units,
)

RESULTS = ROOT / "results" / "builder82"
TABLES = RESULTS / "tables"
CASES = RESULTS / "cases"
QUEUE_LOG = RESULTS / "queue.jsonl"
PRODUCT_RUNS = ROOT / "results" / "builder81" / "product"
SWEEP = ROOT / "results" / "builder79" / "sweep"
PRODUCT = ROOT / "configs" / "clean_room_default.yaml"

# The prompt's limits: a case estimated above 8 GB is not run, and one still
# running after 30 minutes is stopped and recorded as over.
MEMORY_LIMIT = 8 * 2**30
TIME_LIMIT = 30 * 60
# The cap a probe process gives the built set cover: the built masks are int64
# sums of one bit per unit, which hold 63 units (bit 63 is the sign).
PROBE_CAP = 63
# OpenBLAS commits a buffer per thread at import (about 0.8 GB of private bytes
# on this machine's 24 threads); no method here calls BLAS, and one thread
# keeps the baseline under the measured peaks small.
CASE_ENV = os.environ | {"OPENBLAS_NUM_THREADS": "1"}
COUNTS = (1, 2, 3, 4)
METHODS = ("cover", "enum", "greedy", "exhaustive")

# The 320x120 units in the order prompt 81's tables took them (tables80's).
FINE_NAMES = ("door_leak", "filter_breach", "equipment_dust", "seal_leak")
# The 160x60 stand-ins: arm-major, then the scenario. The door's control arm
# never finished on 160x60 (prompt 79's capped flow state) and has no frames.
ARMS = ("control", "i10", "max_nut", "min_nut", "sc02", "widest")
SCENARIOS = ("door_leak", "filter_breach", "equipment_dust")
COARSE_SIZES = (20, 24, 28, 32, 40, 60)
# Added by the builder: the same 160x60 units at the sizes between 24 and 28,
# where the built cover's estimate crosses the memory limit (one release
# position more than the twenty is 25 units).
EDGE_SIZES = (25, 26, 27)
# Added by the builder: a bound, not a realistic table. Every cell and unit
# independent (prompt 75b's most diverse map), on as many candidates as the
# 160x60 tables, so that the enumeration meets many maximal masks.
NOISE_SIZES = (20, 24, 28, 32, 40, 60)
NOISE_SEED = 82
NOISE_UNREACHED = 0.5
NOISE_SPAN = 30.0
NOISE_CAP = 110.0


def table_names() -> list[str]:
    """The tables in the order measured: the prompt's, the edge, the noise bound."""
    return [
        "fine20",
        *(f"coarse{n}" for n in COARSE_SIZES),
        *(f"coarse{n}" for n in EDGE_SIZES),
        *(f"noise{n}" for n in NOISE_SIZES),
    ]


# ---------------------------------------------------------------------------
# The unit tables
# ---------------------------------------------------------------------------


def config_of(path: Path) -> SimConfig:
    """The merged configuration a run file records."""
    return SimConfig.from_dict(RunFile(path).read_config().mapping)


def coarse_runs() -> list[Path]:
    """The 160x60 arm runs with stored frames, in the stated order."""
    out = []
    for arm in ARMS:
        for name in SCENARIOS:
            path = SWEEP / f"{name}_{arm}_160x60.h5"
            if not path.exists():
                continue
            try:
                RunFile(path).read_frames()
            except KeyError:
                continue
            out.append(path)
    return out


def from_frames(path: Path) -> RunArrival:
    """A run's cumulative maps rebuilt from its stored frames on the channels.

    The fallback branch of ``arrival_from_run``, taken for a file whose
    ``/arrival/`` is per-class (prompt 79's): the channels' maps at the frames.
    """
    config = config_of(path)
    events = contaminating_events(config.scenario)
    onsets = np.array([onset for _, onset in events])
    frames = RunFile(path).read_frames()
    maps = arrival_from_frames(
        frames,
        np.array(config.class_thresholds("the arrival maps")),
        Mesh(config).cell_type != SOLID,
        onsets,
        channels=config.channel_classes(),
    )
    return RunArrival(
        maps,
        onsets,
        FRAME_RESOLVED,
        tuple(name for name, _ in events),
        float(frames.times[-1]),
        counting=CUMULATIVE,
        fallback="prompt 82: read from the frames on the channels",
    )


def unit_table(
    arrivals: dict[str, RunArrival], configs: list[SimConfig]
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, list]:
    """The product search's candidates and the table it searches.

    Returns the table [candidates, units] with the unreached units removed (as
    ``search`` removes them), their caps and labels, the candidate cells, and
    the labels removed.
    """
    units = detection_units(arrivals)
    mesh = Mesh(configs[0])
    spec = SimConfig(PRODUCT).placement
    assert spec is not None
    pool = candidates(
        mesh, configs, spec.mount, spec.standoff, stand_height=spec.stand_height
    )
    table = np.ascontiguousarray(units.latencies[:, pool[:, 0], pool[:, 1]].T)
    reached = np.isfinite(table).any(axis=0)
    labels = [list(map(str, label)) for label in units.labels]
    kept = [la for la, hit in zip(labels, reached, strict=True) if hit]
    dropped = [la for la, hit in zip(labels, reached, strict=True) if not hit]
    return (
        np.ascontiguousarray(table[:, reached]),
        units.caps[reached],
        kept,
        pool,
        dropped,
    )


def save_table(name: str, grid: str, parts: tuple, size: int | None) -> dict:
    """Write one table (its first SIZE units, or all) and return its summary."""
    table, caps, labels, pool, dropped = parts
    n = table.shape[1] if size is None else size
    assert n <= table.shape[1], f"{name}: {n} units asked, {table.shape[1]} reached"
    TABLES.mkdir(parents=True, exist_ok=True)
    np.savez(
        TABLES / f"{name}.npz",
        table=table[:, :n],
        caps=caps[:n],
        cells=pool,
    )
    return {
        "name": name,
        "grid": grid,
        "candidates": int(table.shape[0]),
        "units": n,
        "labels": labels[:n],
        "unreached": dropped,
    }


def build() -> None:
    """The unit tables: 320x120's twenty and 160x60's six sizes."""
    summary = []
    t0 = time.perf_counter()
    fine = {n: arrival_from_run(PRODUCT_RUNS / f"{n}.h5") for n in FINE_NAMES}
    for n, a in fine.items():
        assert a.counting == CUMULATIVE and a.resolution == STEP_RESOLVED, n
    configs = [config_of(PRODUCT_RUNS / f"{n}.h5") for n in FINE_NAMES]
    configs.append(config_of(PRODUCT_RUNS / "clean_out.h5"))
    summary.append(save_table("fine20", "320x120", unit_table(fine, configs), None))
    paths = coarse_runs()
    coarse = {p.stem: from_frames(p) for p in paths}
    configs = [config_of(p) for p in paths]
    configs.append(config_of(SWEEP / "clean_out_160x60.h5"))
    parts = unit_table(coarse, configs)
    for n in (*COARSE_SIZES, *EDGE_SIZES):
        summary.append(save_table(f"coarse{n}", "160x60", parts, n))
    noise = noise_table(int(parts[0].shape[0]), max(NOISE_SIZES))
    for n in NOISE_SIZES:
        summary.append(save_table(f"noise{n}", "noise", noise, n))
    record = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": time.perf_counter() - t0,
        "coarse_runs": [p.stem for p in paths],
        "coarse_units_reached": int(parts[0].shape[1]),
        "tables": summary,
    }
    (RESULTS / "tables.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    for row in summary:
        print(
            row["name"], row["grid"], row["candidates"], row["units"], row["unreached"]
        )


def noise_table(n: int, units: int) -> tuple:
    """Independent latencies: uniform over the span, unreached at the stated share."""
    rng = np.random.default_rng(NOISE_SEED)
    table = rng.uniform(0.0, NOISE_SPAN, (n, units))
    table[rng.uniform(0.0, 1.0, (n, units)) < NOISE_UNREACHED] = np.inf
    assert np.isfinite(table).any(axis=0).all()
    labels = [["noise", "unit", str(u)] for u in range(units)]
    cells = np.zeros((n, 2), dtype=np.int64)
    return table, np.full(units, NOISE_CAP), labels, cells, []


def load_table(name: str) -> tuple[np.ndarray, np.ndarray]:
    """A table and its caps."""
    data = np.load(TABLES / f"{name}.npz")
    return np.ascontiguousarray(data["table"]), data["caps"]


# ---------------------------------------------------------------------------
# The exact enumeration
# ---------------------------------------------------------------------------


def masks_at(table: np.ndarray, threshold: float) -> np.ndarray:
    """Each candidate's mask: the units it detects within THRESHOLD, as built."""
    bits = np.left_shift(1, np.arange(table.shape[1], dtype=np.int64))
    return ((table <= threshold) * bits).sum(axis=1)


class Feasibility:
    """Whether at most k of the maximal masks cover a set of units, exactly.

    Every combination of at most k masks is tried, implicitly: a cover of a
    target holds a mask containing each of its units, so the search branches
    on the masks that contain the target's unit fewest masks contain, and
    each branch goes on with the target less that mask and one slot fewer.
    The masks are first cut to the target and reduced to the maximal ones
    among the cuts, which changes no answer (a cover with a contained mask is
    one with its container). One slot left is a test of every mask at once,
    two of every pair at once. The answer depends on the target and the slots
    alone, so it is kept.

    Parameters
    ----------
    maximal : np.ndarray
        [masks] int64, the maximal masks at a threshold (``placement._maximal``).
    """

    def __init__(self, maximal: np.ndarray) -> None:
        self.maximal = maximal
        self.known: dict[tuple[int, int], bool] = {}
        self.nodes = 0
        self.pairs = 0

    def covers(self, target: int, slots: int, masks: np.ndarray | None = None) -> bool:
        """True when at most SLOTS of the masks have a union containing TARGET."""
        if target == 0:
            return True
        if slots == 0:
            return False
        key = (target, slots)
        if key in self.known:
            return self.known[key]
        self.nodes += 1
        cut = np.unique((self.maximal if masks is None else masks) & target)
        cut = cut[cut != 0]
        answer = self._decide(target, slots, cut)
        self.known[key] = answer
        return answer

    def _decide(self, target: int, slots: int, cut: np.ndarray) -> bool:
        """The answer for TARGET and SLOTS over the masks CUT to the target."""
        if cut.size == 0:
            return False
        if (cut == target).any():
            return True
        if slots == 1:
            return False
        cut = placement._maximal(cut)
        if slots * int(np.bitwise_count(cut).max()) < target.bit_count():
            return False
        if slots == 2:
            for start in range(0, cut.size, placement._MAXIMAL_CHUNK):
                rows = cut[start : start + placement._MAXIMAL_CHUNK]
                self.pairs += rows.size * cut.size
                if ((rows[:, None] | cut[None, :]) == target).any():
                    return True
            return False
        units = np.array([b for b in range(target.bit_length()) if target >> b & 1])
        holds = (cut[:, None] >> units[None, :]) & 1
        unit = int(units[int(np.argmin(holds.sum(axis=0)))])
        branch = cut[(cut >> unit) & 1 == 1]
        branch = branch[np.argsort(-np.bitwise_count(branch), kind="stable")]
        return any(self.covers(target & ~int(m), slots - 1, cut) for m in branch)


class Fewest:
    """The built search's subset table, answered by ``Feasibility`` on demand.

    ``placement._covers`` reads ``fewest[remaining]`` only to compare it with
    the slots left; this gives the fewest masks (at most ``count``, else
    ``count + 1``) for the subsets it is asked about, without the 2^units
    table.
    """

    def __init__(self, feasibility: Feasibility, count: int) -> None:
        self.feasibility = feasibility
        self.count = count
        self.asked = 0

    def __getitem__(self, remaining: int) -> int:
        """The fewest masks whose union contains REMAINING, or count + 1."""
        self.asked += 1
        for k in range(self.count + 1):
            if self.feasibility.covers(int(remaining), k):
                return k
        return self.count + 1


def enumerate_cover(
    table: np.ndarray, caps: np.ndarray, count: int, stats: dict
) -> placement._Found:
    """The worst-case optimum by the enumeration, with the built tie-break.

    The bisection and everything after the feasibility test are the built
    ``_worst_case_cover``'s, line for line; only the subset table is replaced.
    STATS receives the counts of the masks and of the search's work.
    """
    units = table.shape[1]
    full = (1 << units) - 1
    values = np.unique(table[np.isfinite(table)])
    stats.update(bisections=0, nodes=0, pairs=0, top=None)

    def at(threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, Feasibility]:
        masks = masks_at(table, threshold)
        distinct = np.unique(masks)
        maximal = placement._maximal(distinct)
        return masks, distinct, maximal, Feasibility(maximal)

    def feasible(f: Feasibility) -> bool:
        stats["bisections"] += 1
        answer = f.covers(full, count)
        stats["nodes"] += f.nodes
        stats["pairs"] += f.pairs
        return answer

    masks, distinct, maximal, f = at(values[-1])
    stats["top"] = (int(distinct.size), int(maximal.size))
    if not feasible(f):
        return placement._Found(
            (), WORST_CASE, placement.SET_COVER, True, distinct.size, maximal.size
        )
    low, high = 0, values.size - 1
    while low < high:
        middle = (low + high) // 2
        if feasible(at(values[middle])[3]):
            high = middle
        else:
            low = middle + 1
    optimum = float(values[high])
    masks, distinct, maximal, f = at(optimum)
    fewest = Fewest(f, count)
    every = np.arange(len(table))
    best: tuple | None = None
    covers = placement._covers([int(m) for m in maximal], fewest, full, count)
    stats.update(covers=len(covers), asked=fewest.asked, optimum_nodes=f.nodes)
    for cover in covers:
        chosen: list[int] = []
        for mask in cover:
            chosen.append(
                placement._added(table, caps, chosen, np.flatnonzero(masks == mask))
            )
        while len(chosen) < count:
            chosen.append(
                placement._added(table, caps, chosen, np.setdiff1d(every, chosen))
            )
        _, mean = placement._rules(table[chosen].min(axis=0)[None, :], caps)
        key = (mean[0], tuple(sorted(chosen)))
        if best is None or key < best[0]:
            best = (key, chosen)
    assert best is not None
    # The exchanges hold the other sensors fixed, and at one sensor there are
    # none: the built code is never called there (``search`` takes the argmin).
    columns = (
        tuple(best[1])
        if count == 1
        else tuple(placement._exchanged(table, caps, best[1], optimum))
    )
    return placement._Found(
        columns, WORST_CASE, placement.SET_COVER, True, distinct.size, maximal.size
    )


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


class _Counters(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS_EX (psapi.h)."""

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
        ("PrivateUsage", ctypes.c_size_t),
    ]


def memory() -> dict[str, int]:
    """This process's private bytes now and at their peak, and its peak working set."""
    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_Counters),
        wintypes.DWORD,
    )
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    ok = kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    assert ok, "K32GetProcessMemoryInfo failed"
    return {
        "private": int(counters.PagefileUsage),
        "peak_private": int(counters.PeakPagefileUsage),
        "peak_working_set": int(counters.PeakWorkingSetSize),
    }


def scored(table: np.ndarray, caps: np.ndarray, columns: tuple) -> dict:
    """A layout's worst case and capped mean over the table's units."""
    if not columns:
        return {"columns": [], "worst": None, "mean": None}
    rows = table[list(columns)].min(axis=0)
    worst, mean = placement._rules(rows[None, :], caps)
    return {
        "columns": sorted(int(c) for c in columns),
        "worst": float(worst[0]),
        "mean": float(mean[0]),
    }


def run_case(method: str, name: str, count: int) -> dict:
    """One method on one table at one count, in this process."""
    table, caps = load_table(name)
    before = memory()
    record: dict = {"method": method, "table": name, "count": count}
    stats: dict = {}
    t0, c0 = time.perf_counter(), time.process_time()
    if method == "cover":
        placement.MAX_SET_COVER_UNITS = PROBE_CAP
        found = placement._worst_case_cover(table, caps, count)
        columns = found.columns
        record.update(distinct=found.distinct, maximal=found.maximal)
    elif method == "enum":
        found = enumerate_cover(table, caps, count, stats)
        columns = found.columns
        record.update(distinct=found.distinct, maximal=found.maximal)
    elif method == "greedy":
        columns = placement._greedy(table, caps, count)
    elif method == "exhaustive":
        assert count in (1, 2), "the exhaustive search is at one and two sensors"
        columns = (
            placement._best_one(table, caps, WORST_CASE)
            if count == 1
            else placement._best_two(table, caps, WORST_CASE)
        )
    else:
        raise ValueError(f"unknown method {method!r}")
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    after = memory()
    record.update(scored(table, caps, columns))
    if method == "enum":
        record["stats"] = stats
        record.update(masks_record(table, record["worst"]))
    record.update(
        status="ok",
        wall=wall,
        cpu=cpu,
        private_before=before["private"],
        peak_private=after["peak_private"],
        peak_working_set=after["peak_working_set"],
        units=int(table.shape[1]),
        candidates=int(table.shape[0]),
        finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    return record


def masks_record(table: np.ndarray, optimum: float | None) -> dict:
    """The distinct and maximal masks at the optimum and at the largest latency."""
    out: dict = {}
    values = table[np.isfinite(table)]
    for key, threshold in (
        ("at_optimum", optimum),
        ("at_largest", float(values.max())),
    ):
        if threshold is None or not math.isfinite(threshold):
            out[key] = None
            continue
        distinct = np.unique(masks_at(table, threshold))
        maximal = placement._maximal(distinct)
        out[key] = {
            "threshold": threshold,
            "distinct": int(distinct.size),
            "maximal": int(maximal.size),
        }
    return out


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------


def estimate(method: str, units: int, n: int) -> int:
    """Bytes a case is expected to peak at, from the method's arrays.

    The built set cover: its subset table and the arrays ``_fewest`` holds at
    once over the 2^units subsets (fewest and the subset index, int64; reach
    and grown, bool; the reachable unions' indices, int64 at most; the
    superset pass's selected index, its gather, its or and its minimum, int64
    over half the subsets), 45 bytes a subset at most, and the union chunk
    ``_UNION_CHUNK`` by the masks, at most the candidates N. The other methods
    hold arrays of the candidates by the units, a few at once.
    """
    table = 8 * n * units
    if method == "cover":
        return 45 * 2**units + 8 * placement._UNION_CHUNK * n + 4 * table
    return 8 * table + 8 * placement._MAXIMAL_CHUNK * n


def cases() -> list[tuple[str, str, int]]:
    """Every case, by table, then count, then method."""
    out = []
    for name in table_names():
        for count in COUNTS:
            for method in METHODS:
                if method == "exhaustive" and count > 2:
                    continue
                # The built cover's and the greedy's exchanges need another
                # sensor to hold fixed, so at one sensor they raise; search
                # takes the argmin there.
                if method in ("cover", "greedy") and count == 1:
                    continue
                out.append((method, name, count))
    return out


def case_path(method: str, name: str, count: int) -> Path:
    """The record of one case."""
    return CASES / f"{name}_{count}_{method}.json"


def queue(redo: bool) -> None:
    """Every case in its own process, one at a time, within the limits."""
    CASES.mkdir(parents=True, exist_ok=True)
    meta = json.loads((RESULTS / "tables.json").read_text(encoding="utf-8"))
    sizes = {t["name"]: (t["units"], t["candidates"]) for t in meta["tables"]}
    for method, name, count in cases():
        path = case_path(method, name, count)
        if path.exists() and not redo:
            continue
        units, n = sizes[name]
        guess = estimate(method, units, n)
        base = {
            "method": method,
            "table": name,
            "count": count,
            "units": units,
            "candidates": n,
            "estimate": guess,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if guess > MEMORY_LIMIT:
            record = base | {"status": "skipped"}
        else:
            t0 = time.perf_counter()
            command = [sys.executable, str(Path(__file__)), "case", method, name]
            try:
                done = subprocess.run(
                    [*command, str(count)],
                    capture_output=True,
                    text=True,
                    timeout=TIME_LIMIT,
                    check=False,
                    env=CASE_ENV,
                )
            except subprocess.TimeoutExpired:
                record = base | {"status": "over", "wall": time.perf_counter() - t0}
            else:
                if done.returncode != 0:
                    record = base | {"status": "error", "stderr": done.stderr[-4000:]}
                else:
                    record = base | json.loads(done.stdout.strip().splitlines()[-1])
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        with QUEUE_LOG.open("a", encoding="utf-8") as log:
            log.write(json.dumps({k: record.get(k) for k in LOG_KEYS}) + "\n")
        print(json.dumps({k: record.get(k) for k in LOG_KEYS}), flush=True)


LOG_KEYS = ("table", "count", "method", "status", "wall", "peak_private", "worst")


def literal_combinations(maximal: int, count: int) -> int:
    """Combinations of at most COUNT of MAXIMAL masks: a literal enumeration's work."""
    return sum(comb(maximal, k) for k in range(1, count + 1))


def main() -> int:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    one = sub.add_parser("case")
    one.add_argument("method", choices=METHODS)
    one.add_argument("table")
    one.add_argument("count", type=int)
    q = sub.add_parser("queue")
    q.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "case":
        print(json.dumps(run_case(args.method, args.table, args.count)))
    else:
        queue(args.redo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
