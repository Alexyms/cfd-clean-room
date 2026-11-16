"""The tables of prompt 85's noise-band probe, from the records in results/builder85/.

    python docs/reports/probe85/tables85.py           # write tables85.md, splice the report
    python docs/reports/probe85/tables85.py --check   # regenerate and compare, write nothing

Every number in the report's probe section comes from here. Each variant is
set against the reference on the same grid and scenario (``probe85.py``'s
docstring names the variants). The comparisons are the report's scoring rule,
committed with the predictions before the runs:

- a field's relative difference is the largest absolute difference over the
  cells (and over the stored frames, for a class) over the largest magnitude of
  the reference's; beside it, for a class, the largest pointwise relative
  difference over the cells where the reference reads at least a hundredth of
  the class's threshold, the decades the alerts and the maps read;
- a latency or an arrival time moves by the difference over the step of the
  reference's interval at the later of the two; one reached and the other
  never is "reached in one only";
- a clear time moves by the difference over the unit's frame spacing.
"""

import argparse
import json
import math
import sys
from functools import cached_property
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import probe85  # noqa: E402 -- follows sys.path.insert

from src.config import (  # noqa: E402
    MEAN,
    MOUNT_FREE,
    MOUNT_STAND,
    WORST_CASE,
    SimConfig,
)
from src.io_manager import RunFile  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.monitor import sensor_cells  # noqa: E402
from src.placement import (  # noqa: E402
    arrival_from_run,
    candidates,
    clear_units,
    detection_units,
    given_layout,
    rank,
    room_clear,
    search,
)

TABLES = HERE / "tables85.md"
REPORT = ROOT / "docs" / "reports" / "phase6_design.md"
REFERENCE = "reference"
OTHERS = [v for v in probe85.VARIANTS if v != REFERENCE]
TOLERANCE = [v for v in OTHERS if v.startswith(("pressure_rtol", "diffusion_tol"))]
INFLOW = "inflow_1e-15"
SEQUENTIAL = "leftover_sequential"
OUTER = "outer_one_later"
FAMILIES = {
    "repeat": ["repeat"],
    "pressure_rtol x 0.99, 1.01": [v for v in TOLERANCE if "pressure" in v],
    "diffusion_tol x 0.99, 1.01": [v for v in TOLERANCE if "diffusion" in v],
    "inflow x (1 + 1e-15)": [INFLOW],
    "leftover summed in row-major order": [SEQUENTIAL],
    "each flow stop one outer iteration later (an addition)": [OUTER],
}
FLOW_FIELDS = ("u", "v", "p", "k", "eps", "nu_t")
# The pointwise measure's floor: a hundredth of the class's threshold.
POINTWISE_FLOOR = 0.01
RULES = (WORST_CASE, MEAN)
FILE, SURF, STAND, FREE = "the file's four", "surface", "product", "free"
LAYOUTS = (FILE, SURF, STAND, FREE)
# The prompt's predictions' limits.
B_FIELD_LIMIT = 1.0e-10
C_FIELD_LIMIT = 1.0e-6
REQ_T05 = 1.0e-4
# Prompt 81's 80x30 runs, which the 80x30 references should equal bit for bit.
PROMPT81 = ROOT / "results" / "builder81" / "placement"
PROMPT81_NAMES = {"dust": "dust", "door": "door", "breach": "breach"}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def e(value: float) -> str:
    """A relative difference: 0, or two significant figures."""
    if value == 0.0:
        return "0"
    if math.isinf(value):
        return "inf"
    return f"{value:.1e}"


def yes(flag: bool) -> str:
    """yes or no."""
    return "yes" if flag else "no"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A markdown table."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


# ---------------------------------------------------------------------------
# The records
# ---------------------------------------------------------------------------


class Run:
    """One run's file and summary, read on demand."""

    def __init__(self, variant: str, scenario: str, grid: str) -> None:
        self.variant, self.scenario, self.grid = variant, scenario, grid
        self.path = probe85.run_path(variant, scenario, grid)
        self.summary = json.loads(self.path.with_suffix(".json").read_text())

    @cached_property
    def file(self) -> RunFile:
        """The run file."""
        return RunFile(self.path)

    @cached_property
    def config(self) -> SimConfig:
        """The merged configuration the run file records."""
        return SimConfig.from_dict(self.file.read_config().mapping)

    @cached_property
    def frames(self) -> object:
        """The stored transport frames."""
        return self.file.read_frames()

    @cached_property
    def states(self) -> list[object]:
        """The flow states in the order the intervals first use them."""
        order: list[str] = []
        for plan in self.summary["intervals"]:
            if plan["fingerprint"] not in order:
                order.append(plan["fingerprint"])
        return [self.file.read_flow_state(fp).state for fp in order]

    @cached_property
    def sensors(self) -> object:
        """The sensor readings."""
        return self.file.read_sensor_readings()

    @cached_property
    def arrival(self) -> object:
        """The step-resolved arrival maps."""
        return arrival_from_run(self.path)

    @property
    def outers(self) -> list[int]:
        """Outer iterations per flow state, in the summary's order."""
        return [f["outer"] for f in self.summary["flows"]]

    def release(self) -> None:
        """Drop the large cached arrays."""
        for name in ("frames", "states", "sensors", "arrival"):
            self.__dict__.pop(name, None)


def runs_of(grid: str, scenario: str) -> dict[str, Run]:
    """Every variant's run of SCENARIO on GRID."""
    return {v: Run(v, scenario, grid) for v in probe85.VARIANTS}


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """max |b - a| over max |a|, over the finite entries of a."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    diff = float(np.abs(b - a)[finite].max()) if finite.any() else 0.0
    scale = float(np.abs(a[finite]).max()) if finite.any() else 0.0
    if diff == 0.0:
        return 0.0
    return diff / scale if scale > 0.0 else math.inf


def pointwise(a: np.ndarray, b: np.ndarray, floor: float) -> float:
    """The largest |b - a| / |a| where |a| >= floor."""
    keep = np.abs(a) >= floor
    if not keep.any():
        return 0.0
    return float((np.abs(b[keep] - a[keep]) / np.abs(a[keep])).max())


def interval_dt(plans: list[dict], t: np.ndarray) -> np.ndarray:
    """The step of the interval each time falls in, (t_start, t_end]."""
    starts = np.array([p["t_start"] for p in plans])
    dts = np.array([p["dt"] for p in plans])
    index = np.clip(np.searchsorted(starts, t, side="left") - 1, 0, len(plans) - 1)
    return dts[index]


def moved(ref: np.ndarray, var: np.ndarray, spacing: np.ndarray | float) -> np.ndarray:
    """How far each time moved, in units of SPACING; inf where one is never.

    NaN where both are NaN (SOLID); 0 where both are never.
    """
    ref = np.asarray(ref, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    out = np.full(ref.shape, np.nan)
    both = np.isfinite(ref) & np.isfinite(var)
    out[both] = (
        np.abs(var[both] - ref[both]) / np.broadcast_to(spacing, ref.shape)[both]
    )
    never = np.isinf(ref) & np.isinf(var)
    out[never] = 0.0
    one = np.isinf(ref) ^ np.isinf(var)
    out[one & ~np.isnan(ref) & ~np.isnan(var)] = math.inf
    return out


def step_moves(ref: np.ndarray, var: np.ndarray, plans: list[dict]) -> np.ndarray:
    """Times moved in steps of the reference's interval at the later of the two."""
    later = np.where(np.isfinite(var) & (var > ref), var, ref)
    later = np.where(np.isfinite(later), later, 0.0)
    return moved(ref, var, interval_dt(plans, later))


def counts(steps: np.ndarray) -> tuple[int, int, int, int, float]:
    """(compared, moved one step, moved more than one, reached in one only, largest finite)."""
    s = steps[~np.isnan(steps)]
    whole = np.rint(s[np.isfinite(s)])
    one = int((whole == 1).sum())
    more = int((whole > 1).sum())
    only = int(np.isinf(s).sum())
    largest = float(s[np.isfinite(s)].max()) if np.isfinite(s).any() else 0.0
    return int(s.size), one, more, only, largest


# ---------------------------------------------------------------------------
# Per run: bitwise, fields, budget
# ---------------------------------------------------------------------------


def same_bytes(a: np.ndarray, b: np.ndarray) -> bool:
    """Equal shapes and bytes."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def budget_rows(frames: object) -> np.ndarray:
    """[frames, classes, terms] of every stored budget, None read as NaN."""
    terms = ("initial", "inflow", "outflow", "source", "current", "leftover")
    rows = []
    for budgets in frames.budgets:
        rows.append(
            [
                [
                    np.nan if getattr(b, t) is None else float(getattr(b, t))
                    for t in terms
                ]
                for b in budgets
            ]
        )
    return np.array(rows)


def compare_run(ref: Run, var: Run) -> dict:
    """Bitwise flags, field differences and the budget of VAR against REF."""
    out: dict = {"outers": var.outers, "ref_outers": ref.outers}
    flow = {name: 0.0 for name in FLOW_FIELDS}
    flow_same = len(ref.states) == len(var.states)
    for a, b in zip(ref.states, var.states, strict=False):
        for name in FLOW_FIELDS:
            x, y = getattr(a, name), getattr(b, name)
            if x is None and y is None:
                continue
            flow[name] = max(flow[name], relative(x, y))
            flow_same = flow_same and same_bytes(x, y)
    out["flow"] = flow
    out["flow_bitwise"] = flow_same
    fr, fv = ref.frames, var.frames
    out["frames_aligned"] = same_bytes(fr.steps, fv.steps)
    thresholds = np.array(ref.config.class_thresholds("the probe"), dtype=np.float64)
    conc, point = [], []
    fields_same = out["frames_aligned"]
    for k in sorted(fr.fields):
        a, b = fr.fields[k], fv.fields[k]
        if a.shape != b.shape:
            conc.append(math.inf)
            point.append(math.inf)
            fields_same = False
            continue
        conc.append(relative(a, b))
        point.append(pointwise(a, b, POINTWISE_FLOOR * thresholds[k]))
        fields_same = fields_same and same_bytes(a, b)
    out["concentration"] = conc
    out["pointwise"] = point
    out["fields_bitwise"] = fields_same
    br, bv = budget_rows(fr), budget_rows(fv)
    out["budgets_bitwise"] = same_bytes(br, bv)
    sr, sv = ref.sensors, var.sensors
    out["sensors"] = (
        relative(sr.channels, sv.channels)
        if sr.channels.shape == sv.channels.shape
        else math.inf
    )
    out["sensors_bitwise"] = same_bytes(sr.channels, sv.channels) and same_bytes(
        sr.readings, sv.readings
    )
    ar, av = ref.arrival, var.arrival
    out["arrival_bitwise"] = same_bytes(ar.maps, av.maps)
    steps = step_moves(ar.maps, av.maps, ref.summary["intervals"])
    out["arrival_moves"] = counts(steps)
    lr = np.array(_floats(ref.summary["latencies"]))
    lv = np.array(_floats(var.summary["latencies"]))
    onset = np.array(ar.onsets)[:, None, None]
    out["sensor_moves"] = counts(
        step_moves(lr + onset, lv + onset, ref.summary["intervals"])
    )
    out["budget"] = [budget_end(b) for b in var.frames.budgets[-1]]
    out["ref_budget"] = [budget_end(b) for b in ref.frames.budgets[-1]]
    out["bitwise"] = all(
        out[key]
        for key in (
            "flow_bitwise",
            "fields_bitwise",
            "budgets_bitwise",
            "sensors_bitwise",
            "arrival_bitwise",
        )
    )
    out["sequential"] = var.summary.get("sequential")
    var.release()
    return out


def _floats(value: object) -> object:
    """JSON's text inf back to floats."""
    if isinstance(value, list):
        return [_floats(v) for v in value]
    if isinstance(value, str):
        return float(value)
    return value


def budget_end(b: object) -> tuple[float, float]:
    """(residual over supply, leftover over supply) at the run's end."""
    return float(b.relative()), float(b.leftover_relative())


# ---------------------------------------------------------------------------
# Per grid: placement and clear times
# ---------------------------------------------------------------------------


class Placement:
    """One grid's pools, and per variant the searches, the file's four and the clear times."""

    def __init__(self, grid: str) -> None:
        self.grid = grid
        self.scenarios = probe85.GRID_SCENARIOS[grid]
        ref = {s: Run(REFERENCE, s, grid) for s in self.scenarios}
        configs = [r.config for r in ref.values()]
        self.config = configs[0]
        self.mesh = Mesh(self.config)
        spec = self.config.placement
        assert spec is not None
        self.spec = spec
        surfaces = tuple(m for m in spec.mount if m != MOUNT_STAND)
        self.pools = {
            SURF: candidates(self.mesh, configs, surfaces, spec.standoff),
            STAND: candidates(
                self.mesh,
                configs,
                spec.mount,
                spec.standoff,
                stand_height=spec.stand_height,
            ),
            FREE: candidates(self.mesh, configs, (MOUNT_FREE,), spec.standoff),
        }
        self.sensors = sensor_cells(self.config, self.mesh)
        self.plans = {s: r.summary["intervals"] for s, r in ref.items()}
        self.onsets = {s: r.arrival.onsets for s, r in ref.items()}
        self.results = {v: self.of(v) for v in probe85.VARIANTS}

    def of(self, variant: str) -> dict:
        """The searches, the layouts' per-unit latencies and order, and the clear times."""
        runs = {s: Run(variant, s, self.grid) for s in self.scenarios}
        units = detection_units({s: r.arrival for s, r in runs.items()})
        out: dict = {"labels": units.labels, "rules": {}}
        for rule in RULES:
            cells = {FILE: tuple(self.sensors)}
            for name, pool in self.pools.items():
                cells[name] = search(units, pool, self.spec.count, rule).cells
            scored = {n: given_layout(units, c) for n, c in cells.items()}
            per_unit = {
                n: np.array(
                    [
                        min(units.latencies[u, j, i] for j, i in c)
                        for u in range(len(units.labels))
                    ]
                )
                for n, c in cells.items()
            }
            # rank sorts the Layout objects it is given, so identity names them
            # even when two searches return the same cells.
            name_of = {id(lay): n for n, lay in scored.items()}
            names = [name_of[id(lay)] for lay in rank(scored.values())]
            out["rules"][rule] = {
                "cells": {n: tuple(sorted(c)) for n, c in cells.items()},
                "worst": {n: s.score.worst for n, s in scored.items()},
                "mean": {n: s.score.mean for n, s in scored.items()},
                "per_unit": per_unit,
                "order": names,
            }
        clear = clear_units({s: r.path for s, r in runs.items()})
        times, _cells = room_clear(clear)
        out["clear"] = clear
        out["room"] = times
        return out

    def compare(self, variant: str) -> dict:
        """VARIANT's decisions against the reference's."""
        ref, var = self.results[REFERENCE], self.results[variant]
        assert ref["labels"] == var["labels"]
        out: dict = {"rules": {}}
        for rule in RULES:
            r, v = ref["rules"][rule], var["rules"][rule]
            same_cells = {n: r["cells"][n] == v["cells"][n] for n in LAYOUTS}
            moves = []
            for n in LAYOUTS:
                for u, (scenario, _event, _k) in enumerate(ref["labels"]):
                    onset = float(self.onsets[scenario][0])
                    a = np.array([r["per_unit"][n][u] + onset])
                    b = np.array([v["per_unit"][n][u] + onset])
                    moves.append(step_moves(a, b, self.plans[scenario])[0])
            out["rules"][rule] = {
                "same_cells": same_cells,
                "same_order": r["order"] == v["order"],
                "order": v["order"],
                "moves": counts(np.array(moves)),
            }
        cr, cv = ref["clear"], var["clear"]
        spacing = cr.spacing[:, None, None]
        out["clear_moves"] = counts(moved(cr.clear, cv.clear, spacing))
        out["room_moves"] = counts(moved(ref["room"], var["room"], cr.spacing))
        return out


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------


def runs_table(results: dict) -> list[str]:
    """P1: the runs, their outer counts and whether each equals the reference bitwise."""
    rows = []
    for (grid, scenario), per in results.items():
        ref = Run(REFERENCE, scenario, grid)
        rows.append(
            [
                grid,
                scenario,
                REFERENCE,
                "-",
                " / ".join(str(o) for o in ref.outers),
                str(ref.summary["steps"]),
                f"{ref.summary['wall_seconds']:.0f}",
                "-",
            ]
        )
        for variant, c in per.items():
            run = Run(variant, scenario, grid)
            changed = run.summary["changed"]
            text = (
                "; ".join(f"{name} {old!r} to {new!r}" for name, old, new in changed)
                if changed
                else {
                    "repeat": "none",
                    SEQUENTIAL: "the leftover's sum in row-major order",
                    OUTER: "each flow solve's first stop deferred one iteration",
                }[variant]
            )
            if variant == INFLOW:
                text = f"{len(changed)} inlet velocities x (1 + 1e-15)"
            rows.append(
                [
                    grid,
                    scenario,
                    variant,
                    text,
                    " / ".join(str(o) for o in c["outers"]),
                    str(run.summary["steps"]),
                    f"{run.summary['wall_seconds']:.0f}",
                    yes(c["bitwise"]),
                ]
            )
    return [
        "### P1. The runs: what each changed, the outer iterations per flow state, the steps, and whether it equals the reference bit for bit",
        "",
        *table(
            [
                "grid",
                "scenario",
                "variant",
                "change",
                "outer per state",
                "steps",
                "wall, s",
                "bitwise the reference",
            ],
            rows,
        ),
        "",
        "Bitwise: every flow state's u, v, p, k, eps and nu_t, every stored frame of every class, every stored budget row, the sensor readings and channels, and the arrival maps, byte for byte. The flow states are taken in the order the intervals first use them, since the inflow variant's fingerprints differ. Each run started from rest in its own process on one BLAS thread, at most three at once.",
    ]


def prompt81_table() -> list[str]:
    """P1b: the 80x30 references against prompt 81's runs of the same files."""
    rows = []
    for scenario, name in PROMPT81_NAMES.items():
        mine = probe85.run_path(REFERENCE, scenario, "80x30")
        theirs = PROMPT81 / f"{name}.h5"
        if not theirs.exists():
            rows.append([scenario, "missing", "", "", ""])
            continue
        a, b = RunFile(mine), RunFile(theirs)
        fa, fb = a.read_frames(), b.read_frames()
        frames = same_bytes(fa.steps, fb.steps) and all(
            same_bytes(fa.fields[k], fb.fields[k]) for k in fa.fields
        )
        budgets = same_bytes(budget_rows(fa), budget_rows(fb))
        arrival = same_bytes(arrival_from_run(mine).maps, arrival_from_run(theirs).maps)
        sa, sb = a.read_sensor_readings(), b.read_sensor_readings()
        sensors = same_bytes(sa.channels, sb.channels)
        rows.append([scenario, yes(frames), yes(budgets), yes(arrival), yes(sensors)])
    return [
        "### P1b. The 80x30 references against prompt 81's runs of the same scenario files (`results/builder81/placement/`), bit for bit",
        "",
        *table(
            ["scenario", "frames", "budget rows", "arrival maps", "sensor channels"],
            rows,
        ),
        "",
        "Prompt 81's runs copied their flow states from prompt 79's; this pass's solved them from rest, on main at this branch's base.",
    ]


def fields_table(results: dict) -> list[str]:
    """P2: the largest relative difference of every field."""
    rows = []
    for (grid, scenario), per in results.items():
        for variant, c in per.items():
            rows.append(
                [
                    grid,
                    scenario,
                    variant,
                    *[e(c["flow"][n]) for n in FLOW_FIELDS],
                    e(max(c["concentration"])),
                    e(max(c["pointwise"])),
                    e(c["sensors"]),
                ]
            )
    return [
        "### P2. Each field's largest relative difference from the reference",
        "",
        *table(
            [
                "grid",
                "scenario",
                "variant",
                *FLOW_FIELDS,
                "concentration",
                "concentration, pointwise",
                "sensor channels",
            ],
            rows,
        ),
        "",
        "Flow fields: max over the cells of |difference| over max |reference|, the largest over the run's flow states. Concentration: the same over every stored frame and cell, the largest over the five classes. Pointwise: the largest |difference| / |reference| over the stored frames' cells where the reference reads at least a hundredth of the class's threshold. Sensor channels: the cumulative readings of the file's four sensors at every step, max |difference| over max |reference|.",
    ]


def informative_table(placements: dict) -> list[str]:
    """P3b: how many of the reference's latencies and clear times say something.

    A latency never reached in the reference and never in a variant cannot
    move, and a clear time equal to the unit's end (no frame from the end on
    at or above) moves only if a variant crosses the threshold there; so the
    zero moves of P3 and P5 are read against these counts.
    """
    rows = []
    for grid, place in placements.items():
        ref = place.results[REFERENCE]
        arrival = reached = 0
        for scenario in probe85.GRID_SCENARIOS[grid]:
            maps = Run(REFERENCE, scenario, grid).arrival.maps
            live = ~np.isnan(maps)
            arrival += int(live.sum())
            reached += int(np.isfinite(maps).sum())
        sensors = sensors_reached = 0
        for scenario in probe85.GRID_SCENARIOS[grid]:
            lat = np.array(_floats(Run(REFERENCE, scenario, grid).summary["latencies"]))
            sensors += lat.size
            sensors_reached += int(np.isfinite(lat).sum())
        units = len(ref["labels"])
        layout_reached = {
            rule: sum(
                int(np.isfinite(ref["rules"][rule]["per_unit"][n]).sum())
                for n in LAYOUTS
            )
            for rule in RULES
        }
        clear = ref["clear"]
        ends = clear.ends[:, None, None]
        live = ~np.isnan(clear.clear)
        after = int((np.isfinite(clear.clear) & (clear.clear > ends)).sum())
        not_clear = int(np.isinf(clear.clear).sum())
        room_after = int((np.isfinite(ref["room"]) & (ref["room"] > clear.ends)).sum())
        room_not = int(np.isinf(ref["room"]).sum())
        rows.append(
            [
                grid,
                f"{reached:,} of {arrival:,}",
                f"{sensors_reached} of {sensors}",
                " / ".join(
                    f"{layout_reached[r]} of {units * len(LAYOUTS)}" for r in RULES
                ),
                f"{after:,} / {not_clear:,} of {int(live.sum()):,}",
                f"{room_after} / {room_not} of {len(ref['room'])}",
            ]
        )
    return [
        "### P3b. What the zero moves are read against: the reference's latencies reached and its clear times after the event's end",
        "",
        *table(
            [
                "grid",
                "arrival times reached (cells x events x channels)",
                "file's sensors' latencies reached",
                "layouts' unit latencies reached, worst case / mean",
                "cells' clear times after the end / not clear",
                "room clear times after the end / not clear",
            ],
            rows,
        ),
        "",
        "Reached: finite in the reference. A clear time after the end is one at a later frame than the scenario's last event's end, so some frame from the end on read at or above there; not clear is still at or above at the run's last frame. A latency or clear time that is neither reached nor after the end is unchanged unless a variant crosses the threshold where the reference never did, which P3's and P5's one-only counts record.",
    ]


def latency_table(results: dict, placements: dict) -> list[str]:
    """P3: the arrival maps', the file's sensors' and the layouts' latencies moved."""
    rows = []
    for grid, place in placements.items():
        for variant in OTHERS:
            per = [results[(grid, s)][variant] for s in probe85.GRID_SCENARIOS[grid]]
            arrival = [c["arrival_moves"] for c in per]
            sensor = [c["sensor_moves"] for c in per]
            cmp = place.compare(variant)
            layout = [cmp["rules"][r]["moves"] for r in RULES]
            rows.append(
                [
                    grid,
                    variant,
                    *_moves_text(arrival),
                    *_moves_text(sensor)[1:],
                    *_moves_text(layout)[1:],
                ]
            )
    return [
        "### P3. Detection latencies moved: every cell's arrival time per event and channel, the file's four sensors' latencies, and the searched layouts' latencies per unit",
        "",
        *table(
            [
                "grid",
                "variant",
                "cells x events x channels",
                "moved 1 step / > 1 / one only",
                "largest move, steps",
                "sensors: moved 1 / > 1 / one only",
                "largest, steps",
                "layouts: moved 1 / > 1 / one only",
                "largest, steps",
            ],
            rows,
        ),
        "",
        "A move is the difference over the step of the reference's interval at the later of the two times, rounded to the nearest step; one only is a latency reached in one run and never in the other. The cells are every non-SOLID cell of every scenario on the grid. The file's sensors: every event, channel and sensor of the monitor's latencies. The layouts: every unit of the four layouts (the file's four and the three searches) under both rules, each run's layouts searched on its own maps.",
    ]


def _moves_text(parts: list[tuple]) -> list[str]:
    """Sum the counts and take the largest move."""
    n = sum(p[0] for p in parts)
    one = sum(p[1] for p in parts)
    more = sum(p[2] for p in parts)
    only = sum(p[3] for p in parts)
    largest = max(p[4] for p in parts)
    return [f"{n:,}", f"{one} / {more} / {only}", f"{largest:.2f}"]


def placement_table(placements: dict) -> list[str]:
    """P4: the searched cells and the layouts' order."""
    rows = []
    for grid, place in placements.items():
        ref = place.results[REFERENCE]
        for rule in RULES:
            rows.append(
                [grid, REFERENCE, rule, "-", ", ".join(ref["rules"][rule]["order"])]
            )
            for variant in OTHERS:
                c = place.compare(variant)["rules"][rule]
                changed = [n for n, same in c["same_cells"].items() if not same]
                rows.append(
                    [
                        grid,
                        variant,
                        rule,
                        ", ".join(changed) if changed else "none",
                        ", ".join(c["order"])
                        + ("" if c["same_order"] else " (differs)"),
                    ]
                )
    return [
        "### P4. The placement: which searched layouts' cells differ from the reference's, and the four layouts' order",
        "",
        *table(
            [
                "grid",
                "variant",
                "rule",
                "layouts whose cells differ",
                "order, best first",
            ],
            rows,
        ),
        "",
        "The searches are `placement.search` over the grid's runs' step-resolved maps with main's placement spec (four sensors): surface, the mounting surfaces without the stand; product, with the stand; free, every non-SOLID cell outside the exclusions. The order is `placement.rank` of the four, each scored over every unit.",
    ]


def clear_table(placements: dict) -> list[str]:
    """P5: the clear times moved."""
    rows = []
    for grid, place in placements.items():
        for variant in OTHERS:
            c = place.compare(variant)
            rows.append(
                [
                    grid,
                    variant,
                    *_moves_text([c["clear_moves"]]),
                    *_moves_text([c["room_moves"]]),
                ]
            )
    return [
        "### P5. Clear times moved: every cell's and the room's, per all-clear unit (scenario, channel)",
        "",
        *table(
            [
                "grid",
                "variant",
                "cells x units",
                "moved 1 frame / > 1 / one only",
                "largest, frames",
                "room: units",
                "room: moved 1 / > 1 / one only",
                "room: largest, frames",
            ],
            rows,
        ),
        "",
        "`placement.clear_units` over the grid's runs, frame-resolved (a frame every 100 steps). A move is the difference over the unit's frame spacing.",
    ]


def budget_table(results: dict) -> list[str]:
    """P6: the budget at the end against REQ-T05."""
    rows = []
    for (grid, scenario), per in results.items():
        ref = next(iter(per.values()))["ref_budget"]
        rows.append(
            [
                grid,
                scenario,
                REFERENCE,
                e(max(abs(r) for r, _ in ref)),
                e(max(abs(lo) for _, lo in ref)),
                "-",
                yes(all(abs(r) < REQ_T05 and abs(lo) < REQ_T05 for r, lo in ref)),
            ]
        )
        for variant, c in per.items():
            b = c["budget"]
            change = max(
                abs(lo - lr)
                for (_, lo), (_, lr) in zip(b, c["ref_budget"], strict=True)
            )
            rows.append(
                [
                    grid,
                    scenario,
                    variant,
                    e(max(abs(r) for r, _ in b)),
                    e(max(abs(lo) for _, lo in b)),
                    e(change),
                    yes(all(abs(r) < REQ_T05 and abs(lo) < REQ_T05 for r, lo in b)),
                ]
            )
    return [
        "### P6. The budget at the run's end against REQ-T05 (the largest over the five classes)",
        "",
        *table(
            [
                "grid",
                "scenario",
                "variant",
                "residual / supplied",
                "leftover / supplied",
                "leftover / supplied, change from the reference",
                "within 1e-4",
            ],
            rows,
        ),
    ]


def sequential_table(results: dict) -> list[str]:
    """P7: what the sequential leftover changed."""
    rows = []
    for (grid, scenario), per in results.items():
        s = per[SEQUENTIAL]["sequential"]
        rows.append(
            [
                grid,
                scenario,
                f"{s['formed']:,}",
                f"{s['differ']:,}",
                e(s["largest_relative"]),
                f"{s['tests']:,}",
                str(s["flipped"]),
            ]
        )
    return [
        "### P7. The leftover summed in row-major order: how many sums and stop verdicts the order changed",
        "",
        *table(
            [
                "grid",
                "scenario",
                "leftovers formed",
                "differing from NumPy's sum",
                "largest relative difference",
                "stop tests against the share",
                "verdicts flipped",
            ],
            rows,
        ),
        "",
        "Formed counts every implicit solve's leftover in the run, the k and eps solves' included (their stop has no share, so they form it once and do not test it). A flipped verdict is a test whose two sums fall on opposite sides of the share.",
    ]


def band_table(results: dict, placements: dict) -> tuple[list[str], dict]:
    """P8: the spread of each output per family of variants, the band's basis."""
    rows, band = [], {}
    for family, variants in FAMILIES.items():
        cells: dict[str, float] = {
            "flow": 0.0,
            "concentration": 0.0,
            "pointwise": 0.0,
            "arrival": 0.0,
            "layouts": 0.0,
            "clear": 0.0,
            "room": 0.0,
            "budget": 0.0,
        }
        only = {"arrival": 0, "layouts": 0, "clear": 0}
        order_same = True
        for (_grid, _scenario), per in results.items():
            for v in variants:
                c = per[v]
                cells["flow"] = max(cells["flow"], max(c["flow"].values()))
                cells["concentration"] = max(
                    cells["concentration"], max(c["concentration"])
                )
                cells["pointwise"] = max(cells["pointwise"], max(c["pointwise"]))
                cells["arrival"] = max(cells["arrival"], c["arrival_moves"][4])
                only["arrival"] += c["arrival_moves"][3]
                change = max(
                    abs(lo - lr)
                    for (_, lo), (_, lr) in zip(
                        c["budget"], c["ref_budget"], strict=True
                    )
                )
                cells["budget"] = max(cells["budget"], change)
        for place in placements.values():
            for v in variants:
                cmp = place.compare(v)
                for rule in RULES:
                    m = cmp["rules"][rule]["moves"]
                    cells["layouts"] = max(cells["layouts"], m[4])
                    only["layouts"] += m[3]
                    order_same = order_same and cmp["rules"][rule]["same_order"]
                cells["clear"] = max(cells["clear"], cmp["clear_moves"][4])
                only["clear"] += cmp["clear_moves"][3]
                cells["room"] = max(cells["room"], cmp["room_moves"][4])
        band[family] = {**cells, "only": only, "order_same": order_same}
        rows.append(
            [
                family,
                e(cells["flow"]),
                e(cells["concentration"]),
                e(cells["pointwise"]),
                f"{cells['arrival']:.2f} ({only['arrival']})",
                f"{cells['layouts']:.2f} ({only['layouts']})",
                yes(order_same),
                f"{cells['clear']:.2f} ({only['clear']})",
                f"{cells['room']:.2f}",
                e(cells["budget"]),
            ]
        )
    lines = [
        "### P8. The spread of each output over each family of variants, both grids and every scenario",
        "",
        *table(
            [
                "variants",
                "flow fields",
                "concentration",
                "concentration, pointwise",
                "arrival times, steps (one only)",
                "layouts' latencies, steps (one only)",
                "layouts' order the same",
                "cells' clear times, frames (one only)",
                "room clear times, frames",
                "leftover / supplied, change",
            ],
            rows,
        ),
        "",
        "Each entry is the largest over the family's variants of P2 to P6's measure; the counts in brackets are the latencies or clear times reached in one run and never in the other.",
    ]
    return lines, band


def predictions_table(
    results: dict, placements: dict, band: dict
) -> tuple[list[str], dict]:
    """P9: the predictions scored by the rule committed with them."""
    # (a): every latency of the four tolerance variants within one step.
    worst_a, only_a = 0.0, 0
    for (_grid, _s), per in results.items():
        for v in TOLERANCE:
            for key in ("arrival_moves", "sensor_moves"):
                worst_a = max(worst_a, per[v][key][4])
                only_a += per[v][key][3]
    lay_a, lay_only = 0.0, 0
    for place in placements.values():
        for v in TOLERANCE:
            for rule in RULES:
                m = place.compare(v)["rules"][rule]["moves"]
                lay_a, lay_only = max(lay_a, m[4]), lay_only + m[3]
    a_holds = max(worst_a, lay_a) <= 1.5 and only_a == 0 and lay_only == 0
    # (b): the inflow variant.
    b_field = 0.0
    b_decisions = True
    for (_grid, _s), per in results.items():
        c = per[INFLOW]
        b_field = max(b_field, max(c["flow"].values()), max(c["concentration"]))
        b_decisions = (
            b_decisions
            and c["arrival_moves"][1:4] == (0, 0, 0)
            and c["sensor_moves"][1:4] == (0, 0, 0)
            and all(
                (abs(r) < REQ_T05 and abs(lo) < REQ_T05)
                == (abs(rr) < REQ_T05 and abs(lr) < REQ_T05)
                for (r, lo), (rr, lr) in zip(c["budget"], c["ref_budget"], strict=True)
            )
        )
    for place in placements.values():
        cmp = place.compare(INFLOW)
        for rule in RULES:
            r = cmp["rules"][rule]
            b_decisions = (
                b_decisions
                and all(r["same_cells"].values())
                and r["same_order"]
                and r["moves"][1:4] == (0, 0, 0)
            )
        b_decisions = (
            b_decisions
            and cmp["clear_moves"][1:4] == (0, 0, 0)
            and cmp["room_moves"][1:4] == (0, 0, 0)
        )
    b_holds = b_decisions and b_field <= B_FIELD_LIMIT
    # (c): the tolerance variants' fields.
    c_field, c_where = 0.0, ""
    for (grid, scenario), per in results.items():
        for v in TOLERANCE:
            c = per[v]
            for name, value in [
                *c["flow"].items(),
                ("concentration", max(c["concentration"])),
            ]:
                if value > c_field:
                    c_field, c_where = value, f"{name}, {v}, {scenario} {grid}"
    c_holds = c_field <= C_FIELD_LIMIT
    rows = [
        [
            "(a)",
            "a 1% tolerance change moves no detection latency by more than one step",
            f"largest move {max(worst_a, lay_a):.2f} steps over the cells and the file's sensors ({only_a} reached in one only); layouts {lay_a:.2f} ({lay_only} one only)",
            "HOLDS" if a_holds else "FAILS",
        ],
        [
            "(b)",
            "the 1e-15 inflow change leaves every decision unchanged and no field above 1e-10",
            f"decisions unchanged: {yes(b_decisions)}; largest field difference {e(b_field)}",
            "HOLDS" if b_holds else "FAILS",
        ],
        [
            "(c)",
            "no field moves by more than 1e-6 under the tolerance changes",
            f"largest {e(c_field)} ({c_where})",
            "HOLDS" if c_holds else "FAILS",
        ],
    ]
    return (
        [
            "### P9. The predictions, scored by the rule committed with them",
            "",
            *table(["", "prediction", "measured", "verdict"], rows),
        ],
        {"a": a_holds, "b": b_holds, "c": c_holds},
    )


# ---------------------------------------------------------------------------
# Build, splice, check
# ---------------------------------------------------------------------------


def build() -> tuple[str, dict[str, str]]:
    """Every table, as the tables file's text and the report's blocks."""
    results: dict = {}
    for grid, scenarios in probe85.GRID_SCENARIOS.items():
        for scenario in scenarios:
            runs = runs_of(grid, scenario)
            ref = runs[REFERENCE]
            results[(grid, scenario)] = {v: compare_run(ref, runs[v]) for v in OTHERS}
            ref.release()
    placements = {grid: Placement(grid) for grid in probe85.GRID_SCENARIOS}
    band_lines, band = band_table(results, placements)
    pred_lines, _verdicts = predictions_table(results, placements, band)
    blocks = {
        "runs": "\n".join([*runs_table(results), "", *prompt81_table()]),
        "fields": "\n".join(fields_table(results)),
        "decisions": "\n".join(
            [
                *latency_table(results, placements),
                "",
                *informative_table(placements),
                "",
                *placement_table(placements),
                "",
                *clear_table(placements),
                "",
                *budget_table(results),
                "",
                *sequential_table(results),
            ]
        ),
        "band": "\n".join(band_lines),
        "predictions": "\n".join(pred_lines),
    }
    text = "# Prompt 85's probe tables\n\nGenerated by `tables85.py` from `results/builder85/`.\n\n"
    text += "\n\n".join(
        blocks[k] for k in ("predictions", "runs", "fields", "decisions", "band")
    )
    return text + "\n", blocks


def splice(report: str, blocks: dict[str, str]) -> str:
    """REPORT with each marked block replaced."""
    for name, body in blocks.items():
        begin, end = f"<!-- tables85 {name} begin -->", f"<!-- tables85 {name} end -->"
        if begin not in report:
            continue
        head, rest = report.split(begin, 1)
        _old, tail = rest.split(end, 1)
        report = f"{head}{begin}\n{body}\n{end}{tail}"
    return report


def main() -> int:
    """Write, or with --check compare."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text, blocks = build()
    report = REPORT.read_text(encoding="utf-8")
    spliced = splice(report, blocks)
    if args.check:
        ok = TABLES.read_text(encoding="utf-8") == text and spliced == report
        print("tables85 --check:", "matches" if ok else "DIFFERS")
        return 0 if ok else 1
    TABLES.write_text(text, encoding="utf-8")
    REPORT.write_text(spliced, encoding="utf-8")
    print("wrote", TABLES, "and spliced", REPORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
