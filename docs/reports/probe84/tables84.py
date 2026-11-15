"""Tables of prompt 84's report from the records in results/builder84.

    python docs/reports/probe84/tables84.py           # write the tables, splice the report
    python docs/reports/probe84/tables84.py --check   # regenerate and compare

Every number the report states is in a table here, generated from the
measurement records (``probe84.py``, ``run84.py``, ``overhead84.py``), the
code-read counts (``counts84.py``) and the stored runs of prompts 79 and 81.
Writes ``tables84.md`` beside it and copies each table into
``docs/reports/profile84.md`` between ``<!-- tables84 NAME begin/end -->``;
``--check`` regenerates both and exits 1 if either differs.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import counts84  # noqa: E402 -- follows sys.path.insert
import probe84  # noqa: E402

ROOT = probe84.ROOT
RESULTS = probe84.RESULTS
OUT = HERE / "tables84.md"
REPORT = ROOT / "docs" / "reports" / "profile84.md"
GRIDS = ["80x30", "160x60", "320x120"]
SWEEPS79 = (
    ROOT / "results" / "builder79" / "product" / "equipment_dust.sweeps.66344.npz"
)
SUMMARY79 = ROOT / "results" / "builder79" / "product" / "equipment_dust.txt"
SUMMARY81 = ROOT / "results" / "builder81" / "product" / "equipment_dust.txt"
# The stored 320x120 run's intervals (its summary): steps of each.
PLANS81 = (2760, 8280, 22079)

FLOW_STAGES = [
    "flow.momentum",
    "flow.pressure",
    "flow.turbulence",
    "flow.boundary",
    "flow.stopping",
    "flow.io",
    "flow.remainder",
]
CLASS_STAGES = [
    "class.advection",
    "class.divergence",
    "class.source",
    "class.implicit",
    "class.booking",
    "class.remainder",
]
STEP_STAGES = ["step.monitor", "step.arrival", "step.frames", "step.checkpoint"]
MARCH_STAGES = [*CLASS_STAGES, *STEP_STAGES, "march.remainder"]
NAMES = {
    "flow.momentum": "momentum predictor",
    "flow.pressure": "pressure correction",
    "flow.turbulence": "turbulence update",
    "flow.boundary": "boundary updates",
    "flow.stopping": "stopping rule",
    "flow.io": "steady checkpoints",
    "flow.remainder": "remainder",
    "class.advection": "advective fluxes and limiter",
    "class.divergence": "divergence update",
    "class.source": "source",
    "class.implicit": "implicit solve",
    "class.booking": "deposition and budget booking",
    "class.remainder": "remainder (input checks)",
    "step.monitor": "monitor (sensor channels, alerts)",
    "step.arrival": "arrival maps",
    "step.frames": "frame writes",
    "step.checkpoint": "checkpoint writes",
    "march.remainder": "march remainder",
}
# Region markers that fire per call (probe84.REGIONS), for the event estimate.
MARKERS_PER_OUTER = len(probe84.REGIONS["solve_steady"]) - 1
MARKERS_PER_CLASS_STEP = len(probe84.REGIONS["solve_timestep"]) - 1


# ---------------------------------------------------------------------------
# Reading the records
# ---------------------------------------------------------------------------


def load(job: str) -> dict:
    """The JSON record of JOB in results/builder84."""
    return json.loads((RESULTS / f"{job}.json").read_text())


# Under the lean instrument a wrapped call directly inside solve_timestep's
# inline lump takes the stage of the region its call line lies in
# (probe84.REGIONS): stable_dt and the nu_t check in the input checks, the
# advecting velocities and fluxes in the advection, the face nu_t and the
# solve in the implicit step, the deposition and the content in the booking.
LEAN_CALLEE = {
    "stable_dt": "class.remainder",
    "check_nu_t": "class.remainder",
    "advecting_velocities": "class.advection",
    "advective_flux": "class.advection",
    "face_nu_t": "class.implicit",
    "implicit_solve": "class.implicit",
    "book_deposition": "class.booking",
    "in_domain": "class.booking",
}
# The stage of each region of solve_timestep (probe84.REGIONS).
REGION_STAGE = {
    region: probe84.STAGES[f"transport:{region}"]
    for _text, region in probe84.REGIONS["solve_timestep"]
}
# The regions-on record each lean record's inline lump is split by: the same
# scope where one exists (the first instrument's whole 80x30 run), else the
# grid's 9 s to 14 s window.
SPLIT_OF = {
    "run_80x30": "run_80x30_regions",
    "run_160x60": "split_window_160x60",
    "window_80x30": "split_window_80x30",
    "window_160x60": "split_window_160x60",
    "window_320x120": "split_window_320x120",
}


def stage_of(path: list[str]) -> str:
    """The stage of PATH, a lean call inside the inline lump by its callee."""
    stage = probe84.stage_of_path(path)
    if stage != "class.inline":
        return stage
    at = next(i for i, label in enumerate(path) if label.endswith(":inline"))
    if at == len(path) - 1:
        return "class.inline"
    return LEAN_CALLEE[path[at + 1]]


def split_fractions(record: dict) -> dict[str, float]:
    """A regions-on record's inline time by stage, as fractions of its total.

    Only the regions' own lines count (the paths ending at a region label),
    the share of solve_timestep's frame that the lean instrument lumps.
    """
    own: dict[str, int] = defaultdict(int)
    for path, ns in record["charge"]:
        head, _, region = path[-1].partition(":")
        if head.startswith("transport") and region in REGION_STAGE:
            own[REGION_STAGE[region]] += ns
    total = sum(own.values())
    return {k: v / total for k, v in own.items()}


def by_stage(record: dict, prefix: str = "", job: str | None = None) -> dict[str, int]:
    """Exclusive ns per stage, over paths whose stage starts with PREFIX.

    A lean record's inline lump is split by ``SPLIT_OF[job]``'s fractions.
    """
    out: dict[str, float] = defaultdict(float)
    lump = 0
    for path, ns in record["charge"]:
        stage = stage_of(path)
        if stage == "class.inline":
            lump += ns
        elif stage.startswith(prefix):
            out[stage] += ns
    if lump:
        fractions = split_fractions(load(SPLIT_OF[job or record["job"]]))
        for stage, f in fractions.items():
            if stage.startswith(prefix):
                out[stage] += lump * f
    return out


def inline_lump(record: dict) -> int:
    """Exclusive nanoseconds charged to the inline lump of a lean record."""
    return path_ns(record, lambda p: p[-1].endswith(":inline"))


def path_ns(record: dict, test) -> int:
    """Exclusive ns over the paths TEST accepts."""
    return sum(ns for path, ns in record["charge"] if test(path))


def calls(record: dict, label: str) -> int:
    """Entries into any path ending in LABEL."""
    return sum(n for path, n in record["calls"] if path[-1].partition(":")[0] == label)


def class_label_calls(record: dict) -> int:
    """Entries into a class step's ``checks`` region of a regions-on record."""
    return sum(
        n
        for path, n in record["calls"]
        if path[-1].startswith("transport") and path[-1].endswith(":checks")
    )


def s(ns: float) -> str:
    """NS nanoseconds as seconds, two decimals."""
    return f"{ns / 1e9:.2f}"


def pct(x: float) -> str:
    """X as a percentage, one decimal."""
    return f"{100 * x:.1f}%"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A Markdown table of HEADER and ROWS, as lines."""
    out = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return [*out, ""]


# ---------------------------------------------------------------------------
# Facts of the configuration that the counts depend on
# ---------------------------------------------------------------------------


def turbulence_sides(grid: str) -> int:
    """Full-field wall sides of the k-epsilon boundary on GRID."""
    from src.boundary_staggered import StaggeredBoundary
    from src.config import SimConfig
    from src.mesh import Mesh
    from src.scenarios import ScenarioManager
    from src.turbulence import TurbulenceBoundary

    config = SimConfig(probe84.config_path(grid))
    mesh = Mesh(config)
    icfg = ScenarioManager(config, mesh).intervals()[0].config
    walls = TurbulenceBoundary(mesh, icfg, StaggeredBoundary(mesh, icfg))
    return len(walls._sides)


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------


def machine(lines: list[str]) -> None:
    """T1: the machine and the runs; appends to LINES."""
    env = load("run_80x30")["environment"]
    lines += ["### T1. The machine and the runs", ""]
    lines += table(
        ["item", "value"],
        [
            ["CPU", env["machine"]],
            ["Python, NumPy", f"{env['python']}, {env['numpy']}"],
            [
                "BLAS",
                "; ".join(
                    f"{b['internal_api']} {b['version']}, {b['num_threads']} thread(s)"
                    for b in env["blas"]
                ),
            ],
            [
                "thread variables",
                ", ".join(f"{k}={v}" for k, v in env["threads"].items()),
            ],
            ["commit", env["commit"]],
            [
                "GPU (CUDA driver attributes)",
                f"NVIDIA GeForce RTX 4070 Laptop GPU, {counts84.SMS} SMs, {counts84.BUS_BITS}-bit bus, memory clock {counts84.MEMORY_CLOCK_HZ / 1e6:.0f} MHz, L2 {counts84.L2_BYTES // 2**20} MiB",
            ],
        ],
    )
    rows = []
    for job, _argv in __import__("run84").JOBS:
        procs = RESULTS / "procs" / f"{job}.json"
        if not procs.exists():
            continue
        top = json.loads(procs.read_text())["top"]
        if isinstance(top, dict):
            top = [top]
        rows.append(
            [
                job,
                json.loads(procs.read_text())["sampled"],
                ", ".join(f"{t['process']} {t['cpu_percent']}" for t in top[:4]),
            ]
        )
    lines += [
        "Processes using the CPU when each job started (percent of one logical processor, a 2 s sample of `% Processor Time`):",
        "",
    ]
    lines += table(["job", "sampled", "top four"], rows)


def calibration(lines: list[str]) -> dict[str, dict[str, float]]:
    """T2; returns the event estimate and the measured overhead per calibration job."""
    cost = json.loads((RESULTS / "overhead.json").read_text())
    call_ns, region_ns = cost["wrapped_call_ns"], cost["region_event_ns"]
    lines += ["### T2. Calibration: wrapped against unwrapped, in one process", ""]
    lines += [
        f"Per event, from `overhead84.py` (median of {len(cost['call_batches_ns'])} batches of "
        f"{cost['n']:,}): a wrapped call {call_ns:.0f} ns, a region marker {region_ns:.0f} ns.",
        "",
    ]
    rows = []
    estimates: dict[str, float] = {}
    differences: dict[str, float] = {}
    jobs = [
        (kind, grid, f"calibrate_{kind}_{grid}")
        for kind in ("flow", "march")
        for grid in GRIDS
    ]
    # The first instrument's window calibrations (line regions in the march),
    # and its first 80x30 window, too short to resolve 5% (run84.py, JOBS).
    jobs += [
        ("march", "80x30", "calibrate_march_80x30_short"),
        ("march", "80x30", "calibrate_march_80x30_regions"),
        ("march", "320x120", "calibrate_march_320x120_regions"),
    ]
    for kind, grid, job in jobs:
        r = load(job)
        walls = np.array(r["wall_ns"], dtype=float)
        wrapped = np.array([o == "wrapped" for o in r["order"]])
        plain = walls[~wrapped]
        measured = walls[wrapped].mean() / plain.mean() - 1.0
        spread = (plain.max() - plain.min()) / plain.mean()
        w = r["wrapped"][0]
        n_calls = sum(n for _p, n in w["calls"])
        if kind == "flow":
            regions = MARKERS_PER_OUTER * r["outer"]
            what = f"{r['outer']} outer iterations from rest"
        else:
            on = r.get("march_regions", True)
            regions = MARKERS_PER_CLASS_STEP * 5 * w["steps"] if on else 0
            instrument = "first, line regions" if on else "lean"
            what = f"window 9 s to {r['t_stop']} s ({w['steps']} steps), {instrument}"
        estimate = (n_calls * call_ns + regions * region_ns) / plain.mean()
        estimates[job] = estimate
        differences[job] = measured
        bitwise = "yes" if not any(r["differ"]) else f"no: {r['differ']}"
        rows.append(
            [
                grid,
                what,
                " / ".join(f"{x / 1e9:.2f}" for x in plain),
                " / ".join(f"{x / 1e9:.2f}" for x in walls[wrapped]),
                f"{100 * measured:+.1f}%",
                pct(spread),
                f"{n_calls:,}",
                f"{regions:,}",
                f"{100 * estimate:.2f}%",
                bitwise,
            ]
        )
    lines += table(
        [
            "grid",
            "run",
            "unwrapped walls, s",
            "wrapped walls, s",
            "wrapped less unwrapped (means)",
            "spread of the unwrapped",
            "wrapped calls",
            "region events (estimate)",
            "event estimate",
            "bitwise",
        ],
        rows,
    )
    lines += [
        "The order was unwrapped, wrapped, unwrapped, wrapped, unwrapped. Bitwise: every run's "
        "fields and budgets (march) or faces, pressure, k, eps, nu_t and residual history (flow) "
        "against the first run's. The region events are the markers per outer iteration "
        f"({MARKERS_PER_OUTER}) or per class step ({MARKERS_PER_CLASS_STEP}) times the calls, "
        "an estimate: a marker on a statement spanning lines may fire twice. The lean "
        "instrument has no region in the march. The prompt's 5% stop fired twice: on the "
        f"160x60 flow calibration ({100 * differences['calibrate_flow_160x60']:+.1f}%) and "
        "on the first instrument's 80x30 march window "
        f"({100 * differences['calibrate_march_80x30_short']:+.1f}% on the 9 s to 10.5 s "
        f"window, {100 * differences['calibrate_march_80x30_regions']:+.1f}% on the 9 s to 40 s "
        "one). The lean windows are within 3%. The flow calibrations' means rest on three "
        "unwrapped and two wrapped runs whose spread is wider than the difference, so they "
        "do not resolve 5%; the paired measurement is in the text.",
        "",
    ]
    rows = []
    for name in ("ablate_80x30_parts", "ablate_80x30_regions3"):
        r = load(name)
        walls = np.array(r["wall_ns"], dtype=float)
        parts = r["parts"]
        none = walls[[i for i, q in enumerate(parts) if q == "none"]]
        for i, q in enumerate(parts):
            if q == "none":
                continue
            rows.append(
                [
                    f"`{name}`",
                    str(i + 1),
                    q,
                    f"{walls[i] / 1e9:.2f}",
                    f"{100 * (walls[i] / none.mean() - 1):+.1f}%",
                ]
            )
        rows.append(
            [
                f"`{name}`",
                "",
                "none (all)",
                " / ".join(f"{x / 1e9:.2f}" for x in none),
                f"spread {pct((none.max() - none.min()) / none.mean())}",
            ]
        )
    lines += [
        "### T2b. Which part of the timers costs: 80x30 windows 9 s to 40 s, one process each "
        "list, in the order run",
        "",
    ]
    lines += table(
        ["record", "run", "timers", "wall, s", "against the unwrapped mean"], rows
    )
    lines += [
        "full: every wrapper and the march's line regions (the first instrument); lean: the "
        "instrument of T3 to T12; regions: the march's line regions alone.",
        "",
    ]
    return {"estimate": estimates, "measured": differences}


def flow_records() -> dict[str, dict]:
    """The flow solves of T3, by grid."""
    return {
        "80x30": load("run_80x30"),
        "160x60": load("run_160x60"),
        "320x120": load("flow_320x120"),
    }


def outer_of(record: dict) -> int:
    """Outer iterations of a flow record (its first solve, for a whole run)."""
    if "flows" in record:
        return record["flows"][0]["outer"]
    return record["outer"]


def flow_table(lines: list[str], calibrated: dict[str, dict[str, float]]) -> dict:
    """T3 and the CG table; returns each grid's stage times and repeat counts."""
    lines += ["### T3. An outer iteration of the flow solve", ""]
    rows, facts = [], {}
    for grid, r in flow_records().items():
        st = by_stage(r, "flow.")
        total = sum(st.values())
        n = outer_of(r)
        facts[grid] = {"total": total, "outer": n, "stages": st}
        what = "whole solve" if grid != "320x120" else "first 300"
        rows.append(
            [grid, what, f"{n:,}", s(total), f"{total / n / 1e6:.1f}"]
            + [f"{s(st.get(k, 0))} ({pct(st.get(k, 0) / total)})" for k in FLOW_STAGES]
        )
    lines += table(
        ["grid", "solve", "outer", "seconds", "ms per outer"]
        + [NAMES[k] for k in FLOW_STAGES],
        rows,
    )
    rows = []
    for grid, r in flow_records().items():
        p = facts[grid]["stages"]["flow.pressure"]
        cg = np.array(r["counters"]["cg"], dtype=float)
        n_corr = len(cg)
        cycles = calls(r, "mg_cycle") / n_corr
        mg = path_ns(r, lambda path: "mg_cycle" in path or "mg_build" in path)
        build = path_ns(r, lambda path: "mg_build" in path)
        loop = path_ns(r, lambda path: "cg_loop" in path)
        keps = np.array(r["counters"]["keps"], dtype=float)
        rows.append(
            [
                grid,
                f"{cg[:, 0].mean():.1f}",
                f"{cycles:.1f}",
                pct(mg / p),
                pct(build / p),
                pct(loop / p),
                f"{keps[:, 0].mean():.1f}",
                f"{int(keps[:, 1].sum())}",
            ]
        )
        facts[grid]["cg_iterations"] = cg[:, 0].mean()
        facts[grid]["cg_checks"] = (cg[:, 1] - cg[:, 0]).mean()
        facts[grid]["mg_cycles"] = cycles
        facts[grid]["keps_iterations"] = 2 * keps[:, 0].mean()
        facts[grid]["imbalance_summaries"] = (
            calls(r, "imbalance_summary") / facts[grid]["outer"]
        )
    lines += table(
        [
            "grid",
            "CG iterations per correction",
            "V-cycles per correction",
            "multigrid share of the pressure correction",
            "of it, the hierarchy's build",
            "CG loop share of the pressure correction",
            "CG iterations per k or eps solve",
            "k, eps fallbacks to Jacobi",
        ],
        rows,
    )
    lines += [
        "Stage times are exclusive and sum to the solve's time. The pressure correction holds the "
        "p' system's assembly, the hierarchy's build, the CG loop with its V-cycles and the velocity "
        "correction; the boundary updates hold the outlet extrapolation, the wall functions' "
        "viscosity and the k-epsilon wall conditions; the stopping rule holds the cell-centre "
        "velocities, the steps it reads and the rule's update. The 80x30 and 160x60 solves are "
        "inside the whole runs of T4, so their wrappers' overhead is those runs'; the 320x120 "
        "rows are its first 300 iterations. The flow timers cost more than the event estimate "
        "of T2 says. The calibrations measured "
        f"{100 * calibrated['measured']['calibrate_flow_80x30']:+.1f}% (80x30), "
        f"{100 * calibrated['measured']['calibrate_flow_160x60']:+.1f}% (160x60) and "
        f"{100 * calibrated['measured']['calibrate_flow_320x120']:+.1f}% (320x120) against "
        f"{pct(calibrated['estimate']['calibrate_flow_80x30'])}, "
        f"{pct(calibrated['estimate']['calibrate_flow_160x60'])} and "
        f"{pct(calibrated['estimate']['calibrate_flow_320x120'])} by the estimate; the paired "
        "measurement of test 84 (section 'The stop condition') gives more. Most wrapped calls "
        "sit in the pressure correction, so its share is inflated most.",
        "",
    ]
    return facts


def march_records() -> dict[str, dict]:
    """The march records of T4, by name."""
    return {
        "80x30 whole": load("run_80x30"),
        "160x60 whole": load("run_160x60"),
        "80x30 window": load("window_80x30"),
        "160x60 window": load("window_160x60"),
        "320x120 window": load("window_320x120"),
    }


def steps_of(record: dict) -> int:
    """March steps of RECORD."""
    return record["steps"]


def march_table(lines: list[str]) -> dict:
    """T4 with its lump and per-class tables; returns each run's stage times."""
    lines += ["### T4. The march", ""]
    rows, facts = [], {}
    for name, r in march_records().items():
        st = {k: v for k, v in by_stage(r).items() if k in MARCH_STAGES}
        lump = inline_lump(r)
        total = sum(st.values())
        n = steps_of(r)
        facts[name] = {
            "total": total,
            "steps": n,
            "stages": st,
            "record": r,
            "lump": lump,
        }
        rows.append(
            [name, f"{n:,}", s(total), f"{total / n / 1e6:.2f}"]
            + [f"{s(st.get(k, 0))} ({pct(st.get(k, 0) / total)})" for k in MARCH_STAGES]
        )
    lines += table(
        ["run", "steps", "seconds", "ms per step"] + [NAMES[k] for k in MARCH_STAGES],
        rows,
    )
    lines += [
        "Whole: the 120 s march of T5's run. Window: 9 s to 14 s, started from zero fields at the "
        "step at or before 9 s (the fields are zero before the 10 s onset), the 320x120 one on the "
        "stored state of prompt 81. Class stages are summed over the five classes.",
        "",
    ]
    inline_stages = [REGION_STAGE[r] for r in dict.fromkeys(REGION_STAGE)]
    inline_stages = list(dict.fromkeys(inline_stages))
    rows = []
    for name, f in facts.items():
        split = SPLIT_OF[f["record"]["job"]]
        fractions = split_fractions(load(split))
        rows.append(
            [name, s(f["lump"]), pct(f["lump"] / f["total"]), f"`{split}`"]
            + [pct(fractions.get(k, 0.0)) for k in inline_stages]
        )
    lines += [
        "The inline lump: solve_timestep's own lines, which the lean instrument times as one "
        "per call and which are split over the stages by the fractions of a run with the line "
        "regions on (the regions' own time, wrapped calls excluded). The split assumes the "
        "regions' slowdown under the line events (T2b) falls on each region in proportion. "
        "The booking region's marker (`inflow = (`) fires after the four inflow edge sums it "
        "is meant to open, because a line event on a statement that spans lines fires when its "
        "first instruction runs, which for an assignment is the store: those sums are charged "
        "to the divergence update, so the divergence update's fraction below is high and the "
        "booking's low by their time. Test 84 timed them at about 35 us per class step on "
        "both grids, which is under 2% of the march:",
        "",
    ]
    lines += table(
        ["run", "lump, s", "of the march", "split by"]
        + [NAMES[k] for k in inline_stages],
        rows,
    )
    rows = []
    for name, f in facts.items():
        r = f["record"]
        sweeps = np.array(r["counters"]["sweeps"], dtype=float)
        per_class = []
        for k in range(5):
            mine = sweeps[sweeps[:, 0] == k]
            ns = path_ns(
                r, lambda path, k=k: any(p.startswith(f"transport{k}:") for p in path)
            )
            per_class.append(f"{s(ns)} s, {mine[:, 1].mean():.1f}")
        rows.append([name, *per_class])
    lines += [
        "Per class: seconds in its class stages, and mean Jacobi sweeps per step:",
        "",
    ]
    lines += table(["run", "0.1 um", "0.3 um", "0.5 um", "1 um", "5 um"], rows)
    return facts


def frames_check(lines: list[str]) -> None:
    """The frames each window shares with the run it starts inside."""
    rows = []
    for grid in GRIDS:
        r = load(f"window_{grid}")
        checked = r["frames_checked"]
        same = sum(1 for v in checked.values() if v["fields"] and v["budgets"])
        source = (
            "prompt 81's stored run"
            if grid == "320x120"
            else "this pass's whole run (T4)"
        )
        rows.append([grid, source, f"{len(checked)}", f"{same}"])
    lines += [
        "Each window's frames against the run it starts inside, bitwise (fields and budget rows):",
        "",
    ]
    lines += table(["grid", "against", "frames shared", "bitwise equal"], rows)


def per_cell(lines: list[str], flow: dict, march: dict) -> dict:
    """T5: cost per cell per call by grid; returns the totals T11 reads."""
    lines += ["### T5. Cost per cell per call, and how it scales", ""]
    cells = {g: probe84.cells_of(g) for g in GRIDS}
    rows = []
    out = {}
    for k in FLOW_STAGES:
        vals = [
            flow[g]["stages"].get(k, 0) / flow[g]["outer"] / cells[g] for g in GRIDS
        ]
        rows.append(
            [NAMES[k], "outer iteration"] + [f"{v:.1f}" for v in vals] + [ratio(vals)]
        )
    totals = [flow[g]["total"] / flow[g]["outer"] / cells[g] for g in GRIDS]
    rows.append(
        ["**outer iteration**", ""] + [f"{v:.1f}" for v in totals] + [ratio(totals)]
    )
    out["outer"] = totals
    windows = [march[f"{g} window"] for g in GRIDS]
    for k in MARCH_STAGES:
        per = "class step" if k.startswith("class.") else "step"
        n = 5 if per == "class step" else 1
        vals = [
            w["stages"].get(k, 0) / (n * w["steps"]) / cells[g]
            for w, g in zip(windows, GRIDS, strict=True)
        ]
        rows.append([NAMES[k], per] + [f"{v:.2f}" for v in vals] + [ratio(vals)])
    totals = [
        w["total"] / w["steps"] / cells[g] for w, g in zip(windows, GRIDS, strict=True)
    ]
    rows.append(
        ["**march step (window)**", "step"]
        + [f"{v:.2f}" for v in totals]
        + [ratio(totals)]
    )
    out["window_step"] = totals
    whole = [
        march[f"{g} whole"]["total"] / march[f"{g} whole"]["steps"] / cells[g]
        for g in ("80x30", "160x60")
    ]
    rows.append(
        [
            "march step (whole 120 s)",
            "step",
            f"{whole[0]:.2f}",
            f"{whole[1]:.2f}",
            "",
            "",
        ]
    )
    out["whole_step"] = whole
    lines += table(
        [
            "stage",
            "per",
            "80x30, ns/cell",
            "160x60, ns/cell",
            "320x120, ns/cell",
            "80x30 over 320x120",
        ],
        rows,
    )
    lines += [
        "Flow: per outer iteration over the solves of T3. March: the 9 s to 14 s windows, the same "
        f"physical interval on each grid; cells {cells['80x30']:,}, {cells['160x60']:,} and {cells['320x120']:,}.",
        "",
    ]
    return out


def ratio(vals: list[float]) -> str:
    """80x30 over 320x120 of VALS, one decimal, or empty without a 320x120 value."""
    return f"{vals[0] / vals[2]:.1f}" if vals[2] else ""


def full_run(lines: list[str], flow: dict, march: dict) -> dict:
    """T6: the 320x120 scenario run's time, from its parts."""
    lines += ["### T6. The 320x120 scenario run, whole", ""]
    w = march["320x120 window"]
    r = w["record"]
    sweeps79 = np.load(SWEEPS79)
    total_sweeps = int(sweeps79["sweeps"].sum())
    class_steps = int(sweeps79["sweeps"].size)
    steps = sum(PLANS81)
    source_steps = PLANS81[1]
    after_onset = PLANS81[1] + PLANS81[2]
    frames = steps // 100 + 1
    checkpoints = steps // 1000 + 1
    # The implicit solve per class step, a + b sweeps, fitted on the window.
    data = np.array(r["counters"]["sweeps"], dtype=float)
    b, a = np.polyfit(data[:, 1], data[:, 2], 1)
    implicit_solve = path_ns(r, lambda path: "implicit_solve" in path)
    implicit_rest = w["stages"]["class.implicit"] - implicit_solve
    n_class = 5 * w["steps"]
    src_calls = source_calls(r)
    # The march builds a TransportSolver and its faces once per interval
    # (probe84.construct), inside the remainder; taken out before scaling.
    construction = float(np.median(load("construct_320x120")["wall_ns"]))
    estimate = {
        "class.advection": w["stages"]["class.advection"] / n_class * class_steps,
        "class.divergence": w["stages"]["class.divergence"] / n_class * class_steps,
        "class.source": w["stages"].get("class.source", 0)
        / max(src_calls, 1)
        * 5
        * source_steps,
        "class.implicit": implicit_rest / n_class * class_steps
        + a * class_steps
        + b * total_sweeps,
        "class.booking": w["stages"]["class.booking"] / n_class * class_steps,
        "class.remainder": w["stages"]["class.remainder"] / n_class * class_steps,
        "step.monitor": w["stages"]["step.monitor"] / w["steps"] * steps,
        "step.arrival": w["stages"]["step.arrival"] / steps_after(r) * after_onset,
        "step.frames": w["stages"]["step.frames"] / calls(r, "write_frame") * frames,
        "step.checkpoint": w["stages"]["step.checkpoint"]
        / calls(r, "write_checkpoint")
        * checkpoints,
        "march.remainder": construction * len(PLANS81)
        + (w["stages"]["march.remainder"] - construction * len(r["plans"]))
        / w["steps"]
        * steps,
    }
    march_s = sum(estimate.values())
    flow_rec = flow["320x120"]
    outer79 = 4389
    flow_here = flow_rec["total"] / flow_rec["outer"] * outer79
    flow_record = record_seconds(
        SUMMARY79, r"22ac4cb0815d: kept, 4389 outer, \S+, ([\d.]+) s"
    )
    march81 = record_seconds(SUMMARY81, r"wall time: ([\d.]+) s")
    whole79 = record_seconds(SUMMARY79, r"wall time: ([\d.]+) s")
    rows = [[NAMES[k], s(v), pct(v / march_s)] for k, v in estimate.items()]
    rows.append(["**march**", s(march_s), "100%"])
    lines += table(["stage", "seconds (estimate)", "share of the march"], rows)
    lines += [
        f"The march: 33,119 steps, {class_steps:,} class steps with {total_sweeps:,} Jacobi sweeps "
        "(prompt 79's record of every solve's sweeps, `equipment_dust.sweeps.66344.npz`), the "
        f"source on for {source_steps:,} steps and the arrival maps live for {after_onset:,}, "
        f"{frames} frame writes and {checkpoints} checkpoints. Each stage is the window's time per "
        "call times the run's calls; the implicit solve is the window's fit of its time per class "
        f"step on the sweeps, {a / 1e3:.0f} us + {b / 1e3:.1f} us per sweep, plus its conductances "
        "per call. The arrival maps' window time is all charged to the steps after the onset. "
        "The march remainder is the per-interval construction (`construct_320x120`, the "
        "median of five) three times plus the window's remainder less two constructions per "
        "step.",
        "",
    ]
    rows = [
        [
            "flow solve, 4,389 outer",
            s(flow_here),
            f"{flow_record:.1f}",
            "this pass: the first 300 iterations' mean times 4,389; record: prompt 79's solve",
        ],
        [
            "march, 120 s",
            s(march_s),
            f"{whole79 - flow_record:.1f} (prompt 79), {march81:.1f} (prompt 81)",
            "record: the run's wall less its flow solve (79), the march-only rerun (81)",
        ],
        [
            "march share of the run",
            pct(march_s / (march_s + flow_here)),
            f"{pct((whole79 - flow_record) / whole79)} (79), {pct(march81 / (march81 + flow_record))} (81 with 79's solve)",
            "",
        ],
    ]
    lines += table(["part", "this pass, s", "the records, s", "note"], rows)
    lines += [
        "The records' walls were taken with another run in the queue beside them (both queues "
        "record `concurrent: 1` at this job's start).",
        "",
    ]
    return {
        "estimate": estimate,
        "march": march_s,
        "flow": flow_here,
        "flow_record": flow_record,
        "march81": march81,
        "whole79": whole79,
        "fit": (a, b),
    }


def source_calls(record: dict) -> int:
    """A window's class steps with the source on: five per step of a plan with an event."""
    return 5 * sum(p["steps"] for p in record["plans"] if p["events"])


def steps_after(record: dict) -> int:
    """The window's steps after the onset."""
    return sum(p["steps"] for p in record["plans"] if p["t_start"] >= probe84.ONSET)


def record_seconds(path: Path, pattern: str) -> float:
    """The number PATTERN captures in the text of PATH."""
    found = re.search(pattern, path.read_text())
    assert found is not None, (path, pattern)
    return float(found.group(1))


def repeats(grid: str, flow: dict, march: dict) -> dict[str, float]:
    """The executions per call of each repeated kernel on GRID, from the records."""
    f = flow[grid]
    w = march[f"{grid} window"]["record"]
    sweeps = np.array(w["counters"]["sweeps"], dtype=float)
    n = counts84.COARSEST_CELLS
    cells = probe84.cells_of(grid)
    return {
        "momentum_sweeps_x2": 20,
        "cg_iterations": f["cg_iterations"],
        "cg_checks": f["cg_checks"],
        "mg_cycles": f["mg_cycles"],
        "keps_iterations": f["keps_iterations"],
        "keps_checks": 4,
        "imbalance_summaries": f["imbalance_summaries"],
        "turbulence_sides": turbulence_sides(grid),
        "jacobi_passes": sweeps[:, 1].mean() + 1,
        "source_steps": 1,
        "started_events": 1,
        "pinv_ops": 11 * n**3 / cells,
        "pinv_bytes": 3 * n * n * 8 / cells,
        "coarse_ops": 2 * n * n / cells,
        "coarse_bytes": n * n * 8 / cells,
    }


def value(x: float | str, rep: dict[str, float]) -> float:
    """X as a number: a count, or the repeat of that name."""
    return rep[x] if isinstance(x, str) else float(x)


def stage_counts(grid: str, rep: dict[str, float]) -> dict[str, tuple[float, float]]:
    """Operations and bytes per cell per call of each stage on GRID."""
    out: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for k in counts84.KERNELS:
        times = value(k.repeat, rep)
        out[k.stage][0] += value(k.ops, rep) * times
        out[k.stage][1] += value(k.bytes, rep) * times
    return {k: (v[0], v[1]) for k, v in out.items()}


def counts_tables(lines: list[str], flow: dict, march: dict) -> dict:
    """T7: the counts per kernel and per stage; returns the repeats and stage totals."""
    lines += [
        "### T7. Operations, bytes and arithmetic intensity, counted from the code",
        "",
    ]
    rep = {g: repeats(g, flow, march) for g in GRIDS}
    rows = []
    for k in counts84.KERNELS:
        r320 = rep["320x120"]
        ops, by = value(k.ops, r320), value(k.bytes, r320)
        rows.append(
            [
                NAMES.get(k.stage, k.stage),
                k.name,
                f"{ops:.1f}",
                f"{by:.1f}",
                f"{ops / by:.2f}",
                str(k.repeat),
            ]
        )
    lines += [
        "Per kernel, per cell per execution (the pseudo-inverse and the coarsest solve per fine cell of 320x120):",
        "",
    ]
    lines += table(
        [
            "stage",
            "kernel",
            "operations",
            "bytes",
            "ops per byte",
            "executions per call",
        ],
        rows,
    )
    rows = []
    stages = {g: stage_counts(g, rep[g]) for g in GRIDS}
    for stage in [*FLOW_STAGES, *CLASS_STAGES, "step.arrival"]:
        if stage not in stages["320x120"]:
            continue
        row = [NAMES[stage]]
        for g in GRIDS:
            ops, by = stages[g][stage]
            row.append(f"{ops:.0f} / {by:.0f} = {ops / by:.2f}")
        rows.append(row)
    lines += [
        "Per stage, per cell per call (operations / bytes = intensity), with each grid's measured repeats:",
        "",
    ]
    lines += table(["stage", "80x30", "160x60", "320x120"], rows)
    rows = [
        [key, " / ".join(f"{rep[g][key]:.2f}" for g in GRIDS)] for key in rep["320x120"]
    ]
    lines += [
        "The repeats (80x30 / 160x60 / 320x120): flow means per correction or per outer iteration from T3's solves, march means per class step from the windows:",
        "",
    ]
    lines += table(["repeat", "value"], rows)
    lines += ["Scalars each stage's call hands back to Python before it can go on:", ""]
    lines += table(
        ["stage", "scalars"], [[NAMES[k], v] for k, v in counts84.SCALARS.items()]
    )
    return {"rep": rep, "stages": stages}


def gpu_table(lines: list[str]) -> None:
    """T8: the GPU's peaks."""
    lines += ["### T8. The GPU's peaks", ""]
    lo, hi = counts84.BOOST_HZ
    bw = counts84.PEAK_BANDWIDTH
    rows = [
        [
            "peak DRAM bandwidth",
            f"{bw / 1e9:.1f} GB/s",
            "2 x 8001 MHz x 128 bit / 8 (CUDA driver attributes; `nvidia-smi -q -d CLOCK` max memory clock 8001 MHz)",
        ],
        [
            "peak FP32",
            f"{counts84.peak_fp32(lo) / 1e12:.2f} to {counts84.peak_fp32(hi) / 1e12:.2f} TFLOP/s",
            "36 SMs x 128 cores x 2 x boost clock, NVIDIA's 1230-2175 MHz",
        ],
        [
            "peak FP64",
            f"{counts84.peak_fp64(lo) / 1e9:.0f} to {counts84.peak_fp64(hi) / 1e9:.0f} GFLOP/s",
            "FP32 / 64 (Ada whitepaper, p. 7)",
        ],
        [
            "FP64 balance",
            f"{counts84.peak_fp64(lo) / bw:.2f} to {counts84.peak_fp64(hi) / bw:.2f} op/byte",
            "peak FP64 over peak bandwidth: below it a kernel is memory-bound",
        ],
        [
            "FP32 balance",
            f"{counts84.peak_fp32(lo) / bw:.0f} to {counts84.peak_fp32(hi) / bw:.0f} op/byte",
            "",
        ],
        [
            "L2",
            f"{counts84.L2_BYTES // 2**20} MiB",
            "CUDA driver attribute; a 320x120 field is 0.29 MiB",
        ],
    ]
    lines += table(["quantity", "value", "source"], rows)


def bound_table(lines: list[str], flow: dict, march: dict, counted: dict) -> dict:
    """T9: each 320x120 stage's time now against its time at peak bandwidth and FP64; returns the bounds."""
    lines += [
        "### T9. 320x120: each stage's time now against its time at peak bandwidth",
        "",
    ]
    cells = probe84.cells_of("320x120")
    bw = counts84.PEAK_BANDWIDTH
    lo, hi = counts84.BOOST_HZ
    stages = counted["stages"]["320x120"]
    rows, out = [], {}
    f = flow["320x120"]
    w = march["320x120 window"]
    for stage in [*FLOW_STAGES, *CLASS_STAGES, "step.arrival"]:
        if stage not in stages:
            continue
        ops, by = stages[stage]
        if stage.startswith("flow."):
            now = f["stages"].get(stage, 0) / f["outer"]
            per = "outer iteration"
        elif stage.startswith("class."):
            n = 5 * w["steps"]
            if stage == "class.source":
                n = source_calls(w["record"])
            now = w["stages"].get(stage, 0) / n
            per = "class step"
        else:
            now = w["stages"].get(stage, 0) / steps_after(w["record"])
            per = "step after the onset"
        t_bw = by * cells / bw * 1e9
        t_lo = ops * cells / counts84.peak_fp64(lo) * 1e9
        t_hi = ops * cells / counts84.peak_fp64(hi) * 1e9
        rows.append(
            [
                NAMES[stage],
                per,
                f"{now / 1e3:.1f}",
                f"{t_bw / 1e3:.2f}",
                f"{t_lo / 1e3:.2f}",
                f"{t_hi / 1e3:.2f}",
                f"{now / max(t_bw, t_lo):.0f}",
                f"{now / max(t_bw, t_hi):.0f}",
            ]
        )
        out[stage] = now / max(t_bw, t_hi)
    lines += table(
        [
            "stage",
            "per",
            "now, us",
            "at 256 GB/s, us",
            f"at peak FP64 ({lo / 1e6:.0f} MHz), us",
            f"at peak FP64 ({hi / 1e6:.0f} MHz), us",
            f"bound, {lo / 1e6:.0f} MHz",
            f"bound, {hi / 1e6:.0f} MHz",
        ],
        rows,
    )
    largest = max(
        counts84.KERNELS,
        key=lambda k: value(k.bytes, counted["rep"]["320x120"]),
    )
    footprint = value(largest.bytes, counted["rep"]["320x120"]) * cells / 2**20
    lines += [
        "The bandwidth time is T7's bytes per call over the peak bandwidth; the FP64 time its "
        "operations over the peak at each end of the boost range. A bound is the time now over "
        "the larger of the two, and the bound at the top of the range is the most favourable one. "
        "Each bound is labelled: it is an upper bound on the speedup only while every byte is DRAM "
        "traffic. Every field of the 320x120 grid is 0.29 MiB and the largest kernel of T7 "
        f"({NAMES.get(largest.stage, largest.stage)}, {largest.name}) moves {footprint:.1f} MiB per execution, "
        f"against an L2 of {counts84.L2_BYTES // 2**20} MiB, so each kernel's data stay in the L2 "
        "and the DRAM bandwidth is not the one it meets (T9b). Nor does a bound count a kernel "
        "launch's latency, or the host reads of T7's scalars between kernels.",
        "",
    ]
    return out


def side_of(intensity: float, bandwidth: float) -> str:
    """Which side of the FP64 balance INTENSITY falls on, over the boost range.

    A kernel is memory-bound when its operations per byte are below peak FP64
    over BANDWIDTH, and peak FP64 grows with the graphics clock, so the answer
    is one side at every clock or changes at one clock.
    """
    lo, hi = counts84.BOOST_HZ
    crossover = intensity * bandwidth / counts84.peak_fp64(1.0)
    if crossover <= lo:
        return "memory-bound at every clock"
    if crossover >= hi:
        return "compute-bound at every clock"
    return f"memory-bound above {crossover / 1e6:.0f} MHz"


def balance_table(lines: list[str], counted: dict) -> dict:
    """T9b: each stage's intensity against the FP64 balance of DRAM and of L2."""
    lo, hi = counts84.BOOST_HZ
    bw = counts84.PEAK_BANDWIDTH
    bandwidths = [
        ("DRAM", bw),
        *[(f"L2 at {b / 1e12:g} TB/s", b) for b in counts84.L2_BANDWIDTHS_ASSUMED],
    ]
    header = ["stage", "ops per byte"] + [
        f"{name}, balance {counts84.peak_fp64(lo) / b:.2f} to {counts84.peak_fp64(hi) / b:.2f}"
        for name, b in bandwidths
    ]
    stages = counted["stages"]["320x120"]
    rows = []
    sides: dict[str, list[str]] = {}
    items = [(NAMES[k], v[0] / v[1]) for k, v in stages.items()]
    variants = [(f"advection, {v.name}", v.ops / v.bytes) for v in counts84.VARIANTS]
    for name, intensity in [*items, *variants]:
        row = [side_of(intensity, b) for _n, b in bandwidths]
        if (name, intensity) not in variants:
            sides[name] = row
        rows.append([name, f"{intensity:.2f}", *row])
    lines += [
        "### T9b. 320x120: each stage's intensity against the FP64 balance of DRAM and of L2",
        "",
    ]
    lines += table(header, rows)
    lines += [
        "A stage is memory-bound in FP64 when its operations per byte are below the balance, "
        "peak FP64 over the bandwidth it meets; the balance rises with the graphics clock, so "
        "the table says whether a stage is on one side at every clock of the boost range "
        f"({lo / 1e6:.0f} to {hi / 1e6:.0f} MHz) or changes sides at one. The DRAM bandwidth is "
        f"{bw / 1e9:.1f} GB/s (T8). NVIDIA publishes no L2 bandwidth for this part, so the L2 "
        "columns are assumed, not measured: the only Ada measurement found is the RTX 4090's, "
        "about 5 TB/s with all 128 SMs active (Chips and Cheese, 'Microbenchmarking Nvidia's "
        "RTX 4090', 2022), which scales by SM count to about 1.4 TB/s for 36 SMs, across dies "
        "whose L2 slices differ. 1, 2 and 4 TB/s bracket that. The intensities are T7's "
        "per-stage ones for 320x120's repeats.",
        "",
    ]
    return sides


def cprofile_table(lines: list[str]) -> None:
    """T10: the profiler's top functions by own time."""
    r = load("cprofile_80x30")
    total = r["total_tt"]
    lines += ["### T10. cProfile of the 80x30 march, top functions by own time", ""]
    rows = []
    for row in r["top"][:25]:
        name = row["function"]
        where = row["file"]
        src = (
            "src/" + where.split("/src/")[-1]
            if "/src/" in where
            else ("numpy" if "numpy" in where else Path(where).name)
        )
        rows.append(
            [
                f"`{name}`",
                src,
                f"{row['calls']:,}",
                f"{row['tottime']:.2f}",
                pct(row["tottime"] / total),
                f"{row['cumtime']:.2f}",
            ]
        )
    lines += table(
        ["function", "where", "calls", "own s", "share of own time", "cumulative s"],
        rows,
    )
    lines += [
        f"The march only (the flow state copied from T4's run), {r['steps']:,} steps; own time "
        f"totals {total:.1f} s, wall {r['wall_ns'] / 1e9:.1f} s under the profiler. Built-in "
        "functions such as `numpy.ufunc` calls appear by their C names.",
        "",
    ]


def tally(sides: dict[str, list[str]], column: int) -> str:
    """Stages by side in COLUMN of T9b: memory / depends on the clock / compute."""
    cells = [row[column] for row in sides.values()]
    memory = sum(c.startswith("memory-bound at") for c in cells)
    compute = sum(c.startswith("compute-bound") for c in cells)
    return f"{memory} / {len(cells) - memory - compute} / {compute}"


def predictions(
    lines: list[str],
    flow: dict,
    march: dict,
    scaling: dict,
    full: dict,
    counted: dict,
    sides: dict[str, list[str]],
) -> dict:
    """T11; returns the measured values the verdicts rest on."""
    lines += ["### T11. The predictions, scored", ""]
    est = full["estimate"]
    march_share = full["march"] / (full["march"] + full["flow"])
    largest = max(
        CLASS_STAGES + STEP_STAGES + ["march.remainder"], key=lambda k: est.get(k, 0)
    )
    a = march_share > 0.70 and largest == "class.implicit"
    pressure_shares = {
        g: flow[g]["stages"]["flow.pressure"] / flow[g]["total"] for g in GRIDS
    }
    b = all(v > 0.5 for v in pressure_shares.values())
    c_window = scaling["window_step"][0] / scaling["window_step"][2]
    c = c_window >= 5.0
    stages320 = counted["stages"]["320x120"]
    intens = {k: v[0] / v[1] for k, v in stages320.items()}
    over = [f"{NAMES[k]} {v:.2f}" for k, v in intens.items() if v >= 1.0]
    kernels_over = [
        k.name
        for k in counts84.KERNELS
        if value(k.ops, counted["rep"]["320x120"])
        / value(k.bytes, counted["rep"]["320x120"])
        >= 1.0
    ]
    d = not over
    densest = max(intens, key=intens.get)
    bw = counts84.PEAK_BANDWIDTH
    lo, hi = counts84.BOOST_HZ
    fused = [(v.name, v.ops / v.bytes) for v in counts84.VARIANTS]
    # The clause "so every stage is memory-bound" holds only if every stage is
    # on the memory side at every clock, against DRAM and against each assumed
    # L2 bandwidth; the first clause (below 1 op/byte) decides on its own when
    # it fails.
    memory_everywhere = all(
        row[col].startswith("memory-bound at")
        for row in sides.values()
        for col in range(4)
    )
    if not d:
        d_verdict = "FAILS"
    elif memory_everywhere:
        d_verdict = "HOLDS"
    else:
        d_verdict = "HOLDS IN PART"
    l2_columns = ", ".join(
        f"L2 at {b / 1e12:g} TB/s {tally(sides, col)}"
        for col, b in enumerate(counts84.L2_BANDWIDTHS_ASSUMED, start=1)
    )
    e_parts = {}
    for name in ("320x120 estimate", "80x30 whole", "160x60 whole", "320x120 window"):
        if name == "320x120 estimate":
            st, tot = est, full["march"]
        else:
            st, tot = march[name]["stages"], march[name]["total"]
        e_parts[name] = (
            st.get("step.monitor", 0)
            + st.get("step.arrival", 0)
            + st.get("step.frames", 0)
        ) / tot
    e = e_parts["320x120 estimate"] < 0.10
    rows = [
        [
            "(a)",
            "320x120: march > 70% of the run; the implicit solve the march's largest stage",
            f"march {pct(march_share)} of the run (this pass's estimate; {pct((full['whole79'] - full['flow_record']) / full['whole79'])} by prompt 79's record); largest stage: {NAMES[largest]} ({pct(est[largest] / full['march'])}), implicit solve {pct(est['class.implicit'] / full['march'])}",
            "HOLDS" if a else "FAILS",
        ],
        [
            "(b)",
            "pressure correction > half an outer iteration",
            ", ".join(f"{g} {pct(v)}" for g, v in pressure_shares.items()),
            "HOLDS" if b else "FAILS",
        ],
        [
            "(c)",
            "cost per cell per step on 80x30 at least 5x that on 320x120",
            f"march step, 9-14 s windows: {c_window:.2f}x; outer iteration: {scaling['outer'][0] / scaling['outer'][2]:.2f}x",
            "HOLDS" if c else "FAILS",
        ],
        [
            "(d)",
            "every stage below 1 op/byte, so every stage memory-bound in FP64",
            ("every stage below 1" if d else "at or above 1: " + ", ".join(over))
            + f"; against DRAM ({counts84.peak_fp64(lo) / bw:.2f} to "
            f"{counts84.peak_fp64(hi) / bw:.2f} op/byte over the boost range of "
            f"{lo / 1e6:.0f}-{hi / 1e6:.0f} MHz, T8) the densest, {NAMES[densest]}, "
            f"{intens[densest]:.2f}, is {side_of(intens[densest], bw)} in FP64"
            "; stages memory-bound at every clock / changing with the clock / compute-bound "
            f"at every clock (T9b): DRAM {tally(sides, 0)}, {l2_columns} (L2 bandwidths assumed, "
            "none published); every kernel's data fit the L2 on 320x120 (T9)"
            + (
                f"; kernels at or above 1: {', '.join(kernels_over)}"
                if kernels_over
                else ""
            )
            + "; the advection with other kernel boundaries: "
            + ", ".join(f"{name} {x:.2f}" for name, x in fused),
            d_verdict,
        ],
        [
            "(e)",
            "monitor, arrival maps and frame writes < 10% of the march",
            "; ".join(f"{k} {pct(v)}" for k, v in e_parts.items()),
            "HOLDS" if e else "FAILS",
        ],
    ]
    lines += table(["", "prediction", "measured", "verdict"], rows)
    lines += [
        "(a) and (e) are scored on T6's whole-run estimate, (c) on the windows, (d) on T7's 320x120 stage counts and T9b.",
        "",
    ]
    return {"pressure": pressure_shares, "e": e_parts, "march_share": march_share}


def ports(lines: list[str], flow: dict, march: dict, full: dict) -> None:
    """T12: what the plan's two kernels would carry."""
    lines += ["### T12. What the plan's two kernels would carry", ""]
    rows = []

    def loop_flow(record: dict) -> int:
        return path_ns(
            record,
            lambda p: "cg_loop" in p and stage_of(p) == "flow.pressure",
        )

    def loop_march(record: dict) -> float:
        st = by_stage(record)
        solve = path_ns(record, lambda p: "implicit_solve" in p)
        return st["class.advection"] + st["class.divergence"] + solve

    for grid in ("80x30", "160x60"):
        r = load(f"run_{grid}")
        total = r["total_ns"]
        moved = loop_flow(r) + loop_march(r)
        share = moved / total
        rows.append(
            [
                f"{grid} whole run (T4)",
                s(total),
                pct(loop_flow(r) / total),
                pct(loop_march(r) / total),
                pct(share),
                f"{1 / (1 - share):.2f}",
            ]
        )
    f = flow["320x120"]
    fr = load("flow_320x120")
    flow_loop = loop_flow(fr) / f["total"] * full["flow"]
    w = march["320x120 window"]["record"]
    est = full["estimate"]
    implicit_solve_share = (
        path_ns(w, lambda p: "implicit_solve" in p)
        / march["320x120 window"]["stages"]["class.implicit"]
    )
    march_loop = (
        est["class.advection"]
        + est["class.divergence"]
        + est["class.implicit"] * implicit_solve_share
    )
    total = full["flow"] + full["march"]
    share = (flow_loop + march_loop) / total
    rows.append(
        [
            "320x120 whole run (T6 estimate)",
            s(total),
            pct(flow_loop / total),
            pct(march_loop / total),
            pct(share),
            f"{1 / (1 - share):.2f}",
        ]
    )
    lines += table(
        [
            "run",
            "seconds",
            "pressure CG loop with its V-cycles",
            "advection, divergence and implicit solve",
            "together",
            "run speedup if they took no time",
        ],
        rows,
    )
    lines += [
        "The plan's kernels (docs/PROJECT_PLAN.md, Phase 6): `pressure_solve.cu`, the CG loop with "
        "the multigrid cycles, and `advection_diffusion.cu`, the transport stencil. Everything else "
        "stays in Python under such a port: the momentum predictor, the hierarchy's build, the "
        "turbulence update, the boundary updates, the stopping rule, the input checks, the source, "
        "the conductances, the booking, the monitor, the arrival maps and the writes. The 320x120 "
        "row's implicit solve is T6's stage less its conductances, by the window's split, at the "
        "window's share of solve time in that stage; the run averages more sweeps per class step "
        "than the window, so that row's share of the run is understated by about a point, and its "
        "speedup by about a tenth.",
        "",
    ]


def build() -> dict[str, list[str]]:
    """Every table, as named blocks in the report's order."""
    blocks: dict[str, list[str]] = {}

    def block[T](name: str, make: Callable[..., T], *args: object) -> T:
        lines: list[str] = []
        result = make(lines, *args)
        blocks[name] = lines
        return result

    block("machine", machine)
    calibrated = block("calibration", calibration)
    flow = block("flow", flow_table, calibrated)
    march = block("march", march_table)
    block("frames", frames_check)
    scaling = block("scaling", per_cell, flow, march)
    full = block("full", full_run, flow, march)
    counted = block("counts", counts_tables, flow, march)
    block("gpu", gpu_table)
    block("bounds", bound_table, flow, march, counted)
    sides = block("balance", balance_table, counted)
    block("cprofile", cprofile_table)
    block("predictions", predictions, flow, march, scaling, full, counted, sides)
    block("ports", ports, flow, march, full)
    return blocks


def text_of(blocks: dict[str, list[str]]) -> str:
    """tables84.md."""
    parts = [
        "# Prompt 84's tables (generated by tables84.py from results/builder84; do not edit)",
        "",
    ]
    for lines in blocks.values():
        parts += lines
    return "\n".join(parts)


def splice(text: str, blocks: dict[str, list[str]]) -> str:
    """The report with each marked block replaced."""
    for name, lines in blocks.items():
        begin, end = f"<!-- tables84 {name} begin -->", f"<!-- tables84 {name} end -->"
        if begin not in text:
            continue
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        body = "\n".join(lines).rstrip("\n")
        text = head + begin + "\n" + body + "\n" + end + tail
    return text


def main() -> int:
    """Write the tables and splice the report, or with --check compare them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    blocks = build()
    tables = text_of(blocks)
    report = REPORT.read_text(encoding="utf-8")
    spliced = splice(report, blocks)
    if args.check:
        ok = True
        if OUT.read_text(encoding="utf-8") != tables:
            print(f"{OUT.name} differs from the records")
            ok = False
        if report != spliced:
            print(f"{REPORT.name}'s copies differ from the records")
            ok = False
        print("match" if ok else "differ")
        return 0 if ok else 1
    OUT.write_text(tables, encoding="utf-8")
    REPORT.write_text(spliced, encoding="utf-8")
    print(f"wrote {OUT} and spliced {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
