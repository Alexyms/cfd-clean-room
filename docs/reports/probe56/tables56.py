"""The report's tables from the prompt 56 records.

Usage:
    python tables56.py timing       the two timing probes and the projection at the cap
    python tables56.py layouts      the three layouts' flows as the boundary layer resolves them
    python tables56.py matrix       every flow run, classified by step 5's rule
    python tables56.py converged    the converged rooms' readings, core and y+
    python tables56.py bounded      the bounded rows, from bounded56.py's output (tables46's)
    python tables56.py transport    every march: the stop, the budget, the sensors, the surfaces,
                                    the hotspots
    python tables56.py score        the layout score per layout, grid, Sc_t and class
    python tables56.py ranking      the order of the layouts by the score
    python tables56.py margins      each neighbouring difference against the grid's and Sc_t's change
    python tables56.py hotspots     the largest location and the shared count, between grids and
                                    between Schmidt numbers
    python tables56.py verdict      predictions (a) to (d) scored from the records
    python tables56.py all          everything above

Reads results/builder56/*.json, prints markdown tables, and keeps every
number it printed in results/builder56/tables_MODE.json. The score is the
report's section 2.6: the deposition on the litho tool's top and the etch
chamber's top, summed over the three sources, per class, lower better.
The classification is tables46's (imported).
"""

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe47"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe44"))

import tables46  # noqa: E402
import tables47  # noqa: E402
from tables46 import classify, fmt, print_table  # noqa: E402
from tables47 import from_when_cell, readings_cell  # noqa: E402

from src.boundary_registry import FIXED_FLOW_OUTLET  # noqa: E402
from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402

OUT = ROOT / "results" / "builder56"
tables46.OUT = OUT
tables47.OUT = OUT
LAYOUTS = ("L0", "L1", "L2")
LAYOUT_FILES = {
    "L0": "clean_room_default.yaml",
    "L1": "clean_room_l1_return3_closed.yaml",
    "L2": "clean_room_l2_left_supply.yaml",
}
CHANGES = {
    "L0": "the room as it is",
    "L1": "return 3 closed, its flow to returns 2 and 4",
    "L2": "the supply over x 0.5 to 4.0 m only, at 0.9 m/s",
}
GRIDS = ("160x60", "320x120")
GRID_CELLS = {"160x60": (160, 60), "320x120": (320, 120)}
PRODUCT_GRID = "320x120"
SOURCES = ("S1", "S2", "S3")
CLASSES = ("2", "4")
SCHMIDTS = (0.2, 0.7, 1.3)
BASE_SC = 0.7
PROCESS_SURFACES = ("litho_tool top", "etch_chamber top")
HOTSPOTS_SHARED = 3
SURFACE_SHARE_FLOOR = 1e-6
POSITION_TOL = 1e-9
NAME = re.compile(r"^(?P<layout>L\d)_(?P<grid>\d+x\d+)(?P<rest>.*)$")
TRANSPORT = re.compile(
    r"^transport_(?P<flow>(?P<layout>L\d)_(?P<grid>\d+x\d+))_(?P<source>S\d)_sc(?P<sc>[\d.]+)(?P<tag>_.*)?$"
)


def load(name: str) -> dict:
    """A record by name."""
    return json.loads((OUT / f"{name}.json").read_text())


def keep(mode: str, data: object) -> None:
    """The printed numbers, as JSON beside the records."""
    (OUT / f"tables_{mode}.json").write_text(json.dumps(data, indent=1, default=float))


def sc_key(sc: float) -> str:
    return f"{sc:g}"


def um(c: dict) -> str:
    return f"{c['diameter'] * 1e6:g}"


def records() -> dict[str, dict]:
    """Every flow record, keyed by name; no timing or transport rows."""
    out = {}
    for path in sorted(OUT.glob("*.json")):
        m = NAME.match(path.stem)
        if not m or path.stem.endswith("_time"):
            continue
        rec = load(path.stem)
        if "stop" not in rec:
            continue
        rec["_layout"], rec["_grid"], rec["_rest"] = m["layout"], m["grid"], m["rest"]
        out[path.stem] = rec
    return out


def order_key(rec: dict) -> tuple:
    """Grid, then layout."""
    return (
        GRIDS.index(rec["_grid"]) if rec["_grid"] in GRIDS else 99,
        rec["_layout"],
        rec["_rest"],
    )


def transport_records() -> dict[str, dict]:
    """Every untagged transport record, keyed by its label, with the name parsed."""
    out = {}
    for path in sorted(OUT.glob("transport_*.json")):
        m = TRANSPORT.match(path.stem)
        if not m or m["tag"]:
            continue
        rec = load(path.stem)
        rec["_flow"], rec["_layout"], rec["_grid"] = m["flow"], m["layout"], m["grid"]
        rec["_source"], rec["_sc"] = m["source"], float(m["sc"])
        out[path.stem] = rec
    return out


def transport_key(layout: str, grid: str, source: str, sc: float) -> str:
    return f"transport_{layout}_{grid}_{source}_sc{sc_key(sc)}"


# ---------------------------------------------------------------------------
# The set
# ---------------------------------------------------------------------------


def timing(_args: argparse.Namespace) -> None:
    """The timing probes: seconds per outer iteration and the projection at the cap."""
    rows = []
    for grid in GRIDS:
        path = OUT / f"L0_{grid}_time.json"
        if not path.exists():
            continue
        r = load(path.stem)
        rows.append(
            {
                "grid": grid,
                "outer": r["outer"],
                "seconds_per_outer": r["seconds_per_outer"],
                "inner_mean": r["inner_mean"],
                "k_sweeps_mean": r["k_sweeps_mean"],
                "eps_sweeps_mean": r["eps_sweeps_mean"],
                "share_pressure": r["stage_seconds"]["pressure"] / r["seconds"],
                "share_turbulence": r["stage_seconds"]["turbulence"] / r["seconds"],
                "minutes_at_cap": r["seconds_per_outer"] * r["cap"] / 60,
                "concurrent_processes": r.get("concurrent_processes"),
                "started": r["started"],
            }
        )
    headers = [
        "Grid",
        "Outer",
        "Seconds per outer",
        "CG per correction (mean)",
        "k, eps iterations (mean)",
        "Share of wall: pressure, turbulence",
        "Minutes at the 10,000 cap",
        "Concurrent processes",
    ]
    table = [
        [
            r["grid"],
            fmt(r["outer"]),
            f"{r['seconds_per_outer']:.3f}",
            f"{r['inner_mean']:.0f}",
            f"{r['k_sweeps_mean']:.1f}, {r['eps_sweeps_mean']:.1f}",
            f"{r['share_pressure']:.2f}, {r['share_turbulence']:.2f}",
            f"{r['minutes_at_cap']:.0f}",
            fmt(r["concurrent_processes"]),
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("timing", rows)


def positions(raw: dict) -> list[float]:
    """Every x and y position the configuration states."""
    out = []
    for seg in raw["boundaries"].values():
        out += [
            float(seg[k]) for k in ("x_start", "x_end", "y_start", "y_end") if k in seg
        ]
    for o in raw["obstacles"]:
        out += [float(o[k]) for k in ("x_start", "x_end", "y_start", "y_end")]
    for s in raw["sensors"]:
        out += [float(s["x"]), float(s["y"])]
    return out


def outlet_cell(g: dict, name: str) -> str:
    """One outlet's range, resolved velocity and flow, or "closed"."""
    o = g["outlets"].get(name)
    if o is None:
        return "closed"
    stated = " (stated)" if o["stated"] is not None else ""
    return (
        f"{o['span'][0]:g} to {o['span'][1]:g}, {o['velocity']:.4g}{stated}, "
        f"{o['flow']:.4g}"
    )


def layouts(_args: argparse.Namespace) -> None:
    """The layouts' flows as the boundary layer resolves them on the product grid."""
    rows = []
    for layout in LAYOUTS:
        raw = yaml.safe_load(
            (ROOT / "configs" / LAYOUT_FILES[layout]).read_text(encoding="utf-8")
        )
        row = {
            "layout": layout,
            "file": LAYOUT_FILES[layout],
            "change": CHANGES[layout],
        }
        for grid in GRIDS:
            nx, ny = GRID_CELLS[grid]
            r = json.loads(json.dumps(raw))
            r["domain"]["nx"], r["domain"]["ny"] = nx, ny
            cfg = SimConfig.from_dict(r)
            mesh = Mesh(cfg)
            bc = StaggeredBoundary(mesh, cfg)
            velocities = bc.fixed_flow_velocities()
            outlets = {}
            for name, spec in cfg.boundaries.items():
                if spec.type != FIXED_FLOW_OUTLET:
                    continue
                if spec.location in ("top", "bottom"):
                    length = spec.x_end - spec.x_start
                    span = (spec.x_start, spec.x_end)
                else:
                    length = spec.y_end - spec.y_start
                    span = (spec.y_start, spec.y_end)
                outlets[name] = {
                    "span": span,
                    "stated": spec.velocity,
                    "velocity": velocities[name],
                    "flow": velocities[name] * length,
                }
            supply = cfg.boundaries["hepa_supply"]
            dx, dy = raw["domain"]["width"] / nx, raw["domain"]["height"] / ny
            rounded = [
                p
                for p in positions(raw)
                if min(abs(p / dx - round(p / dx)), abs(p / dy - round(p / dy)))
                > POSITION_TOL
            ]
            row[grid] = {
                "supply": {
                    "span": (supply.x_start, supply.x_end),
                    "velocity": supply.velocity,
                    "flow": bc.get_total_inlet_flux(),
                },
                "outlets": outlets,
                "total_extract": sum(o["flow"] for o in outlets.values()),
                "positions": len(positions(raw)),
                "rounded_positions": rounded,
            }
        rows.append(row)
    headers = [
        "Layout",
        "Change",
        "Supply: x range (m), velocity (m/s), flow (m^2/s)",
        "Return 1: range, velocity, flow",
        "Return 2",
        "Return 3",
        "Return 4",
        "Hood: y range, velocity, flow",
        "Total extract (m^2/s)",
        "Positions rounded on 160x60, 320x120",
    ]
    table = []
    for r in rows:
        g = r[PRODUCT_GRID]
        s = g["supply"]
        table.append(
            [
                r["layout"],
                r["change"],
                f"{s['span'][0]:g} to {s['span'][1]:g}, {s['velocity']:g}, {s['flow']:.4g}",
                outlet_cell(g, "floor_return_1"),
                outlet_cell(g, "floor_return_2"),
                outlet_cell(g, "floor_return_3"),
                outlet_cell(g, "floor_return_4"),
                outlet_cell(g, "hood_exhaust"),
                f"{g['total_extract']:.4g}",
                ", ".join(str(len(r[grid]["rounded_positions"])) for grid in GRIDS),
            ]
        )
    print_table(headers, table)
    keep("layouts", rows)


def matrix(_args: argparse.Namespace) -> None:
    """Every flow run, one line each."""
    rows = []
    for rec in sorted(records().values(), key=order_key):
        cls, sub = classify(rec)
        rows.append(
            {
                "name": rec["name"],
                "layout": rec["_layout"],
                "grid": rec["_grid"],
                "class": cls + (f" ({sub})" if sub else ""),
                "stop": rec["stop"],
                "outer": rec["outer"],
                "cap": rec.get("cap"),
                "residual_min": rec["residual_min"],
                "residual_min_at": rec["residual_min_at"],
                "residual_end": rec["residual_end"],
                "readings_at_stop": rec.get("readings_at_stop"),
                "from_when": rec.get("from_when"),
                "max_speed_end": rec["max_speed_end"],
                "at_end": rec["at_end"],
                "max_speed_peak": rec["max_speed_peak"],
                "inner_mean": rec["inner_mean"],
                "inner_max": rec["inner_max"],
                "cap_hits": rec["cap_hits"],
                "k_sweeps_mean": rec["k_sweeps_mean"],
                "eps_sweeps_mean": rec["eps_sweeps_mean"],
                "scalar_cap_hits": rec["scalar_cap_hits"],
                "seconds": rec["seconds"],
                "seconds_per_outer": rec["seconds_per_outer"],
                "concurrent_processes": rec.get("concurrent_processes"),
                "face_hash": rec["face_hash"],
                "positivity": rec.get("positivity"),
                "commit": rec["commit"],
                "started": rec["started"],
            }
        )
    headers = [
        "Run",
        "Class",
        "Stop",
        "Outer (cap)",
        "Residual: least (at), end",
        "Readings at stop (a), (b), (c), (d), (e)",
        "Holds from (a), (b), (c), (d), (e)",
        "Largest speed at end (m/s), cell",
        "Peak speed",
        "CG per correction: mean, largest",
        "Cap hits",
        "k, eps iterations (mean)",
        "Wall (s), per outer; processes",
        "Face hash",
    ]
    table = [
        [
            r["name"],
            r["class"],
            r["stop"],
            f"{fmt(r['outer'])} ({fmt(r['cap'])})",
            f"{r['residual_min']:.2e} ({r['residual_min_at']:,}), {r['residual_end']:.2e}",
            readings_cell(r) if r["readings_at_stop"] else "-",
            from_when_cell(r) if r["from_when"] else "-",
            f"{r['max_speed_end']:.3g}, {tuple(r['at_end'])}",
            f"{r['max_speed_peak']:.3g}",
            f"{r['inner_mean']:.0f}, {r['inner_max']}",
            fmt(r["cap_hits"]),
            f"{r['k_sweeps_mean']:.1f}, {r['eps_sweeps_mean']:.1f}",
            f"{r['seconds']:.0f}, {r['seconds_per_outer']:.3f}; {fmt(r['concurrent_processes'])}",
            (r["face_hash"] or "-")[:16],
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("matrix", rows)


def converged(_args: argparse.Namespace) -> None:
    """The converged rooms' readings, core eddy viscosity and y+."""
    rows = []
    for rec in sorted(records().values(), key=order_key):
        if rec["stop"] != "error_estimate_and_continuity" or "core" not in rec:
            continue
        core, y = rec["core"], rec["y_star"]
        rows.append(
            {
                "name": rec["name"],
                "outer": rec["outer"],
                "stop": rec["stop"],
                "readings_at_stop": rec["readings_at_stop"],
                "from_when": rec["from_when"],
                "seconds": rec["seconds"],
                "concurrent_processes": rec.get("concurrent_processes"),
                "nu_scale": rec["nu_scale"],
                "nu_air": rec["nu_air"],
                "inlet_nu_t_over_nu": (rec["nu_scale"] - rec["nu_air"]) / rec["nu_air"],
                "core": core,
                "y_star": y,
                "figure": f"ecr002_step8_{rec['name']}.png",
            }
        )
    headers = [
        "Run",
        "Outer, stop",
        "Readings at stop (a), (b), (c), (d), (e)",
        "Holds from (a), (b), (c), (d), (e)",
        "Wall (s); processes",
        "Core nu_t / nu: median, 95th",
        "y+ nodes",
        "y+: median, least, largest",
        "Share below 11.53: all; domain, tops, sides",
        "Figure",
    ]
    table = []
    for r in rows:
        kinds = r["y_star"]["by_kind"]
        share = ", ".join(
            fmt(kinds.get(k, {}).get("share_below_floor"), 2)
            for k in ("domain", "obstacle top", "obstacle side")
        )
        table.append(
            [
                r["name"],
                f"{r['outer']:,}, {r['stop']}",
                readings_cell(r),
                from_when_cell(r),
                f"{r['seconds']:.0f}; {fmt(r['concurrent_processes'])}",
                f"{r['core']['nu_t_over_nu_median']:.3g}, {r['core']['nu_t_over_nu_p95']:.3g}",
                fmt(r["y_star"]["nodes"]),
                f"{r['y_star']['median']:.3g}, {r['y_star']['least']:.3g}, {r['y_star']['largest']:.3g}",
                f"{r['y_star']['share_below_floor']:.2g}; {share}",
                r["figure"],
            ]
        )
    print_table(headers, table)
    keep("converged", rows)


def bounded(args: argparse.Namespace) -> None:
    """The bounded rows' characterisation: tables46's table on this step's records."""
    tables46.bounded(args)


# ---------------------------------------------------------------------------
# The marches
# ---------------------------------------------------------------------------


def transport(_args: argparse.Namespace) -> None:
    """Every march: the stop, the budget, the sensors, the surfaces and the hotspots."""
    recs = transport_records()
    kept: dict = {"runs": {}, "surfaces": {}, "hotspots": {}}
    headers = [
        "Run, class (um)",
        "Stop, t (s)",
        "Last window: total, sensor change; removal / Q",
        "Exhausted / Q, deposited / Q (faces; budget)",
        "Budget residual",
        "Source-cell C, floor",
        "Sensors above the floor, in order (per m^3)",
        "Wall (s); processes",
    ]
    table = []
    for label, rec in recs.items():
        for k, c in rec["per_class"].items():
            lw = c["last_window"]
            above = c["sensors_above_floor"]
            table.append(
                [
                    f"{label[len('transport_') :]}, {um(c)}",
                    f"{c['stop']}, {c['t_end']:.0f}",
                    f"{fmt(lw['total_change'], 2)}, {fmt(lw['sensor_change'], 2)}; "
                    f"{fmt(lw['removal_over_source'], 5)}",
                    f"{c['outflow_rate_over_source']:.4g}, "
                    f"{c['deposition_rate_over_source']:.4g} "
                    f"({c['budget_deposition_rate_over_source']:.4g})",
                    fmt(c["budget"]["relative_residual"], 2),
                    f"{c['source_cell_concentration']:.3g}, {c['floor']:.3g}",
                    " > ".join(
                        f"{s} ({above[s]:.3g})" for s in c["sensor_order_above_floor"]
                    )
                    or "none",
                    f"{rec['seconds']:.0f}; {fmt(rec.get('concurrent_processes'))}",
                ]
            )
            kept["runs"][f"{label}/{k}"] = {
                key: c[key]
                for key in (
                    "stop",
                    "t_end",
                    "windows",
                    "last_window",
                    "source_cell_concentration",
                    "floor",
                    "sensors",
                    "sensors_above_floor",
                    "sensor_order_above_floor",
                    "surfaces_above_floor",
                    "deposition_rate_over_source",
                    "budget_deposition_rate_over_source",
                    "outflow_rate_over_source",
                    "budget",
                    "hotspots",
                    "hotspot_rates",
                    "by_surface",
                    "by_surface_share",
                )
            }
    print_table(headers, table)
    # Deposition per surface: one table per source and class, a column per
    # layout, grid and Schmidt number.
    columns = [
        (layout, grid, sc)
        for layout in LAYOUTS
        for grid, sc in ((GRIDS[0], BASE_SC), *((PRODUCT_GRID, s) for s in SCHMIDTS))
    ]
    for source in SOURCES:
        for k in CLASSES:
            present = [
                (layout, grid, sc)
                for layout, grid, sc in columns
                if transport_key(layout, grid, source, sc) in recs
            ]
            if not present:
                continue
            surfaces: list[str] = []
            for layout, grid, sc in present:
                shares = recs[transport_key(layout, grid, source, sc)]["per_class"][k][
                    "by_surface_share"
                ]
                for s, share in shares.items():
                    # Below a millionth of the total a surface's value is the
                    # scheme's tail, not transport; such rows are left out.
                    if s not in surfaces and (share or 0.0) >= SURFACE_SHARE_FLOOR:
                        surfaces.append(s)
            diameter = recs[transport_key(*present[0][:2], source, present[0][2])][
                "per_class"
            ][k]["diameter"]
            print(
                f"Deposition per surface, source {source}, {diameter * 1e6:g} um, share of "
                f"the deposition total (surfaces with at least {SURFACE_SHARE_FLOOR:g} of "
                f"it in some column):"
            )
            headers = ["Surface"] + [
                f"{layout} {grid} Sc {sc_key(sc)}" for layout, grid, sc in present
            ]
            table = [
                [s]
                + [
                    fmt(
                        recs[transport_key(layout, grid, source, sc)]["per_class"][k][
                            "by_surface_share"
                        ].get(s),
                        3,
                    )
                    for layout, grid, sc in present
                ]
                for s in surfaces
            ]
            print_table(headers, table)
            kept["surfaces"][f"{source}/{k}"] = {
                transport_key(layout, grid, source, sc): recs[
                    transport_key(layout, grid, source, sc)
                ]["per_class"][k]["by_surface"]
                for layout, grid, sc in present
            }
    print(
        "The five segments of largest deposition (surface | bin start), rate per s per m depth:"
    )
    headers = ["Run, class (um)", "1", "2", "3", "4", "5"]
    table = []
    for label, rec in recs.items():
        for k, c in rec["per_class"].items():
            table.append(
                [f"{label[len('transport_') :]}, {um(c)}"]
                + [
                    f"{s} ({v:.3g})"
                    for s, v in zip(c["hotspots"], c["hotspot_rates"], strict=True)
                ]
            )
            kept["hotspots"][f"{label}/{k}"] = list(
                zip(c["hotspots"], c["hotspot_rates"], strict=True)
            )
    print_table(headers, table)
    keep("transport", kept)


# ---------------------------------------------------------------------------
# The comparative check
# ---------------------------------------------------------------------------


def scores() -> dict:
    """The score of every layout, grid, Schmidt number and class the records allow.

    Keyed "LAYOUT/GRID/SC/CLASS"; each entry holds the per-source deposition
    on the two process surfaces, the per-source sum, and the score, the sum
    over the three sources. An entry exists only when all three sources'
    marches exist.
    """
    recs = transport_records()
    out: dict[str, dict] = {}
    for layout, grid, sc, k in itertools.product(LAYOUTS, GRIDS, SCHMIDTS, CLASSES):
        labels = [transport_key(layout, grid, source, sc) for source in SOURCES]
        if not all(label in recs for label in labels):
            continue
        per_source = {}
        for source, label in zip(SOURCES, labels, strict=True):
            by_surface = recs[label]["per_class"][k]["by_surface"]
            parts = {s: by_surface.get(s, 0.0) for s in PROCESS_SURFACES}
            per_source[source] = {**parts, "sum": sum(parts.values())}
        out[f"{layout}/{grid}/{sc_key(sc)}/{k}"] = {
            "layout": layout,
            "grid": grid,
            "sc": sc,
            "class": k,
            "diameter": recs[labels[0]]["per_class"][k]["diameter"],
            "per_source": per_source,
            "score": sum(v["sum"] for v in per_source.values()),
        }
    return out


def score_key(layout: str, grid: str, sc: float, k: str) -> str:
    return f"{layout}/{grid}/{sc_key(sc)}/{k}"


def cases() -> list[tuple[str, float]]:
    """The (grid, Sc_t) combinations the set ranks: both grids at 0.7, the product grid at each."""
    return [(GRIDS[0], BASE_SC), *((PRODUCT_GRID, sc) for sc in SCHMIDTS)]


def score(_args: argparse.Namespace) -> None:
    """The score per layout, grid, Sc_t and class, with its per-source terms."""
    s = scores()
    for k in CLASSES:
        rows = [v for v in s.values() if v["class"] == k]
        if not rows:
            continue
        print(
            f"The layout score, {rows[0]['diameter'] * 1e6:g} um: deposition on the litho "
            "tool's top and the etch chamber's top, particles per s per m depth, per source "
            "and summed (lower is better):"
        )
        headers = (
            ["Layout, grid, Sc_t"]
            + [f"{source}: litho top, etch top, sum" for source in SOURCES]
            + ["Score"]
        )
        table = []
        for layout in LAYOUTS:
            for grid, sc in cases():
                v = s.get(score_key(layout, grid, sc, k))
                if v is None:
                    continue
                table.append(
                    [f"{layout}, {grid}, {sc_key(sc)}"]
                    + [
                        f"{v['per_source'][src][PROCESS_SURFACES[0]]:.4g}, "
                        f"{v['per_source'][src][PROCESS_SURFACES[1]]:.4g}, "
                        f"{v['per_source'][src]['sum']:.4g}"
                        for src in SOURCES
                    ]
                    + [f"{v['score']:.4g}"]
                )
        print_table(headers, table)
    keep("score", s)


def rankings(s: dict) -> list[dict]:
    """The order of the layouts by the score for every (grid, Sc_t, class) with all three."""
    out = []
    for grid, sc in cases():
        for k in CLASSES:
            entries = {
                layout: s[score_key(layout, grid, sc, k)]["score"]
                for layout in LAYOUTS
                if score_key(layout, grid, sc, k) in s
            }
            if len(entries) < len(LAYOUTS):
                continue
            order = sorted(entries, key=lambda layout: entries[layout])
            out.append(
                {
                    "grid": grid,
                    "sc": sc,
                    "class": k,
                    "diameter": s[score_key(order[0], grid, sc, k)]["diameter"],
                    "scores": entries,
                    "order": order,
                    "l2_last": order[-1] == "L2",
                    "l0_or_l1_first": order[0] in ("L0", "L1"),
                }
            )
    return out


def ranking(_args: argparse.Namespace) -> None:
    """The order of L0, L1 and L2 by the score, per grid, Sc_t and class."""
    rows = rankings(scores())
    headers = [
        "Grid, Sc_t, class (um)",
        "Ranking (best first)",
        "Scores L0, L1, L2",
        "L2 last",
        "L0 or L1 first",
    ]
    table = [
        [
            f"{r['grid']}, {sc_key(r['sc'])}, {r['diameter'] * 1e6:g}",
            " < ".join(r["order"]),
            ", ".join(f"{r['scores'][layout]:.4g}" for layout in LAYOUTS),
            "yes" if r["l2_last"] else "no",
            "yes" if r["l0_or_l1_first"] else "no",
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("ranking", rows)


def changes(s: dict, layout: str, k: str) -> dict:
    """The grid's and the Schmidt number's change to one layout's score, per section 2.6."""
    coarse = s.get(score_key(layout, GRIDS[0], BASE_SC, k))
    fine = s.get(score_key(layout, PRODUCT_GRID, BASE_SC, k))
    grid_change = (
        None if coarse is None or fine is None else abs(fine["score"] - coarse["score"])
    )
    sc_changes = []
    if fine is not None:
        for sc in SCHMIDTS:
            if sc == BASE_SC:
                continue
            other = s.get(score_key(layout, PRODUCT_GRID, sc, k))
            if other is not None:
                sc_changes.append(abs(other["score"] - fine["score"]))
    return {
        "grid": grid_change,
        "schmidt": max(sc_changes) if sc_changes else None,
        "schmidt_each": sc_changes,
    }


def margin_rows(s: dict) -> list[dict]:
    """Each neighbouring difference in each ranking against the changes."""
    out = []
    for r in rankings(s):
        for first, second in zip(r["order"], r["order"][1:], strict=False):
            a, b = r["scores"][first], r["scores"][second]
            diff = b - a
            ch_a, ch_b = changes(s, first, r["class"]), changes(s, second, r["class"])
            grid_changes = [c for c in (ch_a["grid"], ch_b["grid"]) if c is not None]
            sc_changes = (
                [c for c in (ch_a["schmidt"], ch_b["schmidt"]) if c is not None]
                if r["grid"] == PRODUCT_GRID
                else []
            )
            exceeds_all = all(diff > c for c in grid_changes + sc_changes)
            exceeds_grid_both = len(grid_changes) == 2 and all(
                diff > c for c in grid_changes
            )
            out.append(
                {
                    "grid": r["grid"],
                    "sc": r["sc"],
                    "class": r["class"],
                    "diameter": r["diameter"],
                    "pair": [first, second],
                    "scores": [a, b],
                    "difference": diff,
                    "relative": diff / max(a, b) if max(a, b) > 0 else None,
                    "grid_change": [ch_a["grid"], ch_b["grid"]],
                    "schmidt_change": [ch_a["schmidt"], ch_b["schmidt"]],
                    "robust": exceeds_all and len(grid_changes) == 2,
                    "exceeds_grid_change_of_both": exceeds_grid_both,
                    "below_grid_change_of_either": any(diff < c for c in grid_changes),
                    "below_grid_change_of_both": len(grid_changes) == 2
                    and all(diff < c for c in grid_changes),
                }
            )
    return out


def margins(_args: argparse.Namespace) -> None:
    """Each neighbouring difference against the grid's and the Schmidt number's change."""
    rows = margin_rows(scores())
    headers = [
        "Grid, Sc_t, class (um)",
        "Pair (better, worse)",
        "Scores",
        "Difference; relative to the larger",
        "Grid's change to each score (at Sc_t 0.7)",
        "Sc_t's change to each score (320x120, largest of 0.2 and 1.3 against 0.7)",
        "Robust (exceeds every change)",
    ]
    table = [
        [
            f"{r['grid']}, {sc_key(r['sc'])}, {r['diameter'] * 1e6:g}",
            ", ".join(r["pair"]),
            ", ".join(f"{v:.4g}" for v in r["scores"]),
            f"{r['difference']:.4g}; {fmt(r['relative'], 3)}",
            ", ".join(fmt(c, 3) for c in r["grid_change"]),
            ", ".join(fmt(c, 3) for c in r["schmidt_change"])
            if r["grid"] == PRODUCT_GRID
            else "-",
            "yes" if r["robust"] else "no",
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("margins", rows)


def hotspot_rows() -> list[dict]:
    """Per layout, source and class: the largest location and the shared count, pairwise."""
    recs = transport_records()
    out = []
    for layout in LAYOUTS:
        for source in SOURCES:
            for k in CLASSES:
                base = recs.get(transport_key(layout, PRODUCT_GRID, source, BASE_SC))
                if base is None:
                    continue
                b = base["per_class"][k]
                row = {
                    "layout": layout,
                    "source": source,
                    "class": k,
                    "diameter": b["diameter"],
                    "largest_320x120_sc0.7": b["hotspots"][0],
                    "hotspots_320x120_sc0.7": b["hotspots"],
                    "sensors_320x120_sc0.7": b["sensor_order_above_floor"],
                    "pairs": {},
                }
                comparisons = {
                    "grids": transport_key(layout, GRIDS[0], source, BASE_SC),
                    **{
                        f"sc{sc_key(sc)}": transport_key(
                            layout, PRODUCT_GRID, source, sc
                        )
                        for sc in SCHMIDTS
                        if sc != BASE_SC
                    },
                }
                for name, label in comparisons.items():
                    other = recs.get(label)
                    if other is None:
                        continue
                    o = other["per_class"][k]
                    shared = [s for s in b["hotspots"] if s in o["hotspots"]]
                    row["pairs"][name] = {
                        "largest": o["hotspots"][0],
                        "largest_same_segment": o["hotspots"][0] == b["hotspots"][0],
                        "largest_same_surface": o["hotspots"][0].split(" | ")[0]
                        == b["hotspots"][0].split(" | ")[0],
                        "shared": len(shared),
                        "shared_segments": shared,
                        "sensor_order_equal": o["sensor_order_above_floor"]
                        == b["sensor_order_above_floor"],
                    }
                out.append(row)
    return out


def hotspots(_args: argparse.Namespace) -> None:
    """The largest location and the shared count, between grids and between Schmidt numbers."""
    rows = hotspot_rows()
    headers = [
        "Layout, source, class (um)",
        "Largest on 320x120, Sc_t 0.7",
        "160x60: largest; same segment; shared of 5",
        "Sc_t 0.2: largest; same segment; shared of 5",
        "Sc_t 1.3: largest; same segment; shared of 5",
        "Sensors above the floor (320x120, 0.7); order equal on 160x60, 0.2, 1.3",
    ]

    def cell(p: dict | None) -> str:
        if p is None:
            return "-"
        return f"{p['largest']}; {'yes' if p['largest_same_segment'] else 'no'}; {p['shared']}"

    table = []
    for r in rows:
        pairs = r["pairs"]
        orders = ", ".join(
            "yes" if pairs[n]["sensor_order_equal"] else "no"
            for n in ("grids", "sc0.2", "sc1.3")
            if n in pairs
        )
        table.append(
            [
                f"{r['layout']}, {r['source']}, {r['diameter'] * 1e6:g}",
                r["largest_320x120_sc0.7"],
                cell(pairs.get("grids")),
                cell(pairs.get("sc0.2")),
                cell(pairs.get("sc1.3")),
                (" > ".join(r["sensors_320x120_sc0.7"]) or "none") + f"; {orders}",
            ]
        )
    print_table(headers, table)
    print("Per comparison, over the rows (layout, source, class):")
    headers = [
        "Comparison",
        "Rows",
        "Largest the same segment",
        "Largest the same surface",
        "At least three of five shared",
        "Shared: fewest, median",
        "Segments not shared, summed",
    ]
    summary = {}
    table = []
    for name in ("grids", "sc0.2", "sc1.3"):
        pairs = [r["pairs"][name] for r in rows if name in r["pairs"]]
        if not pairs:
            continue
        shared = [p["shared"] for p in pairs]
        summary[name] = {
            "rows": len(pairs),
            "largest_same_segment": sum(p["largest_same_segment"] for p in pairs),
            "largest_same_surface": sum(p["largest_same_surface"] for p in pairs),
            "at_least_three_shared": sum(p["shared"] >= HOTSPOTS_SHARED for p in pairs),
            "fewest_shared": min(shared),
            "median_shared": float(np.median(shared)),
            "not_shared_total": sum(5 - s for s in shared),
        }
        v = summary[name]
        table.append(
            [
                name,
                str(v["rows"]),
                str(v["largest_same_segment"]),
                str(v["largest_same_surface"]),
                str(v["at_least_three_shared"]),
                f"{v['fewest_shared']}, {v['median_shared']:g}",
                str(v["not_shared_total"]),
            ]
        )
    print_table(headers, table)
    keep("hotspots", {"rows": rows, "summary": summary})


def verdict(_args: argparse.Namespace) -> None:
    """Predictions (a) to (d) of section 3.1, scored from the records."""
    flows = records()
    s = scores()
    ranks = rankings(s)
    hot = hotspot_rows()
    out: dict = {}

    def converged_run(name: str) -> bool | None:
        rec = flows.get(name)
        if rec is None:
            return None
        return rec["stop"] == "error_estimate_and_continuity"

    a = {
        "L0_320x120_stops_within_cap": converged_run("L0_320x120"),
        **{
            f"{layout}_{grid}_converges": converged_run(f"{layout}_{grid}")
            for layout in ("L1", "L2")
            for grid in GRIDS
        },
    }
    a["holds"] = all(v is True for v in a.values())
    out["a"] = a

    b = {
        "orderings": len(ranks),
        "expected": len(cases()) * len(CLASSES),
        "l2_last": sum(r["l2_last"] for r in ranks),
        "l0_or_l1_first": sum(r["l0_or_l1_first"] for r in ranks),
        "failing": [
            f"{r['grid']}, {sc_key(r['sc'])}, class {r['class']}: {' < '.join(r['order'])}"
            for r in ranks
            if not (r["l2_last"] and r["l0_or_l1_first"])
        ],
    }
    b["holds"] = b["orderings"] == b["expected"] and not b["failing"]
    out["b"] = b

    c = {}
    for k in CLASSES:
        entry = {}
        for grid in GRIDS:
            l0, l1 = (
                s.get(score_key(layout, grid, BASE_SC, k)) for layout in ("L0", "L1")
            )
            if l0 is None or l1 is None:
                continue
            diff = abs(l0["score"] - l1["score"])
            ch = [changes(s, layout, k)["grid"] for layout in ("L0", "L1")]
            if any(v is None for v in ch):
                continue
            entry[grid] = {
                "difference": diff,
                "grid_change_l0": ch[0],
                "grid_change_l1": ch[1],
                "smaller_than_each": diff < min(ch),
                "smaller_than_at_least_one": diff < max(ch),
                "order": "L0 < L1" if l0["score"] < l1["score"] else "L1 < L0",
            }
        c[k] = entry
    c["holds_strict"] = any(
        e.get(PRODUCT_GRID, {}).get("smaller_than_each", False)
        for e in (c[k] for k in CLASSES)
    )
    c["holds_weak"] = any(
        e.get(PRODUCT_GRID, {}).get("smaller_than_at_least_one", False)
        for e in (c[k] for k in CLASSES)
    )
    out["c"] = c

    d = {}
    for name in ("grids", "sc1.3", "sc0.2"):
        pairs = [r["pairs"][name] for r in hot if name in r["pairs"]]
        d[name] = {
            "rows": len(pairs),
            "largest_same_segment": sum(p["largest_same_segment"] for p in pairs),
            "at_least_three_shared": sum(p["shared"] >= HOTSPOTS_SHARED for p in pairs),
            "not_shared_total": sum(5 - p["shared"] for p in pairs),
            "failing": [
                f"{r['layout']}, {r['source']}, class {r['class']}: largest "
                f"{r['pairs'][name]['largest']} against {r['largest_320x120_sc0.7']}, "
                f"shared {r['pairs'][name]['shared']}"
                for r in hot
                if name in r["pairs"]
                and not (
                    r["pairs"][name]["largest_same_segment"]
                    and r["pairs"][name]["shared"] >= HOTSPOTS_SHARED
                )
            ],
        }
    expected_rows = len(LAYOUTS) * len(SOURCES) * len(CLASSES)
    d["clause_grids_and_1.3_holds"] = all(
        d[n]["rows"] == expected_rows and not d[n]["failing"]
        for n in ("grids", "sc1.3")
    )
    d["sc0.2_moves_more"] = d["sc0.2"]["rows"] > 0 and d["sc0.2"][
        "not_shared_total"
    ] > max(d["grids"]["not_shared_total"], d["sc1.3"]["not_shared_total"])
    d["holds"] = d["clause_grids_and_1.3_holds"] and d["sc0.2_moves_more"]
    out["d"] = d

    headers = ["Prediction", "Holds", "From the records"]
    table = [
        [
            "(a) L0 320x120 stops within the cap; L1 and L2 converge on both grids",
            "yes" if a["holds"] else "no",
            "; ".join(f"{k} {v}" for k, v in a.items() if k != "holds"),
        ],
        [
            "(b) L0 or L1 first and L2 last in every ordering",
            "yes" if b["holds"] else "no",
            f"{b['orderings']} of {b['expected']} orderings; L2 last in {b['l2_last']}, "
            f"L0 or L1 first in {b['l0_or_l1_first']}; failing: {b['failing'] or 'none'}",
        ],
        [
            "(c) L0 and L1's difference smaller than the grid's change to either score, for at least one class (320x120)",
            f"strict {'yes' if c['holds_strict'] else 'no'}, weak {'yes' if c['holds_weak'] else 'no'}",
            "; ".join(
                f"class {k} {grid}: {e['order']}, difference {e['difference']:.4g}, grid change "
                f"L0 {e['grid_change_l0']:.4g}, L1 {e['grid_change_l1']:.4g}"
                for k in CLASSES
                for grid, e in c[k].items()
            ),
        ],
        [
            "(d) largest the same segment and three of five shared between grids and at Sc_t 1.3 for every row; Sc_t 0.2 moves more",
            "yes" if d["holds"] else "no",
            "; ".join(
                f"{n}: {d[n]['rows']} rows, largest same {d[n]['largest_same_segment']}, "
                f"three shared {d[n]['at_least_three_shared']}, not shared {d[n]['not_shared_total']}"
                for n in ("grids", "sc1.3", "sc0.2")
            )
            + f"; failing: {d['grids']['failing'] + d['sc1.3']['failing'] or 'none'}",
        ],
    ]
    print_table(headers, table)
    keep("verdict", out)


def everything(args: argparse.Namespace) -> None:
    """Every table, in the report's order."""
    for f in (
        timing,
        layouts,
        matrix,
        converged,
        bounded,
        transport,
        score,
        ranking,
        margins,
        hotspots,
        verdict,
    ):
        print(f"### {f.__name__}")
        f(args)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, func in (
        ("timing", timing),
        ("layouts", layouts),
        ("matrix", matrix),
        ("converged", converged),
        ("bounded", bounded),
        ("transport", transport),
        ("score", score),
        ("ranking", ranking),
        ("margins", margins),
        ("hotspots", hotspots),
        ("verdict", verdict),
        ("all", everything),
    ):
        sub.add_parser(name).set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
