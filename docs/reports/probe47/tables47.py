"""The report's tables from the prompt 47 records.

Usage:
    python tables47.py matrix       every flow run, classified by step 5's rule
    python tables47.py cycle        measurement 1: the 80x30 arms side by side
    python tables47.py limiter      measurement 1: the limiter diagnostic
    python tables47.py converged    measurement 2: the converged rooms' readings, core and y+
    python tables47.py bounded      the bounded rows, from bounded47.py's output (tables46's)
    python tables47.py checks       measurement 3: the discrimination check and any move
    python tables47.py cfl          measurement 3: the 0.1 against 0.4 Courant check
    python tables47.py transport    measurement 3: the march, the sensors, the surfaces, the hotspots
    python tables47.py compare      measurement 5: between grids and between variants
    python tables47.py all          everything above

Reads results/builder47/*.json, prints markdown tables, and keeps every
number it printed in results/builder47/tables_MODE.json. The
classification is tables46's (imported): diverged, converged, growing,
bounded with step 0's sub-classes, or a positivity error.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe44"))

import bounded44  # noqa: E402
import tables46  # noqa: E402
from tables46 import classify, fmt, print_table  # noqa: E402

OUT = ROOT / "results" / "builder47"
tables46.OUT = OUT
GRIDS = ("80x30", "160x60", "320x120")
VARIANTS = ("standard", "rng")
ARMS = ("control", "upwind", "corner", "alpha", "limiter")
NAME = re.compile(r"^(?P<variant>standard|rng)_(?P<grid>\d+x\d+)(?P<rest>.*)$")
TRANSPORT = re.compile(
    r"^transport_(?P<flow>(?:standard|rng)_\d+x\d+(?:_[a-z]+)?)_(?P<source>S\d+m?)(?P<tag>_.*)?$"
)
# Prompt 46's standard_80x30 row, which the control is expected to reproduce.
PROMPT46_CONTROL_HASH = "6ab5e2c7d5a1fa0c"
SENSOR_NAMES = ("near_door", "above_gap_1", "above_gap_2", "hood_entry")
SURFACE_SHARE_FLOOR = 1e-6
TIMING_GRIDS = ("160x60", "320x120")


def load(name: str) -> dict:
    """A record by name."""
    return json.loads((OUT / f"{name}.json").read_text())


def keep(mode: str, data: object) -> None:
    """The printed numbers, as JSON beside the records."""
    (OUT / f"tables_{mode}.json").write_text(json.dumps(data, indent=1, default=float))


def records() -> dict[str, dict]:
    """Every flow record, keyed by name; no timing, smoke or transport rows."""
    out = {}
    for path in sorted(OUT.glob("*.json")):
        m = NAME.match(path.stem)
        if not m or path.stem.endswith(("_time", "_smoke")):
            continue
        rec = load(path.stem)
        if "stop" not in rec:
            continue
        rec["_variant"], rec["_grid"], rec["_rest"] = m["variant"], m["grid"], m["rest"]
        rec["_arm"] = rec.get("arm", "control")
        out[path.stem] = rec
    return out


def order_key(rec: dict) -> tuple:
    """Grid, then variant, then arm."""
    return (
        GRIDS.index(rec["_grid"]) if rec["_grid"] in GRIDS else 99,
        VARIANTS.index(rec["_variant"]),
        ARMS.index(rec["_arm"]) if rec["_arm"] in ARMS else 99,
        rec["_rest"],
    )


def readings_cell(rec: dict) -> str:
    r = rec.get("readings_at_stop") or {}
    return ", ".join(fmt(r.get(k), 2) for k in ("a", "b", "c", "d", "e"))


def from_when_cell(rec: dict) -> str:
    f = rec.get("from_when") or {}
    return ", ".join(fmt(f.get(k)) for k in ("a", "b", "c", "d", "e"))


# ---------------------------------------------------------------------------
# Flow tables
# ---------------------------------------------------------------------------


def matrix(_args: argparse.Namespace) -> None:
    """Every flow run, one line each."""
    rows = []
    for rec in sorted(records().values(), key=order_key):
        cls, sub = classify(rec)
        rows.append(
            {
                "name": rec["name"],
                "variant": rec["_variant"],
                "grid": rec["_grid"],
                "arm": rec["_arm"],
                "alpha_turbulence": rec.get("alpha_turbulence"),
                "class": cls + (f" ({sub})" if sub else ""),
                "stop": rec["stop"],
                "outer": rec["outer"],
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
        "Outer",
        "Residual: least (at), end",
        "Readings at stop (a), (b), (c), (d), (e)",
        "Holds from (a), (b), (c), (d), (e)",
        "Largest speed at end (m/s), cell",
        "Peak speed",
        "CG per correction: mean, largest",
        "Cap hits",
        "k, eps sweeps (mean)",
        "Wall (s), per outer",
        "Face hash",
    ]
    table = [
        [
            r["name"],
            r["class"],
            r["stop"],
            fmt(r["outer"]),
            f"{r['residual_min']:.2e} ({r['residual_min_at']:,}), {r['residual_end']:.2e}",
            readings_cell(r) if r["readings_at_stop"] else "-",
            from_when_cell(r) if r["from_when"] else "-",
            f"{r['max_speed_end']:.3g}, {tuple(r['at_end'])}",
            f"{r['max_speed_peak']:.3g}",
            f"{r['inner_mean']:.0f}, {r['inner_max']}",
            fmt(r["cap_hits"]),
            f"{r['k_sweeps_mean']:.1f}, {r['eps_sweeps_mean']:.1f}",
            f"{r['seconds']:.0f}, {r['seconds_per_outer']:.3f}",
            (r["face_hash"] or "-")[:16],
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("matrix", rows)


def timing(_args: argparse.Namespace) -> None:
    """The timing probes: seconds per outer iteration and the projection at the cap."""
    rows = []
    for grid in TIMING_GRIDS:
        path = OUT / f"standard_{grid}_time.json"
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
                "minutes_at_cap": r["seconds_per_outer"] * 10000 / 60,
                "started": r["started"],
            }
        )
    headers = [
        "Grid",
        "Outer",
        "Seconds per outer",
        "CG per correction (mean)",
        "k, eps sweeps (mean)",
        "Share of wall: pressure, turbulence",
        "Minutes at the 10,000 cap",
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
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("timing", rows)


def tail_step(rec: dict, n: int = 2000) -> dict:
    """The velocity step over the tail, in m/s and over the velocity scale."""
    n = min(n, max(rec["outer"] - 100, 1))
    steps = np.array(
        [s for s in rec["readings"]["velocity_step"][-n:] if s is not None]
    )
    scale = rec["velocity_scale"]
    if steps.size == 0:
        return {}
    return {
        "median_m_per_s": float(np.median(steps)),
        "largest_m_per_s": float(steps.max()),
        "median_over_scale": float(np.median(steps)) / scale,
        "largest_over_scale": float(steps.max()) / scale,
    }


def cycle(_args: argparse.Namespace) -> None:
    """Measurement 1: the 80x30 arms side by side."""
    rows = []
    for rec in sorted(records().values(), key=order_key):
        if rec["_grid"] != "80x30":
            continue
        cls, sub = classify(rec)
        step = tail_step(rec)
        rows.append(
            {
                "name": rec["name"],
                "arm": rec["_arm"],
                "alpha_turbulence": rec.get("alpha_turbulence"),
                "class": cls + (f" ({sub})" if sub else ""),
                "outer": rec["outer"],
                "residual_end": rec["residual_end"],
                "readings_at_stop": rec.get("readings_at_stop"),
                "from_when": rec.get("from_when"),
                "tail_velocity_step": step,
                "face_hash": rec["face_hash"],
                "matches_prompt46": (rec["face_hash"] or "")[:16]
                == PROMPT46_CONTROL_HASH,
                "seconds": rec["seconds"],
            }
        )
    headers = [
        "Run (arm)",
        "Change",
        "Class",
        "Outer",
        "Residual at end",
        "Readings at stop (a), (b), (c), (d), (e)",
        "Holds from (a), (b), (c), (d), (e)",
        "Tail velocity step: median, largest (over 0.45 m/s)",
        "Face hash (equals prompt 46's row)",
    ]
    changes = {
        "control": "none",
        "upwind": "k and eps advected by upwind",
        "corner": "momentum corner rule removed",
        "alpha": "alpha_turbulence 0.35",
        "limiter": "none (branches recorded)",
    }
    table = [
        [
            f"{r['name']} ({r['arm']})",
            changes.get(r["arm"], r["arm"]),
            r["class"],
            fmt(r["outer"]),
            f"{r['residual_end']:.2e}",
            readings_cell(r),
            from_when_cell(r),
            f"{r['tail_velocity_step'].get('median_over_scale', float('nan')):.2g}, "
            f"{r['tail_velocity_step'].get('largest_over_scale', float('nan')):.2g}"
            if r["tail_velocity_step"]
            else "-",
            f"{(r['face_hash'] or '-')[:16]} ({'yes' if r['matches_prompt46'] else 'no'})",
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("cycle", rows)


def limiter(_args: argparse.Namespace) -> None:
    """Measurement 1: the limiter diagnostic's counts and the faces that switch most."""
    rows = []
    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    boxes = bounded44.regions(raw)
    tops = max(o["y_end"] for o in raw["obstacles"])
    for rec in sorted(records().values(), key=order_key):
        if "limiter" not in rec:
            continue
        lim = rec["limiter"]
        switches = np.array(rec["limiter_branch_switches"])
        signs = np.array(rec["limiter_sign_switches"])
        by_call = np.array(rec["limiter_branch_switches_by_call"])
        tail = min(bounded44.TAIL, max(len(switches) - 100, 0))
        faces = lim["faces_switching_most"]
        by_region = collections.Counter(
            bounded44.region_of(f["x"], f["y"], boxes, tops) for f in faces
        )
        rows.append(
            {
                "name": rec["name"],
                "outer": rec["outer"],
                "faces_total": lim["faces_total"],
                "tail_iterations": lim["tail_iterations"],
                "branch_switches": lim["branch_switches_per_iteration"],
                "sign_switches": lim["sign_switches_per_iteration"],
                "branch_switches_by_call_tail_median": [
                    float(np.median(by_call[-tail:, i]))
                    for i in range(by_call.shape[1])
                ]
                if by_call.size
                else [],
                "switching_faces_in_tail": lim["switching_faces_in_tail"],
                "faces_switching_most": faces,
                "regions_of_top_faces": dict(by_region.most_common()),
                "branch_shares_at_end": lim["branch_shares_at_end"],
                "per_thousand": [
                    {
                        "from": a + 1,
                        "branch_median": float(np.median(switches[a : a + 1000])),
                        "sign_median": float(np.median(signs[a : a + 1000])),
                    }
                    for a in range(0, len(switches), 1000)
                ],
            }
        )
    headers = [
        "Run",
        "Faces per iteration (4 calls)",
        "Tail iterations",
        "Branch switches per iteration: least, median, largest; iterations with none",
        "Flux sign switches per iteration: least, median, largest; iterations with none",
        "Median switches per call (k x, k y, eps x, eps y)",
        "Faces that switched at least once in the tail",
        "Faces switching most (call, x, y, share of tail)",
        "Regions of those faces",
        "Branch shares at the end (zero, 2r, (1+3r)/4, quick, 2, c_c)",
    ]
    table = []
    for r in rows:
        b, s = r["branch_switches"], r["sign_switches"]
        top = "; ".join(
            f"{f['call']} ({f['x']:g}, {f['y']:g}) {f['share_of_tail']:.2f}"
            for f in r["faces_switching_most"][:8]
        )
        table.append(
            [
                r["name"],
                fmt(r["faces_total"]),
                fmt(r["tail_iterations"]),
                f"{b['least']}, {b['median']:g}, {b['largest']}; {b['iterations_with_none']}",
                f"{s['least']}, {s['median']:g}, {s['largest']}; {s['iterations_with_none']}",
                ", ".join(f"{m:g}" for m in r["branch_switches_by_call_tail_median"]),
                fmt(r["switching_faces_in_tail"]),
                top,
                "; ".join(f"{k} {v}" for k, v in r["regions_of_top_faces"].items()),
                ", ".join(f"{v:,}" for v in r["branch_shares_at_end"].values())
                if r["branch_shares_at_end"]
                else "-",
            ]
        )
    print_table(headers, table)
    if rows:
        print(
            "Branch and sign switches per iteration, medians per thousand iterations:"
        )
        headers = ["Run"] + [f"{p['from']:,}" for p in rows[0]["per_thousand"]]
        table = [
            [r["name"] + " (branch)"]
            + [f"{p['branch_median']:g}" for p in r["per_thousand"]]
            for r in rows
        ] + [
            [r["name"] + " (sign)"]
            + [f"{p['sign_median']:g}" for p in r["per_thousand"]]
            for r in rows
        ]
        print_table(headers, table)
    keep("limiter", rows)


def converged_rows(_args: argparse.Namespace) -> None:
    """Measurement 2: the converged rooms' readings, core eddy viscosity and y+."""
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
                "nu_scale": rec["nu_scale"],
                "nu_air": rec["nu_air"],
                "inlet_nu_t_over_nu": (rec["nu_scale"] - rec["nu_air"]) / rec["nu_air"],
                "core": core,
                "y_star": y,
                "figure": f"ecr002_step6_{rec['name']}.png",
            }
        )
    headers = [
        "Run",
        "Outer, stop",
        "Readings at stop (a), (b), (c), (d), (e)",
        "Holds from (a), (b), (c), (d), (e)",
        "Wall (s)",
        "Inlet nu_t / nu",
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
                f"{r['seconds']:.0f}",
                fmt(r["inlet_nu_t_over_nu"]),
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
# Transport tables
# ---------------------------------------------------------------------------


def transport_records() -> dict[str, dict]:
    """Every transport record, keyed by its label, with the flow name, source and tag parsed."""
    out = {}
    for path in sorted(OUT.glob("transport_*.json")):
        m = TRANSPORT.match(path.stem)
        if not m or path.stem.endswith("_smoke"):
            continue
        rec = load(path.stem)
        rec["_flow"], rec["_source"], rec["_tag"] = (
            m["flow"],
            m["source"],
            m["tag"] or "",
        )
        out[path.stem] = rec
    return out


def checks(_args: argparse.Namespace) -> None:
    """Measurement 3: the discrimination check per source, and the move if any."""
    path = OUT / "checks.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    recs = transport_records()
    rows = []
    for key, c in data.items():
        label = f"transport_{c['flow_record']}_{c['source']}"
        rec = recs.get(label)
        per_class = {}
        if rec is not None:
            for k, v in rec["per_class"].items():
                per_class[k] = {
                    "diameter": v["diameter"],
                    "floor": v["floor"],
                    "source_cell_concentration": v["source_cell_concentration"],
                    "sensors_above_floor": v["sensors_above_floor"],
                    "surfaces_above_floor": v["surfaces_above_floor"],
                    "discriminates": v["discriminates"],
                }
        rows.append({"key": key, **c, "per_class": per_class})
    headers = [
        "Source (record)",
        "Square (m)",
        "Class (um)",
        "Source-cell C, floor (per m^3)",
        "Sensors above the floor (excluding one inside the square)",
        "Surfaces above the floor (excluding ones the square touches)",
        "Discriminates",
        "Verdict; move",
    ]
    table = []
    for r in rows:
        first = True
        for v in r["per_class"].values():
            moved = r["moved"]
            table.append(
                [
                    f"{r['source']} ({r['flow_record']})" if first else "",
                    str(r["box"]) if first else "",
                    f"{v['diameter'] * 1e6:g}",
                    f"{v['source_cell_concentration']:.3g}, {v['floor']:.3g}",
                    ", ".join(
                        f"{s} {c:.2g}" for s, c in v["sensors_above_floor"].items()
                    )
                    or "none",
                    ", ".join(
                        f"{s} {c:.2g}" for s, c in v["surfaces_above_floor"].items()
                    )
                    or "none",
                    "yes" if v["discriminates"] else "no",
                    (
                        ("passes" if r["passes"] else "fails")
                        + (
                            f"; moved to {moved['name']} {moved['box']} ({moved['rule']})"
                            if moved
                            else ""
                        )
                    )
                    if first
                    else "",
                ]
            )
            first = False
    print_table(headers, table)
    keep("checks", rows)


def cfl(_args: argparse.Namespace) -> None:
    """Measurement 3: the same march at Courant 0.1 and 0.4, sensors and surfaces compared."""
    recs = transport_records()
    rows = []
    for label, rec in recs.items():
        if not rec["_tag"]:
            continue
        base = f"transport_{rec['_flow']}_{rec['_source']}"
        if base not in recs:
            continue
        a = recs[base]
        for k in a["per_class"]:
            pa, pb = a["per_class"][k], rec["per_class"][k]
            sensor_diff = max(
                abs(pa["sensors"][s] - pb["sensors"][s])
                / max(max(pa["sensors"].values()), 1e-300)
                for s in pa["sensors"]
            )
            surfaces = set(pa["by_surface"]) | set(pb["by_surface"])
            top = max(pa["by_surface"].values())
            surface_diff = max(
                abs(pa["by_surface"].get(s, 0.0) - pb["by_surface"].get(s, 0.0)) / top
                for s in surfaces
            )
            rows.append(
                {
                    "base": base,
                    "check": label,
                    "class": k,
                    "diameter": pa["diameter"],
                    "cfl": [a["cfl_number"], rec["cfl_number"]],
                    "dt": [a["dt"], rec["dt"]],
                    "t_end": [pa["t_end"], pb["t_end"]],
                    "seconds": [a["seconds"], rec["seconds"]],
                    "sensor_max_relative_difference": sensor_diff,
                    "surface_max_relative_difference": surface_diff,
                    "hotspots_agree": pa["hotspots"] == pb["hotspots"],
                    "order_agree": pa["sensor_order_above_floor"]
                    == pb["sensor_order_above_floor"],
                }
            )
    headers = [
        "Rows (0.4, check), class (um)",
        "Courant",
        "dt (s)",
        "Steady at (s)",
        "Wall (s)",
        "Sensors: largest difference over the largest reading",
        "Surfaces: largest difference over the largest rate",
        "Hotspots equal",
        "Sensor order equal",
    ]
    table = [
        [
            f"{r['base']}, {r['check']}, {r['diameter'] * 1e6:g}",
            ", ".join(f"{c:g}" for c in r["cfl"]),
            ", ".join(f"{d:.3g}" for d in r["dt"]),
            ", ".join(f"{t:.0f}" for t in r["t_end"]),
            ", ".join(f"{s:.0f}" for s in r["seconds"]),
            f"{r['sensor_max_relative_difference']:.2e}",
            f"{r['surface_max_relative_difference']:.2e}",
            "yes" if r["hotspots_agree"] else "no",
            "yes" if r["order_agree"] else "no",
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("cfl", rows)


def transport(_args: argparse.Namespace) -> None:
    """Measurement 3: the march, the sensors above the floor, the surfaces and the hotspots."""
    recs = {k: v for k, v in transport_records().items() if not v["_tag"]}
    kept: dict = {"runs": {}, "surfaces": {}, "hotspots": {}}
    headers = [
        "Run, class (um)",
        "Stop, t (s)",
        "Last window: total, sensor change; removal / Q",
        "Deposition / Q (faces; budget), outflow / Q",
        "Budget residual",
        "Source-cell C, floor",
        "Sensors above the floor, in order (per m^3)",
    ]
    table = []
    for label, rec in recs.items():
        for k, c in rec["per_class"].items():
            um = c["diameter"] * 1e6
            lw = c["last_window"]
            above = c["sensors_above_floor"]
            table.append(
                [
                    f"{label[len('transport_') :]}, {um:g}",
                    f"{c['stop']}, {c['t_end']:.0f}",
                    f"{fmt(lw['total_change'], 2)}, {fmt(lw['sensor_change'], 2)}; "
                    f"{fmt(lw['removal_over_source'], 5)}",
                    f"{c['deposition_rate_over_source']:.4g} "
                    f"({c['budget_deposition_rate_over_source']:.4g}), "
                    f"{c['outflow_rate_over_source']:.4g}",
                    fmt(c["budget"]["relative_residual"], 2),
                    f"{c['source_cell_concentration']:.3g}, {c['floor']:.3g}",
                    " > ".join(
                        f"{s} ({above[s]:.3g})" for s in c["sensor_order_above_floor"]
                    )
                    or "none",
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
    # Deposition per surface, one table per source, columns per run and class.
    sources = sorted({r["_source"] for r in recs.values()})
    for source in sources:
        columns = [
            (label, k)
            for label, r in recs.items()
            if r["_source"] == source
            for k in r["per_class"]
        ]
        surfaces: list[str] = []
        for label, k in columns:
            shares = recs[label]["per_class"][k]["by_surface_share"]
            for s, share in shares.items():
                # Below a millionth of the total a surface's value is the
                # scheme's tail, not transport; such rows are left out.
                if s not in surfaces and (share or 0.0) >= SURFACE_SHARE_FLOOR:
                    surfaces.append(s)
        print(
            f"Deposition per surface, source {source}, share of the deposition total "
            f"(surfaces with at least {SURFACE_SHARE_FLOOR:g} of it in some column):"
        )
        headers = ["Surface"] + [
            f"{label[len('transport_') :]}, {recs[label]['per_class'][k]['diameter'] * 1e6:g} um"
            for label, k in columns
        ]
        table = [
            [s]
            + [
                fmt(recs[label]["per_class"][k]["by_surface_share"].get(s), 3)
                for label, k in columns
            ]
            for s in surfaces
        ]
        print_table(headers, table)
        kept["surfaces"][source] = {
            f"{label}/{k}": recs[label]["per_class"][k]["by_surface"]
            for label, k in columns
        }
    print(
        "The five segments of largest deposition (surface | bin start), rate per s per m depth:"
    )
    headers = ["Run, class (um)", "1", "2", "3", "4", "5"]
    table = []
    for label, rec in recs.items():
        for k, c in rec["per_class"].items():
            table.append(
                [f"{label[len('transport_') :]}, {c['diameter'] * 1e6:g}"]
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


def pairs(recs: dict[str, dict]) -> list[tuple[str, str, str, str]]:
    """The comparisons the records allow: between grids (standard) and between variants."""
    by = {(r["_flow"], r["_source"]): label for label, r in recs.items()}
    sources = sorted({r["_source"] for r in recs.values()})
    out = []
    for source in sources:
        a, b = by.get(("standard_160x60", source)), by.get(("standard_320x120", source))
        if a and b:
            out.append(("grids, standard", source, a, b))
        for grid in ("160x60", "320x120"):
            a = by.get((f"standard_{grid}", source))
            b = by.get((f"rng_{grid}", source))
            if a and b:
                out.append((f"variants, {grid}", source, a, b))
    return out


def compare(_args: argparse.Namespace) -> None:
    """Measurement 5: hotspots in common, the largest on each, the sensor orders."""
    recs = {k: v for k, v in transport_records().items() if not v["_tag"]}
    rows = []
    headers = [
        "Pair, source, class (um)",
        "Hotspots in common (of 5)",
        "In common",
        "Largest: first",
        "Largest: second",
        "Same surface",
        "Sensor order above the floor: first",
        "Second",
        "Orders equal",
    ]
    table = []
    for label, source, first, second in pairs(recs):
        for k in recs[first]["per_class"]:
            a, b = recs[first]["per_class"][k], recs[second]["per_class"][k]
            common = [s for s in a["hotspots"] if s in b["hotspots"]]
            top_a, top_b = a["hotspots"][0], b["hotspots"][0]
            surface_a = top_a.split(" | ")[0]
            surface_b = top_b.split(" | ")[0]
            order_a, order_b = (
                a["sensor_order_above_floor"],
                b["sensor_order_above_floor"],
            )
            rows.append(
                {
                    "pair": label,
                    "source": source,
                    "first": first,
                    "second": second,
                    "class": k,
                    "diameter": a["diameter"],
                    "common": common,
                    "common_count": len(common),
                    "only_first": [s for s in a["hotspots"] if s not in b["hotspots"]],
                    "only_second": [s for s in b["hotspots"] if s not in a["hotspots"]],
                    "top_first": top_a,
                    "top_second": top_b,
                    "top_same_surface": surface_a == surface_b,
                    "top_same_segment": top_a == top_b,
                    "order_first": order_a,
                    "order_second": order_b,
                    "orders_equal": order_a == order_b,
                }
            )
            table.append(
                [
                    f"{label}, {source}, {a['diameter'] * 1e6:g}",
                    str(len(common)),
                    "; ".join(common) or "-",
                    top_a,
                    top_b,
                    "yes" if surface_a == surface_b else "no",
                    " > ".join(order_a) or "none",
                    " > ".join(order_b) or "none",
                    "yes" if order_a == order_b else "no",
                ]
            )
    print_table(headers, table)
    keep("compare", rows)


def everything(args: argparse.Namespace) -> None:
    """Every table, in the report's order."""
    for f in (
        timing,
        matrix,
        cycle,
        limiter,
        converged_rows,
        bounded,
        checks,
        cfl,
        transport,
        compare,
    ):
        print(f"### {f.__name__}")
        f(args)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, func in (
        ("timing", timing),
        ("matrix", matrix),
        ("cycle", cycle),
        ("limiter", limiter),
        ("converged", converged_rows),
        ("bounded", bounded),
        ("checks", checks),
        ("cfl", cfl),
        ("transport", transport),
        ("compare", compare),
        ("all", everything),
    ):
        sub.add_parser(name).set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
