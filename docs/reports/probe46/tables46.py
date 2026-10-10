"""The report's tables from the step 6 product records (prompt 46).

Usage:
    python tables46.py matrix       measurement 1 and 2: every flow run, classified
    python tables46.py converged    measurement 3: the converged rooms' readings, core and y+
    python tables46.py inlet        measurement 2: the three 80x30 inlet settings side by side
    python tables46.py bounded      the bounded rows' characterisation, from bounded46.py's output
    python tables46.py transport    measurement 4: sensors, surfaces and hotspots per run and class
    python tables46.py compare      the comparative check: hotspots in common and sensor orders
    python tables46.py all          everything above

Reads results/builder46/*.json, prints markdown tables, and keeps every
number it printed in results/builder46/tables_MODE.json. Each flow run is
classified by step 5's rule in its order (the report's section 2.1):
diverged, converged (the solver's own stop), growing (at the cap with the
median of the largest speed over the last 500 iterations above 5 m/s),
bounded and not converged (at the cap with that median under 5 m/s) with
step 0's sub-classes falling, stalled or neither over those 500 iterations;
a positivity error is its own class.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.mesh import SOLID  # noqa: E402

OUT = ROOT / "results" / "builder46"
GRIDS = ("40x15", "80x30", "200x75")
VARIANTS = ("standard", "rng")
NAME = re.compile(r"^(?P<variant>standard|rng)_(?P<grid>\d+x\d+)(?P<rest>.*)$")
GROWING_SPEED = 5.0
WINDOW = 500
# The prompt's two pairs, then the supplementary pairs the converged rows
# allow when a prompt pair is missing a record (the report's section 5.6).
PAIRS = (
    ("grids, standard", "standard_80x30", "standard_200x75"),
    ("variants, 200x75", "standard_200x75", "rng_200x75"),
    ("supplementary: grids, rng", "rng_40x15", "rng_80x30"),
    (
        "supplementary: rng 80x30 against standard 200x75",
        "rng_80x30",
        "standard_200x75",
    ),
)


def load(name: str) -> dict:
    """A record by name."""
    return json.loads((OUT / f"{name}.json").read_text())


def records() -> dict[str, dict]:
    """Every flow record, keyed by name, with its parsed name; no timing or transport rows."""
    out = {}
    for path in sorted(OUT.glob("*.json")):
        m = NAME.match(path.stem)
        if not m or path.stem.endswith("_time") or path.stem.endswith("_smoke"):
            continue
        rec = load(path.stem)
        if "stop" not in rec:
            continue
        rec["_variant"], rec["_grid"], rec["_rest"] = m["variant"], m["grid"], m["rest"]
        out[path.stem] = rec
    return out


def classify(rec: dict) -> tuple[str, str]:
    """The class and, for a bounded run, step 0's sub-class."""
    if rec["stop"] == "diverged":
        return "diverged", ""
    if rec["stop"] == "positivity_error":
        return "positivity error", ""
    if rec["stop"] == "error_estimate_and_continuity":
        return "converged", ""
    speeds = np.array(rec["max_speed"])[-WINDOW:]
    if float(np.median(speeds)) > GROWING_SPEED:
        return "growing", ""
    res = np.array(rec["residual"])
    w = res[-WINDOW:]
    if w[-1] <= 1.1 * w.min() and w[-1] < 0.5 * w[0]:
        sub = "falling"
    elif w.max() / w.min() < 2.0:
        sub = "stalled"
    else:
        sub = "neither"
    return "bounded", sub


def fmt(x: object, digits: int = 3) -> str:
    """A number for a table cell."""
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int) or (
        isinstance(x, float) and x.is_integer() and abs(x) >= 1000
    ):
        return f"{int(x):,}"
    return f"{x:.{digits}g}"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """A markdown table."""
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


def keep(mode: str, data: object) -> None:
    """The printed numbers, as JSON beside the records."""
    (OUT / f"tables_{mode}.json").write_text(json.dumps(data, indent=1, default=float))


def order_key(rec: dict) -> tuple:
    """Baseline rows first by variant and grid, then the check and inlet rows."""
    return (
        rec["_rest"] != "",
        VARIANTS.index(rec["_variant"]),
        GRIDS.index(rec["_grid"]),
        rec["_rest"],
    )


def readings_cell(rec: dict) -> str:
    """The five readings at the stop, (a) to (e), in order."""
    r = rec.get("readings_at_stop", {})
    return ", ".join(fmt(r.get(k), 2) for k in ("a", "b", "c", "d", "e"))


def from_when_cell(rec: dict) -> str:
    """From which outer iteration each condition holds to the end."""
    f = rec.get("from_when", {})
    return ", ".join(fmt(f.get(k)) for k in ("a", "b", "c", "d", "e"))


def matrix(_args: argparse.Namespace) -> None:
    """Measurements 1 and 2: every flow run, one line each."""
    recs = records()
    rows = []
    for rec in sorted(recs.values(), key=order_key):
        cls, sub = classify(rec)
        rows.append(
            {
                "name": rec["name"],
                "variant": rec["_variant"],
                "grid": rec["_grid"],
                "pressure_rtol": rec["pressure_rtol"],
                "turbulence_intensity": rec["turbulence_intensity"],
                "dissipation_length": rec["dissipation_length"],
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
            ", ".join(
                fmt(r["readings_at_stop"].get(k), 2) for k in ("a", "b", "c", "d", "e")
            )
            if r["readings_at_stop"]
            else "-",
            ", ".join(fmt(r["from_when"].get(k)) for k in ("a", "b", "c", "d", "e"))
            if r["from_when"]
            else "-",
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


def converged_rows(_args: argparse.Namespace) -> None:
    """Measurement 3: the converged rooms' readings, core eddy viscosity and y+."""
    recs = records()
    rows = []
    for rec in sorted(recs.values(), key=order_key):
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
        "Wall (s)",
        "Inlet nu_t / nu",
        "Core nu_t / nu: median, 95th",
        "Core nu_t / nu: 5th, 25th, 75th",
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
                ", ".join(
                    fmt(r["readings_at_stop"].get(k), 2)
                    for k in ("a", "b", "c", "d", "e")
                ),
                f"{r['seconds']:.0f}",
                fmt(r["inlet_nu_t_over_nu"]),
                f"{r['core']['nu_t_over_nu_median']:.3g}, {r['core']['nu_t_over_nu_p95']:.3g}",
                ", ".join(
                    f"{r['core']['nu_t_over_nu_percentiles'][i]:.3g}" for i in (0, 1, 3)
                ),
                fmt(r["y_star"]["nodes"]),
                f"{r['y_star']['median']:.3g}, {r['y_star']['least']:.3g}, {r['y_star']['largest']:.3g}",
                f"{r['y_star']['share_below_floor']:.2g}; {share}",
                r["figure"],
            ]
        )
    print_table(headers, table)
    keep("converged", rows)


def inlet(_args: argparse.Namespace) -> None:
    """Measurement 2: the 80x30 standard rows at the three inlet settings."""
    recs = records()
    rows = []
    for rec in sorted(recs.values(), key=order_key):
        if rec["_grid"] != "80x30" or rec["_variant"] != "standard":
            continue
        if rec["pressure_rtol"] != 1e-4:
            continue
        cls, sub = classify(rec)
        k_in, eps_in = rec["initial_k_eps"]
        rows.append(
            {
                "name": rec["name"],
                "turbulence_intensity": rec["turbulence_intensity"],
                "dissipation_length": rec["dissipation_length"],
                "inlet_k": k_in,
                "inlet_eps": eps_in,
                "inlet_nu_t_over_nu": (rec["nu_scale"] - rec["nu_air"]) / rec["nu_air"],
                "class": cls + (f" ({sub})" if sub else ""),
                "outer": rec["outer"],
                "core": rec.get("core"),
                "y_star": rec.get("y_star"),
                "max_speed_end": rec["max_speed_end"],
                "at_end": rec["at_end"],
            }
        )
    headers = [
        "Run",
        "Intensity, length (m)",
        "Inlet k (m^2/s^2), eps (m^2/s^3)",
        "Inlet nu_t / nu",
        "Class",
        "Outer",
        "Core nu_t / nu: median, 95th",
        "y+: median, share below 11.53",
        "Largest speed, cell",
    ]
    table = [
        [
            r["name"],
            f"{r['turbulence_intensity']}, {r['dissipation_length']}",
            f"{r['inlet_k']:.3g}, {r['inlet_eps']:.3g}",
            fmt(r["inlet_nu_t_over_nu"]),
            r["class"],
            fmt(r["outer"]),
            f"{r['core']['nu_t_over_nu_median']:.3g}, {r['core']['nu_t_over_nu_p95']:.3g}"
            if r["core"]
            else "-",
            f"{r['y_star']['median']:.3g}, {r['y_star']['share_below_floor']:.2g}"
            if r["y_star"]
            else "-",
            f"{r['max_speed_end']:.3g}, {tuple(r['at_end'])}",
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("inlet", rows)


def viscosity_where(name: str) -> dict | None:
    """Where the largest nu_t change sits over a located record's tail (bounded44's regions)."""
    path = OUT / f"{name}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    if "dnu_at" not in rec:
        return None
    sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe44"))
    import collections

    import bounded44
    import yaml

    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    boxes = bounded44.regions(raw)
    tops = max(o["y_end"] for o in raw["obstacles"])
    n = len(rec["residual"])
    tail = min(bounded44.TAIL, max(n - 100, 0))
    cells = [tuple(c) for c in rec["dnu_at"][-tail:]]
    sizes = np.array(rec["dnu"][-tail:])
    by_region = collections.Counter(
        bounded44.region_of(x, y, boxes, tops) for x, y in cells
    )
    return {
        "share_by_region": {k: v / tail for k, v in by_region.most_common()},
        "cells": collections.Counter(cells).most_common(3),
        "change_m2_per_s": {
            "least": float(sizes.min()),
            "median": float(np.median(sizes)),
            "largest": float(sizes.max()),
        },
    }


def bounded(_args: argparse.Namespace) -> None:
    """The bounded rows' characterisation, from bounded46.py's output files."""
    rows = []
    for path in sorted(OUT.glob("bounded_*.json")):
        if path.stem.startswith("bounded_where_"):
            continue
        b = json.loads(path.read_text())
        where_path = OUT / f"bounded_where_{b['name']}.json"
        where = json.loads(where_path.read_text()) if where_path.exists() else None
        rows.append(
            {"history": b, "where": where, "nu_where": viscosity_where(b["name"])}
        )
    headers = [
        "Run",
        "Outer, tail",
        "Residual: least, median, largest",
        "Amplitude p95/p5",
        "Drift (log10 per 1,000)",
        "Period (height); strongest (height)",
        "Largest speed: least, largest; cells",
        "Located: share by region (largest of u, v); cells",
        "Largest nu_t change: share by region; cells; size (m^2/s)",
    ]
    table = []
    for r in rows:
        h = r["history"]
        res, sp = h["residual"], h["max_speed"]
        located, nu_located = "-", "-"
        if r["where"]:
            both = r["where"]["largest_of_both"]
            located = (
                "; ".join(
                    f"{k} {v:.2f}" for k, v in list(both["share_by_region"].items())[:3]
                )
                + "; "
                + ", ".join(str(tuple(c[0])) for c in both["cells"][:3])
            )
        if r["nu_where"]:
            w = r["nu_where"]
            nu_located = (
                "; ".join(
                    f"{k} {v:.2f}" for k, v in list(w["share_by_region"].items())[:3]
                )
                + "; "
                + ", ".join(str(tuple(c[0])) for c in w["cells"][:3])
                + f"; {w['change_m2_per_s']['median']:.2g}"
            )
        table.append(
            [
                h["name"],
                f"{h['outer']:,}, {h['tail_iterations']:,}",
                f"{res['least']:.2e}, {res['median']:.2e}, {res['largest']:.2e}",
                fmt(res["amplitude_p95_over_p5"]),
                fmt(res["drift_log10_per_thousand"]),
                f"{fmt(res['acf_period'])} ({fmt(res['acf_height'], 2)}); "
                f"{fmt(res['acf_strongest'])} ({fmt(res['acf_strongest_height'], 2)})",
                f"{sp['least']:.3g}, {sp['largest']:.3g}; "
                + ", ".join(str(tuple(c[0])) for c in sp["cells"][:2]),
                located,
                nu_located,
            ]
        )
    print_table(headers, table)
    keep("bounded", rows)


def transport_records() -> dict[str, dict]:
    """Every transport record, keyed by the flow record's name."""
    out = {}
    for path in sorted(OUT.glob("transport_*.json")):
        rec = json.loads(path.read_text())
        out[rec["flow_record"]] = rec
    return out


def transport(_args: argparse.Namespace) -> None:
    """Measurement 4: the march, the sensors, the surfaces and the hotspots."""
    recs = transport_records()
    kept: dict = {"runs": {}, "surfaces": {}, "hotspots": {}}
    # The march and the sensors.
    headers = [
        "Run, class (um)",
        "Stop, t (s)",
        "Last window: total, sensor change; removal / Q",
        "Deposition / Q (faces; budget), outflow / Q",
        "Budget residual",
        "Sensors: near_door, above_gap_1, above_gap_2, hood_entry (per m^3)",
        "Sensor order",
    ]
    table = []
    for name, rec in recs.items():
        for k, c in rec["per_class"].items():
            um = c["diameter"] * 1e6
            lw = c["last_window"]
            sensors = c["sensors"]
            table.append(
                [
                    f"{name}, {um:g}",
                    f"{c['stop']}, {c['t_end']:.0f}",
                    f"{fmt(lw['total_change'], 2)}, {fmt(lw['sensor_change'], 2)}; "
                    f"{fmt(lw['removal_over_source'], 5)}",
                    f"{c['deposition_rate_over_source']:.4g} "
                    f"({c['budget_deposition_rate_over_source']:.4g}), "
                    f"{c['outflow_rate_over_source']:.4g}",
                    fmt(c["budget"]["relative_residual"], 2),
                    ", ".join(f"{sensors[s]:.4g}" for s in rec["sensors"]),
                    " > ".join(c["sensor_order"]),
                ]
            )
            kept["runs"][f"{name}/{k}"] = {
                key: c[key]
                for key in (
                    "stop",
                    "t_end",
                    "windows",
                    "last_window",
                    "sensors",
                    "sensor_order",
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
    # Deposition per surface, one column per run and class, as a share of
    # the deposition total and as a rate.
    columns = [(n, k) for n, r in recs.items() for k in r["per_class"]]
    surfaces: list[str] = []
    for n, k in columns:
        for s in recs[n]["per_class"][k]["by_surface"]:
            if s not in surfaces:
                surfaces.append(s)
    for label, key in (
        ("share of the deposition", "by_surface_share"),
        ("rate (per s per m depth)", "by_surface"),
    ):
        print(f"Deposition per surface, {label}:")
        headers = ["Surface"] + [
            f"{n}, {recs[n]['per_class'][k]['diameter'] * 1e6:g} um" for n, k in columns
        ]
        table = [
            [s] + [fmt(recs[n]["per_class"][k][key].get(s), 3) for n, k in columns]
            for s in surfaces
        ]
        print_table(headers, table)
        kept["surfaces"][key] = {
            f"{n}/{k}": recs[n]["per_class"][k][key] for n, k in columns
        }
    # The five hotspots per run and class.
    print(
        "The five segments of largest deposition (surface | bin start), rate per s per m depth:"
    )
    headers = ["Run, class (um)", "1", "2", "3", "4", "5"]
    table = []
    for n, k in columns:
        c = recs[n]["per_class"][k]
        table.append(
            [f"{n}, {c['diameter'] * 1e6:g}"]
            + [
                f"{s} ({v:.3g})"
                for s, v in zip(c["hotspots"], c["hotspot_rates"], strict=True)
            ]
        )
        kept["hotspots"][f"{n}/{k}"] = list(
            zip(c["hotspots"], c["hotspot_rates"], strict=True)
        )
    print_table(headers, table)
    keep("transport", kept)


def compare(_args: argparse.Namespace) -> None:
    """The comparative check: hotspots in common and the sensor orders, per pair and class."""
    recs = transport_records()
    rows = []
    headers = [
        "Pair, class (um)",
        "Hotspots in common (of 5)",
        "In common",
        "Only in the first",
        "Only in the second",
        "Sensor order, first",
        "Sensor order, second",
        "Orders agree",
    ]
    table = []
    for label, first, second in PAIRS:
        if first not in recs or second not in recs:
            rows.append(
                {"pair": label, "first": first, "second": second, "skipped": True}
            )
            table.append(
                [label, "skipped: a record is missing", "", "", "", "", "", ""]
            )
            continue
        for k in recs[first]["per_class"]:
            a, b = recs[first]["per_class"][k], recs[second]["per_class"][k]
            common = [s for s in a["hotspots"] if s in b["hotspots"]]
            only_a = [s for s in a["hotspots"] if s not in b["hotspots"]]
            only_b = [s for s in b["hotspots"] if s not in a["hotspots"]]
            agree = a["sensor_order"] == b["sensor_order"]
            rows.append(
                {
                    "pair": label,
                    "first": first,
                    "second": second,
                    "class": k,
                    "diameter": a["diameter"],
                    "common": common,
                    "only_first": only_a,
                    "only_second": only_b,
                    "order_first": a["sensor_order"],
                    "order_second": b["sensor_order"],
                    "orders_agree": agree,
                    "top_first": a["hotspots"][0],
                    "top_second": b["hotspots"][0],
                    "top_agree": a["hotspots"][0] == b["hotspots"][0],
                }
            )
            table.append(
                [
                    f"{label}, {a['diameter'] * 1e6:g}",
                    str(len(common)),
                    "; ".join(common) or "-",
                    "; ".join(only_a) or "-",
                    "; ".join(only_b) or "-",
                    " > ".join(a["sensor_order"]),
                    " > ".join(b["sensor_order"]),
                    "yes" if agree else "no",
                ]
            )
    print_table(headers, table)
    keep("compare", rows)


def rtol(_args: argparse.Namespace) -> None:
    """The 1e-8 check row against the 1e-4 row on 200x75: counts, CG work and the field difference."""
    pairs = [("standard_200x75", "standard_200x75_r1e-8")]
    rows = []
    for base, check in pairs:
        if not (OUT / f"{base}.json").exists() or not (OUT / f"{check}.json").exists():
            continue
        a, b = load(base), load(check)
        fa, fb = dict(np.load(OUT / f"{base}.npz")), dict(np.load(OUT / f"{check}.npz"))
        live = fa["cell_type"] != SOLID
        rows.append(
            {
                "base": base,
                "check": check,
                "outer": [a["outer"], b["outer"]],
                "stop": [a["stop"], b["stop"]],
                "inner_mean": [a["inner_mean"], b["inner_mean"]],
                "seconds": [a["seconds"], b["seconds"]],
                "faces_max_abs_du": float(np.abs(fb["u_faces"] - fa["u_faces"]).max()),
                "faces_max_abs_dv": float(np.abs(fb["v_faces"] - fa["v_faces"]).max()),
                "cells_max_speed_difference": float(
                    np.hypot(fb["u_c"] - fa["u_c"], fb["v_c"] - fa["v_c"])[live].max()
                ),
                "nu_t_max_abs_difference": float(
                    np.abs(fb["nu_t"] - fa["nu_t"])[live].max()
                ),
                "nu_t_max_relative_difference": float(
                    (
                        np.abs(fb["nu_t"][live] - fa["nu_t"][live]) / fa["nu_t"][live]
                    ).max()
                ),
                "k_max_relative_difference": float(
                    (np.abs(fb["k"][live] - fa["k"][live]) / fa["k"][live]).max()
                ),
                "face_hash": [a["face_hash"], b["face_hash"]],
            }
        )
    headers = [
        "Rows (1e-4, 1e-8)",
        "Outer",
        "Stop",
        "CG per correction (mean)",
        "Wall (s)",
        "Faces: max |du|, max |dv| (m/s)",
        "Cells: max speed difference (m/s)",
        "nu_t: max |difference| (m^2/s), max relative",
        "k: max relative difference",
    ]
    table = [
        [
            f"{r['base']}, {r['check']}",
            ", ".join(f"{x:,}" for x in r["outer"]),
            ", ".join(r["stop"]),
            ", ".join(f"{x:.0f}" for x in r["inner_mean"]),
            ", ".join(f"{x:.0f}" for x in r["seconds"]),
            f"{r['faces_max_abs_du']:.2e}, {r['faces_max_abs_dv']:.2e}",
            f"{r['cells_max_speed_difference']:.2e}",
            f"{r['nu_t_max_abs_difference']:.2e}, {r['nu_t_max_relative_difference']:.2e}",
            f"{r['k_max_relative_difference']:.2e}",
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("rtol", rows)


def everything(args: argparse.Namespace) -> None:
    """Every table, in the report's order."""
    for f in (matrix, converged_rows, inlet, rtol, bounded, transport, compare):
        print(f"### {f.__name__}")
        f(args)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, func in (
        ("matrix", matrix),
        ("converged", converged_rows),
        ("inlet", inlet),
        ("rtol", rtol),
        ("bounded", bounded),
        ("transport", transport),
        ("compare", compare),
        ("all", everything),
    ):
        sub.add_parser(name).set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
