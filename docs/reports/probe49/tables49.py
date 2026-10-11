"""The report's tables from the prompt 49 records.

Usage:
    python tables49.py matrix       item 3: every flow run, classified by step 5's rule
    python tables49.py converged    item 3: the converged rooms' readings, nu_t and y+
    python tables49.py lines        item 4: each converged run on each line, against the
                                    data and against the published prediction
    python tables49.py published    item 1: the published prediction's own departure
    python tables49.py schemes      item 4: upwind against umist on each grid, per line
    python tables49.py grids        item 4: 216x72 against 375x125 under each scheme
    python tables49.py predictions  section 6: (a) to (e) scored from the records
    python tables49.py all          everything above

Reads results/builder49/NAME.json, NAME.npz and lines_NAME.json (annex49.py),
prints markdown tables, and keeps every number it printed in
results/builder49/tables_MODE.json. The classification is tables46's
(imported): diverged, converged, growing, bounded with step 0's sub-classes,
or a positivity error.
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
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))

import tables46  # noqa: E402
from tables46 import classify, fmt, print_table  # noqa: E402

from validation import annex20  # noqa: E402

OUT = ROOT / "results" / "builder49"
tables46.OUT = OUT
GRIDS = ("216x72", "375x125")
SCHEMES = ("upwind", "umist")
VARIANTS = ("standard", "rng")
NAME = re.compile(
    r"^(?P<variant>standard|rng)_(?P<grid>\d+x\d+)_(?P<scheme>upwind|umist)$"
)
CONVERGED = "error_estimate_and_continuity"
JET_LIMIT = 2 * annex20.HEIGHT  # x/H 2, prediction (b)
JET_TOL = 0.1  # u0, prediction (b)
SCHEME_RATIO = 0.5  # prediction (d)


def keep(mode: str, data: object) -> None:
    """The printed numbers, as JSON beside the records."""
    (OUT / f"tables_{mode}.json").write_text(json.dumps(data, indent=1, default=float))


def records() -> dict[str, dict]:
    """Every flow record, keyed by name, in grid, variant, scheme order."""
    found = {}
    for path in sorted(OUT.glob("*.json")):
        m = NAME.match(path.stem)
        if not m:
            continue
        rec = json.loads(path.read_text())
        if "stop" not in rec:
            continue
        rec["_grid"], rec["_variant"], rec["_scheme"] = (
            m["grid"],
            m["variant"],
            m["scheme"],
        )
        found[path.stem] = rec
    return dict(
        sorted(
            found.items(),
            key=lambda kv: (
                GRIDS.index(kv[1]["_grid"]),
                VARIANTS.index(kv[1]["_variant"]),
                SCHEMES.index(kv[1]["_scheme"]),
            ),
        )
    )


def lines_records() -> dict[str, dict]:
    """The line samples of every converged record that has them."""
    out = {}
    for name, rec in records().items():
        path = OUT / f"lines_{name}.json"
        if rec["stop"] == CONVERGED and path.exists():
            out[name] = json.loads(path.read_text())
    return out


def readings_cell(rec: dict) -> str:
    r = rec.get("readings_at_stop") or {}
    return ", ".join(fmt(r.get(k), 2) for k in ("a", "b", "c", "d", "e"))


def from_when_cell(rec: dict) -> str:
    f = rec.get("from_when") or {}
    return ", ".join(fmt(f.get(k)) for k in ("a", "b", "c", "d", "e"))


def matrix(_args: argparse.Namespace) -> None:
    """Item 3: every flow run, one line each."""
    rows = []
    for name, rec in records().items():
        cls, sub = classify(rec)
        rows.append(
            {
                "name": name,
                "grid": rec["_grid"],
                "variant": rec["_variant"],
                "advection_scheme": rec["_scheme"],
                "scheme_key_stated": rec.get("scheme_key_stated"),
                "inlet_velocity": rec.get("inlet_velocity"),
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


def wall_y_plus(name: str, side: str, lo: float, hi: float) -> dict:
    """The y* of one wall's nodes between two x positions, from the record's fields."""
    arrays = np.load(OUT / f"{name}.npz")
    field = arrays[f"y_star_{side}"]
    xc = arrays["xc"]
    row = field[-1, :] if side == "north" else field[0, :]
    sel = row[(xc >= lo) & (xc <= hi) & np.isfinite(row)]
    return {
        "nodes": int(sel.size),
        "median": float(np.median(sel)),
        "least": float(sel.min()),
        "largest": float(sel.max()),
    }


def converged_rows(_args: argparse.Namespace) -> None:
    """Item 3: the converged rooms' readings, eddy viscosity and y+."""
    rows = []
    for name, rec in records().items():
        if rec["stop"] != CONVERGED or "core" not in rec:
            continue
        y = rec["y_star"]
        rows.append(
            {
                "name": name,
                "outer": rec["outer"],
                "seconds": rec["seconds"],
                "readings_at_stop": rec["readings_at_stop"],
                "from_when": rec["from_when"],
                "inlet_nu_t_over_nu": (rec["nu_scale"] - rec["nu_air"]) / rec["nu_air"],
                "room_nu_t_over_nu_median": rec["core"]["nu_t_over_nu_median"],
                "room_nu_t_over_nu_p95": rec["core"]["nu_t_over_nu_p95"],
                "room_nu_t_over_nu_max": rec["core"]["nu_t_over_nu_all_max"],
                "y_star": {k: v for k, v in y.items() if k != "by_kind"},
                "ceiling_jet": wall_y_plus(name, "north", annex20.HEIGHT, JET_LIMIT),
                "floor_x1_to_2H": wall_y_plus(name, "south", annex20.HEIGHT, JET_LIMIT),
                "figure": f"ecr002_val017_{name}.png",
            }
        )
    headers = [
        "Run",
        "Outer, wall (s)",
        "Readings at stop (a), (b), (c), (d), (e)",
        "Holds from (a), (b), (c), (d), (e)",
        "Inlet nu_t / nu",
        "Room nu_t / nu: median, 95th, largest",
        "y+ nodes; median, least, largest; share below 11.53",
        "Ceiling y+ for H < x < 2H: median (least, largest)",
        "Floor y+ for H < x < 2H: median (least, largest)",
        "Figure",
    ]
    table = []
    for r in rows:
        y, c, f = r["y_star"], r["ceiling_jet"], r["floor_x1_to_2H"]
        table.append(
            [
                r["name"],
                f"{r['outer']:,}, {r['seconds']:.0f}",
                readings_cell(r),
                from_when_cell(r),
                fmt(r["inlet_nu_t_over_nu"]),
                f"{r['room_nu_t_over_nu_median']:.3g}, {r['room_nu_t_over_nu_p95']:.3g}, "
                f"{r['room_nu_t_over_nu_max']:.3g}",
                f"{y['nodes']:,}; {y['median']:.3g}, {y['least']:.3g}, {y['largest']:.3g}; "
                f"{y['share_below_floor']:.2g}",
                f"{c['median']:.3g} ({c['least']:.3g}, {c['largest']:.3g})",
                f"{f['median']:.3g} ({f['least']:.3g}, {f['largest']:.3g})",
                r["figure"],
            ]
        )
    print_table(headers, table)
    keep("converged", rows)


def published(_args: argparse.Namespace) -> None:
    """Item 1: Rong and Nielsen's standard k-epsilon departure from the data, per line."""
    rows = []
    for key, line in annex20.LINES.items():
        held = [i for i, p in enumerate(line.published) if p is not None]
        dep = np.array([line.published[i] - line.measured[i] for i in held])
        rows.append(
            {
                "line": key,
                "label": line.label,
                "stations": len(line.stations),
                "held": len(held),
                "max_abs": float(np.abs(dep).max()),
                "rms": float(np.sqrt(np.mean(dep**2))),
                "mean": float(dep.mean()),
                "max_at": float(line.stations[held[int(np.argmax(np.abs(dep)))]]),
            }
        )
    headers = [
        "Line",
        "Stations (held by the published line)",
        "Published minus measured: largest |d| (at, m)",
        "RMS",
        "Mean",
    ]
    table = [
        [
            r["label"],
            f"{r['stations']} ({r['held']})",
            f"{r['max_abs']:.3f} ({r['max_at']:.2f})",
            f"{r['rms']:.3f}",
            f"{r['mean']:+.3f}",
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("published", rows)


def lines(_args: argparse.Namespace) -> None:
    """Item 4: each converged run on each line, against the data and the published line."""
    rows = []
    for name, rec in lines_records().items():
        for key, line in rec["lines"].items():
            values = np.array(line["u_over_u0"])
            meas = np.array(annex20.LINES[key].measured)
            d = values - meas
            rows.append(
                {
                    "name": name,
                    "line": key,
                    "label": annex20.LINES[key].label,
                    "against_data": line["against_data"],
                    "against_published": line["against_published"],
                    "data_max_at": float(
                        annex20.LINES[key].stations[int(np.argmax(np.abs(d)))]
                    ),
                    "mean_minus_data": float(d.mean()),
                    "u_over_u0": line["u_over_u0"],
                }
            )
    headers = [
        "Run",
        "Line",
        "Against the data: largest |d| (at, m)",
        "RMS",
        "Mean",
        "Against the published line: largest |d|",
        "RMS",
        "Stations",
    ]
    table = [
        [
            r["name"],
            r["label"],
            f"{r['against_data']['max_abs']:.3f} ({r['data_max_at']:.2f})",
            f"{r['against_data']['rms']:.3f}",
            f"{r['mean_minus_data']:+.3f}",
            f"{r['against_published']['max_abs']:.3f}",
            f"{r['against_published']['rms']:.3f}",
            f"{r['against_data']['stations']} ({r['against_published']['stations']})",
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("lines", rows)


def pair_rows(pairs: list[tuple[str, str, str]], recs: dict[str, dict]) -> list[dict]:
    """The difference of two runs on each line, beside each run's distance from the data."""
    rows = []
    for label, a, b in pairs:
        if a not in recs or b not in recs:
            rows.append({"pair": label, "a": a, "b": b, "present": False})
            continue
        for key in annex20.LINES:
            va = np.array(recs[a]["lines"][key]["u_over_u0"])
            vb = np.array(recs[b]["lines"][key]["u_over_u0"])
            between = annex20.line_errors(va, vb)
            rows.append(
                {
                    "pair": label,
                    "a": a,
                    "b": b,
                    "present": True,
                    "line": key,
                    "label": annex20.LINES[key].label,
                    "between_max": between.max_abs,
                    "between_rms": between.rms,
                    "a_vs_data_max": recs[a]["lines"][key]["against_data"]["max_abs"],
                    "b_vs_data_max": recs[b]["lines"][key]["against_data"]["max_abs"],
                    "a_vs_data_rms": recs[a]["lines"][key]["against_data"]["rms"],
                    "b_vs_data_rms": recs[b]["lines"][key]["against_data"]["rms"],
                }
            )
    return rows


def print_pairs(rows: list[dict], what: tuple[str, str]) -> None:
    headers = [
        "Pair",
        "Line",
        f"Between {what[0]} and {what[1]}: largest |d|",
        "RMS",
        f"{what[0]} against the data: largest |d|, RMS",
        f"{what[1]} against the data: largest |d|, RMS",
        "Ratio: between over the larger against the data",
    ]
    table = []
    for r in rows:
        if not r["present"]:
            table.append([r["pair"], "not run", "-", "-", "-", "-", "-"])
            continue
        larger = max(r["a_vs_data_max"], r["b_vs_data_max"])
        table.append(
            [
                r["pair"],
                r["label"],
                f"{r['between_max']:.3f}",
                f"{r['between_rms']:.3f}",
                f"{r['a_vs_data_max']:.3f}, {r['a_vs_data_rms']:.3f}",
                f"{r['b_vs_data_max']:.3f}, {r['b_vs_data_rms']:.3f}",
                f"{r['between_max'] / larger:.2f}",
            ]
        )
    print_table(headers, table)


def schemes(_args: argparse.Namespace) -> None:
    """Item 4: upwind against umist on each grid."""
    recs = lines_records()
    pairs = [
        (f"schemes, {g}", f"standard_{g}_upwind", f"standard_{g}_umist") for g in GRIDS
    ]
    rows = pair_rows(pairs, recs)
    print_pairs(rows, ("upwind", "umist"))
    keep("schemes", rows)


def grids(_args: argparse.Namespace) -> None:
    """Item 4: 216x72 against 375x125 under each scheme."""
    recs = lines_records()
    pairs = [
        (f"grids, {s}", f"standard_{GRIDS[0]}_{s}", f"standard_{GRIDS[1]}_{s}")
        for s in SCHEMES
    ]
    rows = pair_rows(pairs, recs)
    print_pairs(rows, (GRIDS[0], GRIDS[1]))
    keep("grids", rows)


def predictions(_args: argparse.Namespace) -> None:
    """Section 6: the orchestrator's (a) to (e), scored from the records."""
    recs = records()
    lrecs = lines_records()
    standard = [f"standard_{g}_{s}" for g in GRIDS for s in SCHEMES]
    out: dict[str, dict] = {}
    # (a) both schemes converge on both grids.
    stops = {n: recs[n]["stop"] if n in recs else "not run" for n in standard}
    out["a"] = {"stops": stops, "holds": all(s == CONVERGED for s in stops.values())}
    # (b) the jet line within 0.1 u0 up to x/H 2, per converged run.
    jet = annex20.LINES["y2916"]
    inside = np.array(jet.stations) <= JET_LIMIT
    b_rows = {}
    for n, lr in lrecs.items():
        d = np.array(lr["lines"]["y2916"]["u_over_u0"]) - np.array(jet.measured)
        worst = float(np.abs(d[inside]).max())
        b_rows[n] = {
            "largest_abs_diff_to_2H": worst,
            "at": float(
                np.array(jet.stations)[inside][int(np.argmax(np.abs(d[inside])))]
            ),
            "within": worst <= JET_TOL,
        }
    out["b"] = {
        "runs": b_rows,
        "holds": bool(b_rows) and all(r["within"] for r in b_rows.values()),
    }
    # (c) the floor line carries the right sign everywhere and departs more than the jet.
    floor = annex20.LINES["y0084"]
    c_rows = {}
    for n, lr in lrecs.items():
        v = np.array(lr["lines"]["y0084"]["u_over_u0"])
        m = np.array(floor.measured)
        signs = int(np.sum(np.sign(v) == np.sign(m)))
        c_rows[n] = {
            "stations": int(m.size),
            "same_sign": signs,
            "floor_max": lr["lines"]["y0084"]["against_data"]["max_abs"],
            "jet_max": lr["lines"]["y2916"]["against_data"]["max_abs"],
            "floor_rms": lr["lines"]["y0084"]["against_data"]["rms"],
            "jet_rms": lr["lines"]["y2916"]["against_data"]["rms"],
            "holds": signs == m.size
            and lr["lines"]["y0084"]["against_data"]["max_abs"]
            > lr["lines"]["y2916"]["against_data"]["max_abs"],
        }
    out["c"] = {
        "runs": c_rows,
        "holds": bool(c_rows) and all(r["holds"] for r in c_rows.values()),
    }
    # (d) on every line the schemes differ by under half the larger distance from the data.
    d_rows = []
    for r in pair_rows(
        [
            (f"schemes, {g}", f"standard_{g}_upwind", f"standard_{g}_umist")
            for g in GRIDS
        ],
        lrecs,
    ):
        if r["present"]:
            larger = max(r["a_vs_data_max"], r["b_vs_data_max"])
            d_rows.append(
                {
                    "pair": r["pair"],
                    "line": r["line"],
                    "between_max": r["between_max"],
                    "larger_vs_data_max": larger,
                    "ratio": r["between_max"] / larger,
                    "holds": r["between_max"] < SCHEME_RATIO * larger,
                }
            )
    out["d"] = {
        "rows": d_rows,
        "holds": bool(d_rows) and all(r["holds"] for r in d_rows),
    }
    # (e) the grids differ by less than either differs from the data.
    e_rows = []
    for r in pair_rows(
        [
            (f"grids, {s}", f"standard_{GRIDS[0]}_{s}", f"standard_{GRIDS[1]}_{s}")
            for s in SCHEMES
        ],
        lrecs,
    ):
        if r["present"]:
            smaller = min(r["a_vs_data_max"], r["b_vs_data_max"])
            e_rows.append(
                {
                    "pair": r["pair"],
                    "line": r["line"],
                    "between_max": r["between_max"],
                    "smaller_vs_data_max": smaller,
                    "holds": r["between_max"] < smaller,
                }
            )
    out["e"] = {
        "rows": e_rows,
        "holds": bool(e_rows) and all(r["holds"] for r in e_rows),
    }

    headers = ["Prediction", "Measured", "Holds"]
    table = [
        [
            "(a) both schemes converge on both grids",
            "; ".join(f"{n}: {s}" for n, s in stops.items()),
            "yes" if out["a"]["holds"] else "no",
        ],
        [
            "(b) the jet within 0.1 u0 up to x/H 2",
            "; ".join(
                f"{n}: {r['largest_abs_diff_to_2H']:.3f} at {r['at']:.2f} m"
                for n, r in b_rows.items()
            )
            or "no converged run",
            "yes" if out["b"]["holds"] else "no",
        ],
        [
            "(c) the floor's sign right everywhere, its departure above the jet's",
            "; ".join(
                f"{n}: {r['same_sign']} of {r['stations']} signs, floor {r['floor_max']:.3f} "
                f"against jet {r['jet_max']:.3f}"
                for n, r in c_rows.items()
            )
            or "no converged run",
            "yes" if out["c"]["holds"] else "no",
        ],
        [
            "(d) schemes differ by under half the larger distance from the data, every line",
            "; ".join(
                f"{r['pair']} {r['line']}: {r['between_max']:.3f} against "
                f"{SCHEME_RATIO:g} x {r['larger_vs_data_max']:.3f}"
                for r in d_rows
            )
            or "no pair",
            "yes" if out["d"]["holds"] else "no",
        ],
        [
            "(e) grids differ by less than either differs from the data, every line",
            "; ".join(
                f"{r['pair']} {r['line']}: {r['between_max']:.3f} against {r['smaller_vs_data_max']:.3f}"
                for r in e_rows
            )
            or "no pair",
            "yes" if out["e"]["holds"] else "no",
        ],
    ]
    print_table(headers, table)
    keep("predictions", out)


def everything(args: argparse.Namespace) -> None:
    """Every table, in the report's order."""
    for name, f in MODES:
        print(f"### {name}")
        f(args)


MODES = (
    ("matrix", matrix),
    ("converged", converged_rows),
    ("published", published),
    ("lines", lines),
    ("schemes", schemes),
    ("grids", grids),
    ("predictions", predictions),
)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, func in (*MODES, ("all", everything)):
        sub.add_parser(name).set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
