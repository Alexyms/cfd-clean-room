"""The report's tables from the prompt 48 records.

Usage:
    python tables48.py parity      item 4: the key's rows against prompt 47's upwind arm
    python tables48.py matrix      item 5: every flow run, classified by step 5's rule
    python tables48.py converged   item 5: the converged rooms' readings, core and y+
    python tables48.py bounded     item 5: the bounded rows, if any (bounded47.py's output)
    python tables48.py transport   item 6: the marches, sensors, surfaces and hotspots
    python tables48.py marches     item 6: this build's standard marches against prompt 47's
    python tables48.py compare     item 6: between variants on each grid, and between grids
    python tables48.py all         everything above

Reads results/builder48/*.json (and, for the parity rows, prompt 47's
results/builder47/), prints markdown tables, and keeps every number it
printed in results/builder48/tables_MODE.json. Every table but `parity`,
`marches` and the pairs of `compare` is prompt 47's (tables47.py,
imported, its record directory redirected); the classification is
tables46's.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe47"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))

import tables46  # noqa: E402
import tables47  # noqa: E402
from tables46 import fmt, print_table  # noqa: E402

OUT = ROOT / "results" / "builder48"
PROMPT47 = ROOT / "results" / "builder47"
tables46.OUT = OUT
tables47.OUT = OUT
GRIDS = ("80x30", "160x60", "320x120")
# Prompt 47's upwind rows (its report, section 5.2 and 5.3) that the key's
# default rows are expected to reproduce bit for bit, with their counts.
PARITY = {
    "standard_80x30_upwind": 1046,
    "standard_160x60_upwind": 2003,
    "standard_320x120_upwind": 4472,
}
FIELDS = ("u_faces", "v_faces", "k", "eps", "nu_t", "p")


def keep(mode: str, data: object) -> None:
    """The printed numbers, as JSON beside the records."""
    (OUT / f"tables_{mode}.json").write_text(json.dumps(data, indent=1, default=float))


def parity(_args: argparse.Namespace) -> None:
    """Item 4: stop, count, face hash and every saved field against prompt 47's upwind arm."""
    rows = []
    for name, expected in PARITY.items():
        path = OUT / f"{name}.json"
        if not path.exists():
            rows.append({"name": name, "present": False})
            continue
        built = json.loads(path.read_text())
        base = json.loads((PROMPT47 / f"{name}.json").read_text())
        a = np.load(OUT / f"{name}.npz")
        b = np.load(PROMPT47 / f"{name}.npz")
        fields_equal = {f: bool(np.array_equal(a[f], b[f])) for f in FIELDS}
        residuals_equal = built["residual"] == base["residual"]
        rows.append(
            {
                "name": name,
                "present": True,
                "advection_scheme": built.get("advection_scheme"),
                "key_stated": built.get("scheme_key_stated"),
                "arm_47": base.get("arm"),
                "stop": (built["stop"], base["stop"]),
                "outer": (built["outer"], base["outer"]),
                "outer_expected": expected,
                "face_hash": (built["face_hash"], base["face_hash"]),
                "fields_equal": fields_equal,
                "residual_history_equal": residuals_equal,
                "reproduces": built["stop"] == base["stop"]
                and built["outer"] == base["outer"] == expected
                and built["face_hash"] == base["face_hash"]
                and all(fields_equal.values())
                and residuals_equal,
                "commit": built["commit"],
            }
        )
    headers = [
        "Row (key absent)",
        "Scheme parsed",
        "Stop: this build, prompt 47",
        "Outer: this build, prompt 47 (expected)",
        "Face hash: this build, prompt 47",
        "u, v, k, eps, nu_t, p and the residual history bitwise equal",
        "Reproduces",
    ]
    table = []
    for r in rows:
        if not r["present"]:
            table.append([r["name"], "-", "not run", "-", "-", "-", "-"])
            continue
        table.append(
            [
                r["name"],
                str(r["advection_scheme"]),
                f"{r['stop'][0]}, {r['stop'][1]}",
                f"{r['outer'][0]:,}, {r['outer'][1]:,} ({r['outer_expected']:,})",
                f"{r['face_hash'][0][:16]}, {r['face_hash'][1][:16]}",
                "yes"
                if all(r["fields_equal"].values()) and r["residual_history_equal"]
                else "no: "
                + ", ".join(f for f, eq in r["fields_equal"].items() if not eq)
                + ("" if r["residual_history_equal"] else ", residuals"),
                "yes" if r["reproduces"] else "no",
            ]
        )
    print_table(headers, table)
    keep("parity", rows)


def marches(_args: argparse.Namespace) -> None:
    """This build's standard marches against prompt 47's on the same rooms."""
    rows = []
    for label in sorted(tables47.transport_records()):
        if not label.startswith("transport_standard_"):
            continue
        base_path = PROMPT47 / f"{label}.json"
        if not base_path.exists():
            continue
        built = json.loads((OUT / f"{label}.json").read_text())
        base = json.loads(base_path.read_text())
        a = np.load(OUT / f"{label}.npz")
        b = np.load(PROMPT47 / f"{label}.npz")
        arrays_equal = sorted(a.files) == sorted(b.files) and all(
            np.array_equal(a[f], b[f]) for f in a.files
        )
        for k, c in built["per_class"].items():
            d = base["per_class"][k]
            rows.append(
                {
                    "label": label,
                    "class": k,
                    "diameter": c["diameter"],
                    "t_end": (c["t_end"], d["t_end"]),
                    "hotspots_equal": c["hotspots"] == d["hotspots"],
                    "rates_equal": c["hotspot_rates"] == d["hotspot_rates"],
                    "sensors_equal": c["sensors"] == d["sensors"],
                    "by_surface_equal": c["by_surface"] == d["by_surface"],
                    "arrays_equal": arrays_equal,
                }
            )
    headers = [
        "March, class (um)",
        "Steady at (s): this build, prompt 47",
        "Hotspots and their rates equal",
        "Sensors equal",
        "Per-surface deposition equal",
        "Saved fields bitwise equal",
    ]
    table = [
        [
            f"{r['label'][len('transport_') :]}, {r['diameter'] * 1e6:g}",
            f"{r['t_end'][0]:.0f}, {r['t_end'][1]:.0f}",
            "yes" if r["hotspots_equal"] and r["rates_equal"] else "no",
            "yes" if r["sensors_equal"] else "no",
            "yes" if r["by_surface_equal"] else "no",
            "yes" if r["arrays_equal"] else "no",
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("marches", rows)


def pairs48(recs: dict[str, dict]) -> list[tuple[str, str, str, str]]:
    """Item 6's comparisons: the variants on each exact grid, then each variant between grids."""
    by = {(r["_flow"], r["_source"]): label for label, r in recs.items()}
    sources = sorted({r["_source"] for r in recs.values()})
    out = []
    for source in sources:
        for grid in ("160x60", "320x120"):
            a = by.get((f"standard_{grid}_upwind", source))
            b = by.get((f"rng_{grid}_upwind", source))
            if a and b:
                out.append((f"variants, {grid}", source, a, b))
        for variant in ("rng", "standard"):
            a = by.get((f"{variant}_160x60_upwind", source))
            b = by.get((f"{variant}_320x120_upwind", source))
            if a and b:
                out.append((f"grids, {variant}", source, a, b))
    return out


def compare(args: argparse.Namespace) -> None:
    """tables47's comparison on item 6's pairs, and the per-pair summary the predictions score."""
    tables47.pairs = pairs48
    tables47.compare(args)
    rows = json.loads((OUT / "tables_compare.json").read_text())
    summary: dict[str, dict] = {}
    for r in rows:
        s = summary.setdefault(
            r["pair"],
            {
                "rows": 0,
                "same_largest_segment": 0,
                "four_or_more": 0,
                "orders_equal": 0,
            },
        )
        s["rows"] += 1
        s["same_largest_segment"] += int(r["top_same_segment"])
        s["four_or_more"] += int(r["common_count"] >= 4)
        s["orders_equal"] += int(r["orders_equal"])
        s.setdefault("least_common", r["common_count"])
        s["least_common"] = min(s["least_common"], r["common_count"])
    print("Per pair, over its sources and classes:")
    headers = [
        "Pair",
        "Rows (source, class)",
        "Largest the same segment",
        "At least four of five shared",
        "Fewest shared",
        "Sensor orders equal",
    ]
    table = [
        [
            pair,
            fmt(s["rows"]),
            fmt(s["same_largest_segment"]),
            fmt(s["four_or_more"]),
            fmt(s["least_common"]),
            fmt(s["orders_equal"]),
        ]
        for pair, s in summary.items()
    ]
    print_table(headers, table)
    keep("compare_summary", summary)


def everything(args: argparse.Namespace) -> None:
    """Every table, in the report's order."""
    for name, f in (
        ("parity", parity),
        ("matrix", tables47.matrix),
        ("converged", tables47.converged_rows),
        ("bounded", tables47.bounded),
        ("transport", tables47.transport),
        ("marches", marches),
        ("compare", compare),
    ):
        print(f"### {name}")
        f(args)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, func in (
        ("parity", parity),
        ("matrix", tables47.matrix),
        ("converged", tables47.converged_rows),
        ("bounded", tables47.bounded),
        ("transport", tables47.transport),
        ("marches", marches),
        ("compare", compare),
        ("all", everything),
    ):
        sub.add_parser(name).set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
