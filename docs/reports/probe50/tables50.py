"""Every table of docs/reports/ecr002_rule_e_scale.md, generated from the records.

Usage:
    python tables50.py            print every table and write results/builder50/tables_all.md
    python tables50.py --splice   also replace each table's block in the report
    python tables50.py --check    exit 1 unless the report's blocks are the generated ones

A block in the report is the text between ``<!-- tables50 NAME begin -->``
and ``<!-- tables50 NAME end -->``. The records are results/builder50/:
report_base_built.json (check50.py), plants.json (plant50.py),
annex_TAG.json (annex50.py), compare_GRID.json and standard_GRID_upwind.json
(product50.py), and the harness rows of val017_216x72 in
benchmarks/results.jsonl.
"""

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "results" / "builder50"
REPORT = ROOT / "docs" / "reports" / "ecr002_rule_e_scale.md"
RESULTS = ROOT / "benchmarks" / "results.jsonl"
ANNEX_TAGS = ("file", "tol1e-13")
GRIDS = ("80x30", "160x60")
LINE_LABELS = {
    "x3": "x = H (3.0 m)",
    "x6": "x = 2H (6.0 m)",
    "y2916": "y = 2.916 m, the ceiling jet",
    "y0084": "y = 0.084 m, the floor",
}


def load(name: str) -> dict:
    """One JSON record under results/builder50/."""
    return json.loads((OUT / name).read_text())


def g(x: float | None, digits: int = 2) -> str:
    """A number in short scientific or plain form, '-' for None or inf."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{digits}g}"


def n(x: int | None) -> str:
    """An integer with thousands separators, '-' for None."""
    return "-" if x is None else f"{x:,}"


def table(header: list[str], rows: list[list[str]]) -> str:
    """A Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def laminar() -> str:
    """Item 3: each laminar case on main and on commit A."""
    rows = []
    for case in json.loads((OUT / "report_base_built.json").read_text()):
        rows.append(
            [
                case["case"],
                " / ".join(case["stop"]),
                " / ".join(n(x) for x in case["outer"]),
                " / ".join(case["face_hash"]),
                "yes" if case["equal"] else "NO",
            ]
        )
    return table(["Case", "Stop: main / commit A", "Outer", "Face hash", "Equal"], rows)


def plants() -> str:
    """The planted defects of commits A and C and the tests that caught them."""
    record = load("plants.json")
    rows = [["(none, the control)", str(record["control"]["exit"]), "-", "-"]]
    for label, r in record.items():
        if label == "control":
            continue
        caught = r.get("failed", [])
        names = sorted({f.split("::")[-1].split("[")[0] for f in caught})
        rows.append(
            [
                label,
                str(r.get("exit", "-")),
                "yes" if r.get("caught") else "NO",
                f"{len(caught)}: " + ", ".join(names),
            ]
        )
    return table(["Plant", "pytest exit", "Caught", "Failing tests"], rows)


def annex_runs() -> str:
    """Item 6: each Annex 20 run's stop, readings and counts."""
    rows = []
    for tag in ANNEX_TAGS:
        r = load(f"annex_{tag}.json")
        row, rule = r["row"], r["rule"]
        nu = r["nu_air"]
        read = rule["readings_at_stop"]
        holds = rule["from_when"]
        scale = rule["nu_scale"][-1]
        rows.append(
            [
                tag,
                g(r["turbulence_tol"]),
                n(r["cap"]),
                row["outcome"]["stop_reason"],
                n(row["work"]["outer_iterations"]),
                str(row["params"]["rule_version"]),
                ", ".join(g(read.get(k)) for k in "abcde"),
                ", ".join(n(holds.get(k)) for k in "abcde"),
                f"{g(scale / nu, 3)}; {g(r['nu_t_over_nu']['fluid_median'], 3)}, "
                f"{g(r['nu_t_over_nu']['fluid_max'], 3)}",
                f"{r['k_sweeps_mean']:.1f}, {r['eps_sweeps_mean']:.1f}; "
                f"{n(r['scalar_cap_hits'])}",
                str(row["outcome"]["pressure_cap_hits"]),
                f"{row['time']['wall_seconds']:.0f}",
                r["face_hash"][:16],
            ]
        )
    return table(
        [
            "Run",
            "k, eps tol",
            "Cap",
            "Stop",
            "Outer",
            "Rule version",
            "Readings at stop (a), (b), (c), (d), (e)",
            "Holds from (a), (b), (c), (d), (e)",
            "(e)'s scale at stop / nu; FLUID nu_t / nu: median, largest",
            "k, eps sweeps (mean); outer iterations with a capped k or eps solve",
            "Pressure cap hits",
            "Wall (s)",
            "Face hash",
        ],
        rows,
    )


def annex_lines() -> str:
    """Item 6: each line's RMS beside the allowance, and the distance from prompt 49."""
    rows = []
    for tag in ANNEX_TAGS:
        r = load(f"annex_{tag}.json")
        acc = r["row"]["accuracy"]
        for key, label in LINE_LABELS.items():
            line = acc["lines"][key]
            verdict = (
                "-" if line["pass"] is None else ("pass" if line["pass"] else "FAIL")
            )
            rows.append(
                [
                    tag,
                    label,
                    f"{line['rms']:.4f} ({line['stations']})",
                    f"{line['allowance']:.4f}",
                    "scored" if line["scored"] else "reported",
                    verdict,
                    g(r["against_49"]["tol1e-13_cap12000"][key]),
                    g(r["against_49"]["file_cap10000"][key]),
                ]
            )
        rows.append(
            [
                tag,
                "the row",
                f"value {acc['value']:.4f}",
                "1",
                "-",
                "pass" if acc["passed"] else "FAIL",
                "-",
                "-",
            ]
        )
    return table(
        [
            "Run",
            "Line",
            "RMS against the data, u / u0 (stations)",
            "Allowance (published RMS)",
            "Scored",
            "Verdict",
            "Largest difference from prompt 49's 1e-13 record (12,000 cap), u0",
            "From prompt 49's file row (10,000 cap), u0",
        ],
        rows,
    )


def harness() -> str:
    """The val017_216x72 rows stored in benchmarks/results.jsonl."""
    rows = []
    for text in RESULTS.read_text(encoding="utf-8").splitlines():
        rec = json.loads(text)
        if rec["case"] != "val017_216x72":
            continue
        acc = rec["accuracy"]
        lines = acc["lines"]
        rows.append(
            [
                rec["git_commit"][:7],
                "yes" if rec["git_dirty"] else "no",
                rec["outcome"]["stop_reason"],
                n(rec["work"]["outer_iterations"]),
                str(rec["params"]["rule_version"]),
                f"{g(rec['params']['turbulence']['tol'])}, "
                f"{n(rec['params']['max_simple_iter'])}",
                ", ".join(
                    f"{lines[k]['rms']:.4f} / {lines[k]['allowance']:.4f}"
                    for k in ("x3", "y2916")
                ),
                f"{acc['value']:.4f}",
                "pass" if acc["passed"] else "FAIL",
                str(rec["outcome"]["pressure_cap_hits"]),
                f"{rec['time']['wall_seconds']:.0f}",
                str(rec["environment"]["concurrent_processes"]),
            ]
        )
    return table(
        [
            "Commit",
            "Dirty",
            "Stop",
            "Outer",
            "Rule version",
            "k, eps tol; max_simple_iter",
            "RMS / allowance: x = H, jet",
            "Value",
            "Verdict",
            "Pressure cap hits",
            "Wall (s)",
            "Processes",
        ],
        rows,
    )


def product() -> str:
    """Item 7: each product row against prompt 48's."""
    rows = []
    for grid in GRIDS:
        c = load(f"compare_{grid}.json")
        rows.append(
            [
                grid,
                " / ".join(c["stop"]),
                " / ".join(n(x) for x in c["outer"]),
                f"{100 * c['outer_change']:+.1f}%",
                " / ".join(str(x) for x in c["rule_version"]),
                " / ".join(g(x, 3) for x in c["nu_scale_over_nu"]),
                ", ".join(n(c["from_when"][1].get(k)) for k in "abcde"),
                g(c["max_face_difference_over_scale"]),
                " / ".join(str(x) for x in c["cap_hits"]),
                " / ".join(c["face_hash"]),
            ]
        )
    return table(
        [
            "Grid",
            "Stop: prompt 48 / now",
            "Outer",
            "Change",
            "Rule version",
            "(e)'s scale at stop / nu",
            "Now, holds from (a), (b), (c), (d), (e)",
            "Largest face difference / velocity scale",
            "Pressure cap hits",
            "Face hash",
        ],
        rows,
    )


def predictions() -> str:
    """The orchestrator's predictions against the records."""
    file, tight = load("annex_file.json"), load("annex_tol1e-13.json")
    compares = [load(f"compare_{grid}.json") for grid in GRIDS]

    def stopped(r: dict) -> bool:
        return r["row"]["outcome"]["converged"]

    outer_tight = tight["row"]["work"]["outer_iterations"]
    acc = tight["row"]["accuracy"]
    worst_49 = max(tight["against_49"]["tol1e-13_cap12000"].values())
    rows = [
        [
            "(a) at 1e-10, 216x72 does not stop",
            f"{file['row']['outcome']['stop_reason']} at "
            f"{n(file['row']['work']['outer_iterations'])}",
            "yes" if not stopped(file) else "no",
        ],
        [
            "(b) at 1e-13 it stops between 11,200 and 12,500",
            f"{tight['row']['outcome']['stop_reason']} at {n(outer_tight)}",
            "yes" if stopped(tight) and 11200 <= outer_tight <= 12500 else "no",
        ],
        [
            "(c) lines within 1e-4 u0 of prompt 49's record; both scored lines pass",
            f"largest line difference from the 1e-13 record {g(worst_49)}; "
            + ", ".join(
                f"{k} {acc['lines'][k]['rms']:.3f} against {acc['lines'][k]['allowance']:.3f}"
                for k in ("x3", "y2916")
            ),
            "yes" if stopped(tight) and worst_49 < 1e-4 and acc["passed"] else "no",
        ],
        [
            "(d) product rows stop; outer count within 20%; fields within 1e-3",
            "; ".join(
                f"{c['grid']}: {c['stop'][1]}, {c['outer_change'] * 100:+.1f}%, "
                f"{g(c['max_face_difference_over_scale'])}"
                for c in compares
            ),
            "yes"
            if all(
                c["stop"][1] == "error_estimate_and_continuity"
                and abs(c["outer_change"]) < 0.2
                and c["max_face_difference_over_scale"] < 1e-3
                for c in compares
            )
            else "no",
        ],
    ]
    return table(["Prediction", "Measured", "Holds"], rows)


TABLES = {
    "laminar": laminar,
    "plants": plants,
    "annex": annex_runs,
    "lines": annex_lines,
    "harness": harness,
    "product": product,
    "predictions": predictions,
}


def splice(text: str, blocks: dict[str, str]) -> str:
    """Replace each named block of the report with its generated table."""
    for name, body in blocks.items():
        begin, end = f"<!-- tables50 {name} begin -->", f"<!-- tables50 {name} end -->"
        if begin not in text:
            continue
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{body}\n{end}{tail}"
    return text


def main() -> None:
    """Generate, and splice or check as asked."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--splice", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    blocks = {}
    for name, build in TABLES.items():
        try:
            blocks[name] = build()
        except FileNotFoundError as missing:
            print(f"{name}: no record yet ({missing.filename})", file=sys.stderr)
    text = "\n\n".join(f"## {name}\n\n{body}" for name, body in blocks.items())
    (OUT / "tables_all.md").write_text(text + "\n")
    report = REPORT.read_text(encoding="utf-8")
    spliced = splice(report, blocks)
    if args.check:
        if spliced != report:
            print("the report's tables differ from the records", file=sys.stderr)
            raise SystemExit(1)
        print("the report's tables match the records")
        return
    if args.splice:
        REPORT.write_text(spliced, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
