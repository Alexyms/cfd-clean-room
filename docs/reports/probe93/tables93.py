"""The tables of prompt 93's pressure report, from results/builder93/ and builder93b/.

    python docs/reports/probe93/tables93.py           # write tables93.md, splice the report
    python docs/reports/probe93/tables93.py --check   # regenerate and compare, write nothing

Run with Windows' ``.venv``: every record it reads is in the checkout.
``--check`` also fails when one of the report's marker pairs is missing.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLES = HERE / "tables93.md"
REPORT = ROOT / "docs" / "reports" / "phase6_pressure.md"
# Prompt 93's records, and those prompt 93b made again at its head.
FIRST = ROOT / "results" / "builder93"
RESULTS = ROOT / "results" / "builder93b"
TOLERANCE = 1e-12
SPEED = 5.0
RUNS = (1, 2, 3)
# Round 1's sessions, kept in builder93b/.
QUIET_RUNS = (4, 5, 6)
CAPS = (1, 4)
CASES = {
    "product80": "80x30 product room",
    "cavity40": "40x40 cavity",
    "vents80": "80x30 vents room",
    "annex216": "216x72 Annex 20 room",
    "backstep": "656x60 backward-facing step",
}
BLOCKS = {
    "coefficients": "the five p' coefficients",
    "face_d": "d on the faces",
    "mass_imbalance": "the mass imbalance (of u*, v* and of the corrected field)",
    "right_hand_side": "the right-hand side, projected on a closed domain",
    "apply_operator": "`apply_operator` on p'",
    "apply_operator_f": "`apply_operator` on the right-hand side",
    "jacobi_preconditioner": "the diagonal preconditioner",
    "levels": "every level's five arrays",
    "smoothers": "every level's smoother",
    "sweep": "a weighted Jacobi sweep",
    "restrict": "`restrict`",
    "add_prolonged": "`add_prolonged`",
    "projection": "the closed domain's projection",
    "coarsest_matrix": "the coarsest level's dense matrix",
    "correction": "u, v, p and p' given the solved p'",
}
PART_NAMES = {
    "assembly": "the coefficients",
    "build": "the hierarchy's build",
    "pseudo_inverse": "of it, the pseudo-inverse",
    "cycle": "one V-cycle",
}
KINDS = {
    "planted": "the parity module's planted systems",
    "worse": "planted, worse conditioned",
    "captured": "the captured corrections' systems",
}
# The planted cases of the first kind, and of the solve, by test function.
FIRST_KIND = (
    "test_the_equation_and_the_correction_are_bit_for_bit",
    "test_the_multigrids_blocks_are_bit_for_bit",
    "test_every_block_is_bit_for_bit_at_every_row_length",
    "test_a_zero_right_hand_side_takes_no_iteration",
)


def table(header: list[str], rows: Iterable[list[object]]) -> list[str]:
    """A markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def load(name: str, folder: Path = RESULTS) -> object:
    """A JSON record of results/builder93b/, or of another folder."""
    return json.loads((folder / name).read_text(encoding="utf-8"))


def junit_cases(path: Path) -> list[tuple[str, str, str]]:
    """(class name, test name, outcome) for every case of a JUnit file."""
    out = []
    for case in ET.parse(path).getroot().iter("testcase"):
        outcome = "passed"
        if case.find("failure") is not None or case.find("error") is not None:
            outcome = "failed"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        out.append((case.get("classname", ""), case.get("name", ""), outcome))
    return out


def counts(outcomes: list[str]) -> list[object]:
    """Run, passed, failed and skipped."""
    return [
        f"{len(outcomes):,}",
        f"{outcomes.count('passed'):,}",
        outcomes.count("failed"),
        outcomes.count("skipped"),
    ]


def power(x: float) -> str:
    """A number as 1.2e-13, zero as 0."""
    return "0" if x == 0.0 else f"{x:.1e}"


# ---------------------------------------------------------------------------


def tests_tables() -> tuple[list[str], list[str], dict[str, list[str]]]:
    """T1: the new parity cases by function; T1b: CTest and the whole suites."""
    by_function: dict[str, list[str]] = {}
    for cls, name, outcome in junit_cases(RESULTS / "parity.xml"):
        if cls.endswith("test_core_pressure"):
            by_function.setdefault(name.split("[")[0], []).append(outcome)
    rows = [[f"`{f}`", *counts(o)] for f, o in by_function.items()]
    rows.append(
        [
            "`tests/test_core_pressure.py`",
            *counts([o for outcomes in by_function.values() for o in outcomes]),
        ]
    )
    functions = table(["Test", "Cases", "Passed", "Failed", "Skipped"], rows)

    ctest = junit_cases(RESULTS / "ctest.xml")
    suites: dict[str, list[str]] = {}
    for _cls, name, outcome in ctest:
        suites.setdefault(name.split(".")[0], []).append(outcome)
    rows = [
        [f"CTest, suite `{suite}`", *counts(o)]
        for suite, o in suites.items()
        if re.match(r"Pressure|Multigrid|PseudoInverse|\w*ThreeDirections", suite)
    ]
    rows.append(["CTest, all suites", *counts([c[2] for c in ctest])])
    for label, name in (
        (
            "the four parity modules, WSL2, `CFD_CORE_REQUIRED=1`",
            "parity_modules.xml",
        ),
        ("the whole suite, WSL2, the core built", "suite_linux.xml"),
        ("the whole suite, Windows, the core not built", "suite_windows.xml"),
    ):
        rows.append([label, *counts([c[2] for c in junit_cases(RESULTS / name)])])
    return (
        functions,
        table(["Tests", "Cases", "Passed", "Failed", "Skipped"], rows),
        by_function,
    )


def captures_table() -> list[str]:
    """T2: the captured cases."""
    rows = []
    for case, label in CASES.items():
        c = load(f"captures/{case}.json", FIRST)
        start = c["start"]
        source = f"`{start['from']}`" if "/" in start["from"] else start["from"]
        if "outer" in start:
            source += f" ({start['outer']:,} outer)"
        low, high = min(c["iterations"]), max(c["iterations"])
        rows.append(
            [
                label,
                source,
                c["calls"],
                c["distinct_calls"],
                c["preconditioner"],
                f"{c['rtol']:g}",
                str(low) if low == high else f"{low} to {high}",
                f"{c['largest_step_u']:.1e}",
            ]
        )
    return table(
        [
            "Case",
            "Warm-started from",
            "Corrections",
            "Distinct",
            "Preconditioner",
            "rtol",
            "CG iterations",
            "Largest change of u* over them, m/s",
        ],
        rows,
    )


def first_table(records: list[dict]) -> tuple[list[str], bool]:
    """T3: the captured corrections' blocks of the first kind."""
    rows = []
    clean = True
    for key, label in BLOCKS.items():
        cells: list[object] = [label]
        for case in CASES:
            differ = [r["blocks"][key] for r in records if r["case"] == case]
            clean = clean and not any(differ)
            cells.append(f"{sum(1 for d in differ if d)} of {len(differ)}")
        rows.append(cells)
    again = [
        sum(1 for r in records if r["case"] == case and r["reference_differs"])
        for case in CASES
    ]
    rows.append(
        [
            "the reference called again, not what was captured",
            *(
                f"{n} of {sum(r['case'] == c for r in records)}"
                for c, n in zip(CASES, again, strict=True)
            ),
        ]
    )
    return table(
        ["Block: corrections with a differing bit", *CASES.values()], rows
    ), clean and not any(again)


def second_table(records: list[dict]) -> list[str]:
    """T3b: the captured pseudo-inverses and cycles."""
    rows = []
    for case, label in CASES.items():
        mine = [r["second"] for r in records if r["case"] == case]
        rows.append(
            [
                label,
                mine[0]["cells"],
                *(
                    power(max(m[key] for m in mine))
                    for key in (
                        "pseudo_inverse",
                        "cycle",
                        "precondition",
                        "smallest_core",
                        "smallest_lapack",
                        "next_lapack",
                    )
                ),
            ]
        )
    return table(
        [
            "Case",
            "Coarsest cells",
            "Pseudo-inverse, largest difference",
            "V-cycle on f",
            "The preconditioner on f, projected where closed",
            "Smallest eigenvalue over the largest, core",
            "LAPACK",
            "The next, LAPACK",
        ],
        rows,
    )


def corrections_table(records: list[dict]) -> tuple[list[str], bool, int]:
    """T3c: the captured corrections whole; whether each is inside (c)."""
    rows = []
    inside = 0
    for case, label in CASES.items():
        mine = [r for r in records if r["case"] == case]
        same = [r for r in mine if r["iterations"][0] == r["iterations"][1]]
        good = [
            r for r in same if max(r["fields"][k] for k in ("u", "v", "p")) <= TOLERANCE
        ]
        inside += len(good)
        rows.append(
            [
                label,
                len(mine),
                len(same),
                sum(1 for r in mine if r["products"][0] == r["products"][1]),
                sum(1 for r in mine if r["reached_cap"][0] != r["reached_cap"][1]),
                *(
                    power(max(r["fields"][k] for r in mine))
                    for k in ("u", "v", "p", "p_prime")
                ),
            ]
        )
    return (
        table(
            [
                "Case",
                "Corrections",
                "Same CG count",
                "Same products",
                "Cap differs",
                "u",
                "v",
                "p",
                "p'",
            ],
            rows,
        ),
        inside == len(records),
        inside,
    )


def corrections_each(records: list[dict]) -> list[str]:
    """T3d: each captured correction."""
    return table(
        [
            "Case",
            "Correction",
            "Reference's count",
            "Core's",
            "u",
            "v",
            "p",
            "Pseudo-inverse",
            "V-cycle",
        ],
        (
            [
                r["case"],
                r["call"],
                r["iterations"][0],
                r["iterations"][1],
                *(power(r["fields"][k]) for k in ("u", "v", "p")),
                power(r["second"]["pseudo_inverse"]),
                power(r["second"]["cycle"]),
            ]
            for r in records
        ),
    )


def plant_records() -> list[dict]:
    """Every stream's plants, one control, in variants93's order."""
    sys.path.insert(0, str(HERE))
    import variants93

    records: dict[str, dict] = {}
    for path in sorted(RESULTS.glob("plants_s*.json")):
        for record in json.loads(path.read_text(encoding="utf-8")):
            records[record["name"]] = record
    return [records[name] for name, _what, _edits in variants93.PLANTS]


def caught_by(record: dict) -> list[str]:
    """Which of the three nets fails on a plant."""
    if "build" in record:
        return []
    nets = []
    if record["ctest_failed"]:
        nets.append(f"CTest ({len(record['ctest_failed'])})")
    failed = record["parity_failed_by_test"]
    if record["pytest_returncode"] != 0:
        cases = sum(failed.values())
        nets.append(
            f"parity ({cases:,} cases of {len(failed)} tests)"
            if cases
            else f"parity (exit {record['pytest_returncode']})"
        )
    differ = [
        line.split()[0]
        for line in record["replay"]
        if " 0 with a block" not in line or " 0 counts that differ" not in line
    ]
    if differ:
        nets.append(f"replay ({', '.join(differ)})")
    elif len(record["replay"]) != len(CASES):
        nets.append("replay (raises)")
    return nets


def plants_table() -> tuple[list[str], int, int, list[str]]:
    """T4: the planted defects."""
    sys.path.insert(0, str(HERE))
    import variants93

    equivalent = variants93.EQUIVALENT
    rows = []
    planted = caught = 0
    survivors = []
    for record in plant_records():
        nets = caught_by(record)
        if record["name"] == "control":
            rows.append(
                ["`control`", record["what"], "; ".join(nets) or "nothing, as it must"]
            )
            continue
        planted += 1
        caught += bool(nets)
        verdict = "; ".join(nets)
        if not nets:
            verdict = "does not build" if "build" in record else "nothing"
            survivors.append(record["name"])
            if record["name"] in equivalent:
                verdict = f"nothing: {equivalent[record['name']]}"
        rows.append([f"`{record['name']}`", record["what"], verdict])
    return (
        table(["Plant", "What it changes", "Fails"], rows),
        planted,
        caught,
        [s for s in survivors if s not in equivalent],
    )


def load_text(lines: list[str]) -> str:
    """Windows' busiest processes and the power state, in one cell."""
    busy = [x for x in lines if not x.startswith(("on mains", "processor"))]
    rest = [x for x in lines if x.startswith(("on mains", "processor"))]
    return ", ".join(busy[:3] + rest) if lines else "not read"


def milliseconds(t: dict[str, float]) -> str:
    """A median in ms."""
    return f"{t['median_s'] * 1e3:.3f}"


def timing_tables() -> tuple[list[str], list[str], dict[int, float]]:
    """T5: a correction; T5b: the parts; the ratios by run."""
    rows, parts, ratios = [], [], {}
    for run in (*RUNS, *QUIET_RUNS):
        t = load(f"timing_{run}.json", FIRST if run in RUNS else RESULTS)
        ratios[run] = t["ratio"]
        rows.append(
            [
                run,
                t["started"],
                f"{t['iterations'][0]} and {t['iterations'][1]}",
                *(
                    f"{milliseconds(t[k])} ({t[k]['q1_s'] * 1e3:.2f} to {t[k]['q3_s'] * 1e3:.2f})"
                    for k in ("reference", "core")
                ),
                f"{t['ratio']:.2f}",
                f"{t['calibration_before']['median_s'] * 1e6:.2f}, {t['calibration_after']['median_s'] * 1e6:.2f}",
                load_text(t["windows_before"]),
                load_text(t["windows_after"]),
            ]
        )
        for key, name in PART_NAMES.items():
            part = t["parts"][key]
            parts.append(
                [
                    run,
                    name,
                    milliseconds(part["reference"]),
                    milliseconds(part["core"]),
                    f"{part['reference']['median_s'] / part['core']['median_s']:.2f}",
                ]
            )
        low, high = (t["parts"][f"capped_{cap}"] for cap in CAPS)
        each = {
            k: (high[k]["median_s"] - low[k]["median_s"]) / (CAPS[1] - CAPS[0])
            for k in ("reference", "core")
        }
        parts.append(
            [
                run,
                f"one CG iteration (a correction capped at {CAPS[1]} less one at {CAPS[0]}, over {CAPS[1] - CAPS[0]})",
                f"{each['reference'] * 1e3:.3f}",
                f"{each['core'] * 1e3:.3f}",
                f"{each['reference'] / each['core']:.2f}",
            ]
        )
    return (
        table(
            [
                "Session",
                "Started",
                "CG iterations, reference and core",
                "Reference, ms (quartiles)",
                "Core, ms (quartiles)",
                "Ratio",
                "Calibration before and after, us",
                "Windows before",
                "Windows after",
            ],
            rows,
        ),
        table(["Session", "Part", "Reference, ms", "Core, ms", "Ratio"], parts),
        ratios,
    )


def conditioning_tables() -> tuple[list[str], list[str], list[dict]]:
    """T6: the inverses against the exact ones by kind; T6b: those beyond 1e-12 of NumPy's."""
    records = load("conditioning.json", FIRST)
    rows = []
    for kind, label in KINDS.items():
        mine = [r for r in records if r["kind"] == kind]
        rows.append(
            [
                label,
                len(mine),
                sum(1 for r in mine if r["null"]),
                f"{min(r['condition'] for r in mine):.0f} to {max(r['condition'] for r in mine):,.0f}",
                sum(1 for r in mine if r["between"] > TOLERANCE),
                *(
                    power(max(r[k] for r in mine))
                    for k in ("between", "numpy_error", "core_error")
                ),
                *(
                    power(max(r[k] for r in mine))
                    for k in ("dropped_exact", "dropped_lapack", "dropped_core")
                ),
            ]
        )
    summary = table(
        [
            "Systems",
            "Count",
            "Singular",
            "Condition on the range",
            "Core and NumPy further apart than 1e-12",
            "Largest distance between them",
            "NumPy's largest error",
            "The core's largest error",
            "Largest dropped eigenvalue over the largest, exact",
            "LAPACK's",
            "The core's",
        ],
        rows,
    )
    beyond = [r for r in records if r["between"] > TOLERANCE]
    each = table(
        [
            "System (rows x columns, SOLID share, spread in decades, outlet term)",
            "Singular",
            "Condition",
            "Between the two",
            "NumPy's error",
            "The core's error",
            "Largest dropped eigenvalue over the largest, exact, LAPACK's, the core's",
        ],
        (
            [
                r["name"],
                "yes" if r["null"] else "no",
                f"{r['condition']:,.0f}",
                power(r["between"]),
                power(r["numpy_error"]),
                power(r["core_error"]),
                ", ".join(
                    power(r[k])
                    for k in ("dropped_exact", "dropped_lapack", "dropped_core")
                ),
            ]
            for r in beyond
        ),
    )
    return summary, each, records


def criterion_tables() -> tuple[list[str], list[str]]:
    """T6c: the pseudo-inverses by their criterion; T6d: the systems nearest its bounds."""
    records = load("inverses.json")
    rows = []
    for kind, label in {**KINDS, "rest": "the first corrections' from rest"}.items():
        mine = [r for r in records if r["kind"] == kind]
        rows.append(
            [
                label,
                len(mine),
                sum(1 for r in mine if r["dropped"]),
                f"{min(r['condition'] for r in mine):.1f} to {max(r['condition'] for r in mine):,.0f}",
                sum(r["core_error"] > r["bound"] for r in mine),
                sum(r["numpy_error"] > r["bound"] for r in mine),
                sum(r["between"] > r["between_bound"] for r in mine),
                *(
                    f"{max(r[e] / r[b] for r in mine):.2g}"
                    for e, b in (
                        ("core_error", "bound"),
                        ("numpy_error", "bound"),
                        ("between", "between_bound"),
                    )
                ),
                sum(r["between_over_numpy"] > TOLERANCE for r in mine),
                power(max(max(r["residuals"]) for r in mine)),
            ]
        )
    summary = table(
        [
            "Systems",
            "Count",
            "With dropped values",
            "kappa",
            "The core beyond 4 n kappa eps max X",
            "NumPy beyond it",
            "The two beyond 8 n kappa eps max X",
            "The core's largest share of its bound",
            "NumPy's",
            "The two's",
            "The two beyond 1e-12 of NumPy's largest",
            "The referee's largest residual of Moore and Penrose's conditions",
        ],
        rows,
    )
    rotations = load("rotations.json")
    nearest = sorted(records, key=lambda r: -r["core_error"] / r["bound"])[:5]
    each = table(
        [
            "System",
            "n",
            "kappa",
            "max X",
            "Bound",
            "The core's error",
            "NumPy's",
            "Between the two, over NumPy's largest",
        ],
        (
            [
                f"{r['kind']}: {r['name']}",
                r["cells"],
                f"{r['condition']:,.2f}",
                f"{r['largest']:.3g}",
                *(
                    power(r[k])
                    for k in (
                        "bound",
                        "core_error",
                        "numpy_error",
                        "between_over_numpy",
                    )
                ),
            ]
            for r in nearest
        ),
    )
    each += [
        "",
        f"The referee against the tester's 50-digit rotations on the {len(rotations)} systems of T6b: "
        f"the largest difference {power(max(r['apart'] for r in rotations))} of the largest magnitude.",
    ]
    return summary, each


def field_records() -> list[dict]:
    """The corrected fields' records, a case each."""
    paths = sorted((RESULTS / "fields").glob("*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def both_components(row: dict) -> float:
    """The largest difference over the largest magnitude of either component's terms."""
    terms = max(max(row[c]["star"], row[c]["correction"]) for c in "uv")
    return max(row[c]["difference"] for c in "uv") / terms


def field_tables() -> tuple[list[str], list[str]]:
    """T8: the corrected fields by both measures; T8b: those beyond 1e-12 by the measure."""
    cases = field_records()
    rows, beyond = [], []
    for kind, label in (("rest", "from rest"), ("captured", "captured")):
        for case in (c for c in cases if c["kind"] == kind):
            mine = case["rows"]
            cells: list[object] = [label, f"`{case['case']}`", len(mine)]
            cells.append(sum(r["iterations"][0] == r["iterations"][1] for r in mine))
            for c in "uv":
                for measure in ("by_field", "by_terms"):
                    cells.append(power(max(r[c][measure] for r in mine)))
            cells.append(power(max(r["p"]["by_field"] for r in mine)))
            cells.append(power(max(both_components(r) for r in mine)))
            rows.append(cells)
            for r in mine:
                for c in "uv":
                    if max(r[c]["by_terms"], r[c]["by_field"]) > TOLERANCE:
                        beyond.append((case["case"], r, c))
    totals = [r for case in cases for r in case["rows"]]
    inside = [r for r in totals if max(r[c]["by_terms"] for c in "uv") <= TOLERANCE]
    summary = table(
        [
            "Corrections",
            "Case and cap",
            "Count",
            "Same CG count",
            "u over the corrected u",
            "u by the measure",
            "v over the corrected v",
            "v by the measure",
            "p over p",
            "Either component over both components' terms (after the fact)",
        ],
        rows,
    )
    summary += [
        "",
        f"{len(inside):,} of {len(totals):,} corrections within 1e-12 by the measure.",
    ]
    each = table(
        [
            "Case and cap",
            "Correction",
            "CG count, both",
            "Component",
            "Largest difference",
            "Predicted (largest)",
            "Correction (largest)",
            "Corrected (largest)",
            "Over the corrected",
            "By the measure",
            "Over both components' terms (after the fact)",
        ],
        (
            [
                f"`{case}`",
                r["call"],
                r["iterations"][1],
                c,
                power(r[c]["difference"]),
                *(f"{r[c][k]:.4g}" for k in ("star", "correction", "corrected")),
                power(r[c]["by_field"]),
                power(r[c]["by_terms"]),
                power(both_components(r)),
            ]
            for case, r, c in beyond
        ),
    )
    return summary, each


def loops_table() -> list[str]:
    """T7: the blocks of the three files and whether GCC vectorises each."""
    record = load("vecinfo.json")
    rows = [
        [
            f"`{b['file']}.cpp`",
            f"`{b['block']}`",
            "yes" if b["vectorised"] else "no",
        ]
        for b in record["blocks"]
    ]
    widths = sorted({w for b in record["blocks"] for w in b["widths"]})
    return [
        *table(["File", "Block", "Vectorised"], rows),
        "",
        f"{record['compiler']}; {' '.join(record['flags'])}. "
        f"{sum(b['vectorised'] for b in record['blocks'])} of {len(record['blocks'])} blocks "
        f"vectorised, with vectors of {', '.join(widths)} bytes.",
    ]


def predictions_table(
    functions: dict[str, list[str]],
    records: list[dict],
    first_clean: bool,
    inside: int,
    conditioning: list[dict],
    ratios: dict[int, float],
) -> list[str]:
    """T0: the predictions by the committed rule."""
    planted = [o for f in FIRST_KIND for o in functions[f]]
    blocks = len(BLOCKS)
    a_holds = first_clean and planted.count("passed") == len(planted) > 0
    parity = [r for r in conditioning if r["kind"] == "planted"]
    caught = [r for r in conditioning if r["kind"] == "captured"]
    worse = [r for r in conditioning if r["kind"] == "worse"]
    beyond = {
        k: sum(1 for r in rows if r["between"] > TOLERANCE)
        for k, rows in (("planted", parity), ("captured", caught), ("worse", worse))
    }
    b_holds = not any(beyond.values())
    apart = [r for r in conditioning if r["between"] > TOLERANCE]
    numpys = sum(1 for r in apart if r["numpy_error"] > r["core_error"])
    singular = sum(1 for r in apart if r["null"])
    worst = max(max(r["fields"][k] for k in ("u", "v", "p")) for r in records)
    same = sum(1 for r in records if r["iterations"][0] == r["iterations"][1])
    d_holds = ratios[1] >= SPEED
    return table(
        ["Prediction", "Measured", "Verdict"],
        [
            [
                "(a) every block of the first kind equals the reference bit for bit on every planted and captured input",
                f"{blocks} blocks on {len(records)} captured corrections, no differing bit (T3); "
                f"{planted.count('passed'):,} of {len(planted):,} planted cases pass, each of several blocks (T1)",
                "holds" if a_holds else "fails",
            ],
            [
                "(b) the pseudo-inverse within 1e-12 of NumPy's on every planted and captured coarsest system",
                f"captured: {beyond['captured']} of {len(caught)} beyond it; the parity module's planted "
                f"systems: {beyond['planted']} of {len(parity)}; worse conditioned planted ones: "
                f"{beyond['worse']} of {len(worse)} (T6). Of the {len(apart)} beyond it {singular} are "
                f"singular, and NumPy's matrix is the one further from the exact inverse on {numpys} (T6b)",
                "holds" if b_holds else "fails as stated",
            ],
            [
                "(c) every captured correction at the reference's count, u, v and p within 1e-12",
                f"{same} of {len(records)} at the reference's count; {inside} of {len(records)} inside, "
                f"the largest difference {power(worst)} (T3c)",
                "holds" if inside == len(records) else "fails",
            ],
            [
                "(d) a correction on the 320x120 state at least five times faster",
                f"{ratios[1]:.2f} times in session 1 (session 2: {ratios[2]:.2f}; session 3: {ratios[3]:.2f}); "
                f"round 1's sessions, beside them: {', '.join(f'{ratios[run]:.2f}' for run in QUIET_RUNS)} (T5)",
                "holds" if d_holds else "fails",
            ],
        ],
    )


# ---------------------------------------------------------------------------


def build() -> tuple[str, dict[str, str]]:
    """The tables' file and the report's blocks."""
    functions, totals, by_function = tests_tables()
    records = load("replay.json")
    first, first_clean = first_table(records)
    corrections, _all, inside = corrections_table(records)
    plants, planted, caught, survivors = plants_table()
    timing, parts, ratios = timing_tables()
    summary, beyond, conditioning = conditioning_tables()
    judged, nearest = criterion_tables()
    fields, fields_beyond = field_tables()
    count = [
        "",
        f"{caught} of {planted} plants fail at least one net"
        + (
            "; not caught: " + ", ".join(f"`{s}`" for s in survivors) + "."
            if survivors
            else "; every other is an equivalent mutant, as its row says."
            if caught < planted
            else "."
        ),
    ]
    sections = [
        (
            "predictions",
            "T0. The predictions, scored by the rule committed with them",
            predictions_table(
                by_function, records, first_clean, inside, conditioning, ratios
            ),
        ),
        ("parity", "T1. The parity cases by function", functions),
        ("tests", "T1b. CTest, the parity modules and the suites", totals),
        ("captures", "T2. The captured corrections", captures_table()),
        (
            "first",
            "T3. The first kind on the captured corrections: corrections whose block differs in a bit",
            first,
        ),
        (
            "second",
            "T3b. The second kind on the captured corrections: the largest differences over the reference's largest magnitude",
            second_table(records),
        ),
        (
            "corrections",
            "T3c. The captured corrections whole: counts, and the fields' largest differences over the reference's largest magnitude",
            corrections,
        ),
        ("each", "T3d. Each captured correction", corrections_each(records)),
        ("plants", "T4. The planted defects: which nets fail on each", plants + count),
        (
            "timing",
            "T5. A correction on the 320x120 product state, one thread",
            timing,
        ),
        ("parts", "T5b. The parts, the same sessions", parts),
        (
            "conditioning",
            "T6. The pseudo-inverses against the exact ones",
            summary,
        ),
        (
            "beyond",
            "T6b. Every system on which the core and NumPy are further apart than 1e-12",
            beyond,
        ),
        ("criterion", "T6c. The pseudo-inverses by their criterion", judged),
        (
            "nearest",
            "T6d. The five systems nearest the core's bound",
            nearest,
        ),
        ("loops", "T7. The blocks GCC vectorises", loops_table()),
        ("fields", "T8. The corrected fields by both measures", fields),
        (
            "beyond_fields",
            "T8b. Every corrected component beyond 1e-12 by either measure",
            fields_beyond,
        ),
    ]
    blocks = {
        name: "\n".join([f"### {title}", "", *lines]) for name, title, lines in sections
    }
    text = "# Prompt 93's tables\n\n" + "\n\n".join(blocks.values()) + "\n"
    return text, blocks


def splice(report: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """The report with each marked block replaced, and the names whose markers are missing."""
    missing = []
    for name, body in blocks.items():
        begin = f"<!-- tables93 {name} begin -->"
        end = f"<!-- tables93 {name} end -->"
        if begin not in report or end not in report:
            missing.append(name)
            continue
        head, rest = report.split(begin, 1)
        _old, tail = rest.split(end, 1)
        report = f"{head}{begin}\n{body}\n{end}{tail}"
    return report, missing


def main() -> int:
    """Write, or with --check compare."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text, blocks = build()
    report = REPORT.read_text(encoding="utf-8")
    spliced, missing = splice(report, blocks)
    note = f"(missing markers: {missing})" if missing else ""
    if args.check:
        ok = (
            TABLES.read_text(encoding="utf-8") == text
            and spliced == report
            and not missing
        )
        print("tables93 --check:", "matches" if ok else "DIFFERS", note)
        return 0 if ok else 1
    TABLES.write_text(text, encoding="utf-8")
    REPORT.write_text(spliced, encoding="utf-8")
    print("wrote", TABLES, "and spliced", REPORT, note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
