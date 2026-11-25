"""The tables of prompt 94's turbulence report, from results/builder94/.

    python docs/reports/probe94/tables94.py           # write tables94.md, splice the report
    python docs/reports/probe94/tables94.py --check   # regenerate and compare, write nothing

Run with Windows' ``.venv``: every record it reads is in the checkout.
``--check`` also fails when one of the report's marker pairs is missing.
"""

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "probe93"))

# The helpers of prompt 93's tables that read no record of their own.
from tables93 import (  # noqa: E402 -- after sys.path
    counts,
    junit_cases,
    load_text,
    milliseconds,
    power,
    table,
)

TABLES = HERE / "tables94.md"
REPORT = ROOT / "docs" / "reports" / "phase6_turbulence.md"
RESULTS = ROOT / "results" / "builder94"
TOLERANCE = 1e-12
SPEED = 5.0
RUNS = (1, 2, 3)
# The sessions with NumPy's AVX-512 loops off, beside the scored ones.
LIBM_RUNS = (4, 5, 6)
CASES = {
    "product80": "80x30 product room",
    "vents80": "80x30 vents room",
    "annex216": "216x72 Annex 20 room",
    "backstep": "656x60 backward-facing step",
    "product80_umist": "80x30 product room, k and eps by UMIST",
}
BLOCKS = {
    "wall_viscosity": "the wall viscosity on both wall stencils (log)",
    "inflow": "the four inflow arrays",
    "tangential": "the four edges' tangential velocities",
    "eps_held": "the held cells and their eps (cube)",
    "production_given": "the given production (log)",
    "pseudo_time_step": "the pseudo-time step",
    "corner_gradients": "the corner gradients du/dy and dv/dx",
    "strain_squared": "the strain squared",
    "terms": "production, R, the two growths and the two decays",
    "k_advected": "k advected",
    "eps_advected": "eps advected",
    "k_conductances": "k's conductances",
    "eps_conductances": "eps's conductances",
    "eddy_viscosity": "the eddy viscosity",
    "k_explicit": "k's explicit field",
    "eps_explicit": "eps's explicit field",
}
PARTS = (
    "two CG solves",
    "conditions",
    "wall viscosity",
    "terms",
    "pseudo-time step",
)
# The planted cases of prediction (a), by test function.
FIRST_KIND = (
    "test_the_blocks_equal_the_reference_bit_for_bit",
    "test_the_standard_terms_equal_the_reference_bit_for_bit",
    "test_the_rng_terms_equal_the_reference_bit_for_bit",
    "test_the_corner_gradients_equal_the_reference_bit_for_bit",
    "test_small_rooms_equal_the_reference_bit_for_bit",
    "test_the_conditions_equal_the_reference_bit_for_bit",
    "test_the_wall_viscosity_equals_the_reference_bit_for_bit",
    "test_wall_viscosity_equals_the_reference_bit_for_bit",
    "test_the_fixed_arrays_equal_the_reference",
)
MODULE = "test_core_turbulence"


def load(name: str) -> object:
    """A JSON record of results/builder94/."""
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------


def tests_tables() -> tuple[list[str], list[str], dict[str, list[str]]]:
    """T1: the parity cases by function; T1b: CTest and the whole suites."""
    by_function: dict[str, list[str]] = {}
    for cls, name, outcome in junit_cases(RESULTS / "parity.xml"):
        if cls.endswith(MODULE):
            by_function.setdefault(name.split("[")[0], []).append(outcome)
    default: dict[str, list[str]] = {}
    for cls, name, outcome in junit_cases(RESULTS / "parity_default.xml"):
        if cls.endswith(MODULE):
            default.setdefault(name.split("[")[0], []).append(outcome)
    rows = [
        [f"`{f}`", *counts(o), default.get(f, []).count("skipped")]
        for f, o in by_function.items()
    ]
    rows.append(
        [
            f"`tests/{MODULE}.py`",
            *counts([o for outcomes in by_function.values() for o in outcomes]),
            sum(o.count("skipped") for o in default.values()),
        ]
    )
    functions = table(
        [
            "Test",
            "Cases",
            "Passed",
            "Failed",
            "Skipped",
            "Skipped where NumPy runs its AVX-512 loops",
        ],
        rows,
    )
    ctest = junit_cases(RESULTS / "ctest.xml")
    suites: dict[str, list[str]] = {}
    for _cls, name, outcome in ctest:
        suites.setdefault(name.split(".")[0], []).append(outcome)
    rows = [
        [f"CTest, suite `{suite}`", *counts(o)]
        for suite, o in suites.items()
        if re.match(r"Turbulence|Walls|Elementary|\w*ThreeDirections", suite)
    ]
    rows.append(["CTest, all suites", *counts([o for _c, _n, o in ctest])])
    for label, name in (
        (
            "the five parity modules, WSL2, `CFD_CORE_REQUIRED=1`, "
            "`NPY_DISABLE_CPU_FEATURES=X86_V4`",
            "parity_modules.xml",
        ),
        ("the whole suite, WSL2, the core built", "suite_linux.xml"),
        ("the whole suite, Windows, the core not built", "suite_windows.xml"),
    ):
        rows.append([label, *counts([o for _c, _n, o in junit_cases(RESULTS / name)])])
    suite = table(["Tests", "Cases", "Passed", "Failed", "Skipped"], rows)
    return functions, suite, by_function


def captures_table() -> list[str]:
    """T2: the captured steps."""
    rows = []
    for case, label in CASES.items():
        r = load(f"captures/{case}.json")
        source = r["start"]["from"]
        if "outer" in r["start"]:
            source += f" of {r['start']['outer']} outer iterations"
        counted = r["iterations"]
        rows.append(
            [
                label,
                source,
                r["calls"],
                r["distinct_calls"],
                f"{r['variant']}, {r['advection']}",
                f"{min(c[0] for c in counted)} to {max(c[0] for c in counted)}",
                f"{min(c[1] for c in counted)} to {max(c[1] for c in counted)}",
                sum(sum(f) for f in r["fallbacks"]),
                power(r["largest_step_k"]),
            ]
        )
    return table(
        [
            "Case",
            "The state the solve starts from",
            "Steps",
            "Of them different",
            "Variant, advection",
            "CG iterations of k",
            "of eps",
            "Fallbacks",
            "Largest change of k over the steps, m^2/s^2",
        ],
        rows,
    )


def first_table(records: list[dict]) -> tuple[list[str], int]:
    """T3: the captured steps on which a block of the first kind differs in a bit."""
    rows, bad = [], 0
    for block, label in BLOCKS.items():
        row = [label]
        for case in CASES:
            mine = [r for r in records if r["case"] == case]
            differ = sum(r["blocks"][block][0] != 0 for r in mine)
            bad += differ
            row.append(f"{differ} of {len(mine)}")
        rows.append(row)
    return table(["Block", *CASES.values()], rows), bad


def default_table(records: list[dict]) -> tuple[list[str], dict[str, int]]:
    """T3b: the same where NumPy runs its AVX-512 loops, the blocks that differ."""
    rows, differ = [], {}
    for block, label in BLOCKS.items():
        entries = sum(r["blocks"][block][0] for r in records)
        steps = sum(r["blocks"][block][0] != 0 for r in records)
        if not entries:
            continue
        differ[block] = entries
        rows.append(
            [
                label,
                f"{steps} of {len(records)}",
                f"{entries:,}",
                power(max(r["blocks"][block][1] for r in records)),
            ]
        )
    return (
        table(
            [
                "Block",
                "Steps on which it differs",
                "Values that differ",
                "Largest difference over the reference's largest magnitude",
            ],
            rows,
        ),
        differ,
    )


def solves_table(records: list[dict]) -> tuple[list[str], dict[str, int]]:
    """T4: the captured CG solves by the rule, and each step's own differences."""
    rows = []
    found = {"solves": 0, "counts": 0, "fallbacks": 0, "beyond": 0}
    for case, label in CASES.items():
        mine = [r for r in records if r["case"] == case]
        for i, quantity in enumerate(("k", "eps")):
            solves = [r["solves"][i] for r in mine]
            counts_differ = sum(
                s["iterations"][0] != s["iterations"][1] for s in solves
            )
            fallbacks = sum(s["fallbacks"][0] != s["fallbacks"][1] for s in solves)
            beyond = sum(s["field"] > TOLERANCE for s in solves)
            found["solves"] += len(solves)
            found["counts"] += counts_differ
            found["fallbacks"] += fallbacks
            found["beyond"] += beyond
            rows.append(
                [
                    label,
                    quantity,
                    len(solves),
                    counts_differ,
                    fallbacks,
                    power(max(s["field"] for s in solves)),
                    power(max(r["step"][quantity] for r in mine)),
                    power(max(r["step"]["nu_t"] for r in mine)),
                    power(max(r["advance"][quantity] for r in mine)),
                ]
            )
    return (
        table(
            [
                "Case",
                "Quantity",
                "Solves",
                "Counts that differ",
                "Fallbacks on one side only",
                "Largest difference of a solve's field",
                "of the step's",
                "of the step's nu_t",
                "of the step's with the core's own conditions",
            ],
            rows,
        ),
        found,
    )


def planted_table() -> list[str]:
    """T5: the CG rule on the parity module's planted steps."""
    rows = []
    labels = {
        "planted": "planted over decades (the module asserts the wiring only)",
        "gentle": "gentle (the module asserts the rule)",
    }
    records = load("planted.json")
    for kind, label in labels.items():
        mine = [s for r in records if r["kind"] == kind for s in r["solves"]]
        same = [s for s in mine if s["iterations"][0] == s["iterations"][1]]
        rows.append(
            [
                label,
                len(mine),
                len(mine) - len(same),
                max(abs(s["iterations"][0] - s["iterations"][1]) for s in mine),
                sum(s["fallbacks"][0] != s["fallbacks"][1] for s in mine),
                sum(s["field"] > TOLERANCE for s in same),
                power(max(s["field"] for s in same)),
                power(
                    max(
                        (
                            s["field"]
                            for s in mine
                            if s["iterations"][0] != s["iterations"][1]
                        ),
                        default=0.0,
                    )
                ),
            ]
        )
    return table(
        [
            "Steps",
            "Solves",
            "Counts that differ",
            "by at most",
            "Fallbacks on one side only",
            "Fields beyond 1e-12 at equal counts",
            "The largest there",
            "The largest at counts that differ",
        ],
        rows,
    )


def rest_table() -> list[str]:
    """T4b: the first steps of each case's solve from rest, not scored."""
    records = load("rest.json")["rows"]
    rows = []
    for case, label in CASES.items():
        mine = [r for r in records if r["case"] == case]
        solves = [s for r in mine for s in r["solves"]]
        rows.append(
            [
                label,
                len(mine),
                sum(any(v[0] for v in r["blocks"].values()) for r in mine),
                sum(s["iterations"][0] != s["iterations"][1] for s in solves),
                sum(s["fallbacks"][0] != s["fallbacks"][1] for s in solves),
                power(max(s["field"] for s in solves)),
            ]
        )
    return table(
        [
            "Case",
            "Steps",
            "With a block that differs in a bit",
            "Counts that differ",
            "Fallbacks on one side only",
            "Largest difference of a solve's field",
        ],
        rows,
    )


def elementary_table() -> list[str]:
    """T6: NumPy's log and cube against the C library's, by NumPy's loops."""
    rows = []
    for label, name in (
        ("X86_V4 off", "elementary.json"),
        ("as it runs by default on this machine", "elementary_default.json"),
    ):
        record = load(name)
        for function, r in record["rows"].items():
            loop = record["loops"]["log" if "log" in function else "power"]
            rows.append(
                [
                    label,
                    f"`{function}`",
                    loop,
                    f"{r['values']:,}",
                    f"{r['differ']:,}",
                    power(r["largest_relative"]),
                    f"{r['largest_ulps']:.0f}",
                ]
            )
    return table(
        [
            "NumPy's AVX-512 loops",
            "Function",
            "NumPy's loop",
            "Values",
            "Differ from the C library's",
            "Largest relative difference",
            "in units of the last place",
        ],
        rows,
    )


def plant_records() -> list[dict]:
    """Every stream's plants, one control, in variants94's order."""
    import variants94

    records: dict[str, dict] = {}
    for path in sorted(RESULTS.glob("plants_s*.json")):
        for record in json.loads(path.read_text(encoding="utf-8")):
            records[record["name"]] = record
    return [records[name] for name, _what, _edits in variants94.PLANTS]


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
        if " 0 with a block" not in line
        or " 0 counts that differ" not in line
        or " 0 fallbacks that differ" not in line
    ]
    if differ:
        nets.append(f"replay ({', '.join(differ)})")
    elif len(record["replay"]) != len(CASES):
        nets.append("replay (raises)")
    return nets


def plants_table() -> tuple[list[str], int, int, list[str]]:
    """T7: the planted defects."""
    import variants94

    equivalent = variants94.EQUIVALENT
    rows, survivors = [], []
    planted = caught = 0
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


def timing_tables() -> tuple[list[str], list[str], dict[int, tuple[float, float]]]:
    """T8: a step with its conditions; T8b: the parts; the ratios by session."""
    rows, parts, ratios = [], [], {}
    for run in (*RUNS, *LIBM_RUNS):
        t = load(f"timing_{run}.json")
        ratios[run] = (t["ratio"], t["ratio_advance"])
        rows.append(
            [
                run,
                t["started"],
                "on" if t["loops"]["power"] == "X86_V4" else "off",
                f"{t['iterations'][0][0]} and {t['iterations'][0][1]}",
                *(
                    f"{milliseconds(t[k])} ({t[k]['q1_s'] * 1e3:.2f} to {t[k]['q3_s'] * 1e3:.2f})"
                    for k in ("reference", "core", "advance")
                ),
                f"{t['ratio']:.2f}",
                f"{t['ratio_advance']:.2f}",
                f"{t['calibration_before']['median_s'] * 1e6:.2f}, "
                f"{t['calibration_after']['median_s'] * 1e6:.2f}",
                load_text(t["windows_before"]),
                load_text(t["windows_after"]),
            ]
        )
        solves = t["parts"]["two CG solves"]
        for name in PARTS:
            part = t["parts"][name]
            parts.append(
                [
                    run,
                    name,
                    milliseconds(part["reference"]),
                    milliseconds(part["core"]),
                    f"{part['reference']['median_s'] / part['core']['median_s']:.2f}",
                ]
            )
        rest = {
            k: t[k]["median_s"]
            - solves["reference" if k == "reference" else "core"]["median_s"]
            for k in ("reference", "core", "advance")
        }
        for label, key in (
            ("the step with its conditions less the two solves", "core"),
            ("the same, the core's two calls in one (`advance`)", "advance"),
        ):
            parts.append(
                [
                    run,
                    label,
                    f"{rest['reference'] * 1e3:.3f}",
                    f"{rest[key] * 1e3:.3f}",
                    f"{rest['reference'] / rest[key]:.2f}",
                ]
            )
    return (
        table(
            [
                "Session",
                "Started",
                "NumPy's AVX-512 loops",
                "CG iterations of k and eps, both sides",
                "Reference, ms (quartiles)",
                "Core, two calls, ms (quartiles)",
                "Core, one call, ms (quartiles)",
                "Ratio",
                "Ratio, one call",
                "Calibration before and after, us",
                "Windows before",
                "Windows after",
            ],
            rows,
        ),
        table(["Session", "Part", "Reference, ms", "Core, ms", "Ratio"], parts),
        ratios,
    )


def loops_table() -> list[str]:
    """T9: the blocks of the two files and whether GCC vectorises each."""
    record = load("vecinfo.json")
    blocks = record["blocks"]
    rows = [
        [f"`{b['file']}.cpp`", f"`{b['block']}`", "yes" if b["vectorised"] else "no"]
        for b in blocks
    ]
    widths = sorted({w for b in blocks for w in b["widths"]})
    return [
        *table(["File", "Block", "Vectorised"], rows),
        "",
        f"{record['compiler']}; {' '.join(record['flags'])}. "
        f"{sum(b['vectorised'] for b in blocks)} of {len(blocks)} blocks "
        f"vectorised, with vectors of {', '.join(widths)} bytes.",
    ]


def predictions_table(
    functions: dict[str, list[str]],
    records: list[dict],
    bad: int,
    differ: dict[str, int],
    found: dict[str, int],
    ratios: dict[int, tuple[float, float]],
) -> list[str]:
    """T0: the predictions, scored by the rule committed with them."""
    planted = [o for f in FIRST_KIND for o in functions[f]]
    passed = planted.count("passed")
    deep: dict[str, list[str]] = {}
    for _cls, name, outcome in junit_cases(RESULTS / "ctest.xml"):
        if name.startswith(
            ("TurbulenceThreeDirections", "MomentumThree", "PressureThree")
        ):
            deep.setdefault(name.split(".")[0], []).append(outcome)
    turbulent = deep["TurbulenceThreeDirections"]
    whole = [o for outcomes in deep.values() for o in outcomes]
    c_holds = all(o == "passed" for o in whole) and turbulent
    b_holds = not (found["counts"] or found["fallbacks"] or found["beyond"])
    names = ", ".join(BLOCKS[b] for b in differ)
    ratio = ratios[1][0]
    others = "; ".join(f"session {r}: {ratios[r][0]:.2f}" for r in RUNS[1:])
    rows = [
        [
            "(a) every block of the first kind equals the reference bit for bit on every "
            "planted and captured input",
            f"as NumPy runs by default on this machine: {len(differ)} blocks of "
            f"{len(BLOCKS)} differ on the captured steps ({names}; "
            f"{sum(differ.values()):,} values, T3b), and the planted cases of a block that "
            "calls one of the two functions are skipped (T1; what it does to the functions "
            "is T6). Where NumPy calls the C library's "
            f"log and pow (Alex, 2026-11-25): {bad} of {len(BLOCKS) * len(records)} block "
            f"comparisons of the captured steps differ (T3), and {passed} of "
            f"{len(planted)} planted cases pass (T1)",
            "fails as stated; "
            + ("holds" if bad == 0 and passed == len(planted) else "fails")
            + " where NumPy calls the C library",
        ],
        [
            "(b) every captured CG solve at the reference's count, its field within 1e-12, "
            "no fallback on one side",
            f"{found['solves']} solves: {found['counts']} counts differ, "
            f"{found['beyond']} fields beyond 1e-12, {found['fallbacks']} fallbacks on one "
            "side only (T4)",
            "holds" if b_holds else "fails",
        ],
        [
            "(c) the D = 3 instantiation builds and its tests pass",
            f"the program of three directions builds; {whole.count('passed')} of "
            f"{len(whole)} of its tests pass, {turbulent.count('passed')} of "
            f"{len(turbulent)} of them this step's (T1b)",
            "holds" if c_holds else "fails",
        ],
        [
            "(d) a step with its conditions on the 320x120 state at least five times faster",
            f"{ratio:.2f} times in session 1 ({others}); the core's two calls in one: "
            + ", ".join(f"{ratios[r][1]:.2f}" for r in RUNS)
            + "; with NumPy's AVX-512 loops off: "
            + ", ".join(f"{ratios[r][0]:.2f}" for r in LIBM_RUNS)
            + ", in one call "
            + ", ".join(f"{ratios[r][1]:.2f}" for r in LIBM_RUNS)
            + " (T8)",
            "holds" if ratio >= SPEED else "fails",
        ],
    ]
    return table(["Prediction", "Measured", "Verdict"], rows)


# ---------------------------------------------------------------------------


def titled(title: str, lines: Iterable[str]) -> str:
    """A table under its title."""
    return "\n".join([f"### {title}", "", *lines])


def build() -> tuple[str, dict[str, str]]:
    """tables94.md's text, and the report's marked blocks."""
    functions, suites, by_function = tests_tables()
    replayed = load("replay.json")
    records = replayed["rows"]
    default = load("replay_default.json")["rows"]
    first, bad = first_table(records)
    by_default, differ = default_table(default)
    solves, found = solves_table(records)
    plants, planted, caught, survivors = plants_table()
    timing, parts, ratios = timing_tables()
    not_made = sum(bool(r["reference_differs"]) for r in records)
    blocks = {
        "predictions": titled(
            "T0. The predictions, scored by the rule committed with them",
            predictions_table(by_function, records, bad, differ, found, ratios),
        ),
        "parity": titled("T1. The parity cases by function", functions),
        "tests": titled("T1b. CTest, the parity modules and the suites", suites),
        "captures": titled("T2. The captured steps", captures_table()),
        "first": titled(
            "T3. The first kind on the captured steps: steps whose block differs in a bit, "
            "NumPy's AVX-512 loops off",
            [
                *first,
                "",
                f"The reference called again on a kept step's arguments gave other bits than "
                f"the kept ones on {not_made} of {len(records)} steps.",
            ],
        ),
        "default": titled(
            "T3b. The blocks that differ where NumPy runs its AVX-512 loops, the same steps",
            [
                *by_default,
                "",
                f"The other {len(BLOCKS) - len(differ)} blocks differ in no bit on any step.",
            ],
        ),
        "solves": titled(
            "T4. The captured CG solves, and the steps' own differences, each over the "
            "reference's largest magnitude",
            solves,
        ),
        "rest": titled(
            "T4b. The first outer iterations from rest, beside the captures and not scored, "
            "NumPy's AVX-512 loops off",
            rest_table(),
        ),
        "planted": titled(
            "T5. The CG rule on the parity module's planted steps", planted_table()
        ),
        "elementary": titled(
            "T6. NumPy's logarithm and cube against the C library's", elementary_table()
        ),
        "plants": titled(
            "T7. The planted defects: which nets fail on each",
            [
                *plants,
                "",
                f"{caught} of {planted} plants caught; not caught and not equivalent: "
                f"{', '.join(survivors) or 'none'}.",
            ],
        ),
        "timing": titled(
            "T8. A step with its conditions on the 320x120 product state, one thread",
            timing,
        ),
        "parts": titled("T8b. The parts, the same sessions", parts),
        "loops": titled("T9. The blocks GCC vectorises", loops_table()),
    }
    text = "# Prompt 94's tables\n\nGenerated by tables94.py.\n\n"
    return text + "\n\n".join(blocks.values()) + "\n", blocks


def splice(report: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """The report with each marked block replaced, and the names whose markers are missing."""
    missing = []
    for name, body in blocks.items():
        begin = f"<!-- tables94 {name} begin -->"
        end = f"<!-- tables94 {name} end -->"
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
        print("tables94 --check:", "matches" if ok else "DIFFERS", note)
        return 0 if ok else 1
    TABLES.write_text(text, encoding="utf-8")
    REPORT.write_text(spliced, encoding="utf-8")
    print("wrote", TABLES, "and spliced", REPORT, note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
