"""The tables of prompt 92's momentum report, from results/builder92/ and builder92b/.

    python docs/reports/probe92/tables92.py           # write tables92.md, splice the report
    python docs/reports/probe92/tables92.py --check   # regenerate and compare, write nothing

Run with Windows' ``.venv``: every record it reads is in the checkout. T2, T5, T5b and T6
are prompt 92's records, of 6ca02d7; T1, T1b, T3, T4, T7 and T8 are prompt 92b's, of the
module written per direction, in ``results/builder92b/``.
``--check`` also fails when one of the report's marker pairs is missing.
"""

import argparse
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLES = HERE / "tables92.md"
REPORT = ROOT / "docs" / "reports" / "phase6_momentum.md"
RESULTS = ROOT / "results" / "builder92"
CURRENT = ROOT / "results" / "builder92b"
# The sessions of the kernels' timing and the ratio above which one is slower.
SESSIONS = (1, 2, 3)
SLOWER = 1.05
SPEED = 5.0
CASES = {
    "product80": "80x30 product room",
    "vents80": "80x30 vents room",
    "cavity40": "40x40 cavity",
    "annex216": "216x72 Annex 20 room",
}
OUTPUTS = {"u_star": "u*", "v_star": "v*", "a_p_u": "a_P u", "a_p_v": "a_P v"}
PART_NAMES = {
    "one_sweep": "predict with one sweep",
    "scalar_path": "predict on the scalar path, ten sweeps",
    "stencil_viscosity": "stencil_viscosity",
}
FORM_PARTS = {
    "predict": "predict, ten sweeps",
    "one_sweep": "one sweep",
    "scalar_path": "scalar path, ten sweeps",
    "scalar_one_sweep": "scalar path, one sweep",
    "stencil_viscosity": "stencil_viscosity",
}


def table(header: list[str], rows: list[list[object]]) -> list[str]:
    """A markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def load(name: str, root: Path = RESULTS) -> object:
    """A JSON record, of results/builder92/ unless another root is given."""
    return json.loads((root / name).read_text(encoding="utf-8"))


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


# ---------------------------------------------------------------------------


def tests_tables() -> tuple[list[str], list[str], dict[str, list[str]]]:
    """T1: the new parity cases by function; T1b: CTest and the whole suites."""
    cases = junit_cases(CURRENT / "parity.xml")
    by_function: dict[str, list[str]] = {}
    for cls, name, outcome in cases:
        if cls.endswith("test_core_momentum"):
            by_function.setdefault(name.split("[")[0], []).append(outcome)
    rows = [[f"`{f}`", *counts(o)] for f, o in by_function.items()]
    rows.append(
        [
            "`tests/test_core_momentum.py`",
            *counts([o for outcomes in by_function.values() for o in outcomes]),
        ]
    )
    functions = table(["Test", "Cases", "Passed", "Failed", "Skipped"], rows)

    ctest = junit_cases(CURRENT / "ctest.xml")
    suites: dict[str, list[str]] = {}
    for _cls, name, outcome in ctest:
        suites.setdefault(name.split(".")[0], []).append(outcome)
    rows = [
        [f"CTest, suite `{suite}`", *counts(o)]
        for suite, o in suites.items()
        if suite.startswith("Momentum")
    ]
    rows.append(["CTest, all suites", *counts([c[2] for c in ctest])])
    for label, name in (
        (
            "the three parity modules, WSL2, `CFD_CORE_REQUIRED=1`",
            "parity_modules.xml",
        ),
        ("the whole suite, WSL2, the core built", "suite_linux.xml"),
        ("the whole suite, Windows, the core not built", "suite_windows.xml"),
    ):
        rows.append([label, *counts([c[2] for c in junit_cases(CURRENT / name)])])
    return (
        functions,
        table(["Tests", "Cases", "Passed", "Failed", "Skipped"], rows),
        by_function,
    )


def captures_table() -> list[str]:
    """T2: the captured cases."""
    rows = []
    for case, label in CASES.items():
        c = load(f"captures/{case}.json")
        start = c["start"]
        source = f"`{start['from']}`" if "/" in start["from"] else start["from"]
        if "outer" in start:
            source += f" ({start['outer']:,} outer, {start['stop_reason']})"
        rows.append(
            [
                label,
                source,
                c["calls"],
                c["distinct_calls"],
                c["sweeps"],
                "yes" if c["field"] else "no",
                "yes" if c["wall"] else "no",
                f"{c['largest_step_u']:.1e}",
            ]
        )
    return table(
        [
            "Case",
            "Warm-started from",
            "predict calls",
            "Distinct",
            "Sweeps",
            "Viscosity field",
            "Wall viscosity",
            "Largest change of u over the calls, m/s",
        ],
        rows,
    )


def replay_table() -> tuple[list[str], bool, int]:
    """T3: the captured calls replayed, by case and output."""
    records = load("replay.json", CURRENT)
    rows = []
    clean = True
    total = 0
    for case, label in CASES.items():
        for what, outputs in (
            ("predict", OUTPUTS),
            ("stencil_viscosity", {"u": "u's faces", "v": "v's faces"}),
        ):
            mine = [r for r in records if r["case"] == case and r["what"] == what]
            if not mine:
                continue
            total += len(mine)
            again = sum(1 for r in mine if any(r["reference_differs"].values()))
            for key, name in outputs.items():
                differ = [r["differ"][key] for r in mine]
                clean = clean and not any(differ) and again == 0
                rows.append(
                    [
                        label,
                        f"`{what}`",
                        name,
                        len(mine),
                        sum(1 for d in differ if d),
                        sum(differ),
                        again,
                        sum(r.get("vent_faces_moving", 0) for r in mine)
                        if what == "predict"
                        else "-",
                    ]
                )
    return (
        table(
            [
                "Case",
                "Call",
                "Output",
                "Calls",
                "Calls with a differing bit",
                "Entries that differ",
                "Calls the reference does not reproduce",
                "Moving vent faces",
            ],
            rows,
        ),
        clean,
        total,
    )


def plant_records() -> list[dict]:
    """Every stream's plants, one control, in variants92's order."""
    sys.path.insert(0, str(HERE))
    import variants92

    records: dict[str, dict] = {}
    for path in sorted(CURRENT.glob("plants_s*.json")):
        for record in json.loads(path.read_text(encoding="utf-8")):
            records[record["name"]] = record
    return [records[name] for name, _what, _edits in variants92.PLANTS]


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
        if not re.search(r" 0 differ, 0 not", line)
    ]
    if differ:
        nets.append(f"replay ({', '.join(differ)})")
    elif len(record["replay"]) != len(CASES):
        nets.append("replay (raises)")
    return nets


def plants_table() -> tuple[list[str], int, int, list[str]]:
    """T4: the planted defects."""
    sys.path.insert(0, str(HERE))
    import variants92

    equivalent = getattr(variants92, "EQUIVALENT", {})
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
            verdict = "nothing"
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


def timing_tables() -> tuple[list[str], list[str], dict[int, float]]:
    """T5: predict per call; T5b: the parts; the ratios by run."""
    rows, parts, ratios = [], [], {}
    for run in (1, 2, 3):
        t = load(f"timing_{run}.json")
        ratios[run] = t["ratio"]
        rows.append(
            [
                run,
                t["started"],
                f"{t['reference']['median_s'] * 1e3:.2f} ({t['reference']['q1_s'] * 1e3:.2f} to {t['reference']['q3_s'] * 1e3:.2f})",
                f"{t['core']['median_s'] * 1e3:.3f} ({t['core']['q1_s'] * 1e3:.3f} to {t['core']['q3_s'] * 1e3:.3f})",
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
                    f"{part['reference']['median_s'] * 1e3:.3f}",
                    f"{part['core']['median_s'] * 1e3:.3f}",
                    f"{part['reference']['median_s'] / part['core']['median_s']:.2f}",
                ]
            )
    return (
        table(
            [
                "Run",
                "Started",
                "Reference, ms (quartiles)",
                "Core, ms (quartiles)",
                "Ratio",
                "Calibration before and after, us",
                "Windows before",
                "Windows after",
            ],
            rows,
        ),
        table(["Run", "Call", "Reference, ms", "Core, ms", "Ratio"], parts),
        ratios,
    )


def forms_table() -> list[str]:
    """T6: the kernels' forms, medians of the rounds' medians and the ratio to the head's."""
    record = load("forms_1.json")
    rows = []
    for name, what in record["what"].items():
        cells: list[object] = [f"`{name}`", what]
        for part in FORM_PARTS:
            times = [r[name][part]["median_s"] for r in record["rounds"]]
            ratios = [
                r[name][part]["median_s"] / r["head"][part]["median_s"]
                for r in record["rounds"]
            ]
            cells.append(
                f"{statistics.median(times) * 1e6:,.0f}"
                + (
                    ""
                    if name == "head"
                    else f" ({min(ratios):.2f} to {max(ratios):.2f})"
                )
            )
        same = not any(any(r[name]["differ"].values()) for r in record["rounds"])
        cells.append("yes" if same else "no")
        rows.append(cells)
    state = [
        "",
        f"Started {record['started']}. Windows before: {load_text(record['windows_before'])}; "
        f"after: {load_text(record['windows_after'])}.",
    ]
    return (
        table(
            [
                "Form",
                "What",
                *(f"{v}, us" for v in FORM_PARTS.values()),
                "Bit for bit",
            ],
            rows,
        )
        + state
    )


def loops_table() -> list[str]:
    """T7: the blocks of momentum.cpp and whether GCC vectorises each."""
    record = load("vecinfo.json", CURRENT)
    rows = [
        [f"`{b['block']}`", "yes" if b["vectorised"] else "no"]
        for b in record["blocks"]
    ]
    return [
        *table(["Block", "Vectorised"], rows),
        "",
        f"{record['compiler']}; {' '.join(record['flags'])}. "
        f"{sum(b['vectorised'] for b in record['blocks'])} of {len(record['blocks'])} blocks "
        f"vectorised, every one with vectors of "
        f"{', '.join(sorted(record['widths']))} bytes. With `np_maximum` (T6) the blocks "
        "left scalar: "
        + ", ".join(
            f"`{b['block']}`"
            for b in record["np_maximum"]["blocks"]
            if not b["vectorised"]
        )
        + ".",
    ]


def kernels_table() -> list[str]:
    """T8: each kernel and pass before and after the restructure, by session."""
    sessions = [load(f"kernels_{s}.json", CURRENT) for s in SESSIONS]
    rows = []
    slower = []
    for key in sessions[0]["rounds"][0]["before"]:
        ratios = []
        times: dict[str, list[float]] = {"before": [], "after": []}
        for session in sessions:
            medians = {}
            for build in times:
                values = [r[build][key] for r in session["rounds"]]
                times[build] += values
                medians[build] = statistics.median(values)
            ratios.append(medians["after"] / medians["before"])
        if min(ratios) > SLOWER:
            slower.append(key)
        rows.append(
            [
                key,
                f"{statistics.median(times['before']):.1f}",
                f"{statistics.median(times['after']):.1f}",
                ", ".join(f"{r:.3f}" for r in ratios),
                f"{statistics.median(times['after']) / statistics.median(times['before']):.3f}",
            ]
        )
    state = "; ".join(
        f"session {s} started {session['started']} (Windows before: "
        f"{load_text(session['windows_before'])})"
        for s, session in zip(SESSIONS, sessions, strict=True)
    )
    return [
        *table(
            [
                "Kernel or pass",
                "6ca02d7, us",
                "Per direction, us",
                "Ratio by session",
                "Ratio, all rounds",
            ],
            rows,
        ),
        "",
        f"Above {SLOWER} in every session: "
        + (", ".join(f"`{k}`" for k in slower) if slower else "none")
        + f". {state}.",
    ]


def predictions_table(
    functions: dict[str, list[str]],
    replay_clean: bool,
    replayed: int,
    ratios: dict[int, float],
) -> list[str]:
    """T0: the predictions by the committed rule."""
    fixed = [
        o
        for f, outcomes in functions.items()
        if f.startswith("test_the_fixed_quantities")
        for o in outcomes
    ]
    planted = [
        o
        for f, outcomes in functions.items()
        if f
        in (
            "test_predict_equals_the_reference_bit_for_bit",
            "test_predict_with_planted_edges_equals_the_reference_bit_for_bit",
            "test_the_sweeps_keep_the_vents_values",
            "test_predict_is_bit_for_bit_at_every_row_length",
            "test_one_predictor_through_many_calls_is_each_calls_own",
            "test_a_nan_lands_where_the_references_does",
        )
        for o in outcomes
    ]
    records = load("replay.json", CURRENT)
    predicts = [r for r in records if r["what"] == "predict"]
    clean = [r for r in predicts if not any(r["differ"].values())]
    a_holds = replay_clean and planted.count("passed") == len(planted) > 0
    b_holds = fixed.count("passed") == len(fixed) > 0
    c_holds = ratios[1] >= SPEED
    return table(
        ["Prediction", "Measured", "Verdict"],
        [
            [
                "(a) every planted and captured `predict` call equals the reference's bit for bit",
                f"{len(clean)} of {len(predicts)} captured `predict` calls with no differing bit in "
                f"the four outputs (T3, with {replayed - len(predicts)} `stencil_viscosity` calls "
                f"beside them); {planted.count('passed'):,} of {len(planted):,} planted `predict` "
                "cases pass, each holding one call or several (T1)",
                "holds" if a_holds else "fails",
            ],
            [
                "(b) the set-up's fixed quantities equal the reference's per-call values on every mesh",
                f"{fixed.count('passed')} of {len(fixed)} cases pass, nine meshes under two "
                "boundaries and the small grids of every row length (T1)",
                "holds" if b_holds else "fails",
            ],
            [
                "(c) `predict` on the 320x120 state at least five times faster per call",
                f"{ratios[1]:.2f} times in run 1 (run 2: {ratios[2]:.2f}; run 3: {ratios[3]:.2f}) (T5)",
                "holds" if c_holds else "fails",
            ],
        ],
    )


# ---------------------------------------------------------------------------


def build() -> tuple[str, dict[str, str]]:
    """The tables' file and the report's blocks."""
    functions, totals, by_function = tests_tables()
    replayed, clean, calls = replay_table()
    plants, planted, caught, survivors = plants_table()
    timing, parts, ratios = timing_tables()
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
            predictions_table(by_function, clean, calls, ratios),
        ),
        ("parity", "T1. The parity cases by function", functions),
        ("tests", "T1b. CTest, the parity modules and the suites", totals),
        ("captures", "T2. The captured calls", captures_table()),
        (
            "replay",
            "T3. The captured calls replayed through the core, by output",
            replayed,
        ),
        ("plants", "T4. The planted defects: which nets fail on each", plants + count),
        (
            "timing",
            "T5. `predict` per call on the 320x120 product state, one thread",
            timing,
        ),
        ("parts", "T5b. The parts, the same sessions", parts),
        (
            "forms",
            "T6. The kernels' forms on the same call (the ratio to the head's over the three rounds)",
            forms_table(),
        ),
        ("loops", "T7. The blocks GCC vectorises", loops_table()),
        (
            "kernels",
            "T8. Each kernel and pass, the build of 6ca02d7 against the per-direction one",
            kernels_table(),
        ),
    ]
    blocks = {
        name: "\n".join([f"### {title}", "", *lines]) for name, title, lines in sections
    }
    text = "# Prompt 92's tables\n\n" + "\n\n".join(blocks.values()) + "\n"
    return text, blocks


def splice(report: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """The report with each marked block replaced, and the names whose markers are missing."""
    missing = []
    for name, body in blocks.items():
        begin = f"<!-- tables92 {name} begin -->"
        end = f"<!-- tables92 {name} end -->"
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
        print("tables92 --check:", "matches" if ok else "DIFFERS", note)
        return 0 if ok else 1
    TABLES.write_text(text, encoding="utf-8")
    REPORT.write_text(spliced, encoding="utf-8")
    print("wrote", TABLES, "and spliced", REPORT, note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
