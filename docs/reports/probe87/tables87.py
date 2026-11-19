"""The tables of prompt 87's report, from the records in results/builder87/.

Prompt 87b added the developed-field timing (``developed_N.json``), the
binding's and the grid's planted refusals, and the tie-rule measurement
(``ties87.json``). The round-0 records are in ``results/builder87/round0/``.

    python docs/reports/probe87/tables87.py           # write tables87.md, splice the report
    python docs/reports/probe87/tables87.py --check   # regenerate and compare, write nothing

Run with Windows' ``.venv`` or the Linux environment: every record it reads is
a text file in the checkout, written under WSL2 by ``probe87.py`` (the parity
run's JUnit record, the captured replay, the timing), by ``mutants87.sh`` (the
planted defects) and by ``scripts/build_core.sh`` (CTest's JUnit record, copied
from the build tree). ``--check`` also fails when one of the report's marker
pairs is missing.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "results" / "builder87"
TABLES = HERE / "tables87.md"
REPORT = ROOT / "docs" / "reports" / "phase6_scalar_flux.md"
MARKERS = (
    "predictions",
    "tests",
    "ties",
    "parity",
    "weights",
    "replay",
    "plants",
    "refusals",
    "timing",
    "developed",
    "suites",
)
# The parity tests' meshes and options, in the order the tables list them
# (tests/test_core_parity.py, MESHES and VENT_MESHES).
MESHES = [
    "80x30",
    "160x60",
    "320x120",
    "160x60-stretched",
    "40x15-uniform-open",
    "80x30-vents",
    "160x60-vents",
    "320x120-vents",
]
FIELDS = [
    "random",
    "uniform",
    "levels",
    "steps-spikes",
    "signed-zeros",
    "negatives-and-zeros",
]
VENT_OPTIONS = {
    "none": "absent",
    "random-mask": "a random mask",
    "vents": "the vents file's",
}
FLUX_ID = re.compile(
    r"^(random|uniform|levels|steps-spikes|signed-zeros|negatives-and-zeros)"
    r"-(True|False)-([01])-(.+)-"
    r"(none|random-mask|vents)$"
)
TIMING_RUNS = (1, 2, 3)
# Prediction (b)'s factor.
SPEEDUP = 10.0
CAPTURES = {
    "dust80": "80x30 dust release, transport march",
    "dust320": "320x120 dust release, transport march",
    "flow80": "80x30 flow solve, k and eps",
}


def cases(path: Path) -> list[tuple[str, str, str]]:
    """(test function, parameter id, outcome) of every case in a JUnit record."""
    out = []
    for case in ET.parse(path).getroot().iter("testcase"):
        name = case.get("name", "")
        function, _, param = name.partition("[")
        outcome = "passed"
        if case.find("failure") is not None or case.find("error") is not None:
            outcome = "failed"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        out.append((function, param.rstrip("]"), outcome))
    return out


def ctest_counts(path: Path) -> tuple[int, int, dict[str, tuple[int, int]]]:
    """CTest's JUnit record: cases, failures, and (cases, passed) per GoogleTest suite."""
    suites: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total = failed = 0
    for case in ET.parse(path).getroot().iter("testcase"):
        total += 1
        suite = suites[case.get("name", "").split(".")[0]]
        suite[0] += 1
        if case.get("status") == "fail" or case.find("failure") is not None:
            failed += 1
        else:
            suite[1] += 1
    return total, failed, {name: (n, ok) for name, (n, ok) in suites.items()}


def flux_outcomes() -> dict[tuple[str, str, int, bool], list[str]]:
    """The advective_flux parity cases' outcomes by (mesh, vents, direction, upwind)."""
    table: dict[tuple[str, str, int, bool], list[str]] = defaultdict(list)
    for function, param, outcome in cases(RESULTS / "parity.xml"):
        if function != "test_advective_flux_equals_the_reference_bit_for_bit":
            continue
        match = FLUX_ID.match(param)
        if match is None:
            raise ValueError(f"unparsed parity id {param}")
        _kind, upwind, d, mesh, vents = match.groups()
        table[(mesh, vents, int(d), upwind == "True")].append(outcome)
    return table


def passed(outcomes: list[str]) -> str:
    """'n of m', or 'n of m, k FAILED'."""
    n = outcomes.count("passed")
    failed = len(outcomes) - n
    return f"{n} of {len(outcomes)}" + (f", {failed} FAILED" if failed else "")


def tests_table() -> list[str]:
    """T1: the core's GoogleTest tests and the parity tests, by suite and function."""
    total, failed, suites = ctest_counts(RESULTS / "ctest.xml")
    lines = [
        "### T1. The tests: CTest by GoogleTest suite, and the parity tests by function",
        "",
        "| Suite or test function | Cases | Passed |",
        "|---|---|---|",
    ]
    for suite, (n, ok) in suites.items():
        lines.append(f"| GoogleTest `{suite}` | {n} | {ok} |")
    lines.append(f"| **CTest, all** | {total} | {total - failed} |")
    by_function: dict[str, list[str]] = defaultdict(list)
    for function, _param, outcome in cases(RESULTS / "parity.xml"):
        by_function[function].append(outcome)
    for function, outcomes in by_function.items():
        lines.append(f"| `{function}` | {len(outcomes)} | {outcomes.count('passed')} |")
    every = [o for outcomes in by_function.values() for o in outcomes]
    lines.append(f"| **parity, all** | {len(every)} | {every.count('passed')} |")
    return lines


def parity_table() -> list[str]:
    """T2: advective_flux against the reference by mesh, vents, direction and upwind."""
    table = flux_outcomes()
    lines = [
        "### T2. `advective_flux` bit for bit against the reference, planted inputs",
        "",
        f"Each cell counts the planted fields ({', '.join(FIELDS)}) that equal the",
        "reference bit for bit on every face, of those run.",
        "",
        "| Mesh | Vents | x, QUICK | x, upwind | y, QUICK | y, upwind |",
        "|---|---|---|---|---|---|",
    ]
    for mesh in MESHES:
        for vents, label in VENT_OPTIONS.items():
            row = [
                table.get((mesh, vents, d, up), [])
                for d in (0, 1)
                for up in (False, True)
            ]
            if not any(row):
                continue
            lines.append(
                f"| {mesh} | {label} | " + " | ".join(passed(r) for r in row) + " |"
            )
    return lines


def weights_table() -> list[str]:
    """T3: the weights at setup, quick_face_values and the clamp, by mesh and direction."""
    by: dict[tuple[str, str], list[str]] = defaultdict(list)
    for function, param, outcome in cases(RESULTS / "parity.xml"):
        if function == "test_quick_weights_at_setup_equal_the_per_call_weights":
            d, mesh = param.split("-", 1)
            by[("weights", f"{mesh}|{d}")].append(outcome)
        elif function == "test_quick_face_values_equal_the_reference_bit_for_bit":
            _kind, d, mesh = re.match(
                r"^(random|levels|signed-zeros)-([01])-(.+)$", param
            ).groups()
            by[("quick", f"{mesh}|{d}")].append(outcome)
        elif function == "test_limited_face_values_equal_the_reference_bit_for_bit":
            by[("limited", param)].append(outcome)
    lines = [
        "### T3. The QUICK weights at set-up, `quick_face_values` and the clamp",
        "",
        "Weights: the six arrays equal bit for bit (one case per mesh and direction).",
        "`quick_face_values`: the planted fields (random, levels, signed zeros) whose",
        "interior face values equal the reference's bit for bit.",
        "",
        "| Mesh | Weights, x | Weights, y | `quick_face_values`, x | `quick_face_values`, y |",
        "|---|---|---|---|---|",
    ]
    for mesh in MESHES:
        cells = [
            by.get((kind, f"{mesh}|{d}"), [])
            for kind in ("weights", "quick")
            for d in (0, 1)
        ]
        if not any(cells):
            continue
        lines.append(f"| {mesh} | " + " | ".join(passed(c) for c in cells) + " |")
    shapes = sorted((k[1] for k in by if k[0] == "limited"), key=len)
    lines += [
        "",
        "`limited_face_values` on planted arrays (half random over twelve decades, half",
        "the levels -1, -0.0, 0.0, 2 and 1e-300), by shape: "
        + "; ".join(
            f"{s.replace('-', 'x')}: {passed(by[('limited', s)])}" for s in shapes
        )
        + ".",
    ]
    return lines


def replay_rows() -> list[dict]:
    """The captured replay's record."""
    return json.loads((RESULTS / "replay.json").read_text(encoding="utf-8"))


def replay_table() -> list[str]:
    """T4: every captured call replayed through the core, by capture, direction and option."""
    rows = replay_rows()
    groups: dict[tuple[str, int, bool, bool], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["capture"], row["d"], row["upwind"], row["vents"])].append(row)
    lines = [
        "### T4. The captured calls replayed through the core",
        "",
        "| Capture | Direction | Face value | Vents | Calls | Faces compared "
        "| Faces differing | Reference again, differing | Non-zero cells per call |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (capture, d, upwind, vents), group in sorted(groups.items()):
        nonzero = [r["nonzero_cells"] for r in group]
        lines.append(
            f"| {CAPTURES[capture]} | {'xy'[d]} | {'upwind' if upwind else 'QUICK'} "
            f"| {'yes' if vents else 'no'} | {len(group)} "
            f"| {sum(r['faces'] for r in group):,} | {sum(r['core_differs'] for r in group)} "
            f"| {sum(r['reference_again_differs'] for r in group)} "
            f"| {min(nonzero):,} to {max(nonzero):,} |"
        )
    lines.append(
        f"| **all** | | | | {len(rows)} | {sum(r['faces'] for r in rows):,} "
        f"| {sum(r['core_differs'] for r in rows)} "
        f"| {sum(r['reference_again_differs'] for r in rows)} | |"
    )
    return lines


def plants() -> list[dict[str, str]]:
    """mutants87.txt, one dict per line: name, GoogleTest and parity."""
    out = []
    for line in (RESULTS / "mutants87.txt").read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split(" | ")]
        entry = {"name": parts[0]}
        for part in parts[1:]:
            key, _, value = part.partition(": ")
            # The run time varies from run to run and is not a finding.
            entry[key] = re.sub(r" in [0-9.]+s", "", value)
        out.append(entry)
    return out


def is_refusal(name: str) -> bool:
    """Whether a plant removes one of the binding's or the grid's checks."""
    return name.startswith(("bind_", "grid_")) or name == "control_end"


def plant_rows(refusals: bool) -> list[str]:
    """The table rows of the planted defects of one group, in the script's order."""
    return [
        f"| `{entry['name']}` | {entry.get('GoogleTest', entry.get('not planted', ''))} "
        f"| {entry.get('parity', '')} |"
        for entry in plants()
        if is_refusal(entry["name"]) == refusals
    ]


def plants_table() -> list[str]:
    """T5: the scalar module's planted defects and which tests catch each."""
    return [
        "### T5. The planted defects: which tests catch each",
        "",
        "| Defect | GoogleTest (CTest) | Parity (pytest) |",
        "|---|---|---|",
        *plant_rows(refusals=False),
    ]


def refusals_table() -> list[str]:
    """T5b: the binding's and the grid's planted refusals and which tests catch each."""
    return [
        "### T5b. The binding's and the grid's checks, each removed: which tests catch it",
        "",
        "Each `bind_` and `grid_` defect removes one check (a shape comparison, "
        "`nb::c_contig` of one array alias, an alignment test, an aliasing test, "
        "a clause of the coordinate check). `control_end` is the unchanged source "
        "built again after the last plant.",
        "",
        "| Defect | GoogleTest (CTest) | Parity (pytest) |",
        "|---|---|---|",
        *plant_rows(refusals=True),
    ]


def timings() -> dict[int, dict]:
    """The three timing runs' records; run 1 is timing.json, the one scored."""
    return {
        n: json.loads((RESULTS / f"timing_{n}.json").read_text(encoding="utf-8"))
        for n in TIMING_RUNS
    }


def timing_table() -> list[str]:
    """T6: the core against the reference per call on the captured 320x120 inputs."""
    lines = [
        "### T6. `advective_flux` per call on captured 320x120 inputs, one thread",
        "",
        "Medians of repeated calls under Linux; the core through the binding with its",
        "output allocated once, the reference as the transport solver calls it (the y",
        "call's transposes included). The empty call is `advective_flux_overhead`: the",
        "binding's argument conversion and checks with the same arguments. Run 1 is the",
        "one scored; runs 2 and 3 repeat it.",
        "",
        "| Run | Direction | Faces | Core, us | Empty call, us | Reference, us | Reference / core |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, record in timings().items():
        for row in record["rows"]:
            lines.append(
                f"| {n} | {'xy'[row['d']]} | {row['faces']:,} "
                f"| {row['core_median_s'] * 1e6:.1f} | {row['overhead_median_s'] * 1e6:.2f} "
                f"| {row['reference_median_s'] * 1e6:.0f} | {row['ratio']:.2f} |"
            )
    first = timings()[1]
    lines += [
        "",
        f"Repeats: the core {first['rows'][0]['core_repeats']}, the reference "
        f"{first['rows'][0]['reference_repeats']}, after five unrecorded calls; "
        f"OPENBLAS_NUM_THREADS={first['openblas_num_threads']}, Python {first['python']}, "
        f"NumPy {first['numpy']}.",
    ]
    return lines


def suite_counts(path: Path) -> dict[str, float]:
    """The totals of a pytest JUnit record: tests, failures, errors, skipped and seconds."""
    suite = ET.parse(path).getroot().find("testsuite")
    return {
        key: float(suite.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped", "time")
    }


def suites_table() -> list[str]:
    """T8: the whole suite on each platform, from pytest's JUnit records."""
    lines = [
        "### T8. The whole suite at the round 1 head",
        "",
        "The two suites ran at the same time, one BLAS thread each, so the wall times",
        "are not those of a suite alone.",
        "",
        "| Platform | Tests | Passed | Failed | Errors | Skipped | Wall time, s |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, name in (
        ("Windows, `.venv`, the core not built", "suite_windows.xml"),
        ("WSL2, the core built, `CFD_CORE_REQUIRED=1`", "suite_linux.xml"),
    ):
        c = suite_counts(RESULTS / name)
        passed = c["tests"] - c["failures"] - c["errors"] - c["skipped"]
        lines.append(
            f"| {label} | {c['tests']:,.0f} | {passed:,.0f} | {c['failures']:.0f} "
            f"| {c['errors']:.0f} | {c['skipped']:.0f} | {c['time']:,.0f} |"
        )
    return lines


def developed_records() -> dict[int, dict]:
    """The three developed-field timing runs' records."""
    return {
        n: json.loads((RESULTS / f"developed_{n}.json").read_text(encoding="utf-8"))
        for n in TIMING_RUNS
    }


def developed_table() -> list[str]:
    """T6b: the core against the reference on developed fields, beside the captured one."""
    runs = developed_records()
    first = runs[1]
    lines = [
        "### T6b. `advective_flux` per call on the stored run's last frame, one thread",
        "",
        "The field is the stored 320x120 run's last frame of each class "
        f"(`{first['stored_run']}`), under that class's last captured flux and inflow. "
        "The captured rows are the scored call, timed again in the same session. Medians "
        "of repeated calls under Linux, as in T6; times are run 1's, the ratio "
        "(reference / core) is given for each run.",
        "",
        "| Field | Class | Direction | Non-zero cells | Core, us | Empty call, us "
        "| Reference, us | Ratio, run 1 | Ratio, run 2 | Ratio, run 3 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(first["rows"]):
        ratios = " | ".join(f"{runs[n]['rows'][i]['ratio']:.2f}" for n in TIMING_RUNS)
        lines.append(
            f"| {row['field']} | {row['class']} | {'xy'[row['d']]} | {row['nonzero_cells']:,} "
            f"| {row['core_median_s'] * 1e6:.1f} | {row['overhead_median_s'] * 1e6:.2f} "
            f"| {row['reference_median_s'] * 1e6:.0f} | {ratios} |"
        )
    return lines


def developed_range(d: int, run: int = 1) -> tuple[float, float]:
    """The least and greatest reference / core ratio over the developed rows of a direction."""
    ratios = [
        row["ratio"]
        for row in developed_records()[run]["rows"]
        if row["field"] == "stored last frame" and row["d"] == d
    ]
    return min(ratios), max(ratios)


def ties_table() -> list[str]:
    """T7: NumPy's tie and NaN rules at every length, from ties87.py's record."""
    record = json.loads((RESULTS / "ties87.json").read_text(encoding="utf-8"))
    first, last = record["lengths"]
    lines = [
        "### T7. NumPy's tie and NaN rules, measured by `ties87.py`",
        "",
        f"NumPy {record['numpy']}. Each check runs at every length from {first} to "
        f"{last:,} and on the shapes {', '.join('x'.join(map(str, s)) for s in record['shapes'])}.",
        "",
        "| Check | Shapes run | Departing from the rule |",
        "|---|---|---|",
    ]
    for row in record["rows"]:
        lines.append(f"| {row['check']} | {row['cases']:,} | {row['departures']} |")
    return lines


def predictions_table() -> list[str]:
    """T0: the predictions, scored by the rule committed with them."""
    flux = [o for outcomes in flux_outcomes().values() for o in outcomes]
    others = [
        outcome
        for function, _p, outcome in cases(RESULTS / "parity.xml")
        if function
        in (
            "test_quick_face_values_equal_the_reference_bit_for_bit",
            "test_limited_face_values_equal_the_reference_bit_for_bit",
        )
    ]
    weights = [
        outcome
        for function, _p, outcome in cases(RESULTS / "parity.xml")
        if function == "test_quick_weights_at_setup_equal_the_per_call_weights"
    ]
    rows = replay_rows()
    replay_differ = sum(r["core_differs"] for r in rows)
    a_holds = all(o == "passed" for o in flux + others) and replay_differ == 0
    ratios = {row["d"]: row["ratio"] for row in timings()[1]["rows"]}
    b_holds = all(r >= SPEEDUP for r in ratios.values())
    c_holds = all(o == "passed" for o in weights)
    x_dev, y_dev = developed_range(0), developed_range(1)
    return [
        "### T0. The predictions, scored by the rule committed with them",
        "",
        "| Prediction | Measured | Verdict |",
        "|---|---|---|",
        f"| (a) every ported function bit for bit, planted and captured "
        f"| planted: `advective_flux` {passed(flux)} cases, `quick_face_values` and the "
        f"clamp {passed(others)}; captured: {len(rows)} calls, "
        f"{sum(r['faces'] for r in rows):,} faces, {replay_differ} differing "
        f"| {'holds' if a_holds else 'FAILS'} |",
        f"| (b) the core at least ten times faster per call on 320x120 "
        f"| x {ratios[0]:.2f} times, y {ratios[1]:.2f} times (T6, run 1); "
        f"on the stored run's developed fields, x {x_dev[0]:.2f} to {x_dev[1]:.2f} times, "
        f"y {y_dev[0]:.2f} to {y_dev[1]:.2f} times (T6b, run 1, not scored) "
        f"| {'holds' if b_holds else 'FAILS'} |",
        f"| (c) the weights at set-up equal the per-call weights "
        f"| {passed(weights)} mesh and direction cases, six arrays each "
        f"| {'holds' if c_holds else 'FAILS'} |",
    ]


def build() -> tuple[str, dict[str, str]]:
    """tables87.md's text and the report's blocks."""
    builders = {
        "predictions": predictions_table,
        "tests": tests_table,
        "parity": parity_table,
        "weights": weights_table,
        "replay": replay_table,
        "plants": plants_table,
        "refusals": refusals_table,
        "ties": ties_table,
        "timing": timing_table,
        "developed": developed_table,
        "suites": suites_table,
    }
    blocks = {name: "\n".join(builders[name]()) for name in MARKERS}
    text = "# Prompt 87's tables (generated by tables87.py; do not edit)\n\n"
    text += "\n\n".join(blocks[name] for name in MARKERS)
    return text + "\n", blocks


def splice(report: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """REPORT with each marked block replaced, and the names whose markers are missing."""
    missing = []
    for name, body in blocks.items():
        begin, end = f"<!-- tables87 {name} begin -->", f"<!-- tables87 {name} end -->"
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
    if args.check:
        same_tables = TABLES.exists() and TABLES.read_text(encoding="utf-8") == text
        ok = same_tables and spliced == report and not missing
        if missing:
            print(f"missing markers: {', '.join(missing)}")
        if not same_tables:
            print(f"{TABLES.name} differs from the records")
        if spliced != report:
            print(f"{REPORT.name}'s tables differ from the records")
        print("tables87 --check:", "ok" if ok else "FAILED")
        return 0 if ok else 1
    TABLES.write_text(text, encoding="utf-8", newline="\n")
    REPORT.write_text(spliced, encoding="utf-8", newline="\n")
    if missing:
        print(f"missing markers (not spliced): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
