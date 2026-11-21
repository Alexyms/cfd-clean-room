"""Prompt 90's tables, generated from the records in ``results/builder90/``.

    python docs/reports/probe90/tables90.py           # write tables90.md and splice the report
    python docs/reports/probe90/tables90.py --check   # regenerate and compare, write nothing

The records are untracked and on this machine: the captures' descriptions
(``captures/*.json``), the replay (``replay.json``), the parity module's and
CTest's JUnit records (``parity.xml``, ``ctest.xml``), the planted defects
(``plants.json``), the three timing sessions (``timing_N.json``), the
vectorisation report (``vecinfo.json``), the vents flow state's solve
(``vents80_flow.json``) and the two suites' logs. Each table goes into
``docs/reports/phase6_class_step.md`` between its markers, and the full set
into ``tables90.md``.
"""

import argparse
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "results" / "builder90"
REPORT = ROOT / "docs" / "reports" / "phase6_class_step.md"
TABLES = HERE / "tables90.md"
SCENARIOS = ("dust80", "door80", "breach80", "vents80", "dust320")
SCENARIO_NAMES = {
    "dust80": "80x30 dust release",
    "door80": "80x30 door leak",
    "breach80": "80x30 filter breach",
    "vents80": "80x30 vents dust release",
    "dust320": "320x120 dust release",
}
MARKERS = ("predictions", "captures", "replay", "tests", "plants", "timing", "loops")
# The records were taken at this commit: the last that changes core/ or the tests.
RECORDS_AT = "969dca9"
RATIO_REQUIRED = 5.0


def load(name: str) -> object:
    """A JSON record of results/builder90/."""
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def captures_table() -> list[str]:
    """T1: what each capture holds."""
    lines = [
        "### T1. The captured class steps",
        "",
        "Each run from its stored flow state (prompt 81's runs; the vents run solved from rest), "
        "every class's steps in two windows: across the onset (two steps before it, three after) "
        "and the run's last three. A solver is one interval's `TransportSolver`.",
        "",
        "| Scenario | Grid | Flow states from | Steps run | Calls kept | Solvers | Vents | Wall time, s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in SCENARIOS:
        path = RESULTS / "captures" / f"{name}.json"
        if not path.exists():
            lines.append(f"| {SCENARIO_NAMES[name]} | | not captured | | | | | |")
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        ny, nx = meta["shape"]
        source = meta["flows_from"] or "solved from rest"
        vents = "yes" if any(s["vents"] for s in meta["solvers"]) else "no"
        lines.append(
            f"| {SCENARIO_NAMES[name]} | {nx}x{ny} | `{source}` | {meta['steps']:,} | "
            f"{len(meta['calls'])} | {len(meta['solvers'])} | {vents} | {meta['wall_seconds']:.0f} |"
        )
    flow = RESULTS / "vents80_flow.json"
    if flow.exists():
        record = json.loads(flow.read_text(encoding="utf-8"))
        lines += [
            "",
            f"The vents run's flow state, solved from rest with `max_simple_iter` raised to "
            f"{record['max_simple_iter']:,}: stopped by `{record['stop_reason']}` after "
            f"{record['outer']:,} outer iterations in {record['wall_seconds']:.0f} s "
            f"(`results/builder90/vents80_flow.json`); the base file's cap is 15,000.",
        ]
    return lines


def replay_rows() -> list[dict]:
    """The replay's rows."""
    rows = load("replay.json")
    assert isinstance(rows, list)
    return rows


def differs(row: dict) -> bool:
    """Whether any output of a replayed call differs."""
    return bool(
        row["field_differ"]
        or row["terms_differ"]
        or row["records_differ"]
        or row["interval_differ"]
        or not row["breach_equal"]
        or not row["reference_reproduces"]
        or not row["inputs_match"]
    )


def replay_table() -> list[str]:
    """T2: the replay by scenario, window and class."""
    lines = [
        "### T2. Every captured call replayed through the core",
        "",
        "Per scenario, window and class: the calls, the cells per call, the Jacobi sweeps the "
        "reference took, and what differs from the reference in any bit: the field, the ten "
        "budget terms, the step's records (sweeps, converged, leftover, bound), the interval's "
        "quantities against the reference's per-step values (prediction (b)), and whether the "
        "reference called again on the captured inputs reproduces the capture and the kept "
        "interval inputs are the call's own.",
        "",
        "| Scenario | Window | Class | Calls | Cells | Sweeps | Vents, recirculated | Source | Field bits differing | Budget terms differing | Records differing | Interval quantities differing | Reference reproduces, inputs the call's |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in replay_rows():
        groups[(row["scenario"], row["window"], row["class"])].append(row)
    for name in SCENARIOS:
        for window in ("onset", "late"):
            for k in range(5):
                rows = groups.get((name, window, k))
                if not rows:
                    continue
                sweeps = sorted({r["sweeps"] for r in rows})
                span = (
                    f"{sweeps[0]}" if len(sweeps) == 1 else f"{sweeps[0]}-{sweeps[-1]}"
                )
                vents = (
                    (
                        "yes, "
                        + ("yes" if any(r["recirculated"] for r in rows) else "no")
                    )
                    if any(r["vents"] for r in rows)
                    else "no"
                )
                source = "yes" if any(r["sources"] for r in rows) else "no"
                field = sum(max(r["field_differ"], 0) for r in rows)
                terms = sum(len(r["terms_differ"]) for r in rows)
                records = sum(len(r["records_differ"]) for r in rows)
                interval = sum(len(r["interval_differ"]) for r in rows)
                ok = all(r["reference_reproduces"] and r["inputs_match"] for r in rows)
                lines.append(
                    f"| {SCENARIO_NAMES[name]} | {window} | {k} | {len(rows)} | "
                    f"{rows[0]['cells']:,} | {span} | {vents} | {source} | {field} | {terms} | "
                    f"{records} | {interval} | {'yes' if ok else 'NO'} |"
                )
    rows = replay_rows()
    bad = [r for r in rows if differs(r)]
    breaches = sum(r["breach"] for r in rows)
    lines += [
        "",
        f"{len(rows)} calls, {sum(r['cells'] for r in rows):,} cell values compared, "
        f"{len(bad)} with any difference; {breaches} raised a breach.",
    ]
    return lines


def junit(path: Path) -> list[tuple[str, str]]:
    """Each test case of a JUnit record: its name and its outcome."""
    root = ET.parse(path).getroot()
    out = []
    for case in root.iter("testcase"):
        outcome = "passed"
        for child in case:
            if child.tag in ("failure", "error"):
                outcome = "failed"
            elif child.tag == "skipped":
                outcome = "skipped"
        out.append((case.get("name", ""), outcome))
    return out


PARITY_GROUPS = {
    "test_class_step_equals_the_reference_bit_for_bit": "planted class steps: field, budget, records, raise",
    "test_the_planted_class_steps_reach_every_rule": "the planted inputs reach every rule",
    "test_class_step_continues_a_restored_budget_bit_for_bit": "a budget restored mid-run",
    "test_class_step_reads_and_sets_budgets_whole": "budgets read and set whole, restore's refusals",
    "test_class_step_raises_the_references_breach": "the breach raised",
    "test_class_step_breaches_at_equality_with_the_limit": "the breach at equality with the limit",
    "test_class_step_interval_quantities_equal_the_per_step_values": "interval quantities against the per-step values",
    "test_class_step_refuses_what_the_reference_refuses": "the refusals",
    "test_buffered_sum_equals_np_sum_on_strided_slices": "NumPy's buffered order on strided slices",
}


def tests_table() -> list[str]:
    """T3: the new tests and the suites, at the records' commit."""
    lines = [
        "### T3. The tests",
        "",
        f"At {RECORDS_AT}, under the project's flags. The parity module's new cases by test "
        "function, then the totals.",
        "",
        "| Test | Cases | Passed | Failed | Skipped |",
        "|---|---|---|---|---|",
    ]
    cases = junit(RESULTS / "parity.xml")
    by_function: dict[str, Counter] = defaultdict(Counter)
    for name, outcome in cases:
        by_function[name.split("[")[0]][outcome] += 1
    for function, what in PARITY_GROUPS.items():
        c = by_function.get(function, Counter())
        lines.append(
            f"| `{function}` ({what}) | {sum(c.values())} | {c['passed']} | {c['failed']} | {c['skipped']} |"
        )
    total = Counter(outcome for _, outcome in cases)
    new = sum(sum(by_function[f].values()) for f in PARITY_GROUPS)
    ctest = junit(RESULTS / "ctest.xml")
    ctest_total = Counter(outcome for _, outcome in ctest)
    transport = [n for n, _ in ctest if n.startswith("TransportStep.")]
    held = [n for n, _ in ctest if "Held" in n and "Length" in n]
    lines += [
        f"| the whole parity module | {len(cases)} | {total['passed']} | {total['failed']} | {total['skipped']} |",
        f"| CTest | {len(ctest)} | {ctest_total['passed']} | {ctest_total['failed']} | {ctest_total['skipped']} |",
        "",
        f"New: {new} parity cases; {len(transport)} GoogleTest cases of `TransportStep` "
        f"({', '.join(n.split('.', 1)[1] for n in transport)}) and {len(held)} for `held`'s "
        f"length ({', '.join(held)}).",
    ]
    for label, log in (("Linux", "suite_linux.log"), ("Windows", "suite_windows.log")):
        path = RESULTS / log
        if path.exists():
            summary = [
                line.strip("= ")
                for line in path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if re.search(r"(passed|failed).* in [0-9.]+s", line)
            ]
            if summary:
                lines.append(f"- The whole suite under {label}: {summary[-1]}.")
    return lines


def plants_table() -> list[str]:
    """T4: the planted defects and what catches each."""
    lines = [
        "### T4. The planted defects: which tests catch each",
        "",
        "Each built alone from the checkout's `core/` with one edit "
        "(`docs/reports/probe90/variants90.py`), then CTest, the parity module's class-step and "
        "buffered-sum cases, and the replay of every capture.",
        "",
        "| Plant | What it changes | CTest | Parity cases failing (by test) | Replay: calls differing |",
        "|---|---|---|---|---|",
    ]
    for record in load("plants.json"):
        if "build" in record:
            lines.append(
                f"| `{record['name']}` | {record['what']} | build failed | | |"
            )
            continue
        failed = record["ctest_failed"]
        ctest = record["ctest"].replace("tests passed, ", "passed, ")
        if failed:
            ctest += ": " + ", ".join(f"`{n}`" for n in failed)
        parity = (
            ", ".join(f"`{t}` {n}" for t, n in record["parity_failed_by_test"].items())
            or "none"
        )
        replayed = []
        for line in record["replay"]:
            m = re.match(r"(\S+) (\d+) calls, (\d+) differ", line)
            if m and m.group(1) in SCENARIO_NAMES:
                replayed.append((m.group(1), int(m.group(2)), int(m.group(3))))
        replay = ", ".join(f"{s} {d}/{c}" for s, c, d in replayed) or "not run"
        lines.append(
            f"| `{record['name']}` | {record['what']} | {ctest} | {parity} | {replay} |"
        )
    caught = [
        r
        for r in load("plants.json")
        if r["name"] != "control"
        and (r.get("ctest_failed") or r.get("parity_failed_by_test"))
    ]
    total = [r for r in load("plants.json") if r["name"] != "control"]
    lines += ["", f"{len(caught)} of {len(total)} plants fail at least one test."]
    return lines


def timing_rows() -> list[dict]:
    """The timing sessions, in run order."""
    out = []
    for path in sorted(RESULTS.glob("timing_*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(out, key=lambda r: r["run"])


def load_line(lines: list[str]) -> str:
    """The busiest Windows processes and the processor state of a windows_load sample."""
    if not lines:
        return "not sampled"
    busy = [line for line in lines if not line.startswith(("on mains", "processor"))][
        :3
    ]
    state = [line for line in lines if line.startswith(("on mains", "processor"))]
    return "; ".join(busy + state)


def timing_table() -> list[str]:
    """T5: the class step on the developed frame, core against reference."""
    lines = [
        "### T5. A class step on the developed 320x120 frame: the core against `solve_timestep`",
        "",
        "The stored 320x120 dust release's last frame (`results/builder81/product/equipment_dust.h5`) "
        "of each class, stepped under its last interval's captured conditions, faces and eddy "
        "viscosity. Medians of repeated calls in one process after warm-up calls: the reference's "
        "whole `solve_timestep`, and the core's `step` with the interval set up beforehand. The "
        "two steps were checked equal bit for bit before each was timed.",
        "",
        "| Run | Class | Sweeps | Reference, ms | Core, ms | Reference over core |",
        "|---|---|---|---|---|---|",
    ]
    sessions = timing_rows()
    for record in sessions:
        for row in record["rows"]:
            lines.append(
                f"| {record['run']} | {row['class']} | {row['sweeps']} | "
                f"{row['reference']['median_s'] * 1e3:.3f} | {row['core']['median_s'] * 1e3:.3f} | "
                f"{row['ratio']:.2f} |"
            )
    lines += [
        "",
        "| Run | Started | Calibration before and after, us | The core's interval set-up, ms | Windows before | Windows after |",
        "|---|---|---|---|---|---|",
    ]
    for record in sessions:
        lines.append(
            f"| {record['run']} | {record['started']} | "
            f"{record['calibration_before']['median_s'] * 1e6:.2f}, "
            f"{record['calibration_after']['median_s'] * 1e6:.2f} | "
            f"{record['set_interval']['median_s'] * 1e3:.2f} | "
            f"{load_line(record['windows_before'])} | {load_line(record['windows_after'])} |"
        )
    return lines


def loops_table() -> list[str]:
    """T6: the loops GCC vectorises in the core's sources."""
    record = load("vecinfo.json")
    lines = [
        "### T6. The loops GCC vectorises",
        "",
        f"`{record['compiler']}`, `{' '.join(record['flags'])}`, `-fopt-info-vec-optimized`: "
        "loops vectorised, each counted once by its place in the source (the report's lines "
        "repeat a loop for each inlined copy).",
        "",
        "| Source | Loops vectorised | Report lines |",
        "|---|---|---|",
    ]
    for file in ("src/transport.cpp", "src/scalar.cpp", "src/reduce.cpp"):
        lines.append(
            f"| `{file}` | {len(record[file]['loops'])} | {record[file]['report_lines']} |"
        )
    text = (
        (ROOT / "core" / "src" / "transport.cpp")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    names = []
    for place in record["src/transport.cpp"]["loops"]:
        line = int(place.split(":")[1])
        for back in range(line - 1, max(line - 12, 0), -1):
            m = re.search(r"inline \w+(?:::\w+)? (\w+_cells)\(", text[back - 1])
            if m:
                names.append(m.group(1))
                break
    lines += [
        "",
        "The block functions of `transport.cpp` whose loop is vectorised: "
        + ", ".join(f"`{n}`" for n in sorted(set(names)))
        + ".",
    ]
    return lines


def predictions_table() -> list[str]:
    """T0: the predictions, scored by the rule committed with them."""
    rows = replay_rows()
    bad = [r for r in rows if differs(r)]
    cases = junit(RESULTS / "parity.xml")
    step_cases = [
        (n, o)
        for n, o in cases
        if n.startswith(("test_class_step", "test_the_planted_class_steps"))
    ]
    step_failed = [n for n, o in step_cases if o == "failed"]
    vents = [r for r in rows if r["vents"] and r["recirculated"]]
    interval_rows = [r for r in rows if r["interval_differ"]]
    interval_cases = [
        (n, o) for n, o in cases if n.startswith("test_class_step_interval_quantities")
    ]
    sessions = timing_rows()
    run1 = next((s for s in sessions if s["run"] == 1), None)
    a_ok = not bad and not step_failed and vents
    b_ok = not interval_rows and all(o == "passed" for _, o in interval_cases)
    lines = [
        "### T0. The predictions, scored by the rule committed with them",
        "",
        "| Prediction | Measured | Verdict |",
        "|---|---|---|",
        f"| (a) every captured and planted class step equals the reference's bit for bit | "
        f"{len(rows) - len(bad)} of {len(rows)} captured calls with no differing bit in the field, "
        f"the ten budget terms or the records, {len(vents)} of them with vents and a discharge "
        f"recirculating (T2); {len(step_cases) - len(step_failed)} of {len(step_cases)} planted "
        f"class-step parity cases pass (T3) | {'holds' if a_ok else 'FAILS'} |",
        f"| (b) the interval set-up's quantities equal the reference's per-step values on every mesh | "
        f"{len(rows) - len(interval_rows)} of {len(rows)} captured calls with every quantity equal; "
        f"{sum(o == 'passed' for _, o in interval_cases)} of {len(interval_cases)} parity cases "
        f"(eight meshes, two variants) pass (T2, T3) | {'holds' if b_ok else 'FAILS'} |",
    ]
    if run1 is None:
        lines.append(
            "| (c) a class step at least five times faster on the developed 320x120 frame | not timed | |"
        )
    else:
        ratios = [row["ratio"] for row in run1["rows"]]
        others = [
            f"run {s['run']}: {min(r['ratio'] for r in s['rows']):.2f} to {max(r['ratio'] for r in s['rows']):.2f}"
            for s in sessions
            if s["run"] != 1
        ]
        verdict = "holds" if min(ratios) >= RATIO_REQUIRED else "FAILS"
        lines.append(
            f"| (c) a class step at least five times faster on the developed 320x120 frame | "
            f"{min(ratios):.2f} to {max(ratios):.2f} times over the five classes in run 1, median "
            f"{statistics.median(ratios):.2f} ({'; '.join(others)}) (T5) | {verdict} |"
        )
    return lines


def build() -> tuple[str, dict[str, str]]:
    """Every table, as the full file and as blocks by marker."""
    blocks = {
        "predictions": predictions_table(),
        "captures": captures_table(),
        "replay": replay_table(),
        "tests": tests_table(),
        "plants": plants_table(),
        "timing": timing_table(),
        "loops": loops_table(),
    }
    assert tuple(blocks) == MARKERS
    text = "# Prompt 90's tables\n\nGenerated by `tables90.py` from `results/builder90/`.\n\n"
    text += "\n\n".join("\n".join(lines) for lines in blocks.values()) + "\n"
    return text, {name: "\n".join(lines) for name, lines in blocks.items()}


def splice(report: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """The report with each block between its markers; the markers it lacks."""
    missing = []
    for name, body in blocks.items():
        begin, end = f"<!-- tables90 {name} begin -->", f"<!-- tables90 {name} end -->"
        if begin not in report or end not in report:
            missing.append(name)
            continue
        head, rest = report.split(begin, 1)
        _, tail = rest.split(end, 1)
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
        print("tables90 --check:", "ok" if ok else "FAILED")
        return 0 if ok else 1
    TABLES.write_text(text, encoding="utf-8", newline="\n")
    REPORT.write_text(spliced, encoding="utf-8", newline="\n")
    if missing:
        print(f"missing markers (not spliced): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
