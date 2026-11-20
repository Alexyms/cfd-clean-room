"""The tables of prompt 88's report, from the records in results/builder88/.

    python docs/reports/probe88/tables88.py           # write tables88.md, splice the report
    python docs/reports/probe88/tables88.py --check   # regenerate and compare, write nothing

Run with Windows' ``.venv`` or the Linux environment: every record it reads is
a text file in the checkout, written under WSL2 by ``probe88.py`` (the parity
run's JUnit record with its per-case properties, the captured replay, the
timing, CTest's JUnit record copied from the build tree), by ``mutants88.py``
(the planted defects), and by the two whole-suite runs (pytest's JUnit
records). ``--check`` also fails when one of the report's marker pairs is
missing.
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
RESULTS = ROOT / "results" / "builder88"
TABLES = HERE / "tables88.md"
REPORT = ROOT / "docs" / "reports" / "phase6_scalar_implicit.md"
MARKERS = (
    "predictions",
    "tests",
    "jacobi",
    "cg",
    "cgbound",
    "fallbacks",
    "replay",
    "replaycg",
    "replaycgsolves",
    "plants",
    "timing",
    "parts",
    "suites",
)
# The parity tests' options, in the order the tables list them
# (tests/test_core_parity.py: MESHES, JACOBI_FIELDS, STEPS, BOUNDS, HOLDS,
# CG_FIELDS, CG_STEPS).
MESHES = ["80x30", "160x60", "320x120", "160x60-stretched", "40x15-uniform-open"]
JACOBI_FIELDS = ["decades", "signed", "plume", "every-face"]
STEPS = ["scalar", "per-cell"]
BOUNDS = ["none", "binds", "infinite", "zero"]
HOLDS = ["none", "held"]
CG_FIELDS = ["positive-decades", "eps-like", "tight"]
CG_STEPS = ["scalar", "per-cell", "per-cell-inf-solid"]
CG_BOUND_FIELDS = ["positive-decades", "tight"]
CG_BOUND_STOPS = ["tol", "loose"]
CG_BOUNDS = ["binds", "infinite", "zero"]
CG_FIELD_RTOL = 1e-12
# The commit the suites, CTest, the parity run and the planted defects were run
# at: the last commit that changes the code or its tests (T7 and T5 name it).
RECORDS_AT = "9e47d21"
JACOBI_TEST = "test_implicit_jacobi_equals_the_reference_bit_for_bit"
CG_RULE_TEST = "test_implicit_cg_meets_prompt_88s_rule"
CG_DOT_TEST = "test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit"
FALLBACK_TEST = "test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit"
CG_BOUND_RULE_TEST = "test_implicit_cg_with_a_leftover_bound_meets_prompt_88s_rule"
CG_BOUND_DOT_TEST = (
    "test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit"
)


def alternatives(names: list[str]) -> str:
    """A regex alternation of names, the longest first so a prefix cannot shadow a name."""
    return "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))


JACOBI_ID = re.compile(
    rf"^({alternatives(MESHES)})-({alternatives(JACOBI_FIELDS)})-({alternatives(STEPS)})"
    rf"-({alternatives(BOUNDS)})-({alternatives(HOLDS)})$"
)
CG_ID = re.compile(
    rf"^({alternatives(MESHES)})-({alternatives(CG_FIELDS)})-({alternatives(CG_STEPS)})"
    rf"-({alternatives(HOLDS)})$"
)
FALLBACK_ID = re.compile(rf"^({alternatives(MESHES)})-(cap|non-positive)$")
CG_BOUND_ID = re.compile(
    rf"^({alternatives(MESHES)})-({alternatives(CG_BOUND_FIELDS)})"
    rf"-({alternatives(CG_BOUND_STOPS)})-({alternatives(CG_BOUNDS)})$"
)


class Case:
    """One parity case of a JUnit record: its function, parameter id, outcome and properties."""

    def __init__(self, element: ET.Element) -> None:
        name = element.get("name", "")
        self.function, _, param = name.partition("[")
        self.param = param.rstrip("]")
        self.outcome = "passed"
        if element.find("failure") is not None or element.find("error") is not None:
            self.outcome = "failed"
        elif element.find("skipped") is not None:
            self.outcome = "skipped"
        self.properties = {
            p.get("name"): p.get("value") for p in element.iter("property")
        }


def cases(path: Path) -> list[Case]:
    """Every case of a JUnit record."""
    return [Case(e) for e in ET.parse(path).getroot().iter("testcase")]


def parity_cases(
    function: str, pattern: re.Pattern
) -> list[tuple[tuple[str, ...], Case]]:
    """One parity test's cases with their parsed parameters."""
    out = []
    for case in cases(RESULTS / "parity.xml"):
        if case.function != function:
            continue
        match = pattern.match(case.param)
        if match is None:
            raise ValueError(f"unparsed parity id {case.param}")
        out.append((match.groups(), case))
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


def passed(outcomes: list[str]) -> str:
    """'n of m', or 'n of m, k FAILED'."""
    n = outcomes.count("passed")
    failed = len(outcomes) - n
    return f"{n} of {len(outcomes)}" + (f", {failed} FAILED" if failed else "")


def span(values: list[int]) -> str:
    """'a to b', or 'a' when they are equal."""
    lo, hi = min(values), max(values)
    return f"{lo:,}" if lo == hi else f"{lo:,} to {hi:,}"


def tests_table() -> list[str]:
    """T1: the core's GoogleTest tests and the parity tests, by suite and function."""
    total, failed, suites = ctest_counts(RESULTS / "ctest.xml")
    lines = [
        "### T1. The tests: CTest by GoogleTest suite, and the parity tests by function",
        "",
        "The parity functions before `test_implicit_*` and `row_dot` are prompt 86's and 87's,",
        "run again.",
        "",
        "| Suite or test function | Cases | Passed |",
        "|---|---|---|",
    ]
    for suite, (n, ok) in suites.items():
        lines.append(f"| GoogleTest `{suite}` | {n} | {ok} |")
    lines.append(f"| **CTest, all** | {total} | {total - failed} |")
    by_function: dict[str, list[str]] = defaultdict(list)
    for case in cases(RESULTS / "parity.xml"):
        by_function[case.function].append(case.outcome)
    for function, outcomes in by_function.items():
        lines.append(f"| `{function}` | {len(outcomes)} | {outcomes.count('passed')} |")
    every = [o for outcomes in by_function.values() for o in outcomes]
    lines.append(f"| **parity, all** | {len(every)} | {every.count('passed')} |")
    return lines


def jacobi_table() -> list[str]:
    """T2: the planted Jacobi solves bit for bit, by mesh and field."""
    rows = parity_cases(JACOBI_TEST, JACOBI_ID)
    by: dict[tuple[str, str], list[Case]] = defaultdict(list)
    for (mesh, field, _step, _bound, _hold), case in rows:
        by[(mesh, field)].append(case)
    lines = [
        "### T2. Planted Jacobi solves against the reference, every output bit for bit",
        "",
        "Each cell counts the cases (two steps, four bounds, held cells or none: 16 per",
        "mesh and field) whose field, sweeps, converged flag, leftover and leftover sums",
        "all equal the reference's.",
        "",
        "| Mesh | " + " | ".join(f"`{f}`" for f in JACOBI_FIELDS) + " | Sweeps |",
        "|---|" + "---|" * (len(JACOBI_FIELDS) + 1),
    ]
    for mesh in MESHES:
        cells = [passed([c.outcome for c in by[(mesh, f)]]) for f in JACOBI_FIELDS]
        sweeps = [
            int(c.properties["sweeps"]) for f in JACOBI_FIELDS for c in by[(mesh, f)]
        ]
        lines.append(f"| {mesh} | " + " | ".join(cells) + f" | {span(sweeps)} |")
    binding = [c for (_m, _f, _s, b, _h), c in rows if b == "binds"]
    added = sum(
        int(c.properties["sweeps"]) > int(c.properties["sweeps_unbound"])
        for c in binding
    )
    zero = [c for (_m, _f, _s, b, _h), c in rows if b == "zero"]
    capped = sum(c.properties["converged"] == "False" for c in zero)
    infinite = [c for (_m, _f, _s, b, _h), c in rows if b == "infinite"]
    same = sum(
        c.properties["sweeps"] == c.properties["sweeps_unbound"] for c in infinite
    )
    lines += [
        "",
        f"The binding bound (an eighth of the unbound leftover) added sweeps in {added} of "
        f"{len(binding)} cases; the zero bound ran to the cap in {capped} of {len(zero)}; "
        f"the infinite bound took the unbound sweeps in {same} of {len(infinite)}.",
    ]
    return lines


def cg_table() -> list[str]:
    """T3: the planted CG solves by prompt 88's rule, and with the core's dot products."""
    rule = parity_cases(CG_RULE_TEST, CG_ID)
    dot = parity_cases(CG_DOT_TEST, CG_ID)
    by: dict[tuple[str, str], list[Case]] = defaultdict(list)
    for (mesh, field, _step, _hold), case in rule:
        by[(mesh, field)].append(case)
    dot_by: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (mesh, field, _step, _hold), case in dot:
        dot_by[(mesh, field)].append(case.outcome)
    lines = [
        "### T3. Planted CG solves: prompt 88's rule, and bit for bit with the core's dot products",
        "",
        "Six cases per mesh and field (three steps, held cells or none). Same count: the",
        "core's iterations equal the reference's. The difference is the field's largest",
        "absolute difference over the reference's largest magnitude, over the cases with",
        "the same count; the rule asks at most 1e-12 there. Differing counts are listed as",
        "reference to core with the core's restarts, and the fallbacks where either fell back.",
        "The last column is the reference's CG with `np.vdot` replaced by the core's `row_dot`,",
        "every output bit for bit.",
        "",
        "| Mesh | Field | Same count | Largest difference, same count | Counts differing "
        "| Restarts (core) | With the core's dot, bit for bit |",
        "|---|---|---|---|---|---|---|",
    ]
    for mesh in MESHES:
        for field in CG_FIELDS:
            group = by[(mesh, field)]
            same = [
                c
                for c in group
                if c.properties["reference_iterations"]
                == c.properties["core_iterations"]
            ]
            largest = max(
                (float(c.properties["relative_difference"]) for c in same), default=0.0
            )
            differing = []
            for c in group:
                p = c.properties
                if p["reference_iterations"] != p["core_iterations"]:
                    note = (
                        f"{p['reference_iterations']} to {p['core_iterations']} "
                        f"(restarts {p['restarts']})"
                    )
                    if p["reference_fallbacks"] != "0" or p["core_fallbacks"] != "0":
                        note += f" (fallbacks {p['reference_fallbacks']} and {p['core_fallbacks']})"
                    differing.append(note)
            restarts = [int(c.properties["restarts"]) for c in group]
            lines.append(
                f"| {mesh} | `{field}` | {len(same)} of {len(group)} | {largest:.2e} "
                f"| {'; '.join(differing) if differing else 'none'} | {span(restarts)} "
                f"| {passed(dot_by[(mesh, field)])} |"
            )
    return lines


def cgbound_table() -> list[str]:
    """T3c: the planted CG solves with a leftover bound, by field, stop and bound."""
    rule = parity_cases(CG_BOUND_RULE_TEST, CG_BOUND_ID)
    dot = parity_cases(CG_BOUND_DOT_TEST, CG_BOUND_ID)
    by: dict[tuple[str, str, str], list[tuple[str, Case]]] = defaultdict(list)
    for (mesh, field, stop, bound), case in rule:
        by[(field, stop, bound)].append((mesh, case))
    dot_by: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for (_mesh, field, stop, bound), case in dot:
        dot_by[(field, stop, bound)].append(case.outcome)
    lines = [
        "### T3c. Planted CG solves with a leftover bound",
        "",
        "Five meshes per row, a per-cell step, no held cells. `binds`: an eighth of the",
        "unbound leftover; `infinite`; `zero`, with the cap at 30 iterations. `loose`: a",
        "stop of 1e3, which the start residual already meets. The rule column counts the",
        "cases that pass prompt 88's rule and the bound's contract (a converged result's",
        "leftover is inside the bound, the core's and the reference's). Same count and the",
        "difference are as in T3; differing counts are listed as mesh, reference to core.",
        "The last column is the reference's CG with `np.vdot` replaced by `row_dot`, every",
        "output bit for bit.",
        "",
        "| Field | Stop | Bound | Rule | Same count | Largest difference, same count "
        "| Counts differing | Restarts (core) | Converged (core) | Fallbacks (core) "
        "| With the core's dot, bit for bit |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for field in CG_BOUND_FIELDS:
        for stop in CG_BOUND_STOPS:
            for bound in CG_BOUNDS:
                group = by.get((field, stop, bound))
                if not group:
                    continue
                members = [c for _m, c in group]
                same = [
                    c
                    for c in members
                    if c.properties["reference_iterations"]
                    == c.properties["core_iterations"]
                ]
                largest = max(
                    (float(c.properties["relative_difference"]) for c in same),
                    default=0.0,
                )
                differing = [
                    f"{m} {c.properties['reference_iterations']} to "
                    f"{c.properties['core_iterations']}"
                    for m, c in group
                    if c.properties["reference_iterations"]
                    != c.properties["core_iterations"]
                ]
                restarts = [int(c.properties["restarts"]) for c in members]
                converged = sum(c.properties["converged"] == "True" for c in members)
                fallbacks = sum(c.properties["core_fallbacks"] != "0" for c in members)
                lines.append(
                    f"| `{field}` | {stop} | {bound} "
                    f"| {passed([c.outcome for c in members])} "
                    f"| {len(same)} of {len(members)} | {largest:.2e} "
                    f"| {'; '.join(differing) if differing else 'none'} "
                    f"| {span(restarts)} | {converged} of {len(members)} "
                    f"| {fallbacks} of {len(members)} "
                    f"| {passed(dot_by[(field, stop, bound)])} |"
                )
    return lines


def fallbacks_table() -> list[str]:
    """T3b: the forced fallbacks, bit for bit."""
    rows = parity_cases(FALLBACK_TEST, FALLBACK_ID)
    lines = [
        "### T3b. Forced fallbacks: the Jacobi solve from C* bit for bit",
        "",
        "`cap`: one CG iteration at 1e-15; `non-positive`: one cell cut off with C* zero,",
        "which CG leaves at zero. The field, fallback sweeps, converged flag and leftover",
        "are compared bit for bit, and the field against a Jacobi solve alone.",
        "",
        "| Mesh | Cause | Passed | CG iterations, reference | CG iterations, core |",
        "|---|---|---|---|---|",
    ]
    for (mesh, cause), case in rows:
        p = case.properties
        lines.append(
            f"| {mesh} | {cause} | {passed([case.outcome])} | {p['reference_iterations']} "
            f"| {p['core_iterations']} |"
        )
    return lines


def replay_rows() -> list[dict]:
    """The captured calls' replay records."""
    return json.loads((RESULTS / "replay.json").read_text(encoding="utf-8"))


CAPTURE_NAMES = {
    "dust80": "80x30 dust release, transport march",
    "dust320": "320x120 dust release, transport march",
    "flow80": "80x30 flow solve, k and eps",
}


def jacobi_differs(r: dict) -> bool:
    """Whether a Jacobi call's replay differs from the reference in any output."""
    return bool(
        r["field_cells_differing"]
        or r["core_iterations"] != r["reference_iterations"]
        or r["core_converged"] != r["reference_converged"]
        or not r["leftover_same_bits"]
        or r["core_forms"] != r["reference_forms"]
        or r["core_fallbacks"] != r["reference_fallbacks"]
    )


def replay_table() -> list[str]:
    """T4: the captured Jacobi calls replayed through the core."""
    rows = [r for r in replay_rows() if r["method"] == "jacobi"]
    lines = [
        "### T4. The captured transport solves replayed through the core (Jacobi)",
        "",
        "Every output is compared bit for bit: the field, the sweeps, converged, the",
        "leftover and the number of leftover sums. The bound added sweeps where the",
        "reference without the bound stopped sooner. Reference again: the reference",
        "called on the captured inputs, its outputs differing from the capture.",
        "",
        "| Capture | Calls | Cells compared | Calls differing | Sweeps | Bound added sweeps "
        "| Converged | Reference again, differing | Non-zero cells of C* |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ("dust80", "dust320"):
        group = [r for r in rows if r["capture"] == name]
        added = sum(
            r["reference_iterations"] > r["sweeps_without_bound"] for r in group
        )
        lines.append(
            f"| {CAPTURE_NAMES[name]} | {len(group)} | {sum(r['cells'] for r in group):,} "
            f"| {sum(jacobi_differs(r) for r in group)} "
            f"| {span([r['reference_iterations'] for r in group])} | {added} "
            f"| {sum(r['reference_converged'] for r in group)} "
            f"| {sum(r['reference_again_differs'] for r in group)} "
            f"| {span([r['nonzero_cells'] for r in group])} |"
        )
    return lines


def replaycg_table() -> list[str]:
    """T4b: the captured CG calls replayed through the core."""
    rows = [r for r in replay_rows() if r["method"] == "cg"]
    lines = [
        "### T4b. The captured k and eps solves replayed through the core (CG)",
        "",
        "Same count and the largest difference as in T3. The leftover's difference is",
        "taken over ADR-016's rounding scale `sum_P dt a_P |C_P|` (the leftover is a",
        "cancelling sum of residuals, and the residuals of two fields that differ in their",
        "last bits differ by about that much). With the core's dot: the reference's CG with",
        "`np.vdot` replaced by `row_dot`, its outputs differing from the core's.",
        "",
        "| Field | Calls | Same count | Iterations | Largest difference | Fallbacks, "
        "reference and core | Leftover difference over its scale | With the core's dot, "
        "differing | Held cells |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for field, label in ((0, "k"), (1, "eps")):
        group = [r for r in rows if r["label"].get("field") == field]
        same = sum(r["core_iterations"] == r["reference_iterations"] for r in group)
        leftover = max(
            abs(r["core_leftover"] - r["reference_leftover"]) / r["leftover_scale"]
            for r in group
        )
        lines.append(
            f"| {label} | {len(group)} | {same} of {len(group)} "
            f"| {span([r['reference_iterations'] for r in group])} "
            f"| {max(r['field_relative_difference'] for r in group):.2e} "
            f"| {sum(r['reference_fallbacks'] for r in group)} and "
            f"{sum(r['core_fallbacks'] for r in group)} | {leftover:.1e} "
            f"| {sum(r['with_cores_dot_bits_differ'] for r in group)} "
            f"| {span([r['held_cells'] for r in group])} |"
        )
    return lines


def replaycgsolves_table() -> list[str]:
    """T4c: each captured k and eps solve, its counts and its largest difference."""
    rows = [r for r in replay_rows() if r["method"] == "cg"]
    lines = [
        "### T4c. Each captured k and eps solve",
        "",
        "One row per solve of T4b, in the order the flow solve made them: the outer",
        "iteration, the field (k or eps), the reference's and the core's iteration counts,",
        "the core's restarts, the field's largest difference over the reference's largest",
        "magnitude, the leftover's difference over its rounding scale (as in T4b) and the",
        "number of held cells.",
        "",
        "| Outer iteration | Field | Reference iterations | Core iterations | Restarts "
        "| Largest difference | Leftover difference over its scale | Held cells |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        field = "k" if r["label"]["field"] == 0 else "eps"
        leftover = (
            abs(r["core_leftover"] - r["reference_leftover"]) / r["leftover_scale"]
        )
        lines.append(
            f"| {r['label']['outer']} | {field} | {r['reference_iterations']} "
            f"| {r['core_iterations']} | {r['core_restarts']} "
            f"| {r['field_relative_difference']:.2e} | {leftover:.1e} "
            f"| {r['held_cells']} |"
        )
    return lines


def plants() -> list[dict]:
    """The planted defects' records."""
    return json.loads((RESULTS / "mutants88.json").read_text(encoding="utf-8"))


def plants_table() -> list[str]:
    """T5: the planted defects and the tests that catch each."""
    lines = [
        "### T5. The planted defects: which tests catch each",
        "",
        "| Defect | What it changes | GoogleTest (CTest) | Parity (pytest) |",
        "|---|---|---|---|",
    ]
    for p in plants():
        if "build" in p:
            lines.append(f"| `{p['name']}` | {p['what']} | build failed | |")
            continue
        gtest = p["ctest"]
        if p["ctest_failed"]:
            gtest += f" [failed: {', '.join(p['ctest_failed'])}]"
        parity = p["pytest"]
        if p["parity_failed_by_test"]:
            parity += (
                " [failed: "
                + ", ".join(f"{k} {v}" for k, v in p["parity_failed_by_test"].items())
                + "]"
            )
        # A "|" in a description would end its table cell.
        what = p["what"].replace("|", "\\|")
        lines.append(f"| `{p['name']}` | {what} | {gtest} | {parity} |")
    return lines


def timings() -> dict[int, dict]:
    """The timing records by run."""
    out = {}
    for path in sorted(RESULTS.glob("timing_[0-9]*.json")):
        out[int(path.stem.split("_")[1])] = json.loads(path.read_text(encoding="utf-8"))
    return out


def timing_table() -> list[str]:
    """T6: the Jacobi solve per call on the developed 320x120 frames."""
    records = timings()
    first = records[1]
    lines = [
        "### T6. The Jacobi solve per call on the stored 320x120 run's last frame, one thread",
        "",
        "Medians of repeated calls under Linux: the core through the binding with its",
        "scratch and output allocated once, the reference's `implicit_step` as the transport",
        "solver calls it. C* is each class's last frame of `results/builder81/product/",
        "equipment_dust.h5`, the other inputs that class's last captured call's, the bound",
        "ADR-016's share of the class's supply at that frame. The core's outputs equal the",
        "reference's bit for bit in every timed call (last column). Run 1 is the one scored;",
        "runs 2 and 3 repeat it.",
        "",
        "| Class | Non-zero cells | Sweeps | Core, ms | Reference, ms | Ratio, run 1 "
        "| Ratio, run 2 | Ratio, run 3 | Outputs differing |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(first["rows"]):
        ratios = [
            f"{records[n]['rows'][i]['ratio']:.2f}" if n in records else ""
            for n in (1, 2, 3)
        ]
        lines.append(
            f"| {row['class']} | {row['nonzero_cells']:,} | {row['sweeps']} "
            f"| {1e3 * row['core_median_s']:.2f} | {1e3 * row['reference_median_s']:.1f} "
            f"| {' | '.join(ratios)} | {row['core_bits_differ']} |"
        )
    lines += [
        "",
        f"Repeats: the core {first['rows'][0]['core_repeats']}, the reference "
        f"{first['rows'][0]['reference_repeats']}, after two unrecorded calls; "
        f"OPENBLAS_NUM_THREADS={first['openblas_num_threads']}, Python {first['python']}, "
        f"NumPy {first['numpy']}.",
    ]
    return lines


def parts_table() -> list[str]:
    """T6b: where the developed solve's time goes, for the core and the reference."""
    records = {}
    for path in sorted(RESULTS.glob("timing_parts_*.json")):
        records[int(path.stem.split("_")[2])] = json.loads(
            path.read_text(encoding="utf-8")
        )
    first = records[1]
    sweeps = first["sweeps"]
    lines = [
        "### T6b. Where the developed solve's time goes, class "
        f"{first['class']}, one thread",
        "",
        "Medians as in T6. `no sweep`: the cap at 0, so the set-up, one residual and the",
        f"leftover; `the full solve`: the cap at its {sweeps} sweeps; `nothing to solve`: no",
        "conductance and no sink, the early return, so the call, its argument checks and the",
        "copy of C*. A sweep is the difference of the first two over the sweeps.",
        "",
        "| Run | Part | Core, ms | Reference, ms | Reference / core |",
        "|---|---|---|---|---|",
    ]
    for n, record in sorted(records.items()):
        by = {row["part"]: row for row in record["rows"]}
        for row in record["rows"]:
            core, ref = row["core_median_s"], row["reference_median_s"]
            lines.append(
                f"| {n} | {row['part']} | {1e3 * core:.3f} | {1e3 * ref:.3f} | {ref / core:.2f} |"
            )
        core = (
            by["the full solve"]["core_median_s"] - by["no sweep"]["core_median_s"]
        ) / sweeps
        ref = (
            by["the full solve"]["reference_median_s"]
            - by["no sweep"]["reference_median_s"]
        ) / sweeps
        lines.append(
            f"| {n} | one sweep | {1e3 * core:.3f} | {1e3 * ref:.3f} | {ref / core:.2f} |"
        )
    return lines


def suite_counts(path: Path) -> dict[str, float]:
    """The totals of a pytest JUnit record: tests, failures, errors, skipped and seconds."""
    suite = ET.parse(path).getroot().find("testsuite")
    return {
        key: float(suite.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped", "time")
    }


def suites_table() -> list[str]:
    """T7: the whole suite on each platform, from pytest's JUnit records."""
    lines = [
        f"### T7. The whole suite at {RECORDS_AT}, the last commit that changes the code or its tests",
        "",
        "The two suites ran at the same time, one BLAS thread each, beside the planted",
        "defects, so the wall times are not those of a suite alone.",
        "",
        "| Platform | Tests | Passed | Failed | Errors | Skipped | Wall time, s |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, name in (
        ("Windows, `.venv`, the core not built", "suite_windows.xml"),
        ("WSL2, the core built, `CFD_CORE_REQUIRED=1`", "suite_linux.xml"),
    ):
        c = suite_counts(RESULTS / name)
        ok = c["tests"] - c["failures"] - c["errors"] - c["skipped"]
        lines.append(
            f"| {label} | {c['tests']:,.0f} | {ok:,.0f} | {c['failures']:.0f} "
            f"| {c['errors']:.0f} | {c['skipped']:.0f} | {c['time']:,.0f} |"
        )
    return lines


def predictions_table() -> list[str]:
    """T0: the three predictions, scored by the rule committed with them."""
    jacobi = parity_cases(JACOBI_TEST, JACOBI_ID)
    fallbacks = parity_cases(FALLBACK_TEST, FALLBACK_ID)
    planted = [c.outcome for _, c in jacobi] + [c.outcome for _, c in fallbacks]
    rows = replay_rows()
    captured_jacobi = [r for r in rows if r["method"] == "jacobi"]
    captured_cg = [r for r in rows if r["method"] == "cg"]
    jacobi_differing = sum(jacobi_differs(r) for r in captured_jacobi)
    fallback_jacobi = sum(r["core_fallbacks"] for r in captured_cg)
    a_holds = planted.count("passed") == len(planted) and jacobi_differing == 0
    same = [r for r in captured_cg if r["core_iterations"] == r["reference_iterations"]]
    largest = max(r["field_relative_difference"] for r in captured_cg)
    b_holds = len(same) == len(captured_cg) and largest <= CG_FIELD_RTOL
    ratios = [row["ratio"] for row in timings()[1]["rows"]]
    c_holds = min(ratios) >= 10.0
    return [
        "### T0. The predictions, scored by the rule committed with them",
        "",
        "| Prediction | Measured | Verdict |",
        "|---|---|---|",
        f"| (a) every Jacobi solve bit for bit, planted and captured "
        f"| planted: {passed(planted)} cases (T2, and the forced fallbacks of T3b); captured: "
        f"{len(captured_jacobi)} calls, {sum(r['cells'] for r in captured_jacobi):,} cells, "
        f"{jacobi_differing} differing in any output (T4); no captured CG call fell back "
        f"({fallback_jacobi}) | {'holds' if a_holds else 'FAILS'} |",
        f"| (b) every captured CG solve at the reference's count, field within 1e-12 "
        f"| {len(same)} of {len(captured_cg)} calls at the reference's count, largest "
        f"difference {largest:.2e} (T4b) | {'holds' if b_holds else 'FAILS'} |",
        f"| (c) the core's Jacobi solve at least ten times faster per call on the developed "
        f"320x120 frame | {min(ratios):.2f} to {max(ratios):.2f} times over the five classes "
        f"(T6, run 1) | {'holds' if c_holds else 'FAILS'} |",
    ]


def build() -> tuple[str, dict[str, str]]:
    """tables88.md's text and the report's blocks."""
    builders = {
        "predictions": predictions_table,
        "tests": tests_table,
        "jacobi": jacobi_table,
        "cg": cg_table,
        "cgbound": cgbound_table,
        "fallbacks": fallbacks_table,
        "replay": replay_table,
        "replaycg": replaycg_table,
        "replaycgsolves": replaycgsolves_table,
        "plants": plants_table,
        "timing": timing_table,
        "parts": parts_table,
        "suites": suites_table,
    }
    blocks = {name: "\n".join(builders[name]()) for name in MARKERS}
    text = "# Prompt 88's tables (generated by tables88.py; do not edit)\n\n"
    text += "\n\n".join(blocks[name] for name in MARKERS)
    return text + "\n", blocks


def splice(report: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """REPORT with each marked block replaced, and the names whose markers are missing."""
    missing = []
    for name, body in blocks.items():
        begin, end = f"<!-- tables88 {name} begin -->", f"<!-- tables88 {name} end -->"
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
        print("tables88 --check:", "ok" if ok else "FAILED")
        return 0 if ok else 1
    TABLES.write_text(text, encoding="utf-8", newline="\n")
    REPORT.write_text(spliced, encoding="utf-8", newline="\n")
    if missing:
        print(f"missing markers (not spliced): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
