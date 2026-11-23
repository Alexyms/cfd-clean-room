"""The tables of prompt 91's march report, from the records in results/builder91/.

    python docs/reports/probe91/tables91.py           # write tables91.md, splice the report
    python docs/reports/probe91/tables91.py --check   # regenerate and compare, write nothing

Run with Windows' ``.venv``: every record it reads is in the checkout. The
whole-march comparison is ``probe91.py compare``'s (``compare.json``, made under
Linux, where the runs were); the comparison with the Windows records is made
here, by prompt 86's means (``tables85.compare_run`` for the fields and the
budget, ``tables85.Placement`` for the searches, the layouts' order and the
clear times), the Windows run as the reference and the core-driven run as the
variant, each field read against ADR-017 H in its three tiers. ``--check`` also
fails when one of the report's marker pairs is missing.
"""

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "probe85"))
sys.path.insert(0, str(HERE.parent / "probe86"))

import probe85  # noqa: E402 -- follows sys.path.insert
import tables85  # noqa: E402 -- follows sys.path.insert
import tables86  # noqa: E402 -- follows sys.path.insert; reads either platform's base path

TABLES = HERE / "tables91.md"
REPORT = ROOT / "docs" / "reports" / "phase6_march.md"
RESULTS = ROOT / "results" / "builder91"
RUNS = RESULTS / "runs"
WINDOWS, CORE = tables85.REFERENCE, tables86.LINUX
GRID = "80x30"
SCENARIOS = tables86.SCENARIOS
NAMES = {"dust": "dust80", "door": "door80", "breach": "breach80"}
LABELS = {
    "dust80": "80x30 dust release",
    "door80": "80x30 door leak",
    "breach80": "80x30 filter breach",
    "clean80": "80x30 clean-out, to t_end",
    "hold80": "80x30 clean-out, with a hold",
    "dust320": "320x120 dust release",
    "restart": "80x30 dust release, restarted at its checkpoint",
}
# The clean-out's restarts: the pair's label in compare.json, the run's name, what it is.
CLEAN_RESTARTS = (
    ("clean restart before", "hold80r_before_restart", "before the stretch"),
    ("clean restart inside", "hold80r_inside_restart", "inside the stretch"),
)
RUN_ORDER = ("dust80", "door80", "breach80", "clean80", "hold80", "dust320")
REQ_T05 = 1.0e-4
SPEED = 5.0
MARKERS = (
    "predictions",
    "runs",
    "march",
    "decisions",
    "restart",
    "cleanrestart",
    "round0",
    "windows",
    "tests",
    "plants",
    "plantsround1",
    "timing",
    "loops",
)
NEW_PARITY = "tests/test_core_march.py"
NEW_GTEST_SUITES = ("Channels", "Arrival", "Recovery", "March")


def e(value: float) -> str:
    """tables85's format for a relative difference."""
    return tables85.e(value)


def yes(flag: bool) -> str:
    """yes or no."""
    return "yes" if flag else "no"


def table(header: list[str], rows: list[list[object]]) -> list[str]:
    """A markdown table."""
    return tables85.table(header, [[str(c) for c in row] for row in rows])


def load(name: str) -> dict:
    """A JSON record of results/builder91/."""
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def summary(side: str, name: str) -> dict:
    """A run's summary."""
    return json.loads((RUNS / side / f"{name}.json").read_text(encoding="utf-8"))


def floats(value: object) -> object:
    """JSON's text inf and NaN back to floats."""
    if isinstance(value, list):
        return [floats(v) for v in value]
    if value in ("inf", "Infinity"):
        return math.inf
    if value in ("-inf", "-Infinity"):
        return -math.inf
    if value in ("nan", "NaN"):
        return math.nan
    return value


# ---------------------------------------------------------------------------
# The Windows comparison: prompt 86's, with the core-driven runs as the variant
# ---------------------------------------------------------------------------


class CoreAt(tables86.RunAt):
    """A Windows record, or a core-driven run of this pass under the name tables86 gives Linux."""

    def __init__(
        self, variant: str, scenario: str, grid: str, location: str = "reference"
    ) -> None:
        if variant != CORE:
            super().__init__(variant, scenario, grid, location)
            return
        self.variant, self.scenario, self.grid = variant, scenario, grid
        self.path = RUNS / "core" / f"{NAMES[scenario]}.h5"
        self.summary = json.loads(self.path.with_suffix(".json").read_text())


class CorePlacement(tables85.Placement):
    """tables85's placement over the Windows records and the core-driven runs."""

    def __init__(self, grid: str) -> None:
        saved_run, saved_variants = tables85.Run, probe85.VARIANTS
        tables85.Run, probe85.VARIANTS = CoreAt, {WINDOWS: None, CORE: None}
        try:
            super().__init__(grid)
        finally:
            tables85.Run, probe85.VARIANTS = saved_run, saved_variants

    def of(self, variant: str) -> dict:
        """tables85's searches, with this module's runs."""
        saved = tables85.Run
        tables85.Run = CoreAt
        try:
            return super().of(variant)
        finally:
            tables85.Run = saved


def windows_comparison() -> tuple[dict, CorePlacement]:
    """Each 80x30 core-driven run against its Windows record, and the placement over both."""
    results = {}
    for scenario in SCENARIOS:
        win, core = CoreAt(WINDOWS, scenario, GRID), CoreAt(CORE, scenario, GRID)
        results[scenario] = tables85.compare_run(win, core)
        win.release()
    return results, CorePlacement(GRID)


def windows_fields(results: dict) -> tuple[list[str], dict]:
    """Each field's largest relative difference from the Windows record, with its tier."""
    rows, tiers = [], {}
    for scenario in SCENARIOS:
        diffs = tables86._differences(results[scenario])
        for field, value in diffs.items():
            spread, limit = tables86.BAND[field]
            t = tables86.tier(value, field)
            tiers[(scenario, field)] = t
            rows.append(
                [LABELS[NAMES[scenario]], field, e(value), e(spread), e(limit), t]
            )
    closure = {s: max(abs(r) for r, _ in results[s]["budget"]) for s in SCENARIOS}
    outers = {s: (results[s]["ref_outers"], results[s]["outers"]) for s in SCENARIOS}
    lines = [
        "### T5. The core-driven runs against the Windows records: each field's largest relative difference, read against ADR-017 H",
        "",
        *table(
            ["scenario", "field", "difference", "measured spread", "limit", "tier"],
            rows,
        ),
        "",
        "Differences as prompt 85's P2 and P6 measure them (`tables85.compare_run`), the Windows run (`results/builder85/runs/reference/`) as the reference: a flow field's over its states, a class's over every stored frame, the pointwise reading where the Windows run reads at least a hundredth of the class's threshold, the sensor channels at every step, the change in the accumulated leftover over the supply at the end. The spreads are P8's (the sensor channels P2's), the limits ten times them rounded up to a decade.",
        "",
        "The budget's closure, the reference's own check: the largest residual over the supply at the run's end, "
        + "; ".join(f"{LABELS[NAMES[s]]} {e(v)}" for s, v in closure.items())
        + f", against {e(tables86.CLOSURE)}. The flow states' outer counts, Windows and the core-driven run's (read from step 1's Linux files): "
        + "; ".join(f"{LABELS[NAMES[s]]} {a} and {b}" for s, (a, b) in outers.items())
        + ".",
    ]
    return lines, {"tiers": tiers, "closure": closure}


def windows_decisions(results: dict, place: CorePlacement) -> tuple[list[str], bool]:
    """The decisions against the Windows records (tables86's T6, the core-driven runs as the variant)."""
    arrival = [results[s]["arrival_moves"] for s in SCENARIOS]
    sensor = [results[s]["sensor_moves"] for s in SCENARIOS]
    cmp = place.compare(CORE)
    rows = [
        ["arrival times (cells x events x channels)", *tables85._moves_text(arrival)],
        ["the file's sensors' latencies", *tables85._moves_text(sensor)],
        *[
            [
                f"the layouts' latencies, {rule}",
                *tables85._moves_text([cmp["rules"][rule]["moves"]]),
            ]
            for rule in tables85.RULES
        ],
        [
            "cells' clear times (cells x units)",
            *tables85._moves_text([cmp["clear_moves"]]),
        ],
        ["room clear times (units)", *tables85._moves_text([cmp["room_moves"]])],
    ]
    moves_equal = all(
        m[1:4] == (0, 0, 0)
        for m in (
            *arrival,
            *sensor,
            *(cmp["rules"][r]["moves"] for r in tables85.RULES),
            cmp["clear_moves"],
            cmp["room_moves"],
        )
    )
    searches = []
    cells_equal = order_equal = True
    for rule in tables85.RULES:
        r = cmp["rules"][rule]
        differ = [n for n, same in r["same_cells"].items() if not same]
        cells_equal = cells_equal and not differ
        order_equal = order_equal and r["same_order"]
        searches.append(
            [
                rule,
                ", ".join(differ) if differ else "none",
                ", ".join(place.results[WINDOWS]["rules"][rule]["order"]),
                ", ".join(r["order"]),
            ]
        )
    verdicts = []
    verdicts_equal = True
    for s in SCENARIOS:
        c = results[s]
        ok = [abs(r) < REQ_T05 and abs(lo) < REQ_T05 for r, lo in c["budget"]]
        ref = [abs(r) < REQ_T05 and abs(lo) < REQ_T05 for r, lo in c["ref_budget"]]
        verdicts_equal = verdicts_equal and ok == ref
        verdicts.append(
            [
                LABELS[NAMES[s]],
                e(max(abs(lo) for _, lo in c["ref_budget"])),
                e(max(abs(lo) for _, lo in c["budget"])),
                yes(ok == ref),
            ]
        )
    equal = moves_equal and cells_equal and order_equal and verdicts_equal
    lines = [
        "### T6. The decisions against the Windows records: latencies and clear times moved, the searches, the layouts' order, the budget's verdicts",
        "",
        *table(
            ["decision", "compared", "moved 1 / > 1 / one only", "largest move"], rows
        ),
        "",
        *table(
            [
                "rule",
                "searched layouts whose cells differ",
                "order, Windows",
                "order, core-driven",
            ],
            searches,
        ),
        "",
        *table(
            [
                "scenario",
                "leftover / supplied, Windows (largest class)",
                "core-driven",
                "every class's verdict against REQ-T05 the same",
            ],
            verdicts,
        ),
        "",
        "Prompt 85's P3 to P6 measures (`tables85.Placement`, `compare_run`), the Windows run as the reference: a latency moves by its difference over the step of the Windows run's interval, a clear time by its difference over the unit's frame spacing; the searches are `placement.search` over the three runs' maps with main's placement spec, under both rules and every pool.",
    ]
    return lines, equal


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------


def runs_table(compare: dict) -> list[str]:
    """T1: the runs made, each twice."""
    rows = []
    for name in RUN_ORDER:
        ref, core = summary("reference", name), summary("core", name)
        flows = ", ".join(f"{f['origin']}" for f in core["flows"])
        rows.append(
            [
                LABELS[name],
                core["steps"],
                compare["pairs"][name]["frames"],
                len(core["intervals"]),
                flows,
                yes(core["held"]),
                f"{ref['wall_seconds']:.0f}",
                f"{core['wall_seconds']:.0f}",
            ]
        )
    step1 = compare["step1"]
    same = all(all(v for k, v in r.items() if k != "frames") for r in step1.values())
    return [
        "### T1. The runs, each made twice under Linux: the reference and the core-driven march",
        "",
        *table(
            [
                "Run",
                "Steps",
                "Frames",
                "Intervals",
                "Flow states (core-driven run)",
                "Held",
                "Reference wall, s",
                "Core-driven wall, s",
            ],
            rows,
        ),
        "",
        "Each from its stored flow states (`flows_from`): the 80x30 runs from step 1's Linux reference (`results/builder86/runs/reference/`), the clean-out's from the dust release's (its flow inputs are the dust release's), the 320x120 run from `results/builder81/product/equipment_dust.h5`. `kept` is a state copied from that file. The wall times are the whole calls, flow states read and files written included, under Linux, with two 320x120 runs and one 80x30 run at a time; they are for information only. The core-driven march hands the monitor the fields at every step, as the reference does (section 1).",
        "",
        f"This pass's three 80x30 references against step 1's Linux runs of the same files: frames, budgets, sensor table and arrival maps equal bit for bit, {yes(same)} ("
        + "; ".join(f"{LABELS[n]} {r['frames']} frames" for n, r in step1.items())
        + ").",
    ]


def march_table(compare: dict) -> tuple[list[str], bool]:
    """T2: the whole march, the core-driven run against the reference, bit for bit."""
    rows = []
    all_bitwise = True
    for name in RUN_ORDER:
        c = compare["pairs"][name]
        digests = c["digests"]
        maps = digests.get("first_above", {"frames_differing": "n/a"})
        recovery = [v for k, v in digests.items() if k.startswith("recovery.")]
        rec = (
            "n/a" if not recovery else str(sum(v["frames_differing"] for v in recovery))
        )
        all_bitwise = all_bitwise and c["bitwise"]
        rows.append(
            [
                LABELS[name],
                c["frames"],
                c["fields_frames_differing"],
                c["budgets_frames_differing"],
                c["sensor_rows"],
                c["sensor_rows_differing"],
                c["digest_frames"],
                maps["frames_differing"],
                rec,
                ", ".join(
                    c["alerts_differing"]
                    + c["arrival_differing"]
                    + c["recovery_differing"]
                )
                or "none",
                yes(c["bitwise"]),
            ]
        )
    lines = [
        "### T2. The whole march: the core-driven run against the Linux reference, bit for bit",
        "",
        *table(
            [
                "Run",
                "Frames",
                "Frames whose class fields differ",
                "Frames whose budget terms differ",
                "Sensor rows",
                "Rows whose readings or channels differ",
                "Frames with tracker digests",
                "Frames whose arrival maps differ",
                "Frames whose recovery fields differ (all six arrays)",
                "End records differing (alerts, arrival, recovery)",
                "Bit for bit",
            ],
            rows,
        ),
        "",
        "`probe91.py compare`: at every frame (every output step and the last), every class field and the ten terms of every class's budget by their 64-bit patterns; every row of the sensor table (each class's reading and each channel's at each sensor, every step); the arrival maps and the six recovery arrays as the run's trackers hold them when each frame is written, by a digest of their bytes (in the core-driven run they are set from the core's state just before); and at the end the alert table, `/arrival/` and `/recovery/` field by field. A 0 is no differing bit.",
    ]
    return lines, all_bitwise


def decisions_table(compare: dict) -> tuple[list[str], bool]:
    """T3: every decision each run reports, the reference's against the core-driven run's."""
    rows = []
    equal = True
    for name in RUN_ORDER:
        c = compare["pairs"][name]
        d = c["decisions_equal"]
        ref = summary("reference", name)
        leaves = [float(x) for x in _leaves(floats(ref["latencies"]))]
        finite = sum(1 for x in leaves if math.isfinite(x))
        verdict_a = [v["within"] for v in c["verdicts"]["a"]]
        verdict_b = [v["within"] for v in c["verdicts"]["b"]]
        largest = max(abs(v["leftover_over_supplied"]) for v in c["verdicts"]["b"])
        closure = max(abs(v["closure_over_supplied"]) for v in c["verdicts"]["b"])
        same = all(d.values()) and verdict_a == verdict_b
        equal = equal and same and not c["recovery_differing"]
        rows.append(
            [
                LABELS[name],
                len(ref["alerts"]),
                yes(d["alerts"]),
                f"{finite} of {len(leaves)}",
                yes(d["latencies"] and d["map_latencies"]),
                "n/a" if not c["recovery"] else yes(not c["recovery_differing"]),
                f"{yes(c['held'])}, step {c['steps']}"
                if c["held"]
                else f"no, {c['steps']} steps",
                yes(d["held"] and d["steps"] and d["t_final"]),
                f"{e(largest)}; {e(closure)}",
                yes(verdict_a == verdict_b and all(verdict_b)),
            ]
        )
    lines = [
        "### T3. The decisions each run reports: the reference's against the core-driven run's",
        "",
        *table(
            [
                "Run",
                "Alerts",
                "Alert table equal",
                "Latencies reached (finite of all)",
                "Latencies and map latencies equal",
                "Recovery record equal (clear times, 100:1 times, ratio, local field)",
                "Held",
                "Hold, steps and final time equal",
                "Largest leftover / supplied; closure / supplied (core-driven)",
                "Every class within REQ-T05, the same on both",
            ],
            rows,
        ),
        "",
        "The latencies are the monitor's per event, channel and sensor (`RunRecord.latencies`, inf where never reached) and the maps' at the sensors' cells (`map_latencies`); equal is equal as JSON, every float to its last digit. A class's verdict is its accumulated leftover and its budget's closure each under REQ-T05's 1e-4 of the particles supplied.",
    ]
    return lines, equal


def _leaves(value: object) -> list[object]:
    """The numbers of a nested list."""
    if isinstance(value, list):
        return [x for v in value for x in _leaves(v)]
    return [value]


def restart_table(compare: dict) -> tuple[list[str], bool]:
    """T4: the restart against the uninterrupted run, both through the core."""
    c = compare["pairs"]["restart"]
    restarted = summary("core", "dust80r_restart")
    digests = c["digests"]
    rows = [
        [
            "frames, from the start",
            c["frames"],
            c["fields_frames_differing"],
            c["budgets_frames_differing"],
        ],
        [
            "sensor rows, from the start",
            c["sensor_rows"],
            c["sensor_rows_differing"],
            "-",
        ],
        [
            "frames after the restart with tracker digests",
            c["digest_frames"],
            sum(v["frames_differing"] for v in digests.values()),
            "-",
        ],
    ]
    lines = [
        "### T4. The restart: the 80x30 dust release stopped after a step and continued through the core, against the run without a stop",
        "",
        *table(
            ["Compared", "Count", "Differing (fields, or any)", "Differing (budgets)"],
            rows,
        ),
        "",
        f"Stopped by an exception from `on_step` after step {restarted['stopped_after']}, restarted from its checkpoint at step {restarted['restarted_from']} (`RunFile.discard_after` keeps the frames and sensor rows up to it), continued to step {restarted['steps']}. End records differing: "
        + (", ".join(c["alerts_differing"] + c["arrival_differing"]) or "none")
        + f"; decisions equal: {yes(all(c['decisions_equal'].values()))}. Both runs write a checkpoint every 1,000 steps; the uninterrupted one against the core-driven dust release of T2, which writes none, bit for bit in every frame, sensor row, tracker digest, end record and decision: {yes(compare['pairs']['checkpoints']['bitwise'])}.",
    ]
    return lines, c["bitwise"]


def _set_count(values: object) -> str:
    """How many of a checkpoint's array hold a time (are not NaN), as 'k of n'."""
    flat = [float(x) for x in _leaves(values)]
    return f"{sum(1 for x in flat if not math.isnan(x))} of {len(flat)}"


def clean_restart_table(compare: dict) -> tuple[list[str], bool]:
    """T4b: the held clean-out restarted from before and from inside its within-spec stretch."""
    rows = []
    bitwise = True
    for label, run, where in CLEAN_RESTARTS:
        c = compare["pairs"][label]
        restarted = summary("core", run)
        carried = restarted["checkpoint_recovery"]
        room = float(carried["room_since"])
        digests = c["digests"]
        bitwise = bitwise and c["bitwise"]
        rows.append(
            [
                where,
                restarted["stopped_after"],
                restarted["restarted_from"],
                "no" if math.isnan(room) else f"yes, since t = {room:.5g} s",
                _set_count(carried["class_since"]),
                _set_count(carried["mean_time"]),
                f"{c['frames']}; {c['fields_frames_differing']} and {c['budgets_frames_differing']}",
                f"{c['sensor_rows']}; {c['sensor_rows_differing']}",
                f"{c['digest_frames']}; {sum(v['frames_differing'] for v in digests.values())}",
                ", ".join(
                    c["alerts_differing"]
                    + c["arrival_differing"]
                    + c["recovery_differing"]
                )
                or "none",
                yes(all(c["decisions_equal"].values())),
            ]
        )
    held = summary("core", "hold80r")
    checkpoints = compare["pairs"]["clean checkpoints"]
    start = float(
        summary("core", "hold80r_inside_restart")["checkpoint_recovery"]["room_since"]
    )
    begins = round(start / held["intervals"][0]["dt"])
    return (
        [
            "### T4b. The clean-out's restarts: the 80x30 held clean-out stopped after a step and continued through the core, from before and from inside its within-spec stretch, against the run without a stop",
            "",
            *table(
                [
                    "Restarted",
                    "Stopped after step",
                    "From step",
                    "Room within spec at the checkpoint",
                    "Classes' starts set",
                    "Mean times set",
                    "Frames; differing (fields and budgets)",
                    "Sensor rows; differing",
                    "Frames after the restart with tracker digests; differing",
                    "Record, recovery and map differing",
                    "Decisions equal",
                ],
                rows,
            ),
            "",
            f"The held clean-out writes a checkpoint every 100 steps and holds at step {held['steps']} (t = {held['t_final']:.5g} s); its within-spec stretch begins at step {begins} (t = {start:.5g} s), read from the checkpoint inside it. Each restart's file against the uninterrupted run's (`hold80r`), bit for bit in every frame, sensor row, tracker digest, end record and decision; the checkpointed run against the core-driven held clean-out of T2, which writes none: {yes(checkpoints['bitwise'])}. The checkpoint's recovery state is read from the file before the restart. The free-area means do not reach their 100:1 targets within the held run, so the means' times are unset at both checkpoints: the restarts pin the room's start and the classes' starts, and the decay case of section 4 pins the means' times.",
        ],
        bitwise,
    )


def round0_table(compare: dict) -> tuple[list[str], bool]:
    """T4c: round 1's core-driven runs against round 0's, made before the binding's changes."""
    rows = []
    bitwise = True
    for name in RUN_ORDER:
        c = compare["pairs"][f"round 0 {name}"]
        digests = c["digests"]
        bitwise = bitwise and c["bitwise"]
        rows.append(
            [
                LABELS[name],
                c["steps"],
                f"{c['frames']}; {c['fields_frames_differing']} and {c['budgets_frames_differing']}",
                f"{c['sensor_rows']}; {c['sensor_rows_differing']}",
                f"{c['digest_frames']}; {sum(v['frames_differing'] for v in digests.values())}",
                ", ".join(
                    c["alerts_differing"]
                    + c["arrival_differing"]
                    + c["recovery_differing"]
                )
                or "none",
                yes(all(c["decisions_equal"].values())),
            ]
        )
    return (
        [
            "### T4c. Round 1's core-driven runs against round 0's: the binding changed under them, the arithmetic did not",
            "",
            *table(
                [
                    "Run",
                    "Steps",
                    "Frames; differing (fields and budgets)",
                    "Sensor rows; differing",
                    "Frames with tracker digests; differing",
                    "Record, recovery and map differing",
                    "Decisions equal",
                ],
                rows,
            ),
            "",
            "Round 0's files were made at e152bbd, round 1's at the commit its summaries name, each from the same flow states and the same configuration; the six runs of T2 and T3 are round 1's.",
        ],
        bitwise,
    )


def junit_cases(path: Path) -> list[tuple[str, str, str]]:
    """(class name, test name, outcome) for every case of a JUnit file."""
    root = ET.parse(path).getroot()
    out = []
    for case in root.iter("testcase"):
        outcome = "passed"
        if case.find("failure") is not None or case.find("error") is not None:
            outcome = "failed"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        out.append((case.get("classname", ""), case.get("name", ""), outcome))
    return out


def tests_table() -> list[str]:
    """T6: the tests: the new parity cases by function, the totals, CTest, the suites."""
    cases = junit_cases(RESULTS / "parity.xml")
    by_function: dict[str, list[str]] = {}
    for cls, name, outcome in cases:
        if cls.endswith("test_core_march"):
            by_function.setdefault(name.split("[")[0], []).append(outcome)
    rows = [
        [f"`{f}`", len(o), o.count("passed"), o.count("failed"), o.count("skipped")]
        for f, o in by_function.items()
    ]
    total = [c[2] for c in cases]
    rows.append(
        [
            "the parity modules (`test_core_parity.py` and `test_core_march.py`)",
            len(total),
            total.count("passed"),
            total.count("failed"),
            total.count("skipped"),
        ]
    )
    ctest = junit_cases(RESULTS / "ctest.xml")
    outcomes = [c[2] for c in ctest]
    rows.append(
        [
            "CTest",
            len(outcomes),
            outcomes.count("passed"),
            outcomes.count("failed"),
            outcomes.count("skipped"),
        ]
    )
    new_gtests = [n for _, n, _ in ctest if n.split(".")[0] in NEW_GTEST_SUITES]
    suites = []
    for label, file in (("Linux", "suite_linux.xml"), ("Windows", "suite_windows.xml")):
        if (RESULTS / file).exists():
            counts = tables86.junit(RESULTS / file)
            suites.append(
                f"- The whole suite under {label}: {counts['passed']} passed, {counts['failures'] + counts['errors']} failed, {counts['skipped']} skipped in {counts['seconds']:.0f} s."
            )
    return [
        "### T7. The tests",
        "",
        *table(["Test", "Cases", "Passed", "Failed", "Skipped"], rows),
        "",
        f"New GoogleTest cases ({len(new_gtests)}): " + ", ".join(new_gtests) + ".",
        *suites,
    ]


def plants_table() -> tuple[list[str], int, int]:
    """T8: the planted defects, which tests and whole runs catch each."""
    records = load("plants.json")
    rows = []
    caught = 0
    for r in records:
        if "build" in r:
            rows.append([f"`{r['name']}`", r["what"], r["build"], "", ""])
            continue
        parity = (
            ", ".join(f"`{k}` {v}" for k, v in r["parity_failed_by_test"].items())
            or "none"
        )
        whole = ", ".join(f"{k} {v}" for k, v in r["whole_runs"].items())
        ctest = r["ctest"]
        if r["ctest_failed"]:
            ctest += ": " + ", ".join(f"`{n}`" for n in r["ctest_failed"])
        failing = bool(r["ctest_failed"]) or bool(r["parity_failed_by_test"])
        if r["name"] != "control":
            caught += failing
        rows.append([f"`{r['name']}`", r["what"], ctest, parity, whole])
    planted = sum(1 for r in records if r["name"] != "control")
    return (
        [
            "### T8. The planted defects: which tests catch each",
            "",
            "Each built alone from the checkout's `core/` with its edits (`docs/reports/probe91/variants91.py`), then CTest, the march's parity module (`tests/test_core_march.py`), and two whole 80x30 runs through the core (the dust release and the held clean-out) against this pass's Linux references.",
            "",
            *table(
                [
                    "Plant",
                    "What it changes",
                    "CTest",
                    "Parity cases failing (by test)",
                    "Whole runs against the reference",
                ],
                rows,
            ),
            "",
            f"{caught} of {planted} plants fail at least one test. In the last column a count of frames is of the frames whose class fields or budget terms differ, then what else differs (the sensor table, `/arrival/`, `/recovery/`); `raised` is a run the driver stopped, its check of the core's readings against the monitor's, or of a march's rows against the frames, failing. The whole-run column cannot see a breach, a refusal or a restore: neither run breaches, is refused or restarts, so a plant in those rules reads `equal` there whatever it changes; they are T8b's, which are not run on whole runs, and T4b's whole restarts.",
        ],
        caught,
        planted,
    )


def plants_round1_table() -> tuple[list[str], int, int, list[str]]:
    """T8b: round 1's planted defects, which tests catch each (CTest and the march's parity module)."""
    sys.path.insert(0, str(HERE))
    import variants91b

    records: dict[str, dict] = {}
    controls = []
    for path in sorted(RESULTS.glob("plants_round1_s*.json")):
        for r in json.loads(path.read_text(encoding="utf-8")):
            if r["name"].startswith("control"):
                controls.append(r)
            else:
                records[r["name"]] = r
    rows = []
    survivors = []
    for name, _what, _edits in variants91b.PLANTS:
        r = records.get(name)
        if r is None:
            rows.append([f"`{name}`", "not run", "", ""])
            survivors.append(name)
            continue
        if "build" in r:
            rows.append([f"`{name}`", r["build"][:200], "", ""])
            survivors.append(name)
            continue
        failed = r["ctest_failed"]
        parity = r["parity_failed_by_test"]
        # A run of the module that died before its summary: the plant crashes the binding.
        crashed = r.get("pytest_returncode", 0) not in (0, 1) or not re.search(
            r" in [0-9.]+s", r["pytest"]
        )
        ctest = f"{len(failed)}" + (
            ": " + ", ".join(f"`{n}`" for n in failed) if failed else ""
        )
        by_test = ", ".join(f"`{k}` {v}" for k, v in parity.items()) or "none"
        if crashed:
            by_test = "the module crashed" + (
                "" if by_test == "none" else f"; {by_test}"
            )
        missed = not (failed or parity or crashed)
        verdict = "NO" if missed else "yes"
        if missed and name in variants91b.EQUIVALENT:
            verdict = "no: an equivalent mutant"
        rows.append([f"`{name}`", ctest, by_test, verdict])
        if missed:
            survivors.append(name)
    planted = len(variants91b.PLANTS)
    equivalent = [n for n in survivors if n in variants91b.EQUIVALENT]
    survivors = [n for n in survivors if n not in variants91b.EQUIVALENT]
    caught = planted - len(survivors) - len(equivalent)
    clean = all(
        c.get("ctest", "").startswith("100%")
        and re.search(r" in [0-9.]+s", c.get("pytest", ""))
        and " failed" not in c.get("pytest", "")
        for c in controls
    )
    return (
        [
            "### T8b. Round 1's planted defects: which tests catch each",
            "",
            "Each built alone from the checkout's `core/` with its edits (`docs/reports/probe91/variants91b.py`), then CTest and the march's parity module (`tests/test_core_march.py`); no whole runs. The plants are the tester's of test 91 (`results/tester91/plant.py`, its anchors updated where the binding changed), the plants of round 1's new tests, one plant per refusal of the binding and one per guard of the plan, time and count inputs. Each stream builds the unplanted core first and last.",
            "",
            *table(
                [
                    "Plant",
                    "CTest cases failing",
                    "Parity cases failing (by test)",
                    "Caught",
                ],
                rows,
            ),
            "",
            f"{caught} of {planted} plants fail at least one test"
            + "".join(
                f"; `{n}` is an equivalent mutant, as {variants91b.EQUIVALENT[n]}"
                for n in equivalent
            )
            + (f"; not caught: {', '.join(survivors)}" if survivors else "")
            + f". A plant that makes the module crash instead of fail is counted as caught. The controls ({len(controls)} runs, at the start and the end of each stream) pass every test: {yes(clean)}.",
        ],
        caught,
        planted,
        survivors,
    )


def timing_table() -> tuple[list[str], dict]:
    """T9: the 320x120 window, three sessions."""
    rows, states = [], []
    ratios = {}
    for n in (1, 2, 3):
        path = RESULTS / f"timing_{n}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        ratios[n] = r["ratios"]
        for row in r["rows"]:
            if row["march"] == "reference":
                continue
            rows.append(
                [
                    n,
                    row["march"],
                    row["steps"],
                    row["core_calls"],
                    f"{r['ratios'][row['march']]:.2f}",
                    yes(r["ends_equal_the_reference"][row["march"]]),
                ]
            )
        calib = r["calibration_after"]["median_s"] / r["calibration_before"]["median_s"]
        states.append(
            [
                n,
                r["started"],
                f"{calib:.2f}",
                "; ".join(r["windows_before"][:3] + r["windows_before"][-2:]),
                "; ".join(r["windows_after"][:3] + r["windows_after"][-2:]),
            ]
        )
    lines = [
        "### T9. The 320x120 dust release's march over 200 steps across the release: the reference's over the core-driven",
        "",
        *table(
            [
                "Session",
                "Core-driven march",
                "Steps",
                "Calls to the core",
                "Reference over core-driven",
                "Ends equal the reference's, bit for bit",
            ],
            rows,
        ),
        "",
        "Each march from the Linux reference's frame at step 2,700 (about 9.8 s) of the 320x120 dust release through the onset at 10 s, with fresh trackers and monitor, frames written as the run writes them: the reference's `_Run.march`, then the core-driven march with the monitor reading the fields every step, then with the monitor reading the core's readings, each once, in one process. Only ratios within a session are given; the times stay in `results/builder91/timing_N.json`.",
        "",
        *table(
            [
                "Session",
                "Started",
                "Calibration, after over before",
                "Windows before",
                "Windows after",
            ],
            states,
        ),
    ]
    return lines, ratios


def loops_table() -> list[str]:
    """T10: the vectorised loops of the new modules."""
    v = load("vecinfo.json")
    rows = [
        [
            f"`{name}`",
            d["loops"],
            d["report_lines"],
            "; ".join(f"`{p}`" for p in d["places"]),
        ]
        for name, d in v["files"].items()
    ]
    return [
        "### T10. The loops GCC vectorises in the new modules",
        "",
        f"`{v['compiler']}`, `{' '.join(v['flags'])}`, `-fopt-info-vec-optimized`: loops vectorised, each counted once by its place in the source.",
        "",
        *table(
            ["Source", "Loops vectorised", "Report lines", "Where (line: source)"], rows
        ),
    ]


def predictions_table(
    march_bitwise: bool,
    decisions_equal: bool,
    windows: dict,
    windows_equal: bool,
    restart_bitwise: bool,
    ratios: dict,
) -> list[str]:
    """T0: the predictions, scored by the rule committed with them."""
    first = [t for t in windows["tiers"].values()]
    inside = sum(1 for t in first if t == tables86.INSIDE)
    closure = max(windows["closure"].values())
    speed = ratios.get(1, {}).get("core, monitor on the readings", math.nan)
    others = [ratios[n]["core, monitor on the readings"] for n in (2, 3) if n in ratios]
    fields_speed = [ratios[n]["core, monitor on the fields"] for n in sorted(ratios)]
    rows = [
        [
            "(a) the core-driven march equals the Linux reference's bit for bit at every output step and in every decision",
            f"six runs, every frame, sensor row and tracker digest without a differing bit: {yes(march_bitwise)} (T2); every decision equal: {yes(decisions_equal)} (T3)",
            "holds" if march_bitwise and decisions_equal else "fails",
        ],
        [
            "(b) against the Windows records every decision equal and every field inside the band's first tier",
            f"every decision equal: {yes(windows_equal)} (T6); {inside} of {len(first)} field readings inside the measured spread, the closure at most {e(closure)} (T5)",
            "holds"
            if windows_equal and inside == len(first) and closure <= tables86.CLOSURE
            else "fails",
        ],
        [
            "(c) the restart continues the uninterrupted run bit for bit",
            f"every frame, sensor row, tracker digest and end record equal: {yes(restart_bitwise)} (T4)",
            "holds" if restart_bitwise else "fails",
        ],
        [
            "(d) the 320x120 window, monitor reads removed, at least five times faster than the reference's",
            f"{speed:.2f} in session 1 (sessions 2 and 3: {', '.join(f'{x:.2f}' for x in others)}); with the monitor's field reads {', '.join(f'{x:.2f}' for x in fields_speed)} (T9)",
            "holds" if speed >= SPEED else "fails",
        ],
    ]
    return [
        "### T0. The predictions, scored by the rule committed with them",
        "",
        *table(["Prediction", "Measured", "Verdict"], rows),
    ]


def build() -> tuple[str, dict[str, str]]:
    """Every table, as the tables file's text and the report's blocks."""
    compare = load("compare.json")
    run_lines = runs_table(compare)
    march_lines, march_bitwise = march_table(compare)
    decision_lines, decisions_equal = decisions_table(compare)
    results, place = windows_comparison()
    field_lines, windows = windows_fields(results)
    windows_lines, windows_equal = windows_decisions(results, place)
    restart_lines, restart_bitwise = restart_table(compare)
    plant_lines, _caught, _planted = plants_table()
    round1_lines, _caught1, _planted1, _survivors = plants_round1_table()
    clean_restart_lines, _clean_bitwise = clean_restart_table(compare)
    round0_lines, _round0_bitwise = round0_table(compare)
    timing_lines, ratios = timing_table()
    blocks = {
        "predictions": "\n".join(
            predictions_table(
                march_bitwise,
                decisions_equal,
                windows,
                windows_equal,
                restart_bitwise,
                ratios,
            )
        ),
        "runs": "\n".join(run_lines),
        "march": "\n".join(march_lines),
        "decisions": "\n".join(decision_lines),
        "windows": "\n".join([*field_lines, "", *windows_lines]),
        "restart": "\n".join(restart_lines),
        "cleanrestart": "\n".join(clean_restart_lines),
        "round0": "\n".join(round0_lines),
        "tests": "\n".join(tests_table()),
        "plants": "\n".join(plant_lines),
        "plantsround1": "\n".join(round1_lines),
        "timing": "\n".join(timing_lines),
        "loops": "\n".join(loops_table()),
    }
    text = "# Prompt 91's tables\n\nGenerated by `tables91.py` from `results/builder91/` and `results/builder85/`.\n\n"
    text += "\n\n".join(blocks[k] for k in MARKERS)
    return text + "\n", blocks


def splice(report: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """REPORT with each marked block replaced, and the names whose markers are missing."""
    missing = []
    for name, body in blocks.items():
        begin, end = f"<!-- tables91 {name} begin -->", f"<!-- tables91 {name} end -->"
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
        ok = (
            TABLES.read_text(encoding="utf-8") == text
            and spliced == report
            and not missing
        )
        print(
            "tables91 --check:",
            "matches" if ok else "DIFFERS",
            f"(missing markers: {missing})" if missing else "",
        )
        return 0 if ok else 1
    TABLES.write_text(text, encoding="utf-8")
    REPORT.write_text(spliced, encoding="utf-8")
    print(
        "wrote",
        TABLES,
        "and spliced",
        REPORT,
        f"(missing markers: {missing})" if missing else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
