"""The tables of prompt 96's report, from the records in results/builder96/.

    python docs/reports/probe96/tables96.py           # write tables96.md
    python docs/reports/probe96/tables96.py --check   # regenerate and compare, write nothing

Every number in ``docs/reports/phase6_backend.md``'s results comes from here.
The records: ``compare.json`` (``probe96.py compare``: every pair by
``scripts/compare_runs.py`` and the placement), each run's record beside its file
under ``runs/``, the timing sessions under ``timing/``, the planted defects
(``plants_windows.json``, ``plants_linux.json``), the hash tests
(``hash_tests.json``) and the suites' JUnit files. Each record names the commit it
was made at, and the tables refuse a record that names another.
"""

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "results" / "builder96"
TABLES = HERE / "tables96.md"
SCENARIOS = ("clean80", "door80", "dust80", "breach80", "seal80")
LABELS = {
    "clean80": "clean-out",
    "door80": "door leak",
    "dust80": "dust release",
    "breach80": "filter breach",
    "seal80": "seal leak",
}
PROMPT91 = ("dust80", "door80", "breach80")
INSIDE = "inside the spread"
SPEEDUP = 4.0
SESSIONS = (1, 2, 3)


def load(path: Path) -> dict:
    """A JSON record."""
    return json.loads(path.read_text(encoding="utf-8"))


def e(value: object) -> str:
    """A difference: 0, inf, or two significant figures."""
    if isinstance(value, str):
        return value
    if value == 0.0:
        return "0"
    if math.isinf(value):
        return "inf"
    return f"{value:.1e}"


def yes(flag: bool) -> str:
    """yes or no."""
    return "yes" if flag else "no"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A markdown table."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


# ---------------------------------------------------------------------------
# The records


class Records:
    """Every record the tables read, checked to name one commit."""

    def __init__(self) -> None:
        self.compare = load(RESULTS / "compare.json")
        self.commit = self.compare["commit"]
        self.pairs = {(p["label"], p["kind"]): p for p in self.compare["pairs"]}
        self.runs = {}
        for side in ("reference", "core", "fields", "prompt91", "restart"):
            folder = RESULTS / "runs" / side
            for path in sorted(folder.glob("*.json")):
                self.runs[(side, path.stem)] = load(path)
        self.timing = {}
        for session in SESSIONS:
            folder = RESULTS / "timing" / f"session_{session}"
            if (folder / "state.json").exists():
                self.timing[session] = {
                    "state": load(folder / "state.json"),
                    **{
                        side: load(folder / f"{side}.json")
                        for side in ("reference", "core")
                    },
                }
        self.plants = [
            load(RESULTS / f"plants_{platform}.json")
            for platform in ("windows", "linux")
        ]
        self.hashes = load(RESULTS / "hash_tests.json")
        commits = {self.commit, self.hashes["head"]}
        commits |= {r["commit"] for r in self.runs.values() if "commit" in r}
        for session in self.timing.values():
            commits |= {session[side]["commit"] for side in ("reference", "core")}
        for platform in self.plants:
            commits.add(platform["commit"])
        if len(commits) != 1:
            raise SystemExit(
                f"the records name more than one commit: {sorted(commits)}"
            )

    def pair(self, label: str, kind: str) -> dict:
        """One pair's comparison."""
        return self.pairs[(label, kind)]


def items(pair: dict, criterion: str) -> list[dict]:
    """A pair's items of one criterion."""
    return [i for i in pair["items"] if i["criterion"] == criterion]


def item(pair: dict, prefix: str) -> dict:
    """A pair's one item whose name starts with PREFIX."""
    (found,) = [i for i in pair["items"] if i["name"].startswith(prefix)]
    return found


def number(value: object) -> float:
    """A record's number, its infinities written as text read back."""
    return float(value)


# ---------------------------------------------------------------------------
# The tables


def runs_table(r: Records) -> list[str]:
    """T1: every run, its exit, its flow states and its steps."""
    rows = []
    for name in SCENARIOS:
        pair = r.pair(name, "backends")
        stops = items(pair, "stop")
        last = (
            item(pair, "the run's last step")["detail"]
            if any(i["name"].startswith("the run's last step") for i in pair["items"])
            else {"ref": "-", "other": "-"}
        )
        for side, key in (("reference", "ref"), ("core", "other")):
            record = r.runs[(side, name)]
            flows = "; ".join(
                f"{s['detail'][f'{key} outer']} ({s['detail'][f'{key} reason']}"
                + (", capped" if s["detail"][f"{key} capped"] else "")
                + ")"
                for s in stops
            )
            rows.append(
                [
                    LABELS[name],
                    side,
                    str(record["exit"]),
                    flows,
                    str(last[key]),
                    f"{record['wall_seconds']:.1f}",
                ]
            )
    return [
        "### T1. The runs: each scenario through `scripts/run_scenario.py` on each backend",
        "",
        *table(
            [
                "scenario",
                "backend",
                "exit",
                "flow states: outer (stop)",
                "last step",
                "wall, s",
            ],
            rows,
        ),
        "",
        "Each from rest, three processes at once at most, one BLAS thread, NumPy's AVX-512 loops off. The flow states in fingerprint order. The wall time is the script's process, set-up included, beside two other runs, so it is not a timing (T9 is).",
    ]


def stops_table(r: Records) -> tuple[list[str], bool, bool]:
    """T2: every flow state's stop on both sides."""
    rows, equal, within = [], True, True
    for name in SCENARIOS:
        for s in items(r.pair(name, "backends"), "stop"):
            d = s["detail"]
            equal = equal and d["apart"] == 0 and d["ref reason"] == d["other reason"]
            within = within and s["passed"]
            rows.append(
                [
                    LABELS[name],
                    s["name"].removeprefix("flow state "),
                    f"{d['ref outer']} / {d['other outer']}",
                    str(d["apart"]),
                    d["ref reason"],
                    d["other reason"],
                ]
            )
    return (
        [
            "### T2. Every flow state's stop: the outer count and the reason, the reference / the core",
            "",
            *table(
                [
                    "scenario",
                    "state",
                    "outer, ref / core",
                    "apart",
                    "reason, ref",
                    "reason, core",
                ],
                rows,
            ),
            "",
            "ADR-017 H item 3: within one is the criterion; prediction (c) asks for none apart. A state capped on a side would say so after its reason.",
        ],
        equal,
        within,
    )


def decisions_table(r: Records) -> tuple[list[str], bool]:
    """T3: every decision item of the five pairs."""
    rows, equal = [], True
    for name in SCENARIOS:
        for i in items(r.pair(name, "backends"), "decision"):
            d = i["detail"]
            equal = equal and i["passed"]
            reach = "-"
            if "compared" in d:
                text = f"{d['compared']:,}"
                moves = f"{d['moved one']} / {d['moved more']} / {d['one only']}"
                if "reached, ref" in d:
                    reach = f"{d['reached, ref']:,} / {d['reached, other']:,}"
            else:
                ref, other = (
                    ("ref", "other") if "ref" in d else ("ref steps", "other steps")
                )
                text, moves = f"{d[ref]} / {d[other]}", "-"
            rows.append([LABELS[name], i["name"], text, moves, reach, yes(i["passed"])])
    return (
        [
            "### T3. Every decision, the core against the reference",
            "",
            *table(
                [
                    "scenario",
                    "decision",
                    "compared, or ref / core",
                    "moved 1 / > 1 / one only",
                    "reached, ref / core",
                    "equal",
                ],
                rows,
            ),
            "",
            "`scripts/compare_runs.py`: a decision is equal only when it falls at the same step or the same frame, so a time moves by the difference of the indexes of the steps it falls on (the arrival maps per event, channel and cell; the file's sensors' latencies per event, channel and sensor, by `AlertMonitor.latencies` on the alert table; the clean-out's recovery times), a clear time by that of the stored frames, an alert's opening and closing step by the step; one only is a time reached in one run and never in the other. The budget verdict is each class's residual and leftover against REQ-T05 at the end. Reached is how many of the compared entries each run reaches: the finite arrival times, latencies and recovery times, the alerts, and the cells and channels read at or above a threshold after t_e (a clear time of t_e itself is a cell nothing reached). A decision no run reaches compares as no move, so the column says what \"every decision equal\" covers.",
        ],
        equal,
    )


def fields_table(r: Records) -> tuple[list[str], bool]:
    """T4: each field's largest difference and its tier, per scenario."""
    rows, inside = [], True
    for name in SCENARIOS:
        pair = r.pair(name, "backends")
        largest: dict[str, tuple[float, dict]] = {}
        for i in items(pair, "field"):
            field = i["name"].split(", state")[0]
            value = number(i["detail"]["difference"])
            if field not in largest or value > largest[field][0]:
                largest[field] = (value, i["detail"])
            inside = inside and i["detail"]["tier"] == INSIDE
        for field, (value, d) in largest.items():
            rows.append(
                [
                    LABELS[name],
                    field,
                    e(value),
                    e(number(d["spread"])),
                    e(number(d["limit"])),
                    d["tier"],
                ]
            )
        closure = item(pair, "the budget's closure")["detail"]
        rows.append(
            [
                LABELS[name],
                "budget closure (ref / core)",
                f"{e(number(closure['ref']))} / {e(number(closure['other']))}",
                "-",
                e(number(closure["limit"])),
                "not tiered",
            ]
        )
    return (
        [
            "### T4. Each field's largest difference from the reference and its tier (ADR-017 H item 2)",
            "",
            *table(
                ["scenario", "field", "difference", "measured spread", "limit", "tier"],
                rows,
            ),
            "",
            "A flow field's over every flow state, max |difference| over max |reference|; the class fields over every stored frame, the largest over the classes; pointwise where the reference reads at least a hundredth of the class's threshold; the sensor channels at every step; the leftover's change over the supply at the end. The spreads are prompt 85's P8 (the sensor channels P2's), the limits ten times them rounded up to a decade. The closure is the largest |residual| over the supply on any stored frame, each side, against the reference's own 1e-12.",
        ],
        inside,
    )


def placement_table(r: Records) -> tuple[list[str], bool]:
    """T5: the placement's searched layouts, order and unit latencies."""
    placement = r.compare["placement"] or []
    rows = []
    for i in placement:
        d = i["detail"]
        if "compared" in d:
            text = f"{d['compared']} units: moved {d['moved one']} / {d['moved more']} / {d['one only']}"
        else:
            ref, other = d["ref"], d["other"]
            text = "same" if ref == other else f"{ref} / {other}"
        # The file's layout is named "the file's four"; the item reads "the {name}".
        rows.append([i["name"].replace("the the ", "the "), text, yes(i["passed"])])
    equal = bool(placement) and all(i["passed"] for i in placement)
    return (
        [
            "### T5. The placement on the four scenarios with a contaminating event, the core's maps against the reference's",
            "",
            *table(["decision", "core against reference", "equal"], rows),
            "",
            "`compare_runs.compare_placement` over the door leak, the dust release, the filter breach and the seal leak: `placement.search` under both rules and every pool of the base file's placement section, the file's four sensors beside them, `placement.rank`'s order, and every unit's latency of every layout in steps.",
        ],
        equal,
    )


BITWISE = (
    "flow states",
    "step times",
    "frames",
    "budgets",
    "sensors",
    "alerts",
    "latencies",
    "arrival maps",
)


def bitwise_table(
    r: Records, title: str, kind: str, labels: tuple[str, ...], names: dict[str, str]
) -> tuple[list[str], bool]:
    """A table of pairs' bitwise flags."""
    rows, every = [], True
    for label in labels:
        pair = r.pair(label, kind)
        flags = pair["bitwise"]
        every = every and all(flags.values())
        rows.append(
            [
                names.get(label, label),
                *[yes(flags.get(f, True)) for f in BITWISE],
                yes(flags["recovery"]) if "recovery" in flags else "n/a",
                pair["verdict"],
            ]
        )
    return (
        [
            title,
            "",
            *table(["run", *BITWISE, "recovery", "verdict"], rows),
            "",
        ],
        every,
    )


def timing_table(r: Records) -> tuple[list[str], list[float]]:
    """T9: the timing sessions."""
    rows, ratios, states = [], [], []
    for session, record in sorted(r.timing.items()):
        ref, core = record["reference"], record["core"]
        ratio = ref["total"] / core["total"]
        ratios.append(ratio)
        for side in ("reference", "core"):
            s = record[side]
            rows.append(
                [
                    str(session),
                    side,
                    f"{s['total']:.2f}",
                    f"{s['flow']:.2f}",
                    f"{s['march']:.2f}",
                    f"{s['rest']:.2f}",
                    f"{ratio:.2f}" if side == "core" else "-",
                ]
            )
        state = record["state"]
        before = "; ".join(
            state["before"][:3]
            + [x for x in state["before"] if x.startswith(("on mains", "processor"))]
        )
        states.append(
            f"- Session {session} ({state['started']}, order {', '.join(state['order'])}): before, {before}."
        )
    return (
        [
            "### T9. The 80x30 dust release end to end on each backend, three sessions",
            "",
            *table(
                [
                    "session",
                    "backend",
                    "total, s",
                    "flow solve, s",
                    "march, s",
                    "rest, s",
                    "reference / core",
                ],
                rows,
            ),
            "",
            "Each run is `run_scenario.main` in a process of its own, alone on the machine, the two backends in alternating order. The flow solve is `_Run._solve` (the solver's set-up, its solve and the flow state's write), the march `march` of the run's class, the rest the set-up, the plans and the end's writes. Windows' busiest processes and the power state, sampled before each session:",
            "",
            *states,
        ],
        ratios,
    )


def plants_table(r: Records) -> tuple[list[str], int, int]:
    """T10: the planted defects."""
    rows, caught, total, equivalent = [], 0, 0, []
    for platform in r.plants:
        for p in platform["plants"]:
            if p.get("equivalent") and not p["caught"]:
                equivalent.append(f"`{p['name']}`: {p['equivalent']}")
                result = "equivalent"
            else:
                total += 1
                caught += p["caught"]
                result = "caught" if p["caught"] else "SURVIVES"
            rows.append(
                [
                    p["name"],
                    p["what"],
                    p["file"],
                    platform["platform"],
                    p["tests"],
                    result,
                    str(p["summary"]),
                ]
            )
    clean = "; ".join(
        f"{p['platform']}: {p['unchanged_tree']['summary']}" for p in r.plants
    )
    return (
        [
            "### T10. The planted defects",
            "",
            *table(
                [
                    "plant",
                    "defect",
                    "file",
                    "platform",
                    "tests run",
                    "result",
                    "pytest's last line",
                ],
                rows,
            ),
            "",
            f"Each plant in a `git archive` copy of the commit, its tests run there, the copy restored after it; caught when pytest exits other than 0. The unchanged copy, every test the plants ran: {clean}. Equivalent, changing no result: "
            + ("; ".join(equivalent) or "none")
            + ".",
        ],
        caught,
        total,
    )


def junit(path: Path) -> dict:
    """A JUnit file's counts."""
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    out = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        for key in out:
            out[key] += int(suite.get(key, 0))
    return out


def tests_table(r: Records) -> list[str]:
    """T11: the suites at the records' commit."""
    rows = []
    for name, path in (
        ("Windows suite", "windows.xml"),
        ("Linux suite, the core required", "linux.xml"),
        ("CTest", "ctest.xml"),
    ):
        counts = junit(RESULTS / path)
        passed = (
            counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
        )
        rows.append(
            [
                name,
                str(counts["tests"]),
                str(passed),
                str(counts["failures"] + counts["errors"]),
                str(counts["skipped"]),
            ]
        )
    h = r.hashes
    return [
        "### T11. The suites, and the stored hash tests",
        "",
        *table(["suite", "tests", "passed", "failed", "skipped"], rows),
        "",
        f"The test modules that assert a stored hash (a 64-digit literal, `sha256` or `hexdigest`): {', '.join(f'`{t}`' for t in h['hash_tests'])}. Edited by this branch ({h['base'][:7]}..{h['head'][:7]}): {', '.join(f'`{t}`' for t in h['hash_tests_edited']) or 'none'}.",
    ]


def predictions_table(scores: dict[str, tuple[str, str]]) -> list[str]:
    """T0: the predictions scored."""
    texts = {
        "a": "Under `reference`, prompt 91's three 80x30 runs reproduce bit for bit, and no stored hash test changes",
        "b": "The update from readings gives the alerts and latencies of the update from fields bit for bit on every run compared",
        "c": "On all five 80x30 scenarios under `core`, every decision equals the reference's, every flow state stops at the reference's outer count, not one apart, and every field is inside the measured spread",
        "d": "A restart under `core` is the uninterrupted core run bit for bit",
        "e": "The 80x30 dust release runs end to end at least four times faster under `core`, same session",
    }
    rows = [[f"({k})", texts[k], scores[k][0], scores[k][1]] for k in "abcde"]
    return [
        "### T0. The predictions scored",
        "",
        *table(["", "prediction", "result", "verdict"], rows),
    ]


def build() -> str:
    """Every table, from the records."""
    r = Records()
    t1 = runs_table(r)
    t2, stops_equal, stops_within = stops_table(r)
    t3, decisions_equal = decisions_table(r)
    t4, inside = fields_table(r)
    t5, placement_equal = placement_table(r)
    t6, a_runs = bitwise_table(
        r,
        "### T6. Prediction (a): prompt 91's three 80x30 reference runs made again through the script, against their records, bit for bit",
        "prompt91",
        PROMPT91,
        LABELS,
    )
    t7, b_runs = bitwise_table(
        r,
        "### T7. Prediction (b): the core's run with the monitor on the fields against the same run with it on the readings, bit for bit",
        "monitor",
        SCENARIOS,
        LABELS,
    )
    t8, d_runs = bitwise_table(
        r,
        "### T8. Prediction (d): the core's restarts against the uninterrupted core runs, bit for bit",
        "restart march",
        ("dust80_c1000",),
        {"dust80_c1000": "dust release, stopped after step 1,500 in the march"},
    )
    t8b, d_flow = bitwise_table(
        r,
        "",
        "restart flow",
        ("dust80_cap600",),
        {"dust80_cap600": "dust release, capped at 600 outer iterations"},
    )
    t8c, cadence = bitwise_table(
        r,
        "",
        "checkpoints",
        ("dust80_c1000",),
        {
            "dust80_c1000": "dust release checkpointed every 1,000 steps, against the run that is not"
        },
    )
    t9, ratios = timing_table(r)
    t10, caught, total = plants_table(r)
    t11 = tests_table(r)
    a_hash = not r.hashes["hash_tests_edited"]
    c_holds = stops_equal and decisions_equal and inside and placement_equal
    restart = r.runs[("restart", "dust80_c1000_restart")]
    scores = {
        "a": (
            f"three runs bitwise: {yes(a_runs)}; stored hash tests edited: {len(r.hashes['hash_tests_edited'])} (T6, T11)",
            "HOLDS" if a_runs and a_hash else "FAILS",
        ),
        "b": (
            f"five pairs, alerts, latencies and sensor tables bitwise: {yes(b_runs)} (T7)",
            "HOLDS" if b_runs else "FAILS",
        ),
        "c": (
            f"stops none apart: {yes(stops_equal)} (T2); decisions equal: {yes(decisions_equal)} (T3), placement: {yes(placement_equal)} (T5); every field inside the spread: {yes(inside)} (T4)",
            "HOLDS" if c_holds else "FAILS",
        ),
        "d": (
            f"march restart from step {restart['checkpoint_step']:,} bitwise: {yes(d_runs)}; flow restart from its steady checkpoint bitwise: {yes(d_flow)} (T8)",
            "HOLDS" if d_runs and d_flow else "FAILS",
        ),
        "e": (
            "reference / core "
            + ", ".join(f"{x:.1f}" for x in ratios)
            + f" in sessions {', '.join(str(s) for s in sorted(r.timing))} (T9)",
            "HOLDS"
            if len(ratios) == len(SESSIONS) and min(ratios) >= SPEEDUP
            else "FAILS",
        ),
    }
    head = [
        "# Prompt 96's tables (generated by `tables96.py` from `results/builder96/`; do not edit)",
        "",
        f"Every record was made at {r.commit}.",
        "",
    ]
    blocks = [
        head,
        predictions_table(scores),
        t1,
        t2,
        t3,
        t4,
        t5,
        t6,
        t7,
        t8,
        t8b[2:],
        t8c[2:],
        t9,
        t10,
        t11,
        [
            f"Planted defects caught: {caught} of {total}. Stops within one: {yes(stops_within)}. A checkpoint cadence changes no bit: {yes(cadence)}."
        ],
    ]
    return "\n".join(line for block in blocks for line in [*block, ""]).rstrip() + "\n"


def main() -> int:
    """Write tables96.md, or with --check compare it with the records' tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = build()
    if args.check:
        same = TABLES.read_text(encoding="utf-8").replace("\r\n", "\n") == text
        print(
            "tables96.md matches the records"
            if same
            else "tables96.md DIFFERS from the records"
        )
        return 0 if same else 1
    TABLES.write_text(text, encoding="utf-8")
    print("wrote", TABLES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
