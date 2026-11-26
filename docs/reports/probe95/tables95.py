"""The tables of docs/reports/phase6_flow.md, generated from prompt 95's records.

    python docs/reports/probe95/tables95.py          # writes tables95.md
    python docs/reports/probe95/tables95.py --check  # exits 1 if tables95.md differs

The records are in ``results/builder95/`` (not committed): the captures
(``captures/<case>.json``), the whole solves (``solves/<case>.json``), the timings
(``timing_<n>.json``), the vectorisation report (``vecinfo.json``), the plants
(``plants_s<n>.json``), CTest's, the parity modules' and the flow module's JUnit
records (``ctest.xml``, ``parity.xml``, ``flow.xml``) and the two suites'
(``linux.xml``, ``windows.xml``). Each JSON record names the commit it was taken at
(``commit``), each JUnit record has a file of that name beside it
(``ctest.commit`` and so on). A record that is absent is an error, not a missing
row, and so are records of two commits: the tables name the one commit.
"""

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "results" / "builder95"
OUT = HERE / "tables95.md"
# The cases that are captured and solved, then those that are only solved (the
# channel, which prompt 95b added for its pressure outlet).
CASES = ("product80", "product160", "cavity40", "door80")
SOLVE_ONLY = ("poiseuille40",)
CASE_NAMES = {
    "product80": "product room 80x30",
    "product160": "product room 160x60",
    "cavity40": "cavity 40x40",
    "door80": "door leak 80x30",
    "poiseuille40": "channel 40x20 (VAL-001's room)",
}
JUNIT_RECORDS = (
    ("ctest.xml", "CTest (GoogleTest, both programs)"),
    ("parity.xml", "the six parity modules, `-m parity`"),
    ("flow.xml", "`tests/test_core_flow.py`"),
    ("linux.xml", "the Linux suite, core built"),
    ("windows.xml", "the Windows suite, no core"),
)
FIELDS = ("u", "v", "p", "k", "eps", "nu_t")
BLOCKS = (
    "outlets",
    "effective_viscosity",
    "relaxation",
    "cell_centres",
    "velocity_step",
    "viscosity_step",
    "residual",
    "median",
    "imbalance_summary",
    "rule_verdict",
)


def load(path: Path) -> object:
    """A JSON record; its absence ends the run, so that no table silently loses a row."""
    if not path.exists():
        raise SystemExit(f"record absent: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def e(x: float | None) -> str:
    """A number in the tables' form."""
    if x is None:
        return "-"
    if x == 0:
        return "0"
    if math.isinf(x):
        return "inf"
    return f"{x:.1e}"


def state_name(case: str, state: dict) -> str:
    """A flow state's name: the door leak's two, every other case's one from rest."""
    first = state["index"] == 0 if "index" in state else state["state"] == "state0"
    if case == "door80":
        return "closed, from rest" if first else "open, warm from closed"
    return "from rest"


def junit_path(name: str) -> Path:
    """A JUnit record's path; its absence ends the run."""
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"record absent: {path.relative_to(ROOT)}")
    return path


def records_commit() -> str:
    """The one commit every record was taken at; records of two, or one without a commit, end the run."""
    named: dict[str, str] = {}
    for case in CASES:
        named[f"captures/{case}.json"] = load(RESULTS / "captures" / f"{case}.json")[
            "commit"
        ]
    for case in (*CASES, *SOLVE_ONLY):
        named[f"solves/{case}.json"] = load(RESULTS / "solves" / f"{case}.json")[
            "commit"
        ]
    for path in sorted(RESULTS.glob("timing_*.json")):
        named[path.name] = load(path)["commit"]
    named["vecinfo.json"] = load(RESULTS / "vecinfo.json")["commit"]
    for path in plant_paths():
        for row in load(path):
            named[f"{path.name}:{row['plant']}"] = row["commit"]
    for name, _ in JUNIT_RECORDS:
        stamp = junit_path(name).with_suffix(".commit")
        if not stamp.exists():
            raise SystemExit(f"record absent: {stamp.relative_to(ROOT)}")
        named[stamp.name] = stamp.read_text(encoding="utf-8").strip()
    commits = set(named.values())
    if len(commits) != 1:
        by_commit: dict[str, list[str]] = {}
        for name, commit in named.items():
            by_commit.setdefault(commit, []).append(name)
        detail = "; ".join(f"{c[:7]}: {', '.join(n[:3])}" for c, n in by_commit.items())
        raise SystemExit(f"the records are of {len(commits)} commits ({detail})")
    return commits.pop()


def timing_paths() -> list[Path]:
    """The timing sessions' records, three of them."""
    paths = [RESULTS / f"timing_{n}.json" for n in (1, 2, 3)]
    for path in paths:
        load(path)
    return paths


def plant_paths() -> list[Path]:
    """The plants' records: one per stream, and the streams' reruns of single plants."""
    paths = [RESULTS / f"plants_s{n}.json" for n in (0, 1, 2)]
    for path in paths:
        load(path)
    return [*paths, *sorted(RESULTS.glob("plants_s*_only.json"))]


def predictions() -> list[str]:
    """The five predictions, each scored by the rule committed with them."""
    states = [
        st
        for case in CASES
        for st in load(RESULTS / "captures" / f"{case}.json")["states"]
    ]
    differing = sum(c[1] for st in states for c in st["blocks"].values())
    calls = sum(c[0] for st in states for c in st["blocks"].values())
    parity = junit_path("parity.xml")
    parity_ok = junit(parity)["failures"] + junit(parity)["errors"] == 0
    a = differing == 0 and parity_ok
    worst = max(
        max(
            st["estimate"]["largest_relative"],
            st["viscosity_estimate"]["largest_relative"],
        )
        for st in states
    )
    opposite = sum(
        len(st["estimate"]["opposite_sides"])
        + len(st["viscosity_estimate"]["opposite_sides"])
        for st in states
    )
    b = worst <= 1e-11 and opposite == 0
    # The prediction names the first four cases' solves; the channel is outside it.
    solves = [
        st
        for case in CASES
        for st in load(RESULTS / "solves" / f"{case}.json")["states"]
    ]
    same_stop = all(st["stop_reason"][0] == st["stop_reason"][1] for st in solves)
    same_outer = all(st["outer"][0] == st["outer"][1] for st in solves)
    inside = all(
        f["tier"] == "inside the spread" for st in solves for f in st["fields"].values()
    )
    c = same_stop and same_outer and inside and len(solves) == 5
    root = ElementTree.parse(junit_path("ctest.xml")).getroot()
    cases = [t for t in root.iter("testcase") if "ThreeDirections" in t.get("name", "")]
    failed = [t for t in cases if t.find("failure") is not None]
    three = (len(cases), len(failed))
    d = three[1] == 0 and three[0] > 0
    ratio = load(RESULTS / "timing_1.json")["ratio"]
    e_ok = ratio >= 4.0

    def verdict(ok: bool) -> str:
        return "HOLDS" if ok else "FAILS"

    return [
        "### T0. The predictions, scored by the rule committed with them",
        "",
        "| prediction | the reading | verdict |",
        "|---|---|---|",
        f"| (a) every block of the first kind bit for bit on every planted and captured input | captured: {calls} calls in {len(states)} solves, {differing} differ in a bit; the parity modules: {'every case passes' if parity_ok else 'see T7'} | {verdict(a)} |",
        f"| (b) every captured estimate within 1e-11, relative, none on the other side of the tolerance | largest relative difference {e(worst)}; {opposite} iterations on opposite sides | {verdict(b)} |",
        f"| (c) every whole solve at the reference's stop and outer count, every flow field inside the measured spread | {len(solves)} solves: stops {'equal' if same_stop else 'differ'}, outer counts {'equal' if same_outer else 'differ'}, fields {'all inside the spread' if inside else 'see T3'} | {verdict(c)} |",
        f"| (d) the D = 3 instantiation builds and its tests pass | {three[0]} three-direction tests, {three[1]} fail | {verdict(d)} |",
        f"| (e) the 160x60 solve from rest at least four times faster, session 1 | {ratio:.2f} | {verdict(e_ok)} |",
    ]


def captured_blocks() -> list[str]:
    """T1: the blocks of the first kind on the captured solves."""
    lines = [
        "### T1. Captured outer iterations: the blocks of the first kind",
        "",
        "Each call of each block in the reference's own solve, replayed through the core on the reference's inputs as it ran. A cell is calls compared / calls that differ in any bit. The last three columns count the outer iterations with any block that differs: the first 50, the 300 before the stop, all.",
        "",
        "| case | state | stop | outer | "
        + " | ".join(BLOCKS)
        + " | first 50 | last 300 | all |",
        "|---|---|---|---|" + "---|" * len(BLOCKS) + "---|---|---|",
    ]
    for case in CASES:
        for state in load(RESULTS / "captures" / f"{case}.json")["states"]:
            cells = []
            for block in BLOCKS:
                counts = state["blocks"].get(block)
                cells.append("-" if counts is None else f"{counts[0]} / {counts[1]}")
            diff = state["iterations_with_a_difference"]
            lines.append(
                f"| {CASE_NAMES[case]} | {state_name(case, state)} | {state['stop_reason']} | {state['outer']} | "
                + " | ".join(cells)
                + f" | {diff['first']} of {state['first_count']} | {diff['last']} of {state['last_count']} | {diff['all']} of {state['outer']} |"
            )
    return lines


def captured_estimates() -> list[str]:
    """T2: the estimates of the second kind on the captured solves."""
    lines = [
        "### T2. Captured outer iterations: the estimates (the second kind)",
        "",
        "The core's estimate formed from the reference's steps against the reference's, at each outer iteration whose window is full: the largest relative difference and the outer iteration it falls at, and the iterations where one estimate is below `iteration_error_tol` and the other is not. At the stop, each side's estimate over the tolerance.",
        "",
        "| case | state | velocity: finite | largest relative | at | opposite sides | viscosity: finite | largest relative | at | opposite sides | at the stop, (a) reference / core | (e) reference / core |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in CASES:
        for state in load(RESULTS / "captures" / f"{case}.json")["states"]:
            a, b = state["estimate"], state["viscosity_estimate"]
            m = state["margins_at_stop"]
            visc = m.get("viscosity_estimate")
            lines.append(
                f"| {CASE_NAMES[case]} | {state_name(case, state)} | {a['finite']} | {e(a['largest_relative'])} | {a['at']} | {len(a['opposite_sides'])} | "
                + (
                    f"{b['finite']} | {e(b['largest_relative'])} | {b['at']} | {len(b['opposite_sides'])} | "
                    if b["finite"]
                    else "- | - | - | - | "
                )
                + f"{m['estimate'][0]:.6f} / {m['estimate'][1]:.6f} | "
                + ("-" if visc is None else f"{visc[0]:.6f} / {visc[1]:.6f}")
                + " |"
            )
    return lines


def whole_solves() -> list[str]:
    """T3: the whole solves, core against reference."""
    lines = [
        "### T3. Whole solves, core against reference (the step's gate)",
        "",
        "Each flow field's largest difference over the reference's largest magnitude, and its tier against its measured spread (prompt 85's P2 for that field) and the limit 1e-8 (ADR-017 H). The pressure corrections' conjugate gradient iterations summed over the solve, and the outer iterations whose count differs. The door leak's closed state is the product room's 80x30 solve again (the same configuration, solved twice), so the five rows of the first four cases are four distinct solves. The channel is outside the predictions: the one whole solve with a pressure outlet, which the stage that extrapolates the outlets' faces is read by.",
        "",
        "| case | state | stop, reference / core | outer, reference / core | "
        + " | ".join(FIELDS)
        + " | tiers | CG iterations, reference / core | counts that differ | first at |",
        "|---|---|---|---|" + "---|" * len(FIELDS) + "---|---|---|---|",
    ]
    for case in (*CASES, *SOLVE_ONLY):
        for state in load(RESULTS / "solves" / f"{case}.json")["states"]:
            fields = state["fields"]
            tiers = sorted({f["tier"] for f in fields.values()})
            stops = state["stop_reason"]
            outer = state["outer"]
            lines.append(
                f"| {CASE_NAMES[case]} | {state_name(case, state)} | {stops[0]} / {stops[1]} | {outer[0]} / {outer[1]} | "
                + " | ".join(
                    e(fields[f]["difference"]) if f in fields else "-" for f in FIELDS
                )
                + f" | {', '.join(tiers)} | {state['pressure_iterations'][0]} / {state['pressure_iterations'][1]} | {state['counts_differ']} | "
                + (
                    "-"
                    if state["first_count_differs"] is None
                    else str(state["first_count_differs"])
                )
                + " |"
            )
    return lines


def timings() -> list[str]:
    """T4: the 160x60 solve's three timing sessions."""
    lines = [
        "### T4. The 160x60 product solve from rest, by stage (for information)",
        "",
        "The reference's `solve_steady` then the core's driven solve, in one process per session; seconds of wall time, and the time in each of the reference's stages (the core's rule apart). Session 1 is scored.",
        "",
        "| session | started | outer, reference / core | reference, s | core, s | ratio | momentum | pressure | turbulence | correct | core's rule, s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for path in timing_paths():
        t = load(path)
        ref, core = t["reference"], t["core"]

        lines.append(
            f"| {t['run']} | {t['started']} | {ref['outer']} / {core['outer']} | {ref['wall_seconds']:.2f} | {core['wall_seconds']:.2f} | {t['ratio']:.2f} | "
            + " | ".join(
                f"{ref['stage_seconds'].get(n, 0.0):.2f} / "
                f"{core['stage_seconds'].get(n, 0.0):.2f}"
                for n in ("momentum", "pressure", "turbulence", "correct")
            )
            + f" | {core['core_seconds']['stopping']:.3f} |"
        )
    lines += [
        "",
        "Each stage is reference / core, s. The machine's state, Windows' busiest processes (per cent of one core), the power and the processor's performance, before and after each session:",
        "",
    ]
    for path in timing_paths():
        t = load(path)
        lines.append(
            f"- session {t['run']}: before {', '.join(t['windows_before'])}; after {', '.join(t['windows_after'])}."
        )
    return lines


def vectorisation() -> list[str]:
    """T5: the vectorisation report on the new blocks."""
    lines = [
        "### T5. The new blocks under the project's flags (GCC's report)",
        "",
        "| block | file | lines | loops vectorised |",
        "|---|---|---|---|",
    ]
    for name, v in load(RESULTS / "vecinfo.json")["blocks"].items():
        lines.append(
            f"| {name} | `{v['file']}` | {v['lines'][0]} to {v['lines'][1]} | {v['vectorised_loops']} |"
        )
    return lines


def planted() -> list[str]:
    """T6: the planted defects and whether a test fails on each."""
    lines = [
        "### T6. Planted defects",
        "",
        "Each built in a scratch tree from the branch at the commit the records name; CTest, `tests/test_core_flow.py` (to its first failure) and the turbulence module's section 0 tests are run on it. Caught where any of them fails; the first failing tests are named. A plant no test can tell from the code is an equivalent mutant, with its reason in `variants95.EQUIVALENT`.",
        "",
        "| plant | what it changes | caught | first failing tests |",
        "|---|---|---|---|",
    ]
    # A plant rerun alone (``--only``) supersedes its stream's record.
    by_name: dict[str, dict] = {}
    paths = sorted(plant_paths(), key=lambda p: "_only" in p.name)
    for path in paths:
        for row in load(path):
            by_name[row["plant"]] = row
    rows = list(by_name.values())
    sys.path.insert(0, str(HERE))
    import variants95

    order = [p[0] for p in variants95.PLANTS]
    missing = [name for name in order if name not in by_name]
    if missing:
        raise SystemExit(f"plants without a record: {', '.join(missing)}")
    rows.sort(
        key=lambda r: order.index(r["plant"]) if r["plant"] in order else len(order)
    )
    caught = equivalent = 0
    for r in rows:
        if r["plant"] in variants95.EQUIVALENT:
            verdict = "no (equivalent)"
            equivalent += 1
        else:
            verdict = (
                "no build"
                if r.get("caught") is None
                else ("yes" if r["caught"] else "no")
            )
            if r["plant"] != "control":
                caught += bool(r.get("caught"))
        failing = (
            "; ".join(f"`{f.split('::')[-1][:90]}`" for f in r.get("failing", [])[:2])
            or "-"
        )
        lines.append(f"| `{r['plant']}` | {r['description']} | {verdict} | {failing} |")
    total = len([r for r in rows if r["plant"] != "control"])
    lines += [
        "",
        f"{caught} of {total} plants caught, {equivalent} equivalent; the control is caught by nothing.",
    ]
    for name, reason in variants95.EQUIVALENT.items():
        lines.append(f"- `{name}` is equivalent: {reason}.")
    return lines


def junit(path: Path) -> dict[str, float]:
    """The counts a JUnit file holds (tests, failures, errors, skipped) and its seconds."""
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    out = {"tests": 0.0, "failures": 0.0, "errors": 0.0, "skipped": 0.0, "time": 0.0}
    for suite in suites:
        for key in out:
            out[key] += float(suite.get(key, 0))
    return out


def counts() -> list[str]:
    """T7: the tests' counts, each run's JUnit record."""
    lines = [
        "### T7. The tests",
        "",
        "| run | tests | failures | errors | skipped | seconds |",
        "|---|---|---|---|---|---|",
    ]
    for name, label in JUNIT_RECORDS:
        c = junit(junit_path(name))
        lines.append(
            f"| {label} | {c['tests']:.0f} | {c['failures']:.0f} | {c['errors']:.0f} | {c['skipped']:.0f} | {c['time']:.0f} |"
        )
    return lines


def tables() -> str:
    """The whole of tables95.md."""
    commit = records_commit()
    parts = [
        "# Prompt 95's tables",
        "",
        "Generated by `docs/reports/probe95/tables95.py` from `results/builder95/`; do not edit by hand.",
        "",
        f"Every record is of commit `{commit}`.",
        "",
    ]
    for section in (
        predictions,
        captured_blocks,
        captured_estimates,
        whole_solves,
        timings,
        vectorisation,
        planted,
        counts,
    ):
        parts += [*section(), ""]
    return "\n".join(parts)


def main() -> None:
    """Write tables95.md, or with ``--check`` compare it with the records'."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = tables()
    if args.check:
        if OUT.read_text(encoding="utf-8") != text:
            print("tables95.md differs from the records")
            raise SystemExit(1)
        print("tables95.md is the records'")
        return
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
