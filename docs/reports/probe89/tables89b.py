"""The tables of prompt 89b's round, from the records in results/builder89b/.

    python docs/reports/probe89/tables89b.py           # write tables89b.md, splice the report
    python docs/reports/probe89/tables89b.py --check   # regenerate and compare, write nothing

The records are ``probe89b.py``'s (the builds, the checks, the timings, the
planted defects, the vectorisation report) and the two whole-suite runs'
JUnit records. The report's section 9 holds the marked blocks. Round 0's
tables are ``tables89.py``'s and are not touched here; this file reads its
helpers.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import tables89  # noqa: E402 -- after sys.path

RESULTS = ROOT / "results" / "builder89b"
ROUND_0 = tables89.RESULTS
TABLES = HERE / "tables89b.md"
REPORT = tables89.REPORT
MARKERS = (
    "sessions",
    "timings",
    "ratios",
    "checks",
    "loops",
    "plants",
    "suites",
)
# The three sessions of every build and the Python reference (T13), and every session
# that timed the three builds the decisions rest on (T12, T13b).
RUNS = (1, 2, 3)
ALL_RUNS = tuple(n for n in range(1, 100) if (RESULTS / f"timing_v3_{n}.json").exists())
# The builds of a session, as probe89b.BUILDS names them, and the columns they head.
LABELS = {
    "base": "before the pass (3c8d1ec), `-O2`",
    "head_o2": "round 0's head (eacc71c), `-O2`",
    "head_v3": "round 0's head under `-march=x86-64-v3`",
    "v3": "the head now (`-march=x86-64-v3`, the flux in blocks)",
}
# The overlays Windows 11 keeps for its power mode setting (probe89b.POWER_MODES).
POWER_MODES = {
    "00000000-0000-0000-0000-000000000000": "Balanced",
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
    "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "Better performance",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
}
# The commit the checks, the planted defects and the suites were run at.
RECORDS_AT = "the tree built for round 1"
# The kernels of T12, and (for the first three) the prediction each carried in round 0.
KERNELS = tables89.KERNELS


def load(name: str, directory: Path = RESULTS) -> dict:
    """One JSON record."""
    return json.loads((directory / name).read_text(encoding="utf-8"))


def timing(build: str, run: int) -> dict:
    """One build's timing record of one session."""
    return load(f"timing_{build}_{run}.json")


def sessions_table() -> list[str]:
    """What the machine was doing in each session: power, load, calibration."""
    out = [
        "### T12. The sessions: the machine's state and the order of the builds",
        "",
        "Each session times the Python reference and each build in its own process,"
        " one after another. Before each process the probe samples Windows for two"
        " seconds (the busiest processes in percent of one logical processor, the"
        " processors' performance percentage, whether the machine is on mains) and"
        " records the Windows power scheme and the power mode. The calibration is the"
        " median time of `np.sqrt` over 4,096 doubles in that process, the same code"
        " whatever build is timed.",
        "",
        "| Run | Started | Order of the builds | On mains in every process | Power scheme and mode"
        " | Busiest process in any process of the session (percent of one logical processor)"
        " | Processor performance, percent | Calibration, us (lowest to highest) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in ALL_RUNS:
        records = [
            (b, timing(b, run))
            for b in LABELS
            if (RESULTS / f"timing_{b}_{run}.json").exists()
        ]
        if (RESULTS / f"timing_reference_{run}.json").exists():
            records.append(("reference", load(f"timing_reference_{run}.json")))
        records.sort(key=lambda item: item[1]["started"])
        order = ", ".join(f"`{b}`" for b, _ in records)
        mains = [
            line.endswith("True")
            for _, r in records
            for line in r["windows_load"]
            if line.startswith("on mains")
        ]
        schemes = sorted(
            {
                re.search(r"\(([^)]*)\)", line).group(1)
                for _, r in records
                for line in r["windows_load"]
                if line.startswith("Power Scheme GUID")
            }
        )
        modes = sorted(
            {
                POWER_MODES.get(
                    line.split("power mode overlay on mains ")[1].strip(), "unknown"
                )
                for _, r in records
                for line in r["windows_load"]
                if "power mode overlay on mains " in line
            }
        )
        busiest: dict[str, int] = {}
        for _, r in records:
            for line in r["windows_load"]:
                name, _, share = line.rpartition(" ")
                if share.isdigit() and not line.startswith(
                    ("processor performance", "power")
                ):
                    busiest[name] = max(busiest.get(name, 0), int(share))
        top = ", ".join(
            f"{name} {share}"
            for name, share in sorted(busiest.items(), key=lambda kv: -kv[1])[:4]
        )
        performance = [
            int(line.rsplit(" ", 1)[1])
            for _, r in records
            for line in r["windows_load"]
            if line.startswith("processor performance")
        ]
        calibrations = [1e6 * r["calibration"]["median_s"] for _, r in records]
        out.append(
            f"| {run} | {records[0][1]['started']} | {order} |"
            f" {'yes' if all(mains) and mains else 'NO'} ({sum(mains)} of {len(records)}) |"
            f" {', '.join(schemes)}; {', '.join(modes)} | {top} |"
            f" {min(performance)} to {max(performance)} |"
            f" {min(calibrations):.2f} to {max(calibrations):.2f} |"
        )
    return out


def kernel_rows(builds: list[str]) -> dict[str, dict[int, dict[str, float]]]:
    """Each build's kernel times in each run."""
    return {b: {run: tables89.kernels(timing(b, run)) for run in RUNS} for b in builds}


def timings_table() -> list[str]:
    """The kernels under the new flags against round 0's head, in the same sessions."""
    builds = [b for b in LABELS if (RESULTS / f"timing_{b}_1.json").exists()]
    rows = kernel_rows(builds)
    reference = {
        run: tables89.kernels(load(f"timing_reference_{run}.json")) for run in RUNS
    }
    out = [
        "### T13. Each kernel under the new flags against round 0's head, on the developed 320x120 frames",
        "",
        "Medians of repeated calls through the binding, in microseconds, as T4 gives"
        " them, in sessions 1 to 3 of T12. The inputs are prompt 88's developed"
        " frame of class 0 for the implicit kernels and prompt 87's ten developed rows"
        " for the flux, whose median over the five classes is given. A sweep is the"
        " full solve less the set-up with one residual, over the solve's sweeps; a CG"
        " iteration likewise. The two ratios are the time of round 0's head at `-O2`"
        " (the build the pass's numbers were taken on) and of the core before the pass"
        " over the time of the head now.",
        "",
        "| Kernel | Run | Reference | Before the pass | Round 0's head, `-O2` | Round 0's head,"
        " v3 | Now, v3 | `-O2` head / now | Before the pass / now |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for kernel in KERNELS:
        for run in RUNS:
            row = {b: rows[b][run][kernel] for b in builds}
            out.append(
                f"| {kernel} | {run} | {tables89.us(reference[run][kernel])} |"
                f" {tables89.us(row['base'])} | {tables89.us(row['head_o2'])} |"
                f" {tables89.us(row['head_v3'])} | {tables89.us(row['v3'])} |"
                f" {row['head_o2'] / row['v3']:.2f} | {row['base'] / row['v3']:.2f} |"
            )
    out += [
        "",
        "The calibration of each process (T12), in microseconds, in the order reference,"
        " before the pass, round 0's head at `-O2`, round 0's head at v3, now:",
        "",
    ]
    for run in RUNS:
        values = [1e6 * load(f"timing_reference_{run}.json")["calibration"]["median_s"]]
        values += [1e6 * timing(b, run)["calibration"]["median_s"] for b in builds]
        out.append(f"- run {run}: " + ", ".join(f"{v:.2f}" for v in values))
    out += [
        "",
        "Every timed Jacobi solve's outputs equal the reference's bit for bit in every"
        " build and session (the number of differing outputs over the five classes: "
        + ", ".join(
            f"{b} {sum(s['outputs_differing'] for run in RUNS for s in timing(b, run)['solves'])}"
            for b in builds
        )
        + ").",
        "",
        "The flux per class and direction, microseconds, run 1, and the range of the"
        " ratio over the ten rows in each run:",
        "",
        "| Class | Flux x, `-O2` head | now | ratio | Flux y, `-O2` head | now | ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    old, new = timing("head_o2", 1)["flux"], timing("v3", 1)["flux"]
    for k in range(5):
        x_old = next(r for r in old if r["class"] == k and r["d"] == 0)["median_s"]
        x_new = next(r for r in new if r["class"] == k and r["d"] == 0)["median_s"]
        y_old = next(r for r in old if r["class"] == k and r["d"] == 1)["median_s"]
        y_new = next(r for r in new if r["class"] == k and r["d"] == 1)["median_s"]
        out.append(
            f"| {k} | {tables89.us(1e6 * x_old)} | {tables89.us(1e6 * x_new)} | {x_old / x_new:.2f} |"
            f" {tables89.us(1e6 * y_old)} | {tables89.us(1e6 * y_new)} | {y_old / y_new:.2f} |"
        )
    spans = []
    for run in RUNS:
        ratios = [
            a["median_s"] / b["median_s"]
            for a, b in zip(
                timing("head_o2", run)["flux"], timing("v3", run)["flux"], strict=True
            )
        ]
        spans.append(f"run {run}: {min(ratios):.2f} to {max(ratios):.2f}")
    out += ["", "Over the ten rows: " + "; ".join(spans) + "."]
    # The form the flux was switched from, under the same flags.
    loop = [
        statistics.median(
            a["median_s"] / b["median_s"]
            for a, b in zip(
                timing("head_v3", run)["flux"], timing("v3", run)["flux"], strict=True
            )
        )
        for run in RUNS
    ]
    out += [
        "",
        "The flux's loop (round 0's head under v3) over its block form (now), median over"
        f" the ten rows in each run: {loop[0]:.2f}, {loop[1]:.2f}, {loop[2]:.2f}.",
    ]
    return out


def ratios_table() -> list[str]:
    """The ratios of the builds that matter, in every session that timed them."""
    pairs = [
        ("a Jacobi sweep", "sweep"),
        ("the set-up with one residual", "set-up"),
        ("a CG iteration", "CG iteration"),
        ("the flux along x", "flux x"),
        ("the flux along y", "flux y"),
    ]
    out = [
        "### T13b. The ratios in every session that timed the three builds",
        "",
        "Each cell is the time of round 0's head at `-O2` over the time of the head now"
        " (`-O2` head / now), and for the flux also the time of round 0's head under v3,"
        " the flux a loop of one face at a time, over the time of the head now (loop /"
        " block): above 1 the head now is the faster. Sessions 1 to 3 are T13's; 4 to 9"
        " were taken afterwards, with the three builds only and their order rotated,"
        " because in session 3 the loop's process ran at twice its speed in every other"
        " session (the cause was not found).",
        "",
        "| Run | " + " | ".join(label for _, label in pairs) + " | flux x, loop / block"
        " | flux y, loop / block |",
        "|---|" + "---|" * (len(pairs) + 2),
    ]
    columns: list[list[float]] = [[] for _ in range(len(pairs) + 2)]
    for run in ALL_RUNS:
        old = tables89.kernels(timing("head_o2", run))
        loop = tables89.kernels(timing("head_v3", run))
        new = tables89.kernels(timing("v3", run))
        values = [old[kernel] / new[kernel] for kernel, _ in pairs]
        values += [loop[k] / new[k] for k in ("the flux along x", "the flux along y")]
        for column, value in zip(columns, values, strict=True):
            column.append(value)
        out.append(f"| {run} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    out.append(
        "| median | "
        + " | ".join(f"{statistics.median(c):.2f}" for c in columns)
        + " |"
    )
    out.append(
        "| range | "
        + " | ".join(f"{min(c):.2f} to {max(c):.2f}" for c in columns)
        + " |"
    )
    return out


def check_summary(build: str) -> tuple[str, bool]:
    """Whether a build's CTest, parity module and replays all pass."""
    check = load(f"check_{build}.json")
    replay = check["replay"]
    clean = (
        not check["ctest_failed"]
        and "100% tests passed" in check["ctest"]
        and check["pytest_exit"] == 0
        and isinstance(replay, dict)
        and replay["flux_calls_differing"] == 0
        and replay["jacobi_calls_differing"] == 0
        and replay["cg_calls_at_the_reference_count"] == replay["cg_calls"]
        and replay["cg_calls_differing_with_the_cores_dot"] == 0
    )
    return ("all pass" if clean else "NOT ALL PASS"), clean


def checks_table() -> list[str]:
    """CTest, the parity module and the replays under the new flags."""
    ctest = tables89.junit_cases(RESULTS / "ctest_v3.xml")
    parity = tables89.junit_cases(RESULTS / "parity_v3.xml")
    before_ctest = {
        name for _, name, _ in tables89.junit_cases(ROUND_0 / "ctest_o2.xml")
    }
    before_parity = {
        name for _, name, _ in tables89.junit_cases(ROUND_0 / "parity_o2.xml")
    }
    kept_ctest = [c for c in ctest if c[1] in before_ctest]
    new_ctest = [c for c in ctest if c[1] not in before_ctest]
    kept_parity = [c for c in parity if c[1] in before_parity]
    new_parity = [c for c in parity if c[1] not in before_parity]
    replay = load("check_v3.json")["replay"]
    out = [
        "### T14. The tests under the new flags",
        "",
        "| Tests | Cases | Passed |",
        "|---|---|---|",
        f"| GoogleTest cases of round 0's head (eacc71c) | {len(before_ctest)} |"
        f" {sum(c[2] for c in kept_ctest)} of those present |",
        f"| GoogleTest cases added by this round | {len(new_ctest)} |"
        f" {sum(c[2] for c in new_ctest)} |",
        f"| Parity cases of round 0's head | {len(before_parity):,} |"
        f" {sum(c[2] for c in kept_parity):,} of those present |",
        f"| Parity cases added by this round | {len(new_parity)} |"
        f" {sum(c[2] for c in new_parity)} |",
        f"| Captured flux calls of prompt 87 replayed, bit for bit | {replay['flux_calls']} |"
        f" {replay['flux_calls'] - replay['flux_calls_differing']} |",
        "| Captured Jacobi solves of prompt 88 replayed, every output bit for bit |"
        f" {replay['jacobi_calls']} |"
        f" {replay['jacobi_calls'] - replay['jacobi_calls_differing']} |",
        "| Captured CG solves of prompt 88 at the reference's count |"
        f" {replay['cg_calls']} | {replay['cg_calls_at_the_reference_count']} |",
        "| the same against the reference with the core's dot products, bit for bit |"
        f" {replay['cg_calls']} |"
        f" {replay['cg_calls'] - replay['cg_calls_differing_with_the_cores_dot']} |",
        "",
        f"Totals: {len(ctest)} GoogleTest cases and {len(parity):,} parity cases, of which"
        f" {sum(c[2] for c in ctest)} and {sum(c[2] for c in parity):,} passed. The"
        " captured CG solves' largest difference from the reference, over its largest"
        f" magnitude: {replay['cg_largest_relative_difference']:.2e}. The build had no"
        f" warning (`-Werror`: {len(load('build_v3.json')['warnings'])} warnings). The"
        f" build's flags: `{load('build_v3.json')['flags']}`.",
        "",
        "The cases added:",
        "",
        "| Added | Cases |",
        "|---|---|",
    ]
    for _, name, passed in new_ctest:
        out.append(
            f"| GoogleTest `{name}` | {'1 (passes)' if passed else '1 (FAILS)'} |"
        )
    functions: dict[str, int] = {}
    for _, name, _ in new_parity:
        functions[name.split("[")[0]] = functions.get(name.split("[")[0], 0) + 1
    for name, count in sorted(functions.items()):
        out.append(f"| `{name}` | {count} |")
    head_v3, _ = check_summary("head_v3")
    out += [
        "",
        f"Round 0's head under v3 (the flux still a loop), the checks of this round: {head_v3}.",
    ]
    return out


def vector_names(lines: list[str], source: str) -> tuple[list[str], list[str]]:
    """The block functions whose loop GCC vectorised, and those it did not."""
    declared = [
        (number, re.search(r"(\w+)\(lanes::cells<N>", line))
        for number, line in enumerate(source.splitlines(), 1)
    ]
    blocks = [(number, match.group(1)) for number, match in declared if match]
    loops, _, _ = tables89.vectorised_places(lines)
    done = []
    for place in loops:
        parts = place.split(":")
        if not parts[0].endswith("scalar.cpp"):
            continue
        above = [name for number, name in blocks if number < int(parts[1])]
        if above:
            done.append(above[-1])
    names = sorted({name for _, name in blocks})
    return sorted(set(done)), [n for n in names if n not in done]


def loops_table() -> list[str]:
    """Which loops GCC vectorises, counted by place, before and after."""
    report = load("vecinfo.json")
    out = [
        "### T15. The loops GCC vectorises, round 1",
        "",
        f"`{report['compiler']}`, `-fopt-info-vec-optimized`, per source file: the number of"
        " loops vectorised whole and of straight-line parts, each counted once for its"
        " place in the source, however many vector widths GCC vectorised it at.",
        "",
        "| Source | Build | Loops vectorised | Vector width, bytes | Straight-line parts |",
        "|---|---|---|---|---|",
    ]
    for file in ("src/scalar.cpp", "src/reduce.cpp"):
        for build, label in LABELS.items():
            loops, parts, widths = tables89.vectorised_places(
                report["reports"][build][file]
            )
            out.append(
                f"| `{file}` | {label} | {len(loops)} |"
                f" {', '.join(map(str, widths)) or 'none'} | {len(parts)} |"
            )
    source = (RESULTS / "src_head_scalar.cpp").read_text(encoding="utf-8")
    done, missing = vector_names(report["reports"]["v3"]["src/scalar.cpp"], source)
    out += [
        "",
        "The block functions of `scalar.cpp` whose loop GCC vectorised under the new flags: "
        + ", ".join(f"`{n}`" for n in done)
        + ". "
        + (
            "Every block function of the file is among them."
            if not missing
            else "Not vectorised: " + ", ".join(f"`{n}`" for n in missing) + "."
        ),
    ]
    old_source = (ROUND_0 / "src_eacc71c_scalar.cpp").read_text(encoding="utf-8")
    old_done, old_missing = vector_names(
        report["reports"]["head_v3"]["src/scalar.cpp"], old_source
    )
    out += [
        "",
        "Round 0's head under the same flags: "
        + ", ".join(f"`{n}`" for n in old_done)
        + "; not vectorised: "
        + (", ".join(f"`{n}`" for n in old_missing) or "none")
        + ".",
    ]
    return out


def plants_table() -> list[str]:
    """The planted defects of this round and the tests that catch each."""
    records = load("plants.json")
    caught = []
    survivors = []
    for record in records:
        if record["name"] in ("control", "control_end"):
            continue
        failed = (
            "build" in record
            or bool(record["ctest_failed"])
            or record["pytest_exit"] != 0
        )
        (caught if failed else survivors).append(record["name"])
    out = [
        "### T16. The planted defects, round 1: which tests catch each",
        "",
        f"{len(caught) + len(survivors)} defects, each built in a copy of the head under the new"
        f" flags: {len(caught)} fail a GoogleTest case or a parity case or crash a run;"
        f" {len(survivors)} pass both"
        + (": " + ", ".join(f"`{s}`" for s in survivors) if survivors else "")
        + ". The controls, before and after, pass.",
        "",
        "| Defect | What it changes | GoogleTest (CTest) | Parity (pytest) |",
        "|---|---|---|---|",
    ]
    for record in records:
        if "build" in record:
            out.append(
                f"| `{record['name']}` | {record['what']} | does not build | does not build |"
            )
            continue
        ctest = record["ctest"]
        if record["ctest_failed"]:
            ctest += " [failed: " + ", ".join(record["ctest_failed"]) + "]"
        parity = record["pytest"]
        if record["pytest_exit"] not in (0, 1):
            parity = f"the run ended without a summary, exit code {record['pytest_exit']}: a crash"
        if record["parity_failed_by_test"]:
            parity += (
                " [failed: "
                + ", ".join(
                    f"{n} {c}" for n, c in record["parity_failed_by_test"].items()
                )
                + "]"
            )
        out.append(f"| `{record['name']}` | {record['what']} | {ctest} | {parity} |")
    return out


def suites_table() -> list[str]:
    """The whole suite on each platform."""
    out = [
        "### T17. The whole suite, round 1",
        "",
        "| Platform | Tests | Passed | Failed | Errors | Skipped | Wall time, s |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, file in (
        ("Windows, `.venv`, the core not built", "suite_windows.xml"),
        ("WSL2, the core built, `CFD_CORE_REQUIRED=1`", "suite_linux.xml"),
    ):
        counts = tables89.suite_counts(RESULTS / file)
        tests, failed = int(counts["tests"]), int(counts["failures"])
        errors, skipped = int(counts["errors"]), int(counts["skipped"])
        out.append(
            f"| {label} | {tests:,} | {tests - failed - errors - skipped:,} | {failed} |"
            f" {errors} | {skipped} | {counts['time']:,.0f} |"
        )
    return out


def build() -> tuple[str, dict[str, str]]:
    """Every table: the whole text, and each marked block."""
    blocks = {
        "sessions": sessions_table(),
        "timings": timings_table(),
        "ratios": ratios_table(),
        "checks": checks_table(),
        "loops": loops_table(),
        "plants": plants_table(),
        "suites": suites_table(),
    }
    assert tuple(blocks) == MARKERS
    text = "# Prompt 89b: the tables\n\n" + "\n\n".join(
        "\n".join(lines) for lines in blocks.values()
    )
    return text + "\n", {name: "\n".join(lines) for name, lines in blocks.items()}


def splice(report: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """REPORT with each marked block replaced, and the names whose markers are missing."""
    missing = []
    for name, body in blocks.items():
        begin, end = (
            f"<!-- tables89b {name} begin -->",
            f"<!-- tables89b {name} end -->",
        )
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
        print("tables89b --check:", "ok" if ok else "FAILED")
        return 0 if ok else 1
    TABLES.write_text(text, encoding="utf-8", newline="\n")
    REPORT.write_text(spliced, encoding="utf-8", newline="\n")
    if missing:
        print(f"missing markers (not spliced): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
