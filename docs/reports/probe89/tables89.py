"""The tables of prompt 89's report, from the records in results/builder89/.

    python docs/reports/probe89/tables89.py           # write tables89.md, splice the report
    python docs/reports/probe89/tables89.py --check   # regenerate and compare, write nothing

Run with Windows' ``.venv`` or the Linux environment: every record it reads is
a text file in the checkout, written under WSL2 by ``probe89.py`` (the builds,
the checks with their JUnit records, the timings, the planted defects, the
vectorisation report, the floor's rates and the loop styles) and by the two
whole-suite runs (pytest's JUnit records). ``--check`` also fails when one of
the report's marker pairs is missing.
"""

import argparse
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import variants89  # noqa: E402 -- after sys.path

RESULTS = ROOT / "results" / "builder89"
TABLES = HERE / "tables89.md"
REPORT = ROOT / "docs" / "reports" / "phase6_scalar_perf.md"
MARKERS = (
    "predictions",
    "machine",
    "floor",
    "styles",
    "kernels",
    "changes",
    "candidates",
    "flags",
    "vectorised",
    "tests",
    "plants",
    "suites",
)
RUNS = (1, 2, 3)
# The builds a session times, as probe89.BUILDS names them.
COMMIT_BUILDS = ["base", "jacobi", "setup", "cg", "o2"]
COMMIT_LABELS = {
    "base": "before this pass (3c8d1ec)",
    "jacobi": "after the Jacobi sweep (101b1ad)",
    "setup": "after the set-up (34e11fe)",
    "cg": "after the conjugate gradient (104a02d)",
    "o2": "after the flux (1b98cf7) and the end cells' pointers: the head (6a13cfc)",
}
FLAG_BUILDS = ["o2", "o3", "native", "o3native"]
# The prediction's sets, and one more measured beside them.
ALL_FLAG_BUILDS = [*FLAG_BUILDS, "v3"]
FLAG_LABELS = {
    "o2": "`-O2` (the project's)",
    "o3": "`-O3`",
    "native": "`-O2 -march=native`",
    "o3native": "`-O3 -march=native`",
    "v3": "`-O2 -march=x86-64-v3`",
}
# The kernels of the predictions, in the order the tables list them.
KERNELS = [
    "a Jacobi sweep",
    "the set-up with one residual",
    "the Jacobi solve",
    "nothing to solve",
    "a CG iteration",
    "the CG solve",
    "the flux along x",
    "the flux along y",
]
# Prediction (e)'s kernels: (b)'s, (c)'s and (d)'s, and the CG solve per call.
FLAG_KERNELS = [
    "a Jacobi sweep",
    "the set-up with one residual",
    "the CG solve",
    "the flux along x",
    "the flux along y",
]
# Section 1's counts from the code: the bytes a kernel moves per cell or face,
# a written array's line read before it is written, and the faces of one flux
# call on 320x120 (the x direction's interior and boundary faces).
SWEEP_BYTES = 73
SETUP_BYTES = 72
FLUX_BYTES = 33
FLUX_FACES = 38_520
SWEEP_LIMIT_US = 70.0
SETUP_LIMIT_US = 150.0
FLUX_RATIO = 1.5
FLAG_CHANGE = 0.25
# The parity functions this pass added (tests/test_core_parity.py's last section).
NEW_PARITY = (
    "test_the_row_widths_leave_every_number_of_cells_after_the_blocks",
    "test_implicit_jacobi_is_bit_for_bit_at_every_row_length",
    "test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length",
    "test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length",
    "test_implicit_system_is_the_references_set_up_at_every_row_length",
    "test_advective_flux_is_bit_for_bit_at_every_row_length",
)
# The GoogleTest suites this pass added, and the suite an old one gained cases in.
NEW_SUITES = ("Lanes", "BlockDot")
# The commit the checks, the planted defects and the suites were run at.
RECORDS_AT = "6a13cfc"


def load(name: str) -> dict:
    """One JSON record of results/builder89/."""
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def timing(build: str, run: int) -> dict:
    """One build's timing record of one session."""
    return load(f"timing_{build}_{run}.json")


def kernels(record: dict) -> dict[str, float]:
    """A timing record's kernel times, in microseconds."""
    parts = record["parts"]
    full, none = parts["the full solve"], parts["no sweep"]
    cg, start = parts["cg"], parts["cg at the start"]
    flux = record["flux"]
    return {
        "a Jacobi sweep": 1e6
        * (full["median_s"] - none["median_s"])
        / max(full["iterations"], 1),
        "the set-up with one residual": 1e6 * none["median_s"],
        "the Jacobi solve": 1e6 * full["median_s"],
        "nothing to solve": 1e6 * parts["nothing to solve"]["median_s"],
        "a CG iteration": 1e6
        * (cg["median_s"] - start["median_s"])
        / max(cg["iterations"], 1),
        "the CG solve": 1e6 * cg["median_s"],
        "the flux along x": 1e6
        * statistics.median(r["median_s"] for r in flux if r["d"] == 0),
        "the flux along y": 1e6
        * statistics.median(r["median_s"] for r in flux if r["d"] == 1),
    }


def calibration(build: str, run: int) -> str:
    """A timed process's calibration: its median np.sqrt time, in microseconds."""
    return f"{1e6 * timing(build, run)['calibration']['median_s']:.2f}"


def us(value: float) -> str:
    """Microseconds to three significant figures."""
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def junit_cases(path: Path) -> list[tuple[str, str, bool]]:
    """A JUnit record's cases: (class or suite, name, passed)."""
    cases = []
    for case in ET.parse(path).getroot().iter("testcase"):
        bad = any(child.tag in ("failure", "error", "skipped") for child in case)
        status = case.get("status", "run")
        cases.append(
            (
                case.get("classname", ""),
                case.get("name", ""),
                not bad and status in ("run", "passed"),
            )
        )
    return cases


def machine_table() -> list[str]:
    """The machine as WSL2 sees it, and what else ran in each session."""
    lines = (RESULTS / "machine.txt").read_text(encoding="utf-8").splitlines()

    def field(key: str) -> str:
        return next(
            line.split(":", 1)[1].strip() for line in lines if line.startswith(key)
        )

    caches = [line for line in lines if line[:2] in ("1 ", "2 ", "3 ")]
    native = next(
        line.split()[-1] for line in lines if line.strip().startswith("-march=")
    )
    compiler = next(line for line in lines if line.startswith("g++"))
    out = [
        "### T1. The machine, and what else ran",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| CPU | {field('Model name')}, {field('CPU(s)')} logical processors |",
        *(
            f"| Cache level {c.split()[0]} ({c.split()[1].lower()}) | {c.split()[2]},"
            f" shared by logical processors {c.split()[-1]} |"
            for c in caches
        ),
        f"| Compiler | {compiler} |",
        f"| `-march=native` resolves to | `-march={native}` |",
        f"| Kernel | {lines[-1].strip()} |",
        "",
        "Each session of `probe89.py session` times the reference and every build, one"
        " process after another. Before each process the probe samples Windows for two"
        " seconds: the busiest processes, in percent of one logical processor, and the"
        " processors' performance percentage. The calibration is the median time of"
        " `np.sqrt` over 4,096 doubles in that process, the same compiled code whatever"
        " build is timed.",
        "",
        "| Run | Started | Calibration, us (lowest to highest over the processes) |"
        " Busiest processes at the first process | Processor performance |",
        "|---|---|---|---|---|",
    ]
    for run in RUNS:
        records = [
            load(path.name) for path in sorted(RESULTS.glob(f"timing_*_{run}.json"))
        ]
        first = min(records, key=lambda r: r["started"])
        calibrations = [1e6 * r["calibration"]["median_s"] for r in records]
        load_lines = first["windows_load"]
        busiest = ", ".join(
            line for line in load_lines[:4] if not line.startswith(("on ", "proc"))
        )
        performance = next(
            (
                line.split()[-1]
                for line in load_lines
                if line.startswith("processor performance")
            ),
            "not read",
        )
        out.append(
            f"| {run} | {first['started']} | {min(calibrations):.2f} to"
            f" {max(calibrations):.2f} | {busiest} | {performance} |"
        )
    return out


def floor_table() -> list[str]:
    """The machine's rates and the floors they give."""
    floor = load("floor.json")
    cells = floor["grid_cells"]
    cycle = floor["chain_step"]["best_ns"] / 4.0

    def best(name: str) -> float:
        return floor[name]["best_ns"]

    rate = 1.0 / best("stream_byte")
    out = [
        "### T2. The floor: this machine's rates, measured by `floor89.cpp`",
        "",
        "Each rate is the best of repeated timings of a kernel written with intrinsics"
        " at a stated width: 2 doubles an instruction (SSE2, what the project's flags"
        " allow), 4 (AVX2) and 8 (AVX-512). A cycle is a quarter of a step of a"
        " dependent 64-bit multiply-add chain.",
        "",
        "| Rate | 2 doubles | 4 doubles | 8 doubles |",
        "|---|---|---|---|",
        f"| a cycle, ns ({1.0 / cycle:.2f} GHz) | {cycle:.4f} | | |",
        "| a division, ns per element, arrays in the first-level cache |"
        f" {best('divide_sse2'):.3f} | {best('divide_avx2'):.3f} |"
        f" {best('divide_avx512'):.3f} |",
        "| the residual's nine operations, ns per cell, first-level cache |"
        f" {best('arithmetic_sse2'):.3f} | {best('arithmetic_avx2'):.3f} |"
        f" {best('arithmetic_avx512'):.3f} |",
        "| the whole sweep, ns per cell, first-level cache |"
        f" {best('sweep_small_sse2'):.3f} | {best('sweep_small_avx2'):.3f} |"
        f" {best('sweep_small_avx512'):.3f} |",
        "| the whole sweep, ns per cell, 320x120 cells |"
        f" {best('sweep_grid_sse2'):.3f} | {best('sweep_grid_avx2'):.3f} |"
        f" {best('sweep_grid_avx512'):.3f} |",
        "| **the sweep's floor, us per sweep of 38,400 cells** |"
        f" **{best('sweep_grid_sse2') * cells / 1e3:.1f}** |"
        f" **{best('sweep_grid_avx2') * cells / 1e3:.1f}** |"
        f" **{best('sweep_grid_avx512') * cells / 1e3:.1f}** |",
        "| the sweep with memory free (first-level cache rate), us per sweep |"
        f" {best('sweep_small_sse2') * cells / 1e3:.1f} |"
        f" {best('sweep_small_avx2') * cells / 1e3:.1f} |"
        f" {best('sweep_small_avx512') * cells / 1e3:.1f} |",
        "| **the flux's floor from its two divisions a face, us per call of"
        f" {FLUX_FACES:,} faces** |"
        f" **{2 * best('divide_sse2') * FLUX_FACES / 1e3:.1f}** |"
        f" **{2 * best('divide_avx2') * FLUX_FACES / 1e3:.1f}** |"
        f" **{2 * best('divide_avx512') * FLUX_FACES / 1e3:.1f}** |",
        "| the set-up pass from its one division a cell, us per pass |"
        f" {best('divide_sse2') * cells / 1e3:.1f} |"
        f" {best('divide_avx2') * cells / 1e3:.1f} |"
        f" {best('divide_avx512') * cells / 1e3:.1f} |",
        "",
        f"A scalar division takes {best('divide_scalar'):.3f} ns"
        f" ({best('divide_scalar') / cycle:.1f} cycles), so a flux loop that divides one"
        f" face at a time cannot take less than"
        f" {2 * best('divide_scalar') * FLUX_FACES / 1e3:.1f} us a call. Nine arrays of"
        " 38,400 doubles, read and written as the sweep's are, move"
        f" {rate:.0f} GB/s: at that rate a sweep's {SWEEP_BYTES} bytes a cell take"
        f" {SWEEP_BYTES * cells / rate / 1e3:.1f} us, a flux call's {FLUX_BYTES} bytes a"
        f" face {FLUX_BYTES * FLUX_FACES / rate / 1e3:.1f} us and the set-up pass's"
        f" {SETUP_BYTES} bytes a cell {SETUP_BYTES * cells / rate / 1e3:.1f} us.",
    ]
    return out


def styles_table() -> list[str]:
    """One sweep written six ways, under each flag set."""
    styles = load("styles.json")
    names = list(styles["o2"]["styles"])
    out = [
        "### T3. One sweep written six ways (`styles89.cpp`), us per sweep of 320x120 cells",
        "",
        "The best of 2,000 sweeps of a synthetic field, the cells 1 to 318 of each row."
        " `blocks` is the style kept. The checksums of the six agree under every flag"
        " set.",
        "",
        "| Style | " + " | ".join(FLAG_LABELS[f] for f in ALL_FLAG_BUILDS) + " |",
        "|---|" + "---|" * len(ALL_FLAG_BUILDS),
    ]
    for name in names:
        out.append(
            f"| `{name}` | "
            + " | ".join(
                f"{styles[f]['styles'][name]['best_us']:.1f}" for f in ALL_FLAG_BUILDS
            )
            + " |"
        )
    checksums = {
        style["checksum"]
        for f in ALL_FLAG_BUILDS
        for style in styles[f]["styles"].values()
    }
    out += [
        "",
        f"Distinct checksums over the {len(ALL_FLAG_BUILDS) * len(names)} runs: {len(checksums)}.",
    ]
    return out


def kernels_table() -> list[str]:
    """Each kernel before and after, with the reference, three sessions."""
    out = [
        "### T4. Each kernel before and after, on the developed 320x120 frames",
        "",
        "Medians of repeated calls through the binding under the project's flags, in"
        " microseconds: `before` is the core at 3c8d1ec, `after` the head, the"
        " reference `src/scalar_scheme.py`. The inputs are prompt 88's developed frame"
        " of class 0 for the implicit kernels and prompt 87's ten developed rows for the"
        " flux, whose median over the five classes is given. A sweep is the full solve"
        " less the set-up with one residual, over the solve's sweeps; a CG iteration"
        " likewise from the CG solve and the same solve with a stop its first residual"
        " meets. Run 1 is the one scored.",
        "",
        "| Kernel | Run | Reference | Before | After | Before / after | Reference / after |",
        "|---|---|---|---|---|---|---|",
    ]
    for kernel in KERNELS:
        for run in RUNS:
            reference = kernels(timing("reference", run))[kernel]
            before = kernels(timing("base", run))[kernel]
            after = kernels(timing("o2", run))[kernel]
            out.append(
                f"| {kernel} | {run} | {us(reference)} | {us(before)} | {us(after)} |"
                f" {before / after:.2f} | {reference / after:.2f} |"
            )
    out += [
        "",
        "The calibration of each process (T1), in microseconds, reference, before and"
        " after: "
        + "; ".join(
            f"run {run}: {calibration('reference', run)}, {calibration('base', run)},"
            f" {calibration('o2', run)}"
            for run in RUNS
        )
        + ".",
    ]
    record = timing("o2", 1)
    parts = record["parts"]
    out += [
        "",
        f"The Jacobi solve takes {parts['the full solve']['iterations']} sweeps and the"
        f" CG solve {parts['cg']['iterations']} iterations with"
        f" {parts['cg']['restarts']} restarts and {parts['cg']['fallbacks']} fallbacks."
        " Every timed Jacobi solve's outputs equal the reference's bit for bit"
        f" ({sum(s['outputs_differing'] for s in record['solves'])} differing over the"
        " five classes). On a 2x2 grid, where the arithmetic is nothing, a call of"
        f" `implicit_step` takes {us(1e6 * record['small_grid']['implicit_step']['median_s'])}"
        " us and one of `advective_flux`"
        f" {us(1e6 * record['small_grid']['advective_flux']['median_s'])} us: the"
        " binding's cost per call.",
        "",
        "The Jacobi solve per class and the flux per class and direction, run 1:",
        "",
        "| Class | Jacobi solve, before | after | ratio | Flux x, before | after | ratio |"
        " Flux y, before | after | ratio |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    before, after = timing("base", 1), timing("o2", 1)
    for k in range(len(after["solves"])):
        cells = [str(k)]
        b, a = before["solves"][k]["median_s"], after["solves"][k]["median_s"]
        cells += [us(1e6 * b), us(1e6 * a), f"{b / a:.2f}"]
        for d in (0, 1):
            b = next(
                r["median_s"] for r in before["flux"] if r["class"] == k and r["d"] == d
            )
            a = next(
                r["median_s"] for r in after["flux"] if r["class"] == k and r["d"] == d
            )
            cells += [us(1e6 * b), us(1e6 * a), f"{b / a:.2f}"]
        out.append("| " + " | ".join(cells) + " |")
    return out


def changes_table() -> list[str]:
    """Each kernel's time after each commit of the pass."""
    out = [
        "### T5. Each change with its effect: the kernels after each commit, run 1",
        "",
        "One build per commit, all under the project's flags and timed in one session,"
        " in microseconds. A kernel moves at its own commit, and at another's only by"
        " what the two share (the block walk, the masks) or by the session's noise:"
        " runs 2 and 3 are in the records.",
        "",
        "| Kernel | " + " | ".join(COMMIT_LABELS[b] for b in COMMIT_BUILDS) + " |",
        "|---|" + "---|" * len(COMMIT_BUILDS),
    ]
    for kernel in KERNELS:
        out.append(
            f"| {kernel} | "
            + " | ".join(us(kernels(timing(b, 1))[kernel]) for b in COMMIT_BUILDS)
            + " |"
        )
    out.append(
        "| *the process's calibration* | "
        + " | ".join(calibration(b, 1) for b in COMMIT_BUILDS)
        + " |"
    )
    for run in RUNS[1:]:
        out += [
            "",
            f"Run {run}:",
            "",
            "| Kernel | " + " | ".join(COMMIT_LABELS[b] for b in COMMIT_BUILDS) + " |",
            "|---|" + "---|" * len(COMMIT_BUILDS),
        ]
        for kernel in KERNELS:
            out.append(
                f"| {kernel} | "
                + " | ".join(us(kernels(timing(b, run))[kernel]) for b in COMMIT_BUILDS)
                + " |"
            )
        out.append(
            "| *the process's calibration* | "
            + " | ".join(calibration(b, run) for b in COMMIT_BUILDS)
            + " |"
        )
    return out


def check_summary(build: str) -> tuple[str, bool]:
    """Whether a build's CTest, parity module and replays all pass, in words."""
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
    if clean:
        return "all pass", True
    failed = sum(check["parity_failed_by_test"].values())
    return (
        f"FAILS: {len(check['ctest_failed'])} GoogleTest and {failed} parity cases",
        False,
    )


def candidates_table() -> list[str]:
    """The forms measured against the kept ones."""
    out = [
        "### T6. The forms tried against the kept ones, run 1 (runs 2 and 3 in brackets)",
        "",
        "Each candidate is the head with one form changed (`variants89.py`), built"
        " under the project's flags and timed in the same session as the head. The"
        " ratio is the candidate's time over the head's for the kernels the change"
        " reaches: above 1 the kept form is the faster. Every candidate computes the"
        " same values; the check column is CTest, the parity module and the replays on"
        " the candidate.",
        "",
        "| Candidate | What it is | Kernel | Candidate, us | Head, us | Candidate / head |"
        " Calibration, candidate and head | Checks |",
        "|---|---|---|---|---|---|---|---|",
    ]
    reach = {
        "flux_blocks": ["the flux along x", "the flux along y"],
        "flux_blocks_by_sign": ["the flux along x", "the flux along y"],
        "flux_paired_division": ["the flux along x", "the flux along y"],
        "jacobi_copies": ["the Jacobi solve"],
        "jacobi_two_masks": ["a Jacobi sweep", "the Jacobi solve"],
        "neighbours_by_pointer": ["a Jacobi sweep", "a CG iteration"],
        "helpers_not_forced_inline": ["a Jacobi sweep", "a CG iteration"],
        "row_loops_not_blocks": [
            "a Jacobi sweep",
            "the set-up with one residual",
            "a CG iteration",
        ],
    }
    for name, candidate in variants89.CANDIDATES.items():
        verdict, _ = check_summary(name)
        for kernel in reach[name]:
            ratios = []
            for run in RUNS:
                tried = kernels(timing(name, run))[kernel]
                head = kernels(timing("o2", run))[kernel]
                ratios.append(tried / head)
            tried = kernels(timing(name, 1))[kernel]
            head = kernels(timing("o2", 1))[kernel]
            out.append(
                f"| `{name}` | {candidate['what']} | {kernel} | {us(tried)} | {us(head)} |"
                f" {ratios[0]:.2f} ({ratios[1]:.2f}, {ratios[2]:.2f}) |"
                f" {calibration(name, 1)}, {calibration('o2', 1)} | {verdict} |"
            )
    return out


def flag_changes(run: int) -> dict[str, dict[str, float]]:
    """Each flag set's change of each of (e)'s kernels against the project's flags."""
    project = kernels(timing("o2", run))
    return {
        build: {
            kernel: kernels(timing(build, run))[kernel] / project[kernel] - 1.0
            for kernel in FLAG_KERNELS
        }
        for build in ALL_FLAG_BUILDS[1:]
    }


def flags_table() -> list[str]:
    """The restructured kernels under each flag set, and the flux's block form."""
    out = [
        "### T7. The flags measured: the head's kernels under each set, run 1",
        "",
        "Microseconds, and in brackets the change against the project's flags in runs"
        " 1, 2 and 3. The checks are CTest, the parity module and the captured replays"
        " on that build. The last column is not one of prediction (e)'s sets:"
        " `x86-64-v3` is AVX2 by a name that means the same on every machine, where"
        " `native` means this processor's AVX-512.",
        "",
        "| Kernel | " + " | ".join(FLAG_LABELS[f] for f in ALL_FLAG_BUILDS) + " |",
        "|---|" + "---|" * len(ALL_FLAG_BUILDS),
    ]
    changes = {run: flag_changes(run) for run in RUNS}
    for kernel in KERNELS:
        cells = [us(kernels(timing("o2", 1))[kernel])]
        for build in ALL_FLAG_BUILDS[1:]:
            value = kernels(timing(build, 1))[kernel]
            if kernel in FLAG_KERNELS:
                moves = ", ".join(f"{changes[run][build][kernel]:+.0%}" for run in RUNS)
                cells.append(f"{us(value)} ({moves})")
            else:
                cells.append(us(value))
        out.append(f"| {kernel} | " + " | ".join(cells) + " |")
    out.append(
        "| *the process's calibration, runs 1, 2 and 3* | "
        + " | ".join(
            ", ".join(calibration(f, run) for run in RUNS) for f in ALL_FLAG_BUILDS
        )
        + " |"
    )
    out.append(
        "| **checks** | "
        + " | ".join(check_summary(f)[0] for f in ALL_FLAG_BUILDS)
        + " |"
    )
    out += [
        "",
        "The limited QUICK flux in its block form (`flux_blocks` of T6) under each set,"
        " against the kept loop under the same set, run 1 (runs 2 and 3 in brackets):",
        "",
        "| Flags | Flux x: kept, us | block form, us | block / kept | Flux y: kept, us |"
        " block form, us | block / kept | Checks on the block form |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for build in ALL_FLAG_BUILDS:
        block = "flux_blocks" if build == "o2" else f"flux_blocks_{build}"
        cells = [FLAG_LABELS[build]]
        for kernel in ("the flux along x", "the flux along y"):
            kept = kernels(timing(build, 1))[kernel]
            tried = kernels(timing(block, 1))[kernel]
            others = ", ".join(
                f"{kernels(timing(block, run))[kernel] / kernels(timing(build, run))[kernel]:.2f}"
                for run in RUNS[1:]
            )
            cells += [us(kept), us(tried), f"{tried / kept:.2f} ({others})"]
        cells.append(check_summary(block)[0])
        out.append("| " + " | ".join(cells) + " |")
    return out


def vectorised_table() -> list[str]:
    """Which loops GCC vectorises, before and after."""
    report = load("vecinfo.json")
    out = [
        "### T8. The loops GCC vectorises: its report before and after",
        "",
        f"`{report['compiler']}`, `-fopt-info-vec-optimized`, per source file: the"
        " number of loops vectorised whole, and of straight-line parts (which are not"
        " loops).",
        "",
        "| Source | Build | Loops vectorised | Vector width, bytes | Straight-line parts |",
        "|---|---|---|---|---|",
    ]
    for file in ("src/scalar.cpp", "src/reduce.cpp"):
        for build in ("base", *ALL_FLAG_BUILDS):
            lines = report["reports"][build][file]
            loops = [line for line in lines if "loop vectorized" in line]
            widths = sorted({int(line.split("using ")[1].split()[0]) for line in loops})
            label = "before (3c8d1ec), `-O2`" if build == "base" else FLAG_LABELS[build]
            out.append(
                f"| `{file}` | {label} | {len(loops)} |"
                f" {', '.join(map(str, widths)) or 'none'} |"
                f" {len(lines) - len(loops)} |"
            )
    head = [
        line
        for line in report["reports"]["o2"]["src/scalar.cpp"]
        if "loop vectorized" in line
    ]
    # A block function is named by the declaration nearest above its loop.
    source = (ROOT / "core" / "src" / "scalar.cpp").read_text(encoding="utf-8")
    declared = [
        (number, re.search(r"(\w+)\(lanes::cells<N>", line))
        for number, line in enumerate(source.splitlines(), 1)
    ]
    blocks = [(number, match.group(1)) for number, match in declared if match]
    names = []
    for line in head:
        at = int(line.split(":")[1])
        above = [name for number, name in blocks if number < at]
        names.append(f"`{above[-1]}`" if above else f"line {at}")
    out += [
        "",
        "The head's loops vectorised under the project's flags, by the block function"
        " each is in: " + ", ".join(names) + ". Every block function of the file is"
        " among them; the limited QUICK flux is not a block (section 3.4).",
    ]
    return out


def tests_table() -> list[str]:
    """CTest and the parity module at the head, the old cases apart from the new."""
    ctest = junit_cases(RESULTS / "ctest_o2.xml")
    parity = junit_cases(RESULTS / "parity_o2.xml")
    base = {name for _, name, _ in junit_cases(RESULTS / "ctest_base.xml")}
    old_ctest = [c for c in ctest if c[1] in base]
    new_ctest = [c for c in ctest if c[1] not in base]

    def function(name: str) -> str:
        return name.split("[")[0]

    new_parity = [c for c in parity if function(c[1]) in NEW_PARITY]
    old_parity = [c for c in parity if function(c[1]) not in NEW_PARITY]
    replay = load("check_o2.json")["replay"]
    out = [
        f"### T9. The tests at {RECORDS_AT}, under the project's flags",
        "",
        "| Tests | Cases | Passed |",
        "|---|---|---|",
        f"| GoogleTest cases that 3c8d1ec has | {len(old_ctest)} |"
        f" {sum(c[2] for c in old_ctest)} |",
        f"| GoogleTest cases added by this pass | {len(new_ctest)} |"
        f" {sum(c[2] for c in new_ctest)} |",
        f"| Parity cases that 3c8d1ec has | {len(old_parity)} |"
        f" {sum(c[2] for c in old_parity)} |",
        f"| Parity cases added by this pass | {len(new_parity)} |"
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
        "The captured CG solves' largest difference from the reference, over its"
        f" largest magnitude: {replay['cg_largest_relative_difference']:.2e}.",
        "",
        "The cases added:",
        "",
        "| Added | Cases |",
        "|---|---|",
    ]
    suites = Counter(name.split(".")[0] for _, name, _ in new_ctest)
    for suite, count in sorted(suites.items()):
        out.append(f"| GoogleTest `{suite}` | {count} |")
    functions = Counter(function(name) for _, name, _ in new_parity)
    for name in NEW_PARITY:
        out.append(f"| `{name}` | {functions[name]} |")
    return out


def plants_table() -> list[str]:
    """The planted defects and the tests that fail on each."""
    out = [
        "### T10. The planted defects: which tests catch each",
        "",
        "| Defect | What it changes | GoogleTest (CTest) | Parity (pytest) |",
        "|---|---|---|---|",
    ]
    for record in load("plants.json"):
        if "build" in record:
            out.append(
                f"| `{record['name']}` | {record['what']} | does not build |"
                " does not build |"
            )
            continue
        ctest = record["ctest"]
        if record["ctest_failed"]:
            ctest += " [failed: " + ", ".join(record["ctest_failed"]) + "]"
        parity = record["pytest"]
        if record["pytest_exit"] not in (0, 1):
            parity = (
                f"the run ended without a summary, exit code {record['pytest_exit']}:"
                " a crash"
            )
        if record["parity_failed_by_test"]:
            parity += (
                " [failed: "
                + ", ".join(
                    f"{name} {count}"
                    for name, count in record["parity_failed_by_test"].items()
                )
                + "]"
            )
        out.append(f"| `{record['name']}` | {record['what']} | {ctest} | {parity} |")
    return out


def suite_counts(path: Path) -> dict[str, float]:
    """A pytest JUnit record's totals."""
    suite = ET.parse(path).getroot()
    if suite.tag == "testsuites":
        suite = suite[0]
    return {
        key: float(suite.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped", "time")
    }


def suites_table() -> list[str]:
    """The whole suite on each platform."""
    out = [
        f"### T11. The whole suite at {RECORDS_AT}",
        "",
        "The two suites ran at the same time, one BLAS thread each, beside the planted"
        " defects, so the wall times are not those of a suite alone.",
        "",
        "| Platform | Tests | Passed | Failed | Errors | Skipped | Wall time, s |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, file in (
        ("Windows, `.venv`, the core not built", "suite_windows.xml"),
        ("WSL2, the core built, `CFD_CORE_REQUIRED=1`", "suite_linux.xml"),
    ):
        counts = suite_counts(RESULTS / file)
        tests, failed = int(counts["tests"]), int(counts["failures"])
        errors, skipped = int(counts["errors"]), int(counts["skipped"])
        out.append(
            f"| {label} | {tests:,} | {tests - failed - errors - skipped:,} | {failed} |"
            f" {errors} | {skipped} | {counts['time']:,.0f} |"
        )
    return out


def predictions_table() -> list[str]:
    """The five predictions scored by the rule committed with them."""
    _, clean = check_summary("o2")
    ctest = junit_cases(RESULTS / "ctest_o2.xml")
    base = {name for _, name, _ in junit_cases(RESULTS / "ctest_base.xml")}
    old_ctest = [c for c in ctest if c[1] in base]
    parity = junit_cases(RESULTS / "parity_o2.xml")
    old_parity = [c for c in parity if c[1].split("[")[0] not in NEW_PARITY]
    replay = load("check_o2.json")["replay"]
    holds_a = (
        clean and all(c[2] for c in old_ctest) and all(c[2] for c in old_parity)
    ) and len(old_ctest) == len(base)
    after = {run: kernels(timing("o2", run)) for run in RUNS}
    before = {run: kernels(timing("base", run)) for run in RUNS}
    sweep = [after[run]["a Jacobi sweep"] for run in RUNS]
    sweep_ratio = [
        before[run]["a Jacobi sweep"] / after[run]["a Jacobi sweep"] for run in RUNS
    ]
    setup = [after[run]["the set-up with one residual"] for run in RUNS]

    def flux_ratios(run: int) -> list[float]:
        old, new = timing("base", run)["flux"], timing("o2", run)["flux"]
        return [b["median_s"] / a["median_s"] for b, a in zip(old, new, strict=True)]

    flux = {run: flux_ratios(run) for run in RUNS}
    changes = {run: flag_changes(run) for run in RUNS}
    worst = {
        run: max(
            (
                (abs(change), build, kernel)
                for build, by_kernel in changes[run].items()
                for kernel, change in by_kernel.items()
                if build in FLAG_BUILDS
            )
        )
        for run in RUNS
    }
    flags_pass = all(check_summary(f)[1] for f in FLAG_BUILDS[1:])
    beyond = sorted(
        f"{kernel} under {FLAG_LABELS[build]} ({change:+.0%})"
        for build, by_kernel in changes[1].items()
        for kernel, change in by_kernel.items()
        if abs(change) >= FLAG_CHANGE and build in FLAG_BUILDS
    )
    holds_e = worst[1][0] < FLAG_CHANGE and flags_pass
    return [
        "### T0. The predictions, scored by the rule committed with them",
        "",
        "| Prediction | Measured | Verdict |",
        "|---|---|---|",
        "| (a) every parity test, GoogleTest case and captured replay of prompts 87 and"
        f" 88 passes unchanged | {sum(c[2] for c in old_ctest)} of {len(base)}"
        f" GoogleTest cases and {sum(c[2] for c in old_parity):,} of"
        f" {len(old_parity):,} parity cases that 3c8d1ec has;"
        f" {replay['flux_calls'] - replay['flux_calls_differing']} of"
        f" {replay['flux_calls']} captured flux calls and"
        f" {replay['jacobi_calls'] - replay['jacobi_calls_differing']} of"
        f" {replay['jacobi_calls']} captured Jacobi solves bit for bit;"
        f" {replay['cg_calls_at_the_reference_count']} of {replay['cg_calls']} captured"
        " CG solves at the reference's count, largest difference"
        f" {replay['cg_largest_relative_difference']:.2e}, and"
        f" {replay['cg_calls'] - replay['cg_calls_differing_with_the_cores_dot']} of"
        f" {replay['cg_calls']} bit for bit with the core's dot products (T9); no line"
        " of an existing case edited or removed (section 6) |"
        f" {'holds' if holds_a else 'FAILS'} |",
        "| (b) a Jacobi sweep at most 0.07 ms under the current flags |"
        f" {sweep[0] / 1e3:.4f} ms in run 1 ({sweep[1] / 1e3:.4f} and"
        f" {sweep[2] / 1e3:.4f} in runs 2 and 3); {sweep_ratio[0]:.2f} times faster"
        f" than the core before the pass in the same session ({sweep_ratio[1]:.2f},"
        f" {sweep_ratio[2]:.2f}), not the four times of the prediction's parenthesis"
        f" (T4) | {'holds' if sweep[0] <= SWEEP_LIMIT_US else 'FAILS'} |",
        "| (c) the set-up with one residual at most 0.15 ms |"
        f" {setup[0] / 1e3:.4f} ms in run 1 ({setup[1] / 1e3:.4f},"
        f" {setup[2] / 1e3:.4f}) (T4) |"
        f" {'holds' if setup[0] <= SETUP_LIMIT_US else 'FAILS'} |",
        "| (d) the flux at least 1.5 times faster per call than the core's before |"
        f" {min(flux[1]):.2f} to {max(flux[1]):.2f} times over the ten rows in run 1"
        f" ({min(flux[2]):.2f} to {max(flux[2]):.2f}, {min(flux[3]):.2f} to"
        f" {max(flux[3]):.2f}) (T4) |"
        f" {'holds' if min(flux[1]) >= FLUX_RATIO else 'FAILS'} |",
        "| (e) `-O3` and `-march=native` change the kernels' time by less than 25% |"
        f" the largest change in run 1 is {worst[1][0]:.0%}, {worst[1][2]} under"
        f" {FLAG_LABELS[worst[1][1]]}"
        + (
            "; at or beyond 25%: " + "; ".join(beyond)
            if beyond
            else "; none reaches 25%"
        )
        + f" (runs 2 and 3: {worst[2][0]:.0%}, {worst[3][0]:.0%}); the checks under"
        f" each set: {'all pass' if flags_pass else 'not all pass'} (T7) |"
        f" {'holds' if holds_e else 'FAILS'} |",
    ]


def build() -> tuple[str, dict[str, str]]:
    """Every table: the whole text, and each marked block."""
    blocks = {
        "predictions": predictions_table(),
        "machine": machine_table(),
        "floor": floor_table(),
        "styles": styles_table(),
        "kernels": kernels_table(),
        "changes": changes_table(),
        "candidates": candidates_table(),
        "flags": flags_table(),
        "vectorised": vectorised_table(),
        "tests": tests_table(),
        "plants": plants_table(),
        "suites": suites_table(),
    }
    assert tuple(blocks) == MARKERS
    text = "# Prompt 89: the tables\n\n" + "\n\n".join(
        "\n".join(lines) for lines in blocks.values()
    )
    return text + "\n", {name: "\n".join(lines) for name, lines in blocks.items()}


def splice(report: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """REPORT with each marked block replaced, and the names whose markers are missing."""
    missing = []
    for name, body in blocks.items():
        begin, end = f"<!-- tables89 {name} begin -->", f"<!-- tables89 {name} end -->"
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
        print("tables89 --check:", "ok" if ok else "FAILED")
        return 0 if ok else 1
    TABLES.write_text(text, encoding="utf-8", newline="\n")
    REPORT.write_text(spliced, encoding="utf-8", newline="\n")
    if missing:
        print(f"missing markers (not spliced): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
