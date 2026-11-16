"""The tables of prompt 86's toolchain report, from the records in results/builder86/.

    python docs/reports/probe86/tables86.py           # write tables86.md, splice the report
    python docs/reports/probe86/tables86.py --check   # regenerate and compare, write nothing

Run with Windows' ``.venv``: every record it reads is in the checkout. The
Linux reference runs are set against prompt 85's Windows records
(``results/builder85/runs/reference/``) by ``tables85.py``'s comparison,
unchanged: ``compare_run`` for the fields, the budget and the bitwise flags,
and ``Placement`` for the searches, the layouts' order and the clear times, the
Windows run as the reference and the Linux run as the variant. Each field is
read against ADR-017 H in its three tiers, with the spreads and limits below.
``--check`` also fails when one of the report's marker pairs is missing.

A run file records its scenario's base file by the path it ran with
(``/mnt/c/...`` under WSL2, ``C:\\...`` on Windows), and the readers open
that file again (``SimConfig.from_dict``). Both name the same file of this
checkout, so this process reads either form as the local one
(``read_config_here``); nothing else in a record is changed.
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "probe85"))

import probe85  # noqa: E402 -- follows sys.path.insert
import probe86  # noqa: E402 -- follows sys.path.insert
import tables85  # noqa: E402 -- follows sys.path.insert

from src.io_manager import RunConfig, RunFile  # noqa: E402 -- follows sys.path.insert

TABLES = HERE / "tables86.md"
REPORT = ROOT / "docs" / "reports" / "phase6_toolchain.md"
RESULTS = probe86.RESULTS
GRID = probe86.GRID
SCENARIOS = probe86.SCENARIOS
WINDOWS, LINUX = tables85.REFERENCE, "linux"
REQ_T05 = tables85.REQ_T05
# ADR-017 H: (the measured spread, the limit) per field. The spreads are the
# largest the probe measured as its report prints them (P8; the sensor
# channels P2's largest, as ADR-017 H takes it); the limits ten times them
# rounded up to a decade.
FLOW_SPREAD, FLOW_LIMIT = 7.8e-10, 1.0e-8
BAND = {
    **{name: (FLOW_SPREAD, FLOW_LIMIT) for name in tables85.FLOW_FIELDS},
    "concentration": (2.7e-8, 1.0e-6),
    "concentration, pointwise": (5.8e-8, 1.0e-6),
    "sensor channels": (4.7e-8, 1.0e-6),
    "leftover / supplied, change": (3.6e-10, 1.0e-8),
}
# The budget's closure, the reference's own check, not tiered.
CLOSURE = 1.0e-12
# Predictions (c) and (d): wall times within this fraction.
WALL_FRACTION = 0.10
INSIDE, BETWEEN, BEYOND = (
    "inside the spread",
    "between the spread and the limit",
    "beyond the limit",
)
MARKERS = (
    "predictions",
    "versions",
    "environment",
    "core",
    "runs",
    "fields",
    "decisions",
    "walls",
    "gate",
)


# ---------------------------------------------------------------------------
# The records
# ---------------------------------------------------------------------------


def checkout_forms() -> tuple[str, str]:
    """This checkout's root as Windows names it and as WSL2 names it."""
    if ROOT.drive:
        windows = PureWindowsPath(ROOT)
        linux = PurePosixPath("/mnt", windows.drive[0].lower(), *windows.parts[1:])
    else:
        linux = PurePosixPath(ROOT)
        parts = linux.parts
        windows = PureWindowsPath(f"{parts[2].upper()}:\\", *parts[3:])
    return str(windows), str(linux)


def local_path(name: str) -> str:
    """NAME, a path in this checkout in either platform's form, in the local form."""
    windows, linux = checkout_forms()
    if name.lower().startswith(windows.lower() + "\\"):
        return str(ROOT.joinpath(*PureWindowsPath(name[len(windows) + 1 :]).parts))
    if name.startswith(linux + "/"):
        return str(ROOT.joinpath(*PurePosixPath(name[len(linux) + 1 :]).parts))
    return name


_READ_CONFIG = RunFile.read_config


def read_config_here(run_file: RunFile) -> RunConfig:
    """RunFile.read_config with the scenario's base read as the local path."""
    record = _READ_CONFIG(run_file)
    section = record.mapping.get("scenario")
    if isinstance(section, dict) and isinstance(section.get("base"), str):
        section["base"] = local_path(section["base"])
    return record


RunFile.read_config = read_config_here


class RunAt(tables85.Run):
    """A run of either platform: Windows' records as tables85 reads them, or Linux's."""

    def __init__(
        self, variant: str, scenario: str, grid: str, location: str = "reference"
    ) -> None:
        self.variant, self.scenario, self.grid = variant, scenario, grid
        if variant == LINUX:
            self.path = probe86.run_path(location, scenario)
        else:
            self.path = probe85.run_path(variant, scenario, grid)
        self.summary = json.loads(self.path.with_suffix(".json").read_text())


class Placement(tables85.Placement):
    """tables85's placement over the two platforms' runs only."""

    def __init__(self, grid: str) -> None:
        # tables85.Placement builds its runs through its module's name Run and
        # searches every variant probe85 lists; here the name is RunAt and the
        # variants are the two platforms, for the length of the construction.
        saved_run, saved_variants = tables85.Run, probe85.VARIANTS
        tables85.Run, probe85.VARIANTS = RunAt, {WINDOWS: None, LINUX: None}
        try:
            super().__init__(grid)
        finally:
            tables85.Run, probe85.VARIANTS = saved_run, saved_variants

    def of(self, variant: str) -> dict:
        """tables85's searches, with this module's runs."""
        saved = tables85.Run
        tables85.Run = RunAt
        try:
            return super().of(variant)
        finally:
            tables85.Run = saved


def tier(value: float, field: str) -> str:
    """Which of ADR-017 H's three tiers VALUE falls in for FIELD."""
    spread, limit = BAND[field]
    if value <= spread:
        return INSIDE
    if value <= limit:
        return BETWEEN
    return BEYOND


def junit(path: Path) -> dict:
    """A pytest JUnit file's counts and time."""
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    assert suite is not None
    counts = {
        k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")
    }
    counts["passed"] = (
        counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    )
    counts["seconds"] = float(suite.get("time", "nan"))
    return counts


def freeze(path: Path) -> dict[str, str]:
    """A pip freeze as name to version, names lower-cased."""
    out = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "==" in line:
            name, version = line.strip().split("==", 1)
            out[name.lower().replace("_", "-")] = version
    return out


def yes(flag: bool) -> str:
    """yes or no."""
    return "yes" if flag else "no"


def e(value: float) -> str:
    """tables85's format for a relative difference."""
    return tables85.e(value)


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------


def blas(config: dict) -> str:
    """The BLAS np.show_config names, with its version and build options."""
    info = config.get("Build Dependencies", {}).get("blas", {})
    parts = [str(info.get(k, "")) for k in ("name", "version") if info.get(k)]
    opts = info.get("openblas configuration", "")
    return " ".join(parts) + (f" ({opts})" if opts else "")


def versions_table(versions: dict) -> list[str]:
    """T1: the versions."""
    win, lin = versions["windows"], versions["linux"]

    def after(text: str, prefix: str) -> str:
        return text.removeprefix(prefix).strip().strip('"')

    rows = [
        ["Windows build", win["build"]],
        ["WSL", "; ".join(win["wsl"])],
        ["distribution", after(lin["os"], "PRETTY_NAME=")],
        ["kernel", lin["kernel"]],
        ["glibc", lin["glibc"]],
        ["GCC", lin["gcc"]],
        ["CMake", lin["cmake"]],
        ["Ninja", lin["ninja"]],
        ["pipx", lin["pipx"]],
        ["uv", lin["uv"]],
        [
            "Python, Windows / Linux",
            f"{win['python']} / {after(lin['python'], 'Python ')}",
        ],
        ["NumPy, Windows / Linux", f"{win['numpy']} / {lin['numpy']}"],
        ["NumPy's BLAS, Windows", blas(win["numpy_config"])],
        ["NumPy's BLAS, Linux", blas(lin["numpy_config"])],
        ["NumPy's compiler, Windows", _compiler(win["numpy_config"])],
        ["NumPy's compiler, Linux", _compiler(lin["numpy_config"])],
        ["nanobind", after(lin["nanobind"], "Version:")],
        ["scikit-build-core", after(lin["scikit-build-core"], "Version:")],
        ["GoogleTest", versions["googletest"]],
    ]
    return [
        "### T1. The versions",
        "",
        *tables85.table(["", "version"], rows),
        "",
        "From `versions86.py` (`results/builder86/versions.json`): Windows' `ver` and `wsl --version`; inside `Ubuntu-24.04` the commands it names; NumPy's build from `np.show_config(mode=\"dicts\")` in each environment; GoogleTest the release `core/CMakeLists.txt` fetches.",
    ]


def _compiler(config: dict) -> str:
    """NumPy's C compiler as show_config names it."""
    c = config.get("Compilers", {}).get("c", {})
    return f"{c.get('name', '')} {c.get('version', '')}".strip()


def environment_table() -> list[str]:
    """T2: the Linux environment's packages against Windows', and the suites."""
    win = freeze(RESULTS / "windows-freeze.txt")
    lin = freeze(RESULTS / "linux-freeze.txt")
    now = freeze(RESULTS / "linux-freeze-core.txt")
    same = sum(1 for k, v in win.items() if lin.get(k) == v)
    added = sorted(f"{k} {v}" for k, v in now.items() if k not in win)
    rows = [
        ["the Windows environment's packages", str(len(win))],
        ["installed under Linux at the same version", str(same)],
        ["at another version, or missing", str(len(win) - same)],
        ["added under Linux for the core's build, and the core", ", ".join(added)],
    ]
    suites = []
    for name in ("windows", "linux"):
        c = junit(RESULTS / f"suite_{name}.xml")
        suites.append(
            [
                name.capitalize(),
                f"{c['tests']:,}",
                f"{c['passed']:,}",
                str(c["failures"]),
                str(c["errors"]),
                str(c["skipped"]),
                f"{c['seconds']:.0f}",
            ]
        )
    return [
        "### T2. The Linux environment, and the whole suite on each platform",
        "",
        *tables85.table(["", ""], rows),
        "",
        *tables85.table(
            [
                "environment",
                "tests",
                "passed",
                "failed",
                "errors",
                "skipped",
                "wall, s",
            ],
            suites,
        ),
        "",
        "Packages: `results/builder86/windows-freeze.txt` (the Windows `.venv`), `linux-freeze.txt` (the Linux environment as installed from it) and `linux-freeze-core.txt` (after `requirements-dev.txt`'s build dependencies and `pip install --no-build-isolation ./core`). Suites: `pytest tests/ -q -p no:cacheprovider` with its JUnit file, each run alone with the default BLAS threads; the time is pytest's.",
    ]


def core_table(parity: dict, ctest: dict) -> list[str]:
    """T3: CTest and prediction (b)'s comparison."""
    v, f = parity["vectors"], parity["fields"]
    rows = [
        [
            "1-D, lengths 0 to 1,000 and 2,400, 9,600, 38,400",
            f"{v['compared']:,}",
            str(len(v["differ"])),
            f"{v['order_shows']:,}",
        ],
        [
            "2-D, "
            + ", ".join(f"{a}x{b}" for a, b in parity["shapes"])
            + ", summed whole",
            str(f["compared"]),
            str(len(f["differ"])),
            str(f["order_shows"]),
        ],
    ]
    return [
        "### T3. The core's tests: CTest, and `pairwise_sum` against `np.sum`",
        "",
        f"CTest under WSL2: {ctest['tests']} tests, {ctest['passed']} passed, {ctest['failures']} failed.",
        "",
        *tables85.table(
            [
                "arrays",
                "compared",
                "differing from np.sum (64-bit pattern)",
                "where a left-to-right sum differs from np.sum",
            ],
            rows,
        ),
        "",
        "From `probe86.py parity` (`results/builder86/parity86.json`), which repeats `tests/test_core_parity.py`'s comparison on the same seeded values, and CTest's JUnit file (`results/builder86/ctest.xml`). The last column says how many arrays the comparison could catch a wrong order on.",
    ]


def runs_table(results: dict) -> list[str]:
    """T4: the runs, their stops and whether each is bitwise the Windows record."""
    rows = []
    for scenario in SCENARIOS:
        c = results[scenario]
        win, lin = RunAt(WINDOWS, scenario, GRID), RunAt(LINUX, scenario, GRID)
        reasons = [
            f"{a['stop_reason']} / {b['stop_reason']}"
            for a, b in zip(win.summary["flows"], lin.summary["flows"], strict=True)
        ]
        rows.append(
            [
                scenario,
                " / ".join(str(o) for o in c["ref_outers"]),
                " / ".join(str(o) for o in c["outers"]),
                "; ".join(reasons),
                f"{win.summary['steps']} / {lin.summary['steps']}",
                yes(c["flow_bitwise"]),
                yes(c["fields_bitwise"]),
                yes(c["budgets_bitwise"]),
                yes(c["sensors_bitwise"]),
                yes(c["arrival_bitwise"]),
                yes(c["bitwise"]),
            ]
        )
    return [
        "### T4. The runs: the stops, and whether each Linux run is bitwise the Windows record",
        "",
        *tables85.table(
            [
                "scenario",
                "outer per state, Windows",
                "Linux",
                "stop reasons, Windows / Linux",
                "steps, Windows / Linux",
                "flow states bitwise",
                "class frames",
                "budget rows",
                "sensors",
                "arrival maps",
                "bitwise",
            ],
            rows,
        ),
        "",
        "Bitwise as `tables85.compare_run` defines it (prompt 85's P1). The run files record the outer count of each flow state and its stop reason, not the pressure CG's or the implicit solves' counts (T4c counts those).",
    ]


def readings_table() -> list[str]:
    """T4d: the readings each flow state's run file records at its stop, on both platforms."""
    rows = []
    for scenario in SCENARIOS:
        win, lin = RunAt(WINDOWS, scenario, GRID), RunAt(LINUX, scenario, GRID)
        tol = win.config.mass_imbalance_tol
        pairs = zip(_records(win), _records(lin), strict=True)
        for state, (a, b) in enumerate(pairs):
            if a.rule_readings.keys() != b.rule_readings.keys():
                raise ValueError(f"{scenario}: the two runs record different readings")
            for name, x in a.rule_readings.items():
                y = b.rule_readings[name]
                scale = e(tol) if name.endswith("imbalance") else "-"
                rows.append(
                    [
                        scenario,
                        str(state),
                        name,
                        f"{x:.6e}",
                        f"{y:.6e}",
                        e(abs(y - x)),
                        scale,
                    ]
                )
    return [
        "### T4d. The readings at each flow state's stop, Windows against Linux",
        "",
        *tables85.table(
            [
                "scenario",
                "flow state",
                "reading",
                "Windows",
                "Linux",
                "|difference|",
                "the stop's tolerance for it",
            ],
            rows,
        ),
        "",
        "The readings `time_integration` records with each flow state: the last outer residual, the largest per-cell mass imbalance, the imbalance's signed sum over the cells and the corrections that reached the CG's cap. The two imbalances are tested against `mass_imbalance_tol` (`stopping.py`, conditions (b) and (d)). Each is a small quantity at a converged stop, the signed sum a cancelling one of rounding size, so its difference is read against its tolerance, not against itself (ADR-017 G).",
    ]


def repeats_table(repeats: dict) -> list[str]:
    """T4b: the Linux reference reproduces itself."""
    rows = [
        [loc, " / ".join(str(o) for o in c["outers"]), yes(c["bitwise"])]
        for loc, c in repeats.items()
    ]
    return [
        "### T4b. The Linux dust release's other runs on /mnt/c against the Linux reference's",
        "",
        *tables85.table(
            ["run", "outer per state", "bitwise the Linux reference"], rows
        ),
        "",
        "Each from rest in its own process, alone, where the reference ran three at once. The runs on the Linux file system are not compared: Windows does not read that file system.",
    ]


COUNT_KINDS = {
    "cg": "pressure CG iterations",
    "transport": "transport implicit solves (iterations, fallback sweeps)",
    "turbulence": "k and eps implicit solves (iterations, fallback sweeps)",
}


def counts_table() -> list[str]:
    """T4c: every inner solve's count, Windows against Linux, from the counted runs."""
    rows, checks = [], []
    for scenario in SCENARIOS:
        win = np.load(probe86.counts_path("windows_counted", scenario))
        lin = np.load(probe86.counts_path("counted", scenario))
        for kind, label in COUNT_KINDS.items():
            a, b = win[kind], lin[kind]
            n = min(len(a), len(b))
            differ = np.flatnonzero(np.any(a[:n] != b[:n], axis=1))
            first = f"{int(differ[0]):,}" if differ.size else "-"
            total = int(a[:, 0].sum()), int(b[:, 0].sum())
            rows.append(
                [
                    scenario,
                    label,
                    f"{len(a):,} / {len(b):,}",
                    f"{differ.size:,}",
                    first,
                    f"{total[0]:,} / {total[1]:,}",
                ]
            )
        for platform_name, variant, location in (
            ("Windows", WINDOWS, "windows_counted"),
            ("Linux", LINUX, "counted"),
        ):
            ref = RunAt(variant, scenario, GRID)
            counted = RunAt(LINUX, scenario, GRID, location)
            c = tables85.compare_run(ref, counted)
            ref.release()
            checks.append([scenario, platform_name, yes(c["bitwise"])])
    return [
        "### T4c. Every inner solve's iteration count, Windows against Linux",
        "",
        *tables85.table(
            [
                "scenario",
                "solve",
                "calls, Windows / Linux",
                "calls whose counts differ",
                "the first, by call",
                "iterations in all, Windows / Linux",
            ],
            rows,
        ),
        "",
        *tables85.table(
            ["scenario", "platform", "the counted run bitwise the uncounted"], checks
        ),
        "",
        "From the counted runs (`probe86.py`, `StopCounts`): the three references again on each platform, three at once, with every call's count recorded in call order. A call compared is the same call in both runs while the counts before it agree; once one differs, later calls may be different calls. Each counted run is bitwise its platform's uncounted one (the second table), so counting changed nothing.",
    ]


def _records(run: RunAt) -> list[object]:
    """The run's flow records in the order its intervals first use them."""
    order: list[str] = []
    for plan in run.summary["intervals"]:
        if plan["fingerprint"] not in order:
            order.append(plan["fingerprint"])
    return [run.file.read_flow_state(fp) for fp in order]


def _differences(c: dict) -> dict[str, float]:
    """One run's comparison as field name to its relative difference."""
    change = max(
        abs(lo - lr)
        for (_, lo), (_, lr) in zip(c["budget"], c["ref_budget"], strict=True)
    )
    return {
        **c["flow"],
        "concentration": max(c["concentration"]),
        "concentration, pointwise": max(c["pointwise"]),
        "sensor channels": c["sensors"],
        "leftover / supplied, change": change,
    }


def fields_table(results: dict) -> tuple[list[str], dict]:
    """T5: each field's difference and its tier."""
    rows, tiers = [], {}
    for scenario in SCENARIOS:
        diffs = _differences(results[scenario])
        for field, value in diffs.items():
            spread, limit = BAND[field]
            t = tier(value, field)
            tiers[(scenario, field)] = t
            rows.append([scenario, field, e(value), e(spread), e(limit), t])
    closure = {s: max(abs(r) for r, _ in results[s]["budget"]) for s in SCENARIOS}
    lines = [
        "### T5. Each field's largest relative difference from the Windows record, read against ADR-017 H",
        "",
        *tables85.table(
            ["scenario", "field", "difference", "measured spread", "limit", "tier"],
            rows,
        ),
        "",
        "Differences as prompt 85's P2 and P6 measure them (`tables85.compare_run`): a flow field's over its states, a class's over every stored frame, the pointwise reading where the Windows run reads at least a hundredth of the class's threshold, the sensor channels at every step, and the change in the accumulated leftover over the supply at the run's end. The spreads are P8's (the sensor channels P2's), the limits ten times them rounded up to a decade (ADR-017 H).",
        "",
        "The budget's closure, the reference's own check: the largest residual over the supply at the run's end, "
        + "; ".join(f"{s} {e(v)}" for s, v in closure.items())
        + f", against {e(CLOSURE)}.",
    ]
    return lines, {"tiers": tiers, "closure": closure}


def decisions_table(results: dict, place: Placement) -> tuple[list[str], bool]:
    """T6: the decisions against the Windows records."""
    arrival = [results[s]["arrival_moves"] for s in SCENARIOS]
    sensor = [results[s]["sensor_moves"] for s in SCENARIOS]
    cmp = place.compare(LINUX)
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
                s,
                e(max(abs(lo) for _, lo in c["ref_budget"])),
                e(max(abs(lo) for _, lo in c["budget"])),
                yes(ok == ref),
            ]
        )
    equal = moves_equal and cells_equal and order_equal and verdicts_equal
    lines = [
        "### T6. The decisions: latencies and clear times moved, the searches, the layouts' order, the budget's verdicts",
        "",
        *tables85.table(
            ["decision", "compared", "moved 1 / > 1 / one only", "largest move"],
            rows,
        ),
        "",
        *tables85.table(
            [
                "rule",
                "searched layouts whose cells differ",
                "order, Windows",
                "order, Linux",
            ],
            searches,
        ),
        "",
        *tables85.table(
            [
                "scenario",
                "leftover / supplied, Windows (largest class)",
                "Linux",
                "every class's verdict against REQ-T05 the same",
            ],
            verdicts,
        ),
        "",
        "Prompt 85's P3 to P6 measures, the Windows run as the reference: a latency moves by its difference over the step of the Windows run's interval, a clear time by its difference over the unit's frame spacing; the searches are `placement.search` over the three runs' maps with main's placement spec, under both rules and every pool.",
    ]
    return lines, equal


def walls_table() -> tuple[list[str], dict]:
    """T7: the wall times, and predictions (c) and (d)."""
    windows = RunAt(WINDOWS, "dust", GRID).summary
    summaries = {"reference": RunAt(LINUX, "dust", GRID).summary}
    for loc in LOCATED:
        path = RESULTS / "runs" / loc / f"dust_{GRID}.json"
        summaries[loc] = json.loads(path.read_text())
    labels = {
        "reference": ("Linux reference", "/mnt/c", "three at once"),
        "mnt_c": ("Linux, run file on /mnt/c", "/mnt/c", "alone"),
        "linux_fs": ("Linux, run file on the Linux file system", "~/cfd-runs", "alone"),
        "mnt_c_repeat": ("the same again (an addition)", "/mnt/c", "alone"),
        "linux_fs_repeat": ("the same again (an addition)", "~/cfd-runs", "alone"),
    }
    w = windows["wall_seconds"]
    rows = [
        _wall_row("Windows (prompt 85's record)", "C:", "three at once", windows, w)
    ]
    rows += [_wall_row(*labels[k], s, w) for k, s in summaries.items()]

    def wall(name: str) -> float:
        return summaries[name]["wall_seconds"]

    c = abs(wall("mnt_c") - wall("linux_fs")) / wall("linux_fs")
    c_repeat = abs(wall("mnt_c_repeat") - wall("linux_fs_repeat")) / wall(
        "linux_fs_repeat"
    )
    d = abs(wall("reference") - w) / w
    lines = [
        "### T7. The 80x30 dust release's wall time",
        "",
        *tables85.table(
            [
                "run",
                "run file",
                "processes",
                "wall, s",
                "the flow solve, s",
                "the rest, s",
                "over Windows'",
            ],
            rows,
        ),
        "",
        f"The run file's placement, (c): |W(/mnt/c) - W(Linux)| / W(Linux) = {c:.3f} (the repeated pair {c_repeat:.3f}). The Linux reference against Windows, (d): |W(Linux) - W(Windows)| / W(Windows) = {d:.3f}. The wall time is the summary's `wall_seconds`, `time_integration.run` on one BLAS thread; the flow solve is its flow state's own `wall_seconds`, and the rest is the march with its frame, sensor and checkpoint writes.",
    ]
    return lines, {"c": c, "c_repeat": c_repeat, "d": d}


# The located runs of the dust release, the repeats an addition to the prompt's pair.
LOCATED = ("mnt_c", "linux_fs", "mnt_c_repeat", "linux_fs_repeat")


def _wall_row(name: str, where: str, procs: str, summary: dict, windows: float) -> list:
    """One row of T7."""
    total = summary["wall_seconds"]
    flow = sum(f["wall_seconds"] for f in summary["flows"])
    return [
        name,
        where,
        procs,
        f"{total:.1f}",
        f"{flow:.1f}",
        f"{total - flow:.1f}",
        f"{total / windows:.3f}",
    ]


def predictions_table(
    results: dict, fields: dict, decisions_equal: bool, parity: dict, walls: dict
) -> tuple[list[str], dict]:
    """T0: the predictions scored by the rule committed with them."""
    not_bitwise = [s for s in SCENARIOS if not results[s]["bitwise"]]
    inside = all(t == INSIDE for t in fields["tiers"].values())
    closes = all(v <= CLOSURE for v in fields["closure"].values())
    a = bool(not_bitwise) and decisions_equal and inside and closes
    outside = sorted(
        {f"{f} ({s})" for (s, f), t in fields["tiers"].items() if t != INSIDE}
    )
    b_differ = len(parity["vectors"]["differ"]) + len(parity["fields"]["differ"])
    b = b_differ == 0
    c, d = walls["c"] <= WALL_FRACTION, walls["d"] <= WALL_FRACTION
    rows = [
        [
            "(a)",
            "not bitwise on at least one run; every decision equal; every field inside the measured spread",
            f"not bitwise: {', '.join(not_bitwise) if not_bitwise else 'none'}; decisions equal: {yes(decisions_equal)}; "
            f"every field inside the spread: {yes(inside)}"
            + (f" (not: {', '.join(outside)})" if outside else "")
            + f"; budget closes: {yes(closes)}",
            "HOLDS" if a else "FAILS",
        ],
        [
            "(b)",
            "`pairwise_sum` equals `np.sum` bit for bit on every array tested",
            f"{parity['vectors']['compared'] + parity['fields']['compared']:,} arrays, {b_differ} differing",
            "HOLDS" if b else "FAILS",
        ],
        [
            "(c)",
            "the run file on /mnt/c within 10% of the Linux file system",
            f"{walls['c']:.3f}",
            "HOLDS" if c else "FAILS",
        ],
        [
            "(d)",
            "the Linux reference within 10% of the Windows run",
            f"{walls['d']:.3f}",
            "HOLDS" if d else "FAILS",
        ],
    ]
    return (
        [
            "### T0. The predictions, scored by the rule committed with them",
            "",
            *tables85.table(["", "prediction", "measured", "verdict"], rows),
        ],
        {"a": a, "b": b, "c": c, "d": d, "inside": inside},
    )


def gate_table(
    fields: dict, decisions_equal: bool, ctest: dict, walls: dict
) -> list[str]:
    """T8: Phase 6 step 1's gate, row by row."""
    worst = {INSIDE: 0, BETWEEN: 1, BEYOND: 2}
    tiers = list(fields["tiers"].values())
    worst_tier = max(tiers, key=lambda t: worst[t])
    in_band = worst_tier != BEYOND and all(
        v <= CLOSURE for v in fields["closure"].values()
    )
    rows = [
        [
            "CTest passes on WSL2",
            f"{ctest['passed']} of {ctest['tests']}",
            "MET" if ctest["failures"] == 0 and ctest["tests"] > 0 else "NOT MET",
        ],
        [
            "CTest passes in CI",
            "the core job of `.github/workflows/ci.yml`, on the pull request",
            "on the pull request's CI",
        ],
        [
            "the reference under WSL2: every decision equal to the Windows records",
            yes(decisions_equal),
            "MET" if decisions_equal else "NOT MET",
        ],
        [
            "the reference under WSL2: every field inside the band",
            f"the worst field is {worst_tier}",
            "MET"
            if in_band and worst_tier == INSIDE
            else ("MET if the stop counts explain it" if in_band else "NOT MET"),
        ],
        [
            "a reference run's wall time with its run file on /mnt/c and on the Linux file system recorded",
            f"T7; their difference {walls['c']:.3f} of the Linux file system's",
            "MET",
        ],
    ]
    return [
        "### T8. Phase 6 step 1's gate (`docs/PROJECT_PLAN.md`), row by row",
        "",
        *tables85.table(["criterion", "measured", "verdict"], rows),
    ]


# ---------------------------------------------------------------------------
# Build, splice, check
# ---------------------------------------------------------------------------


def ctest_counts() -> dict:
    """CTest's JUnit file's counts."""
    root = ET.parse(RESULTS / "ctest.xml").getroot()
    tests = int(root.get("tests", 0))
    failures = int(root.get("failures", 0)) + int(root.get("errors", 0))
    return {
        "tests": tests,
        "failures": failures,
        "passed": tests - failures - int(root.get("skipped", 0)),
    }


def build() -> tuple[str, dict[str, str]]:
    """Every table, as the tables file's text and the report's blocks."""
    results = {}
    for scenario in SCENARIOS:
        win, lin = RunAt(WINDOWS, scenario, GRID), RunAt(LINUX, scenario, GRID)
        results[scenario] = tables85.compare_run(win, lin)
        win.release()
    repeats = {}
    for loc in ("mnt_c", "mnt_c_repeat"):
        ref = RunAt(LINUX, "dust", GRID)
        repeats[loc] = tables85.compare_run(ref, RunAt(LINUX, "dust", GRID, loc))
        ref.release()
    place = Placement(GRID)
    versions = json.loads((RESULTS / "versions.json").read_text(encoding="utf-8"))
    parity = json.loads((RESULTS / "parity86.json").read_text())
    ctest = ctest_counts()
    field_lines, fields = fields_table(results)
    decision_lines, decisions_equal = decisions_table(results, place)
    wall_lines, walls = walls_table()
    pred_lines, _verdicts = predictions_table(
        results, fields, decisions_equal, parity, walls
    )
    blocks = {
        "predictions": "\n".join(pred_lines),
        "versions": "\n".join(versions_table(versions)),
        "environment": "\n".join(environment_table()),
        "core": "\n".join(core_table(parity, ctest)),
        "runs": "\n".join(
            [
                *runs_table(results),
                "",
                *repeats_table(repeats),
                "",
                *counts_table(),
                "",
                *readings_table(),
            ]
        ),
        "fields": "\n".join(field_lines),
        "decisions": "\n".join(decision_lines),
        "walls": "\n".join(wall_lines),
        "gate": "\n".join(gate_table(fields, decisions_equal, ctest, walls)),
    }
    text = "# Prompt 86's tables\n\nGenerated by `tables86.py` from `results/builder86/` and `results/builder85/`.\n\n"
    text += "\n\n".join(blocks[k] for k in MARKERS)
    return text + "\n", blocks


def splice(report: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """REPORT with each marked block replaced, and the names whose markers are missing."""
    missing = []
    for name, body in blocks.items():
        begin, end = f"<!-- tables86 {name} begin -->", f"<!-- tables86 {name} end -->"
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
            "tables86 --check:",
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
