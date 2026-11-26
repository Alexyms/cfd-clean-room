"""Prompt 96's planted defects, and the runner that shows the tests catch each.

    python docs/reports/probe96/variants96.py check
    python docs/reports/probe96/variants96.py run --platform windows
    python docs/reports/probe96/variants96.py run --platform linux

A plant is one exact-text edit of one file (found exactly once, or the plant is
refused), made in a copy of the commit's tree (``git archive``), with the tests
that should catch it run there. Caught: pytest exits other than 0. The first
nine are the prompt's list; the rest are the builder's, one per rule the step
added that the prompt's list does not reach. ``windows`` plants run on Windows'
environment from ``results/builder96/plants/tree``; ``linux`` plants need the
core and run under WSL2 from ``~/scratch96/tree`` with the build tree on
``PYTHONPATH``. ``check`` applies every plant to the sources in memory and
reports one whose text is not found once. The records go to
``results/builder96/plants_<platform>.json``.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


@dataclasses.dataclass(frozen=True)
class Plant:
    """One planted defect: the file, the text replaced, the tests that catch it."""

    name: str
    what: str
    path: str
    old: str
    new: str
    platform: str
    tests: tuple[str, ...]


WINDOWS_BACKEND_RUN = ("tests/test_time_integration.py", "-k", "BackendKeyInTheRun")
CORE_BACKEND = ("tests/test_core_backend.py",)
CONFIG_BACKEND = ("tests/test_config.py", "-k", "BackendKey")
MONITOR = ("tests/test_monitor.py", "-k", "FromReadings")
COMPARATOR = ("tests/test_compare_runs.py",)
HARNESS = ("tests/test_benchmark.py", "-k", "backend or factory")

PLANTS = (
    Plant(
        "restart_hash_without_key",
        "the key left out of the restart hash",
        "src/io_manager.py",
        '    ("solver", "checkpoint_interval"),\n)',
        '    ("solver", "checkpoint_interval"),\n    ("solver", "backend"),\n)',
        "windows",
        WINDOWS_BACKEND_RUN,
    ),
    Plant(
        "restart_hash_without_key_core",
        "the key left out of the restart hash (the core's whole runs)",
        "src/io_manager.py",
        '    ("solver", "checkpoint_interval"),\n)',
        '    ("solver", "checkpoint_interval"),\n    ("solver", "backend"),\n)',
        "linux",
        CORE_BACKEND,
    ),
    Plant(
        "default_core",
        "the default core",
        "src/config.py",
        "DEFAULT_BACKEND = REFERENCE_BACKEND",
        "DEFAULT_BACKEND = CORE_BACKEND",
        "windows",
        CONFIG_BACKEND,
    ),
    Plant(
        "readings_one_step_late",
        "the monitor reading the readings one step late",
        "src/core_backend.py",
        'step, t, result["readings"][r], result["channels"][r]',
        'step, t, result["readings"][max(r - 1, 0)], result["channels"][max(r - 1, 0)]',
        "linux",
        CORE_BACKEND,
    ),
    Plant(
        "channel_wrong_class",
        "a channel read from the wrong class",
        "src/monitor.py",
        'return self._advance(step, t, arrays["readings"], arrays["channels"])',
        'return self._advance(\n            step, t, arrays["readings"], np.roll(arrays["channels"], 1, axis=0)\n        )',
        "windows",
        MONITOR,
    ),
    Plant(
        "channel_wrong_class_core",
        "a channel read from the wrong class (the core's whole runs)",
        "src/monitor.py",
        'return self._advance(step, t, arrays["readings"], arrays["channels"])',
        'return self._advance(\n            step, t, arrays["readings"], np.roll(arrays["channels"], 1, axis=0)\n        )',
        "linux",
        CORE_BACKEND,
    ),
    Plant(
        "sign_reversed",
        "the comparator's sign reversed",
        "scripts/compare_runs.py",
        "diff = float(np.abs(b[finite] - a[finite]).max())",
        "diff = float((b[finite] - a[finite]).max())",
        "windows",
        COMPARATOR,
    ),
    Plant(
        "tier_bounds_swapped",
        "its tier bounds swapped",
        "scripts/compare_runs.py",
        "    spread, limit = BAND[name]\n    if value <= spread:",
        "    limit, spread = BAND[name]\n    if value <= spread:",
        "windows",
        COMPARATOR,
    ),
    Plant(
        "straddle_as_no_move",
        "a straddling move read as no move",
        "scripts/compare_runs.py",
        "    out = np.full(ref.shape, math.inf)\n    both =",
        "    out = np.full(ref.shape, 0.0)\n    both =",
        "windows",
        COMPARATOR,
    ),
    Plant(
        "decision_wrong_step",
        "a decision compared at the wrong step (the earlier time's)",
        "scripts/compare_runs.py",
        "later = np.where(np.isfinite(other) & (other > ref), other, ref)",
        "later = np.where(np.isfinite(other) & (other < ref), other, ref)",
        "windows",
        COMPARATOR,
    ),
    Plant(
        "factory_reference_under_core",
        "the factory building the reference under core",
        "src/core_backend.py",
        "        return CoreSolver(mesh, config, boundary)\n    return reference(",
        "        return reference(mesh, config, boundary)\n    return reference(",
        "linux",
        CORE_BACKEND,
    ),
    # The builder's.
    Plant(
        "core_with_transient",
        "the core accepted with a transient section",
        "src/config.py",
        "        if self.transient is not None:\n            raise ValueError(\n"
        "                f\"solver.backend '{CORE_BACKEND}' is refused with a transient",
        "        if False:\n            raise ValueError(\n"
        "                f\"solver.backend '{CORE_BACKEND}' is refused with a transient",
        "windows",
        CONFIG_BACKEND,
    ),
    Plant(
        "core_without_its_module",
        "the core accepted where _cfdcore does not import",
        "src/config.py",
        "            importlib.import_module(CORE_MODULE)\n",
        "            pass\n",
        "windows",
        CONFIG_BACKEND,
    ),
    Plant(
        "run_state_reference_under_core",
        "time_integration building the reference's run under core",
        "src/time_integration.py",
        "    if config.backend == CORE_BACKEND:\n        # Imported here: core_backend builds",
        "    if False:\n        # Imported here: core_backend builds",
        "linux",
        CORE_BACKEND,
    ),
    Plant(
        "update_resums_channels",
        "the update from readings summing the channels itself",
        "src/monitor.py",
        'return self._advance(step, t, arrays["readings"], arrays["channels"])',
        'return self._advance(\n            step, t, arrays["readings"], self._sums(arrays["readings"])\n        )',
        "windows",
        MONITOR,
    ),
    Plant(
        "missing_alert_as_none",
        "an alert in one run only read as no move",
        "scripts/compare_runs.py",
        "        moves.extend([math.inf] * (2 * abs(len(xs) - len(ys))))",
        "        moves.extend([])",
        "windows",
        COMPARATOR,
    ),
    Plant(
        "closure_unchecked",
        "the closure not checked",
        "scripts/compare_runs.py",
        "            closure <= CLOSURE,",
        "            True,",
        "windows",
        COMPARATOR,
    ),
    Plant(
        "explained_by_equal_stops",
        "a middle-tier field explained by equal stops",
        "scripts/compare_runs.py",
        "    return bool(aparts) and max(aparts) == 1",
        "    return bool(aparts) and max(aparts) <= 1",
        "windows",
        COMPARATOR,
    ),
    Plant(
        "kind_change_as_equal",
        "a value gone NaN compared on the finite entries only",
        "scripts/compare_runs.py",
        "    if not np.array_equal(finite, np.isfinite(b)):\n        return math.inf",
        "    finite = finite & np.isfinite(b)",
        "windows",
        COMPARATOR,
    ),
    Plant(
        "kind_change_unchecked",
        "a value gone NaN compared on the finite entries only, both checks removed",
        "scripts/compare_runs.py",
        "    if not np.array_equal(finite, np.isfinite(b)):\n        return math.inf\n"
        "    odd = ~finite\n"
        "    if not np.array_equal(a[odd], b[odd], equal_nan=True):\n"
        "        return math.inf\n",
        "    finite = finite & np.isfinite(b)\n",
        "windows",
        COMPARATOR,
    ),
    Plant(
        "harness_builds_staggered",
        "the harness building StaggeredSolver itself",
        "scripts/benchmark.py",
        "    solver = steady_solver(\n        mesh, config, StaggeredBoundary(mesh, config), reference=StaggeredSolver\n    )",
        "    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))",
        "windows",
        HARNESS,
    ),
    Plant(
        "old_rows_as_core",
        "an old harness row read as the core's",
        "scripts/benchmark.py",
        'return str(row.get("params", {}).get("backend", DEFAULT_BACKEND))',
        'return str(row.get("params", {}).get("backend", "core"))',
        "windows",
        HARNESS,
    ),
)


# Plants that change no result, with the reason. ``relative`` refuses a change of
# kind twice: the finite masks differ, or the non-finite entries do; a value
# finite in one run and not in the other is in the second set too, so removing
# the first check alone leaves the second to refuse it. ``kind_change_unchecked``
# removes both, and is caught.
EQUIVALENT = {
    "kind_change_as_equal": (
        "the second check (the non-finite entries compared) refuses every change of "
        "kind the first does"
    ),
}


def commit() -> str:
    """HEAD's hash read from the checkout's files (git runs on Windows only)."""
    head = (ROOT / ".git" / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = ROOT / ".git" / head[5:]
    if ref.exists():
        return ref.read_text().strip()
    for line in (ROOT / ".git" / "packed-refs").read_text().splitlines():
        if line.endswith(head[5:]):
            return line.split()[0]
    return "unknown"


def _read(path: Path) -> str:
    """A file's text with LF line ends."""
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def check(tree: Path = ROOT) -> list[str]:
    """Every plant whose text is not found exactly once in TREE's file."""
    bad = []
    for plant in PLANTS:
        count = _read(tree / plant.path).count(plant.old)
        if count != 1:
            bad.append(f"{plant.name}: found {count} times")
    return bad


def apply(tree: Path, plant: Plant) -> str:
    """Write the plant into TREE; return the file's text before it."""
    path = tree / plant.path
    text = _read(path)
    if text.count(plant.old) != 1:
        raise SystemExit(f"{plant.name}: its text is not found once")
    path.write_bytes(text.replace(plant.old, plant.new).encode("utf-8"))
    return text


def run(
    platform: str, tree: Path, python: str, out: Path, only: list[str] | None = None
) -> int:
    """Each plant of PLATFORM in TREE (those named in ONLY), its tests run there.

    The records go to OUT; with ONLY they are merged into OUT's earlier ones.
    """
    records = []
    chosen = [p for p in PLANTS if p.platform == platform]
    if only:
        chosen = [p for p in chosen if p.name in only]
    for plant in chosen:
        before = apply(tree, plant)
        started = time.perf_counter()
        try:
            done = subprocess.run(
                [
                    python,
                    "-m",
                    "pytest",
                    *plant.tests,
                    "-q",
                    "--tb=no",
                    "-x",
                    "-p",
                    "no:cacheprovider",
                ],
                cwd=tree,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
        finally:
            (tree / plant.path).write_bytes(before.encode("utf-8"))
        tail = [line for line in done.stdout.splitlines() if line.strip()][-1:]
        record = {
            "name": plant.name,
            "what": plant.what,
            "file": plant.path,
            "tests": " ".join(plant.tests),
            "returncode": done.returncode,
            "caught": done.returncode != 0,
            "equivalent": EQUIVALENT.get(plant.name),
            "summary": tail[0] if tail else done.stderr.strip().splitlines()[-1:],
            "seconds": round(time.perf_counter() - started, 1),
        }
        records.append(record)
        print(
            plant.name,
            "caught" if record["caught"] else "SURVIVES",
            record["summary"],
            flush=True,
        )
    # The tree unchanged: the same tests pass there.
    tests = sorted(
        {
            t
            for p in PLANTS
            if p.platform == platform
            for t in p.tests
            if t.startswith("tests/")
        }
    )
    clean = subprocess.run(
        [python, "-m", "pytest", *tests, "-q", "--tb=no", "-p", "no:cacheprovider"],
        cwd=tree,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    lines = [line for line in clean.stdout.splitlines() if line.strip()]
    if only and out.exists():
        earlier = json.loads(out.read_text(encoding="utf-8"))["plants"]
        names = {r["name"] for r in records}
        records = [r for r in earlier if r["name"] not in names] + records
        order = [p.name for p in PLANTS]
        records.sort(key=lambda r: order.index(r["name"]))
        for record in records:
            record["equivalent"] = EQUIVALENT.get(record["name"])
    result = {
        "platform": platform,
        "commit": commit(),
        "plants": records,
        "unchanged_tree": {
            "returncode": clean.returncode,
            "summary": lines[-1] if lines else "",
        },
    }
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print("unchanged tree:", result["unchanged_tree"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--platform", choices=("windows", "linux"), default="windows")
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "check":
        bad = check(args.tree or ROOT)
        print("\n".join(bad) if bad else f"all {len(PLANTS)} plants found once")
        return 1 if bad else 0
    out = args.out or ROOT / "results" / "builder96" / f"plants_{args.platform}.json"
    return run(args.platform, args.tree, args.python, out, args.only)


if __name__ == "__main__":
    sys.exit(main())
