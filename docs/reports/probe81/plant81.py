"""Plant each defect prompt 81's tests name, run the tests that should catch it, restore.

    python docs/reports/probe81/plant81.py --root WORKTREE

WORKTREE is a git worktree of this branch at the commit under test (the
plants edit its ``src/`` files, never this tree's, so the queue's runs,
which import this tree's ``src/``, are untouched). Each plant is one text
replacement in one file, made at the stated occurrence, and must change the
file; the named tests then run there with the venv's interpreter, one BLAS
thread, and the file is restored byte for byte whatever happens. A plant is
caught when the tests fail. A clean run of every named test first must pass.

The record goes to ``results/builder81/plants.json``: per plant, the file,
the replacement, the tests, pytest's exit code and the last summary line.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
OUT = ROOT / "results" / "builder81" / "plants.json"
ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

CHANNELS = "tests/test_cumulative_counting.py::TestChannels"
MONITOR = "tests/test_cumulative_counting.py::TestMonitorOnTheSum"
MAPS = "tests/test_cumulative_counting.py::TestMapsAndRecoveryOnTheSum"
FRAMES = "tests/test_cumulative_counting.py::TestFramesOnTheSum"
RECORDS = "tests/test_cumulative_counting.py::TestRecords"
HOLD = (
    "tests/test_cumulative_counting.py::test_the_hold_reads_the_start_on_the_channels"
)
RUN_MAPS = "tests/test_alert_latency.py::TestCumulativeMaps"
RESTART = (
    "tests/test_time_integration.py::TestArrivalMaps::"
    "test_a_per_class_checkpoint_refuses_the_restart"
)

# (name, what the defect is, file, old, new, occurrence (0-based), tests)
PLANTS: list[tuple[str, str, str, str, str, int, list[str]]] = [
    (
        "smaller",
        "each channel sums its class and the smaller classes",
        "src/config.py",
        "if diameters[j] >= diameters[k])",
        "if diameters[j] <= diameters[k])",
        0,
        [CHANNELS, MONITOR, MAPS, FRAMES, HOLD, RUN_MAPS],
    ),
    (
        "class_alone",
        "each channel is its class alone (the per-class comparison)",
        "src/config.py",
        "if diameters[j] >= diameters[k])",
        "if j == k)",
        0,
        [CHANNELS, MONITOR, MAPS, FRAMES, HOLD, RUN_MAPS],
    ),
    (
        "strict",
        "the monitor opens a channel's alert above its limit, not at it",
        "src/monitor.py",
        "above = sums >= self._thresholds[:, None]",
        "above = sums > self._thresholds[:, None]",
        0,
        [MONITOR],
    ),
    (
        "mix_allowed",
        "detection_units stacks per-class maps beside cumulative ones",
        "src/placement.py",
        "if len(set(countings.values())) > 1:",
        "if False:",
        0,
        [RECORDS],
    ),
    (
        "unmarked_cumulative",
        "an /arrival/ without the mark reads as cumulative",
        "src/io_manager.py",
        "counting=str(attrs.get(COUNTING_ATTR, PER_CLASS)),",
        "counting=str(attrs.get(COUNTING_ATTR, CUMULATIVE)),",
        1,
        [RECORDS],
    ),
    (
        "mark_not_written",
        "write_arrival writes no counting",
        "src/io_manager.py",
        "            group.attrs[COUNTING_ATTR] = record.counting\n",
        "",
        1,
        [RECORDS],
    ),
    (
        "flag_dropped",
        "arrival_from_run drops the file's counting",
        "src/placement.py",
        "            counting=record.counting,\n",
        "",
        0,
        [RECORDS],
    ),
    (
        "restart_unchecked",
        "a per-class checkpoint continues on the channels",
        "src/time_integration.py",
        "and COUNTING_STATE not in checkpoint.recovery",
        "and False",
        0,
        [RESTART],
    ),
    (
        "hold_per_class",
        "the hold's start check compares each class alone",
        "src/config.py",
        "for members in self.channel_classes():",
        "for members in ((k,) for k in range(len(thresholds))):",
        0,
        [HOLD],
    ),
]


def replace_at(text: str, old: str, new: str, occurrence: int) -> str:
    """TEXT with the OCCURRENCE-th OLD replaced by NEW; ValueError if it is not there.

    A worktree checked out on Windows holds CRLF line ends, so a newline in
    OLD or NEW is matched and written as the file's own.
    """
    if "\r\n" in text:
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(old, start + 1)
        if start < 0:
            raise ValueError(f"occurrence {occurrence} of {old!r} not found")
    return text[:start] + new + text[start + len(old) :]


def pytest(root: Path, tests: list[str]) -> tuple[int, str]:
    """Run TESTS under ROOT; return the exit code and pytest's last line."""
    done = subprocess.run(
        [str(PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        cwd=root,
        env={**os.environ, **ENV},
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [ln for ln in done.stdout.splitlines() if ln.strip()]
    return done.returncode, lines[-1] if lines else ""


def main() -> int:
    """Plant, test and restore each defect; write the record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    every = sorted({t for plant in PLANTS for t in plant[6]})
    clean_code, clean_line = pytest(root, every)
    record = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "clean": {"tests": every, "exit": clean_code, "summary": clean_line},
        "plants": [],
    }
    print(f"clean: exit {clean_code}, {clean_line}", flush=True)
    if clean_code != 0:
        print("the tests fail without a plant; nothing planted")
        OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return 1
    for name, what, file, old, new, occurrence, tests in PLANTS:
        path = root / file
        original = path.read_bytes()
        try:
            text = original.decode("utf-8")
            planted = replace_at(text, old, new, occurrence)
            if planted == text:
                raise ValueError("the plant changes nothing")
            path.write_bytes(planted.encode("utf-8"))
            code, line = pytest(root, tests)
        finally:
            path.write_bytes(original)
        caught = code != 0
        record["plants"].append(
            {
                "name": name,
                "defect": what,
                "file": file,
                "old": old,
                "new": new,
                "occurrence": occurrence,
                "tests": tests,
                "exit": code,
                "summary": line,
                "caught": caught,
            }
        )
        print(
            f"{name}: exit {code}, {'caught' if caught else 'SURVIVED'}; {line}",
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    survived = [p["name"] for p in record["plants"] if not p["caught"]]
    print(
        f"{len(PLANTS) - len(survived)} of {len(PLANTS)} caught; survived: {survived}"
    )
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
