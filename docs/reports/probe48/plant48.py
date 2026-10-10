"""Plant each defect of prompt 48's commit A in a scratch copy and list the tests that fail.

Usage:
    python plant48.py PYTHON

PYTHON is the interpreter that runs pytest. The script copies src/, tests/,
validation/, configs/ and pyproject.toml of the tree it sits in to
results/builder48/plant_tree/ (scratch, rebuilt every run), so the working
tree is never edited. For each plant it changes one line of the copy, runs
the scheme tests there, and puts the file back byte for byte. The output is
JSON: the failing tests with no defect (the control) and under each plant,
also written to results/builder48/plants.json. A plant counts as caught only
when pytest exits 1 (tests ran and failed).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "results" / "builder48" / "plant_tree"
OUT = ROOT / "results" / "builder48" / "plants.json"
COPIED = ("src", "tests", "validation", "configs")

# The tests the plants are aimed at: the key's parsing and the scheme tests.
TESTS = (
    "tests/test_config.py::TestTurbulenceSection",
    "tests/test_turbulence.py::TestAdvectionScheme",
    "tests/test_turbulence.py::TestBoundaries::test_nothing_crosses_a_wall_or_an_obstacle_face",
)

PLANTS = {
    "the limited value used under upwind": (
        "src/turbulence.py",
        "        self._upwind = spec.advection_scheme == UPWIND",
        "        self._upwind = False",
    ),
    "upwind used under umist": (
        "src/turbulence.py",
        "        self._upwind = spec.advection_scheme == UPWIND",
        "        self._upwind = True",
    ),
    "the default umist": (
        "src/config.py",
        '        scheme = UPWIND\n        if "advection_scheme" in section:\n'
        "            scheme = cls._require_choice(\n"
        '                section, "advection_scheme", "turbulence", ADVECTION_SCHEMES',
        '        scheme = UMIST\n        if "advection_scheme" in section:\n'
        "            scheme = cls._require_choice(\n"
        '                section, "advection_scheme", "turbulence", ADVECTION_SCHEMES',
    ),
    "any string accepted": (
        "src/config.py",
        "            scheme = cls._require_choice(\n"
        '                section, "advection_scheme", "turbulence", ADVECTION_SCHEMES\n'
        "            )",
        "            scheme = cls._require_string(\n"
        '                section, "advection_scheme", "turbulence"\n'
        "            )",
    ),
}


def run_tests() -> tuple[int, list[str]]:
    """Run TESTS in the copy; the exit code and the failing node ids."""
    result = subprocess.run(
        [
            sys.argv[1],
            "-m",
            "pytest",
            *TESTS,
            "-q",
            "-p",
            "no:cacheprovider",
            "-rf",
        ],
        cwd=TREE,
        capture_output=True,
        text=True,
    )
    failed = [
        line.split(" ", 1)[1].split(" - ")[0]
        for line in result.stdout.splitlines()
        if line.startswith("FAILED ")
    ]
    return result.returncode, failed


def main() -> None:
    """Copy the tree, run the control and every plant, write the record."""
    if TREE.exists():
        shutil.rmtree(TREE)
    TREE.mkdir(parents=True)
    for name in COPIED:
        shutil.copytree(
            ROOT / name, TREE / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    shutil.copy2(ROOT / "pyproject.toml", TREE / "pyproject.toml")
    record: dict[str, dict] = {}
    code, failed = run_tests()
    record["control"] = {"exit": code, "failed": failed}
    for label, (path, old, new) in PLANTS.items():
        target = TREE / path
        original = target.read_bytes()
        text = original.decode("utf-8").replace("\r\n", "\n")
        if text.count(old) != 1:
            record[label] = {"applicable": False}
            continue
        target.write_bytes(text.replace(old, new).encode("utf-8"))
        try:
            code, failed = run_tests()
        finally:
            target.write_bytes(original)
        record[label] = {"exit": code, "caught": code == 1, "failed": failed}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
