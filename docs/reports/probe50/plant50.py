"""Plant each defect of prompt 50's commit A in a scratch copy and list the tests that fail.

Usage:
    python plant50.py PYTHON

PYTHON is the interpreter that runs pytest. The script copies src/, tests/,
validation/, configs/ and pyproject.toml of the tree it sits in to
results/builder50/plant_tree/ (scratch, rebuilt every run), so the working
tree is never edited. For each plant it changes one line of the copy, runs
the rule's and the coupled solve's tests there, and puts the file back byte
for byte. The output is JSON: the failing tests with no defect (the control)
and under each plant, also written to results/builder50/plants.json. A plant
counts as caught only when pytest exits 1 (tests ran and failed).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "results" / "builder50" / "plant_tree"
OUT = ROOT / "results" / "builder50" / "plants.json"
COPIED = ("src", "tests", "validation", "configs")

# The tests the plants are aimed at: condition (e) on synthetic histories, and
# what the coupled solve hands the rule.
TESTS = (
    "tests/test_stopping.py",
    "tests/test_coupled_solve.py",
)

SCALE_LINE = "                nu_scale = viscosity_scale(self._mu / self._rho, state.nu_t, fluid)"
ESTIMATE_LINE = "            nu_estimate = self._estimate_of(self._nu_steps, nu_scale)"

PLANTS = {
    "the maximum in place of the median": (
        "src/stopping.py",
        "    return nu + float(np.median(nu_t[fluid]))",
        "    return nu + float(np.max(nu_t[fluid]))",
    ),
    "the median over every cell": (
        "src/stopping.py",
        "    return nu + float(np.median(nu_t[fluid]))",
        "    return nu + float(np.median(nu_t))",
    ),
    "nu left out of the scale": (
        "src/stopping.py",
        "    return nu + float(np.median(nu_t[fluid]))",
        "    return float(np.median(nu_t[fluid]))",
    ),
    "the first iteration's median kept (solver)": (
        "src/solver_staggered.py",
        SCALE_LINE,
        "                nu_scale = self.__dict__.setdefault("
        '"_frozen", viscosity_scale(self._mu / self._rho, state.nu_t, fluid))',
    ),
    "the inlet scale restored (solver)": (
        "src/solver_staggered.py",
        SCALE_LINE,
        "                nu_scale = self._mu / self._rho + walls.largest_inlet_eddy_viscosity()",
    ),
    "the first scale kept (rule)": (
        "src/stopping.py",
        ESTIMATE_LINE,
        "            nu_estimate = self._estimate_of("
        "self._nu_steps, self.viscosity_scale_history[0])",
    ),
    "the scale not applied (rule)": (
        "src/stopping.py",
        ESTIMATE_LINE,
        "            nu_estimate = self._estimate_of(self._nu_steps, 1.0)",
    ),
    "version 4 with (e)": (
        "src/stopping.py",
        "RULE_VERSION_WITH_E = 5",
        "RULE_VERSION_WITH_E = 4",
    ),
    "the per-step scale unchecked": (
        "src/stopping.py",
        '            _check_positive("nu_scale", nu_scale)',
        "            pass",
    ),
    "a missing scale accepted": (
        "src/stopping.py",
        "        if (viscosity_step is None) == on or (nu_scale is None) == on:",
        "        if (viscosity_step is None) == on:",
    ),
    "a truthy flag accepted": (
        "src/stopping.py",
        "        if not isinstance(viscosity_condition, bool):",
        "        if False:",
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
