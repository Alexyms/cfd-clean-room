"""Plant each defect of prompt 45 in a scratch worktree and list the tests that fail.

Usage:
    python plant45.py ROOT PYTHON

ROOT is a scratch git worktree of the branch (the script edits files under
its src/ and puts them back byte for byte); PYTHON is the interpreter that
runs pytest there. The output is JSON: the failing tests of the step's test
files with no defect (the control) and under each planted defect. A defect
whose code is not in the worktree's commit is reported as not applicable. A
plant counts as caught only when pytest exits 1 (tests ran and failed); exit
4 or 5 is a missing test file, not a kill.
"""

import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
python = sys.argv[2]

PLANTS = {
    # Commit A
    "y*_0 typed as 11.06": (
        "src/turbulence.py",
        "Y_STAR_FLOOR = log_law_floor(KAPPA, E_WALL)",
        "Y_STAR_FLOOR = 11.06",
    ),
    "an inlet face given a wall function": (
        "src/boundary_staggered.py",
        "            condition.bc_type == VELOCITY_INLET\n"
        "            and StaggeredBoundary._normal_component(condition, edge) == 0.0",
        "            condition.bc_type == VELOCITY_INLET",
    ),
    "corner cell: one wall's contribution dropped": (
        "src/turbulence.py",
        "        for mask, distance, wall_speed, component in self._sides:",
        "        for mask, distance, wall_speed, component in self._sides[:3]:",
    ),
    "inflow eps in the other convention, C_mu^(3/4)": (
        "src/turbulence.py",
        "    return k, k**1.5 / dissipation_length",
        "    return k, 0.09**0.75 * k**1.5 / dissipation_length",
    ),
    "refusal: velocity_step with the model on accepted": (
        "src/config.py",
        "            if self.stopping_rule != ERROR_ESTIMATE:",
        "            if False:",
    ),
    "refusal: an admitting inlet without its keys accepted": (
        "src/config.py",
        "        for key in keys:\n            if key not in spec:\n                raise ValueError(",
        "        for key in keys:\n            if False:\n                raise ValueError(",
    ),
    "refusal: the keys with the model off accepted": (
        "src/config.py",
        "            if self.turbulence is None:\n                raise ValueError(\n"
        '                    f"{ctx}.{key} is read only',
        "            if False:\n                raise ValueError(\n"
        '                    f"{ctx}.{key} is read only',
    ),
    "refusal: the keys on a zero-normal inlet accepted": (
        "src/config.py",
        "            if not admits_air:\n                raise ValueError(",
        "            if False:\n                raise ValueError(",
    ),
    # Commit B
    "relaxation: alpha_turbulence ignored": (
        "src/solver_staggered.py",
        "        nu_t = (1.0 - a_t) * state.nu_t + a_t * stepped.nu_t",
        "        nu_t = np.array(stepped.nu_t)",
    ),
    "refusal: eddy_viscosity with the model on accepted": (
        "src/solver_staggered.py",
        "        if model is not None and eddy_viscosity is not None:",
        "        if False:",
    ),
    # Commit C
    "condition (e): nu_scale from the field's maximum": (
        "src/solver_staggered.py",
        "        nu_scale = self._mu / self._rho + self._walls.largest_inlet_eddy_viscosity()",
        "        nu_scale = self._mu / self._rho + float(\n"
        "            self._model.initial(*self._initial_turbulence).nu_t.max()\n"
        "        )",
    ),
    "condition (e): not required for the stop": (
        "src/stopping.py",
        "            viscosity_ok = nu_estimate < self._error_tol",
        "            viscosity_ok = True",
    ),
}
TESTS = [
    "tests/test_wall_functions.py",
    "tests/test_config.py",
    "tests/test_coupled_solve.py",
    "tests/test_momentum.py",
    "tests/test_stopping.py",
]


def run() -> tuple[int, list[str], str]:
    """Run the step's test files; return pytest's exit code, the failing ids and the summary."""
    proc = subprocess.run(
        [
            python,
            "-m",
            "pytest",
            *[t for t in TESTS if (root / t).exists()],
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "-rf",
            "--tb=no",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    out = proc.stdout
    failed = [
        line.split(" ")[1] for line in out.splitlines() if line.startswith("FAILED")
    ]
    lines = out.strip().splitlines()
    return proc.returncode, failed, lines[-1] if lines else ""


def main() -> None:
    record: dict[str, dict] = {}
    code, failed, summary = run()
    record["control"] = {"exit": code, "failed": failed, "summary": summary}
    for name, (rel, old, new) in PLANTS.items():
        path = root / rel
        original = path.read_bytes()
        text = original.decode("utf-8")
        nl = "\r\n" if "\r\n" in text else "\n"
        old_n, new_n = old.replace("\n", nl), new.replace("\n", nl)
        count = text.count(old_n)
        if count == 0:
            record[name] = {"applicable": False}
            continue
        if count != 1:
            raise SystemExit(f"{name}: {count} matches in {rel}")
        try:
            path.write_bytes(text.replace(old_n, new_n).encode("utf-8"))
            code, failed, summary = run()
        finally:
            path.write_bytes(original)
        record[name] = {
            "applicable": True,
            "caught": code == 1,
            "exit": code,
            "failed": failed,
            "summary": summary,
        }
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
