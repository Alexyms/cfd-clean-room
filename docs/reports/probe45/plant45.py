"""Plant each defect of prompt 45 in a scratch worktree and list the tests that fail.

Usage:
    python plant45.py ROOT PYTHON [VAL016_CAP]

ROOT is a scratch git worktree of the branch (the script edits files under
its src/ and puts them back byte for byte); PYTHON is the interpreter that
runs pytest there. The output is JSON: the failing tests of the step's test
files with no defect (the control) and under each planted defect. A defect
whose code is not in the worktree's commit is reported as not applicable. A
plant counts as caught only when pytest exits 1 (tests ran and failed); exit
4 or 5 is a missing test file, not a kill.

VAL016_CAP, when given, lowers VAL-016's max_simple_iter in ROOT's copy of
tests/couette_reference.py (left lowered: ROOT is scratch), so a wall-function
plant that stops the Couette solve converging fails in minutes instead of
running to 20,000 outer iterations (prompt 45b).
"""

import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
python = sys.argv[2]
if len(sys.argv) > 3:
    reference = root / "tests" / "couette_reference.py"
    text = reference.read_text(encoding="utf-8")
    if text.count('"max_simple_iter": 20000') != 1:
        raise SystemExit("VAL-016's cap is not where plant45.py expects it")
    reference.write_text(
        text.replace('"max_simple_iter": 20000', f'"max_simple_iter": {sys.argv[3]}'),
        encoding="utf-8",
    )

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
    # Prompt 45b (test 45 B1 and B2, review 45 B1)
    "condition (e)'s step scaled by 1e-3": (
        "src/solver_staggered.py",
        "np.max(np.abs(state.nu_t[self._live] - nu_t_old[self._live]))",
        "np.max(np.abs(state.nu_t[self._live] - nu_t_old[self._live])) * 1e-3",
    ),
    "wall viscosity: k_P from one cell": (
        "src/turbulence.py",
        "                + frame[walls.frame_t, walls.frame_s]\n",
        "                + frame[walls.frame_t, walls.frame_s - 1]\n",
    ),
    "wall viscosity: y_P the full cell (south faces)": (
        "src/turbulence.py",
        "(t_centers[t_s] - t_faces[t_s], t_faces[t_n + 1] - t_centers[t_n])",
        "(t_faces[t_s + 1] - t_faces[t_s], t_faces[t_n + 1] - t_centers[t_n])",
    ),
    "wall-cell production without the y*_0 floor": (
        "src/turbulence.py",
        "        log_term = np.log(E_WALL * np.maximum(y_star, Y_STAR_FLOOR))\n"
        "        return u_k**3",
        "        log_term = np.log(E_WALL * y_star)\n        return u_k**3",
    ),
    "moving walls: bottom and top swapped": (
        "src/turbulence.py",
        '        moving["s"][0, :] = mean["bottom"]\n        moving["n"][-1, :] = mean["top"]',
        '        moving["s"][0, :] = mean["top"]\n        moving["n"][-1, :] = mean["bottom"]',
    ),
    "moving walls: the left edge's line dropped": (
        "src/turbulence.py",
        '        moving["w"][:, 0] = mean["left"]\n',
        "",
    ),
    # Commit D
    "VAL-016 (a): the wall-cell production scaled by 1.01": (
        "src/turbulence.py",
        "        return u_k**3 / (KAPPA * y_p), u_k**2 * slip / (y_p * log_term)",
        "        return u_k**3 / (KAPPA * y_p), 1.01 * u_k**2 * slip / (y_p * log_term)",
    ),
}
TESTS = [
    "tests/test_wall_functions.py",
    "tests/test_config.py",
    "tests/test_coupled_solve.py",
    "tests/test_momentum.py",
    "tests/test_stopping.py",
    "tests/test_turbulent_channel.py",
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
