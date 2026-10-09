"""Plant each defect of prompt 43 in a scratch worktree and list the tests that fail.

Usage:
    python plant43.py ROOT PYTHON

ROOT is a scratch git worktree of the branch (the script edits files under
its src/ and puts them back byte for byte); PYTHON is the interpreter that
runs pytest there. The output is JSON: the failing tests of the step's test
files with no defect (the control) and under each planted defect. A defect
whose code is not in the worktree's commit (the obstacle stencil before
commit B1, its QUICK form before B2) is reported as not applicable.
"""

import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
python = sys.argv[2]

PLANTS = {
    "arithmetic mean in place of harmonic across the row boundary": (
        "src/momentum.py",
        "        rows[1:-1, :] = _harmonic(lo_eff, hi_eff, frac_lo)",
        "        rows[1:-1, :] = _lerp(lo_eff, hi_eff, 1.0 - frac_lo)",
    ),
    "stress source sign": (
        "src/momentum.py",
        "            b_deferred = b_deferred + self._stress_source(",
        "            b_deferred = b_deferred - self._stress_source(",
    ),
    "sweep count ignored": (
        "src/momentum.py",
        "        if self._n_sweeps == 1:\n            return self._sweep(phi, c, b_pressure)",
        "        return self._sweep(phi, c, b_pressure)",
    ),
    "full distance in place of the half in the obstacle stencil": (
        "src/momentum.py",
        "mu_wall * o.ds_face[None, :] / walls.distance",
        "mu_wall * o.ds_face[None, :] / o.dt_face[:, None]",
    ),
    "QUICK at obstacles reverted to the stored SOLID-row value": (
        "src/momentum.py",
        "        obstacles = bool(o.solid.any())",
        "        obstacles = False",
    ),
}
TESTS = [
    "tests/test_momentum.py",
    "tests/test_viscosity_channel.py",
    "tests/test_config.py",
    "tests/test_solver_staggered.py",
]


def run() -> tuple[list[str], str]:
    """Run the step's test files; return the failing test ids and the summary line."""
    out = subprocess.run(
        [
            python,
            "-m",
            "pytest",
            *TESTS,
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
    ).stdout
    failed = [
        line.split(" ")[1] for line in out.splitlines() if line.startswith("FAILED")
    ]
    summary = out.strip().splitlines()[-1]
    return failed, summary


results = {}
base_failed, base_summary = run()
results["control (no defect)"] = {"failed": base_failed, "summary": base_summary}
for label, (rel, old, new) in PLANTS.items():
    path = root / rel
    original = path.read_bytes()
    text = original.decode().replace("\r\n", "\n")
    if text.count(old) != 1:
        results[label] = {"failed": [], "summary": "not applicable at this commit"}
        continue
    try:
        path.write_bytes(text.replace(old, new).encode())
        failed, summary = run()
    finally:
        path.write_bytes(original)
    results[label] = {"failed": failed, "summary": summary}
print(json.dumps(results, indent=1))
