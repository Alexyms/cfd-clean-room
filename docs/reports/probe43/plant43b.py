"""Plant the obstacle-path mutants of review 43 and test 43 and list the tests that fail.

Usage:
    python plant43b.py ROOT PYTHON

ROOT is a scratch git worktree of the branch (the script edits src/momentum.py
there and puts it back byte for byte); PYTHON is the interpreter that runs
pytest there. The test files are the three review 43 named. The mutants:
review 43 B2's three (a) to (c), test 43's M12, two more on the high side
of the obstacle stencil and two on the low side. Output is JSON: the failing tests with no
mutant (the control) and under each.
"""

import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
python = sys.argv[2]
TARGET = "src/momentum.py"

PLANTS = {
    "review (a): blocked_neg never set": (
        "        blocked_neg[:, :-1] = o.solid[:, 1:]",
        "        blocked_neg[:, :-1] = False",
    ),
    "review (b): below and above swapped": (
        "    below = (t_faces[1:] - t_centers)[:, None]\n"
        "    above = (t_centers - t_faces[:-1])[:, None]",
        "    above = (t_faces[1:] - t_centers)[:, None]\n"
        "    below = (t_centers - t_faces[:-1])[:, None]",
    ),
    "review (c): wall_neg's far node at t_faces[r]": (
        "            o.t_faces[r + 1][:, None],",
        "            o.t_faces[r][:, None],",
    ),
    "test M12: the obstacle face's QUICK correction not zeroed": (
        "            dq_t[o.walls.face] = 0.0",
        "            pass",
    ),
    "high side: the north wall at the full distance": (
        "    below = (t_faces[1:] - t_centers)[:, None]",
        "    below = 2.0 * (t_faces[1:] - t_centers)[:, None]",
    ),
    "high side: wall_neg takes the stored value as far node": (
        "        return np.where(~pos_t & north[r, :], wall_neg, q_t)",
        "        return q_t",
    ),
    "low side: blocked_pos never set": (
        "        blocked_pos[:, 1:] = o.solid[:, :-1]",
        "        blocked_pos[:, 1:] = False",
    ),
    "low side: wall_pos takes the stored value as far node": (
        "        q_t = np.where(pos_t & south[r - 1, :], wall_pos, q_t)",
        "        q_t = q_t",
    ),
}
TESTS = [
    "tests/test_momentum.py",
    "tests/test_solver_staggered.py",
    "tests/test_viscosity_channel.py",
]


def run() -> tuple[list[str], str]:
    """Run the test files; return the failing test ids and the summary line."""
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
    return failed, out.strip().splitlines()[-1]


results = {}
failed, summary = run()
results["control (no mutant)"] = {"failed": failed, "summary": summary}
path = root / TARGET
original = path.read_bytes()
for label, (old, new) in PLANTS.items():
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
