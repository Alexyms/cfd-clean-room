"""Plant each defect of prompt 42 item 6 in a scratch worktree and list the tests that fail.

Usage:
    python plant42.py ROOT PYTHON

ROOT is a scratch git worktree of the branch (the script edits files under its
src/ and puts them back byte for byte); PYTHON is the interpreter that runs
pytest there. The output is JSON: the failing tests of the two new test files
with no defect (the control) and under each planted defect.
"""

import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
python = sys.argv[2]

PLANTS = {
    "remainder over the configured length": (
        "src/boundary_staggered.py",
        "            lengths[name] = float(self._face_widths(spec.location)[covered].sum())",
        "            lengths[name] = (\n"
        "                spec.y_end - spec.y_start\n"
        "                if spec.location in ('left', 'right')\n"
        "                else spec.x_end - spec.x_start\n"
        "            )",
    ),
    "hood counted as inflow (flux)": (
        "src/boundary_staggered.py",
        "            if point.condition.bc_type != VELOCITY_INLET:\n                continue",
        "            if point.condition.bc_type not in (VELOCITY_INLET, FIXED_FLOW_OUTLET):\n"
        "                continue",
    ),
    "hood counted as inflow (total)": (
        "src/boundary_staggered.py",
        "            if spec.type == VELOCITY_INLET:\n                total += self.get_inlet_flux(name)",
        "            if spec.type in (VELOCITY_INLET, FIXED_FLOW_OUTLET):\n"
        "                total += self.get_inlet_flux(name)",
    ),
    "nonzero tangential value": (
        "src/boundary_registry.py",
        '    if edge == "top":\n        return EdgeCondition(FIXED_FLOW_OUTLET, 0.0, speed)\n'
        '    if edge == "bottom":\n        return EdgeCondition(FIXED_FLOW_OUTLET, 0.0, -speed)\n'
        '    if edge == "left":\n        return EdgeCondition(FIXED_FLOW_OUTLET, -speed, 0.0)\n'
        "    return EdgeCondition(FIXED_FLOW_OUTLET, speed, 0.0)",
        '    if edge == "top":\n        return EdgeCondition(FIXED_FLOW_OUTLET, 0.1, speed)\n'
        '    if edge == "bottom":\n        return EdgeCondition(FIXED_FLOW_OUTLET, 0.1, -speed)\n'
        '    if edge == "left":\n        return EdgeCondition(FIXED_FLOW_OUTLET, -speed, 0.1)\n'
        "    return EdgeCondition(FIXED_FLOW_OUTLET, speed, 0.1)",
    ),
    "concentration carried in": (
        "src/boundary_concentration.py",
        "            if bc_type == VELOCITY_INLET and self._normal(point, edge) != 0.0:",
        "            if (\n"
        "                bc_type == VELOCITY_INLET and self._normal(point, edge) != 0.0\n"
        "            ) or bc_type == 'fixed_flow_outlet':",
    ),
    "concentration carried in (value)": (
        "src/boundary_concentration.py",
        "        if spec.concentration is None:\n            return 0.0",
        "        if spec.type == 'fixed_flow_outlet':\n            return 1.0\n"
        "        if spec.concentration is None:\n            return 0.0",
    ),
}
# The last two together are the one defect: the outlet admits a unit concentration.
GROUPS = {
    "remainder over the configured length": ["remainder over the configured length"],
    "hood counted as inflow": [
        "hood counted as inflow (flux)",
        "hood counted as inflow (total)",
    ],
    "nonzero tangential value": ["nonzero tangential value"],
    "concentration carried in": [
        "concentration carried in",
        "concentration carried in (value)",
    ],
}
TESTS = ["tests/test_fixed_flow_outlet.py", "tests/test_fixed_flow_product.py"]


def run() -> tuple[list[str], str]:
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
for label, keys in GROUPS.items():
    originals: dict[Path, bytes] = {}
    try:
        for key in keys:
            rel, old, new = PLANTS[key]
            path = root / rel
            # The true original, once: a second plant in the same file must
            # not overwrite it with the file the first plant has edited.
            originals.setdefault(path, path.read_bytes())
            text = path.read_bytes().decode().replace("\r\n", "\n")
            assert text.count(old) == 1, (key, text.count(old))
            path.write_bytes(text.replace(old, new).replace("\n", "\r\n").encode())
        failed, summary = run()
    finally:
        for path, data in originals.items():
            path.write_bytes(data)
    results[label] = {"failed": failed, "summary": summary}
print(json.dumps(results, indent=1))
