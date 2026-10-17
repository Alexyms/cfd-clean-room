"""Prompt 61: the planted defects. Each guard step 0b added, removed, and the test that fails.

Usage:
    python plants61.py --tree TREE

TREE is a worktree of the branch (results/builder61/P), never the tree the
runs read. For each plant the file is edited in TREE by one exact string
replacement, the named tests run there, and the file is restored from git
before the next plant; a control run of the same tests on the unplanted file
must pass first, so a failure is the plant's. A plant is caught when at least
one named test fails. The record is results/builder61/plants.json of the main
checkout.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

CONFIG = "tests/test_config.py::TestVents"
STAG = "tests/test_boundary_staggered.py::TestVents"
MOM = "tests/test_momentum.py::TestVents"
SOLVE = "tests/test_solver_staggered.py::TestVents"
WALLS = "tests/test_wall_functions.py::TestVents"
NO_DIFFUSION = (
    "tests/test_wall_functions.py::"
    "test_no_diffusion_crosses_a_vent_face_into_the_obstacle"
)
SCHEME = "tests/test_scalar_scheme.py::test_a_vent_face_carries_the_inflow_in_and_the_room_cell_out"
COUPLED = (
    "tests/test_coupled_solve.py::"
    "test_the_coupled_solve_converges_with_a_discharge_an_intake_and_an_exhaust"
)
CONC = "tests/test_boundary_concentration.py::TestVents"
TRANS = "tests/test_solver_transport.py::TestVents"

# (name, file, old, new, tests)
PLANTS: list[tuple[str, str, str, str, list[str]]] = [
    (
        "config: vents not checked against the obstacles",
        "src/config.py",
        "        self._check_vents()\n",
        "",
        [CONFIG],
    ),
    (
        "config: a face on the domain edge accepted",
        "src/config.py",
        "            if self._face_on_edge(obs, spec.face):",
        "            if False:",
        [CONFIG],
    ),
    (
        "config: an obstacle against the face accepted",
        "src/config.py",
        "                if other is not obs and self._stands_against(obs, spec, other):",
        "                if False:",
        [CONFIG],
    ),
    (
        "config: overlap read by edge only, not by obstacle face",
        "src/config.py",
        "if (b.location, b.obstacle, b.face) != (a.location, a.obstacle, a.face):",
        "if b.location != a.location:",
        [CONFIG],
    ),
    (
        "config: a vent without a velocity accepted",
        "src/config.py",
        '        if "velocity" not in spec:\n            raise ValueError(\n                f"Missing required key: \'{ctx}.velocity\'. A vent states its flow: "',
        "        if False:\n            raise ValueError(\n                f\"Missing required key: '{ctx}.velocity'. A vent states its flow: \"",
        [CONFIG],
    ),
    (
        "config: velocity components accepted on a vent",
        "src/config.py",
        '        for key in ("u_velocity", "v_velocity"):\n            if key in spec:\n                raise ValueError(\n                    f"{ctx}.{key} is not valid on a vent',
        '        for key in ():\n            if key in spec:\n                raise ValueError(\n                    f"{ctx}.{key} is not valid on a vent',
        [CONFIG],
    ),
    (
        "config: stated outlets accepted unbalanced",
        "src/config.py",
        "            if abs(inflow - outflow) > BALANCE_TOLERANCE * max(abs(inflow), outflow):",
        "            if False:",
        [CONFIG, "tests/test_fixed_flow_outlet.py::TestConfiguration"],
    ),
    (
        "config: recirculate_from not checked",
        "src/config.py",
        "            if spec.recirculate_from is not None:\n                self._check_recirculation(name, spec)",
        "            pass",
        [CONFIG],
    ),
    (
        "config: recirculate_from accepted beside a concentration",
        "src/config.py",
        '            if "concentration" in spec:\n                raise ValueError(\n                    f"{ctx}.concentration is not valid with recirculate_from',
        '            if False:\n                raise ValueError(\n                    f"{ctx}.concentration is not valid with recirculate_from',
        [CONFIG],
    ),
    (
        "velocity: vent faces not written",
        "src/boundary_staggered.py",
        "            target[run.rows, run.cols] = run.normal\n",
        "            pass\n",
        [STAG],
    ),
    (
        "velocity: a discharge left out of the inflow",
        "src/boundary_staggered.py",
        "            return self._vent_inlet_flux(boundary_name)",
        "            return 0.0",
        [STAG],
    ),
    (
        "velocity: an exhaust left out of the remainder rule",
        "src/boundary_staggered.py",
        "                lengths[name] = vent_widths[name]\n",
        "                lengths[name] = 0.0\n",
        [STAG],
    ),
    (
        "velocity: the mesh balance of stated outlets not checked",
        "src/boundary_staggered.py",
        "            self._check_stated_balance(resolved, lengths)\n",
        "",
        [STAG],
    ),
    (
        "velocity: a vent covering no face accepted",
        "src/boundary_staggered.py",
        "        if not mine.any():\n            raise ValueError(",
        "        if False:\n            raise ValueError(",
        [STAG],
    ),
    (
        "velocity: a discharge not in the velocity scale",
        "src/boundary_staggered.py",
        "            if run.is_inlet:\n                max_vel = max(max_vel, abs(run.normal))",
        "            if False:\n                max_vel = max(max_vel, abs(run.normal))",
        [STAG],
    ),
    (
        "momentum: vent faces predicted as zero",
        "src/momentum.py",
        "        if vents.any:\n            self._held = {",
        "        if False:\n            self._held = {",
        [MOM, SOLVE],
    ),
    (
        "turbulence: the vent span left a wall-function face",
        "src/turbulence.py",
        "    if vent is not None:\n        south &= ~vent[:-1, :]",
        "    if False:\n        south &= ~vent[:-1, :]",
        [WALLS],
    ),
    (
        "turbulence: a vent face left a wall side of its cell",
        "src/turbulence.py",
        "        south[1:, :] = solid[:-1, :] & ~vents.normal_v[1:-1, :]",
        "        south[1:, :] = solid[:-1, :]",
        [WALLS],
    ),
    (
        "turbulence: a discharge carries no inflow k and eps",
        "src/turbulence.py",
        "            if run.is_inlet:\n                target[run.rows, run.cols] = getattr(",
        "            if False:\n                target[run.rows, run.cols] = getattr(",
        [WALLS, COUPLED],
    ),
    (
        "turbulence: the model does not read the vent faces",
        "src/turbulence.py",
        "        if vents.any:\n            live_u = live_u | vents.normal_u",
        "        if False:\n            live_u = live_u | vents.normal_u",
        [WALLS, COUPLED],
    ),
    (
        "turbulence: diffusion crosses the vent faces",
        "src/turbulence.py",
        "        # Diffusion crosses interior faces between two non-SOLID cells only.\n        self._inner_u = np.zeros(self._u_shape, dtype=bool)\n        self._inner_u[:, 1:-1] = live_u[:, 1:-1]",
        "        # Diffusion crosses interior faces between two non-SOLID cells only.\n        self._inner_u = np.zeros(self._u_shape, dtype=bool)\n        self._inner_u[:, 1:-1] = (live_u | vent_face_masks(mesh, vent_runs(mesh, config, BoundaryRegistry(config))).normal_u)[:, 1:-1]",
        [WALLS, COUPLED, NO_DIFFUSION],
    ),
    (
        "scheme: the vents argument ignored",
        "src/scalar_scheme.py",
        "        if vents is not None:\n            k = axis.left",
        "        if False:\n            k = axis.left",
        [SCHEME, TRANS],
    ),
    (
        "concentration: vent faces deposit",
        "src/boundary_concentration.py",
        "            surface_u[masks.normal_u] = SURFACE_NONE\n            surface_v[masks.normal_v] = SURFACE_NONE\n",
        "",
        [CONC, TRANS],
    ),
    (
        "concentration: a discharge carries no concentration",
        "src/boundary_concentration.py",
        "            if run.is_inlet and run.spec.recirculate_from is None:",
        "            if False:",
        [CONC, TRANS],
    ),
    (
        "transport: vent faces not booked in the budget",
        "src/solver_transport.py",
        "            inflow += vent_in\n            outflow += vent_out\n",
        "",
        [TRANS],
    ),
    (
        "transport: a recirculating discharge carries nothing",
        "src/solver_transport.py",
        "        if not conditions.recirculation:\n            return conditions.inflow_u, conditions.inflow_v",
        "        if True:\n            return conditions.inflow_u, conditions.inflow_v",
        [TRANS],
    ),
    (
        "transport: a recirculation unfiltered",
        "src/solver_transport.py",
        "            inflow_u[pair.discharge_u] = pair.passed * carried\n            inflow_v[pair.discharge_v] = pair.passed * carried",
        "            inflow_u[pair.discharge_u] = carried\n            inflow_v[pair.discharge_v] = carried",
        [TRANS],
    ),
    (
        "transport: the vents not opened to the advection",
        "src/solver_transport.py",
        "            self._live_u = live_u | self._vent_u\n            self._live_v = live_v | self._vent_v",
        "            pass",
        [TRANS],
    ),
    (
        "transport: diffusion crosses the vent faces",
        "src/solver_transport.py",
        "            self._live_u = live_u | self._vent_u\n            self._live_v = live_v | self._vent_v",
        "            self._live_u = live_u | self._vent_u\n            self._live_v = live_v | self._vent_v\n            self._conductance_u = np.where(inner_u | self._vent_u, self._area_u / mesh.dx_face[None, :], 0.0)",
        [TRANS],
    ),
]


def run_tests(tree: Path, tests: list[str]) -> tuple[bool, str]:
    """Run the tests in TREE; True when every one passed, and the summary line."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests],
        cwd=tree,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return result.returncode == 0, lines[-1] if lines else result.stderr[-300:]


def main() -> None:
    """Every plant, its control, its tests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    args = parser.parse_args()
    tree = Path(args.tree).resolve()
    records = []
    for name, rel, old, new, tests in PLANTS:
        path = tree / rel
        control_ok, control_line = run_tests(tree, tests)
        text = path.read_bytes().decode("utf-8")
        crlf = "\r\n" in text
        old_n = old.replace("\n", "\r\n") if crlf else old
        new_n = new.replace("\n", "\r\n") if crlf else new
        count = text.count(old_n)
        if count != 1:
            records.append(
                {"plant": name, "file": rel, "error": f"anchor found {count}"}
            )
            print(f"ANCHOR {count}: {name}", flush=True)
            continue
        path.write_bytes(text.replace(old_n, new_n).encode("utf-8"))
        try:
            planted_ok, planted_line = run_tests(tree, tests)
        finally:
            subprocess.run(["git", "checkout", "--", rel], cwd=tree, check=True)
        caught = control_ok and not planted_ok
        records.append(
            {
                "plant": name,
                "file": rel,
                "tests": tests,
                "control_passed": control_ok,
                "control": control_line,
                "planted": planted_line,
                "caught": caught,
            }
        )
        print(
            f"{'CAUGHT' if caught else 'SURVIVED'}: {name} -- {planted_line}",
            flush=True,
        )
    out = ROOT / "results" / "builder61" / "plants.json"
    out.write_text(json.dumps(records, indent=1))
    print(f"{sum(r.get('caught', False) for r in records)} of {len(records)} caught")


if __name__ == "__main__":
    main()
