"""The planted defects of prompt 54, commits A and B: each is applied, the tests run, and the source restored.

Usage:
    python docs/reports/probe54/plant54.py --root TREE --group A|B [NAME ...]

TREE is a worktree of the group's commit made for this purpose (git worktree
add --detach results/builder54/plants COMMIT), never the checkout that
measures. Group A is commit A (the fixed smoother weight), group B commit B
(the default and the harness labels). Each plant replaces exact text in one
source file of TREE, runs the test files that guard it there, records whether
any test failed and which, and restores the file's bytes before the next
plant, also on an error. A plant whose text is not found exactly once is an
error, not a pass. The result goes to results/builder54/plants_GROUP.json of
the main checkout.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]
MAIN = Path(
    subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
).parent
OUT = MAIN / "results" / "builder54"
PYTHON = MAIN / ".venv" / "Scripts" / "python"
MULTIGRID_TESTS = "tests/test_multigrid.py"
CONFIG_TESTS = "tests/test_config.py"
PRESSURE_TESTS = "tests/test_pressure.py"
BENCHMARK_TESTS = "tests/test_benchmark.py"
SELECTION_TESTS = "tests/test_solver_selection.py"

# The weight as prompt 53 formed it: three power iterations from the
# checkerboard, in place of the fixed weight.
OLD_ESTIMATE = """        weight = 0.0
        if smoothed:
            ny, nx = c.a_p.shape
            x = np.where(
                (np.arange(ny)[:, None] + np.arange(nx)[None, :]) % 2 == 0, 1.0, -1.0
            )
            x[inverse == 0.0] = 0.0
            quotient = 0.0
            for _ in range(3):
                ax = self._apply(c, x)
                mass = float(np.vdot(x, c.a_p * x))
                if mass == 0.0:
                    break
                quotient = float(np.vdot(x, ax)) / mass
                y = inverse * ax
                scale = float(np.abs(y).max())
                if scale == 0.0:
                    break
                x = y / scale
            weight = SMOOTHER_FACTOR / quotient if quotient > 0.0 else 0.0
"""
FIXED_WEIGHT = "        weight = SMOOTHER_FACTOR / 2.0 if smoothed else 0.0\n"

# group: name: (file, [(old text, new text)], test files, what the plant stands for)
PLANTS: dict[str, dict[str, tuple[str, list[tuple[str, str]], list[str], str]]] = {
    "A": {
        "old_estimate_restored": (
            "src/multigrid.py",
            [(FIXED_WEIGHT, OLD_ESTIMATE)],
            [MULTIGRID_TESTS],
            "the three-power-iteration weight restored, on levels whose estimate differs from 2",
        ),
        "presmoothing_dropped": (
            "src/multigrid.py",
            [
                (
                    "        for _ in range(SMOOTHING_SWEEPS - 1):\n",
                    "        for _ in range(0):\n",
                )
            ],
            [MULTIGRID_TESTS],
            "the extra pre-smoothing sweeps of V(nu, nu) dropped, so nu above 1 is unbalanced",
        ),
        "dominance_unchecked": (
            "src/multigrid.py",
            [
                (
                    "        if np.any(c.a_p * (1.0 + DOMINANCE_TOLERANCE) < couplings):\n",
                    "        if False:\n",
                )
            ],
            [MULTIGRID_TESTS],
            "the finest system's diagonal dominance not checked",
        ),
        "dominance_tolerance_wide": (
            "src/multigrid.py",
            [("DOMINANCE_TOLERANCE = 1.0e-12", "DOMINANCE_TOLERANCE = 0.5")],
            [MULTIGRID_TESTS],
            "the dominance check's rounding allowance widened to 50%",
        ),
        "coarsest_smoothed": (
            "src/multigrid.py",
            [(FIXED_WEIGHT, "        weight = SMOOTHER_FACTOR / 2.0\n")],
            [MULTIGRID_TESTS],
            "the coarsest level given a weight although it is solved exactly",
        ),
        "weight_beyond_contraction": (
            "src/multigrid.py",
            [("SMOOTHER_FACTOR = 4.0 / 3.0", "SMOOTHER_FACTOR = 3.0")],
            [MULTIGRID_TESTS],
            "a smoother weight of 1.5, past the energy-norm contraction limit 2 / lambda_max",
        ),
    },
    "B": {
        "default_left_jacobi": (
            "src/config.py",
            [
                (
                    "DEFAULT_PRESSURE_PRECONDITIONER = MULTIGRID",
                    "DEFAULT_PRESSURE_PRECONDITIONER = PRESSURE_JACOBI",
                )
            ],
            [CONFIG_TESTS, MULTIGRID_TESTS],
            "the default constant left at the diagonal",
        ),
        "default_read_as_jacobi": (
            "src/config.py",
            [
                (
                    "        self.pressure_preconditioner: str = DEFAULT_PRESSURE_PRECONDITIONER\n",
                    "        self.pressure_preconditioner: str = PRESSURE_JACOBI\n",
                )
            ],
            [CONFIG_TESTS, MULTIGRID_TESTS],
            "the default constant changed but the configuration still reading the diagonal",
        ),
        "label_ignored": (
            "scripts/benchmark.py",
            [
                (
                    "    config = with_pressure_preconditioner(\n"
                    "        load_preset(case_id), METHOD_PRECONDITIONERS[method]\n"
                    "    )\n",
                    "    config = load_preset(case_id)\n",
                )
            ],
            [BENCHMARK_TESTS],
            "run_case leaving the case file's preconditioner, so a label can claim a solver it did not run",
        ),
        "diagonal_label_runs_multigrid": (
            "scripts/benchmark.py",
            [
                (
                    "    CG_JACOBI_METHOD: PRESSURE_JACOBI,\n}",
                    "    CG_JACOBI_METHOD: MULTIGRID,\n}",
                )
            ],
            [BENCHMARK_TESTS],
            "the staggered-cg label mapped to the multigrid preconditioner",
        ),
        "labels_pooled": (
            "src/pressure.py",
            [('    3: "staggered-mgcg",\n', '    3: "staggered-cg",\n')],
            [PRESSURE_TESTS, BENCHMARK_TESTS, SELECTION_TESTS],
            "both preconditioners filed under one label, so their rows would summarize together",
        ),
        "version_not_raised": (
            "src/pressure.py",
            [("PRESSURE_SOLVER_VERSION = 3", "PRESSURE_SOLVER_VERSION = 2")],
            [PRESSURE_TESTS, BENCHMARK_TESTS, SELECTION_TESTS],
            "the version left at 2, so a saved diagonal-preconditioned history is read as current",
        ),
        "work_definition_unchanged": (
            "scripts/benchmark.py",
            [
                (
                    "and neither are the multigrid V-cycle's own products and transfers, ",
                    "and neither are the V-cycle's own products and transfers, ",
                )
            ],
            [BENCHMARK_TESTS],
            "the multigrid row's work definition not saying what it leaves out",
        ),
        "helper_unvalidated": (
            "validation/cases.py",
            [
                (
                    "    if preconditioner not in PRESSURE_PRECONDITIONERS:\n",
                    "    if False:\n",
                )
            ],
            [BENCHMARK_TESTS],
            "with_pressure_preconditioner accepting any name",
        ),
    },
}


def run_plant(root: Path, group: str, name: str) -> dict:
    """Apply one plant in TREE, run its tests there, restore, and report."""
    path_name, edits, tests, meaning = PLANTS[group][name]
    path = root / path_name
    original = path.read_bytes()
    text = original.decode()
    for old, new in edits:
        for variant in (old, old.replace("\n", "\r\n")):
            if text.count(variant) == 1:
                text = text.replace(
                    variant, new if variant == old else new.replace("\n", "\r\n")
                )
                break
        else:
            raise SystemExit(
                f"{name}: text not found exactly once in {path_name}: {old!r}"
            )
    try:
        path.write_bytes(text.encode())
        run = subprocess.run(
            [str(PYTHON), "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"],
            cwd=root,
            capture_output=True,
            text=True,
        )
    finally:
        path.write_bytes(original)
    failed = [
        line.split(" ", 1)[1].split(" - ")[0]
        for line in run.stdout.splitlines()
        if line.startswith("FAILED ") or line.startswith("ERROR ")
    ]
    last = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
    return {
        "plant": name,
        "group": group,
        "meaning": meaning,
        "file": path_name,
        "tests": tests,
        "caught": run.returncode != 0,
        "failures": failed,
        "summary": last,
    }


def main() -> None:
    """Run the named plants of one group, or every one, in TREE and write the record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--group", required=True, choices=sorted(PLANTS))
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if root == MAIN.resolve():
        raise SystemExit(
            "plants edit the tree they run in; give a worktree made for them"
        )
    clean = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if clean:
        raise SystemExit(f"{root} is not clean:\n{clean}")
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    results = []
    for name in args.names or list(PLANTS[args.group]):
        outcome = run_plant(root, args.group, name)
        outcome["commit"] = head
        print(
            f"{name}: caught {outcome['caught']} {outcome['failures'][:4]}", flush=True
        )
        results.append(outcome)
    after = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if after:
        sys.exit(f"{root} was not restored:\n{after}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"plants_{args.group}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
