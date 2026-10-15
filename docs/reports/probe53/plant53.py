"""The planted defects of prompt 53, commit A: each is applied, the tests run, and the source restored.

Usage:
    python docs/reports/probe53/plant53.py --root TREE [NAME ...]

TREE is a worktree of commit A made for this purpose (git worktree add
--detach results/builder53/plants A), never the checkout that measures: each
plant replaces exact text in one source file of TREE, runs the test files
that guard it there, records whether any test failed and which, and
restores the file's bytes before the next plant, also on an error. A plant
whose text is not found exactly once is an error, not a pass. The result
goes to results/builder53/plants.json of the main checkout.
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
OUT = MAIN / "results" / "builder53"
PYTHON = MAIN / ".venv" / "Scripts" / "python"
MULTIGRID_TESTS = "tests/test_multigrid.py"
CONFIG_TESTS = "tests/test_config.py"
PRESSURE_TESTS = "tests/test_pressure.py"

# name: (file, [(old text, new text)], test files, what the plant stands for)
PLANTS: dict[str, tuple[str, list[tuple[str, str]], list[str], str]] = {
    "inside_coupling_once": (
        "src/multigrid.py",
        [
            (
                "a_p=_rows(_columns(a_p)) - 2.0 * inside,",
                "a_p=_rows(_columns(a_p)) - inside,",
            )
        ],
        [MULTIGRID_TESTS],
        "a coupling inside a group subtracted from the coarse diagonal once, not twice",
    ),
    "inside_coupling_kept": (
        "src/multigrid.py",
        [("a_p=_rows(_columns(a_p)) - 2.0 * inside,", "a_p=_rows(_columns(a_p)),")],
        [MULTIGRID_TESTS],
        "a coupling inside a group not subtracted from the coarse diagonal",
    ),
    "east_face_from_inside": (
        "src/multigrid.py",
        [("a_e=_rows(a_e[:, 1::2]),", "a_e=_rows(a_e[:, 0::2]),")],
        [MULTIGRID_TESTS],
        "the coarse east coefficient summed from the couplings inside the group",
    ),
    "no_post_smoothing": (
        "src/multigrid.py",
        [
            (
                "        add_prolonged(x, coarse)\n        for _ in range(SMOOTHING_SWEEPS):\n",
                "        add_prolonged(x, coarse)\n        for _ in range(0):\n",
            )
        ],
        [MULTIGRID_TESTS],
        "post-smoothing omitted",
    ),
    "weight_beyond_contraction": (
        "src/multigrid.py",
        [("SMOOTHER_FACTOR = 4.0 / 3.0", "SMOOTHER_FACTOR = 3.0")],
        [MULTIGRID_TESTS],
        "a smoother weight of 1.5, past the energy-norm contraction limit 2 / lambda_max",
    ),
    "no_closed_projection": (
        "src/pressure.py",
        [
            (
                "    if not project:\n        return hierarchy.cycle\n",
                "    return hierarchy.cycle\n",
            )
        ],
        [MULTIGRID_TESTS],
        "the closed domain's preconditioned vector not projected onto the range",
    ),
    "default_multigrid": (
        "src/config.py",
        [
            (
                "        self.pressure_preconditioner: str = PRESSURE_JACOBI\n",
                "        self.pressure_preconditioner: str = MULTIGRID\n",
            )
        ],
        [CONFIG_TESTS, MULTIGRID_TESTS],
        "the default preconditioner multigrid",
    ),
    "jacobi_divides": (
        "src/pressure.py",
        [
            (
                "    return lambda r: inverse_diagonal * r\n",
                "    return lambda r: np.where(active, r / np.where(active, coefficients.a_p, 1.0), 0.0)\n",
            )
        ],
        [MULTIGRID_TESTS, PRESSURE_TESTS],
        "the diagonal preconditioner as a division, the same values to rounding but not to the bit",
    ),
}


def run_plant(root: Path, name: str) -> dict:
    """Apply one plant in TREE, run its tests there, restore, and report."""
    path_name, edits, tests, meaning = PLANTS[name]
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
        "meaning": meaning,
        "file": path_name,
        "tests": tests,
        "caught": run.returncode != 0,
        "failures": failed,
        "summary": last,
    }


def main() -> None:
    """Run the named plants, or every one, in TREE and write the record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
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
    for name in args.names or list(PLANTS):
        outcome = run_plant(root, name)
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
    (OUT / "plants.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
