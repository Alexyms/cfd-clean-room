"""Prompts 67 and 67b: the planted defects of step 6a, each applied to a worktree, its tests run.

Usage:
    python docs/reports/probe67/plant67.py WORKTREE [PLANT_ID ...]

WORKTREE is a scratch checkout of the commit under test (``git worktree add
--detach``), never the main checkout: the runner edits src/ there and restores
it with git checkout.

Each plant is one or more exact text replacements in src/ and the pytest
selection that should catch it. A plant is caught when pytest exits 1 (tests
failed); exit 2 or more (collection error) is BROKEN, never caught, and a
plant whose text does not match exactly once is NOT APPLIED. The worktree is
restored with git checkout after every plant and checked clean. A control
run of every selection on the clean worktree comes first and must pass.
Records: results/builder67/plants[_IDS].json in the main checkout (gitignored).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECORDS = ROOT / "results" / "builder67"
MO = "src/momentum.py"
SS = "src/solver_staggered.py"
T_MO = ("tests/test_momentum.py", "TransientTerm")
T_TR = ("tests/test_transient.py", "")
T_FP = ("tests/test_transient.py", "FixedPoint")
T_ORD = ("tests/test_time_order.py", "")
T_BD = ("tests/test_transient.py", "Boundaries")

# (id, what, [(file, old, new), ...], [(test file, -k expression), ...])
PLANTS: list[tuple[str, str, list[tuple[str, str, str]], list[tuple[str, str]]]] = [
    (
        "M1",
        "BDF2 weights perturbed, still summing to zero",
        [(MO, "BDF2_WEIGHTS = (1.5, -2.0, 0.5)", "BDF2_WEIGHTS = (1.5, -1.9, 0.4)")],
        [T_MO, T_ORD],
    ),
    (
        "M2",
        "BDF2 weights perturbed on the new level",
        [(MO, "BDF2_WEIGHTS = (1.5, -2.0, 0.5)", "BDF2_WEIGHTS = (1.6, -2.0, 0.5)")],
        [T_MO, T_FP, T_ORD],
    ),
    (
        "M3",
        "old levels swapped in the term",
        [
            (
                MO,
                "zip(weights[1:], old, strict=True)",
                "zip(weights[1:], old[::-1], strict=True)",
            )
        ],
        [T_MO, T_ORD],
    ),
    (
        "M4",
        "the transient term on the source only",
        [(MO, "a_p=c.a_p + weights[0] * rate,", "a_p=c.a_p,")],
        [T_MO, T_FP, T_ORD],
    ),
    (
        "S1",
        "old levels swapped by the loop",
        [
            (
                SS,
                "transient = (dt, old_u, old_v) if bdf2",
                "transient = (dt, old_u[::-1], old_v[::-1]) if bdf2",
            )
        ],
        [T_TR, T_ORD],
    ),
    (
        "S2",
        "the first step left BDF2 with a zero old level",
        [
            (SS, "bdf2 = scheme == BDF2 and n > 1", "bdf2 = scheme == BDF2"),
            (
                SS,
                "old_u: tuple[np.ndarray, ...] = (u.copy(),)\n"
                "        old_v: tuple[np.ndarray, ...] = (v.copy(),)",
                "old_u: tuple[np.ndarray, ...] = (u.copy(), np.zeros_like(u))\n"
                "        old_v: tuple[np.ndarray, ...] = (v.copy(), np.zeros_like(v))",
            ),
        ],
        [T_TR, T_ORD],
    ),
    (
        "S3",
        "the step cap ignored",
        [(SS, "            if not met:\n", "            if False:\n")],
        [T_TR],
    ),
    (
        "S4",
        "a capped correction allowed to end a step",
        [
            (
                SS,
                "if residual < step_tol and not corrected.reached_cap:",
                "if residual < step_tol:",
            )
        ],
        [T_TR],
    ),
    (
        "S5",
        "the criterion scaled by reference_velocity",
        [
            (
                SS,
                "residual = max(du, dv) / scale\n",
                "residual = max(du, dv) / self.reference_velocity\n",
            )
        ],
        [T_TR],
    ),
    (
        "S6",
        "a non-finite residual left to the cap",
        [
            (
                SS,
                "                if not math.isfinite(residual):\n"
                "                    break\n",
                "",
            )
        ],
        [T_TR],
    ),
    (
        "S7",
        "the older level never advanced",
        [
            (
                SS,
                "old_u = (u.copy(), old_u[0])",
                "old_u = (u.copy(), u.copy())",
            ),
            (
                SS,
                "old_v = (v.copy(), old_v[0])",
                "old_v = (v.copy(), v.copy())",
            ),
        ],
        [T_TR, T_ORD],
    ),
    (
        "B1a",
        "the march's normal velocities not written at its start",
        [
            (
                SS,
                "        self._boundary.apply_normal_velocity(u, v)\n\n"
                "        self.transient_steps = []\n",
                "        self.transient_steps = []\n",
            )
        ],
        [T_BD],
    ),
    (
        "B1b",
        "the march's outlets not extrapolated before each prediction",
        [
            (
                SS,
                "                self._extrapolate_outlets(u, v)\n"
                "                prediction = self._predictor.predict(u, v, p, transient=transient)",
                "                prediction = self._predictor.predict(u, v, p, transient=transient)",
            )
        ],
        [T_BD],
    ),
]


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def pytest(root: Path, test_file: str, expression: str) -> tuple[int, str]:
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    python = str(ROOT / ".venv" / "Scripts" / "python.exe")
    command = [python, "-m", "pytest", test_file, "-q", "-x", "-p", "no:cacheprovider"]
    if expression:
        command += ["-k", expression]
    done = subprocess.run(command, cwd=root, capture_output=True, text=True, env=env)
    tail = "\n".join(done.stdout.strip().splitlines()[-3:])
    return done.returncode, tail


def apply(root: Path, path: str, old: str, new: str) -> bool:
    target = root / path
    raw = target.read_bytes().decode("ascii")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    if text.count(old) != 1 or old == new:
        return False
    text = text.replace(old, new)
    target.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("ascii"))
    return True


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    wanted = set(sys.argv[2:])
    plants = [p for p in PLANTS if not wanted or p[0] in wanted]
    if git(root, "status", "--porcelain").strip():
        raise SystemExit("the worktree is not clean")
    controls = sorted({selection for p in plants for selection in p[3]})
    for test_file, expression in controls:
        code, tail = pytest(root, test_file, expression)
        last = tail.splitlines()[-1] if tail else ""
        print(f"control {test_file} -k {expression!r}: {code} {last}", flush=True)
        if code != 0:
            raise SystemExit("control failed")
    results = []
    for pid, what, edits, selections in plants:
        started = time.perf_counter()
        applied = all(apply(root, path, old, new) for path, old, new in edits)
        caught_by = []
        outcome = "NOT APPLIED"
        if applied:
            outcome = "SURVIVED"
            for test_file, expression in selections:
                code, tail = pytest(root, test_file, expression)
                verdict = {0: "passed", 1: "failed"}.get(code, f"broken ({code})")
                caught_by.append(
                    {
                        "tests": f"{test_file} -k {expression!r}",
                        "verdict": verdict,
                        "tail": tail,
                    }
                )
                if code >= 2:
                    outcome = f"BROKEN ({code})"
                elif code == 1 and outcome == "SURVIVED":
                    outcome = "CAUGHT"
        git(root, "checkout", "--", "src")
        if git(root, "status", "--porcelain").strip():
            raise SystemExit(f"{pid}: the worktree is not clean after restoring")
        seconds = time.perf_counter() - started
        verdicts = ", ".join(f"{c['tests']}: {c['verdict']}" for c in caught_by)
        print(f"{pid} {outcome} {seconds:.0f} s  {what}  [{verdicts}]", flush=True)
        results.append(
            {
                "id": pid,
                "what": what,
                "outcome": outcome,
                "selections": caught_by,
                "seconds": seconds,
            }
        )
    RECORDS.mkdir(parents=True, exist_ok=True)
    out = RECORDS / f"plants{'_' + '_'.join(sorted(wanted)) if wanted else ''}.json"
    out.write_text(
        json.dumps(
            {
                "commit": git(root, "rev-parse", "--short", "HEAD").strip(),
                "plants": results,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
