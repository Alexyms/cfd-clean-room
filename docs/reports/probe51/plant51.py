"""The planted defects of prompt 51, commit A: each is applied, the tests run, and the source restored.

Usage:
    python docs/reports/probe51/plant51.py [NAME ...]

Each plant replaces exact text in one source file (CRLF in the working tree,
so the match is made on the file as it is), runs the test files that guard
it, records whether any test failed, and restores the file's bytes before the
next plant, also on an error. A plant whose text is not found once is an
error, not a pass. The result goes to results/builder51/plants.json. Run it
alone: it edits the tree it measures.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "builder51"
PYTHON = ROOT / ".venv" / "Scripts" / "python"
SCALAR = "tests/test_scalar_scheme.py"
TURBULENCE = "tests/test_turbulence.py"

# name: (file, old text, new text, test files, what the plant stands for)
PLANTS: dict[str, tuple[str, list[tuple[str, str]], list[str], str]] = {
    "two_norm_stop": (
        "src/scalar_scheme.py",
        [
            (
                "            c, a_p, b, g_u, g_v, fixed, stop, max_iter\n",
                "            c, a_p, b, g_u, g_v, fixed, tol * float(np.sqrt(np.vdot(b, b))), max_iter\n",
            ),
            (
                "    r = true_residual()\n    if float(np.abs(r).max()) <= stop:\n",
                "    r = true_residual()\n    if float(np.sqrt(np.vdot(r, r))) <= stop:\n",
            ),
            (
                "        if float(np.abs(r).max()) <= stop or iterations == max_iter:\n"
                "            r = true_residual()\n"
                "            if float(np.abs(r).max()) <= stop:\n",
                "        if float(np.sqrt(np.vdot(r, r))) <= stop or iterations == max_iter:\n"
                "            r = true_residual()\n"
                "            if float(np.sqrt(np.vdot(r, r))) <= stop:\n",
            ),
        ],
        [SCALAR],
        "CG's own 2-norm stop, ||r|| <= tol ||b||, in place of the max-norm one",
    ),
    "held_unknown": (
        "src/scalar_scheme.py",
        [
            (
                "    fixed = solid if held is None else solid | held\n",
                "    fixed = solid\n",
            )
        ],
        [SCALAR, TURBULENCE],
        "a held cell left as an unknown under CG",
    ),
    "transport_on_cg": (
        "src/solver_transport.py",
        [("            method=JACOBI,\n", '            method="cg",\n')],
        [SCALAR],
        "the transport solver on CG",
    ),
    "turbulence_on_jacobi": (
        "src/turbulence.py",
        [
            (
                "            self._max_iter,\n            method=CONJUGATE_GRADIENT,\n        )\n        eps_result",
                "            self._max_iter,\n        )\n        eps_result",
            )
        ],
        [TURBULENCE],
        "the model's k solve left on the default, Jacobi",
    ),
    "fallback_from_cg_iterate": (
        "src/scalar_scheme.py",
        [
            (
                "    c, sweeps, converged = _jacobi(\n"
                "        c_star, a_p, b, g_u, g_v, solid, held, stop, max_iter\n"
                "    )\n"
                "    return ImplicitResult(\n"
                "        field=c,\n"
                "        method=method,\n"
                "        iterations=iterations,\n",
                "    c, sweeps, converged = _jacobi(\n"
                "        c, a_p, b, g_u, g_v, solid, held, stop, max_iter\n"
                "    )\n"
                "    return ImplicitResult(\n"
                "        field=c,\n"
                "        method=method,\n"
                "        iterations=iterations,\n",
            )
        ],
        [SCALAR, TURBULENCE],
        "the fallback restarts from the CG iterate, not C*",
    ),
    "fallback_on_cap_only": (
        "src/scalar_scheme.py",
        [
            (
                "    if converged and bool(np.all(c[~fixed] > 0.0)):\n",
                "    if converged:\n",
            )
        ],
        [SCALAR, TURBULENCE],
        "the fallback on the cap only, not on a non-positive value",
    ),
    "fallback_below_zero_only": (
        "src/scalar_scheme.py",
        [
            (
                "    if converged and bool(np.all(c[~fixed] > 0.0)):\n",
                "    if converged and bool(np.all(c[~fixed] >= 0.0)):\n",
            )
        ],
        [SCALAR],
        "the fallback on a negative value only, a zero kept",
    ),
    "no_fallback": (
        "src/scalar_scheme.py",
        [
            (
                "    if converged and bool(np.all(c[~fixed] > 0.0)):\n",
                "    if True:\n",
            )
        ],
        [SCALAR, TURBULENCE],
        "no fallback at all, the CG iterate returned whatever it is",
    ),
}


def run_plant(name: str) -> dict[str, object]:
    """Apply one plant, run its tests, restore the file; the outcome."""
    path_name, edits, tests, meaning = PLANTS[name]
    path = ROOT / path_name
    original = path.read_bytes()
    text = original.decode()
    for old, new in edits:
        old_crlf = old.replace("\n", "\r\n")
        if text.count(old_crlf) != 1:
            raise SystemExit(
                f"{name}: text found {text.count(old_crlf)} times in {path_name}"
            )
        text = text.replace(old_crlf, new.replace("\n", "\r\n"))
    try:
        path.write_bytes(text.encode())
        run = subprocess.run(
            [str(PYTHON), "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        path.write_bytes(original)
    failed = [
        line.split(" ", 1)[1].split(" - ")[0]
        for line in run.stdout.splitlines()
        if line.startswith("FAILED ")
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
    """Run the named plants, or every one, and write the record."""
    names = sys.argv[1:] or list(PLANTS)
    results = []
    for name in names:
        outcome = run_plant(name)
        print(f"{name}: caught {outcome['caught']} {outcome['failures']}", flush=True)
        results.append(outcome)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plants.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
