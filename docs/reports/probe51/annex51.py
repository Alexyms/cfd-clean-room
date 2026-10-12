"""Item 5 of prompt 51: the harness row val017_216x72 on main and on this branch.

Usage:
    python annex51.py run LABEL [--root TREE] [--concurrent P] [--append]
    python annex51.py compare BASE BUILT

``run`` is scripts/benchmark.py's ``run_case`` on the committed case file
(``load_preset("val017_216x72")``), every setting the file's, imported from
TREE (default this checkout) so that a worktree of origin/main measures main.
One probe-side substitution measures without changing the solve: the solver
is TREE's StaggeredSolver with its callback wrapped to keep, every outer
iteration, the k and eps solves' iteration counts, their fallbacks and
whether both converged. Main's model names the count ``last_sweeps`` (Jacobi
sweeps) and has no fallback; the branch's names it ``last_iterations`` (CG
iterations) beside ``last_fallbacks``; the trace reads whichever the tree
has. The record, results/builder51/annex_LABEL.json of the main checkout,
holds the harness row as run_case returns it (stop, outer count, wall time,
the pressure correction's CG iterations, the scored lines and their RMS),
the trace's means and totals, the four lines and the face hash (prompt 41's
measure); the final faces go to annex_LABEL.npz. ``--append`` appends the
row to TREE's benchmarks/results.jsonl as the harness would; ``--cap N``
stops at N outer iterations, a smoke run of the driver and never a
measurement. One BLAS thread.

``compare`` reads two records and writes compare_BASE_BUILT.json: each
row's stop, outer count, seconds per outer iteration and mean iteration
counts, and the largest line difference between them in u / u0.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np

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
OUT = MAIN / "results" / "builder51"
CASE = "val017_216x72"


def face_hash(u: np.ndarray, v: np.ndarray) -> str:
    """SHA-256 over u's bytes then v's (docs/reports/probe41/outlet41.py)."""
    return hashlib.sha256(
        np.ascontiguousarray(u, dtype="<f8").tobytes()
        + np.ascontiguousarray(v, dtype="<f8").tobytes()
    ).hexdigest()


def commit(root: Path) -> str:
    """The commit checked out at TREE, so the record names what it measured."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def traced_solver(solver_class: type) -> type:
    """TREE's solver with the k and eps counts kept per outer iteration."""

    class TracedSolver(solver_class):  # type: ignore[misc, valid-type]
        instances: ClassVar[list] = []

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.trace: dict[str, list] = {
                "k_iterations": [],
                "eps_iterations": [],
                "k_fallbacks": [],
                "eps_fallbacks": [],
                "scalar_converged": [],
            }
            TracedSolver.instances.append(self)

        def solve_steady(
            self, on_iteration: Callable | None = None, **kwargs: object
        ) -> tuple:
            """The committed solve, with the callback wrapped."""

            def traced(state: object) -> None:
                model = self._model
                if model is not None:
                    counts = getattr(model, "last_iterations", None)
                    if counts is None:
                        counts = model.last_sweeps
                    fallbacks = getattr(model, "last_fallbacks", (0, 0))
                    self.trace["k_iterations"].append(int(counts[0]))
                    self.trace["eps_iterations"].append(int(counts[1]))
                    self.trace["k_fallbacks"].append(int(fallbacks[0]))
                    self.trace["eps_fallbacks"].append(int(fallbacks[1]))
                    self.trace["scalar_converged"].append(bool(model.solves_converged))
                if on_iteration is not None:
                    on_iteration(state)

            return super().solve_steady(on_iteration=traced, **kwargs)

    return TracedSolver


def run(args: argparse.Namespace) -> None:
    """One row: run_case on the case file, from TREE, traced."""
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    benchmark = importlib.import_module("benchmark")
    if Path(benchmark.__file__).resolve().parents[1] != root:
        raise SystemExit(
            f"benchmark was imported from {benchmark.__file__}, not {root}"
        )
    annex20 = importlib.import_module("validation.annex20")
    mesh_module = importlib.import_module("src.mesh")
    cases = importlib.import_module("validation.cases")
    traced = traced_solver(benchmark.StaggeredSolver)
    benchmark.StaggeredSolver = traced
    if args.cap is not None:
        # A smoke run of the driver itself, never a measurement.
        config = cases.load_preset(CASE)
        config.max_simple_iter = args.cap
        benchmark.load_preset = lambda case_id: config
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    tree = commit(root)
    row = benchmark.run_case(CASE, benchmark.DEFAULT_METHOD, 100, args.concurrent)
    if args.append:
        with benchmark.RESULTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    solver = traced.instances[-1]
    faces = solver.face_velocities
    config = cases.load_preset(CASE)
    mesh = mesh_module.Mesh(config)
    lines = annex20.sample_lines(mesh, np.array(faces.u))
    trace = solver.trace
    outer = row["work"]["outer_iterations"]
    record = {
        "label": args.label,
        "commit": tree,
        "root": str(root),
        "started": started,
        "appended": bool(args.append),
        "row": row,
        "trace": trace,
        "k_iterations_mean": float(np.mean(trace["k_iterations"])),
        "eps_iterations_mean": float(np.mean(trace["eps_iterations"])),
        "fallbacks": int(sum(trace["k_fallbacks"]) + sum(trace["eps_fallbacks"])),
        "scalar_cap_hits": int(
            len(trace["scalar_converged"]) - sum(trace["scalar_converged"])
        ),
        "pressure_iterations_mean": row["work"]["inner_iterations"] / outer,
        "seconds_per_outer": row["time"]["wall_seconds"] / outer,
        "lines": {key: values.tolist() for key, values in lines.items()},
        "face_hash": face_hash(faces.u, faces.v),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"annex_{args.label}"
    (OUT / f"{name}.json").write_text(json.dumps(record))
    np.savez_compressed(
        OUT / f"{name}.npz", u_faces=np.array(faces.u), v_faces=np.array(faces.v)
    )
    print(
        f"{name}: {row['outcome']['stop_reason']} at {outer}, "
        f"{row['time']['wall_seconds']:.0f} s, value {row['accuracy']['value']:.4f}, "
        f"passed {row['accuracy']['passed']}, fallbacks {record['fallbacks']}",
        flush=True,
    )


def compare(args: argparse.Namespace) -> None:
    """Two records side by side, and their largest line difference."""
    a, b = (
        json.loads((OUT / f"annex_{lab}.json").read_text())
        for lab in (args.base, args.built)
    )

    def summary(rec: dict) -> dict:
        row = rec["row"]
        return {
            "label": rec["label"],
            "commit": rec["commit"],
            "stop": row["outcome"]["stop_reason"],
            "outer": row["work"]["outer_iterations"],
            "wall_seconds": row["time"]["wall_seconds"],
            "seconds_per_outer": rec["seconds_per_outer"],
            "k_iterations_mean": rec["k_iterations_mean"],
            "eps_iterations_mean": rec["eps_iterations_mean"],
            "pressure_iterations_mean": rec["pressure_iterations_mean"],
            "pressure_cap_hits": row["outcome"]["pressure_cap_hits"],
            "fallbacks": rec["fallbacks"],
            "scalar_cap_hits": rec["scalar_cap_hits"],
            "lines": {
                key: {
                    "rms": line["rms"],
                    "allowance": line["allowance"],
                    "scored": line["scored"],
                    "pass": line["pass"],
                }
                for key, line in row["accuracy"]["lines"].items()
            },
            "value": row["accuracy"]["value"],
            "passed": row["accuracy"]["passed"],
        }

    line_diff = {
        key: float(np.abs(np.array(a["lines"][key]) - np.array(b["lines"][key])).max())
        for key in a["lines"]
    }
    out = {
        "base": summary(a),
        "built": summary(b),
        "outer_change": (
            b["row"]["work"]["outer_iterations"] - a["row"]["work"]["outer_iterations"]
        )
        / a["row"]["work"]["outer_iterations"],
        "seconds_per_outer_change": (b["seconds_per_outer"] - a["seconds_per_outer"])
        / a["seconds_per_outer"],
        "line_difference_u0": line_diff,
        "largest_line_difference_u0": max(line_diff.values()),
    }
    text = json.dumps(out, indent=1)
    (OUT / f"compare_{args.base}_{args.built}.json").write_text(text)
    print(text)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    p.add_argument("label")
    p.add_argument("--root", default=str(HERE))
    p.add_argument("--concurrent", type=int, default=1)
    p.add_argument("--append", action="store_true")
    p.add_argument("--cap", type=int, default=None)
    p.set_defaults(func=run)
    p = sub.add_parser("compare")
    p.add_argument("base")
    p.add_argument("built")
    p.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
