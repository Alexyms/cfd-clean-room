"""Item 7 of prompt 53: a harness row under a chosen pressure preconditioner.

Usage:
    python row53.py CASE LABEL [--root TREE] [--preconditioner P] [--concurrent N]

Runs scripts/benchmark.py's ``run_case`` on the committed case file of CASE
(``val017_216x72`` or ``val019_656x60``), imported from TREE (default this
checkout), with one probe-side substitution: ``load_preset`` returns the
case's configuration with ``pressure_preconditioner`` set to P (default
``multigrid``), so the row's params record it. Nothing else differs from the
harness's row. The record, results/builder53/row_CASE_LABEL.json of the main
checkout, holds the row as run_case returns it, the face hash, seconds per
outer iteration and, for the Annex 20 room, its four lines
(``validation.annex20.sample_lines``); the faces go to row_CASE_LABEL.npz.
The row is not appended to benchmarks/results.jsonl: the default stays the
diagonal, and a row under another preconditioner there would be read as the
case's latest. One BLAS thread.
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
from pathlib import Path

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
OUT = MAIN / "results" / "builder53"
CASES = ("val017_216x72", "val019_656x60")


def face_hash(u: np.ndarray, v: np.ndarray) -> str:
    """SHA-256 over u's bytes then v's (docs/reports/probe41/outlet41.py)."""
    return hashlib.sha256(
        np.ascontiguousarray(u, dtype="<f8").tobytes()
        + np.ascontiguousarray(v, dtype="<f8").tobytes()
    ).hexdigest()


def main() -> None:
    """One row from TREE under P; the record and the faces."""
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("label")
    parser.add_argument("--root", default=str(HERE))
    parser.add_argument("--preconditioner", default="multigrid")
    parser.add_argument("--concurrent", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    benchmark = importlib.import_module("benchmark")
    if Path(benchmark.__file__).resolve().parents[1] != root:
        raise SystemExit(
            f"benchmark was imported from {benchmark.__file__}, not {root}"
        )
    cases = importlib.import_module("validation.cases")
    mesh_module = importlib.import_module("src.mesh")
    solvers: list = []

    class KeptSolver(benchmark.StaggeredSolver):  # type: ignore[misc, name-defined]
        """The committed solver, kept so its faces can be read after the row."""

        def __init__(self, *a: object, **k: object) -> None:
            super().__init__(*a, **k)
            solvers.append(self)

    def preset(case_id: str) -> object:
        config = cases.load_preset(case_id)
        config.pressure_preconditioner = args.preconditioner
        return config

    benchmark.StaggeredSolver = KeptSolver
    benchmark.load_preset = preset
    tree = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    row = benchmark.run_case(args.case, benchmark.DEFAULT_METHOD, 100, args.concurrent)
    solver = solvers[-1]
    faces = solver.face_velocities
    outer = row["work"]["outer_iterations"]
    record = {
        "case": args.case,
        "label": args.label,
        "commit": tree,
        "root": str(root),
        "started": started,
        "preconditioner": args.preconditioner,
        "row": row,
        "face_hash": face_hash(faces.u, faces.v),
        "seconds_per_outer": row["time"]["wall_seconds"] / outer,
        "pressure_seconds_per_outer": row["time"]["stages"]["pressure"] / outer,
        "pressure_iterations_mean": row["work"]["inner_iterations"] / outer,
    }
    if args.case.startswith("val017"):
        annex20 = importlib.import_module("validation.annex20")
        mesh = mesh_module.Mesh(cases.load_preset(args.case))
        lines = annex20.sample_lines(mesh, np.array(faces.u))
        record["lines"] = {key: values.tolist() for key, values in lines.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"row_{args.case}_{args.label}"
    (OUT / f"{name}.json").write_text(json.dumps(record))
    np.savez_compressed(
        OUT / f"{name}.npz", u_faces=np.array(faces.u), v_faces=np.array(faces.v)
    )
    print(
        f"{name}: {row['outcome']['stop_reason']} at {outer}, "
        f"{row['time']['wall_seconds']:.0f} s, {record['seconds_per_outer']:.3f} s/outer, "
        f"value {row['accuracy']['value']:.4f}, cap hits {row['outcome']['pressure_cap_hits']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
