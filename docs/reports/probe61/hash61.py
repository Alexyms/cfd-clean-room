"""Prompt 61, prediction (a): the hashes of record at step 0b, no vents.

hash60.py of prompt 60, unchanged but for the records directory
(results/builder61/) and this paragraph: the same cases, the same
wrapper, the same SHA-256 over the faces.

Usage:
    python hash60.py row CASE --root TREE [--concurrent N]
        a harness row (scripts/benchmark.py run_case, the default method,
        multigrid) of TREE: the VAL-001 and VAL-002 presets, val017_216x72,
        val019_656x60
    python hash60.py product GRID --root TREE [--concurrent N]
        prompt 51's product row (docs/reports/probe47/coupled47.py
        ``product_raw(GRID, "standard", "control", 10000)``), GRID 80x30 or
        160x60, as prompt 54's inject54.py ran it
    python hash60.py record NAME PATH --root TREE [--concurrent N]
        the configuration a kept record names in its ``raw`` key, solved
        from rest: the step 8 room L0_320x120 (results/builder56/) and T1's
        rest arm (results/builder59/rest.json)

Each solve runs from rest through the committed StaggeredSolver of TREE,
imported from TREE, with nothing replaced but the class, wrapped so its
faces can be read; ``solve_steady`` is called with no ``initial`` and no
``resume``. The record, results/builder61/hash_LABEL.json of the main
checkout (LABEL ending in --suffix when given, as the base
commit's control runs do), holds the face hash (SHA-256 over u's bytes then v's, as
docs/reports/probe41/outlet41.py), the stop, the outer count, the wall
time and the process count. The hash of record each is compared with is
docs/reports/multigrid_default.md's table (tables54 hashes) or the kept
record's own ``face_hash``; tables60.py makes the comparison. One BLAS
thread, as every product run since prompt 46.
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
PRODUCT_CAP = 10000


def main_results() -> Path:
    """The main checkout's results/builder61, through git's common directory."""
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).parent / "results" / "builder61"


def face_hash(u: np.ndarray, v: np.ndarray) -> str:
    """SHA-256 over u's bytes then v's (docs/reports/probe41/outlet41.py)."""
    return hashlib.sha256(
        np.ascontiguousarray(u, dtype="<f8").tobytes()
        + np.ascontiguousarray(v, dtype="<f8").tobytes()
    ).hexdigest()


def tree_commit(root: Path) -> str:
    """The short commit of TREE."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def kept_solvers(root: Path) -> list:
    """Import TREE's src and wrap its solver class so every instance is kept."""
    sys.path.insert(0, str(root))
    solver_module = importlib.import_module("src.solver_staggered")
    if Path(solver_module.__file__).resolve().parents[1] != root:
        raise SystemExit(f"src was imported from {solver_module.__file__}, not {root}")
    kept: list = []
    committed = solver_module.StaggeredSolver

    class KeptSolver(committed):  # type: ignore[misc, valid-type]
        """The committed solver, kept so its faces can be read after the solve."""

        def __init__(self, *a: object, **k: object) -> None:
            super().__init__(*a, **k)
            kept.append(self)

    solver_module.StaggeredSolver = KeptSolver
    return kept


def solve_raw(root: Path, raw: dict) -> object:
    """Solve the mapping from rest with TREE's code; the solver."""
    config = importlib.import_module("src.config").SimConfig.from_dict(raw)
    mesh = importlib.import_module("src.mesh").Mesh(config)
    boundary = importlib.import_module("src.boundary_staggered").StaggeredBoundary(
        mesh, config
    )
    solver = importlib.import_module("src.solver_staggered").StaggeredSolver(
        mesh, config, boundary
    )
    solver.solve_steady()
    return solver


def main() -> None:
    """One solve from rest; its face hash and stop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("row", "product", "record"))
    parser.add_argument("name")
    parser.add_argument("path", nargs="?")
    parser.add_argument("--root", required=True)
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument("--suffix", default="", help="appended to the label")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    kept = kept_solvers(root)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    if args.mode == "row":
        sys.path.insert(0, str(root / "scripts"))
        benchmark = importlib.import_module("benchmark")
        benchmark.StaggeredSolver = importlib.import_module(
            "src.solver_staggered"
        ).StaggeredSolver
        benchmark.run_case(args.name, benchmark.DEFAULT_METHOD, 100, args.concurrent)
        solver = kept[-1]
        label = args.name
    elif args.mode == "product":
        sys.path.insert(0, str(root / "docs" / "reports" / "probe47"))
        coupled47 = importlib.import_module("coupled47")
        raw = coupled47.product_raw(args.name, "standard", "control", PRODUCT_CAP)
        solver = solve_raw(root, raw)
        label = f"product_{args.name}"
    else:
        if args.path is None:
            raise SystemExit("record needs the kept record's path")
        record = json.loads(Path(args.path).read_text(encoding="utf-8"))
        solver = solve_raw(root, record["raw"])
        label = args.name
    wall = time.perf_counter() - t0
    label += args.suffix
    faces = solver.face_velocities
    out = {
        "label": label,
        "mode": args.mode,
        "source": args.path,
        "commit": tree_commit(root),
        "root": str(root),
        "started": started,
        "concurrent_processes": args.concurrent,
        "stop": solver.stop_reason,
        "outer": len(solver.residual_history),
        "cap_hits": int(solver.pressure_cap_hits),
        "wall_seconds": wall,
        "face_hash": face_hash(faces.u, faces.v),
    }
    target = main_results()
    target.mkdir(parents=True, exist_ok=True)
    (target / f"hash_{label}.json").write_text(json.dumps(out, indent=1))
    print(
        f"{label}: {out['stop']} at {out['outer']}, {wall:.0f} s, "
        f"hash {out['face_hash'][:16]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
