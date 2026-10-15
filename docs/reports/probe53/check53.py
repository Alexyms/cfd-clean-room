"""Items 5 and 7 of prompt 53: the laminar cases' hashes under the default, and their criteria under multigrid.

Usage:
    python check53.py val CASE --label L [--root TREE] [--preconditioner P]
    python check53.py report --base B --built L

``val`` runs a VAL preset (``validation.cases.load_preset``) through TREE's
committed StaggeredSolver from rest to its own stop, under the preset's
settings with ``pressure_preconditioner`` set to P when it is given (absent,
the key is the preset's own, the default), and records the stop, the outer
count, the SHA-256 over u's face bytes then v's (prompt 43's ``val``), the
pressure correction's CG iterations in total, the harness's accuracy figure
(``scripts/benchmark.py``'s ``accuracy_of``) and the criterion it is read
against: VAL-001's L2 error below 1% and VAL-002's centerline errors below
2% of the lid speed, each the committed test's bound. TREE (default this
checkout) is a worktree of another commit when the base is measured. The
record is results/builder53/val_CASE_L.json of the main checkout.

``report`` compares each case's records under labels B and L: stop, count
and face hash, equal or not.
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
CASES = ("val001_80x40", "val001_80x40_stretched", "val002_80x80")
# The committed tests' bounds: tests/test_poiseuille.py and tests/test_lid_cavity.py.
CRITERIA = {"poiseuille": 0.01, "cavity": 0.02}


def face_hash(u: np.ndarray, v: np.ndarray) -> str:
    """SHA-256 over u's bytes then v's (docs/reports/probe41/outlet41.py)."""
    return hashlib.sha256(
        np.ascontiguousarray(u, dtype="<f8").tobytes()
        + np.ascontiguousarray(v, dtype="<f8").tobytes()
    ).hexdigest()


def commit(root: Path) -> str:
    """The commit checked out at TREE."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def val(root: Path, case: str, label: str, preconditioner: str | None) -> None:
    """One preset from TREE to its stop; the record."""
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    benchmark = importlib.import_module("benchmark")
    if Path(benchmark.__file__).resolve().parents[1] != root:
        raise SystemExit(
            f"benchmark was imported from {benchmark.__file__}, not {root}"
        )
    cases = importlib.import_module("validation.cases")
    mesh_module = importlib.import_module("src.mesh")
    boundary_module = importlib.import_module("src.boundary_staggered")
    solver_module = importlib.import_module("src.solver_staggered")
    config = cases.load_preset(case)
    if preconditioner is not None:
        config.pressure_preconditioner = preconditioner
    kind = benchmark.CASES[case][0]
    mesh = mesh_module.Mesh(config)
    solver = solver_module.StaggeredSolver(
        mesh, config, boundary_module.StaggeredBoundary(mesh, config)
    )
    counts = {"outer": 0, "pressure_iterations": 0, "cap_hits": 0}

    def observe(state: object) -> None:
        counts["outer"] += 1
        counts["pressure_iterations"] += int(state.pressure_iterations)

    started = time.perf_counter()
    u, v, _p = solver.solve_steady(on_iteration=observe)
    seconds = time.perf_counter() - started
    faces = solver.face_velocities
    accuracy = benchmark.accuracy_of(kind, config, mesh, u, v)
    if kind == "cavity":
        worst = max(accuracy["components"].values())
    else:
        worst = accuracy["value"]
    record = {
        "case": case,
        "label": label,
        "commit": commit(root),
        "root": str(root),
        "preconditioner": getattr(config, "pressure_preconditioner", "jacobi"),
        "stop": solver.stop_reason,
        "outer": counts["outer"],
        "pressure_iterations": counts["pressure_iterations"],
        "pressure_cap_hits": solver.pressure_cap_hits,
        "face_hash": face_hash(faces.u, faces.v),
        "accuracy": accuracy,
        "criterion": CRITERIA[kind],
        "passes": worst < CRITERIA[kind],
        "seconds": seconds,
        "stage_seconds": dict(solver.stage_seconds),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"val_{case}_{label}.json").write_text(json.dumps(record, indent=1))
    np.savez_compressed(
        OUT / f"val_{case}_{label}.npz",
        u_faces=np.array(faces.u),
        v_faces=np.array(faces.v),
    )
    print(
        f"{case} {label}: {record['stop']} at {record['outer']}, hash "
        f"{record['face_hash'][:16]}, value {accuracy['value']:.4e}, passes "
        f"{record['passes']}, {seconds:.0f} s",
        flush=True,
    )


def report(base: str, built: str) -> None:
    """Each case's stop, count and face hash under BUILT against BASE."""
    summary = []
    for case in CASES:
        paths = {lab: OUT / f"val_{case}_{lab}.json" for lab in (base, built)}
        if not all(p.exists() for p in paths.values()):
            summary.append({"case": case, "present": False})
            continue
        a, b = (json.loads(paths[lab].read_text()) for lab in (base, built))
        summary.append(
            {
                "case": case,
                "commits": (a["commit"], b["commit"]),
                "stop": (a["stop"], b["stop"]),
                "outer": (a["outer"], b["outer"]),
                "face_hash": (a["face_hash"][:16], b["face_hash"][:16]),
                "equal": a["stop"] == b["stop"]
                and a["outer"] == b["outer"]
                and a["face_hash"] == b["face_hash"],
            }
        )
    text = json.dumps(summary, indent=1)
    (OUT / f"report_{base}_{built}.json").write_text(text)
    print(text)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(HERE))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("val")
    p.add_argument("case", choices=CASES)
    p.add_argument("--label", required=True)
    p.add_argument("--preconditioner", choices=("jacobi", "multigrid"), default=None)
    p = sub.add_parser("report")
    p.add_argument("--base", required=True)
    p.add_argument("--built", required=True)
    args = parser.parse_args()
    if args.command == "report":
        report(args.base, args.built)
        return
    val(Path(args.root).resolve(), args.case, args.label, args.preconditioner)


if __name__ == "__main__":
    main()
