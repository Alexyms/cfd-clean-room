"""Prompt 60, item 6: the warm start on the product room, through the argument.

Usage:
    python warm60.py self --root TREE [--concurrent N]
        prediction (b): the door-closed room (configs/clean_room_default.yaml,
        320x120) warm-started from its own converged state, ECR-002 step 8's
        L0 (results/builder56/L0_320x120.npz, commit b9e8191, 4,389 outer):
        does it stop within one rule window (100 outer iterations)?
    python warm60.py open --root TREE [--concurrent N]
        prediction (c): T1's warm arm again, the open-door state
        (docs/reports/probe59/warm59.py ``door_open(product_raw())``) from
        the same L0 state, through ``solve_steady(initial=...)`` in place
        of the probe's patch of the two allocations; its outer count and
        faces against results/builder59/warm.json (3,872 outer, commit
        809d1c6)

The kept state is read into a FlowState whole (faces, p, k, eps, nu_t); the
warm start reads its faces, p, k and eps and rebuilds nu_t, as the probe
did. Nothing under src/ is replaced. The record,
results/builder60/warm60_ARM.json of the main checkout, holds the stop, the
outer count, the wall time, the face hash, the largest face change from the
start and, for ``open``, the same fields of T1's warm record beside them;
the faces go to warm60_ARM.npz. One BLAS thread.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hash60  # noqa: E402

L0 = "results/builder56/L0_320x120"
T1_WARM = "results/builder59/warm.json"


def main() -> None:
    """One warm-started product solve; the record and the faces."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("self", "open"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--concurrent", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_dir = hash60.main_results()
    results = out_dir.parent
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "docs" / "reports" / "probe59"))
    solver_module = importlib.import_module("src.solver_staggered")
    if Path(solver_module.__file__).resolve().parents[1] != root:
        raise SystemExit(f"src was imported from {solver_module.__file__}, not {root}")
    warm59 = importlib.import_module("warm59")
    config_module = importlib.import_module("src.config")
    mesh_module = importlib.import_module("src.mesh")
    boundary_module = importlib.import_module("src.boundary_staggered")

    kept = np.load(results.parent / f"{L0}.npz")
    start = solver_module.FlowState(
        u=kept["u_faces"],
        v=kept["v_faces"],
        p=kept["p"],
        k=kept["k"],
        eps=kept["eps"],
        nu_t=kept["nu_t"],
    )
    l0 = json.loads((results.parent / f"{L0}.json").read_text(encoding="utf-8"))
    if hash60.face_hash(start.u, start.v) != l0["face_hash"]:
        raise SystemExit("the kept L0 faces do not hash to their record's face_hash")
    raw = warm59.product_raw()
    if args.arm == "open":
        raw = warm59.door_open(raw)
    config = config_module.SimConfig.from_dict(raw)
    mesh = mesh_module.Mesh(config)
    boundary = boundary_module.StaggeredBoundary(mesh, config)
    solver = solver_module.StaggeredSolver(mesh, config, boundary)

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    solver.solve_steady(initial=start)
    wall = time.perf_counter() - t0
    faces = solver.face_velocities
    record = {
        "arm": args.arm,
        "commit": hash60.tree_commit(root),
        "root": str(root),
        "started": started,
        "concurrent_processes": args.concurrent,
        "start": f"{L0}.npz (commit {l0['commit']}, {l0['stop']} at {l0['outer']})",
        "door": "open" if args.arm == "open" else "closed",
        "max_simple_iter": config.max_simple_iter,
        "stop": solver.stop_reason,
        "outer": len(solver.residual_history),
        "wall_seconds": wall,
        "seconds_per_outer": wall / max(len(solver.residual_history), 1),
        "cap_hits": int(solver.pressure_cap_hits),
        "rule_version": solver.rule_version,
        "residual_first": float(solver.residual_history[0]),
        "residual_end": float(solver.residual_history[-1]),
        "face_hash": hash60.face_hash(faces.u, faces.v),
        "largest_face_change_from_start": {
            "u": float(np.max(np.abs(np.asarray(faces.u) - start.u))),
            "v": float(np.max(np.abs(np.asarray(faces.v) - start.v))),
        },
        "velocity_scale": boundary.get_max_boundary_velocity(),
        "iteration_error_tol": config.iteration_error_tol,
        "residual_history": [float(r) for r in solver.residual_history],
    }
    if args.arm == "open":
        t1 = json.loads((results.parent / T1_WARM).read_text(encoding="utf-8"))
        record["t1_warm"] = {
            key: t1[key]
            for key in (
                "commit",
                "stop",
                "outer",
                "face_hash",
                "wall_seconds",
                "concurrent_processes",
                "residual_first",
                "residual_end",
            )
        }
        record["matches_t1"] = {
            "outer": record["outer"] == t1["outer"],
            "face_hash": record["face_hash"] == t1["face_hash"],
            "residual_history": record["residual_history"] == t1["residual_history"],
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"warm60_{args.arm}.json").write_text(json.dumps(record, indent=1))
    np.savez_compressed(
        out_dir / f"warm60_{args.arm}.npz",
        u_faces=np.array(faces.u),
        v_faces=np.array(faces.v),
    )
    print(
        f"warm60 {args.arm}: {record['stop']} at {record['outer']}, {wall:.0f} s, "
        f"hash {record['face_hash'][:16]}"
        + (f", matches T1 {record['matches_t1']}" if args.arm == "open" else ""),
        flush=True,
    )


if __name__ == "__main__":
    main()
