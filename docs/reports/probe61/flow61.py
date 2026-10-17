"""Prompt 61: one steady solve from rest of a room, recorded for the step 0b report.

Usage:
    python flow61.py room NAME CONFIG --root TREE [--concurrent N]
        the configuration file CONFIG (a path under TREE), solved as it stands:
        the vented room configs/clean_room_vents.yaml and the default
        configs/clean_room_default.yaml, both 320x120 under the stopping rule
        with the cap 15,000
    python flow61.py box NAME --root TREE [--concurrent N]
        prediction (c)'s closed box: tests/vent_cases.py's room with only the
        discharge and an exhaust of equal flow, as
        tests/test_solver_staggered.py solves it (air 55 times as viscous)

The solve is TREE's committed StaggeredSolver from rest, nothing replaced. The
corrector's output is read once per outer iteration (its ``correct`` wrapped,
the arithmetic untouched) for the signed domain sum of the corrected faces'
imbalance, which prediction (c) is about. The record, results/builder61/NAME.json
of the main checkout, holds the stop, the outer count, the wall time, the
stage times, the face hash (hash60.face_hash), the residual history, the signed
sums and the configuration mapping; NAME.npz the faces, the pressure and, with
the turbulence section, k, eps and nu_t. One BLAS thread, as every product run
since prompt 46.
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
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hash61 import face_hash, main_results, tree_commit  # noqa: E402


def box_raw(root: Path) -> dict:
    """The closed box of tests/test_solver_staggered.py's balance test."""
    vent_cases = importlib.import_module("tests.vent_cases")
    exhaust = {**vent_cases.EXHAUST, "velocity": 0.5}
    raw = vent_cases.vent_raw(
        {"discharge": vent_cases.DISCHARGE, "exhaust": exhaust},
        supply=False,
        returns=False,
    )
    raw["fluid"]["viscosity"] = 1.0e-3
    raw["solver"].update(
        stopping_rule="error_estimate",
        max_simple_iter=3000,
        iteration_error_tol=1.0e-6,
        mass_imbalance_tol=1.0e-10,
    )
    return raw


def main() -> None:
    """One solve from rest and its record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("room", "box"))
    parser.add_argument("name")
    parser.add_argument("config", nargs="?")
    parser.add_argument("--root", required=True)
    parser.add_argument("--concurrent", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    solver_module = importlib.import_module("src.solver_staggered")
    if Path(solver_module.__file__).resolve().parents[1] != root:
        raise SystemExit(f"src was imported from {solver_module.__file__}, not {root}")
    if args.mode == "room":
        if args.config is None:
            raise SystemExit("room needs the configuration file")
        raw = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    else:
        raw = box_raw(root)
    config = importlib.import_module("src.config").SimConfig.from_dict(raw)
    mesh = importlib.import_module("src.mesh").Mesh(config)
    boundary = importlib.import_module("src.boundary_staggered").StaggeredBoundary(
        mesh, config
    )
    solver = solver_module.StaggeredSolver(mesh, config, boundary)
    corrector = solver._corrector
    correct = corrector.correct
    sums: list[float] = []

    def read(prediction: object, p: np.ndarray) -> object:
        result = correct(prediction, p)
        sums.append(float(corrector.mass_imbalance(result.u, result.v).sum()))
        return result

    corrector.correct = read
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{args.name} start {started} concurrent {args.concurrent}", flush=True)
    t0 = time.perf_counter()
    solver.solve_steady()
    wall = time.perf_counter() - t0
    faces = solver.face_velocities
    outer = len(solver.residual_history)
    out = {
        "name": args.name,
        "mode": args.mode,
        "config_file": args.config,
        "commit": tree_commit(root),
        "root": str(root),
        "started": started,
        "concurrent_processes": args.concurrent,
        "grid": f"{config.nx}x{config.ny}",
        "stop": solver.stop_reason,
        "converged": bool(solver.converged),
        "outer": outer,
        "max_simple_iter": config.max_simple_iter,
        "cap_hits": int(solver.pressure_cap_hits),
        "wall_seconds": wall,
        "seconds_per_outer": wall / max(outer, 1),
        "stage_seconds": dict(solver.stage_seconds),
        "flux_scale": solver.flux_scale,
        "velocity_scale": boundary.get_max_boundary_velocity(),
        "total_inlet_flux": boundary.get_total_inlet_flux(),
        "fixed_flow_velocities": boundary.fixed_flow_velocities(),
        "face_hash": face_hash(faces.u, faces.v),
        "residual_history": [float(r) for r in solver.residual_history],
        "signed_sums": sums,
        "raw": raw,
    }
    arrays = {
        "u_faces": np.asarray(faces.u),
        "v_faces": np.asarray(faces.v),
        "p": np.asarray(solver.flow_state.p) if solver.flow_state else np.zeros(0),
        "xc": mesh.xc,
        "yc": mesh.yc,
        "x": mesh.x,
        "y": mesh.y,
        "cell_type": mesh.cell_type,
    }
    state = solver.turbulence_state
    if state is not None:
        arrays.update(k=state.k, eps=state.eps, nu_t=state.nu_t)
    target = main_results()
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{args.name}.json").write_text(json.dumps(out, indent=1, default=float))
    np.savez(target / f"{args.name}.npz", **arrays)
    print(
        f"{args.name}: {out['stop']} at {outer}, {wall:.0f} s, hash "
        f"{out['face_hash'][:16]}, worst signed sum {max(map(abs, sums)):.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
