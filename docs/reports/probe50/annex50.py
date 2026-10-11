"""Item 6 of prompt 50: the harness row val017_216x72 under rule version 5.

Usage:
    python annex50.py run TAG [--tol T] [--cap N] [--concurrent P]

The row is scripts/benchmark.py's ``run_case`` on the committed case file
(``load_preset("val017_216x72")``), with ``turbulence.tol`` set to T (default
the file's) and ``max_simple_iter`` to N (default the file's); every other
setting is the file's. Two probe-side substitutions measure without changing
the solve: the rule is rule50.RecordingRule50 (every outer iteration's
readings and scales kept, the decision the committed rule's) and the solver
rule50.TracedSolver (the k and eps solve counts kept). The record, under
results/builder50/annex_TAG.json, holds the harness row as run_case returns
it, the rule's per-iteration readings, the four lines sampled from the final
faces, their largest difference from prompt 49's records of the case
(results/builder49/lines_*.json) and the face hash (prompt 41's measure);
the final faces go to annex_TAG.npz. One BLAS thread.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

import benchmark  # noqa: E402
import rule50  # noqa: E402

from src.mesh import FLUID, Mesh  # noqa: E402
from validation import annex20  # noqa: E402
from validation.cases import load_preset  # noqa: E402

OUT = ROOT / "results" / "builder50"
CASE = "val017_216x72"
# Prompt 49's records of the same case under the default scheme: the 1e-13
# diagnostic at its 12,000 cap, and the file's row at its 10,000 cap.
RECORDS_49 = {
    "tol1e-13_cap12000": "lines_standard_216x72_upwind_tol1e-13.json",
    "file_cap10000": "lines_standard_216x72_upwind.json",
}


def face_hash(u: np.ndarray, v: np.ndarray) -> str:
    """SHA-256 over u's bytes then v's (docs/reports/probe41/outlet41.py)."""
    return hashlib.sha256(
        np.ascontiguousarray(u, dtype="<f8").tobytes()
        + np.ascontiguousarray(v, dtype="<f8").tobytes()
    ).hexdigest()


def commit() -> str:
    """The checked-out commit, so the record names the tree it measured."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def against_49(lines: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """The largest |difference| of each line from each of prompt 49's records, in u0."""
    out: dict[str, dict[str, float]] = {}
    for label, name in RECORDS_49.items():
        record = json.loads((ROOT / "results" / "builder49" / name).read_text())
        out[label] = {
            key: float(
                np.abs(lines[key] - np.array(record["lines"][key]["u_over_u0"])).max()
            )
            for key in annex20.LINES
        }
    return out


def run(args: argparse.Namespace) -> None:
    """One row: run_case on the case file with the tolerance and cap given."""
    config = load_preset(CASE)
    assert config.turbulence is not None
    tol = args.tol if args.tol is not None else config.turbulence.tol
    cap = args.cap if args.cap is not None else config.max_simple_iter
    config.turbulence = dataclasses.replace(config.turbulence, tol=tol)
    config.max_simple_iter = cap
    benchmark.load_preset = lambda case_id: config
    benchmark.StaggeredSolver = rule50.TracedSolver
    rule50.install()
    rule50.TracedSolver.instances.clear()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    tree = commit()
    row = benchmark.run_case(CASE, benchmark.DEFAULT_METHOD, 100, args.concurrent)
    solver = rule50.TracedSolver.instances[-1]
    rule = rule50.RecordingRule50.instances[-1]
    faces = solver.face_velocities
    assert faces is not None
    mesh = Mesh(config)
    lines = annex20.sample_lines(mesh, np.array(faces.u))
    state = solver.turbulence_state
    assert state is not None
    fluid = mesh.cell_type == FLUID
    nu = config.mu / config.rho
    trace = solver.trace
    record = {
        "tag": args.tag,
        "commit": tree,
        "started": started,
        "turbulence_tol": tol,
        "turbulence_max_iter": config.turbulence.max_iter,
        "cap": cap,
        "nu_air": nu,
        "row": row,
        "rule": rule.summary((config.iteration_error_tol, config.mass_imbalance_tol)),
        "trace": trace,
        "scalar_cap_hits": int(
            len(trace["scalar_converged"]) - sum(trace["scalar_converged"])
        ),
        "k_sweeps_mean": float(np.mean(trace["k_sweeps"])),
        "eps_sweeps_mean": float(np.mean(trace["eps_sweeps"])),
        "nu_t_over_nu": {
            "fluid_median": float(np.median(state.nu_t[fluid])) / nu,
            "fluid_max": float(state.nu_t[fluid].max()) / nu,
        },
        "lines": {key: values.tolist() for key, values in lines.items()},
        "against_49": against_49(lines),
        "face_hash": face_hash(faces.u, faces.v),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"annex_{args.tag}"
    (OUT / f"{name}.json").write_text(json.dumps(record))
    np.savez_compressed(
        OUT / f"{name}.npz",
        u_faces=np.array(faces.u),
        v_faces=np.array(faces.v),
        nu_t=np.array(state.nu_t),
        cell_type=mesh.cell_type,
    )
    print(
        f"{name}: {row['outcome']['stop_reason']} at {row['work']['outer_iterations']}, "
        f"value {row['accuracy']['value']:.4f}, passed {row['accuracy']['passed']}",
        flush=True,
    )


def main() -> None:
    """Dispatch the one mode the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    p.add_argument("tag")
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--cap", type=int, default=None)
    p.add_argument("--concurrent", type=int, default=1)
    p.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
