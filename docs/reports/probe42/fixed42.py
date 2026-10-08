"""Demonstrations of prompt 42 (ECR-002 step 3): the built fixed-flow outlets against probe arm D0.

Usage:
    python fixed42.py coverage                        built faces against D0's, face for face
    python fixed42.py drift RTOL [--long]             the 80x30 drift case, RTOL 1e-8, 1e-4 or 1e-2
    python fixed42.py ladder RUNG                     40x15 at Re 895 or 8950
    python fixed42.py compare NAME D0_NAME            residual history and hash against a D0 record

The solver is the committed StaggeredSolver on the committed configuration
(configs/clean_room_default.yaml, whose outlets are fixed-flow); nothing is
subclassed for the physics. Only the two measuring devices of the probe are
reused from docs/reports/probe41/outlet41.py: the ten-sweep predictor (the
drift case's momentum step, equal to the committed predictor at one sweep) and
the corrector that records its right-hand side. The run, the rooms' solver
keys, the stopping block and the face hash are the probe's, so every number
here is taken the way D0's was. Records go to results/builder42/ and are
compared with the probe's records in results/builder41/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe41"))

import outlet41 as probe  # noqa: E402

from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.solver_staggered import StaggeredSolver  # noqa: E402
from src.staggered import allocate_fields  # noqa: E402

OUT = ROOT / "results" / "builder42"
PROBE_OUT = ROOT / "results" / "builder41"
probe.OUT = OUT


def built_room(
    name: str,
    nx: int,
    ny: int,
    mu_factor: float,
    n_outer: int,
    rtol: float,
    sweeps: int,
    stopping: dict | None,
) -> probe.Room:
    """The product room as the committed configuration builds it, wrapped for the probe's runner."""
    raw = probe.product_raw(nx, ny, mu_factor, n_outer, rtol, stopping)
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    solver = StaggeredSolver(mesh, cfg, boundary)
    solver._predictor = probe.SweepPredictor(mesh, cfg, boundary, sweeps)
    solver._corrector = probe.RecordingCorrector(mesh, cfg, boundary)
    speeds = boundary.fixed_flow_velocities()
    segments = []
    for segment_name, speed in speeds.items():
        edge = cfg.boundaries[segment_name].location
        mask = probe.segment_names(mesh, cfg, edge) == segment_name
        segments.append(probe.Segment(segment_name, edge, mask, "fixed", speed))
    meta = {
        "case": "product",
        "arm": "built",
        "nx": nx,
        "ny": ny,
        "mu_factor": mu_factor,
        "sweeps": sweeps,
        "pressure_rtol": rtol,
        "stopping": stopping or {"stopping_rule": "velocity_step"},
        "resolved_velocities": speeds,
    }
    return probe.Room(name, cfg, mesh, boundary, solver, segments, meta)


def drift(args: argparse.Namespace) -> None:
    """The drift case: 80x30, a thousand times air's viscosity, ten sweeps."""
    stopping = probe.stopping_for(80, 30, 8.0, 3.0, 1.2, 60.0)
    tag = f"drift_built_{args.rtol}"
    n_outer = 20000
    if args.long:
        stopping.update(iteration_error_tol=1e-14, mass_imbalance_tol=1e-16)
        tag += "_long"
        n_outer = 3000
    room = built_room(tag, 80, 30, 1000.0, n_outer, float(args.rtol), 10, stopping)
    probe.run(room)


def ladder(args: argparse.Namespace) -> None:
    """One rung of prompt 33b's ladder: 40x15, one sweep, velocity_step, to 3,000."""
    factor = probe.LADDER_FACTORS[args.rung]
    room = built_room(f"ladder_built_{args.rung}", 40, 15, factor, 3000, 1e-8, 1, None)
    probe.run(room, log_every=25)


def coverage(_args: argparse.Namespace) -> None:
    """The built faces against D0's, per segment and per face, on 80x30 and 40x15."""
    result = {}
    for nx, ny, d0_name in ((80, 30, "drift_D0_1e-8"), (40, 15, "ladder_D0_895")):
        room = built_room("coverage", nx, ny, 1.0, 1, 1e-8, 1, None)
        record = json.loads((PROBE_OUT / f"{d0_name}.json").read_text())
        u, v, _ = allocate_fields(room.mesh)
        room.boundary.apply_normal_velocity(u, v)
        split = record["split"]
        built = {s.name: int(s.mask.sum()) for s in room.segments}
        faces_equal = {}
        for seg in room.segments:
            written = probe.face_row(u, v, seg.edge)[seg.mask]
            wanted = probe.outward_value(
                0.5 if seg.name == "hood_exhaust" else split["fixed_speed"], seg.edge
            )
            faces_equal[seg.name] = bool(np.all(written == wanted))
        written_bottom = np.flatnonzero(v[0, :] != 0.0)
        written_right = np.flatnonzero(u[:, -1] != 0.0)
        probe_bottom = np.flatnonzero(
            np.any([s.mask for s in room.segments if s.edge == "bottom"], axis=0)
        )
        probe_right = np.flatnonzero(
            np.any([s.mask for s in room.segments if s.edge == "right"], axis=0)
        )
        result[f"{nx}x{ny}"] = {
            "faces_per_segment_built": built,
            "faces_per_segment_d0": dict(
                zip(record["segments"], record["faces_per_segment"], strict=True)
            ),
            "counts_equal": built
            == dict(zip(record["segments"], record["faces_per_segment"], strict=True)),
            "return_speed_built": room.boundary.fixed_flow_velocities()[
                "floor_return_1"
            ],
            "return_speed_d0": split["fixed_speed"],
            "return_speed_bitwise_equal": room.boundary.fixed_flow_velocities()[
                "floor_return_1"
            ]
            == split["fixed_speed"],
            "every_face_equals_d0_value": faces_equal,
            "written_faces_are_exactly_the_segments": bool(
                np.array_equal(written_bottom, probe_bottom)
                and np.array_equal(written_right, probe_right)
            ),
        }
    print(json.dumps(result, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage.json").write_text(json.dumps(result))


def compare(args: argparse.Namespace) -> None:
    """Stop, counts, residual history and face hash of a built record against a D0 record."""
    built = json.loads((OUT / f"{args.name}.json").read_text())
    base = json.loads((PROBE_OUT / f"{args.d0}.json").read_text())
    a, b = np.array(built["residual"]), np.array(base["residual"])
    common = min(len(a), len(b))
    relative = np.abs(a[:common] - b[:common]) / np.abs(b[:common])
    result = {
        "built": args.name,
        "d0": args.d0,
        "stop": [built["stop"], base["stop"]],
        "outer": [built["outer"], base["outer"]],
        "velocity_step_outer": [
            built["velocity_step_outer"],
            base["velocity_step_outer"],
        ],
        "residual_histories_bitwise_equal": built["residual"] == base["residual"],
        "largest_relative_residual_difference": float(relative.max()),
        "at_outer_iteration": int(relative.argmax()) + 1,
        "face_hash": [built["face_hash"], base["face_hash"]],
        "face_hash_equal": built["face_hash"] == base["face_hash"],
        "cg_iterations_equal": built["inner"] == base["inner"],
        "p_mean_equal": built["p_mean"] == base["p_mean"],
        "p_drift_last100": [built["p_drift_last100"], base["p_drift_last100"]],
    }
    print(json.dumps(result, indent=1))
    (OUT / f"compare_{args.name}_vs_{args.d0}.json").write_text(json.dumps(result))


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("coverage")
    p.set_defaults(func=coverage)
    p = sub.add_parser("drift")
    p.add_argument("rtol")
    p.add_argument("--long", action="store_true")
    p.set_defaults(func=drift)
    p = sub.add_parser("ladder")
    p.add_argument("rung", choices=sorted(probe.LADDER_FACTORS))
    p.set_defaults(func=ladder)
    p = sub.add_parser("compare")
    p.add_argument("name")
    p.add_argument("d0")
    p.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
