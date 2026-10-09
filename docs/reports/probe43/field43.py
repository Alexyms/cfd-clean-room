"""Demonstrations of prompt 43 (ECR-002 step 4): momentum with a viscosity field.

Usage:
    python field43.py val CASE [--zero] [--label L]      a VAL preset's stop, count and face hash
    python field43.py ladder RUNG [--label L]            40x15 at Re 895 or 8950, one sweep
    python field43.py drift RTOL [--long] [--label L]    80x30 drift case, momentum_sweeps 10
    python field43.py compare NAME D0_NAME               a record against a D0 record
    python field43.py frozen                             predict with mu_eff against frozen34.py
    python field43.py floor [--ny N] [--label L]         obstacle-floor channel against domain floor

The solver is the committed StaggeredSolver on the committed configuration;
nothing under src/ is subclassed for the physics. The product runs reuse the
measuring devices of docs/reports/probe41/outlet41.py (the corrector that
records its right-hand side, the runner, the stopping block and the face
hash), so every number is taken the way probe arm D0's was. Unlike
docs/reports/probe42/fixed42.py, the drift case's ten sweeps are the
committed `solver.momentum_sweeps: 10`, not the probe's SweepPredictor.
The script imports src/ from the tree it sits in, so a copy placed in a
worktree of another commit measures that commit. Records go to
results/builder43/ under that tree; D0's are read from results/builder41/ of
the main checkout (MAIN_RESULTS, found through git's common directory).

`val --zero` passes a field of zeros as `eddy_viscosity`, which the predictor
receives as a uniform field equal to air's viscosity. `frozen` compares
`MomentumPredictor.predict(u, v, p, mu_eff=mu + mu_t)` with frozen34.py's
FrozenPredictor (read from the committed report's appendix B by
tests/frozen34_reference.py) on three random states of the 40x15 product
room, at one sweep and at ten, and once more with NaN in every SOLID cell of
the field the built path receives. `floor` solves a channel whose floor is
one row of SOLID cells and the same channel with the domain edge as its
floor, both under one inlet and one outlet over the fluid rows, and reports
the largest velocity difference over the fluid faces.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
# The checkout the script was committed in: from a worktree copy, the main
# tree's results/ is still where D0's records live (git's common directory).
MAIN_RESULTS = (
    Path(
        subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    ).parent
    / "results"
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe41"))

import outlet41 as probe  # noqa: E402

from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.momentum import MomentumPredictor  # noqa: E402
from src.solver_staggered import StaggeredSolver  # noqa: E402
from validation.cases import load_preset  # noqa: E402

OUT = ROOT / "results" / "builder43"
PROBE_OUT = MAIN_RESULTS / "builder41"
probe.OUT = OUT


def face_hash(u: np.ndarray, v: np.ndarray) -> str:
    """Section 2.3 of docs/reports/ecr003_step2_baseline.md: SHA-256 over u's bytes then v's."""
    return hashlib.sha256(
        np.ascontiguousarray(u, dtype="<f8").tobytes()
        + np.ascontiguousarray(v, dtype="<f8").tobytes()
    ).hexdigest()


def write(name: str, record: dict) -> None:
    """Print a record and keep it as NAME.json under OUT."""
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps(record, indent=1))
    (OUT / f"{name}.json").write_text(json.dumps(record))


def val(args: argparse.Namespace) -> None:
    """A VAL preset through the committed solver, with or without a zero field."""
    config = load_preset(args.case)
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    count = [0]
    started = time.perf_counter()
    kwargs = {}
    if args.zero:
        kwargs["eddy_viscosity"] = np.zeros(mesh.cell_type.shape)
    solver.solve_steady(
        on_iteration=lambda _s: count.__setitem__(0, count[0] + 1), **kwargs
    )
    faces = solver.face_velocities
    assert faces is not None
    name = f"val_{args.case}" + ("_zero" if args.zero else "") + f"_{args.label}"
    write(
        name,
        {
            "case": args.case,
            "zero_field": args.zero,
            "label": args.label,
            "stop": solver.stop_reason,
            "outer": count[0],
            "face_hash": face_hash(faces.u, faces.v),
            "seconds": time.perf_counter() - started,
        },
    )


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
    """The product room as the committed configuration builds it, sweeps from the config."""
    raw = probe.product_raw(nx, ny, mu_factor, n_outer, rtol, stopping)
    raw["solver"]["momentum_sweeps"] = sweeps
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    solver = StaggeredSolver(mesh, cfg, boundary)
    solver._corrector = probe.RecordingCorrector(mesh, cfg, boundary)
    segments = []
    for segment_name, speed in boundary.fixed_flow_velocities().items():
        edge = cfg.boundaries[segment_name].location
        mask = probe.segment_names(mesh, cfg, edge) == segment_name
        segments.append(probe.Segment(segment_name, edge, mask, "fixed", speed))
    meta = {
        "case": "product",
        "arm": "built43",
        "nx": nx,
        "ny": ny,
        "mu_factor": mu_factor,
        "sweeps": sweeps,
        "momentum_sweeps_key": cfg.momentum_sweeps,
        "pressure_rtol": rtol,
        "stopping": stopping or {"stopping_rule": "velocity_step"},
    }
    return probe.Room(name, cfg, mesh, boundary, solver, segments, meta)


def drift(args: argparse.Namespace) -> None:
    """The drift case: 80x30, a thousand times air's viscosity, momentum_sweeps 10."""
    stopping = probe.stopping_for(80, 30, 8.0, 3.0, 1.2, 60.0)
    tag = f"drift_{args.label}_{args.rtol}"
    n_outer = 20000
    if args.long:
        stopping.update(iteration_error_tol=1e-14, mass_imbalance_tol=1e-16)
        tag += "_long"
        n_outer = 3000
    probe.run(built_room(tag, 80, 30, 1000.0, n_outer, float(args.rtol), 10, stopping))


def ladder(args: argparse.Namespace) -> None:
    """One rung of prompt 33b's ladder: 40x15, one sweep, velocity_step, to 3,000."""
    factor = probe.LADDER_FACTORS[args.rung]
    room = built_room(
        f"ladder_{args.label}_{args.rung}", 40, 15, factor, 3000, 1e-8, 1, None
    )
    probe.run(room, log_every=25)


def compare(args: argparse.Namespace) -> None:
    """Stop, counts, residual history and face hash of a record against a D0 record."""
    built = json.loads((OUT / f"{args.name}.json").read_text())
    base = json.loads((PROBE_OUT / f"{args.d0}.json").read_text())
    a, b = np.array(built["residual"]), np.array(base["residual"])
    common = min(len(a), len(b))
    relative = np.abs(a[:common] - b[:common]) / np.abs(b[:common])
    write(
        f"compare_{args.name}_vs_{args.d0}",
        {
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
            "face_hash": [built["face_hash"], base["face_hash"]],
            "face_hash_equal": built["face_hash"] == base["face_hash"],
            "cg_iterations_equal": built["inner"] == base["inner"],
            "p_drift_last100": [built["p_drift_last100"], base["p_drift_last100"]],
        },
    )


def product_40x15(sweeps: int) -> SimConfig:
    """The committed product configuration on the ladder's 40x15 grid."""
    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    raw["domain"]["nx"], raw["domain"]["ny"] = 40, 15
    raw["solver"]["alpha_velocity"] = 0.5
    raw["solver"]["momentum_sweeps"] = sweeps
    return SimConfig.from_dict(raw)


def frozen(_args: argparse.Namespace) -> None:
    """predict with mu_eff against frozen34.py's FrozenPredictor, face for face."""
    from tests.frozen34_reference import load_frozen_predictor

    reference = load_frozen_predictor()
    rows = []
    for sweeps in (1, 10):
        config = product_40x15(sweeps)
        mesh = Mesh(config)
        boundary = StaggeredBoundary(mesh, config)
        built = MomentumPredictor(mesh, config, boundary)
        solid = mesh.cell_type == SOLID
        rng = np.random.default_rng(4300 + sweeps)
        for state in range(3):
            u = rng.normal(0.0, 0.5, (mesh.yc.size, mesh.x.size))
            v = rng.normal(0.0, 0.5, (mesh.y.size, mesh.xc.size))
            p = rng.normal(0.0, 0.1, (mesh.yc.size, mesh.xc.size))
            boundary.apply_normal_velocity(u, v)
            # Four decades above air, so neighbours across an obstacle face
            # differ by up to 1e4.
            mu_t = config.mu * 10.0 ** rng.uniform(-2.0, 2.0, solid.shape)
            theirs = reference(mesh, config, boundary, mu_t, sweeps=sweeps).predict(
                u, v, p
            )
            field = config.mu + mu_t
            holes = np.where(solid, np.nan, field)
            for label, mu_eff in (("field", field), ("nan_in_solid", holes)):
                ours = built.predict(u, v, p, mu_eff=mu_eff)
                row = {"sweeps": sweeps, "state": state, "built_field": label}
                for name in ("u_star", "v_star", "a_p_u", "a_p_v"):
                    a, b = getattr(ours, name), getattr(theirs, name)
                    row[f"{name}_bitwise"] = bool(np.array_equal(a, b))
                    row[f"{name}_largest_difference"] = float(np.max(np.abs(a - b)))
                rows.append(row)
    write("frozen_parity", {"rows": rows})


def channel(floor: str, ny_fluid: int, nx: int) -> tuple[Mesh, StaggeredSolver]:
    """VAL-001's fluid and inflow in a 2 by 0.5 channel; floor 'edge' or 'solid'."""
    height, width = 0.5, 2.0
    dy = height / ny_fluid
    base = dy if floor == "solid" else 0.0
    raw = yaml.safe_load((ROOT / "configs/validation_poiseuille.yaml").read_text())
    raw["domain"].update(
        width=width,
        height=height + base,
        nx=nx,
        ny=ny_fluid + (1 if floor == "solid" else 0),
    )
    raw["boundaries"]["inlet"].update(y_start=base, y_end=height + base)
    raw["boundaries"]["outlet"].update(y_start=base, y_end=height + base)
    raw["obstacles"] = (
        [{"name": "floor", "x_start": 0.0, "x_end": width, "y_start": 0.0, "y_end": dy}]
        if floor == "solid"
        else []
    )
    raw["sensors"] = [{"name": "center", "x": width / 2, "y": base + height / 2}]
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    solver.solve_steady()
    return mesh, solver


def floor(args: argparse.Namespace) -> None:
    """The obstacle stencil's test: a SOLID floor row against the domain edge as floor."""
    ny_fluid = args.ny
    nx = 4 * ny_fluid
    _mesh_e, edge = channel("edge", ny_fluid, nx)
    mesh_s, solid = channel("solid", ny_fluid, nx)
    assert np.all(mesh_s.cell_type[0, :] == SOLID)
    assert not np.any(mesh_s.cell_type[1:, :] == SOLID)
    fe, fs = edge.face_velocities, solid.face_velocities
    assert fe is not None and fs is not None
    du = np.abs(fs.u[1:, :] - fe.u)
    dv = np.abs(fs.v[1:, :] - fe.v)
    row = int(np.unravel_index(np.argmax(du), du.shape)[0])
    write(
        f"floor_{args.label}" + ("" if ny_fluid == 20 else f"_{ny_fluid}"),
        {
            "label": args.label,
            "grid_fluid": [nx, ny_fluid],
            "stops": [edge.stop_reason, solid.stop_reason],
            "outer": [len(edge.residual_history), len(solid.residual_history)],
            "largest_u_difference": float(du.max()),
            "at_fluid_row": row,
            "largest_v_difference": float(dv.max()),
            "floor_row_u_difference": float(du[0].max()),
            # The developed half, x >= L/2, away from the inlet corner where
            # the near-wall gradient steepens with refinement.
            "developed_half_u_difference": float(du[:, nx // 2 :].max()),
            "developed_half_v_difference": float(dv[:, nx // 2 :].max()),
            "inflow_velocity": 0.1,
            "iteration_error_tol_times_inflow": 1e-6 * 0.1,
        },
    )


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("val")
    p.add_argument("case")
    p.add_argument("--zero", action="store_true")
    p.add_argument("--label", default="run")
    p.set_defaults(func=val)
    p = sub.add_parser("ladder")
    p.add_argument("rung", choices=sorted(probe.LADDER_FACTORS))
    p.add_argument("--label", default="run")
    p.set_defaults(func=ladder)
    p = sub.add_parser("drift")
    p.add_argument("rtol")
    p.add_argument("--long", action="store_true")
    p.add_argument("--label", default="run")
    p.set_defaults(func=drift)
    p = sub.add_parser("compare")
    p.add_argument("name")
    p.add_argument("d0")
    p.set_defaults(func=compare)
    p = sub.add_parser("frozen")
    p.set_defaults(func=frozen)
    p = sub.add_parser("floor")
    p.add_argument("--label", default="run")
    p.add_argument("--ny", type=int, default=20)
    p.set_defaults(func=floor)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
