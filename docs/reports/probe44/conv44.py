"""ECR-002 step 5 (prompt 44): the convergence measurement's rooms, fields and runner.

Usage:
    python conv44.py base GRID                       the Re 90 base field, ten sweeps, to the stop
    python conv44.py run GRID FIELD SWEEPS [--rtol R] [--corner-free] [--cap N] [--tag T] [--locate]
    python conv44.py cavity N                        the lid-driven cavity at Re 1,000 on N x N

GRID is 40x15, 80x30 or 200x75. FIELD is U1, U2, U3 (uniform effective
viscosities of 1.5e-3, 1.5e-4 and 6.5e-5 m^2/s), Z2, Z3 or Z4 (the
zero-equation field of docs/reports/ecr002_step0_frozen_viscosity.md built
on that grid's base field and scaled to core medians of 1.5e-3, 5e-4 and
1.5e-4 m^2/s), or L (laminar, no field). SWEEPS is the committed
`solver.momentum_sweeps` key.

Every run goes through the committed StaggeredSolver on the committed
product configuration (configs/clean_room_default.yaml regridded through
outlet41.product_raw, the ladder's solver keys, error_estimate with
stopping_for's tolerances) and `solve_steady(eddy_viscosity=...)`. Nothing
under src/ is edited or patched. The measuring devices are the probe's
corrector that records its right-hand side (outlet41.RecordingCorrector),
the face hash, and a hook that keeps the solver's stopping rule so its
estimate history can be recorded. `--corner-free` is measurement 4's one
counterfactual: the predictor is tail43.corner_free_class(), the committed
`_deferred_correction` with the two lines that zero the QUICK correction at
obstacle faces removed. `--locate` adds to the record the cell of the
largest change of each component between successive iterates (round 2,
prompt 44b item 4). Records go to results/builder44/ as NAME.json and
NAME.npz; the base field of each grid to base_GRID.npz and .json.

One BLAS thread: the CG solve runs under src/pressure.py's limit, and this
process sets OPENBLAS_NUM_THREADS=1 before NumPy loads as well; the record
carries threadpoolctl's view of the loaded BLAS.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_info

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe41"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe43"))

import outlet41 as probe  # noqa: E402
import tail43  # noqa: E402

from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.solver_staggered import StaggeredSolver  # noqa: E402
from src.staggered import FaceVelocities  # noqa: E402
from src.stopping import ErrorEstimateRule, IterationState  # noqa: E402
from validation.metrics import cavity_true_centerline_profiles  # noqa: E402

OUT = ROOT / "results" / "builder44"
GRIDS = {"40x15": (40, 15), "80x30": (80, 30), "200x75": (200, 75)}
CAPS = {"40x15": 5000, "80x30": 5000, "200x75": 10000}
# Effective kinematic viscosity each field's core carries, m^2/s. The uniform
# fields' eddy viscosity is this less air's; the Z fields' core median of the
# eddy viscosity is this (step 0's convention, section 2.5 there).
TARGETS = {
    "U1": 1.5e-3,
    "U2": 1.5e-4,
    "U3": 6.5e-5,
    "Z2": 1.5e-3,
    "Z3": 5e-4,
    "Z4": 1.5e-4,
}
BASE_MU_FACTOR = 1000.0
BASE_SWEEPS = 10
CHEN_XU = 0.03874
DIVERGED_SPEED = 100.0
CAVITY_VISCOSITY = 1e-3
CAVITY_CAP = 40000
PRODUCT_T_END = 60.0


class DivergedError(Exception):
    """Raised from the callback to end a run whose largest speed passed the bound."""


def commit() -> str:
    """The checked-out commit, so every record names the tree it measured."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def blas_threads() -> list[dict]:
    """threadpoolctl's view of the loaded BLAS libraries and their thread counts."""
    return [
        {
            k: v
            for k, v in info.items()
            if k in ("internal_api", "version", "num_threads")
        }
        for info in threadpool_info()
        if info.get("user_api") == "blas"
    ]


# ---------------------------------------------------------------------------
# Rooms
# ---------------------------------------------------------------------------


def product_config(
    grid: str, mu_factor: float, n_outer: int, rtol: float, sweeps: int
) -> tuple[dict, SimConfig]:
    """The product room on a grid, the ladder's keys, error_estimate with the grid's tolerances."""
    nx, ny = GRIDS[grid]
    raw0 = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    width, height = raw0["domain"]["width"], raw0["domain"]["height"]
    stopping = probe.stopping_for(
        nx, ny, width, height, raw0["fluid"]["density"], PRODUCT_T_END
    )
    raw = probe.product_raw(nx, ny, mu_factor, n_outer, rtol, stopping)
    raw["solver"]["momentum_sweeps"] = sweeps
    return raw, SimConfig.from_dict(raw)


def build_room(
    name: str, cfg: SimConfig, meta: dict, corner_free: bool = False
) -> probe.Room:
    """The committed solver on a configuration, with the recording corrector in place."""
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    solver = StaggeredSolver(mesh, cfg, boundary)
    solver._corrector = probe.RecordingCorrector(mesh, cfg, boundary)
    if corner_free:
        solver._predictor = tail43.corner_free_class()(mesh, cfg, boundary)
    segments = []
    for segment_name, speed in boundary.fixed_flow_velocities().items():
        edge = cfg.boundaries[segment_name].location
        mask = probe.segment_names(mesh, cfg, edge) == segment_name
        segments.append(probe.Segment(segment_name, edge, mask, "fixed", speed))
    return probe.Room(name, cfg, mesh, boundary, solver, segments, meta)


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


def staircase_distance(mesh: Mesh) -> np.ndarray:
    """Distance from each cell centre to the nearest domain edge or SOLID cell, [ny, nx].

    base34.wall_distance with staircase=True: every SOLID cell is a box.
    """
    xc, yc = np.meshgrid(mesh.xc, mesh.yc)
    width, height = float(mesh.x[-1]), float(mesh.y[-1])
    dist = np.minimum.reduce([xc, width - xc, yc, height - yc])
    solid = mesh.cell_type == SOLID
    zero = np.zeros_like(xc)
    for j, i in zip(*np.nonzero(solid), strict=True):
        x0, x1, y0, y1 = mesh.x[i], mesh.x[i + 1], mesh.y[j], mesh.y[j + 1]
        dx = np.maximum.reduce([x0 - xc, zero, xc - x1])
        dy = np.maximum.reduce([y0 - yc, zero, yc - y1])
        dist = np.minimum(dist, np.hypot(dx, dy))
    return dist


def core_mask(mesh: Mesh, raw: dict) -> np.ndarray:
    """Non-SOLID cells whose centre lies above the equipment tops."""
    tops = max(o["y_end"] for o in raw["obstacles"])
    _, yc = np.meshgrid(mesh.xc, mesh.yc)
    return (mesh.cell_type != SOLID) & (yc > tops)


def percentiles(a: np.ndarray) -> list[float]:
    """The 5th, 25th, 50th, 75th and 95th percentiles."""
    return [float(np.percentile(a, q)) for q in (5, 25, 50, 75, 95)]


def eddy_field(
    grid: str, field: str, mesh: Mesh, cfg: SimConfig
) -> tuple[np.ndarray | None, dict]:
    """The run's eddy_viscosity (kinematic, [ny, nx]) and how it was made."""
    nu_air = cfg.mu / cfg.rho
    if field == "L":
        return None, {"kind": "laminar", "nu_air": nu_air}
    target = TARGETS[field]
    if field.startswith("U"):
        value = target - nu_air
        return np.full(mesh.cell_type.shape, value), {
            "kind": "uniform",
            "nu_air": nu_air,
            "effective": target,
            "eddy_viscosity": value,
        }
    base = np.load(OUT / f"base_{grid}.npz")
    base_json = json.loads((OUT / f"base_{grid}.json").read_text())
    if base_json["stop"] != "error_estimate_and_continuity":
        raise ValueError(
            f"the {grid} base field did not converge ({base_json['stop']})"
        )
    nut0 = base["nut0"]
    core = base["core"]
    scale = target / float(np.median(nut0[core]))
    nut = scale * nut0
    non_solid = mesh.cell_type != SOLID
    return nut, {
        "kind": "zero_equation",
        "nu_air": nu_air,
        "base": f"base_{grid}",
        "base_commit": base_json["commit"],
        "scale": scale,
        "core_median_eddy": float(np.median(nut[core])),
        "core_median_effective": float(np.median(nut[core])) + nu_air,
        "eddy_percentiles_core": percentiles(nut[core]),
        "eddy_percentiles_all": percentiles(nut[non_solid]),
        "eddy_max": float(nut[non_solid].max()),
    }


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def run_room(
    room: probe.Room,
    eddy: np.ndarray | None,
    log_every: int = 100,
    locate: bool = False,
) -> dict:
    """Run a room to its stop, divergence or cap; write NAME.json and NAME.npz.

    With locate, the record also carries, per outer iteration, the cell of
    the largest change of each cell-centred component between successive
    iterates (the change the residual measures) and that change in m/s.
    The commit is read when the run starts, so the record names the tree
    it measured.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    tree = commit()
    solver = room.solver
    corrector: probe.RecordingCorrector = solver._corrector  # type: ignore[assignment]
    mesh = room.mesh
    not_solid = mesh.cell_type != SOLID
    xc, yc = np.meshgrid(mesh.xc, mesh.yc)
    tol = room.cfg.convergence_tol
    started = datetime.now().isoformat(timespec="seconds")
    print(f"{room.name} start {started} {json.dumps(room.meta)}", flush=True)

    # The stopping rule is local to solve_steady; keep the one it builds so
    # its estimate history can be recorded.
    held: dict = {}
    new_rule = solver._new_rule

    def keep_rule() -> ErrorEstimateRule | None:
        rule = new_rule()
        held["rule"] = rule
        return rule

    solver._new_rule = keep_rule  # type: ignore[method-assign]

    rec: dict[str, list] = {
        k: [] for k in ("residual", "inner", "cap", "max_speed", "at")
    }
    if locate:
        rec.update({k: [] for k in ("du_at", "du", "dv_at", "dv")})
    previous: dict[str, np.ndarray] = {}
    vs_stop: dict[str, int] = {}
    last: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()

    def callback(state: IterationState) -> None:
        speed = np.hypot(state.u, state.v)
        speed[~not_solid] = 0.0
        k = int(np.argmax(speed))
        top = float(speed.flat[k])
        record = corrector.records[-1]
        rec["residual"].append(float(state.residual))
        rec["inner"].append(int(record.iterations))
        rec["cap"].append(bool(record.reached_cap))
        rec["max_speed"].append(top)
        rec["at"].append((round(float(xc.flat[k]), 4), round(float(yc.flat[k]), 4)))
        if locate:
            for comp, field in (("u", state.u), ("v", state.v)):
                change = (
                    np.abs(field - previous[comp])
                    if comp in previous
                    else np.abs(field)
                )
                change[~not_solid] = 0.0
                kc = int(np.argmax(change))
                rec[f"d{comp}_at"].append(
                    (round(float(xc.flat[kc]), 4), round(float(yc.flat[kc]), 4))
                )
                rec[f"d{comp}"].append(float(change.flat[kc]))
            previous["u"], previous["v"] = state.u.copy(), state.v.copy()
        last["u_c"], last["v_c"], last["p"] = (
            state.u.copy(),
            state.v.copy(),
            state.p.copy(),
        )
        if (
            "velocity_step" not in vs_stop
            and state.residual < tol
            and not record.reached_cap
        ):
            vs_stop["velocity_step"] = state.iteration + 1
        it = state.iteration
        if it % log_every == 0:
            print(
                f"{room.name} it {it:5d} res {state.residual:.3e} inner {record.iterations:5d} "
                f"max|U| {top:.4g} at {rec['at'][-1]} t {time.perf_counter() - t0:8.1f}s",
                flush=True,
            )
        if not math.isfinite(top) or top > DIVERGED_SPEED:
            raise DivergedError

    try:
        if eddy is None:
            solver.solve_steady(on_iteration=callback)
        else:
            solver.solve_steady(on_iteration=callback, eddy_viscosity=eddy)
        stop = solver.stop_reason
    except DivergedError:
        stop = "diverged"
    seconds = time.perf_counter() - t0
    n = len(rec["residual"])
    u_f, v_f = corrector.last_u, corrector.last_v
    assert u_f is not None and v_f is not None
    faces = solver.face_velocities
    if faces is None:
        # A diverged run leaves the loop before the solver keeps its faces;
        # the last correction's are the ones to hash.
        faces = FaceVelocities.copy_of(u_f, v_f)
    imbalance = corrector.mass_imbalance(u_f, v_f)
    rule = held.get("rule")
    estimate = list(rule.estimate_history) if rule is not None else []
    out = {
        "name": room.name,
        "commit": tree,
        "started": started,
        "seconds": seconds,
        "seconds_per_outer": seconds / max(n, 1),
        "stage_seconds": solver.stage_seconds,
        "blas": blas_threads(),
        "openblas_num_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
        **room.meta,
        "segments": [s.name for s in room.segments],
        "faces_per_segment": [int(s.mask.sum()) for s in room.segments],
        "stop": stop,
        "outer": n,
        "velocity_step_outer": vs_stop.get("velocity_step"),
        "residual_min": float(np.min(rec["residual"])),
        "residual_min_at": int(np.argmin(rec["residual"])) + 1,
        "residual_end": rec["residual"][-1],
        "max_speed_end": rec["max_speed"][-1],
        "at_end": rec["at"][-1],
        "max_speed_peak": float(np.max(rec["max_speed"])),
        "worst_end": float(np.max(np.abs(imbalance))),
        "signed_end": float(np.sum(imbalance)),
        "cap_hits": int(sum(rec["cap"])),
        "inner_mean": float(np.mean(rec["inner"])),
        "inner_median": float(np.median(rec["inner"])),
        "inner_max": int(np.max(rec["inner"])),
        "inner_end": int(rec["inner"][-1]),
        "estimate_end": estimate[-1] if estimate else None,
        "face_hash": probe.face_hash(faces.u, faces.v),
        "faces_match_last_correction": bool(
            np.array_equal(faces.u, u_f) and np.array_equal(faces.v, v_f)
        ),
        "estimate": [e if math.isfinite(e) else None for e in estimate],
        **rec,
    }
    (OUT / f"{room.name}.json").write_text(json.dumps(out))
    np.savez(
        OUT / f"{room.name}.npz",
        u_faces=faces.u,
        v_faces=faces.v,
        p=last["p"],
        u_c=last["u_c"],
        v_c=last["v_c"],
        xc=mesh.xc,
        yc=mesh.yc,
        cell_type=mesh.cell_type,
    )
    print(
        f"{room.name} done {datetime.now().isoformat(timespec='seconds')} stop {stop} after {n} "
        f"(velocity_step at {out['velocity_step_outer']}); res min {out['residual_min']:.3e} "
        f"end {out['residual_end']:.3e}; max|U| {out['max_speed_end']:.4g} at {out['at_end']}; "
        f"CG mean {out['inner_mean']:.0f} max {out['inner_max']} cap hits {out['cap_hits']}; "
        f"{seconds:.0f}s ({out['seconds_per_outer']:.3f} s/outer); hash {out['face_hash'][:16]}",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def base(args: argparse.Namespace) -> None:
    """The Re 90 base field of a grid, and the zero-equation field made from it."""
    grid = args.grid
    raw, cfg = product_config(grid, BASE_MU_FACTOR, CAPS[grid], 1e-8, BASE_SWEEPS)
    meta = {
        "case": "product",
        "grid": grid,
        "field": "base",
        "mu_factor": BASE_MU_FACTOR,
        "sweeps": BASE_SWEEPS,
        "pressure_rtol": 1e-8,
        "stopping": raw["solver"]["stopping_rule"],
        "iteration_error_tol": cfg.iteration_error_tol,
        "mass_imbalance_tol": cfg.mass_imbalance_tol,
        "cap": CAPS[grid],
    }
    room = build_room(f"base_{grid}", cfg, meta)
    out = run_room(room, None)
    mesh = room.mesh
    kept = np.load(OUT / f"base_{grid}.npz")
    speed = np.hypot(kept["u_c"], kept["v_c"])
    dist = staircase_distance(mesh)
    non_solid = mesh.cell_type != SOLID
    nut0 = CHEN_XU * speed * dist
    nut0[~non_solid] = 0.0
    core = core_mask(mesh, raw)
    nu_air = cfg.mu / cfg.rho / BASE_MU_FACTOR
    summary = {
        "fluid_cells": int(non_solid.sum()),
        "core_cells": int(core.sum()),
        "nu_air": nu_air,
        "nut0_core": percentiles(nut0[core]),
        "nut0_all": percentiles(nut0[non_solid]),
        "nut0_max": float(nut0[non_solid].max()),
        "core_median_over_nu_air": float(np.median(nut0[core]) / nu_air),
        "speed_all": percentiles(speed[non_solid]),
        "L_all": percentiles(dist[non_solid]),
        "scales": {
            f: TARGETS[f] / float(np.median(nut0[core])) for f in ("Z2", "Z3", "Z4")
        },
    }
    out.update(summary)
    (OUT / f"base_{grid}.json").write_text(json.dumps(out))
    np.savez(
        OUT / f"base_{grid}.npz",
        **{k: kept[k] for k in kept.files},
        nut0=nut0,
        L=dist,
        speed=speed,
        core=core,
    )
    print(f"base_{grid} field: {json.dumps(summary)}", flush=True)


def run(args: argparse.Namespace) -> None:
    """One row of the matrix, or one counterfactual of measurements 4 and 5."""
    grid, field, sweeps, rtol = args.grid, args.field, args.sweeps, float(args.rtol)
    cap = args.cap or CAPS[grid]
    name = f"{field}_{grid}_s{sweeps}"
    if rtol != 1e-8:
        name += f"_r{args.rtol}"
    if args.corner_free:
        name += "_cf"
    if args.locate:
        name += "_loc"
    if args.tag:
        name += f"_{args.tag}"
    raw, cfg = product_config(grid, 1.0, cap, rtol, sweeps)
    mesh = Mesh(cfg)
    eddy, how = eddy_field(grid, field, mesh, cfg)
    meta = {
        "case": "product",
        "grid": grid,
        "field": field,
        "field_made": how,
        "mu_factor": 1.0,
        "sweeps": sweeps,
        "momentum_sweeps_key": cfg.momentum_sweeps,
        "pressure_rtol": rtol,
        "corner_free": bool(args.corner_free),
        "locate": bool(args.locate),
        "stopping": raw["solver"]["stopping_rule"],
        "iteration_error_tol": cfg.iteration_error_tol,
        "mass_imbalance_tol": cfg.mass_imbalance_tol,
        "velocity_scale": 0.45,
        "cap": cap,
    }
    room = build_room(name, cfg, meta, corner_free=args.corner_free)
    run_room(room, eddy, log_every=args.log_every, locate=args.locate)


def extremes(positions: list[float], values: list[float]) -> dict:
    """The discrete extremes of a profile and the vertex of the parabola through each."""

    def vertex(k: int) -> tuple[float, float]:
        if k == 0 or k == len(values) - 1:
            return positions[k], values[k]
        x0, x1, x2 = positions[k - 1], positions[k], positions[k + 1]
        y0, y1, y2 = values[k - 1], values[k], values[k + 1]
        a, b, c = np.polyfit([x0, x1, x2], [y0, y1, y2], 2)
        if a == 0.0:
            return x1, y1
        xv = -b / (2.0 * a)
        return float(xv), float(a * xv * xv + b * xv + c)

    k_min, k_max = int(np.argmin(values)), int(np.argmax(values))
    return {
        "min": {
            "value": values[k_min],
            "at": positions[k_min],
            "parabola": vertex(k_min),
        },
        "max": {
            "value": values[k_max],
            "at": positions[k_max],
            "parabola": vertex(k_max),
        },
    }


def cavity(args: argparse.Namespace) -> None:
    """The lid-driven cavity at Re 1,000 on N x N, VAL-002's stopping keys, cap 40,000."""
    n = args.n
    raw = yaml.safe_load((ROOT / "configs/validation_cavity.yaml").read_text())
    raw["domain"]["nx"], raw["domain"]["ny"] = n, n
    raw["fluid"]["viscosity"] = CAVITY_VISCOSITY
    raw["solver"]["max_simple_iter"] = CAVITY_CAP
    cfg = SimConfig.from_dict(raw)
    reynolds = (
        cfg.rho
        * raw["boundaries"]["lid"]["u_velocity"]
        * raw["domain"]["width"]
        / cfg.mu
    )
    meta = {
        "case": "cavity",
        "grid": f"{n}x{n}",
        "field": "L",
        "reynolds": reynolds,
        "viscosity": cfg.mu,
        "sweeps": cfg.momentum_sweeps,
        "pressure_rtol": cfg.pressure_rtol,
        "alpha_velocity": cfg.alpha_velocity,
        "stopping": cfg.stopping_rule,
        "iteration_error_tol": cfg.iteration_error_tol,
        "mass_imbalance_tol": cfg.mass_imbalance_tol,
        "cap": CAVITY_CAP,
    }
    room = build_room(f"cavity_{n}x{n}", cfg, meta)
    out = run_room(room, None, log_every=args.log_every)
    kept = np.load(OUT / f"cavity_{n}x{n}.npz")
    y, u, x, v = cavity_true_centerline_profiles(
        cfg, room.mesh, kept["u_c"], kept["v_c"]
    )
    out["centerline"] = {
        "u_on_vertical": extremes(list(map(float, y)), list(map(float, u))),
        "v_on_horizontal": extremes(list(map(float, x)), list(map(float, v))),
    }
    (OUT / f"cavity_{n}x{n}.json").write_text(json.dumps(out))
    print(f"cavity_{n}x{n} centerline: {json.dumps(out['centerline'])}", flush=True)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("base")
    p.add_argument("grid", choices=sorted(GRIDS))
    p.set_defaults(func=base)
    p = sub.add_parser("run")
    p.add_argument("grid", choices=sorted(GRIDS))
    p.add_argument("field", choices=[*sorted(TARGETS), "L"])
    p.add_argument("sweeps", type=int)
    p.add_argument("--rtol", default="1e-8")
    p.add_argument("--corner-free", action="store_true")
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--locate", action="store_true")
    p.set_defaults(func=run)
    p = sub.add_parser("cavity")
    p.add_argument("n", type=int)
    p.add_argument("--log-every", type=int, default=500)
    p.set_defaults(func=cavity)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
