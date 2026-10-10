"""ECR-002 step 6 (prompt 46): the product room under the coupled k-epsilon model.

Usage:
    python coupled46.py run GRID VARIANT [--rtol R] [--intensity I] [--length L]
                            [--cap N] [--tag T] [--locate] [--log-every N]
    python coupled46.py time GRID        a 20-iteration timing probe, standard variant

GRID is 40x15, 80x30 or 200x75; VARIANT is standard or rng. Every run goes
through the committed StaggeredSolver on the committed product configuration
(configs/clean_room_default.yaml regridded, with the turbulence section and
the solver keys of the report's section 2.1 set in the probe) and
`solve_steady()`, from rest. Nothing under src/ is edited or patched. The
measuring devices are a recording subclass of ErrorEstimateRule that asks
the corrector for the imbalance every outer iteration (VAL-016's
couette45.py did the same; the rule's decision is unchanged), a hook that
keeps the solver's stopping rule so its histories can be read, and, with
--locate, a wrapper on the solver's turbulence step that keeps the latest
state so the cell of the largest nu_t change can be recorded. Records go to
results/builder46/ as NAME.json and NAME.npz.

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

import outlet41 as probe  # noqa: E402

from src import solver_staggered  # noqa: E402
from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.stopping import ErrorEstimateRule, ImbalanceSummary, IterationState  # noqa: E402
from src.turbulence import (  # noqa: E402
    VARIANTS,
    Y_STAR_FLOOR,
    PositivityError,
    TurbulenceBoundary,
    TurbulenceState,
)

OUT = ROOT / "results" / "builder46"
GRIDS = {"40x15": (40, 15), "80x30": (80, 30), "200x75": (200, 75)}
CAP = 10000
SWEEPS = 10
ALPHA_VELOCITY = 0.5
PRESSURE_RTOL = 1e-4
INTENSITY = 0.05
DISSIPATION_LENGTH = 0.1
# VAL-016's turbulence keys (tests/couette_reference.py, case_raw) at the
# Courant number every VAL-016 run used (ADR-012 C's note of 2026-10-09).
TURBULENCE = {
    "model": "k_epsilon",
    "wall_treatment": "scalable_wall_functions",
    "cfl_number": 0.25,
    "alpha_turbulence": 0.7,
    "max_iter": 500,
    "tol": 1e-10,
}
DIVERGED_SPEED = 100.0
TIME_CAP = 20


class DivergedError(Exception):
    """Raised from the callback to end a run whose largest speed passed the bound."""


class RecordingRule(ErrorEstimateRule):
    """The committed rule, asking for the imbalance every outer iteration.

    The rule itself asks only when (a) and (e) hold; here every iteration's
    readings are kept so the record can say from which iteration each of
    the five conditions holds. The decision is the parent's on the same
    readings.
    """

    instances: list["RecordingRule"] = []

    def __init__(self, *args: float, **kwargs: float | None) -> None:
        super().__init__(*args, **kwargs)
        self.readings: list[ImbalanceSummary] = []
        self.steps: list[float] = []
        self.viscosity_steps: list[float] = []
        RecordingRule.instances.append(self)

    def update(
        self,
        step: float,
        imbalance: object,
        viscosity_step: float | None = None,
    ) -> bool:
        found = imbalance()  # type: ignore[operator]
        self.readings.append(found)
        self.steps.append(float(step))
        self.viscosity_steps.append(
            float("nan") if viscosity_step is None else float(viscosity_step)
        )
        return super().update(step, lambda: found, viscosity_step=viscosity_step)


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
# The room
# ---------------------------------------------------------------------------


def product_raw(
    grid: str,
    variant: str,
    rtol: float,
    intensity: float,
    length: float,
    cap: int,
) -> dict:
    """The product configuration mapping of the report's section 2.1."""
    nx, ny = GRIDS[grid]
    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    width, height = raw["domain"]["width"], raw["domain"]["height"]
    block = raw["solver"]
    block.update(
        probe.stopping_for(
            nx, ny, width, height, raw["fluid"]["density"], block["t_end"]
        )
    )
    block["max_simple_iter"] = cap
    block["alpha_velocity"] = ALPHA_VELOCITY
    block["pressure_rtol"] = rtol
    block["momentum_sweeps"] = SWEEPS
    raw["turbulence"] = {**TURBULENCE, "variant": variant}
    supply = raw["boundaries"]["hepa_supply"]
    supply["turbulence_intensity"] = intensity
    supply["dissipation_length"] = length
    return raw


def core_mask(mesh: Mesh, raw: dict) -> np.ndarray:
    """Non-SOLID cells whose centre lies above the equipment tops (step 5's core)."""
    tops = max(o["y_end"] for o in raw["obstacles"])
    _, yc = np.meshgrid(mesh.xc, mesh.yc)
    return (mesh.cell_type != SOLID) & (yc > tops)


def percentiles(a: np.ndarray) -> list[float]:
    """The 5th, 25th, 50th, 75th and 95th percentiles."""
    return [float(np.percentile(a, q)) for q in (5, 25, 50, 75, 95)]


def wall_y_star(
    walls: TurbulenceBoundary, k: np.ndarray, c_mu: float, nu: float
) -> tuple[dict[str, np.ndarray], dict]:
    """y* of every wall node, per wall side, and its statistics.

    The quantity the wall functions evaluate in the wall cells (ADR-012 B,
    `TurbulenceBoundary._wall_cell_values`): ``y* = C_mu^(1/4) k_P^(1/2)
    y_P / nu`` with k_P the cell's k and y_P the distance from its centre to
    that wall. The sides are the boundary's own (south, north, west, east),
    read from its `_sides`, a probe-side read of a private attribute. A
    south wall on the domain's bottom row is the floor and elsewhere an
    obstacle top; a west or east wall on the domain's edge is a domain wall
    and elsewhere an obstacle side; a north wall on the top row is the
    ceiling and elsewhere an obstacle underside.
    """
    ny, nx = k.shape
    fields: dict[str, np.ndarray] = {}
    kinds: dict[str, np.ndarray] = {}
    u_k = c_mu**0.25 * np.sqrt(np.where(k > 0.0, k, np.nan))
    names = ("south", "north", "west", "east")
    rows = np.arange(ny)[:, None] * np.ones((1, nx), dtype=int)
    cols = np.ones((ny, 1), dtype=int) * np.arange(nx)[None, :]
    for name, (mask, distance, _moving, _component) in zip(
        names, walls._sides, strict=True
    ):
        y_star = np.where(mask, u_k * distance / nu, np.nan)
        fields[name] = y_star
        if name == "south":
            kind = np.where(rows == 0, "domain", "obstacle top")
        elif name == "north":
            kind = np.where(rows == ny - 1, "domain", "obstacle underside")
        elif name == "west":
            kind = np.where(cols == 0, "domain", "obstacle side")
        else:
            kind = np.where(cols == nx - 1, "domain", "obstacle side")
        kinds[name] = np.where(mask, kind, "")
    values = np.concatenate([f[np.isfinite(f)] for f in fields.values()])
    kind_values = np.concatenate(
        [kinds[n][np.isfinite(fields[n])] for n in names]
    )
    stats = {
        "nodes": int(values.size),
        "median": float(np.median(values)),
        "least": float(values.min()),
        "largest": float(values.max()),
        "percentiles": percentiles(values),
        "floor": Y_STAR_FLOOR,
        "share_below_floor": float(np.mean(values < Y_STAR_FLOOR)),
        "by_kind": {},
    }
    for kind in ("domain", "obstacle top", "obstacle side", "obstacle underside"):
        sel = values[kind_values == kind]
        if sel.size == 0:
            continue
        stats["by_kind"][kind] = {
            "nodes": int(sel.size),
            "median": float(np.median(sel)),
            "least": float(sel.min()),
            "largest": float(sel.max()),
            "share_below_floor": float(np.mean(sel < Y_STAR_FLOOR)),
        }
    return fields, stats


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def run_room(
    name: str,
    raw: dict,
    meta: dict,
    log_every: int = 100,
    locate: bool = False,
) -> dict:
    """Run the room to its stop, divergence, positivity error or cap; write the record.

    The commit is read when the run starts, so the record names the tree
    it measured. The recording rule is substituted on the module before
    the solver is built, as couette45.py did, so the solver's own
    `_new_rule` makes it with the solver's scales.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    tree = commit()
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    solver_staggered.ErrorEstimateRule = RecordingRule  # type: ignore[misc]
    RecordingRule.instances.clear()
    solver = solver_staggered.StaggeredSolver(mesh, cfg, boundary)
    model, walls = solver._model, solver._walls
    assert model is not None and walls is not None
    not_solid = mesh.cell_type != SOLID
    xc, yc = np.meshgrid(mesh.xc, mesh.yc)
    nu = cfg.mu / cfg.rho
    started = datetime.now().isoformat(timespec="seconds")
    print(f"{name} start {started} {json.dumps(meta)}", flush=True)

    latest: dict[str, TurbulenceState] = {}
    if locate:
        real_step = solver._turbulence_step

        def keep_state(
            state: TurbulenceState, u: np.ndarray, v: np.ndarray, iteration: int
        ) -> TurbulenceState:
            stepped = real_step(state, u, v, iteration)
            latest["state"] = stepped
            return stepped

        solver._turbulence_step = keep_state  # type: ignore[method-assign]

    rec: dict[str, list] = {
        k: []
        for k in (
            "residual",
            "inner",
            "cap",
            "max_speed",
            "at",
            "k_sweeps",
            "eps_sweeps",
            "scalar_converged",
        )
    }
    if locate:
        rec.update({k: [] for k in ("du_at", "du", "dv_at", "dv", "dnu_at", "dnu")})
    previous: dict[str, np.ndarray] = {}
    last: dict[str, np.ndarray] = {}
    cap_hits = {"seen": 0}
    t0 = time.perf_counter()

    def callback(state: IterationState) -> None:
        speed = np.hypot(state.u, state.v)
        speed[~not_solid] = 0.0
        k = int(np.argmax(speed))
        top = float(speed.flat[k])
        reached_cap = solver.pressure_cap_hits > cap_hits["seen"]
        cap_hits["seen"] = solver.pressure_cap_hits
        rec["residual"].append(float(state.residual))
        rec["inner"].append(int(state.pressure_iterations))
        rec["cap"].append(bool(reached_cap))
        rec["max_speed"].append(top)
        rec["at"].append((round(float(xc.flat[k]), 4), round(float(yc.flat[k]), 4)))
        rec["k_sweeps"].append(int(model.last_sweeps[0]))
        rec["eps_sweeps"].append(int(model.last_sweeps[1]))
        rec["scalar_converged"].append(bool(model.solves_converged))
        if locate:
            fields = {"u": state.u, "v": state.v, "nu": latest["state"].nu_t}
            for comp, field in fields.items():
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
                previous[comp] = field.copy()
        last["u_c"], last["v_c"], last["p"] = (
            state.u.copy(),
            state.v.copy(),
            state.p.copy(),
        )
        it = state.iteration
        if it % log_every == 0:
            print(
                f"{name} it {it:5d} res {state.residual:.3e} inner {state.pressure_iterations:5d} "
                f"k/eps sweeps {model.last_sweeps} max|U| {top:.4g} at {rec['at'][-1]} "
                f"t {time.perf_counter() - t0:8.1f}s",
                flush=True,
            )
        if not math.isfinite(top) or top > DIVERGED_SPEED:
            raise DivergedError

    positivity: dict | None = None
    try:
        solver.solve_steady(on_iteration=callback)
        stop = solver.stop_reason
    except DivergedError:
        stop = "diverged"
    except PositivityError as err:
        stop = "positivity_error"
        positivity = {
            "message": str(err),
            "minimum": err.minimum,
            "outer_iteration_zero_based": len(rec["residual"]),
            "pressure_cap_hits_before": solver.pressure_cap_hits,
            "cap_hit_iterations": [i for i, c in enumerate(rec["cap"]) if c],
        }
        print(f"{name} POSITIVITY {err}", flush=True)
    seconds = time.perf_counter() - t0
    n = len(rec["residual"])
    rule = RecordingRule.instances[-1] if RecordingRule.instances else None
    tols = (cfg.iteration_error_tol, cfg.mass_imbalance_tol)
    flux = solver.flux_scale or float("nan")
    holds: dict[str, list[bool]] = {}
    readings: dict[str, list[float]] = {}
    if rule is not None:
        est = list(rule.estimate_history)
        nu_est = list(rule.viscosity_estimate_history)
        readings = {
            "a": [e if math.isfinite(e) else None for e in est],
            "e": [e if math.isfinite(e) else None for e in nu_est],
            "b": [r.worst for r in rule.readings],
            "c": [r.absolute_sum / flux for r in rule.readings],
            "d": [abs(r.signed_sum) for r in rule.readings],
            "velocity_step": rule.steps,
            "viscosity_step": [
                s if math.isfinite(s) else None for s in rule.viscosity_steps
            ],
        }
        holds = {
            "a": [e < tols[0] for e in est],
            "e": [e < tols[0] for e in nu_est],
            "b": [r.worst < tols[1] for r in rule.readings],
            "c": [r.absolute_sum / flux < tols[0] for r in rule.readings],
            "d": [abs(r.signed_sum) < tols[1] for r in rule.readings],
        }

    def from_when(flags: list[bool]) -> int | None:
        """The first outer iteration (one-based) from which the flag holds to the end."""
        if not flags or not flags[-1]:
            return None
        i = len(flags) - 1
        while i > 0 and flags[i - 1]:
            i -= 1
        return i + 1

    # The final faces and fields. A run that left the loop early (diverged
    # or positivity) has no solver faces; the record then keeps what the
    # callback saw last and says so.
    faces = solver.face_velocities
    state = solver.turbulence_state
    if state is None and "state" in latest:
        state = latest["state"]
    out: dict = {
        "name": name,
        "commit": tree,
        "started": started,
        "seconds": seconds,
        "seconds_per_outer": seconds / max(n, 1),
        "stage_seconds": solver.stage_seconds,
        "blas": blas_threads(),
        "openblas_num_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
        **meta,
        "raw": raw,
        "rule_version": solver.rule_version,
        "velocity_scale": boundary.get_max_boundary_velocity(),
        "flux_scale": flux,
        "nu_scale": None if rule is None else rule._nu_scale,
        "nu_air": nu,
        "initial_k_eps": list(solver._initial_turbulence),
        "stop": stop,
        "outer": n,
        "positivity": positivity,
        "residual_min": float(np.min(rec["residual"])) if n else None,
        "residual_min_at": int(np.argmin(rec["residual"])) + 1 if n else None,
        "residual_end": rec["residual"][-1] if n else None,
        "max_speed_end": rec["max_speed"][-1] if n else None,
        "at_end": rec["at"][-1] if n else None,
        "max_speed_peak": float(np.max(rec["max_speed"])) if n else None,
        "cap_hits": int(solver.pressure_cap_hits),
        "inner_mean": float(np.mean(rec["inner"])) if n else None,
        "inner_median": float(np.median(rec["inner"])) if n else None,
        "inner_max": int(np.max(rec["inner"])) if n else None,
        "k_sweeps_mean": float(np.mean(rec["k_sweeps"])) if n else None,
        "eps_sweeps_mean": float(np.mean(rec["eps_sweeps"])) if n else None,
        "scalar_cap_hits": int(n - sum(rec["scalar_converged"])) if n else None,
        "readings_at_stop": {k: v[-1] for k, v in readings.items() if v},
        "from_when": {k: from_when(v) for k, v in holds.items()},
        "faces_kept": faces is not None,
        "face_hash": None if faces is None else probe.face_hash(faces.u, faces.v),
        "readings": readings,
        **rec,
    }
    if faces is not None:
        imbalance = solver.last_mass_imbalance
        out["worst_end"] = float(np.max(np.abs(imbalance)))
        out["signed_end"] = float(np.sum(imbalance))
    arrays: dict[str, np.ndarray] = {
        "xc": mesh.xc,
        "yc": mesh.yc,
        "cell_type": mesh.cell_type,
    }
    if last:
        arrays.update(u_c=last["u_c"], v_c=last["v_c"], p=last["p"])
    if faces is not None:
        arrays.update(u_faces=np.array(faces.u), v_faces=np.array(faces.v))
    if state is not None:
        k_field = np.array(state.k)
        nut = np.array(state.nu_t)
        core = core_mask(mesh, raw)
        ratio = nut / nu
        out["core"] = {
            "cells": int(core.sum()),
            "nu_t_over_nu_median": float(np.median(ratio[core])),
            "nu_t_over_nu_p95": float(np.percentile(ratio[core], 95)),
            "nu_t_over_nu_percentiles": percentiles(ratio[core]),
            "nu_t_over_nu_all_median": float(np.median(ratio[not_solid])),
            "nu_t_over_nu_all_max": float(ratio[not_solid].max()),
            "k_median": float(np.median(k_field[core])),
            "k_max_all": float(k_field[not_solid].max()),
        }
        y_fields, y_stats = wall_y_star(
            walls, k_field, VARIANTS[raw["turbulence"]["variant"]].c_mu, nu
        )
        out["y_star"] = y_stats
        arrays.update(
            k=k_field,
            eps=np.array(state.eps),
            nu_t=nut,
            core=core,
            **{f"y_star_{side}": field for side, field in y_fields.items()},
        )
    (OUT / f"{name}.json").write_text(json.dumps(out))
    np.savez(OUT / f"{name}.npz", **arrays)
    summary = {
        k: out.get(k)
        for k in (
            "stop",
            "outer",
            "residual_end",
            "max_speed_end",
            "at_end",
            "cap_hits",
            "inner_mean",
            "readings_at_stop",
            "from_when",
            "core",
            "y_star",
        )
    }
    print(
        f"{name} done {datetime.now().isoformat(timespec='seconds')} {seconds:.0f}s "
        f"({out['seconds_per_outer']:.3f} s/outer) {json.dumps(summary, default=float)}",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """One row of the set."""
    grid, variant, rtol = args.grid, args.variant, float(args.rtol)
    intensity, length = float(args.intensity), float(args.length)
    cap = args.cap or CAP
    name = f"{variant}_{grid}"
    if rtol != PRESSURE_RTOL:
        name += f"_r{args.rtol}"
    if (intensity, length) != (INTENSITY, DISSIPATION_LENGTH):
        name += f"_i{args.intensity}_l{args.length}"
    if args.locate:
        name += "_loc"
    if args.tag:
        name += f"_{args.tag}"
    raw = product_raw(grid, variant, rtol, intensity, length, cap)
    meta = {
        "case": "product",
        "grid": grid,
        "variant": variant,
        "sweeps": SWEEPS,
        "pressure_rtol": rtol,
        "alpha_velocity": ALPHA_VELOCITY,
        "turbulence_intensity": intensity,
        "dissipation_length": length,
        "stopping": raw["solver"]["stopping_rule"],
        "iteration_error_tol": raw["solver"]["iteration_error_tol"],
        "mass_imbalance_tol": raw["solver"]["mass_imbalance_tol"],
        "cap": cap,
        "locate": bool(args.locate),
    }
    run_room(name, raw, meta, log_every=args.log_every, locate=args.locate)


def timing(args: argparse.Namespace) -> None:
    """A short run whose seconds per outer iteration sizes the set."""
    args.variant = "standard"
    args.rtol = str(PRESSURE_RTOL)
    args.intensity = str(INTENSITY)
    args.length = str(DISSIPATION_LENGTH)
    args.cap = TIME_CAP
    args.tag = "time"
    args.locate = False
    args.log_every = 5
    run(args)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    p.add_argument("grid", choices=sorted(GRIDS))
    p.add_argument("variant", choices=("standard", "rng"))
    p.add_argument("--rtol", default=str(PRESSURE_RTOL))
    p.add_argument("--intensity", default=str(INTENSITY))
    p.add_argument("--length", default=str(DISSIPATION_LENGTH))
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--locate", action="store_true")
    p.add_argument("--log-every", type=int, default=100)
    p.set_defaults(func=run)
    p = sub.add_parser("time")
    p.add_argument("grid", choices=sorted(GRIDS))
    p.set_defaults(func=timing)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
