"""ECR-002 step 6 (prompt 47): the product room on the exact grids, and the 80x30 cycle diagnosed.

Usage:
    python coupled47.py run GRID VARIANT [--arm ARM] [--cap N] [--tag T] [--log-every N]
    python coupled47.py time GRID        a 20-iteration timing probe, standard variant

GRID is 80x30, 160x60 or 320x120; VARIANT is standard or rng. ARM is one of
    control   the committed solver as prompt 46 ran it (the default)
    upwind    k and eps advected by plain upwind (a KEpsilonModel subclass)
    corner    momentum's corner rule removed (tail43.corner_free_class)
    alpha     alpha_turbulence 0.35 in place of 0.7 (a configuration key)
    limiter   the control with the UMIST branch of every face recorded

Every run goes through the committed StaggeredSolver on the committed
product configuration regridded (configs/clean_room_default.yaml with the
turbulence section and the solver keys of prompt 46's report, section 2.1),
from rest, with the cell of the largest change of u, v and nu_t recorded
every outer iteration (prompt 46's --locate, on every row here). Nothing
under src/ is edited. The measuring devices are prompt 46's
(coupled46.RecordingRule and its helpers, imported) plus the arms above,
each a probe-side substitution on the built solver. Records go to
results/builder47/ as NAME.json and NAME.npz, NAME = VARIANT_GRID[_ARM].

One BLAS thread, as prompt 46: OPENBLAS_NUM_THREADS=1 before NumPy loads.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe43"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe41"))

import coupled46  # noqa: E402
import outlet41 as probe  # noqa: E402
from coupled46 import (  # noqa: E402
    ALPHA_VELOCITY,
    CAP,
    DISSIPATION_LENGTH,
    DIVERGED_SPEED,
    INTENSITY,
    PRESSURE_RTOL,
    SWEEPS,
    TURBULENCE,
    DivergedError,
    RecordingRule,
    blas_threads,
    commit,
    core_mask,
    percentiles,
    wall_y_star,
)

from src import scalar_scheme, solver_staggered, turbulence  # noqa: E402
from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.stopping import IterationState  # noqa: E402
from src.turbulence import (  # noqa: E402
    VARIANTS,
    KEpsilonModel,
    PositivityError,
    TurbulenceState,
)

OUT = ROOT / "results" / "builder47"
GRIDS = {"80x30": (80, 30), "160x60": (160, 60), "320x120": (320, 120)}
ARMS = ("control", "upwind", "corner", "alpha", "limiter")
ALPHA_HALVED = 0.35
TIME_CAP = 20
LIMITER_TAIL = 2000
# The four advection calls of one k-epsilon step, in the order step() makes
# them: _advect(k) along x then y, _advect(eps) along x then y.
ADVECTION_CALLS = ("k_x", "k_y", "eps_x", "eps_y")
# Branch codes of the UMIST clamp, psi = max(0, min(2r, (1 + 3r) / 4,
# psi_quick, 2)), per face: which term set the face value.
BRANCHES = ("zero", "2r", "(1+3r)/4", "quick", "2", "c_c")


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------


class UpwindModel(KEpsilonModel):
    """The committed model with k and eps advected by plain upwind.

    `_advect` is the committed method with ``upwind=True`` passed to the
    shared `advective_flux`; the production, the growth, the diffusion, the
    decay and the wall functions are the committed ones.
    """

    def _advect(
        self,
        q: np.ndarray,
        flux_u: np.ndarray,
        flux_v: np.ndarray,
        inflow_u: np.ndarray,
        inflow_v: np.ndarray,
        dt_cell: float | np.ndarray,
    ) -> np.ndarray:
        adv_u = turbulence.advective_flux(q, flux_u, inflow_u, self._axis_x, True)
        adv_v = turbulence.advective_flux(
            q.T, flux_v.T, inflow_v.T, self._axis_y, True
        ).T
        divergence = adv_u[:, 1:] - adv_u[:, :-1] + adv_v[1:, :] - adv_v[:-1, :]
        return q - dt_cell * divergence / self._volume


class LimiterRecorder:
    """Which UMIST branch set each face value, and each flux's sign, per advection call.

    Wraps `scalar_scheme.limited_face_values` (looked up by name inside
    `advective_flux`, so the wrap is seen) to classify every face, and
    `turbulence.advective_flux` (the name `KEpsilonModel._advect` reads) to
    keep the flux signs and to count the calls. Both wrappers return the
    committed functions' values unchanged. `end_iteration` compares the
    four calls' branch and sign arrays with the previous iteration's and
    keeps the counts; over the last LIMITER_TAIL iterations it accumulates
    per-face switch counts so the faces that switch most can be placed.
    """

    def __init__(self) -> None:
        self._real_limited = scalar_scheme.limited_face_values
        self._real_advect = turbulence.advective_flux
        self._branch: np.ndarray | None = None
        self.calls: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.previous: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.branch_switches: list[int] = []
        self.sign_switches: list[int] = []
        self.branch_switches_by_call: list[list[int]] = []
        self.tail_counts: dict[str, np.ndarray] = {}
        self.branch_shares: list[list[int]] = []
        self._call = 0

    def install(self) -> None:
        scalar_scheme.limited_face_values = self._limited  # type: ignore[assignment]
        turbulence.advective_flux = self._advect  # type: ignore[assignment]

    def _limited(
        self, c_up: np.ndarray, c_c: np.ndarray, c_d: np.ndarray, quick: np.ndarray
    ) -> np.ndarray:
        d_down = c_d - c_c
        defined = d_down != 0.0
        safe = np.where(defined, d_down, 1.0)
        r = (c_c - c_up) / safe
        psi_quick = 2.0 * (quick - c_c) / safe
        terms = np.stack(
            [2.0 * r, (1.0 + 3.0 * r) / 4.0, psi_quick, np.full_like(r, 2.0)]
        )
        which = np.argmin(terms, axis=0) + 1
        smallest = terms.min(axis=0)
        branch = np.where(smallest <= 0.0, 0, which)
        self._branch = np.where(defined, branch, 5).astype(np.int8)
        return self._real_limited(c_up, c_c, c_d, quick)

    def _advect(
        self,
        c: np.ndarray,
        flux: np.ndarray,
        inflow: np.ndarray,
        axis: object,
        upwind: bool,
    ) -> np.ndarray:
        self._branch = None
        out = self._real_advect(c, flux, inflow, axis, upwind)
        name = ADVECTION_CALLS[self._call % len(ADVECTION_CALLS)]
        self._call += 1
        branch = (
            np.full(flux[:, 1:-1].shape, 5, dtype=np.int8)
            if self._branch is None
            else self._branch
        )
        self.calls[name] = (branch, np.sign(flux[:, 1:-1]).astype(np.int8))
        return out

    def end_iteration(self, in_tail: bool) -> None:
        """Compare this iteration's four calls with the previous iteration's."""
        if len(self.calls) != len(ADVECTION_CALLS):
            raise RuntimeError(
                f"expected {len(ADVECTION_CALLS)} advection calls, saw {sorted(self.calls)}"
            )
        shares = np.zeros(len(BRANCHES), dtype=int)
        per_call = []
        total_branch = total_sign = 0
        for name in ADVECTION_CALLS:
            branch, sign = self.calls[name]
            shares += np.bincount(branch.ravel(), minlength=len(BRANCHES))
            if name in self.previous:
                was_branch, was_sign = self.previous[name]
                changed = branch != was_branch
                n_branch = int(changed.sum())
                n_sign = int((sign != was_sign).sum())
                if in_tail:
                    if name not in self.tail_counts:
                        self.tail_counts[name] = np.zeros(branch.shape, dtype=np.int32)
                    self.tail_counts[name] += changed
            else:
                n_branch = n_sign = 0
            per_call.append(n_branch)
            total_branch += n_branch
            total_sign += n_sign
        self.branch_switches.append(total_branch)
        self.sign_switches.append(total_sign)
        self.branch_switches_by_call.append(per_call)
        self.branch_shares.append([int(s) for s in shares])
        self.previous = dict(self.calls)
        self.calls = {}

    def summary(self, mesh: Mesh) -> dict:
        """The counts per iteration and the tail's most-switching faces, placed."""
        tail = np.array(self.branch_switches[-LIMITER_TAIL:])
        signs = np.array(self.sign_switches[-LIMITER_TAIL:])
        faces: list[dict] = []
        for name, counts in self.tail_counts.items():
            along_x = name.endswith("_x")
            flat = np.argsort(counts.ravel())[::-1][:10]
            for f in flat:
                j, i = np.unravel_index(int(f), counts.shape)
                if counts[j, i] == 0:
                    continue
                # Interior faces along x: [ny, nx-1], face i+1 at mesh.x[i+1],
                # row j at mesh.yc[j]. Along y the arrays are transposed:
                # [nx, ny-1], column j at mesh.xc[j], face i+1 at mesh.y[i+1].
                if along_x:
                    x, y = float(mesh.x[i + 1]), float(mesh.yc[j])
                else:
                    x, y = float(mesh.xc[j]), float(mesh.y[i + 1])
                faces.append(
                    {
                        "call": name,
                        "x": round(x, 4),
                        "y": round(y, 4),
                        "switches": int(counts[j, i]),
                        "share_of_tail": float(counts[j, i]) / max(tail.size, 1),
                    }
                )
        faces.sort(key=lambda d: -d["switches"])
        faces_total = sum(int(c.size) for c in self.tail_counts.values())
        switching_faces = sum(int((c > 0).sum()) for c in self.tail_counts.values())
        return {
            "tail_iterations": int(tail.size),
            "faces_per_call": {n: list(c.shape) for n, c in self.tail_counts.items()},
            "faces_total": faces_total,
            "branch_switches_per_iteration": {
                "least": int(tail.min()) if tail.size else None,
                "median": float(np.median(tail)) if tail.size else None,
                "largest": int(tail.max()) if tail.size else None,
                "iterations_with_none": int((tail == 0).sum()) if tail.size else None,
            },
            "sign_switches_per_iteration": {
                "least": int(signs.min()) if signs.size else None,
                "median": float(np.median(signs)) if signs.size else None,
                "largest": int(signs.max()) if signs.size else None,
                "iterations_with_none": int((signs == 0).sum()) if signs.size else None,
            },
            "switching_faces_in_tail": switching_faces,
            "faces_switching_most": faces[:20],
            "branch_shares_at_end": dict(
                zip(BRANCHES, self.branch_shares[-1], strict=True)
            )
            if self.branch_shares
            else None,
        }


def corner_free_predictor_class() -> type:
    """tail43's MomentumPredictor subclass with the corner QUICK zeroing removed."""
    import tail43

    return tail43.corner_free_class()


# ---------------------------------------------------------------------------
# The room
# ---------------------------------------------------------------------------


def product_raw(grid: str, variant: str, arm: str, cap: int) -> dict:
    """Prompt 46's product configuration mapping (its report, section 2.1) on GRID."""
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
    block["pressure_rtol"] = PRESSURE_RTOL
    block["momentum_sweeps"] = SWEEPS
    raw["turbulence"] = {**TURBULENCE, "variant": variant}
    if arm == "alpha":
        raw["turbulence"]["alpha_turbulence"] = ALPHA_HALVED
    supply = raw["boundaries"]["hepa_supply"]
    supply["turbulence_intensity"] = INTENSITY
    supply["dissipation_length"] = DISSIPATION_LENGTH
    return raw


# ---------------------------------------------------------------------------
# The runner (prompt 46's run_room with the arms and --locate always on)
# ---------------------------------------------------------------------------


def run_room(name: str, raw: dict, meta: dict, arm: str, log_every: int = 100) -> dict:
    """Run the room to its stop, divergence, positivity error or cap; write the record.

    The arm's substitution is made on the built solver, before the first
    iteration: `upwind` replaces the solver's model, `corner` its predictor,
    `limiter` wraps the two scheme functions in this process; `alpha` is in
    `raw` already and `control` changes nothing.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    tree = commit()
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    solver_staggered.ErrorEstimateRule = RecordingRule  # type: ignore[misc]
    RecordingRule.instances.clear()
    solver = solver_staggered.StaggeredSolver(mesh, cfg, boundary)
    recorder: LimiterRecorder | None = None
    if arm == "upwind":
        solver._model = UpwindModel(mesh, cfg)
    elif arm == "corner":
        solver._predictor = corner_free_predictor_class()(mesh, cfg, boundary)
    elif arm == "limiter":
        recorder = LimiterRecorder()
        recorder.install()
    model, walls = solver._model, solver._walls
    assert model is not None and walls is not None
    not_solid = mesh.cell_type != SOLID
    xc, yc = np.meshgrid(mesh.xc, mesh.yc)
    nu = cfg.mu / cfg.rho
    started = datetime.now().isoformat(timespec="seconds")
    print(f"{name} start {started} {json.dumps(meta)}", flush=True)

    latest: dict[str, TurbulenceState] = {}
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
            "du_at",
            "du",
            "dv_at",
            "dv",
            "dnu_at",
            "dnu",
        )
    }
    previous: dict[str, np.ndarray] = {}
    last: dict[str, np.ndarray] = {}
    cap_hits = {"seen": 0}
    cap = cfg.max_simple_iter
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
        fields = {"u": state.u, "v": state.v, "nu": latest["state"].nu_t}
        for comp, field in fields.items():
            change = (
                np.abs(field - previous[comp]) if comp in previous else np.abs(field)
            )
            change[~not_solid] = 0.0
            kc = int(np.argmax(change))
            rec[f"d{comp}_at"].append(
                (round(float(xc.flat[kc]), 4), round(float(yc.flat[kc]), 4))
            )
            rec[f"d{comp}"].append(float(change.flat[kc]))
            previous[comp] = field.copy()
        if recorder is not None:
            recorder.end_iteration(in_tail=state.iteration >= cap - LIMITER_TAIL)
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
                f"dnu {rec['dnu'][-1]:.2e} at {rec['dnu_at'][-1]} "
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
        if not flags or not flags[-1]:
            return None
        i = len(flags) - 1
        while i > 0 and flags[i - 1]:
            i -= 1
        return i + 1

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
    if recorder is not None:
        out["limiter"] = recorder.summary(mesh)
        out["limiter_branch_switches"] = recorder.branch_switches
        out["limiter_sign_switches"] = recorder.sign_switches
        out["limiter_branch_switches_by_call"] = recorder.branch_switches_by_call
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
    if recorder is not None:
        for call, counts in recorder.tail_counts.items():
            arrays[f"limiter_switches_{call}"] = counts
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
            "face_hash",
            "core",
            "y_star",
            "limiter",
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
    grid, variant, arm = args.grid, args.variant, args.arm
    cap = args.cap or CAP
    name = f"{variant}_{grid}"
    if arm != "control":
        name += f"_{arm}"
    if args.tag:
        name += f"_{args.tag}"
    raw = product_raw(grid, variant, arm, cap)
    meta = {
        "case": "product",
        "grid": grid,
        "variant": variant,
        "arm": arm,
        "sweeps": SWEEPS,
        "pressure_rtol": PRESSURE_RTOL,
        "alpha_velocity": ALPHA_VELOCITY,
        "alpha_turbulence": raw["turbulence"]["alpha_turbulence"],
        "turbulence_intensity": INTENSITY,
        "dissipation_length": DISSIPATION_LENGTH,
        "stopping": raw["solver"]["stopping_rule"],
        "iteration_error_tol": raw["solver"]["iteration_error_tol"],
        "mass_imbalance_tol": raw["solver"]["mass_imbalance_tol"],
        "cap": cap,
        "locate": True,
    }
    run_room(name, raw, meta, arm, log_every=args.log_every)


def timing(args: argparse.Namespace) -> None:
    """A short run whose seconds per outer iteration sizes the set."""
    args.variant = "standard"
    args.arm = "control"
    args.cap = TIME_CAP
    args.tag = "time"
    args.log_every = 5
    run(args)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    p.add_argument("grid", choices=sorted(GRIDS))
    p.add_argument("variant", choices=("standard", "rng"))
    p.add_argument("--arm", choices=ARMS, default="control")
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--log-every", type=int, default=100)
    p.set_defaults(func=run)
    p = sub.add_parser("time")
    p.add_argument("grid", choices=sorted(GRIDS))
    p.set_defaults(func=timing)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    coupled46.OUT = OUT
    main()
