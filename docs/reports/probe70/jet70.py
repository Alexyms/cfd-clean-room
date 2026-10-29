"""Prompt 70 probe A: the plane jet's spreading rate under standard k-epsilon.

A plane jet of width b issues from the left wall into a coflow, between
no-slip walls, and leaves through a fixed-flow outlet over the whole right
edge that states no velocity, so the domain has no pressure outlet and no
face can draw air in (#61). The flow is solved steady with the product's
model and solver keys; then one passive class (0.1 um, no settling to speak
of) is marched on the converged faces with the jet carrying C = 1 and the
coflow clean, to a steady field.

    python docs/reports/probe70/jet70.py [--grid NX NY] [--height H] [--coflow U]
                                         [--concurrent N]

The first version ran a 30 b high domain with a 0.05 U_j coflow
(jet_240x120.json, jet_120x60.json): the plane jet's volume flux grows as
sqrt(x) and by 50 b it carries about 4.4 times the slot's flow, more than the
coflow supplied, so the walls carried a return flow, the eddy viscosity
filled the channel and the scalar homogenised. The second version (the
defaults) runs 50 b high with a 0.1 U_j coflow, which supplies about 4.9
slot flows against the 3.2 the jet entrains by the fit range's end; its
records are jet_<nx>x<ny>_h<H>_c<U>.json.

What it writes (results/builder70/): the .json (the half-width series, the
fits, the stop, the wall time) and the .npz (the fields). The readings,
fixed before the run:

- The half-width y_1/2(x) of a field: at each column, the excess over the
  local background (the mean of the column outside 1.0 m of the
  centreline, the four wall cells excluded; the coflow's velocity, the
  scalar's zero) is taken, and y_1/2 is half the distance between the two
  points where the excess falls to half its centreline value, each found by
  linear interpolation.
- The spreading rate K1 is the slope of the least-squares line through
  y_1/2(x) over 15 b <= x <= 45 b (the self-similar region clear of the
  outlet); the fit's intercept gives the virtual origin.
- The scalar march stops when the largest change in C over a step is below
  STEADY_TOL of the inlet value, or at T_CAP.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.boundary_concentration import ConcentrationBoundary  # noqa: E402
from src.boundary_registry import BoundaryRegistry  # noqa: E402
from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.particles import ParticlePhysics  # noqa: E402
from src.solver_staggered import StaggeredSolver  # noqa: E402
from src.solver_transport import TransportSolver  # noqa: E402

OUT = ROOT / "results" / "builder70"
B = 0.1  # slot width, m
U_JET = 10.0  # m/s
WIDTH = 6.0  # 60 b
DEFAULT_HEIGHT = 5.0  # 50 b
DEFAULT_COFLOW = 1.0  # m/s, 0.1 U_j
FIT_RANGE = (15.0 * B, 45.0 * B)
BACKGROUND_FROM = 1.0  # m from the centreline: the coflow's cells
WALL_CELLS = 4
STEADY_TOL = 1e-7
T_CAP = 40.0


def commit() -> str:
    """The short commit of the tree the probe runs in."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.strip()


def jet_raw(
    nx: int, ny: int, height: float, coflow: float, k_advection: str = "upwind"
) -> dict:
    """The case as a mapping: the product's model and solver keys on the jet's domain."""
    v_min = (WIDTH / nx) * (height / ny)
    y_lo, y_hi = height / 2 - B / 2, height / 2 + B / 2
    return {
        "domain": {"width": WIDTH, "height": height, "nx": nx, "ny": ny},
        "mesh": {"x": {"stretch_ratio": 1.0}, "y": {"stretch_ratio": 1.0}},
        "fluid": {"density": 1.2, "viscosity": 1.81e-5, "temperature": 293.0},
        "particles": {
            "density": 1000.0,
            "sizes": [0.1e-6],
            "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {"diameters": [0.1e-6], "efficiencies": [0.99999]},
        },
        "solver": {
            "dt": 0.01,
            "t_end": 60.0,
            "output_interval": 10,
            "convergence_tol": 1.0e-6,
            "stopping_rule": "error_estimate",
            "iteration_error_tol": 1.0e-6,
            "mass_imbalance_tol": 1e-4 * 1.2 * v_min / 60.0,
            "max_simple_iter": 15000,
            "momentum_sweeps": 10,
            "pressure_rtol": 1.0e-4,
            "alpha_velocity": 0.5,
            "alpha_pressure": 0.3,
            "max_pressure_iter": 5000,
            "pressure_preconditioner": "multigrid",
        },
        "transport": {
            "cfl_number": 0.4,
            "advection_scheme": "umist",
            "max_diffusion_iter": 200,
            "diffusion_tol": 1.0e-8,
            "turbulent_schmidt": 0.7,
        },
        "turbulence": {
            "model": "k_epsilon",
            "variant": "standard",
            "wall_treatment": "scalable_wall_functions",
            "cfl_number": 0.25,
            "advection_scheme": k_advection,
            "alpha_turbulence": 0.7,
            "max_iter": 500,
            "tol": 1.0e-10,
        },
        "boundaries": {
            "coflow_low": {
                "type": "velocity_inlet",
                "location": "left",
                "y_start": 0.0,
                "y_end": y_lo,
                "velocity": coflow,
                "hepa_filtered": True,
                "turbulence_intensity": 0.01,
                "dissipation_length": 0.1,
            },
            "jet": {
                "type": "velocity_inlet",
                "location": "left",
                "y_start": y_lo,
                "y_end": y_hi,
                "velocity": U_JET,
                "hepa_filtered": False,
                "concentration": [1.0],
                "turbulence_intensity": 0.05,
                "dissipation_length": 0.1 * B,
            },
            "coflow_high": {
                "type": "velocity_inlet",
                "location": "left",
                "y_start": y_hi,
                "y_end": height,
                "velocity": coflow,
                "hepa_filtered": True,
                "turbulence_intensity": 0.01,
                "dissipation_length": 0.1,
            },
            "exit": {
                "type": "fixed_flow_outlet",
                "location": "right",
                "y_start": 0.0,
                "y_end": height,
            },
        },
        "obstacles": [],
        "sensors": [{"name": "axis", "x": 3.0, "y": height / 2}],
        "thresholds": {"0.1e-6": 0.5},
    }


def half_width(yc: np.ndarray, profile: np.ndarray, background: float) -> float:
    """Half the distance between the two half-excess points of one column."""
    excess = profile - background
    j0 = int(np.argmax(excess))
    half = 0.5 * excess[j0]
    if half <= 0.0:
        return float("nan")

    def crossing(direction: int) -> float:
        j = j0
        while 0 <= j + direction < excess.size and excess[j + direction] >= half:
            j += direction
        k = j + direction
        if not 0 <= k < excess.size:
            return float("nan")
        frac = (excess[j] - half) / (excess[j] - excess[k])
        return float(yc[j] + frac * (yc[k] - yc[j]))

    return 0.5 * (crossing(+1) - crossing(-1))


def series(mesh: Mesh, field: np.ndarray, height: float) -> np.ndarray:
    """y_1/2 at every column, from the cell-centred field over its far-field background."""
    yc = np.asarray(mesh.yc)
    far = np.abs(yc - height / 2) > BACKGROUND_FROM
    far[:WALL_CELLS] = False
    far[-WALL_CELLS:] = False
    out = np.empty(field.shape[1])
    for i in range(field.shape[1]):
        column = field[:, i]
        out[i] = half_width(yc, column, float(column[far].mean()))
    return out


def fit(xc: np.ndarray, widths: np.ndarray) -> dict:
    """The least-squares line through y_1/2 over FIT_RANGE, in slot widths."""
    sel = (xc >= FIT_RANGE[0]) & (xc <= FIT_RANGE[1]) & np.isfinite(widths)
    slope, intercept = np.polyfit(xc[sel], widths[sel], 1)
    residual = widths[sel] - (slope * xc[sel] + intercept)
    return {
        "k1": float(slope),
        "k2_slot_widths": float(intercept / B),
        "virtual_origin_slot_widths": float(-intercept / slope / B),
        "rms_residual_slot_widths": float(np.sqrt(np.mean(residual**2)) / B),
        "columns": int(sel.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=int, nargs=2, default=(120, 100))
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT)
    parser.add_argument("--coflow", type=float, default=DEFAULT_COFLOW)
    parser.add_argument("--k-advection", default="upwind", help="k and eps scheme")
    parser.add_argument("--concurrent", type=int, default=0)
    args = parser.parse_args()
    nx, ny = args.grid
    height, coflow = args.height, args.coflow
    tag = f"{nx}x{ny}_h{height:g}_c{coflow:g}"
    if args.k_advection != "upwind":
        tag += f"_k{args.k_advection}"
    cfg = SimConfig.from_dict(jet_raw(nx, ny, height, coflow, args.k_advection))
    mesh = Mesh(cfg)
    xc = np.asarray(mesh.xc)
    started = datetime.now().isoformat(timespec="seconds")
    tree = commit()
    print(
        f"jet {tag}: start {started} commit {tree} concurrent {args.concurrent}",
        flush=True,
    )

    boundary = StaggeredBoundary(mesh, cfg)
    solver = StaggeredSolver(mesh, cfg, boundary)
    t0 = perf_counter()
    _u, _v, p = solver.solve_steady()
    flow_wall = perf_counter() - t0
    state = solver.flow_state
    faces = solver.face_velocities
    print(
        f"jet {tag}: flow stop {state.stop_reason} after "
        f"{len(solver.residual_history)} outer, {flow_wall:.0f} s",
        flush=True,
    )
    u_c = 0.5 * (faces.u[:, :-1] + faces.u[:, 1:])
    widths_u = series(mesh, u_c, height)
    fit_u = fit(xc, widths_u)
    print(f"jet {tag}: velocity K1 {fit_u['k1']:.4f}", flush=True)

    physics = ParticlePhysics(cfg)
    transport = TransportSolver(
        mesh,
        cfg,
        physics,
        ConcentrationBoundary(mesh, cfg, physics, BoundaryRegistry(cfg)),
    )
    nu_t = np.ascontiguousarray(state.nu_t, dtype=np.float64)
    dt = transport.stable_dt(faces, 0)
    field = np.zeros((ny, nx))
    t1 = perf_counter()
    t, step, change = 0.0, 0, float("inf")
    history = []
    while t < T_CAP and change > STEADY_TOL:
        step += 1
        new = transport.solve_timestep(field, faces, 0, dt, eddy_viscosity=nu_t)
        change = float(np.abs(new - field).max())
        field = new
        t = step * dt
        if step % 2000 == 0:
            k1 = fit(xc, series(mesh, field, height))["k1"]
            history.append({"t": t, "step": step, "change": change, "k1": k1})
            print(
                f"jet {tag}: scalar t {t:.2f} s change {change:.2e} K1 {k1:.4f}",
                flush=True,
            )
    scalar_wall = perf_counter() - t1
    widths_c = series(mesh, field, height)
    fit_c = fit(xc, widths_c)
    centre_j = int(np.argmin(np.abs(np.asarray(mesh.yc) - height / 2)))

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / f"jet_{tag}.npz",
        u=faces.u,
        v=faces.v,
        p=p,
        k=state.k,
        eps=state.eps,
        nu_t=state.nu_t,
        C=field,
        widths_u=widths_u,
        widths_c=widths_c,
        xc=xc,
    )
    record = {
        "grid": [nx, ny],
        "slot_width_m": B,
        "cells_across_slot": round(B / (height / ny)),
        "u_jet": U_JET,
        "u_coflow": coflow,
        "k_advection": args.k_advection,
        "domain_m": [WIDTH, height],
        "fit_range_slot_widths": [FIT_RANGE[0] / B, FIT_RANGE[1] / B],
        "started": started,
        "commit": tree,
        "concurrent_processes": args.concurrent,
        "flow": {
            "stop_reason": state.stop_reason,
            "outer_iterations": len(solver.residual_history),
            "wall_seconds": flow_wall,
        },
        "velocity": fit_u,
        "scalar": fit_c,
        "scalar_to_velocity_ratio": fit_c["k1"] / fit_u["k1"],
        "scalar_march": {
            "dt": dt,
            "steps": step,
            "t_end": t,
            "stopped_by": "steady" if change <= STEADY_TOL else "cap",
            "last_change": change,
            "wall_seconds": scalar_wall,
            "history": history,
        },
        "centreline_excess_over_coflow": [
            float(u_c[centre_j, i] - coflow) for i in range(nx)
        ],
        "centreline_c": [float(field[centre_j, i]) for i in range(nx)],
        "xc": xc.tolist(),
        "widths_u": widths_u.tolist(),
        "widths_c": widths_c.tolist(),
    }
    (OUT / f"jet_{tag}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(
        f"jet {tag}: velocity K1 {fit_u['k1']:.4f} "
        f"(x0 {fit_u['virtual_origin_slot_widths']:.1f} b), "
        f"scalar K1 {fit_c['k1']:.4f}, ratio {record['scalar_to_velocity_ratio']:.3f}; "
        f"scalar {record['scalar_march']['stopped_by']} at {t:.2f} s, {scalar_wall:.0f} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
