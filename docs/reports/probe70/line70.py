"""Prompt 70b probe: a line source in decaying grid turbulence (VAL-024 as revised).

Warhaft's (1984, JFM 144) single line source, as Viswanathan and Pope (2008,
Phys. Fluids 20, table I) tabulate it: grid mesh M = 2.54 cm, mean speed
U = 7 m/s, the heated wire at x_o = 52 M from the grid, the velocity standard
deviations at one mesh length sigma_u 2.44 and sigma_v = sigma_w 2.27 m/s
decaying as (x / M)^(-m / 2) with m = 1.4, molecular diffusivity
kappa = 2.1e-5 m^2/s, source size 2.5e-5 m. The reference plume width is
their equations 11 to 13, Anand and Pope's displacement variance for the
Langevin model (C_0 = 2.1) in that decaying turbulence, plus 2 kappa t.

The case here: a uniform inflow at U over the whole left edge carrying k and
eps of the source's station (k_o = (sigma_u^2 + 2 sigma_v^2) / 2 at 52 M,
eps_o = m k_o U / x_o, the exact decay rate of that power law), walls top and
bottom 25 cm from the centreline (the plume stays under 3 cm wide), a
fixed-flow outlet over the whole right edge stating no velocity (no pressure
outlet). The flow is solved steady with the product's model and solver keys;
one passive class is then marched on the converged faces with a unit source
in one cell on the centreline two cells from the inlet until the field is
steady.

    python docs/reports/probe70/line70.py [--grid NX NY] [--concurrent N]

Writes results/builder70/line_<nx>x<ny>.json and .npz. The readings, fixed
before the run:

- The model's decay exponent n: the slope of ln k against ln(t_v + x / U)
  along the centreline row over the middle 80% of the domain, t_v the
  virtual origin k_o / ((C_e2 - 1) eps_o) that the model's own solution
  k = k_o (1 + t / t_v)^(-n) carries; standard k-epsilon's n is
  1 / (C_e2 - 1) = 1.087.
- The plume's width sigma(x): the second moment of the column's profile
  about its own mean, which for a Gaussian is its standard deviation (the
  experiments fit Gaussians).
- The reference sigma_ref(x) at the same flight time t = (x - x_s) / U:
  sqrt(sigma_o^2 + 2 kappa t + Delta_o^2(t)), Delta_o^2 from equation 13
  with r and s from equations 14 and 15; the ratio sigma / sigma_ref is
  reported at 5, 10, 20, 50 and 100 M from the source.
- The model's near-field width also against the cell's own second moment
  dx / sqrt(12), which is the source size the grid imposes.
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
# Warhaft (1984) as tabulated by Viswanathan and Pope (2008), table I.
M = 0.0254  # grid mesh, m
U = 7.0  # m/s
X_O_OVER_M = 52.0  # the source's station
SIGMA_U_AT_M = 2.44  # m/s, at one mesh length from the grid
SIGMA_V_AT_M = 2.27  # m/s, sigma_v = sigma_w
M_EXP = 1.4  # the velocity variance decay exponent, Sawford's refit
C_0 = 2.1  # the Langevin constant of the reference curve
KAPPA = 2.1e-5  # m^2/s, molecular diffusivity of heat
SIGMA_O = 2.5e-5  # m, the wire's effective source size
# The model's constants (ADR-012 A).
C_E2 = 1.92
C_MU = 0.09
# The case.
LENGTH, HEIGHT = 2.8, 0.5  # m: 110 M downstream, walls 25 cm off the centreline
SOURCE_COLUMN = 2  # the source cell's column (the inlet is the left edge)
STATIONS_M = (5.0, 10.0, 20.0, 50.0, 100.0)  # (x - x_s) / M where the ratio is read
STEADY_RTOL = 1e-6
T_CAP = 3.0


def commit() -> str:
    """The short commit of the tree the probe runs in."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.strip()


def source_turbulence() -> dict:
    """k, eps, the intensity and dissipation length at the source's station, and t_o."""
    decay = X_O_OVER_M ** (-M_EXP / 2)
    sigma_u, sigma_v = SIGMA_U_AT_M * decay, SIGMA_V_AT_M * decay
    k_o = 0.5 * (sigma_u**2 + 2.0 * sigma_v**2)
    t_o = X_O_OVER_M * M / U
    eps_o = M_EXP * k_o / t_o
    return {
        "sigma_u": sigma_u,
        "sigma_v": sigma_v,
        "k": k_o,
        "eps": eps_o,
        "t_o": t_o,
        "intensity": float(np.sqrt(k_o / 1.5) / U),
        "dissipation_length": float(k_o**1.5 / eps_o),
        "k_over_eps": k_o / eps_o,
        "t_virtual_model": k_o / ((C_E2 - 1.0) * eps_o),
        "nu_t_model": C_MU * k_o**2 / eps_o,
    }


def reference_sigma(t: np.ndarray, sigma_v: float, t_o: float) -> np.ndarray:
    """Viswanathan and Pope's equations 11 to 15 with the source size and 2 kappa t."""
    r = 0.5 * M_EXP * (1.5 * C_0 - 1.0) + 1.0
    s = 0.5 * M_EXP * (1.5 * C_0 + 1.0) - 1.0
    tau = t / t_o
    bracket = (
        (1.0 + tau) ** (r - s) / (r * (r - s))
        + (1.0 + tau) ** (-s) / (r * s)
        - 1.0 / (s * (r - s))
    )
    delta2 = 2.0 * sigma_v**2 * t_o**2 * bracket
    return np.sqrt(SIGMA_O**2 + 2.0 * KAPPA * t + delta2)


def case_raw(nx: int, ny: int, turb: dict) -> dict:
    v_min = (LENGTH / nx) * (HEIGHT / ny)
    return {
        "domain": {"width": LENGTH, "height": HEIGHT, "nx": nx, "ny": ny},
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
            "advection_scheme": "upwind",
            "alpha_turbulence": 0.7,
            "max_iter": 500,
            "tol": 1.0e-10,
        },
        "boundaries": {
            "stream": {
                "type": "velocity_inlet",
                "location": "left",
                "y_start": 0.0,
                "y_end": HEIGHT,
                "velocity": U,
                "hepa_filtered": True,
                "turbulence_intensity": turb["intensity"],
                "dissipation_length": turb["dissipation_length"],
            },
            "exit": {
                "type": "fixed_flow_outlet",
                "location": "right",
                "y_start": 0.0,
                "y_end": HEIGHT,
            },
        },
        "obstacles": [],
        "sensors": [{"name": "axis", "x": LENGTH / 2, "y": HEIGHT / 2}],
        "thresholds": {"0.1e-6": 1.0},
    }


def widths(mesh: Mesh, field: np.ndarray) -> np.ndarray:
    """The second moment of each column's profile about its own mean."""
    yc = np.asarray(mesh.yc)
    out = np.full(field.shape[1], np.nan)
    for i in range(field.shape[1]):
        column = field[:, i]
        total = column.sum()
        if total <= 0.0:
            continue
        mean = (column * yc).sum() / total
        out[i] = float(np.sqrt((column * (yc - mean) ** 2).sum() / total))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=int, nargs=2, default=(280, 50))
    parser.add_argument("--concurrent", type=int, default=0)
    args = parser.parse_args()
    nx, ny = args.grid
    tag = f"{nx}x{ny}"
    turb = source_turbulence()
    cfg = SimConfig.from_dict(case_raw(nx, ny, turb))
    mesh = Mesh(cfg)
    xc, yc = np.asarray(mesh.xc), np.asarray(mesh.yc)
    dx = float(mesh.dx_cell[0])
    started = datetime.now().isoformat(timespec="seconds")
    tree = commit()
    print(
        f"line {tag}: start {started} commit {tree} concurrent {args.concurrent}; "
        f"k_o {turb['k']:.4e} eps_o {turb['eps']:.4e} I {turb['intensity']:.4f} "
        f"l_e {turb['dissipation_length']:.4e}",
        flush=True,
    )

    solver = StaggeredSolver(mesh, cfg, StaggeredBoundary(mesh, cfg))
    t0 = perf_counter()
    _u, _v, p = solver.solve_steady()
    flow_wall = perf_counter() - t0
    state = solver.flow_state
    faces = solver.face_velocities
    print(
        f"line {tag}: flow stop {state.stop_reason} after "
        f"{len(solver.residual_history)} outer, {flow_wall:.0f} s",
        flush=True,
    )
    j_mid = ny // 2
    k_line = np.asarray(state.k[j_mid, :])
    sel = (xc >= 0.1 * LENGTH) & (xc <= 0.9 * LENGTH)
    t_model = turb["t_virtual_model"] + xc / U
    slope, _intercept = np.polyfit(np.log(t_model[sel]), np.log(k_line[sel]), 1)
    n_model = -float(slope)
    print(
        f"line {tag}: k decay exponent {n_model:.4f} (model's own 1/(C_e2-1) {1 / (C_E2 - 1):.4f})",
        flush=True,
    )

    physics = ParticlePhysics(cfg)
    transport = TransportSolver(
        mesh,
        cfg,
        physics,
        ConcentrationBoundary(mesh, cfg, physics, BoundaryRegistry(cfg)),
    )
    nu_t = np.ascontiguousarray(state.nu_t, dtype=np.float64)
    dt = transport.stable_dt(faces, 0)
    sources = np.zeros((ny, nx))
    sources[j_mid, SOURCE_COLUMN] = 1.0
    x_s = float(xc[SOURCE_COLUMN])
    field = np.zeros((ny, nx))
    t1 = perf_counter()
    t, step, rel = 0.0, 0, float("inf")
    while t < T_CAP and rel > STEADY_RTOL:
        step += 1
        new = transport.solve_timestep(
            field, faces, 0, dt, sources=sources, eddy_viscosity=nu_t
        )
        rel = float(np.abs(new - field).max() / max(new.max(), 1e-300))
        field = new
        t = step * dt
        if step % 2000 == 0:
            print(
                f"line {tag}: scalar t {t:.3f} s relative change {rel:.2e}", flush=True
            )
    scalar_wall = perf_counter() - t1

    sigma = widths(mesh, field)
    downstream = xc > x_s
    flight = (xc - x_s) / U
    sigma_ref = reference_sigma(
        np.where(downstream, flight, 0.0), turb["sigma_v"], turb["t_o"]
    )
    stations = {}
    for station in STATIONS_M:
        i = int(np.argmin(np.abs(xc - (x_s + station * M))))
        stations[f"{station:g}"] = {
            "x_m": float(xc[i]),
            "flight_time_s": float(flight[i]),
            "t_over_T_L": float(flight[i] / (4.0 / (3.0 * C_0) * turb["k_over_eps"])),
            "sigma_model_m": float(sigma[i]),
            "sigma_ref_m": float(sigma_ref[i]),
            "ratio": float(sigma[i] / sigma_ref[i]),
            "sigma_model_over_M": float(sigma[i] / M),
            "sigma_ref_over_M": float(sigma_ref[i] / M),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / f"line_{tag}.npz",
        u=faces.u,
        v=faces.v,
        p=p,
        k=state.k,
        eps=state.eps,
        nu_t=state.nu_t,
        C=field,
        sigma=sigma,
        sigma_ref=sigma_ref,
        xc=xc,
        yc=yc,
    )
    record = {
        "grid": [nx, ny],
        "dx_m": dx,
        "cell_second_moment_m": dx / np.sqrt(12.0),
        "domain_m": [LENGTH, HEIGHT],
        "source": {"column": SOURCE_COLUMN, "x_m": x_s, "row": j_mid},
        "experiment": {
            "M_m": M,
            "U_m_per_s": U,
            "x_o_over_M": X_O_OVER_M,
            "m": M_EXP,
            "C_0": C_0,
            "kappa_m2_per_s": KAPPA,
            "sigma_o_m": SIGMA_O,
            "Re_M": U * M / (1.81e-5 / 1.2),
        },
        "source_turbulence": turb,
        "T_L_s": 4.0 / (3.0 * C_0) * turb["k_over_eps"],
        "started": started,
        "commit": tree,
        "concurrent_processes": args.concurrent,
        "flow": {
            "stop_reason": state.stop_reason,
            "outer_iterations": len(solver.residual_history),
            "wall_seconds": flow_wall,
        },
        "decay": {
            "n_model": n_model,
            "n_standard_k_epsilon": 1.0 / (C_E2 - 1.0),
            "fit_range_m": [0.1 * LENGTH, 0.9 * LENGTH],
            "k_inlet": float(k_line[0]),
            "k_end": float(k_line[-1]),
        },
        "scalar_march": {
            "dt": dt,
            "steps": step,
            "t_end": t,
            "stopped_by": "steady" if rel <= STEADY_RTOL else "cap",
            "last_relative_change": rel,
            "wall_seconds": scalar_wall,
        },
        "stations": stations,
        "xc": xc.tolist(),
        "sigma": sigma.tolist(),
        "sigma_ref": sigma_ref.tolist(),
    }
    (OUT / f"line_{tag}.json").write_text(
        json.dumps(record, indent=1), encoding="utf-8"
    )
    summary = ", ".join(f"{k} M: {v['ratio']:.3f}" for k, v in stations.items())
    print(
        f"line {tag}: n {n_model:.4f}; sigma / sigma_ref at {summary}; "
        f"scalar {record['scalar_march']['stopped_by']} at {t:.3f} s, {scalar_wall:.0f} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
