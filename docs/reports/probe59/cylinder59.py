"""Prompt 59 probe B: the cost of one converged implicit time step on the square cylinder at Re 100.

ADR-014 section A, tier 2. The channel of Sohankar, Norberg and Davidson
(1998): a unit square cylinder in a channel of height 20 (5% blockage), the
inlet 10 units upstream of the body's front face at U = 1, the outlet 26
units downstream (the paper's Neumann-outlet extent, since this solver's
pressure outlet is the zero-gradient copy), the lateral boundaries at
U = 1, V = 0 (a freestream condition in place of the paper's symmetry
condition, which the boundary layer does not offer), rho = 1 and mu = 0.01
so Re = U d / nu = 100. A uniform grid of DX per cell, ten cells across the
body at the default 0.1, because mesh.py clusters toward the domain walls
only. Nothing under src/ changes: the steady predictor is subclassed here
with the implicit Euler term rho V / dt on the diagonal and rho V u_old / dt
in the source, and the steady loop is driven one time step at a time by
replacing its fields at rest with the previous step's faces.

    python cylinder59.py [--dx 0.1] [--startup 200] [--steps 20] [--arms ABC]
                         [--perturb] [--tag T] [--concurrent N]

The run: STARTUP outer iterations of the steady loop from rest give a
developing, non-trivial field; from that one field each arm takes STEPS
implicit Euler time steps, the outer iterations of each step stopped when
the largest cell-centred velocity change over an outer iteration is below
the arm's step tolerance times U:

    A  dt 0.025, pressure_rtol 1e-4, step tol 1e-4 U (the paper's step; the product's rtol)
    B  dt 0.050, pressure_rtol 1e-4, step tol 1e-4 U
    C  dt 0.025, pressure_rtol 1e-8, step tol 1e-4 U (the validation cases' rtol)
    D  dt 0.050, pressure_rtol 1e-4, step tol 1e-5 U
    E  dt 0.050, pressure_rtol 1e-4, step tol 1e-6 U
    F  dt 0.025, pressure_rtol 1e-4, step tol 1e-6 U

The counts at every threshold of THRESHOLDS are read off each step's
history as well. With --perturb the start-up's lower lateral boundary runs
at 0.9 U, an asymmetry that lets the shedding instability grow during a
long march (the march itself is symmetric); the wake velocity's sign
changes in the record show whether it did.

Each record carries every step's outer count, wall time, stage seconds, the
mass imbalance at the step's end and the vertical velocity one unit behind
the body on the centreline. One BLAS thread. Records go to
results/builder59/cylinder_<arm><tag>.json and the start-up state to
cylinder_startup[_perturbed].npz.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import src.solver_staggered as solver_module  # noqa: E402
from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.momentum import MomentumCoefficients, MomentumPredictor  # noqa: E402
from src.solver_staggered import StaggeredSolver  # noqa: E402

OUT = ROOT / "results" / "builder59"
D = 1.0
H = 20.0
X_UP = 10.0
X_DOWN = 26.0
U_INF = 1.0
RE = 100.0
STEP_CAP = 500
THRESHOLDS = (1e-3, 1e-4, 1e-5, 1e-6)
ARMS = {
    "A": (0.025, 1e-4, 1e-4),
    "B": (0.05, 1e-4, 1e-4),
    "C": (0.025, 1e-8, 1e-4),
    "D": (0.05, 1e-4, 1e-5),
    "E": (0.05, 1e-4, 1e-6),
    "F": (0.025, 1e-4, 1e-6),
}


def commit() -> str:
    """The short commit of the tree the probe runs in."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.strip()


def cylinder_raw(
    dx: float, max_iter: int, tol: float, rtol: float, bottom_u: float = U_INF
) -> dict:
    """The channel's configuration mapping on a uniform grid of DX."""
    width = X_UP + D + X_DOWN
    nx, ny = round(width / dx), round(H / dx)
    if abs(nx * dx - width) > 1e-9 or abs(ny * dx - H) > 1e-9:
        raise ValueError(f"dx {dx} does not divide the domain {width} x {H}")
    return {
        "domain": {"width": width, "height": H, "nx": nx, "ny": ny},
        "mesh": {"x": {"stretch_ratio": 1.0}, "y": {"stretch_ratio": 1.0}},
        "fluid": {"density": 1.0, "viscosity": U_INF * D / RE, "temperature": 293.0},
        "particles": {
            "density": 1000.0,
            "sizes": [0.1e-6],
            "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {"diameters": [0.1e-6], "efficiencies": [0.99999]},
        },
        "solver": {
            "dt": 0.025,
            "t_end": 1.0,
            "output_interval": 10,
            "convergence_tol": tol,
            "max_simple_iter": max_iter,
            "momentum_sweeps": 10,
            "alpha_velocity": 0.7,
            "alpha_pressure": 0.3,
            "max_pressure_iter": 5000,
            "pressure_rtol": rtol,
            "pressure_preconditioner": "multigrid",
            "stopping_rule": "velocity_step",
        },
        "boundaries": {
            "inlet": {
                "type": "velocity_inlet",
                "location": "left",
                "y_start": 0.0,
                "y_end": H,
                "velocity": U_INF,
            },
            "outlet": {
                "type": "pressure_outlet",
                "location": "right",
                "y_start": 0.0,
                "y_end": H,
            },
            "top": {
                "type": "velocity_inlet",
                "location": "top",
                "x_start": 0.0,
                "x_end": width,
                "u_velocity": U_INF,
                "v_velocity": 0.0,
            },
            "bottom": {
                "type": "velocity_inlet",
                "location": "bottom",
                "x_start": 0.0,
                "x_end": width,
                "u_velocity": bottom_u,
                "v_velocity": 0.0,
            },
        },
        "obstacles": [
            {
                "name": "cylinder",
                "x_start": X_UP,
                "x_end": X_UP + D,
                "y_start": (H - D) / 2,
                "y_end": (H + D) / 2,
            }
        ],
        "sensors": [{"name": "wake", "x": X_UP + 2 * D, "y": H / 2}],
        "thresholds": {"0.1e-6": 100.0},
    }


class ImplicitEulerPredictor(MomentumPredictor):
    """The steady predictor with the implicit Euler term of one time step.

    ``u_old`` and ``v_old`` are the previous time level's faces; the term
    ``rho V / dt`` joins the diagonal at every unknown face and
    ``rho V u_old / dt`` the source, V the face's control volume (the
    centre-to-centre width times the cell height).
    """

    def __init__(
        self, mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary, dt: float
    ) -> None:
        super().__init__(mesh, config, boundary)
        self.dt = dt
        self.u_old: np.ndarray | None = None
        self.v_old: np.ndarray | None = None

    def _assemble(self, u, v, o, mu_cells=None, wall=None) -> MomentumCoefficients:
        c = super()._assemble(u, v, o, mu_cells=mu_cells, wall=wall)
        assert self.u_old is not None and self.v_old is not None
        old = self.u_old if o is self._for_u else np.ascontiguousarray(self.v_old.T)
        unknown = c.a_p > 0.0
        volume = o.ds_face[None, :] * o.dt_cell[:, None]
        term = np.where(unknown, self._rho * volume / self.dt, 0.0)
        return MomentumCoefficients(
            a_p=c.a_p + term,
            a_s_plus=c.a_s_plus,
            a_s_minus=c.a_s_minus,
            a_t_plus=c.a_t_plus,
            a_t_minus=c.a_t_minus,
            b_boundary=c.b_boundary + term * old,
            b_deferred=c.b_deferred,
        )


def fields_of(
    u: np.ndarray, v: np.ndarray, p: np.ndarray
) -> Callable[[Mesh], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A stand-in for allocate_fields that hands the loop copies of a kept state."""

    def fields(_mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return u.copy(), v.copy(), p.copy()

    return fields


def wake_probe(mesh: Mesh) -> tuple[int, int]:
    """The cell nearest the sensor one unit behind the body on the centreline."""
    i = int(np.argmin(np.abs(mesh.xc - (X_UP + 2 * D))))
    j = int(np.argmin(np.abs(mesh.yc - H / 2)))
    return j, i


def startup(
    dx: float, outer: int, concurrent: int, perturb: bool
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """OUTER steady outer iterations from rest; the developing field every arm starts from."""
    raw = cylinder_raw(dx, outer, 1e-12, 1e-4, 0.9 * U_INF if perturb else U_INF)
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    solver = StaggeredSolver(mesh, cfg, StaggeredBoundary(mesh, cfg))
    t0 = perf_counter()
    u_c, v_c, p = solver.solve_steady()
    wall = perf_counter() - t0
    faces = solver.face_velocities
    assert faces is not None
    j, i = wake_probe(mesh)
    record = {
        "kind": "startup",
        "perturbed": perturb,
        "dx": dx,
        "grid": [cfg.nx, cfg.ny],
        "outer": len(solver.residual_history),
        "wall_seconds": wall,
        "seconds_per_outer": wall / max(len(solver.residual_history), 1),
        "stage_seconds": dict(solver.stage_seconds),
        "reference_velocity": solver.reference_velocity,
        "residual_end": float(solver.residual_history[-1]),
        "wake_v": float(v_c[j, i]),
        "max_speed": float(np.max(np.hypot(u_c, v_c)[mesh.cell_type != 2])),
        "concurrent_processes": concurrent,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    label = "cylinder_startup_perturbed" if perturb else "cylinder_startup"
    np.savez(
        OUT / f"{label}.npz",
        u_faces=np.array(faces.u),
        v_faces=np.array(faces.v),
        p=p,
    )
    (OUT / f"{label}.json").write_text(
        json.dumps({**record, "raw": raw}, indent=1), encoding="utf-8"
    )
    print(
        f"startup: {record['outer']} outer in {wall:.0f} s "
        f"({record['seconds_per_outer']:.3f} s per outer), reference velocity "
        f"{solver.reference_velocity:g}, wake v {record['wake_v']:+.4f}",
        flush=True,
    )
    return record, np.array(faces.u), np.array(faces.v), p


def march(
    arm: str,
    dx: float,
    steps: int,
    u0: np.ndarray,
    v0: np.ndarray,
    p0: np.ndarray,
    concurrent: int,
    tag: str = "",
    perturbed: bool = False,
) -> dict:
    """STEPS implicit Euler steps of one arm from the start-up field, each to its tolerance."""
    dt, rtol, step_tol = ARMS[arm]
    probe_cfg = SimConfig.from_dict(cylinder_raw(dx, STEP_CAP, 1e-12, rtol))
    mesh = Mesh(probe_cfg)
    boundary = StaggeredBoundary(mesh, probe_cfg)
    reference = StaggeredSolver(mesh, probe_cfg, boundary).reference_velocity
    tol = step_tol * U_INF / reference
    raw = cylinder_raw(dx, STEP_CAP, tol, rtol)
    cfg = SimConfig.from_dict(raw)
    solver = StaggeredSolver(mesh, cfg, boundary)
    predictor = ImplicitEulerPredictor(mesh, cfg, boundary, dt)
    solver._predictor = predictor
    j, i = wake_probe(mesh)
    u_f, v_f, p = u0.copy(), v0.copy(), p0.copy()
    rows = []
    started = datetime.now().isoformat(timespec="seconds")
    every = max(1, steps // 50)
    print(
        f"arm {arm}{tag}: dt {dt} rtol {rtol:g} step tol {step_tol:g} U = {tol:.3e} "
        f"(reference velocity {reference:g}) start {started}",
        flush=True,
    )
    for n in range(steps):
        predictor.u_old, predictor.v_old = u_f.copy(), v_f.copy()
        solver_module.allocate_fields = fields_of(u_f, v_f, p)
        t0 = perf_counter()
        _, v_c, p = solver.solve_steady()
        wall = perf_counter() - t0
        faces = solver.face_velocities
        assert faces is not None
        u_f, v_f = np.array(faces.u), np.array(faces.v)
        history = solver.residual_history
        at = {
            f"{level:g}": next(
                (k + 1 for k, r in enumerate(history) if r * reference < level * U_INF),
                None,
            )
            for level in THRESHOLDS
        }
        live = mesh.cell_type != 2
        rows.append(
            {
                "step": n + 1,
                "t": (n + 1) * dt,
                "outer": len(history),
                "outer_at": at,
                "converged": bool(solver.converged),
                "wall_seconds": wall,
                "stage_seconds": dict(solver.stage_seconds),
                "cap_hits": int(solver.pressure_cap_hits),
                "residual_end_over_U": float(history[-1] * reference / U_INF),
                "worst_imbalance": float(
                    np.max(np.abs(solver.last_mass_imbalance[live]))
                ),
                "wake_v": float(v_c[j, i]),
            }
        )
        if (n + 1) % every == 0 or n + 1 == steps:
            print(
                f"arm {arm}{tag} step {n + 1}: {rows[-1]['outer']} outer (to 1e-3 at "
                f"{at['0.001']}), {wall:.1f} s, imbalance "
                f"{rows[-1]['worst_imbalance']:.1e}, wake v {rows[-1]['wake_v']:+.4f}",
                flush=True,
            )
    outer = np.array([r["outer"] for r in rows])
    seconds = np.array([r["wall_seconds"] for r in rows])
    record = {
        "kind": "march",
        "arm": arm,
        "tag": tag,
        "perturbed_startup": perturbed,
        "dt": dt,
        "pressure_rtol": rtol,
        "dx": dx,
        "grid": [cfg.nx, cfg.ny],
        "step_tol_over_U": step_tol,
        "reference_velocity": reference,
        "courant": U_INF * dt / dx,
        "steps": steps,
        "started": started,
        "commit": commit(),
        "concurrent_processes": concurrent,
        "outer_mean": float(outer.mean()),
        "outer_median": float(np.median(outer)),
        "outer_max": int(outer.max()),
        "outer_at_mean": {
            key: float(
                np.mean([r["outer_at"][key] for r in rows if r["outer_at"][key]])
            )
            for key in rows[0]["outer_at"]
            if any(r["outer_at"][key] for r in rows)
        },
        "seconds_per_step_mean": float(seconds.mean()),
        "seconds_per_outer": float(seconds.sum() / outer.sum()),
        "all_converged": all(r["converged"] for r in rows),
        "wake_sign_changes": int(
            np.sum(np.diff(np.sign([r["wake_v"] for r in rows])) != 0)
        ),
        "rows": rows,
        "raw": raw,
    }
    (OUT / f"cylinder_{arm}{tag}.json").write_text(
        json.dumps(record, indent=1), encoding="utf-8"
    )
    print(
        f"arm {arm}{tag}: outer per step mean {record['outer_mean']:.1f} (median "
        f"{record['outer_median']:g}, max {record['outer_max']}), "
        f"{record['seconds_per_step_mean']:.2f} s per step, "
        f"{record['seconds_per_outer']:.3f} s per outer, wake sign changes "
        f"{record['wake_sign_changes']}",
        flush=True,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 59 probe B, the cylinder step")
    parser.add_argument("--dx", type=float, default=0.1)
    parser.add_argument("--startup", type=int, default=200)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--arms", default="ABC")
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--concurrent", type=int, default=1)
    args = parser.parse_args()
    print(
        f"cylinder59 commit {commit()} dx {args.dx} startup {args.startup} steps "
        f"{args.steps} arms {args.arms} perturb {args.perturb}",
        flush=True,
    )
    _, u0, v0, p0 = startup(args.dx, args.startup, args.concurrent, args.perturb)
    for arm in args.arms:
        march(
            arm,
            args.dx,
            args.steps,
            u0,
            v0,
            p0,
            args.concurrent,
            args.tag,
            args.perturb,
        )


if __name__ == "__main__":
    main()
