"""VAL-016 measurement of prompt 45 (ECR-002 step 6, item 6): plane Couette flow and the channel.

Usage:
    python couette45.py run KIND VARIANT NY [--length L] [--dx DX] [--cfl C] [--label T]
    python couette45.py reference VARIANT NY REFINE
    python couette45.py table

KIND is "couette" (lower wall at rest, upper wall a velocity inlet with zero
normal velocity moving at U_W) or "channel" (both walls at rest). The inlet is
uniform, U_W / 2 for Couette and U_M for the channel, with the intensity and
dissipation length below; a pressure outlet closes the far end. Air, rho 1.2,
a 0.3 m gap. The solver is the committed StaggeredSolver; the only probe-side
change is a recording subclass of ErrorEstimateRule that asks for the
imbalance every outer iteration (it is a pure function of the faces), so the
iteration from which each of the five conditions holds is known.

The reference is a one-dimensional steady solve of the same k-epsilon
equations with the same wall treatment across the gap, written here with
NumPy only (nothing from src/; the constants are copied from the published
values and checked against src/turbulence.py at import). Its grid has the 2D
grid's wall cells (height H / NY, so the same y_P, eps held and production
given there, the wall viscosity on the momentum's wall faces) and splits each
interior 2D cell into REFINE cells; with REFINE 1 it is the 2D stencil's own
x-invariant limit, with REFINE odd every 2D cell centre is a reference cell
centre. Face diffusivities are the distance-weighted harmonic mean, the strain
at a centre is the mean of the gradients at its two faces, as in 2D.

Records go to results/builder45/couette/ of the main checkout.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[3]
MAIN = Path(
    subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
).parent
OUT = MAIN / "results" / "builder45" / "couette"
sys.path.insert(0, str(HERE))

GAP = 0.3
RHO = 1.2
MU = 1.81e-5
NU = MU / RHO
# U_W for Re_tau about 3,000 on the half gap (ADR-012 G (ii)); off a round
# number so nothing aligns with the grid.
U_W = 15.7
# The channel's bulk velocity, the Couette inlet's.
U_M = U_W / 2.0
# The inlet's turbulence: 6% of the inflow speed and a dissipation length of a
# third of the gap, in ADR-012 C's convention, which put the inflow k and nu_t
# near the developed core's (k about 0.33 against 0.28 m^2/s^2).
INTENSITY = 0.06
DISSIPATION_LENGTH = 0.1
KAPPA, E_WALL = 0.41, 9.793
# Published constants (ADR-012 A), copied; checked against src at import.
VARIANTS = {
    "standard": {"c_mu": 0.09, "c_1": 1.44, "c_2": 1.92, "sigma_k": 1.0,
                 "sigma_e": 1.3, "eta_0": None, "beta": None},
    "rng": {"c_mu": 0.0845, "c_1": 1.42, "c_2": 1.68, "sigma_k": 0.7194,
            "sigma_e": 0.7194, "eta_0": 4.38, "beta": 0.012},
}  # fmt: skip


def log_law_floor() -> float:
    """y*_0 for KAPPA and E_WALL, by Newton's method from 2 / KAPPA."""
    y = 2.0 / KAPPA
    for _ in range(100):
        step = (KAPPA * y - math.log(E_WALL * y)) / (KAPPA - 1.0 / y)
        y -= step
        if abs(step) <= 4.0 * math.ulp(y):
            break
    return y


Y_FLOOR = log_law_floor()


def _check_constants() -> None:
    from src import turbulence

    assert (turbulence.KAPPA, turbulence.E_WALL) == (KAPPA, E_WALL)
    assert turbulence.Y_STAR_FLOOR == Y_FLOOR
    for name, const in VARIANTS.items():
        built = turbulence.VARIANTS[name]
        for key, value in const.items():
            assert getattr(built, key) == value, (name, key)


# ---------------------------------------------------------------------------
# The one-dimensional reference
# ---------------------------------------------------------------------------


def _thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = b.size
    cp, dp = np.empty(n), np.empty(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _conductance(gamma: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Harmonic face value over the centre-to-centre distance, interior faces."""
    da, db = half[:-1], half[1:]
    return 1.0 / (da / gamma[:-1] + db / gamma[1:])


def reference(
    variant: str, ny: int, refine: int, kind: str = "couette", tol: float = 1e-13
) -> dict:
    """The steady 1D solve (module docstring); k and eps by a local pseudo-time step."""
    const = VARIANTS[variant]
    h = GAP / ny
    widths = np.concatenate(([h], np.full((ny - 2) * refine, h / refine), [h]))
    faces = np.concatenate(([0.0], np.cumsum(widths)))
    yc = 0.5 * (faces[:-1] + faces[1:])
    half = 0.5 * widths
    n = widths.size
    c_mu = const["c_mu"]
    top = U_W if kind == "couette" else 0.0
    k = np.full(n, 0.3)
    eps = np.full(n, 1.0)
    y_p = 0.5 * h
    # A channel is driven by a pressure gradient set so the bulk velocity is U_M.
    dpdx = 0.0
    u = np.full(n, U_M)
    for iteration in range(200000):  # noqa: B007 -- read after the loop as the count
        nu_t = c_mu * k**2 / eps
        g = _conductance(RHO * (NU + nu_t), half)
        u_k = c_mu**0.25 * np.sqrt(k[[0, -1]])
        y_star = u_k * y_p / NU
        log_term = np.log(E_WALL * np.maximum(y_star, Y_FLOOR))
        g_wall = RHO * u_k * KAPPA / log_term  # mu_w / y_p
        a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)
        a[1:], c[:-1] = -g, -g
        b[1:] += g
        b[:-1] += g
        b[0] += g_wall[0]
        b[-1] += g_wall[1]
        d = np.zeros(n)
        d[-1] += g_wall[1] * top
        if kind == "channel":
            # Solve for the unit-gradient response and scale to the bulk flux.
            unit = _thomas(a, b, c, widths.copy())
            dpdx = -U_M * GAP / float(unit @ widths)
            u = -dpdx * unit
        else:
            u = _thomas(a, b, c, d)
        gradient = np.diff(u) / (half[:-1] + half[1:])
        low = np.concatenate(([u[0] / y_p], gradient))
        high = np.concatenate((gradient, [(top - u[-1]) / y_p]))
        s = np.abs(0.5 * (low + high))
        production = nu_t * s**2
        slip = np.abs(np.array([u[0], top - u[-1]]))
        production[[0, -1]] = u_k**2 * slip / (y_p * log_term)
        eps_wall = u_k**3 / (KAPPA * y_p)
        ratio = eps / k
        r_term = np.zeros(n)
        if const["eta_0"] is not None:
            eta = s * k / eps
            r_term = (
                c_mu * eta**3 * (1.0 - eta / const["eta_0"])
                / (1.0 + const["beta"] * eta**3) * eps * ratio
            )  # fmt: skip
        step = 1.0 / ratio
        gk = _conductance(NU + nu_t / const["sigma_k"], half)
        a, c = np.zeros(n), np.zeros(n)
        b = widths / step + ratio * widths
        a[1:], c[:-1] = -gk, -gk
        b[1:] += gk
        b[:-1] += gk
        k_new = _thomas(a, b, c, widths / step * k + production * widths)
        ge = _conductance(NU + nu_t / const["sigma_e"], half)
        a, c = np.zeros(n), np.zeros(n)
        b = (
            widths / step
            + (const["c_2"] * ratio + np.maximum(r_term, 0.0) / eps) * widths
        )
        a[1:], c[:-1] = -ge, -ge
        b[1:] += ge
        b[:-1] += ge
        d = (
            widths / step * eps
            + (const["c_1"] * ratio * production - np.minimum(r_term, 0.0)) * widths
        )
        for idx, value in ((0, eps_wall[0]), (n - 1, eps_wall[1])):
            a[idx], c[idx], b[idx], d[idx] = 0.0, 0.0, 1.0, value
        d[1] -= a[1] * eps_wall[0]
        a[1] = 0.0
        d[-2] -= c[-2] * eps_wall[1]
        c[-2] = 0.0
        eps_new = _thomas(a, b, c, d)
        change = max(
            np.max(np.abs(k_new / k - 1.0)), np.max(np.abs(eps_new / eps - 1.0))
        )
        k, eps = k_new, eps_new
        if change < tol:
            break
    nu_t = c_mu * k**2 / eps
    centres = (
        [0] + [1 + (j - 1) * refine + refine // 2 for j in range(1, ny - 1)] + [n - 1]
    )
    m = int(np.searchsorted(faces, GAP / 2.0)) - 1
    face_nu = (half[m] + half[m + 1]) / (
        half[m] / (NU + nu_t[m]) + half[m + 1] / (NU + nu_t[m + 1])
    )
    stress = face_nu * (u[m + 1] - u[m]) / (yc[m + 1] - yc[m])
    return {
        "variant": variant, "ny": ny, "refine": refine, "kind": kind,
        "iterations": iteration + 1, "change": float(change),
        "yc": yc[centres].tolist(), "u": u[centres].tolist(), "k": k[centres].tolist(),
        "u_tau_mid": math.sqrt(abs(stress)), "dpdx": dpdx,
        "tau_wall": [float(g_wall[0] * u[0]), float(g_wall[1] * abs(top - u[-1]))],
    }  # fmt: skip


# ---------------------------------------------------------------------------
# The two-dimensional solve
# ---------------------------------------------------------------------------


def config_raw(
    kind: str, variant: str, ny: int, length: float, dx: float, cfl: float
) -> dict:
    inlet_speed = U_W / 2.0 if kind == "couette" else U_M
    boundaries = {
        "inlet": {
            "type": "velocity_inlet", "location": "left", "y_start": 0.0, "y_end": GAP,
            "velocity": inlet_speed, "turbulence_intensity": INTENSITY,
            "dissipation_length": DISSIPATION_LENGTH,
        },
        "outlet": {"type": "pressure_outlet", "location": "right", "y_start": 0.0, "y_end": GAP},
    }  # fmt: skip
    if kind == "couette":
        boundaries["belt"] = {
            "type": "velocity_inlet", "location": "top", "x_start": 0.0, "x_end": length,
            "u_velocity": U_W, "v_velocity": 0.0,
        }  # fmt: skip
    return {
        "domain": {"width": length, "height": GAP, "nx": round(length / dx), "ny": ny},
        "fluid": {"density": RHO, "viscosity": MU, "temperature": 293.0},
        "particles": {
            "density": 1000.0, "sizes": [5.0e-6], "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {"diameters": [5.0e-6], "efficiencies": [0.99999]},
        },
        "solver": {
            "dt": 0.01, "t_end": 1.0, "output_interval": 10, "convergence_tol": 1e-6,
            "max_simple_iter": 20000, "alpha_velocity": 0.7, "alpha_pressure": 0.3,
            "max_pressure_iter": 5000, "pressure_rtol": 1e-8,
            "stopping_rule": "error_estimate", "momentum_sweeps": 10,
        },
        "turbulence": {
            "model": "k_epsilon", "variant": variant,
            "wall_treatment": "scalable_wall_functions", "cfl_number": cfl,
            "alpha_turbulence": 0.7, "max_iter": 500, "tol": 1e-10,
        },
        "boundaries": boundaries, "obstacles": [],
        "sensors": [{"name": "c", "x": length / 2.0, "y": GAP / 2.0}],
        "thresholds": {"5e-06": 100.0},
    }  # fmt: skip


def run(args: argparse.Namespace) -> None:
    _check_constants()
    from src import solver_staggered
    from src.boundary_staggered import StaggeredBoundary
    from src.config import SimConfig
    from src.mesh import Mesh
    from src.stopping import ErrorEstimateRule

    readings: list = []

    class RecordingRule(ErrorEstimateRule):
        def update(self, step, imbalance, viscosity_step=None):  # type: ignore[no-untyped-def]
            found = imbalance()
            readings.append(found)
            return super().update(step, lambda: found, viscosity_step=viscosity_step)

    solver_staggered.ErrorEstimateRule = RecordingRule
    raw = config_raw(args.kind, args.variant, args.ny, args.length, args.dx, args.cfl)
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    solver = solver_staggered.StaggeredSolver(
        mesh, config, StaggeredBoundary(mesh, config)
    )
    rule = solver._new_rule()
    tols = (config.iteration_error_tol, config.mass_imbalance_tol)
    started = time.perf_counter()
    rules: list = []
    real_new_rule = solver._new_rule

    def keep_rule():  # type: ignore[no-untyped-def]
        made = real_new_rule()
        rules.append(made)
        return made

    solver._new_rule = keep_rule
    _u_c, _v_c, pressure = solver.solve_steady()
    seconds = time.perf_counter() - started
    used = rules[-1]
    n = len(solver.residual_history)
    flux = solver.flux_scale
    holds = {
        "a": [e < tols[0] for e in used.estimate_history],
        "e": [e < tols[0] for e in used.viscosity_estimate_history],
        "b": [r.worst < tols[1] for r in readings],
        "c": [r.absolute_sum / flux < tols[0] for r in readings],
        "d": [abs(r.signed_sum) < tols[1] for r in readings],
    }

    def from_when(flags: list) -> int | None:
        """The first outer iteration from which the flag holds to the end."""
        if not flags or not flags[-1]:
            return None
        i = len(flags) - 1
        while i > 0 and flags[i - 1]:
            i -= 1
        return i

    faces, state = solver.face_velocities, solver.turbulence_state
    u, k, nu_t = np.array(faces.u), np.array(state.k), np.array(state.nu_t)
    nx = k.shape[1]
    yc = mesh.yc

    def profile(i: int) -> tuple:
        return (
            u[:, i],
            0.5 * (k[:, i - 1] + k[:, i]),
            0.5 * (nu_t[:, i - 1] + nu_t[:, i]),
        )

    station = int(0.8 * nx)
    us, ks, ns = profile(station)
    j = args.ny // 2
    a_, b_ = NU + ns[j - 1], NU + ns[j]
    stress = 2.0 * a_ * b_ / (a_ + b_) * (us[j] - us[j - 1]) / (yc[j] - yc[j - 1])
    along = {}
    for frac in (0.6, 0.7, 0.9):
        uo, ko, _ = profile(int(frac * nx))
        along[str(frac)] = {
            "du_over_scale": float(
                np.abs(uo - us).max() / (U_W if args.kind == "couette" else U_M)
            ),
            "dk_over_kmax": float(np.abs(ko - ks).max() / ks.max()),
        }
    record = {
        "kind": args.kind, "variant": args.variant, "ny": args.ny, "nx": nx,
        "length": args.length, "dx": args.dx, "cfl": args.cfl, "label": args.label,
        "stop": solver.stop_reason, "outer": n, "seconds": seconds,
        "rule_version": solver.rule_version, "nu_scale": rule._nu_scale,
        "from_when": {key: from_when(flags) for key, flags in holds.items()},
        "station_x": float(mesh.x[station]), "profile_change_against_station": along,
        "yc": yc.tolist(), "u": us.tolist(), "k": ks.tolist(), "nu_t": ns.tolist(),
        "u_tau_mid": math.sqrt(abs(stress)),
        "k_min": float(k[mesh.cell_type != 1].min()), "eps_min": float(state.eps.min()),
    }  # fmt: skip
    # The wall shear from the wall function at the station's two wall cells,
    # and for the channel the streamwise pressure gradient there (developed
    # flow: tau_w = -(GAP / 2) dp/dx; p is p + (2/3) rho k, and k does not
    # vary along x where the flow is developed).
    c_mu = VARIANTS[args.variant]["c_mu"]
    u_k = c_mu**0.25 * np.sqrt(ks[[0, -1]])
    y_p = 0.5 * GAP / args.ny
    log_term = np.log(E_WALL * np.maximum(u_k * y_p / NU, Y_FLOOR))
    top = U_W if args.kind == "couette" else 0.0
    slip = np.abs(np.array([us[0], top - us[-1]]))
    record["tau_wall"] = (RHO * u_k * KAPPA * slip / log_term).tolist()
    i = station
    record["dpdx"] = float(
        np.mean((pressure[:, i] - pressure[:, i - 2]) / (mesh.xc[i] - mesh.xc[i - 2]))
    )
    OUT.mkdir(parents=True, exist_ok=True)
    name = (
        f"{args.kind}_{args.variant}_{args.ny}{'_' + args.label if args.label else ''}"
    )
    (OUT / f"{name}.json").write_text(json.dumps(record, indent=1))
    np.savez(
        OUT / f"{name}.npz",
        u=u,
        v=np.array(faces.v),
        k=k,
        eps=np.array(state.eps),
        nu_t=nu_t,
    )
    print(
        json.dumps(
            {
                key: record[key]
                for key in ("stop", "outer", "seconds", "from_when", "u_tau_mid")
            }
        )
    )


def reference_command(args: argparse.Namespace) -> None:
    record = reference(args.variant, args.ny, args.refine, args.kind)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"ref_{args.kind}_{args.variant}_{args.ny}_r{args.refine}"
    (OUT / f"{name}.json").write_text(json.dumps(record, indent=1))
    print(name, record["iterations"], record["change"], record["u_tau_mid"])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("kind", choices=["couette", "channel"])
    p.add_argument("variant", choices=sorted(VARIANTS))
    p.add_argument("ny", type=int)
    p.add_argument("--length", type=float, default=90.0)
    p.add_argument("--dx", type=float, default=0.3)
    p.add_argument("--cfl", type=float, default=0.25)
    p.add_argument("--label", default="")
    p = sub.add_parser("reference")
    p.add_argument("variant", choices=sorted(VARIANTS))
    p.add_argument("ny", type=int)
    p.add_argument("refine", type=int)
    p.add_argument("--kind", default="couette", choices=["couette", "channel"])
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        reference_command(args)


if __name__ == "__main__":
    main()
