"""VAL-016 measurement of prompt 45 (ECR-002 step 6, item 6): plane Couette flow and the channel.

Usage:
    python couette45.py run KIND VARIANT NY [--length L] [--dx DX] [--cfl C]
                            [--tol T] [--label NAME]
    python couette45.py reference VARIANT NY REFINE [--kind KIND]

Defaults: a 120 m channel (400 gaps) in 1 m cells along x, cfl_number 0.25,
iteration_error_tol 1e-6. tables45.py tabulates the records and mech45.py
measures the second-cell strain.

KIND is "couette" (lower wall at rest, upper wall a velocity inlet with zero
normal velocity moving at U_W) or "channel" (both walls at rest). The inlet is
uniform, U_W / 2 for Couette and U_M for the channel, with the intensity and
dissipation length of tests/couette_reference.py, which holds the case's
numbers; a pressure outlet closes the far end. Air, rho 1.2, a 0.3 m gap. The solver is the committed StaggeredSolver; the only probe-side
change is a recording subclass of ErrorEstimateRule that asks for the
imbalance every outer iteration (it is a pure function of the faces), so the
iteration from which each of the five conditions holds is known.

The reference is tests/couette_reference.py's one-dimensional solve, which
imports nothing from src/; the run checks its copied constants against
src/turbulence.py first.

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

from tests.couette_reference import (  # noqa: E402 -- follows sys.path.insert
    E_WALL,
    GAP,
    KAPPA,
    NU,
    RHO,
    U_M,
    U_W,
    VARIANTS,
    Y_FLOOR,
    case_raw,
    reference,
)


def _check_constants() -> None:
    from src import turbulence

    assert (turbulence.KAPPA, turbulence.E_WALL) == (KAPPA, E_WALL)
    assert turbulence.Y_STAR_FLOOR == Y_FLOOR
    for name, const in VARIANTS.items():
        built = turbulence.VARIANTS[name]
        for key, value in const.items():
            assert getattr(built, key) == value, (name, key)


# ---------------------------------------------------------------------------
# The two-dimensional solve
# ---------------------------------------------------------------------------


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
    raw = case_raw(
        args.kind,
        args.variant,
        args.ny,
        args.length,
        args.dx,
        args.cfl,
        args.tol,
        args.max_pressure_iter,
    )
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
        "iteration_error_tol": args.tol,
        "stop": solver.stop_reason, "outer": n, "seconds": seconds,
        "rule_version": solver.rule_version, "nu_scale": rule._nu_scale,
        "pressure_cap_hits": solver.pressure_cap_hits,
        "max_pressure_iter": args.max_pressure_iter,
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
    p.add_argument("--length", type=float, default=120.0)
    p.add_argument("--dx", type=float, default=1.0)
    p.add_argument("--cfl", type=float, default=0.25)
    p.add_argument("--label", default="")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--max-pressure-iter", type=int, default=5000)
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
