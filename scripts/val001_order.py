"""VAL-001 observed order under uniform refinement, ECR-001 acceptance criterion 4.

Solves the channel at 40x20, 80x40 and 160x80 with the staggered solver under
the case file's stopping rule. Fields are saved under results/val001_order/
(gitignored) under reuse_key, the solver parameters they were solved with, the
stopping rule's RULE_VERSION and the pressure solve's PRESSURE_SOLVER_VERSION,
and a saved field is re-solved only when that key differs from the current one.
A solve that reaches its cap stops the script.

Each profile is u at x = L/2 exactly, the mean of the two cell columns either
side of that face, and at x = 3L/4 the same way; every nx here is a multiple
of four. Fine profiles are restricted to the coarse rows by averaging pairs,
as scripts/self_convergence.py restricts its blocks. Three orders:

- reference-free, the one criterion 4 is judged by (ECR-001 section 9 note):
  p = log2(||f40 - R f80|| / ||R(f80 - R f160)||), RMS, with the max norm beside it;
- against the parabola at L/2 and at 3L/4: log2(e_n / e_2n) for each pair of
  grids, e the relative L2 error over every row of the profile.

The control, synthetic fields of known order q through the identical pipeline,
runs first and stops the script if any order misses q by CONTROL_TOL or more.

Limit of the instrument (review 25 S2). The pair restriction and the
two-column station each carry an O(h^2) error of their own, about half the
real signal at 40x20. For a true order q between 1 and 2 that pulls the
reference-free order toward 2: synthetic q = 1.5 and 1.8 at about twice the
real signal's amplitude read 1.62 and 1.85, so near 1.8 the judged order can
read about 0.05 high. The control above runs q = 2 and q = 1, where the pull
is absent, and does not cover that band. On the recorded fields the pull is
absent (1.993 judged, 1.996 with the parabola subtracted before restriction,
docs/reports/val001_revalidation_step7.md section 2), and the orders on record
are not retaken. The instrument cannot show an order above 2.

Run:

    python scripts/val001_order.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark import solver_parameters  # noqa: E402 -- scripts/ is on sys.path

from src.boundary_staggered import (  # noqa: E402 -- follows sys.path.insert
    StaggeredBoundary,
)
from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.mesh import Mesh  # noqa: E402 -- follows sys.path.insert
from src.pressure import (  # noqa: E402 -- follows sys.path.insert
    PRESSURE_SOLVER_VERSION,
)
from src.solver_staggered import (  # noqa: E402 -- follows sys.path.insert
    StaggeredSolver,
)
from src.stopping import RULE_VERSION  # noqa: E402 -- follows sys.path.insert
from validation.cases import load_case  # noqa: E402 -- follows sys.path.insert
from validation.metrics import (  # noqa: E402 -- follows sys.path.insert
    inlet_velocity,
    poiseuille_l2_error,
    poiseuille_reference,
)

GRIDS = ((40, 20), (80, 40), (160, 80))
STATIONS = {"L/2": 0.5, "3L/4": 0.75}
FIELD_DIR = REPO_ROOT / "results" / "val001_order"
CONTROL_TOL = 0.05


def reuse_key(config: SimConfig) -> str:
    """The key a saved solve is reused under: the case, its mesh, the solver and both versions.

    Parameters
    ----------
    config : SimConfig
        The channel configuration the solve would run under.

    Returns
    -------
    str
        A JSON string with sorted keys.

    Notes
    -----
    A saved field is a function of the domain, the grid and its clustering,
    the fluid and the inlet as well as of the solver block (review 25 S8), so
    a case file edited in any of them is solved again. A new stopping
    condition changes no solver parameter, so RULE_VERSION joins them. Nor
    need a new pressure solve: ECR-003 changed the field only because
    pressure_rtol replaced pressure_tol, so PRESSURE_SOLVER_VERSION joins
    them too (review 37 S4).
    """
    case = {
        "width": config.room_width,
        "height": config.room_height,
        "nx": config.nx,
        "ny": config.ny,
        "stretch_x": dataclasses.asdict(config.stretch_x),
        "stretch_y": dataclasses.asdict(config.stretch_y),
        "density": config.rho,
        "viscosity": config.mu,
        "inlet_velocity": inlet_velocity(config),
    }
    key = solver_parameters(config) | {
        "case": case,
        "rule_version": RULE_VERSION,
        "pressure_solver_version": PRESSURE_SOLVER_VERSION,
    }
    return json.dumps(key, sort_keys=True)


def solve(nx: int, ny: int) -> dict[str, np.ndarray]:
    """The saved staggered solve at nx x ny, solved first if absent or stale.

    Parameters
    ----------
    nx, ny : int
        Cells along x and y.

    Returns
    -------
    dict[str, np.ndarray]
        The saved arrays: params, u, outer, seconds, stop_reason, imbalance,
        signed_sum and metric.

    Raises
    ------
    SystemExit
        If the solve reaches its cap.
    """
    config = load_case("poiseuille", grid=(nx, ny))
    params = reuse_key(config)
    path = FIELD_DIR / f"poiseuille_{nx}x{ny}.npz"
    if path.exists():
        with np.load(path) as saved:
            if str(saved["params"]) == params:
                return {key: saved[key] for key in saved.files}
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    start = time.perf_counter()
    u, _v, _p = solver.solve_steady()
    seconds = time.perf_counter() - start
    if not solver.converged:
        raise SystemExit(f"{nx}x{ny} stopped at its cap, {config.max_simple_iter}")
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        params=params,
        u=u,
        outer=len(solver.residual_history),
        seconds=seconds,
        stop_reason=solver.stop_reason,
        imbalance=np.abs(solver.last_mass_imbalance).max(),
        signed_sum=solver.last_mass_imbalance.sum(),
        metric=poiseuille_l2_error(config, mesh, u).value,
    )
    print(f"solved {nx}x{ny}: {len(solver.residual_history)} outer, {seconds:.0f} s")
    return solve(nx, ny)


def station(u: np.ndarray, fraction: float) -> np.ndarray:
    """u on the face at x = fraction * L of a uniform grid: the two columns' mean.

    Parameters
    ----------
    u : np.ndarray
        Cell-centered x-velocity, shape [ny, nx].
    fraction : float
        Position along the channel as a fraction of its length, strictly
        between 0 and 1 and a multiple of 1 / nx.

    Returns
    -------
    np.ndarray
        The mean of the two columns either side of the face, shape [ny].

    Raises
    ------
    ValueError
        If the position is not an interior face: at 0 the column before the
        face would wrap to the last one, and at 1 there is no column after.
    """
    i = fraction * u.shape[1]
    if not i.is_integer():
        raise ValueError(f"x = {fraction} L is not a face of {u.shape[1]} columns")
    if not 0 < i < u.shape[1]:
        raise ValueError(
            f"x = {fraction} L is not an interior face of {u.shape[1]} columns"
        )
    return 0.5 * (u[:, int(i) - 1] + u[:, int(i)])


def restrict(f: np.ndarray) -> np.ndarray:
    """Average each pair of rows onto the grid with half as many.

    Parameters
    ----------
    f : np.ndarray
        A profile with an even number of rows, shape [2m].

    Returns
    -------
    np.ndarray
        The pair means, shape [m].
    """
    return f.reshape(-1, 2).mean(axis=1)


def developed(ny: int, u_mean: float) -> np.ndarray:
    """The fully developed profile at ny uniform row centres.

    Parameters
    ----------
    ny : int
        Rows.
    u_mean : float
        Mean speed, in m/s.

    Returns
    -------
    np.ndarray
        validation.metrics.poiseuille_reference on the unit height, shape [ny].
        The profile in s = y / H does not depend on H, so the study works in s.
    """
    return poiseuille_reference((np.arange(ny) + 0.5) / ny, 1.0, u_mean)


def orders(fields: list[np.ndarray], u_mean: float) -> dict[str, dict]:
    """Every order, and the errors against the parabola, from three u fields, coarse first.

    Parameters
    ----------
    fields : list[np.ndarray]
        Cell-centered x-velocity on the 40x20, 80x40 and 160x80 grids.
    u_mean : float
        The inlet speed, in m/s.

    Returns
    -------
    dict[str, dict]
        "orders": reference_free, reference_free_max and, per station, the
        two orders against the parabola; "errors": the relative L2 error at
        each station on each grid.
    """
    f = [station(u, STATIONS["L/2"]) for u in fields]
    d1, d2 = f[0] - restrict(f[1]), restrict(f[1] - restrict(f[2]))
    found = {
        "reference_free": [float(np.log2(np.sqrt(np.mean(d1**2) / np.mean(d2**2))))],
        "reference_free_max": [float(np.log2(np.abs(d1).max() / np.abs(d2).max()))],
    }
    errors = {}
    for name, fraction in STATIONS.items():
        e = []
        for u in fields:
            ref = developed(u.shape[0], u_mean)
            e.append(
                float(
                    np.sqrt(np.sum((station(u, fraction) - ref) ** 2) / np.sum(ref**2))
                )
            )
        found[f"parabola_{name}"] = [float(np.log2(e[k] / e[k + 1])) for k in (0, 1)]
        errors[name] = e
    return {"orders": found, "errors": errors}


def run_control(u_mean: float) -> dict[str, dict]:
    """Synthetic P + A sin(4 pi x / L)(1 + s) + u_mean h^q G through orders().

    Parameters
    ----------
    u_mean : float
        The inlet speed, in m/s, setting the profile and the amplitudes.

    Returns
    -------
    dict[str, dict]
        The orders found at q = 2.0 and q = 1.0, keyed by str(q).

    Raises
    ------
    SystemExit
        Unless every order is within CONTROL_TOL of q.

    Notes
    -----
    The sine is zero on both stations but not beside them, so a station read
    off one column carries an O(h) error that the two columns' mean does not.
    P restricted by pairs is off by O(h^2), so q above 2 would read as 2 and
    is not a control.
    """
    results = {}
    for q in (2.0, 1.0):
        fields = []
        for nx, ny in GRIDS:
            t = (np.arange(nx) + 0.5)[None, :] / nx
            s = (np.arange(ny) + 0.5)[:, None] / ny
            g = (1.0 + t) * (1.0 + 0.5 * np.cos(np.pi * s))
            wave = 0.05 * u_mean * np.sin(4.0 * np.pi * t) * (1.0 + s)
            fields.append(developed(ny, u_mean)[:, None] + wave + u_mean * g / ny**q)
        results[str(q)] = found = orders(fields, u_mean)["orders"]
        worst = max(abs(p - q) for values in found.values() for p in values)
        print(f"control q={q}: worst |p - q| = {worst:.4f}")
        if not worst < CONTROL_TOL:
            raise SystemExit(f"control failed at q={q}: {found}")
    return results


def main() -> int:
    """Run the control, the three solves and the orders; print and save summary.json.

    Returns
    -------
    int
        Exit status, 0 when the run completes.

    Raises
    ------
    SystemExit
        If the control fails or a solve reaches its cap.
    """
    u_mean = inlet_velocity(load_case("poiseuille"))
    summary: dict = {"control": run_control(u_mean)}
    saved = [solve(nx, ny) for nx, ny in GRIDS]
    fields = [s["u"] for s in saved]
    summary |= orders(fields, u_mean)
    # Refinement does not remove the flow's development between the two stations.
    summary["l2_minus_3l4"] = [
        float(np.sqrt(np.sum((station(u, 0.5) - station(u, 0.75)) ** 2)))
        / float(np.sqrt(np.sum(developed(u.shape[0], u_mean) ** 2)))
        for u in fields
    ]
    summary["solves"] = {
        f"{nx}x{ny}": {k: s[k].item() for k in s if k not in ("u", "params")}
        for (nx, ny), s in zip(GRIDS, saved, strict=True)
    }
    text = json.dumps(summary, indent=2)
    (FIELD_DIR / "summary.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
