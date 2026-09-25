"""VAL-001 observed order under uniform refinement, ECR-001 acceptance criterion 4.

Solves the channel at 40x20, 80x40 and 160x80 with the staggered solver under
the case file's stopping rule. Fields are saved under results/val001_order/
(gitignored) with the solver parameters they were solved with, and a saved
field is re-solved only when those differ from the case file's. A solve that
reaches its cap stops the script.

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

Run:

    python scripts/val001_order.py
"""

from __future__ import annotations

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
from src.mesh import Mesh  # noqa: E402 -- follows sys.path.insert
from src.solver_staggered import (  # noqa: E402 -- follows sys.path.insert
    StaggeredSolver,
)
from validation.cases import load_case  # noqa: E402 -- follows sys.path.insert
from validation.metrics import (  # noqa: E402 -- follows sys.path.insert
    _inlet_velocity,
    poiseuille_l2_error,
)

GRIDS = ((40, 20), (80, 40), (160, 80))
STATIONS = {"L/2": 0.5, "3L/4": 0.75}
FIELD_DIR = REPO_ROOT / "results" / "val001_order"
CONTROL_TOL = 0.05


def solve(nx: int, ny: int) -> dict[str, np.ndarray]:
    """The saved staggered solve at nx x ny, solved first if absent or stale.

    Raises SystemExit if the solve reaches its cap.
    """
    config = load_case("poiseuille", grid=(nx, ny))
    params = json.dumps(solver_parameters(config), sort_keys=True)
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
        metric=poiseuille_l2_error(config, mesh, u).value,
    )
    print(f"solved {nx}x{ny}: {len(solver.residual_history)} outer, {seconds:.0f} s")
    return solve(nx, ny)


def station(u: np.ndarray, fraction: float) -> np.ndarray:
    """u on the face at x = fraction * L of a uniform grid: the two columns' mean."""
    i = fraction * u.shape[1]
    if not i.is_integer():
        raise ValueError(f"x = {fraction} L is not a face of {u.shape[1]} columns")
    return 0.5 * (u[:, int(i) - 1] + u[:, int(i)])


def restrict(f: np.ndarray) -> np.ndarray:
    """Average each pair of rows onto the grid with half as many."""
    return f.reshape(-1, 2).mean(axis=1)


def parabola(ny: int, u_mean: float) -> np.ndarray:
    """The fully developed profile, 1.5 u_mean 4 s (1 - s), at ny uniform row centres."""
    s = (np.arange(ny) + 0.5) / ny
    return 6.0 * u_mean * s * (1.0 - s)


def orders(fields: list[np.ndarray], u_mean: float) -> dict[str, dict]:
    """Every order, and the errors against the parabola, from three u fields, coarse first."""
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
            ref = parabola(u.shape[0], u_mean)
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

    The sine is zero on both stations but not beside them, so a station read
    off one column carries an O(h) error that the two columns' mean does not.
    P restricted by pairs is off by O(h^2), so q above 2 would read as 2 and
    is not a control. Raises SystemExit unless every order is within
    CONTROL_TOL of q.
    """
    results = {}
    for q in (2.0, 1.0):
        fields = []
        for nx, ny in GRIDS:
            t = (np.arange(nx) + 0.5)[None, :] / nx
            s = (np.arange(ny) + 0.5)[:, None] / ny
            g = (1.0 + t) * (1.0 + 0.5 * np.cos(np.pi * s))
            wave = 0.05 * u_mean * np.sin(4.0 * np.pi * t) * (1.0 + s)
            fields.append(parabola(ny, u_mean)[:, None] + wave + u_mean * g / ny**q)
        results[str(q)] = found = orders(fields, u_mean)["orders"]
        worst = max(abs(p - q) for values in found.values() for p in values)
        print(f"control q={q}: worst |p - q| = {worst:.4f}")
        if not worst < CONTROL_TOL:
            raise SystemExit(f"control failed at q={q}: {found}")
    return results


def main() -> int:
    """Run the control, the three solves and the orders; print and save summary.json."""
    u_mean = _inlet_velocity(load_case("poiseuille"))
    summary: dict = {"control": run_control(u_mean)}
    saved = [solve(nx, ny) for nx, ny in GRIDS]
    fields = [s["u"] for s in saved]
    summary |= orders(fields, u_mean)
    # Refinement does not remove the flow's development between the two stations.
    summary["l2_minus_3l4"] = [
        float(np.sqrt(np.sum((station(u, 0.5) - station(u, 0.75)) ** 2)))
        / float(np.sqrt(np.sum(parabola(u.shape[0], u_mean) ** 2)))
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
