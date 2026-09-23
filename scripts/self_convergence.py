"""Reference-free self-convergence of the lid-driven cavity, for both solvers.

Measures each solver's observed order against itself, with no reference, at
20x20, 40x40 and 80x80 with the committed case settings. Fields are saved under
results/self_convergence/ (gitignored) and never re-solved once saved. Fine
fields are restricted by averaging 2x2 blocks, which is exact tiling and
second-order for cell-centered values: d1 = R(f40) - f20, d2 = R(f80) - f40,
and p = log2(||d1|| / ||R(d2)||). Pressure has its domain mean removed first,
because each grid pins it at a different point. The control, synthetic fields
F + C h^q G with known q through the identical pipeline, runs first and stops
the script if any estimate misses q by 0.05 or more.

--extrapolate asks, from saved fields and with no solve, whether the staggered
solver's remaining distance to Ghia on the true centerlines is Ghia's:
pointwise Richardson extrapolation at Ghia's stations, after its own control,
with the true-centerline metric for both solvers. The fields saved at the
case's convergence_tol carry iteration error as large as the steps it reads,
so --solve-tight first continues the staggered solves to 1e-9.

Run:

    python scripts/self_convergence.py             # control, solves, analysis
    python scripts/self_convergence.py --solve-only
    python scripts/self_convergence.py --solve-tight   # staggered to 1e-9
    python scripts/self_convergence.py --extrapolate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.boundary import BoundaryManager  # noqa: E402 -- follows sys.path.insert
from src.boundary_staggered import (  # noqa: E402 -- follows sys.path.insert
    StaggeredBoundary,
)
from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.mesh import FLUID, Mesh  # noqa: E402 -- follows sys.path.insert
from src.solver_ns import (  # noqa: E402 -- follows sys.path.insert
    IterationState,
    NavierStokesSolver,
)
from src.solver_staggered import (  # noqa: E402 -- follows sys.path.insert
    StaggeredSolver,
)
from validation.cases import (  # noqa: E402 -- follows sys.path.insert
    case_path,
    load_case,
)
from validation.metrics import (  # noqa: E402 -- follows sys.path.insert
    GHIA_U_VAL,
    GHIA_U_Y,
    GHIA_V_VAL,
    GHIA_V_X,
    _lid_velocity,
    cavity_centerline_errors,
    cavity_true_centerline_errors,
    cavity_true_centerline_profiles,
)

GRIDS = (20, 40, 80)
METHODS = ("collocated-jacobi", "staggered-jacobi")
FIELD_DIR = REPO_ROOT / "results" / "self_convergence"
MAP_DIR = REPO_ROOT / "docs" / "reports"
CONTROL_TOL = 0.05
# |p - 2| below this counts as second order. Inside it the Richardson factor
# 1 / (2^p - 1) stays within 27% of the 1/3 the extrapolation applies.
ORDER_BAND = 0.25
# The case's convergence_tol stops the staggered cavity with iteration error
# larger than the grid-to-grid steps Richardson reads (test 21, C20), so those
# solves are continued to TIGHT_TOL, set in memory. 80x80 needs about 12800
# outer iterations to get there.
TIGHT_TOL = 1.0e-9
TIGHT_MAX_ITER = 40000
SNAPSHOT_TOLS = (1.0e-7, 1.0e-8)


def solve_and_save(method: str, n: int) -> Path:
    """Solve the n x n cavity with one solver and save u, v, p, unless already saved."""
    path = FIELD_DIR / f"{method}_{n}.npz"
    if path.exists():
        return path
    config = load_case("cavity", grid=(n, n))
    mesh = Mesh(config)
    solver: NavierStokesSolver | StaggeredSolver
    if method == "collocated-jacobi":
        solver = NavierStokesSolver(mesh, config, BoundaryManager(mesh, config))
    else:
        solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    start = time.perf_counter()
    u, v, p = solver.solve_steady()
    seconds = time.perf_counter() - start
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, u=u, v=v, p=p, outer=len(solver.residual_history), seconds=seconds)
    print(f"solved {method} {n}x{n} in {seconds:.0f} s")
    return path


def solve_tight(n: int) -> Path:
    """Continue the staggered n x n solve to TIGHT_TOL, snapshotting u and v on the way.

    The tolerance and iteration cap are set in memory; the case file is not
    touched. A snapshot is taken the first time the residual falls below the
    case's own convergence_tol, each of SNAPSHOT_TOLS, and TIGHT_TOL.

    Parameters
    ----------
    n : int
        Cells per side.

    Returns
    -------
    Path
        results/self_convergence/staggered-jacobi_<n>_tol1e-9.npz, holding
        u_<tol>, v_<tol> and outer_<tol> per snapshot (tol formatted as 1e-06)
        and the wall time in seconds. An existing file is returned unsolved.

    Raises
    ------
    SystemExit
        If the solve stops before TIGHT_TOL, or if the snapshot at the case's
        tolerance is not bitwise identical to the saved staggered field, in
        which case this is not a continuation of that computation.
    """
    path = FIELD_DIR / f"staggered-jacobi_{n}_tol1e-9.npz"
    if path.exists():
        return path
    raw = yaml.safe_load(case_path("cavity").read_text(encoding="utf-8"))
    raw["domain"]["nx"] = raw["domain"]["ny"] = n
    case_tol = float(raw["solver"]["convergence_tol"])
    raw["solver"]["convergence_tol"] = TIGHT_TOL
    raw["solver"]["max_simple_iter"] = TIGHT_MAX_ITER
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    levels = [case_tol, *SNAPSHOT_TOLS, TIGHT_TOL]
    snaps: dict[str, np.ndarray] = {}

    def snapshot(state: IterationState) -> None:
        while levels and state.residual < levels[0]:
            tag = f"{levels.pop(0):.0e}"
            snaps[f"u_{tag}"], snaps[f"v_{tag}"] = state.u.copy(), state.v.copy()
            snaps[f"outer_{tag}"] = np.array(state.iteration + 1)

    start = time.perf_counter()
    solver.solve_steady(on_iteration=snapshot)
    seconds = time.perf_counter() - start
    if levels:
        raise SystemExit(f"{n}x{n} stopped before reaching {levels[0]:.0e}")
    tag = f"{case_tol:.0e}"
    with np.load(FIELD_DIR / f"staggered-jacobi_{n}.npz") as saved:
        same = np.array_equal(snaps[f"u_{tag}"], saved["u"]) and np.array_equal(
            snaps[f"v_{tag}"], saved["v"]
        )
    if not same:
        raise SystemExit(f"{n}x{n}: the {tag} snapshot differs from the saved field")
    np.savez(path, seconds=seconds, **snaps)
    print(f"solved staggered {n}x{n} to {TIGHT_TOL:.0e} in {seconds:.0f} s")
    return path


def restrict(f: np.ndarray) -> np.ndarray:
    """Average each 2x2 block of a [2m, 2m] cell-centered field onto the [m, m] grid."""
    m = f.shape[0] // 2
    return f.reshape(m, 2, m, 2).mean(axis=(1, 3))


def centerline_extremum(profile: np.ndarray, kind: str) -> float:
    """Extremum of a sampled profile, refined by the parabola through three samples.

    The bare sample extremum has an O(h^2) error that depends on where the true
    one falls between samples, so it is not a power law in h; the fit is not.
    """
    k = int(np.argmin(profile) if kind == "min" else np.argmax(profile))
    if k in (0, len(profile) - 1):
        return float(profile[k])
    a, b, c = profile[k - 1], profile[k], profile[k + 1]
    return float(b - (c - a) ** 2 / (8.0 * (a - 2.0 * b + c)))


def functionals(u: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """Min u on x = 0.5, max and min v on y = 0.5, and the midpoint-sum kinetic energy.

    On an even grid the centerlines are face lines, so each profile is the
    mean of the two cell rows or columns either side of it.
    """
    n = u.shape[0]
    u_mid = 0.5 * (u[:, n // 2 - 1] + u[:, n // 2])
    v_mid = 0.5 * (v[n // 2 - 1, :] + v[n // 2, :])
    energy = 0.5 * (u**2 + v**2) / n**2
    out = {
        "u_min": centerline_extremum(u_mid, "min"),
        "v_max": centerline_extremum(v_mid, "max"),
        "v_min": centerline_extremum(v_mid, "min"),
        "kinetic_energy": float(energy.sum()),
    }
    # The whole-domain energy is no order instrument when the wall strip and the
    # interior change with opposite signs, so both parts are reported apart. The
    # strips are fixed in space: 0.05 and 0.1 are whole cells on every grid.
    for strip in (0.05, 0.1):
        k = round(strip * n)
        inner = float(energy[k : n - k, k : n - k].sum())
        out[f"ke_inside_{strip}"] = inner
        out[f"ke_strip_{strip}"] = out["kinetic_energy"] - inner
    return out


def differences(
    fields: dict[int, dict[str, np.ndarray]], name: str
) -> tuple[np.ndarray, np.ndarray]:
    """d1 = R(f40) - f20 on the 20x20 grid and d2 = R(f80) - f40 on the 40x40 grid."""
    f = {n: fields[n][name] for n in GRIDS}
    if name == "p":
        f = {n: g - g.mean() for n, g in f.items()}
    n1, n2, n3 = GRIDS
    return restrict(f[n2]) - f[n1], restrict(f[n3]) - f[n2]


def orders(fields: dict[int, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Observed orders of every field (max and L2 norms) and every functional.

    ``fields`` maps each grid size n to its "u", "v" and "p" arrays, [n, n].
    Per field the result holds p_max, p_l2 and the norms of d1 and R(d2); per
    functional its three values and p, which is NaN if the differences did not
    shrink monotonically.
    """
    n1 = GRIDS[0]
    out: dict[str, dict[str, float]] = {}
    for name in ("u", "v", "p"):
        d1, d2 = differences(fields, name)
        d2c = restrict(d2)
        norms = {
            "max_d1": float(np.abs(d1).max()),
            "max_d2": float(np.abs(d2c).max()),
            "l2_d1": float(np.sqrt(np.mean(d1**2))),
            "l2_d2": float(np.sqrt(np.mean(d2c**2))),
        }
        norms["p_max"] = float(np.log2(norms["max_d1"] / norms["max_d2"]))
        norms["p_l2"] = float(np.log2(norms["l2_d1"] / norms["l2_d2"]))
        out[name] = norms
    values = {n: functionals(fields[n]["u"], fields[n]["v"]) for n in GRIDS}
    for key in values[n1]:
        a, b, c = (values[n][key] for n in GRIDS)
        ratio = (a - b) / (b - c)
        p = float(np.log2(ratio)) if ratio > 0 else float("nan")
        out[key] = {"f20": a, "f40": b, "f80": c, "p": p}
    return out


def region_masks(n: int) -> dict[str, np.ndarray]:
    """The 2x2 blocks at the two top corners, the two-cell wall band without them,
    and the interior, on an n x n grid whose row 0 is the floor."""
    corner = np.zeros((n, n), dtype=bool)
    corner[n - 2 :, :2] = corner[n - 2 :, n - 2 :] = True
    band = np.zeros((n, n), dtype=bool)
    band[:2, :] = band[n - 2 :, :] = band[:, :2] = band[:, n - 2 :] = True
    band &= ~corner
    return {"top_corners": corner, "wall_band": band, "interior": ~(corner | band)}


def wall_split_masks(n: int) -> dict[str, np.ndarray]:
    """Supplementary regions: the wall band of region_masks split by wall, with the
    side walls halved at y = 0.5, and the cells within 0.15 of either top corner."""
    band = region_masks(n)["wall_band"]
    j, i = np.indices((n, n))
    lid, floor = band & (j >= n - 2), band & (j < 2)
    sides = band & ~lid & ~floor
    xc, yc = (i + 0.5) / n, (j + 0.5) / n
    near = np.minimum(np.hypot(xc, 1 - yc), np.hypot(1 - xc, 1 - yc)) < 0.15
    return {
        "band_lid": lid,
        "band_sides_upper": sides & (j >= n // 2),
        "band_sides_lower": sides & (j < n // 2),
        "band_floor": floor,
        "within_0.15_of_top_corner": near,
    }


def region_shares(d: np.ndarray) -> dict[str, float]:
    """Share of sum(d^2) in each region of region_masks and wall_split_masks."""
    total = float(np.sum(d**2))
    masks = region_masks(d.shape[0]) | wall_split_masks(d.shape[0])
    return {k: float(np.sum(d[m] ** 2)) / total for k, m in masks.items()}


def synthetic(n: int, q: float, c: float) -> dict[str, np.ndarray]:
    """F + C h^q G at the cell centers of an n x n grid, one smooth pair per field.

    F has interior extrema on the centerlines where the functionals look, and
    G is flat across each centerline pair, so the functionals respond to the
    error linearly and the control tests the instrument, not min().
    """
    h = 1.0 / n
    xc = (np.arange(n) + 0.5) * h
    x, y = np.meshgrid(xc, xc)
    eps = c * h**q
    s, co, px, py = np.sin, np.cos, np.pi * x, np.pi * y
    return {
        "u": 4.0 - 0.1 * s(px) * s(py) + eps * (1.0 + 0.5 * co(2 * py) * co(px)),
        "v": 4.0 + 0.1 * s(2 * px) * s(py) + eps * (1.0 + 0.5 * co(4 * px) * co(py)),
        "p": 0.1 * s(px) * co(py) + eps * co(px) * co(2 * py),
    }


def run_control(c: float = 1.0) -> dict[str, dict[str, float]]:
    """Run the pipeline on synthetic fields for q = 2, 1, 0.5; raise if any order misses."""
    results: dict[str, dict[str, float]] = {}
    for q in (2.0, 1.0, 0.5):
        result = orders({n: synthetic(n, q, c) for n in GRIDS})
        flat = {
            f"{key}.{name}": entry[name]
            for key, entry in result.items()
            for name in ("p_max", "p_l2", "p")
            if name in entry
        }
        results[str(q)] = flat
        worst = max(abs(p - q) for p in flat.values())
        print(f"control q={q}: worst |p - q| = {worst:.4f} over {len(flat)} estimates")
        if not worst < CONTROL_TOL:
            raise SystemExit(f"control failed at q={q}: {flat}")
    return results


def save_map(method: str, d2: dict[str, np.ndarray], out: Path) -> None:
    """Write |d2| on the 40x40 grid for u, v and p as one three-panel PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, name in zip(axes, ("u", "v", "p"), strict=True):
        image = ax.imshow(np.abs(d2[name]), origin="lower", extent=(0, 1, 0, 1))
        ax.set_title(f"|R(f80) - f40|, {name}")
        fig.colorbar(image, ax=ax, shrink=0.85)
    fig.suptitle(f"{method}: grid-to-grid difference on the 40x40 grid (lid on top)")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def upwind_faces(u: np.ndarray, v: np.ndarray, re_h: float) -> dict[str, np.ndarray]:
    """Faces where the collocated hybrid scheme is upwind: |F| / D >= 2.

    F / D is rho |u_face| h / mu with u_face the two-cell mean, the solver's
    own face value; ``re_h`` is rho h / mu. Shapes [n, n-1] (x) and [n-1, n] (y).
    """
    return {
        "x": np.abs(0.5 * (u[:, :-1] + u[:, 1:])) * re_h >= 2.0,
        "y": np.abs(0.5 * (v[:-1, :] + v[1:, :])) * re_h >= 2.0,
    }


def centerline_faces(
    u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """The staggered solver's face values on x = 0.5 and y = 0.5, from its cell means.

    Its cell-centered u is the mean of the two bounding faces and the wall face
    is zero exactly, so the faces follow by recurrence from the left wall (the
    floor for v). Returns (u on x = 0.5 at each yc, v on y = 0.5 at each xc) and
    the residual left on the far wall, which is zero when the recovery is exact.
    """
    n = u.shape[0]
    uf, vf = np.zeros((n, n + 1)), np.zeros((n + 1, n))
    for i in range(n):
        uf[:, i + 1] = 2.0 * u[:, i] - uf[:, i]
        vf[i + 1, :] = 2.0 * v[i, :] - vf[i, :]
    residual = max(float(np.abs(uf[:, n]).max()), float(np.abs(vf[n, :]).max()))
    return uf[:, n // 2], vf[n // 2, :], residual


def offset_errors(method: str, n: int, fields: dict[str, np.ndarray]) -> dict:
    """The Ghia errors as validation.metrics takes them, half a cell off the
    centerline, against the same profile taken on the centerline itself."""
    config = load_case("cavity", grid=(n, n))
    mesh = Mesh(config)
    u, v = fields["u"], fields["v"]
    out = {"metric": cavity_centerline_errors(config, mesh, u, v).components}
    if method == "staggered-jacobi":
        u_line, v_line, out["recovery_residual"] = centerline_faces(u, v)
    else:  # no faces: the mean of the two cell columns or rows either side
        u_line = 0.5 * (u[:, n // 2 - 1] + u[:, n // 2])
        v_line = 0.5 * (v[n // 2 - 1, :] + v[n // 2, :])
    col, row = mesh.cell_type[:, n // 2] == FLUID, mesh.cell_type[n // 2, :] == FLUID
    y = [0.0, *np.asarray(mesh.yc)[col], 1.0]
    x = [0.0, *np.asarray(mesh.xc)[row], 1.0]
    u_lid = _lid_velocity(config)
    u_err = np.interp(GHIA_U_Y, y, [0.0, *u_line[col], u_lid]) / u_lid - GHIA_U_VAL
    v_err = np.interp(GHIA_V_X, x, [0.0, *v_line[row], 0.0]) / u_lid - GHIA_V_VAL
    out["on_centerline"] = {
        "u": float(np.abs(u_err).max()),
        "v": float(np.abs(v_err).max()),
    }
    return out


def upwind_summary() -> dict:
    """Where the collocated hybrid scheme is upwind at each grid, with a map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out: dict = {}
    fig, axes = plt.subplots(1, len(GRIDS), figsize=(12, 3.8))
    for ax, n in zip(axes, GRIDS, strict=True):
        config = load_case("cavity", grid=(n, n))
        with np.load(FIELD_DIR / f"collocated-jacobi_{n}.npz") as data:
            faces = upwind_faces(data["u"], data["v"], config.rho / (config.mu * n))
        cell = np.zeros((n, n), dtype=bool)
        cell[:, :-1] |= faces["x"]
        cell[:, 1:] |= faces["x"]
        cell[:-1, :] |= faces["y"]
        cell[1:, :] |= faces["y"]
        rows = np.flatnonzero(cell.any(axis=1))
        out[n] = {
            "x_faces": float(faces["x"].mean()),
            "y_faces": float(faces["y"].mean()),
            "cells_touched": float(cell.mean()),
            "lowest_row_center_y": float((rows.min() + 0.5) / n) if rows.size else None,
        }
        ax.imshow(cell, origin="lower", extent=(0, 1, 0, 1), cmap="Greys")
        ax.set_title(f"{n}x{n}: {100 * cell.mean():.0f}% of cells touch an upwind face")
    fig.suptitle("collocated hybrid scheme: cells with an upwind face (|F|/D >= 2)")
    fig.tight_layout()
    fig.savefig(MAP_DIR / "cavity_self_convergence_upwind.png", dpi=80)
    plt.close(fig)
    return out


def face_gaps(
    config: SimConfig,
    mesh: Mesh,
    u: np.ndarray,
    v: np.ndarray,
    profiles: Callable[..., tuple[list[float], ...]] = cavity_true_centerline_profiles,
) -> dict[str, float]:
    """Largest gap between a metric's centerline profiles and the exact staggered faces.

    Every interior column and row of the cavity has the same FLUID cells, so one
    mask serves.

    Parameters
    ----------
    config : SimConfig
        Case configuration.
    mesh : Mesh
        Mesh of the solve.
    u, v : np.ndarray
        The staggered solver's cell-centered fields [n, n].
    profiles : Callable
        A validation.metrics profile function, the true-centerline one by
        default.

    Returns
    -------
    dict[str, float]
        The largest abs(profile - face) over the sampled rows, for u and v.
    """
    _y, u_prof, _x, v_prof = profiles(config, mesh, u, v)
    u_line, v_line, _ = centerline_faces(u, v)
    fluid = mesh.cell_type[:, u.shape[1] // 2] == FLUID
    return {
        "u": float(np.abs(np.asarray(u_prof[1:-1]) - u_line[fluid]).max()),
        "v": float(np.abs(np.asarray(v_prof[1:-1]) - v_line[fluid]).max()),
    }


def lagrange(
    nodes: np.ndarray, values: np.ndarray, targets: np.ndarray, k: int = 4
) -> np.ndarray:
    """Evaluate at each target the degree k - 1 polynomial through the k nodes around it.

    Parameters
    ----------
    nodes : np.ndarray
        Increasing node positions.
    values : np.ndarray
        Values at the nodes.
    targets : np.ndarray
        Positions to evaluate at, in any order.
    k : int
        Number of nodes per stencil, about half on each side of the target and
        shifted inward at the ends.

    Returns
    -------
    np.ndarray
        The interpolated values, one per target.
    """
    out = np.empty(len(targets))
    for t, target in enumerate(targets):
        i = int(np.clip(np.searchsorted(nodes, target) - k // 2, 0, len(nodes) - k))
        xs = nodes[i : i + k]
        weights = [
            np.prod([(target - xs[m]) / (xs[j] - xs[m]) for m in range(k) if m != j])
            for j in range(k)
        ]
        out[t] = np.dot(weights, values[i : i + k])
    return out


def richardson(
    profiles: dict[int, tuple[np.ndarray, np.ndarray]], stations: np.ndarray, k: int = 4
) -> dict[str, np.ndarray]:
    """Pointwise observed order at each station and, where it is second order, the limit.

    Parameters
    ----------
    profiles : dict[int, tuple[np.ndarray, np.ndarray]]
        (nodes, values) of one profile for each n in GRIDS.
    stations : np.ndarray
        Positions to evaluate at.
    k : int
        Interpolation stencil width, passed to lagrange.

    Returns
    -------
    dict[str, np.ndarray]
        Per station: f20, f40 and f80; the order log2((f20 - f40) / (f40 - f80)),
        NaN where the two steps have opposite signs; licensed, True where the
        order is within ORDER_BAND of 2; value, the order-2 extrapolation
        f80 + (f80 - f40) / 3 at every station; and limit, that value where
        licensed and NaN elsewhere.
    """
    f20, f40, f80 = (lagrange(*profiles[n], stations, k) for n in GRIDS)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (f20 - f40) / (f40 - f80)
        order = np.log2(np.where(ratio > 0, ratio, np.nan))
        licensed = np.abs(order - 2.0) < ORDER_BAND
    value = f80 + (f80 - f40) / 3.0
    return {
        "f20": f20,
        "f40": f40,
        "f80": f80,
        "order": order,
        "licensed": licensed,
        "value": value,
        "limit": np.where(licensed, value, np.nan),
    }


def order_change(a: np.ndarray, b: np.ndarray, stations: np.ndarray) -> dict:
    """Largest change between two readings of the orders at the same stations.

    Parameters
    ----------
    a, b : np.ndarray
        Orders at the same stations, NaN where the steps reversed.
    stations : np.ndarray
        The station positions, to name where the change is.

    Returns
    -------
    dict
        max, the largest abs(a - b) over stations where both orders are finite,
        and at, its station (both None when there is none); undefined_in_one,
        the stations where exactly one order is not finite. A maximum over the
        differences alone would drop those silently.
    """
    both = np.isfinite(a) & np.isfinite(b)
    change = np.where(both, np.abs(a - b), -1.0)
    k = int(np.argmax(change))
    return {
        "max": float(change[k]) if both.any() else None,
        "at": float(stations[k]) if both.any() else None,
        "undefined_in_one": stations[np.isfinite(a) ^ np.isfinite(b)].tolist(),
    }


def synthetic_profiles(q: float) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Known limit s^3 - s/2 plus h^q (1 + s), at each grid's cell centers and walls.

    Both parts are cubics, which four-point interpolation reproduces exactly, so
    the control tests the order and extrapolation arithmetic by itself.

    Parameters
    ----------
    q : float
        Order of the synthetic error term.

    Returns
    -------
    dict[int, tuple[np.ndarray, np.ndarray]]
        (nodes, values) for each n in GRIDS, in the layout richardson takes.
    """
    out = {}
    for n in GRIDS:
        s = np.array([0.0, *(np.arange(n) + 0.5) / n, 1.0])
        out[n] = (s, s**3 - 0.5 * s + (1.0 / n) ** q * (1.0 + s))
    return out


def run_extrapolation_control(stations: np.ndarray) -> None:
    """Check richardson on synthetic profiles of known order before any real one.

    Parameters
    ----------
    stations : np.ndarray
        Positions to run the control at.

    Raises
    ------
    SystemExit
        Unless the h^2 profile gives back its known limit to rounding and the
        h profile reads as order 1 with no station licensed.
    """
    second = richardson(synthetic_profiles(2.0), stations)
    first = richardson(synthetic_profiles(1.0), stations)
    exact = stations**3 - 0.5 * stations
    if not (
        np.allclose(second["limit"], exact, rtol=0.0, atol=1e-13)
        and np.allclose(first["order"], 1.0, rtol=0.0, atol=1e-9)
        and np.isnan(first["limit"]).all()
    ):
        raise SystemExit("extrapolation control failed")


def face_profiles(
    mesh: Mesh, u: np.ndarray, v: np.ndarray, u_lid: float
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """The staggered faces on both centerlines, every row, walls appended, over u_lid.

    Parameters
    ----------
    mesh : Mesh
        Mesh of the solve; supplies the node and wall positions.
    u, v : np.ndarray
        The staggered solver's cell-centered fields [n, n].
    u_lid : float
        Lid speed, the normalization of Ghia's tables.

    Returns
    -------
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
        (y nodes, u on x = 0.5) and (x nodes, v on y = 0.5).
    """
    u_line, v_line, _ = centerline_faces(u, v)
    y = np.array([mesh.y[0], *mesh.yc, mesh.y[-1]])
    x = np.array([mesh.x[0], *mesh.xc, mesh.x[-1]])
    return (
        (y, np.array([0.0, *u_line, u_lid]) / u_lid),
        (x, np.array([0.0, *v_line, 0.0]) / u_lid),
    )


def station_orders(
    profiles: dict[int, tuple[np.ndarray, np.ndarray]],
    positions: tuple[float, ...],
    ghia: tuple[float, ...],
) -> dict:
    """Richardson at Ghia's stations under the cubic and the quintic interpolation.

    Parameters
    ----------
    profiles : dict[int, tuple[np.ndarray, np.ndarray]]
        (nodes, values) of one normalized profile for each n in GRIDS.
    positions, ghia : tuple[float, ...]
        Ghia's stations and values. The two wall stations are left out: the
        walls are exact on every grid.

    Returns
    -------
    dict
        Per station, as lists: the order and licensed under each interpolation,
        f80 - Ghia, f80 - f40, and the order-2 value minus Ghia under each.
        Then the largest cubic-to-quintic change in value per grid, and in
        order. That change estimates the cubic's interpolation error; it does
        not bound it.
    """
    stations, ref = np.array(positions[1:-1]), np.array(ghia[1:-1])
    cubic = richardson(profiles, stations)
    quintic = richardson(profiles, stations, k=6)
    shift = {n: np.abs(quintic[f"f{n}"] - cubic[f"f{n}"]) for n in GRIDS}
    return {
        "station": stations.tolist(),
        "order": cubic["order"].tolist(),
        "order_quintic": quintic["order"].tolist(),
        "licensed": cubic["licensed"].tolist(),
        "licensed_quintic": quintic["licensed"].tolist(),
        "f80_minus_ghia": (cubic["f80"] - ref).tolist(),
        "step_40_to_80": (cubic["f80"] - cubic["f40"]).tolist(),
        "order2_minus_ghia": (cubic["value"] - ref).tolist(),
        "order2_quintic_minus_ghia": (quintic["value"] - ref).tolist(),
        "interpolation_estimate": {
            n: {"max": float(d.max()), "at": float(stations[d.argmax()])}
            for n, d in shift.items()
        },
        "quintic_order_change": order_change(
            cubic["order"], quintic["order"], stations
        ),
    }


def extrapolation() -> dict:
    """Richardson on the staggered true centerlines at every snapshot tolerance.

    The control runs first. Reads saved fields only, the six at the case's
    tolerance and the three staggered files from solve_tight, and stops if any
    is missing.

    Returns
    -------
    dict
        metric: the true-centerline metric for both solvers at the case's
        tolerance and for the staggered solver at TIGHT_TOL. iteration: per
        grid, the outer count at each snapshot, the wall time, and the largest
        change in u and v from the case's tolerance to TIGHT_TOL. stations:
        station_orders at each snapshot tolerance. settling: order_change
        between the last two tolerances.
    """
    for positions in (GHIA_U_Y, GHIA_V_X):
        run_extrapolation_control(np.array(positions[1:-1]))
    saved = [FIELD_DIR / f"{m}_{n}.npz" for m in METHODS for n in GRIDS]
    tight = [FIELD_DIR / f"staggered-jacobi_{n}_tol1e-9.npz" for n in GRIDS]
    if missing := [p.name for p in saved + tight if not p.exists()]:
        raise SystemExit(f"saved fields missing, nothing re-solved: {missing}")
    configs = {n: load_case("cavity", grid=(n, n)) for n in GRIDS}
    meshes = {n: Mesh(configs[n]) for n in GRIDS}
    # The metric's own reader, so the script and the metric normalize alike.
    u_lid = _lid_velocity(configs[GRIDS[0]])
    case_tol = configs[GRIDS[0]].convergence_tol
    tags = [f"{t:.0e}" for t in (case_tol, *SNAPSHOT_TOLS, TIGHT_TOL)]
    out: dict = {"metric": {}, "iteration": {}, "stations": {}, "settling": {}}
    for path in saved:
        n = int(path.stem.rsplit("_", 1)[1])
        with np.load(path) as data:
            u, v = data["u"], data["v"]
        entry = cavity_true_centerline_errors(configs[n], meshes[n], u, v).components
        if path.stem.startswith("staggered"):
            entry["face_gap"] = face_gaps(configs[n], meshes[n], u, v)
        out["metric"][path.stem] = entry
    snaps = {}
    for n, path in zip(GRIDS, tight, strict=True):
        with np.load(path) as data:
            snaps[n] = s = {key: data[key] for key in data.files}
        first, last = tags[0], tags[-1]
        out["iteration"][n] = {
            "outer": {t: int(s[f"outer_{t}"]) for t in tags},
            "seconds": float(s["seconds"]),
            "u_change": float(np.abs(s[f"u_{last}"] - s[f"u_{first}"]).max()),
            "v_change": float(np.abs(s[f"v_{last}"] - s[f"v_{first}"]).max()),
        }
        metric = cavity_true_centerline_errors(
            configs[n], meshes[n], s[f"u_{last}"], s[f"v_{last}"]
        )
        out["metric"][f"staggered-jacobi_{n}_tol{last}"] = metric.components
    references = {"u": (GHIA_U_Y, GHIA_U_VAL), "v": (GHIA_V_X, GHIA_V_VAL)}
    for t in tags:
        lines = {
            n: face_profiles(meshes[n], snaps[n][f"u_{t}"], snaps[n][f"v_{t}"], u_lid)
            for n in GRIDS
        }
        out["stations"][t] = {
            axis: station_orders({n: lines[n][k] for n in GRIDS}, *references[axis])
            for k, axis in enumerate("uv")
        }
    for axis in "uv":
        before, after = (out["stations"][t][axis] for t in tags[-2:])
        out["settling"][axis] = order_change(
            np.array(before["order"]),
            np.array(after["order"]),
            np.array(after["station"]),
        )
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the control, the six solves, and the analysis; print and save a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--solve-only", action="store_true")
    parser.add_argument("--extrapolate", action="store_true")
    parser.add_argument("--solve-tight", action="store_true")
    args = parser.parse_args(argv)
    if args.solve_tight:
        for n in GRIDS:
            solve_tight(n)
        return 0
    if args.extrapolate:
        text = json.dumps(extrapolation(), indent=2)
        (FIELD_DIR / "extrapolation.json").write_text(text, encoding="utf-8")
        print(text)
        return 0

    # A difference spread evenly over the grid would put these shares in each region.
    masks = region_masks(GRIDS[1]) | wall_split_masks(GRIDS[1])
    uniform = {k: float(m.mean()) for k, m in masks.items()}
    summary: dict = {"control": run_control(), "uniform_shares": uniform}
    for n in GRIDS:  # the 80x80 solves come last
        for method in METHODS:
            solve_and_save(method, n)
    if args.solve_only:
        return 0
    for method in METHODS:
        fields = {}
        for n in GRIDS:
            with np.load(FIELD_DIR / f"{method}_{n}.npz") as data:
                fields[n] = {k: data[k] for k in ("u", "v", "p")}
        d2 = {name: differences(fields, name)[1] for name in ("u", "v", "p")}
        summary[method] = {
            "orders": orders(fields),
            "shares": {name: region_shares(d) for name, d in d2.items()},
        }
        save_map(method, d2, MAP_DIR / f"cavity_self_convergence_{method}.png")
        summary[method]["centerline_offset"] = {
            n: offset_errors(method, n, fields[n]) for n in GRIDS
        }
    summary["upwind"] = upwind_summary()
    text = json.dumps(summary, indent=2)
    (FIELD_DIR / "summary.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
