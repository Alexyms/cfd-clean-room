"""Render a validation case's converged fields to a PNG.

A development instrument for reading the field during the rebuild, not a
deliverable. Default matplotlib styling, three panels: streamlines coloured
by speed, the pressure field, and the centerline profiles against their
reference (Ghia et al. for the cavity, the analytical parabola for
Poiseuille). Phase 7 owns presentation-quality visuals.

Run:

    python scripts/view_field.py val002_40x40            # solve, save, draw
    python scripts/view_field.py val002_40x40 --method staggered-jacobi
    python scripts/view_field.py results/val002_40x40.npz  # draw a saved solve

A solve writes results/<case_id>.npz, or results/<case_id>_<method>.npz for
a method other than the default, so the next look costs no solver time and
the two solvers' fields sit side by side. The method names are the
benchmark harness's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.boundary import BoundaryManager  # noqa: E402 -- follows sys.path.insert
from src.boundary_staggered import (  # noqa: E402 -- follows sys.path.insert
    StaggeredBoundary,
)
from src.mesh import Mesh  # noqa: E402 -- follows sys.path.insert
from src.solver_ns import NavierStokesSolver  # noqa: E402 -- follows sys.path.insert
from src.solver_staggered import (  # noqa: E402 -- follows sys.path.insert
    StaggeredSolver,
)
from validation.cases import (  # noqa: E402 -- follows sys.path.insert
    CASE_GRIDS,
    load_case,
)
from validation.metrics import (  # noqa: E402 -- follows sys.path.insert
    GHIA_U_VAL,
    GHIA_U_Y,
    GHIA_V_VAL,
    GHIA_V_X,
    cavity_centerline_errors,
    cavity_centerline_profiles,
    poiseuille_l2_error,
    poiseuille_profiles,
)

RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_METHOD = "collocated-jacobi"
STAGGERED_METHOD = "staggered-jacobi"
METHODS = (DEFAULT_METHOD, STAGGERED_METHOD)


def solve_and_save(case_id: str, out_dir: Path, method: str = DEFAULT_METHOD) -> Path:
    """Run a preset case with the named solver and write its fields to an .npz.

    Raises
    ------
    ValueError
        If the method is not one of METHODS.
    """
    kind, nx, ny = CASE_GRIDS[case_id]
    config = load_case(kind, grid=(nx, ny))
    mesh = Mesh(config)
    solver: NavierStokesSolver | StaggeredSolver
    if method == DEFAULT_METHOD:
        solver = NavierStokesSolver(mesh, config, BoundaryManager(mesh, config))
        stem = case_id
    elif method == STAGGERED_METHOD:
        solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
        stem = f"{case_id}_{method}"
    else:
        raise ValueError(f"unknown method {method!r}; known: {list(METHODS)}")
    u, v, p = solver.solve_steady()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.npz"
    np.savez(
        path,
        u=u,
        v=v,
        p=p,
        kind=np.array(kind),
        case_id=np.array(case_id),
        method=np.array(method),
        nx=np.array(nx),
        ny=np.array(ny),
        outer_iterations=np.array(len(solver.residual_history)),
    )
    print(f"solved {case_id} in {len(solver.residual_history)} iterations -> {path}")
    return path


def render(npz_path: Path, out_dir: Path) -> Path:
    """Draw the three panels for a saved solve and write a PNG beside it."""
    data = np.load(npz_path)
    kind = str(data["kind"])
    case_id = str(data["case_id"])
    nx, ny = int(data["nx"]), int(data["ny"])
    u, v, p = data["u"], data["v"], data["p"]

    config = load_case(kind, grid=(nx, ny))
    mesh = Mesh(config)
    xc, yc = np.asarray(mesh.xc), np.asarray(mesh.yc)
    speed = np.hypot(u, v)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    strm = ax.streamplot(xc, yc, u, v, color=speed, density=1.2, linewidth=0.8)
    fig.colorbar(strm.lines, ax=ax, label="|velocity|")
    ax.set_title("streamlines")
    ax.set_aspect("equal")

    ax = axes[1]
    cf = ax.contourf(xc, yc, p, 30)
    fig.colorbar(cf, ax=ax, label="p")
    ax.set_title("pressure")
    ax.set_aspect("equal")

    ax = axes[2]
    if kind.startswith("cavity"):
        y_prof, u_prof, x_prof, v_prof = cavity_centerline_profiles(config, mesh, u, v)
        ax.plot(y_prof, u_prof, "-", label="u along x = 0.5 (solver)")
        ax.plot(GHIA_U_Y, GHIA_U_VAL, "o", label="u Ghia 1982")
        ax.plot(x_prof, v_prof, "-", label="v along y = 0.5 (solver)")
        ax.plot(GHIA_V_X, GHIA_V_VAL, "s", label="v Ghia 1982")
        ax.set_xlabel("y for u, x for v")
        ax.set_ylabel("velocity / U_lid")
        metric = cavity_centerline_errors(config, mesh, u, v)
        summary = (
            f"max err u {metric.components['u']:.3f}, v {metric.components['v']:.3f}"
        )
    elif kind.startswith("poiseuille"):
        y, u_num, u_ref = poiseuille_profiles(config, mesh, u)
        ax.plot(u_num, y, "-", label="u at x = L/2 (solver)")
        ax.plot(u_ref, y, "o", label="analytical parabola")
        ax.set_xlabel("u")
        ax.set_ylabel("y")
        metric = poiseuille_l2_error(config, mesh, u)
        summary = f"L2 error {metric.value:.4f}"
    else:
        # A third family would otherwise be drawn against the parabola with a
        # confident L2 error for it. Refusing is the useful behaviour.
        raise ValueError(
            f"unrecognised case kind {kind!r}: the viewer has reference panels "
            "for cavity and poiseuille only"
        )
    ax.legend(fontsize=8)
    ax.set_title(f"centerline vs reference: {summary}")

    # Files saved before the method was recorded all came from the collocated solver.
    method = str(data["method"]) if "method" in data.files else DEFAULT_METHOD
    fig.suptitle(
        f"{case_id}  ({method}, {nx}x{ny}, "
        f"{int(data['outer_iterations'])} outer iterations)"
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{npz_path.stem}.png"
    fig.savefig(png, dpi=110)
    print(f"wrote {png}")
    return png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "target",
        help=f"a case id ({', '.join(sorted(CASE_GRIDS))}) or a path to a saved .npz",
    )
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--method",
        choices=METHODS,
        default=DEFAULT_METHOD,
        help="Solver for a case id; ignored when drawing a saved .npz",
    )
    args = parser.parse_args(argv)

    if args.target.endswith(".npz"):
        npz_path = Path(args.target)
    elif args.target in CASE_GRIDS:
        npz_path = solve_and_save(args.target, args.out, args.method)
    else:
        parser.error(f"unknown case id {args.target!r}")
    render(npz_path, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
