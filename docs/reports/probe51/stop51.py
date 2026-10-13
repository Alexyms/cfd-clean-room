"""The stop of prompt 51's first pass: a truncated CG iterate of k goes negative.

Usage:
    python stop51.py SCHEME            SCHEME upwind or umist

Replays tests/test_turbulence.py's long standard run (the random field, the
patchy start, 300 pseudo-time steps) with the model's implicit solve as the
first pass built it: CG with no fallback, the CG iterate returned whatever
it is (``scalar_scheme._conjugate_gradient`` called in place of
``implicit_step``'s CG branch). At steps 4, 17, 60 and 120, and at the step
whose k is not positive, the k system is assembled densely and recorded: the
largest conductance over the smallest V / dt, the extreme eigenvalues and the
condition number of the matrix over the unknowns, the dense solution's least
value, and the least value of each method on the same system: Jacobi, CG
with no fallback, and the committed ``implicit_step`` under CG with its
fallback. The record is results/builder51/stop_SCHEME.json of the main
checkout. Seconds per scheme.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
MAIN = Path(
    subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
).parent
OUT = MAIN / "results" / "builder51"
STEPS = (4, 17, 60, 120)

logging.disable(logging.WARNING)

from src import scalar_scheme, turbulence  # noqa: E402
from src.mesh import SOLID  # noqa: E402
from src.turbulence import PositivityError  # noqa: E402
from tests import test_turbulence as tt  # noqa: E402

COMMITTED = scalar_scheme.implicit_step


def first_pass(*args: object, **kwargs: object) -> scalar_scheme.ImplicitResult:
    """implicit_step's CG branch with no fallback: the CG iterate as it ends."""
    c_star, volume, dt, g_u, g_v, diagonal, solid, tol, max_iter = args
    held = kwargs.get("held")
    diag_extra = g_u[:, :-1] + g_u[:, 1:] + g_v[:-1, :] + g_v[1:, :] + diagonal
    over_dt = volume / dt
    a_p = over_dt + diag_extra
    b = over_dt * c_star
    stop = tol * float(np.abs(b).max())
    fixed = solid if held is None else solid | held
    c = c_star.copy()
    c[solid] = 0.0
    c, iterations, converged = scalar_scheme._conjugate_gradient(
        c, a_p, b, g_u, g_v, fixed, stop, max_iter
    )
    return scalar_scheme.ImplicitResult(
        field=c,
        method="cg",
        iterations=iterations,
        converged=converged,
        fallbacks=0,
        fallback_sweeps=0,
    )


def describe(args: tuple, kwargs: dict) -> dict:
    """The k system densely, and each method's least value on it."""
    c_star, volume, dt, g_u, g_v, diagonal, solid = args[:7]
    ny, nx = solid.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    over_dt = volume / dt
    a = np.zeros((ny * nx, ny * nx))
    a[idx, idx] = over_dt + diagonal
    for j in range(ny):
        for i in range(1, nx):
            p, q = idx[j, i - 1], idx[j, i]
            a[[p, q], [p, q]] += g_u[j, i]
            a[p, q] -= g_u[j, i]
            a[q, p] -= g_u[j, i]
    for j in range(1, ny):
        for i in range(nx):
            p, q = idx[j - 1, i], idx[j, i]
            a[[p, q], [p, q]] += g_v[j, i]
            a[p, q] -= g_v[j, i]
            a[q, p] -= g_v[j, i]
    unknown = ~solid.ravel()
    au = a[np.ix_(unknown, unknown)]
    exact = np.linalg.solve(au, (over_dt * c_star).ravel()[unknown])
    eig = np.linalg.eigvalsh(au)
    live = ~solid
    jacobi = COMMITTED(*args, method="jacobi")
    bare = first_pass(*args)
    committed = COMMITTED(*args, method="cg")
    return {
        "conductance_over_v_dt": float(max(g_u.max(), g_v.max()) / over_dt[live].min()),
        "eig_min": float(eig.min()),
        "eig_max": float(eig.max()),
        "condition": float(eig.max() / eig.min()),
        "c_star_min": float(c_star[live].min()),
        "dense_min": float(exact.min()),
        "jacobi": [
            jacobi.iterations,
            jacobi.converged,
            float(jacobi.field[live].min()),
        ],
        "cg_no_fallback": [
            bare.iterations,
            bare.converged,
            float(bare.field[live].min()),
        ],
        "cg_committed": [
            committed.iterations,
            committed.fallbacks,
            float(committed.field[live].min()),
        ],
    }


def main() -> None:
    """Replay the first pass on the long standard run and record the systems."""
    scheme = sys.argv[1]
    calls: list[tuple] = []

    def recording(*args: object, **kwargs: object) -> scalar_scheme.ImplicitResult:
        calls.append((args, kwargs))
        return first_pass(*args, **kwargs)

    turbulence.implicit_step = recording
    mesh, model, faces, conditions_for = tt.CASES["random"]("standard", scheme)
    state = model.state(*tt._patchy_start(model, mesh.cell_type.shape, seed=3))
    record: dict = {"scheme": scheme, "steps": {}, "failure": None}
    live = mesh.cell_type != SOLID
    for n in range(tt.POSITIVITY_STEPS):
        calls.clear()
        try:
            state = model.step(state, faces, conditions_for(state))
        except PositivityError as err:
            record["failure"] = {"step": n, "message": str(err)}
            record["steps"][str(n)] = describe(*calls[0])
            break
        if n in STEPS:
            record["steps"][str(n)] = describe(*calls[0])
            record["steps"][str(n)]["k_range"] = [
                float(state.k[live].min()),
                float(state.k[live].max()),
            ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stop_{scheme}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record["failure"]))


if __name__ == "__main__":
    main()
