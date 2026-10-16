"""Prompt 59 probe C: the observed order of the limited scheme under refinement, for VAL-008.

ADR-014 section E. A Gaussian pulse is advected by a uniform face field over
a fixed travel on a sequence of grids, and the relative L2 error against the
exactly translated pulse (validation.metrics.relative_l2, VAL-004's metric)
gives the observed order between successive grids, log2 of the error ratio.
Three cases, each for the UMIST-limited QUICK scheme and for upwind:

    aligned   a channel 2.0 by 0.4 m, the flow along x at the supply speed,
              sigma 0.1 m, travel 1.0 m; grids of 0.04, 0.02, 0.01 and 0.005 m
    front, front_wide
              the aligned channel with a smooth monotone front in place of
              the pulse, 0.5 (1 + tanh((x - x0) / w)) with w = 0.1 m (0.2 m for
              front_wide, the same front twice as resolved on each grid, started
              at x 0.9 m and moved 0.6 m so that its upstream tail at the inlet is
              below 1e-4 of its height, since the inflow value is zero), so the
              limiter has no extremum to clip and the face value is QUICK's
              wherever the data are smooth (ADR-011 B's "second order on
              smooth monotone data")
    oblique   VAL-004 row 1's geometry (2.0 by 1.2 m, 30 degrees), sigma 0.1 m,
              travel 1.0 m; grids 50x30, 100x60, 200x120
    stretched the oblique case on a mesh clustered toward the top and bottom
              walls (y stretch_ratio 1.05), the gap issue 53 D6 names

and two time-step rules: the Courant number fixed at 0.1 (VAL-004's), and
the Courant number halved with the spacing (0.1 on the coarsest grid), so
the forward Euler error falls as dx^2 and the spatial order shows. Output:
results/builder59/order59.json and a markdown table on stdout.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.config import UMIST, UPWIND  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.solver_transport import TransportSolver  # noqa: E402
from validation.metrics import peak_retention, relative_l2  # noqa: E402
from validation.transport_cases import (  # noqa: E402
    SUPPLY_SPEED,
    FixedConditions,
    ScalarPhysics,
    gaussian_cell_averages,
    transport_config,
    uniform_face_field,
    zero_conditions,
)

OUT = ROOT / "results" / "builder59"
SIGMA = 0.1
TRAVEL = 1.0
COURANT = 0.1
FRONT_WIDTH = 0.1
CASES = {
    "aligned": {
        "width": 2.0,
        "height": 0.4,
        "angle": 0.0,
        "start": (0.45, 0.2),
        "grids": [0.04, 0.02, 0.01, 0.005],
        "mesh": None,
    },
    "front": {
        "width": 2.0,
        "height": 0.4,
        "angle": 0.0,
        "start": (0.45, 0.2),
        "grids": [0.04, 0.02, 0.01, 0.005],
        "mesh": None,
        "front": FRONT_WIDTH,
    },
    "front_wide": {
        "width": 2.0,
        "height": 0.4,
        "angle": 0.0,
        "start": (0.9, 0.2),
        "travel": 0.6,
        "grids": [0.04, 0.02, 0.01, 0.005],
        "mesh": None,
        "front": 2.0 * FRONT_WIDTH,
    },
    "oblique": {
        "width": 2.0,
        "height": 1.2,
        "angle": 30.0,
        "start": (0.4, 0.25),
        "grids": [0.04, 0.02, 0.01],
        "mesh": None,
    },
    "stretched": {
        "width": 2.0,
        "height": 1.2,
        "angle": 30.0,
        "start": (0.4, 0.25),
        "grids": [0.04, 0.02, 0.01],
        "mesh": {"x": {"stretch_ratio": 1.0}, "y": {"stretch_ratio": 1.05}},
    },
}


def front_cell_averages(
    x: np.ndarray, y: np.ndarray, x0: float, width: float
) -> np.ndarray:
    """Exact cell averages of 0.5 (1 + tanh((x - x0) / width)), uniform in y.

    The antiderivative of the profile is 0.5 (x + width log cosh((x - x0) / width)),
    evaluated with log cosh written stably for large arguments.
    """

    def primitive(s: np.ndarray) -> np.ndarray:
        z = (s - x0) / width
        log_cosh = np.abs(z) + np.log1p(np.exp(-2.0 * np.abs(z))) - np.log(2.0)
        return 0.5 * (s + width * log_cosh)

    column = (primitive(x[1:]) - primitive(x[:-1])) / np.diff(x)
    return np.ascontiguousarray(
        np.broadcast_to(column[None, :], (y.size - 1, x.size - 1))
    )


def run_one(case: dict, dx: float, scheme: str, courant: float) -> dict:
    """One advection of the pulse on the grid of DX at the Courant number given."""
    nx, ny = round(case["width"] / dx), round(case["height"] / dx)
    config = transport_config(
        case["width"],
        case["height"],
        nx,
        ny,
        cfl_number=courant,
        scheme=scheme,
        mesh=case["mesh"],
    )
    mesh = Mesh(config)
    angle = math.radians(case["angle"])
    u, v = SUPPLY_SPEED * math.cos(angle), SUPPLY_SPEED * math.sin(angle)
    t_end = case.get("travel", TRAVEL) / SUPPLY_SPEED
    start = case["start"]
    end = (start[0] + u * t_end, start[1] + v * t_end)
    faces = uniform_face_field(mesh, u, v)
    solver = TransportSolver(
        mesh,
        config,
        ScalarPhysics(settling=0.0, diffusion=0.0),
        FixedConditions(zero_conditions(mesh)),
    )
    if case.get("front"):
        initial = front_cell_averages(mesh.x, mesh.y, start[0], case["front"])
        exact = front_cell_averages(mesh.x, mesh.y, end[0], case["front"])
    else:
        initial = gaussian_cell_averages(mesh.x, mesh.y, start, SIGMA)
        exact = gaussian_cell_averages(mesh.x, mesh.y, end, SIGMA)
    n_steps = math.ceil(t_end / solver.stable_dt(faces, 0))
    dt = t_end / n_steps
    c = initial.copy()
    t0 = perf_counter()
    for _ in range(n_steps):
        c = solver.solve_timestep(c, faces, 0, dt)
    return {
        "dx": dx,
        "grid": [nx, ny],
        "courant": courant,
        "steps": n_steps,
        "dt": dt,
        "l2": float(relative_l2(c, exact, mesh)),
        "peak": float(peak_retention(c, exact, mesh)),
        "minimum": float(c[mesh.cell_type != 2].min()),
        "seconds": perf_counter() - t0,
    }


def main() -> None:
    """Run every case, or those named on the command line, merging into the record."""
    wanted = sys.argv[1:] or list(CASES)
    path = OUT / "order59.json"
    results = []
    if path.exists():
        results = [
            r
            for r in json.loads(path.read_text(encoding="utf-8"))
            if r["case"] not in wanted
        ]
    for name, case in CASES.items():
        if name not in wanted:
            continue
        for scheme in (UMIST, UPWIND):
            for rule in ("fixed", "halving"):
                rows = []
                for k, dx in enumerate(case["grids"]):
                    courant = COURANT if rule == "fixed" else COURANT / 2**k
                    row = run_one(case, dx, scheme, courant)
                    rows.append(row)
                    print(
                        f"{name} {scheme} {rule} dx {dx}: L2 {row['l2']:.4e} peak {row['peak']:.4f} steps {row['steps']} ({row['seconds']:.0f} s)",
                        flush=True,
                    )
                orders = [
                    math.log2(rows[i - 1]["l2"] / rows[i]["l2"])
                    for i in range(1, len(rows))
                ]
                results.append(
                    {
                        "case": name,
                        "scheme": scheme,
                        "rule": rule,
                        "rows": rows,
                        "orders": orders,
                    }
                )
                print(f"  orders: {', '.join(f'{o:.2f}' for o in orders)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda r: list(CASES).index(r["case"]))
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print()
    print(
        "| Case | Scheme | Courant rule | L2 error per grid (coarse to fine) | Observed order between grids |"
    )
    print("|---|---|---|---|---|")
    for r in results:
        errors = ", ".join(f"{row['l2']:.3e}" for row in r["rows"])
        orders = ", ".join(f"{o:.2f}" for o in r["orders"])
        print(f"| {r['case']} | {r['scheme']} | {r['rule']} | {errors} | {orders} |")


if __name__ == "__main__":
    main()
