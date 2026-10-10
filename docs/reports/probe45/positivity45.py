"""Positivity of the coupled step under a capped pressure correction (prompt 45, ECR-002 step 6).

Usage:
    python positivity45.py MAX_PRESSURE_ITER [--outer N] [--ny NY] [--dx DX]

The first outer iterations of the 48-row plane Couette case of couette45.py
(standard model, a 120 m channel in 0.25 m cells), with the pressure
correction capped at MAX_PRESSURE_ITER conjugate-gradient iterations. Per
outer iteration it records the correction's iteration count and whether it
reached the cap, the worst absolute per-cell mass imbalance of the corrected
faces over the flux scale, and the least k and eps after the step, or the
PositivityError that stops the solve. The step's positivity argument (ADR-012
C, ADR-011 B's lemma) holds for faces that close every cell; this shows what
happens when a capped correction leaves them open. The solver is the
committed one; the probe wraps the corrector and the step to read them.
Records go to results/builder45/couette/positivity_<cap>.json.
"""

import argparse
import json
import sys

import couette45 as c45
import numpy as np

from src import solver_staggered
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import Mesh
from src.turbulence import PositivityError
from tests.couette_reference import case_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cap", type=int)
    parser.add_argument("--outer", type=int, default=12)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--dx", type=float, default=0.25)
    args = parser.parse_args()
    raw = case_raw("couette", "standard", args.ny, 120.0, args.dx, 0.25, 1e-6, args.cap)
    raw["solver"]["max_simple_iter"] = args.outer
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    solver = solver_staggered.StaggeredSolver(
        mesh, config, StaggeredBoundary(mesh, config)
    )
    rows: list[dict] = []
    corrector = solver._corrector
    real_correct = corrector.correct
    real_step = solver._turbulence_step

    def correct(prediction, p):  # type: ignore[no-untyped-def]
        out = real_correct(prediction, p)
        imbalance = corrector.mass_imbalance(out.u, out.v)
        rows.append(
            {
                "outer": len(rows),
                "cg_iterations": out.iterations,
                "reached_cap": bool(out.reached_cap),
                "worst_imbalance_over_flux": float(
                    np.abs(imbalance).max() / solver.flux_scale
                ),
            }
        )
        return out

    def step(state, u, v, iteration):  # type: ignore[no-untyped-def]
        new = real_step(state, u, v, iteration)
        rows[-1]["k_min"] = float(new.k[new.k > 0].min())
        rows[-1]["eps_min"] = float(new.eps[new.eps > 0].min())
        return new

    corrector.correct = correct
    solver._turbulence_step = step
    stopped = None
    try:
        solver.solve_steady()
    except PositivityError as err:
        stopped = str(err)
    record = {
        "cap": args.cap,
        "ny": args.ny,
        "dx": args.dx,
        "rows": rows,
        "stopped": stopped,
    }
    c45.OUT.mkdir(parents=True, exist_ok=True)
    (c45.OUT / f"positivity_{args.cap}.json").write_text(json.dumps(record, indent=1))
    for row in rows:
        print(row)
    print("stopped:", stopped)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
