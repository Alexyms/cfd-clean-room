"""Measurement 1 of prompt 56 (ECR-002 step 8): the three layouts on the exact grids.

Usage:
    python coupled56.py run LAYOUT GRID [--cap N] [--tag T] [--log-every N] [--concurrent N]
    python coupled56.py time GRID [--concurrent N]    20 outer iterations of L0

LAYOUT is L0, L1 or L2, the committed configuration files LAYOUT_FILES
names (the report's section 2.3); GRID is 160x60 or 320x120. The run is
prompt 47's (docs/reports/probe47/coupled47.py, its `control` arm, nothing
substituted): the committed StaggeredSolver on the layout file regridded,
from rest, with the cell of the largest change of u, v and nu_t recorded
every outer iteration. The configuration mapping is the file's with
domain.nx and domain.ny set and, on a grid other than the file's,
solver.mass_imbalance_tol recomputed by ADR-011 G's formula at that grid's
cell volume; on the file's own grid the file's value is kept and checked
against the formula, so the L0 320x120 record runs the committed file
exactly as the harness row val018_320x120 does. Records go to
results/builder56/ as NAME.json and NAME.npz, NAME = LAYOUT_GRID[_TAG],
each carrying the number of processes it shared the machine with.

One BLAS thread, as prompts 46 to 48: OPENBLAS_NUM_THREADS=1 before NumPy
loads.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe47"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe43"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe41"))

import coupled46  # noqa: E402
import coupled47  # noqa: E402

from src.stopping import ErrorEstimateRule  # noqa: E402
from src.turbulence import KEpsilonModel  # noqa: E402

OUT = ROOT / "results" / "builder56"
coupled46.OUT = OUT
coupled47.OUT = OUT
# Prompt 51 renamed the model's per-step count from last_sweeps (Jacobi
# sweeps) to last_iterations (CG iterations). Prompt 47's runner reads the
# old name every outer iteration, so this process gives the model the old
# name as a read-only alias of the new one; the record's k_sweeps and
# eps_sweeps columns are CG iteration counts.
KEpsilonModel.last_sweeps = property(lambda self: self.last_iterations)  # type: ignore[attr-defined]


class RecordingRule(coupled46.RecordingRule):
    """Prompt 46's recording rule under rule version 5 (prompt 50).

    The committed rule's ``update`` takes the eddy-viscosity scale of the
    iterate just computed (``nu_scale``) beside the viscosity step; prompt
    46's subclass predates it. This one keeps the same readings (every
    iteration's imbalance, the velocity and viscosity steps) and hands both
    keywords to the committed rule, whose decision is unchanged. Prompt
    47's runner records ``rule._nu_scale`` as the record's ``nu_scale``;
    under version 5 the scale moves every iteration, so the property gives
    the last one (the full history is ``viscosity_scale_history``).
    """

    def update(
        self,
        step: float,
        imbalance: object,
        viscosity_step: float | None = None,
        nu_scale: float | None = None,
    ) -> bool:
        found = imbalance()  # type: ignore[operator]
        self.readings.append(found)
        self.steps.append(float(step))
        self.viscosity_steps.append(
            float("nan") if viscosity_step is None else float(viscosity_step)
        )
        return ErrorEstimateRule.update(
            self, step, lambda: found, viscosity_step=viscosity_step, nu_scale=nu_scale
        )

    @property
    def _nu_scale(self) -> float | None:
        history = self.viscosity_scale_history
        return float(history[-1]) if history else None


coupled47.RecordingRule = RecordingRule  # type: ignore[attr-defined]

LAYOUT_FILES = {
    "L0": "clean_room_default.yaml",
    "L1": "clean_room_l1_return3_closed.yaml",
    "L2": "clean_room_l2_left_supply.yaml",
}
GRIDS = {"160x60": (160, 60), "320x120": (320, 120)}
TIME_CAP = 20
# ADR-011 G: mass_imbalance_tol = IMBALANCE_FACTOR rho V_min / t_end.
IMBALANCE_FACTOR = 1e-4
FORMULA_TOL = 1e-12


def imbalance_tol(raw: dict) -> float:
    """ADR-011 G's bound at the mapping's own grid."""
    domain, block = raw["domain"], raw["solver"]
    v_min = (domain["width"] / domain["nx"]) * (domain["height"] / domain["ny"])
    return IMBALANCE_FACTOR * raw["fluid"]["density"] * v_min / block["t_end"]


def layout_raw(layout: str, grid: str, cap: int = 0) -> dict:
    """The layout file's mapping on GRID, nothing else changed (the report's section 2.2)."""
    path = ROOT / "configs" / LAYOUT_FILES[layout]
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    nx, ny = GRIDS[grid]
    file_grid = (raw["domain"]["nx"], raw["domain"]["ny"])
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    block = raw["solver"]
    formula = imbalance_tol(raw)
    if (nx, ny) == file_grid:
        stated = block["mass_imbalance_tol"]
        if abs(stated - formula) > FORMULA_TOL * formula:
            raise ValueError(
                f"{path.name} states mass_imbalance_tol {stated:g}; ADR-011 G's formula "
                f"gives {formula:g} on its own grid"
            )
    else:
        block["mass_imbalance_tol"] = formula
    if cap:
        block["max_simple_iter"] = cap
    return raw


def meta_for(layout: str, grid: str, raw: dict, concurrent: int) -> dict:
    """What the record says about its configuration beside the mapping itself."""
    block, turb = raw["solver"], raw["turbulence"]
    supply = raw["boundaries"]["hepa_supply"]
    return {
        "case": "product",
        "layout": layout,
        "layout_file": LAYOUT_FILES[layout],
        "grid": grid,
        "variant": turb["variant"],
        "arm": "control",
        "advection_scheme": turb["advection_scheme"],
        "sweeps": block["momentum_sweeps"],
        "pressure_rtol": block["pressure_rtol"],
        "pressure_preconditioner": block["pressure_preconditioner"],
        "alpha_velocity": block["alpha_velocity"],
        "alpha_turbulence": turb["alpha_turbulence"],
        "turbulence_cfl": turb["cfl_number"],
        "turbulence_intensity": supply["turbulence_intensity"],
        "dissipation_length": supply["dissipation_length"],
        "stopping": block["stopping_rule"],
        "iteration_error_tol": block["iteration_error_tol"],
        "mass_imbalance_tol": block["mass_imbalance_tol"],
        "max_simple_iter": block["max_simple_iter"],
        "concurrent_processes": concurrent,
        "locate": True,
    }


def run(args: argparse.Namespace) -> None:
    """One row of the set."""
    raw = layout_raw(args.layout, args.grid, args.cap)
    name = f"{args.layout}_{args.grid}" + (f"_{args.tag}" if args.tag else "")
    meta = meta_for(args.layout, args.grid, raw, args.concurrent)
    coupled47.run_room(name, raw, meta, "control", log_every=args.log_every)


def timing(args: argparse.Namespace) -> None:
    """A short run of L0 whose seconds per outer iteration sizes the set."""
    args.layout = "L0"
    args.cap = TIME_CAP
    args.tag = "time"
    args.log_every = 5
    run(args)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    p.add_argument("layout", choices=sorted(LAYOUT_FILES))
    p.add_argument("grid", choices=sorted(GRIDS))
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--concurrent", type=int, default=1)
    p.set_defaults(func=run)
    p = sub.add_parser("time")
    p.add_argument("grid", choices=sorted(GRIDS))
    p.add_argument("--concurrent", type=int, default=1)
    p.set_defaults(func=timing)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
