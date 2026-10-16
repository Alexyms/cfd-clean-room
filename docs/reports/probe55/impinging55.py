"""Prompt 55 (ECR-002 step 7, VAL-020): the plane impinging jet runs.

Usage:
    python impinging55.py run GRID [--scheme S] [--variant V] [--cap N] [--log-every N]
                                   [--tag TAG] [--width-w W] [--preconditioner P]
    python impinging55.py time GRID          a 100-iteration timing probe, standard upwind

A run is the validation case as the suite loads it (validation.cases.
load_impinging_jet: configs/validation_impinging_jet.yaml regridded, y
clustered toward the plate and the nozzle's plane) through the committed
StaggeredSolver, by prompt 46's runner (docs/reports/probe46/coupled46.py,
run_room, imported) with prompt 52's four probe-side additions
(docs/reports/probe52/backstep52.py: the mean-pressure history, rule
version 5's update signature, the CG iteration counts, the whole domain
as the core) and the divergence guard at ten times V0. The record is
written as NAME.json and NAME.npz under results/builder55/, the mean
pressure history beside it as NAME_pmean.json. With --scheme absent the
turbulence section carries no `advection_scheme` key, so the committed
default (upwind) holds; --scheme umist states the key. --variant rng
states the variant in place of the file's standard. NAME is
VARIANT_GRID_SCHEME plus the tag.

--width-w W is item 2's check of the side distance: the domain is made W
nozzle widths wide at the same cell size, the nozzle and the sensor kept
at its centre and the outlets over the whole sides, so the run differs
from the case only in how far the sides are. The grid name then names
the cells across the nozzle (the case's 4 on the coarse grid).

One BLAS thread, as prompts 46 to 54: OPENBLAS_NUM_THREADS=1 before NumPy
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
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe52"))

import backstep52  # noqa: E402  (installs the probe-side subclasses on coupled46)
import coupled46  # noqa: E402

from src.config import SimConfig  # noqa: E402
from validation.cases import (  # noqa: E402
    IMPINGING_JET_INLET,
    case_path,
    load_impinging_jet,
)
from validation.impinging_jet import H_JET, V0, W_JET  # noqa: E402

OUT = ROOT / "results" / "builder55"
GRIDS = {"272x90": (272, 90), "408x135": (408, 135)}
coupled46.OUT = OUT
backstep52.OUT = OUT
coupled46.DIVERGED_SPEED = 10.0 * V0
TIME_CAP = 100


def case_raw(
    grid: str, variant: str, scheme: str, cap: int, width_w: float, preconditioner: str
) -> dict:
    """The case file regridded, y clustered as the loader clusters it."""
    nx, ny = GRIDS[grid]
    raw = yaml.safe_load(case_path("impinging_jet").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    loaded = load_impinging_jet("impinging_jet", grid=(nx, ny))
    raw["mesh"]["y"] = {"min_wall_spacing": loaded.stretch_y.min_spacing}
    if width_w:
        # The same cell size, the nozzle and the sensor recentred.
        dx = raw["domain"]["width"] / nx
        width = width_w * W_JET
        new_nx = round(width / dx)
        if abs(new_nx * dx - width) > 1e-9:
            raise SystemExit(f"{width_w} w is not a whole number of {dx} m cells")
        raw["domain"]["width"], raw["domain"]["nx"] = width, new_nx
        inlet = raw["boundaries"][IMPINGING_JET_INLET]
        inlet["x_start"], inlet["x_end"] = (
            0.5 * width - 0.5 * W_JET,
            0.5 * width + 0.5 * W_JET,
        )
        for sensor in raw["sensors"]:
            sensor["x"] = 0.5 * width
    raw["turbulence"]["variant"] = variant
    if scheme:
        raw["turbulence"]["advection_scheme"] = scheme
    if cap:
        raw["solver"]["max_simple_iter"] = cap
    if preconditioner:
        raw["solver"]["pressure_preconditioner"] = preconditioner
    return raw


def run(args: argparse.Namespace) -> None:
    """Item 3: one run through prompt 46's runner."""
    raw = case_raw(
        args.grid,
        args.variant,
        args.scheme,
        args.cap,
        args.width_w,
        args.preconditioner,
    )
    config = SimConfig.from_dict(raw)
    scheme = config.turbulence.advection_scheme
    name = f"{args.variant}_{args.grid}_{scheme}" + (f"_{args.tag}" if args.tag else "")
    inlet = raw["boundaries"][IMPINGING_JET_INLET]
    meta = {
        "case": "impinging_jet",
        "grid": args.grid,
        "variant": args.variant,
        "advection_scheme": scheme,
        "scheme_key_stated": bool(args.scheme),
        "w_jet": W_JET,
        "h_jet": H_JET,
        "v0": V0,
        "width": raw["domain"]["width"],
        "width_w": raw["domain"]["width"] / W_JET,
        "nx": raw["domain"]["nx"],
        "ny": raw["domain"]["ny"],
        "inlet_velocity": inlet["velocity"],
        "turbulence_intensity": inlet["turbulence_intensity"],
        "dissipation_length": inlet["dissipation_length"],
        "sweeps": raw["solver"]["momentum_sweeps"],
        "pressure_rtol": raw["solver"]["pressure_rtol"],
        "pressure_preconditioner": config.pressure_preconditioner,
        "alpha_velocity": raw["solver"]["alpha_velocity"],
        "alpha_turbulence": raw["turbulence"]["alpha_turbulence"],
        "stopping": raw["solver"]["stopping_rule"],
        "iteration_error_tol": raw["solver"]["iteration_error_tol"],
        "mass_imbalance_tol": raw["solver"]["mass_imbalance_tol"],
        "cap": raw["solver"]["max_simple_iter"],
        "turbulence_tol": raw["turbulence"]["tol"],
        "turbulence_max_iter": raw["turbulence"]["max_iter"],
        "wall_spacing": raw["mesh"]["y"]["min_wall_spacing"],
        "tag": args.tag,
        "locate": True,
    }
    rec = coupled46.run_room(name, raw, meta, log_every=args.log_every, locate=True)
    backstep52.finish(name, rec)


def timing(args: argparse.Namespace) -> None:
    """A short run whose seconds per outer iteration sizes the set."""
    args.variant = "standard"
    args.scheme = ""
    args.cap = TIME_CAP
    args.tag = "time"
    args.log_every = 10
    args.width_w = 0.0
    args.preconditioner = ""
    run(args)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    p.add_argument("grid", choices=sorted(GRIDS))
    p.add_argument("--scheme", choices=("upwind", "umist"), default="")
    p.add_argument("--variant", choices=("standard", "rng"), default="standard")
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--tag", default="")
    p.add_argument("--width-w", type=float, default=0.0)
    p.add_argument("--preconditioner", default="")
    p.set_defaults(func=run)
    p = sub.add_parser("time")
    p.add_argument("grid", choices=sorted(GRIDS))
    p.set_defaults(func=timing)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
