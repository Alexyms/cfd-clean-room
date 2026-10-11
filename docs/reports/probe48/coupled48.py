"""Items 4 and 5 of prompt 48: the product room through the turbulence key.

Usage:
    python coupled48.py run GRID VARIANT [--scheme S] [--cap N] [--log-every N]

GRID is 80x30, 160x60 or 320x120; VARIANT is standard or rng. The run is
prompt 47's (docs/reports/probe47/coupled47.py, its `control` arm: the
committed StaggeredSolver on the product configuration regridded, the
cell of the largest change recorded every outer iteration) with nothing
substituted: k and eps are advected by whatever the configuration's
turbulence section says. With --scheme absent the section carries no
`advection_scheme` key, so the committed default holds; --scheme upwind or
umist states the key. The record is VARIANT_GRID_SCHEME under
results/builder48/, SCHEME the scheme the parsed configuration holds, so
a default row and a stated row of the same scheme share a name and the
default row is the one this build runs.

One BLAS thread, as prompts 46 and 47: OPENBLAS_NUM_THREADS=1 before NumPy
loads.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe47"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))

import coupled46  # noqa: E402
import coupled47  # noqa: E402

from src.config import SimConfig  # noqa: E402

OUT = ROOT / "results" / "builder48"
coupled46.OUT = OUT
coupled47.OUT = OUT


def run(args: argparse.Namespace) -> None:
    """One row: prompt 47's control arm with the key as given."""
    cap = args.cap or coupled46.CAP
    raw = coupled47.product_raw(args.grid, args.variant, "control", cap)
    if args.scheme:
        raw["turbulence"]["advection_scheme"] = args.scheme
    scheme = SimConfig.from_dict(raw).turbulence.advection_scheme
    name = f"{args.variant}_{args.grid}_{scheme}"
    meta = {
        "case": "product",
        "grid": args.grid,
        "variant": args.variant,
        "arm": "control",
        "advection_scheme": scheme,
        "scheme_key_stated": bool(args.scheme),
        "sweeps": coupled46.SWEEPS,
        "pressure_rtol": coupled46.PRESSURE_RTOL,
        "alpha_velocity": coupled46.ALPHA_VELOCITY,
        "alpha_turbulence": raw["turbulence"]["alpha_turbulence"],
        "turbulence_intensity": coupled46.INTENSITY,
        "dissipation_length": coupled46.DISSIPATION_LENGTH,
        "stopping": raw["solver"]["stopping_rule"],
        "iteration_error_tol": raw["solver"]["iteration_error_tol"],
        "mass_imbalance_tol": raw["solver"]["mass_imbalance_tol"],
        "cap": cap,
        "locate": True,
    }
    coupled47.run_room(name, raw, meta, "control", log_every=args.log_every)


def main() -> None:
    """Dispatch the one mode the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    p.add_argument("grid", choices=sorted(coupled47.GRIDS))
    p.add_argument("variant", choices=("standard", "rng"))
    p.add_argument("--scheme", choices=("upwind", "umist"), default="")
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
