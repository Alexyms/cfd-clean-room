"""Prompt 49 (ECR-002 step 7, VAL-017): the Annex 20 room runs.

Usage:
    python annex49.py run GRID [--scheme S] [--variant V] [--cap N] [--log-every N]
    python annex49.py lines NAME       the four lines of a record, u / u0 at the stations

GRID is 216x72 or 375x125. The run is the validation case as the suite loads
it (validation.cases.load_annex20: configs/validation_annex20.yaml regridded,
the inlet velocity scaled so the covered faces carry u0 h) through the
committed StaggeredSolver, by prompt 46's runner (docs/reports/probe46/
coupled46.py, run_room, imported): the recording rule in place of the
solver's, the cell of the largest change of u, v and nu_t recorded every
outer iteration, the record written as NAME.json and NAME.npz under
results/builder49/. With --scheme absent the turbulence section carries no
`advection_scheme` key, so the committed default (upwind) holds; --scheme
umist states the key. --variant rng states the variant in place of the
file's standard. NAME is VARIANT_GRID_SCHEME, SCHEME the scheme the parsed
configuration holds.

The room has no equipment, so prompt 46's core (the cells above the
equipment tops) is replaced by every non-SOLID cell. After a run whose faces
were kept, the four lines are sampled by validation.annex20 at the
measurement stations and written as lines_NAME.json, the record the report's
comparison tables and the VAL-017 test read.

One BLAS thread, as prompts 46 to 48: OPENBLAS_NUM_THREADS=1 before NumPy
loads.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))

import coupled46  # noqa: E402

from src.config import SimConfig  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from validation import annex20  # noqa: E402
from validation.cases import case_path, load_annex20  # noqa: E402

OUT = ROOT / "results" / "builder49"
GRIDS = {"216x72": (216, 72), "375x125": (375, 125)}
coupled46.OUT = OUT


def whole_room(mesh: Mesh, _raw: dict) -> np.ndarray:
    """Every non-SOLID cell: the room has no equipment to stand above."""
    return mesh.cell_type != SOLID


coupled46.core_mask = whole_room


def annex_raw(grid: str, variant: str, scheme: str, cap: int) -> dict:
    """The case file regridded, the inlet scaled as the loader scales it."""
    nx, ny = GRIDS[grid]
    raw = yaml.safe_load(case_path("annex20").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    loaded = load_annex20("annex20", grid=(nx, ny))
    raw["boundaries"]["inlet"]["velocity"] = loaded.boundaries["inlet"].velocity
    raw["turbulence"]["variant"] = variant
    if scheme:
        raw["turbulence"]["advection_scheme"] = scheme
    if cap:
        raw["solver"]["max_simple_iter"] = cap
    return raw


def lines_of(name: str) -> dict:
    """The four lines of a record's final faces, written beside it."""
    rec = json.loads((OUT / f"{name}.json").read_text())
    arrays = np.load(OUT / f"{name}.npz")
    if "u_faces" not in arrays:
        raise SystemExit(f"{name}: no faces kept (stop {rec['stop']})")
    config = SimConfig.from_dict(rec["raw"])
    mesh = Mesh(config)
    sampled = annex20.sample_lines(mesh, arrays["u_faces"])
    out = {
        "name": name,
        "commit": rec["commit"],
        "stop": rec["stop"],
        "outer": rec["outer"],
        "u0": annex20.U0,
        "lines": {
            key: {
                "coordinate": line.stations,
                "u_over_u0": values.tolist(),
                "against_data": annex20.line_errors(values, line.measured).__dict__,
                "against_published": annex20.line_errors(
                    values, line.published
                ).__dict__,
            }
            for (key, line), values in zip(
                annex20.LINES.items(), sampled.values(), strict=True
            )
        },
    }
    (OUT / f"lines_{name}.json").write_text(json.dumps(out, indent=1))
    return out


def run(args: argparse.Namespace) -> None:
    """One row through prompt 46's runner."""
    raw = annex_raw(args.grid, args.variant, args.scheme, args.cap)
    config = SimConfig.from_dict(raw)
    scheme = config.turbulence.advection_scheme
    name = f"{args.variant}_{args.grid}_{scheme}"
    inlet = raw["boundaries"]["inlet"]
    meta = {
        "case": "annex20",
        "grid": args.grid,
        "variant": args.variant,
        "advection_scheme": scheme,
        "scheme_key_stated": bool(args.scheme),
        "u0": annex20.U0,
        "slot_height": inlet["y_end"] - inlet["y_start"],
        "inlet_velocity": inlet["velocity"],
        "sweeps": raw["solver"]["momentum_sweeps"],
        "pressure_rtol": raw["solver"]["pressure_rtol"],
        "alpha_velocity": raw["solver"]["alpha_velocity"],
        "alpha_turbulence": raw["turbulence"]["alpha_turbulence"],
        "turbulence_intensity": inlet["turbulence_intensity"],
        "dissipation_length": inlet["dissipation_length"],
        "stopping": raw["solver"]["stopping_rule"],
        "iteration_error_tol": raw["solver"]["iteration_error_tol"],
        "mass_imbalance_tol": raw["solver"]["mass_imbalance_tol"],
        "cap": raw["solver"]["max_simple_iter"],
        "locate": True,
    }
    rec = coupled46.run_room(name, raw, meta, log_every=args.log_every, locate=True)
    if rec["faces_kept"]:
        lines = lines_of(name)
        for key, line in lines["lines"].items():
            print(
                f"{name} {key}: against data max {line['against_data']['max_abs']:.4f} "
                f"rms {line['against_data']['rms']:.4f}",
                flush=True,
            )


def lines(args: argparse.Namespace) -> None:
    """Sample a finished record's lines again."""
    out = lines_of(args.name)
    print(json.dumps({k: v["against_data"] for k, v in out["lines"].items()}, indent=1))


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
    p.set_defaults(func=run)
    p = sub.add_parser("lines")
    p.add_argument("name")
    p.set_defaults(func=lines)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
