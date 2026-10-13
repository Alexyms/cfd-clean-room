"""Prompt 52 (ECR-002 step 7, VAL-019): the backward-facing step runs.

Usage:
    python backstep52.py precursor [--length-h L] [--dx-h D] [--ny N] [--u-in U]
                                   [--cap N] [--tag TAG]
    python backstep52.py run GRID [--scheme S] [--variant V] [--cap N] [--log-every N]
                                  [--tag TAG]
    python backstep52.py time GRID          a 100-iteration timing probe, standard upwind

The precursor is item 2: the upstream channel alone, the step's SOLID block
spanning the whole length so the channel runs on the case's own y grid (the
loader's clustering, validation.cases.wall_spacing_for_face), coarse in x,
a uniform inlet above the block and a pressure outlet. Its record is read by
precursor52.py for the boundary layer's growth along x.

A run is the validation case as the suite loads it (validation.cases.
load_backstep: configs/validation_backstep.yaml regridded, y clustered so a
face falls on the step height) through the committed StaggeredSolver, by
prompt 46's runner (docs/reports/probe46/coupled46.py, run_room, imported):
the recording rule in place of the solver's, the cell of the largest change
of u, v and nu_t recorded every outer iteration, the record written as
NAME.json and NAME.npz under results/builder52/. With --scheme absent the
turbulence section carries no `advection_scheme` key, so the committed
default (upwind) holds; --scheme umist states the key. --variant rng states
the variant in place of the file's standard. NAME is VARIANT_GRID_SCHEME.

Four probe-side additions to prompt 46's runner, which predates rule version
5 and the CG k and eps solve. The solver run_room builds is a subclass that
records the mean pressure over the non-SOLID cells every outer iteration,
the reading the outlet probe (prompt 41) used for the copy rule's drift; it
is written beside the record as NAME_pmean.json. The recording rule is a
subclass of prompt 46's taking rule version 5's per-step ``nu_scale`` (the
decision is the committed rule's, on the same readings) and answering the
runner's ``_nu_scale`` with the scale of the last step. The model the solver
builds is a subclass that answers prompt 46's ``last_sweeps`` with the CG
solve's ``last_iterations`` (prompt 51 renamed the count), so the record's
k_sweeps and eps_sweeps columns hold CG iterations. And the core of prompt
46 (the cells above the equipment tops) is every non-SOLID cell, as prompt
49's runner had it. Its divergence guard, 100 m/s for a room, becomes ten
times U_ref: the coarse grid's first iteration passes 100 m/s at the inlet
corner and settles.

One BLAS thread, as prompts 46 to 51: OPENBLAS_NUM_THREADS=1 before NumPy
loads.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
from collections.abc import Callable
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
from src.solver_staggered import StaggeredSolver  # noqa: E402
from src.stopping import (  # noqa: E402
    ErrorEstimateRule,
    ImbalanceSummary,
    IterationState,
)
from src.turbulence import KEpsilonModel  # noqa: E402
from validation.cases import (  # noqa: E402
    case_path,
    load_backstep,
    step_face_index,
    wall_spacing_for_face,
)

OUT = ROOT / "results" / "builder52"
GRIDS = {"410x40": (410, 40), "656x60": (656, 60), "984x90": (984, 90)}
coupled46.OUT = OUT

# The case's fixed quantities (ADR-012 G (ii); the data's header).
STEP_HEIGHT = 0.0127
U_REF = 44.2
RE_H = 36000.0
RHO = 1.2
NU = U_REF * STEP_HEIGHT / RE_H
CHANNEL_HEIGHTS = 8.0  # the upstream channel, in step heights
DOMAIN_HEIGHTS = 9.0  # the downstream channel, in step heights
INLET_INTENSITY = 0.01
INLET_LENGTH = 0.1 * STEP_HEIGHT
TIME_CAP = 100

# The precursor's defaults: 160 H long in half-height cells on the medium
# y grid, the inlet at a first guess of the speed that makes the centreline
# U_ref at x/H = -4 (the channel accelerates as its layers grow).
PRECURSOR_LENGTH_H = 160.0
PRECURSOR_DX_H = 0.5
PRECURSOR_NY = 60
PRECURSOR_U_IN = 42.0
PRECURSOR_CAP = 20000

P_MEAN: list[float] = []


class PressureMeanSolver(StaggeredSolver):
    """The committed solver, keeping the mean pressure of every outer iteration.

    The reading prompt 41 used for the pressure outlet's drift: a copy rule
    that never corrects its copy for continuity walks the pressure datum
    at a steady rate per outer iteration. The decision and every field are
    the parent's; only the list P_MEAN is added to.
    """

    def solve_steady(
        self,
        on_iteration: Callable[[IterationState], None] | None = None,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        live = self._live
        P_MEAN.clear()

        def record(state: IterationState) -> None:
            P_MEAN.append(float(np.mean(state.p[live])))
            if on_iteration is not None:
                on_iteration(state)

        return super().solve_steady(on_iteration=record, **kwargs)


class RecordingRule52(coupled46.RecordingRule):
    """Prompt 46's recording rule under rule version 5's update signature.

    Every outer iteration's imbalance readings, velocity step and viscosity
    step are kept, as prompt 46 kept them; the decision is the committed
    rule's on the same readings, with the per-step scale passed through.
    """

    def update(
        self,
        step: float,
        imbalance: Callable[[], ImbalanceSummary],
        viscosity_step: float | None = None,
        nu_scale: float | None = None,
    ) -> bool:
        found = imbalance()
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
        """The scale of the last step, where prompt 46's runner reads a fixed one."""
        history = self.viscosity_scale_history
        return float(history[-1]) if history else None


class SweepsAliasModel(KEpsilonModel):
    """The committed model, naming its CG iteration counts as prompt 46's runner reads them."""

    @property
    def last_sweeps(self) -> tuple[int, int]:
        return self.last_iterations


def whole_room(mesh: Mesh, _raw: dict) -> np.ndarray:
    """Every non-SOLID cell: the step has no equipment to stand above."""
    return mesh.cell_type != SOLID


coupled46.core_mask = whole_room
# Prompt 46's divergence guard is 100 m/s, a room's bound; the inlet here is
# 42 m/s and the first iteration's transient at the inlet corner passes 100
# on the coarse grid before settling. Ten times U_ref is the bound here.
coupled46.DIVERGED_SPEED = 10.0 * U_REF
coupled46.solver_staggered.StaggeredSolver = PressureMeanSolver  # type: ignore[misc]
coupled46.solver_staggered.KEpsilonModel = SweepsAliasModel  # type: ignore[misc]
coupled46.RecordingRule = RecordingRule52  # type: ignore[misc]


def solver_block(cap: int, v_min: float, t_end: float = 60.0) -> dict:
    """The case file's solver section as the precursor needs it, before the file exists.

    The keys are VAL-017's (configs/validation_annex20.yaml) with the cap the
    prompt names, and the mass-imbalance tolerance ADR-011 G's bound on the
    smallest cell.
    """
    return {
        "dt": 0.01,
        "t_end": t_end,
        "output_interval": 10,
        "convergence_tol": 1.0e-6,
        "max_simple_iter": cap,
        "alpha_velocity": 0.5,
        "alpha_pressure": 0.3,
        "max_pressure_iter": 5000,
        "pressure_rtol": 1.0e-4,
        "momentum_sweeps": 10,
        "stopping_rule": "error_estimate",
        "iteration_error_tol": 1.0e-6,
        "mass_imbalance_tol": 1e-4 * RHO * v_min / t_end,
    }


TURBULENCE = {
    "model": "k_epsilon",
    "variant": "standard",
    "wall_treatment": "scalable_wall_functions",
    "cfl_number": 0.25,
    "alpha_turbulence": 0.7,
    "max_iter": 500,
    "tol": 1.0e-10,
}


def precursor_raw(length_h: float, dx_h: float, ny: int, u_in: float, cap: int) -> dict:
    """The upstream channel alone: the block over the whole length, on the case's y grid."""
    h = STEP_HEIGHT
    length = length_h * h
    nx = round(length_h / dx_h)
    if abs(nx * dx_h - length_h) > 1e-9:
        raise SystemExit(f"{length_h} H is not a whole number of {dx_h} H cells")
    height = DOMAIN_HEIGHTS * h
    spacing = wall_spacing_for_face(height, ny, step_face_index(ny), h)
    v_min = (length / nx) * spacing
    return {
        "domain": {"width": length, "height": height, "nx": nx, "ny": ny},
        "mesh": {"x": {"stretch_ratio": 1.0}, "y": {"min_wall_spacing": spacing}},
        "fluid": {"density": RHO, "viscosity": RHO * NU, "temperature": 293.0},
        "particles": {
            "density": 1000.0,
            "sizes": [0.1e-6],
            "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {"diameters": [0.1e-6], "efficiencies": [0.99999]},
        },
        "solver": solver_block(cap, v_min),
        "turbulence": dict(TURBULENCE),
        "boundaries": {
            "inlet": {
                "type": "velocity_inlet",
                "location": "left",
                "y_start": h,
                "y_end": height,
                "velocity": u_in,
                "turbulence_intensity": INLET_INTENSITY,
                "dissipation_length": INLET_LENGTH,
            },
            "outlet": {
                "type": "pressure_outlet",
                "location": "right",
                "y_start": 0.0,
                "y_end": height,
            },
        },
        "obstacles": [
            {
                "name": "step",
                "x_start": 0.0,
                "x_end": length,
                "y_start": 0.0,
                "y_end": h,
            }
        ],
        "sensors": [{"name": "centre", "x": 0.5 * length, "y": 5.0 * h}],
        "thresholds": {"0.1e-6": 100.0},
    }


def case_raw(grid: str, variant: str, scheme: str, cap: int) -> dict:
    """The case file regridded, y clustered as the loader clusters it."""
    nx, ny = GRIDS[grid]
    raw = yaml.safe_load(case_path("backstep").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    loaded = load_backstep("backstep", grid=(nx, ny))
    raw["mesh"]["y"] = {"min_wall_spacing": loaded.stretch_y.min_spacing}
    raw["turbulence"]["variant"] = variant
    if scheme:
        raw["turbulence"]["advection_scheme"] = scheme
    if cap:
        raw["solver"]["max_simple_iter"] = cap
    return raw


def finish(name: str, rec: dict) -> None:
    """Write the mean-pressure history beside the record and print the drift reading."""
    p_mean = np.array(P_MEAN)
    window = min(200, max(p_mean.size - 1, 0))
    rate = float(np.mean(np.diff(p_mean[-(window + 1) :]))) if window > 0 else None
    out = {
        "name": name,
        "outer": int(p_mean.size),
        "p_mean_end": float(p_mean[-1]) if p_mean.size else None,
        "p_mean_rate_last_200": rate,
        "p_mean": p_mean.tolist(),
    }
    (OUT / f"{name}_pmean.json").write_text(json.dumps(out))
    print(
        f"{name} p_mean end {out['p_mean_end']} rate over the last {window} outer "
        f"iterations {rate} Pa per iteration; stop {rec['stop']} outer {rec['outer']}",
        flush=True,
    )


def precursor(args: argparse.Namespace) -> None:
    """Item 2: the channel alone, to its stop."""
    raw = precursor_raw(args.length_h, args.dx_h, args.ny, args.u_in, args.cap)
    name = "precursor" + (f"_{args.tag}" if args.tag else "")
    meta = {
        "case": "precursor",
        "length_h": args.length_h,
        "dx_h": args.dx_h,
        "ny": args.ny,
        "step_height": STEP_HEIGHT,
        "u_in": args.u_in,
        "u_ref": U_REF,
        "nu": NU,
        "turbulence_intensity": INLET_INTENSITY,
        "dissipation_length": INLET_LENGTH,
        "cap": args.cap,
        "stopping": raw["solver"]["stopping_rule"],
        "mass_imbalance_tol": raw["solver"]["mass_imbalance_tol"],
        "locate": True,
    }
    rec = coupled46.run_room(name, raw, meta, log_every=args.log_every, locate=True)
    finish(name, rec)


def run(args: argparse.Namespace) -> None:
    """Item 4: one row through prompt 46's runner."""
    raw = case_raw(args.grid, args.variant, args.scheme, args.cap)
    config = SimConfig.from_dict(raw)
    scheme = config.turbulence.advection_scheme
    name = f"{args.variant}_{args.grid}_{scheme}" + (f"_{args.tag}" if args.tag else "")
    inlet = raw["boundaries"]["inlet"]
    meta = {
        "case": "backstep",
        "grid": args.grid,
        "variant": args.variant,
        "advection_scheme": scheme,
        "scheme_key_stated": bool(args.scheme),
        "step_height": STEP_HEIGHT,
        "u_ref": U_REF,
        "inlet_velocity": inlet["velocity"],
        "turbulence_intensity": inlet["turbulence_intensity"],
        "dissipation_length": inlet["dissipation_length"],
        "sweeps": raw["solver"]["momentum_sweeps"],
        "pressure_rtol": raw["solver"]["pressure_rtol"],
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
    finish(name, rec)


def timing(args: argparse.Namespace) -> None:
    """A short run whose seconds per outer iteration sizes the set."""
    args.variant = "standard"
    args.scheme = ""
    args.cap = TIME_CAP
    args.tag = "time"
    args.log_every = 10
    run(args)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("precursor")
    p.add_argument("--length-h", type=float, default=PRECURSOR_LENGTH_H)
    p.add_argument("--dx-h", type=float, default=PRECURSOR_DX_H)
    p.add_argument("--ny", type=int, default=PRECURSOR_NY)
    p.add_argument("--u-in", type=float, default=PRECURSOR_U_IN)
    p.add_argument("--cap", type=int, default=PRECURSOR_CAP)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--tag", default="")
    p.set_defaults(func=precursor)
    p = sub.add_parser("run")
    p.add_argument("grid", choices=sorted(GRIDS))
    p.add_argument("--scheme", choices=("upwind", "umist"), default="")
    p.add_argument("--variant", choices=("standard", "rng"), default="standard")
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--tag", default="")
    p.set_defaults(func=run)
    p = sub.add_parser("time")
    p.add_argument("grid", choices=sorted(GRIDS))
    p.set_defaults(func=timing)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
