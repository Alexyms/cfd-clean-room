"""Prompt 68: does pressure_rtol 1e-4 move VAL-022's Strouhal number? (Alex, 2026-10-18).

ADR-013 decision 3 keeps pressure_rtol 1e-8 for the laminar validation
cases; the cylinder's file runs 1e-4, T2's setting. Alex asked for a
measurement before the decision is amended for tier-2 cases: from the Re 100
clustered row's developed shedding (its final checkpoint, step 4000 at t = 200,
with step 3999 as the previous level), march about twelve more periods at
1e-8 and compare the Strouhal number of the last ten with the 1e-4 value; the
amendment stands if they agree within 0.5%. A separate script, not a restart:
the restart hash refuses the changed key.

Two arms from the same checkpoint, run side by side: RTOL 1e-8, and 1e-4 as
the control, so the two readings share their window. Each is the case file's
configuration (validation.cases.load_preset("val022_re100")) with only
pressure_rtol changed, marched by solve_transient(resume=...) to t = 200 +
STEPS dt; every step's lift and drag by forces.ForceGauge; the Strouhal number
by forces.strouhal over the last ten periods.

    python rtol68.py RTOL [--checkpoint PATH]

Writes results/builder68/rtol68_<RTOL>.json: the reading, the drag
coefficient over the window, the outer counts, the worst imbalance and the
wall time. One BLAS thread.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.forces import ForceGauge, strouhal  # noqa: E402
from src.io_manager import RunFile  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.solver_staggered import FlowState, StaggeredSolver, TransientStep  # noqa: E402
from validation.cases import load_preset  # noqa: E402

OUT = ROOT / "results" / "builder68"
CHECKPOINT = (
    ROOT
    / "results"
    / "builder68"
    / "R"
    / "results"
    / "val022"
    / "val022_re100_b53bab68.h5"
)
# About twelve periods at St 0.146 (6.85 units each) in steps of 0.05.
STEPS = 1640


def commit() -> str:
    """The short commit of the tree the probe runs in."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.strip()


def main() -> None:
    """March one arm from the Re 100 row's checkpoint and write its record."""
    parser = argparse.ArgumentParser(description="Prompt 68, the rtol probe")
    parser.add_argument("rtol", type=float)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    raw = copy.deepcopy(load_preset("val022_re100").merged_mapping())
    raw["solver"]["pressure_rtol"] = args.rtol
    t0_checkpoint, resume = RunFile(args.checkpoint).read_transient_checkpoint()
    raw["solver"]["t_end"] = (resume.step + STEPS) * raw["solver"]["dt"]
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    spec = config.transient
    gauge = ForceGauge(
        mesh, config, spec.forces.obstacle, spec.forces.reference_velocity
    )
    rows: list[tuple[int, float, int, float, float, float]] = []

    def on_step(record: TransientStep, state: FlowState) -> None:
        c_d, c_l = gauge.coefficients(gauge.forces(state.u, state.v, state.p))
        rows.append(
            (record.step, record.t, record.outer, record.worst_imbalance, c_d, c_l)
        )
        if record.step % 200 == 0:
            print(
                f"rtol {args.rtol:g} step {record.step}: {record.outer} outer, C_L {c_l:+.4f}",
                flush=True,
            )

    started = datetime.now().isoformat(timespec="seconds")
    clock = perf_counter()
    solver.solve_transient(
        config.dt,
        config.t_end,
        resume=resume,
        step_tol=spec.step_tol,
        step_cap=spec.step_cap,
        scheme=spec.scheme,
        on_step=on_step,
    )
    wall = perf_counter() - clock
    t = np.array([r[1] for r in rows])
    c_d = np.array([r[4] for r in rows])
    c_l = np.array([r[5] for r in rows])
    outer = np.array([r[2] for r in rows])
    reading = strouhal(t, c_l, gauge.height, spec.forces.reference_velocity)
    window = t >= reading.crossings[0]
    record = {
        "kind": "rtol",
        "commit": commit(),
        "started": started,
        "pressure_rtol": args.rtol,
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "from_step": resume.step,
        "from_t": t0_checkpoint,
        "steps": len(rows),
        "t_end": config.t_end,
        "strouhal": reading.strouhal,
        "period": reading.period,
        "spectral_strouhal": reading.spectral_strouhal,
        "resolution": reading.resolution,
        "crossings": reading.crossings.tolist(),
        "drag_coefficient": float(np.mean(c_d[window])),
        "lift_amplitude": 0.5 * float(c_l[window].max() - c_l[window].min()),
        "outer_mean": float(outer.mean()),
        "outer_max": int(outer.max()),
        "worst_imbalance_max": max(r[3] for r in rows),
        "wall_seconds": wall,
        "seconds_per_step": wall / len(rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rtol68_{args.rtol:g}.json").write_text(
        json.dumps(record, indent=1), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in record.items() if k != "crossings"}, indent=1),
        flush=True,
    )


if __name__ == "__main__":
    main()
