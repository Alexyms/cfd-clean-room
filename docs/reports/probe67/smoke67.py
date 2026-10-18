"""Prompt 67: a smoke run of solve_transient on the probe's cylinder (ADR-014 step 6a).

The configuration is docs/reports/probe59/cylinder59.py's ``cylinder_raw`` on
its uniform 0.1 d grid (370x200): the square cylinder at Re 100 with the
freestream lateral boundaries. The start-up field is the probe's, 200 steady
outer iterations from rest, solved again here and compared, face for face,
with the probe's kept field (results/builder59/cylinder_startup_dx0.1.npz).
From it each arm takes STEPS steps of solve_transient at the per-step
criterion 1e-4 U (decision 5) and the product's pressure_rtol of 1e-4, as
T2's arms A and B did:

    A_ie    implicit Euler, dt 0.025 (T2's arm A, the probe's own scheme)
    A_bdf2  BDF2, dt 0.025 (the paper's step)
    B_bdf2  BDF2, dt 0.05 (decision 3's proposed step)

A_ie's per-step counts and wake velocity are compared with the probe's
record of arm A (results/builder59/cylinder_A.json, its 20 steps). No
VAL-022 row: that is step 6b's.

    python smoke67.py [--steps 40] [--arms A_ie,A_bdf2,B_bdf2] [--concurrent N]

Writes results/builder67/smoke67.json: per arm and step the outer count, the
worst imbalance, the cap hits, the wake's cell-centred v and the wall time.
One BLAS thread.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import importlib.util
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
from src.mesh import Mesh  # noqa: E402
from src.solver_staggered import (  # noqa: E402
    FlowState,
    StaggeredSolver,
    TransientStep,
)
from src.staggered import to_cell_centers  # noqa: E402

OUT = ROOT / "results" / "builder67"
PROBE_RECORDS = ROOT / "results" / "builder59"
DX = 0.1
STARTUP_OUTER = 200
STEP_TOL = 1e-4
PRESSURE_RTOL = 1e-4
STEP_CAP = 500
ARMS = {
    "A_ie": ("implicit_euler", 0.025),
    "A_bdf2": ("bdf2", 0.025),
    "B_bdf2": ("bdf2", 0.05),
}


def commit() -> str:
    """The short commit of the tree the probe runs in."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.strip()


def probe59() -> object:
    """cylinder59.py as a module, for its configuration and wake cell only."""
    path = ROOT / "docs" / "reports" / "probe59" / "cylinder59.py"
    spec = importlib.util.spec_from_file_location("cylinder59", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(raw: dict[str, object]) -> tuple[Mesh, StaggeredSolver]:
    """The mesh and a solver of the configuration mapping."""
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    return mesh, StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))


def startup(cyl: object) -> tuple[FlowState, dict]:
    """The probe's start-up: STARTUP_OUTER steady outer iterations from rest."""
    raw = cyl.cylinder_raw(DX, STARTUP_OUTER, 1e-12, PRESSURE_RTOL)
    _mesh, solver = build(raw)
    t0 = perf_counter()
    solver.solve_steady()
    wall = perf_counter() - t0
    state = solver.flow_state
    assert state is not None
    kept = np.load(PROBE_RECORDS / f"cylinder_startup_dx{DX:g}.npz")
    same = bool(
        np.array_equal(state.u, kept["u_faces"])
        and np.array_equal(state.v, kept["v_faces"])
        and np.array_equal(state.p, kept["p"])
    )
    largest = float(
        max(
            np.max(np.abs(state.u - kept["u_faces"])),
            np.max(np.abs(state.v - kept["v_faces"])),
        )
    )
    print(
        f"startup: {len(solver.residual_history)} outer, {wall:.0f} s, "
        f"equal to the probe's kept field: {same}",
        flush=True,
    )
    return state, {
        "outer": len(solver.residual_history),
        "wall_seconds": wall,
        "equal_to_probe59_startup": same,
        "largest_face_difference": largest,
    }


def march(
    cyl: object, arm: str, start: FlowState, steps: int, concurrent: int
) -> dict[str, object]:
    """STEPS steps of one arm from the start-up field."""
    scheme, dt = ARMS[arm]
    raw = cyl.cylinder_raw(DX, STEP_CAP, 1e-12, PRESSURE_RTOL)
    mesh, solver = build(raw)
    j, i = cyl.wake_probe(mesh)
    rows: list[dict] = []
    clock = [perf_counter()]

    def on_step(record: TransientStep, state: FlowState) -> None:
        """Record the step's count, imbalance, wake velocity and wall time."""
        now = perf_counter()
        _u_c, v_c = to_cell_centers(np.array(state.u), np.array(state.v))
        rows.append(
            {
                "step": record.step,
                "t": record.t,
                "scheme": record.scheme,
                "outer": record.outer,
                "residual_end": record.residual_history[-1],
                "worst_imbalance": record.worst_imbalance,
                "cap_hits": record.pressure_cap_hits,
                "wake_v": float(v_c[j, i]),
                "wall_seconds": now - clock[0],
            }
        )
        clock[0] = now
        if record.step % 10 == 0:
            print(
                f"{arm} step {record.step}: {record.outer} outer, imbalance "
                f"{record.worst_imbalance:.1e}, wake v {rows[-1]['wake_v']:+.5f}",
                flush=True,
            )

    started = datetime.now().isoformat(timespec="seconds")
    solver.solve_transient(
        dt,
        steps * dt,
        initial=start,
        step_tol=STEP_TOL,
        step_cap=STEP_CAP,
        on_step=on_step,
        scheme=scheme,
    )
    outer = np.array([r["outer"] for r in rows])
    seconds = np.array([r["wall_seconds"] for r in rows])
    return {
        "arm": arm,
        "scheme": scheme,
        "dt": dt,
        "courant": dt / DX,
        "steps": steps,
        "started": started,
        "concurrent_processes": concurrent,
        "outer_mean": float(outer.mean()),
        "outer_median": float(np.median(outer)),
        "outer_min": int(outer.min()),
        "outer_max": int(outer.max()),
        "outer_counts": {str(k): int(np.sum(outer == k)) for k in np.unique(outer)},
        "seconds_per_step_mean": float(seconds.mean()),
        "seconds_per_outer": float(seconds.sum() / outer.sum()),
        "cap_hits": int(sum(r["cap_hits"] for r in rows)),
        "rows": rows,
    }


def against_probe(arm: dict[str, object]) -> dict[str, object]:
    """A_ie's steps against the probe's arm A, over the steps both took."""
    probe = json.loads((PROBE_RECORDS / "cylinder_A.json").read_text(encoding="utf-8"))
    pairs = list(zip(arm["rows"], probe["rows"], strict=False))
    return {
        "probe_commit": probe["commit"],
        "steps_compared": len(pairs),
        "outer_equal": sum(a["outer"] == b["outer"] for a, b in pairs),
        "wake_v_equal": sum(a["wake_v"] == b["wake_v"] for a, b in pairs),
        "wake_v_largest_difference": max(
            abs(a["wake_v"] - b["wake_v"]) for a, b in pairs
        ),
    }


def main() -> None:
    """Solve the start-up, march every arm and write the record."""
    parser = argparse.ArgumentParser(description="Prompt 67, the cylinder smoke run")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--concurrent", type=int, default=1)
    args = parser.parse_args()
    cyl = probe59()
    start, start_record = startup(cyl)
    record: dict = {
        "kind": "smoke",
        "commit": commit(),
        "dx": DX,
        "grid": [round(37.0 / DX), round(20.0 / DX)],
        "step_tol_over_U": STEP_TOL,
        "pressure_rtol": PRESSURE_RTOL,
        "startup": start_record,
        "arms": {},
    }
    for arm in args.arms.split(","):
        result = march(cyl, arm, start, args.steps, args.concurrent)
        if arm == "A_ie":
            result["against_probe59_arm_A"] = against_probe(result)
        record["arms"][arm] = result
        print(
            f"{arm}: outer per step mean {result['outer_mean']:.2f}, median "
            f"{result['outer_median']:g}, range {result['outer_min']} to "
            f"{result['outer_max']}, {result['seconds_per_step_mean']:.2f} s per step",
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke67.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
