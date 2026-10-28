"""Prompt 68b: is Re 40's lift bounded by the step tolerance? (orchestrator, 2026-10-28).

The report reads the Re 40 row's stall from t = 101 as the instrument's: the
per-step criterion of 1e-4 U is met on each step's first outer iteration, so
nothing below that level is corrected. Below onset the wake is stable and a
resolved march should keep decaying. The probe lowers the floor: from the
row's final checkpoint (step 4000, t = 200, both levels), it marches on at a
step tolerance of STEP_TOL U with the cap at STEP_CAP and the row's dt, for
SPAN time units or WALL_LIMIT seconds, whichever comes first. A separate
script, not a restart: the restart check refuses a changed step tolerance.

The readings, defined before the probe runs (prompt 68b):

- Peaks. Between two successive sign changes of C_L, the largest |C_L| and
  its time: one peak per half period.
- The envelope's rate. The least-squares slope of ln(peak) against time,
  per time unit (negative for a decay).
- The original rate: the row's own peaks over its last three whole periods
  before t = 101, t from 63.70 to 101.30 (the report's table V7), the last
  periods whose steps still iterated.
- The probe's rate: its peaks from t = 200 + SETTLE on, so the first
  adjustment from the floor is not read as the decay.
- Prediction (e) holds when the probe's rate is within 30% of the original's
  (their ratio in [0.7, 1.3]) and the last peak is below 1e-5.
- Prediction (f) holds when the mean C_D over the probe's last ten units and
  over the row's last ten (t = 190 to 200) agree to four significant figures.
- The row's measure on the continuation: validation.cylinder.decays on the
  probe's series alone, as the orchestrator's rule of 2026-10-28 scores it.

    python floor68.py [--checkpoint PATH] [--span 40] [--wall 7200]

Writes results/builder68/floor68.json: every step's t, outer count, C_L and
C_D, the peaks, both rates, the readings of (e) and (f), the measure's
verdict, the wall time and whether the wall limit stopped it. One BLAS thread.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.forces import ForceGauge  # noqa: E402
from src.io_manager import RunFile  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.solver_staggered import FlowState, StaggeredSolver, TransientStep  # noqa: E402
from validation import cylinder  # noqa: E402
from validation.cases import load_preset  # noqa: E402

OUT = ROOT / "results" / "builder68"
CHECKPOINT = (
    ROOT
    / "results"
    / "builder68"
    / "R"
    / "results"
    / "val022"
    / "val022_re40_78fae737.h5"
)
ROW_RECORD = HERE / "val022_re40.json"
STEP_TOL = 1e-6
STEP_CAP = 500
SPAN = 40.0
WALL_LIMIT = 7200.0
SETTLE = 10.0
ORIGINAL_WINDOW = (63.70, 101.30)
RATE_TOLERANCE = 0.3
ENVELOPE_LIMIT = 1e-5


class _WallLimitError(Exception):
    """The wall limit reached: the march stops and what it took is kept."""


def commit() -> str:
    """The short commit of the tree the probe runs in."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.strip()


def peaks(t: np.ndarray, c_l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The largest |C_L| between successive sign changes, with its time."""
    change = np.nonzero(np.sign(c_l[:-1]) != np.sign(c_l[1:]))[0]
    times, values = [], []
    for a, b in pairwise(change):
        span = np.abs(c_l[a + 1 : b + 1])
        k = int(np.argmax(span))
        times.append(float(t[a + 1 + k]))
        values.append(float(span[k]))
    return np.array(times), np.array(values)


def rate(times: np.ndarray, values: np.ndarray) -> float | None:
    """The least-squares slope of ln(peak) against time; None with fewer than three."""
    if times.size < 3:
        return None
    return float(np.polyfit(times, np.log(values), 1)[0])


def main() -> None:
    """March the continuation and write its record."""
    parser = argparse.ArgumentParser(description="Prompt 68b, the Re 40 floor probe")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--span", type=float, default=SPAN)
    parser.add_argument("--wall", type=float, default=WALL_LIMIT)
    args = parser.parse_args()
    t0_checkpoint, resume = RunFile(args.checkpoint).read_transient_checkpoint()
    raw = copy.deepcopy(load_preset("val022_re40").merged_mapping())
    raw["transient"]["step_tol"] = STEP_TOL
    raw["transient"]["step_cap"] = STEP_CAP
    dt = raw["solver"]["dt"]
    steps = round(args.span / dt)
    raw["solver"]["t_end"] = (resume.step + steps) * dt
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    spec = config.transient
    gauge = ForceGauge(
        mesh, config, spec.forces.obstacle, spec.forces.reference_velocity
    )
    rows: list[tuple[int, float, int, float, float]] = []
    clock = perf_counter()

    def on_step(record: TransientStep, state: FlowState) -> None:
        c_d, c_l = gauge.coefficients(gauge.forces(state.u, state.v, state.p))
        rows.append((record.step, record.t, record.outer, c_d, c_l))
        if record.step % 100 == 0:
            print(
                f"step {record.step} t {record.t:g}: {record.outer} outer, "
                f"C_L {c_l:+.3e}, C_D {c_d:.6f}",
                flush=True,
            )
        if perf_counter() - clock > args.wall:
            raise _WallLimitError

    started = datetime.now().isoformat(timespec="seconds")
    stopped_by_wall = False
    try:
        solver.solve_transient(
            config.dt,
            config.t_end,
            resume=resume,
            step_tol=spec.step_tol,
            step_cap=spec.step_cap,
            scheme=spec.scheme,
            on_step=on_step,
        )
    except _WallLimitError:
        stopped_by_wall = True
    wall = perf_counter() - clock

    t = np.array([r[1] for r in rows])
    outer = np.array([r[2] for r in rows])
    c_d = np.array([r[3] for r in rows])
    c_l = np.array([r[4] for r in rows])
    row = json.loads(ROW_RECORD.read_text(encoding="utf-8"))
    t_row = np.array(row["series"]["t"])
    c_l_row = np.array(row["series"]["c_l"])
    c_d_row = np.array(row["series"]["c_d"])

    row_times, row_peaks = peaks(t_row, c_l_row)
    inside = (row_times >= ORIGINAL_WINDOW[0]) & (row_times <= ORIGINAL_WINDOW[1])
    original_rate = rate(row_times[inside], row_peaks[inside])
    probe_times, probe_peaks = peaks(t, c_l)
    settled = probe_times >= t0_checkpoint + SETTLE
    probe_rate = rate(probe_times[settled], probe_peaks[settled])
    ratio = (
        None
        if original_rate is None or probe_rate is None
        else probe_rate / original_rate
    )
    last_peak = float(probe_peaks[-1]) if probe_peaks.size else None
    e_holds = (
        ratio is not None
        and abs(ratio - 1.0) <= RATE_TOLERANCE
        and last_peak is not None
        and last_peak < ENVELOPE_LIMIT
    )
    probe_drag = float(np.mean(c_d[t > t[-1] - 10.0]))
    row_drag = float(np.mean(c_d_row[t_row > t_row[-1] - 10.0]))
    f_holds = f"{probe_drag:.4g}" == f"{row_drag:.4g}"
    measure = cylinder.decays(t, c_l)

    record = {
        "kind": "floor",
        "commit": commit(),
        "started": started,
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "from_step": resume.step,
        "from_t": t0_checkpoint,
        "step_tol": STEP_TOL,
        "step_cap": STEP_CAP,
        "dt": dt,
        "steps": len(rows),
        "t_end": float(t[-1]),
        "stopped_by_wall": stopped_by_wall,
        "wall_seconds": wall,
        "original_window": list(ORIGINAL_WINDOW),
        "original_peaks": {
            "t": row_times[inside].tolist(),
            "peak": row_peaks[inside].tolist(),
        },
        "original_rate": original_rate,
        "probe_peaks": {"t": probe_times.tolist(), "peak": probe_peaks.tolist()},
        "settle": SETTLE,
        "probe_rate": probe_rate,
        "rate_ratio": ratio,
        "last_peak": last_peak,
        "prediction_e": e_holds,
        "probe_drag": probe_drag,
        "row_drag": row_drag,
        "prediction_f": f_holds,
        "measure_on_continuation": measure,
        "series": {
            "t": t.tolist(),
            "outer": outer.tolist(),
            "c_l": c_l.tolist(),
            "c_d": c_d.tolist(),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "floor68.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    summary = {
        k: v
        for k, v in record.items()
        if k not in ("series", "probe_peaks", "original_peaks")
    }
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
