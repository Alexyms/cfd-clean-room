"""Benchmark harness: append one JSON record per solver run to benchmarks/results.jsonl.

Three independent changes are coming: the staggered rebuild changes the
accuracy obtained on a given grid, the linear solver choice changes how much
work is needed to reach that accuracy, and a GPU port changes how fast that
work executes. A record of wall time alone cannot attribute an improvement
to any of the three. Each record therefore carries accuracy, work in
hardware-independent cell updates, wall time, and a trajectory of error
against cumulative work sampled during the solve, so any fixed-accuracy
comparison can be read off later without rerunning anything.

The results file is append-only and never rewritten. Repeats are stored as
separate records; averaging is a presentation decision.

Run:

    python scripts/benchmark.py                       # seed cases, 3 repeats
    python scripts/benchmark.py --cases val002_20x20  # one case
    python scripts/benchmark.py --summary             # table of what is stored
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.boundary import BoundaryManager  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import FLUID, Mesh  # noqa: E402
from src.solver_ns import IterationState, NavierStokesSolver  # noqa: E402
from tests.test_lid_cavity import (  # noqa: E402
    GHIA_U_VAL,
    GHIA_U_Y,
    GHIA_V_VAL,
    GHIA_V_X,
    _make_cavity_config,
)
from tests.test_poiseuille import _make_poiseuille_config  # noqa: E402

SCHEMA_VERSION = 1
RESULTS_PATH = REPO_ROOT / "benchmarks" / "results.jsonl"
DEFAULT_METHOD = "collocated-jacobi"

# Case id -> (kind, nx, ny). Settings come from the validation tests themselves
# so the harness and the tests cannot drift apart; only the grid is overridden.
CASES: dict[str, tuple[str, int, int]] = {
    "val001_80x40": ("poiseuille", 80, 40),
    "val002_20x20": ("cavity", 20, 20),
    "val002_40x40": ("cavity", 40, 40),
    "val002_80x80": ("cavity", 80, 80),
}
DEFAULT_CASES = ["val001_80x40", "val002_20x20", "val002_40x40"]


def build_config(kind: str, nx: int, ny: int, workdir: Path) -> tuple[SimConfig, dict]:
    """Build the validation test's config with the grid overridden.

    Returns the loaded SimConfig and the raw solver block for the record.
    """
    if kind == "poiseuille":
        _make_poiseuille_config(workdir)
        path = workdir / "poiseuille.yaml"
    else:
        _make_cavity_config(workdir)
        path = workdir / "cavity.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["domain"]["nx"] = nx
    raw["domain"]["ny"] = ny
    path.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")
    return SimConfig(str(path)), raw["solver"]


def poiseuille_error(config: SimConfig, mesh: Mesh, u: np.ndarray) -> dict:
    """L2 relative error of the mid-channel u profile against the parabola."""
    i_mid = config.nx // 2
    fluid = mesh.cell_type[:, i_mid] == FLUID
    y = mesh.yc[fluid]
    u_num = u[fluid, i_mid]
    height = config.room_height
    u_ref = 1.5 * 0.1 * 4.0 * y * (height - y) / height**2
    value = float(np.sqrt(np.sum((u_num - u_ref) ** 2) / np.sum(u_ref**2)))
    return {
        "metric": "l2_relative_error_u_midchannel",
        "value": value,
        "reference": "analytical_poiseuille",
    }


def cavity_error(config: SimConfig, mesh: Mesh, u: np.ndarray, v: np.ndarray) -> dict:
    """Max normalized centerline error against Ghia et al. (1982), Re=100."""
    i_mid = config.nx // 2
    fluid_col = mesh.cell_type[:, i_mid] == FLUID
    y_profile = [0.0, *mesh.yc[fluid_col], 1.0]
    u_profile = [0.0, *u[fluid_col, i_mid], 1.0]
    u_err = float(
        np.max(np.abs(np.interp(GHIA_U_Y, y_profile, u_profile) - GHIA_U_VAL))
    )

    j_mid = config.ny // 2
    fluid_row = mesh.cell_type[j_mid, :] == FLUID
    x_profile = [0.0, *mesh.xc[fluid_row], 1.0]
    v_profile = [0.0, *v[j_mid, fluid_row], 0.0]
    v_err = float(
        np.max(np.abs(np.interp(GHIA_V_X, x_profile, v_profile) - GHIA_V_VAL))
    )
    return {
        "metric": "max_normalized_centerline_error",
        "value": max(u_err, v_err),
        "components": {"u": u_err, "v": v_err},
        "reference": "ghia_1982_re100",
    }


def git_state() -> tuple[str, bool]:
    """Identify the commit a run is measured on and whether the tree matches it.

    Returns
    -------
    tuple[str, bool]
        The HEAD commit hash, and True when any tracked file other than
        benchmarks/results.jsonl differs from that commit.
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.splitlines()
    dirty = any(not line.endswith("benchmarks/results.jsonl") for line in status)
    return commit, dirty


CELL_UPDATE_DEFINITION = (
    "stencil evaluations at FLUID cells: two momentum sweeps per outer iteration "
    "plus one per Jacobi pressure sweep; SOLID cells are not counted"
)


def fluid_cells_per_sweep(mesh: Mesh) -> int:
    """Number of unknowns one solver sweep updates: the FLUID cells.

    The work axis counts what the algorithm updates, not what a vectorised
    implementation touches. Whether an implementation spends arithmetic on
    masked SOLID cells is throughput, which the time axis already carries as
    seconds per cell update; counting them here would put an implementation
    detail inside the axis built to exclude it. The seed cases have no
    obstacles, so for them this equals the interior count (ny - 2) * (nx - 2)
    and the stored rows are unchanged.

    Parameters
    ----------
    mesh : Mesh
        Classified mesh of the case.

    Returns
    -------
    int
        Count of cells typed FLUID.
    """
    return int(np.count_nonzero(mesh.cell_type == FLUID))


class WorkCounter:
    """Accumulate the work record from the solver's per-iteration callbacks.

    Parameters
    ----------
    cells_per_sweep : int
        Unknowns updated by one sweep, from fluid_cells_per_sweep.
    """

    def __init__(self, cells_per_sweep: int) -> None:
        self.cells_per_sweep = cells_per_sweep
        self.work: dict[str, int] = {
            "outer_iterations": 0,
            "inner_sweeps": 0,
            "cell_updates": 0,
        }

    def record(self, state: IterationState) -> None:
        """Add one SIMPLE iteration: two momentum sweeps plus its pressure sweeps.

        Parameters
        ----------
        state : IterationState
            Snapshot handed to the solve_steady callback.
        """
        self.work["outer_iterations"] = state.iteration + 1
        self.work["inner_sweeps"] += state.pressure_sweeps
        self.work["cell_updates"] += self.cells_per_sweep * (2 + state.pressure_sweeps)


def positive_int(text: str) -> int:
    """Parse a command line integer that must be at least 1.

    Parameters
    ----------
    text : str
        The raw argument.

    Returns
    -------
    int
        The parsed value.

    Raises
    ------
    argparse.ArgumentTypeError
        When the value is not an integer or is below 1. A caller who asked
        for zero samples or a negative repeat count hears about it instead
        of getting a ZeroDivisionError or a silent no-op.
    """
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def cpu_name() -> str:
    """Best available CPU model string without a third-party dependency."""
    name = ""
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            name = os.environ.get("PROCESSOR_IDENTIFIER", "")
    cpuinfo = Path("/proc/cpuinfo")
    if not name and cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    return name or platform.processor() or platform.machine()


def run_case(
    case_id: str, method: str, sample_every: int, concurrent: int, workdir: Path
) -> dict:
    """Run one case once and return its record."""
    kind, nx, ny = CASES[case_id]
    config, solver_block = build_config(kind, nx, ny, workdir)
    mesh = Mesh(config)
    boundary = BoundaryManager(mesh, config)
    solver = NavierStokesSolver(mesh, config, boundary)

    def error_of(u: np.ndarray, v: np.ndarray) -> dict:
        if kind == "poiseuille":
            return poiseuille_error(config, mesh, u)
        return cavity_error(config, mesh, u, v)

    # One cell update = one stencil evaluation at one fluid cell. Momentum
    # counts two sweeps (u and v) per outer iteration, pressure one per Jacobi
    # sweep. Comparable across Jacobi, Krylov, multigrid, CPU and GPU.
    counter = WorkCounter(fluid_cells_per_sweep(mesh))
    work = counter.work
    trajectory: list[dict] = []
    t_start = time.perf_counter()

    def observe(state: IterationState) -> None:
        counter.record(state)
        if state.iteration % sample_every == 0:
            trajectory.append(
                {
                    "outer_iteration": state.iteration,
                    "cell_updates": work["cell_updates"],
                    "elapsed_seconds": time.perf_counter() - t_start,
                    "residual": state.residual,
                    "error": error_of(state.u, state.v)["value"],
                }
            )

    u, v, _p = solver.solve_steady(on_iteration=observe)
    wall = time.perf_counter() - t_start
    accuracy = error_of(u, v)
    final_iter = work["outer_iterations"] - 1
    if not trajectory or trajectory[-1]["outer_iteration"] != final_iter:
        trajectory.append(
            {
                "outer_iteration": final_iter,
                "cell_updates": work["cell_updates"],
                "elapsed_seconds": wall,
                "residual": solver.residual_history[-1],
                "error": accuracy["value"],
            }
        )

    converged = solver.residual_history[-1] < config.convergence_tol
    commit, dirty = git_state()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "method": method,
        "case": case_id,
        "grid": {"nx": nx, "ny": ny},
        "params": solver_block,
        "environment": {
            "cpu": cpu_name(),
            "cpu_count": os.cpu_count(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "concurrent_processes": concurrent,
        },
        "outcome": {
            "converged": bool(converged),
            "stop_reason": "residual_below_tol" if converged else "max_simple_iter",
        },
        "accuracy": accuracy,
        "work": {**work, "cell_update_definition": CELL_UPDATE_DEFINITION},
        "time": {"wall_seconds": wall, "stages": dict(solver.stage_seconds)},
        "trajectory": trajectory,
    }


def print_summary(path: Path) -> None:
    """Group stored records by method and case and print one row per group."""
    if not path.exists():
        print(f"{path} does not exist; nothing recorded yet.")
        return
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["method"], record["case"]), []).append(record)

    header = (
        f"{'method':<20} {'case':<14} {'n':>2} {'outer':>10} {'wall s (min/med/max)':>24} "
        f"{'cell updates':>14} {'error (min..max)':>20} {'conv':>5}"
    )
    print(header)
    print("-" * len(header))
    for (method, case), runs in sorted(groups.items()):
        outer = [r["work"]["outer_iterations"] for r in runs]
        wall = [r["time"]["wall_seconds"] for r in runs]
        updates = [r["work"]["cell_updates"] for r in runs]
        error = [r["accuracy"]["value"] for r in runs]
        conv = sum(r["outcome"]["converged"] for r in runs)
        print(
            f"{method:<20} {case:<14} {len(runs):>2} "
            f"{min(outer):>4}..{max(outer):<4} "
            f"{min(wall):>7.1f}/{statistics.median(wall):>7.1f}/{max(wall):>7.1f} "
            f"{statistics.median(updates):>14.3e} "
            f"{min(error):>9.3e}..{max(error):<9.3e} {conv:>2}/{len(runs)}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the selected cases and append one record each, or print the summary.

    Parameters
    ----------
    argv : list[str] | None
        Command line arguments. None reads sys.argv.

    Returns
    -------
    int
        Process exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cases", nargs="+", choices=sorted(CASES), default=DEFAULT_CASES
    )
    parser.add_argument(
        "--repeats", type=positive_int, default=3, help="Runs per case, at least 1"
    )
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument(
        "--sample-every",
        type=positive_int,
        default=10,
        help="Trajectory sample interval in outer iterations, at least 1",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Processes this run shared the machine with, this one included",
    )
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--summary", action="store_true", help="Print stored results")
    args = parser.parse_args(argv)

    if args.summary:
        print_summary(args.results)
        return 0

    args.results.parent.mkdir(parents=True, exist_ok=True)
    for case_id in args.cases:
        for repeat in range(args.repeats):
            with tempfile.TemporaryDirectory() as tmp:
                record = run_case(
                    case_id, args.method, args.sample_every, args.concurrent, Path(tmp)
                )
            with args.results.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"{case_id} repeat {repeat + 1}/{args.repeats}: "
                f"{record['work']['outer_iterations']} outer, "
                f"{record['time']['wall_seconds']:.1f} s, "
                f"error {record['accuracy']['value']:.3e}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
