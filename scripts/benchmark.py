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

Each row runs the case file's stopping rule, except that the collocated
solver, which refuses error_estimate, runs velocity_step
(validation.cases.with_velocity_step) and its params say so. A velocity-step
stop is labelled residual_below_tol whichever solver ran it.

Run:

    python scripts/benchmark.py                       # seed cases, 3 repeats
    python scripts/benchmark.py --cases val002_20x20  # one case
    python scripts/benchmark.py --method staggered-jacobi  # the staggered solver
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
import time
import uuid
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.boundary import BoundaryManager  # noqa: E402 -- follows sys.path.insert
from src.boundary_staggered import (  # noqa: E402 -- follows sys.path.insert
    StaggeredBoundary,
)
from src.config import (  # noqa: E402 -- follows sys.path.insert
    VELOCITY_STEP,
    SimConfig,
)
from src.mesh import FLUID, Mesh  # noqa: E402 -- follows sys.path.insert
from src.momentum import MomentumPredictor  # noqa: E402 -- follows sys.path.insert
from src.pressure import PressureCorrector  # noqa: E402 -- follows sys.path.insert
from src.solver_ns import (  # noqa: E402 -- follows sys.path.insert
    IterationState,
    NavierStokesSolver,
)
from src.solver_staggered import (  # noqa: E402 -- follows sys.path.insert
    StaggeredSolver,
)
from src.staggered import allocate_fields  # noqa: E402 -- follows sys.path.insert
from validation.cases import (  # noqa: E402 -- follows sys.path.insert
    CASE_GRIDS,
    WALL_CLUSTERED_GRIDS,
    load_case,
    load_wall_clustered,
    with_velocity_step,
)
from validation.metrics import (  # noqa: E402 -- follows sys.path.insert
    cavity_true_centerline_errors,
    poiseuille_l2_error,
)

SCHEMA_VERSION = 1
RESULTS_PATH = REPO_ROOT / "benchmarks" / "results.jsonl"
# The method label selects the solver, so a row cannot claim one it did not run.
DEFAULT_METHOD = "collocated-jacobi"
STAGGERED_METHOD = "staggered-jacobi"
METHODS = (DEFAULT_METHOD, STAGGERED_METHOD)

# Grid presets come from validation.cases so the harness, the tests and the
# field viewer name the same solve the same way. Every other setting comes
# from the committed case file.
CASES = CASE_GRIDS
DEFAULT_CASES = ["val001_80x40", "val002_20x20", "val002_40x40"]

# One stop_reason per rule across both solvers. The velocity-step label is the
# one every row stored before the staggered solver reported its own stop.
STOP_LABELS = {"velocity_step_below_tol": "residual_below_tol"}


SOLVER_PARAMETERS = (
    "dt",
    "t_end",
    "output_interval",
    "convergence_tol",
    "max_simple_iter",
    "alpha_velocity",
    "alpha_pressure",
    "max_pressure_iter",
    "pressure_tol",
    "stopping_rule",
    "iteration_error_tol",
    "mass_imbalance_tol",
)


def solver_parameters(config: SimConfig) -> dict:
    """Every solver parameter as loaded, read back from the config object."""
    return {name: getattr(config, name) for name in SOLVER_PARAMETERS}


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


# Recorded with every row, so each row says what its cell_updates number counts.
CELL_UPDATE_DEFINITIONS: dict[str, str] = {
    DEFAULT_METHOD: (
        "stencil evaluations at FLUID cells: two momentum sweeps per outer iteration "
        "plus one per Jacobi pressure sweep; SOLID cells are not counted"
    ),
    STAGGERED_METHOD: (
        "stencil evaluations at unknowns: the unknown u and v faces (a_p > 0) once "
        "per outer iteration for momentum, plus the cells with a pressure equation "
        "(a_P > 0) once per weighted Jacobi pressure sweep"
    ),
}


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


def staggered_updates(mesh: Mesh, config: SimConfig) -> tuple[int, int]:
    """Unknowns the staggered solver updates, read from its own coefficient masks.

    On the staggered grid the BOUNDARY ring cells are ordinary solution
    cells, so the FLUID count would understate the work: a 20x20 cavity
    has 400 cells with a pressure equation against 324 FLUID cells. The
    masks are structural, so one prediction from the initial field gives
    them for the whole solve.

    Parameters
    ----------
    mesh : Mesh
        Classified mesh of the case.
    config : SimConfig
        Case configuration.

    Returns
    -------
    tuple[int, int]
        (momentum updates per outer iteration, the unknown u faces plus the
        unknown v faces with a_p > 0; cells per pressure sweep, those with
        a_P > 0).
    """
    boundary = StaggeredBoundary(mesh, config)
    u, v, p = allocate_fields(mesh)
    boundary.apply_normal_velocity(u, v)
    prediction = MomentumPredictor(mesh, config, boundary).predict(u, v, p)
    coefficients = PressureCorrector(mesh, config, boundary).coefficients(
        prediction.a_p_u, prediction.a_p_v
    )
    momentum = np.count_nonzero(prediction.a_p_u > 0.0) + np.count_nonzero(
        prediction.a_p_v > 0.0
    )
    return int(momentum), int(np.count_nonzero(coefficients.a_p > 0.0))


class WorkCounter:
    """Accumulate the work record from the solver's per-iteration callbacks.

    Parameters
    ----------
    cells_per_sweep : int
        Unknowns updated by one pressure sweep: fluid_cells_per_sweep for
        the collocated solver, the second value of staggered_updates for
        the staggered one.
    momentum_updates : int, optional
        Unknowns the momentum step updates per outer iteration. Defaults to
        ``2 * cells_per_sweep``, the collocated u and v sweeps over the
        same cells.
    """

    def __init__(
        self, cells_per_sweep: int, momentum_updates: int | None = None
    ) -> None:
        self.cells_per_sweep = cells_per_sweep
        self.momentum_updates = (
            2 * cells_per_sweep if momentum_updates is None else momentum_updates
        )
        self.work: dict[str, int] = {
            "outer_iterations": 0,
            "inner_sweeps": 0,
            "cell_updates": 0,
        }

    def record(self, state: IterationState) -> None:
        """Add one SIMPLE iteration: its momentum updates plus its pressure sweeps.

        Parameters
        ----------
        state : IterationState
            Snapshot handed to the solve_steady callback.
        """
        self.work["outer_iterations"] = state.iteration + 1
        self.work["inner_sweeps"] += state.pressure_sweeps
        self.work["cell_updates"] += (
            self.momentum_updates + self.cells_per_sweep * state.pressure_sweeps
        )


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


def accuracy_of(
    kind: str, config: SimConfig, mesh: Mesh, u: np.ndarray, v: np.ndarray
) -> dict:
    """The accuracy a record carries for one field of one case family.

    Parameters
    ----------
    kind : str
        Case family, "poiseuille" or "cavity".
    config : SimConfig
        Case configuration.
    mesh : Mesh
        Mesh the field was computed on.
    u, v : np.ndarray
        Cell-centered velocity fields [ny, nx].

    Returns
    -------
    dict
        The metric's as_dict: its name, value, reference and components.
    """
    if kind == "poiseuille":
        return poiseuille_l2_error(config, mesh, u).as_dict()
    return cavity_true_centerline_errors(config, mesh, u, v).as_dict()


def run_case(case_id: str, method: str, sample_every: int, concurrent: int) -> dict:
    """Run one case once with the solver the method names and return its record.

    Raises
    ------
    ValueError
        If the method is not one of METHODS.
    """
    kind, nx, ny = CASES[case_id]
    loader = load_wall_clustered if case_id in WALL_CLUSTERED_GRIDS else load_case
    config = loader(kind, grid=(nx, ny))
    if method == DEFAULT_METHOD:
        config = with_velocity_step(config)
    solver_block = solver_parameters(config)
    mesh = Mesh(config)

    # One cell update = one stencil evaluation at one unknown, counted per
    # method as CELL_UPDATE_DEFINITIONS states. Comparable across Jacobi,
    # Krylov, multigrid, CPU and GPU.
    solver: NavierStokesSolver | StaggeredSolver
    if method == DEFAULT_METHOD:
        solver = NavierStokesSolver(mesh, config, BoundaryManager(mesh, config))
        counter = WorkCounter(fluid_cells_per_sweep(mesh))
    elif method == STAGGERED_METHOD:
        solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
        momentum_updates, pressure_cells = staggered_updates(mesh, config)
        counter = WorkCounter(pressure_cells, momentum_updates)
    else:
        raise ValueError(f"unknown method {method!r}; known: {list(METHODS)}")

    def error_of(u: np.ndarray, v: np.ndarray) -> dict:
        return accuracy_of(kind, config, mesh, u, v)

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

    # The staggered solver knows which rule it ran and whether the cap stopped
    # it; the collocated one has only the velocity-step rule, read back here.
    if isinstance(solver, StaggeredSolver):
        converged = solver.converged
        stop_reason = STOP_LABELS.get(solver.stop_reason, solver.stop_reason)
    else:
        converged = solver.residual_history[-1] < config.convergence_tol
        stop_reason = "residual_below_tol" if converged else "max_simple_iter"
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
        "outcome": {"converged": bool(converged), "stop_reason": stop_reason},
        "accuracy": accuracy,
        "work": {**work, "cell_update_definition": CELL_UPDATE_DEFINITIONS[method]},
        "time": {"wall_seconds": wall, "stages": dict(solver.stage_seconds)},
        "trajectory": trajectory,
    }


def print_summary(path: Path) -> None:
    """Print one row per method, case, concurrency, rule, metric and reference; flag mixed.

    Wall time is only comparable between runs that shared the machine with
    the same number of processes, so ``concurrent_processes`` is part of the
    grouping key and shown as a column. Accuracy, outer iterations and cell
    updates are deterministic and do not depend on load. When the table
    holds more than one load a note says so, and any single case recorded
    under more than one load is named: the 80x80 cavity row was taken with
    two processes where the 40x40 rows used one, and nothing noticed until a
    reviewer read the raw file.

    An error is only comparable with errors against the same reference, so
    the accuracy reference is part of the row key too and shown last, and a
    case scored against more than one reference is named. The rows stored
    before revision r2 of the Ghia table carry ``ghia_1982_re100`` and would
    otherwise be pooled into one error range with the corrected rows.

    The metric is part of the key for the same reason. The cavity rows stored
    as ``max_normalized_centerline_error`` sample half a cell off the
    centerlines; ``max_normalized_centerline_error_r2`` samples on them.

    So is the stopping rule, ``params.stopping_rule``, shown after the case:
    it sets how far each solve iterates, so outer counts, wall times and
    errors compare only within one rule. A row without the key predates it
    and ran velocity_step.
    """
    if not path.exists():
        print(f"{path} does not exist; nothing recorded yet.")
        return
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    groups: dict[tuple[str, str, int], list[dict]] = {}
    rows: dict[tuple[str, str, int, str, str, str], list[dict]] = {}
    for record in records:
        key = (
            record["method"],
            record["case"],
            record["environment"]["concurrent_processes"],
        )
        groups.setdefault(key, []).append(record)
        # Every harness row names its metric and reference; a hand-built record may not.
        metric = record["accuracy"].get("metric", "-")
        reference = record["accuracy"].get("reference", "-")
        rule = record.get("params", {}).get("stopping_rule", VELOCITY_STEP)
        rows.setdefault((*key, rule, metric, reference), []).append(record)

    header = (
        f"{'method':<20} {'case':<22} {'rule':<14} {'procs':>5} {'n':>2} "
        f"{'outer':>10} {'wall s (min/med/max)':>24} {'cell updates':>14} "
        f"{'error (min..max)':>20} {'conv':>5} {'metric':<34} reference"
    )
    print(header)
    print("-" * len(header))
    for (method, case, procs, rule, metric, reference), runs in sorted(rows.items()):
        outer = [r["work"]["outer_iterations"] for r in runs]
        wall = [r["time"]["wall_seconds"] for r in runs]
        updates = [r["work"]["cell_updates"] for r in runs]
        error = [r["accuracy"]["value"] for r in runs]
        conv = sum(r["outcome"]["converged"] for r in runs)
        print(
            f"{method:<20} {case:<22} {rule:<14} {procs:>5} {len(runs):>2} "
            f"{min(outer):>4}..{max(outer):<4} "
            f"{min(wall):>7.1f}/{statistics.median(wall):>7.1f}/{max(wall):>7.1f} "
            f"{statistics.median(updates):>14.3e} "
            f"{min(error):>9.3e}..{max(error):<9.3e} {conv:>2}/{len(runs)} "
            f"{metric:<34} {reference}"
        )

    references: dict[tuple[str, str], set[str]] = {}
    metrics: dict[tuple[str, str], set[str]] = {}
    rules: dict[tuple[str, str], set[str]] = {}
    for method, case, _procs, rule, metric, reference in rows:
        references.setdefault((method, case), set()).add(reference)
        metrics.setdefault((method, case), set()).add(metric)
        rules.setdefault((method, case), set()).add(rule)
    for (method, case), seen in sorted(references.items()):
        if len(seen) > 1:
            print(
                f"note: {method} {case} has rows scored against {sorted(seen)}; "
                "errors are comparable only within one reference"
            )
    for (method, case), seen in sorted(metrics.items()):
        if len(seen) > 1:
            print(
                f"note: {method} {case} has rows measured by {sorted(seen)}; "
                "errors are comparable only within one metric"
            )
    for (method, case), seen in sorted(rules.items()):
        if len(seen) > 1:
            print(
                f"note: {method} {case} has rows under {sorted(seen)}; outer "
                "iterations, wall times and errors are comparable only within one rule"
            )

    loads_seen = sorted({procs for _, _, procs in groups})
    if len(loads_seen) > 1:
        print(
            f"note: rows above were recorded at concurrent_processes {loads_seen}; "
            "compare wall times only within one value. Accuracy, outer iterations "
            "and cell updates do not depend on load."
        )
    for method, case in sorted(mixed_load_cases(groups)):
        loads = sorted(p for m, c, p in groups if (m, c) == (method, case))
        print(
            f"note: {method} {case} has rows at concurrent_processes "
            f"{loads}; wall times are not comparable across them"
        )


def mixed_load_cases(
    groups: dict[tuple[str, str, int], list[dict]],
) -> set[tuple[str, str]]:
    """Return the (method, case) pairs recorded under more than one load."""
    loads: dict[tuple[str, str], set[int]] = {}
    for method, case, procs in groups:
        loads.setdefault((method, case), set()).add(procs)
    return {key for key, seen in loads.items() if len(seen) > 1}


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
    parser.add_argument(
        "--method",
        choices=METHODS,
        default=DEFAULT_METHOD,
        help="Solver to run; the label is recorded with each row",
    )
    parser.add_argument(
        "--sample-every",
        type=positive_int,
        default=10,
        help="Trajectory sample interval in outer iterations, at least 1",
    )
    parser.add_argument(
        "--concurrent",
        type=positive_int,
        default=1,
        help="Processes this run shared the machine with, itself included, at least 1",
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
            record = run_case(case_id, args.method, args.sample_every, args.concurrent)
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
