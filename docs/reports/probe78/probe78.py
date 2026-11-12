"""Make one of prompt 78's runs on one side, plain or with the step recorder.

    python docs/reports/probe78/probe78.py RUN SIDE [--concurrent N]

SIDE is ``main`` (origin/main's code, from configs78.MAIN_TREE), ``branch``
(the working tree's) or ``probe`` (the working tree's, recorded). Each run goes
through its tree's ``scripts/run_scenario.py`` ``main`` in this process, the
summary written to ``results/builder78/SIDE/RUN.txt`` and the exit code, the
wall time and the process count beside it to ``RUN.meta.json``.

The ``probe`` side wraps two names in this process only, each calling the
original with the arguments it was given and returning what it returned:
``TransportSolver.solve_timestep``, after which it records the class's
budget (leftover, supply, residual) and the solver's counters (sweeps, the
step's leftover and its share, whether the stop held), and
``solver_transport.implicit_step``, which also solves the same system without
``leftover_bound``, the max-norm stop alone, and records that solve's sweeps:
how many more the share asked for. The second solve is a pure function of
copies of its inputs, so the run's fields are the plain run's, which the
calibration table checks against the ``branch`` run file at every frame. The
rows go to ``results/builder78/probe/RUN.steps.npz``, one array per class.
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import configs78  # noqa: E402 -- follows sys.path.insert

COLUMNS: tuple[str, ...] = (
    "step",
    "dt",
    "leftover",
    "supplied",
    "residual",
    "current",
    "sweeps",
    "sweeps_free",
    "step_leftover",
    "share",
    "converged",
)


def _install(rows: dict[int, list[list[float]]]) -> None:
    """Wrap the two names of the working tree's transport solver (probe side)."""
    from src import solver_transport

    original_step = solver_transport.TransportSolver.solve_timestep
    original_solve = solver_transport.implicit_step
    free: list[int] = []
    counts: dict[int, int] = {}

    def solve(*args: object, **kwargs: object) -> object:
        result = original_solve(*args, **kwargs)
        unbounded = dict(kwargs)
        unbounded.pop("leftover_bound", None)
        free.append(original_solve(*args, **unbounded).iterations)
        return result

    def step(
        self: solver_transport.TransportSolver,
        c_k: np.ndarray,
        faces: object,
        size_class: int,
        dt: float,
        *args: object,
        **kwargs: object,
    ) -> np.ndarray:
        free.clear()
        out = original_step(self, c_k, faces, size_class, dt, *args, **kwargs)
        budget = self.budget[size_class]
        counts[size_class] = counts.get(size_class, 0) + 1
        rows.setdefault(size_class, []).append(
            [
                counts[size_class],
                dt,
                budget.leftover,
                budget.supplied(),
                budget.residual(),
                budget.current,
                self.last_diffusion_sweeps,
                free[0] if free else 0,
                self.last_leftover,
                self.last_leftover_bound,
                float(self.diffusion_converged),
            ]
        )
        return out

    solver_transport.implicit_step = solve
    solver_transport.TransportSolver.solve_timestep = step


def make(name: str, side: str, concurrent: int) -> int:
    """Make one run; returns the runner's exit code."""
    tree = configs78.tree("main" if side == "main" else "branch")
    sys.path.insert(0, str(tree / "scripts"))
    sys.path.insert(0, str(tree))
    import run_scenario

    folder = configs78.RESULTS / side
    folder.mkdir(parents=True, exist_ok=True)
    target = configs78.run_path(name, side)
    for stale in (target, target.with_suffix(".checkpoint.h5")):
        if stale.exists():
            stale.unlink()
    rows: dict[int, list[list[float]]] = {}
    if side == "probe":
        _install(rows)
    run = configs78.BY_NAME[name]
    argv = [
        str(configs78.scenario_path(name, side)),
        "--out",
        str(target),
        "--flows-from",
        str(run.flows_from),
        "--concurrent",
        str(concurrent),
        "--quiet",
    ]
    buffer = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        status = run_scenario.main(argv)
    wall = time.perf_counter() - started
    (folder / f"{name}.txt").write_text(buffer.getvalue(), encoding="utf-8")
    meta = {
        "run": name,
        "side": side,
        "tree": str(tree),
        "exit": status,
        "wall_seconds": wall,
        "concurrent": concurrent,
    }
    (folder / f"{name}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    if side == "probe":
        np.savez(
            folder / f"{name}.steps.npz",
            columns=np.array(COLUMNS),
            **{f"class_{k}": np.array(v, dtype=np.float64) for k, v in rows.items()},
        )
    return status


def main() -> int:
    """Make the named run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", choices=sorted(configs78.BY_NAME))
    parser.add_argument("side", choices=configs78.SIDES)
    parser.add_argument("--concurrent", type=int, default=0)
    args = parser.parse_args()
    return make(args.run, args.side, args.concurrent)


if __name__ == "__main__":
    sys.exit(main())
