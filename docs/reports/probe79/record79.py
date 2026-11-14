"""Run scripts/run_scenario.py with a recorder of the transport solve's sweeps (prompt 79).

    python docs/reports/probe79/record79.py SCENARIO --out RUN.h5 [run_scenario's options]

The arguments are run_scenario.py's, and its ``main`` runs in this process
after one change: ``TransportSolver.solve_timestep`` is wrapped so that after
each call the class and ``last_diffusion_sweeps`` (the Jacobi or conjugate
gradient sweeps of the implicit solve the call made) are appended to a list.
The wrapper reads an attribute after the call returns and passes the call's
arguments and result through untouched, so the run is the runner's; the
calibration in tables79.py compares a recorded run with an unrecorded one,
bit for bit. Run files do not hold the sweeps, and the prompt's cost table
asks for sweeps per solve.

At exit the list goes to ``RUN.sweeps.<pid>.npz`` beside the run file
(``class`` and ``sweeps``, one entry per solve in call order), one file per
process so a restarted run's attempts do not overwrite each other. A process
killed before it exits leaves no file for its attempt.
"""

import atexit
import os
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.solver_transport import (  # noqa: E402 -- follows sys.path.insert
    TransportSolver,
)

CLASSES: list[int] = []
SWEEPS: list[int] = []


def run_file(argv: list[str]) -> Path:
    """The run file the arguments name, after --out or --restart."""
    for flag in ("--out", "--restart"):
        if flag in argv:
            return Path(argv[argv.index(flag) + 1])
    raise SystemExit("record79: give --out or --restart")


def install() -> None:
    """Wrap solve_timestep with the recorder."""
    original = TransportSolver.solve_timestep

    def recorded(self: TransportSolver, *args: object, **kwargs: object) -> np.ndarray:
        result = original(self, *args, **kwargs)
        size_class = kwargs.get("size_class", args[2] if len(args) > 2 else -1)
        CLASSES.append(int(size_class))
        SWEEPS.append(int(self.last_diffusion_sweeps))
        return result

    TransportSolver.solve_timestep = recorded


def save(path: Path) -> None:
    """Write the recorded solves beside the run file."""
    target = path.with_name(f"{path.stem}.sweeps.{os.getpid()}.npz")
    np.savez(
        target,
        **{
            "class": np.array(CLASSES, dtype=np.int8),
            "sweeps": np.array(SWEEPS, dtype=np.int32),
        },
    )


def main() -> None:
    """Install the recorder and run run_scenario.py's main with the arguments."""
    path = run_file(sys.argv[1:])
    install()
    atexit.register(save, path)
    sys.argv = [str(ROOT / "scripts" / "run_scenario.py"), *sys.argv[1:]]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
