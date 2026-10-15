"""Shared pieces of prompt 53's probes: a captured p' system and one timed solve of it.

``load_system`` reads a record of capture53.py and forms the right-hand side
exactly as ``PressureCorrector.correct`` forms it (``f = -b``, projected onto
the range on a closed domain, zero where there is no equation). ``solve``
runs the committed ``conjugate_gradient`` on it under one preconditioner,
timing the setup (building the preconditioner) and the solve apart, with
the stop the corrector uses: the system's ``pressure_rtol``, the floor
``RESIDUAL_FLOOR`` times its flux scale and its cap. Everything is imported
from the ``src`` on sys.path, so the caller decides which tree is measured.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import multigrid
from src.pressure import (
    RESIDUAL_FLOOR,
    PressureCoefficients,
    apply_operator,
    conjugate_gradient,
    jacobi_preconditioner,
    multigrid_preconditioner,
)


@dataclass
class System:
    """One captured p' system and the stop its case solves it to."""

    name: str
    c: PressureCoefficients
    f: np.ndarray
    active: np.ndarray
    needs_pin: bool
    floor: float
    rtol: float
    max_iter: int
    meta: dict


def load_system(path: Path) -> System:
    """A capture53.py record, with f formed as the corrector forms it."""
    data = np.load(path)
    c = PressureCoefficients(
        a_p=data["a_p"],
        a_e=data["a_e"],
        a_w=data["a_w"],
        a_n=data["a_n"],
        a_s=data["a_s"],
    )
    active = c.a_p > 0.0
    needs_pin = bool(data["needs_pin"])
    f = -data["b"]
    if needs_pin:
        f[active] -= f[active].mean()
    f[~active] = 0.0
    return System(
        name=path.stem,
        c=c,
        f=f,
        active=active,
        needs_pin=needs_pin,
        floor=RESIDUAL_FLOOR * float(data["flux_scale"]),
        rtol=float(data["rtol"]),
        max_iter=int(data["max_iter"]),
        meta=json.loads(str(data["meta"])),
    )


def solve(system: System, preconditioner: str, rtol: float | None = None) -> dict:
    """One solve under ``jacobi`` or ``multigrid``: iterations, setup and solve seconds.

    The residual history is not kept; ``conjugate_gradient`` has no hook for
    it and the committed loop is what is timed.
    """
    c = system.c
    t0 = time.perf_counter()
    if preconditioner == "jacobi":
        precondition = jacobi_preconditioner(c)
    else:
        precondition = multigrid_preconditioner(c, system.needs_pin)
    t1 = time.perf_counter()
    result = conjugate_gradient(
        lambda x: apply_operator(c, x),
        precondition,
        system.f,
        system.rtol if rtol is None else rtol,
        system.floor,
        system.max_iter,
    )
    t2 = time.perf_counter()
    f_norm = float(np.sqrt(np.vdot(system.f, system.f)))
    return {
        "iterations": result.iterations,
        "reached_cap": result.reached_cap,
        "products": result.products,
        "relative_residual": result.residual_norm / f_norm if f_norm > 0 else 0.0,
        "residual_norm": result.residual_norm,
        "setup_seconds": t1 - t0,
        "solve_seconds": t2 - t1,
        "total_seconds": t2 - t0,
        "x": result.x,
    }


def hierarchy_summary(system: System) -> list[dict]:
    """Each level's shape, cells with an equation, eigenvalue estimate and weight."""
    hierarchy = multigrid.Multigrid(system.c, apply_operator)
    return [
        {
            "shape": list(level.coefficients.a_p.shape),
            "active": int(level.active.sum()),
            "eigenvalue": level.eigenvalue,
            "weight": level.weight,
        }
        for level in hierarchy.levels
    ]
