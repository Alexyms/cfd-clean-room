"""Split the budget residual's increment into the parts of a transport step (prompt 77).

    python docs/reports/probe77/probe77.py RUN [--concurrent N]
    python docs/reports/probe77/probe77.py --calibrate

RUN is a name of configs77.RUNS or RERUNS. The run goes through
scripts/run_scenario.py's ``main``, in this process, with its configuration
from configs77.py and its flow states from the source run (--flows-from).
Before it starts, this module wraps three names as ``src.solver_transport``
sees them: ``TransportSolver.solve_timestep``, ``advective_flux`` and
``implicit_step``. The wrappers call the originals with the arguments they were
given and hand back what the originals returned; they read the arrays and
change none, so the field's arithmetic is the solver's. ``--calibrate`` shows
it: a 2,000-step window of the 80x30 dust release (R1's configuration) run
unwrapped and then wrapped in this one process, compared at every step.

Per step and class the wrapper of ``solve_timestep`` books the increment of the
budget's residual ``R = initial + inflow + source - outflow - deposited -
current`` to these terms, every total being ``MassBudget.in_domain`` (the sum
of C V over the non-SOLID cells) of the field named:

- T1, between steps: the budget's ``current`` before the step less the total of
  the field handed in, after the start-of-step SOLID zeroing.
- T2, advection: the booked inflow less outflow of the step (the budget's
  increments) less the change in total from the advection update. The update's
  field, ``c - dt * divergence / V``, is rebuilt from the two face-flux arrays
  ``advective_flux`` returned; the rebuild plus the source, SOLID cells zeroed,
  is compared bit for bit with the field the solver hands ``implicit_step``
  (column ``rebuilt``), so the field measured is the solver's.
- T3, the SOLID zeroing after advection: its change to the total, which the
  budget's non-SOLID sum makes zero by construction; the particles it removes
  are kept apart (``solid_removed``).
- T4, source: the booked source increment less the source's change in total.
- T5, the implicit solve's leftover: ``dt`` times the sum over the non-SOLID
  cells of the final residual ``b + sum_f G_f C_N - a_P C`` of the system
  ``implicit_step`` was given, at the field it returned.
- T6, deposition booking: the implicit stage's increment (less its change in
  total, less the booked deposition) less T5.
- gap: the residual's increment less T1 to T6.

Also per step and class: the Jacobi sweeps, whether the solve converged, the
final residual's largest magnitude over its stop, and T5's bound ``dt`` times
the non-SOLID cell count times the stop. The series go to
results/builder77/RUN.steps.npz, and the terms accumulated to every stored
frame (every ``output_interval`` steps and the last) to RUN.frames.npz.
"""

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import configs77  # noqa: E402 -- follows sys.path.insert

import src.solver_transport as solver_transport  # noqa: E402 -- follows sys.path.insert
from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.scalar_scheme import (  # noqa: E402 -- follows sys.path.insert
    JACOBI,
    _neighbour_sum,
)
from src.solver_transport import MassBudget  # noqa: E402 -- follows sys.path.insert
from src.time_integration import MarchStep, run  # noqa: E402 -- follows sys.path.insert

RESULTS = ROOT / "results" / "builder77"
CALIBRATION_STEPS = 2000
PROGRESS_EVERY = 1000
COLUMNS: tuple[str, ...] = (
    "t",
    "dt",
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "gap",
    "dR",
    "R",
    "content",
    "sweeps",
    "converged",
    "res_over_stop",
    "bound",
    "solid_removed",
    "rebuilt",
)
COLUMN = {name: index for index, name in enumerate(COLUMNS)}

logger = logging.getLogger("probe77")

_ORIGINAL_STEP = solver_transport.TransportSolver.solve_timestep
_ORIGINAL_ADVECTIVE = solver_transport.advective_flux
_ORIGINAL_IMPLICIT = solver_transport.implicit_step


@dataclass
class _Capture:
    """What the inner wrappers saw during one ``solve_timestep`` call."""

    advective: list[np.ndarray] = field(default_factory=list)
    implicit: tuple | None = None


@dataclass
class Recorder:
    """The per-step terms of every class, filled by the wrappers.

    Parameters
    ----------
    rows : dict[int, list[list[float]]]
        Class to its rows, one per step, in COLUMNS order.
    """

    rows: dict[int, list[list[float]]] = field(default_factory=dict)
    capture: _Capture | None = None
    started: float = field(default_factory=time.perf_counter)

    def steps(self, k: int) -> int:
        """Steps booked for class k."""
        return len(self.rows.get(k, ()))

    def array(self, k: int) -> np.ndarray:
        """Class k's rows, [steps, len(COLUMNS)]."""
        return np.array(self.rows.get(k, []), dtype=np.float64).reshape(
            -1, len(COLUMNS)
        )


RECORDER = Recorder()


def _snapshot(
    budget: MassBudget,
) -> tuple[float | None, float, float, float, float, float | None, float | None]:
    """(initial, inflow, outflow, source, deposited, current, residual) of a budget."""
    residual = None
    if budget.initial is not None and budget.current is not None:
        residual = budget.residual()
    return (
        budget.initial,
        budget.inflow,
        budget.outflow,
        budget.source,
        sum(budget.deposited.values()),
        budget.current,
        residual,
    )


def advective_flux_observed(*args: object, **kwargs: object) -> np.ndarray:
    """``scalar_scheme.advective_flux``, its result kept for the step's terms."""
    out = _ORIGINAL_ADVECTIVE(*args, **kwargs)
    if RECORDER.capture is not None:
        RECORDER.capture.advective.append(out)
    return out


def implicit_step_observed(*args: object, **kwargs: object) -> object:
    """``scalar_scheme.implicit_step``, its arguments and result kept."""
    result = _ORIGINAL_IMPLICIT(*args, **kwargs)
    if RECORDER.capture is not None:
        RECORDER.capture.implicit = (args, kwargs, result)
    return result


def solve_timestep_observed(
    self: solver_transport.TransportSolver,
    C_k: np.ndarray,  # noqa: N803 -- the solver's argument name
    faces: object,
    size_class: int,
    dt: float,
    v_ext: object = None,
    sources: np.ndarray | None = None,
    *,
    eddy_viscosity: np.ndarray | None = None,
) -> np.ndarray:
    """``TransportSolver.solve_timestep``, with the step's terms booked to RECORDER."""
    k = int(size_class)
    before = _snapshot(self.budget[k])
    RECORDER.capture = _Capture()
    try:
        out = _ORIGINAL_STEP(
            self,
            C_k,
            faces,
            size_class,
            dt,
            v_ext,
            sources,
            eddy_viscosity=eddy_viscosity,
        )
        capture = RECORDER.capture
    finally:
        RECORDER.capture = None
    after = _snapshot(self.budget[k])
    RECORDER.rows.setdefault(k, []).append(
        _terms(self, C_k, sources, float(dt), before, after, capture)
    )
    n = RECORDER.steps(k)
    if k == 0 and n % PROGRESS_EVERY == 0:
        logger.info(
            "probe77: step %d, %.1f s wall", n, time.perf_counter() - RECORDER.started
        )
    return out


def _terms(
    solver: solver_transport.TransportSolver,
    C_k: np.ndarray,  # noqa: N803 -- the solver's argument name
    sources: np.ndarray | None,
    dt: float,
    before: tuple,
    after: tuple,
    capture: _Capture,
) -> list[float]:
    """One row of COLUMNS for one step of one class."""
    mesh = solver._mesh
    solid = solver._solid
    volume = solver._volume
    total = MassBudget.in_domain
    # The field the step starts from, as the solver forms it.
    c_in = np.array(C_k, dtype=np.float64, order="C", copy=True)
    c_in[solid] = 0.0
    content_in = total(c_in, mesh)
    if len(capture.advective) != 2 or capture.implicit is None:
        raise RuntimeError(
            "the step did not call advective_flux twice and implicit_step once"
        )
    adv_u = capture.advective[0]
    adv_v = capture.advective[1].T
    divergence = adv_u[:, 1:] - adv_u[:, :-1] + adv_v[1:, :] - adv_v[:-1, :]
    c_adv = c_in - dt * divergence / volume
    c_src = c_adv.copy()
    if sources is not None:
        c_src += dt * np.asarray(sources, dtype=np.float64)
    c_zeroed = c_src.copy()
    solid_removed = float(np.sum((c_src * volume)[solid]))
    c_zeroed[solid] = 0.0
    args, kwargs, result = capture.implicit
    c_star, vol, dt_solve, g_u, g_v, diagonal, solid_mask, tol, _max_iter = args[:9]
    method = kwargs.get("method", JACOBI)
    held = kwargs.get("held")
    if method != JACOBI or held is not None or len(args) > 9:
        raise RuntimeError(
            "the transport step's implicit solve is Jacobi with nothing held"
        )
    rebuilt = bool(np.array_equal(c_zeroed, c_star))
    content_adv = total(c_adv, mesh)
    content_src = total(c_src, mesh)
    content_star = total(c_star, mesh)
    c_new = result.field
    content_new = total(c_new, mesh)
    # The residual of the system the solve was given, at the field it returned,
    # in the order _jacobi forms it.
    over_dt = vol / dt_solve
    diag_extra = g_u[:, :-1] + g_u[:, 1:] + g_v[:-1, :] + g_v[1:, :] + diagonal
    a_p = over_dt + diag_extra
    b = over_dt * c_star
    stop = tol * float(np.abs(b).max())
    residual = b + _neighbour_sum(np.pad(c_new, 1), g_u, g_v) - a_p * c_new
    residual[solid_mask] = 0.0
    largest = float(np.abs(residual).max())
    live = int(np.count_nonzero(~solid_mask))
    t5 = dt * float(np.sum(residual[~solid_mask]))

    _initial0, in0, out0, src0, dep0, current0, r0 = before
    _, in1, out1, src1, dep1, current1, r1 = after
    if current1 != content_new:
        raise RuntimeError(
            "the budget's current is not the total of the returned field"
        )
    d_in, d_out, d_src, d_dep = in1 - in0, out1 - out0, src1 - src0, dep1 - dep0
    t1 = 0.0 if current0 is None else current0 - content_in
    t2 = (d_in - d_out) - (content_adv - content_in)
    t3 = -(content_star - content_src)
    t4 = d_src - (content_src - content_adv)
    stage = -(content_new - content_star) - d_dep
    t6 = stage - t5
    d_r = r1 - (0.0 if r0 is None else r0)
    gap = d_r - (t1 + t2 + t3 + t4 + t5 + t6)
    return [
        0.0,  # t, filled when the series is written
        dt,
        t1,
        t2,
        t3,
        t4,
        t5,
        t6,
        gap,
        d_r,
        r1,
        content_new,
        float(result.iterations),
        1.0 if result.converged else 0.0,
        largest / stop if stop > 0.0 else 0.0,
        dt * live * stop,
        solid_removed,
        1.0 if rebuilt else 0.0,
    ]


def install() -> None:
    """Wrap the three names as src.solver_transport sees them."""
    solver_transport.TransportSolver.solve_timestep = solve_timestep_observed
    solver_transport.advective_flux = advective_flux_observed
    solver_transport.implicit_step = implicit_step_observed


def uninstall() -> None:
    """Put the originals back."""
    solver_transport.TransportSolver.solve_timestep = _ORIGINAL_STEP
    solver_transport.advective_flux = _ORIGINAL_ADVECTIVE
    solver_transport.implicit_step = _ORIGINAL_IMPLICIT


def series(recorder: Recorder) -> dict[str, np.ndarray]:
    """Every class's rows, with t the running sum of the class's steps."""
    out = {}
    for k in sorted(recorder.rows):
        rows = recorder.array(k)
        rows[:, COLUMN["t"]] = np.cumsum(rows[:, COLUMN["dt"]])
        out[f"class_{k}"] = rows
    return out


def frames(rows: dict[str, np.ndarray], every: int) -> dict[str, np.ndarray]:
    """The terms accumulated to every stored frame: step % every == 0 and the last."""
    out: dict[str, np.ndarray] = {}
    for name, table in rows.items():
        n = table.shape[0]
        steps = np.array(sorted({s for s in range(every, n + 1, every)} | {n}))
        cumulative = np.cumsum(table, axis=0)[steps - 1]
        # Per-step quantities are taken at the frame, not summed.
        for column in ("t", "R", "content"):
            cumulative[:, COLUMN[column]] = table[steps - 1, COLUMN[column]]
        out[name] = cumulative
        out[f"{name}_steps"] = steps
    return out


def load_runner() -> object:
    """scripts/run_scenario.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "run_scenario", ROOT / "scripts" / "run_scenario.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def probe(name: str, concurrent: int) -> int:
    """Run one configuration wrapped, and write its series and frames."""
    selected = configs77.BY_NAME[name]
    scenario = configs77.OUT / f"scenario_{name}.yaml"
    out = RESULTS / f"{name}.h5"
    runner = load_runner()
    install()
    started = time.perf_counter()
    try:
        status = runner.main(
            [
                str(scenario),
                "--out",
                str(out),
                "--flows-from",
                str(selected.flows_from),
                "--concurrent",
                str(concurrent),
            ]
        )
    finally:
        uninstall()
    wall = time.perf_counter() - started
    rows = series(RECORDER)
    every = SimConfig(scenario).output_interval
    np.savez(RESULTS / f"{name}.steps.npz", columns=np.array(COLUMNS), **rows)
    np.savez(
        RESULTS / f"{name}.frames.npz", columns=np.array(COLUMNS), **frames(rows, every)
    )
    meta = {
        "run": name,
        "scenario": selected.scenario,
        "grid": configs77.grid_tag(selected.grid),
        "diffusion_tol": selected.tolerance,
        "max_diffusion_iter": selected.sweep_cap,
        "flows_from": str(selected.flows_from.relative_to(ROOT)),
        "exit": status,
        "wall_seconds": wall,
        "steps": {k: RECORDER.steps(k) for k in sorted(RECORDER.rows)},
    }
    (RESULTS / f"{name}.probe.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    logger.info("probe77: %s done in %.1f s, exit %d", name, wall, status)
    return status


class _WindowEndError(Exception):
    """Raised by the calibration's step hook after the window's last step."""


def _window(path: Path, wrapped: bool) -> list[tuple]:
    """The calibration window's fields and budgets, every step."""
    for stale in (path, path.with_suffix(".checkpoint.h5")):
        if stale.exists():
            stale.unlink()
    seen: list[tuple] = []

    def hook(step: MarchStep) -> None:
        row = []
        for k in sorted(step.fields):
            c = step.fields[k]
            b = step.budgets[k]
            row.append(
                (
                    hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest(),
                    b.initial,
                    b.inflow,
                    b.outflow,
                    b.source,
                    tuple(b.deposited.values()),
                    b.current,
                )
            )
        seen.append((step.step, tuple(row)))
        if step.step >= CALIBRATION_STEPS:
            raise _WindowEndError

    config = SimConfig(configs77.OUT / "scenario_R1.yaml")
    if wrapped:
        install()
    try:
        run(config, path, flows_from=configs77.BY_NAME["R1"].flows_from, on_step=hook)
    except _WindowEndError:
        pass
    finally:
        uninstall()
    return seen


def calibrate() -> int:
    """The in-process check: the window unwrapped, then wrapped, equal at every step."""
    folder = RESULTS / "calibration"
    folder.mkdir(parents=True, exist_ok=True)
    plain = _window(folder / "plain.h5", wrapped=False)
    RECORDER.rows.clear()
    wrapped = _window(folder / "wrapped.h5", wrapped=True)
    differing = [a[0] for a, b in zip(plain, wrapped, strict=False) if a != b]
    rows = series(RECORDER)
    rebuilt = all(bool(np.all(r[:, COLUMN["rebuilt"]] == 1.0)) for r in rows.values())
    record = {
        "steps_plain": len(plain),
        "steps_wrapped": len(wrapped),
        "classes": len(plain[0][1]) if plain else 0,
        "steps_differing": len(differing),
        "first_differing": differing[0] if differing else None,
        "bitwise": len(plain) == len(wrapped) == CALIBRATION_STEPS and not differing,
        "rebuilt_every_step": rebuilt,
        "last_step_hashes": [row[0] for row in plain[-1][1]] if plain else [],
    }
    (RESULTS / "calibration.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return 0 if record["bitwise"] and rebuilt else 1


def main() -> int:
    """Probe one run, or calibrate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", choices=sorted(configs77.BY_NAME))
    parser.add_argument("--concurrent", type=int, default=0)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.calibrate:
        return calibrate()
    if args.run is None:
        parser.error("name a run or pass --calibrate")
    return probe(args.run, args.concurrent)


if __name__ == "__main__":
    sys.exit(main())
