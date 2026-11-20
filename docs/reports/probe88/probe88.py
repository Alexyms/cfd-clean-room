"""Prompt 88's probe: the reference's own implicit_step calls, captured and replayed.

    python docs/reports/probe88/probe88.py capture-march dust80
    python docs/reports/probe88/probe88.py capture-march dust320
    python docs/reports/probe88/probe88.py capture-flow
    python docs/reports/probe88/probe88.py replay
    python docs/reports/probe88/probe88.py timing --run 1
    python docs/reports/probe88/probe88.py timing-parts --run 1
    python docs/reports/probe88/probe88.py parity
    python docs/reports/probe88/probe88.py ctest

Run under WSL2 with the Linux environment (``~/.venvs/cfd_clean_room``) from
the checkout's root, one BLAS thread (``OPENBLAS_NUM_THREADS=1``). Nothing in
``src/`` changes: the probe wraps ``implicit_step`` where its two callers
imported it (``src.solver_transport`` and ``src.turbulence``), and the wrapper
calls the reference, counts the leftover's sums it forms, and keeps a copy of
every input and output.

- ``capture-march``: the 80x30 dust release (``results/builder81/placement``)
  or the 320x120 one (``results/builder81/product``), run from its stored flow
  state (``flows_from``) to the release onset and ten steps beyond it; every
  class's solve in those ten steps is kept (Jacobi, with ADR-016's bound). The
  run file is written on the Linux file system and copied into
  ``results/builder88/runs/`` (ADR-017 A).
- ``capture-flow``: the 80x30 dust release's flow solve, warm-started from its
  stored state; the k and eps solves of its first twenty outer iterations (CG,
  held cells for eps, a per-cell step).
- ``replay``: every captured call through ``_cfdcore.implicit_step``. A Jacobi
  call is compared bit for bit in every output; a CG call by prompt 88's rule
  (the count, and the field's largest difference over the reference's largest
  magnitude), and again against the reference's CG with the core's dot
  products, bit for bit. The reference is called again on each captured input,
  a check that the capture is what the reference gives, and without the bound,
  to count the calls whose bound added sweeps.
- ``timing``: the core's and the reference's Jacobi solve per call on the
  stored 320x120 run's last frame of each class (prediction (c)), with the
  sweeps per call. ``--run N`` names the record, ``timing_N.json``.
- ``timing-parts``: where the time of class 0's developed solve goes: no
  sweep, the full solve and an empty call, for the core and the reference.
  ``--run N`` names the record, ``timing_parts_N.json``.
- ``parity``: the parity tests with a JUnit record, for the report's tables.
- ``ctest``: the main build's CTest record copied into the records.

The captures stay in ``results/builder88/captures/`` (not committed).
"""

import argparse
import contextlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "probe87"))

import probe87  # noqa: E402 -- after sys.path; its scenario readers and sources

import src.scalar_scheme as scalar_scheme  # noqa: E402 -- after sys.path
import src.solver_transport as solver_transport  # noqa: E402 -- after sys.path
import src.turbulence as turbulence  # noqa: E402 -- after sys.path
from src.boundary_staggered import StaggeredBoundary  # noqa: E402 -- after sys.path
from src.io_manager import RunFile  # noqa: E402 -- after sys.path
from src.mesh import SOLID, Mesh  # noqa: E402 -- after sys.path
from src.scalar_scheme import (  # noqa: E402 -- after sys.path
    ImplicitResult,
    implicit_step,
)
from src.solver_staggered import StaggeredSolver  # noqa: E402 -- after sys.path
from src.time_integration import MarchStep, run  # noqa: E402 -- after sys.path

RESULTS = ROOT / "results" / "builder88"
CAPTURES = RESULTS / "captures"
LINUX_RUNS = Path.home() / "scratch88" / "runs"
# The release's onset in both scenario files, s, and the steps kept after it.
ONSET = probe87.ONSET
STEPS_AFTER_ONSET = 10
# The flow solve's outer iterations whose k and eps solves are kept; one
# turbulence step per outer iteration solves k, then eps.
FLOW_OUTER = 20
SOLVES_PER_OUTER = 2
# Prompt 88's rule for a CG solve whose count matches the reference's.
CG_FIELD_RTOL = 1e-12
# The timing's repeats: the median of these many calls, after WARMUP unrecorded.
CORE_REPEATS = 40
REFERENCE_REPEATS = 10
WARMUP = 2
# The arrays and scalars of one implicit_step call, as the capture keeps them.
INPUTS = (
    "c_star",
    "volume",
    "dt",
    "g_u",
    "g_v",
    "diagonal",
    "solid",
    "tol",
    "max_iter",
)


class _WindowClosedError(Exception):
    """Raised once the capture window has closed, to end the run there."""


@contextlib.contextmanager
def _counting_forms() -> Iterator[list[int]]:
    """Count the reference's leftover sums (``_Acceptance._form``) while the block runs."""
    count = [0]
    form = scalar_scheme._Acceptance._form

    def counted(self: object, residual: np.ndarray) -> float:
        count[0] += 1
        return form(self, residual)

    scalar_scheme._Acceptance._form = counted
    try:
        yield count
    finally:
        scalar_scheme._Acceptance._form = form


class Recorder:
    """A stand-in for implicit_step that calls the reference and keeps the call."""

    def __init__(
        self,
        limit: int | None = None,
        labeller: Callable[[int], dict[str, int]] | None = None,
    ) -> None:
        self.active = False
        self.limit = limit
        self.labeller = labeller
        self.calls: list[dict[str, np.ndarray]] = []
        self.label: dict[str, int] = {}

    def __call__(
        self,
        c_star: np.ndarray,
        volume: np.ndarray,
        dt: float | np.ndarray,
        g_u: np.ndarray,
        g_v: np.ndarray,
        diagonal: np.ndarray,
        solid: np.ndarray,
        tol: float,
        max_iter: int,
        held: np.ndarray | None = None,
        method: str = scalar_scheme.JACOBI,
        leftover_bound: float | None = None,
    ) -> ImplicitResult:
        args = (c_star, volume, dt, g_u, g_v, diagonal, solid, tol, max_iter)
        if not self.active:
            return implicit_step(
                *args, held=held, method=method, leftover_bound=leftover_bound
            )
        if self.limit is not None and len(self.calls) >= self.limit:
            raise _WindowClosedError
        if self.labeller is not None:
            self.label = self.labeller(len(self.calls))
        kept = {
            name: np.array(value, copy=True)
            for name, value in zip(INPUTS, args, strict=True)
        }
        with _counting_forms() as forms:
            result = implicit_step(
                *args, held=held, method=method, leftover_bound=leftover_bound
            )
        call = {
            **kept,
            "method": np.array(method),
            "has_bound": np.array(leftover_bound is not None),
            "leftover_bound": np.array(
                np.nan if leftover_bound is None else leftover_bound
            ),
            "out_field": np.array(result.field, copy=True),
            "out_iterations": np.array(result.iterations),
            "out_converged": np.array(result.converged),
            "out_fallbacks": np.array(result.fallbacks),
            "out_fallback_sweeps": np.array(result.fallback_sweeps),
            "out_leftover": np.array(result.leftover),
            "out_forms": np.array(forms[0]),
            **{k: np.array(v) for k, v in self.label.items()},
        }
        if held is not None:
            call["held"] = np.array(held, copy=True)
        self.calls.append(call)
        return result


def save(name: str, calls: list[dict[str, np.ndarray]], meta: dict) -> None:
    """The captured calls as one .npz, and their description as JSON."""
    CAPTURES.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{i}/{key}": value
        for i, call in enumerate(calls)
        for key, value in call.items()
    }
    np.savez(CAPTURES / f"{name}.npz", **arrays)
    meta = {**meta, "calls": len(calls)}
    (CAPTURES / f"{name}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{name}: {len(calls)} calls kept")


def load(name: str) -> tuple[list[dict[str, np.ndarray]], dict]:
    """The captured calls of save(name, ...)."""
    meta = json.loads((CAPTURES / f"{name}.json").read_text(encoding="utf-8"))
    with np.load(CAPTURES / f"{name}.npz") as data:
        calls: list[dict[str, np.ndarray]] = [{} for _ in range(meta["calls"])]
        for key in data.files:
            i, field = key.split("/")
            calls[int(i)][field] = data[key]
    return calls, meta


def capture_march(name: str) -> None:
    """Every class's solve in the first ten steps after the release's onset."""
    config = probe87.scenario_config(name)
    mesh = Mesh(config)
    recorder = Recorder()
    solver_transport.implicit_step = recorder
    original = solver_transport.TransportSolver.solve_timestep
    state = {"after_onset": 0}

    def labelled(self: object, *args: object, **kwargs: object) -> object:
        k = kwargs["size_class"] if "size_class" in kwargs else args[2]
        recorder.label = {"class": int(k), "step": state["after_onset"]}
        return original(self, *args, **kwargs)

    solver_transport.TransportSolver.solve_timestep = labelled

    def on_step(step: MarchStep) -> None:
        if recorder.active:
            state["after_onset"] += 1
            if state["after_onset"] == STEPS_AFTER_ONSET:
                raise _WindowClosedError
        # The step ending at the onset is the last before it; the next starts the release.
        if step.t >= ONSET:
            recorder.active = True

    LINUX_RUNS.mkdir(parents=True, exist_ok=True)
    run_file = LINUX_RUNS / f"{name}.h5"
    for path in (run_file, run_file.with_suffix(".checkpoint.h5")):
        path.unlink(missing_ok=True)
    started = time.perf_counter()
    with contextlib.suppress(_WindowClosedError):
        run(
            config,
            run_file,
            flows_from=probe87.SOURCES[name].with_suffix(".h5"),
            on_step=on_step,
        )
    wall = time.perf_counter() - started
    (RESULTS / "runs").mkdir(parents=True, exist_ok=True)
    for path in (run_file, run_file.with_suffix(".checkpoint.h5")):
        if path.exists():
            # A plain copy: the C: mount refuses copy2's file times (ADR-017 A).
            shutil.copyfile(path, RESULTS / "runs" / path.name)
    save(
        name,
        recorder.calls,
        {
            "kind": "march",
            "config": str(probe87.CONFIGS[name].relative_to(ROOT)),
            "flows_from": str(
                probe87.SOURCES[name].with_suffix(".h5").relative_to(ROOT)
            ),
            "shape": list(mesh.cell_type.shape),
            "onset": ONSET,
            "steps_after_onset": STEPS_AFTER_ONSET,
            "wall_seconds": wall,
        },
    )


def capture_flow() -> None:
    """The k and eps solves of the 80x30 flow solve's first outer iterations, warm-started."""
    name = "dust80"
    config = probe87.scenario_config(name)
    mesh = Mesh(config)
    stored = RunFile(probe87.SOURCES[name].with_suffix(".h5"))
    (fingerprint,) = stored.flow_fingerprints()
    initial = stored.read_flow_state(fingerprint).state
    recorder = Recorder(
        limit=FLOW_OUTER * SOLVES_PER_OUTER,
        labeller=lambda n: {
            "outer": n // SOLVES_PER_OUTER,
            "field": n % SOLVES_PER_OUTER,
        },
    )
    recorder.active = True
    turbulence.implicit_step = recorder
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    started = time.perf_counter()
    with contextlib.suppress(_WindowClosedError):
        solver.solve_steady(initial=initial)
    wall = time.perf_counter() - started
    save(
        "flow80",
        recorder.calls,
        {
            "kind": "flow",
            "config": str(probe87.CONFIGS[name].relative_to(ROOT)),
            "warm_from": str(
                probe87.SOURCES[name].with_suffix(".h5").relative_to(ROOT)
            ),
            "fingerprint": fingerprint,
            "shape": list(mesh.cell_type.shape),
            "outer_iterations": FLOW_OUTER,
            "wall_seconds": wall,
        },
    )


CAPTURE_MESHES = {"dust80": "dust80", "dust320": "dust320", "flow80": "dust80"}


def core_scratch(name: str) -> tuple[Mesh, object]:
    """The mesh of a capture and the core's implicit scratch on it."""
    import _cfdcore

    mesh = Mesh(probe87.scenario_config(CAPTURE_MESHES[name]))
    grid = _cfdcore.Grid2(mesh.x, mesh.xc, mesh.y, mesh.yc, mesh.cell_type, SOLID)
    return mesh, _cfdcore.ImplicitScratch2(grid)


def arguments_of(call: dict[str, np.ndarray]) -> tuple[tuple, dict]:
    """A captured call's positional and keyword arguments, as its caller passed them."""
    dt = call["dt"]
    args = (
        call["c_star"],
        call["volume"],
        float(dt) if dt.ndim == 0 else dt,
        call["g_u"],
        call["g_v"],
        call["diagonal"],
        call["solid"],
        float(call["tol"]),
        int(call["max_iter"]),
    )
    kwargs = {
        "held": call.get("held"),
        "method": str(call["method"]),
        "leftover_bound": float(call["leftover_bound"])
        if bool(call["has_bound"])
        else None,
    }
    return args, kwargs


def core_call(
    scratch: object, call: dict[str, np.ndarray]
) -> tuple[np.ndarray, object]:
    """The core's solve of a captured call: its field and its result."""
    import _cfdcore

    args, kwargs = arguments_of(call)
    out = np.full(call["c_star"].shape, np.nan)
    result = _cfdcore.implicit_step(scratch, *args, out, **kwargs)
    return out, result


def bits_differ(a: np.ndarray, b: np.ndarray) -> int:
    """The number of elements whose 64-bit patterns differ."""
    return int(
        np.count_nonzero(
            np.ascontiguousarray(a).view(np.uint64)
            != np.ascontiguousarray(b).view(np.uint64)
        )
    )


def same_bits(x: float, y: float) -> bool:
    """Whether two floats have the same 64-bit pattern."""
    return bits_differ(np.array([x]), np.array([y])) == 0


class _NumpyWithTheCoresDot:
    """NumPy, with ``vdot`` the core's row-blocked dot product (``_cfdcore.row_dot``)."""

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)

    @staticmethod
    def vdot(a: np.ndarray, b: np.ndarray) -> float:
        """The core's dot product of two [ny, nx] arrays, a block per row."""
        import _cfdcore

        return _cfdcore.row_dot(a, b)


def _with_the_cores_dot(fn: Callable[[], ImplicitResult]) -> ImplicitResult:
    """The reference's solve with np.vdot replaced by the core's dot product."""
    saved = scalar_scheme.np
    scalar_scheme.np = _NumpyWithTheCoresDot()
    try:
        return fn()
    finally:
        scalar_scheme.np = saved


def replay() -> None:
    """Every captured call through the core, by the comparison its method takes."""
    rows = []
    for name in ("dust80", "dust320", "flow80"):
        calls, meta = load(name)
        _, scratch = core_scratch(name)
        for i, call in enumerate(calls):
            args, kwargs = arguments_of(call)
            out, core = core_call(scratch, call)
            again = implicit_step(*args, **kwargs)
            row: dict[str, object] = {
                "capture": name,
                "call": i,
                "method": kwargs["method"],
                "label": {
                    k: int(call[k])
                    for k in ("class", "step", "outer", "field")
                    if k in call
                },
                "cells": int(out.size),
                "nonzero_cells": int(np.count_nonzero(call["c_star"])),
                "held_cells": int(call["held"].sum()) if "held" in call else 0,
                "per_cell_dt": bool(call["dt"].ndim > 0),
                "has_bound": bool(call["has_bound"]),
                "reference_again_differs": bits_differ(again.field, call["out_field"])
                + int(again.iterations != int(call["out_iterations"]))
                + int(not same_bits(again.leftover, float(call["out_leftover"]))),
                "reference_iterations": int(call["out_iterations"]),
                "core_iterations": int(core.iterations),
                "reference_converged": bool(call["out_converged"]),
                "core_converged": bool(core.converged),
                "reference_fallbacks": int(call["out_fallbacks"]),
                "core_fallbacks": int(core.fallbacks),
                "reference_fallback_sweeps": int(call["out_fallback_sweeps"]),
                "core_fallback_sweeps": int(core.fallback_sweeps),
                "reference_leftover": float(call["out_leftover"]),
                "core_leftover": float(core.leftover),
                "leftover_same_bits": same_bits(
                    core.leftover, float(call["out_leftover"])
                ),
                "reference_forms": int(call["out_forms"]),
                "core_forms": int(core.leftover_sums),
                "core_restarts": int(core.restarts),
                "field_cells_differing": bits_differ(out, call["out_field"]),
                "field_relative_difference": float(
                    np.abs(out - call["out_field"]).max()
                )
                / max(float(np.abs(call["out_field"]).max()), np.finfo(float).tiny),
            }
            if kwargs["method"] == scalar_scheme.JACOBI:
                if kwargs["leftover_bound"] is not None:
                    free = implicit_step(*args, **{**kwargs, "leftover_bound": None})
                    row["sweeps_without_bound"] = int(free.iterations)
            else:
                with_dot = _with_the_cores_dot(
                    lambda a=args, k=kwargs: implicit_step(*a, **k)
                )
                row["with_cores_dot_bits_differ"] = (
                    bits_differ(with_dot.field, out)
                    + int(with_dot.iterations != core.iterations)
                    + int(with_dot.converged != core.converged)
                    + int(with_dot.fallbacks != core.fallbacks)
                    + int(not same_bits(with_dot.leftover, core.leftover))
                )
                # The leftover's rounding scale, ADR-016's floor sum_P dt a_P C_P:
                # a CG leftover is a cancelling sum of residuals, and the two
                # solves' residuals differ by the fields' last bits times a_P.
                dt = args[2]
                over_dt = args[1] / dt
                a_p = (
                    over_dt
                    + args[3][:, :-1]
                    + args[3][:, 1:]
                    + args[4][:-1, :]
                    + args[4][1:, :]
                    + args[5]
                )
                dt_cells = np.where(args[6], 0.0, dt)
                row["leftover_scale"] = float(
                    np.sum(dt_cells * a_p * np.abs(call["out_field"]))
                )
            rows.append(row)
        print(name, meta["calls"], "calls replayed")
    (RESULTS / "replay.json").write_text(
        json.dumps(rows, indent=1) + "\n", encoding="utf-8"
    )


def _median_seconds(
    fn: Callable[[], object], repeats: int
) -> tuple[float, list[float]]:
    """The median wall time of repeats calls of fn, after WARMUP unrecorded calls."""
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append(time.perf_counter() - started)
    return statistics.median(times), times


def developed_inputs() -> list[dict[str, np.ndarray]]:
    """Each class's last captured 320x120 call with the stored run's last frame as C*.

    The bound is ADR-016's share, ``leftover_share * conservation_limit *
    S * dt / t_end``, with S the class's supply (initial, inflow and source)
    booked at the stored run's last frame.
    """
    import h5py

    calls, _ = load("dust320")
    config = probe87.scenario_config("dust320")
    spec = config.transport
    with h5py.File(probe87.STORED_RUN, "r") as stored:
        terms = [str(t) for t in stored["transport/budgets"].attrs["terms"]]
        last = stored["transport/budgets"][-1]
        classes = 1 + max(int(c["class"]) for c in calls)
        frames = [
            np.ascontiguousarray(stored[f"transport/C_{k}"][-1], dtype=np.float64)
            for k in range(classes)
        ]
    inputs = []
    for k, frame in enumerate(frames):
        call = dict([c for c in calls if int(c["class"]) == k][-1])
        supplied = sum(
            float(last[k][terms.index(t)]) for t in ("initial", "inflow", "source")
        )
        dt = float(call["dt"])
        bound = (
            spec.leftover_share * spec.conservation_limit * supplied * dt / config.t_end
        )
        call.update(
            c_star=frame,
            has_bound=np.array(True),
            leftover_bound=np.array(bound),
            supplied=np.array(supplied),
            captured_bound=call["leftover_bound"],
        )
        inputs.append(call)
    return inputs


def timing(run_number: int) -> None:
    """The core's Jacobi solve against the reference's per call on each developed frame."""
    import _cfdcore

    _, scratch = core_scratch("dust320")
    rows = []
    for call in developed_inputs():
        args, kwargs = arguments_of(call)
        out, core = core_call(scratch, call)
        reference = implicit_step(*args, **kwargs)
        out_buffer = np.empty(call["c_star"].shape)

        def core_once(
            a: tuple = args, k: dict = kwargs, o: np.ndarray = out_buffer
        ) -> object:
            return _cfdcore.implicit_step(scratch, *a, o, **k)

        core_s, core_all = _median_seconds(core_once, CORE_REPEATS)
        ref_s, ref_all = _median_seconds(
            lambda a=args, k=kwargs: implicit_step(*a, **k), REFERENCE_REPEATS
        )
        rows.append(
            {
                "class": int(call["class"]),
                "nonzero_cells": int(np.count_nonzero(call["c_star"])),
                "sweeps": int(reference.iterations),
                "converged": bool(reference.converged),
                "core_bits_differ": bits_differ(out, reference.field)
                + int(core.iterations != reference.iterations)
                + int(not same_bits(core.leftover, reference.leftover)),
                "bound": float(call["leftover_bound"]),
                "captured_bound": float(call["captured_bound"]),
                "supplied": float(call["supplied"]),
                "core_median_s": core_s,
                "core_min_s": min(core_all),
                "reference_median_s": ref_s,
                "reference_min_s": min(ref_all),
                "core_repeats": CORE_REPEATS,
                "reference_repeats": REFERENCE_REPEATS,
                "ratio": ref_s / core_s,
            }
        )
        print(rows[-1])
    record = {
        "rows": rows,
        "stored_run": str(probe87.STORED_RUN.relative_to(ROOT)),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    (RESULTS / f"timing_{run_number}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )


def timing_parts(run_number: int) -> None:
    """Where the Jacobi solve's time goes, on class 0's developed frame.

    Each of the core and the reference is timed with the cap at 0 (the set-up,
    one residual and the leftover: no sweep), at the cap that the full solve
    takes (its sweeps), and on an empty call (no conductance and no sink, the
    early return: the call and its argument checks). A sweep's cost is the
    difference of the first two over the sweeps.
    """
    import _cfdcore

    _, scratch = core_scratch("dust320")
    call = developed_inputs()[0]
    args, kwargs = arguments_of(call)
    sweeps = implicit_step(*args, **kwargs).iterations
    empty = dict(call)
    empty.update(
        g_u=np.zeros_like(call["g_u"]),
        g_v=np.zeros_like(call["g_v"]),
        diagonal=np.zeros_like(call["diagonal"]),
    )
    out = np.empty(call["c_star"].shape)
    rows = []
    for label, variant, cap in (
        ("no sweep", call, 0),
        ("the full solve", call, sweeps),
        ("nothing to solve", empty, sweeps),
    ):
        a, k = arguments_of({**variant, "max_iter": np.array(cap)})
        core_s, _ = _median_seconds(
            lambda a=a, k=k: _cfdcore.implicit_step(scratch, *a, out, **k), CORE_REPEATS
        )
        ref_s, _ = _median_seconds(
            lambda a=a, k=k: implicit_step(*a, **k), REFERENCE_REPEATS
        )
        rows.append(
            {
                "part": label,
                "cap": cap,
                "core_median_s": core_s,
                "reference_median_s": ref_s,
            }
        )
        print(rows[-1])
    record = {
        "rows": rows,
        "sweeps": sweeps,
        "class": int(call["class"]),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }
    (RESULTS / f"timing_parts_{run_number}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )


def parity() -> None:
    """The parity tests with a JUnit record, the core required."""
    env = {**os.environ, "CFD_CORE_REQUIRED": "1"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_core_parity.py",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={RESULTS / 'parity.xml'}",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )


def ctest() -> None:
    """The main build's CTest record (scripts/build_core.sh writes it) into the records."""
    build = Path(os.environ.get("CFD_CORE_BUILD", Path.home() / "build" / "cfd-core"))
    shutil.copyfile(build / "ctest.xml", RESULTS / "ctest.xml")


def main(argv: list[str] | None = None) -> int:
    """Run one of the probe's commands."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=[
            "capture-march",
            "capture-flow",
            "replay",
            "timing",
            "timing-parts",
            "parity",
            "ctest",
        ],
    )
    parser.add_argument("name", nargs="?", choices=list(probe87.SOURCES))
    parser.add_argument("--run", type=int, default=1, help="timing's record")
    args = parser.parse_args(argv)
    if args.command == "capture-march":
        if args.name is None:
            parser.error("capture-march needs dust80 or dust320")
        capture_march(args.name)
    elif args.command == "capture-flow":
        capture_flow()
    elif args.command == "replay":
        replay()
    elif args.command == "timing":
        timing(args.run)
    elif args.command == "timing-parts":
        timing_parts(args.run)
    elif args.command == "parity":
        parity()
    else:
        ctest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
