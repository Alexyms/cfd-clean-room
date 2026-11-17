"""Prompt 87's probe: the reference's own advective_flux calls, captured and replayed.

    python docs/reports/probe87/probe87.py capture-march dust80
    python docs/reports/probe87/probe87.py capture-march dust320
    python docs/reports/probe87/probe87.py capture-flow
    python docs/reports/probe87/probe87.py replay
    python docs/reports/probe87/probe87.py timing
    python docs/reports/probe87/probe87.py parity

Run under WSL2 with the Linux environment (``~/.venvs/cfd_clean_room``) from
the checkout's root, one BLAS thread (``OPENBLAS_NUM_THREADS=1``). Nothing in
``src/`` changes: the probe wraps ``advective_flux`` where its two callers
imported it (``src.solver_transport`` and ``src.turbulence``), and the wrapper
calls the reference and keeps a copy of every input and of the output, in the
core's layout (the y call's transposed arrays transposed back).

- ``capture-march``: the 80x30 dust release (``results/builder81/placement``)
  or the 320x120 one (``results/builder81/product``), run from its stored flow
  state (``flows_from``) to the release onset and five steps beyond it; every
  class's two calls in those five steps are kept. The run file is written on
  the Linux file system and copied into ``results/builder87/runs/`` (ADR-017 A).
- ``capture-flow``: the 80x30 dust release's flow solve, warm-started from its
  stored state; the k and eps calls of its first five outer iterations.
- ``replay``: every captured call through ``_cfdcore.advective_flux``,
  compared bit for bit with the captured output, and the reference called
  again on the captured inputs (a check that the capture is what the
  reference gives).
- ``timing``: the core's and the reference's ``advective_flux`` on one
  captured 320x120 call per direction, the median of repeated calls, and the
  binding's overhead on an empty call (``advective_flux_overhead``).
- ``parity``: the parity tests with a JUnit record, for the report's tables.

The captures stay in ``results/builder87/captures/`` (not committed).
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
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import src.solver_transport as solver_transport  # noqa: E402 -- after sys.path
import src.turbulence as turbulence  # noqa: E402 -- after sys.path
from src.boundary_staggered import StaggeredBoundary  # noqa: E402 -- after sys.path
from src.config import SimConfig  # noqa: E402 -- after sys.path
from src.io_manager import RunFile  # noqa: E402 -- after sys.path
from src.mesh import SOLID, Mesh  # noqa: E402 -- after sys.path
from src.scalar_scheme import advective_flux, mesh_axes  # noqa: E402 -- after sys.path
from src.solver_staggered import StaggeredSolver  # noqa: E402 -- after sys.path
from src.time_integration import MarchStep, run  # noqa: E402 -- after sys.path

RESULTS = ROOT / "results" / "builder87"
CAPTURES = RESULTS / "captures"
LINUX_RUNS = Path.home() / "scratch87" / "runs"
# The stored runs whose flow states the captures start from (prompt 81's).
SOURCES = {
    "dust80": ROOT / "results" / "builder81" / "placement" / "dust",
    "dust320": ROOT / "results" / "builder81" / "product" / "equipment_dust",
}
CONFIGS = {
    "dust80": ROOT
    / "results"
    / "builder81"
    / "placement"
    / "configs"
    / "scenario_dust.yaml",
    "dust320": ROOT
    / "results"
    / "builder81"
    / "product"
    / "configs"
    / "scenario_equipment_dust.yaml",
}
# The release's onset in both scenario files, s, and the steps kept after it.
ONSET = 10.0
STEPS_AFTER_ONSET = 5
# The flow solve's outer iterations whose k and eps calls are kept.
FLOW_OUTER = 5
# One turbulence step per outer iteration advects k and eps along x and y.
CALLS_PER_OUTER = 4
# The timing's repeats: the median of these many calls, after WARMUP unrecorded.
CORE_REPEATS = 400
REFERENCE_REPEATS = 60
WARMUP = 5


class _WindowClosedError(Exception):
    """Raised by on_step once the capture window has closed, to end the run there."""


def local_path(name: str) -> str:
    """A path of this checkout in Windows' or WSL2's form, in the local form."""
    windows = PureWindowsPath("C:\\Users\\tomat\\Projects\\cfd_clean_room")
    linux = PurePosixPath("/mnt/c/Users/tomat/Projects/cfd_clean_room")
    if name.lower().startswith(str(windows).lower() + "\\"):
        return str(ROOT.joinpath(*PureWindowsPath(name[len(str(windows)) + 1 :]).parts))
    if name.startswith(str(linux) + "/"):
        return str(ROOT.joinpath(*PurePosixPath(name[len(str(linux)) + 1 :]).parts))
    return name


_READ_CONFIG = RunFile.read_config


def _read_config_here(run_file: RunFile) -> object:
    """RunFile.read_config with the scenario's base read as the local path.

    The stored runs were made on Windows and name their base file by its
    Windows path; both forms name the same file of this checkout.
    """
    record = _READ_CONFIG(run_file)
    section = record.mapping.get("scenario")
    if isinstance(section, dict) and isinstance(section.get("base"), str):
        section["base"] = local_path(section["base"])
    return record


RunFile.read_config = _read_config_here


def scenario_config(name: str) -> SimConfig:
    """The stored run's scenario file, its base read as the local path."""
    raw = yaml.safe_load(CONFIGS[name].read_text(encoding="utf-8"))
    raw["scenario"]["base"] = local_path(raw["scenario"]["base"])
    return SimConfig.from_dict(raw, CONFIGS[name].parent)


class Recorder:
    """A stand-in for advective_flux that calls the reference and keeps the call."""

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh
        self.active = False
        self.calls: list[dict[str, np.ndarray]] = []
        self.label: dict[str, int] = {}
        ny, nx = mesh.cell_type.shape
        self.x_shape = (ny, nx + 2)

    def __call__(
        self,
        c: np.ndarray,
        flux: np.ndarray,
        inflow: np.ndarray,
        axis: object,
        upwind: bool,
        vents: np.ndarray | None = None,
    ) -> np.ndarray:
        out = advective_flux(c, flux, inflow, axis, upwind, vents)
        if self.active:
            d = 0 if axis.solid_ext.shape == self.x_shape else 1

            def own(a: np.ndarray) -> np.ndarray:
                return np.ascontiguousarray(a if d == 0 else a.T)

            call = {
                "d": np.array(d),
                "upwind": np.array(bool(upwind)),
                "c": own(c),
                "flux": own(flux),
                "inflow": own(inflow),
                "out": own(out),
                **{k: np.array(v) for k, v in self.label.items()},
            }
            if vents is not None:
                call["vents"] = own(vents)
            self.calls.append(call)
        return out


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
    """Every class's two calls in the first five steps after the release's onset."""
    config = scenario_config(name)
    mesh = Mesh(config)
    recorder = Recorder(mesh)
    solver_transport.advective_flux = recorder
    # The class whose step is running, for the record: the transport solver
    # steps the classes in order, two calls each.
    original = solver_transport.TransportSolver.solve_timestep

    def labelled(self: object, *args: object, **kwargs: object) -> object:
        k = kwargs["size_class"] if "size_class" in kwargs else args[2]
        recorder.label = {"class": int(k), "step": state["after_onset"]}
        return original(self, *args, **kwargs)

    solver_transport.TransportSolver.solve_timestep = labelled
    state = {"after_onset": 0}

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
            flows_from=SOURCES[name].with_suffix(".h5"),
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
            "config": str(CONFIGS[name].relative_to(ROOT)),
            "flows_from": str(SOURCES[name].with_suffix(".h5").relative_to(ROOT)),
            "shape": list(mesh.cell_type.shape),
            "onset": ONSET,
            "steps_after_onset": STEPS_AFTER_ONSET,
            "wall_seconds": wall,
        },
    )


def capture_flow() -> None:
    """The k and eps calls of the 80x30 flow solve's first outer iterations, warm-started."""
    name = "dust80"
    config = scenario_config(name)
    mesh = Mesh(config)
    stored = RunFile(SOURCES[name].with_suffix(".h5"))
    (fingerprint,) = stored.flow_fingerprints()
    initial = stored.read_flow_state(fingerprint).state
    recorder = Recorder(mesh)
    recorder.active = True
    turbulence.advective_flux = recorder
    original = turbulence.KEpsilonModel._advect

    def labelled(
        self: object, q: np.ndarray, *args: object, **kwargs: object
    ) -> np.ndarray:
        index = len(recorder.calls) // CALLS_PER_OUTER
        recorder.label = {"outer": index, "field": (len(recorder.calls) // 2) % 2}
        if len(recorder.calls) >= FLOW_OUTER * CALLS_PER_OUTER:
            raise _WindowClosedError
        return original(self, q, *args, **kwargs)

    turbulence.KEpsilonModel._advect = labelled
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
            "config": str(CONFIGS[name].relative_to(ROOT)),
            "warm_from": str(SOURCES[name].with_suffix(".h5").relative_to(ROOT)),
            "fingerprint": fingerprint,
            "shape": list(mesh.cell_type.shape),
            "outer_iterations": FLOW_OUTER,
            "wall_seconds": wall,
        },
    )


CAPTURE_MESHES = {"dust80": "dust80", "dust320": "dust320", "flow80": "dust80"}


def core_scheme(name: str) -> tuple[Mesh, object]:
    """The mesh of a capture and the core's scheme on it."""
    import _cfdcore

    mesh = Mesh(scenario_config(CAPTURE_MESHES[name]))
    grid = _cfdcore.Grid2(mesh.x, mesh.xc, mesh.y, mesh.yc, mesh.cell_type, SOLID)
    return mesh, _cfdcore.ScalarScheme2(grid)


def reference_call(mesh: Mesh, call: dict[str, np.ndarray]) -> np.ndarray:
    """The reference on a captured call's inputs, as its callers call it."""
    axis_x, axis_y = mesh_axes(mesh)
    d, upwind = int(call["d"]), bool(call["upwind"])
    vents = call.get("vents")
    if d == 0:
        return advective_flux(
            call["c"], call["flux"], call["inflow"], axis_x, upwind, vents
        )
    v = None if vents is None else np.ascontiguousarray(vents.T)
    return advective_flux(
        call["c"].T, call["flux"].T, call["inflow"].T, axis_y, upwind, v
    ).T


def bits_differ(a: np.ndarray, b: np.ndarray) -> int:
    """The number of elements whose 64-bit patterns differ."""
    return int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))


def replay() -> None:
    """Every captured call through the core, bit for bit against the captured output."""
    import _cfdcore

    rows = []
    for name in ("dust80", "dust320", "flow80"):
        calls, meta = load(name)
        mesh, scheme = core_scheme(name)
        for i, call in enumerate(calls):
            out = np.full(call["flux"].shape, np.nan)
            _cfdcore.advective_flux(
                scheme,
                int(call["d"]),
                call["c"],
                call["flux"],
                call["inflow"],
                bool(call["upwind"]),
                out,
                call.get("vents"),
            )
            again = reference_call(mesh, call)
            c = call["c"]
            rows.append(
                {
                    "capture": name,
                    "call": i,
                    "d": int(call["d"]),
                    "upwind": bool(call["upwind"]),
                    "vents": "vents" in call,
                    "label": {
                        k: int(call[k])
                        for k in ("class", "step", "outer", "field")
                        if k in call
                    },
                    "faces": int(out.size),
                    "core_differs": bits_differ(out, call["out"]),
                    "reference_again_differs": bits_differ(again, call["out"]),
                    "nonzero_cells": int(np.count_nonzero(c)),
                    "nonzero_flux_faces": int(np.count_nonzero(out)),
                    "c_max": float(np.abs(c).max()),
                }
            )
        print(name, meta["calls"], "calls replayed")
    (RESULTS / "replay.json").write_text(
        json.dumps(rows, indent=1) + "\n", encoding="utf-8"
    )


def _median_seconds(fn: object, repeats: int) -> tuple[float, list[float]]:
    """The median wall time of repeats calls of fn, after WARMUP unrecorded calls."""
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append(time.perf_counter() - started)
    return statistics.median(times), times


def timing() -> None:
    """The core against the reference on one captured 320x120 call per direction."""
    import _cfdcore

    calls, _ = load("dust320")
    mesh, scheme = core_scheme("dust320")
    axis_x, axis_y = mesh_axes(mesh)
    rows = []
    for d, axis in enumerate((axis_x, axis_y)):
        # The last captured call in this direction: the fifth step, the last class.
        i, call = [(i, c) for i, c in enumerate(calls) if int(c["d"]) == d][-1]
        row = _time_direction(_cfdcore, scheme, axis, call)
        rows.append({"call": i, **row})
        print(rows[-1])
    record = {
        "rows": rows,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    (RESULTS / "timing.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )


def _time_direction(
    core: object, scheme: object, axis: object, call: dict[str, np.ndarray]
) -> dict[str, object]:
    """The medians of the core, its empty call and the reference on one captured call."""
    d = int(call["d"])
    c, flux, inflow = call["c"], call["flux"], call["inflow"]
    upwind = bool(call["upwind"])
    out = np.empty(flux.shape)
    args = (scheme, d, c, flux, inflow, upwind, out, None)

    def reference() -> np.ndarray:
        # As the transport solver calls it: y on the transposed arrays.
        if d == 0:
            return advective_flux(c, flux, inflow, axis, upwind)
        return advective_flux(c.T, flux.T, inflow.T, axis, upwind).T

    core_s, core_all = _median_seconds(lambda: core.advective_flux(*args), CORE_REPEATS)
    empty_s, _ = _median_seconds(
        lambda: core.advective_flux_overhead(*args), CORE_REPEATS
    )
    ref_s, ref_all = _median_seconds(reference, REFERENCE_REPEATS)
    return {
        "d": d,
        "upwind": upwind,
        "faces": int(flux.size),
        "core_median_s": core_s,
        "core_min_s": min(core_all),
        "overhead_median_s": empty_s,
        "reference_median_s": ref_s,
        "reference_min_s": min(ref_all),
        "core_repeats": CORE_REPEATS,
        "reference_repeats": REFERENCE_REPEATS,
        "ratio": ref_s / core_s,
    }


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


def main(argv: list[str] | None = None) -> int:
    """Run one of the probe's commands."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=["capture-march", "capture-flow", "replay", "timing", "parity"],
    )
    parser.add_argument("name", nargs="?", choices=list(SOURCES))
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
        timing()
    else:
        parity()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
