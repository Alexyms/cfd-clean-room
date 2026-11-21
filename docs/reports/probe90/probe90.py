"""Prompt 90's probe: the reference's own class steps, captured and replayed through the core.

    python docs/reports/probe90/probe90.py capture dust80
    python docs/reports/probe90/probe90.py capture door80
    python docs/reports/probe90/probe90.py capture breach80
    python docs/reports/probe90/probe90.py capture vents80
    python docs/reports/probe90/probe90.py capture dust320
    python docs/reports/probe90/probe90.py replay
    python docs/reports/probe90/probe90.py timing --run 1
    python docs/reports/probe90/probe90.py plants

Run under WSL2 with the Linux environment (``~/.venvs/cfd_clean_room``) from
the checkout's root, one BLAS thread (``OPENBLAS_NUM_THREADS=1``), through
``results/builder90/probe.sh``. Nothing in ``src/`` changes: the probe wraps
``TransportSolver.solve_timestep`` and four of its helpers on the class, and
each wrapper calls the reference and keeps a copy of what it was handed and
what it returned.

- ``capture``: one stored scenario run from its stored flow states
  (``flows_from``; prompt 81's runs), or the vents file's short dust release
  at 80x30 with its flow state solved from rest. Two windows are kept, every
  class: the five steps across the onset at 10 s (two before it, three
  after; 1 s for the vents run) and the run's last three steps. A kept call
  holds the step's inputs, the budget before and after, the field returned
  (or the breach raised), the step's records, and the values the reference
  formed inside the step that the core computes once per interval
  (``_advecting_velocities``, ``_face_eddy_viscosity``, the conductances and
  the bound ``_implicit_step`` is handed, ``stable_dt``): the arrays in full
  at a solver's first call per class, and a digest at every call. Each
  interval's solver is kept once: its conditions per class, physics and
  settings. The run file is written on the Linux file system with frames
  only at the end (``output_interval`` raised, which changes no arithmetic of
  the march) and copied into ``results/builder90/runs/``.
- ``replay``: every kept call through the core, every output compared bit for
  bit, and the core's interval quantities against the reference's.
- ``timing``: a class step on the stored 320x120 run's last frame, every
  class, the core's against ``solve_timestep``, in one process, with the
  machine's state.
- ``plants``: each planted defect of ``variants90.py`` built in turn, with
  CTest and the parity module run against it.

The captures stay in ``results/builder90/captures/`` (not committed).
"""

import argparse
import contextlib
import dataclasses
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.config import SimConfig  # noqa: E402 -- after sys.path
from src.io_manager import RunFile  # noqa: E402 -- after sys.path
from src.solver_transport import (  # noqa: E402 -- after sys.path
    DEPOSIT_SURFACES,
    ConservationBreachError,
    MassBudget,
    TransportSolver,
)
from src.time_integration import MarchStep, run  # noqa: E402 -- after sys.path
from tests import core_mapping  # noqa: E402 -- after sys.path

RESULTS = ROOT / "results" / "builder90"
CAPTURES = RESULTS / "captures"
LINUX_RUNS = Path.home() / "scratch90" / "runs"
STORED = ROOT / "results" / "builder81"
# The stored runs whose flow states the captures start from (prompt 81's), each
# with its scenario file; the vents run has none and solves its state from rest.
SCENARIOS: dict[str, dict[str, object]] = {
    "dust80": {
        "config": STORED / "placement" / "configs" / "scenario_dust.yaml",
        "flows_from": STORED / "placement" / "dust.h5",
        "onset": 10.0,
    },
    "door80": {
        "config": STORED / "placement" / "configs" / "scenario_door.yaml",
        "flows_from": STORED / "placement" / "door.h5",
        "onset": 10.0,
    },
    "breach80": {
        "config": STORED / "placement" / "configs" / "scenario_breach.yaml",
        "flows_from": STORED / "placement" / "breach.h5",
        "onset": 10.0,
    },
    "dust320": {
        "config": STORED / "product" / "configs" / "scenario_equipment_dust.yaml",
        "flows_from": STORED / "product" / "equipment_dust.h5",
        "onset": 10.0,
    },
    "vents80": {
        "config": HERE / "scenario_vents80.yaml",
        "flows_from": None,
        "onset": 1.0,
    },
}
# The window across the onset: steps starting from BEFORE steps before it to
# AFTER steps after; and the run's last LATE steps.
BEFORE = 2
AFTER = 3
LATE = 3
# Frames only at the end of a run: a write, never an input of the march.
OUTPUT_INTERVAL = 10**9


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
    """A scenario's configuration, its base read as the local path, frames at the end."""
    path = Path(str(SCENARIOS[name]["config"]))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["scenario"]["base"] = local_path(raw["scenario"]["base"])
    raw.setdefault("solver", {})["output_interval"] = OUTPUT_INTERVAL
    raw["solver"].pop("checkpoint_interval", None)
    return SimConfig.from_dict(raw, path.parent)


# ---------------------------------------------------------------------------
# Records


def hexed(value: float | None) -> str | None:
    """A float as its exact hex form, None as None."""
    return None if value is None else float(value).hex()


def unhexed(value: str | None) -> float | None:
    """The inverse of hexed."""
    return None if value is None else float.fromhex(value)


def budget_record(budget: MassBudget) -> dict[str, object]:
    """A budget's ten terms, each exact."""
    return {
        "initial": hexed(budget.initial),
        "inflow": hexed(budget.inflow),
        "outflow": hexed(budget.outflow),
        "source": hexed(budget.source),
        "deposited": {name: hexed(budget.deposited[name]) for name in DEPOSIT_SURFACES},
        "current": hexed(budget.current),
        "leftover": hexed(budget.leftover),
    }


def budget_from(record: dict[str, object]) -> MassBudget:
    """The budget of a budget_record."""
    deposited = record["deposited"]
    assert isinstance(deposited, dict)
    return MassBudget(
        initial=unhexed(record["initial"]),
        inflow=unhexed(record["inflow"]),
        outflow=unhexed(record["outflow"]),
        source=unhexed(record["source"]),
        deposited={name: unhexed(deposited[name]) for name in DEPOSIT_SURFACES},
        current=unhexed(record["current"]),
        leftover=unhexed(record["leftover"]),
    )


def digest(array: np.ndarray | None) -> str | None:
    """A SHA-256 of an array's shape, dtype and bytes, None for None."""
    if array is None:
        return None
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{a.shape}{a.dtype}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


CONDITION_ARRAYS = (
    "inflow_u",
    "inflow_v",
    "deposition_u",
    "deposition_v",
    "surface_u",
    "surface_v",
    "settling_v",
)


# ---------------------------------------------------------------------------
# Capture


class Recorder:
    """Wraps solve_timestep and the helpers whose values the core forms once per interval."""

    def __init__(self) -> None:
        self.active = False
        self.window = ""
        self.arrays: dict[str, np.ndarray] = {}
        self.solvers: list[dict[str, object]] = []
        self.solver_index: dict[int, int] = {}
        # The solvers themselves, held so that a freed one's id is never reused
        # by the next interval's (the first vents capture merged two that way).
        self.held: list[TransportSolver] = []
        self.calls: list[dict[str, object]] = []
        self.seen: dict[str, object] | None = None
        self.full_kept: set[tuple[int, int]] = set()
        self.step = 0

    def solver(self, solver: TransportSolver) -> int:
        """The index of a solver, kept with its conditions on first sight."""
        key = id(solver)
        if key in self.solver_index:
            return self.solver_index[key]
        s = len(self.solvers)
        self.solver_index[key] = s
        self.held.append(solver)
        spec: dict[str, object] = {
            "cfl": hexed(solver._cfl),
            "upwind": bool(solver._upwind),
            "max_sweeps": int(solver._max_sweeps),
            "tol": hexed(solver._tol),
            "limit": hexed(solver._limit),
            "share": hexed(solver._share),
            "horizon": hexed(solver._horizon),
            "schmidt": hexed(solver._schmidt),
            "settling": [hexed(x) for x in solver._settling],
            "diffusion": [hexed(x) for x in solver._diffusion],
            "vents": bool(solver._vents),
            "recirculation": [],
        }
        recirculation: list[list[str]] = []
        for k, faces in enumerate(solver._faces):
            for name in CONDITION_ARRAYS:
                self.arrays[f"s{s}/c{k}/{name}"] = np.array(getattr(faces, name))
            if faces.vent_u is not None:
                self.arrays[f"s{s}/c{k}/vent_u"] = np.array(faces.vent_u)
                self.arrays[f"s{s}/c{k}/vent_v"] = np.array(faces.vent_v)
            passed = []
            for p, pair in enumerate(faces.recirculation):
                for name in ("intake_u", "intake_v", "discharge_u", "discharge_v"):
                    self.arrays[f"s{s}/c{k}/r{p}/{name}"] = np.array(
                        getattr(pair, name)
                    )
                passed.append(hexed(pair.passed))
            recirculation.append(passed)
        spec["recirculation"] = recirculation
        self.solvers.append(spec)
        return s

    def keep_once(self, key: str, array: np.ndarray | None) -> None:
        """Keep an interval input in full the first time it is seen."""
        if array is not None and key not in self.arrays:
            self.arrays[key] = np.array(array)


def install(recorder: Recorder) -> None:
    """Wrap the reference's step and helpers on the class, for every solver."""
    original_step = TransportSolver.solve_timestep
    original_adv = TransportSolver._advecting_velocities
    original_nu = TransportSolver._face_eddy_viscosity
    original_implicit = TransportSolver._implicit_step
    original_stable = TransportSolver.stable_dt

    def advecting(self: TransportSolver, *args: object, **kwargs: object) -> object:
        out = original_adv(self, *args, **kwargs)
        if recorder.seen is not None:
            recorder.seen["u_adv"], recorder.seen["v_adv"] = out
        return out

    def eddy(self: TransportSolver, *args: object, **kwargs: object) -> object:
        out = original_nu(self, *args, **kwargs)
        if recorder.seen is not None:
            recorder.seen["nu_u"], recorder.seen["nu_v"] = out
        return out

    def implicit(
        self: TransportSolver,
        c_star: np.ndarray,
        dt: float,
        g_u: np.ndarray,
        g_v: np.ndarray,
        k: int,
        bound: float,
    ) -> object:
        if recorder.seen is not None:
            recorder.seen["g_u"], recorder.seen["g_v"] = g_u, g_v
            recorder.seen["bound"] = bound
        return original_implicit(self, c_star, dt, g_u, g_v, k, bound)

    def stable(self: TransportSolver, *args: object, **kwargs: object) -> float:
        out = original_stable(self, *args, **kwargs)
        if recorder.seen is not None:
            recorder.seen["stable_dt"] = out
        return out

    def step(
        self: TransportSolver,
        C_k: np.ndarray,  # noqa: N803 -- the reference's name
        faces: object,
        size_class: int,
        dt: float,
        v_ext: object = None,
        sources: np.ndarray | None = None,
        *,
        eddy_viscosity: np.ndarray | None = None,
    ) -> np.ndarray:
        if not recorder.active:
            return original_step(
                self,
                C_k,
                faces,
                size_class,
                dt,
                v_ext,
                sources,
                eddy_viscosity=eddy_viscosity,
            )
        s = recorder.solver(self)
        k = int(size_class)
        i = len(recorder.calls)
        call: dict[str, object] = {
            "solver": s,
            "k": k,
            "dt": hexed(dt),
            "step": recorder.step + 1,
            "window": recorder.window,
            "before": budget_record(self.budget[k]),
            "v_ext": v_ext is not None,
        }
        recorder.arrays[f"call{i}/c_in"] = np.array(C_k)
        # The interval's inputs, in full once per solver, and a digest every call.
        recorder.keep_once(f"s{s}/u", faces.u)
        recorder.keep_once(f"s{s}/v", faces.v)
        recorder.keep_once(f"s{s}/nu_t", eddy_viscosity)
        recorder.keep_once(f"s{s}/sources{k}", sources)
        call["inputs"] = {
            "u": digest(faces.u),
            "v": digest(faces.v),
            "nu_t": digest(eddy_viscosity),
            "sources": digest(sources),
        }
        recorder.seen = {}
        try:
            out = original_step(
                self,
                C_k,
                faces,
                size_class,
                dt,
                v_ext,
                sources,
                eddy_viscosity=eddy_viscosity,
            )
            call["breach"] = None
        except ConservationBreachError as breach:
            out = None
            call["breach"] = [
                breach.size_class,
                hexed(breach.leftover),
                hexed(breach.supplied),
                hexed(breach.limit),
            ]
            raise
        finally:
            seen, recorder.seen = recorder.seen, None
            call["after"] = budget_record(self.budget[k])
            call["records"] = {
                "sweeps": int(self.last_diffusion_sweeps),
                "converged": bool(self.diffusion_converged),
                "leftover": hexed(self.last_leftover),
                "bound": hexed(self.last_leftover_bound),
            }
            call["stable_dt"] = hexed(seen.get("stable_dt"))
            call["seen_bound"] = hexed(seen.get("bound"))
            call["seen"] = {
                name: digest(seen.get(name))
                for name in ("u_adv", "v_adv", "nu_u", "nu_v", "g_u", "g_v")
            }
            if (s, k) not in recorder.full_kept:
                recorder.full_kept.add((s, k))
                for name in ("u_adv", "v_adv", "g_u", "g_v"):
                    recorder.arrays[f"s{s}/c{k}/seen_{name}"] = np.array(seen[name])
                for name in ("nu_u", "nu_v"):
                    if seen.get(name) is not None:
                        recorder.keep_once(f"s{s}/seen_{name}", seen[name])
            recorder.calls.append(call)
        if out is not None:
            recorder.arrays[f"call{i}/c_out"] = np.array(out)
        return out

    TransportSolver.solve_timestep = step
    TransportSolver._advecting_velocities = advecting
    TransportSolver._face_eddy_viscosity = eddy
    TransportSolver._implicit_step = implicit
    TransportSolver.stable_dt = stable


def save(name: str, recorder: Recorder, meta: dict[str, object]) -> None:
    """The kept calls as one .npz, and their description as JSON."""
    CAPTURES.mkdir(parents=True, exist_ok=True)
    np.savez(CAPTURES / f"{name}.npz", **recorder.arrays)
    record = {
        **meta,
        "solvers": recorder.solvers,
        "calls": recorder.calls,
    }
    (CAPTURES / f"{name}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print(f"{name}: {len(recorder.calls)} calls kept, {len(recorder.solvers)} solvers")


def load(name: str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """A capture's description and its arrays."""
    meta = json.loads((CAPTURES / f"{name}.json").read_text(encoding="utf-8"))
    with np.load(CAPTURES / f"{name}.npz") as data:
        arrays = {key: data[key] for key in data.files}
    return meta, arrays


def capture(name: str) -> None:
    """Run one scenario and keep every class's steps in its two windows."""
    spec = SCENARIOS[name]
    config = scenario_config(name)
    onset = float(spec["onset"])  # type: ignore[arg-type]
    t_end = float(config.t_end)
    recorder = Recorder()
    install(recorder)

    def on_step(step: MarchStep) -> None:
        # Decides whether the next step is kept, from the time it starts at.
        recorder.step = step.step
        start, dt = step.t, step.dt
        if onset - (BEFORE + 0.5) * dt <= start < onset + (AFTER - 0.5) * dt:
            recorder.active, recorder.window = True, "onset"
        elif start >= t_end - (LATE + 0.5) * dt:
            recorder.active, recorder.window = True, "late"
        else:
            recorder.active = False

    LINUX_RUNS.mkdir(parents=True, exist_ok=True)
    run_file = LINUX_RUNS / f"{name}.h5"
    for path in (run_file, run_file.with_suffix(".checkpoint.h5")):
        path.unlink(missing_ok=True)
    flows_from = spec["flows_from"]
    started = time.perf_counter()
    record = run(
        config,
        run_file,
        flows_from=None if flows_from is None else Path(str(flows_from)),
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
        recorder,
        {
            "kind": "class steps",
            "config": str(Path(str(spec["config"])).relative_to(ROOT)),
            "flows_from": None
            if flows_from is None
            else str(Path(str(flows_from)).relative_to(ROOT)),
            "shape": [config.ny, config.nx],
            "onset": onset,
            "t_end": t_end,
            "windows": {"before": BEFORE, "after": AFTER, "late": LATE},
            "steps": int(getattr(record, "steps", 0) or 0),
            "wall_seconds": wall,
        },
    )


# The vents run's flow state solved from rest with the cap raised, its residual
# history kept: the record of how many outer iterations it needs.
VENTS_FLOW = RESULTS / "vents80_flow"
VENTS_FLOW_CAP = 60000


def vents_flow() -> None:
    """Solve the vents run's flow state from rest with the cap raised, and keep it."""
    from src.boundary_staggered import StaggeredBoundary
    from src.mesh import Mesh
    from src.solver_staggered import StaggeredSolver

    path = Path(str(SCENARIOS["vents80"]["config"]))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["scenario"]["base"] = local_path(raw["scenario"]["base"])
    raw["solver"]["max_simple_iter"] = VENTS_FLOW_CAP
    config = SimConfig.from_dict(raw, path.parent)
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    started = time.perf_counter()
    solver.solve_steady()
    wall = time.perf_counter() - started
    state = solver.flow_state
    history = list(solver.residual_history)
    record = {
        "config": str(path.relative_to(ROOT)),
        "max_simple_iter": VENTS_FLOW_CAP,
        "converged": bool(solver.converged),
        "stop_reason": solver.stop_reason,
        "outer": len(history),
        "wall_seconds": wall,
        "residual_every_1000": [history[i] for i in range(0, len(history), 1000)],
        "residual_last": history[-1] if history else None,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "vents80_flow.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    if state is not None:
        arrays = {
            name: getattr(state, name)
            for name in ("u", "v", "p", "k", "eps", "nu_t")
            if getattr(state, name) is not None
        }
        np.savez(VENTS_FLOW.with_suffix(".npz"), **arrays)
    print(record["converged"], record["stop_reason"], record["outer"], f"{wall:.0f} s")


# ---------------------------------------------------------------------------
# Replay


@dataclasses.dataclass
class Physics:
    """The captured settling and diffusion per class, as ParticleProperties."""

    settling: list[float]
    diffusion: list[float]

    def settling_velocity(self, size_class: int) -> float:
        """The class's settling velocity, m/s."""
        return self.settling[size_class]

    def diffusion_coeff(self, size_class: int) -> float:
        """The class's diffusion coefficient, m^2/s."""
        return self.diffusion[size_class]


@dataclasses.dataclass
class Conditions:
    """The captured ConcentrationFaces per class, as ScalarConditions."""

    faces: list[object]

    def faces_for(self, size_class: int) -> object:
        """The class's captured conditions."""
        return self.faces[size_class]


def captured_conditions(
    arrays: dict[str, np.ndarray], s: int, spec: dict[str, object]
) -> list[object]:
    """Each class's ConcentrationFaces as the solver s held them."""
    from src.boundary_concentration import ConcentrationFaces, Recirculation

    out = []
    for k, passed in enumerate(spec["recirculation"]):  # type: ignore[arg-type]
        prefix = f"s{s}/c{k}/"
        pairs = tuple(
            Recirculation(
                *(
                    arrays[f"{prefix}r{p}/{n}"]
                    for n in ("intake_u", "intake_v", "discharge_u", "discharge_v")
                ),
                unhexed(value),
            )
            for p, value in enumerate(passed)
        )
        vents = {}
        if f"{prefix}vent_u" in arrays:
            vents = {
                "vent_u": arrays[f"{prefix}vent_u"],
                "vent_v": arrays[f"{prefix}vent_v"],
            }
        out.append(
            ConcentrationFaces(
                *(arrays[prefix + n] for n in CONDITION_ARRAYS),
                **vents,
                recirculation=pairs,
            )
        )
    return out


def core_solver(mesh: object, spec: dict[str, object]) -> object:
    """The core's solver on a mesh with a captured solver's keys and physics."""
    import _cfdcore

    from src.boundary_concentration import SURFACE_CEILING, SURFACE_FLOOR, SURFACE_WALL
    from src.mesh import SOLID

    grid = _cfdcore.Grid2(mesh.x, mesh.xc, mesh.y, mesh.yc, mesh.cell_type, SOLID)
    return _cfdcore.TransportSolver2(
        _cfdcore.ScalarScheme2(grid),
        mesh.dx_cell,
        mesh.dy_cell,
        mesh.dx_face,
        mesh.dy_face,
        unhexed(spec["cfl"]),
        bool(spec["upwind"]),
        int(spec["max_sweeps"]),
        unhexed(spec["tol"]),
        unhexed(spec["limit"]),
        unhexed(spec["share"]),
        unhexed(spec["horizon"]),
        unhexed(spec["schmidt"]),
        [unhexed(x) for x in spec["settling"]],
        [unhexed(x) for x in spec["diffusion"]],
        (SURFACE_FLOOR, SURFACE_CEILING, SURFACE_WALL),
    )


def interval_inputs(
    arrays: dict[str, np.ndarray], s: int, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[np.ndarray | None]]:
    """A captured solver's faces, eddy viscosity and sources per class."""
    return (
        arrays[f"s{s}/u"],
        arrays[f"s{s}/v"],
        arrays.get(f"s{s}/nu_t"),
        [arrays.get(f"s{s}/sources{k}") for k in range(n)],
    )


def same_bits(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    """Whether two float64 arrays have one shape and equal 64-bit patterns (None equals None)."""
    if a is None or b is None:
        return a is None and b is None
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(
        np.array_equal(a.view(np.uint64), b.view(np.uint64))
    )


def budget_terms(budget: MassBudget) -> dict[str, str | None]:
    """A budget's ten terms by name, each exact."""
    record = budget_record(budget)
    deposited = record.pop("deposited")
    assert isinstance(deposited, dict)
    return {**record, **{f"deposited.{k}": v for k, v in deposited.items()}}  # type: ignore[dict-item]


def reference_interval(
    reference: TransportSolver,
    faces: object,
    k: int,
    dt: float,
    rate: np.ndarray | None,
) -> dict[str, object]:
    """The interval quantities as the reference forms them inside one step.

    The values formed by helpers the step calls (the advecting velocities, the
    conductances, the face eddy viscosity) are the capture's; the ones the
    step forms inline are formed here by the step's own expressions on the
    reference solver built from the captured conditions.
    """
    u_adv, v_adv = reference._advecting_velocities(faces, k, None)
    flux_u = u_adv * reference._area_u
    flux_v = v_adv * reference._area_v
    values: dict[str, object] = {
        "volume_flux": (flux_u, flux_v),
        "deposition_step": (reference._deposit_u[k] * dt, reference._deposit_v[k] * dt),
        "deposition_cell": reference._deposit_cell[k],
        "unit_conductance": (reference._conductance_u, reference._conductance_v),
        "domain": (
            (
                flux_u[:, 0] > 0.0,
                flux_u[:, 0] < 0.0,
                flux_u[:, -1] < 0.0,
                flux_u[:, -1] > 0.0,
            ),
            (
                flux_v[0, :] > 0.0,
                flux_v[0, :] < 0.0,
                flux_v[-1, :] < 0.0,
                flux_v[-1, :] > 0.0,
            ),
        ),
        "source_step": None if rate is None else dt * rate,
        "source_total": 0.0
        if rate is None
        else float(np.sum(rate * reference._volume)),
    }
    if reference._vents:
        conditions = reference._faces[k]
        values["drawn"] = [
            float(np.abs(flux_u[p.intake_u]).sum() + np.abs(flux_v[p.intake_v]).sum())
            for p in conditions.recirculation
        ]
        values["vent"] = tuple(
            tuple(
                (vents & side & sign).reshape(-1)
                for side, sign in (
                    (low, flux > 0.0),
                    (high, flux < 0.0),
                    (high, flux > 0.0),
                    (low, flux < 0.0),
                )
            )
            for vents, low, high, flux in (
                (
                    reference._vent_u,
                    reference._solid_low_u,
                    reference._solid_high_u,
                    flux_u,
                ),
                (
                    reference._vent_v,
                    reference._solid_low_v,
                    reference._solid_high_v,
                    flux_v,
                ),
            )
        )
    return values


def interval_differences(
    core: object,
    reference: TransportSolver,
    arrays: dict[str, np.ndarray],
    s: int,
    k: int,
    call: dict[str, object],
    faces: object,
    rate: np.ndarray | None,
) -> list[str]:
    """The core's interval quantities of class k that differ from the reference's per-step values."""
    differ = []
    dt = unhexed(call["dt"])
    seen = call["seen"]
    assert isinstance(seen, dict)
    full = {
        name: arrays[f"s{s}/c{k}/seen_{name}"]
        for name in ("u_adv", "v_adv", "g_u", "g_v")
    }
    for name, d, key in (
        ("advecting_velocity", 0, "u_adv"),
        ("advecting_velocity", 1, "v_adv"),
        ("conductance", 0, "g_u"),
        ("conductance", 1, "g_v"),
    ):
        got = core.interval_quantity(name, k, d)
        # The capture's array of the solver's first call, and its digest at this call.
        if not same_bits(got, full[key]) or digest(got) != seen[key]:
            differ.append(f"{name} {d}")
    for name, d, key in (
        ("face_eddy_viscosity", 0, "nu_u"),
        ("face_eddy_viscosity", 1, "nu_v"),
    ):
        got = core.interval_quantity(name, k, d)
        if seen[key] is None:
            if got is not None:
                differ.append(f"{name} {d} present")
        elif digest(got) != seen[key]:
            differ.append(f"{name} {d}")
    if (
        core.stable_dt(k) != unhexed(call["stable_dt"])
        or hexed(core.stable_dt(k)) != call["stable_dt"]
    ):
        differ.append("stable_dt")
    expected = reference_interval(reference, faces, k, dt, rate)
    for name in ("volume_flux", "deposition_step", "unit_conductance"):
        for d in (0, 1):
            if not same_bits(core.interval_quantity(name, k, d), expected[name][d]):
                differ.append(f"{name} {d}")
    for name in ("deposition_cell", "source_step"):
        if not same_bits(core.interval_quantity(name, k, 0), expected[name]):
            differ.append(name)
    if hexed(core.source_total(k)) != hexed(expected["source_total"]):
        differ.append("source_total")
    for d in (0, 1):
        got = [m.tolist() for m in core.masks("domain", k, d)]
        if got != [m.tolist() for m in expected["domain"][d]]:
            differ.append(f"domain masks {d}")
    if "vent" in expected:
        for d in (0, 1):
            got = [m.tolist() for m in core.masks("vent", k, d)]
            if got != [m.tolist() for m in expected["vent"][d]]:
                differ.append(f"vent masks {d}")
        if [hexed(x) for x in core.drawn(k)] != [hexed(x) for x in expected["drawn"]]:
            differ.append("drawn")
    return differ


def replay(out_name: str = "replay") -> None:
    """Every captured call through the core, every output compared bit for bit."""
    from src.mesh import Mesh
    from src.staggered import FaceVelocities

    rows = []
    for name in SCENARIOS:
        if not (CAPTURES / f"{name}.json").exists():
            print(name, "not captured")
            continue
        meta, arrays = load(name)
        mesh = Mesh(scenario_config(name))
        cores, references, inputs = {}, {}, {}
        for s, spec in enumerate(meta["solvers"]):
            n = len(spec["settling"])
            conditions = captured_conditions(arrays, s, spec)
            u, v, nu_t, sources = interval_inputs(arrays, s, n)
            core = core_solver(mesh, spec)
            core.set_interval(conditions, u, v, None, nu_t, sources)
            cores[s] = core
            config = scenario_config(name)
            reference = TransportSolver(
                mesh,
                config,
                Physics(
                    [unhexed(x) for x in spec["settling"]],
                    [unhexed(x) for x in spec["diffusion"]],
                ),
                Conditions(conditions),
            )
            # The reference rebuilt from the capture holds the captured solver's keys.
            for key, attribute in (
                ("cfl", "_cfl"),
                ("tol", "_tol"),
                ("limit", "_limit"),
                ("share", "_share"),
                ("horizon", "_horizon"),
            ):
                if hexed(getattr(reference, attribute)) != spec[key]:
                    raise SystemExit(
                        f"{name} solver {s}: {key} is not the captured solver's"
                    )
            references[s] = reference
            inputs[s] = (
                FaceVelocities.copy_of(np.array(u), np.array(v)),
                nu_t,
                sources,
            )
        for i, call in enumerate(meta["calls"]):
            s, k = call["solver"], call["k"]
            core, reference = cores[s], references[s]
            faces, nu_t, sources = inputs[s]
            dt = unhexed(call["dt"])
            core.set_step(dt)
            budgets = [MassBudget() for _ in range(core.classes)]
            budgets[k] = budget_from(call["before"])
            core_mapping.restore_budgets(core, budgets)
            reference.restore_budgets(budgets)
            c_in = arrays[f"call{i}/c_in"]
            expected = arrays.get(f"call{i}/c_out")
            out = np.full(c_in.shape, np.nan)
            breach = None
            try:
                record = core_mapping.step(core, k, c_in, out)
            except ConservationBreachError as error:
                record = None
                breach = [
                    error.size_class,
                    hexed(error.leftover),
                    hexed(error.supplied),
                    hexed(error.limit),
                ]
            after = budget_terms(core_mapping.budgets_of(core)[k])
            wanted = budget_terms(budget_from(call["after"]))
            terms_differ = [term for term in after if after[term] != wanted[term]]
            records_differ = []
            if record is not None:
                got = {
                    "sweeps": record.sweeps,
                    "converged": record.converged,
                    "leftover": hexed(record.leftover),
                    "bound": hexed(record.bound),
                }
                records_differ = [r for r in got if got[r] != call["records"][r]]
            # The reference again on the captured inputs: the capture is what it gives.
            again = None
            with contextlib.suppress(ConservationBreachError):
                again = reference.solve_timestep(
                    c_in, faces, k, dt, None, sources[k], eddy_viscosity=nu_t
                )
            reproduces = (
                same_bits(again, expected)
                and budget_terms(reference.budget[k]) == wanted
            )
            # The interval's inputs kept once per solver are the ones this call
            # was handed (its digests), so no call is replayed on another's.
            kept = {
                "u": digest(arrays[f"s{s}/u"]),
                "v": digest(arrays[f"s{s}/v"]),
                "nu_t": digest(nu_t),
                "sources": digest(sources[k]),
            }
            inputs_match = kept == call["inputs"]
            interval = interval_differences(
                core, reference, arrays, s, k, call, faces, sources[k]
            )
            field_differ = (
                int(np.count_nonzero(out.view(np.uint64) != expected.view(np.uint64)))
                if expected is not None
                else (0 if record is None else -1)
            )
            rows.append(
                {
                    "scenario": name,
                    "window": call["window"],
                    "step": call["step"],
                    "class": k,
                    "solver": s,
                    "cells": int(c_in.size),
                    "vents": bool(meta["solvers"][s]["vents"]),
                    "recirculated": bool(core.drawn(k))
                    and any(x > 0.0 for x in core.drawn(k)),
                    "sources": sources[k] is not None,
                    "eddy": nu_t is not None,
                    "sweeps": call["records"]["sweeps"],
                    "converged": call["records"]["converged"],
                    "breach": breach is not None or call["breach"] is not None,
                    "breach_equal": breach == call["breach"],
                    "field_differ": field_differ,
                    "terms_differ": terms_differ,
                    "records_differ": records_differ,
                    "interval_differ": interval,
                    "reference_reproduces": bool(reproduces),
                    "inputs_match": bool(inputs_match),
                    "budget_after": wanted,
                }
            )
        bad = [
            r
            for r in rows
            if r["scenario"] == name
            and (
                r["field_differ"]
                or r["terms_differ"]
                or r["records_differ"]
                or r["interval_differ"]
                or not r["breach_equal"]
                or not r["reference_reproduces"]
                or not r["inputs_match"]
            )
        ]
        print(
            name,
            len([r for r in rows if r["scenario"] == name]),
            "calls,",
            len(bad),
            "differ",
        )
    # A plant's replay goes beside the plants' record, never over the head's.
    target = RESULTS / f"{out_name}.json"
    if out_name != "replay":
        target = RESULTS / "plants" / f"{out_name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Timing

STORED_RUN = STORED / "product" / "equipment_dust.h5"
REFERENCE_REPEATS = 30
CORE_REPEATS = 300
WARMUP = 3
CALIBRATION_LENGTH = 4096
CALIBRATION_REPEATS = 2000


def timed(fn: Callable[[], object], repeats: int) -> dict[str, float]:
    """The median, minimum and quartiles of repeats calls of fn, after WARMUP calls."""
    import statistics

    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append(time.perf_counter() - started)
    quartiles = statistics.quantiles(times, n=4)
    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "q1_s": quartiles[0],
        "q3_s": quartiles[2],
        "repeats": repeats,
    }


def windows_load() -> list[str]:
    """Windows' busiest processes, the power state and the processor's performance.

    Prompt 89's probe (``probe89.windows_load``), sampled over two seconds
    through ``powershell.exe``; empty when Windows cannot be reached.
    """
    import subprocess

    script = (
        "$s = Get-Counter '\\Process(*)\\% Processor Time' -SampleInterval 2 -MaxSamples 1"
        " -ErrorAction SilentlyContinue;"
        " $s.CounterSamples | Where-Object { $_.InstanceName -notin '_total','idle' }"
        " | Sort-Object CookedValue -Descending | Select-Object -First 8"
        " | ForEach-Object { '{0} {1:N0}' -f $_.InstanceName, $_.CookedValue };"
        " 'on mains ' + ((Get-CimInstance Win32_Battery).BatteryStatus -eq 2);"
        " 'processor performance ' + [int](Get-Counter"
        " '\\Processor Information(_Total)\\% Processor Performance')"
        ".CounterSamples[0].CookedValue"
    )
    try:
        done = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def calibration() -> dict[str, float]:
    """A fixed NumPy loop that fits the first-level cache (prompt 89's)."""
    values = np.linspace(1.0, 2.0, CALIBRATION_LENGTH)
    out = np.empty_like(values)
    return timed(lambda: np.sqrt(values, out=out), CALIBRATION_REPEATS)


def timing(run_number: int) -> None:
    """A class step on the developed 320x120 frame, every class, core against reference."""
    import _cfdcore
    import h5py

    from src.mesh import Mesh
    from src.staggered import FaceVelocities

    started_load = windows_load()
    calib_before = calibration()
    name = "dust320"
    meta, arrays = load(name)
    late = [c for c in meta["calls"] if c["window"] == "late"]
    s = late[-1]["solver"]
    spec = meta["solvers"][s]
    n = len(spec["settling"])
    mesh = Mesh(scenario_config(name))
    conditions = captured_conditions(arrays, s, spec)
    u, v, nu_t, sources = interval_inputs(arrays, s, n)
    faces = FaceVelocities.copy_of(np.array(u), np.array(v))
    dt = unhexed(late[-1]["dt"])
    with h5py.File(STORED_RUN, "r") as stored:
        developed = [
            np.ascontiguousarray(stored[f"transport/C_{k}"][-1], dtype=np.float64)
            for k in range(n)
        ]
    reference = TransportSolver(
        mesh,
        scenario_config(name),
        Physics(
            [unhexed(x) for x in spec["settling"]],
            [unhexed(x) for x in spec["diffusion"]],
        ),
        Conditions(conditions),
    )
    core = core_solver(mesh, spec)
    set_up = timed(lambda: core.set_interval(conditions, u, v, None, nu_t, sources), 5)
    core.set_step(dt)
    rows = []
    for k in range(n):
        c = developed[k]
        out = np.empty_like(c)
        # The two give the same step on this frame: checked before it is timed.
        first = reference.solve_timestep(
            c, faces, k, dt, None, sources[k], eddy_viscosity=nu_t
        )
        record = core.step(k, c, out)
        if (
            not same_bits(first, out)
            or record.sweeps != reference.last_diffusion_sweeps
        ):
            raise SystemExit(
                f"class {k}: the core's step is not the reference's on the frame"
            )
        ref_time = timed(
            lambda c=c, k=k: reference.solve_timestep(
                c, faces, k, dt, None, sources[k], eddy_viscosity=nu_t
            ),
            REFERENCE_REPEATS,
        )
        core_time = timed(lambda c=c, k=k, out=out: core.step(k, c, out), CORE_REPEATS)
        rows.append(
            {
                "class": k,
                "sweeps": int(record.sweeps),
                "reference": ref_time,
                "core": core_time,
                "ratio": ref_time["median_s"] / core_time["median_s"],
            }
        )
        print(
            k,
            record.sweeps,
            f"{ref_time['median_s'] * 1e3:.3f} ms",
            f"{core_time['median_s'] * 1e3:.3f} ms",
            f"{rows[-1]['ratio']:.2f}",
        )
    record = {
        "run": run_number,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame": str(STORED_RUN.relative_to(ROOT)),
        "dt": hexed(dt),
        "module": str(_cfdcore.__file__),
        "windows_before": started_load,
        "calibration_before": calib_before,
        "set_interval": set_up,
        "rows": rows,
        "calibration_after": calibration(),
        "windows_after": windows_load(),
    }
    (RESULTS / f"timing_{run_number}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# The vectorisation report


def vecinfo() -> None:
    """GCC's vectorisation report for the core's sources, its loops counted by place.

    The report names a loop once for each inlined copy and each version it
    makes, so the count is of distinct source places (file:line:column) whose
    loop is vectorised, which is what CLAUDE.md asks to read (prompt 89's T8).
    """
    import re
    import subprocess

    flags = ["-std=c++20", "-O2", "-ffp-contract=off", "-march=x86-64-v3", "-DNDEBUG"]
    record: dict[str, object] = {"flags": flags}
    for file in ("src/transport.cpp", "src/scalar.cpp", "src/reduce.cpp"):
        report = subprocess.run(
            [
                "g++",
                *flags,
                f"-I{ROOT / 'core' / 'include'}",
                "-c",
                str(ROOT / "core" / file),
                "-o",
                "/dev/null",
                "-fopt-info-vec-optimized",
            ],
            capture_output=True,
            text=True,
        )
        lines = [
            re.sub(r"^.*?/core/", "", line)
            for line in (report.stdout + report.stderr).splitlines()
            if "optimized:" in line and "loop vectorized" in line
        ]
        places = sorted({line.split(": optimized")[0] for line in lines})
        record[file] = {"report_lines": len(lines), "loops": places}
        print(file, len(lines), "report lines,", len(places), "loops")
    version = subprocess.run(["g++", "--version"], capture_output=True, text=True)
    record["compiler"] = version.stdout.splitlines()[0]
    (RESULTS / "vecinfo.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Planted defects

PLANT_WORK = Path.home() / "scratch90" / "plants"
GTEST_SRC = Path.home() / "build" / "cfd-core" / "_deps" / "googletest-src"


def plant_sources(edits: list[tuple[str, str, str]]) -> dict[str, str]:
    """The checkout's core/ with a plant's edits made, path to text with LF line ends."""
    files = {
        path.relative_to(ROOT / "core").as_posix(): path.read_text(
            encoding="utf-8"
        ).replace("\r\n", "\n")
        for path in sorted((ROOT / "core").rglob("*"))
        if path.is_file()
    }
    for file, old, new in edits:
        if files[file].count(old) != 1:
            raise SystemExit(f"{file}: found {files[file].count(old)} times: {old!r}")
        files[file] = files[file].replace(old, new)
    return files


def plants() -> None:
    """Each planted defect built in turn, with the tests and the replay that fail on it."""
    import re
    import subprocess
    from collections import Counter

    sys.path.insert(0, str(HERE))
    import variants90

    source = PLANT_WORK / "core"
    build = PLANT_WORK / "build"
    records = []
    for name, what, edits in variants90.PLANTS:
        files = plant_sources(edits)
        for relative, text in files.items():
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Written only when it differs, so ninja rebuilds what changed (a copy
            # keeps old file times and ninja would not: prompt 88's lesson).
            if not target.exists() or target.read_text(encoding="utf-8") != text:
                target.write_text(text, encoding="utf-8")
        record: dict[str, object] = {"name": name, "what": what}
        configure = subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-G",
                "Ninja",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST={GTEST_SRC}",
            ],
            capture_output=True,
            text=True,
        )
        made = subprocess.run(
            ["ninja", "-C", str(build)], capture_output=True, text=True
        )
        if configure.returncode or made.returncode:
            record["build"] = "failed: " + (made.stdout + made.stderr)[-1500:]
            records.append(record)
            print(name, "build failed")
            continue
        ctest = subprocess.run(
            ["ctest", "--test-dir", str(build)], capture_output=True, text=True
        )
        summary = [line for line in ctest.stdout.splitlines() if "tests passed" in line]
        record["ctest"] = summary[0].strip() if summary else ctest.stdout[-300:]
        # A test that crashes is listed as (SEGFAULT) or (Subprocess aborted), not (Failed).
        record["ctest_failed"] = re.findall(r"\d+ - (\S+) \([^)]+\)", ctest.stdout)
        env = {
            **__import__("os").environ,
            "PYTHONPATH": str(build),
            "CFD_CORE_REQUIRED": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        pytest = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_core_parity.py",
                "-q",
                "-rf",
                "--tb=no",
                "-p",
                "no:cacheprovider",
                "-k",
                "class_step or buffered_sum",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )
        lines = pytest.stdout.splitlines()
        record["pytest"] = next(
            (
                line.strip("= ")
                for line in reversed(lines)
                if re.search(r" in [0-9.]+s", line)
            ),
            lines[-1] if lines else "no output",
        )
        failed = [
            line.split("::", 1)[1].split(" ")[0]
            for line in lines
            if line.startswith("FAILED tests/test_core_parity.py::")
        ]
        record["parity_failed_by_test"] = dict(
            sorted(Counter(case.split("[")[0] for case in failed).items())
        )
        replayed = subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "replay",
                "--out",
                f"plant_{name}",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )
        record["replay"] = [
            line for line in replayed.stdout.splitlines() if "calls," in line
        ]
        records.append(record)
        print(
            name,
            "|",
            record["ctest"],
            "|",
            record["pytest"],
            "|",
            "; ".join(record["replay"]),
        )
        (RESULTS / "plants.json").write_text(
            json.dumps(records, indent=1) + "\n", encoding="utf-8"
        )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run one command of the probe."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("name", choices=list(SCENARIOS))
    commands.add_parser("vents-flow")
    replayer = commands.add_parser("replay")
    replayer.add_argument("--out", default="replay")
    commands.add_parser("plants")
    commands.add_parser("vecinfo")
    timer = commands.add_parser("timing")
    timer.add_argument("--run", type=int, required=True)
    args = parser.parse_args(argv)
    commands_run: dict[str, Callable[[], None]] = {
        "capture": lambda: capture(args.name),
        "vents-flow": vents_flow,
        "replay": lambda: replay(args.out),
        "plants": plants,
        "vecinfo": vecinfo,
        "timing": lambda: timing(args.run),
    }
    commands_run[args.command]()
    return 0


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        raise SystemExit(main())
