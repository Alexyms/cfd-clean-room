"""Stage timers for prompt 84: where a scenario run's time goes.

    python docs/reports/probe84/probe84.py configs
    python docs/reports/probe84/probe84.py calibrate-flow GRID OUTER
    python docs/reports/probe84/probe84.py calibrate-march GRID T_STOP
    python docs/reports/probe84/probe84.py run GRID
    python docs/reports/probe84/probe84.py flow GRID OUTER
    python docs/reports/probe84/probe84.py window GRID T_START T_STOP
    python docs/reports/probe84/probe84.py cprofile GRID
    python docs/reports/probe84/probe84.py construct GRID

Each command is one measurement process; ``run84.py`` starts them one at a
time with one BLAS thread each and records what else was running.

The timers wrap the functions that make up each stage, in this process only,
and never copy a body. Every call boundary of a wrapped function reads
``time.perf_counter_ns`` once and charges the time since the last reading to
the label path that was running (exclusive time), so every nanosecond between
``Clock.start`` and ``Clock.stop`` is charged to exactly one path. Code that is
inline in ``StaggeredSolver.solve_steady`` and ``TransportSolver.solve_timestep``
(the divergence update, the source, the booking, the residual) is not a
function, so it is labelled by region: ``sys.monitoring`` LINE events fire on
the first line of each region, found by its text in the function's source, and
relabel the running path; every other line returns ``DISABLE`` and costs
nothing after its first execution.

The window march starts a run part way: the dust release's fields and budgets
are exactly zero before its onset at 10 s (the stored 320x120 run's frames say
so, step 0 to 2700), so a march started at a step before the onset from zero
fields, fresh budgets and a fresh monitor and arrival maps is the stored run's
march from that step. ``window`` checks its frames against the stored run's.

Records go to ``results/builder84/<job>.json``.
"""

import argparse
import cProfile
import dataclasses
import inspect
import json
import math
import os
import platform
import pstats
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402 -- follows sys.path.insert
    multigrid,
    pressure,
    solver_staggered,
    solver_transport,
    time_integration,
    turbulence,
)
from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.io_manager import RunFile  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.momentum import MomentumPredictor  # noqa: E402
from src.scenarios import ScenarioManager  # noqa: E402
from src.solver_staggered import StaggeredSolver  # noqa: E402
from src.solver_transport import MassBudget, TransportSolver  # noqa: E402
from src.stopping import ErrorEstimateRule  # noqa: E402
from src.time_integration import Provenance, _Run  # noqa: E402

RESULTS = ROOT / "results" / "builder84"
CONFIGS = RESULTS / "configs"
STORED = ROOT / "results" / "builder81" / "product" / "equipment_dust.h5"
STORED_CONFIG = ROOT / "results" / "builder81" / "configs" / "product"
STORED_CONFIG = STORED_CONFIG / "scenario_equipment_dust.yaml"
# The release's onset: the fields are zero before it (module docstring).
ONSET = 10.0
# mass_imbalance_tol per grid: the base file's rule, conservation_limit times
# rho times the smallest cell's area over 60 s (configs/clean_room_default.yaml,
# solver section), which gives the product grid's 1.25e-9 and prompt 75's
# 80x30 value 2e-8 (docs/reports/probe79/configs79.py, PLACEMENT_TOL).
GRIDS = {
    "80x30": (80, 30, 2.0e-8),
    "160x60": (160, 60, 5.0e-9),
    "320x120": (320, 120, 1.25e-9),
}
# The calibration's order: unwrapped and wrapped in turn, so a drift in the
# machine's speed over the runs falls on both.
ORDER = (False, True, False, True, False)
# Line regions in solve_timestep cost the march 7% to 10% on 80x30 (an
# ablation, overhead84.py ablate), well above the few percent their callbacks
# account for, so the march runs without them: solve_timestep's own lines are
# one lump, "transport<k>:inline", which tables84.py splits by the regions of a
# separate run of the same window with them on (``--regions``). The detail
# wrappers inside the advection and the implicit step run only with them on.
MARCH_REGIONS = False
DETAIL_ONLY = {"limiter", "quick", "implicit"}
TOOL = 3  # a sys.monitoring tool id no debugger, coverage tool or profiler takes


# ---------------------------------------------------------------------------
# The clock
# ---------------------------------------------------------------------------


class Clock:
    """Exclusive time per label path, and per-call counters.

    ``path`` is the tuple of labels of the wrapped calls (and regions) that
    are running, outermost first. ``charge[path]`` is the time spent with that
    path innermost; ``calls[path]`` the number of entries into it.
    """

    def __init__(self) -> None:
        self.path: tuple[str, ...] = ("probe",)
        self.last = perf_counter_ns()
        self.started = self.last
        self.stopped: int | None = None
        self.charge: dict[tuple[str, ...], int] = defaultdict(int)
        self.calls: Counter[tuple[str, ...]] = Counter()
        self.counters: dict[str, list] = defaultdict(list)
        self.size_class: int | None = None

    def tick(self) -> int:
        """Charge the time since the last reading to the running path."""
        now = perf_counter_ns()
        self.charge[self.path] += now - self.last
        self.last = now
        return now

    def stop(self) -> None:
        """Charge the tail and freeze the total."""
        self.stopped = self.tick()

    def record(self) -> dict:
        """The charges, calls and counters as JSON-ready lists."""
        assert self.stopped is not None
        return {
            "total_ns": self.stopped - self.started,
            "charge": [[list(p), ns] for p, ns in self.charge.items()],
            "calls": [[list(p), n] for p, n in self.calls.items()],
            "counters": dict(self.counters),
        }


CLOCK: Clock | None = None


def _first_region(
    label: str | Callable[[tuple, dict], str], region: str
) -> Callable[[tuple, dict], str]:
    """LABEL with ``:REGION`` appended, the region a function's frame starts in."""

    def labelled(args: tuple, kwargs: dict) -> str:
        name = label(args, kwargs) if callable(label) else label
        return f"{name}:{region}"

    return labelled


@dataclasses.dataclass
class Wrap:
    """One wrapped attribute: ``owner.name`` timed under ``label``.

    ``label`` may be a callable of the call's positional and keyword
    arguments (the transport step takes its class from them). ``counter``,
    if given, is called with the arguments, the result and the call's
    inclusive time in ns, and returns a value appended to
    ``CLOCK.counters[key]``.
    """

    owner: object
    name: str
    label: str | Callable[[tuple, dict], str]
    key: str | None = None
    counter: Callable[[tuple, dict, object, int], object] | None = None
    original: object = None


def _wrapper(spec: Wrap, function: Callable) -> Callable:
    """The timing wrapper of FUNCTION under SPEC."""
    label = spec.label
    counter = spec.counter
    key = spec.key

    def timed(*args: object, **kwargs: object) -> object:
        clock = CLOCK
        if clock is None:
            return function(*args, **kwargs)
        entered = clock.tick()
        saved = clock.path
        name = label(args, kwargs) if callable(label) else label
        clock.path = (*saved, name)
        clock.calls[clock.path] += 1
        try:
            result = function(*args, **kwargs)
        finally:
            left = clock.tick()
            clock.path = saved
        if counter is not None:
            clock.counters[key].append(counter(args, kwargs, result, left - entered))
        return result

    timed.__wrapped__ = function  # type: ignore[attr-defined]
    return timed


def _transport_label(args: tuple, kwargs: dict) -> str:
    """``transport<k>`` from solve_timestep's size_class (args: self, C_k, faces, k)."""
    k = args[3] if len(args) > 3 else kwargs["size_class"]
    if CLOCK is not None:
        CLOCK.size_class = int(k)
    return f"transport{int(k)}"


def _cg_counter(args: tuple, kwargs: dict, result: object, ns: int) -> list:
    return [result.iterations, result.products, ns]  # type: ignore[attr-defined]


def _scalar_counter(args: tuple, kwargs: dict, result: object, ns: int) -> list:
    r = result
    return [r.iterations, r.fallbacks, r.fallback_sweeps, ns]  # type: ignore[attr-defined]


def _sweeps_counter(args: tuple, kwargs: dict, result: object, ns: int) -> list:
    k = -1 if CLOCK is None or CLOCK.size_class is None else CLOCK.size_class
    return [k, result.iterations, ns]  # type: ignore[attr-defined]


def _observe_counter(args: tuple, kwargs: dict, result: object, ns: int) -> list:
    # args: self, step, t, fields, pending; the reading is taken after the call.
    return [int(args[1]), perf_counter_ns()]


def _flow_counter(args: tuple, kwargs: dict, result: object, ns: int) -> list:
    solver = args[0]
    return [len(solver.residual_history), solver.stop_reason, ns]  # type: ignore[attr-defined]


def _step_counter(args: tuple, kwargs: dict, result: object, ns: int) -> list:
    return [CLOCK.size_class if CLOCK else -1, ns]


def wraps() -> list[Wrap]:
    """Every wrapped attribute, outermost stages first.

    Labels name the function; ``STAGES`` (below) says which stage a label
    opens. A label without a stage inherits the stage of the innermost label
    above it on the path that has one.
    """
    st, tr, ti = solver_staggered, solver_transport, time_integration
    every = [
        # The run and its phases.
        Wrap(ti, "run", "run"),
        Wrap(_Run, "plans", "plans"),
        Wrap(_Run, "march", "march"),
        # The flow solve.
        Wrap(StaggeredSolver, "solve_steady", "flow", "flow", _flow_counter),
        Wrap(StaggeredSolver, "_extrapolate_outlets", "outlets"),
        Wrap(StaggeredSolver, "_effective_viscosity", "mu_eff"),
        Wrap(MomentumPredictor, "predict", "predict"),
        Wrap(MomentumPredictor, "_assemble", "assemble"),
        Wrap(MomentumPredictor, "_sweep_n", "momentum_sweeps"),
        Wrap(MomentumPredictor, "stencil_viscosity", "stencil_mu"),
        Wrap(turbulence.TurbulenceBoundary, "wall_viscosity", "wall_mu"),
        Wrap(turbulence.TurbulenceBoundary, "conditions", "turbulence_conditions"),
        Wrap(pressure.PressureCorrector, "correct", "correct", "cg", _cg_counter),
        Wrap(pressure.PressureCorrector, "coefficients", "p_coefficients"),
        Wrap(pressure.PressureCorrector, "mass_imbalance", "mass_imbalance"),
        Wrap(pressure, "multigrid_preconditioner", "mg_build"),
        Wrap(pressure, "conjugate_gradient", "cg_loop"),
        Wrap(pressure, "apply_operator", "apply"),
        Wrap(multigrid.Multigrid, "cycle", "mg_cycle"),
        Wrap(StaggeredSolver, "_turbulence_step", "turbulence_step"),
        Wrap(turbulence.KEpsilonModel, "step", "keps_step"),
        Wrap(turbulence, "implicit_step", "keps_implicit", "keps", _scalar_counter),
        Wrap(ErrorEstimateRule, "update", "rule_update"),
        Wrap(StaggeredSolver, "_imbalance_summary", "imbalance_summary"),
        Wrap(st, "to_cell_centers", "cell_centers"),
        Wrap(st, "viscosity_scale", "viscosity_scale"),
        Wrap(RunFile, "write_steady_checkpoint", "steady_checkpoint"),
        # The march.
        Wrap(
            TransportSolver, "solve_timestep", _transport_label, "step", _step_counter
        ),
        Wrap(TransportSolver, "stable_dt", "stable_dt"),
        Wrap(TransportSolver, "_check_eddy_viscosity", "check_nu_t"),
        Wrap(TransportSolver, "_advecting_velocities", "advecting_velocities"),
        Wrap(tr, "advective_flux", "advective_flux"),
        Wrap(sys.modules["src.scalar_scheme"], "limited_face_values", "limiter"),
        Wrap(sys.modules["src.scalar_scheme"], "quick_face_values", "quick"),
        Wrap(TransportSolver, "_face_eddy_viscosity", "face_nu_t"),
        Wrap(TransportSolver, "_implicit_step", "implicit"),
        Wrap(tr, "implicit_step", "implicit_solve", "sweeps", _sweeps_counter),
        Wrap(TransportSolver, "_book_deposition", "book_deposition"),
        Wrap(MassBudget, "in_domain", "in_domain"),
        Wrap(_Run, "observe", "observe", "observe", _observe_counter),
        Wrap(_Run, "_flush", "flush"),
        Wrap(sys.modules["src.monitor"].AlertMonitor, "update", "monitor_update"),
        Wrap(sys.modules["src.monitor"].ArrivalTracker, "update", "arrival_update"),
        Wrap(RunFile, "write_frame", "write_frame"),
        Wrap(RunFile, "write_arrival", "write_arrival"),
        Wrap(RunFile, "checkpoint", "write_checkpoint"),
    ]
    if MARCH_REGIONS:
        return every
    return [w for w in every if w.label not in DETAIL_ONLY]


# The stage each label opens. Regions of the two inline functions carry the
# stage of their region; ``transport<k>`` regions are looked up without k.
STAGES = {
    "run": "run.remainder",
    "plans": "run.plans",
    "march": "march.remainder",
    "flow": "flow.remainder",
    "flow:setup": "flow.remainder",
    "flow:loop": "flow.remainder",
    "flow:finish": "flow.remainder",
    "flow:momentum": "flow.momentum",
    "flow:pressure": "flow.pressure",
    "flow:turbulence": "flow.turbulence",
    "flow:stopping": "flow.stopping",
    "outlets": "flow.boundary",
    "predict": "flow.momentum",
    "stencil_mu": "flow.boundary",
    "wall_mu": "flow.boundary",
    "turbulence_conditions": "flow.boundary",
    "correct": "flow.pressure",
    "turbulence_step": "flow.turbulence",
    "rule_update": "flow.stopping",
    "steady_checkpoint": "flow.io",
    "transport": "class.remainder",
    "transport:checks": "class.remainder",
    "transport:inline": "class.inline",
    "transport:advection": "class.advection",
    "transport:divergence": "class.divergence",
    "transport:source": "class.source",
    "transport:booking": "class.booking",
    "transport:implicit": "class.implicit",
    "observe": "step.monitor",
    "arrival_update": "step.arrival",
    "write_frame": "step.frames",
    "flush": "step.frames",
    "write_arrival": "step.checkpoint",
    "write_checkpoint": "step.checkpoint",
}


def stage_of_label(label: str) -> str | None:
    """The stage LABEL opens, or None when it inherits one."""
    if label.startswith("transport"):
        head, _, region = label.partition(":")
        key = "transport" if not region else f"transport:{region}"
        if head[len("transport") :].isdigit() or head == "transport":
            return STAGES.get(key)
    return STAGES.get(label)


def stage_of_path(path: list[str] | tuple[str, ...]) -> str:
    """The stage of the innermost label on PATH that opens one ("outside" if none)."""
    for label in reversed(path):
        stage = stage_of_label(label)
        if stage is not None:
            return stage
    return "outside"


# Regions: (marker text, region) per function; the marker is the start of the
# first line of the region's first statement, unique in the function.
REGIONS = {
    "solve_steady": [
        ("for iteration in range(start, self._max_simple_iter):", "loop"),
        ("if state is not None and walls is not None:", "momentum"),
        ("t1 = perf_counter()", "loop"),
        ("corrected = self._corrector.correct(prediction, p)", "pressure"),
        ("t2 = perf_counter()", "loop"),
        ("nu_t_old = state.nu_t", "turbulence"),
        ("viscosity_step = float(", "stopping"),
        ("t3 = perf_counter()", "loop"),
        ("u_prev, v_prev = u_c, v_c", "stopping"),
        ("if on_iteration is not None:", "loop"),
        ("if rule is None:", "stopping"),
        ("if iteration % 50 == 0 or stop:", "loop"),
        ("if not self.converged:", "finish"),
    ],
    "solve_timestep": [
        ("k = self._check_class(size_class)", "checks"),
        ("conditions = self._faces[k]", "advection"),
        ("divergence = adv_u[:, 1:]", "divergence"),
        ("c_star += dt * rate", "source"),
        ("c_star[self._solid] = 0.0", "divergence"),
        ("inflow = (", "booking"),
        ("if nu_t is None:", "implicit"),
        ("supplied = budget.initial", "booking"),
        ("c_new, leftover = self._implicit_step(", "implicit"),
        ("budget.leftover += leftover", "booking"),
    ],
}
# The label a function's own frame starts under, before its first marker.
FIRST_REGION = {"solve_steady": "setup", "solve_timestep": "checks"}
FIRST_REGION_LEAN = {"solve_steady": "setup", "solve_timestep": "inline"}

_MARKS: dict[tuple[object, int], str] = {}
_INSTALLED: list[Wrap] = []


def marker_lines(function: Callable, markers: list[tuple[str, str]]) -> dict[int, str]:
    """Line number of each marker in FUNCTION's source, refusing an absent or repeated one."""
    lines, first = inspect.getsourcelines(function)
    out: dict[int, str] = {}
    for text, region in markers:
        found = [
            first + i for i, line in enumerate(lines) if line.strip().startswith(text)
        ]
        if len(found) != 1:
            raise RuntimeError(f"marker {text!r} found {len(found)} times")
        out[found[0]] = region
    return out


def _on_line(code: object, line: int) -> object:
    region = _MARKS.get((code, line))
    if region is None:
        return sys.monitoring.DISABLE
    clock = CLOCK
    if clock is not None:
        clock.tick()
        head = clock.path[-1].partition(":")[0]
        clock.path = (*clock.path[:-1], f"{head}:{region}")
    return None


def install(
    specs: list[Wrap] | None = None,
    inline: dict[str, Callable] | None = None,
    regions: dict[str, list[tuple[str, str]]] | None = None,
    first_region: dict[str, str] | None = None,
) -> Clock:
    """Wrap every stage function, turn the region markers on, start a clock.

    The defaults are the solver's stages (``wraps``, ``REGIONS``,
    ``FIRST_REGION``); a test passes its own. ``inline`` maps a name in
    ``regions`` to the function whose lines carry the markers.
    """
    global CLOCK
    if _INSTALLED:
        raise RuntimeError("the timers are already installed")
    if specs is None:
        specs = wraps()
    if inline is None:
        inline = {"solve_steady": StaggeredSolver.solve_steady}
        if MARCH_REGIONS:
            inline["solve_timestep"] = TransportSolver.solve_timestep
    regions = REGIONS if regions is None else regions
    if first_region is None:
        first_region = FIRST_REGION if MARCH_REGIONS else FIRST_REGION_LEAN
    mon = sys.monitoring
    mon.use_tool_id(TOOL, "probe84")
    mon.register_callback(TOOL, mon.events.LINE, _on_line)
    for name, function in inline.items():
        code = function.__code__
        for line, region in marker_lines(function, regions[name]).items():
            _MARKS[(code, line)] = region
        mon.set_local_events(TOOL, code, mon.events.LINE)
    for spec in specs:
        raw = spec.owner.__dict__[spec.name] if isinstance(spec.owner, type) else None
        if raw is None:
            raw = getattr(spec.owner, spec.name)
        spec.original = raw
        if spec.name in first_region:
            # The inline functions' own frames start under their first region.
            spec.label = _first_region(spec.label, first_region[spec.name])
        if isinstance(raw, staticmethod):
            setattr(spec.owner, spec.name, staticmethod(_wrapper(spec, raw.__func__)))
        else:
            setattr(spec.owner, spec.name, _wrapper(spec, raw))
        _INSTALLED.append(spec)
    CLOCK = Clock()
    return CLOCK


def uninstall() -> Clock:
    """Restore every original and stop the clock; returns the stopped clock."""
    global CLOCK
    clock = CLOCK
    assert clock is not None
    clock.stop()
    CLOCK = None
    for spec in reversed(_INSTALLED):
        setattr(spec.owner, spec.name, spec.original)
    _INSTALLED.clear()
    mon = sys.monitoring
    for code, _line in list(_MARKS):
        mon.set_local_events(TOOL, code, 0)
    _MARKS.clear()
    mon.register_callback(TOOL, mon.events.LINE, None)
    mon.free_tool_id(TOOL)
    mon.restart_events()
    return clock


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


def config_path(grid: str) -> Path:
    """The scenario file of GRID: the stored run's own on 320x120."""
    if grid == "320x120":
        return STORED_CONFIG
    return CONFIGS / f"scenario_equipment_dust_{grid}.yaml"


def write_configs() -> None:
    """The stored 320x120 run's scenario with the grid and its continuity tolerance."""
    CONFIGS.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(STORED_CONFIG.read_text())
    for grid, (nx, ny, tol) in GRIDS.items():
        if grid == "320x120":
            continue
        out = json.loads(json.dumps(raw))
        out["domain"] = {"nx": nx, "ny": ny}
        out["solver"]["mass_imbalance_tol"] = tol
        config_path(grid).write_text(yaml.safe_dump(out, sort_keys=False))
        SimConfig(config_path(grid))  # refuses an invalid file now


def capped(grid: str, outer: int) -> SimConfig:
    """GRID's scenario with ``max_simple_iter`` set to OUTER."""
    raw = yaml.safe_load(config_path(grid).read_text())
    raw["solver"]["max_simple_iter"] = outer
    return SimConfig.from_dict(raw, base_dir=config_path(grid).parent)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def environment() -> dict:
    """What the process ran on."""
    from threadpoolctl import threadpool_info

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        commit = "unknown"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "machine": platform.processor() or platform.machine(),
        "commit": commit,
        "threads": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "blas": [
            {k: d.get(k) for k in ("internal_api", "num_threads", "version")}
            for d in threadpool_info()
        ],
    }


def flow_solver(config: SimConfig) -> tuple[StaggeredSolver, Mesh]:
    """The solver of the scenario's first interval, as ``_Run._solve`` builds it."""
    mesh = Mesh(config)
    interval = ScenarioManager(config, mesh).intervals()[0]
    icfg = interval.config
    return StaggeredSolver(mesh, icfg, StaggeredBoundary(mesh, icfg)), mesh


def flow_arrays(solver: StaggeredSolver) -> dict[str, np.ndarray]:
    """The arrays a flow solve ends with, for the bitwise comparison."""
    state = solver.steady_checkpoint.state if solver.steady_checkpoint else None
    assert state is not None
    out = {"u": state.u, "v": state.v, "p": state.p}
    for name in ("k", "eps", "nu_t"):
        value = getattr(state, name)
        if value is not None:
            out[name] = value
    out["residual_history"] = np.array(solver.residual_history)
    return out


def timed_flow(grid: str, outer: int, timers: bool) -> tuple[dict, dict]:
    """OUTER outer iterations from rest on GRID; the arrays and the record."""
    solver, _mesh = flow_solver(capped(grid, outer))
    assert solver._max_simple_iter == outer
    clock = install() if timers else None
    started = perf_counter_ns()
    solver.solve_steady()
    wall = perf_counter_ns() - started
    record = {"wall_ns": wall, "outer": len(solver.residual_history)}
    if clock is not None:
        record.update(uninstall().record())
    return flow_arrays(solver), record


def window_march(
    grid: str, flows_from: Path, t_start: float, t_stop: float, out: Path, timers: bool
) -> tuple[dict, dict]:
    """The march of GRID's run from the step at or before T_START to T_STOP.

    Starts from zero fields and fresh budgets, monitor and arrival maps,
    which is the run's own state at any step before the onset (module
    docstring). Returns the final fields and budgets and the record.
    """
    if not t_start < ONSET:
        raise ValueError("a window starts before the onset, where the fields are zero")
    for path in (out, out.with_name(out.stem + ".checkpoint.h5")):
        if path.exists():
            path.unlink()
    config = SimConfig(config_path(grid))
    run_file = RunFile(out)
    state = _Run(config, run_file, None)
    given = Provenance.here(environment()["commit"], 0)
    run_file.write_config(
        config.merged_mapping(),
        commit=given.commit,
        started=given.started,
        machine=given.machine,
        concurrent=given.concurrent,
    )
    state.flow_states(RunFile(flows_from), False)
    plans = state.plans()
    first = next(
        p for p, plan in enumerate(plans) if plan.t_start <= t_start < plan.t_end
    )
    i0 = math.floor((t_start - plans[first].t_start) / plans[first].dt)
    last = next(
        p for p, plan in enumerate(plans) if plan.t_start < t_stop <= plan.t_end
    )
    stop = plans[last]
    i1 = round((t_stop - stop.t_start) / stop.dt)
    truncated = list(plans[: last + 1])
    if i1 < stop.steps:
        truncated[last] = dataclasses.replace(stop, steps=i1, t_end=stop.time(i1))
    step = sum(plan.steps for plan in plans[:first]) + i0
    t = plans[first].time(i0)
    fields = state.manager.initial_fields()
    if any(np.any(c != 0.0) for c in fields.values()):
        raise ValueError("the scenario's initial fields are not zero")
    for c in fields.values():
        c.flags.writeable = False
    budgets = tuple(MassBudget() for _ in range(state.n_classes))
    clock = install() if timers else None
    started = perf_counter_ns()
    fields, budgets, end_step, end_t, _held = state.march(
        truncated, (first, i0), fields, budgets, step, t
    )
    wall = perf_counter_ns() - started
    record = {
        "wall_ns": wall,
        "start_step": step,
        "start_t": t,
        "end_step": end_step,
        "end_t": end_t,
        "steps": end_step - step,
        "plans": [dataclasses.asdict(p) for p in truncated[first:]],
    }
    if clock is not None:
        record.update(uninstall().record())
    return {"fields": fields, "budgets": budgets}, record


def same_fields(a: dict, b: dict) -> list[str]:
    """Names of the arrays or budget terms that differ bitwise (empty when none)."""
    differ = []
    for k in a["fields"]:
        if a["fields"][k].tobytes() != b["fields"][k].tobytes():
            differ.append(f"C_{k}")
    for k, (x, y) in enumerate(zip(a["budgets"], b["budgets"], strict=True)):
        for f in dataclasses.fields(x):
            u, w = getattr(x, f.name), getattr(y, f.name)
            if isinstance(u, dict):
                same = u.keys() == w.keys() and all(
                    np.float64(u[s]).tobytes() == np.float64(w[s]).tobytes() for s in u
                )
            else:
                same = (
                    np.float64(np.nan if u is None else u).tobytes()
                    == np.float64(np.nan if w is None else w).tobytes()
                )
            if not same:
                differ.append(f"budget {k}.{f.name}")
    return differ


def save(job: str, record: dict) -> None:
    """Write RECORD with the environment to results/builder84/JOB.json."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    record = {
        "job": job,
        "environment": environment(),
        "march_regions": MARCH_REGIONS,
        **record,
    }
    (RESULTS / f"{job}.json").write_text(json.dumps(record, indent=1, default=float))
    print(f"wrote {RESULTS / f'{job}.json'}")


def cells_of(grid: str) -> int:
    nx, ny, _ = GRIDS[grid]
    return nx * ny


def calibrate_flow(grid: str, outer: int) -> None:
    """OUTER outer iterations from rest, unwrapped and wrapped in turn (U W U W U).

    Every run's arrays are compared bitwise with the first's; the overhead is
    the wrapped runs' mean wall less the unwrapped runs' mean.
    """
    runs = [timed_flow(grid, outer, timers) for timers in ORDER]
    first = runs[0][0]
    differ = [
        [n for n in first if first[n].tobytes() != a[n].tobytes()] for a, _ in runs
    ]
    save(
        f"calibrate_flow_{grid}",
        {
            "grid": grid,
            "cells": cells_of(grid),
            "outer": outer,
            "order": ["wrapped" if t else "unwrapped" for t in ORDER],
            "wall_ns": [r["wall_ns"] for _, r in runs],
            "differ": differ,
            "wrapped": [r for (_, r), t in zip(runs, ORDER, strict=True) if t],
        },
    )
    print(grid, "differ", differ, "walls", [r["wall_ns"] for _, r in runs])


def calibrate_march(grid: str, t_stop: float, flows_from: Path) -> None:
    """A window from 9 s to T_STOP, unwrapped and wrapped in turn (U W U W U)."""
    out = RESULTS / "windows"
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for n, timers in enumerate(ORDER):
        fields, record = window_march(
            grid, flows_from, 9.0, t_stop, out / f"calibrate_{grid}_{n}.h5", timers
        )
        runs.append((fields, record))
    differ = [same_fields(runs[0][0], f) for f, _ in runs]
    save(
        f"calibrate_march_{grid}",
        {
            "grid": grid,
            "cells": cells_of(grid),
            "t_stop": t_stop,
            "order": ["wrapped" if t else "unwrapped" for t in ORDER],
            "wall_ns": [r["wall_ns"] for _, r in runs],
            "differ": differ,
            "wrapped": [r for (_, r), t in zip(runs, ORDER, strict=True) if t],
        },
    )
    print(grid, "differ", differ, [r["wall_ns"] for _, r in runs])


def whole_run(grid: str) -> None:
    """The scenario on GRID with the timers on, flow solve from rest."""
    out = RESULTS / "runs" / f"equipment_dust_{grid}.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    for path in (out, out.with_name(out.stem + ".checkpoint.h5")):
        if path.exists():
            path.unlink()
    config = SimConfig(config_path(grid))
    install()
    started = perf_counter_ns()
    record = time_integration.run(
        config, out, provenance=Provenance.here(environment()["commit"], 0)
    )
    wall = perf_counter_ns() - started
    timed = uninstall().record()
    flows = [dataclasses.asdict(f) for f in record.flows]
    save(
        f"run_{grid}",
        {
            "grid": grid,
            "cells": cells_of(grid),
            "wall_ns": wall,
            "steps": record.steps,
            "t_final": record.t_final,
            "flows": flows,
            "plans": [dataclasses.asdict(p) for p in record.intervals],
            "budgets": [dataclasses.asdict(b) for b in record.budgets],
            **timed,
        },
    )


def first_outer(grid: str, outer: int) -> None:
    """The first OUTER outer iterations from rest on GRID, timed."""
    _arrays, record = timed_flow(grid, outer, True)
    save(f"flow_{grid}", {"grid": grid, "cells": cells_of(grid), **record})


def window(grid: str, t_start: float, t_stop: float) -> None:
    """The timed window on GRID from the stored state, checked against the stored frames.

    With the march's regions on (``--regions``) it is the split run of the
    window, ``split_window_<grid>``.
    """
    job = f"{'split_window' if MARCH_REGIONS else 'window'}_{grid}"
    out = RESULTS / "windows" / f"{job}.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    flows = (
        STORED if grid == "320x120" else RESULTS / "runs" / f"equipment_dust_{grid}.h5"
    )
    _fields, record = window_march(grid, flows, t_start, t_stop, out, True)
    record["frames_checked"] = compare_frames(out, flows)
    save(job, {"grid": grid, "cells": cells_of(grid), **record})


def compare_frames(window_file: Path, stored: Path) -> dict:
    """Per frame step the window shares with the stored run: whether every class is bitwise equal."""
    import h5py

    out = {}
    with h5py.File(window_file, "r") as w, h5py.File(stored, "r") as s:
        steps_w = list(w["transport/steps"][:])
        steps_s = list(s["transport/steps"][:])
        for i, step in enumerate(steps_w):
            if step not in steps_s:
                continue
            j = steps_s.index(step)
            same = all(
                w[f"transport/C_{k}"][i].tobytes() == s[f"transport/C_{k}"][j].tobytes()
                for k in range(5)
            )
            budgets = (
                w["transport/budgets"][i].tobytes()
                == s["transport/budgets"][j].tobytes()
            )
            out[str(int(step))] = {"fields": same, "budgets": budgets}
    return out


def construct(grid: str, repeats: int = 5) -> None:
    """The march's per-interval work on GRID: building its TransportSolver and faces.

    ``_Run.march`` does it once per interval, inside its own frame, so the
    march remainder holds it beside the per-step glue; tables84.py takes it
    out before scaling the remainder by the steps.
    """
    config = SimConfig(config_path(grid))
    out = RESULTS / "windows" / f"construct_{grid}.h5"
    for path in (out, out.with_name(out.stem + ".checkpoint.h5")):
        if path.exists():
            path.unlink()
    run_file = RunFile(out)
    state = _Run(config, run_file, None)
    run_file.write_config(
        config.merged_mapping(), commit="", started="", machine="", concurrent=0
    )
    flows = (
        STORED if grid == "320x120" else RESULTS / "runs" / f"equipment_dust_{grid}.h5"
    )
    state.flow_states(RunFile(flows), False)
    interval = state.intervals[1]
    walls = []
    for _ in range(repeats):
        started = perf_counter_ns()
        solver = state.transport(interval)
        solver.restore_budgets(tuple(MassBudget() for _ in range(state.n_classes)))
        state.faces(interval.fingerprint)
        state.manager.sources_at(interval.t_start)
        walls.append(perf_counter_ns() - started)
    save(f"construct_{grid}", {"grid": grid, "wall_ns": walls})


def profile_march(grid: str) -> None:
    """cProfile of GRID's march, its flow state copied from the timed run."""
    out = RESULTS / "runs" / f"cprofile_{grid}.h5"
    for path in (out, out.with_name(out.stem + ".checkpoint.h5")):
        if path.exists():
            path.unlink()
    config = SimConfig(config_path(grid))
    flows = RESULTS / "runs" / f"equipment_dust_{grid}.h5"
    profiler = cProfile.Profile()
    started = perf_counter_ns()
    profiler.enable()
    record = time_integration.run(config, out, flows_from=flows)
    profiler.disable()
    wall = perf_counter_ns() - started
    stats = pstats.Stats(profiler)
    stats.dump_stats(str(RESULTS / f"cprofile_{grid}.prof"))
    rows = []
    for (path, line, name), (cc, nc, tt, ct, _callers) in stats.stats.items():
        rows.append(
            {
                "function": f"{Path(path).name}:{line}({name})",
                "file": str(Path(path).as_posix()),
                "primitive_calls": cc,
                "calls": nc,
                "tottime": tt,
                "cumtime": ct,
            }
        )
    rows.sort(key=lambda r: r["tottime"], reverse=True)
    save(
        f"cprofile_{grid}",
        {
            "grid": grid,
            "wall_ns": wall,
            "steps": record.steps,
            "total_tt": sum(r["tottime"] for r in rows),
            "top": rows[:60],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command")
    parser.add_argument("args", nargs="*")
    parser.add_argument(
        "--regions", action="store_true", help="the march's line regions on"
    )
    args = parser.parse_args(argv)
    global MARCH_REGIONS
    MARCH_REGIONS = args.regions
    a = args.args
    if args.command == "configs":
        write_configs()
    elif args.command == "calibrate-flow":
        calibrate_flow(a[0], int(a[1]))
    elif args.command == "calibrate-march":
        flows = (
            STORED
            if a[0] == "320x120"
            else RESULTS / "runs" / f"equipment_dust_{a[0]}.h5"
        )
        calibrate_march(a[0], float(a[1]), flows)
    elif args.command == "run":
        whole_run(a[0])
    elif args.command == "flow":
        first_outer(a[0], int(a[1]))
    elif args.command == "window":
        window(a[0], float(a[1]), float(a[2]))
    elif args.command == "cprofile":
        profile_march(a[0])
    elif args.command == "construct":
        construct(a[0])
    else:
        parser.error(f"unknown command {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
