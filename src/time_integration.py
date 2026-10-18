"""Tier 1 time integration: piecewise-steady flow, time-accurate transport (ADR-014 A).

A scenario's run (ADR-014 G, step 4). The scenario manager turns the events
into intervals, each with one steady flow state keyed by its fingerprint.
Each distinct state is solved once and kept in the run file: the first from
rest unless a run file holds it, each later one warm-started from the state
of the interval before it (``solve_steady(initial=...)``), and a return to a
state already held costs nothing. The particles are then marched through the
intervals at one step for every class, so the frames and the sensors share a
clock. Nothing is discretised here: the flow is the steady solver's, the
step is the transport solver's, and the event model is the scenario
manager's.

The hand-over. At a switch the class fields are carried across unchanged and
a transport solver is built on the new interval's concentration boundary
with the budgets the last one held (``TransportSolver.restore_budgets``), so
the budget's current mass is the same number. The eddy viscosity the
transport reads switches with the faces. ``stable_dt`` is recomputed on the
new faces, the smallest over the classes, and the interval is divided into
the fewest equal steps no larger: step i of an interval from t0 is at
``t0 + i dt`` and its last step is at the interval's end exactly, so a
switch lands on a step boundary and no time is accumulated.

The sources. An interval's sources are ``ScenarioManager.sources_at`` at its
start, constant over it since every event starts and ends on an interval
boundary, passed as ``solve_timestep``'s ``sources`` so the budget books them
in its source slot; a class with no active source is passed None, the step
without one, bitwise.

The records (ADR-014 D). The run file holds the merged mapping, every flow
state, the frames every ``output_interval`` steps and at the last step, the
budgets with them, and the sensor readings of every step, held and appended
at each frame and before each checkpoint (prompt 64's decision 3), so a
restart leaves no gap whatever the two intervals are. With
``checkpoint_interval`` the march's checkpoint is rewritten every that many
steps and at the last, after that step's frame, and a steady solve hands its
own checkpoint at the same cadence. A restart checks the configuration,
reuses the flow states, resumes a capped steady solve, and continues the
march from its checkpoint, bitwise the uninterrupted run; with a later
``t_end`` the interval the checkpoint lies in is divided again from the
checkpoint's time.

The clean-out (ADR-014 C, a scenario with an initial event). A sensor reads
the cell containing it, the one below or to the left of it when it sits on
a face (``sensor_cells``). A class is within spec when every sensor reads below
its threshold, the room when every class is. Updated every step and carried
in the checkpoint (prompt 64's decision 1): the last step time each cell was
at or above its class's threshold, when each sensor and the free-area mean
first fell to a hundredth of the initial level, and the start of the current
within-spec stretch per class and for the room. With ``scenario.hold_time``
the run stops once the room has held within spec that long (decision 2). At
the end the recovery outputs are written to ``/recovery/``.
"""

import logging
import math
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from src.boundary_staggered import StaggeredBoundary
from src.config import Initial, SimConfig
from src.io_manager import RecoveryRecord, RunFile
from src.mesh import SOLID, Mesh
from src.particles import ParticlePhysics
from src.scenarios import Interval, ScenarioManager
from src.solver_staggered import FlowState, StaggeredSolver, SteadyCheckpoint
from src.solver_transport import MassBudget, TransportSolver
from src.staggered import FaceVelocities

logger = logging.getLogger(__name__)

# ISO 14644-3's recovery challenge: the 100:1 time is the time a reading takes
# to fall to a hundredth of its initial level (ADR-014 C).
HUNDRED_TO_ONE = 100.0
# The 100:1 time is defined where the initial level is at least 100 times the
# threshold. The clean-out sets it to exactly that, and an initial stated as a
# concentration can carry the rounding of the product (35.2 x 100 is
# 3520.0000000000005), so the comparison is read to this relative tolerance.
DEFINED_RTOL = 1.0e-9
# A sensor this fraction of the room's extent from a face is on it. The faces
# carry the rounding of the mesh's coordinates (about 1e-16 of the extent),
# and the room's sensors are stated on multiples of 0.05 m that fall on faces.
SENSOR_FACE_RTOL = 1.0e-9
# The sections and solver keys a steady flow solve does not read, so a state
# kept in another run file is this run's state when every other key agrees
# and the fingerprint does. The boundaries are compared by the fingerprint.
FLOW_FREE_SECTIONS: tuple[str, ...] = (
    "scenario",
    "boundaries",
    "transport",
    "particles",
    "sensors",
    "thresholds",
)
FLOW_FREE_SOLVER_KEYS: tuple[str, ...] = (
    "dt",
    "t_end",
    "output_interval",
    "checkpoint_interval",
    "max_simple_iter",
)
# How a run came by each of its flow states.
REST = "rest"
WARM = "warm"
RESUMED = "resumed"
RUN_FILE = "run file"
KEPT = "kept"
SOLVED_ORIGINS: tuple[str, ...] = (REST, WARM, RESUMED)


class FlowNotConvergedError(RuntimeError):
    """A flow state stopped without converging; the march does not start on it.

    A state capped at ``max_simple_iter`` leaves its steady checkpoint in the
    checkpoint file, so a restart under a larger cap continues the solve. A
    state whose fields went non-finite leaves nothing.

    Parameters
    ----------
    fingerprint : str
        The state's key.
    stop_reason : str or None
        The solve's ``stop_reason``.
    outer : int
        Outer iterations taken.
    """

    def __init__(self, fingerprint: str, stop_reason: str | None, outer: int) -> None:
        self.fingerprint = fingerprint
        self.stop_reason = stop_reason
        self.outer = outer
        super().__init__(
            f"flow state {fingerprint[:12]} stopped by {stop_reason} after {outer} "
            "outer iterations without converging; the march needs a converged "
            "state (raise solver.max_simple_iter and restart to continue a capped "
            "solve)"
        )


@dataclass(frozen=True)
class Provenance:
    """What a run file's ``/config`` records about where the run was made.

    Parameters
    ----------
    commit, started, machine : str
        The commit, the start time and the machine.
    concurrent : int
        Processes running beside the run, non-negative.
    """

    commit: str
    started: str
    machine: str
    concurrent: int

    @classmethod
    def here(cls, commit: str = "unknown", concurrent: int = 0) -> "Provenance":
        """This machine now, with the commit and process count given.

        Parameters
        ----------
        commit : str, optional
            The commit the code was taken from; ``scripts/run_scenario.py``
            asks git.
        concurrent : int, optional
            Processes running beside the run.

        Returns
        -------
        Provenance
            The start time to the second and ``platform.node()``.
        """
        return cls(
            commit=commit,
            started=datetime.now().isoformat(timespec="seconds"),
            machine=platform.node(),
            concurrent=concurrent,
        )


@dataclass(frozen=True)
class FlowSolve:
    """How a run came by one of its flow states.

    Parameters
    ----------
    fingerprint : str
        The state's key.
    origin : str
        ``REST``, ``WARM`` (from the state named by ``warm_from``),
        ``RESUMED`` (a capped solve continued from the checkpoint file),
        ``RUN_FILE`` (read from the run file a restart continues) or
        ``KEPT`` (read from another run's file, ``flows_from``).
    outer : int
        Outer iterations of the solve that produced it (all of them for a
        resumed solve).
    stop_reason : str or None
        That solve's stop.
    wall_seconds : float
        Wall time of the solve in this run; the kept solve's for a state
        read from a file.
    warm_from : str or None
        The fingerprint a warm start began from.
    """

    fingerprint: str
    origin: str
    outer: int
    stop_reason: str | None
    wall_seconds: float
    warm_from: str | None = None

    @property
    def solved(self) -> bool:
        """Whether this run solved the state rather than reading it."""
        return self.origin in SOLVED_ORIGINS


@dataclass(frozen=True)
class IntervalPlan:
    """An interval as the march steps it.

    Parameters
    ----------
    index : int
        Its place in ``ScenarioManager.intervals()``.
    t_start, t_end : float
        Seconds. A restart whose checkpoint lies inside an interval it must
        divide again starts that interval's plan at the checkpoint's time.
    fingerprint : str
        The flow state it steps on.
    events : tuple[str, ...]
        The events active over it.
    steps : int
        Equal steps from ``t_start`` to ``t_end``.
    dt : float
        ``(t_end - t_start) / steps``, at most ``stable_dt``.
    stable_dt : float
        The smallest ``stable_dt`` over the classes on its faces.
    """

    index: int
    t_start: float
    t_end: float
    fingerprint: str
    events: tuple[str, ...]
    steps: int
    dt: float
    stable_dt: float

    def time(self, i: int) -> float:
        """The time of step i of this interval, its end exactly at i = steps."""
        if i == self.steps:
            return self.t_end
        return self.t_start + i * self.dt


@dataclass(frozen=True, eq=False)
class MarchStep:
    """What ``on_step`` is handed after each step's writes.

    Parameters
    ----------
    step : int
        Steps taken, the run's count.
    t : float
        Simulated time after the step, seconds.
    dt : float
        The step just taken.
    interval : int
        The interval's index.
    fields : Mapping[int, np.ndarray]
        Class to its field after the step, read-only.
    budgets : tuple[MassBudget, ...]
        Copies of the transport solver's budgets after the step.
    """

    step: int
    t: float
    dt: float
    interval: int
    fields: Mapping[int, np.ndarray]
    budgets: tuple[MassBudget, ...]


@dataclass(frozen=True, eq=False)
class RunRecord:
    """What a run did, as ``scripts/run_scenario.py`` summarises it.

    Parameters
    ----------
    path : Path
        The run file.
    intervals : tuple[IntervalPlan, ...]
        Every interval as planned from t = 0; a restart that divided one
        again from its checkpoint lists that part as ``resumed_plan``.
    flows : tuple[FlowSolve, ...]
        One per distinct flow state, in the order the intervals first meet
        them.
    steps : int
        Steps taken by the end, the run's count.
    t_final : float
        Simulated time at the end, seconds.
    held : bool
        Whether a clean-out stopped because the room held within spec.
    wall_seconds : float
        Wall time of this call.
    budgets : tuple[MassBudget, ...]
        Copies of the budgets at the end, one per class.
    recovery : RecoveryRecord or None
        A clean-out's outputs; None for a scenario without an initial event.
    restarted_from : int or None
        The checkpoint step a restart continued from, 0 for a restart that
        found none and marched again; None for a new run.
    resumed_plan : IntervalPlan or None
        The interval a restart divided again from its checkpoint's time.
    """

    path: Path
    intervals: tuple[IntervalPlan, ...]
    flows: tuple[FlowSolve, ...]
    steps: int
    t_final: float
    held: bool
    wall_seconds: float
    budgets: tuple[MassBudget, ...]
    recovery: RecoveryRecord | None
    restarted_from: int | None = None
    resumed_plan: IntervalPlan | None = None


def _cell_holding(faces: np.ndarray, position: float) -> int:
    """The index of the cell whose faces hold POSITION, the lower one on a face."""
    tolerance = SENSOR_FACE_RTOL * float(faces[-1] - faces[0])
    index = int(np.searchsorted(faces, position - tolerance, side="left")) - 1
    return min(max(index, 0), faces.size - 2)


def sensor_cells(config: SimConfig, mesh: Mesh) -> tuple[tuple[int, int], ...]:
    """The (j, i) of the cell each sensor reads, in the configuration's order.

    The cell containing the sensor (ADR-014 C). The room's sensors sit on
    cell faces on every grid that puts its positions on faces, so a sensor
    within ``SENSOR_FACE_RTOL`` of the room's extent of a face reads the
    cell below it or to its left, on each axis. On 160x60 that is the cell
    the probe of ADR-014 T7 read for every sensor, by the nearest centre;
    the nearest centre itself is not a rule on a face, since which of the
    two equidistant centres is nearer is decided by rounding (on 80x30 it
    puts above_gap_1 to the right of its face and the other three to the
    left).

    Parameters
    ----------
    config : SimConfig
        Supplies the sensors.
    mesh : Mesh
        The run's mesh.

    Returns
    -------
    tuple[tuple[int, int], ...]
        One (j, i) per sensor.

    Raises
    ------
    ValueError
        If a sensor's cell is SOLID.
    """
    cells = []
    for sensor in config.sensors:
        i = _cell_holding(mesh.x, sensor.x)
        j = _cell_holding(mesh.y, sensor.y)
        if mesh.cell_type[j, i] == SOLID:
            raise ValueError(f"sensor {sensor.name} reads a SOLID cell ({j}, {i})")
        cells.append((j, i))
    return tuple(cells)


def air_change_rate(state: FlowState, mesh: Mesh) -> float:
    """N, the inflow through the domain's edges over the free area, 1/s.

    Parameters
    ----------
    state : FlowState
        A flow state of the run's mesh.
    mesh : Mesh
        Supplies the face lengths and the cell types.

    Returns
    -------
    float
        The volumetric inflow per metre of depth over the non-SOLID area
        (ADR-014 C's well-mixed rate, T6). A vent's discharge is the room's
        own air and is not counted.
    """
    u, v = state.u, state.v
    dy, dx = mesh.dy_cell, mesh.dx_cell
    inflow = (
        np.sum(np.clip(u[:, 0], 0.0, None) * dy)
        + np.sum(np.clip(-u[:, -1], 0.0, None) * dy)
        + np.sum(np.clip(v[0, :], 0.0, None) * dx)
        + np.sum(np.clip(-v[-1, :], 0.0, None) * dx)
    )
    free = float(np.outer(dy, dx)[mesh.cell_type != SOLID].sum())
    return float(inflow) / free


def flow_inputs(mapping: Mapping[str, object]) -> dict:
    """The parts of a configuration mapping a steady flow solve reads, but its boundaries.

    Parameters
    ----------
    mapping : Mapping[str, object]
        A merged configuration mapping.

    Returns
    -------
    dict
        A copy without ``FLOW_FREE_SECTIONS`` and the solver's
        ``FLOW_FREE_SOLVER_KEYS``. Two runs whose flow inputs are equal
        solve the same state for the same fingerprint.
    """
    out = {k: v for k, v in mapping.items() if k not in FLOW_FREE_SECTIONS}
    solver = out.get("solver")
    if isinstance(solver, Mapping):
        out["solver"] = {
            k: v for k, v in solver.items() if k not in FLOW_FREE_SOLVER_KEYS
        }
    return out


def _first_flow_difference(kept: dict, ours: dict) -> str | None:
    """The first section or section.key where two flow inputs differ, in KEPT's order."""
    for section in list(kept) + [s for s in ours if s not in kept]:
        a, b = kept.get(section), ours.get(section)
        if a == b:
            continue
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in list(a) + [k for k in b if k not in a]:
                if a.get(key) != b.get(key) or (key in a) != (key in b):
                    return f"{section}.{key}"
        return section
    return None


class RecoveryTracker:
    """A clean-out's recovery readings, updated every step (ADR-014 C).

    Parameters
    ----------
    thresholds, initial : Sequence[float]
        Per class: the threshold and the initial level C0, particles per
        cubic metre.
    cells : Sequence[tuple[int, int]]
        The (j, i) each sensor reads (``sensor_cells``); at least one.
    live : np.ndarray
        [ny, nx] bool, the non-SOLID cells.
    hold_time : float or None
        ``scenario.hold_time``.

    Raises
    ------
    ValueError
        If there is no sensor, or the per-class sequences differ in length.

    Notes
    -----
    Within spec means below the threshold, so the local field's "above" is
    "not below": a cell exactly at its threshold has not recovered. The
    tracker's arrays are its whole state (``state`` and ``restore``), so a
    restart from a checkpoint continues it bitwise.
    """

    def __init__(
        self,
        thresholds: Sequence[float],
        initial: Sequence[float],
        cells: Sequence[tuple[int, int]],
        live: np.ndarray,
        hold_time: float | None,
    ) -> None:
        if not cells:
            raise ValueError(
                "a clean-out needs at least one sensor: the room is within spec when "
                "every sensor reads below its threshold (ADR-014 C)"
            )
        if len(thresholds) != len(initial):
            raise ValueError(
                f"{len(thresholds)} thresholds and {len(initial)} initial levels"
            )
        self._thresholds = np.asarray(thresholds, dtype=np.float64)
        self._initial = np.asarray(initial, dtype=np.float64)
        self._target = self._initial / HUNDRED_TO_ONE
        self._defined = self._initial >= HUNDRED_TO_ONE * self._thresholds * (
            1.0 - DEFINED_RTOL
        )
        self._rows = np.array([j for j, _ in cells])
        self._cols = np.array([i for _, i in cells])
        self._live = live
        self._hold = hold_time
        n = self._thresholds.shape[0]
        self.last_above = np.full((n, *live.shape), np.nan)
        self.sensor_time = np.full((n, len(cells)), np.nan)
        self.mean_time = np.full(n, np.nan)
        self.class_since = np.full(n, np.nan)
        self.room_since = np.full((), np.nan)
        self.t = np.full((), np.nan)

    def update(
        self, t: float, fields: Mapping[int, np.ndarray], means: Sequence[float]
    ) -> None:
        """Read the fields after a step (or the initial fields at t = 0).

        Parameters
        ----------
        t : float
            The step's time, seconds, above the last update's.
        fields : Mapping[int, np.ndarray]
            Class to its field, [ny, nx].
        means : Sequence[float]
            Per class, the free-area mean: the content over the non-SOLID
            area.
        """
        readings = np.array([fields[k][self._rows, self._cols] for k in fields])
        for k, threshold in enumerate(self._thresholds):
            above = (fields[k] >= threshold) & self._live
            self.last_above[k][above] = t
        within = (readings < self._thresholds[:, None]).all(axis=1)
        self.class_since = np.where(
            within, np.where(np.isnan(self.class_since), t, self.class_since), np.nan
        )
        if within.all():
            if np.isnan(self.room_since):
                self.room_since = np.full((), t)
        else:
            self.room_since = np.full((), np.nan)
        reached = (readings <= self._target[:, None]) & self._defined[:, None]
        first = reached & np.isnan(self.sensor_time)
        self.sensor_time[first] = t
        fell = (np.asarray(means) <= self._target) & self._defined
        self.mean_time[fell & np.isnan(self.mean_time)] = t
        self.t = np.full((), t)

    def held(self) -> bool:
        """Whether the room has been within spec for the hold time (False without one)."""
        if self._hold is None or np.isnan(self.room_since):
            return False
        return bool(self.t - self.room_since >= self._hold)

    def state(self) -> dict[str, np.ndarray]:
        """The tracker's arrays, for the march's checkpoint."""
        return {
            "last_above": self.last_above,
            "sensor_time": self.sensor_time,
            "mean_time": self.mean_time,
            "class_since": self.class_since,
            "room_since": self.room_since,
            "t": self.t,
        }

    def restore(self, state: Mapping[str, np.ndarray]) -> None:
        """Continue from ``state()`` as a checkpoint carried it.

        Raises
        ------
        KeyError
            If an array is missing.
        ValueError
            If an array's shape is not this tracker's.
        """
        for name, mine in self.state().items():
            given = np.array(state[name], dtype=np.float64)
            if given.shape != mine.shape:
                raise ValueError(
                    f"the checkpoint's {name} has shape {given.shape}; this run's "
                    f"is {mine.shape}"
                )
            setattr(self, name, given)

    def _recovered(self, since: np.ndarray) -> np.ndarray:
        """The start of a stretch that reaches the end, if it lasted the hold."""
        if self._hold is None:
            return since
        return np.where(self.t - since >= self._hold, since, np.nan)

    def record(self, names: Sequence[str], rate: float, held: bool) -> RecoveryRecord:
        """The recovery outputs at the end of the run.

        Parameters
        ----------
        names : Sequence[str]
            The sensors' names.
        rate : float
            N, ``air_change_rate`` of the first interval's flow state.
        held : bool
            Whether the run stopped on the hold.

        Returns
        -------
        RecoveryRecord
            The room's and each class's recovery time: the start of the
            within-spec stretch that reaches the end of the run, NaN when
            the room is not within spec at the end or, with a hold, when the
            stretch is shorter than it. The 100:1 times, NaN where not
            defined or not reached; their ratio to ln(100) / N for the
            free-area mean; and the local field, zero where a cell was never
            at or above its threshold, inf where it still is, NaN in SOLID
            cells, with its maximum over the non-SOLID cells.
        """
        local = self.last_above.copy()
        live = np.broadcast_to(self._live, local.shape)
        local[live & (local == self.t)] = np.inf
        local[live & np.isnan(local)] = 0.0
        largest = np.array([np.max(field[self._live]) for field in local])
        well_mixed = math.log(HUNDRED_TO_ONE) / rate
        return RecoveryRecord(
            sensor_names=tuple(names),
            thresholds=self._thresholds,
            initial=self._initial,
            hold_time=self._hold,
            held=held,
            t_final=float(self.t),
            air_change_rate=rate,
            well_mixed_time=well_mixed,
            room_recovery_time=float(self._recovered(self.room_since)),
            recovery_time=self._recovered(self.class_since),
            defined=self._defined,
            sensor_time=self.sensor_time,
            mean_time=self.mean_time,
            ratio=self.mean_time / well_mixed,
            local_recovery=local,
            local_recovery_max=largest,
        )


def smallest_stable_dt(
    solver: TransportSolver, faces: FaceVelocities, n_classes: int
) -> float:
    """The smallest ``stable_dt`` over the classes, the one step every class takes.

    Parameters
    ----------
    solver : TransportSolver
        Answers ``stable_dt(faces, k)``.
    faces : FaceVelocities
        The interval's faces.
    n_classes : int
        The configured classes.

    Returns
    -------
    float
        Seconds; inf when nothing moves. The classes differ by their
        settling velocities, so the largest would step a faster-settling
        class past its bound.
    """
    return min(solver.stable_dt(faces, k) for k in range(n_classes))


def plan_interval(
    index: int,
    interval: Interval,
    stable_dt: float,
    t_start: float | None = None,
) -> IntervalPlan:
    """Divide an interval into the fewest equal steps no larger than ``stable_dt``.

    Parameters
    ----------
    index : int
        The interval's index.
    interval : Interval
        One of ``ScenarioManager.intervals()``.
    stable_dt : float
        The smallest ``stable_dt`` over the classes on its faces; inf when
        nothing moves.
    t_start : float, optional
        Where the plan starts; the interval's start when None (a restart
        divides the rest of an interval from its checkpoint's time).

    Returns
    -------
    IntervalPlan
        At least one step.
    """
    t0 = interval.t_start if t_start is None else t_start
    span = interval.t_end - t0
    steps = max(1, math.ceil(span / stable_dt))
    # The division can round dt one ulp above the bound, which solve_timestep
    # refuses; one more step brings it back under.
    while span / steps > stable_dt:
        steps += 1
    return IntervalPlan(
        index=index,
        t_start=t0,
        t_end=interval.t_end,
        fingerprint=interval.fingerprint,
        events=interval.events,
        steps=steps,
        dt=span / steps,
        stable_dt=stable_dt,
    )


def _copies(budgets: Sequence[MassBudget]) -> tuple[MassBudget, ...]:
    """Copies of budgets that share nothing with them."""
    return tuple(replace(b, deposited=dict(b.deposited)) for b in budgets)


def _rule_readings(solver: StaggeredSolver, mesh: Mesh) -> dict[str, float]:
    """What the solver shows of its stop: the last residual and the continuity."""
    live = mesh.cell_type != SOLID
    imbalance = solver.last_mass_imbalance[live]
    return {
        "residual": float(solver.residual_history[-1]),
        "worst_imbalance": float(np.max(np.abs(imbalance))),
        "signed_imbalance": float(np.sum(imbalance)),
        "pressure_cap_hits": float(solver.pressure_cap_hits),
    }


class _Run:
    """One call of ``run``: the state the phases share."""

    def __init__(
        self,
        config: SimConfig,
        run: RunFile,
        on_step: Callable[[MarchStep], None] | None,
    ) -> None:
        self.config = config
        self.run = run
        self.on_step = on_step
        self.mesh = Mesh(config)
        self.manager = ScenarioManager(config, self.mesh)
        self.intervals = self.manager.intervals()
        self.n_classes = len(config.particle_sizes)
        self.live = self.mesh.cell_type != SOLID
        self.free_area = float(
            np.outer(self.mesh.dy_cell, self.mesh.dx_cell)[self.live].sum()
        )
        self.cells = sensor_cells(config, self.mesh)
        self.names = [sensor.name for sensor in config.sensors]
        self.states: dict[str, FlowState] = {}
        self.flows: list[FlowSolve] = []
        spec = config.scenario
        assert spec is not None
        self.tracker: RecoveryTracker | None = None
        if any(isinstance(event.change, Initial) for event in spec.events):
            initial = self.manager.initial_fields()
            self.tracker = RecoveryTracker(
                config.class_thresholds("the clean-out's recovery"),
                [float(initial[k][self.live][0]) for k in range(self.n_classes)],
                self.cells,
                self.live,
                spec.hold_time,
            )

    # -- Flow states ---------------------------------------------------------

    def flow_states(self, kept: RunFile | None, restart: bool) -> None:
        """Every distinct flow state, read, kept, resumed or solved, in interval order."""
        held = set(self.run.flow_fingerprints())
        kept_held = set(kept.flow_fingerprints()) if kept is not None else set()
        steady: tuple[str, SteadyCheckpoint] | None = None
        if restart:
            try:
                steady = self.run.read_steady_checkpoint()
            except KeyError:
                steady = None
        previous: tuple[str, FlowState] | None = None
        for interval in self.intervals:
            key = interval.fingerprint
            if key not in self.states:
                if key in held:
                    self._read(self.run, key, RUN_FILE)
                elif key in kept_held:
                    assert kept is not None
                    self._read(kept, key, KEPT, copy=True)
                else:
                    resume = None
                    if steady is not None and steady[0] == key:
                        resume = steady[1]
                    self._solve(interval, previous, resume)
            previous = (key, self.states[key])

    def _read(self, source: RunFile, key: str, origin: str, copy: bool = False) -> None:
        """A state from a run file; with ``copy`` written into this run's."""
        record = source.read_flow_state(key)
        if copy:
            self.run.write_flow_state(
                key,
                record.state,
                wall_seconds=record.wall_seconds,
                stage_seconds=record.stage_seconds,
                rule_readings=record.rule_readings,
            )
        self.states[key] = record.state
        self.flows.append(
            FlowSolve(
                key,
                origin,
                record.state.outer,
                record.state.stop_reason,
                record.wall_seconds,
            )
        )
        logger.info("flow state %s: %s", key[:12], origin)

    def _solve(
        self,
        interval: Interval,
        previous: tuple[str, FlowState] | None,
        resume: SteadyCheckpoint | None,
    ) -> None:
        """Solve an interval's state: resumed, warm from the one before it, or from rest."""
        key = interval.fingerprint
        config = interval.config
        solver = StaggeredSolver(
            self.mesh, config, StaggeredBoundary(self.mesh, config)
        )
        writer = None
        if config.checkpoint_interval is not None:
            writer = partial(self.run.write_steady_checkpoint, fingerprint=key)
        if resume is not None:
            origin, initial, warm_from = RESUMED, None, None
        elif previous is None:
            origin, initial, warm_from = REST, None, None
        else:
            origin, initial, warm_from = WARM, previous[1], previous[0]
        logger.info("flow state %s: solving (%s)", key[:12], origin)
        started = perf_counter()
        solver.solve_steady(initial=initial, resume=resume, on_checkpoint=writer)
        wall = perf_counter() - started
        outer = len(solver.residual_history)
        state = solver.flow_state
        if state is None:
            raise FlowNotConvergedError(key, solver.stop_reason, outer)
        if not solver.converged:
            assert solver.steady_checkpoint is not None
            self.run.write_steady_checkpoint(solver.steady_checkpoint, key)
            raise FlowNotConvergedError(key, solver.stop_reason, outer)
        self.run.write_flow_state(
            key,
            state,
            wall_seconds=wall,
            stage_seconds=solver.stage_seconds,
            rule_readings=_rule_readings(solver, self.mesh),
        )
        self.states[key] = state
        self.flows.append(
            FlowSolve(key, origin, outer, solver.stop_reason, wall, warm_from)
        )
        logger.info(
            "flow state %s: %s after %d outer iterations, %.1f s",
            key[:12],
            solver.stop_reason,
            outer,
            wall,
        )

    # -- The march -----------------------------------------------------------

    def transport(self, interval: Interval) -> TransportSolver:
        """A transport solver on an interval's concentration boundary."""
        config = interval.config
        return TransportSolver(
            self.mesh,
            config,
            ParticlePhysics(config),
            self.manager.conditions_for(interval),
        )

    def faces(self, key: str) -> FaceVelocities:
        """A flow state's faces, as the transport solver takes them."""
        state = self.states[key]
        return FaceVelocities.copy_of(np.array(state.u), np.array(state.v))

    def plans(self) -> list[IntervalPlan]:
        """Every interval divided from its start, stable_dt on each state's faces."""
        stable: dict[str, float] = {}
        out = []
        for index, interval in enumerate(self.intervals):
            key = interval.fingerprint
            if key not in stable:
                solver = self.transport(interval)
                faces = self.faces(key)
                stable[key] = smallest_stable_dt(solver, faces, self.n_classes)
            out.append(plan_interval(index, interval, stable[key]))
        return out

    def readings(self, fields: Mapping[int, np.ndarray]) -> list[list[float]]:
        """Every sensor's reading per class."""
        return [[float(fields[k][j, i]) for j, i in self.cells] for k in fields]

    def means(self, fields: Mapping[int, np.ndarray]) -> list[float]:
        """Every class's free-area mean."""
        return [
            MassBudget.in_domain(fields[k], self.mesh) / self.free_area for k in fields
        ]

    def march(
        self,
        plans: list[IntervalPlan],
        start: tuple[int, int],
        fields: dict[int, np.ndarray],
        budgets: Sequence[MassBudget],
        step: int,
        t: float,
    ) -> tuple[dict[int, np.ndarray], tuple[MassBudget, ...], int, float, bool]:
        """Step every class from (plan, step within it) to the end or the hold.

        Returns the fields, budgets, step count and time at the end, and
        whether the hold stopped it.
        """
        config = self.config
        output, cadence = config.output_interval, config.checkpoint_interval
        pending: list[tuple[int, float, list[list[float]]]] = []
        held = self.tracker is not None and self.tracker.held()
        first_plan, first_i = start
        current = tuple(budgets)
        for p in range(first_plan, len(plans)):
            if held:
                break
            plan = plans[p]
            interval = self.intervals[plan.index]
            solver = self.transport(interval)
            solver.restore_budgets(current)
            faces = self.faces(plan.fingerprint)
            nu_t = self.states[plan.fingerprint].nu_t
            rates = self.manager.sources_at(interval.t_start)
            sources = {k: rates[k] if rates[k].any() else None for k in rates}
            for i in range(first_i + 1 if p == first_plan else 1, plan.steps + 1):
                fields = {
                    k: solver.solve_timestep(
                        c,
                        faces,
                        k,
                        plan.dt,
                        sources=sources[k],
                        eddy_viscosity=nu_t,
                    )
                    for k, c in fields.items()
                }
                for c in fields.values():
                    c.flags.writeable = False
                step += 1
                t = plan.time(i)
                pending.append((step, t, self.readings(fields)))
                if self.tracker is not None:
                    means = [b.current / self.free_area for b in solver.budget]
                    self.tracker.update(t, fields, means)
                    held = self.tracker.held()
                last = held or (p == len(plans) - 1 and i == plan.steps)
                if step % output == 0 or last:
                    self.run.write_frame(step, t, fields, solver.budget)
                    self._flush(pending)
                if cadence is not None and (step % cadence == 0 or last):
                    self._flush(pending)
                    self.run.checkpoint(
                        step,
                        t,
                        fields,
                        solver.budget,
                        plan.fingerprint,
                        recovery=None if self.tracker is None else self.tracker.state(),
                    )
                if self.on_step is not None:
                    self.on_step(
                        MarchStep(
                            step, t, plan.dt, plan.index, fields, _copies(solver.budget)
                        )
                    )
                if held:
                    break
            current = tuple(solver.budget)
        self._flush(pending)
        return fields, _copies(current), step, t, held

    def _flush(self, pending: list[tuple[int, float, list[list[float]]]]) -> None:
        """Append the held sensor readings to the run file and empty the list."""
        if pending and self.cells:
            self.run.write_sensor_readings(
                np.array([row[0] for row in pending]),
                np.array([row[1] for row in pending]),
                np.array([row[2] for row in pending]),
                self.names,
            )
        pending.clear()

    def resume_point(
        self, plans: list[IntervalPlan], step: int, t: float
    ) -> tuple[list[IntervalPlan], tuple[int, int], IntervalPlan | None]:
        """Where a march checkpointed at (step, t) continues.

        The interval holding t, at the step whose time t is; or, when t is
        not on that interval's plan (a later t_end moved the last interval's
        end), the rest of the interval divided again from t.
        """
        for p, plan in enumerate(plans):
            if not plan.t_start <= t < plan.t_end:
                continue
            i = round((t - plan.t_start) / plan.dt)
            if 0 <= i < plan.steps and plan.time(i) == t:
                return plans, (p, i), None
            interval = self.intervals[plan.index]
            again = plan_interval(plan.index, interval, plan.stable_dt, t_start=t)
            logger.info(
                "restart: interval %d divided again from %r s into %d steps",
                plan.index,
                t,
                again.steps,
            )
            return [*plans[:p], again, *plans[p + 1 :]], (p, 0), again
        return plans, (len(plans), 0), None


def run(
    config: SimConfig,
    run_file: str | os.PathLike[str] | None = None,
    *,
    restart: str | os.PathLike[str] | None = None,
    flows_from: str | os.PathLike[str] | None = None,
    on_step: Callable[[MarchStep], None] | None = None,
    provenance: Provenance | None = None,
) -> RunRecord:
    """Run a scenario under tier 1 (ADR-014 A), writing its run file (ADR-014 D).

    Parameters
    ----------
    config : SimConfig
        A configuration with a scenario section and a transport section.
    run_file : str or Path, optional
        A new run's file; neither it nor its checkpoint file may exist.
    restart : str or Path, optional
        The run file of a run to continue: its configuration must be this
        one but for ``io_manager.RESTART_EXCLUDED_KEYS``. Exactly one of
        ``run_file`` and ``restart``.
    flows_from : str or Path, optional
        A new run's source of kept flow states: another run file whose
        configuration agrees with this one on everything the flow solve
        reads (``flow_inputs``). A state it holds under an interval's
        fingerprint is copied in and not solved.
    on_step : Callable[[MarchStep], None], optional
        Called after every step's writes. It must not change what it is
        handed; an exception it raises ends the run there, as a crash would.
    provenance : Provenance, optional
        Written to a new run's ``/config``; ``Provenance.here()`` when None.

    Returns
    -------
    RunRecord

    Raises
    ------
    ValueError
        Without a scenario section; with both or neither of ``run_file`` and
        ``restart``; when a new run's files exist; with ``flows_from`` on a
        restart, or a ``flows_from`` file whose flow inputs differ (the
        message names the first key); for a restart under another
        configuration (``RunFile.check_restart``); for a sensor in a SOLID
        cell, or a clean-out without sensors or thresholds.
    FlowNotConvergedError
        When a flow state stops at its cap or goes non-finite.
    """
    if config.scenario is None:
        raise ValueError(
            "time_integration.run runs a scenario under tier 1 (ADR-014 A); the "
            "configuration has no scenario section, and tier 2 arrives with step 6"
        )
    if (run_file is None) == (restart is None):
        raise ValueError("give exactly one of run_file (a new run) and restart")
    if restart is not None and flows_from is not None:
        raise ValueError(
            "flows_from is for a new run; a restart reuses the flow states its run "
            "file holds"
        )
    started = perf_counter()
    mapping = config.merged_mapping()
    if restart is not None:
        run_path = RunFile(restart)
        run_path.check_restart(mapping)
    else:
        assert run_file is not None
        run_path = RunFile(run_file)
        for path in (run_path.path, run_path.checkpoint_path):
            if path.exists():
                raise ValueError(
                    f"{path} exists; a new run writes new files (pass restart= to "
                    "continue a run)"
                )
    kept = None
    if flows_from is not None:
        kept = RunFile(flows_from)
        difference = _first_flow_difference(
            flow_inputs(kept.read_config().mapping), flow_inputs(mapping)
        )
        if difference is not None:
            raise ValueError(
                f"flows_from: {difference} differs from the configuration "
                f"{kept.path} was run with, so its flow states are not this run's"
            )
    state = _Run(config, run_path, on_step)
    if restart is None:
        given = provenance if provenance is not None else Provenance.here()
        run_path.write_config(
            mapping,
            commit=given.commit,
            started=given.started,
            machine=given.machine,
            concurrent=given.concurrent,
        )
    state.flow_states(kept, restart is not None)
    plans = state.plans()
    logger.info(
        "scenario: %d intervals, %d steps planned, %d flow states",
        len(plans),
        sum(p.steps for p in plans),
        len(state.flows),
    )

    restarted_from = None
    resumed_plan = None
    checkpoint = None
    if restart is not None:
        try:
            checkpoint = run_path.read_checkpoint()
        except KeyError:
            checkpoint = None
    if checkpoint is not None:
        restarted_from = checkpoint.step
        run_path.discard_after(checkpoint.step)
        fields = checkpoint.fields
        for c in fields.values():
            c.flags.writeable = False
        budgets: Sequence[MassBudget] = checkpoint.budgets
        if state.tracker is not None:
            state.tracker.restore(checkpoint.recovery)
        steps, start, resumed_plan = state.resume_point(
            plans, checkpoint.step, checkpoint.t
        )
        step, t = checkpoint.step, checkpoint.t
    else:
        if restart is not None:
            restarted_from = 0
            run_path.discard_march()
        fields = state.manager.initial_fields()
        budgets = tuple(MassBudget() for _ in range(state.n_classes))
        run_path.write_frame(0, 0.0, fields, budgets)
        if state.cells:
            run_path.write_sensor_readings(
                np.array([0]),
                np.array([0.0]),
                np.array([state.readings(fields)]),
                state.names,
            )
        if state.tracker is not None:
            state.tracker.update(0.0, fields, state.means(fields))
        steps, start, step, t = plans, (0, 0), 0, 0.0

    fields, budgets, step, t, held = state.march(steps, start, fields, budgets, step, t)

    recovery = None
    if state.tracker is not None:
        rate = air_change_rate(state.states[state.intervals[0].fingerprint], state.mesh)
        recovery = state.tracker.record(state.names, rate, held)
        run_path.write_recovery(recovery)
    return RunRecord(
        path=run_path.path,
        intervals=tuple(plans),
        flows=tuple(state.flows),
        steps=step,
        t_final=t,
        held=held,
        wall_seconds=perf_counter() - started,
        budgets=tuple(budgets),
        recovery=recovery,
        restarted_from=restarted_from,
        resumed_plan=resumed_plan,
    )
