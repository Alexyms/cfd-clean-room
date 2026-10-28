"""The run file and its checkpoint file, written and read back (ADR-014 D).

Every run keeps two HDF5 files. The run file holds what the run produced and
grows by appending: it is opened, written and closed at each output, so a
process killed between two outputs leaves it readable with every frame
written before. Its groups:

- ``/config``: the merged mapping as YAML text, the text's SHA-256, the
  restart hash (the same over the mapping less ``RESTART_EXCLUDED_KEYS``),
  and the commit, start time, machine and process count;
- ``/flow/<fingerprint>/``: per flow state solved, datasets ``u_faces``,
  ``v_faces``, ``p`` and with the turbulence section ``k``, ``eps``,
  ``nu_t``; attributes ``outer``, ``stop_reason`` (when not None) and
  ``wall_seconds``; child groups ``stage_seconds`` and ``rule_readings``
  whose attributes are those mappings;
- ``/transport/``: ``steps`` (int64), ``times`` and ``C_<k>`` [n, ny, nx]
  per class (``FieldHistory``'s keys), ``budgets`` [n, classes, terms] with
  the term names ``BUDGET_TERMS`` as its attribute, and ``/transport/sensors/``
  with ``steps``, ``times``, ``readings`` [n, classes, sensors] and the
  sensor names as an attribute;
- ``/recovery/``: a clean-out's recovery outputs (ADR-014 C), written once
  at the end of the run (``RecoveryRecord``); ``discard_after`` does not
  touch it, and a restart that runs to a new end replaces it;
- ``/transient/``, a tier-2 run's (ADR-014 D, tier 2; step 6b): ``steps``
  (int64), ``times`` and the frames ``u`` [n, ny, nx+1], ``v`` [n, ny+1, nx]
  and ``p`` [n, ny, nx], and ``/transient/series/``, one row per step:
  ``step`` and ``outer`` and ``pressure_cap_hits`` (int64), ``t``,
  ``worst_imbalance``, ``drag``, ``lift``, ``c_d`` and ``c_l`` (float64,
  NaN for a run without forces).

The checkpoint file, ``<run>.checkpoint.h5`` beside it (the run file's path
less its suffix), holds what a restart continues from: ``/checkpoint/``, the
march's step, simulated time, class fields, budgets and the fingerprint of
the flow state it was marching on, with any named arrays the march carries
besides (the recovery tracker's, ``recovery``), and ``/steady_checkpoint/``, a steady
solve's SteadyCheckpoint (step 0's layout: ``u``, ``v``, ``p`` and ``k``,
``eps``, ``nu_t``, the three series, ``outer``, ``pressure_cap_hits``,
``stop_reason`` and ``rule_version`` when not None) with the fingerprint of
the flow state being solved and the eddy viscosity digest when not None,
and ``/transient_checkpoint/``, a tier-2 march's step, time and two levels:
``u``, ``v``, ``p`` and ``outer`` of its last step and, when it has one, the
``previous_u``, ``previous_v`` and ``previous_p`` of the step before.
HDF5 keeps no journal, and a process killed while a file is open for writing
can leave the whole file unreadable, so the checkpoint file is never written
in place: each checkpoint writes a new file whole under
``<run>.checkpoint.h5.tmp``, the group it does not replace copied across,
closes it and moves it over the old one with ``os.replace``, which is atomic
on one volume. A temporary file left by an earlier crash is truncated, never
read.

HDF5 stores float64 arrays as their bytes, so what is read back is what was
written, bitwise. Nothing here decides anything about a solve or a march: the
solvers validate what they are given, and a restart reads the checkpoint and
calls ``check_restart`` first. The class fields, the budgets and the step
are all a march carries: a recirculating discharge reads its intake from the
field the step starts from, so it needs nothing more.
"""

import hashlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import yaml

from src.solver_staggered import FlowState, SteadyCheckpoint, TransientResume
from src.solver_transport import DEPOSIT_SURFACES, MassBudget

CONFIG = "config"
FLOW = "flow"
TRANSPORT = "transport"
SENSORS = "sensors"
CHECKPOINT = "checkpoint"
STEADY_CHECKPOINT = "steady_checkpoint"
RECOVERY = "recovery"
TRANSIENT = "transient"
SERIES = "series"
TRANSIENT_CHECKPOINT = "transient_checkpoint"
# The per-step columns of /transient/series/, the integer ones first.
STEP_SERIES_INTEGERS: tuple[str, ...] = ("step", "outer", "pressure_cap_hits")
STEP_SERIES_REALS: tuple[str, ...] = (
    "t",
    "worst_imbalance",
    "drag",
    "lift",
    "c_d",
    "c_l",
)
CHECKPOINT_SUFFIX = ".checkpoint.h5"
TEMPORARY_SUFFIX = ".tmp"
# The keys a continuation exists to change (prompt 63, decision 2): a capped
# steady solve continues under a larger cap, a capped march under a later
# t_end, and either may checkpoint at another cadence. Every other key is in
# the restart hash, so any other change refuses the restart.
RESTART_EXCLUDED_KEYS: tuple[tuple[str, str], ...] = (
    ("solver", "max_simple_iter"),
    ("solver", "t_end"),
    ("solver", "checkpoint_interval"),
)
# One column per MassBudget term; deposition by surface in DEPOSIT_SURFACES'
# order. A budget before its first step has initial and current None, stored
# as NaN, which no booked term can be (the solver refuses non-finite input).
BUDGET_TERMS: tuple[str, ...] = (
    "initial",
    "inflow",
    "outflow",
    "source",
    *(f"deposited_{surface}" for surface in DEPOSIT_SURFACES),
    "current",
)
_FIELDS = ("u", "v", "p")
_FLOW_FIELDS = ("u_faces", "v_faces", "p")
_TURBULENCE = ("k", "eps", "nu_t")
_SERIES = ("residual_history", "velocity_steps", "viscosity_steps")
_SHA256_DIGITS = 64
_HEX_DIGITS = "0123456789abcdef"
# RecoveryRecord's float64 arrays, stored as datasets of /recovery/ beside its
# defined array; its scalars are the group's attributes.
_RECOVERY_ARRAYS = (
    "thresholds",
    "initial",
    "recovery_time",
    "sensor_time",
    "mean_time",
    "ratio",
    "local_recovery",
    "local_recovery_max",
)
_RECOVERY_SCALARS = (
    "t_final",
    "air_change_rate",
    "well_mixed_time",
    "room_recovery_time",
)


@dataclass(frozen=True)
class RunConfig:
    """The run file's ``/config``, read back.

    Parameters
    ----------
    mapping : dict
        The merged mapping, as ``yaml.safe_load`` reads its text.
    sha256 : str
        SHA-256 of the YAML text.
    restart_sha256 : str
        SHA-256 of the YAML text of the mapping less RESTART_EXCLUDED_KEYS.
    commit, started, machine : str
        The commit the run was made on, its start time and the machine.
    concurrent : int
        Processes running on the machine beside it.
    """

    mapping: dict
    sha256: str
    restart_sha256: str
    commit: str
    started: str
    machine: str
    concurrent: int


@dataclass(frozen=True)
class FlowRecord:
    """A flow state of ``/flow/<fingerprint>/``, read back.

    Parameters
    ----------
    state : FlowState
        The faces, pressure and turbulence fields with the stop reason and
        outer count, bitwise as written.
    wall_seconds : float
        Wall time of the solve.
    stage_seconds : dict[str, float]
        The solver's ``stage_seconds``.
    rule_readings : dict[str, float]
        The stopping rule's readings at the stop.
    """

    state: FlowState
    wall_seconds: float
    stage_seconds: dict[str, float]
    rule_readings: dict[str, float]


@dataclass(frozen=True)
class TransportFrames:
    """The run file's ``/transport/`` frames, read back.

    Parameters
    ----------
    steps : np.ndarray
        [n] int64, increasing.
    times : np.ndarray
        [n] float64, seconds.
    fields : dict[int, np.ndarray]
        Class to its frames, [n, ny, nx] float64.
    budgets : tuple[tuple[MassBudget, ...], ...]
        Per frame, one budget per class.
    """

    steps: np.ndarray
    times: np.ndarray
    fields: dict[int, np.ndarray]
    budgets: tuple[tuple[MassBudget, ...], ...]


@dataclass(frozen=True)
class SensorReadings:
    """The run file's ``/transport/sensors/``, read back.

    Parameters
    ----------
    names : tuple[str, ...]
        The sensors, in the order of the last axis of ``readings``.
    steps : np.ndarray
        [n] int64.
    times : np.ndarray
        [n] float64, seconds.
    readings : np.ndarray
        [n, classes, sensors] float64.
    """

    names: tuple[str, ...]
    steps: np.ndarray
    times: np.ndarray
    readings: np.ndarray


@dataclass(frozen=True)
class MarchCheckpoint:
    """The checkpoint file's ``/checkpoint/``: what a march resumes from.

    Parameters
    ----------
    step : int
        Steps taken.
    t : float
        Simulated time, seconds, as the march held it.
    fields : dict[int, np.ndarray]
        Class to its field, [ny, nx] float64.
    budgets : tuple[MassBudget, ...]
        One per class, in class order.
    fingerprint : str
        The flow state the march was stepping on (``/flow/<fingerprint>/``).
    recovery : dict[str, np.ndarray]
        The named float64 arrays the march carried besides (the recovery
        tracker's state, ADR-014 C); empty when it carried none.
    plan_start : float or None
        The start of the plan the march was stepping when it wrote the
        checkpoint: the interval's start, or the time a restart divided the
        interval again from. None in a file written without it.
    """

    step: int
    t: float
    fields: dict[int, np.ndarray]
    budgets: tuple[MassBudget, ...]
    fingerprint: str
    recovery: dict[str, np.ndarray] = field(default_factory=dict)
    plan_start: float | None = None


@dataclass(frozen=True, eq=False)
class TransientFrames:
    """A tier-2 run's frames, as ``/transient/`` holds them.

    Parameters
    ----------
    steps : np.ndarray
        [n] int64.
    times : np.ndarray
        [n] float64, seconds.
    u, v, p : np.ndarray
        [n, ny, nx+1], [n, ny+1, nx] and [n, ny, nx] float64.
    """

    steps: np.ndarray
    times: np.ndarray
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray


@dataclass(frozen=True, eq=False)
class StepSeries:
    """One row per step of a tier-2 march (ADR-014 D, tier 2).

    Parameters
    ----------
    step, outer, pressure_cap_hits : np.ndarray
        [n] int64: the step, its outer iterations and its capped corrections.
    t, worst_imbalance : np.ndarray
        [n] float64: the time reached, s, and the step's largest per-cell
        mass imbalance over non-SOLID cells, kg/s per m.
    drag, lift : np.ndarray
        [n] float64, N/m, on the configured obstacle; NaN without one.
    c_d, c_l : np.ndarray
        [n] float64, their coefficients; NaN without an obstacle.

    Raises
    ------
    ValueError
        If the columns are not 1-D of one length.
    """

    step: np.ndarray
    outer: np.ndarray
    pressure_cap_hits: np.ndarray
    t: np.ndarray
    worst_imbalance: np.ndarray
    drag: np.ndarray
    lift: np.ndarray
    c_d: np.ndarray
    c_l: np.ndarray

    def __post_init__(self) -> None:
        columns = STEP_SERIES_INTEGERS + STEP_SERIES_REALS
        lengths = {np.asarray(getattr(self, name)).shape for name in columns}
        if len(lengths) != 1 or len(next(iter(lengths))) != 1:
            raise ValueError(
                f"the step series' columns must be 1-D of one length, got {lengths}"
            )
        for name in STEP_SERIES_INTEGERS:
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype="<i8"))
        for name in STEP_SERIES_REALS:
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype="<f8"))


@dataclass(frozen=True, eq=False)
class RecoveryRecord:
    """A clean-out's recovery outputs (ADR-014 C), as ``/recovery/`` holds them.

    Times are seconds of simulated time, NaN where an output is not defined
    or not reached. The arrays are stored as float64, ``defined`` as bool,
    when the record is built.

    Parameters
    ----------
    sensor_names : tuple[str, ...]
        The sensors, in the order of the last axis of ``sensor_time``.
    thresholds, initial : np.ndarray
        [classes]: each class's threshold and its initial level C0,
        particles per cubic metre.
    hold_time : float or None
        ``scenario.hold_time``; None when the run went to t_end.
    held : bool
        Whether the run stopped because the room held within spec.
    t_final : float
        The run's last step time.
    air_change_rate : float
        N, the inflow through the domain's edges over the free area, 1/s.
    well_mixed_time : float
        ln(100) / N, the well-mixed slice's 100:1 time.
    room_recovery_time : float
        When every sensor of every class fell below its threshold and
        stayed below (ADR-014 C).
    recovery_time : np.ndarray
        [classes], the same per class.
    defined : np.ndarray
        [classes] bool: C0 at least 100 times the threshold, where the
        100:1 times are defined.
    sensor_time : np.ndarray
        [classes, sensors], the 100:1 time at each sensor.
    mean_time : np.ndarray
        [classes], the 100:1 time of the free-area mean.
    ratio : np.ndarray
        [classes], ``mean_time / well_mixed_time``.
    local_recovery : np.ndarray
        [classes, ny, nx], the last step time each cell was at or above its
        threshold: zero for a cell never so, inf for one still so at the
        end, NaN in SOLID cells.
    local_recovery_max : np.ndarray
        [classes], the largest over the non-SOLID cells.

    Raises
    ------
    TypeError
        If a scalar time is not a real number, ``held`` is not a bool, or a
        sensor name is not a str.
    ValueError
        If the arrays' shapes do not agree on the classes and the sensors,
        or ``local_recovery`` is not [classes, ny, nx].
    """

    sensor_names: tuple[str, ...]
    thresholds: np.ndarray
    initial: np.ndarray
    hold_time: float | None
    held: bool
    t_final: float
    air_change_rate: float
    well_mixed_time: float
    room_recovery_time: float
    recovery_time: np.ndarray
    defined: np.ndarray
    sensor_time: np.ndarray
    mean_time: np.ndarray
    ratio: np.ndarray
    local_recovery: np.ndarray
    local_recovery_max: np.ndarray

    def __post_init__(self) -> None:
        names = tuple(self.sensor_names)
        for name in names:
            if not isinstance(name, str):
                raise TypeError(
                    f"a sensor name must be a str, got {type(name).__name__}"
                )
        object.__setattr__(self, "sensor_names", names)
        if not isinstance(self.held, bool):
            raise TypeError(f"held must be a bool, got {type(self.held).__name__}")
        scalars = list(_RECOVERY_SCALARS)
        if self.hold_time is not None:
            scalars.append("hold_time")
        for name in scalars:
            object.__setattr__(self, name, _real(name, getattr(self, name)))
        for name in _RECOVERY_ARRAYS:
            array = np.asarray(getattr(self, name), dtype="<f8")
            object.__setattr__(self, name, array)
        object.__setattr__(self, "defined", np.asarray(self.defined, dtype=bool))
        classes = self.thresholds.shape
        if len(classes) != 1:
            raise ValueError(f"thresholds must be [classes], got shape {classes}")
        for name in (
            "initial",
            "recovery_time",
            "defined",
            "mean_time",
            "ratio",
            "local_recovery_max",
        ):
            if getattr(self, name).shape != classes:
                raise ValueError(
                    f"{name} has shape {getattr(self, name).shape} for "
                    f"{classes[0]} classes"
                )
        if self.sensor_time.shape != (classes[0], len(names)):
            raise ValueError(
                f"sensor_time has shape {self.sensor_time.shape} for {classes[0]} "
                f"classes and {len(names)} sensors"
            )
        local = self.local_recovery.shape
        if len(local) != 3 or local[0] != classes[0]:
            raise ValueError(
                f"local_recovery has shape {local}; expected [{classes[0]}, ny, nx]"
            )


def config_text(mapping: Mapping[str, object]) -> str:
    """The YAML text of a configuration mapping, keys in their given order.

    Parameters
    ----------
    mapping : Mapping[str, object]
        A configuration mapping; nested mappings and sequences are written
        as YAML mappings and lists.

    Returns
    -------
    str
        ASCII. Floats are written as ``repr`` writes them, so the text reads
        back to the same values.
    """
    return yaml.safe_dump(_plain(mapping), sort_keys=False, allow_unicode=False)


def restart_hash(mapping: Mapping[str, object]) -> str:
    """SHA-256 of a mapping's YAML text less RESTART_EXCLUDED_KEYS.

    Parameters
    ----------
    mapping : Mapping[str, object]
        A configuration mapping.

    Returns
    -------
    str
        Hexadecimal. Equal for two mappings that differ at most in the
        excluded keys, present or absent; key order counts, since a
        boundary segment's place in the order is part of its meaning.
    """
    text = config_text(_without_excluded(_plain(mapping)))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class RunFile:
    """A run's HDF5 run file and its checkpoint file (ADR-014 D, step 3).

    Each method opens the file it needs, writes or reads, and closes it, so
    no handle is held between calls and a crash between two calls loses
    nothing already written.

    Parameters
    ----------
    path : str or Path
        The run file. Neither file needs to exist: the first write creates
        it.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path: Path = Path(path)
        self.checkpoint_path: Path = self.path.with_suffix(CHECKPOINT_SUFFIX)
        self.temporary_path: Path = self.checkpoint_path.with_name(
            self.checkpoint_path.name + TEMPORARY_SUFFIX
        )

    # ------------------------------------------------------------------
    # /config
    # ------------------------------------------------------------------

    def write_config(
        self,
        mapping: Mapping[str, object],
        *,
        commit: str,
        started: str,
        machine: str,
        concurrent: int,
    ) -> None:
        """Write ``/config``: the merged mapping, its hashes and the provenance.

        Parameters
        ----------
        mapping : Mapping[str, object]
            The merged configuration mapping the run loads.
        commit, started, machine : str
            The commit, the start time and the machine, as the probe records
            carry them.
        concurrent : int
            Processes running on the machine beside this one, non-negative.

        Raises
        ------
        TypeError
            If ``mapping`` is not a mapping, a provenance string is not a
            str, or ``concurrent`` is not an int (a bool is not one here).
        ValueError
            If ``concurrent`` is negative, or the file holds a configuration
            already: a restart checks its own with ``check_restart``.
        """
        if not isinstance(mapping, Mapping):
            raise TypeError(f"mapping must be a mapping, got {type(mapping).__name__}")
        for name, value in (
            ("commit", commit),
            ("started", started),
            ("machine", machine),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a str, got {type(value).__name__}")
        if isinstance(concurrent, bool) or not isinstance(concurrent, int):
            raise TypeError(
                f"concurrent must be an int, got {type(concurrent).__name__}"
            )
        if concurrent < 0:
            raise ValueError(f"concurrent must be non-negative, got {concurrent}")
        text = config_text(mapping)
        with h5py.File(self.path, "a") as run_file:
            if CONFIG in run_file:
                raise ValueError(
                    f"{self.path} holds a configuration already; a restart checks "
                    "its configuration against it with check_restart"
                )
            group = run_file.create_group(CONFIG)
            group.create_dataset("yaml", data=text)
            group.attrs["sha256"] = hashlib.sha256(text.encode("ascii")).hexdigest()
            group.attrs["restart_sha256"] = restart_hash(mapping)
            group.attrs["commit"] = commit
            group.attrs["started"] = started
            group.attrs["machine"] = machine
            group.attrs["concurrent"] = concurrent

    def read_config(self) -> RunConfig:
        """Read back ``/config``.

        Returns
        -------
        RunConfig
            The mapping, its hashes and the provenance.

        Raises
        ------
        KeyError
            If the file holds no ``/config``.
        """
        with h5py.File(self.path, "r") as run_file:
            if CONFIG not in run_file:
                raise KeyError(f"{self.path} holds no /{CONFIG}")
            group = run_file[CONFIG]
            text = group["yaml"].asstr()[()]
            attrs = dict(group.attrs)
        return RunConfig(
            mapping=yaml.safe_load(text),
            sha256=str(attrs["sha256"]),
            restart_sha256=str(attrs["restart_sha256"]),
            commit=str(attrs["commit"]),
            started=str(attrs["started"]),
            machine=str(attrs["machine"]),
            concurrent=int(attrs["concurrent"]),
        )

    def check_restart(self, mapping: Mapping[str, object]) -> None:
        """Refuse a restart whose configuration is not the run file's.

        Parameters
        ----------
        mapping : Mapping[str, object]
            The merged mapping the restart would load.

        Raises
        ------
        KeyError
            If the file holds no ``/config``.
        TypeError
            If ``mapping`` is not a mapping.
        ValueError
            If the restart hash differs from the file's: the message names
            the first key, in the file's order, that differs, was added or
            was removed. The keys in RESTART_EXCLUDED_KEYS may differ.

        Notes
        -----
        The file's side is recomputed from its stored text under the
        RESTART_EXCLUDED_KEYS in force now, not read from ``restart_sha256``,
        which holds the set in force when the file was written, so a key
        added to the set widens what an existing file's restart accepts
        (review 63 S2, prompt 64's decision 3). ``restart_sha256`` stays as
        the record of what the run was hashed under.
        """
        if not isinstance(mapping, Mapping):
            raise TypeError(f"mapping must be a mapping, got {type(mapping).__name__}")
        stored = self.read_config()
        if restart_hash(mapping) == restart_hash(stored.mapping):
            return
        first = _first_difference(
            _without_excluded(_plain(stored.mapping)),
            _without_excluded(_plain(mapping)),
            (),
        )
        raise ValueError(
            f"restart refused: {first or 'the configuration'} differs from the "
            f"configuration {self.path} was run with; only "
            f"{', '.join('.'.join(path) for path in RESTART_EXCLUDED_KEYS)} may "
            "change on a restart"
        )

    # ------------------------------------------------------------------
    # /flow/<fingerprint>/
    # ------------------------------------------------------------------

    def write_flow_state(
        self,
        fingerprint: str,
        state: FlowState,
        *,
        wall_seconds: float,
        stage_seconds: Mapping[str, float],
        rule_readings: Mapping[str, float],
    ) -> None:
        """Write a solved flow state under its fingerprint.

        Parameters
        ----------
        fingerprint : str
            The state's key (``scenarios.flow_fingerprint``), 64 lowercase
            hexadecimal digits.
        state : FlowState
            The solver's ``flow_state``.
        wall_seconds : float
            Wall time of the solve.
        stage_seconds : Mapping[str, float]
            The solver's ``stage_seconds``.
        rule_readings : Mapping[str, float]
            The stopping rule's readings at the stop; empty is allowed.

        Raises
        ------
        TypeError
            If ``state`` is not a FlowState, the fingerprint is not a str, or
            a time or reading is not a real number.
        ValueError
            If the fingerprint is not a SHA-256, or the file holds that flow
            state already: each is solved once and reused.
        """
        _check_fingerprint(fingerprint)
        if not isinstance(state, FlowState):
            raise TypeError(f"state must be a FlowState, got {type(state).__name__}")
        wall = _real("wall_seconds", wall_seconds)
        stages = {
            str(k): _real(f"stage_seconds[{k}]", v) for k, v in stage_seconds.items()
        }
        readings = {
            str(k): _real(f"rule_readings[{k}]", v) for k, v in rule_readings.items()
        }
        with h5py.File(self.path, "a") as run_file:
            flows = run_file.require_group(FLOW)
            if fingerprint in flows:
                raise ValueError(f"{self.path} holds flow state {fingerprint} already")
            group = flows.create_group(fingerprint)
            arrays = dict(zip(_FLOW_FIELDS, (state.u, state.v, state.p), strict=True))
            if state.turbulent:
                arrays |= {name: getattr(state, name) for name in _TURBULENCE}
            for name, array in arrays.items():
                group.create_dataset(name, data=array, dtype="<f8")
            group.attrs["outer"] = state.outer
            if state.stop_reason is not None:
                group.attrs["stop_reason"] = state.stop_reason
            group.attrs["wall_seconds"] = wall
            for name, values in (
                ("stage_seconds", stages),
                ("rule_readings", readings),
            ):
                child = group.create_group(name)
                for key, value in values.items():
                    child.attrs[key] = value

    def flow_fingerprints(self) -> tuple[str, ...]:
        """The fingerprints of the flow states the file holds, sorted."""
        if not self.path.exists():
            return ()
        with h5py.File(self.path, "r") as run_file:
            if FLOW not in run_file:
                return ()
            return tuple(sorted(run_file[FLOW]))

    def read_flow_state(self, fingerprint: str) -> FlowRecord:
        """Read back a flow state written by ``write_flow_state``.

        Raises
        ------
        KeyError
            If the file holds no flow state of that fingerprint.
        ValueError
            If the group holds some but not all of ``k``, ``eps`` and ``nu_t``.
        """
        with h5py.File(self.path, "r") as run_file:
            if fingerprint not in run_file.get(FLOW, {}):
                raise KeyError(f"{self.path} holds no flow state {fingerprint}")
            group = run_file[FLOW][fingerprint]
            arrays = dict(
                zip(_FIELDS, (group[name][()] for name in _FLOW_FIELDS), strict=True)
            )
            arrays |= {name: group[name][()] for name in _TURBULENCE if name in group}
            attrs = dict(group.attrs)
            stages = {k: float(v) for k, v in group["stage_seconds"].attrs.items()}
            readings = {k: float(v) for k, v in group["rule_readings"].attrs.items()}
        reason = attrs.get("stop_reason")
        state = FlowState(
            **arrays,
            stop_reason=None if reason is None else str(reason),
            outer=int(attrs["outer"]),
        )
        return FlowRecord(state, float(attrs["wall_seconds"]), stages, readings)

    # ------------------------------------------------------------------
    # /transport/
    # ------------------------------------------------------------------

    def write_frame(
        self,
        step: int,
        t: float,
        fields: Mapping[int, np.ndarray],
        budgets: Sequence[MassBudget],
    ) -> None:
        """Append one output frame: every class field and every class budget.

        Parameters
        ----------
        step : int
            The step index, zero at the initial field; above the last frame's.
        t : float
            Simulated time, seconds.
        fields : Mapping[int, np.ndarray]
            Class index to its field, [ny, nx], for classes 0 to n - 1.
        budgets : Sequence[MassBudget]
            The transport solver's ``budget``, one per class.

        Raises
        ------
        TypeError
            If ``step`` is not an int (a bool is not one here), ``t`` is not
            a real number, or a budget is not a MassBudget.
        ValueError
            If ``step`` is negative, the classes are not 0 to n - 1 with one
            budget each, a field is not 2-D, the classes or shapes differ
            from the first frame's, or ``step`` is not above the last frame's.
        """
        step = _step(step)
        t = _real("t", t)
        arrays = _class_fields(fields, budgets)
        classes = list(arrays)
        row = np.array([_budget_row(b) for b in budgets], dtype="<f8")
        with h5py.File(self.path, "a") as run_file:
            group = run_file.require_group(TRANSPORT)
            if "steps" not in group:
                _new_series(group, "steps", (), "<i8")
                _new_series(group, "times", (), "<f8")
                for k, array in arrays.items():
                    _new_series(group, f"C_{k}", array.shape, "<f8")
                _new_series(group, "budgets", row.shape, "<f8")
                group["budgets"].attrs["terms"] = list(BUDGET_TERMS)
            stored = sorted(int(name[2:]) for name in group if name.startswith("C_"))
            if stored != classes:
                raise ValueError(
                    f"the frame carries classes {classes} and the file's {stored}"
                )
            for k, array in arrays.items():
                if group[f"C_{k}"].shape[1:] != array.shape:
                    raise ValueError(
                        f"class {k}'s field has shape {array.shape} and the file's "
                        f"frames {group[f'C_{k}'].shape[1:]}"
                    )
            steps = group["steps"]
            if steps.shape[0] and step <= int(steps[-1]):
                raise ValueError(
                    f"step {step} is not above the last frame's ({int(steps[-1])}); "
                    "a restart discards the frames after its checkpoint first"
                )
            _append(group["steps"], np.int64(step))
            _append(group["times"], np.float64(t))
            for k, array in arrays.items():
                _append(group[f"C_{k}"], array)
            _append(group["budgets"], row)

    def read_frames(self) -> TransportFrames:
        """Read back the frames ``write_frame`` appended.

        Raises
        ------
        KeyError
            If the file holds no frames.
        """
        with h5py.File(self.path, "r") as run_file:
            group = run_file.get(TRANSPORT)
            if group is None or "steps" not in group:
                raise KeyError(f"{self.path} holds no /{TRANSPORT}/ frames")
            steps = group["steps"][()]
            times = group["times"][()]
            fields = {
                int(name[2:]): group[name][()]
                for name in group
                if name.startswith("C_")
            }
            rows = group["budgets"][()]
        budgets = tuple(tuple(_budget_of(r) for r in frame) for frame in rows)
        return TransportFrames(steps, times, dict(sorted(fields.items())), budgets)

    def write_sensor_readings(
        self,
        steps: np.ndarray,
        times: np.ndarray,
        readings: np.ndarray,
        names: Sequence[str],
    ) -> None:
        """Append sensor readings for one or more steps.

        Parameters
        ----------
        steps : np.ndarray
            [n] integers, increasing and above the last appended step.
        times : np.ndarray
            [n] seconds.
        readings : np.ndarray
            [n, classes, sensors], the concentration each sensor reads.
        names : Sequence[str]
            The sensors in the order of the last axis; the first write sets
            them and every later write must give the same.

        Raises
        ------
        ValueError
            If the shapes disagree with each other or with the file's, the
            names differ from the file's, or the steps are not increasing
            and above the last appended step.
        """
        steps = np.asarray(steps, dtype="<i8")
        times = np.asarray(times, dtype="<f8")
        readings = np.asarray(readings, dtype="<f8")
        names = [str(name) for name in names]
        if steps.ndim != 1 or times.shape != steps.shape or readings.ndim != 3:
            raise ValueError(
                "steps and times must be [n] and readings [n, classes, sensors]"
            )
        n = steps.shape[0]
        if readings.shape[0] != n or readings.shape[2] != len(names):
            raise ValueError(
                f"readings of shape {readings.shape} for {n} steps and "
                f"{len(names)} sensors"
            )
        if np.any(np.diff(steps) <= 0):
            raise ValueError("the steps must be increasing")
        with h5py.File(self.path, "a") as run_file:
            group = run_file.require_group(TRANSPORT).require_group(SENSORS)
            if "steps" not in group:
                group.attrs["names"] = names
                _new_series(group, "steps", (), "<i8")
                _new_series(group, "times", (), "<f8")
                _new_series(group, "readings", readings.shape[1:], "<f8")
            if [str(x) for x in group.attrs["names"]] != names:
                raise ValueError(
                    f"the sensors are {names} and the file's "
                    f"{list(group.attrs['names'])}"
                )
            if group["readings"].shape[1:] != readings.shape[1:]:
                raise ValueError(
                    f"readings of {readings.shape[1:]} per step and the file's "
                    f"{group['readings'].shape[1:]}"
                )
            stored = group["steps"]
            if n and stored.shape[0] and steps[0] <= stored[-1]:
                raise ValueError(
                    f"step {int(steps[0])} is not above the last reading's "
                    f"({int(stored[-1])})"
                )
            _append(stored, steps, many=True)
            _append(group["times"], times, many=True)
            _append(group["readings"], readings, many=True)

    def read_sensor_readings(self) -> SensorReadings:
        """Read back the readings ``write_sensor_readings`` appended.

        Raises
        ------
        KeyError
            If the file holds no sensor readings.
        """
        with h5py.File(self.path, "r") as run_file:
            group = run_file.get(TRANSPORT, {}).get(SENSORS)
            if group is None:
                raise KeyError(f"{self.path} holds no /{TRANSPORT}/{SENSORS}/")
            return SensorReadings(
                names=tuple(str(x) for x in group.attrs["names"]),
                steps=group["steps"][()],
                times=group["times"][()],
                readings=group["readings"][()],
            )

    def discard_after(self, step: int) -> None:
        """Drop the frames, sensor readings and step rows after ``step``.

        A march restarted from its checkpoint at ``step`` writes them again;
        those a crash left after the checkpoint would otherwise be written
        twice.

        Parameters
        ----------
        step : int
            The checkpoint's step; frames and readings at it are kept.

        Raises
        ------
        TypeError
            If ``step`` is not an int (a bool is not one here).
        ValueError
            If ``step`` is negative.
        """
        step = _step(step)
        if not self.path.exists():
            return
        with h5py.File(self.path, "a") as run_file:
            holders = []
            for name, child, key in (
                (TRANSPORT, None, "steps"),
                (TRANSPORT, SENSORS, "steps"),
                (TRANSIENT, None, "steps"),
                (TRANSIENT, SERIES, "step"),
            ):
                group = run_file.get(name)
                if group is not None and child is not None:
                    group = group.get(child)
                if group is not None and key in group:
                    holders.append((group, key))
            for holder, key in holders:
                keep = int(np.searchsorted(holder[key][()], step, side="right"))
                for dataset in holder.values():
                    if isinstance(dataset, h5py.Dataset):
                        dataset.resize(keep, axis=0)

    def discard_march(self) -> None:
        """Drop every frame and sensor reading: the whole ``/transport/`` group.

        A restart that finds no march checkpoint (none written yet, or no
        ``checkpoint_interval``) marches again from step 0, so what an
        interrupted march wrote is written again. ``/config``, ``/flow/``
        and ``/recovery/`` are kept.
        """
        if not self.path.exists():
            return
        with h5py.File(self.path, "a") as run_file:
            if TRANSPORT in run_file:
                del run_file[TRANSPORT]

    # ------------------------------------------------------------------
    # /transient/ (tier 2)
    # ------------------------------------------------------------------

    def write_transient_frame(
        self, step: int, t: float, u: np.ndarray, v: np.ndarray, p: np.ndarray
    ) -> None:
        """Append one frame of a tier-2 run: the faces and the pressure.

        Parameters
        ----------
        step : int
            The step, zero at the initial state; above the last frame's.
        t : float
            Simulated time, seconds.
        u, v, p : np.ndarray
            The faces [ny, nx+1], [ny+1, nx] and the pressure [ny, nx].

        Raises
        ------
        TypeError
            If ``step`` is not an int (a bool is not one here) or ``t`` not
            a real number.
        ValueError
            If ``step`` is negative or not above the last frame's, or a
            field's shape differs from the first frame's.
        """
        step = _step(step)
        t = _real("t", t)
        arrays = {
            name: np.asarray(array, dtype="<f8")
            for name, array in (("u", u), ("v", v), ("p", p))
        }
        with h5py.File(self.path, "a") as run_file:
            group = run_file.require_group(TRANSIENT)
            if "steps" not in group:
                _new_series(group, "steps", (), "<i8")
                _new_series(group, "times", (), "<f8")
                for name, array in arrays.items():
                    _new_series(group, name, array.shape, "<f8")
            for name, array in arrays.items():
                if group[name].shape[1:] != array.shape:
                    raise ValueError(
                        f"{name} has shape {array.shape} and the file's frames "
                        f"{group[name].shape[1:]}"
                    )
            steps = group["steps"]
            if steps.shape[0] and step <= int(steps[-1]):
                raise ValueError(
                    f"step {step} is not above the last frame's ({int(steps[-1])}); "
                    "a restart discards the frames after its checkpoint first"
                )
            _append(group["steps"], np.int64(step))
            _append(group["times"], np.float64(t))
            for name, array in arrays.items():
                _append(group[name], array)

    def read_transient_frames(self) -> TransientFrames:
        """Read back the frames ``write_transient_frame`` appended.

        Raises
        ------
        KeyError
            If the file holds no tier-2 frames.
        """
        with h5py.File(self.path, "r") as run_file:
            group = run_file.get(TRANSIENT)
            if group is None or "steps" not in group:
                raise KeyError(f"{self.path} holds no /{TRANSIENT}/ frames")
            return TransientFrames(
                **{name: group[name][()] for name in ("steps", "times", "u", "v", "p")}
            )

    def write_step_series(self, rows: StepSeries) -> None:
        """Append rows of a tier-2 march's per-step series.

        Parameters
        ----------
        rows : StepSeries
            The rows, steps above the last stored; an empty series writes
            nothing.

        Raises
        ------
        TypeError
            If ``rows`` is not a StepSeries.
        ValueError
            If the first step is not above the last stored.
        """
        if not isinstance(rows, StepSeries):
            raise TypeError(f"rows must be a StepSeries, got {type(rows).__name__}")
        if rows.step.size == 0:
            return
        with h5py.File(self.path, "a") as run_file:
            group = run_file.require_group(TRANSIENT).require_group(SERIES)
            if "step" not in group:
                for name in STEP_SERIES_INTEGERS:
                    _new_series(group, name, (), "<i8")
                for name in STEP_SERIES_REALS:
                    _new_series(group, name, (), "<f8")
            stored = group["step"]
            if stored.shape[0] and int(rows.step[0]) <= int(stored[-1]):
                raise ValueError(
                    f"step {int(rows.step[0])} is not above the last stored "
                    f"({int(stored[-1])}); a restart discards the rows after its "
                    "checkpoint first"
                )
            for name in STEP_SERIES_INTEGERS + STEP_SERIES_REALS:
                _append(group[name], getattr(rows, name), many=True)

    def read_step_series(self) -> StepSeries:
        """Read back the per-step series, every row written.

        Raises
        ------
        KeyError
            If the file holds no step series.
        """
        with h5py.File(self.path, "r") as run_file:
            group = run_file.get(TRANSIENT)
            group = None if group is None else group.get(SERIES)
            if group is None or "step" not in group:
                raise KeyError(f"{self.path} holds no /{TRANSIENT}/{SERIES}/ rows")
            return StepSeries(
                **{
                    name: group[name][()]
                    for name in STEP_SERIES_INTEGERS + STEP_SERIES_REALS
                }
            )

    def write_transient_checkpoint(
        self, step: int, t: float, state: FlowState, previous: FlowState | None
    ) -> None:
        """Replace the tier-2 march's checkpoint, atomically, keeping the others.

        Parameters
        ----------
        step : int
            Steps taken, non-negative.
        t : float
            Simulated time, seconds.
        state : FlowState
            The faces and pressure after ``step``, ``outer`` the run's count.
        previous : FlowState or None
            The state after ``step - 1``, which BDF2 reads; None for none.

        Raises
        ------
        TypeError
            If ``step`` is not an int, ``t`` not a real number, or a state
            not a FlowState.
        """
        step = _step(step)
        t = _real("t", t)
        for name, given in (("state", state), ("previous", previous)):
            if given is not None and not isinstance(given, FlowState):
                raise TypeError(
                    f"{name} must be a FlowState, got {type(given).__name__}"
                )

        def write(group: h5py.Group) -> None:
            group.attrs["step"] = step
            group.attrs["t"] = np.float64(t)
            group.attrs["outer"] = state.outer
            for name in _FIELDS:
                group.create_dataset(name, data=getattr(state, name), dtype="<f8")
            if previous is not None:
                group.attrs["previous_outer"] = previous.outer
                for name in _FIELDS:
                    group.create_dataset(
                        f"previous_{name}", data=getattr(previous, name), dtype="<f8"
                    )

        self._replace_group(TRANSIENT_CHECKPOINT, write)

    def read_transient_checkpoint(self) -> tuple[float, TransientResume]:
        """Read back the tier-2 checkpoint: its time and where the march resumes.

        Returns
        -------
        tuple[float, TransientResume]
            The time, s, and the step, the state and the previous level, the
            arrays bitwise as written.

        Raises
        ------
        KeyError
            If there is no checkpoint file or it holds no
            ``/transient_checkpoint/``.
        """
        with self._open_checkpoint_file(TRANSIENT_CHECKPOINT) as group:
            attrs = dict(group.attrs)
            state = FlowState(
                **{name: group[name][()] for name in _FIELDS}, outer=int(attrs["outer"])
            )
            previous = None
            if "previous_u" in group:
                previous = FlowState(
                    **{name: group[f"previous_{name}"][()] for name in _FIELDS},
                    outer=int(attrs["previous_outer"]),
                )
        return float(attrs["t"]), TransientResume(
            state=state, previous=previous, step=int(attrs["step"])
        )

    # ------------------------------------------------------------------
    # /recovery/
    # ------------------------------------------------------------------

    def write_recovery(self, record: RecoveryRecord) -> None:
        """Write a clean-out's recovery outputs, replacing any the file holds.

        Parameters
        ----------
        record : RecoveryRecord
            The outputs at the end of the run. A restart that runs on to a
            later end writes its own over the earlier run's.

        Raises
        ------
        TypeError
            If ``record`` is not a RecoveryRecord.
        """
        if not isinstance(record, RecoveryRecord):
            raise TypeError(
                f"record must be a RecoveryRecord, got {type(record).__name__}"
            )
        with h5py.File(self.path, "a") as run_file:
            if RECOVERY in run_file:
                del run_file[RECOVERY]
            group = run_file.create_group(RECOVERY)
            for name in _RECOVERY_ARRAYS:
                group.create_dataset(name, data=getattr(record, name), dtype="<f8")
            group.create_dataset("defined", data=record.defined.astype(np.int8))
            group.attrs["sensor_names"] = list(record.sensor_names)
            group.attrs["held"] = record.held
            for name in _RECOVERY_SCALARS:
                group.attrs[name] = np.float64(getattr(record, name))
            if record.hold_time is not None:
                group.attrs["hold_time"] = np.float64(record.hold_time)

    def read_recovery(self) -> RecoveryRecord:
        """Read back the recovery outputs ``write_recovery`` wrote.

        Returns
        -------
        RecoveryRecord
            As written, its arrays bitwise.

        Raises
        ------
        KeyError
            If the file holds no ``/recovery/``.
        """
        with h5py.File(self.path, "r") as run_file:
            if RECOVERY not in run_file:
                raise KeyError(f"{self.path} holds no /{RECOVERY}/")
            group = run_file[RECOVERY]
            arrays = {name: group[name][()] for name in _RECOVERY_ARRAYS}
            defined = group["defined"][()].astype(bool)
            attrs = dict(group.attrs)
        hold = attrs.get("hold_time")
        return RecoveryRecord(
            sensor_names=tuple(str(x) for x in attrs["sensor_names"]),
            hold_time=None if hold is None else float(hold),
            held=bool(attrs["held"]),
            defined=defined,
            **{name: float(attrs[name]) for name in _RECOVERY_SCALARS},
            **arrays,
        )

    # ------------------------------------------------------------------
    # The checkpoint file
    # ------------------------------------------------------------------

    def checkpoint(
        self,
        step: int,
        t: float,
        fields: Mapping[int, np.ndarray],
        budgets: Sequence[MassBudget],
        fingerprint: str,
        recovery: Mapping[str, np.ndarray] | None = None,
        plan_start: float | None = None,
    ) -> None:
        """Replace the march's checkpoint, atomically, keeping the steady one.

        Parameters
        ----------
        step : int
            Steps taken, non-negative.
        t : float
            Simulated time, seconds, as the march holds it.
        fields : Mapping[int, np.ndarray]
            Class index to its field, [ny, nx], classes 0 to n - 1.
        budgets : Sequence[MassBudget]
            The transport solver's ``budget``, one per class.
        fingerprint : str
            The flow state the march steps on.
        recovery : Mapping[str, np.ndarray], optional
            Named arrays the march carries besides, stored as float64 and
            read back bitwise (the recovery tracker's state, ADR-014 C).
        plan_start : float, optional
            The start of the plan the march steps, so a restart can find the
            checkpoint on a plan a restart before it divided again.

        Raises
        ------
        TypeError
            If ``step`` is not an int, ``t`` or ``plan_start`` not a real
            number, a budget not a MassBudget, or a recovery name not a str.
        ValueError
            If ``step`` is negative, the classes are not 0 to n - 1 with
            one budget each, a field is not 2-D, or the fingerprint is not a
            SHA-256.
        """
        step = _step(step)
        t = _real("t", t)
        arrays = _class_fields(fields, budgets)
        _check_fingerprint(fingerprint)
        if plan_start is not None:
            plan_start = _real("plan_start", plan_start)
        rows = np.array([_budget_row(b) for b in budgets], dtype="<f8")
        carried = {}
        for name, array in (recovery or {}).items():
            if not isinstance(name, str):
                raise TypeError(
                    f"a recovery name must be a str, got {type(name).__name__}"
                )
            carried[name] = np.asarray(array, dtype="<f8")

        def write(group: h5py.Group) -> None:
            group.attrs["step"] = step
            group.attrs["t"] = np.float64(t)
            group.attrs["fingerprint"] = fingerprint
            if plan_start is not None:
                group.attrs["plan_start"] = np.float64(plan_start)
            for k, array in arrays.items():
                group.create_dataset(f"C_{k}", data=array, dtype="<f8")
            budgets_set = group.create_dataset("budgets", data=rows, dtype="<f8")
            budgets_set.attrs["terms"] = list(BUDGET_TERMS)
            if carried:
                held = group.create_group(RECOVERY)
                for name, array in carried.items():
                    held.create_dataset(name, data=array, dtype="<f8")

        self._replace_group(CHECKPOINT, write)

    def read_checkpoint(self) -> MarchCheckpoint:
        """Read back the march's checkpoint.

        Raises
        ------
        KeyError
            If there is no checkpoint file or it holds no ``/checkpoint/``.
        """
        with self._open_checkpoint_file(CHECKPOINT) as group:
            attrs = dict(group.attrs)
            fields = {
                int(name[2:]): group[name][()]
                for name in group
                if name.startswith("C_")
            }
            rows = group["budgets"][()]
            carried = (
                {name: data[()] for name, data in group[RECOVERY].items()}
                if RECOVERY in group
                else {}
            )
        return MarchCheckpoint(
            step=int(attrs["step"]),
            t=float(attrs["t"]),
            fields=dict(sorted(fields.items())),
            budgets=tuple(_budget_of(row) for row in rows),
            fingerprint=str(attrs["fingerprint"]),
            recovery=carried,
            plan_start=float(attrs["plan_start"]) if "plan_start" in attrs else None,
        )

    def write_steady_checkpoint(
        self, checkpoint: SteadyCheckpoint, fingerprint: str
    ) -> None:
        """Replace the steady solve's checkpoint, atomically, keeping the march's.

        The writer ``solve_steady(on_checkpoint=...)`` is handed, with the
        fingerprint bound: ``functools.partial(run_file.write_steady_checkpoint,
        fingerprint=key)``.

        Parameters
        ----------
        checkpoint : SteadyCheckpoint
            The solve's checkpoint, periodic or at its stop.
        fingerprint : str
            The flow state the solve is solving for.

        Raises
        ------
        TypeError
            If ``checkpoint`` is not a SteadyCheckpoint.
        ValueError
            If the fingerprint is not a SHA-256.
        """
        if not isinstance(checkpoint, SteadyCheckpoint):
            raise TypeError(
                f"checkpoint must be a SteadyCheckpoint, got {type(checkpoint).__name__}"
            )
        _check_fingerprint(fingerprint)
        state = checkpoint.state

        def write(group: h5py.Group) -> None:
            names = _FIELDS + (_TURBULENCE if state.turbulent else ())
            for name in names:
                group.create_dataset(name, data=getattr(state, name), dtype="<f8")
            for name in _SERIES:
                values = np.asarray(getattr(checkpoint, name), dtype=np.float64)
                group.create_dataset(name, data=values, dtype="<f8")
            group.attrs["outer"] = state.outer
            group.attrs["pressure_cap_hits"] = checkpoint.pressure_cap_hits
            group.attrs["fingerprint"] = fingerprint
            if state.stop_reason is not None:
                group.attrs["stop_reason"] = state.stop_reason
            if checkpoint.rule_version is not None:
                group.attrs["rule_version"] = checkpoint.rule_version
            if checkpoint.eddy_viscosity_sha256 is not None:
                group.attrs["eddy_viscosity_sha256"] = checkpoint.eddy_viscosity_sha256

        self._replace_group(STEADY_CHECKPOINT, write)

    def read_steady_checkpoint(self) -> tuple[str, SteadyCheckpoint]:
        """Read back the steady checkpoint and the flow state it solves for.

        Returns
        -------
        tuple[str, SteadyCheckpoint]
            The fingerprint, and the checkpoint as written, its arrays and
            series bitwise.

        Raises
        ------
        KeyError
            If there is no checkpoint file or it holds no
            ``/steady_checkpoint/``.
        ValueError
            If the group holds some but not all of ``k``, ``eps`` and ``nu_t``.
        """
        with self._open_checkpoint_file(STEADY_CHECKPOINT) as group:
            arrays = {name: group[name][()] for name in _FIELDS}
            # Whichever of k, eps and nu_t the group holds: FlowState refuses
            # a partial set, where keying on all three would read a damaged
            # coupled checkpoint back as a laminar one, with no error.
            arrays |= {name: group[name][()] for name in _TURBULENCE if name in group}
            series = {
                name: tuple(float(x) for x in group[name][()]) for name in _SERIES
            }
            attrs = dict(group.attrs)
        reason = attrs.get("stop_reason")
        rule_version = attrs.get("rule_version")
        digest = attrs.get("eddy_viscosity_sha256")
        state = FlowState(
            **arrays,
            stop_reason=None if reason is None else str(reason),
            outer=int(attrs["outer"]),
        )
        checkpoint = SteadyCheckpoint(
            state=state,
            residual_history=series["residual_history"],
            velocity_steps=series["velocity_steps"],
            viscosity_steps=series["viscosity_steps"],
            rule_version=None if rule_version is None else int(rule_version),
            pressure_cap_hits=int(attrs["pressure_cap_hits"]),
            eddy_viscosity_sha256=None if digest is None else str(digest),
        )
        return str(attrs["fingerprint"]), checkpoint

    def _replace_group(self, name: str, write: Callable[[h5py.Group], None]) -> None:
        """Write the checkpoint file anew with group NAME replaced, then swap it in.

        Mode "w" truncates a temporary file an earlier crash left, so it is
        never read. Both files are closed before ``os.replace``, which
        Windows needs as well as the atomic swap.
        """
        with h5py.File(self.temporary_path, "w") as new:
            if self.checkpoint_path.exists():
                with h5py.File(self.checkpoint_path, "r") as old:
                    for kept in old:
                        if kept != name:
                            old.copy(old[kept], new, name=kept)
            write(new.create_group(name))
        os.replace(self.temporary_path, self.checkpoint_path)

    def _open_checkpoint_file(self, name: str) -> "_GroupReader":
        """The checkpoint file's group NAME, open for reading in a with block."""
        return _GroupReader(self.checkpoint_path, name)


class _GroupReader:
    """Open one group of an HDF5 file for reading, closing the file on exit."""

    def __init__(self, path: Path, name: str) -> None:
        self._path = path
        self._name = name
        self._file: h5py.File | None = None

    def __enter__(self) -> h5py.Group:
        if not self._path.exists():
            raise KeyError(f"{self._path} does not exist; no checkpoint was written")
        self._file = h5py.File(self._path, "r")
        if self._name not in self._file:
            self._file.close()
            raise KeyError(f"{self._path} holds no /{self._name}/ group")
        return self._file[self._name]

    def __exit__(self, *exc: object) -> None:
        if self._file is not None:
            self._file.close()


def _plain(value: object) -> object:
    """Mappings as dicts and sequences other than strings as lists, recursively."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _without_excluded(mapping: object) -> object:
    """A copy of a plain mapping with RESTART_EXCLUDED_KEYS removed where present."""
    if not isinstance(mapping, dict):
        return mapping
    out = {
        key: dict(item) if isinstance(item, dict) else item
        for key, item in mapping.items()
    }
    for section, key in RESTART_EXCLUDED_KEYS:
        block = out.get(section)
        if isinstance(block, dict):
            block.pop(key, None)
    return out


def _first_difference(old: object, new: object, path: tuple[str, ...]) -> str | None:
    """The dotted name of the first key, in OLD's order, where two mappings differ."""
    name = ".".join(path)
    if isinstance(old, dict) and isinstance(new, dict):
        for key in list(old) + [k for k in new if k not in old]:
            inner = (*path, str(key))
            if key not in old or key not in new:
                return ".".join(inner)
            found = _first_difference(old[key], new[key], inner)
            if found is not None:
                return found
        if list(old) != list(new):
            return f"the order of the keys under {name or 'the top level'}"
        return None
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return name
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            # An item is named after its list: obstacles[0], not obstacles.[0].
            item = (*path[:-1], f"{path[-1]}[{i}]") if path else (f"[{i}]",)
            found = _first_difference(a, b, item)
            if found is not None:
                return found
        return None
    if type(old) is not type(new) or old != new:
        return name
    return None


def _check_fingerprint(fingerprint: object) -> None:
    """A SHA-256 written as 64 lowercase hexadecimal digits, or a refusal."""
    if not isinstance(fingerprint, str):
        raise TypeError(f"fingerprint must be a str, got {type(fingerprint).__name__}")
    if len(fingerprint) != _SHA256_DIGITS or fingerprint.strip(_HEX_DIGITS):
        raise ValueError(
            f"fingerprint must be 64 lowercase hexadecimal digits, got {fingerprint!r}"
        )


def _real(name: str, value: object) -> float:
    """A real number that is not a bool, as a float, or a refusal."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _step(step: object) -> int:
    """A non-negative int that is not a bool, or a refusal."""
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
        raise TypeError(f"step must be an int, got {type(step).__name__}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    return int(step)


def _class_fields(
    fields: Mapping[int, np.ndarray], budgets: Sequence[MassBudget]
) -> dict[int, np.ndarray]:
    """Each class's field as 2-D float64, classes 0 to n - 1 with a budget each.

    Raises TypeError for a budget that is not a MassBudget and ValueError for
    other classes or a field that is not 2-D.
    """
    classes = sorted(fields)
    if classes != list(range(len(budgets))):
        raise ValueError(
            f"fields for classes {classes} and {len(budgets)} budgets: each class "
            "0 to n - 1 needs its field and its budget"
        )
    for k, budget in enumerate(budgets):
        if not isinstance(budget, MassBudget):
            raise TypeError(
                f"budget {k} must be a MassBudget, got {type(budget).__name__}"
            )
    arrays = {k: np.asarray(fields[k], dtype="<f8") for k in classes}
    for k, array in arrays.items():
        if array.ndim != 2:
            raise ValueError(f"class {k}'s field must be 2-D, got {array.ndim}-D")
    return arrays


def _budget_row(budget: MassBudget) -> list[float]:
    """A budget's terms in BUDGET_TERMS' order, None as NaN."""
    return [
        np.nan if budget.initial is None else budget.initial,
        budget.inflow,
        budget.outflow,
        budget.source,
        *(budget.deposited[surface] for surface in DEPOSIT_SURFACES),
        np.nan if budget.current is None else budget.current,
    ]


def _budget_of(row: np.ndarray) -> MassBudget:
    """The budget a row of BUDGET_TERMS holds, NaN in initial or current as None."""
    values = dict(zip(BUDGET_TERMS, (float(x) for x in row), strict=True))
    initial, current = values["initial"], values["current"]
    return MassBudget(
        initial=None if np.isnan(initial) else initial,
        inflow=values["inflow"],
        outflow=values["outflow"],
        source=values["source"],
        deposited={s: values[f"deposited_{s}"] for s in DEPOSIT_SURFACES},
        current=None if np.isnan(current) else current,
    )


def _new_series(
    group: h5py.Group, name: str, shape: tuple[int, ...], dtype: str
) -> None:
    """An empty dataset that grows along its first axis, one row per append."""
    group.create_dataset(
        name,
        shape=(0, *shape),
        maxshape=(None, *shape),
        dtype=dtype,
        chunks=(1, *shape) if shape else (64,),
    )


def _append(dataset: h5py.Dataset, value: np.ndarray, *, many: bool = False) -> None:
    """Append one row, or with ``many`` every row of VALUE, to a growing dataset."""
    rows = np.asarray(value)
    if not many:
        rows = rows[np.newaxis, ...]
    start = dataset.shape[0]
    dataset.resize(start + rows.shape[0], axis=0)
    dataset[start:] = rows
