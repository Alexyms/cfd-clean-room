"""YAML configuration loader with validation for the CFD simulation.

Loads simulation parameters from a YAML file, validates all values at
load time, and provides typed attribute access. Satisfies REQ-C01
(single source of truth) and REQ-C02 (fail-fast validation).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from os import PathLike
from pathlib import Path
from types import MappingProxyType

import yaml


@dataclass(frozen=True)
class BoundarySpec:
    """Specification for a single boundary condition.

    Parameters
    ----------
    type : str
        Boundary condition type: "velocity_inlet", "pressure_outlet",
        "fixed_flow_outlet" or "wall".
    location : str
        Domain edge: "top", "bottom", "left", or "right"; or "obstacle"
        for an equipment vent on an obstacle face (ADR-014 J).
    x_start : float or None
        Start of boundary segment along x axis (for top/bottom, a domain
        edge or an obstacle face).
    x_end : float or None
        End of boundary segment along x axis (for top/bottom).
    y_start : float or None
        Start of boundary segment along y axis (for left/right).
    y_end : float or None
        End of boundary segment along y axis (for left/right).
    velocity : float or None
        Prescribed velocity magnitude for velocity_inlet type.
        Applied normal to the wall surface. On a fixed_flow_outlet, the
        outward normal velocity it holds, or None for an outlet that
        shares the inflow the stated outlets leave (ADR-012 D, amended
        2026-10-08).
    u_velocity : float or None
        Explicit u-component for velocity_inlet. When present,
        overrides the automatic normal decomposition of velocity.
    v_velocity : float or None
        Explicit v-component for velocity_inlet. When present,
        overrides the automatic normal decomposition of velocity.
    concentration : tuple[float, ...] or None
        Upstream concentration carried by a velocity_inlet, one
        non-negative value per configured particle class, particles per
        cubic meter. None means a clean supply (ADR-011 E). A tuple, so
        the loaded configuration cannot be edited in place.
    hepa_filtered : bool
        velocity_inlet only. When True the carried concentration is
        reduced by the class's HEPA efficiency. Default False.
    deposition_surface : str or None
        wall only: "floor", "ceiling", "wall" or "none", overriding the
        edge's default surface for deposition. None means the edge
        decides (bottom is floor, top is ceiling, left and right are
        walls).
    turbulence_intensity : float or None
        velocity_inlet only, with the turbulence section present: the
        intensity I of the air entering, a fraction in (0, 1), which sets
        the inflow ``k = 1.5 (I |u_n|)^2`` (ADR-012 C). None otherwise.
    dissipation_length : float or None
        Likewise: the length l_e in ``eps = k^(3/2) / l_e``, metres,
        positive, in ADR-012 C's convention, with no ``C_mu^(3/4)``. None
        otherwise.
    obstacle : str or None
        location "obstacle" only: the name of the obstacle the vent sits
        on. None on a domain-edge segment.
    face : str or None
        location "obstacle" only: the obstacle's face, "top", "bottom",
        "left" or "right". None on a domain-edge segment.
    recirculate_from : str or None
        A velocity_inlet vent only: the name of a fixed_flow_outlet vent on
        the same obstacle whose intake the discharge returns to the room,
        so the discharge carries the intake's concentration, lagged one
        transport step (prompt 61's amendment to ADR-014 J). None means
        the discharge carries ``concentration``, or nothing.

    The three concentration keys, the two turbulence keys and the three
    vent keys are keyword-only. A velocity_inlet whose
    prescribed normal component is zero (a tangential lid) admits no air
    and is a wall to the scalar layer (ADR-011 E, amended 2026-10-03), so
    it may carry neither ``concentration`` nor ``hepa_filtered``.
    """

    type: str
    location: str
    x_start: float | None = None
    x_end: float | None = None
    y_start: float | None = None
    y_end: float | None = None
    velocity: float | None = None
    u_velocity: float | None = None
    v_velocity: float | None = None
    concentration: tuple[float, ...] | None = field(default=None, kw_only=True)
    hepa_filtered: bool = field(default=False, kw_only=True)
    deposition_surface: str | None = field(default=None, kw_only=True)
    turbulence_intensity: float | None = field(default=None, kw_only=True)
    dissipation_length: float | None = field(default=None, kw_only=True)
    obstacle: str | None = field(default=None, kw_only=True)
    face: str | None = field(default=None, kw_only=True)
    recirculate_from: str | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class ObstacleSpec:
    """Specification for an internal obstacle (solid region).

    Parameters
    ----------
    name : str
        Descriptive name for the obstacle.
    x_start : float
        Left edge x coordinate in meters.
    x_end : float
        Right edge x coordinate in meters.
    y_start : float
        Bottom edge y coordinate in meters.
    y_end : float
        Top edge y coordinate in meters.
    """

    name: str
    x_start: float
    x_end: float
    y_start: float
    y_end: float


@dataclass(frozen=True)
class SensorSpec:
    """Specification for a contamination sensor location.

    Parameters
    ----------
    name : str
        Sensor identifier.
    x : float
        Sensor x coordinate in meters.
    y : float
        Sensor y coordinate in meters.
    """

    name: str
    x: float
    y: float


@dataclass(frozen=True)
class HepaReference:
    """HEPA filter reference efficiency data for interpolation.

    Parameters
    ----------
    diameters : list[float]
        Reference particle diameters in meters, sorted ascending.
    efficiencies : list[float]
        Single-pass collection efficiency for each diameter.
    """

    diameters: list[float] = field(default_factory=list)
    efficiencies: list[float] = field(default_factory=list)


# The time schemes of a transient run (ADR-014 decision 3): BDF2 after an
# implicit Euler first step, or implicit Euler throughout as the control.
BDF2 = "bdf2"
IMPLICIT_EULER = "implicit_euler"
TRANSIENT_SCHEMES = (BDF2, IMPLICIT_EULER)
# How far solver.t_end / solver.dt may sit from a whole number of steps,
# relative: rounding in the division, not a shortened last step, which BDF2
# does not take.
STEP_COUNT_ROUNDING = 1e-9
TRANSIENT_KEY = "transient"
_TRANSIENT_KEYS = frozenset({"scheme", "step_tol", "step_cap", "forces"})
_FORCE_KEYS = frozenset({"obstacle", "reference_velocity"})


@dataclass(frozen=True)
class ForceSpec:
    """The obstacle a transient run's lift and drag are read on (ADR-014 F).

    Parameters
    ----------
    obstacle : str
        The name of one of the configuration's obstacles.
    reference_velocity : float
        U of the coefficients, m/s, positive.
    """

    obstacle: str
    reference_velocity: float


@dataclass(frozen=True)
class TransientSpec:
    """A tier-2 run: the flow marched through time (ADR-014 decisions 3 to 5).

    The step and the end are ``solver.dt`` and ``solver.t_end``, the output
    and checkpoint cadences ``solver.output_interval`` and
    ``solver.checkpoint_interval``, in steps.

    Parameters
    ----------
    scheme : str
        "bdf2" or "implicit_euler" (``TRANSIENT_SCHEMES``).
    step_tol : float
        The per-step criterion over the largest prescribed boundary
        velocity, positive.
    step_cap : int
        The most outer iterations a step may take, positive.
    forces : ForceSpec or None
        The obstacle whose lift and drag the run records; None for none.
    """

    scheme: str
    step_tol: float
    step_cap: int
    forces: ForceSpec | None = None


@dataclass(frozen=True)
class Breakpoint:
    """One point of an axis's interior clustering (ADR-014 decision 11).

    Parameters
    ----------
    at : float
        The coordinate, m; a face of the mesh.
    spacing : float
        The cell width wanted there, m, positive.
    """

    at: float
    spacing: float


@dataclass(frozen=True)
class StretchSpec:
    """Geometric clustering for one mesh axis: at the walls, or between breakpoints.

    Wall clustering gives one of two quantities and the mesh derives the
    other because the cell count fixes their relationship (see
    src/mesh.py). Interior clustering gives ``breakpoints`` instead, and the
    loader derives the faces from them (``breakpoint_faces``). The default is
    a uniform axis.

    Parameters
    ----------
    ratio : float
        Geometric ratio between adjacent cell widths from each wall toward
        the center, >= 1. A ratio of 1 is a uniform axis.
    min_spacing : float or None
        Width of the wall-adjacent cell in meters. When given, the ratio
        is derived from it and ``ratio`` is ignored.
    breakpoints : tuple[Breakpoint, ...]
        Interior clustering: from the first breakpoint, at 0, to the last,
        at the axis length, each with the spacing wanted there; empty for
        wall clustering or a uniform axis.
    faces : tuple[float, ...]
        The faces ``breakpoint_faces`` derived from ``breakpoints``, which
        the mesh takes as they are; empty without breakpoints.
    """

    ratio: float = 1.0
    min_spacing: float | None = None
    breakpoints: tuple[Breakpoint, ...] = ()
    faces: tuple[float, ...] = ()


# The largest ratio between two adjacent cell widths an axis with breakpoints
# may have. Each stencil's truncation error grows with the jump in width
# between neighbours, and 1.2 is the common ceiling for second-order
# finite-volume grids (Ferziger and Peric, ch. 3); the wall-clustered presets
# stay below it too.
MAX_GROWTH_RATIO = 1.2
# Relative tolerance of the bisection for a segment's growth ratio, as the wall
# clustering's in src/mesh.py.
_GROWTH_TOLERANCE = 1e-14
# A count within this of a whole number is that number, not the next: a length
# of 7 at a spacing of 0.35 is 20 cells, though 7 / 0.35 rounds above 20.
_COUNT_ROUNDING = 1e-9


def _segment_widths(length: float, start: float, end: float) -> list[float]:
    """Cell widths over one segment, growing geometrically from its finer end.

    The finer end's spacing is kept exactly; the count is the smallest whole
    number at least that of the geometric series growing from it to the
    coarser end's spacing, and the ratio is then the one whose widths sum to
    the length. A count at or above the series' makes the ratio at or below
    the series', so no width exceeds the coarser end's spacing. Equal
    spacings give equal widths, the fewest no wider than the spacing.
    """
    fine, coarse = min(start, end), max(start, end)
    if fine == coarse:
        n = max(1, math.ceil(length / fine - _COUNT_ROUNDING))
        return [length / n] * n
    ideal_ratio = (length - fine) / (length - coarse)
    ideal = 1.0 + math.log(coarse / fine) / math.log(ideal_ratio)
    n = max(2, math.ceil(ideal - _COUNT_ROUNDING))
    while n > 1 and n * fine >= length:
        n -= 1
    if n == 1:
        return [length]

    def span(ratio: float) -> float:
        return fine * (ratio**n - 1.0) / (ratio - 1.0)

    lo, hi = 1.0, 2.0
    while span(hi) < length:
        hi *= 2.0
    while hi - lo > _GROWTH_TOLERANCE * hi:
        mid = 0.5 * (lo + hi)
        if span(mid) < length:
            lo = mid
        else:
            hi = mid
    ratio = 0.5 * (lo + hi)
    widths = [fine * ratio**k for k in range(n)]
    return widths if start <= end else widths[::-1]


def breakpoint_faces(breakpoints: Sequence[Breakpoint]) -> tuple[float, ...]:
    """The faces of an axis clustered between breakpoints (ADR-014 decision 11).

    Parameters
    ----------
    breakpoints : Sequence[Breakpoint]
        At least two, in increasing ``at``, the first at the axis's start
        and the last at its end, each spacing positive and below the length
        of the segments beside it (the loader checks all of it).

    Returns
    -------
    tuple[float, ...]
        Every face, increasing, with each breakpoint a face exactly, so an
        obstacle edge placed on a breakpoint lies on a face.

    Notes
    -----
    Between two breakpoints the widths grow geometrically from the finer
    end's spacing, which is kept exactly; the count is fixed by the series
    that would reach the coarser end's spacing, and the ratio is then solved
    for so that the widths fill the segment (``_segment_widths``).
    """
    faces = [breakpoints[0].at]
    for left, right in pairwise(breakpoints):
        widths = _segment_widths(right.at - left.at, left.spacing, right.spacing)
        x = left.at
        for width in widths[:-1]:
            x += width
            faces.append(x)
        faces.append(right.at)
    return tuple(faces)


@dataclass(frozen=True)
class TransportSpec:
    """The transport section: what the scalar solver reads (ADR-011 I).

    Parameters
    ----------
    cfl_number : float
        Courant number the explicit advection step runs at, in
        (0, CFL_NUMBER_BOUND]. An accuracy choice below the bound: the
        forward Euler error grows with it.
    advection_scheme : str
        "umist" (QUICK bounded by the UMIST limiter, the default) or
        "upwind" (first order, the comparison scheme).
    max_diffusion_iter : int
        Cap on Jacobi sweeps of the implicit diffusion and deposition
        system per class per step.
    diffusion_tol : float
        Tolerance the implicit diffusion solve iterates to.
    turbulent_schmidt : float or None
        Sc_t of ``D_t = nu_t / Sc_t``, the turbulent particle diffusivity
        (REQ-T13, ADR-012 F and decision 8): positive and finite, with no
        default in code. None when the key is absent, and the solver then
        refuses an eddy viscosity field. A laminar configuration needs none.
    """

    cfl_number: float
    advection_scheme: str
    max_diffusion_iter: int
    diffusion_tol: float
    turbulent_schmidt: float | None = None


@dataclass(frozen=True)
class TurbulenceSpec:
    """The turbulence section: what the k-epsilon model reads (ADR-012 I).

    Present means the model is on: the flow solver runs the coupled k and
    eps iteration (ECR-002 step 6), every velocity inlet that admits air
    states ``turbulence_intensity`` and ``dissipation_length``, and the
    stopping rule must be ``error_estimate``. Absent, the model is off and
    every other result is unchanged.

    Parameters
    ----------
    model : str
        "k_epsilon", the one model ECR-002 builds.
    variant : str
        "standard" (the default) or "rng" (ADR-012 A, decision 2).
    wall_treatment : str
        "scalable_wall_functions" (ADR-012 B, decision 3).
    cfl_number : float
        The pseudo-time Courant number of the k and eps step, in
        (0, CFL_NUMBER_BOUND] (ADR-012 C).
    advection_scheme : str
        The face value k and eps are advected with: "upwind" (the default,
        ADR-012 C's note of 2026-10-10) or "umist" (QUICK bounded by the
        UMIST limiter, the step as first built).
    alpha_turbulence : float
        Under-relaxation of the eddy viscosity in the coupled solve, in
        (0, 1].
    max_iter : int
        Cap on conjugate gradient iterations of each implicit k and eps
        solve (Jacobi sweeps until prompt 51).
    tol : float
        Tolerance the implicit k and eps solves iterate to.
    """

    model: str
    variant: str
    wall_treatment: str
    cfl_number: float
    advection_scheme: str
    alpha_turbulence: float
    max_iter: int
    tol: float


@dataclass(frozen=True)
class Replace:
    """An event's replacement of one boundary segment (ADR-014 B).

    Parameters
    ----------
    segment : str
        The name of the segment of the merged base that is replaced.
    segments : tuple[tuple[str, Mapping[str, object]], ...]
        The segments that take its place for the event's duration, in
        configuration order, each a name and its keys as the file states
        them (read-only; a list value is a tuple). Empty removes the
        segment, and the coverage rule's default, a wall, takes its faces
        (the amendment of 2026-10-17: a vent's removal is how a fan stops
        or an exhaust fails, since a wall is refused on an obstacle face).
    """

    segment: str
    segments: tuple[tuple[str, Mapping[str, object]], ...]


@dataclass(frozen=True)
class Set:
    """An event's change of keys of one boundary segment in place (ADR-014 B).

    Parameters
    ----------
    segment : str
        The name of the segment of the merged base whose keys change.
    keys : Mapping[str, object]
        The keys and their new values, read-only, merged over the
        segment's own for the event's duration.
    """

    segment: str
    keys: Mapping[str, object]


@dataclass(frozen=True)
class Source:
    """An event's particle source over a box (ADR-014 B).

    Exactly one of ``rate`` and ``emission`` is given. The box's non-SOLID
    cells are the mesh's, so the manager (src/scenarios.py) turns an
    emission into a rate and refuses a box with no non-SOLID cell centre.

    Parameters
    ----------
    x_start, x_end, y_start, y_end : float
        The box, metres, within the domain and not empty.
    rate : tuple[float, ...] or None
        Per class, particles per cubic metre per second, uniform over the
        non-SOLID cells whose centres lie in the box.
    emission : tuple[float, ...] or None
        Per class, particles per second per metre of depth, divided by the
        box's non-SOLID area (prompt 47's convention).
    """

    x_start: float
    x_end: float
    y_start: float
    y_end: float
    rate: tuple[float, ...] | None = None
    emission: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Initial:
    """The initial concentration of every class (ADR-014 B).

    Parameters
    ----------
    values : tuple[float, ...]
        Per class, particles per cubic metre, uniform over the non-SOLID
        cells: the stated ``concentration``, or the class's threshold times
        ``threshold_factor``.
    threshold_factor : float or None
        The factor when the event states one (the clean-out's 100, the
        recovery test's challenge), else None.
    """

    values: tuple[float, ...]
    threshold_factor: float | None = None


@dataclass(frozen=True)
class EventSpec:
    """One event of a scenario (ADR-014 B).

    Parameters
    ----------
    name : str
        The event's name, unique in the scenario.
    start : float
        Start time, seconds, finite and non-negative; zero for an initial.
    duration : float or None
        Seconds, finite and positive; None for an initial.
    change : Replace, Set, Source or Initial
        What the event changes.
    """

    name: str
    start: float
    duration: float | None
    change: Replace | Set | Source | Initial

    @property
    def end(self) -> float | None:
        """``start + duration``, seconds; None for an initial."""
        if self.duration is None:
            return None
        return self.start + self.duration

    @property
    def segment(self) -> str | None:
        """The base segment a replace or a set touches; None otherwise."""
        if isinstance(self.change, (Replace, Set)):
            return self.change.segment
        return None


@dataclass(frozen=True, eq=False)
class ScenarioSpec:
    """The scenario section, validated (ADR-014 B, decision 7).

    Parameters
    ----------
    base_path : Path
        The base file the scenario extends, resolved.
    base : Mapping[str, object]
        The merged base: the base file's mapping with the scenario file's
        own sections merged over it recursively (lists replace whole),
        read-only. ``mapping(())`` gives an editable copy.
    events : tuple[EventSpec, ...]
        The events, in file order.
    t_end : float
        ``solver.t_end`` of the merged base, seconds.
    hold_time : float or None
        ``scenario.hold_time``, seconds, positive and below ``t_end``: a
        clean-out stops once the room has been within spec for this long
        (ADR-014 C, prompt 64's decision 2). None runs to ``t_end``. Given
        only with an initial event.
    path : Path or None
        The scenario file, resolved, when the configuration was loaded from
        one; None from ``SimConfig.from_dict``.

    Notes
    -----
    ADR-014 G lists ``checkpoint_interval`` here; it is the solver block's
    key, ``SimConfig.checkpoint_interval`` (ADR-014 D, step 3), the cadence
    of a steady solve's checkpoint as well as a march's.
    """

    base_path: Path
    base: Mapping[str, object]
    events: tuple[EventSpec, ...]
    t_end: float
    hold_time: float | None = None
    path: Path | None = None

    @property
    def time_tolerance(self) -> float:
        """Seconds within which two event times are one instant.

        ``EVENT_TIME_RTOL`` times ``t_end``: the overlap check, ``times`` and
        ``active`` read the same instant, so they cannot disagree.
        """
        return EVENT_TIME_RTOL * self.t_end

    def times(self) -> tuple[float, ...]:
        """Every interval boundary: 0, each event's start and end, and t_end.

        Returns
        -------
        tuple[float, ...]
            Sorted. A time within ``time_tolerance`` of the boundary before
            it, or of t_end, is that boundary, so no interval is shorter
            than the tolerance. An initial event adds nothing.
        """
        tolerance = self.time_tolerance
        inner = sorted(
            point
            for event in self.events
            if event.end is not None
            for point in (event.start, event.end)
        )
        out = [0.0]
        for point in inner:
            if point - out[-1] > tolerance and self.t_end - point > tolerance:
                out.append(point)
        out.append(self.t_end)
        return tuple(out)

    def is_active(self, event: EventSpec, t: float) -> bool:
        """Whether ``event`` is in force at time t, to ``time_tolerance``.

        Active from its start and expired at its end, each read as the
        instant ``times`` keeps: ``start - tol <= t < end - tol``. An initial
        event is never active; it sets the fields at t = 0.

        Parameters
        ----------
        event : EventSpec
            One of ``events``.
        t : float
            Seconds.

        Returns
        -------
        bool
        """
        if event.end is None:
            return False
        tolerance = self.time_tolerance
        return event.start - tolerance <= t < event.end - tolerance

    def active(self, t: float) -> tuple[EventSpec, ...]:
        """The events in force at time t: ``start <= t < start + duration``.

        An event is active from its start and has expired at its end, so a
        time on a boundary belongs to the interval that begins there; both
        are read to ``time_tolerance`` (``is_active``). An initial event is
        never active; it sets the fields at t = 0.

        Parameters
        ----------
        t : float
            Seconds.

        Returns
        -------
        tuple[EventSpec, ...]
            In file order.
        """
        return tuple(event for event in self.events if self.is_active(event, t))

    def mapping(self, events: tuple[EventSpec, ...]) -> dict:
        """The merged base with the boundary changes of ``events`` applied.

        A replacement's segments take the replaced segment's place in
        configuration order; a set merges its keys over the segment's.
        Sources and initials change no mapping.

        Parameters
        ----------
        events : tuple[EventSpec, ...]
            Usually ``active(t)``.

        Returns
        -------
        dict
            A new mapping with the structure of a configuration file.

        Raises
        ------
        ValueError
            If two active replacements give a segment the same name.
        """
        raw = _thaw(self.base)
        boundaries: dict = raw["boundaries"]
        for event in events:
            change = event.change
            if isinstance(change, Set):
                boundaries[change.segment] = {
                    **boundaries[change.segment],
                    **_thaw(change.keys),
                }
            elif isinstance(change, Replace):
                rebuilt: dict = {}
                for name, spec in boundaries.items():
                    pairs = (
                        [(n, _thaw(s)) for n, s in change.segments]
                        if name == change.segment
                        else [(name, spec)]
                    )
                    for new_name, new_spec in pairs:
                        if new_name in rebuilt:
                            raise ValueError(
                                f"scenario.events: '{event.name}' names a segment "
                                f"'{new_name}' that another active event's segment "
                                "already has"
                            )
                        rebuilt[new_name] = new_spec
                boundaries = rebuilt
        raw["boundaries"] = boundaries
        return raw


def _freeze(value: object) -> object:
    """A read-only copy: a mapping becomes a MappingProxyType, a list a tuple."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: object) -> object:
    """An editable copy of ``_freeze``'s output, as YAML would load it."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(v) for v in value]
    return value


def _read_mapping(path: Path) -> dict:
    """The YAML mapping a configuration file holds."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Configuration file must contain a YAML mapping")
    return raw


def merge_mappings(base: dict, override: dict) -> dict:
    """``override`` merged over ``base``, recursively (ADR-014 decision 7).

    Where both hold a mapping under a key the two are merged; anything
    else in ``override``, a list included, replaces the base's value
    whole. Keys keep the base's order, with the override's new keys after.

    Parameters
    ----------
    base, override : dict
        Configuration mappings; neither is modified.

    Returns
    -------
    dict
        A new mapping.
    """
    merged = {k: _thaw(v) for k, v in base.items()}
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_mappings(merged[key], value)
        else:
            merged[key] = _thaw(value)
    return merged


_VALID_BOUNDARY_TYPES: set[str] = {
    "velocity_inlet",
    "pressure_outlet",
    "fixed_flow_outlet",
    "wall",
}
_VALID_BOUNDARY_LOCATIONS: set[str] = {"top", "bottom", "left", "right"}

# Equipment vents (ADR-014 J, 2026-10-16): a segment on an obstacle face. Two
# existing types only: a fixed-flow outlet is a process exhaust, a velocity
# inlet a cooling discharge. A pressure outlet's zero-gradient copy would need
# the face beyond it, inside the obstacle, and an obstacle face is a wall
# already.
OBSTACLE_LOCATION = "obstacle"
OBSTACLE_FACES: tuple[str, ...] = ("top", "bottom", "left", "right")
VENT_TYPES: tuple[str, ...] = ("velocity_inlet", "fixed_flow_outlet")
_VENT_KEYS: tuple[str, ...] = ("obstacle", "face", "recirculate_from")
# The relative difference between the configured inflow and the stated
# outflow below which a room with no outlet to take a remainder is read as
# balanced, and a recirculating pair's two flows as equal. Far above the
# rounding of a product of two configured numbers and far below any imbalance
# a configuration means.
BALANCE_TOLERANCE = 1.0e-9
# The same rule on the mesh's faces (boundary_staggered.py), which the solver
# runs: the rounding of two face sums, not a tolerance a case chooses. It is
# the stricter of the two, so the configured one never admits a room the
# solver would run unbalanced (review 61 S3 co-located them).
MESH_BALANCE_TOLERANCE = 1.0e-12

# Stopping rules the solver block may name (src/stopping.py). velocity_step is
# the default, the rule every stored result was produced under.
VELOCITY_STEP = "velocity_step"
ERROR_ESTIMATE = "error_estimate"
STOPPING_RULES: tuple[str, ...] = (VELOCITY_STEP, ERROR_ESTIMATE)
DEFAULT_ITERATION_ERROR_TOL = 1.0e-6
# ECR-001 acceptance criterion 6: the per-cell imbalance and its signed domain
# sum each below 1e-10, absolute.
DEFAULT_MASS_IMBALANCE_TOL = 1.0e-10
# Momentum Jacobi sweeps per outer iteration (ECR-002 step 4). One is the
# sweep every stored result was produced with.
DEFAULT_MOMENTUM_SWEEPS = 1
# The relative residual the pressure correction solves to (REQ-S08 as amended
# 2026-10-06, ADR-013 D). The lower bound: on the product's first correction
# from rest the true residual cannot fall below about 1.3e-13 of the flux
# scale, so a level at or below about 1e-12 would run every such correction to
# its cap; 1e-10 keeps two orders above that and two below the default 1e-8.
# The upper bound excludes a level that asks for nothing.
PRESSURE_RTOL_BOUNDS: tuple[float, float] = (1.0e-10, 1.0)
# The key the weighted Jacobi solve read until ECR-003 step 1: pascals of
# change per sweep, which means nothing for the conjugate gradient solve. It is
# refused by name so a saved configuration cannot be read as the new key.
RETIRED_PRESSURE_TOL_KEY = "pressure_tol"
# The preconditioner of the pressure correction's conjugate gradients
# (src/pressure.py): one V-cycle of the additive-correction multigrid of
# src/multigrid.py, ADR-013 decision 1's named successor (prompt 53) and the
# default since prompt 54 (2026-10-14), or the diagonal, the solve every result
# stored before that was produced with, kept for parity work and the Phase 6
# port.
PRESSURE_JACOBI = "jacobi"
MULTIGRID = "multigrid"
PRESSURE_PRECONDITIONERS: tuple[str, ...] = (PRESSURE_JACOBI, MULTIGRID)
DEFAULT_PRESSURE_PRECONDITIONER = MULTIGRID
# Every key the solver block accepts, nine required and six optional, in the
# order the harness records them: it reads this tuple, so a key added here is
# in every harness row (GitHub issue 38). With optional keys a misspelt one
# would otherwise fall back to its default.
SOLVER_KEYS: tuple[str, ...] = (
    "dt",
    "t_end",
    "output_interval",
    "convergence_tol",
    "max_simple_iter",
    "alpha_velocity",
    "alpha_pressure",
    "max_pressure_iter",
    "pressure_rtol",
    "momentum_sweeps",
    "stopping_rule",
    "iteration_error_tol",
    "mass_imbalance_tol",
    "pressure_preconditioner",
    "checkpoint_interval",
)

# Transport section (ADR-011 I), optional: the velocity-only validation cases
# do not carry it. The limited face scheme is bounded under forward Euler at
# a Courant number of at most 1/2 (ADR-011 B); the bound is a property of the
# scheme and lives here as a constant. cfl_number is the Courant number a run
# advects at, at most the bound.
CFL_NUMBER_BOUND = 0.5
UMIST = "umist"
UPWIND = "upwind"
ADVECTION_SCHEMES: tuple[str, ...] = (UMIST, UPWIND)
_TRANSPORT_KEYS: frozenset[str] = frozenset(
    {
        "cfl_number",
        "advection_scheme",
        "max_diffusion_iter",
        "diffusion_tol",
        "turbulent_schmidt",
    }
)

# Turbulence section (ADR-012 I), optional: absent means the model is off.
# The k and eps step uses the transport scheme in pseudo-time, so its Courant
# number has the scheme's bound, CFL_NUMBER_BOUND. The model constants are
# not configuration: they define the published variants and live in
# src/turbulence.py.
K_EPSILON = "k_epsilon"
TURBULENCE_MODELS: tuple[str, ...] = (K_EPSILON,)
STANDARD = "standard"
RNG = "rng"
TURBULENCE_VARIANTS: tuple[str, ...] = (STANDARD, RNG)
SCALABLE_WALL_FUNCTIONS = "scalable_wall_functions"
WALL_TREATMENTS: tuple[str, ...] = (SCALABLE_WALL_FUNCTIONS,)
_TURBULENCE_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "variant",
        "wall_treatment",
        "cfl_number",
        "advection_scheme",
        "alpha_turbulence",
        "max_iter",
        "tol",
    }
)

# Every key a boundary segment accepts. With optional keys a misspelt one
# would otherwise change the physics with no message (REQ-C02).
_SEGMENT_KEYS: frozenset[str] = frozenset(
    {
        "type",
        "location",
        "x_start",
        "x_end",
        "y_start",
        "y_end",
        "velocity",
        "u_velocity",
        "v_velocity",
        "concentration",
        "hepa_filtered",
        "deposition_surface",
        "turbulence_intensity",
        "dissipation_length",
        *_VENT_KEYS,
    }
)

# Surfaces a wall segment may name for deposition (ADR-011 E). The first
# three are the orientations ParticlePhysics.deposition_velocity accepts;
# "none" switches deposition off on the segment.
DEPOSITION_FLOOR = "floor"
DEPOSITION_CEILING = "ceiling"
DEPOSITION_WALL = "wall"
DEPOSITION_NONE = "none"
DEPOSITION_SURFACES: tuple[str, ...] = (
    DEPOSITION_FLOOR,
    DEPOSITION_CEILING,
    DEPOSITION_WALL,
    DEPOSITION_NONE,
)

# The scenario section (ADR-014 B, decision 7). A file with it extends the
# file its base names: the file's own sections merge over the base's, and
# the events change the merged boundaries, the sources and the initial field.
SCENARIO_KEY = "scenario"
_SCENARIO_KEYS: frozenset[str] = frozenset({"base", "events", "hold_time"})
# The four changes; an event states exactly one.
REPLACE = "replace"
SET = "set"
SOURCE = "source"
INITIAL = "initial"
EVENT_CHANGES: tuple[str, ...] = (REPLACE, SET, SOURCE, INITIAL)
_EVENT_KEYS: frozenset[str] = frozenset({"name", "start", "duration", *EVENT_CHANGES})
_REPLACE_KEYS: frozenset[str] = frozenset({"segment", "with"})
_SET_KEYS: frozenset[str] = frozenset({"segment", "keys"})
_SOURCE_BOX_KEYS: tuple[str, ...] = ("x_start", "x_end", "y_start", "y_end")
_SOURCE_KEYS: frozenset[str] = frozenset({*_SOURCE_BOX_KEYS, "rate", "emission"})
_INITIAL_KEYS: frozenset[str] = frozenset({"concentration", "threshold_factor"})
# The run keys a scenario states in its own sections rather than inherits
# (ADR-014 B: each reference scenario has its t_end, cfl_number and
# output_interval of its own), as (section, key).
SCENARIO_RUN_KEYS: tuple[tuple[str, str], ...] = (
    ("solver", "t_end"),
    ("solver", "output_interval"),
    ("transport", "cfl_number"),
)
# The relative tolerance a threshold key's diameter is matched to a class's
# at: both are parsed from the same decimal text, so they agree to rounding.
_DIAMETER_MATCH_RTOL = 1.0e-9
# Two event times closer than this times solver.t_end are one instant (review
# 62 S5). start + duration carries the rounding of a sum, about 1e-16 of the
# times (0.1 + 0.2 is above 0.3), so two events written to meet would otherwise
# overlap by that much or leave a sliver interval between them. 1e-9 of t_end
# is 1.2e-7 s at t_end 120 s, far below a transport step (a few milliseconds on
# the product grid at Courant 0.4) and far above the rounding.
EVENT_TIME_RTOL = 1.0e-9


class SimConfig:
    """Simulation configuration loaded and validated from a YAML file.

    All simulation parameters are validated at load time. Missing keys,
    out-of-range values, and type mismatches raise immediately with
    clear error messages (REQ-C02).

    Parameters
    ----------
    yaml_path : str or Path
        Path to the YAML configuration file.
    """

    def __init__(self, yaml_path: str | PathLike[str]) -> None:
        path = Path(yaml_path)
        raw = _read_mapping(path)
        self._validate_and_load(raw, path.resolve().parent)
        self._keep_mapping(raw)
        if self.scenario is not None:
            self.scenario = replace(self.scenario, path=path.resolve())

    @classmethod
    def from_dict(
        cls, raw: dict, base_dir: str | PathLike[str] | None = None
    ) -> "SimConfig":
        """Build a configuration from an already-parsed mapping.

        Applies exactly the validation the file constructor applies. Used
        where a committed configuration needs one field overridden before
        loading, such as the grid in a refinement study.

        Parameters
        ----------
        raw : dict
            Mapping with the same structure as a configuration file.
        base_dir : str or Path, optional
            The directory a scenario's relative ``base`` is read from, as
            the file constructor reads it from the file's own directory.
            None refuses a relative base.

        Returns
        -------
        SimConfig
            Validated configuration.

        Raises
        ------
        ValueError
            If raw is not a mapping or any parameter fails validation.
        """
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a mapping")
        config = cls.__new__(cls)
        config._validate_and_load(
            raw, None if base_dir is None else Path(base_dir).resolve()
        )
        config._keep_mapping(raw)
        return config

    def _keep_mapping(self, raw: dict) -> None:
        """Keep the merged mapping this configuration loaded, read-only.

        A file without a scenario section is kept as read. A scenario is
        kept as its merged base with the scenario section appended as the
        file states it (``base``, ``events`` and ``hold_time``), so the
        mapping names every event and loads again from the scenario file's
        directory to the same configuration: the base's sections merged
        under the merged base are the merged base.
        """
        if self.scenario is None:
            self._mapping = _freeze(raw)
        else:
            self._mapping = _freeze(
                {**_thaw(self.scenario.base), SCENARIO_KEY: _thaw(raw[SCENARIO_KEY])}
            )

    def merged_mapping(self) -> dict:
        """The merged mapping this configuration was built from (ADR-014 D).

        What a run records in its run file's ``/config`` and what a restart
        is checked against (prompt 64's decision 3).

        Returns
        -------
        dict
            A new editable copy with the structure of a configuration file:
            the file as read, or for a scenario the merged base followed by
            the scenario section as the file states it. A copy made and
            changed after loading (``validation.cases.with_velocity_step``)
            keeps the mapping it was copied from.
        """
        return _thaw(self._mapping)

    def _validate_and_load(self, raw: dict, base_dir: Path | None = None) -> None:
        """Validate all parameters and store as typed attributes."""
        # A scenario (ADR-014 B): the merged base loads as a file of its own,
        # and the events are read against it.
        self.scenario: ScenarioSpec | None = None
        if SCENARIO_KEY in raw:
            merged, base_path = self._merged_base(raw, base_dir)
            self._validate_and_load(merged)
            if self.transient is not None or TRANSIENT_KEY in raw:
                raise ValueError(
                    "a scenario runs under tier 1 and a transient section selects "
                    "tier 2 (ADR-014 A); give one, not both"
                )
            self.scenario = self._parse_scenario(raw, merged, base_path)
            return

        # Domain
        domain = self._require_section(raw, "domain")
        if not isinstance(domain, dict):
            raise ValueError("domain must be a mapping")
        self.room_width: float = self._require_positive_float(domain, "width", "domain")
        self.room_height: float = self._require_positive_float(
            domain, "height", "domain"
        )
        self.nx: int = self._require_positive_int(domain, "nx", "domain")
        self.ny: int = self._require_positive_int(domain, "ny", "domain")

        # Mesh stretching. Optional section; absent means uniform on both axes.
        mesh_raw = raw.get("mesh", {})
        if mesh_raw is None:
            mesh_raw = {}
        if not isinstance(mesh_raw, dict):
            raise ValueError("mesh must be a mapping")
        for key in mesh_raw:
            if key not in ("x", "y"):
                raise ValueError(f"mesh.{key} is not a recognised axis; use x or y")
        self.stretch_x: StretchSpec = self._parse_stretch(
            mesh_raw.get("x"), "mesh.x", self.room_width, self.nx
        )
        self.stretch_y: StretchSpec = self._parse_stretch(
            mesh_raw.get("y"), "mesh.y", self.room_height, self.ny
        )

        # Fluid
        fluid = self._require_section(raw, "fluid")
        if not isinstance(fluid, dict):
            raise ValueError("fluid must be a mapping")
        self.rho: float = self._require_positive_float(fluid, "density", "fluid")
        self.mu: float = self._require_positive_float(fluid, "viscosity", "fluid")
        self.temperature: float = self._require_positive_float(
            fluid, "temperature", "fluid"
        )

        # Particles
        particles = self._require_section(raw, "particles")
        if not isinstance(particles, dict):
            raise ValueError("particles must be a mapping")
        self.particle_density: float = self._require_positive_float(
            particles, "density", "particles"
        )
        self.particle_sizes: list[float] = self._require_positive_float_list(
            particles, "sizes", "particles"
        )
        self.mean_free_path: float = self._require_positive_float(
            particles, "mean_free_path", "particles"
        )
        self.boundary_layer_thickness: float = self._require_positive_float(
            particles, "boundary_layer_thickness", "particles"
        )

        # HEPA reference data
        if "hepa_reference" not in particles:
            raise ValueError("Missing required key: 'particles.hepa_reference'")
        hepa_raw = particles["hepa_reference"]
        if not isinstance(hepa_raw, dict):
            raise ValueError("particles.hepa_reference must be a mapping")
        hepa_diameters = self._require_positive_float_list(
            hepa_raw, "diameters", "particles.hepa_reference"
        )
        hepa_efficiencies = self._require_float_list(
            hepa_raw, "efficiencies", "particles.hepa_reference"
        )
        if len(hepa_diameters) != len(hepa_efficiencies):
            raise ValueError(
                "particles.hepa_reference.diameters and efficiencies "
                "must have the same length"
            )
        for i in range(len(hepa_diameters) - 1):
            if hepa_diameters[i] >= hepa_diameters[i + 1]:
                raise ValueError(
                    "particles.hepa_reference.diameters must be sorted "
                    f"ascending, but index {i} ({hepa_diameters[i]}) >= "
                    f"index {i + 1} ({hepa_diameters[i + 1]})"
                )
        for eff in hepa_efficiencies:
            if not 0.0 <= eff <= 1.0:
                raise ValueError(f"HEPA efficiency must be in [0, 1], got {eff}")
        self.hepa_reference: HepaReference = HepaReference(
            diameters=hepa_diameters, efficiencies=hepa_efficiencies
        )

        # Solver
        solver = self._require_section(raw, "solver")
        if not isinstance(solver, dict):
            raise ValueError("solver must be a mapping")
        if RETIRED_PRESSURE_TOL_KEY in solver:
            lower, upper = PRESSURE_RTOL_BOUNDS
            raise ValueError(
                f"solver.{RETIRED_PRESSURE_TOL_KEY} was the weighted Jacobi solve's "
                "stop, pascals of change per sweep, and was retired with it "
                "(ECR-003, 2026-10-06); the conjugate gradient solve stops on the "
                f"relative residual solver.pressure_rtol, in [{lower}, {upper}), with "
                "max_pressure_iter its iteration cap"
            )
        for key in solver:
            if key not in SOLVER_KEYS:
                raise ValueError(
                    f"solver.{key} is not a recognised solver key; "
                    f"known: {sorted(SOLVER_KEYS)}"
                )
        self.dt: float = self._require_positive_float(solver, "dt", "solver")
        self.t_end: float = self._require_positive_float(solver, "t_end", "solver")
        self.output_interval: int = self._require_positive_int(
            solver, "output_interval", "solver"
        )
        self.convergence_tol: float = self._require_positive_float(
            solver, "convergence_tol", "solver"
        )
        self.max_simple_iter: int = self._require_positive_int(
            solver, "max_simple_iter", "solver"
        )
        self.alpha_velocity: float = self._require_relaxation_factor(
            solver, "alpha_velocity", "solver"
        )
        self.alpha_pressure: float = self._require_relaxation_factor(
            solver, "alpha_pressure", "solver"
        )
        self.max_pressure_iter: int = self._require_positive_int(
            solver, "max_pressure_iter", "solver"
        )
        self.pressure_rtol: float = self._require_positive_float(
            solver, "pressure_rtol", "solver"
        )
        lower, upper = PRESSURE_RTOL_BOUNDS
        if not lower <= self.pressure_rtol < upper:
            raise ValueError(
                f"solver.pressure_rtol must be in [{lower}, {upper}), the relative "
                f"residual the pressure correction solves to, got {self.pressure_rtol}"
            )
        self.momentum_sweeps: int = self._optional_positive_int(
            solver, "momentum_sweeps", "solver", DEFAULT_MOMENTUM_SWEEPS
        )
        # Optional: absent keys give the velocity-step rule and its behaviour.
        self.stopping_rule: str = VELOCITY_STEP
        if "stopping_rule" in solver:
            self.stopping_rule = self._require_string(solver, "stopping_rule", "solver")
            if self.stopping_rule not in STOPPING_RULES:
                raise ValueError(
                    f"solver.stopping_rule must be one of {list(STOPPING_RULES)}, "
                    f"got '{self.stopping_rule}'"
                )
        self.iteration_error_tol: float = self._optional_positive_float(
            solver, "iteration_error_tol", "solver", DEFAULT_ITERATION_ERROR_TOL
        )
        self.mass_imbalance_tol: float = self._optional_positive_float(
            solver, "mass_imbalance_tol", "solver", DEFAULT_MASS_IMBALANCE_TOL
        )
        self.pressure_preconditioner: str = DEFAULT_PRESSURE_PRECONDITIONER
        if "pressure_preconditioner" in solver:
            self.pressure_preconditioner = self._require_choice(
                solver, "pressure_preconditioner", "solver", PRESSURE_PRECONDITIONERS
            )
        # The run file's checkpoint cadence (ADR-014 D): outer iterations of a
        # steady solve, steps of the transport march. Absent writes no periodic
        # checkpoint, so every run before step 3 reads as it did.
        self.checkpoint_interval: int | None = None
        if "checkpoint_interval" in solver:
            self.checkpoint_interval = self._require_positive_int(
                solver, "checkpoint_interval", "solver"
            )

        # Transport (optional). Unknown keys raise as they do in the solver
        # block, so a misspelt key cannot fall back to a default.
        self.transport: TransportSpec | None = None
        if "transport" in raw:
            self.transport = self._parse_transport(raw["transport"])

        # Turbulence (optional, ADR-012 I). Absent means the model is off.
        self.turbulence: TurbulenceSpec | None = None
        if "turbulence" in raw:
            self.turbulence = self._parse_turbulence(raw["turbulence"])
            if self.stopping_rule != ERROR_ESTIMATE:
                raise ValueError(
                    "with the turbulence section present solver.stopping_rule must be "
                    f"'{ERROR_ESTIMATE}', got '{self.stopping_rule}': the turbulent "
                    "validation (VAL-018, ECR-002 criterion 10) is scored under rule "
                    "version 5, whose condition (e) refuses to stop while the eddy "
                    "viscosity is still moving, which the velocity step cannot see"
                )

        # Boundaries
        boundaries_raw = self._require_section(raw, "boundaries")
        if not isinstance(boundaries_raw, dict):
            raise ValueError("boundaries must be a mapping")
        self.boundaries: dict[str, BoundarySpec] = {}
        for name, spec in boundaries_raw.items():
            if not isinstance(spec, dict):
                raise ValueError(f"boundaries.{name} must be a mapping")
            ctx = f"boundaries.{name}"
            for key in spec:
                if key not in _SEGMENT_KEYS:
                    raise ValueError(
                        f"{ctx}.{key} is not a recognised boundary key; "
                        f"known: {sorted(_SEGMENT_KEYS)}"
                    )
            bc_type = self._require_string(spec, "type", ctx)
            if bc_type not in _VALID_BOUNDARY_TYPES:
                raise ValueError(
                    f"{ctx}.type must be one of "
                    f"{sorted(_VALID_BOUNDARY_TYPES)}, got '{bc_type}'"
                )
            bc_location = self._require_string(spec, "location", ctx)
            if (
                bc_location not in _VALID_BOUNDARY_LOCATIONS
                and bc_location != OBSTACLE_LOCATION
            ):
                raise ValueError(
                    f"{ctx}.location must be one of "
                    f"{sorted(_VALID_BOUNDARY_LOCATIONS)} or '{OBSTACLE_LOCATION}', "
                    f"got '{bc_location}'"
                )
            bc_obstacle, bc_face, bc_recirculate = self._vent_keys(
                spec, ctx, bc_type, bc_location
            )
            # The axis the extent runs along: x on a top or bottom edge or face.
            along_x = (bc_face or bc_location) in ("top", "bottom")
            bc_velocity = None
            bc_u_velocity = None
            bc_v_velocity = None
            admits_air = False
            if bc_type == "velocity_inlet":
                # Support explicit u/v components or normal-direction magnitude
                raw_u = spec.get("u_velocity")
                raw_v = spec.get("v_velocity")
                raw_vel = spec.get("velocity")

                has_components = raw_u is not None or raw_v is not None
                if not has_components and raw_vel is None:
                    raise ValueError(
                        f"{ctx}: velocity_inlet requires 'velocity' or "
                        f"'u_velocity'/'v_velocity' fields"
                    )

                if raw_vel is not None:
                    bc_velocity = self._finite_number(raw_vel, f"{ctx}.velocity")
                    if bc_velocity <= 0:
                        raise ValueError(
                            f"{ctx}.velocity must be positive, got {raw_vel}"
                        )

                bc_u_velocity = (
                    self._finite_number(raw_u, f"{ctx}.u_velocity")
                    if raw_u is not None
                    else None
                )
                bc_v_velocity = (
                    self._finite_number(raw_v, f"{ctx}.v_velocity")
                    if raw_v is not None
                    else None
                )

                # A zero normal component (a tangential lid) admits no air: the
                # scalar layer treats the segment as a wall, so the inlet
                # concentration keys have nothing to describe.
                normal = bc_v_velocity if along_x else bc_u_velocity
                admits_air = bool(normal) if has_components else True
                if has_components and not normal:
                    for key in ("concentration", "hepa_filtered"):
                        if key in spec:
                            raise ValueError(
                                f"{ctx}.{key} is not valid on a velocity_inlet whose "
                                "normal velocity is zero: no air crosses it and the "
                                "scalar layer treats it as a wall"
                            )

            if bc_type == "pressure_outlet" and "velocity" in spec:
                # velocity means an outward flow on a fixed_flow_outlet, so on a
                # pressure outlet, whose velocity the room finds, it is a slip.
                raise ValueError(
                    f"{ctx}.velocity is not valid on a pressure_outlet; the room "
                    "sets its velocity (use a fixed_flow_outlet to hold one)"
                )
            if bc_type == "fixed_flow_outlet":
                # A fixed flow is a normal velocity and nothing else. The inlet's
                # component and concentration keys describe air entering, and
                # the outlet carries nothing in (ADR-011 E).
                for key in (
                    "u_velocity",
                    "v_velocity",
                    "concentration",
                    "hepa_filtered",
                    "deposition_surface",
                ):
                    if key in spec:
                        raise ValueError(
                            f"{ctx}.{key} is not valid on a fixed_flow_outlet; "
                            "it holds an outward normal 'velocity' only"
                        )
                # "States none" is written by leaving the key out; a blank value
                # (YAML null) is a slip, and reading it as "none" would let the
                # hood quietly take a share of the remainder.
                if "velocity" in spec and spec["velocity"] is None:
                    raise ValueError(
                        f"{ctx}.velocity is blank; give a positive outward "
                        "velocity, or leave the key out to share the remaining flow"
                    )
                if "velocity" in spec:
                    bc_velocity = self._finite_number(
                        spec["velocity"], f"{ctx}.velocity"
                    )
                    if bc_velocity <= 0:
                        raise ValueError(
                            f"{ctx}.velocity must be positive (outward), "
                            f"got {spec['velocity']}"
                        )

            # Coordinate validation based on boundary orientation
            bc_x_start = None
            bc_x_end = None
            bc_y_start = None
            bc_y_end = None
            if bc_location == OBSTACLE_LOCATION:
                # A vent's extent runs along its face; the other axis's keys
                # would be ignored, so they are refused as a slip.
                for key in ("y_start", "y_end") if along_x else ("x_start", "x_end"):
                    if key in spec:
                        raise ValueError(
                            f"{ctx}.{key} is not valid on a vent on the obstacle's "
                            f"{bc_face} face, whose extent is "
                            f"{'x_start, x_end' if along_x else 'y_start, y_end'}"
                        )
            if along_x:
                bc_x_start = self._require_float(spec, "x_start", ctx)
                bc_x_end = self._require_float(spec, "x_end", ctx)
                if bc_x_start >= bc_x_end:
                    raise ValueError(
                        f"{ctx}: x_start ({bc_x_start}) must be less "
                        f"than x_end ({bc_x_end})"
                    )
                if bc_x_start < 0 or bc_x_end > self.room_width:
                    raise ValueError(
                        f"{ctx}: x range [{bc_x_start}, {bc_x_end}] "
                        f"outside domain [0, {self.room_width}]"
                    )
            else:  # left, right
                bc_y_start = self._require_float(spec, "y_start", ctx)
                bc_y_end = self._require_float(spec, "y_end", ctx)
                if bc_y_start >= bc_y_end:
                    raise ValueError(
                        f"{ctx}: y_start ({bc_y_start}) must be less "
                        f"than y_end ({bc_y_end})"
                    )
                if bc_y_start < 0 or bc_y_end > self.room_height:
                    raise ValueError(
                        f"{ctx}: y range [{bc_y_start}, {bc_y_end}] "
                        f"outside domain [0, {self.room_height}]"
                    )

            # Concentration keys (ADR-011 E). Each is allowed on one segment
            # type only, so a key on the wrong type is an error, not ignored.
            bc_concentration = None
            if "concentration" in spec:
                self._require_segment_type(
                    spec, "concentration", ctx, bc_type, "velocity_inlet"
                )
                bc_concentration = tuple(
                    self._require_float_list(spec, "concentration", ctx)
                )
                n_classes = len(self.particle_sizes)
                if len(bc_concentration) != n_classes:
                    raise ValueError(
                        f"{ctx}.concentration must have one value per particle "
                        f"class ({n_classes}), got {len(bc_concentration)}"
                    )
                for i, value in enumerate(bc_concentration):
                    if value < 0:
                        raise ValueError(
                            f"{ctx}.concentration[{i}] must be non-negative, "
                            f"got {value}"
                        )
            bc_hepa_filtered = False
            if "hepa_filtered" in spec:
                self._require_segment_type(
                    spec, "hepa_filtered", ctx, bc_type, "velocity_inlet"
                )
                raw_flag = spec["hepa_filtered"]
                if not isinstance(raw_flag, bool):
                    raise TypeError(
                        f"{ctx}.hepa_filtered must be a bool, "
                        f"got {type(raw_flag).__name__}"
                    )
                bc_hepa_filtered = raw_flag
            bc_deposition_surface = None
            if "deposition_surface" in spec:
                self._require_segment_type(
                    spec, "deposition_surface", ctx, bc_type, "wall"
                )
                bc_deposition_surface = self._require_string(
                    spec, "deposition_surface", ctx
                )
                if bc_deposition_surface not in DEPOSITION_SURFACES:
                    raise ValueError(
                        f"{ctx}.deposition_surface must be one of "
                        f"{list(DEPOSITION_SURFACES)}, got '{bc_deposition_surface}'"
                    )

            bc_intensity, bc_length = self._inlet_turbulence(
                spec, ctx, bc_type, admits_air
            )

            self.boundaries[name] = BoundarySpec(
                type=bc_type,
                location=bc_location,
                x_start=bc_x_start,
                x_end=bc_x_end,
                y_start=bc_y_start,
                y_end=bc_y_end,
                velocity=bc_velocity,
                u_velocity=bc_u_velocity,
                v_velocity=bc_v_velocity,
                concentration=bc_concentration,
                hepa_filtered=bc_hepa_filtered,
                deposition_surface=bc_deposition_surface,
                turbulence_intensity=bc_intensity,
                dissipation_length=bc_length,
                obstacle=bc_obstacle,
                face=bc_face,
                recirculate_from=bc_recirculate,
            )

        self._reject_overlapping_segments()
        self._check_fixed_flow_outlets()

        # Obstacles (optional)
        obstacles_raw = raw.get("obstacles", [])
        if not isinstance(obstacles_raw, list):
            raise ValueError("obstacles must be a list")
        self.obstacles: list[ObstacleSpec] = []
        for i, obs in enumerate(obstacles_raw):
            if not isinstance(obs, dict):
                raise ValueError(f"obstacles[{i}] must be a mapping")
            ctx = f"obstacles[{i}]"
            x0 = self._require_float(obs, "x_start", ctx)
            x1 = self._require_float(obs, "x_end", ctx)
            y0 = self._require_float(obs, "y_start", ctx)
            y1 = self._require_float(obs, "y_end", ctx)
            if x0 >= x1:
                raise ValueError(
                    f"{ctx}: x_start ({x0}) must be less than x_end ({x1})"
                )
            if y0 >= y1:
                raise ValueError(
                    f"{ctx}: y_start ({y0}) must be less than y_end ({y1})"
                )
            if x0 < 0 or x1 > self.room_width:
                raise ValueError(
                    f"{ctx}: x range [{x0}, {x1}] outside domain [0, {self.room_width}]"
                )
            if y0 < 0 or y1 > self.room_height:
                raise ValueError(
                    f"{ctx}: y range [{y0}, {y1}] outside domain "
                    f"[0, {self.room_height}]"
                )
            self.obstacles.append(
                ObstacleSpec(
                    name=self._require_string(obs, "name", ctx),
                    x_start=x0,
                    x_end=x1,
                    y_start=y0,
                    y_end=y1,
                )
            )

        self._check_vents()

        # A tier-2 run (ADR-014 G, step 6b). Laminar, a whole number of steps,
        # and the forces on an obstacle that exists.
        self.transient: TransientSpec | None = None
        if TRANSIENT_KEY in raw:
            self.transient = self._parse_transient(raw[TRANSIENT_KEY])

        # Sensors
        sensors_raw = self._require_section(raw, "sensors")
        if not isinstance(sensors_raw, list):
            raise ValueError("sensors must be a list")
        self.sensors: list[SensorSpec] = []
        for i, sensor in enumerate(sensors_raw):
            if not isinstance(sensor, dict):
                raise ValueError(f"sensors[{i}] must be a mapping")
            ctx = f"sensors[{i}]"
            sx = self._require_float(sensor, "x", ctx)
            sy = self._require_float(sensor, "y", ctx)
            if not 0 <= sx <= self.room_width:
                raise ValueError(f"{ctx}: x={sx} outside domain [0, {self.room_width}]")
            if not 0 <= sy <= self.room_height:
                raise ValueError(
                    f"{ctx}: y={sy} outside domain [0, {self.room_height}]"
                )
            self.sensors.append(
                SensorSpec(
                    name=self._require_string(sensor, "name", ctx),
                    x=sx,
                    y=sy,
                )
            )

        # Thresholds
        thresholds_raw = self._require_section(raw, "thresholds")
        if not isinstance(thresholds_raw, dict):
            raise ValueError("thresholds must be a mapping")
        self.thresholds: dict[str, float] = {}
        for key, val in thresholds_raw.items():
            number = self._finite_number(val, f"thresholds.{key}")
            if number < 0:
                raise ValueError(f"thresholds.{key} must be non-negative, got {val}")
            self.thresholds[str(key)] = number

    def _inlet_turbulence(
        self, spec: dict, ctx: str, bc_type: str, admits_air: bool
    ) -> tuple[float | None, float | None]:
        """The two inlet turbulence keys of one segment, checked (ADR-012 C, I).

        Air entering carries k and eps, so with the turbulence section present
        every velocity inlet that admits air states both. Nothing else may: on
        another segment type, or an inlet whose normal velocity is zero (a
        moving wall), no air enters, and with the model off nothing reads them.

        Returns
        -------
        tuple[float | None, float | None]
            ``turbulence_intensity`` in (0, 1) and ``dissipation_length``,
            positive, in metres; both None where the segment takes neither.
        """
        keys = ("turbulence_intensity", "dissipation_length")
        for key in keys:
            if key not in spec:
                continue
            self._require_segment_type(spec, key, ctx, bc_type, "velocity_inlet")
            if self.turbulence is None:
                raise ValueError(
                    f"{ctx}.{key} is read only by the k-epsilon model, and the "
                    "configuration has no turbulence section"
                )
            if not admits_air:
                raise ValueError(
                    f"{ctx}.{key} is not valid on a velocity_inlet whose normal "
                    "velocity is zero: no air crosses it, so it carries no k or eps"
                )
        if self.turbulence is None or not admits_air:
            return None, None
        for key in keys:
            if key not in spec:
                raise ValueError(
                    f"Missing required key: '{ctx}.{key}'. With the turbulence "
                    "section present every velocity_inlet that admits air states "
                    "turbulence_intensity and dissipation_length, from which the "
                    "model takes the inflow k and eps (ADR-012 C)"
                )
        intensity = self._finite_number(
            spec["turbulence_intensity"], f"{ctx}.turbulence_intensity"
        )
        if not 0.0 < intensity < 1.0:
            raise ValueError(
                f"{ctx}.turbulence_intensity must be a fraction in (0, 1), got "
                f"{intensity}"
            )
        length = self._require_positive_float(spec, "dissipation_length", ctx)
        return intensity, length

    def _vent_keys(
        self, spec: dict, ctx: str, bc_type: str, location: str
    ) -> tuple[str | None, str | None, str | None]:
        """The vent keys of one segment, checked (ADR-014 J).

        On a domain-edge segment none of them may appear. On a vent,
        ``obstacle`` and ``face`` are required, the type is a velocity inlet
        or a fixed-flow outlet, the velocity is stated and normal to the
        face, and ``recirculate_from`` sits on an inlet without a
        ``concentration``. What needs the obstacles is checked in
        ``_check_vents`` once they are read.

        Returns
        -------
        tuple[str | None, str | None, str | None]
            ``obstacle``, ``face`` and ``recirculate_from``; all None on a
            domain-edge segment.
        """
        if location != OBSTACLE_LOCATION:
            for key in _VENT_KEYS:
                if key in spec:
                    raise ValueError(
                        f"{ctx}.{key} is valid only on a vent (location "
                        f"'{OBSTACLE_LOCATION}'), not on the {location} edge"
                    )
            return None, None, None
        if bc_type not in VENT_TYPES:
            raise ValueError(
                f"{ctx}.type must be one of {list(VENT_TYPES)} on an obstacle face, "
                f"got '{bc_type}': a pressure outlet's zero-gradient copy needs the "
                "face beyond it, which is inside the obstacle, and an obstacle face "
                "is a wall already"
            )
        obstacle = self._require_string(spec, "obstacle", ctx)
        face = self._require_choice(spec, "face", ctx, OBSTACLE_FACES)
        # A vent is a normal jet or a normal draw. The obstacle layers read the
        # tangential velocity on an obstacle face as zero (the momentum wall
        # stencil, the k-epsilon corner gradients, the QUICK boundary form), so
        # an angled jet's tangential component would be dropped there.
        for key in ("u_velocity", "v_velocity"):
            if key in spec:
                raise ValueError(
                    f"{ctx}.{key} is not valid on a vent: its velocity is normal to "
                    "the obstacle face, given as 'velocity'"
                )
        if "velocity" not in spec:
            raise ValueError(
                f"Missing required key: '{ctx}.velocity'. A vent states its flow: "
                "a discharge's or an exhaust's velocity is the tool's, not a share "
                "of the room's remainder"
            )
        recirculate = None
        if "recirculate_from" in spec:
            self._require_segment_type(
                spec, "recirculate_from", ctx, bc_type, "velocity_inlet"
            )
            recirculate = self._require_string(spec, "recirculate_from", ctx)
            if "concentration" in spec:
                raise ValueError(
                    f"{ctx}.concentration is not valid with recirculate_from: the "
                    "discharge carries what its intake draws"
                )
        return obstacle, face, recirculate

    def _check_vents(self) -> None:
        """Raise unless every vent sits on a free face of a named obstacle (ADR-014 J).

        The obstacle exists and is named once; the face is not on a domain
        edge (an obstacle standing on the floor has no bottom face to vent
        from); the extent lies within the face; no other obstacle stands
        against the face along the extent; a ``recirculate_from`` names a
        fixed-flow vent on the same obstacle. Overlapping vents on one face
        are refused with the edge segments (``_reject_overlapping_segments``).
        What needs the mesh, that the extent covers a face of it, is checked
        when the boundary is built.
        """
        by_name: dict[str, list[ObstacleSpec]] = {}
        for obs in self.obstacles:
            by_name.setdefault(obs.name, []).append(obs)
        for name, spec in self.boundaries.items():
            if spec.location != OBSTACLE_LOCATION:
                continue
            ctx = f"boundaries.{name}"
            found = by_name.get(spec.obstacle or "", [])
            if len(found) != 1:
                raise ValueError(
                    f"{ctx}.obstacle names '{spec.obstacle}', which "
                    + (
                        "no obstacle in the obstacles list has"
                        if not found
                        else f"{len(found)} obstacles share"
                    )
                )
            obs = found[0]
            lo, hi = self._segment_range(spec)
            face_lo, face_hi = self._face_span(obs, spec.face)
            if lo < face_lo or hi > face_hi:
                raise ValueError(
                    f"{ctx}: the extent [{lo}, {hi}] does not lie within the "
                    f"{spec.face} face of '{obs.name}', [{face_lo}, {face_hi}]"
                )
            if self._face_on_edge(obs, spec.face):
                raise ValueError(
                    f"{ctx}: the {spec.face} face of '{obs.name}' lies on the domain "
                    "edge, so no air reaches it"
                )
            for other in self.obstacles:
                if other is not obs and self._stands_against(obs, spec, other):
                    raise ValueError(
                        f"{ctx}: obstacle '{other.name}' stands against the "
                        f"{spec.face} face of '{obs.name}' along the vent's extent "
                        f"[{lo}, {hi}], so no air reaches it"
                    )
            if spec.recirculate_from is not None:
                self._check_recirculation(name, spec)

    def _check_recirculation(self, name: str, spec: BoundarySpec) -> None:
        """Raise unless recirculate_from names its own tool's balanced exhaust.

        The named segment is a fixed-flow vent on the discharge's obstacle,
        and its configured flow is the discharge's.
        """
        ctx = f"boundaries.{name}.recirculate_from"
        source = self.boundaries.get(spec.recirculate_from or "")
        if source is None:
            raise ValueError(
                f"{ctx} names '{spec.recirculate_from}', which is not a boundary "
                "segment"
            )
        if source.type != "fixed_flow_outlet" or source.location != OBSTACLE_LOCATION:
            raise ValueError(
                f"{ctx} names '{spec.recirculate_from}', which is not a "
                "fixed_flow_outlet vent: a discharge returns what a vent on its own "
                "tool draws in"
            )
        if source.obstacle != spec.obstacle:
            raise ValueError(
                f"{ctx} names '{spec.recirculate_from}', a vent on "
                f"'{source.obstacle}', not on '{spec.obstacle}': a discharge returns "
                "what its own tool draws in"
            )
        # A discharge of more flow than its intake would carry the intake's
        # concentration at the larger flow and create particles (review 61 S6).
        lo, hi = self._segment_range(spec)
        source_lo, source_hi = self._segment_range(source)
        returned = (spec.velocity or 0.0) * (hi - lo)
        drawn = (source.velocity or 0.0) * (source_hi - source_lo)
        if abs(returned - drawn) > BALANCE_TOLERANCE * max(returned, drawn):
            raise ValueError(
                f"{ctx}: the discharge's flow {returned:.6g} m^2/s differs from "
                f"its intake '{spec.recirculate_from}''s {drawn:.6g} m^2/s; a fan "
                "returns what it draws, so state equal flows"
            )

    @staticmethod
    def _face_span(obs: ObstacleSpec, face: str | None) -> tuple[float, float]:
        """An obstacle face's span along it: x on the top or bottom, y on a side."""
        if face in ("top", "bottom"):
            return obs.x_start, obs.x_end
        return obs.y_start, obs.y_end

    def _face_on_edge(self, obs: ObstacleSpec, face: str | None) -> bool:
        """Whether an obstacle face lies on the domain edge it faces."""
        if face == "top":
            return obs.y_end >= self.room_height
        if face == "bottom":
            return obs.y_start <= 0.0
        if face == "left":
            return obs.x_start <= 0.0
        return obs.x_end >= self.room_width

    @classmethod
    def _stands_against(
        cls, obs: ObstacleSpec, spec: BoundarySpec, other: ObstacleSpec
    ) -> bool:
        """Whether ``other`` fills the space just outside the vent's face on its extent.

        It does when it reaches across the face's plane on the room's side
        and its span along the face meets the vent's extent in more than a
        point.
        """
        lo, hi = cls._segment_range(spec)
        if spec.face == "top":
            across = other.y_start <= obs.y_end < other.y_end
        elif spec.face == "bottom":
            across = other.y_start < obs.y_start <= other.y_end
        elif spec.face == "left":
            across = other.x_start < obs.x_start <= other.x_end
        else:
            across = other.x_start <= obs.x_end < other.x_end
        other_lo, other_hi = cls._face_span(other, spec.face)
        return across and other_lo < hi and lo < other_hi

    def _reject_overlapping_segments(self) -> None:
        """Raise if two segments on one edge, or on one obstacle face, overlap.

        Where two ranges intersect in more than a point the file describes
        two conditions for one face; the registry would give the face to
        the first segment and a reader counting by segment would count it
        twice. Ranges that meet at one coordinate are allowed, and the
        first segment in configuration order decides there.
        """
        names = list(self.boundaries)
        for i, a_name in enumerate(names):
            a = self.boundaries[a_name]
            a0, a1 = self._segment_range(a)
            for b_name in names[i + 1 :]:
                b = self.boundaries[b_name]
                if (b.location, b.obstacle, b.face) != (a.location, a.obstacle, a.face):
                    continue
                b0, b1 = self._segment_range(b)
                if a0 < b1 and b0 < a1:
                    where = (
                        f"the {a.face} face of '{a.obstacle}'"
                        if a.location == OBSTACLE_LOCATION
                        else f"the {a.location} edge"
                    )
                    raise ValueError(
                        f"boundaries.{a_name} and boundaries.{b_name} overlap on "
                        f"{where} ([{a0}, {a1}] and [{b0}, {b1}]); "
                        "segments on one edge or face must not overlap"
                    )

    def _check_fixed_flow_outlets(self) -> None:
        """Raise unless the fixed-flow outlets leave the flow balance determined.

        Without a pressure outlet the outflow must equal the inflow, so
        at least one fixed-flow outlet has to state no velocity and take
        the remainder; if every one stated a velocity the configuration
        would over-determine the balance. With a pressure outlet, that
        outlet balances the flow and every fixed-flow outlet must state
        its velocity. What needs the mesh (a remainder that is not
        positive) is checked when the boundary is built. A fixed-flow
        outlet also needs a velocity inlet: the room's velocity scale comes
        from what drives it in, not from what leaves.
        """
        fixed = {
            name: spec
            for name, spec in self.boundaries.items()
            if spec.type == "fixed_flow_outlet"
        }
        if not fixed:
            return
        if not any(spec.type == "velocity_inlet" for spec in self.boundaries.values()):
            raise ValueError(
                f"fixed_flow_outlet segments {sorted(fixed)} need a velocity_inlet "
                "to drive the room. Without one, make-up air would enter through a "
                "pressure outlet, the regime the outlet's zero-gradient copy "
                "handles badly (GitHub issue 61), and this code does not support it"
            )
        unstated = [name for name, spec in fixed.items() if spec.velocity is None]
        has_pressure_outlet = any(
            spec.type == "pressure_outlet" for spec in self.boundaries.values()
        )
        if has_pressure_outlet and unstated:
            raise ValueError(
                f"boundaries {unstated} are fixed_flow_outlet segments with no "
                "velocity, but the configuration has a pressure_outlet that "
                "balances the flow; give each fixed_flow_outlet a velocity"
            )
        if not has_pressure_outlet and not unstated:
            # With every outlet stated, the stated flows must balance the inflow
            # themselves: a closed box whose only openings are a discharge and
            # an exhaust of equal flow (ADR-014 J's test). On the mesh the
            # boundary checks the same balance on its faces.
            inflow = sum(
                self._configured_inflow(spec)
                for spec in self.boundaries.values()
                if spec.type == "velocity_inlet"
            )
            outflow = sum(
                spec.velocity * (hi - lo)
                for spec in fixed.values()
                for lo, hi in (self._segment_range(spec),)
            )
            if abs(inflow - outflow) > BALANCE_TOLERANCE * max(abs(inflow), outflow):
                raise ValueError(
                    f"every fixed_flow_outlet {sorted(fixed)} states a velocity and "
                    "the configuration has no pressure_outlet, so nothing balances "
                    f"the flow: the stated outflow {outflow:.6g} m^2/s differs from "
                    f"the inflow {inflow:.6g} m^2/s; leave the velocity off at least "
                    "one domain-edge outlet, or state flows that balance"
                )

    @staticmethod
    def _segment_range(spec: BoundarySpec) -> tuple[float, float]:
        """The segment's range along its edge or obstacle face."""
        if (spec.face or spec.location) in ("top", "bottom"):
            return spec.x_start, spec.x_end
        return spec.y_start, spec.y_end

    @classmethod
    def _configured_inflow(cls, spec: BoundarySpec) -> float:
        """A velocity inlet's inflow per metre of depth on the configured extent.

        The normal component signed into the room times the extent's
        length: ``velocity`` is inward by definition; explicit components
        take the sign the edge gives them (a lid's normal component is zero).
        """
        lo, hi = cls._segment_range(spec)
        if spec.u_velocity is None and spec.v_velocity is None:
            return (spec.velocity or 0.0) * (hi - lo)
        u = spec.u_velocity or 0.0
        v = spec.v_velocity or 0.0
        inward = {"bottom": v, "top": -v, "left": u, "right": -u}[spec.location]
        return inward * (hi - lo)

    # -- Scenarios (ADR-014 B) -----------------------------------------------

    @staticmethod
    def _merged_base(raw: dict, base_dir: Path | None) -> tuple[dict, Path]:
        """The base file's mapping with the scenario file's sections merged over it.

        Returns
        -------
        tuple[dict, Path]
            The merged mapping, with no scenario section, and the base's
            resolved path.
        """
        section = raw[SCENARIO_KEY]
        if not isinstance(section, dict):
            raise ValueError(f"{SCENARIO_KEY} must be a mapping")
        for key in section:
            if key not in _SCENARIO_KEYS:
                raise ValueError(
                    f"{SCENARIO_KEY}.{key} is not a recognised scenario key; "
                    f"known: {sorted(_SCENARIO_KEYS)}"
                )
        name = SimConfig._require_string(section, "base", SCENARIO_KEY)
        path = Path(name)
        if not path.is_absolute():
            if base_dir is None:
                raise ValueError(
                    f"{SCENARIO_KEY}.base '{name}' is relative and the mapping has "
                    "no file to be relative to; pass base_dir to from_dict"
                )
            path = base_dir / path
        base = _read_mapping(path)
        if SCENARIO_KEY in base:
            raise ValueError(
                f"{SCENARIO_KEY}.base '{name}' is itself a scenario; a scenario "
                "extends a configuration, not another scenario"
            )
        own = {k: v for k, v in raw.items() if k != SCENARIO_KEY}
        for section_name, key in SCENARIO_RUN_KEYS:
            block = own.get(section_name)
            if not isinstance(block, dict) or key not in block:
                raise ValueError(
                    f"Missing required key: '{section_name}.{key}' in the scenario "
                    "file itself. A scenario states its own run length, output "
                    "interval and Courant number rather than inheriting the base's "
                    "(ADR-014 B)"
                )
        return merge_mappings(base, own), path.resolve()

    def _parse_scenario(self, raw: dict, merged: dict, base_path: Path) -> ScenarioSpec:
        """The events, validated against the merged base, and every interval's file.

        The merged base has loaded already, so this configuration's
        attributes are the base's. Each distinct set of active boundary
        events is applied to the merged base and the result validated as a
        file of its own, so a replacement that overlaps a neighbour or
        omits a required key fails here, before anything runs (REQ-C02).
        """
        section = raw[SCENARIO_KEY]
        if "events" not in section:
            raise ValueError(f"Missing required key: '{SCENARIO_KEY}.events'")
        events_raw = section["events"]
        if not isinstance(events_raw, list):
            raise TypeError(
                f"{SCENARIO_KEY}.events must be a list, got {type(events_raw).__name__}"
            )
        segments = list(merged["boundaries"])
        events: list[EventSpec] = []
        for i, event_raw in enumerate(events_raw):
            events.append(
                self._parse_event(event_raw, f"{SCENARIO_KEY}.events[{i}]", segments)
            )
        names = [event.name for event in events]
        for name in names:
            if names.count(name) > 1:
                raise ValueError(
                    f"{SCENARIO_KEY}.events: two events are named '{name}'; an "
                    "event's name identifies it in the run's records"
                )
        initials = [e.name for e in events if isinstance(e.change, Initial)]
        if len(initials) > 1:
            raise ValueError(
                f"{SCENARIO_KEY}.events: {initials} each set the initial field; a "
                "run has one"
            )
        for i, a in enumerate(events):
            for b in events[i + 1 :]:
                if a.segment is None or a.segment != b.segment:
                    continue
                overlap = min(a.end, b.end) - max(a.start, b.start)
                if overlap > EVENT_TIME_RTOL * self.t_end:
                    raise ValueError(
                        f"{SCENARIO_KEY}.events: '{a.name}' [{a.start}, {a.end}) and "
                        f"'{b.name}' [{b.start}, {b.end}) both change segment "
                        f"'{a.segment}' and overlap in time"
                    )
        hold_time = None
        if "hold_time" in section:
            hold_time = self._require_positive_float(section, "hold_time", SCENARIO_KEY)
            if not initials:
                raise ValueError(
                    f"{SCENARIO_KEY}.hold_time needs an initial event: the hold ends a "
                    "clean-out once the room is within spec, and a room that starts "
                    "clean is within spec from t = 0 (ADR-014 C)"
                )
            if hold_time >= self.t_end:
                raise ValueError(
                    f"{SCENARIO_KEY}.hold_time ({hold_time} s) must be below "
                    f"solver.t_end ({self.t_end} s)"
                )
            # Within spec is every class below its threshold at every sensor,
            # and an initial field is uniform, so a start below the thresholds
            # in every class is within spec at t = 0 and the hold would end
            # the run at hold_time, before any later event (review 64 S5).
            start = next(e for e in events if isinstance(e.change, Initial))
            assert isinstance(start.change, Initial)
            thresholds = self.class_thresholds(f"{SCENARIO_KEY}.hold_time")
            if all(c < t for c, t in zip(start.change.values, thresholds, strict=True)):
                raise ValueError(
                    f"{SCENARIO_KEY}.hold_time needs an initial event at or above "
                    f"the thresholds in some class: '{start.name}' starts every class "
                    "below its threshold, so the room is within spec from t = 0 and "
                    f"the hold would end the run at {hold_time} s (ADR-014 C)"
                )
        # A turbulent flow state carries an eddy viscosity, and the transport
        # step refuses one without a Schmidt number. A scenario is marched, so
        # the refusal belongs here rather than after the flow solves (review
        # 64 S3).
        assert self.transport is not None
        if self.turbulence is not None and self.transport.turbulent_schmidt is None:
            raise ValueError(
                "a scenario on a configuration with the turbulence section needs "
                "transport.turbulent_schmidt: the march reads the flow state's eddy "
                "viscosity, which the transport step turns into a diffusivity by it"
            )
        spec = ScenarioSpec(
            base_path=base_path,
            base=_freeze(merged),
            events=tuple(events),
            t_end=self.t_end,
            hold_time=hold_time,
        )
        self._check_intervals(spec)
        return spec

    def _parse_event(self, event: object, ctx: str, segments: list[str]) -> EventSpec:
        """One event: its name, its times and its one change."""
        if not isinstance(event, dict):
            raise ValueError(f"{ctx} must be a mapping")
        for key in event:
            if key not in _EVENT_KEYS:
                raise ValueError(
                    f"{ctx}.{key} is not a recognised event key; known: "
                    f"{sorted(_EVENT_KEYS)}"
                )
        name = self._require_string(event, "name", ctx)
        if not name.strip():
            raise ValueError(f"{ctx}.name must not be empty")
        ctx = f"{ctx} ('{name}')"
        changes = [key for key in EVENT_CHANGES if key in event]
        if len(changes) != 1:
            raise ValueError(
                f"{ctx} must state exactly one of {list(EVENT_CHANGES)}, got "
                f"{changes or 'none'}"
            )
        kind = changes[0]
        start = self._require_float(event, "start", ctx)
        if start < 0:
            raise ValueError(f"{ctx}.start must be non-negative, got {start}")
        duration = None
        if kind == INITIAL:
            if "duration" in event:
                raise ValueError(
                    f"{ctx}.duration is not valid on an initial event: it sets the "
                    "field at the start of the run"
                )
            if start != 0.0:
                raise ValueError(
                    f"{ctx}.start must be 0 on an initial event, got {start}"
                )
        else:
            duration = self._require_positive_float(event, "duration", ctx)
            if start + duration - self.t_end > EVENT_TIME_RTOL * self.t_end:
                raise ValueError(
                    f"{ctx} ends at {start + duration} s, after solver.t_end "
                    f"({self.t_end} s)"
                )
        body = event[kind]
        if not isinstance(body, dict):
            raise ValueError(f"{ctx}.{kind} must be a mapping")
        body_ctx = f"{ctx}.{kind}"
        change: Replace | Set | Source | Initial
        if kind == REPLACE:
            change = self._parse_replace(body, body_ctx, segments)
        elif kind == SET:
            change = self._parse_set(body, body_ctx, segments)
        elif kind == SOURCE:
            change = self._parse_source(body, body_ctx)
        else:
            change = self._parse_initial(body, body_ctx)
        return EventSpec(name=name, start=start, duration=duration, change=change)

    @classmethod
    def _event_keys(cls, body: dict, ctx: str, known: frozenset[str]) -> None:
        """Raise on a key of an event's change that the change does not take."""
        for key in body:
            if key not in known:
                raise ValueError(
                    f"{ctx}.{key} is not a recognised key; known: {sorted(known)}"
                )

    @classmethod
    def _target_segment(cls, body: dict, ctx: str, segments: list[str]) -> str:
        """The base segment a replace or a set names."""
        segment = cls._require_string(body, "segment", ctx)
        if segment not in segments:
            raise ValueError(
                f"{ctx}.segment names '{segment}', which is not a segment of the "
                f"base; known: {segments}"
            )
        return segment

    def _parse_replace(self, body: dict, ctx: str, segments: list[str]) -> Replace:
        """A replace: the base segment and its replacements, possibly none."""
        self._event_keys(body, ctx, _REPLACE_KEYS)
        segment = self._target_segment(body, ctx, segments)
        if "with" not in body:
            raise ValueError(f"Missing required key: '{ctx}.with'")
        replacements = body["with"]
        if not isinstance(replacements, list):
            raise TypeError(
                f"{ctx}.with must be a list of segments (empty removes the segment), "
                f"got {type(replacements).__name__}"
            )
        out: list[tuple[str, Mapping[str, object]]] = []
        for j, item in enumerate(replacements):
            item_ctx = f"{ctx}.with[{j}]"
            if not isinstance(item, dict):
                raise ValueError(f"{item_ctx} must be a mapping")
            name = self._require_string(item, "name", item_ctx)
            if not name.strip():
                raise ValueError(f"{item_ctx}.name must not be empty")
            if name in segments and name != segment:
                raise ValueError(
                    f"{item_ctx}.name '{name}' is the name of another segment of the "
                    "base"
                )
            if name in (n for n, _ in out):
                raise ValueError(f"{item_ctx}.name '{name}' is given twice")
            keys = {k: v for k, v in item.items() if k != "name"}
            out.append((name, _freeze(keys)))
        return Replace(segment=segment, segments=tuple(out))

    def _parse_set(self, body: dict, ctx: str, segments: list[str]) -> Set:
        """A set: the base segment and the keys it changes."""
        self._event_keys(body, ctx, _SET_KEYS)
        segment = self._target_segment(body, ctx, segments)
        if "keys" not in body:
            raise ValueError(f"Missing required key: '{ctx}.keys'")
        keys = body["keys"]
        if not isinstance(keys, dict) or not keys:
            raise ValueError(f"{ctx}.keys must be a non-empty mapping of segment keys")
        for key in keys:
            if key not in _SEGMENT_KEYS:
                raise ValueError(
                    f"{ctx}.keys.{key} is not a recognised boundary key; known: "
                    f"{sorted(_SEGMENT_KEYS)}"
                )
        return Set(segment=segment, keys=_freeze(keys))

    def _parse_source(self, body: dict, ctx: str) -> Source:
        """A source: the box and its rate or emission per class."""
        self._event_keys(body, ctx, _SOURCE_KEYS)
        x0, x1, y0, y1 = (self._require_float(body, k, ctx) for k in _SOURCE_BOX_KEYS)
        if x0 >= x1 or y0 >= y1:
            raise ValueError(
                f"{ctx}: the box [{x0}, {x1}] x [{y0}, {y1}] is empty; x_start and "
                "y_start must be less than x_end and y_end"
            )
        if x0 < 0 or x1 > self.room_width or y0 < 0 or y1 > self.room_height:
            raise ValueError(
                f"{ctx}: the box [{x0}, {x1}] x [{y0}, {y1}] is outside the domain "
                f"[0, {self.room_width}] x [0, {self.room_height}]"
            )
        stated = [key for key in ("rate", "emission") if key in body]
        if len(stated) != 1:
            raise ValueError(
                f"{ctx} must state exactly one of rate (per cubic metre per second) "
                f"and emission (per second per metre of depth), got {stated or 'none'}"
            )
        values = self._class_values(body, stated[0], ctx)
        return Source(
            x_start=x0,
            x_end=x1,
            y_start=y0,
            y_end=y1,
            **{stated[0]: values},
        )

    def _parse_initial(self, body: dict, ctx: str) -> Initial:
        """An initial: a concentration per class, or the thresholds times a factor."""
        self._event_keys(body, ctx, _INITIAL_KEYS)
        stated = [key for key in ("concentration", "threshold_factor") if key in body]
        if len(stated) != 1:
            raise ValueError(
                f"{ctx} must state exactly one of concentration and threshold_factor, "
                f"got {stated or 'none'}"
            )
        if stated[0] == "concentration":
            return Initial(values=self._class_values(body, "concentration", ctx))
        factor = self._require_positive_float(body, "threshold_factor", ctx)
        return Initial(
            values=tuple(t * factor for t in self.class_thresholds(ctx)),
            threshold_factor=factor,
        )

    def class_thresholds(self, context: str = "thresholds") -> tuple[float, ...]:
        """Each class's threshold, the ``thresholds`` key whose diameter is its size.

        Parameters
        ----------
        context : str
            Names the caller in the error.

        Returns
        -------
        tuple[float, ...]
            One per configured particle class, particles per cubic metre.

        Raises
        ------
        ValueError
            If a class has no threshold key of its diameter.
        """
        by_diameter: list[tuple[float, float]] = []
        for key, value in self.thresholds.items():
            try:
                by_diameter.append((float(key), value))
            except ValueError:
                continue
        out = []
        for size in self.particle_sizes:
            found = [
                value
                for diameter, value in by_diameter
                if math.isclose(diameter, size, rel_tol=_DIAMETER_MATCH_RTOL)
            ]
            if len(found) != 1:
                raise ValueError(
                    f"{context}: the class of diameter {size} m has "
                    f"{'no' if not found else len(found)} threshold keys of that "
                    f"diameter; thresholds has {sorted(self.thresholds)}"
                )
            out.append(found[0])
        return tuple(out)

    def _class_values(self, section: dict, key: str, ctx: str) -> tuple[float, ...]:
        """One finite non-negative number per configured particle class."""
        values = self._require_float_list(section, key, ctx)
        n_classes = len(self.particle_sizes)
        if len(values) != n_classes:
            raise ValueError(
                f"{ctx}.{key} must have one value per particle class ({n_classes}), "
                f"got {len(values)}"
            )
        for i, value in enumerate(values):
            if value < 0:
                raise ValueError(f"{ctx}.{key}[{i}] must be non-negative, got {value}")
        return tuple(values)

    @staticmethod
    def _check_intervals(spec: ScenarioSpec) -> None:
        """Raise unless every interval's configuration loads (ADR-014 B).

        Each distinct set of active events is applied to the merged base and
        validated as a file. A recirculating pair of the base also stays a
        pair: an interval in which its intake still draws while no discharge
        returns what it draws is refused, since a fan's two vents stop
        together (the fan stop of prompt 62's fixtures).
        """
        base_boundaries = spec.base["boundaries"]
        intakes = {
            seg["recirculate_from"]: name
            for name, seg in base_boundaries.items()
            if "recirculate_from" in seg
        }
        times = spec.times()
        checked: set[tuple[str, ...]] = set()
        for t0, t1 in pairwise(times):
            active = spec.active(t0)
            names = tuple(e.name for e in active)
            if names in checked:
                continue
            checked.add(names)
            where = (
                f"scenario: the configuration from {t0} s to {t1} s with "
                f"{list(names) if names else 'no events'} active"
            )
            try:
                mapping = spec.mapping(active)
                SimConfig.from_dict(mapping)
            except (TypeError, ValueError) as err:
                raise type(err)(f"{where} does not load: {err}") from err
            boundaries = mapping["boundaries"]
            returned = {seg.get("recirculate_from") for seg in boundaries.values()}
            for intake, discharge in intakes.items():
                if intake in boundaries and intake not in returned:
                    raise ValueError(
                        f"{where}: the intake '{intake}' still draws while no "
                        f"discharge returns it (the base's '{discharge}' does); a "
                        "fan's intake and discharge stop together"
                    )

    # -- Validation helpers --------------------------------------------------

    @classmethod
    def _parse_stretch(
        cls, section: dict | None, context: str, length: float, n: int
    ) -> StretchSpec:
        """Parse one axis of the mesh section into a StretchSpec.

        Accepts ``stretch_ratio`` (>= 1) or ``min_wall_spacing`` (in
        (0, length / n]) or ``breakpoints`` (``_parse_breakpoints``), one of
        the three. An absent or empty axis is uniform.
        """
        if section is None:
            return StretchSpec()
        if not isinstance(section, dict):
            raise ValueError(f"{context} must be a mapping")
        for key in section:
            if key not in ("stretch_ratio", "min_wall_spacing", "breakpoints"):
                raise ValueError(
                    f"{context}.{key} is not recognised; use stretch_ratio, "
                    "min_wall_spacing or breakpoints"
                )
        if "breakpoints" in section:
            if len(section) > 1:
                raise ValueError(
                    f"{context}: breakpoints cluster the axis on their own; give "
                    "neither stretch_ratio nor min_wall_spacing with them"
                )
            return cls._parse_breakpoints(
                section["breakpoints"], f"{context}.breakpoints", length, n
            )
        has_ratio = "stretch_ratio" in section
        has_spacing = "min_wall_spacing" in section
        if has_ratio and has_spacing:
            raise ValueError(
                f"{context}: give stretch_ratio or min_wall_spacing, not both; the "
                "cell count fixes the other"
            )
        if has_spacing:
            spacing = cls._require_positive_float(section, "min_wall_spacing", context)
            uniform = length / n
            if spacing > uniform:
                raise ValueError(
                    f"{context}.min_wall_spacing must not exceed the uniform spacing "
                    f"{uniform} for {n} cells on {length}, got {spacing}"
                )
            return StretchSpec(ratio=1.0, min_spacing=spacing)
        if has_ratio:
            ratio = cls._require_positive_float(section, "stretch_ratio", context)
            if ratio < 1.0:
                raise ValueError(
                    f"{context}.stretch_ratio must be >= 1 (1 is uniform), got {ratio}"
                )
            return StretchSpec(ratio=ratio, min_spacing=None)
        return StretchSpec()

    def _parse_transient(self, section: object) -> TransientSpec:
        """Parse the transient section (ADR-014 G, step 6b).

        ``scheme`` one of TRANSIENT_SCHEMES (default "bdf2"), ``step_tol`` a
        positive number, ``step_cap`` a positive int, ``forces`` optional
        with ``obstacle`` (a configured obstacle's name) and
        ``reference_velocity`` (positive). Refused with the turbulence
        section (tier 2 is laminar, ADR-014 decision 1) and unless
        solver.t_end is a whole number of steps of solver.dt to a relative
        STEP_COUNT_ROUNDING. Unknown keys raise.
        """
        context = TRANSIENT_KEY
        if not isinstance(section, dict):
            raise ValueError(f"{context} must be a mapping")
        for key in section:
            if key not in _TRANSIENT_KEYS:
                raise ValueError(
                    f"{context}.{key} is not recognised; known: {sorted(_TRANSIENT_KEYS)}"
                )
        if self.turbulence is not None:
            raise ValueError(
                f"{context}: a transient run is laminar (ADR-014 decision 1) and the "
                "turbulence section is present"
            )
        scheme = section.get("scheme", BDF2)
        if not isinstance(scheme, str) or scheme not in TRANSIENT_SCHEMES:
            raise ValueError(
                f"{context}.scheme must be one of {list(TRANSIENT_SCHEMES)}, got {scheme!r}"
            )
        step_tol = self._require_positive_float(section, "step_tol", context)
        step_cap = self._require_positive_int(section, "step_cap", context)
        steps = round(self.t_end / self.dt)
        if (
            steps < 1
            or abs(steps * self.dt - self.t_end) > STEP_COUNT_ROUNDING * self.t_end
        ):
            raise ValueError(
                f"{context}: solver.t_end {self.t_end} is not a whole number of steps "
                f"of solver.dt {self.dt}"
            )
        forces = None
        if "forces" in section:
            given = section["forces"]
            where = f"{context}.forces"
            if not isinstance(given, dict):
                raise ValueError(f"{where} must be a mapping")
            if set(given) != _FORCE_KEYS:
                raise ValueError(
                    f"{where} must have exactly the keys {sorted(_FORCE_KEYS)}, got "
                    f"{sorted(given)}"
                )
            name = self._require_string(given, "obstacle", where)
            names = [o.name for o in self.obstacles]
            if name not in names:
                raise ValueError(
                    f"{where}.obstacle {name!r} is not an obstacle of this "
                    f"configuration; it has {names}"
                )
            forces = ForceSpec(
                obstacle=name,
                reference_velocity=self._require_positive_float(
                    given, "reference_velocity", where
                ),
            )
        return TransientSpec(
            scheme=scheme, step_tol=step_tol, step_cap=step_cap, forces=forces
        )

    @classmethod
    def _parse_breakpoints(
        cls, entries: object, context: str, length: float, n: int
    ) -> StretchSpec:
        """Parse an axis's interior clustering and derive its faces (ADR-014 decision 11).

        A list of at least two mappings with keys ``at`` and ``spacing``,
        real numbers (a bool is refused): the first at 0 and the last at the
        axis length, strictly increasing, every spacing positive and below
        the length of each segment beside it. The faces must give exactly
        the domain's cell count, and no two adjacent widths may differ by a
        ratio above MAX_GROWTH_RATIO; either refusal names what the
        breakpoints give, so the file can be corrected.
        """
        if not isinstance(entries, list) or len(entries) < 2:
            raise ValueError(f"{context} must be a list of at least two breakpoints")
        points = []
        for index, entry in enumerate(entries):
            where = f"{context}[{index}]"
            if not isinstance(entry, dict):
                raise ValueError(f"{where} must be a mapping with at and spacing")
            if set(entry) != {"at", "spacing"}:
                raise ValueError(
                    f"{where} must have exactly the keys at and spacing, got "
                    f"{sorted(entry)}"
                )
            at = cls._finite_number(entry["at"], f"{where}.at")
            spacing = cls._require_positive_float(entry, "spacing", where)
            points.append(Breakpoint(at=at, spacing=spacing))
        if points[0].at != 0.0 or points[-1].at != length:
            raise ValueError(
                f"{context} must run from 0 to the axis length {length}: the first "
                f"is at {points[0].at} and the last at {points[-1].at}"
            )
        for left, right in pairwise(points):
            if right.at <= left.at:
                raise ValueError(
                    f"{context} must be strictly increasing: {right.at} follows "
                    f"{left.at}"
                )
            segment = right.at - left.at
            if max(left.spacing, right.spacing) >= segment:
                raise ValueError(
                    f"{context}: the spacing at {left.at} or {right.at} is not below "
                    f"the {segment} between them"
                )
        faces = breakpoint_faces(points)
        if len(faces) - 1 != n:
            raise ValueError(
                f"{context} give {len(faces) - 1} cells and the domain states {n}; "
                f"set the domain's count to {len(faces) - 1}"
            )
        widths = [b - a for a, b in pairwise(faces)]
        worst = max(max(a / b, b / a) for a, b in pairwise(widths)) if n > 1 else 1.0
        if worst > MAX_GROWTH_RATIO:
            raise ValueError(
                f"{context}: two adjacent cells differ by a ratio of {worst:.4f}, "
                f"above MAX_GROWTH_RATIO {MAX_GROWTH_RATIO}; move the breakpoints "
                "apart or bring their spacings closer"
            )
        return StretchSpec(breakpoints=tuple(points), faces=faces)

    @classmethod
    def _parse_transport(cls, section: object) -> TransportSpec:
        """Parse the transport section into a TransportSpec.

        ``cfl_number`` must lie in (0, CFL_NUMBER_BOUND]; ``advection_scheme``
        defaults to umist; ``turbulent_schmidt`` is optional, positive and
        finite when present, and None when absent; the other two keys are
        required. Any other key raises.
        """
        if not isinstance(section, dict):
            raise ValueError("transport must be a mapping")
        for key in section:
            if key not in _TRANSPORT_KEYS:
                raise ValueError(
                    f"transport.{key} is not a recognised transport key; "
                    f"known: {sorted(_TRANSPORT_KEYS)}"
                )
        cfl_number = cls._require_positive_float(section, "cfl_number", "transport")
        if cfl_number > CFL_NUMBER_BOUND:
            raise ValueError(
                f"transport.cfl_number must be in (0, {CFL_NUMBER_BOUND}], the "
                f"limited scheme's stability bound, got {cfl_number}"
            )
        scheme = UMIST
        if "advection_scheme" in section:
            scheme = cls._require_string(section, "advection_scheme", "transport")
            if scheme not in ADVECTION_SCHEMES:
                raise ValueError(
                    f"transport.advection_scheme must be one of "
                    f"{list(ADVECTION_SCHEMES)}, got '{scheme}'"
                )
        # No default: the key is needed exactly where an eddy viscosity field
        # is used, and a value assumed here would be a parameter defined in
        # code (REQ-C01).
        turbulent_schmidt = None
        if "turbulent_schmidt" in section:
            turbulent_schmidt = cls._require_positive_float(
                section, "turbulent_schmidt", "transport"
            )
        return TransportSpec(
            cfl_number=cfl_number,
            advection_scheme=scheme,
            max_diffusion_iter=cls._require_positive_int(
                section, "max_diffusion_iter", "transport"
            ),
            diffusion_tol=cls._require_positive_float(
                section, "diffusion_tol", "transport"
            ),
            turbulent_schmidt=turbulent_schmidt,
        )

    @classmethod
    def _parse_turbulence(cls, section: object) -> TurbulenceSpec:
        """Parse the turbulence section into a TurbulenceSpec.

        ``variant`` defaults to standard and ``advection_scheme`` to upwind;
        every other key is required. ``cfl_number`` must lie in
        (0, CFL_NUMBER_BOUND] and ``alpha_turbulence`` in (0, 1]. Any other
        key raises.
        """
        if not isinstance(section, dict):
            raise ValueError("turbulence must be a mapping")
        for key in section:
            if key not in _TURBULENCE_KEYS:
                raise ValueError(
                    f"turbulence.{key} is not a recognised turbulence key; "
                    f"known: {sorted(_TURBULENCE_KEYS)}"
                )
        model = cls._require_choice(section, "model", "turbulence", TURBULENCE_MODELS)
        variant = STANDARD
        if "variant" in section:
            variant = cls._require_choice(
                section, "variant", "turbulence", TURBULENCE_VARIANTS
            )
        wall_treatment = cls._require_choice(
            section, "wall_treatment", "turbulence", WALL_TREATMENTS
        )
        cfl_number = cls._require_positive_float(section, "cfl_number", "turbulence")
        if cfl_number > CFL_NUMBER_BOUND:
            raise ValueError(
                f"turbulence.cfl_number must be in (0, {CFL_NUMBER_BOUND}], the "
                f"limited scheme's stability bound, got {cfl_number}"
            )
        # Upwind, not the transport section's umist: the limited value's branch
        # switching accompanies the coupled solve's non-convergence on the
        # product room, and upwind converges it (ADR-012 C, note of 2026-10-10).
        scheme = UPWIND
        if "advection_scheme" in section:
            scheme = cls._require_choice(
                section, "advection_scheme", "turbulence", ADVECTION_SCHEMES
            )
        return TurbulenceSpec(
            model=model,
            variant=variant,
            wall_treatment=wall_treatment,
            cfl_number=cfl_number,
            advection_scheme=scheme,
            alpha_turbulence=cls._require_relaxation_factor(
                section, "alpha_turbulence", "turbulence"
            ),
            max_iter=cls._require_positive_int(section, "max_iter", "turbulence"),
            tol=cls._require_positive_float(section, "tol", "turbulence"),
        )

    @classmethod
    def _require_choice(
        cls, section: dict, key: str, context: str, choices: tuple[str, ...]
    ) -> str:
        """Require a string that is one of ``choices``."""
        value = cls._require_string(section, key, context)
        if value not in choices:
            raise ValueError(
                f"{context}.{key} must be one of {list(choices)}, got '{value}'"
            )
        return value

    @staticmethod
    def _require_segment_type(
        section: dict, key: str, context: str, bc_type: str, allowed: str
    ) -> None:
        """Raise unless a segment key sits on the one segment type it is valid for."""
        if key in section and bc_type != allowed:
            raise ValueError(
                f"{context}.{key} is only valid on a {allowed} segment, "
                f"got type '{bc_type}'"
            )

    @staticmethod
    def _require_section(raw: dict, key: str) -> dict | list:
        """Require a top-level section exists in the config."""
        if key not in raw:
            raise ValueError(f"Missing required config section: '{key}'")
        return raw[key]

    @staticmethod
    def _require_string(section: dict, key: str, context: str) -> str:
        """Require a string value in a config section."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = section[key]
        if not isinstance(val, str):
            raise TypeError(
                f"{context}.{key} must be a string, got {type(val).__name__}"
            )
        return val

    @staticmethod
    def _finite_number(val: object, label: str) -> float:
        """A finite float from a non-bool number; TypeError or ValueError otherwise.

        Every numeric key passes through here (REQ-C02): a bool is an int in
        Python and NaN compares false against every bound, so neither is
        caught by a range check alone.
        """
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{label} must be a number, got {type(val).__name__}")
        if not math.isfinite(val):
            raise ValueError(f"{label} must be finite, got {val}")
        return float(val)

    @classmethod
    def _require_float(cls, section: dict, key: str, context: str) -> float:
        """Require a finite numeric value, return as float."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        return cls._finite_number(section[key], f"{context}.{key}")

    @classmethod
    def _require_positive_float(cls, section: dict, key: str, context: str) -> float:
        """Require a positive finite numeric value, return as float."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = cls._finite_number(section[key], f"{context}.{key}")
        if val <= 0:
            raise ValueError(f"{context}.{key} must be positive, got {val}")
        return val

    @classmethod
    def _optional_positive_float(
        cls, section: dict, key: str, context: str, default: float
    ) -> float:
        """A positive numeric value if the key is present, else the default."""
        if key not in section:
            return default
        return cls._require_positive_float(section, key, context)

    @classmethod
    def _optional_positive_int(
        cls, section: dict, key: str, context: str, default: int
    ) -> int:
        """A positive integer value if the key is present, else the default."""
        if key not in section:
            return default
        return cls._require_positive_int(section, key, context)

    @staticmethod
    def _require_positive_int(section: dict, key: str, context: str) -> int:
        """Require a positive integer value."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = section[key]
        if not isinstance(val, int) or isinstance(val, bool):
            raise TypeError(
                f"{context}.{key} must be an integer, got {type(val).__name__}"
            )
        if val <= 0:
            raise ValueError(f"{context}.{key} must be positive, got {val}")
        return val

    @classmethod
    def _require_float_list(cls, section: dict, key: str, context: str) -> list[float]:
        """Require a list of finite numeric values, return as list of floats."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = section[key]
        if not isinstance(val, list):
            raise TypeError(f"{context}.{key} must be a list, got {type(val).__name__}")
        return [
            cls._finite_number(item, f"{context}.{key}[{i}]")
            for i, item in enumerate(val)
        ]

    @classmethod
    def _require_positive_float_list(
        cls, section: dict, key: str, context: str
    ) -> list[float]:
        """Require a list of positive finite numeric values, return as list of floats."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = section[key]
        if not isinstance(val, list):
            raise TypeError(f"{context}.{key} must be a list, got {type(val).__name__}")
        if len(val) == 0:
            raise ValueError(f"{context}.{key} must not be empty")
        result = []
        for i, item in enumerate(val):
            number = cls._finite_number(item, f"{context}.{key}[{i}]")
            if number <= 0:
                raise ValueError(f"{context}.{key}[{i}] must be positive, got {item}")
            result.append(number)
        return result

    @classmethod
    def _require_relaxation_factor(cls, section: dict, key: str, context: str) -> float:
        """Require a relaxation factor in the range (0.0, 1.0].

        Zero is rejected because it causes division by zero in
        SIMPLE. Values above 1.0 cause divergence.
        """
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = cls._finite_number(section[key], f"{context}.{key}")
        if val <= 0.0 or val > 1.0:
            raise ValueError(f"{context}.{key} must be in (0.0, 1.0], got {val}")
        return val
