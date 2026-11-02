"""Sensor placement by arrival maps (ADR-015 D): the readers, the scoring, the search.

A run's arrival maps are the march's (``monitor.ArrivalTracker``), written
to the run file's ``/arrival/`` at each checkpoint and at the end, resolved
to the step. A run file written before ADR-015 step 3 (Phase 4's, and the
sweep's of step 1) has no ``/arrival/``; its maps are rebuilt from the stored
frames by the same rule, which resolves an arrival only to the output
cadence: a cell's time is the first stored frame at or after the onset at
which the cell is at or above its threshold. That is never before the
march's time, and on the runs measured within one frame spacing after it (a
cell can be below at one frame and above at a later one, so that is a
measurement and not a bound). A cell at or above threshold only between two
frames is not seen, so the map misses cells the steps reach (15 of 520 on
the coarse door run). ``arrival_from_run`` says which of the two a result
is, and why a file's ``/arrival/`` was not used.

Step 4 scores and searches layouts on the maps (section D's "Scoring a
layout", "Candidates" and "The search"). The unit a layout is scored over is
(scenario, event, class): every contaminating event of every scenario, in
every class, is a unit to detect, so a door opened twice is two units per
class. ``detection_units`` stacks the runs' latencies into those units,
``score`` and ``given_layout`` read a layout by lookup, ``candidates`` states
where a sensor may go, ``search`` finds the best layout of a count under a
rule, and ``rank`` orders layouts scored over the same units.
"""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple

import numpy as np

from src.boundary_registry import EDGES, WALL, BoundaryRegistry
from src.config import (
    MEAN,
    MOUNT_CEILING,
    MOUNT_FREE,
    MOUNT_OBSTACLE_SIDES,
    MOUNT_OBSTACLE_TOPS,
    MOUNT_SURFACES,
    MOUNT_WALLS,
    PLACEMENT_RULES,
    WORST_CASE,
    SimConfig,
)
from src.io_manager import (
    RunFile,
    TransportFrames,
    checked_onsets,
    checked_thresholds,
    real_number,
)
from src.mesh import SOLID, Mesh
from src.monitor import contaminating_events
from src.staggered import edge_cell_inputs, obstacle_face_inputs

# What resolves an arrival map's times: the march's every step, or the run
# file's stored frames.
STEP_RESOLVED = "step"
FRAME_RESOLVED = "frame"

# How a layout was found: lookup of a layout given to it, the argmin over
# single cells, every pair, the worst-case rule's set cover, or the mean
# rule's greedy additions and one-sensor exchanges.
GIVEN = "given"
ARGMIN = "argmin"
EXHAUSTIVE = "exhaustive"
SET_COVER = "set_cover"
GREEDY = "greedy_exchange"

# The set cover's dynamic programme holds one entry per subset of the
# coverable units, 2^units of them: 2^20 is a megabyte of flags. The
# reference set has fifteen units (three scenarios, five classes).
MAX_SET_COVER_UNITS = 20
# Members of the reachable unions combined with the masks at once in the
# programme, which bounds its working array to this times the masks.
_UNION_CHUNK = 1 << 14
# A cell centre exactly a standoff from a surface (a standoff of half a cell)
# meets it to within an ulp, and the inclusive comparison would then decide by
# rounding; the tolerance is far above rounding and far below any standoff
# meant.
STANDOFF_RTOL = 1.0e-9


class RunArrival(NamedTuple):
    """A run file's arrival maps and how finely they are resolved.

    Parameters
    ----------
    maps : np.ndarray
        [events, classes, ny, nx]: the first time at or after the event's
        onset at which the cell is at or above the class's threshold; inf
        for a non-SOLID cell never reached, NaN in SOLID cells.
    onsets : np.ndarray
        [events], seconds.
    resolution : str
        ``STEP_RESOLVED`` for the file's ``/arrival/``; ``FRAME_RESOLVED``
        for maps rebuilt from its frames.
    event_names : tuple[str, ...]
        The contaminating events, in the onsets' order.
    t_final : float
        The last time the maps read, seconds: ``/arrival/``'s, or the last
        stored frame's. The mean rule caps an undetected unit's latency at
        ``t_final`` minus its onset.
    fallback : str or None
        Why the file's ``/arrival/`` was not used, for ``FRAME_RESOLVED``:
        absent, or the check that refused it. None for ``STEP_RESOLVED``.
    """

    maps: np.ndarray
    onsets: np.ndarray
    resolution: str
    event_names: tuple[str, ...]
    t_final: float
    fallback: str | None = None


class Units(NamedTuple):
    """The units a layout is scored over: (scenario, event, class).

    Parameters
    ----------
    latencies : np.ndarray
        [units, ny, nx]: the unit's latency at each cell, seconds after its
        event's onset; inf for a non-SOLID cell never reached, NaN in SOLID
        cells.
    caps : np.ndarray
        [units]: the run's remaining time after the unit's onset, seconds,
        the latency the mean rule gives an undetected unit.
    labels : tuple[tuple[str, str, int], ...]
        (scenario, event name, class index) per unit.
    """

    latencies: np.ndarray
    caps: np.ndarray
    labels: tuple[tuple[str, str, int], ...]


class Score(NamedTuple):
    """A layout's three rules over a set of units (ADR-015 D).

    Parameters
    ----------
    worst : float
        The worst case: the largest over the units of the layout's
        latency, the least over its cells; inf when a unit is never
        detected.
    mean : float
        The mean over the units of the latency capped at the unit's cap.
    undetected : int
        The units the layout never detects.
    coverage : float or None
        The fraction of the units detected at or within ``within``; None
        when no time is stated.
    """

    worst: float
    mean: float
    undetected: int
    coverage: float | None


class Layout(NamedTuple):
    """A sensor layout, its score and how it was found.

    Parameters
    ----------
    cells : tuple[tuple[int, int], ...]
        (j, i) per sensor, in row-major order for a searched layout.
    score : Score
        Over ``units``.
    rule : str or None
        The rule the layout is best under, ``WORST_CASE`` or ``MEAN``; a
        worst-case search with no layout of the count detecting every unit
        falls to ``MEAN`` and says so here. None for a given layout.
    method : str
        ``GIVEN``, ``ARGMIN``, ``EXHAUSTIVE``, ``SET_COVER`` or ``GREEDY``.
    exact : bool
        True when the method proves the layout best under ``rule``: the
        argmin and every pair under either rule, the set cover in the worst
        case (its tie-break by the mean is the best its enumeration finds).
    units : tuple[tuple[str, str, int], ...]
        The labels of the units scored over.
    uncovered : tuple[tuple[str, str, int], ...]
        The units no candidate reaches, removed before the search and
        reported; empty for a given layout.
    distinct_masks : int or None
        For the set cover: the distinct masks of the candidates (the units
        each covers within the optimum), measured at the optimum, or at the
        largest latency when no cover of the count exists. None otherwise.
    maximal_masks : int or None
        Of those, the ones no other contains, which the programme runs on.
    """

    cells: tuple[tuple[int, int], ...]
    score: Score
    rule: str | None
    method: str
    exact: bool
    units: tuple[tuple[str, str, int], ...]
    uncovered: tuple[tuple[str, str, int], ...] = ()
    distinct_masks: int | None = None
    maximal_masks: int | None = None


def arrival_from_frames(
    frames: TransportFrames,
    thresholds: np.ndarray,
    live: np.ndarray,
    onsets: np.ndarray,
) -> np.ndarray:
    """The arrival maps at the stored frames' resolution (ADR-015 D).

    Parameters
    ----------
    frames : TransportFrames
        The run file's ``/transport/`` frames.
    thresholds : np.ndarray
        [classes], particles per cubic metre, finite and non-negative, one
        per class the frames hold.
    live : np.ndarray
        [ny, nx] bool, the non-SOLID cells.
    onsets : np.ndarray
        [events], the contaminating events' onsets, seconds, finite,
        non-negative and in order.

    Returns
    -------
    np.ndarray
        [events, classes, ny, nx]: per event, the first frame time at or
        after its onset at which the cell is at or above the class's
        threshold; inf for a non-SOLID cell no such frame reaches, NaN in
        SOLID cells.

    Raises
    ------
    TypeError
        If a threshold or an onset is not a real number (a bool is not).
    ValueError
        If the thresholds are not one per class of the frames, finite and
        non-negative, ``live`` is not the frames' [ny, nx], or the onsets are
        not [events], finite, non-negative and in order.

    Notes
    -----
    The march's own map (``ArrivalTracker``) reads every step, and the
    frames are a subset of its steps, so where both reach a cell this map
    is at or after the march's, and it reaches no cell the march's does not.
    It misses the cells at or above their threshold only between two frames.
    """
    thresholds = checked_thresholds(thresholds)
    onsets = checked_onsets(onsets)
    classes = sorted(frames.fields)
    if thresholds.shape != (len(classes),) or classes != list(range(len(classes))):
        raise ValueError(
            f"thresholds of shape {thresholds.shape} for the frames' classes {classes}"
        )
    times = np.asarray(frames.times, dtype=np.float64)
    out = np.full((onsets.size, len(classes), *live.shape), np.inf)
    for k in classes:
        stack = frames.fields[k]
        if stack.shape[1:] != live.shape:
            raise ValueError(
                f"class {k}'s frames are {stack.shape[1:]}; live is {live.shape}"
            )
        above = (stack >= thresholds[k]) & live
        for e, onset in enumerate(onsets):
            counted = above & (times >= onset)[:, None, None]
            reached = counted.any(axis=0)
            first = times[np.argmax(counted, axis=0)]
            out[e, k][reached] = first[reached]
    out[:, :, ~live] = np.nan
    return out


def arrival_from_run(run_file: RunFile | str | os.PathLike[str]) -> RunArrival:
    """A run file's arrival maps: its ``/arrival/``, else rebuilt from its frames.

    Parameters
    ----------
    run_file : RunFile or str or Path
        A tier-1 run's file.

    Returns
    -------
    RunArrival
        The maps, onsets, event names and ``t_final`` of ``/arrival/``,
        ``STEP_RESOLVED``; or, for a file whose ``/arrival/`` is absent or
        fails ``read_arrival``'s checks, ``arrival_from_frames`` on its
        frames, with the thresholds, mesh and contaminating events of the
        merged configuration it records and the last frame's time,
        ``FRAME_RESOLVED``, with ``fallback`` saying why.

    Raises
    ------
    KeyError
        If the file has no usable ``/arrival/`` and no frames.
    ValueError
        If its recorded configuration has no scenario section (a tier-2
        run's) or no threshold for a class.
    """
    run = run_file if isinstance(run_file, RunFile) else RunFile(run_file)
    try:
        record = run.read_arrival()
    except (KeyError, TypeError, ValueError) as error:
        # A KeyError is the group or one of its members missing; a TypeError
        # or ValueError is ArrivalRecord's check on what the group holds.
        reason = error.args[0] if error.args else repr(error)
        fallback = f"{type(error).__name__}: {reason}"
    else:
        return RunArrival(
            record.first_above,
            record.onsets,
            STEP_RESOLVED,
            record.event_names,
            record.t_final,
        )
    config = SimConfig.from_dict(run.read_config().mapping)
    if config.scenario is None:
        raise ValueError(
            f"{run.path} records no scenario section: the arrival maps are a tier-1 "
            "run's"
        )
    live = Mesh(config).cell_type != SOLID
    events = contaminating_events(config.scenario)
    onsets = np.array([onset for _, onset in events])
    frames = run.read_frames()
    maps = arrival_from_frames(
        frames,
        np.array(config.class_thresholds("the arrival maps")),
        live,
        onsets,
    )
    return RunArrival(
        maps,
        onsets,
        FRAME_RESOLVED,
        tuple(name for name, _ in events),
        float(frames.times[-1]),
        fallback,
    )


def latency(maps: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """The latencies of arrival maps: arrival minus the event's onset (ADR-015 D).

    Parameters
    ----------
    maps : np.ndarray
        [events, classes, ny, nx], as ``RunArrival.maps``.
    onsets : np.ndarray
        [events], seconds, finite, non-negative and in order.

    Returns
    -------
    np.ndarray
        [events, classes, ny, nx], seconds: inf where the map is inf (never
        reached), NaN where it is NaN (SOLID).

    Raises
    ------
    TypeError
        If an onset is not a real number.
    ValueError
        If the maps are not 4-D with one event per onset, or an arrival lies
        before its event's onset.
    """
    onsets = checked_onsets(onsets)
    maps = np.asarray(maps, dtype=np.float64)
    if maps.ndim != 4 or maps.shape[0] != onsets.size:
        raise ValueError(
            f"maps of shape {maps.shape} for {onsets.size} onsets: they must be "
            "[events, classes, ny, nx]"
        )
    out = maps - onsets[:, None, None, None]
    if (out < 0.0).any():
        raise ValueError(
            "an arrival lies before its event's onset: the map is not the onsets'"
        )
    return out


def detection_units(arrivals: Mapping[str, RunArrival]) -> Units:
    """Stack runs' latencies into the (scenario, event, class) units (ADR-015 D).

    Parameters
    ----------
    arrivals : Mapping[str, RunArrival]
        Scenario name to its run's maps, all on one mesh.

    Returns
    -------
    Units
        One unit per scenario, event and class, in the mapping's order, then
        the events', then the classes'.

    Raises
    ------
    TypeError
        If a scenario name is not a str, or a ``t_final`` is not a real
        number.
    ValueError
        If there is no run, the maps are not on one mesh, the event names do
        not match the events, a ``t_final`` is not finite, or an onset lies
        after its run's ``t_final``.
    """
    if not arrivals:
        raise ValueError("no run to score: give at least one scenario's maps")
    blocks: list[np.ndarray] = []
    caps: list[float] = []
    labels: list[tuple[str, str, int]] = []
    shape: tuple[int, ...] | None = None
    for name, arrival in arrivals.items():
        if not isinstance(name, str):
            raise TypeError(f"a scenario name must be a str, got {type(name).__name__}")
        lat = latency(arrival.maps, arrival.onsets)
        if shape is None:
            shape = lat.shape[2:]
        elif lat.shape[2:] != shape:
            raise ValueError(
                f"{name}'s maps are {lat.shape[2:]} and the first run's {shape}: the "
                "runs must share a mesh"
            )
        events = tuple(arrival.event_names)
        if len(events) != lat.shape[0]:
            raise ValueError(
                f"{name} names {len(events)} events for {lat.shape[0]} maps"
            )
        t_final = real_number("t_final", arrival.t_final)
        remaining = t_final - np.asarray(arrival.onsets, dtype=np.float64)
        if not np.isfinite(t_final) or (remaining < 0.0).any():
            raise ValueError(
                f"{name}'s t_final {t_final} must be finite and at or after every onset"
            )
        for e, event in enumerate(events):
            for k in range(lat.shape[1]):
                blocks.append(lat[e, k])
                caps.append(float(remaining[e]))
                labels.append((name, event, k))
    return Units(np.stack(blocks), np.array(caps), tuple(labels))


def exclude_units(units: Units, labels: Iterable[tuple[str, str, int]]) -> Units:
    """The units without those labelled, as a search reports its uncovered ones.

    Parameters
    ----------
    units : Units
        The units.
    labels : Iterable[tuple[str, str, int]]
        Labels of ``units`` to leave out.

    Returns
    -------
    Units
        The rest, in their order.

    Raises
    ------
    ValueError
        If a label is not one of the units', or none would be left.
    """
    drop = set(labels)
    unknown = drop - set(units.labels)
    if unknown:
        raise ValueError(f"no such units to leave out: {sorted(unknown)}")
    keep = np.array([label not in drop for label in units.labels])
    if not keep.any():
        raise ValueError("leaving those out leaves no unit to score")
    return Units(
        units.latencies[keep],
        units.caps[keep],
        tuple(label for label, kept in zip(units.labels, keep, strict=True) if kept),
    )


def score(
    units: Units,
    cells: Sequence[tuple[int, int]] | np.ndarray,
    *,
    within: float | None = None,
) -> Score:
    """A layout's worst case, mean and coverage over the units (ADR-015 D).

    For each unit the layout's latency is the least over its cells; the
    unit is (scenario, event, class), so every event of a scenario with
    several is a unit of its own.

    Parameters
    ----------
    units : Units
        From ``detection_units``.
    cells : Sequence[tuple[int, int]] or np.ndarray
        [sensors, 2]: (j, i) per sensor, non-SOLID, none twice.
    within : float or None
        Seconds, positive and finite: the coverage rule's time.

    Returns
    -------
    Score

    Raises
    ------
    TypeError
        If a cell index is not an integer, or ``within`` not a real number.
    ValueError
        If ``cells`` is empty, not [sensors, 2], off the grid, names a cell
        twice or a SOLID cell, or ``within`` is not positive and finite.
    """
    picked = _checked_cells(units, cells, "cells")
    rows = units.latencies[:, picked[:, 0], picked[:, 1]].T
    return _score_of(rows.min(axis=0), units.caps, _checked_within(within))


def given_layout(
    units: Units,
    cells: Sequence[tuple[int, int]] | np.ndarray,
    *,
    within: float | None = None,
) -> Layout:
    """A layout given to the scoring (the file's sensors), as a Layout to rank.

    Parameters
    ----------
    units : Units
        From ``detection_units``, or a search's units (``exclude_units``).
    cells : Sequence[tuple[int, int]] or np.ndarray
        As ``score`` takes them.
    within : float or None
        As ``score`` takes it.

    Returns
    -------
    Layout
        ``GIVEN``, not exact, no rule, the cells in the order given.

    Raises
    ------
    TypeError, ValueError
        As ``score``.
    """
    picked = _checked_cells(units, cells, "cells")
    return Layout(
        cells=tuple((int(j), int(i)) for j, i in picked),
        score=score(units, picked, within=within),
        rule=None,
        method=GIVEN,
        exact=False,
        units=units.labels,
    )


def candidates(
    mesh: Mesh, config: SimConfig, mount: Sequence[str], standoff: float
) -> np.ndarray:
    """The cells a sensor may be placed in (ADR-015 D, "Candidates").

    A candidate is a non-SOLID cell whose centre lies within ``standoff`` of
    a named surface, and not beside an inlet or outlet face: a sensor in the
    supply's or a return's face reads the boundary, not the room. The
    surfaces are the mesh faces between the room and a wall or obstacle, as
    the solver sees them: ``walls`` the left and right domain edges,
    ``ceiling`` the top edge, ``obstacle_tops`` and ``obstacle_sides`` the
    obstacles' top faces and their left and right faces; the floor is not a
    mounting surface. ``free`` is every non-SOLID cell, whatever the
    standoff, the set the free optimum is searched over.

    Parameters
    ----------
    mesh : Mesh
        The mesh of ``config``.
    config : SimConfig
        Its segments set the inlet and outlet faces; for a scenario, every
        interval's segments (an open door's two halves).
    mount : Sequence[str]
        Names from ``MOUNT_SURFACES``, at least one, none twice.
    standoff : float
        Metres, positive and finite: the largest distance from a cell's
        centre to the nearest point of a named surface.

    Returns
    -------
    np.ndarray
        [candidates, 2] int64, (j, i) in row-major order.

    Raises
    ------
    TypeError
        If a name is not a str or ``standoff`` not a real number.
    ValueError
        If ``mount`` is empty, names an unknown surface or one twice, the
        standoff is not positive and finite, the mesh is not the
        configuration's, or no cell is a candidate.
    """
    names = tuple(mount)
    if not names:
        raise ValueError(f"mount must name at least one of {list(MOUNT_SURFACES)}")
    for i, name in enumerate(names):
        if not isinstance(name, str):
            raise TypeError(f"mount[{i}] must be a str, got {type(name).__name__}")
        if name not in MOUNT_SURFACES:
            raise ValueError(
                f"mount[{i}] must be one of {list(MOUNT_SURFACES)}, got '{name}'"
            )
        if name in names[:i]:
            raise ValueError(f"mount names '{name}' twice")
    standoff = real_number("standoff", standoff)
    if not (np.isfinite(standoff) and standoff > 0.0):
        raise ValueError(f"standoff must be positive and finite, got {standoff}")
    if mesh.cell_type.shape != (config.ny, config.nx):
        raise ValueError(
            f"the mesh is {mesh.cell_type.shape} and the configuration "
            f"({config.ny}, {config.nx}): candidates need the configuration's mesh"
        )
    live = mesh.cell_type != SOLID
    if MOUNT_FREE in names:
        near = live.copy()
    else:
        xc, yc = np.meshgrid(mesh.xc, mesh.yc)
        distance = np.full(live.shape, np.inf)
        for name in names:
            for x0, y0, x1, y1 in _surface_faces(mesh, config, name):
                gap_x = np.maximum(np.maximum(x0 - xc, xc - x1), 0.0)
                gap_y = np.maximum(np.maximum(y0 - yc, yc - y1), 0.0)
                np.minimum(distance, np.hypot(gap_x, gap_y), out=distance)
        near = live & (distance <= standoff * (1.0 + STANDOFF_RTOL))
    near &= ~_beside_openings(mesh, config)
    j, i = np.nonzero(near)
    if j.size == 0:
        raise ValueError(
            f"no cell is a candidate: none within {standoff} m of {list(names)} that "
            "is not SOLID or beside an inlet or outlet face"
        )
    return np.stack([j, i], axis=1).astype(np.int64)


def search(
    units: Units,
    candidates: np.ndarray | Sequence[tuple[int, int]],
    count: int,
    rule: str,
    *,
    within: float | None = None,
) -> Layout:
    """The best layout of ``count`` sensors among the candidates under a rule (ADR-015 D).

    The units no candidate reaches are removed first and reported as
    ``uncovered``; the layout is scored over the rest. One sensor is the
    argmin and two every pair, exact under either rule. Above two, the
    worst-case rule is exact by the threshold sweep over a set cover, and
    the mean rule is greedy with one-sensor exchanges. Under the worst-case
    rule a tie is broken by the mean, and under the mean rule by the worst
    case; a tie left after both goes to the first in candidate order.

    Parameters
    ----------
    units : Units
        From ``detection_units``.
    candidates : np.ndarray or Sequence[tuple[int, int]]
        [n, 2], (j, i), as ``candidates`` returns them.
    count : int
        Sensors, positive, at most the candidates.
    rule : str
        ``WORST_CASE`` or ``MEAN``.
    within : float or None
        The coverage rule's time for the layout's score; it does not steer
        the search.

    Returns
    -------
    Layout
        Its ``rule`` is ``MEAN`` when the worst-case rule was asked and no
        layout of the count detects every coverable unit (the worst case is
        then inf for every layout, and the ranking falls to the mean).

    Raises
    ------
    TypeError
        If ``count`` is not an int, or a cell index not an integer.
    ValueError
        If the rule is unknown, the count is not positive or exceeds the
        candidates, a candidate is SOLID, off the grid or given twice, no
        candidate reaches any unit, or the set cover would need more than
        ``MAX_SET_COVER_UNITS`` units.

    Notes
    -----
    The set cover (worst-case rule, three or more sensors): at a threshold T
    each candidate's mask is the set of units it detects within T. The
    smallest T at which ``count`` masks cover every unit is the optimum;
    feasibility only grows with T, so the sorted distinct latencies are
    bisected. Feasibility is settled by dynamic programming over the subsets
    of the units, the fewest masks whose union contains each subset, built
    from the empty set (which needs none) over the masks no other contains.
    The covers attaining the optimum are enumerated by a depth-first search
    over those masks bounded by the count; each is made a layout by the
    cells of its masks that lower the mean most, the spare sensors added the
    same way, and the best of them by the mean is improved by one-sensor
    exchanges that keep the worst case at the optimum. The worst case is
    exact; the mean among the layouts attaining it is the best this finds.
    """
    if rule not in PLACEMENT_RULES:
        raise ValueError(f"rule must be one of {list(PLACEMENT_RULES)}, got {rule!r}")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError(f"count must be an int, got {type(count).__name__}")
    pool = _checked_cells(units, candidates, "candidates")
    if not 0 < count <= len(pool):
        raise ValueError(
            f"count must be positive and at most the {len(pool)} candidates, got "
            f"{count}"
        )
    within = _checked_within(within)
    # [candidates, units]: a candidate's latencies contiguous, so every mean
    # below is the same row reduction and two equal layouts score equally.
    table = np.ascontiguousarray(units.latencies[:, pool[:, 0], pool[:, 1]].T)
    reached = np.isfinite(table).any(axis=0)
    if not reached.any():
        raise ValueError("no candidate reaches any unit: there is nothing to search")
    uncovered = tuple(
        label for label, hit in zip(units.labels, reached, strict=True) if not hit
    )
    kept = tuple(label for label, hit in zip(units.labels, reached, strict=True) if hit)
    table = np.ascontiguousarray(table[:, reached])
    caps = units.caps[reached]
    found = _find(table, caps, int(count), rule)
    columns = sorted(found.columns)
    return Layout(
        cells=tuple((int(pool[c, 0]), int(pool[c, 1])) for c in columns),
        score=_score_of(table[columns].min(axis=0), caps, within),
        rule=found.rule,
        method=found.method,
        exact=found.exact,
        units=kept,
        uncovered=uncovered,
        distinct_masks=found.distinct,
        maximal_masks=found.maximal,
    )


def rank(layouts: Iterable[Layout]) -> list[Layout]:
    """Layouts ordered by the worst case, ties by the mean (ADR-015 D).

    Parameters
    ----------
    layouts : Iterable[Layout]
        Scored over the same units, at least one.

    Returns
    -------
    list[Layout]
        Best first; layouts equal under both rules keep their given order.

    Raises
    ------
    ValueError
        If there is no layout, or two are scored over different units.
    """
    out = list(layouts)
    if not out:
        raise ValueError("no layout to rank")
    for layout in out[1:]:
        if layout.units != out[0].units:
            raise ValueError(
                "the layouts are scored over different units; score them over one "
                "set (exclude_units) before ranking"
            )
    return sorted(out, key=lambda layout: (layout.score.worst, layout.score.mean))


# ---------------------------------------------------------------------------
# The scoring's and the search's helpers
# ---------------------------------------------------------------------------


class _Found(NamedTuple):
    """A search's columns of the candidate table and how they were found."""

    columns: tuple[int, ...]
    rule: str
    method: str
    exact: bool
    distinct: int | None = None
    maximal: int | None = None


def _checked_cells(
    units: Units, cells: Sequence[tuple[int, int]] | np.ndarray, label: str
) -> np.ndarray:
    """CELLS as [n, 2] int64, each a non-SOLID cell of the units' grid, none twice."""
    picked = np.asarray(cells)
    if picked.size == 0:
        raise ValueError(f"{label} is empty: give at least one (j, i)")
    if picked.dtype == np.bool_ or not np.issubdtype(picked.dtype, np.integer):
        raise TypeError(f"{label} must hold integer (j, i) indices, got {picked.dtype}")
    if picked.ndim != 2 or picked.shape[1] != 2:
        raise ValueError(f"{label} must be [n, 2], (j, i) per cell, got {picked.shape}")
    ny, nx = units.latencies.shape[1:]
    j, i = picked[:, 0], picked[:, 1]
    if ((j < 0) | (j >= ny) | (i < 0) | (i >= nx)).any():
        raise ValueError(f"{label} has a cell off the {ny}x{nx} grid")
    if len({(int(a), int(b)) for a, b in picked}) != len(picked):
        raise ValueError(f"{label} names a cell twice")
    if np.isnan(units.latencies[0, j, i]).any():
        raise ValueError(f"{label} has a SOLID cell, where no sensor reads air")
    return picked.astype(np.int64)


def _checked_within(within: float | None) -> float | None:
    """WITHIN as a positive finite float, or None."""
    if within is None:
        return None
    value = real_number("within", within)
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"within must be positive and finite, got {value}")
    return value


def _rules(rows: np.ndarray, caps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row of ROWS [layouts, units]: the worst case and the capped mean."""
    return rows.max(axis=1), np.minimum(rows, caps).mean(axis=1)


def _score_of(latencies: np.ndarray, caps: np.ndarray, within: float | None) -> Score:
    """The Score of one layout's per-unit latencies."""
    worst, mean = _rules(latencies[None, :], caps)
    coverage = None if within is None else float(np.mean(latencies <= within))
    return Score(
        worst=float(worst[0]),
        mean=float(mean[0]),
        undetected=int(np.isinf(latencies).sum()),
        coverage=coverage,
    )


def _best_row(rows: np.ndarray, caps: np.ndarray, rule: str) -> tuple[int, tuple]:
    """The best row of ROWS under RULE (ties by the other rule, then the first) and its key."""
    worst, mean = _rules(rows, caps)
    first, second = (worst, mean) if rule == WORST_CASE else (mean, worst)
    best = int(np.lexsort((np.arange(len(rows)), second, first))[0])
    return best, (first[best], second[best])


def _find(table: np.ndarray, caps: np.ndarray, count: int, rule: str) -> _Found:
    """The search on the candidate table [candidates, units], every unit reached."""
    if count <= 2:
        columns = (
            _best_one(table, caps, rule) if count == 1 else _best_two(table, caps, rule)
        )
        worst = table[list(columns)].min(axis=0).max()
        # Every layout of the count leaves a unit undetected: the worst case is
        # inf for all, and the (worst, mean) order is the mean's.
        found_rule = MEAN if rule == WORST_CASE and np.isinf(worst) else rule
        method = ARGMIN if count == 1 else EXHAUSTIVE
        return _Found(columns, found_rule, method, True)
    if rule == WORST_CASE:
        cover = _worst_case_cover(table, caps, count)
        if cover.columns:
            return cover
        columns = _greedy(table, caps, count)
        return _Found(columns, MEAN, GREEDY, False, cover.distinct, cover.maximal)
    return _Found(_greedy(table, caps, count), MEAN, GREEDY, False)


def _best_one(table: np.ndarray, caps: np.ndarray, rule: str) -> tuple[int, ...]:
    """The argmin over single candidates."""
    return (_best_row(table, caps, rule)[0],)


def _best_two(table: np.ndarray, caps: np.ndarray, rule: str) -> tuple[int, ...]:
    """Every pair of candidates, the best under RULE; ties to the first pair in order."""
    best: tuple | None = None
    for a in range(len(table) - 1):
        row, key = _best_row(np.minimum(table[a], table[a + 1 :]), caps, rule)
        if best is None or key < best[0]:
            best = (key, (a, a + 1 + row))
    assert best is not None
    return best[1]


def _added(
    table: np.ndarray, caps: np.ndarray, chosen: list[int], pool: np.ndarray
) -> int:
    """The candidate of POOL whose addition to CHOSEN lowers the mean most (ties by the worst)."""
    rows = table[pool]
    if chosen:
        rows = np.minimum(table[chosen].min(axis=0), rows)
    return int(pool[_best_row(rows, caps, MEAN)[0]])


def _exchanged(
    table: np.ndarray, caps: np.ndarray, chosen: list[int], limit: float | None
) -> list[int]:
    """CHOSEN improved by one-sensor exchanges until none lowers (mean, worst).

    With LIMIT, an exchange must keep the worst case at or below it.
    """
    chosen = list(chosen)
    worst, mean = _rules(table[chosen].min(axis=0)[None, :], caps)
    current = (mean[0], worst[0])
    while True:
        move: tuple | None = None
        pool = np.setdiff1d(np.arange(len(table)), chosen)
        for p in range(len(chosen)):
            others = chosen[:p] + chosen[p + 1 :]
            rows = table[pool]
            if others:
                rows = np.minimum(table[others].min(axis=0), rows)
            worst, mean = _rules(rows, caps)
            allowed = (
                np.ones(len(pool), dtype=bool) if limit is None else worst <= limit
            )
            if not allowed.any():
                continue
            order = np.lexsort((np.arange(len(pool)), worst, mean, ~allowed))
            best = int(order[0])
            key = (mean[best], worst[best])
            if key < current and (move is None or key < move[0]):
                move = (key, p, int(pool[best]))
        if move is None:
            return chosen
        current, p, cell = move
        chosen[p] = cell


def _greedy(table: np.ndarray, caps: np.ndarray, count: int) -> tuple[int, ...]:
    """The mean rule above two: greedy additions, then one-sensor exchanges."""
    chosen: list[int] = []
    for _ in range(count):
        chosen.append(
            _added(table, caps, chosen, np.setdiff1d(np.arange(len(table)), chosen))
        )
    return tuple(_exchanged(table, caps, chosen, None))


def _maximal(masks: np.ndarray) -> np.ndarray:
    """The non-empty distinct masks no other distinct mask contains."""
    masks = masks[masks != 0]
    inside = (masks[:, None] & masks[None, :]) == masks[:, None]
    np.fill_diagonal(inside, False)
    return masks[~inside.any(axis=1)]


def _fewest(masks: np.ndarray, units: int, count: int) -> np.ndarray:
    """Per subset of the units, the fewest MASKS whose union contains it (count + 1 for more).

    Built from the empty set, which needs no mask: the unions of k masks
    are the unions of k - 1 masks with one more, and a subset needs as few
    masks as the fewest of any union containing it.
    """
    size = 1 << units
    fewest = np.full(size, count + 1, dtype=np.int64)
    reach = np.zeros(size, dtype=bool)
    fewest[0] = 0
    reach[0] = True
    for k in range(1, count + 1):
        grown = reach.copy()
        members = np.flatnonzero(reach)
        for start in range(0, members.size, _UNION_CHUNK):
            chunk = members[start : start + _UNION_CHUNK]
            grown[(chunk[:, None] | masks[None, :]).ravel()] = True
        fewest[grown & ~reach] = k
        reach = grown
    subsets = np.arange(size)
    for b in range(units):
        lower = subsets[(subsets & (1 << b)) == 0]
        fewest[lower] = np.minimum(fewest[lower], fewest[lower | (1 << b)])
    return fewest


def _covers(
    masks: list[int], fewest: np.ndarray, full: int, count: int
) -> list[tuple[int, ...]]:
    """Every set of at most COUNT of MASKS covering FULL, by a bounded depth-first search.

    Each branch covers the lowest unit still uncovered, so every cover is
    reached; a branch whose remainder needs more masks than it has left is
    cut.
    """
    found: set[tuple[int, ...]] = set()

    def visit(remaining: int, chosen: tuple[int, ...], slots: int) -> None:
        if remaining == 0:
            found.add(tuple(sorted(chosen)))
            return
        if fewest[remaining] > slots:
            return
        lowest = remaining & -remaining
        for mask in masks:
            if mask & lowest:
                visit(remaining & ~mask, (*chosen, mask), slots - 1)

    visit(full, (), count)
    return sorted(found)


def _worst_case_cover(table: np.ndarray, caps: np.ndarray, count: int) -> _Found:
    """The worst-case rule's set cover; no columns when no cover of the count exists."""
    units = table.shape[1]
    if units > MAX_SET_COVER_UNITS:
        raise ValueError(
            f"the set cover over {units} units needs 2^{units} subsets; at most "
            f"MAX_SET_COVER_UNITS = {MAX_SET_COVER_UNITS} are searched exactly"
        )
    full = (1 << units) - 1
    bits = np.left_shift(1, np.arange(units, dtype=np.int64))
    values = np.unique(table[np.isfinite(table)])

    def at(threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        masks = ((table <= threshold) * bits).sum(axis=1)
        distinct = np.unique(masks)
        maximal = _maximal(distinct)
        return masks, distinct, maximal, _fewest(maximal, units, count)

    masks, distinct, maximal, fewest = at(values[-1])
    if fewest[full] > count:
        return _Found((), WORST_CASE, SET_COVER, True, distinct.size, maximal.size)
    low, high = 0, values.size - 1
    while low < high:
        middle = (low + high) // 2
        if at(values[middle])[3][full] <= count:
            high = middle
        else:
            low = middle + 1
    optimum = float(values[high])
    masks, distinct, maximal, fewest = at(optimum)
    every = np.arange(len(table))
    best: tuple | None = None
    for cover in _covers([int(m) for m in maximal], fewest, full, count):
        chosen: list[int] = []
        for mask in cover:
            chosen.append(_added(table, caps, chosen, np.flatnonzero(masks == mask)))
        while len(chosen) < count:
            chosen.append(_added(table, caps, chosen, np.setdiff1d(every, chosen)))
        _, mean = _rules(table[chosen].min(axis=0)[None, :], caps)
        key = (mean[0], tuple(sorted(chosen)))
        if best is None or key < best[0]:
            best = (key, chosen)
    assert best is not None
    columns = tuple(_exchanged(table, caps, best[1], optimum))
    return _Found(columns, WORST_CASE, SET_COVER, True, distinct.size, maximal.size)


def _surface_faces(
    mesh: Mesh, config: SimConfig, name: str
) -> list[tuple[float, float, float, float]]:
    """The faces of a mounting surface between it and the room, as (x0, y0, x1, y1)."""
    x, y = mesh.x, mesh.y
    live = mesh.cell_type != SOLID
    faces: list[tuple[float, float, float, float]] = []
    if name == MOUNT_WALLS:
        for j in range(len(y) - 1):
            if live[j, 0]:
                faces.append((x[0], y[j], x[0], y[j + 1]))
            if live[j, -1]:
                faces.append((x[-1], y[j], x[-1], y[j + 1]))
    elif name == MOUNT_CEILING:
        for i in range(len(x) - 1):
            if live[-1, i]:
                faces.append((x[i], y[-1], x[i + 1], y[-1]))
    elif name == MOUNT_OBSTACLE_TOPS:
        for obstacle in config.obstacles:
            top = obstacle_face_inputs(mesh, obstacle, "top")
            for r, c in zip(top.rows, top.cols, strict=True):
                faces.append((x[c], y[r], x[c + 1], y[r]))
    elif name == MOUNT_OBSTACLE_SIDES:
        for obstacle in config.obstacles:
            for face in ("left", "right"):
                side = obstacle_face_inputs(mesh, obstacle, face)
                for r, c in zip(side.rows, side.cols, strict=True):
                    faces.append((x[c], y[r], x[c], y[r + 1]))
    return faces


def _interval_configs(config: SimConfig) -> list[SimConfig]:
    """CONFIG and, for a scenario, the configuration of each distinct set of events."""
    out = [config]
    spec = config.scenario
    if spec is None:
        return out
    seen: set[tuple[str, ...]] = set()
    for t in spec.times()[:-1]:
        active = spec.active(t)
        names = tuple(event.name for event in active)
        if names and names not in seen:
            seen.add(names)
            out.append(SimConfig.from_dict(spec.mapping(active)))
    return out


def _beside_openings(mesh: Mesh, config: SimConfig) -> np.ndarray:
    """[ny, nx] bool: the cells with a face on an inlet or outlet segment or vent."""
    out = np.zeros(mesh.cell_type.shape, dtype=bool)
    ny, nx = out.shape
    for interval in _interval_configs(config):
        registry = BoundaryRegistry(interval)
        for edge in EDGES:
            coverage = registry.coverage_along(edge, *edge_cell_inputs(mesh, edge))
            for k, point in enumerate(coverage):
                if point.spec is None or point.condition.bc_type == WALL:
                    continue
                cell = {
                    "bottom": (0, k),
                    "top": (ny - 1, k),
                    "left": (k, 0),
                    "right": (k, nx - 1),
                }[edge]
                out[cell] = True
        for obstacle in interval.obstacles:
            for face in ("top", "bottom", "left", "right"):
                faces = obstacle_face_inputs(mesh, obstacle, face)
                coverage = registry.coverage_on(obstacle.name, face, faces.coordinates)
                for r, c, point in zip(faces.rows, faces.cols, coverage, strict=True):
                    if point.spec is None or point.condition.bc_type == WALL:
                        continue
                    # The face sits between the obstacle's cell and the room's,
                    # at index r (v) or c (u): the room's is above, below, left or
                    # right of it.
                    room = {
                        "top": (r, c),
                        "bottom": (r - 1, c),
                        "left": (r, c - 1),
                        "right": (r, c),
                    }[face]
                    out[room] = True
    return out
