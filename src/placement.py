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

Prompt 76b adds the all-clear role (section D's "Two roles"). An all-clear
unit is (scenario, class) for a scenario whose events all end before its run
does; ``clear_units`` reads each cell's clear time from the stored frames,
``clear_score`` reads a layout's against the room's, and ``search_roles``
searches the detection sensors and then adds the all-clear ones in turn.
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
    MOUNT_OBSTACLE_TOPS,
    MOUNT_STAND,
    MOUNT_SURFACES,
    MOUNT_WALLS,
    PLACEMENT_RULES,
    WORST_CASE,
    ScenarioSpec,
    SimConfig,
    Source,
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
# Masks tested for containment at once in ``_maximal``: the broadcast it forms
# is this many rows of the distinct masks, 8 bytes each. On independent noise
# at the product's size (22,731 distinct masks) the unchunked [d, d] form
# peaked at 4.7 GB (prompt 75b).
_MAXIMAL_CHUNK = 256
# A cell centre exactly a standoff from a surface (a standoff of half a cell)
# meets it to within an ulp, and the inclusive comparison would then decide by
# rounding; the tolerance is far above rounding and far below any standoff
# meant.
STANDOFF_RTOL = 1.0e-9


class ClearUnits(NamedTuple):
    """The all-clear units, (scenario, class), and their cells' clear times (ADR-015 D).

    Parameters
    ----------
    clear : np.ndarray
        [units, ny, nx], seconds: per cell, the earliest stored frame time at
        or after the unit's ``ends`` from which every stored frame to the
        run's end reads below the class's threshold; ``ends`` itself when no
        frame from then on reads at or above; inf ("not clear") when the
        last frame does; NaN in SOLID cells.
    ends : np.ndarray
        [units], seconds: t_e, the end of the scenario's last event (0 for
        the clean-out's initial condition).
    times : tuple[np.ndarray, ...]
        Per unit, [frames]: the stored frames' times at or after t_e.
    fractions : tuple[np.ndarray, ...]
        Per unit, [frames]: the fraction of the live area at or above the
        class's threshold at each of those frames.
    spacing : np.ndarray
        [units], seconds: the largest spacing between consecutive stored
        frames from t_e on, the clear times' resolution.
    labels : tuple[tuple[str, int], ...]
        (scenario, class index) per unit.

    Notes
    -----
    The clear times are frame-resolved: a cell above its threshold only
    between two stored frames is not seen, and a clear time is late by up to
    ``spacing``.
    """

    clear: np.ndarray
    ends: np.ndarray
    times: tuple[np.ndarray, ...]
    fractions: tuple[np.ndarray, ...]
    spacing: np.ndarray
    labels: tuple[tuple[str, int], ...]


class ClearScore(NamedTuple):
    """A layout's all-clear reading over the all-clear units (ADR-015 D).

    Parameters
    ----------
    layout : np.ndarray
        [units], seconds: the layout's clear time, the latest of its cells';
        inf when a cell is not clear by the run's end.
    room : np.ndarray
        [units], seconds: the room's clear time, the latest over the live
        cells; inf when some live cell is not clear.
    room_cells : tuple[tuple[int, int], ...]
        Per unit, (j, i) of the live cell that sets the room's clear time:
        the first in row-major order of those that are not clear, or of
        those at the latest time.
    interval : np.ndarray
        [units], seconds: the premature interval, the room's clear time less
        the layout's, zero or more; NaN where either is not clear (a "not
        clear" never enters arithmetic).
    fraction : np.ndarray
        [units]: the fraction of the live area at or above threshold at the
        first stored frame at or after the layout's clear time; NaN where the
        layout is not clear.
    worst : float
        The largest over the units of the ranking key: the interval; inf for
        a unit the layout clears and the room does not; zero for a unit the
        layout does not clear (it never declares the room clean, so it
        declares nothing early).
    mean : float
        The mean of the same keys.
    worst_fraction : float
        The largest over the units of the fraction, zero where the layout is
        not clear.
    """

    layout: np.ndarray
    room: np.ndarray
    room_cells: tuple[tuple[int, int], ...]
    interval: np.ndarray
    fraction: np.ndarray
    worst: float
    mean: float
    worst_fraction: float


class Roles(NamedTuple):
    """A layout searched in two roles: detection, then all-clear (ADR-015 D).

    Parameters
    ----------
    detection : Layout
        The detection sensors' search, ``count`` less ``all_clear`` of them,
        over the detection units it covers.
    all_clear : tuple[tuple[int, int], ...]
        (j, i) of the all-clear sensors, in the order they were added.
    layout : Layout
        The whole layout, the detection cells then the all-clear cells, as a
        given layout over the detection search's units.
    clear : ClearScore
        The whole layout's all-clear reading.
    """

    detection: "Layout"
    all_clear: tuple[tuple[int, int], ...]
    layout: "Layout"
    clear: ClearScore


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
    frame_resolved : tuple[bool, ...]
        Per unit, whether its maps were rebuilt from the run file's stored
        frames (``FRAME_RESOLVED``) and not read from the march's: such a
        latency is an upper bound on the step's, and a cell the steps reach
        only between two frames reads as never.
    """

    latencies: np.ndarray
    caps: np.ndarray
    labels: tuple[tuple[str, str, int], ...]
    frame_resolved: tuple[bool, ...]


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
    frame_resolved : bool
        True when some unit scored was read from maps rebuilt from stored
        frames (``Units.frame_resolved``): its latencies are upper bounds
        on the march's, and the layout's exactness, where it has any, is
        over those maps.
    """

    worst: float
    mean: float
    undetected: int
    coverage: float | None
    frame_resolved: bool = False


class Layout(NamedTuple):
    """A sensor layout, its score and how it was found.

    Parameters
    ----------
    cells : tuple[tuple[int, int], ...]
        (j, i) per sensor: sorted by (j, i), the row-major order, for a
        searched layout; in the order given for a given one.
    score : Score
        Over ``units``.
    rule : str or None
        The rule the layout is best under, ``WORST_CASE`` or ``MEAN``; a
        worst-case search with no layout of the count detecting every unit
        falls to ``MEAN`` and says so here. None for a given layout.
    method : str
        ``GIVEN``, ``ARGMIN``, ``EXHAUSTIVE``, ``SET_COVER`` or ``GREEDY``.
    exact : bool
        True when the method proves the layout best under ``rule``, over the
        maps scored: the argmin and every pair under either rule, the set
        cover in the worst case. Its tie-break by the mean is the best the
        enumeration finds and not always the least (ADR-015 D: test 75
        measured a mean above the least in 4 of 724 random searches at
        three sensors and 1 of 430 at four).
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

    @property
    def frame_resolved(self) -> bool:
        """Whether some unit scored was read from frame-resolved maps."""
        return self.score.frame_resolved


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
        not match the events, a ``resolution`` is neither ``STEP_RESOLVED``
        nor ``FRAME_RESOLVED``, a ``t_final`` is not finite, or an onset
        lies after its run's ``t_final``.

    Notes
    -----
    A run read from its stored frames (``arrival_from_run`` flags it) makes
    frame-resolved units, and the flag reaches every Score and Layout built
    on them. Runs of both resolutions may be scored together; the flag says
    some of the units are upper bounds.
    """
    if not arrivals:
        raise ValueError("no run to score: give at least one scenario's maps")
    blocks: list[np.ndarray] = []
    caps: list[float] = []
    labels: list[tuple[str, str, int]] = []
    flags: list[bool] = []
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
        if arrival.resolution not in (STEP_RESOLVED, FRAME_RESOLVED):
            raise ValueError(
                f"{name}'s resolution is {arrival.resolution!r}; it must be "
                f"{STEP_RESOLVED!r} or {FRAME_RESOLVED!r}"
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
                flags.append(arrival.resolution == FRAME_RESOLVED)
    return Units(np.stack(blocks), np.array(caps), tuple(labels), tuple(flags))


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
        tuple(
            flag for flag, kept in zip(units.frame_resolved, keep, strict=True) if kept
        ),
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
    return _score_of(
        rows.min(axis=0), units.caps, _checked_within(within), any(units.frame_resolved)
    )


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
    mesh: Mesh,
    configs: SimConfig | Sequence[SimConfig],
    mount: Sequence[str],
    standoff: float,
    *,
    stand_height: tuple[float, float] | None = None,
) -> np.ndarray:
    """The cells a sensor may be placed in (ADR-015 D, "Candidates").

    A candidate is a non-SOLID cell whose centre lies within ``standoff`` of
    a named surface, or a stand cell when ``stand`` is named, and that no
    exclusion removes. An opening excludes the cells beside its faces and
    within ``standoff`` of one: an opening is a face on a segment or vent that
    is not a wall in some interval of some scenario given, since a sensor in
    the supply's or a return's face reads the boundary, not the room, and an
    open doorway or a filter face has nothing to mount on. A ``source``
    event's box excludes the cells whose centres lie inside it or within
    ``standoff`` of it: the search must not use knowledge of where a release
    happens (Alex, 2026-11-07). The surfaces are the mesh faces between the
    room and a wall or obstacle, as the solver sees them: ``walls`` the left
    and right domain edges, ``ceiling`` the top edge, ``obstacle_tops`` and
    ``obstacle_sides`` the obstacles' top faces and their left and right
    faces; the floor is not a mounting surface. A ``stand`` cell is a
    non-SOLID cell whose centre's height above the floor lies in
    ``stand_height`` and that has no SOLID cell below it in its column (the
    stand stands on the floor); the standoff is not its rule. ``free`` is
    every non-SOLID cell outside the exclusions, the set the free optimum is
    searched over.

    Parameters
    ----------
    mesh : Mesh
        The mesh of every configuration given.
    configs : SimConfig or Sequence[SimConfig]
        The configurations of every scenario whose units are scored, at least
        one: a cell excluded under any of them is out. For a scenario, every
        interval's segments (an open door's two halves) and every source
        event's box.
    mount : Sequence[str]
        Names from ``MOUNT_SURFACES``, at least one, none twice.
    standoff : float
        Metres, positive and finite: the largest distance from a cell's
        centre to the nearest point of a named surface, and the distance
        inside which an opening's face or a source's box excludes a cell.
    stand_height : tuple[float, float] or None
        (low, high), metres above the floor, finite, 0 <= low < high: given
        exactly when ``mount`` names ``stand``.

    Returns
    -------
    np.ndarray
        [candidates, 2] int64, (j, i) in row-major order.

    Raises
    ------
    TypeError
        If a name is not a str, a configuration is not a SimConfig, or
        ``standoff`` or a stand height not a real number.
    ValueError
        If ``mount`` is empty, names an unknown surface or one twice, no
        configuration is given, the standoff is not positive and finite, the
        mesh is not every configuration's, ``stand_height`` is missing with
        ``stand``, given without it or not finite with 0 <= low < high, or
        no cell is a candidate.

    Notes
    -----
    An opening's face is never a surface, and the one exclusion is enough
    to make it so: every cell such a face would bring in lies within the
    standoff of that opening and is excluded (ADR-015 D, prompt 76). On
    320x120 the 0.05 m standoff holds two columns, so a rule of "beside the
    face" alone left the second column of an open doorway, the second row
    under the supply and the second column of the hood's exhaust as
    candidates. A source's ring is there because a cell one row outside the
    box reads the source as fast as the box's own cells do (prompt 76b).
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
    band = _checked_stand_height(names, stand_height)
    given = _checked_configs(mesh, configs)
    live = mesh.cell_type != SOLID
    limit = standoff * (1.0 + STANDOFF_RTOL)
    if MOUNT_FREE in names:
        near = live.copy()
    else:
        # The surfaces are the room's: every configuration given shares its
        # domain and obstacles (the mesh check), so the first one's serve.
        surfaces = [name for name in names if name != MOUNT_STAND]
        faces = [f for name in surfaces for f in _surface_faces(mesh, given[0], name)]
        near = live & (_distance_to(mesh, faces) <= limit)
        if band is not None:
            near |= _stand_cells(mesh, band)
    beside = np.zeros(live.shape, dtype=bool)
    openings: list[tuple[float, float, float, float]] = []
    boxes: list[tuple[float, float, float, float]] = []
    for config in given:
        cells, faces = _openings(mesh, config)
        beside |= cells
        openings += faces
        boxes += _source_boxes(config)
    near &= ~beside & (_distance_to(mesh, sorted(set(openings))) > limit)
    near &= _distance_to(mesh, sorted(set(boxes))) > limit
    j, i = np.nonzero(near)
    if j.size == 0:
        raise ValueError(
            f"no cell is a candidate: none within {standoff} m of {list(names)} that "
            "is not SOLID, beside an inlet or outlet face or within that distance of "
            "one or of a source's box"
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
    exact; the mean among the layouts attaining it is the best this finds,
    not always the least (see ``Layout.exact``).
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
    frame_resolved = any(np.asarray(units.frame_resolved, dtype=bool)[reached])
    found = _find(table, caps, int(count), rule)
    columns = list(found.columns)
    return Layout(
        cells=tuple(sorted((int(pool[c, 0]), int(pool[c, 1])) for c in columns)),
        score=_score_of(table[columns].min(axis=0), caps, within, frame_resolved),
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


def scenario_end(config: SimConfig) -> float | None:
    """t_e of an all-clear unit: the end of the scenario's last event (ADR-015 D).

    Parameters
    ----------
    config : SimConfig
        A scenario's configuration.

    Returns
    -------
    float or None
        Seconds: the latest over the events of each one's end, an initial
        condition's being its start (0); None when the configuration has no
        scenario or that time is not before ``t_end`` (the run does not see
        the room clear after its last event).
    """
    spec = config.scenario
    if spec is None or not spec.events:
        return None
    end = max(e.start if e.end is None else e.end for e in spec.events)
    return float(end) if end < config.t_end else None


def clear_times(
    frames: TransportFrames,
    thresholds: np.ndarray,
    live: np.ndarray,
    areas: np.ndarray,
    end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each cell's clear time from t_e on, at the stored frames' resolution (ADR-015 D).

    Parameters
    ----------
    frames : TransportFrames
        The run file's ``/transport/`` frames.
    thresholds : np.ndarray
        [classes], particles per cubic metre, finite and non-negative, one
        per class the frames hold.
    live : np.ndarray
        [ny, nx] bool, the non-SOLID cells.
    areas : np.ndarray
        [ny, nx], square metres per metre of depth: each cell's area.
    end : float
        t_e, seconds, finite and non-negative, at or before the last frame.

    Returns
    -------
    clear : np.ndarray
        [classes, ny, nx], seconds: the earliest stored frame time at or
        after ``end`` from which every stored frame reads below the class's
        threshold; ``end`` where no frame from ``end`` on reads at or above;
        inf ("not clear") where the last frame does; NaN in SOLID cells.
    times : np.ndarray
        [n], the stored frames' times at or after ``end``.
    fractions : np.ndarray
        [classes, n]: the fraction of the live area at or above the class's
        threshold at each of those frames.

    Raises
    ------
    TypeError
        If a threshold or ``end`` is not a real number.
    ValueError
        If the thresholds are not one per class of the frames, finite and
        non-negative, ``live`` or ``areas`` is not the frames' [ny, nx],
        ``end`` is not finite and non-negative, or no stored frame lies at or
        after it.

    Notes
    -----
    A cell that dips below its threshold and rises again is not clear at
    the dip: its clear time follows the last frame at or above.
    """
    thresholds = checked_thresholds(thresholds)
    end = real_number("end", end)
    if not (np.isfinite(end) and end >= 0.0):
        raise ValueError(f"end must be finite and non-negative, got {end}")
    classes = sorted(frames.fields)
    if thresholds.shape != (len(classes),) or classes != list(range(len(classes))):
        raise ValueError(
            f"thresholds of shape {thresholds.shape} for the frames' classes {classes}"
        )
    areas = np.asarray(areas, dtype=np.float64)
    if areas.shape != live.shape:
        raise ValueError(f"areas are {areas.shape}; live is {live.shape}")
    times = np.asarray(frames.times, dtype=np.float64)
    kept = times >= end
    if not kept.any():
        raise ValueError(
            f"no stored frame lies at or after t_e = {end} s (the last is at "
            f"{times[-1]} s)"
        )
    times = times[kept]
    n = times.size
    total = float(areas[live].sum())
    clear = np.full((len(classes), *live.shape), end)
    fractions = np.empty((len(classes), n))
    for k in classes:
        stack = frames.fields[k][kept]
        if stack.shape[1:] != live.shape:
            raise ValueError(
                f"class {k}'s frames are {stack.shape[1:]}; live is {live.shape}"
            )
        above = (stack >= thresholds[k]) & live
        fractions[k] = (above * areas).sum(axis=(1, 2)) / total
        ever = above.any(axis=0)
        last = n - 1 - np.argmax(above[::-1], axis=0)
        after = np.where(last < n - 1, times[np.minimum(last + 1, n - 1)], np.inf)
        clear[k][ever] = after[ever]
    clear[:, ~live] = np.nan
    return clear, times, fractions


def clear_units(runs: Mapping[str, RunFile | str | os.PathLike[str]]) -> ClearUnits:
    """The all-clear units of runs whose scenarios' events all end before the run does.

    Parameters
    ----------
    runs : Mapping[str, RunFile or str or Path]
        Scenario name to its tier-1 run's file, all on one mesh.

    Returns
    -------
    ClearUnits
        One unit per class of each run whose ``scenario_end`` is not None,
        in the mapping's order, then the classes'.

    Raises
    ------
    TypeError
        If a scenario name is not a str.
    ValueError
        If there is no run, no run's scenario ends before its run does, the
        runs are not on one mesh, or a run records no threshold for a class.
    """
    if not runs:
        raise ValueError("no run to read: give at least one scenario's file")
    clear: list[np.ndarray] = []
    ends: list[float] = []
    times: list[np.ndarray] = []
    fractions: list[np.ndarray] = []
    spacing: list[float] = []
    labels: list[tuple[str, int]] = []
    shape: tuple[int, ...] | None = None
    for name, source in runs.items():
        if not isinstance(name, str):
            raise TypeError(f"a scenario name must be a str, got {type(name).__name__}")
        run = source if isinstance(source, RunFile) else RunFile(source)
        config = SimConfig.from_dict(run.read_config().mapping)
        end = scenario_end(config)
        if end is None:
            continue
        mesh = Mesh(config)
        if shape is None:
            shape = mesh.cell_type.shape
        elif mesh.cell_type.shape != shape:
            raise ValueError(
                f"{name}'s mesh is {mesh.cell_type.shape} and the first run's {shape}: "
                "the runs must share a mesh"
            )
        live = mesh.cell_type != SOLID
        areas = np.outer(mesh.dy_cell, mesh.dx_cell)
        maps, kept, share = clear_times(
            run.read_frames(),
            np.array(config.class_thresholds("the all-clear units")),
            live,
            areas,
            end,
        )
        gap = float(np.diff(kept).max()) if kept.size > 1 else 0.0
        for k in range(maps.shape[0]):
            clear.append(maps[k])
            ends.append(end)
            times.append(kept)
            fractions.append(share[k])
            spacing.append(gap)
            labels.append((name, k))
    if not labels:
        raise ValueError(
            "no run's scenario ends before its run does: there is no all-clear unit"
        )
    return ClearUnits(
        np.stack(clear),
        np.array(ends),
        tuple(times),
        tuple(fractions),
        np.array(spacing),
        tuple(labels),
    )


def clear_score(
    units: ClearUnits, cells: Sequence[tuple[int, int]] | np.ndarray
) -> ClearScore:
    """A layout's clear time, the room's and the premature interval per unit (ADR-015 D).

    The layout declares the room clean when every sensor reads below, so its
    clear time is the latest of its cells'.

    Parameters
    ----------
    units : ClearUnits
        From ``clear_units``.
    cells : Sequence[tuple[int, int]] or np.ndarray
        [sensors, 2]: (j, i) per sensor, non-SOLID, none twice.

    Returns
    -------
    ClearScore

    Raises
    ------
    TypeError
        If a cell index is not an integer.
    ValueError
        If ``cells`` is empty, not [sensors, 2], off the grid, names a cell
        twice or a SOLID cell.
    """
    picked = _checked_grid_cells(units.clear[0], cells, "cells")
    layout = units.clear[:, picked[:, 0], picked[:, 1]].max(axis=1)
    return _clear_score_of(units, layout)


def add_all_clear(
    units: ClearUnits,
    candidates: np.ndarray | Sequence[tuple[int, int]],
    cells: Sequence[tuple[int, int]],
    count: int,
) -> tuple[tuple[int, int], ...]:
    """All-clear sensors added to a layout one at a time (ADR-015 D, "Two roles").

    Each is the candidate, not already in the layout, whose addition gives
    the least worst premature interval over the units, ties by the mean
    interval, then by the least largest area fraction, then the first in
    candidate order. The search is sequential, not joint.

    Parameters
    ----------
    units : ClearUnits
        From ``clear_units``.
    candidates : np.ndarray or Sequence[tuple[int, int]]
        [n, 2], (j, i), as ``candidates`` returns them.
    cells : Sequence[tuple[int, int]]
        The layout's cells so far (the detection sensors'), possibly none.
    count : int
        Sensors to add, from 0 to the candidates not in the layout.

    Returns
    -------
    tuple[tuple[int, int], ...]
        The added cells, in the order they were added.

    Raises
    ------
    TypeError
        If ``count`` is not an int, or a cell index not an integer.
    ValueError
        If ``count`` is negative or more than the candidates left, or a
        candidate or a cell is SOLID, off the grid or given twice.
    """
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError(f"count must be an int, got {type(count).__name__}")
    pool = _checked_grid_cells(units.clear[0], candidates, "candidates")
    chosen = [(int(j), int(i)) for j, i in cells]
    if chosen:
        _checked_grid_cells(units.clear[0], chosen, "cells")
    free = np.array([(int(j), int(i)) not in set(chosen) for j, i in pool])
    if not 0 <= count <= int(free.sum()):
        raise ValueError(
            f"count must be from 0 to the {int(free.sum())} candidates not in the "
            f"layout, got {count}"
        )
    # [candidates, units]: each candidate's clear time.
    table = units.clear[:, pool[:, 0], pool[:, 1]].T
    current = (
        units.clear[:, [c[0] for c in chosen], [c[1] for c in chosen]].max(axis=1)
        if chosen
        else np.full(len(units.labels), -np.inf)
    )
    added: list[tuple[int, int]] = []
    for _ in range(int(count)):
        layout = np.maximum(current[None, :], table)
        worst, mean, fraction = _clear_keys(units, layout)
        order = np.lexsort((np.arange(len(pool)), fraction, mean, worst, ~free))
        best = int(order[0])
        added.append((int(pool[best, 0]), int(pool[best, 1])))
        free[best] = False
        current = layout[best]
    return tuple(added)


def search_roles(
    units: Units,
    clear: ClearUnits,
    candidates: np.ndarray | Sequence[tuple[int, int]],
    count: int,
    all_clear: int,
    rule: str,
    *,
    within: float | None = None,
) -> Roles:
    """A layout of COUNT sensors in two roles, detection and all-clear (ADR-015 D).

    The detection sensors, ``count`` less ``all_clear``, are searched by
    ``search`` under ``rule`` over the detection units; then the all-clear
    sensors are added in turn by ``add_all_clear`` over the all-clear units.

    Parameters
    ----------
    units : Units
        The detection units, from ``detection_units``.
    clear : ClearUnits
        The all-clear units, from ``clear_units``, on the same mesh.
    candidates : np.ndarray or Sequence[tuple[int, int]]
        [n, 2], (j, i), as ``candidates`` returns them.
    count : int
        Sensors, positive.
    all_clear : int
        The all-clear sensors, from 0 to ``count`` minus 1.
    rule : str
        ``WORST_CASE`` or ``MEAN``, the detection search's rule.
    within : float or None
        The coverage rule's time for the layouts' scores.

    Returns
    -------
    Roles

    Raises
    ------
    TypeError
        If ``count`` or ``all_clear`` is not an int.
    ValueError
        If ``all_clear`` is not from 0 to ``count`` minus 1, the units are
        not on one mesh, or as ``search`` and ``add_all_clear`` raise.
    """
    for label, value in (("count", count), ("all_clear", all_clear)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{label} must be an int, got {type(value).__name__}")
    if not 0 <= all_clear < count:
        raise ValueError(
            f"all_clear must be from 0 to count - 1 = {count - 1}, got {all_clear}"
        )
    if units.latencies.shape[1:] != clear.clear.shape[1:]:
        raise ValueError(
            f"the detection units are {units.latencies.shape[1:]} and the all-clear "
            f"units {clear.clear.shape[1:]}: they must share a mesh"
        )
    detection = search(units, candidates, int(count - all_clear), rule, within=within)
    added = add_all_clear(clear, candidates, detection.cells, int(all_clear))
    cells = detection.cells + added
    scored = exclude_units(units, detection.uncovered) if detection.uncovered else units
    return Roles(
        detection=detection,
        all_clear=added,
        layout=given_layout(scored, cells, within=within),
        clear=clear_score(clear, cells),
    )


def room_clear(units: ClearUnits) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """The room's clear time per unit, the latest over the live cells, and its cell.

    Parameters
    ----------
    units : ClearUnits
        From ``clear_units``.

    Returns
    -------
    times : np.ndarray
        [units], seconds; inf when some live cell is not clear.
    cells : tuple[tuple[int, int], ...]
        Per unit, (j, i) of the cell that sets it: the first in row-major
        order of the cells not clear, or of those at the latest time.
    """
    times = np.empty(len(units.labels))
    cells: list[tuple[int, int]] = []
    for u, field in enumerate(units.clear):
        # SOLID cells are NaN and read as -inf, so no SOLID cell sets the time.
        flat = np.where(np.isnan(field), -np.inf, field).ravel()
        at = int(np.argmax(flat))
        times[u] = flat[at]
        cells.append(tuple(int(v) for v in np.unravel_index(at, field.shape)))
    return times, tuple(cells)


def _clear_keys(
    units: ClearUnits, layout: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row of LAYOUT [layouts, units] of clear times: worst key, mean key, worst fraction."""
    room, _ = room_clear(units)
    cleared = np.isfinite(layout)
    # A layout that is not clear is read as 0 before the subtraction, so no
    # "not clear" enters the arithmetic.
    gap = room[None, :] - np.where(cleared, layout, 0.0)
    keys = np.where(~cleared, 0.0, np.where(np.isinf(room)[None, :], np.inf, gap))
    fraction = np.zeros(layout.shape)
    for u in range(layout.shape[1]):
        column = layout[:, u]
        finite = np.isfinite(column)
        index = np.searchsorted(units.times[u], column[finite], side="left")
        fraction[finite, u] = units.fractions[u][index]
    return keys.max(axis=1), keys.mean(axis=1), fraction.max(axis=1)


def _clear_score_of(units: ClearUnits, layout: np.ndarray) -> ClearScore:
    """The ClearScore of one layout's per-unit clear times."""
    room, room_cells = room_clear(units)
    both = np.isfinite(layout) & np.isfinite(room)
    interval = np.full(layout.shape, np.nan)
    interval[both] = room[both] - layout[both]
    fraction = np.full(layout.shape, np.nan)
    for u in np.flatnonzero(np.isfinite(layout)):
        index = int(np.searchsorted(units.times[u], layout[u], side="left"))
        fraction[u] = units.fractions[u][index]
    worst, mean, worst_fraction = _clear_keys(units, layout[None, :])
    return ClearScore(
        layout=layout,
        room=room,
        room_cells=room_cells,
        interval=interval,
        fraction=fraction,
        worst=float(worst[0]),
        mean=float(mean[0]),
        worst_fraction=float(worst_fraction[0]),
    )


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
    return _checked_grid_cells(units.latencies[0], cells, label)


def _checked_grid_cells(
    grid: np.ndarray, cells: Sequence[tuple[int, int]] | np.ndarray, label: str
) -> np.ndarray:
    """CELLS as [n, 2] int64, each a cell of GRID [ny, nx] that is not NaN, none twice."""
    picked = np.asarray(cells)
    if picked.size == 0:
        raise ValueError(f"{label} is empty: give at least one (j, i)")
    if picked.dtype == np.bool_ or not np.issubdtype(picked.dtype, np.integer):
        raise TypeError(f"{label} must hold integer (j, i) indices, got {picked.dtype}")
    if picked.ndim != 2 or picked.shape[1] != 2:
        raise ValueError(f"{label} must be [n, 2], (j, i) per cell, got {picked.shape}")
    ny, nx = grid.shape
    j, i = picked[:, 0], picked[:, 1]
    if ((j < 0) | (j >= ny) | (i < 0) | (i >= nx)).any():
        raise ValueError(f"{label} has a cell off the {ny}x{nx} grid")
    if len({(int(a), int(b)) for a, b in picked}) != len(picked):
        raise ValueError(f"{label} names a cell twice")
    if np.isnan(grid[j, i]).any():
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


def _score_of(
    latencies: np.ndarray,
    caps: np.ndarray,
    within: float | None,
    frame_resolved: bool,
) -> Score:
    """The Score of one layout's per-unit latencies."""
    worst, mean = _rules(latencies[None, :], caps)
    coverage = None if within is None else float(np.mean(latencies <= within))
    return Score(
        worst=float(worst[0]),
        mean=float(mean[0]),
        undetected=int(np.isinf(latencies).sum()),
        coverage=coverage,
        frame_resolved=frame_resolved,
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
            # A layout exchanged here has three or more sensors (counts one and
            # two are exhaustive), so the others are never empty.
            others = chosen[:p] + chosen[p + 1 :]
            rows = np.minimum(table[others].min(axis=0), table[pool])
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
    """The non-empty distinct masks no other distinct mask contains.

    The masks must be distinct (``np.unique``'s), so that a mask contains
    itself and no other.
    """
    masks = masks[masks != 0]
    contained = np.zeros(masks.size, dtype=bool)
    for start in range(0, masks.size, _MAXIMAL_CHUNK):
        rows = masks[start : start + _MAXIMAL_CHUNK]
        inside = (rows[:, None] & masks[None, :]) == rows[:, None]
        inside[np.arange(rows.size), np.arange(start, start + rows.size)] = False
        contained[start : start + rows.size] = inside.any(axis=1)
    return masks[~contained]


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


def _checked_configs(
    mesh: Mesh, configs: SimConfig | Sequence[SimConfig]
) -> list[SimConfig]:
    """CONFIGS as a non-empty list of SimConfigs, each on MESH's grid."""
    given = [configs] if isinstance(configs, SimConfig) else list(configs)
    if not given:
        raise ValueError("give at least one configuration: the scenarios scored")
    for n, config in enumerate(given):
        if not isinstance(config, SimConfig):
            raise TypeError(
                f"configs[{n}] must be a SimConfig, got {type(config).__name__}"
            )
        if mesh.cell_type.shape != (config.ny, config.nx):
            raise ValueError(
                f"the mesh is {mesh.cell_type.shape} and configuration {n}'s "
                f"({config.ny}, {config.nx}): candidates need the configurations' mesh"
            )
    return given


def _checked_stand_height(
    names: tuple[str, ...], stand_height: tuple[float, float] | None
) -> tuple[float, float] | None:
    """STAND_HEIGHT checked against whether MOUNT names a stand."""
    if MOUNT_STAND not in names:
        if stand_height is not None:
            raise ValueError(
                f"stand_height is given but mount does not name '{MOUNT_STAND}'"
            )
        return None
    if stand_height is None:
        raise ValueError(f"mount names '{MOUNT_STAND}': give its stand_height")
    values = tuple(stand_height)
    if len(values) != 2:
        raise ValueError(f"stand_height must be (low, high), got {stand_height!r}")
    low, high = (real_number(f"stand_height[{n}]", v) for n, v in enumerate(values))
    if not (np.isfinite(low) and np.isfinite(high) and 0.0 <= low < high):
        raise ValueError(
            f"stand_height must be finite with 0 <= low < high, got ({low}, {high})"
        )
    return low, high


def _stand_cells(mesh: Mesh, band: tuple[float, float]) -> np.ndarray:
    """[ny, nx] bool: the cells a stand on the floor can hold an inlet in.

    The centre's height above the floor lies in BAND, inclusive, and no cell
    below it in its column is SOLID.
    """
    solid = mesh.cell_type == SOLID
    # A column's cells at and below row j, SOLID or not, by a running count.
    below = np.cumsum(solid, axis=0)
    height = np.asarray(mesh.yc) - mesh.y[0]
    in_band = (height >= band[0]) & (height <= band[1])
    return in_band[:, None] & (below == 0)


def _source_boxes(config: SimConfig) -> list[tuple[float, float, float, float]]:
    """The boxes of CONFIG's source events, as (x0, y0, x1, y1)."""
    spec: ScenarioSpec | None = config.scenario
    if spec is None:
        return []
    return [
        (e.change.x_start, e.change.y_start, e.change.x_end, e.change.y_end)
        for e in spec.events
        if isinstance(e.change, Source)
    ]


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
    else:
        # MOUNT_OBSTACLE_SIDES: candidates handles MOUNT_FREE without surfaces.
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


def _distance_to(
    mesh: Mesh, faces: list[tuple[float, float, float, float]]
) -> np.ndarray:
    """[ny, nx]: from each cell's centre to the nearest point of FACES, inf with none.

    A face is a closed rectangle (x0, y0, x1, y1), a segment when it is thin,
    and the distance is Euclidean: zero inside it.
    """
    xc, yc = np.meshgrid(mesh.xc, mesh.yc)
    distance = np.full(xc.shape, np.inf)
    for x0, y0, x1, y1 in faces:
        gap_x = np.maximum(np.maximum(x0 - xc, xc - x1), 0.0)
        gap_y = np.maximum(np.maximum(y0 - yc, yc - y1), 0.0)
        np.minimum(distance, np.hypot(gap_x, gap_y), out=distance)
    return distance


def _openings(
    mesh: Mesh, config: SimConfig
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """The cells beside an opening, [ny, nx] bool, and the openings' faces.

    An opening is a face on a segment or vent that is not a wall in some
    interval of the configuration. Each face is (x0, y0, x1, y1), the form
    ``_surface_faces`` gives.
    """
    x, y = mesh.x, mesh.y
    out = np.zeros(mesh.cell_type.shape, dtype=bool)
    ny, nx = out.shape
    found: set[tuple[float, float, float, float]] = set()
    for interval in _interval_configs(config):
        registry = BoundaryRegistry(interval)
        for edge in EDGES:
            coverage = registry.coverage_along(edge, *edge_cell_inputs(mesh, edge))
            for k, point in enumerate(coverage):
                if point.spec is None or point.condition.bc_type == WALL:
                    continue
                if edge in ("bottom", "top"):
                    row = 0 if edge == "bottom" else ny - 1
                    level = y[0] if edge == "bottom" else y[-1]
                    out[row, k] = True
                    found.add((x[k], level, x[k + 1], level))
                else:
                    column = 0 if edge == "left" else nx - 1
                    level = x[0] if edge == "left" else x[-1]
                    out[k, column] = True
                    found.add((level, y[k], level, y[k + 1]))
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
                    if face in ("top", "bottom"):
                        found.add((x[c], y[r], x[c + 1], y[r]))
                    else:
                        found.add((x[c], y[r], x[c], y[r + 1]))
    return out, sorted(found)
