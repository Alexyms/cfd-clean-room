"""Sensor placement by arrival maps (ADR-015 D): the maps' readers (step 3).

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
is.

The scoring rules, the candidates and the search (section D's "Scoring a
layout", "Candidates" and "The search") are step 4's.
"""

import os
from typing import NamedTuple

import numpy as np

from src.config import SimConfig
from src.io_manager import (
    RunFile,
    TransportFrames,
    checked_onsets,
    checked_thresholds,
)
from src.mesh import SOLID, Mesh
from src.monitor import contaminating_events

# What resolves an arrival map's times: the march's every step, or the run
# file's stored frames.
STEP_RESOLVED = "step"
FRAME_RESOLVED = "frame"


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
    """

    maps: np.ndarray
    onsets: np.ndarray
    resolution: str


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
        The maps and onsets of ``/arrival/``, ``STEP_RESOLVED``; or, for a
        file without it, ``arrival_from_frames`` on its frames, with the
        thresholds, mesh and contaminating events of the merged
        configuration it records, ``FRAME_RESOLVED``.

    Raises
    ------
    KeyError
        If the file has no ``/arrival/`` and no frames.
    ValueError
        If its recorded configuration has no scenario section (a tier-2
        run's) or no threshold for a class.
    """
    run = run_file if isinstance(run_file, RunFile) else RunFile(run_file)
    try:
        record = run.read_arrival()
    except KeyError:
        record = None
    if record is not None:
        return RunArrival(record.first_above, record.onsets, STEP_RESOLVED)
    config = SimConfig.from_dict(run.read_config().mapping)
    if config.scenario is None:
        raise ValueError(
            f"{run.path} records no scenario section: the arrival maps are a tier-1 "
            "run's"
        )
    live = Mesh(config).cell_type != SOLID
    onsets = np.array([onset for _, onset in contaminating_events(config.scenario)])
    maps = arrival_from_frames(
        run.read_frames(),
        np.array(config.class_thresholds("the arrival maps")),
        live,
        onsets,
    )
    return RunArrival(maps, onsets, FRAME_RESOLVED)
