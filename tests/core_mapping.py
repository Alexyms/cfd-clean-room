"""The mapping between the compiled core's plain data and the reference's types (ADR-017 B).

The binding is handed data and returns data: a class's budget as its ten terms
by name, and a breach as ``_cfdcore.ConservationBreach`` carrying the class, the
leftover, the particles supplied and the limit. The reference's types for the
same things are ``MassBudget`` and ``ConservationBreachError``. Step 6's
``src/core_backend.py`` will hold this mapping; until it exists the parity tests
and prompt 90's probe share it here, so that every comparison is made on the
reference's own types and the binding imports nothing from ``src``.

Prompt 91 adds the march's: a ``_cfdcore.March2`` built from a configuration and
its mesh, its resident state set from the reference's fields, budgets and
trackers, an interval handed over from the reference's plan and boundary, a
step's budget row read as ``MassBudget`` objects, and the reference's trackers
set from the core's maps and recovery fields (``ArrivalTracker.restore`` and
``RecoveryTracker.restore``, whose checks then read the core's state).
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.boundary_concentration import (
    SURFACE_CEILING,
    SURFACE_FLOOR,
    SURFACE_WALL,
    ConcentrationBoundary,
)
from src.config import UPWIND, SimConfig
from src.mesh import SOLID, Mesh
from src.monitor import COUNTING_STATE, ArrivalTracker
from src.particles import ParticlePhysics
from src.solver_transport import DEPOSIT_SURFACES, ConservationBreachError, MassBudget
from src.staggered import FaceVelocities
from src.time_integration import (
    DEFINED_RTOL,
    HUNDRED_TO_ONE,
    IntervalPlan,
    RecoveryTracker,
)

if TYPE_CHECKING:
    import _cfdcore


def budget_from_terms(terms: Mapping[str, float | None]) -> MassBudget:
    """A MassBudget from the ten terms the core returns by name.

    Parameters
    ----------
    terms : Mapping[str, float | None]
        ``initial``, ``inflow``, ``outflow``, ``source``, one term per name of
        ``DEPOSIT_SURFACES``, ``current`` and ``leftover``. ``initial`` and
        ``current`` are None before the first step.

    Returns
    -------
    MassBudget
        A new budget holding the same values.
    """
    return MassBudget(
        initial=terms["initial"],
        inflow=terms["inflow"],
        outflow=terms["outflow"],
        source=terms["source"],
        deposited={surface: terms[surface] for surface in DEPOSIT_SURFACES},
        current=terms["current"],
        leftover=terms["leftover"],
    )


def terms_of(budget: MassBudget) -> dict[str, float | None]:
    """A MassBudget's ten terms by name, as the core takes them.

    Parameters
    ----------
    budget : MassBudget
        A budget whose ``deposited`` keys are ``DEPOSIT_SURFACES``.

    Returns
    -------
    dict[str, float | None]
        ``initial``, ``inflow``, ``outflow``, ``source``, the four deposition
        terms by surface name, ``current`` and ``leftover``.
    """
    return {
        "initial": budget.initial,
        "inflow": budget.inflow,
        "outflow": budget.outflow,
        "source": budget.source,
        **{surface: budget.deposited[surface] for surface in DEPOSIT_SURFACES},
        "current": budget.current,
        "leftover": budget.leftover,
    }


def budgets_of(core: "_cfdcore.TransportSolver2") -> list[MassBudget]:
    """Every class's budget of a core solver as new MassBudget objects."""
    return [budget_from_terms(terms) for terms in core.budgets()]


def restore_budgets(
    core: "_cfdcore.TransportSolver2", budgets: Sequence[MassBudget]
) -> None:
    """Set a core solver's budgets whole from MassBudget objects.

    The refusals and their messages are ``TransportSolver.restore_budgets``'s:
    they concern ``MassBudget`` and a checkpoint's leftover, which the core knows
    nothing of. The terms are then handed to the core as plain data.

    Parameters
    ----------
    core : _cfdcore.TransportSolver2
        The solver.
    budgets : Sequence[MassBudget]
        One per class, in class order.

    Raises
    ------
    TypeError
        If an element is not a MassBudget.
    ValueError
        If there is not one per class, a budget's ``deposited`` keys are not the
        four deposition surfaces, or its leftover is unknown.
    """
    check_budgets(core.classes, budgets)
    core.restore_budgets([terms_of(budget) for budget in budgets])


def check_budgets(classes: int, budgets: Sequence[MassBudget]) -> None:
    """``TransportSolver.restore_budgets``' refusals of MassBudget objects.

    Parameters
    ----------
    classes : int
        The classes the core holds.
    budgets : Sequence[MassBudget]
        One per class, in class order.

    Raises
    ------
    TypeError
        If an element is not a MassBudget.
    ValueError
        If there is not one per class, a budget's ``deposited`` keys are not the
        four deposition surfaces, or its leftover is unknown.
    """
    if len(budgets) != classes:
        raise ValueError(
            f"restore_budgets needs one budget per class ({classes}), "
            f"got {len(budgets)}"
        )
    for k, budget in enumerate(budgets):
        if not isinstance(budget, MassBudget):
            raise TypeError(
                f"budget {k} must be a MassBudget, got {type(budget).__name__}"
            )
        if tuple(budget.deposited) != DEPOSIT_SURFACES:
            raise ValueError(
                f"budget {k} books deposition to {list(budget.deposited)}, not "
                f"{list(DEPOSIT_SURFACES)}"
            )
        if budget.leftover is None:
            raise ValueError(
                f"budget {k}'s leftover is unknown: it was written before the "
                "implicit solve's leftover was booked (prompt 78), so the "
                "accumulated leftover before the restart is unknown and REQ-T05's "
                "bound cannot be carried across it (ADR-016)"
            )


def step(
    core: "_cfdcore.TransportSolver2", size_class: int, c: np.ndarray, out: np.ndarray
) -> "_cfdcore.TransportStep2":
    """One class step of a core solver, a breach raised as the reference's error.

    Parameters
    ----------
    core : _cfdcore.TransportSolver2
        The solver, its interval and step set.
    size_class : int
        The class.
    c : np.ndarray
        The class field, [ny, nx].
    out : np.ndarray
        The field advanced, [ny, nx].

    Returns
    -------
    _cfdcore.TransportStep2
        The step's records.

    Raises
    ------
    ConservationBreachError
        When the core raises ``_cfdcore.ConservationBreach``: the class's
        accumulated leftover reached the limit (ADR-016).
    """
    # Imported here, so that this module loads where the core is not built.
    import _cfdcore

    try:
        return core.step(size_class, c, out)
    except _cfdcore.ConservationBreach as breach:
        raise ConservationBreachError(
            breach.size_class, breach.leftover, breach.supplied, breach.limit
        ) from breach


# ---------------------------------------------------------------------------
# The march (prompt 91)


def recovery_keys(
    thresholds: Sequence[float], initial: Sequence[float], hold: float | None
) -> dict[str, object]:
    """The core's recovery keys, formed as ``RecoveryTracker`` forms its own.

    Parameters
    ----------
    thresholds, initial : Sequence[float]
        Per class: the threshold and the initial level, particles per m^3.
    hold : float or None
        ``scenario.hold_time``, s.

    Returns
    -------
    dict[str, object]
        ``target`` (the initial level over ``HUNDRED_TO_ONE``), ``defined``
        (the initial level at least ``HUNDRED_TO_ONE`` times the threshold,
        less ``DEFINED_RTOL``) and ``hold``.
    """
    levels = np.asarray(initial, dtype=np.float64)
    limits = np.asarray(thresholds, dtype=np.float64)
    target = levels / HUNDRED_TO_ONE
    defined = levels >= HUNDRED_TO_ONE * limits * (1.0 - DEFINED_RTOL)
    return {
        "target": [float(x) for x in target],
        "defined": [bool(x) for x in defined],
        "hold": hold,
    }


def core_march(
    config: SimConfig,
    mesh: Mesh,
    cells: Sequence[tuple[int, int]],
    onsets: Sequence[float],
    recovery: Mapping[str, object] | None,
) -> "_cfdcore.March2":
    """A core run on a configuration's mesh, transport keys, physics and channels.

    Parameters
    ----------
    config : SimConfig
        Supplies the transport section, ``t_end``, the particles, the
        thresholds and the channels.
    mesh : Mesh
        The run's mesh.
    cells : Sequence[tuple[int, int]]
        The (j, i) each sensor reads (``sensor_cells``), possibly none.
    onsets : Sequence[float]
        The contaminating events' onsets (``contaminating_events``).
    recovery : Mapping[str, object] or None
        ``recovery_keys``' dict for a clean-out, None otherwise.

    Returns
    -------
    _cfdcore.March2
        A run whose state ``restore_march`` sets before its first interval.
    """
    import _cfdcore

    spec = config.transport
    assert spec is not None
    physics = ParticlePhysics(config)
    n = len(config.particle_sizes)
    grid = _cfdcore.Grid2(mesh.x, mesh.xc, mesh.y, mesh.yc, mesh.cell_type, SOLID)
    return _cfdcore.March2(
        _cfdcore.ScalarScheme2(grid),
        mesh.dx_cell,
        mesh.dy_cell,
        mesh.dx_face,
        mesh.dy_face,
        spec.cfl_number,
        spec.advection_scheme == UPWIND,
        spec.max_diffusion_iter,
        spec.diffusion_tol,
        spec.conservation_limit,
        spec.leftover_share,
        config.t_end,
        spec.turbulent_schmidt,
        [physics.settling_velocity(k) for k in range(n)],
        [physics.diffusion_coeff(k) for k in range(n)],
        (SURFACE_FLOOR, SURFACE_CEILING, SURFACE_WALL),
        [list(members) for members in config.channel_classes()],
        list(config.class_thresholds("the core's march")),
        [(int(j), int(i)) for j, i in cells],
        [float(t) for t in onsets],
        None if recovery is None else dict(recovery),
    )


def stacked(fields: Mapping[int, np.ndarray]) -> np.ndarray:
    """The class fields as one new [classes, ny, nx] float64 array, class 0 first."""
    return np.ascontiguousarray(
        np.stack([np.asarray(fields[k], dtype=np.float64) for k in range(len(fields))])
    )


def unstacked(cube: np.ndarray) -> dict[int, np.ndarray]:
    """A [classes, ny, nx] array as class to its read-only field, views of it."""
    cube.flags.writeable = False
    return {k: cube[k] for k in range(cube.shape[0])}


def restore_march(
    core: "_cfdcore.March2",
    fields: Mapping[int, np.ndarray],
    budgets: Sequence[MassBudget],
    step: int,
    t: float,
    arrival: ArrivalTracker,
    tracker: RecoveryTracker | None,
) -> None:
    """Set a core run's resident state from the reference's (ADR-017 B, ``restore``).

    Parameters
    ----------
    core : _cfdcore.March2
        The run.
    fields : Mapping[int, np.ndarray]
        Class to its field, [ny, nx].
    budgets : Sequence[MassBudget]
        One per class; refused as ``restore_budgets`` refuses them.
    step : int
        The run's step count.
    t : float
        The time, s.
    arrival : ArrivalTracker
        The reference's maps, whose state is handed over.
    tracker : RecoveryTracker or None
        The reference's recovery tracker, whose state is handed over.
    """
    check_budgets(core.classes, budgets)
    maps = arrival.state()
    core.restore(
        stacked(fields),
        [terms_of(b) for b in budgets],
        int(step),
        float(t),
        np.ascontiguousarray(maps["first_above"]),
        float(maps["arrival_t"]),
        None
        if tracker is None
        else {k: np.ascontiguousarray(v) for k, v in tracker.state().items()},
    )


def set_interval(
    core: "_cfdcore.March2",
    boundary: ConcentrationBoundary,
    faces: FaceVelocities,
    eddy_viscosity: np.ndarray | None,
    sources: Sequence[np.ndarray | None],
    plan: IntervalPlan,
    first: int,
) -> None:
    """Hand a core run one interval of the reference's plan, from step FIRST.

    Parameters
    ----------
    core : _cfdcore.March2
        The run.
    boundary : ConcentrationBoundary
        The interval's concentration boundary (``conditions_for``).
    faces : FaceVelocities
        The interval's flow state's faces.
    eddy_viscosity : np.ndarray or None
        Its eddy viscosity, None on the laminar path.
    sources : Sequence[np.ndarray or None]
        Per class, the source rate, None for none.
    plan : IntervalPlan
        The interval as the march steps it.
    first : int
        The step within the plan the march continues from.
    """
    core.interval(
        [boundary.faces_for(k) for k in range(core.classes)],
        faces.u,
        faces.v,
        None,
        eddy_viscosity,
        list(sources),
        dt=plan.dt,
        steps=plan.steps,
        t_start=plan.t_start,
        t_end=plan.t_end,
        first=first,
    )


def budgets_of_row(row: np.ndarray) -> list[MassBudget]:
    """A march's budget row [classes, 10] as new MassBudget objects.

    The terms are in ``_cfdcore.budget_terms``' order; a NaN ``initial`` or
    ``current`` is a term not yet set (None).
    """
    import _cfdcore

    out = []
    for terms in row:
        named: dict[str, float | None] = dict(
            zip(_cfdcore.budget_terms, (float(x) for x in terms), strict=True)
        )
        for optional in ("initial", "current"):
            value = named[optional]
            if value is not None and np.isnan(value):
                named[optional] = None
        out.append(budget_from_terms(named))
    return out


def breach_error(result: Mapping[str, object]) -> ConservationBreachError | None:
    """The reference's error for a march that stopped on a breach, else None."""
    breach = result["breach"]
    if breach is None:
        return None
    assert isinstance(breach, Mapping)
    return ConservationBreachError(
        breach["size_class"], breach["leftover"], breach["supplied"], breach["limit"]
    )


def arrival_state(core: "_cfdcore.March2", onsets: np.ndarray) -> dict[str, np.ndarray]:
    """The core's maps in ``ArrivalTracker.state``'s form.

    Parameters
    ----------
    core : _cfdcore.March2
        The run.
    onsets : np.ndarray
        The tracker's onsets (``ArrivalTracker.onsets``).

    Returns
    -------
    dict[str, np.ndarray]
        ``first_above``, ``arrival_onsets``, ``arrival_t`` and
        ``COUNTING_STATE``, new arrays.
    """
    maps = core.arrival()
    return {
        "first_above": maps["first_above"],
        "arrival_onsets": np.array(onsets, dtype=np.float64),
        "arrival_t": np.array(maps["arrival_t"]),
        COUNTING_STATE: np.array(1.0),
    }


def sync_trackers(
    core: "_cfdcore.March2", arrival: ArrivalTracker, tracker: RecoveryTracker | None
) -> None:
    """Set the reference's trackers from the core's state, through their restores.

    The restores copy the arrays and make their checks of them (a map's time
    before its event's onset or in a SOLID cell, a shape not the tracker's),
    so a core state the reference could not have reached is refused here.

    Parameters
    ----------
    core : _cfdcore.March2
        The run.
    arrival : ArrivalTracker
        The reference's maps, set from the core's.
    tracker : RecoveryTracker or None
        The reference's recovery tracker, set from the core's state.
    """
    arrival.restore(arrival_state(core, arrival.onsets))
    if tracker is not None:
        state = core.recovery()
        assert state is not None
        tracker.restore(state)
