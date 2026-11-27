"""The compiled core behind ``solver.backend: core`` (ADR-017 B and E).

A run under ``solver.backend: core`` (``SimConfig.backend``) is
``time_integration.run`` with two parts in the core: every steady flow solve is a
``CoreSolver``, ``StaggeredSolver`` with ``solve_steady``'s outer iterations in a
``_cfdcore.Flow2``, and the march is a ``CoreRun``, ``time_integration._Run``
with its march stepped by a ``_cfdcore.March2``. The plans, the events, the flow
states' bookkeeping, the alert monitor, the frames, the checkpoints, the restart
and the end records stay the reference's own code (ADR-017 E). ``steady_solver``
is the one factory both ``time_integration`` and ``scripts/benchmark.py`` build a
steady solver with, so a harness row runs on either backend.

The binding is handed data and returns data, and imports nothing from ``src``;
this module maps between the two. A class's budget crosses as its ten terms by
name and returns as ``MassBudget``; a condition a run can meet returns as a status
or a record, raised here as the reference's exception (``ConservationBreachError``,
``PositivityError``, the pressure solve's ``ZeroDivisionError`` and
``LinAlgError``, ``FlowNotConvergedError`` through ``time_integration``), so
``scripts/run_scenario.py`` exits as it does on the reference. The modules' set-ups
are built from the reference's own arguments: the momentum predictor (prompt 92),
the pressure correction and its multigrid (prompt 93), the k-epsilon model and the
wall functions (prompt 94), the stopping rule and the outer iteration (prompt 95),
and the march's run, its intervals and its trackers (prompt 91). The parity tests
build the same set-ups through these functions, so every comparison is made on
the reference's own types.

``_cfdcore`` is imported inside the functions that call it, so this module loads
where the core is not built (on Windows, say) and nothing imports the core unless
the key asks for it. A configuration that names ``core`` loads everywhere, so that
a core run's file is read and reported on wherever it is; ``require_core`` refuses
it where a core solver or march is built and the module cannot be imported. The
march's monitor reads the readings and channels the core returns
per step (``AlertMonitor.update_from_readings``), so the march goes to the core in
batches to the next frame, checkpoint or interval end; ``CoreRun.monitor_reads_fields``
puts the monitor on the fields instead, copied out of the core after every step,
which the parity tests and the probe use to show that the two give the same alerts.

Moved here from ``tests/core_mapping.py`` and ``tests/core_driver.py`` (prompt 96),
where prompts 90 to 95 built them.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.boundary_concentration import (
    SURFACE_CEILING,
    SURFACE_FLOOR,
    SURFACE_WALL,
    ConcentrationBoundary,
)
from src.boundary_registry import VELOCITY_INLET, BoundaryRegistry
from src.boundary_staggered import StaggeredBoundary
from src.config import (
    CORE_BACKEND,
    ERROR_ESTIMATE,
    UPWIND,
    BoundarySpec,
    SimConfig,
    check_core_module,
)
from src.io_manager import RunFile
from src.mesh import FLUID, SOLID, Mesh
from src.monitor import COUNTING_STATE, Alert, ArrivalTracker
from src.multigrid import (
    COARSEST_CELLS,
    DOMINANCE_TOLERANCE,
    SMOOTHER_FACTOR,
    SMOOTHING_SWEEPS,
)
from src.particles import ParticlePhysics
from src.pressure import (
    RESIDUAL_FLOOR,
    ZERO_SCALE,
    PressureCorrection,
    PressureCorrector,
)
from src.solver_staggered import (
    _STOP_REASONS,
    CAP_STOP,
    FlowState,
    StaggeredSolver,
    SteadyCheckpoint,
    _digest,
)
from src.solver_staggered import logger as solver_logger
from src.solver_transport import DEPOSIT_SURFACES, ConservationBreachError, MassBudget
from src.staggered import (
    FaceVelocities,
    allocate_fields,
    edge_cell_inputs,
    to_cell_centers,
)
from src.stopping import (
    RATE_WINDOW,
    RULE_VERSION_WITH_E,
    RULE_VERSION_WITHOUT_E,
    IterationState,
)
from src.time_integration import (
    DEFINED_RTOL,
    HUNDRED_TO_ONE,
    IntervalPlan,
    MarchStep,
    RecoveryTracker,
    _copies,
    _Run,
)
from src.turbulence import (
    E_WALL,
    KAPPA,
    VARIANTS,
    Y_STAR_FLOOR,
    PositivityError,
    TurbulenceConditions,
    TurbulenceState,
)
from src.turbulence import logger as turbulence_logger

if TYPE_CHECKING:
    import _cfdcore


def require_core() -> None:
    """Refuse to build a core solver or march where ``_cfdcore`` cannot be imported.

    Called first by ``CoreSolver`` and ``CoreRun``, so a run that names the core
    where it is not built is refused before it writes a file, with the reason
    and where it is built.

    Raises
    ------
    ValueError
        Naming ``scripts/build_core.sh`` (ADR-017 A); Windows has no build.
    """
    check_core_module()


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


def march_arguments(
    config: SimConfig,
    mesh: Mesh,
    cells: Sequence[tuple[int, int]],
    onsets: Sequence[float],
    recovery: Mapping[str, object] | None,
) -> dict[str, object]:
    """The arguments ``_cfdcore.March2`` is built from, by the binding's names.

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
    dict[str, object]
        Keyword arguments for ``_cfdcore.March2``, in its order, so a test can
        change one and build the run.
    """
    import _cfdcore

    spec = config.transport
    assert spec is not None
    physics = ParticlePhysics(config)
    n = len(config.particle_sizes)
    grid = _cfdcore.Grid2(mesh.x, mesh.xc, mesh.y, mesh.yc, mesh.cell_type, SOLID)
    return {
        "scheme": _cfdcore.ScalarScheme2(grid),
        "dx_cell": mesh.dx_cell,
        "dy_cell": mesh.dy_cell,
        "dx_face": mesh.dx_face,
        "dy_face": mesh.dy_face,
        "cfl": spec.cfl_number,
        "upwind": spec.advection_scheme == UPWIND,
        "max_sweeps": spec.max_diffusion_iter,
        "tol": spec.diffusion_tol,
        "limit": spec.conservation_limit,
        "share": spec.leftover_share,
        "horizon": config.t_end,
        "schmidt": spec.turbulent_schmidt,
        "settling": [physics.settling_velocity(k) for k in range(n)],
        "diffusion": [physics.diffusion_coeff(k) for k in range(n)],
        "surface_codes": (SURFACE_FLOOR, SURFACE_CEILING, SURFACE_WALL),
        "channels": [list(members) for members in config.channel_classes()],
        "thresholds": list(config.class_thresholds("the core's march")),
        "sensors": [(int(j), int(i)) for j, i in cells],
        "onsets": [float(t) for t in onsets],
        "recovery": None if recovery is None else dict(recovery),
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

    return _cfdcore.March2(**march_arguments(config, mesh, cells, onsets, recovery))


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


def momentum_predictor(
    mesh: Mesh,
    config: SimConfig,
    boundary: StaggeredBoundary,
    grid: "_cfdcore.Grid2 | None" = None,
) -> "_cfdcore.MomentumPredictor2":
    """The core's momentum predictor from ``MomentumPredictor``'s own arguments (prompt 92).

    Parameters
    ----------
    mesh : Mesh
        The mesh.
    config : SimConfig
        Supplies rho, mu, alpha_velocity and momentum_sweeps.
    boundary : StaggeredBoundary
        Supplies the edges' tangential conditions and the vent faces, which
        are handed over only where there are any, as the reference holds them.
    grid : _cfdcore.Grid2, optional
        The mesh's grid where the caller has one, else a new one.

    Returns
    -------
    _cfdcore.MomentumPredictor2
        The predictor; its set-up is fixed, and its grid lives as long.
    """
    import _cfdcore

    edges = boundary.tangential_conditions()
    vents = boundary.vent_faces()
    return _cfdcore.MomentumPredictor2(
        core_grid(mesh) if grid is None else grid,
        mesh.dx_cell,
        mesh.dy_cell,
        mesh.dx_face,
        mesh.dy_face,
        config.rho,
        config.mu,
        config.alpha_velocity,
        config.momentum_sweeps,
        [
            (edges[e].is_dirichlet, edges[e].value, edges[e].wall_distance)
            for e in ("bottom", "top", "left", "right")
        ],
        vents.normal_u if vents.any else None,
        vents.normal_v if vents.any else None,
    )


# np.linalg.pinv's default cutoff, relative to the largest eigenvalue's magnitude.
PINV_RCOND = 1e-15

# The keys of the core's multigrid, from the reference's constants: the coarsest
# level's cells, the sweeps, their weight, the dominance test's tolerance and the
# pseudo-inverse's cutoff.
MULTIGRID_LEVELS = (
    COARSEST_CELLS,
    SMOOTHING_SWEEPS,
    SMOOTHER_FACTOR / 2.0,
    DOMINANCE_TOLERANCE,
    PINV_RCOND,
)


def pressure_corrector(
    mesh: Mesh,
    config: SimConfig,
    boundary: StaggeredBoundary,
    reference: PressureCorrector,
    grid: "_cfdcore.Grid2 | None" = None,
) -> "_cfdcore.PressureCorrector2":
    """The core's pressure correction from ``PressureCorrector``'s own arguments (prompt 93).

    Parameters
    ----------
    mesh : Mesh
        The mesh.
    config : SimConfig
        Supplies rho, alpha_pressure, pressure_rtol, max_pressure_iter and
        pressure_preconditioner.
    boundary : StaggeredBoundary
        Supplies the pressure outlets.
    reference : PressureCorrector
        Supplies the flux scale, which the boundary's inflow makes and which
        stays in Python (ADR-017 E).
    grid : _cfdcore.Grid2, optional
        The mesh's grid where the caller has one, else a new one.

    Returns
    -------
    _cfdcore.PressureCorrector2
        The corrector; its set-up is fixed, and its grid lives as long.
    """
    import _cfdcore

    outlets = boundary.pressure_outlets()
    return _cfdcore.PressureCorrector2(
        core_grid(mesh) if grid is None else grid,
        mesh.dx_cell,
        mesh.dy_cell,
        [outlets[edge].is_outlet for edge in ("left", "right", "bottom", "top")],
        FLUID,
        config.rho,
        config.alpha_pressure,
        config.pressure_rtol,
        RESIDUAL_FLOOR * reference.flux_scale,
        config.max_pressure_iter,
        config.pressure_preconditioner,
        MULTIGRID_LEVELS,
    )


def outlet_faces(
    mesh: Mesh, boundary: StaggeredBoundary, grid: "_cfdcore.Grid2 | None" = None
) -> "_cfdcore.OutletFaces2":
    """The core's outlet faces from the boundary's pressure outlets (prompt 95).

    Parameters
    ----------
    mesh : Mesh
        The mesh.
    boundary : StaggeredBoundary
        Supplies the pressure outlets, as ``StaggeredSolver`` reads them.
    grid : _cfdcore.Grid2, optional
        The mesh's grid where the caller has one, else a new one.

    Returns
    -------
    _cfdcore.OutletFaces2
        The faces ``StaggeredSolver._extrapolate_outlets`` writes.
    """
    import _cfdcore

    outlets = boundary.pressure_outlets()
    return _cfdcore.OutletFaces2(
        core_grid(mesh) if grid is None else grid,
        [outlets[edge].is_outlet for edge in ("left", "right", "bottom", "top")],
    )


def correction_from(
    core: Mapping[str, np.ndarray | int | bool | float],
) -> PressureCorrection:
    """A correction of the core's, by name, as the reference's ``PressureCorrection``."""
    return PressureCorrection(
        u=core["u"],
        v=core["v"],
        p=core["p"],
        p_prime=core["p_prime"],
        iterations=core["iterations"],
        reached_cap=core["reached_cap"],
        products=core["products"],
    )


def breakdown_error(breakdown: "_cfdcore.PressureBreakdown") -> Exception:
    """The error the reference's arithmetic raises where a core solve broke down.

    Parameters
    ----------
    breakdown : _cfdcore.PressureBreakdown
        What the core raised.

    Returns
    -------
    Exception
        A ZeroDivisionError, which Python's float division raises, or the
        LinAlgError of an eigenproblem that does not converge.
    """
    if breakdown.reason == "division_by_zero":
        return ZeroDivisionError("division by zero in the conjugate gradients")
    return np.linalg.LinAlgError("Eigenvalues did not converge")


# The law of the wall as the core reads it: kappa, E and the scalable form's floor.
WALL_LAW = (KAPPA, E_WALL, Y_STAR_FLOOR)

# The domain's edges in the order the core's wall functions take them.
WALL_EDGES = ("bottom", "top", "left", "right")


def core_grid(mesh: Mesh) -> "_cfdcore.Grid2":
    """The core's grid of a mesh; two modules built on one share it."""
    import _cfdcore

    return _cfdcore.Grid2(mesh.x, mesh.xc, mesh.y, mesh.yc, mesh.cell_type, SOLID)


def k_epsilon(
    mesh: Mesh,
    config: SimConfig,
    boundary: StaggeredBoundary,
    grid: "_cfdcore.Grid2 | None" = None,
) -> "_cfdcore.KEpsilon2":
    """The core's k-epsilon model from ``KEpsilonModel``'s own arguments (prompt 94).

    Parameters
    ----------
    mesh : Mesh
        The mesh.
    config : SimConfig
        Supplies the turbulence section, rho and mu.
    boundary : StaggeredBoundary
        Supplies the vent faces, the runs the reference derives itself, handed
        over only where there are any.
    grid : _cfdcore.Grid2, optional
        The mesh's grid where the caller has one, else a new one.

    Returns
    -------
    _cfdcore.KEpsilon2
        The model; its grid lives as long.
    """
    import _cfdcore

    spec = config.turbulence
    assert spec is not None
    constants = VARIANTS[spec.variant]
    vents = boundary.vent_faces()
    return _cfdcore.KEpsilon2(
        core_grid(mesh) if grid is None else grid,
        mesh.dx_cell,
        mesh.dy_cell,
        mesh.dx_face,
        mesh.dy_face,
        (
            constants.c_mu,
            constants.c_1,
            constants.c_2,
            constants.sigma_k,
            constants.sigma_e,
            constants.eta_0,
            constants.beta,
        ),
        spec.cfl_number,
        spec.advection_scheme == UPWIND,
        spec.max_iter,
        spec.tol,
        config.mu / config.rho,
        vents.normal_u if vents.any else None,
        vents.normal_v if vents.any else None,
    )


# One inlet face as the core takes it: direction, row, column, intensity, normal
# speed, dissipation length and width.
InletFace = tuple[int, int, int, float, float, float, float]


def inlet_faces(
    mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary
) -> tuple[list[InletFace], list[InletFace], list[tuple[int, int, int]]]:
    """The faces that admit air and the exhausts', in ``TurbulenceBoundary``'s order.

    Parameters
    ----------
    mesh : Mesh
        The mesh.
    config : SimConfig
        Supplies the segments.
    boundary : StaggeredBoundary
        Supplies the vents.

    Returns
    -------
    tuple
        The domain faces that admit air, the discharge vents' faces, and the
        exhausts' faces as (direction, row, column).

    Raises
    ------
    ValueError
        If a segment that admits air states no turbulence, as the reference.
    """
    registry = BoundaryRegistry(config)
    ny, nx = mesh.cell_type.shape
    domain: list[InletFace] = []
    for edge in WALL_EDGES:
        horizontal = edge in ("bottom", "top")
        widths = mesh.dx_cell if horizontal else mesh.dy_cell
        coverage = registry.coverage_along(edge, *edge_cell_inputs(mesh, edge))
        for index, point in enumerate(coverage):
            if point.spec is None or point.condition.bc_type != VELOCITY_INLET:
                continue
            c = point.condition
            normal = c.v_prescribed if horizontal else c.u_prescribed
            if normal == 0.0:
                continue
            intensity, length = _turbulence_of(point.name, point.spec)
            face = {
                "bottom": (1, 0, index),
                "top": (1, ny, index),
                "left": (0, index, 0),
                "right": (0, index, nx),
            }[edge]
            domain.append((*face, intensity, normal, length, float(widths[index])))
    discharges: list[InletFace] = []
    exhausts: list[tuple[int, int, int]] = []
    for run in boundary.vent_runs():
        direction = 0 if run.component == "u" else 1
        faces = [
            (direction, int(j), int(i)) for j, i in zip(run.rows, run.cols, strict=True)
        ]
        if not run.is_inlet:
            exhausts.extend(faces)
            continue
        intensity, length = _turbulence_of(run.name, run.spec)
        discharges.extend(
            (*face, intensity, run.normal, length, float(width))
            for face, width in zip(faces, run.widths, strict=True)
        )
    return domain, discharges, exhausts


def _turbulence_of(name: str | None, spec: BoundarySpec) -> tuple[float, float]:
    """A segment's turbulence intensity and dissipation length, or the reference's refusal."""
    intensity, length = spec.turbulence_intensity, spec.dissipation_length
    if intensity is None or length is None:
        raise ValueError(
            f"boundary '{name}' admits air but states no "
            "turbulence_intensity or dissipation_length"
        )
    return intensity, length


def turbulence_boundary(
    mesh: Mesh,
    config: SimConfig,
    boundary: StaggeredBoundary,
    grid: "_cfdcore.Grid2 | None" = None,
) -> "_cfdcore.TurbulenceBoundary2":
    """The core's wall functions from ``TurbulenceBoundary``'s own arguments (prompt 94).

    Parameters
    ----------
    mesh : Mesh
        The mesh.
    config : SimConfig
        Supplies the variant, rho, mu and the segments' turbulence keys.
    boundary : StaggeredBoundary
        Supplies the wall faces, the tangential conditions and the vents.
    grid : _cfdcore.Grid2, optional
        The mesh's grid where the caller has one, else a new one.

    Returns
    -------
    _cfdcore.TurbulenceBoundary2
        The boundary; its grid lives as long.
    """
    import _cfdcore

    assert config.turbulence is not None
    walls = boundary.wall_faces()
    edges = boundary.tangential_conditions()
    vents = boundary.vent_faces()
    domain, discharges, exhausts = inlet_faces(mesh, config, boundary)
    return _cfdcore.TurbulenceBoundary2(
        core_grid(mesh) if grid is None else grid,
        [walls[e].cells for e in WALL_EDGES],
        [walls[e].corners for e in WALL_EDGES],
        [
            (edges[e].is_dirichlet, edges[e].value, edges[e].wall_distance)
            for e in WALL_EDGES
        ],
        (vents.normal_u, vents.normal_v, vents.tangential_u, vents.tangential_v)
        if vents.any
        else None,
        domain,
        discharges,
        exhausts,
        VARIANTS[config.turbulence.variant].c_mu,
        config.rho,
        config.mu / config.rho,
        WALL_LAW,
    )


def turbulence_state(core: Mapping[str, np.ndarray]) -> TurbulenceState:
    """A step of the core's, by name, as the reference's state, its arrays read-only."""
    arrays = {name: core[name] for name in ("k", "eps", "nu_t")}
    for array in arrays.values():
        array.setflags(write=False)
    return TurbulenceState(**arrays)


def turbulence_conditions(core: Mapping[str, np.ndarray]) -> TurbulenceConditions:
    """The core's conditions, by name, as the reference's."""
    return TurbulenceConditions(**core)


def positivity_error(failure: "_cfdcore.PositivityFailure") -> PositivityError:
    """The reference's error for a step that left k or eps not positive (REQ-S15).

    Parameters
    ----------
    failure : _cfdcore.PositivityFailure
        What the core raised.

    Returns
    -------
    PositivityError
        With the reference's message and its ``minimum``.
    """
    minimum = float(failure.minimum)
    return PositivityError(
        f"{failure.quantity} is not positive and finite at {failure.count} "
        f"non-SOLID cell(s), first at row {failure.row}, column {failure.column} "
        f"({np.float64(failure.value)!r}); least {minimum!r}",
        minimum,
    )


# ---- The stopping rule and the outer iteration (prompt 95) --------------------


def error_estimate_rule(
    velocity_scale: float,
    flux_scale: float,
    iteration_error_tol: float,
    mass_imbalance_tol: float,
    *,
    viscosity_condition: bool = False,
) -> "_cfdcore.StoppingRule2":
    """The core's rule from ``ErrorEstimateRule``'s own arguments.

    The window is RATE_WINDOW and the two versions are the reference's, handed in
    so that the core holds no copy of any of them.
    """
    import _cfdcore

    return _cfdcore.StoppingRule2(
        RATE_WINDOW,
        RULE_VERSION_WITHOUT_E,
        RULE_VERSION_WITH_E,
        velocity_scale,
        flux_scale,
        iteration_error_tol,
        mass_imbalance_tol,
        viscosity_condition,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class FlowModules:
    """The core's modules of a steady solve, built on one grid, which a ``Flow2`` calls."""

    grid: "_cfdcore.Grid2"
    predictor: "_cfdcore.MomentumPredictor2"
    corrector: "_cfdcore.PressureCorrector2"
    outlets: "_cfdcore.OutletFaces2"
    model: "_cfdcore.KEpsilon2 | None"
    walls: "_cfdcore.TurbulenceBoundary2 | None"


def flow_modules(
    mesh: Mesh,
    config: SimConfig,
    boundary: StaggeredBoundary,
    corrector: PressureCorrector,
) -> FlowModules:
    """The modules ``StaggeredSolver`` builds, the core's, on one grid (prompt 95).

    Parameters
    ----------
    mesh, config, boundary
        The solver's own arguments.
    corrector : PressureCorrector
        The reference solver's corrector, which holds the flux scale.

    Returns
    -------
    FlowModules
        The predictor, the corrector, the outlets and, with the turbulence
        section, the model and its boundary.
    """
    grid = core_grid(mesh)
    turbulent = config.turbulence is not None
    return FlowModules(
        grid,
        momentum_predictor(mesh, config, boundary, grid),
        pressure_corrector(mesh, config, boundary, corrector, grid),
        outlet_faces(mesh, boundary, grid),
        k_epsilon(mesh, config, boundary, grid) if turbulent else None,
        turbulence_boundary(mesh, config, boundary, grid) if turbulent else None,
    )


def flow_solver(
    modules: FlowModules,
    config: SimConfig,
    boundary: StaggeredBoundary,
    reference_velocity: float,
    flux_scale: float | None,
    mu_eff: np.ndarray | None = None,
) -> "_cfdcore.Flow2":
    """A ``_cfdcore.Flow2`` with the numbers ``StaggeredSolver`` forms (prompt 95).

    Parameters
    ----------
    modules : FlowModules
        The core's modules.
    config : SimConfig
        Supplies rho, mu, the turbulence section and the stopping keys.
    boundary : StaggeredBoundary
        Supplies the rule's velocity scale, the largest prescribed velocity.
    reference_velocity : float
        The solver's ``reference_velocity``.
    flux_scale : float or None
        The solver's ``flux_scale``; None under velocity_step.
    mu_eff : np.ndarray, optional
        The effective viscosity a fixed eddy viscosity makes
        (``_effective_viscosity``), held for the solve.

    Returns
    -------
    _cfdcore.Flow2
        The solve, its modules kept alive.

    Raises
    ------
    ValueError
        A fixed viscosity that is not finite or not positive in a non-SOLID cell,
        in ``predict``'s words (``mu_eff must be finite in every non-SOLID cell``),
        and the core's own refusal of what it cannot build from.
    """
    import _cfdcore

    if config.stopping_rule == ERROR_ESTIMATE:
        assert flux_scale is not None
        rule: tuple = (
            "error_estimate",
            RATE_WINDOW,
            RULE_VERSION_WITHOUT_E,
            RULE_VERSION_WITH_E,
            boundary.get_max_boundary_velocity(),
            flux_scale,
            config.iteration_error_tol,
            config.mass_imbalance_tol,
        )
    else:
        rule = ("velocity_step", config.convergence_tol)
    spec = config.turbulence
    try:
        return _cfdcore.Flow2(
            modules.grid,
            modules.predictor,
            modules.corrector,
            modules.outlets,
            modules.model,
            modules.walls,
            FLUID,
            config.mu,
            config.rho,
            config.mu / config.rho,
            1.0 if spec is None else spec.alpha_turbulence,
            max(reference_velocity, ZERO_SCALE),
            rule,
            mu_eff,
        )
    except ValueError as error:
        # The core reads a fixed viscosity when the solver is built; the reference
        # reads it at its first prediction, in the same words.
        text = str(error)
        if text.startswith("Flow2: mu_eff must be "):
            raise ValueError(text.removeprefix("Flow2: ")) from None
        raise


def flow_error(failure: Mapping[str, object]) -> Exception:
    """The exception the reference's solve raises where a core chunk failed.

    Parameters
    ----------
    failure : Mapping[str, object]
        A chunk's ``failure``: its type's name, its message and, for k or eps
        not positive, the bad cells' numbers.

    Returns
    -------
    Exception
        A PositivityError naming the outer iteration, as ``_turbulence_step``
        raises it; the pressure solve's breakdown as ``breakdown_error`` makes
        it; otherwise the ValueError of the reference's message.
    """
    kind = failure["type"]
    if kind == "PositivityError":
        minimum = float(failure["minimum"])  # type: ignore[arg-type]
        base = (
            f"{failure['quantity']} is not positive and finite at {failure['count']} "
            f"non-SOLID cell(s), first at row {failure['row']}, column "
            f"{failure['column']} ({np.float64(failure['value'])!r}); least {minimum!r}"
        )
        return PositivityError(
            f"outer iteration {failure['iteration']}: {base}", minimum
        )
    if kind == "ZeroDivisionError":
        return ZeroDivisionError("division by zero in the conjugate gradients")
    if kind == "LinAlgError":
        return np.linalg.LinAlgError("Eigenvalues did not converge")
    return ValueError(str(failure["message"]))


# ---- The scenario run and the steady solve on the core (prompts 91, 95, 96) ----


class CoreRun(_Run):
    """``_Run`` with the march stepped through the core (``solver.backend: core``).

    Parameters
    ----------
    config : SimConfig
        The run's configuration (a scenario section and a transport section).
    run : RunFile
        The run file.
    on_step : Callable[[MarchStep], None] or None
        Called after every step's writes, as the reference calls it.

    Notes
    -----
    ``monitor_reads_fields`` (class attribute, False) puts the monitor on the
    readings and channels the core returns per step
    (``AlertMonitor.update_from_readings``), so the march goes to the core in
    batches to the next frame, checkpoint or interval end, or one step at a
    time when ``on_step`` is given. True keeps it on the fields, as the
    reference hands them: they are read from the core after every step and
    handed to ``AlertMonitor.update``, and the core's readings and channels are
    checked against the monitor's bit for bit. ``core_calls`` counts the core's
    ``march`` calls.
    """

    monitor_reads_fields: bool = False

    def __init__(
        self,
        config: SimConfig,
        run: RunFile,
        on_step: Callable[[MarchStep], None] | None,
    ) -> None:
        require_core()
        super().__init__(config, run, on_step)
        recovery = None
        if self.tracker is not None:
            spec = config.scenario
            assert spec is not None
            initial = self.manager.initial_fields()
            recovery = recovery_keys(
                config.class_thresholds("the clean-out's recovery"),
                [float(initial[k][self.live][0]) for k in range(self.n_classes)],
                spec.hold_time,
            )
        self.core = core_march(
            config, self.mesh, self.cells, [onset for _, onset in self.events], recovery
        )
        self.core_calls = 0

    def _check_interval(self, plan: IntervalPlan) -> None:
        """Refuse an interval whose transport inputs the core was not built with."""
        theirs = self.intervals[plan.index].config
        if (
            theirs.transport != self.config.transport
            or theirs.particle_sizes != self.config.particle_sizes
            or theirs.t_end != self.config.t_end
        ):
            raise ValueError(
                f"interval {plan.index}'s transport keys, particles or t_end are not "
                "the run's, which the core was built with"
            )

    @staticmethod
    def _stop_of(
        core: "_cfdcore.March2", result: Mapping[str, object], rows: int
    ) -> str:
        """The reason a core march returned, checked against the core's own state.

        The reason is the core's claim, so the driver reads it and refuses one
        the state does not bear out: the interval's end when no step is left and
        a count when some are, a hold the core holds, a breach that carries its
        record. A march that took no step must have held or breached, or the
        driver would ask it again for the same.

        Parameters
        ----------
        core : _cfdcore.March2
            The run, after the march.
        result : Mapping[str, object]
            The march's returned dict.
        rows : int
            The steps it returned.

        Returns
        -------
        str
            ``result["stopped"]``.

        Raises
        ------
        AssertionError
            If the reason is not one the core's state allows.
        """
        stopped = result["stopped"]
        bears_out = {
            "interval": core.remaining == 0,
            "count": core.remaining > 0,
            "held": core.held(),
            "breach": result["breach"] is not None,
        }
        if not isinstance(stopped, str) or not bears_out.get(stopped, False):
            raise AssertionError(
                f"the core's march stopped as {stopped!r} with {core.remaining} "
                f"steps left, held {core.held()}"
            )
        if (result["breach"] is not None) != (stopped == "breach"):
            raise AssertionError(f"a breach record with the stop {stopped!r}")
        if rows == 0 and stopped not in ("held", "breach"):
            raise AssertionError(
                f"a march that returned no step stopped as {stopped!r}, "
                "neither held nor breached"
            )
        return stopped

    def _observe_row(
        self,
        step: int,
        t: float,
        result: Mapping[str, np.ndarray],
        r: int,
        fields: Mapping[int, np.ndarray] | None,
        pending: list[tuple[int, float, np.ndarray, np.ndarray]],
    ) -> tuple[Alert, ...]:
        """The monitor's update after row R of a march, on the fields or the readings."""
        if self.monitor is None:
            return ()
        if fields is None:
            alerts = self.monitor.update_from_readings(
                step, t, result["readings"][r], result["channels"][r]
            )
            return self.noted(alerts, step, t, pending)
        alerts = self.observe(step, t, fields, pending)
        # The core's readings are the monitor's, read from the same fields.
        if (
            self.monitor.readings.tobytes() != result["readings"][r].tobytes()
            or self.monitor.channel_readings.tobytes()
            != result["channels"][r].tobytes()
        ):
            raise AssertionError(
                f"the core's readings at step {step} are not the monitor's"
            )
        return alerts

    def march(
        self,
        plans: list[IntervalPlan],
        start: tuple[int, int],
        fields: dict[int, np.ndarray],
        budgets: Sequence[MassBudget],
        step: int,
        t: float,
    ) -> tuple[dict[int, np.ndarray], tuple[MassBudget, ...], int, float, bool]:
        """Step every class from (plan, step within it) to the end or the hold, in the core.

        Returns the fields, budgets, step count and time at the end, and
        whether the hold stopped it, as ``_Run.march`` does.
        """
        core = self.core
        config = self.config
        output, cadence = config.output_interval, config.checkpoint_interval
        restore_march(core, fields, budgets, step, t, self.arrival, self.tracker)
        pending: list[tuple[int, float, np.ndarray, np.ndarray]] = []
        held = core.held()
        current = tuple(budgets)
        every_step = self.on_step is not None or (
            self.monitor is not None and self.monitor_reads_fields
        )
        first_plan, first_i = start
        for p in range(first_plan, len(plans)):
            if held:
                break
            plan = plans[p]
            self._check_interval(plan)
            interval = self.intervals[plan.index]
            rates = self.manager.sources_at(interval.t_start)
            set_interval(
                core,
                self.manager.conditions_for(interval),
                self.faces(plan.fingerprint),
                self.states[plan.fingerprint].nu_t,
                [rates[k] if rates[k].any() else None for k in range(self.n_classes)],
                plan,
                first_i if p == first_plan else 0,
            )
            while core.remaining > 0 and not held:
                n = core.remaining
                if every_step:
                    n = 1
                else:
                    n = min(n, output - step % output)
                    if cadence is not None:
                        n = min(n, cadence - step % cadence)
                result = core.march(n)
                self.core_calls += 1
                rows = len(result["step"])
                stopped = self._stop_of(core, result, rows)
                for r in range(rows):
                    step, t = int(result["step"][r]), float(result["t"][r])
                    current = tuple(budgets_of_row(result["budgets"][r]))
                    held = stopped == "held" and r == rows - 1
                    last = held or (
                        p == len(plans) - 1 and core.remaining == 0 and r == rows - 1
                    )
                    frame = step % output == 0 or last
                    checkpoint = cadence is not None and (step % cadence == 0 or last)
                    end_of_call = r == rows - 1
                    if (frame or checkpoint or every_step) and not end_of_call:
                        raise AssertionError(
                            f"step {step} needs the fields but is not a march's last"
                        )
                    fields_now = None
                    if end_of_call and (frame or checkpoint or every_step):
                        fields_now = unstacked(core.class_fields())
                        fields = fields_now
                    monitor_fields = fields_now if self.monitor_reads_fields else None
                    alerts = self._observe_row(
                        step, t, result, r, monitor_fields, pending
                    )
                    if frame or checkpoint:
                        sync_trackers(core, self.arrival, self.tracker)
                    if frame:
                        self.run.write_frame(step, t, fields, current)
                        self._flush(pending)
                    if checkpoint:
                        self._flush(pending)
                        self.run.write_arrival(self.arrival_record(t))
                        self.run.checkpoint(
                            step,
                            t,
                            fields,
                            current,
                            plan.fingerprint,
                            recovery=self.carried(),
                            plan_start=plan.t_start,
                        )
                    if self.on_step is not None:
                        self.on_step(
                            MarchStep(
                                step,
                                t,
                                plan.dt,
                                plan.index,
                                fields,
                                _copies(current),
                                alerts,
                            )
                        )
                error = breach_error(result)
                if error is not None:
                    raise error
                # Read once per call: a march that took no step and held ends the loop.
                held = stopped == "held"
        fields = unstacked(core.class_fields())
        current = tuple(budgets_of(core))
        sync_trackers(core, self.arrival, self.tracker)
        self._flush(pending)
        return fields, _copies(current), int(core.step), float(core.t), held


# ---- The steady solve (prompt 95) ---------------------------------------------


class _CoreWindow:
    """The core's window as ``StaggeredSolver._checkpoint`` reads a rule's."""

    def __init__(self, core: "_cfdcore.Flow2") -> None:
        self._core = core

    def window(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The velocity and the viscosity steps, oldest first."""
        return self._core.window()


def _state_of(fields: Mapping[str, np.ndarray]) -> TurbulenceState | None:
    """The core's k, eps and nu_t as the reference's state, read-only; None when laminar."""
    if "k" not in fields:
        return None
    arrays = {name: fields[name] for name in ("k", "eps", "nu_t")}
    for array in arrays.values():
        array.setflags(write=False)
    return TurbulenceState(**arrays)


class CoreSolver(StaggeredSolver):
    """``StaggeredSolver`` with ``solve_steady``'s outer iteration in the core.

    The set-up, the refusals, the warm start's and the resume's checks, the
    checkpoints, the log lines and the end of a solve are the reference's own code:
    the reference's methods where they are separate (``_effective_viscosity``,
    ``_warm``, ``_resumed``, ``_new_rule``, ``_checkpoint``, ``_finite``) and the
    loop's lines here otherwise. The outer iterations run in a ``_cfdcore.Flow2``
    in chunks: to the next checkpoint (an outer iteration whose
    ``(iteration + 1) % checkpoint_interval == 0``) when ``on_checkpoint`` is given,
    one at a time when ``on_iteration`` is, else to the cap.

    Parameters
    ----------
    mesh, config, boundary
        As ``StaggeredSolver`` takes them.

    Attributes
    ----------
    rule_readings : dict[str, list]
        The last solve's records per outer iteration, the resumed solve's from its
        resume: the velocity and the viscosity steps, the rule's estimates and
        scales, and the corrections' iterations and products.
    core_seconds : dict[str, float]
        The last solve's seconds by the core's stages, the rule's among them.
    core_calls : int
        The last solve's chunks.
    """

    def __init__(
        self, mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary
    ) -> None:
        require_core()
        super().__init__(mesh, config, boundary)
        self._config = config
        self._modules = flow_modules(mesh, config, boundary, self._corrector)
        self.rule_readings: dict[str, list] = {}
        self.core_seconds: dict[str, float] = {}
        self.core_calls = 0

    def solve_transient(self, *args: object, **kwargs: object) -> object:
        """Refused: tier 2's true-time step stays on the reference (ADR-017 F).

        Raises
        ------
        NotImplementedError
            Always. ``SimConfig`` refuses ``solver.backend: core`` with a
            transient section, so a run never reaches this; a direct call would
            otherwise run the reference's own loop under the core's name.
        """
        raise NotImplementedError(
            "the core carries the steady solve; tier 2's true-time step "
            "(solve_transient) stays on the reference (ADR-017 F)"
        )

    def solve_steady(
        self,
        on_iteration: Callable[[IterationState], None] | None = None,
        *,
        eddy_viscosity: np.ndarray | None = None,
        initial: FlowState | None = None,
        resume: SteadyCheckpoint | None = None,
        on_checkpoint: Callable[[SteadyCheckpoint], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``StaggeredSolver.solve_steady``, its outer iterations in the core.

        Parameters and returns as the reference's; so are its refusals, which are
        its own code.
        """
        model = self._model
        if model is not None and eddy_viscosity is not None:
            raise ValueError(
                "eddy_viscosity is refused with the turbulence section present: the "
                "k-epsilon model is the solve's one source of nu_t"
            )
        if initial is not None and resume is not None:
            raise ValueError(
                "initial and resume are exclusive: a warm start begins a solve, a "
                "resume continues one"
            )
        mu_eff = (
            None
            if eddy_viscosity is None
            else self._effective_viscosity(eddy_viscosity)
        )
        digest = None if eddy_viscosity is None else _digest(eddy_viscosity)
        start = 0
        state: TurbulenceState | None = None
        if resume is not None:
            u, v, p, state = self._resumed(resume, digest)
            start = resume.state.outer
        elif initial is not None:
            u, v, p, state = self._warm(initial)
        else:
            if model is not None:
                state = model.initial(*self._initial_turbulence)
            u, v, p = allocate_fields(self._mesh)
        self.turbulence_state = None
        self._boundary.apply_normal_velocity(u, v)

        self.residual_history = [] if resume is None else list(resume.residual_history)
        self.stage_seconds = self._zero_stage_seconds()
        self.last_pressure_iterations = 0
        self.pressure_cap_hits = 0 if resume is None else resume.pressure_cap_hits
        self.converged, self.stop_reason = False, None
        rule = self._new_rule()
        if rule is not None and resume is not None:
            # The reference's refusals of the window, before the core is seeded.
            rule.restore(resume.velocity_steps, resume.viscosity_steps)

        core = flow_solver(
            self._modules,
            self._config,
            self._boundary,
            self.reference_velocity,
            self.flux_scale,
            mu_eff,
        )
        if state is None:
            core.flow_state(u, v, p)
        else:
            core.flow_state(u, v, p, state.k, state.eps, state.nu_t)
        if resume is not None:
            core.restore(resume.velocity_steps, resume.viscosity_steps)
        window = _CoreWindow(core)
        self.rule_readings = {name: [] for name in _READINGS}
        self.core_seconds = dict.fromkeys(_CORE_STAGES, 0.0)
        self.core_calls = 0

        interval = self._checkpoint_interval
        cadence = interval if on_checkpoint is not None else None
        iteration = start
        while iteration < self._max_simple_iter:
            n = self._max_simple_iter - iteration
            if on_iteration is not None:
                n = 1
            elif cadence is not None:
                n = min(n, cadence - iteration % cadence)
            chunk = core.solve_steady(n, iteration)
            self.core_calls += 1
            self._add_seconds(chunk["stage_seconds"])
            rows = int(chunk["outer"])
            stopped = bool(chunk["stopped"])
            for r in range(rows):
                at = iteration + r
                stop = stopped and r == rows - 1
                self._record(chunk, r, at)
                if on_iteration is not None:
                    self._notify(
                        on_iteration,
                        core,
                        at,
                        float(chunk["residual"][r]),
                        int(chunk["pressure_iterations"][r]),
                        int(chunk["pressure_products"][r]),
                    )
                residual = self.residual_history[-1]
                if at % 50 == 0 or stop:
                    solver_logger.info("SIMPLE iter %4d: residual = %.6e", at, residual)
                if stop:
                    self.converged = True
                    self.stop_reason = _STOP_REASONS[self._stopping_rule]
                    solver_logger.info("Converged at iteration %d", at)
            failure = chunk["failure"]
            if failure is not None:
                failing = chunk["failing_record"]
                if failing is not None:
                    self._fail_after_correction(failure, failing, core, on_iteration)
                raise flow_error(failure)
            iteration += rows
            if self.converged:
                break
            if (
                cadence is not None
                and interval is not None
                and rows > 0
                and iteration % interval == 0
            ):
                fields = core.flow_fields()
                now = _state_of(fields)
                if self._finite(fields["u"], fields["v"], fields["p"], now):
                    kept = window if rule is not None else None
                    on_checkpoint(
                        self._checkpoint(
                            fields["u"],
                            fields["v"],
                            fields["p"],
                            now,
                            kept,
                            digest,
                            None,
                        )
                    )

        if not self.converged:
            self.stop_reason = CAP_STOP
            solver_logger.warning("Not converged: stopped at max_simple_iter")
        fields = core.flow_fields()
        u, v, p = fields["u"], fields["v"], fields["p"]
        state = _state_of(fields)
        u_c, v_c = to_cell_centers(u, v)
        self.last_mass_imbalance = self._corrector.mass_imbalance(u, v)
        self.face_velocities = FaceVelocities.copy_of(u, v)
        self.turbulence_state = state
        p = np.ascontiguousarray(p)
        self.flow_state = self.steady_checkpoint = None
        if not self._finite(u, v, p, state):
            solver_logger.warning(
                "Fields are not finite after %d outer iterations: no flow_state "
                "or steady_checkpoint is kept",
                len(self.residual_history),
            )
            return u_c, v_c, p
        self.steady_checkpoint = self._checkpoint(
            u,
            v,
            p,
            state,
            window if rule is not None else None,
            digest,
            self.stop_reason,
        )
        self.flow_state = self.steady_checkpoint.state
        return u_c, v_c, p

    def _fail_after_correction(
        self,
        failure: Mapping[str, object],
        failing: Mapping[str, object],
        core: "_cfdcore.Flow2",
        on_iteration: Callable[[IterationState], None] | None,
    ) -> None:
        """Record a failing iteration as the reference has by the time it raises.

        The reference has counted the correction and its cap, and logged the
        k-epsilon step's warnings. A refused scale of condition (e) is raised by
        the rule, after the residual is appended and ``on_iteration`` called, so
        those are done here.

        Parameters
        ----------
        failure : Mapping[str, object]
            The chunk's failure, naming the outer iteration.
        failing : Mapping[str, object]
            The chunk's record of that iteration: its correction's count, products
            and cap, the step's fallbacks and convergence, and the residual when the
            scale was refused (None otherwise).
        core : _cfdcore.Flow2
            The solve, whose fields are the failing iteration's.
        on_iteration : Callable[[IterationState], None] or None
            The caller's callback.
        """
        at = int(failure["iteration"])
        iterations = int(failing["pressure_iterations"])
        self._count_correction(at, iterations, bool(failing["reached_cap"]))
        self._warn_of_the_step(
            int(failing["k_fallbacks"]),
            int(failing["eps_fallbacks"]),
            bool(failing["solves_converged"]),
        )
        residual = failing["residual"]
        if residual is None:
            return
        self.residual_history.append(float(residual))
        if on_iteration is not None:
            self._notify(
                on_iteration,
                core,
                at,
                float(residual),
                iterations,
                int(failing["pressure_products"]),
            )

    def _notify(
        self,
        on_iteration: Callable[[IterationState], None],
        core: "_cfdcore.Flow2",
        iteration: int,
        residual: float,
        pressure_iterations: int,
        pressure_products: int,
    ) -> None:
        """Call ``on_iteration`` with the core's fields as the reference's loop does."""
        fields = core.flow_fields()
        u_c, v_c = to_cell_centers(fields["u"], fields["v"])
        on_iteration(
            IterationState(
                iteration=iteration,
                residual=residual,
                pressure_iterations=pressure_iterations,
                u=u_c,
                v=v_c,
                p=fields["p"],
                pressure_products=pressure_products,
            )
        )

    def _warn_of_the_step(
        self, k_fallbacks: int, eps_fallbacks: int, converged: bool
    ) -> None:
        """The k-epsilon step's two warnings, as ``KEpsilonModel.step`` logs them.

        The reference logs them before it tests the step's signs, so an iteration
        that then fails the test has logged them too. A laminar solve and an
        iteration that ended before its step hand in no fallback and a converged
        pair of solves, which log nothing.
        """
        if k_fallbacks or eps_fallbacks:
            turbulence_logger.warning(
                "CG k or eps solve fell back to Jacobi (k %d, eps %d)",
                k_fallbacks,
                eps_fallbacks,
            )
        if not converged:
            spec = self._config.turbulence
            assert spec is not None
            turbulence_logger.warning(
                "implicit k or eps solve stopped at max_iter %d CG iterations",
                spec.max_iter,
            )

    def _count_correction(self, iteration: int, iterations: int, capped: bool) -> None:
        """A correction's count and cap, with the reference's warning at the first cap."""
        self.last_pressure_iterations = iterations
        if capped:
            self.pressure_cap_hits += 1
            if self.pressure_cap_hits == 1:
                solver_logger.warning(
                    "SIMPLE iter %d: pressure correction stopped at "
                    "max_pressure_iter (%d) before its residual met the stop",
                    iteration,
                    iterations,
                )

    def _record(self, chunk: Mapping[str, np.ndarray], r: int, iteration: int) -> None:
        """Row R of a chunk into the solver's records, in the reference's order."""
        self._count_correction(
            iteration,
            int(chunk["pressure_iterations"][r]),
            bool(chunk["reached_cap"][r]),
        )
        self._warn_of_the_step(
            int(chunk["k_fallbacks"][r]),
            int(chunk["eps_fallbacks"][r]),
            bool(chunk["solves_converged"][r]),
        )
        self.residual_history.append(float(chunk["residual"][r]))
        for name in _READINGS:
            self.rule_readings[name].append(chunk[name][r].item())

    def _add_seconds(self, seconds: Mapping[str, float]) -> None:
        """A chunk's seconds into the reference's stages and the core's."""
        for name, value in seconds.items():
            self.core_seconds[name] += value
            if name in self.stage_seconds:
                self.stage_seconds[name] += value


# The records a chunk returns per outer iteration, kept in ``rule_readings``.
_READINGS = (
    "step",
    "viscosity_step",
    "estimate",
    "viscosity_estimate",
    "nu_scale",
    "pressure_iterations",
    "pressure_products",
)
# The core's stages, the reference's four and the rule's.
_CORE_STAGES = ("momentum", "pressure", "turbulence", "correct", "stopping")


def steady_solver(
    mesh: Mesh,
    config: SimConfig,
    boundary: StaggeredBoundary,
    *,
    reference: type[StaggeredSolver] = StaggeredSolver,
) -> StaggeredSolver:
    """The steady solver of a configuration's backend (ADR-017 E).

    The one factory ``time_integration`` and ``scripts/benchmark.py`` build a
    steady solver with, so a scenario run and a harness row run on the backend
    ``solver.backend`` names.

    Parameters
    ----------
    mesh, config, boundary
        As ``StaggeredSolver`` takes them.
    reference : type[StaggeredSolver]
        The class built under ``reference``: ``StaggeredSolver`` unless the
        caller names its own, which ``time_integration`` does with the name its
        module holds, so that a test's substitute for it is built.

    Returns
    -------
    StaggeredSolver
        A ``CoreSolver`` under ``core``, else ``reference``'s instance.
    """
    if config.backend == CORE_BACKEND:
        return CoreSolver(mesh, config, boundary)
    return reference(mesh, config, boundary)
