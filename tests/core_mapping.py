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
``RecoveryTracker.restore``, whose checks then read the core's state). Prompt 92
adds the momentum predictor's construction, and prompt 93 the pressure correction's:
its construction, the multigrid's keys, its result as ``PressureCorrection`` and its
breakdown as the error the reference's own arithmetic raises. Prompt 94 adds the turbulence
model's and the wall functions': their construction (which face admits air, read from the
shared coverage as ``TurbulenceBoundary`` reads it, with its refusal of a segment that states
no turbulence), their results as ``TurbulenceState`` and ``TurbulenceConditions``, and a
positivity failure as ``PositivityError``. Prompt 95 adds the stopping rule's and the
outer iteration's: a ``_cfdcore.StoppingRule2`` from ``ErrorEstimateRule``'s own
arguments, the modules of a steady solve on one grid, a ``_cfdcore.Flow2`` from a
``StaggeredSolver``'s set-up, and a chunk's failure as the exception the reference's
solve raises there.
"""

import dataclasses
from collections.abc import Mapping, Sequence
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
from src.config import ERROR_ESTIMATE, UPWIND, BoundarySpec, SimConfig
from src.mesh import FLUID, SOLID, Mesh
from src.monitor import COUNTING_STATE, ArrivalTracker
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
from src.solver_transport import DEPOSIT_SURFACES, ConservationBreachError, MassBudget
from src.staggered import FaceVelocities, edge_cell_inputs
from src.stopping import RATE_WINDOW
from src.time_integration import (
    DEFINED_RTOL,
    HUNDRED_TO_ONE,
    IntervalPlan,
    RecoveryTracker,
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
    """The core's rule from ``ErrorEstimateRule``'s own arguments, its window RATE_WINDOW."""
    import _cfdcore

    return _cfdcore.StoppingRule2(
        RATE_WINDOW,
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
    """
    import _cfdcore

    if config.stopping_rule == ERROR_ESTIMATE:
        assert flux_scale is not None
        rule: tuple = (
            "error_estimate",
            RATE_WINDOW,
            boundary.get_max_boundary_velocity(),
            flux_scale,
            config.iteration_error_tol,
            config.mass_imbalance_tol,
        )
    else:
        rule = ("velocity_step", config.convergence_tol)
    spec = config.turbulence
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
