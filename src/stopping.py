"""Stopping rule for the steady SIMPLE loop: estimated iteration error and continuity.

A small velocity step between outer iterations is not a small error. While
the iteration contracts geometrically at a rate rho, the error left after a
step s is about s rho / (1 - rho), and rho approaches 1 as the grid is
refined, so a fixed bound on the step leaves more error on every finer grid.
On an open domain part of the error is a drift of the through-flow set by
the per-cell mass imbalance, which the step does not see. Both are measured
in docs/reports/stopping_rule_evidence.md, sections 4 and 5.

ErrorEstimateRule stops when (a) the estimated iteration error over a
physical velocity scale, (b) the worst absolute per-cell mass imbalance,
(c) the summed absolute imbalance over the through-flow and (d) the signed
imbalance summed over the domain, in absolute value, are all below their
tolerances. (b) alone lets the through-flow drift by up to the per-cell
bound times the cells upstream, more on every finer grid (section 9). The
flux through any cross-section differs from the inflow by at most the summed
imbalance on one side of it, so (c) bounds that drift on any grid. The rule
knows nothing of the solver: it is fed one outer iteration at a time, so it
can be tested on synthetic histories.

(b) and (d) are ECR-001 criterion 6, its per-cell and its domain-sum clause,
at the bound it names; (a) and (c) are no clause of it. The signed sum is the
net mass flux out of the domain: (c) holds it to iteration_error_tol of
flux_scale, (d) to criterion 6's absolute bound (section 10).

With the turbulence model on the iterate carries k and eps as well, and a slow
mode in eps moves the eddy viscosity, and through it the velocity, more
slowly than the velocity's own rate shows. Condition (e) (ADR-012 E, ECR-002
step 6) is (a)'s estimate on the eddy viscosity: the largest change of nu_t
over the outer iteration, the same rate fit, over ``nu_scale``, which the
solver passes with each step: the molecular nu plus the median of nu_t over
the FLUID cells of the iterate just computed (``viscosity_scale``). Until
2026-10-11 the scale was the molecular nu plus the largest inlet eddy
viscosity, fixed per solve, which on a room with weak inlet turbulence is
hundreds of times smaller than what the room makes, so (e) could not hold
(ADR-012 E, amended; docs/reports/ecr002_val017_annex20.md, section 5.2).
The rule's version (``ErrorEstimateRule.version``) is 3 without (e) and 5
with it (4 was (e) on the inlet scale), and the solver reports it, so a
laminar solve, which applies version 3's conditions, still records 3.

IterationState, the snapshot a solver hands its on_iteration callback once
per outer iteration, is defined here as well: the module that defines when
an outer iteration ends also defines what each iteration reports.
"""

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Trailing window of the rate fit, in outer iterations. While the iteration is
# geometric the length hardly matters: from 1e-5 to 1e-8, halving or doubling
# it moved no estimate-to-error ratio by more than 0.015, except 0.095 on
# VAL-001 80x40 near 1e-5 (evidence report, section 4). Off that regime a long
# window averages across a change of rate, so it must be short against a
# decade. On the fastest case, VAL-001 40x20 at about 71 outer iterations per
# decade, 100 spans 1.4 decades and is full by outer iteration 142, the
# residual's 1e-5, where 200 was not.
RATE_WINDOW = 100
# Which conditions update applies: 1 was (a) and (b), 2 added (c), 3 added (d),
# 4 added (e), which applies only with the turbulence model on, scaled by the
# inlet's eddy viscosity, and 5 scales (e) by the room's median. A rule's version
# is ErrorEstimateRule.version; the scripts read it through the solver and store
# it with saved solves, since a new condition or scale leaves the tolerances alike.
RULE_VERSION_WITHOUT_E = 3
RULE_VERSION_WITH_E = 5


def viscosity_scale(nu: float, nu_t: np.ndarray, fluid: np.ndarray) -> float:
    """Condition (e)'s scale: the molecular nu plus the median of nu_t over FLUID cells.

    Parameters
    ----------
    nu : float
        Molecular kinematic viscosity, m^2/s.
    nu_t : np.ndarray
        Kinematic eddy viscosity of the iterate just computed, m^2/s, shape
        [ny, nx].
    fluid : np.ndarray
        Boolean mask of the FLUID cells, the shape of ``nu_t``; at least one
        True.

    Returns
    -------
    float
        ``nu + median(nu_t[fluid])``, m^2/s.

    Notes
    -----
    The median and not the largest value: one cell's overshoot (the
    stagnation anomaly of ADR-012 A) would move a maximum and loosen (e)
    everywhere, and the median ignores single cells. It moves with the
    iterate but converges with it, so at the stop it is the converged
    field's (ADR-012 E, amended 2026-10-11).
    """
    return nu + float(np.median(nu_t[fluid]))


def _check_positive(name: str, value: float) -> None:
    """Raise ValueError naming ``name`` unless ``value`` is positive and finite."""
    # bool is an int; True would pass as 1.0.
    if isinstance(value, bool) or not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {value}")


# eq=False because the default __eq__ compares the array fields elementwise
# and raises on the ambiguous truth value, and the __hash__ that frozen=True
# would then generate raises on an array. Identity is the only comparison that
# makes sense for a snapshot of the solver's working fields.
@dataclass(frozen=True, eq=False)
class IterationState:
    """Snapshot handed to the solve_steady callback after each SIMPLE iteration.

    The arrays are the solver's working fields, not copies. A callback
    that needs to keep them must copy them and must never modify them.

    Parameters
    ----------
    iteration : int
        Zero-based SIMPLE iteration index.
    residual : float
        Scaled velocity-change residual for this iteration.
    pressure_iterations : int
        Conjugate gradient iterations performed by the pressure correction
        this iteration.
    u, v, p : np.ndarray
        Current velocity and pressure fields, each shape [ny, nx].
    pressure_products : int
        Products with the pressure operator that correction formed: its
        iterations plus the true-residual checks. The harness counts its
        work in them.

    Notes
    -----
    Moved here from src/solver_ns.py on 2026-10-02 (PR 29), fields unchanged.
    The count field was renamed from pressure_sweeps on 2026-10-06 (ECR-003
    step 1), when the Jacobi sweep gave way to conjugate gradients.
    pressure_products was added last the same day (review 37 S5), so every
    other field keeps its position.
    """

    iteration: int
    residual: float
    pressure_iterations: int
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    pressure_products: int


@dataclass(frozen=True, kw_only=True, slots=True)
class ImbalanceSummary:
    """One field's per-cell mass imbalance read three ways, kg/s per unit depth.

    Keyword-only, so no reading can be passed in another's place.

    Attributes
    ----------
    worst : float
        Largest absolute per-cell imbalance.
    absolute_sum : float
        Sum of the absolute per-cell imbalances.
    signed_sum : float
        Sum of the signed per-cell imbalances: net mass flux out of the domain.
    """

    worst: float
    absolute_sum: float
    signed_sum: float


class ErrorEstimateRule:
    """Stop on the estimated iteration error and on continuity, per cell and summed.

    Condition (a): ``step * rho_hat / (1 - rho_hat) / velocity_scale`` is below
    ``iteration_error_tol``, with rho_hat the exp of the least-squares slope of
    log(step) over the last RATE_WINDOW steps. Condition (b): the worst
    absolute per-cell mass imbalance is below ``mass_imbalance_tol``, ECR-001
    criterion 6's per-cell clause. Condition (c): the summed absolute
    imbalance over ``flux_scale`` is below ``iteration_error_tol``, the
    tolerance of (a), since both bound a velocity error relative to its
    scale. Condition (d): the absolute value of the signed imbalance summed
    over the domain is below ``mass_imbalance_tol``, criterion 6's domain-sum
    clause. Condition (e), only with ``viscosity_condition``: ``viscosity_step
    * rho_hat_nu / (1 - rho_hat_nu) / nu_scale`` is below
    ``iteration_error_tol``, rho_hat_nu fitted to the viscosity steps as
    rho_hat is to the velocity steps and nu_scale the one passed with the
    step. The imbalance is asked for only when (a) and, with it on, (e) hold.

    Parameters
    ----------
    velocity_scale : float
        Physical velocity the estimate is divided by, m/s: the largest
        prescribed boundary velocity. Positive and finite.
    flux_scale : float
        Mass flux (c) is relative to, kg/s per unit depth: rho times the
        inflow, or on a closed domain rho times velocity_scale times the
        longer side. A flux through the domain, not one through a cell: a
        scale that moved with the grid would move (c) with it. Positive and
        finite.
    iteration_error_tol : float
        Bound on the estimate over velocity_scale, dimensionless. Positive
        and finite.
    mass_imbalance_tol : float
        Bound on the worst absolute per-cell imbalance and on the absolute
        signed domain sum, kg/s per unit depth
        (PressureCorrector.mass_imbalance). Positive and finite.
    viscosity_condition : bool, optional
        True applies condition (e), and every update then takes a viscosity
        step and its scale; False, the default, leaves (e) off.

    Attributes
    ----------
    estimate_history : list[float]
        The estimate over velocity_scale after each step. ``inf`` where there
        is none: fewer than RATE_WINDOW steps, a step in the window that is
        zero or not finite, or rho_hat not strictly between 0 and 1. A stall
        therefore never reads as convergence.
    viscosity_estimate_history : list[float]
        (e)'s left side after each step, ``inf`` likewise; empty with (e) off.
    viscosity_scale_history : list[float]
        The nu_scale passed with each step, m^2/s; empty with (e) off.

    Raises
    ------
    ValueError
        If any scale or tolerance is not positive and finite.
    TypeError
        If ``viscosity_condition`` is not a bool.
    """

    def __init__(
        self,
        velocity_scale: float,
        flux_scale: float,
        iteration_error_tol: float,
        mass_imbalance_tol: float,
        *,
        viscosity_condition: bool = False,
    ) -> None:
        checked = [
            ("velocity_scale", velocity_scale),
            ("flux_scale", flux_scale),
            ("iteration_error_tol", iteration_error_tol),
            ("mass_imbalance_tol", mass_imbalance_tol),
        ]
        for name, value in checked:
            _check_positive(name, value)
        if not isinstance(viscosity_condition, bool):
            raise TypeError(
                f"viscosity_condition must be a bool, got {type(viscosity_condition).__name__}"
            )
        self._scale = velocity_scale
        self._flux_scale = flux_scale
        self._error_tol = iteration_error_tol
        self._imbalance_tol = mass_imbalance_tol
        self._viscosity_condition = viscosity_condition
        self._steps: deque[float] = deque(maxlen=RATE_WINDOW)
        self._nu_steps: deque[float] = deque(maxlen=RATE_WINDOW)
        self.estimate_history: list[float] = []
        self.viscosity_estimate_history: list[float] = []
        self.viscosity_scale_history: list[float] = []

    @property
    def version(self) -> int:
        """Which conditions update applies: 3 without (e), 5 with it."""
        if self._viscosity_condition:
            return RULE_VERSION_WITH_E
        return RULE_VERSION_WITHOUT_E

    def window(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The steps the rate fits read, oldest first: velocity, then viscosity.

        Returns
        -------
        tuple[tuple[float, ...], tuple[float, ...]]
            The last RATE_WINDOW velocity steps (m/s) and, with (e) on, the
            last RATE_WINDOW viscosity steps (m^2/s); the second is empty
            with (e) off. With ``restore`` on a new rule they continue a
            solve's stop decisions bitwise (ADR-014 D's steady checkpoint).
        """
        return tuple(self._steps), tuple(self._nu_steps)

    def restore(
        self, steps: tuple[float, ...], viscosity_steps: tuple[float, ...] = ()
    ) -> None:
        """Seed a rule that has taken no step with a window another rule kept.

        Parameters
        ----------
        steps : tuple[float, ...]
            Velocity steps, m/s, oldest first, at most RATE_WINDOW of them.
        viscosity_steps : tuple[float, ...], optional
            Viscosity steps, m^2/s, likewise; given exactly when (e) is on
            and ``steps`` is not empty, and as many as ``steps``.

        Raises
        ------
        ValueError
            If the rule has already taken a step, a window is longer than
            RATE_WINDOW, a step is not a finite non-negative number, or the
            viscosity steps do not match the condition and the velocity steps.
        """
        if self._steps or self.estimate_history:
            raise ValueError("restore seeds a new rule; this one has taken steps")
        windows = (("steps", steps), ("viscosity_steps", viscosity_steps))
        for name, values in windows:
            if len(values) > RATE_WINDOW:
                raise ValueError(
                    f"{name} holds {len(values)} steps, more than RATE_WINDOW "
                    f"({RATE_WINDOW})"
                )
            for value in values:
                if isinstance(value, bool) or not isinstance(value, float):
                    raise ValueError(f"{name} must hold floats, got {value!r}")
                if not (math.isfinite(value) and value >= 0.0):
                    raise ValueError(f"{name} must be finite and non-negative")
        expected = len(steps) if self._viscosity_condition else 0
        if len(viscosity_steps) != expected:
            raise ValueError(
                f"viscosity_steps must hold {expected} steps with condition (e) "
                f"{'on' if self._viscosity_condition else 'off'}, got "
                f"{len(viscosity_steps)}"
            )
        self._steps.extend(steps)
        self._nu_steps.extend(viscosity_steps)

    @staticmethod
    def _estimate_of(steps: deque[float], scale: float) -> float:
        """The estimate of one history over its scale, or inf when there is none."""
        if len(steps) < RATE_WINDOW:
            return math.inf
        values = np.array(steps)
        if not np.all(np.isfinite(values) & (values > 0.0)):
            return math.inf
        x = np.arange(values.size) - (values.size - 1) / 2.0
        rho = math.exp(float(x @ np.log(values)) / float(x @ x))
        if not 0.0 < rho < 1.0:
            return math.inf
        return float(values[-1]) * rho / (1.0 - rho) / scale

    def update(
        self,
        step: float,
        imbalance: Callable[[], ImbalanceSummary],
        viscosity_step: float | None = None,
        nu_scale: float | None = None,
    ) -> bool:
        """Record one outer iteration and answer whether the solve has converged.

        Parameters
        ----------
        step : float
            Largest change of the velocity over the outer iteration, m/s.
        imbalance : Callable[[], ImbalanceSummary]
            Returns the current field's imbalance readings from one
            evaluation. Called only when condition (a) and, with it on,
            condition (e) hold.
        viscosity_step : float, optional
            Largest change of the kinematic eddy viscosity over the outer
            iteration, m^2/s. Required with (e) on and refused with it off.
        nu_scale : float, optional
            The kinematic viscosity (e)'s estimate is relative to at this
            step, m^2/s: ``viscosity_scale`` of the iterate just computed.
            Required with (e) on and refused with it off; positive and finite.

        Returns
        -------
        bool
            True when conditions (a), (b), (c) and (d), and (e) when it is
            on, all hold.

        Raises
        ------
        ValueError
            If ``viscosity_step`` or ``nu_scale`` is given with (e) off or
            missing with it on, or ``nu_scale`` is not positive and finite.
        """
        on = self._viscosity_condition
        if (viscosity_step is None) == on or (nu_scale is None) == on:
            raise ValueError(
                "viscosity_step and nu_scale are given exactly when the rule "
                "applies condition (e) (ADR-012 E)"
            )
        if nu_scale is not None:
            # Checked before anything is recorded, so a refused step leaves no trace.
            _check_positive("nu_scale", nu_scale)
        self._steps.append(float(step))
        estimate = self._estimate_of(self._steps, self._scale)
        self.estimate_history.append(estimate)
        viscosity_ok = True
        if viscosity_step is not None and nu_scale is not None:
            self.viscosity_scale_history.append(float(nu_scale))
            self._nu_steps.append(float(viscosity_step))
            nu_estimate = self._estimate_of(self._nu_steps, nu_scale)
            self.viscosity_estimate_history.append(nu_estimate)
            viscosity_ok = nu_estimate < self._error_tol
        if not (estimate < self._error_tol and viscosity_ok):
            return False
        found = imbalance()
        return (
            found.worst < self._imbalance_tol
            and found.absolute_sum / self._flux_scale < self._error_tol
            and abs(found.signed_sum) < self._imbalance_tol
        )
