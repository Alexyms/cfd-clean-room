"""Steady SIMPLE on the staggered (MAC) grid: the step 3 to 5 modules in one loop.

ECR-001 step 6 (REQ-S04, REQ-S07, REQ-S08, REQ-S09, REQ-S12). Each outer
iteration makes the two calls that already exist, MomentumPredictor.predict
and PressureCorrector.correct, and keeps the corrected fields. Nothing is
discretised here. The solver was built beside the collocated one it
replaced, so the two could be compared from one commit; the collocated
solver was retired on 2026-10-02 (tag collocated-final).

The public shape is the collocated solver's: the constructor,
``solve_steady(on_iteration=None)`` returning cell-centered (u, v, p),
``residual_history``, ``last_pressure_iterations`` (``last_pressure_sweeps``
until ECR-003 step 1) and ``stage_seconds``, the three the ECR-001 section
7.1 interface obligation names.

Boundary faces. ``apply_normal_velocity`` writes the wall and inlet faces
once, before the loop. Neither the predictor nor the corrector writes them,
so they are not re-imposed. Pressure outlet faces are the caller's to
extrapolate (the momentum.py contract): before every prediction each outlet
face takes the value of the interior face next to it, and the corrector then
corrects it against p' = 0 so the outlet cell closes. The returned faces are
the corrected ones.

The residual is the collocated one, identical in definition: the
largest change of the cell-centered u and v between outer iterations over
FLUID cells, divided by the reference velocity ``F_ref / (rho h)``. F_ref is
rho times the inlet volumetric flux, or for a closed domain rho times the
largest boundary velocity times h, with h the larger mean spacing
``x[nx] / nx`` or ``y[ny] / ny``, which is ``max(dx, dy)`` on a uniform
mesh. The inlet flux is the staggered layer's, the exact face sum. The
collocated layer's is two corner cells short on VAL-001, so the two
reference velocities differ by 40/38 there and agree exactly on a closed
domain (docs/reports/staggered_integration_step6.md).

``solve_steady(eddy_viscosity=...)`` holds a prescribed kinematic eddy
viscosity for the whole solve and hands the predictor ``mu + rho nu_t``
(ECR-002 step 4); step 5 drove prescribed fields through it.

With the configuration's turbulence section the solve is the coupled one
(ADR-012 D, ECR-002 step 6), and a prescribed field is refused: one source of
nu_t per solve. Each outer iteration predicts the momentum with ``mu + rho
nu_t`` and the wall functions' viscosity on the wall faces
(``TurbulenceBoundary.wall_viscosity`` over
``MomentumPredictor.stencil_viscosity``), corrects the pressure, takes one
pseudo-time step of k and eps on the corrected faces
(``KEpsilonModel.step`` with ``TurbulenceBoundary.conditions``), then
under-relaxes the eddy viscosity, ``nu_t = (1 - a_t) nu_t_old + a_t C_mu
k^2 / eps`` with a_t ``alpha_turbulence``. The solve starts from k and eps
uniform at the inlets' inflow-weighted means and nu_t from them. The step
raises PositivityError unless k and eps are positive and finite in every
non-SOLID cell (REQ-S15); the solve stops there, naming the outer iteration.
The last solve's k, eps and nu_t are ``turbulence_state``. Without the
section every line runs as before.

The faces of the last solve are kept as ``face_velocities`` (REQ-S13):
the transport solver advects with them because the cell means the contract
returns do not carry the continuity the stopping rule enforced on the faces.

Two starts besides rest (ADR-014 step 0). ``solve_steady(initial=state)``
is the warm start of ADR-014 A: a new solve from a FlowState's faces,
pressure, k and eps, the eddy viscosity rebuilt from k and eps and the rule
fresh, as the probe of ADR-014 T1 started its warm arm. A converged state
warm-started under its own boundaries therefore stops when the rule's window
first fills, not at once. ``solve_steady(resume=checkpoint)`` is not a start
but a continuation (ADR-014 D): the relaxed nu_t, the rule's window, the
residual history, the cap hits and the outer count are those the
checkpoint's solve held, so a solve stopped at its cap and resumed under a
larger one reaches what one solve with that cap reaches, bit for bit. Every
solve leaves ``flow_state`` and ``steady_checkpoint``, except one whose fields
went non-finite: that solve returns them as it always did and leaves both
None, since a diverged state is neither a warm start nor a resume point.
With ``solver.checkpoint_interval`` set, ``solve_steady(on_checkpoint=...)``
also hands the caller the checkpoint every that many outer iterations, so a
crash loses at most that many (ADR-014 D, step 3). A checkpoint of a solve
under a prescribed ``eddy_viscosity`` carries the field's SHA-256, and a
resume under another field, or under none, is refused: the field is an input
of the solve that no configuration hash covers. src/io_manager.py writes the
checkpoints to the run's checkpoint file and reads them back.

``stopping_rule`` picks the stop. ``velocity_step``, the default, is the
collocated rule: the residual below ``convergence_tol``. ``error_estimate``
(src/stopping.py) needs the estimated iteration error over the largest
prescribed boundary velocity below ``iteration_error_tol``, the worst
per-cell mass imbalance below ``mass_imbalance_tol``, the summed imbalance
over the through-flow below ``iteration_error_tol``, and the absolute signed
domain sum of the imbalance below ``mass_imbalance_tol``; with the turbulence
section, condition (e) as well, the eddy viscosity's estimated iteration
error over the molecular nu plus the median nu_t of the FLUID cells, formed
each outer iteration from the iterate just computed. Reaching
``max_simple_iter`` is not convergence under either.

A pressure correction that stops at ``max_pressure_iter`` is reported, not
hidden: the solver counts such corrections in ``pressure_cap_hits``, warns at
the first, and under velocity_step does not stop on an outer iteration whose
correction reached the cap. A truncated correction leaves the faces unbalanced
while the velocity barely moves, which is how the committed cap of 200 sweeps
froze the 80x30 probe room with 47% of the supply unaccounted and the
velocity-step rule called it converged (docs/reports/pressure_solver_ecr003.md,
section 8.2; ADR-013 B). Under error_estimate the continuity conditions refuse
that state on their own.

The time-accurate solve (ADR-014 tier 2, decisions 3 to 5; step 6a).
``solve_transient`` marches a FlowState through time steps of ``dt``. Each
step is BDF2 (the predictor's transient term, ADR-014 decision 3) after an
implicit Euler first step, or implicit Euler throughout as the control. Within
a step the outer iteration is the steady one (extrapolate the outlets,
predict, correct) until the largest change of the cell-centred velocity over
an outer iteration is below ``step_tol`` times the velocity scale, the largest
prescribed boundary velocity (the error_estimate rule's scale, U on the
cylinder), and not on an outer iteration whose correction reached its cap. The
scale is not ``reference_velocity``, the inflow over one cell spacing: on the
cylinder's 0.1 d grid that is 200 U, and decision 5's criterion is a fraction
of U. A step that reaches ``step_cap`` outer iterations, or whose fields go
non-finite, raises TransientStepError with the step and every step's record,
so a stalled step fails the run. The solve is laminar: tier 2 has no eddy
viscosity (ADR-014 decision 1). ``solve_steady`` is untouched by it.
"""

import hashlib
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from time import perf_counter

import numpy as np

from src.boundary_staggered import StaggeredBoundary
from src.config import ERROR_ESTIMATE, VELOCITY_STEP, SimConfig
from src.mesh import FLUID, SOLID, Mesh
from src.momentum import MomentumPredictor
from src.pressure import ZERO_SCALE, PressureCorrector
from src.staggered import (
    FaceVelocities,
    allocate_fields,
    p_shape,
    to_cell_centers,
    u_shape,
    v_shape,
)
from src.stopping import (
    RULE_VERSION_WITH_E,
    RULE_VERSION_WITHOUT_E,
    ErrorEstimateRule,
    ImbalanceSummary,
    IterationState,
    viscosity_scale,
)
from src.turbulence import (
    KEpsilonModel,
    PositivityError,
    TurbulenceBoundary,
    TurbulenceState,
)

logger = logging.getLogger(__name__)


def rule_version(config: SimConfig) -> int | None:
    """The stopping rule's version a solver built from ``config`` applies.

    Parameters
    ----------
    config : SimConfig
        The configuration a solver would be built from.

    Returns
    -------
    int or None
        ``StaggeredSolver(...).rule_version`` without building a solver, for
        a script that decides from it whether a saved solve can be reused:
        None under velocity_step, 5 under error_estimate with the turbulence
        section (condition (e) on, ADR-012 E), 3 without it.
    """
    if config.stopping_rule != ERROR_ESTIMATE:
        return None
    if config.turbulence is not None:
        return RULE_VERSION_WITH_E
    return RULE_VERSION_WITHOUT_E


_STOP_REASONS = {
    VELOCITY_STEP: "velocity_step_below_tol",
    ERROR_ESTIMATE: "error_estimate_and_continuity",
}
CAP_STOP = "max_simple_iter"
# The stop of a transient solve that reached t_end, every step within its
# criterion.
TRANSIENT_STOP = "transient_t_end"
BDF2 = "bdf2"
IMPLICIT_EULER = "implicit_euler"
TRANSIENT_SCHEMES = (BDF2, IMPLICIT_EULER)
# How far t_end / dt may sit from a whole number of steps, relative: rounding
# in the division, not a shortened last step, which BDF2 does not take.
_STEP_COUNT_ROUNDING = 1e-9
# A SHA-256 written as hexadecimal: 64 of these digits.
_SHA256_DIGITS = 64
_HEX_DIGITS = "0123456789abcdef"


def _frozen_field(name: str, value: object, ndim: int = 2) -> np.ndarray:
    """A read-only C-contiguous float64 copy of a finite array, or a refusal."""
    if not isinstance(value, np.ndarray) or value.dtype != np.float64:
        got = value.dtype if isinstance(value, np.ndarray) else type(value).__name__
        raise TypeError(f"{name} must be a float64 ndarray, got {got}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite everywhere")
    copy = np.array(value, dtype=np.float64, order="C")
    copy.setflags(write=False)
    return copy


def _count(name: str, value: object) -> int:
    """A non-negative int that is not a bool, or a refusal."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _positive_float(name: str, value: object) -> float:
    """A finite positive float that is not a bool, or a refusal."""
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(f"{name} must be a float, got {type(value).__name__}")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value


@dataclass(frozen=True, eq=False)
class FlowState:
    """A steady solve's fields and stop, as the warm start reads them (ADR-014 A).

    The arrays are validated and stored as read-only C-contiguous copies,
    so a state cannot change after it is built, whoever else holds the
    arrays it was built from.

    Parameters
    ----------
    u : np.ndarray
        x-velocity on the vertical faces, m/s, [ny, nx+1], float64, finite.
    v : np.ndarray
        y-velocity on the horizontal faces, m/s, [ny+1, nx], likewise.
    p : np.ndarray
        Pressure at the cell centres, Pa, [ny, nx], likewise (the modified
        pressure of ADR-012 D with the turbulence section).
    k, eps, nu_t : np.ndarray or None
        With the turbulence section: k (m^2/s^2), eps (m^2/s^3) and the
        under-relaxed eddy viscosity (m^2/s) of the solve, [ny, nx],
        float64, finite. All three or none.
    stop_reason : str or None
        The solve's ``stop_reason``; None for a state no solve finished.
    outer : int
        Outer iterations behind the state, non-negative.

    Raises
    ------
    TypeError
        If an array is not a float64 ndarray, ``stop_reason`` is not a str
        or None, or ``outer`` is not an int (a bool is refused).
    ValueError
        If the shapes do not form one staggered grid, an array is not
        finite, k, eps and nu_t are not all present or all absent, or
        ``outer`` is negative.
    """

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    k: np.ndarray | None = None
    eps: np.ndarray | None = None
    nu_t: np.ndarray | None = None
    stop_reason: str | None = None
    outer: int = 0

    def __post_init__(self) -> None:
        for name in ("u", "v", "p"):
            object.__setattr__(self, name, _frozen_field(name, getattr(self, name)))
        ny, nx = self.p.shape
        if self.u.shape != (ny, nx + 1) or self.v.shape != (ny + 1, nx):
            raise ValueError(
                f"u {self.u.shape} and v {self.v.shape} are not the faces of p "
                f"{self.p.shape}: expected u {(ny, nx + 1)} and v {(ny + 1, nx)}"
            )
        present = [
            name for name in ("k", "eps", "nu_t") if getattr(self, name) is not None
        ]
        if present and len(present) != 3:
            raise ValueError(
                f"k, eps and nu_t are given together or not at all, got {present}"
            )
        for name in present:
            field_ = _frozen_field(name, getattr(self, name))
            if field_.shape != self.p.shape:
                raise ValueError(
                    f"{name} must have p's shape {self.p.shape}, got {field_.shape}"
                )
            object.__setattr__(self, name, field_)
        reason = self.stop_reason
        if reason is not None and not isinstance(reason, str):
            raise TypeError(
                f"stop_reason must be a str or None, got {type(reason).__name__}"
            )
        _count("outer", self.outer)

    @property
    def turbulent(self) -> bool:
        """Whether the state carries k, eps and nu_t."""
        return self.k is not None


@dataclass(frozen=True, eq=False)
class SteadyCheckpoint:
    """Everything a steady solve carries between outer iterations (ADR-014 D).

    A solve resumed from it (``solve_steady(resume=...)``) continues the
    iteration it was taken from: every later iterate, the residual history
    and the stop are those of the solve that was never interrupted, bit for
    bit. The warm start (``initial=``) is not that: it restarts the rule and
    rebuilds the eddy viscosity from k and eps (``solve_steady``).

    Parameters
    ----------
    state : FlowState
        The fields after the last outer iteration, nu_t the under-relaxed
        field the next iteration reads; ``state.outer`` the iterations taken.
    residual_history : tuple[float, ...]
        The solve's ``residual_history``, one entry per outer iteration, so
        exactly ``state.outer`` entries.
    velocity_steps, viscosity_steps : tuple[float, ...]
        The error_estimate rule's window (``ErrorEstimateRule.window``),
        both empty under velocity_step.
    rule_version : int or None
        The solver's ``rule_version``; a resume needs the same.
    pressure_cap_hits : int
        Corrections so far that stopped at max_pressure_iter.
    eddy_viscosity_sha256 : str or None
        SHA-256, hexadecimal, of the bytes of the prescribed
        ``eddy_viscosity`` field the solve ran under; None for a solve
        without one. A resume needs the same (ADR-014 D, step 3).

    Raises
    ------
    TypeError
        If ``state`` is not a FlowState, a count is not an int, or the
        digest is neither None nor a str.
    ValueError
        If the residual history's length is not ``state.outer``, a count is
        negative, or the digest is not 64 lowercase hexadecimal digits.
    """

    state: FlowState
    residual_history: tuple[float, ...]
    velocity_steps: tuple[float, ...]
    viscosity_steps: tuple[float, ...]
    rule_version: int | None
    pressure_cap_hits: int
    eddy_viscosity_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, FlowState):
            raise TypeError(
                f"state must be a FlowState, got {type(self.state).__name__}"
            )
        for name in ("residual_history", "velocity_steps", "viscosity_steps"):
            object.__setattr__(
                self, name, tuple(float(value) for value in getattr(self, name))
            )
        if len(self.residual_history) != self.state.outer:
            raise ValueError(
                f"residual_history holds {len(self.residual_history)} entries for "
                f"{self.state.outer} outer iterations"
            )
        if self.rule_version is not None:
            _count("rule_version", self.rule_version)
        _count("pressure_cap_hits", self.pressure_cap_hits)
        digest = self.eddy_viscosity_sha256
        if digest is not None:
            if not isinstance(digest, str):
                raise TypeError(
                    "eddy_viscosity_sha256 must be a str or None, got "
                    f"{type(digest).__name__}"
                )
            if len(digest) != _SHA256_DIGITS or digest.strip(_HEX_DIGITS):
                raise ValueError(
                    "eddy_viscosity_sha256 must be 64 lowercase hexadecimal digits, "
                    f"got {digest!r}"
                )


@dataclass(frozen=True)
class TransientStep:
    """One time step of ``solve_transient`` as it ended (ADR-014 decision 5).

    Parameters
    ----------
    step : int
        The step's number, 1 for the first.
    t : float
        The time the step reached, ``step * dt``, in s.
    scheme : str
        "implicit_euler" or "bdf2": the time derivative the step took.
    outer : int
        Outer iterations the step took.
    residual_history : tuple[float, ...]
        The largest change of the cell-centred u and v over FLUID cells per
        outer iteration, over the velocity scale, one entry per outer.
    worst_imbalance : float
        The largest absolute per-cell mass imbalance of the step's last
        faces over non-SOLID cells, kg/s per unit depth.
    pressure_cap_hits : int
        The step's corrections that stopped at max_pressure_iter.
    """

    step: int
    t: float
    scheme: str
    outer: int
    residual_history: tuple[float, ...]
    worst_imbalance: float
    pressure_cap_hits: int


class TransientStepError(RuntimeError):
    """A time step that did not meet its criterion: the run fails here.

    Parameters
    ----------
    message : str
        Names the step, its time and why it failed.
    step : TransientStep
        The failing step's record, its residual history included.
    steps : tuple[TransientStep, ...]
        Every step's record up to and including the failing one.
    """

    def __init__(
        self, message: str, step: TransientStep, steps: tuple[TransientStep, ...]
    ) -> None:
        super().__init__(message)
        self.step = step
        self.steps = steps


def _digest(field: np.ndarray) -> str:
    """SHA-256 of a prescribed eddy viscosity's bytes, little-endian float64."""
    data = np.ascontiguousarray(field, dtype="<f8").tobytes()
    return hashlib.sha256(data).hexdigest()


class StaggeredSolver:
    """SIMPLE solver for steady incompressible flow on the staggered grid.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh, uniform or stretched.
    config : SimConfig
        Supplies rho, max_simple_iter and the stopping keys, and through the
        predictor and corrector every other solver parameter.
    boundary : StaggeredBoundary
        Imposes the normal velocities and supplies the outlet masks and the
        reference flux inputs.

    Attributes
    ----------
    reference_velocity : float
        The velocity the residual divides the largest change by.
    residual_history : list[float]
        Scaled residual after each outer iteration of the last steady solve;
        empty after a transient one, whose residuals are in
        ``transient_steps``.
    converged : bool
        Whether the last solve met its stopping rule rather than the cap.
    stop_reason : str or None
        "velocity_step_below_tol", "error_estimate_and_continuity" or
        "max_simple_iter"; None until a solve completes. Both reset per solve.
    last_pressure_iterations : int
        Conjugate gradient iterations of the most recent pressure correction.
    pressure_cap_hits : int
        Corrections of the last solve that stopped at max_pressure_iter
        rather than at their stop; reset per solve, recorded by the harness.
    stage_seconds : dict[str, float]
        Wall time of the last solve in each stage: "momentum" (outlet
        extrapolation and prediction), "pressure" (the p' solve and the
        velocity correction, which the corrector does in one call) and
        "correct" (the conversion to cell centers and the residual). This
        grid has no separate flux stage: the right-hand side is read off
        the face velocities inside the correction. Observability only.
    last_mass_imbalance : np.ndarray
        Per-cell mass imbalance of the returned face velocities, shape
        [ny, nx]. Observability only; the solver never reads it.
    face_velocities : FaceVelocities or None
        The final faces of the last solve, converged or not, as read-only
        copies (REQ-S13, ADR-011 A); None until a solve completes. Their
        two-face averages are the returned cell means bitwise and their
        ``mass_imbalance`` is ``last_mass_imbalance`` bitwise. When
        ``stop_reason`` is "error_estimate_and_continuity" that imbalance
        meets REQ-S04's per-cell and domain-sum clauses; under
        velocity_step the faces carry no continuity promise.
    flux_scale : float or None
        Read-only. The flux scale the error_estimate rule was built with, kg/s
        per unit depth; None under velocity_step.
    rule_version : int or None
        Read-only. The version of the stopping rule a solve applies, its
        rule's ``version``: 3, or 5 with the turbulence section, whose
        condition (e) bounds the eddy viscosity's iteration error; None
        under velocity_step. The harness and the scripts store it with what
        they save (``rule_version`` gives it from a configuration).
    turbulence_state : TurbulenceState or None
        k, eps and the under-relaxed nu_t at the end of the last coupled
        solve, converged or not, read-only arrays; None without the
        turbulence section or before a solve completes.
    flow_state : FlowState or None
        The last solve's faces, pressure and, with the turbulence section,
        k, eps and the under-relaxed nu_t, with its stop reason and outer
        count, converged or not; None before a solve completes and after one
        whose fields are not finite (never a stale earlier state). What
        ``solve_steady(initial=...)`` takes (ADR-014 A).
    steady_checkpoint : SteadyCheckpoint or None
        The last solve's state with its residual history, the rule's window
        and its cap hits; None before a solve completes and after one whose
        fields are not finite. A solve that stopped at max_simple_iter
        continues from it with ``solve_steady(resume=...)`` (ADR-014 D).
        None after a transient solve, which is no steady solve to continue.
    transient_steps : list[TransientStep]
        One record per time step of the last ``solve_transient``, failed
        step included; empty until one runs and reset by each.

    Raises
    ------
    ValueError
        Under error_estimate, if no boundary prescribes a velocity to scale by.
        The message names stopping_rule and the boundaries. With the
        turbulence section, if no velocity inlet admits air: the coupled
        solve starts from the inlets' k and eps.
    """

    def __init__(
        self, mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary
    ) -> None:
        self._mesh = mesh
        self._boundary = boundary
        self._predictor = MomentumPredictor(mesh, config, boundary)
        self._corrector = PressureCorrector(mesh, config, boundary)
        self._rho = config.rho
        self._mu = config.mu
        self._live = mesh.cell_type != SOLID
        self._convergence_tol = config.convergence_tol
        self._max_simple_iter = config.max_simple_iter
        self._checkpoint_interval = config.checkpoint_interval
        self._stopping_rule = config.stopping_rule
        self._rule_tols = (config.iteration_error_tol, config.mass_imbalance_tol)
        self._fluid = mesh.cell_type == FLUID
        self._outlets = boundary.pressure_outlets()
        self._model: KEpsilonModel | None = None
        self._walls: TurbulenceBoundary | None = None
        if config.turbulence is not None:
            self._model = KEpsilonModel(mesh, config)
            self._walls = TurbulenceBoundary(mesh, config, boundary)
            self._alpha_turbulence = config.turbulence.alpha_turbulence
            # Raises here, before any solve, when no inlet sets the supply's k.
            self._initial_turbulence = self._walls.initial_values()
        self.turbulence_state: TurbulenceState | None = None
        self.flow_state: FlowState | None = None
        self.steady_checkpoint: SteadyCheckpoint | None = None
        self.transient_steps: list[TransientStep] = []

        self.reference_velocity: float = self._reference_velocity()
        self.residual_history: list[float] = []
        self.last_pressure_iterations: int = 0
        self.pressure_cap_hits: int = 0
        self.stage_seconds: dict[str, float] = self._zero_stage_seconds()
        self.last_mass_imbalance: np.ndarray = np.zeros(p_shape(mesh))
        self.face_velocities: FaceVelocities | None = None
        self.converged: bool = False
        self.stop_reason: str | None = None
        self._flux_scale: float | None = None
        # Built once here so a zero velocity scale raises at construction.
        rule = self._new_rule()
        self._rule_version = None if rule is None else rule.version

    @property
    def flux_scale(self) -> float | None:
        """The error_estimate rule's flux scale, kg/s per unit depth; None otherwise."""
        return self._flux_scale

    @property
    def rule_version(self) -> int | None:
        """The stopping rule's version a solve applies; None under velocity_step."""
        return self._rule_version

    def _zero_stage_seconds(self) -> dict[str, float]:
        stages = {"momentum": 0.0, "pressure": 0.0, "correct": 0.0}
        if self._model is not None:
            stages["turbulence"] = 0.0
        return stages

    def _reference_velocity(self) -> float:
        """The collocated solver's reference velocity, from the staggered inputs."""
        nx, ny = self._mesh.xc.shape[0], self._mesh.yc.shape[0]
        h = max(float(self._mesh.x[nx]) / nx, float(self._mesh.y[ny]) / ny)
        vol_flux = self._boundary.get_total_inlet_flux()
        if abs(vol_flux) > ZERO_SCALE:
            f_ref = self._rho * abs(vol_flux)
        else:
            max_vel = self._boundary.get_max_boundary_velocity()
            f_ref = max(self._rho * max_vel * h, ZERO_SCALE)
        return f_ref / (self._rho * h)

    def _new_rule(self) -> ErrorEstimateRule | None:
        """A fresh error_estimate rule; None under velocity_step.

        The velocity scale is the largest prescribed boundary velocity. The
        flux scale F is rho times the total inflow, or on a closed domain rho
        times that velocity times the longer side: the corrector's flux_scale,
        formed once there so the pressure solve's rounding floor and this
        rule read the same F. It is not built from reference_velocity, the
        inflow over one cell spacing: that moves with the grid, and so would
        the bound on the summed imbalance. With the turbulence section the
        rule takes condition (e), and solve_steady passes it each outer
        iteration's scale (``viscosity_scale``, ADR-012 E).
        """
        if self._stopping_rule != ERROR_ESTIMATE:
            return None
        scale = self._boundary.get_max_boundary_velocity()
        if scale <= 0.0:
            raise ValueError(
                "stopping_rule error_estimate needs a velocity scale, and no "
                "boundary prescribes a velocity; give one or use velocity_step"
            )
        self._flux_scale = self._corrector.flux_scale
        # Without the section the call is the laminar one, so a rule a test or
        # a probe substitutes still fits.
        if self._walls is None:
            return ErrorEstimateRule(scale, self._flux_scale, *self._rule_tols)
        return ErrorEstimateRule(
            scale, self._flux_scale, *self._rule_tols, viscosity_condition=True
        )

    def _imbalance_summary(self, u: np.ndarray, v: np.ndarray) -> ImbalanceSummary:
        """Worst, absolute-summed and signed-summed cell imbalance, evaluated once."""
        imbalance = self._corrector.mass_imbalance(u, v)
        cells = np.abs(imbalance)
        return ImbalanceSummary(
            worst=float(cells.max()),
            absolute_sum=float(cells.sum()),
            signed_sum=float(imbalance.sum()),
        )

    def _extrapolate_outlets(self, u: np.ndarray, v: np.ndarray) -> None:
        """Give each pressure outlet face its interior neighbour's value, in place."""
        left = self._outlets["left"].is_outlet
        right = self._outlets["right"].is_outlet
        bottom = self._outlets["bottom"].is_outlet
        top = self._outlets["top"].is_outlet
        u[left, 0] = u[left, 1]
        u[right, -1] = u[right, -2]
        v[0, bottom] = v[1, bottom]
        v[-1, top] = v[-2, top]

    def _effective_viscosity(self, eddy_viscosity: np.ndarray) -> np.ndarray:
        """``mu + rho nu_t`` per cell, or a refusal before any arithmetic.

        SOLID cells take air's viscosity; the predictor reads none of them.
        """
        field = eddy_viscosity
        if not isinstance(field, np.ndarray) or field.dtype != np.float64:
            got = field.dtype if isinstance(field, np.ndarray) else type(field).__name__
            raise TypeError(f"eddy_viscosity must be a float64 ndarray, got {got}")
        shape = p_shape(self._mesh)
        if field.shape != shape:
            raise ValueError(
                f"expected eddy_viscosity of shape {shape}, got {field.shape}"
            )
        live = field[self._live]
        if not np.isfinite(live).all():
            raise ValueError("eddy_viscosity must be finite in every non-SOLID cell")
        if np.any(live < 0.0):
            raise ValueError("eddy_viscosity must be non-negative (ADR-012 D)")
        return self._mu + self._rho * np.where(self._live, field, 0.0)

    def solve_steady(
        self,
        on_iteration: Callable[[IterationState], None] | None = None,
        *,
        eddy_viscosity: np.ndarray | None = None,
        initial: FlowState | None = None,
        resume: SteadyCheckpoint | None = None,
        on_checkpoint: Callable[[SteadyCheckpoint], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve for steady-state velocity and pressure fields.

        Parameters
        ----------
        on_iteration : Callable[[IterationState], None], optional
            Called after every outer iteration with the cell-centered u and
            v, the pressure, the residual and the corrector's iteration
            count for that iteration. The arrays are fresh each iteration and
            must not be modified.
        eddy_viscosity : np.ndarray, optional
            Kinematic eddy viscosity nu_t per cell, m^2/s, shape [ny, nx],
            float64, finite and non-negative in every non-SOLID cell (the
            transport solver's convention), held fixed for the solve. The
            predictor receives ``mu_eff = mu + rho nu_t`` each outer
            iteration (ADR-012 D). None is the laminar path, bitwise.
            Refused with the turbulence section, whose model is the solve's
            one source of nu_t.
        initial : FlowState, optional
            The warm start (ADR-014 A): the solve starts from the state's
            faces and pressure, and with the turbulence section from
            ``KEpsilonModel.state(k, eps)``, in place of rest and the inlets'
            uniform k and eps. The eddy viscosity is rebuilt from k and eps
            and the state's nu_t is not read, as the probe of ADR-014 T1 did,
            and the rule starts fresh, so a converged state warm-started
            under its own boundaries stops once the rule's window has filled.
            The boundary's normal velocities are written over the state's, so
            a state of other boundaries starts this solve. None is the start
            from rest, bitwise.
        resume : SteadyCheckpoint, optional
            Continue the solve the checkpoint was taken from (ADR-014 D):
            the faces, pressure, k, eps and the under-relaxed nu_t as they
            stood, the residual history, the rule's window and the cap hits
            carried on, and the outer iterations counted from the
            checkpoint's, so ``max_simple_iter`` is the total. Under the
            configuration the checkpoint's solve ran with, a larger cap
            reaches what one solve with that cap reaches, bitwise. Refused
            with ``initial``, for a checkpoint whose solve converged, and
            under an ``eddy_viscosity`` other than the one the checkpoint's
            solve ran under (none included).
        on_checkpoint : Callable[[SteadyCheckpoint], None], optional
            Called every ``solver.checkpoint_interval`` outer iterations,
            counted from the solve's start (a resumed solve's from the
            checkpoint's), with the SteadyCheckpoint a solve stopped there
            would leave, its stop reason None: a resume from it reaches what
            the uninterrupted solve reaches, bitwise (ADR-014 D, step 3).
            Not called on the iteration the solve converges at, when a field
            is not finite, or without the key. It reads the iterates and
            changes none of them, so the solve is the same with or without
            it. The writer is the caller's (``io_manager.RunFile``).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (u, v, p) at cell centers, each shape [ny, nx], dtype float64,
            C-contiguous. A solve whose fields went non-finite returns them
            too, as before, and leaves ``flow_state`` and ``steady_checkpoint``
            None.

        Raises
        ------
        TypeError
            If ``eddy_viscosity`` is not a float64 ndarray.
        ValueError
            If ``eddy_viscosity`` has the wrong shape, or a non-finite or
            negative value in a non-SOLID cell, or is given with the
            turbulence section present. If ``initial`` or ``resume`` holds
            a state of another grid, carries k and eps for a laminar solve or
            lacks them for a coupled one, or has k or eps not positive in a
            non-SOLID cell; if both are given; if ``resume`` comes from a
            solve that converged, under another rule version, under another
            ``eddy_viscosity`` (none where it had one, or one where it had
            none), or with a negative nu_t in a non-SOLID cell.
        TypeError
            If ``initial`` is not a FlowState or ``resume`` not a
            SteadyCheckpoint.
        PositivityError
            In the coupled solve, if k or eps is not positive and finite at a
            non-SOLID cell after an outer iteration's step; the message names
            the outer iteration and the first such cell.
        """
        model, walls = self._model, self._walls
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
        # Reset with the timers: a solve that stops before its first
        # correction must not report the previous call's counts.
        self.last_pressure_iterations = 0
        self.pressure_cap_hits = 0 if resume is None else resume.pressure_cap_hits
        self.converged, self.stop_reason = False, None
        rule = self._new_rule()
        if rule is not None and resume is not None:
            rule.restore(resume.velocity_steps, resume.viscosity_steps)
        u_c, v_c = to_cell_centers(u, v)
        fluid = self._fluid

        for iteration in range(start, self._max_simple_iter):
            t0 = perf_counter()
            self._extrapolate_outlets(u, v)
            # Without a field the call is the laminar one, so a predictor a
            # probe substitutes (docs/reports/probe41/outlet41.py) still fits.
            if state is not None and walls is not None:
                coupled = self._effective_viscosity(state.nu_t)
                prediction = self._predictor.predict(
                    u,
                    v,
                    p,
                    mu_eff=coupled,
                    wall_mu=walls.wall_viscosity(
                        state.k, self._predictor.stencil_viscosity(coupled)
                    ),
                )
            elif mu_eff is None:
                prediction = self._predictor.predict(u, v, p)
            else:
                prediction = self._predictor.predict(u, v, p, mu_eff=mu_eff)
            t1 = perf_counter()
            self.stage_seconds["momentum"] += t1 - t0

            corrected = self._corrector.correct(prediction, p)
            u, v, p = corrected.u, corrected.v, corrected.p
            self.last_pressure_iterations = corrected.iterations
            if corrected.reached_cap:
                self.pressure_cap_hits += 1
                if self.pressure_cap_hits == 1:
                    logger.warning(
                        "SIMPLE iter %d: pressure correction stopped at "
                        "max_pressure_iter (%d) before its residual met the stop",
                        iteration,
                        corrected.iterations,
                    )
            t2 = perf_counter()
            self.stage_seconds["pressure"] += t2 - t1

            viscosity_step = nu_scale = None
            if state is not None:
                nu_t_old = state.nu_t
                state = self._turbulence_step(state, u, v, iteration)
                viscosity_step = float(
                    np.max(np.abs(state.nu_t[self._live] - nu_t_old[self._live]))
                )
                nu_scale = viscosity_scale(self._mu / self._rho, state.nu_t, fluid)
                t3 = perf_counter()
                self.stage_seconds["turbulence"] += t3 - t2
                t2 = t3

            u_prev, v_prev = u_c, v_c
            u_c, v_c = to_cell_centers(u, v)
            du = float(np.max(np.abs(u_c[fluid] - u_prev[fluid])))
            dv = float(np.max(np.abs(v_c[fluid] - v_prev[fluid])))
            residual = max(du, dv) / max(self.reference_velocity, ZERO_SCALE)
            self.residual_history.append(residual)
            self.stage_seconds["correct"] += perf_counter() - t2

            if on_iteration is not None:
                on_iteration(
                    IterationState(
                        iteration=iteration,
                        residual=residual,
                        pressure_iterations=corrected.iterations,
                        u=u_c,
                        v=v_c,
                        p=p,
                        pressure_products=corrected.products,
                    )
                )

            if rule is None:
                # A capped correction makes the step small without balancing
                # the faces; the velocity-step rule cannot tell, so it is not
                # allowed to stop on one (ADR-013 B).
                stop = residual < self._convergence_tol and not corrected.reached_cap
            else:
                imbalance = partial(self._imbalance_summary, u, v)
                if viscosity_step is None:
                    stop = rule.update(max(du, dv), imbalance)
                else:
                    stop = rule.update(
                        max(du, dv),
                        imbalance,
                        viscosity_step=viscosity_step,
                        nu_scale=nu_scale,
                    )

            if iteration % 50 == 0 or stop:
                logger.info("SIMPLE iter %4d: residual = %.6e", iteration, residual)

            if stop:
                self.converged = True
                self.stop_reason = _STOP_REASONS[self._stopping_rule]
                logger.info("Converged at iteration %d", iteration)
                break

            interval = self._checkpoint_interval
            if (
                on_checkpoint is not None
                and interval is not None
                and (iteration + 1) % interval == 0
                and self._finite(u, v, p, state)
            ):
                on_checkpoint(self._checkpoint(u, v, p, state, rule, digest, None))

        if not self.converged:
            self.stop_reason = CAP_STOP
            logger.warning("Not converged: stopped at max_simple_iter")
        self.last_mass_imbalance = self._corrector.mass_imbalance(u, v)
        self.face_velocities = FaceVelocities.copy_of(u, v)
        self.turbulence_state = state
        p = np.ascontiguousarray(p)
        # A diverged solve returns its fields as it always did, for the probe
        # runners that classify a run by what comes back; FlowState refuses
        # non-finite input, and a diverged state is no warm start or resume.
        self.flow_state = self.steady_checkpoint = None
        if not self._finite(u, v, p, state):
            logger.warning(
                "Fields are not finite after %d outer iterations: no flow_state "
                "or steady_checkpoint is kept",
                len(self.residual_history),
            )
            return u_c, v_c, p
        self.steady_checkpoint = self._checkpoint(
            u, v, p, state, rule, digest, self.stop_reason
        )
        self.flow_state = self.steady_checkpoint.state
        return u_c, v_c, p

    def solve_transient(
        self,
        dt: float,
        t_end: float,
        *,
        initial: FlowState,
        step_tol: float,
        step_cap: int,
        on_step: Callable[[TransientStep, FlowState], None] | None = None,
        scheme: str = BDF2,
    ) -> FlowState:
        """March a flow state through time steps of dt to t_end (ADR-014 tier 2).

        Parameters
        ----------
        dt : float
            The time step, s, finite and positive.
        t_end : float
            The time the march ends at, s, a whole number of steps of ``dt``
            to rounding (a relative ``1e-9``); the march starts at 0.
        initial : FlowState
            The time level 0: faces and pressure of this grid, laminar. The
            boundary's normal velocities are written over its faces, as the
            warm start does, and the result is the first old level.
        step_tol : float
            The per-step criterion (ADR-014 decision 5): a step's outer
            iterations stop when the largest change of the cell-centred u
            and v over FLUID cells in one outer iteration is below
            ``step_tol`` times the velocity scale, the largest prescribed
            boundary velocity; finite and positive.
        step_cap : int
            The most outer iterations a step may take, at least 1. A step
            that reaches it fails the run (TransientStepError).
        on_step : Callable[[TransientStep, FlowState], None], optional
            Called after every step that met its criterion with the step's
            record and the state it reached (``stop_reason`` None, ``outer``
            the run's outer iterations so far). It reads; the march is the
            same with or without it.
        scheme : str
            "bdf2" (ADR-014 decision 3): BDF2 from the second step, the
            first implicit Euler, which needs one old level. Or
            "implicit_euler" every step, the first-order control.

        Returns
        -------
        FlowState
            The faces and pressure at ``t_end``, ``stop_reason``
            "transient_t_end", ``outer`` the run's outer iterations. The same
            state is ``flow_state``, and its faces ``face_velocities``.

        Raises
        ------
        TypeError
            If ``dt``, ``t_end`` or ``step_tol`` is not a float, ``step_cap``
            not an int (a bool is refused for each), ``scheme`` not a str, or
            ``initial`` not a FlowState.
        ValueError
            With the turbulence section; if ``dt``, ``t_end`` or ``step_tol``
            is not finite and positive, ``step_cap`` is below 1, ``scheme`` is
            neither name, ``t_end`` is not a whole number of steps, no
            boundary prescribes a velocity to scale the criterion by, or
            ``initial`` is a state of another grid or carries k and eps.
        TransientStepError
            When a step reaches ``step_cap`` outer iterations, or its residual
            is not finite, before meeting its criterion. It carries the
            failing step's record and every record before it; the solver's
            ``flow_state`` is then None.

        Notes
        -----
        Steps n >= 2 under "bdf2" take ``(3/2 phi - 2 phi_n + 1/2 phi_(n-1))
        / dt``; step 1, and every step under "implicit_euler", ``(phi -
        phi_n) / dt``. A step's iterations start from the last step's faces
        and pressure. Inside a step the stop is the velocity step's, with
        the same refusal of an outer iteration whose correction reached
        ``max_pressure_iter`` (ADR-013 B); ``convergence_tol``,
        ``max_simple_iter`` and ``stopping_rule`` are the steady solve's
        and are not read. ``residual_history`` is left empty: the per-outer
        residuals, over the velocity scale, are each step's record in
        ``transient_steps``.
        """
        if self._model is not None:
            raise ValueError(
                "solve_transient is laminar (ADR-014 decision 1): the configuration "
                "has a turbulence section"
            )
        dt = _positive_float("dt", dt)
        t_end = _positive_float("t_end", t_end)
        step_tol = _positive_float("step_tol", step_tol)
        if _count("step_cap", step_cap) < 1:
            raise ValueError(f"step_cap must be at least 1, got {step_cap}")
        if not isinstance(scheme, str):
            raise TypeError(f"scheme must be a str, got {type(scheme).__name__}")
        if scheme not in TRANSIENT_SCHEMES:
            raise ValueError(
                f"scheme must be one of {TRANSIENT_SCHEMES}, got {scheme!r}"
            )
        n_steps = round(t_end / dt)
        if n_steps < 1 or abs(n_steps * dt - t_end) > _STEP_COUNT_ROUNDING * t_end:
            raise ValueError(f"t_end {t_end} is not a whole number of steps of dt {dt}")
        scale = self._boundary.get_max_boundary_velocity()
        if scale <= 0.0:
            raise ValueError(
                "solve_transient scales step_tol by the largest prescribed boundary "
                "velocity, and no boundary prescribes one"
            )
        flow = self._check_state(initial, "initial", FlowState)
        u, v, p = np.array(flow.u), np.array(flow.v), np.array(flow.p)
        self._boundary.apply_normal_velocity(u, v)

        self.transient_steps = []
        self.residual_history = []
        self.stage_seconds = self._zero_stage_seconds()
        self.last_pressure_iterations = 0
        self.pressure_cap_hits = 0
        self.converged, self.stop_reason = False, None
        self.flow_state = self.steady_checkpoint = None
        self.face_velocities = None
        self.turbulence_state = None
        fluid = self._fluid
        # The old levels, latest first; one until the first step ends.
        old_u: tuple[np.ndarray, ...] = (u.copy(),)
        old_v: tuple[np.ndarray, ...] = (v.copy(),)
        total_outer = 0

        for n in range(1, n_steps + 1):
            bdf2 = scheme == BDF2 and n > 1
            # Two old levels from the second step under BDF2, else the latest.
            transient = (dt, old_u, old_v) if bdf2 else (dt, old_u[0], old_v[0])
            history: list[float] = []
            cap_hits = 0
            met = False
            u_c, v_c = to_cell_centers(u, v)
            for _ in range(step_cap):
                t0 = perf_counter()
                self._extrapolate_outlets(u, v)
                prediction = self._predictor.predict(u, v, p, transient=transient)
                t1 = perf_counter()
                self.stage_seconds["momentum"] += t1 - t0
                corrected = self._corrector.correct(prediction, p)
                u, v, p = corrected.u, corrected.v, corrected.p
                self.last_pressure_iterations = corrected.iterations
                if corrected.reached_cap:
                    cap_hits += 1
                    self.pressure_cap_hits += 1
                t2 = perf_counter()
                self.stage_seconds["pressure"] += t2 - t1
                u_prev, v_prev = u_c, v_c
                u_c, v_c = to_cell_centers(u, v)
                du = float(np.max(np.abs(u_c[fluid] - u_prev[fluid])))
                dv = float(np.max(np.abs(v_c[fluid] - v_prev[fluid])))
                residual = max(du, dv) / scale
                history.append(residual)
                self.stage_seconds["correct"] += perf_counter() - t2
                if not math.isfinite(residual):
                    break
                if residual < step_tol and not corrected.reached_cap:
                    met = True
                    break
            total_outer += len(history)
            imbalance = self._corrector.mass_imbalance(u, v)
            record = TransientStep(
                step=n,
                t=n * dt,
                scheme=BDF2 if bdf2 else IMPLICIT_EULER,
                outer=len(history),
                residual_history=tuple(history),
                worst_imbalance=float(np.max(np.abs(imbalance[self._live]))),
                pressure_cap_hits=cap_hits,
            )
            self.transient_steps.append(record)
            if not met:
                why = (
                    f"its residual is {history[-1]} (not finite)"
                    if not math.isfinite(history[-1])
                    else f"it reached step_cap ({step_cap} outer iterations) with "
                    f"the velocity step at {history[-1]:.3e} of the scale against "
                    f"step_tol {step_tol:g}"
                )
                raise TransientStepError(
                    f"time step {n} (t = {n * dt:g} s) failed: {why}",
                    record,
                    tuple(self.transient_steps),
                )
            logger.debug("time step %d: %d outer", n, record.outer)
            old_u = (u.copy(), old_u[0])
            old_v = (v.copy(), old_v[0])
            if on_step is not None:
                on_step(record, FlowState(u=u, v=v, p=p, outer=total_outer))

        self.converged, self.stop_reason = True, TRANSIENT_STOP
        self.last_mass_imbalance = self._corrector.mass_imbalance(u, v)
        self.face_velocities = FaceVelocities.copy_of(u, v)
        self.flow_state = FlowState(
            u=u,
            v=v,
            p=np.ascontiguousarray(p),
            stop_reason=TRANSIENT_STOP,
            outer=total_outer,
        )
        logger.info(
            "Transient solve: %d steps of %g s, %d outer iterations",
            n_steps,
            dt,
            total_outer,
        )
        return self.flow_state

    @staticmethod
    def _finite(
        u: np.ndarray, v: np.ndarray, p: np.ndarray, state: TurbulenceState | None
    ) -> bool:
        """Whether every field a FlowState would hold is finite."""
        fields = [u, v, p]
        if state is not None:
            fields += [state.k, state.eps, state.nu_t]
        return all(np.isfinite(field).all() for field in fields)

    def _checkpoint(
        self,
        u: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        state: TurbulenceState | None,
        rule: ErrorEstimateRule | None,
        digest: str | None,
        stop_reason: str | None,
    ) -> SteadyCheckpoint:
        """The solve as it stands after its last outer iteration, copied."""
        flow = FlowState(
            u=u,
            v=v,
            p=p,
            k=None if state is None else state.k,
            eps=None if state is None else state.eps,
            nu_t=None if state is None else state.nu_t,
            stop_reason=stop_reason,
            outer=len(self.residual_history),
        )
        steps, viscosity_steps = ((), ()) if rule is None else rule.window()
        return SteadyCheckpoint(
            state=flow,
            residual_history=tuple(self.residual_history),
            velocity_steps=steps,
            viscosity_steps=viscosity_steps,
            rule_version=self._rule_version,
            pressure_cap_hits=self.pressure_cap_hits,
            eddy_viscosity_sha256=digest,
        )

    def _check_state(self, given: object, name: str, kind: type) -> FlowState:
        """The FlowState in ``initial`` or ``resume``, refused unless it fits."""
        if not isinstance(given, kind):
            raise TypeError(
                f"{name} must be a {kind.__name__}, got {type(given).__name__}"
            )
        flow = given if isinstance(given, FlowState) else given.state
        mesh = self._mesh
        expected = (u_shape(mesh), v_shape(mesh), p_shape(mesh))
        if (flow.u.shape, flow.v.shape, flow.p.shape) != expected:
            ny, nx = flow.p.shape
            raise ValueError(
                f"{name} is a state of a {nx}x{ny} grid and this mesh is "
                f"{mesh.xc.shape[0]}x{mesh.yc.shape[0]}"
            )
        if self._model is None and flow.turbulent:
            raise ValueError(
                f"{name} carries k, eps and nu_t and this solve is laminar (no "
                "turbulence section)"
            )
        if self._model is not None and not flow.turbulent:
            raise ValueError(
                f"{name} has no k and eps and this solve is coupled (the "
                "turbulence section is present)"
            )
        return flow

    def _warm(
        self, initial: FlowState
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, TurbulenceState | None]:
        """Writable copies of a state's fields, k and eps checked (ADR-014 A)."""
        flow = self._check_state(initial, "initial", FlowState)
        state = None
        if self._model is not None:
            assert flow.k is not None and flow.eps is not None
            state = self._model.state(np.array(flow.k), np.array(flow.eps))
        return np.array(flow.u), np.array(flow.v), np.array(flow.p), state

    def _resumed(
        self, resume: SteadyCheckpoint, digest: str | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, TurbulenceState | None]:
        """A checkpoint's fields as they stood, the relaxed nu_t kept (ADR-014 D)."""
        flow = self._check_state(resume, "resume", SteadyCheckpoint)
        if resume.eddy_viscosity_sha256 != digest:
            raise ValueError(
                "resume continues a solve under the eddy_viscosity it ran under: "
                f"the checkpoint's field has SHA-256 {resume.eddy_viscosity_sha256} "
                f"and this solve's {digest} (None is no field)"
            )
        if flow.stop_reason in _STOP_REASONS.values():
            raise ValueError(
                "resume continues a solve, and this checkpoint's stopped converged "
                f"({flow.stop_reason}); warm-start from its state with initial="
            )
        if resume.rule_version != self._rule_version:
            raise ValueError(
                f"resume's rule version is {resume.rule_version} and this solve "
                f"applies {self._rule_version}"
            )
        state = None
        if self._model is not None:
            assert flow.k is not None and flow.eps is not None
            assert flow.nu_t is not None
            checked = self._model.state(np.array(flow.k), np.array(flow.eps))
            if np.any(flow.nu_t[self._live] < 0.0):
                raise ValueError("resume's nu_t is negative in a non-SOLID cell")
            state = TurbulenceState(k=checked.k, eps=checked.eps, nu_t=flow.nu_t)
        return np.array(flow.u), np.array(flow.v), np.array(flow.p), state

    def _turbulence_step(
        self, state: TurbulenceState, u: np.ndarray, v: np.ndarray, iteration: int
    ) -> TurbulenceState:
        """One k and eps step on the corrected faces, then relaxed nu_t (ADR-012 D)."""
        assert self._model is not None and self._walls is not None
        faces = FaceVelocities.copy_of(u, v)
        try:
            stepped = self._model.step(
                state, faces, self._walls.conditions(state, faces)
            )
        except PositivityError as err:
            raise PositivityError(
                f"outer iteration {iteration}: {err}", err.minimum
            ) from err
        a_t = self._alpha_turbulence
        nu_t = (1.0 - a_t) * state.nu_t + a_t * stepped.nu_t
        nu_t.setflags(write=False)
        return TurbulenceState(k=stepped.k, eps=stepped.eps, nu_t=nu_t)
