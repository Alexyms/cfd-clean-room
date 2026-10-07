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

The faces of the last solve are kept as ``face_velocities`` (REQ-S13):
the transport solver advects with them because the cell means the contract
returns do not carry the continuity the stopping rule enforced on the faces.

``stopping_rule`` picks the stop. ``velocity_step``, the default, is the
collocated rule: the residual below ``convergence_tol``. ``error_estimate``
(src/stopping.py) needs the estimated iteration error over the largest
prescribed boundary velocity below ``iteration_error_tol``, the worst
per-cell mass imbalance below ``mass_imbalance_tol``, the summed imbalance
over the through-flow below ``iteration_error_tol``, and the absolute signed
domain sum of the imbalance below ``mass_imbalance_tol``. Reaching
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
"""

import logging
from collections.abc import Callable
from functools import partial
from time import perf_counter

import numpy as np

from src.boundary_staggered import StaggeredBoundary
from src.config import ERROR_ESTIMATE, VELOCITY_STEP, SimConfig
from src.mesh import FLUID, Mesh
from src.momentum import MomentumPredictor
from src.pressure import ZERO_SCALE, PressureCorrector
from src.staggered import FaceVelocities, allocate_fields, p_shape, to_cell_centers
from src.stopping import ErrorEstimateRule, ImbalanceSummary, IterationState

logger = logging.getLogger(__name__)

_STOP_REASONS = {
    VELOCITY_STEP: "velocity_step_below_tol",
    ERROR_ESTIMATE: "error_estimate_and_continuity",
}


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
        Scaled residual after each outer iteration of the last solve.
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

    Raises
    ------
    ValueError
        Under error_estimate, if no boundary prescribes a velocity to scale by.
        The message names stopping_rule and the boundaries.
    """

    def __init__(
        self, mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary
    ) -> None:
        self._mesh = mesh
        self._boundary = boundary
        self._predictor = MomentumPredictor(mesh, config, boundary)
        self._corrector = PressureCorrector(mesh, config, boundary)
        self._rho = config.rho
        self._convergence_tol = config.convergence_tol
        self._max_simple_iter = config.max_simple_iter
        self._stopping_rule = config.stopping_rule
        self._rule_tols = (config.iteration_error_tol, config.mass_imbalance_tol)
        self._fluid = mesh.cell_type == FLUID
        self._outlets = boundary.pressure_outlets()

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
        self._new_rule()

    @property
    def flux_scale(self) -> float | None:
        """The error_estimate rule's flux scale, kg/s per unit depth; None otherwise."""
        return self._flux_scale

    @staticmethod
    def _zero_stage_seconds() -> dict[str, float]:
        return {"momentum": 0.0, "pressure": 0.0, "correct": 0.0}

    def _reference_velocity(self) -> float:
        """The collocated solver's reference velocity, from the staggered layer's inputs."""
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
        the bound on the summed imbalance.
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
        return ErrorEstimateRule(scale, self._flux_scale, *self._rule_tols)

    def _imbalance_summary(self, u: np.ndarray, v: np.ndarray) -> ImbalanceSummary:
        """Worst, absolute-summed and signed-summed per-cell imbalance, one evaluation."""
        imbalance = self._corrector.mass_imbalance(u, v)
        cells = np.abs(imbalance)
        return ImbalanceSummary(
            worst=float(cells.max()),
            absolute_sum=float(cells.sum()),
            signed_sum=float(imbalance.sum()),
        )

    def _extrapolate_outlets(self, u: np.ndarray, v: np.ndarray) -> None:
        """Give each pressure outlet face the value of its interior neighbour, in place."""
        left = self._outlets["left"].is_outlet
        right = self._outlets["right"].is_outlet
        bottom = self._outlets["bottom"].is_outlet
        top = self._outlets["top"].is_outlet
        u[left, 0] = u[left, 1]
        u[right, -1] = u[right, -2]
        v[0, bottom] = v[1, bottom]
        v[-1, top] = v[-2, top]

    def solve_steady(
        self, on_iteration: Callable[[IterationState], None] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve for steady-state velocity and pressure fields.

        Parameters
        ----------
        on_iteration : Callable[[IterationState], None], optional
            Called after every outer iteration with the cell-centered u and
            v, the pressure, the residual and the corrector's iteration
            count for that iteration. The arrays are fresh each iteration and
            must not be modified.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (u, v, p) at cell centers, each shape [ny, nx], dtype float64,
            C-contiguous.
        """
        u, v, p = allocate_fields(self._mesh)
        self._boundary.apply_normal_velocity(u, v)

        self.residual_history = []
        self.stage_seconds = self._zero_stage_seconds()
        # Reset with the timers: a solve that stops before its first
        # correction must not report the previous call's counts.
        self.last_pressure_iterations = 0
        self.pressure_cap_hits = 0
        self.converged, self.stop_reason = False, None
        rule = self._new_rule()
        u_c, v_c = to_cell_centers(u, v)
        fluid = self._fluid

        for iteration in range(self._max_simple_iter):
            t0 = perf_counter()
            self._extrapolate_outlets(u, v)
            prediction = self._predictor.predict(u, v, p)
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
                    )
                )

            if rule is None:
                # A capped correction makes the step small without balancing
                # the faces; the velocity-step rule cannot tell, so it is not
                # allowed to stop on one (ADR-013 B).
                stop = residual < self._convergence_tol and not corrected.reached_cap
            else:
                stop = rule.update(max(du, dv), partial(self._imbalance_summary, u, v))

            if iteration % 50 == 0 or stop:
                logger.info("SIMPLE iter %4d: residual = %.6e", iteration, residual)

            if stop:
                self.converged = True
                self.stop_reason = _STOP_REASONS[self._stopping_rule]
                logger.info("Converged at iteration %d", iteration)
                break

        if not self.converged:
            self.stop_reason = "max_simple_iter"
            logger.warning("Not converged: stopped at max_simple_iter")
        self.last_mass_imbalance = self._corrector.mass_imbalance(u, v)
        self.face_velocities = FaceVelocities.copy_of(u, v)
        return u_c, v_c, np.ascontiguousarray(p)
