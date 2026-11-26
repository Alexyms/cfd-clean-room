"""A scenario run whose march steps through the compiled core (prompt 91, ADR-017 B).

``CoreRun`` is ``time_integration._Run`` with its ``march`` replaced: the plans,
the events, the flow states, the alert monitor, the frames, the checkpoints, the
restart and the end records stay the reference's own code, and each interval is
handed to a ``_cfdcore.March2``, which steps it with the fields resident and
returns per step what ADR-017 B's exchange table names. Python calls the core
between output steps: to the next frame, checkpoint or interval end, or one step
at a time when something reads the fields every step (``on_step``, or the
monitor while ``monitor_reads_fields`` is set, as it reads them today).

``run`` calls ``time_integration.run`` with ``CoreRun`` in place of ``_Run`` for
the call, so a scenario runs as the reference's ``run`` runs it and only the
march differs. Step 6 moves this into ``src/time_integration.py`` behind the
``solver.backend`` key; until then the parity tests and prompt 91's probe share
it here, and nothing under ``src/`` changes.

Prompt 95 adds the steady solve: ``CoreSolver`` is ``StaggeredSolver`` with
``solve_steady``'s outer iterations in a ``_cfdcore.Flow2``, chunked so that the
reference's checkpoints, callbacks and log lines fall where they fall in its loop,
and ``steady_solves_in_the_core`` has ``time_integration`` build it in place of the
reference's solver.
"""

import contextlib
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING
from unittest import mock

import numpy as np

import src.time_integration as time_integration
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.io_manager import RunFile
from src.mesh import Mesh
from src.monitor import Alert
from src.solver_staggered import (
    _STOP_REASONS,
    CAP_STOP,
    FlowState,
    StaggeredSolver,
    SteadyCheckpoint,
    _digest,
)
from src.solver_staggered import logger as solver_logger
from src.solver_transport import MassBudget
from src.staggered import FaceVelocities, allocate_fields, to_cell_centers
from src.stopping import IterationState
from src.time_integration import (
    IntervalPlan,
    MarchStep,
    RunRecord,
    TransientRunRecord,
    _copies,
    _Run,
)
from src.turbulence import TurbulenceState
from src.turbulence import logger as turbulence_logger
from tests import core_mapping

if TYPE_CHECKING:
    import _cfdcore


class CoreRun(_Run):
    """``_Run`` with the march stepped through the core.

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
    ``monitor_reads_fields`` (class attribute, True) keeps the monitor on the
    fields: they are read from the core after every step and handed to
    ``AlertMonitor.update``, as the reference hands them, and the core's
    readings and channels are checked against the monitor's bit for bit. False
    hands the monitor the core's readings instead, on fields that hold them at
    the sensors' cells and zero elsewhere (the monitor reads nothing else), so
    the march goes to the core in batches. ``core_calls`` counts the core's
    ``march`` calls.
    """

    monitor_reads_fields: bool = True

    def __init__(
        self,
        config: SimConfig,
        run: RunFile,
        on_step: Callable[[MarchStep], None] | None,
    ) -> None:
        super().__init__(config, run, on_step)
        recovery = None
        if self.tracker is not None:
            spec = config.scenario
            assert spec is not None
            initial = self.manager.initial_fields()
            recovery = core_mapping.recovery_keys(
                config.class_thresholds("the clean-out's recovery"),
                [float(initial[k][self.live][0]) for k in range(self.n_classes)],
                spec.hold_time,
            )
        self.core = core_mapping.core_march(
            config, self.mesh, self.cells, [onset for _, onset in self.events], recovery
        )
        self.core_calls = 0
        self._stand_in: dict[int, np.ndarray] | None = None

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

    def _readings_fields(self, readings: np.ndarray) -> dict[int, np.ndarray]:
        """Fields holding the core's readings at the sensors' cells and zero elsewhere."""
        if self._stand_in is None:
            shape = self.mesh.cell_type.shape
            self._stand_in = {k: np.zeros(shape) for k in range(self.n_classes)}
        rows = [j for j, _ in self.cells]
        cols = [i for _, i in self.cells]
        for k, field in self._stand_in.items():
            field[rows, cols] = readings[k]
        return self._stand_in

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
            return self.observe(
                step, t, self._readings_fields(result["readings"][r]), pending
            )
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
        core_mapping.restore_march(
            core, fields, budgets, step, t, self.arrival, self.tracker
        )
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
            core_mapping.set_interval(
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
                    current = tuple(core_mapping.budgets_of_row(result["budgets"][r]))
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
                        fields_now = core_mapping.unstacked(core.class_fields())
                        fields = fields_now
                    monitor_fields = fields_now if self.monitor_reads_fields else None
                    alerts = self._observe_row(
                        step, t, result, r, monitor_fields, pending
                    )
                    if frame or checkpoint:
                        core_mapping.sync_trackers(core, self.arrival, self.tracker)
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
                error = core_mapping.breach_error(result)
                if error is not None:
                    raise error
                # Read once per call: a march that took no step and held ends the loop.
                held = stopped == "held"
        fields = core_mapping.unstacked(core.class_fields())
        current = tuple(core_mapping.budgets_of(core))
        core_mapping.sync_trackers(core, self.arrival, self.tracker)
        self._flush(pending)
        return fields, _copies(current), int(core.step), float(core.t), held


def run(
    config: SimConfig,
    run_file: str | os.PathLike[str] | None = None,
    *,
    restart: str | os.PathLike[str] | None = None,
    flows_from: str | os.PathLike[str] | None = None,
    on_step: Callable[[MarchStep], None] | None = None,
    provenance: time_integration.Provenance | None = None,
    monitor_reads_fields: bool = True,
) -> RunRecord | TransientRunRecord:
    """``time_integration.run`` with the march stepped through the core.

    Parameters
    ----------
    config, run_file, restart, flows_from, on_step, provenance
        As ``time_integration.run`` takes them (tier 1).
    monitor_reads_fields : bool
        ``CoreRun.monitor_reads_fields`` for the call.

    Returns
    -------
    RunRecord or TransientRunRecord
        ``time_integration.run``'s record.
    """
    driver = type(
        "CoreRunForCall", (CoreRun,), {"monitor_reads_fields": monitor_reads_fields}
    )
    with mock.patch.object(time_integration, "_Run", driver):
        return time_integration.run(
            config,
            run_file,
            restart=restart,
            flows_from=flows_from,
            on_step=on_step,
            provenance=provenance,
        )


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
        super().__init__(mesh, config, boundary)
        self._config = config
        self._modules = core_mapping.flow_modules(
            mesh, config, boundary, self._corrector
        )
        self.rule_readings: dict[str, list] = {}
        self.core_seconds: dict[str, float] = {}
        self.core_calls = 0

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

        core = core_mapping.flow_solver(
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
                raise core_mapping.flow_error(failure)
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


@contextlib.contextmanager
def steady_solves_in_the_core() -> Iterator[None]:
    """Every ``StaggeredSolver`` that ``time_integration`` builds a ``CoreSolver``."""
    with mock.patch.object(time_integration, "StaggeredSolver", CoreSolver):
        yield
