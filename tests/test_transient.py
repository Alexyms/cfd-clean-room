"""The time-accurate flow solve, ``StaggeredSolver.solve_transient`` (ADR-014 tier 2, step 6a).

The march's arithmetic is the steady outer iteration with the predictor's
transient term (tests/test_momentum.py's TestTransientTerm pins the term);
what this file pins is the loop around it, on VAL-002's cavity at a few cells:

- The arguments that cannot be a march are refused before any iteration,
  the turbulence section included (tier 2 is laminar, ADR-014 decision 1).
- The first step is implicit Euler, the rest BDF2, and the implicit Euler
  march is the probe's (docs/reports/probe59/cylinder59.py, which drove the
  steady loop one step at a time) to the byte.
- A step that reaches its cap, or goes non-finite, fails the run with the
  step's record and every record before it.
- Each step's record holds what the step did, and ``on_step`` reads without
  changing the march.
- A steady solution is a fixed point of the time step: a march from a
  converged steady state takes one outer iteration per step and moves the
  faces by less than the per-step criterion, on a closed cavity and on a
  channel with a pressure outlet.
- A steady solve after a transient one is the steady solve, to the byte.

The order in time is tests/test_time_order.py's. The planted defects each
test was shown to catch are named in its docstring
(docs/reports/probe67/plant67.py).
"""

import math
from dataclasses import replace

import numpy as np
import pytest
import yaml

import src.solver_staggered as solver_module
from src.config import SimConfig
from src.mesh import FLUID, SOLID, Mesh
from src.pressure import PressureCorrector
from src.solver_staggered import (
    BDF2,
    IMPLICIT_EULER,
    TRANSIENT_STOP,
    FlowState,
    StaggeredSolver,
    TransientResume,
    TransientStep,
    TransientStepError,
)
from src.staggered import allocate_fields, to_cell_centers
from tests.cylinder59_reference import load_fields_of, load_implicit_euler_predictor
from tests.helpers import channel_raw, product_config, staggered_solver
from validation.cases import case_path

DT = 0.05
STEP_TOL = 1e-6
# The channel's step: a twentieth of the 1 m channel's through-flow time at 0.1 m/s.
CHANNEL_DT = 0.5
GOOD_ARGUMENTS = {"dt": DT, "t_end": 2 * DT, "step_tol": STEP_TOL, "step_cap": 50}


def _cavity(n: int = 8, **solver: object) -> SimConfig:
    """VAL-002's cavity on n x n under the velocity-step rule, solver keys replaced."""
    raw = yaml.safe_load(case_path("cavity").read_text(encoding="utf-8"))
    raw["domain"]["nx"] = raw["domain"]["ny"] = n
    raw["solver"]["stopping_rule"] = "velocity_step"
    raw["solver"].update(solver)
    return SimConfig.from_dict(raw)


def _rest(solver: StaggeredSolver) -> FlowState:
    """The state at rest on the solver's grid."""
    u, v, p = allocate_fields(solver._mesh)
    return FlowState(u=u, v=v, p=p)


def _march(
    config: SimConfig, steps: int, scheme: str = BDF2, dt: float = DT
) -> tuple[StaggeredSolver, list[tuple[TransientStep, FlowState]]]:
    """``steps`` steps from rest; the solver and what on_step received."""
    solver = staggered_solver(config)
    seen: list[tuple[TransientStep, FlowState]] = []
    solver.solve_transient(
        dt,
        steps * dt,
        initial=_rest(solver),
        step_tol=STEP_TOL,
        step_cap=200,
        on_step=lambda record, state: seen.append((record, state)),
        scheme=scheme,
    )
    return solver, seen


def _same_state(a: FlowState, b: FlowState) -> bool:
    """Whether two states hold the same faces and pressure, to the byte."""
    return all(np.array_equal(getattr(a, n), getattr(b, n)) for n in ("u", "v", "p"))


def _same_as_the_probe(
    config: SimConfig,
    dt: float,
    seen: list[tuple[TransientStep, FlowState]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert an implicit Euler march from rest is cylinder59.py's loop, step by step.

    The probe solved each step with solve_steady from the last step's faces
    (its allocate_fields patched to hand them over), the steady loop writing
    the boundary's normal velocities at the start and extrapolating the
    outlets before every prediction, with convergence_tol the step
    criterion on reference_velocity's scale. Every step's faces, pressure
    and outer count must agree.
    """
    probe = staggered_solver(config)
    predictor = load_implicit_euler_predictor()(
        probe._mesh, config, probe._boundary, dt
    )
    probe._predictor = predictor
    fields_of = load_fields_of()
    rest = _rest(probe)
    u, v, p = np.array(rest.u), np.array(rest.v), np.array(rest.p)
    for record, state in seen:
        predictor.u_old, predictor.v_old = u.copy(), v.copy()
        monkeypatch.setattr(solver_module, "allocate_fields", fields_of(u, v, p))
        _uc, _vc, p = probe.solve_steady()
        u, v = np.array(probe.face_velocities.u), np.array(probe.face_velocities.v)
        assert probe.converged
        assert len(probe.residual_history) == record.outer
        assert np.array_equal(u, state.u) and np.array_equal(v, state.v)
        assert np.array_equal(p, state.p)


@pytest.mark.unit
class TestArguments:
    """What cannot be a march is refused before any iteration."""

    @pytest.mark.parametrize(
        ("key", "value", "error"),
        [
            ("dt", True, TypeError),
            ("dt", 1, TypeError),
            ("dt", 0.0, ValueError),
            ("dt", -DT, ValueError),
            ("dt", math.nan, ValueError),
            ("dt", math.inf, ValueError),
            ("t_end", False, TypeError),
            ("t_end", 2, TypeError),
            ("t_end", 0.0, ValueError),
            ("t_end", -1.0, ValueError),
            ("t_end", math.nan, ValueError),
            ("t_end", math.inf, ValueError),
            ("t_end", 2.5 * DT, ValueError),
            ("t_end", 0.4 * DT, ValueError),
            ("step_tol", True, TypeError),
            ("step_tol", 1, TypeError),
            ("step_tol", 0.0, ValueError),
            ("step_tol", -1e-6, ValueError),
            ("step_tol", math.nan, ValueError),
            ("step_cap", True, TypeError),
            ("step_cap", 5.0, TypeError),
            ("step_cap", 0, ValueError),
            ("step_cap", -3, ValueError),
            ("scheme", 2, TypeError),
            ("scheme", "crank_nicolson", ValueError),
        ],
    )
    def test_a_bad_argument_is_refused(
        self, key: str, value: object, error: type
    ) -> None:
        """Each of dt, t_end, step_tol, step_cap and scheme, alone, refused by name."""
        solver = staggered_solver(_cavity())
        kwargs = {**GOOD_ARGUMENTS, key: value}
        dt, t_end = kwargs.pop("dt"), kwargs.pop("t_end")
        with pytest.raises(error, match=key):
            solver.solve_transient(dt, t_end, initial=_rest(solver), **kwargs)
        assert solver.transient_steps == []

    def test_a_state_that_is_not_one_of_this_solve_is_refused(self) -> None:
        """Not a FlowState, another grid, or k and eps for a laminar solve."""
        solver = staggered_solver(_cavity())
        other = staggered_solver(_cavity(6))
        rest = _rest(solver)
        k = np.ones_like(rest.p)
        turbulent = FlowState(u=rest.u, v=rest.v, p=rest.p, k=k, eps=k, nu_t=k)
        for initial, error in (
            ((rest.u, rest.v, rest.p), TypeError),
            (_rest(other), ValueError),
            (turbulent, ValueError),
        ):
            with pytest.raises(error, match="initial"):
                solver.solve_transient(
                    DT, DT, initial=initial, step_tol=1e-6, step_cap=5
                )

    def test_the_turbulence_section_is_refused(self) -> None:
        """Tier 2 is laminar (ADR-014 decision 1)."""
        solver = staggered_solver(product_config())
        with pytest.raises(ValueError, match="laminar"):
            solver.solve_transient(DT, DT, initial=None, step_tol=1e-6, step_cap=5)  # type: ignore[arg-type]

    def test_no_prescribed_velocity_to_scale_by_is_refused(self) -> None:
        """A cavity whose lid stands still has no U for the criterion."""
        raw = yaml.safe_load(case_path("cavity").read_text(encoding="utf-8"))
        raw["domain"]["nx"] = raw["domain"]["ny"] = 6
        raw["solver"]["stopping_rule"] = "velocity_step"
        raw["boundaries"]["lid"]["u_velocity"] = 0.0
        solver = staggered_solver(SimConfig.from_dict(raw))
        with pytest.raises(ValueError, match="prescribes"):
            solver.solve_transient(
                DT, DT, initial=_rest(solver), step_tol=1e-6, step_cap=5
            )


@pytest.mark.integration
class TestSteps:
    """The march's steps on the 8x8 cavity from rest."""

    def test_the_first_step_is_implicit_euler(self) -> None:
        """BDF2's first step is the implicit Euler step, to the byte; the rest are BDF2.

        Catches: the first step left BDF2 with a zero old level.
        """
        bdf2, seen_bdf2 = _march(_cavity(), 3)
        _ie, seen_ie = _march(_cavity(), 1, scheme=IMPLICIT_EULER)
        assert [r.scheme for r in bdf2.transient_steps] == [IMPLICIT_EULER, BDF2, BDF2]
        assert _same_state(seen_bdf2[0][1], seen_ie[0][1])
        assert seen_bdf2[0][0] == seen_ie[0][0]

    def test_bdf2_after_the_first_step_is_not_implicit_euler(self) -> None:
        """The second step's state differs from the implicit Euler march's."""
        _bdf2, seen_bdf2 = _march(_cavity(), 2)
        ie, seen_ie = _march(_cavity(), 2, scheme=IMPLICIT_EULER)
        assert [r.scheme for r in ie.transient_steps] == [IMPLICIT_EULER] * 2
        assert not np.array_equal(seen_bdf2[1][1].u, seen_ie[1][1].u)

    def test_implicit_euler_is_the_probes_march_to_the_byte(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cylinder59.py's loop: solve_steady per step from the last step's faces.

        On a closed cavity the probe's criterion, convergence_tol at the step
        tolerance times U over reference_velocity, is the march's: both
        velocities are the lid's. Every step's faces, pressure and outer count
        agree.
        """
        config = _cavity(8, convergence_tol=STEP_TOL, max_simple_iter=200)
        solver, seen = _march(config, 4, scheme=IMPLICIT_EULER)
        assert solver.reference_velocity == 1.0
        _same_as_the_probe(config, DT, seen, monkeypatch)

    def test_a_step_at_its_cap_fails_the_run(self) -> None:
        """The error names the step and carries its record and those before it.

        Catches: the cap ignored.
        """
        solver = staggered_solver(_cavity())
        with pytest.raises(TransientStepError, match="step_cap") as caught:
            solver.solve_transient(
                DT, 3 * DT, initial=_rest(solver), step_tol=1e-14, step_cap=3
            )
        err = caught.value
        assert err.step.step == 1 and err.step.outer == 3
        assert err.steps == (err.step,) and solver.transient_steps == [err.step]
        assert err.step.residual_history[-1] >= 1e-14
        assert solver.flow_state is None and not solver.converged

    def test_a_failed_march_leaves_its_own_imbalance(self) -> None:
        """After a steady solve, a march that fails at its cap replaces the imbalance.

        last_mass_imbalance is the failing step's last faces', whose worst
        non-SOLID value is the step's record, not the steady solve's array;
        face_velocities and stop_reason are cleared (review 67 S6).
        """
        config = _cavity()
        solver = staggered_solver(config)
        solver.solve_steady()
        steady = solver.last_mass_imbalance
        with pytest.raises(TransientStepError) as caught:
            solver.solve_transient(
                DT, DT, initial=_rest(solver), step_tol=1e-14, step_cap=2
            )
        imbalance = solver.last_mass_imbalance
        live = Mesh(config).cell_type != SOLID
        assert imbalance is not steady and imbalance.shape == steady.shape
        assert (
            float(np.max(np.abs(imbalance[live]))) == caught.value.step.worst_imbalance
        )
        assert not np.array_equal(imbalance, steady)
        assert solver.face_velocities is None and solver.stop_reason is None

    def test_a_capped_correction_cannot_end_a_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """From step 3 every correction reports its cap: step 3 runs to step_cap.

        The velocity step falls below the criterion, but a truncated
        correction cannot stop a step (ADR-013 B), so the step fails, and the
        error carries steps 1 and 2 as they ended. Catches: the cap ignored.
        """
        solver = staggered_solver(_cavity())
        correct = solver._corrector.correct
        done: list[int] = []

        def capped_from_step_3(*args: object, **kwargs: object) -> object:
            """The correction, reported as capped once two steps have ended."""
            corrected = correct(*args, **kwargs)
            return replace(corrected, reached_cap=True) if len(done) >= 2 else corrected

        monkeypatch.setattr(solver._corrector, "correct", capped_from_step_3)
        with pytest.raises(TransientStepError, match="time step 3") as caught:
            solver.solve_transient(
                DT,
                4 * DT,
                initial=_rest(solver),
                step_tol=STEP_TOL,
                step_cap=40,
                on_step=lambda record, _state: done.append(record.step),
            )
        err = caught.value
        assert done == [1, 2] and [r.step for r in err.steps] == [1, 2, 3]
        assert err.step is err.steps[-1] and err.step.outer == 40
        assert err.step.pressure_cap_hits == 40
        assert min(err.step.residual_history) < STEP_TOL
        assert solver.pressure_cap_hits == 40

    def test_a_non_finite_step_fails_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prediction that goes non-finite stops the run at that outer iteration."""
        solver = staggered_solver(_cavity())
        predict = solver._predictor.predict

        def poisoned(*args: object, **kwargs: object) -> object:
            """The prediction with one u face made NaN."""
            prediction = predict(*args, **kwargs)
            u_star = prediction.u_star.copy()
            u_star[3, 3] = np.nan
            return replace(prediction, u_star=u_star)

        monkeypatch.setattr(solver._predictor, "predict", poisoned)
        with pytest.raises(TransientStepError, match="not finite") as caught:
            solver.solve_transient(
                DT, DT, initial=_rest(solver), step_tol=STEP_TOL, step_cap=50
            )
        assert caught.value.step.outer == 1

    def test_each_record_holds_what_its_step_did(self) -> None:
        """Count, time, residuals against the criterion, the worst imbalance."""
        config = _cavity()
        solver, seen = _march(config, 4)
        mesh = Mesh(config)
        corrector = PressureCorrector(mesh, config, solver._boundary)
        live = mesh.cell_type != SOLID
        assert solver.transient_steps == [record for record, _state in seen]
        outer = 0
        for n, (record, state) in enumerate(seen, start=1):
            outer += record.outer
            assert record.step == n and record.t == n * DT
            assert record.outer == len(record.residual_history) >= 1
            assert record.residual_history[-1] < STEP_TOL
            assert all(r >= STEP_TOL for r in record.residual_history[:-1])
            assert record.pressure_cap_hits == 0
            worst = np.max(np.abs(corrector.mass_imbalance(state.u, state.v)[live]))
            assert record.worst_imbalance == float(worst)
            assert state.outer == outer and state.stop_reason is None

    def test_the_end_state_and_on_step_reads_only(self) -> None:
        """The return is flow_state and the last step's state; with or without on_step."""
        config = _cavity()
        solver, seen = _march(config, 3)
        quiet = staggered_solver(config)
        end = quiet.solve_transient(
            DT, 3 * DT, initial=_rest(quiet), step_tol=STEP_TOL, step_cap=200
        )
        assert end is quiet.flow_state
        assert _same_state(end, seen[-1][1]) and _same_state(end, solver.flow_state)
        assert end.stop_reason == TRANSIENT_STOP and end.outer == seen[-1][1].outer
        assert quiet.converged and quiet.stop_reason == TRANSIENT_STOP
        assert quiet.residual_history == [] and quiet.steady_checkpoint is None
        assert np.array_equal(quiet.face_velocities.u, end.u)
        assert np.array_equal(quiet.face_velocities.v, end.v)
        assert quiet.transient_steps == solver.transient_steps

    def test_a_steady_solve_after_a_transient_one_is_unchanged(self) -> None:
        """No state of the march reaches the steady path: the same bytes as a fresh solver."""
        config = _cavity(8, convergence_tol=1e-8)
        marched, _seen = _march(config, 2)
        marched.solve_steady()
        fresh = staggered_solver(config)
        fresh.solve_steady()
        assert marched.residual_history == fresh.residual_history
        for name in ("u", "v"):
            assert np.array_equal(
                getattr(marched.face_velocities, name),
                getattr(fresh.face_velocities, name),
            )
        assert np.array_equal(marched.flow_state.p, fresh.flow_state.p)


def _channel_from_rest(**solver: object) -> SimConfig:
    """VAL-001's channel on 20x10: a velocity inlet left, a pressure outlet right."""
    raw = channel_raw()
    raw["domain"]["nx"], raw["domain"]["ny"] = 20, 10
    raw["solver"]["stopping_rule"] = "velocity_step"
    raw["solver"].update(solver)
    return SimConfig.from_dict(raw)


@pytest.mark.integration
class TestBoundaries:
    """The march's own boundary lines, from rest on a channel (review 67 B1).

    A march from a converged state already holds the inlet's normal velocity
    and the outlet's extrapolation, and the cavity has neither, so only a
    march from rest through an inlet and an outlet tells whether
    solve_transient writes the one and applies the other.
    """

    def test_implicit_euler_from_rest_is_the_probes_march_on_a_channel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every step's faces, pressure and count, to the byte, as on the cavity.

        The channel's reference_velocity is not its inlet velocity, so the
        probe's convergence_tol is the step criterion times U over
        reference_velocity. Catches: the normal velocities not written at
        the start, the outlets not extrapolated before each prediction.
        """
        solver = staggered_solver(_channel_from_rest())
        scale = solver._boundary.get_max_boundary_velocity()
        tol = STEP_TOL * scale / solver.reference_velocity
        config = _channel_from_rest(convergence_tol=tol, max_simple_iter=500)
        _marched, seen = _march(config, 4, scheme=IMPLICIT_EULER, dt=CHANNEL_DT)
        _same_as_the_probe(config, CHANNEL_DT, seen, monkeypatch)

    def test_the_predictor_sees_the_inlet_and_the_extrapolated_outlet(self) -> None:
        """BDF2 from rest, every prediction's input faces and every step's end.

        At every prediction the inlet faces hold the prescribed velocity and
        each outlet face its interior neighbour's value, exactly; at every
        step's end the inlet faces still hold it. Catches: the normal
        velocities not written at the start, the outlets not extrapolated
        before each prediction.
        """
        config = _channel_from_rest()
        solver = staggered_solver(config)
        inlet = channel_raw()["boundaries"]["inlet"]["velocity"]
        outlet = solver._outlets["right"].is_outlet
        assert outlet.all()
        predict = solver._predictor.predict
        calls: list[tuple[bool, bool]] = []

        def watched(
            u: np.ndarray, v: np.ndarray, p: np.ndarray, **kwargs: object
        ) -> object:
            """The predictor, after noting what its input faces hold."""
            calls.append(
                (
                    bool(np.all(u[:, 0] == inlet)),
                    bool(np.array_equal(u[outlet, -1], u[outlet, -2])),
                )
            )
            return predict(u, v, p, **kwargs)

        solver._predictor.predict = watched  # type: ignore[method-assign]
        ends: list[FlowState] = []
        solver.solve_transient(
            CHANNEL_DT,
            4 * CHANNEL_DT,
            initial=_rest(solver),
            step_tol=STEP_TOL,
            step_cap=500,
            on_step=lambda _record, state: ends.append(state),
        )
        assert [r.scheme for r in solver.transient_steps] == [IMPLICIT_EULER] + [
            BDF2
        ] * 3
        assert len(calls) == sum(r.outer for r in solver.transient_steps)
        assert all(at_inlet for at_inlet, _at_outlet in calls)
        assert all(at_outlet for _at_inlet, at_outlet in calls)
        assert all(np.all(state.u[:, 0] == inlet) for state in ends)
        # The flow developed from rest: the interior moves at the inlet's scale.
        assert np.max(np.abs(ends[-1].u[:, 1:-1])) > 0.5 * inlet


@pytest.mark.integration
class TestResume:
    """A march continued from a step's two levels is the uninterrupted march (step 6b)."""

    @pytest.mark.parametrize("scheme", [BDF2, IMPLICIT_EULER])
    def test_a_resumed_march_is_the_uninterrupted_one_to_the_byte(
        self, scheme: str
    ) -> None:
        """From rest on the channel: six steps, against three and three resumed.

        Catches: the previous level not carried (BDF2 restarted with implicit
        Euler), the levels swapped, the step count restarted at 1.
        """
        config = _channel_from_rest()
        whole, seen = _march(config, 6, scheme=scheme, dt=CHANNEL_DT)
        state_2, state_3 = seen[1][1], seen[2][1]
        resumed = staggered_solver(config)
        later: list[tuple[TransientStep, FlowState]] = []
        end = resumed.solve_transient(
            CHANNEL_DT,
            6 * CHANNEL_DT,
            resume=TransientResume(state=state_3, previous=state_2, step=3),
            step_tol=STEP_TOL,
            step_cap=200,
            on_step=lambda record, state: later.append((record, state)),
            scheme=scheme,
        )
        assert [r.step for r, _s in later] == [4, 5, 6]
        for (mine, state), (theirs, kept) in zip(later, seen[3:], strict=True):
            assert mine == theirs
            assert _same_state(state, kept) and state.outer == kept.outer
        assert _same_state(end, whole.flow_state)

    def test_an_implicit_euler_resume_needs_no_previous_level(self) -> None:
        """Implicit Euler reads one old level, so ``previous=None`` resumes exactly.

        Catches: the previous level read when it is None, and a resume that
        refuses a march whose scheme has no use for the second level.
        """
        config = _channel_from_rest()
        whole, seen = _march(config, 6, scheme=IMPLICIT_EULER, dt=CHANNEL_DT)
        end = staggered_solver(config).solve_transient(
            CHANNEL_DT,
            6 * CHANNEL_DT,
            resume=TransientResume(state=seen[2][1], previous=None, step=3),
            step_tol=STEP_TOL,
            step_cap=200,
            scheme=IMPLICIT_EULER,
        )
        assert _same_state(end, whole.flow_state)

    def test_a_resume_at_step_zero_is_a_new_march(self) -> None:
        """A BDF2 resume at step 0 has no previous level and takes the first step.

        The first step of a BDF2 march is implicit Euler, so the resume at
        step 0 from the rest state must be the march begun with ``initial``.
        Catches: a resume at step 0 refused for the missing level, and the
        step count of a resume restarted from the wrong index.
        """
        config = _channel_from_rest()
        whole, _seen = _march(config, 3, dt=CHANNEL_DT)
        resumed = staggered_solver(config)
        end = resumed.solve_transient(
            CHANNEL_DT,
            3 * CHANNEL_DT,
            resume=TransientResume(state=_rest(resumed), previous=None, step=0),
            step_tol=STEP_TOL,
            step_cap=200,
        )
        assert [r.step for r in resumed.transient_steps] == [1, 2, 3]
        assert _same_state(end, whole.flow_state)

    def test_what_cannot_be_resumed_is_refused(self) -> None:
        config = _channel_from_rest()
        _whole, seen = _march(config, 2, dt=CHANNEL_DT)
        solver = staggered_solver(config)
        rest = _rest(solver)
        cases = [
            (
                {"initial": rest, "resume": TransientResume(seen[1][1], seen[0][1], 2)},
                ValueError,
                "exactly one",
            ),
            ({}, ValueError, "exactly one"),
            (
                {"resume": TransientResume(seen[1][1], seen[0][1], 4)},
                ValueError,
                "later t_end",
            ),
            (
                {"resume": TransientResume(seen[1][1], None, 2)},
                ValueError,
                "previous level",
            ),
            ({"resume": (seen[1][1], seen[0][1], 2)}, TypeError, "TransientResume"),
        ]
        for kwargs, error, match in cases:
            with pytest.raises(error, match=match):
                solver.solve_transient(
                    CHANNEL_DT,
                    4 * CHANNEL_DT,
                    step_tol=STEP_TOL,
                    step_cap=200,
                    **kwargs,
                )

    @pytest.mark.parametrize(
        ("state", "previous", "step", "error"),
        [
            ("rest", "rest", -1, ValueError),
            ("rest", "rest", True, TypeError),
            ("rest", "rest", 1.0, TypeError),
            ((), "rest", 1, TypeError),
            ("rest", (), 1, TypeError),
            ("rest", "other", 1, ValueError),
        ],
    )
    def test_a_resume_point_that_is_not_one_is_refused(
        self, state: object, previous: object, step: object, error: type
    ) -> None:
        here = _rest(staggered_solver(_channel_from_rest()))
        other = _rest(staggered_solver(_cavity(6)))
        given = {"rest": here, "other": other}
        with pytest.raises(error):
            TransientResume(
                state=given.get(state, state) if isinstance(state, str) else state,  # type: ignore[arg-type]
                previous=given.get(previous, previous)
                if isinstance(previous, str)
                else previous,  # type: ignore[arg-type]
                step=step,  # type: ignore[arg-type]
            )


def _converged(raw: dict[str, object]) -> StaggeredSolver:
    """A steady solve to the velocity step 1e-12, the pressure to 1e-10."""
    raw["solver"].update(
        stopping_rule="velocity_step",
        convergence_tol=1e-12,
        pressure_rtol=1e-10,
        max_simple_iter=20000,
    )
    solver = staggered_solver(SimConfig.from_dict(raw))
    solver.solve_steady()
    assert solver.stop_reason == "velocity_step_below_tol"
    return solver


def _cavity_raw() -> dict[str, object]:
    """VAL-002's cavity on 16x16, as a mapping."""
    raw = yaml.safe_load(case_path("cavity").read_text(encoding="utf-8"))
    raw["domain"]["nx"] = raw["domain"]["ny"] = 16
    return raw


def _channel_raw() -> dict[str, object]:
    """VAL-001's channel on 20x10, as a mapping."""
    raw = channel_raw()
    raw["domain"]["nx"], raw["domain"]["ny"] = 20, 10
    return raw


@pytest.fixture(scope="module")
def steady() -> dict[str, StaggeredSolver]:
    """The 16x16 cavity and the 20x10 channel, each converged."""
    return {"cavity": _converged(_cavity_raw()), "channel": _converged(_channel_raw())}


@pytest.mark.integration
class TestFixedPoint:
    """A steady solution is a fixed point of the time step."""

    @pytest.mark.parametrize("case", ["cavity", "channel"])
    @pytest.mark.parametrize("scheme", [BDF2, IMPLICIT_EULER])
    def test_a_march_from_a_steady_state_stays_there(
        self, steady: dict[str, StaggeredSolver], case: str, scheme: str
    ) -> None:
        """Five steps of 0.1 s: one outer iteration each, faces within the criterion.

        The steady solve stopped at a velocity step of 1e-12 of U; each time
        step's first outer iteration then moves the field by about that, so
        it meets a criterion of 1e-10 at once and the faces drift by less
        than 1e-10 U over the march. Catches: the term on the source only,
        the BDF2 weights perturbed.
        """
        solver = steady[case]
        start = solver.flow_state
        scale = solver._boundary.get_max_boundary_velocity()
        step_tol = 1e-10
        end = solver.solve_transient(
            0.1, 0.5, initial=start, step_tol=step_tol, step_cap=50, scheme=scheme
        )
        assert [r.outer for r in solver.transient_steps] == [1] * 5
        drift = max(np.max(np.abs(end.u - start.u)), np.max(np.abs(end.v - start.v)))
        assert drift < step_tol * scale

    def test_the_criterion_is_a_fraction_of_the_prescribed_velocity(
        self, steady: dict[str, StaggeredSolver]
    ) -> None:
        """A step's residual is its cell-centred change over U, not reference_velocity.

        On the channel the reference velocity is the inflow over one cell
        spacing, ten times the inlet's, so the two scales tell apart. Each
        one-outer step's recorded residual is computed from the two states
        on either side of it. Catches: step_tol scaled by reference_velocity.
        """
        solver = steady["channel"]
        scale = solver._boundary.get_max_boundary_velocity()
        assert solver.reference_velocity > 5.0 * scale
        states = [solver.flow_state]
        solver.solve_transient(
            0.1,
            0.3,
            initial=states[0],
            step_tol=1e-10,
            step_cap=50,
            on_step=lambda _record, state: states.append(state),
        )
        fluid = solver._mesh.cell_type == FLUID
        for record, before, after in zip(
            solver.transient_steps, states[:-1], states[1:], strict=True
        ):
            assert record.outer == 1
            change = [
                np.max(np.abs(a[fluid] - b[fluid]))
                for a, b in zip(
                    to_cell_centers(after.u, after.v),
                    to_cell_centers(before.u, before.v),
                    strict=True,
                )
            ]
            assert record.residual_history[0] == pytest.approx(
                max(change) / scale, rel=1e-12
            )
