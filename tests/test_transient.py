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
(results/builder67/plant67.py).
"""

import math
from dataclasses import replace

import numpy as np
import pytest
import yaml

import src.solver_staggered as solver_module
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.pressure import PressureCorrector
from src.solver_staggered import (
    BDF2,
    IMPLICIT_EULER,
    TRANSIENT_STOP,
    FlowState,
    StaggeredSolver,
    TransientStep,
    TransientStepError,
)
from src.staggered import allocate_fields
from tests.cylinder59_reference import load_fields_of, load_implicit_euler_predictor
from tests.helpers import channel_raw, product_config, staggered_solver
from validation.cases import case_path

DT = 0.05
STEP_TOL = 1e-6
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
    return all(np.array_equal(getattr(a, n), getattr(b, n)) for n in ("u", "v", "p"))


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
        probe = staggered_solver(config)
        predictor = load_implicit_euler_predictor()(
            probe._mesh, config, probe._boundary, DT
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


def _converged(raw: dict) -> StaggeredSolver:
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


def _cavity_raw() -> dict:
    raw = yaml.safe_load(case_path("cavity").read_text(encoding="utf-8"))
    raw["domain"]["nx"] = raw["domain"]["ny"] = 16
    return raw


def _channel_raw() -> dict:
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
