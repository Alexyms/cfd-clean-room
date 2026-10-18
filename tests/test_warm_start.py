"""The staggered solver's warm start and steady checkpoint (ADR-014 A, D; G step 0).

``solve_steady(initial=...)`` starts a solve from a FlowState in place of
rest; ``solve_steady(resume=...)`` continues the solve a SteadyCheckpoint
was taken from. The two fixtures are the smallest committed rooms that stop
by the error_estimate rule in seconds: the VAL-001 channel on 40x20
(laminar, version 3 of the rule, 408 outer iterations) and the product file
on 40x15 (the coupled k-epsilon solve, version 5, 353 outer iterations).

What is pinned, with the planted defect each test was shown to catch in its
docstring:

- FlowState holds read-only copies and refuses what cannot be one state.
- The start from rest is one case of the warm start: a FlowState holding
  rest and the inlets' uniform k and eps gives the solve from rest bitwise,
  so ``initial=None`` and the warm path run one loop.
- The argument is the probe's start (docs/reports/probe59/warm59.py, which
  patched the two allocations): bitwise the same iterates.
- A converged state warm-started under its own boundaries stops when the
  rule's window first fills, with the faces moved by less than the
  iteration error the rule allowed.
- A warm start of the wrong grid, or with turbulence present or absent
  against the solve, is refused before any iteration.
- A solve cut at a cap and resumed reaches, bitwise, the uninterrupted
  solve: every field, the residual history, the stop and its iteration.
- A solve whose fields went non-finite returns them and keeps no state: its
  flow_state and steady_checkpoint are None, not the previous solve's.
- A resume refuses the checkpoint it cannot continue: a negative nu_t in a
  live cell, k or eps not positive, a rule version or a count that is not
  one, an eddy viscosity digest that is not one.
- With ``checkpoint_interval`` set, ``on_checkpoint`` receives at every
  multiple of it what a solve capped there leaves, counted from a resumed
  solve's checkpoint, never on the converging iteration or a non-finite
  one, and the solve is the same with or without it (ADR-014 step 3; the
  restart through the run's checkpoint file is tests/test_io_manager.py's).
"""

from dataclasses import replace

import numpy as np
import pytest
import yaml

import src.solver_staggered as solver_module
from src.boundary_staggered import StaggeredBoundary
from src.config import ERROR_ESTIMATE, VELOCITY_STEP, SimConfig
from src.mesh import SOLID, Mesh
from src.momentum import MomentumPrediction
from src.pressure import PressureCorrection
from src.solver_staggered import (
    CAP_STOP,
    FlowState,
    StaggeredSolver,
    SteadyCheckpoint,
)
from src.stopping import RATE_WINDOW
from src.turbulence import TurbulenceState
from validation.cases import case_path

FIELDS = ("u", "v", "p", "k", "eps", "nu_t")


def _channel_raw(cap: int = 5000, interval: int | None = None) -> dict:
    """VAL-001's channel on 40x20 under its own error_estimate rule, as a mapping.

    INTERVAL, when given, is ``solver.checkpoint_interval``.
    """
    raw = yaml.safe_load(case_path("poiseuille").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = 40, 20
    raw["solver"]["max_simple_iter"] = cap
    if interval is not None:
        raw["solver"]["checkpoint_interval"] = interval
    return raw


def _channel(cap: int = 5000, interval: int | None = None) -> SimConfig:
    """``_channel_raw`` loaded."""
    return SimConfig.from_dict(_channel_raw(cap, interval))


def _product(
    cap: int = 5000, inlet: float | None = None, interval: int | None = None
) -> SimConfig:
    """``_product_raw`` loaded."""
    return SimConfig.from_dict(_product_raw(cap, inlet, interval))


def _product_raw(
    cap: int = 5000, inlet: float | None = None, interval: int | None = None
) -> dict:
    """The product file on 40x15, its imbalance tolerance by the file's formula."""
    raw = yaml.safe_load(case_path("product").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = 40, 15
    cell = (raw["domain"]["width"] / 40) * (raw["domain"]["height"] / 15)
    raw["solver"]["mass_imbalance_tol"] = (
        1e-4 * raw["fluid"]["density"] * cell / raw["solver"]["t_end"]
    )
    raw["solver"]["max_simple_iter"] = cap
    if inlet is not None:
        raw["boundaries"]["hepa_supply"]["velocity"] = inlet
    if interval is not None:
        raw["solver"]["checkpoint_interval"] = interval
    return raw


def _cavity(rule: str, cap: int = 8, interval: int | None = None) -> SimConfig:
    """VAL-002's cavity on 6x6 under either stopping rule, a few outer iterations."""
    raw = yaml.safe_load(case_path("cavity").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = 6, 6
    raw["solver"]["max_simple_iter"] = cap
    raw["solver"]["stopping_rule"] = rule
    if interval is not None:
        raw["solver"]["checkpoint_interval"] = interval
    return SimConfig.from_dict(raw)


FIXTURES = {"channel": _channel, "product": _product}
RAW_FIXTURES = {"channel": _channel_raw, "product": _product_raw}


def _solver(config: SimConfig) -> StaggeredSolver:
    mesh = Mesh(config)
    return StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))


def _solved(config: SimConfig, **kwargs: object) -> StaggeredSolver:
    solver = _solver(config)
    solver.solve_steady(**kwargs)
    return solver


def _same_bits(a: StaggeredSolver, b: StaggeredSolver) -> None:
    """Every field, the history and the stop of two solves, bit for bit."""
    assert a.flow_state is not None and b.flow_state is not None
    for name in FIELDS:
        x, y = getattr(a.flow_state, name), getattr(b.flow_state, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert x.tobytes() == y.tobytes(), name
    assert a.residual_history == b.residual_history
    assert (a.stop_reason, a.converged) == (b.stop_reason, b.converged)
    assert a.flow_state.outer == b.flow_state.outer == len(b.residual_history)
    assert a.pressure_cap_hits == b.pressure_cap_hits


@pytest.fixture(scope="module")
def converged() -> dict[str, StaggeredSolver]:
    """Each fixture solved from rest to its rule's stop."""
    return {name: _solved(make()) for name, make in FIXTURES.items()}


@pytest.fixture(scope="module")
def capped_checkpoint() -> SteadyCheckpoint:
    """The product fixture cut at 5 outer iterations, for edits into bad ones."""
    return _solved(_product(cap=5)).steady_checkpoint


def _live_cell() -> tuple[int, int]:
    """The first non-SOLID cell of the product fixture's mesh, as an index."""
    live = np.argwhere(Mesh(_product()).cell_type != SOLID)[0]
    return int(live[0]), int(live[1])


@pytest.mark.unit
class TestFlowState:
    def _arrays(self, ny: int = 3, nx: int = 4) -> dict[str, np.ndarray]:
        return {
            "u": np.zeros((ny, nx + 1)),
            "v": np.zeros((ny + 1, nx)),
            "p": np.zeros((ny, nx)),
        }

    def test_the_arrays_are_read_only_copies(self) -> None:
        """Defect caught: the caller's array kept, so a later write changes it."""
        arrays = self._arrays()
        state = FlowState(
            **arrays, k=np.ones((3, 4)), eps=np.ones((3, 4)), nu_t=np.ones((3, 4))
        )
        arrays["u"][0, 0] = 5.0
        assert state.u[0, 0] == 0.0
        for name in FIELDS:
            assert not getattr(state, name).flags.writeable
        assert state.turbulent and not FlowState(**self._arrays()).turbulent

    @pytest.mark.parametrize(
        ("change", "error", "match"),
        [
            ({"u": np.zeros((3, 4))}, ValueError, "are not the faces of p"),
            ({"v": np.zeros((3, 4))}, ValueError, "are not the faces of p"),
            ({"p": np.zeros(12)}, ValueError, "must have 2 dimensions"),
            ({"u": np.zeros((3, 5), dtype=np.float32)}, TypeError, "float64 ndarray"),
            ({"p": [[0.0] * 4] * 3}, TypeError, "float64 ndarray"),
            ({"p": np.full((3, 4), np.nan)}, ValueError, "finite"),
            ({"k": np.ones((3, 4))}, ValueError, "together or not at all"),
            (
                {"k": np.ones((3, 4)), "eps": np.ones((3, 4)), "nu_t": np.ones((4, 3))},
                ValueError,
                "must have p's shape",
            ),
            ({"stop_reason": 3}, TypeError, "stop_reason must be a str"),
            ({"outer": True}, TypeError, "outer must be an int"),
            ({"outer": -1}, ValueError, "outer must be non-negative"),
        ],
    )
    def test_what_cannot_be_one_state_is_refused(
        self, change: dict, error: type, match: str
    ) -> None:
        """Defects caught: each refusal removed (shape, dtype, NaN, trio, counts)."""
        with pytest.raises(error, match=match):
            FlowState(**(self._arrays() | change))


@pytest.mark.unit
class TestSteadyCheckpoint:
    def _fields(self, **change: object) -> dict:
        """A valid checkpoint's arguments (version 3, nothing run), with CHANGE."""
        state = FlowState(np.zeros((3, 5)), np.zeros((4, 4)), np.zeros((3, 4)))
        return {
            "state": state,
            "residual_history": (),
            "velocity_steps": (),
            "viscosity_steps": (),
            "rule_version": 3,
            "pressure_cap_hits": 0,
        } | change

    @pytest.mark.parametrize(
        ("change", "error", "match"),
        [
            ({"rule_version": True}, TypeError, "rule_version must be an int"),
            ({"rule_version": 3.0}, TypeError, "rule_version must be an int"),
            ({"rule_version": -1}, ValueError, "rule_version must be non-negative"),
            (
                {"pressure_cap_hits": True},
                TypeError,
                "pressure_cap_hits must be an int",
            ),
            (
                {"pressure_cap_hits": 2.0},
                TypeError,
                "pressure_cap_hits must be an int",
            ),
            (
                {"pressure_cap_hits": -1},
                ValueError,
                "pressure_cap_hits must be non-negative",
            ),
        ],
    )
    def test_a_rule_version_or_cap_hit_count_that_is_not_one_is_refused(
        self, change: dict, error: type, match: str
    ) -> None:
        """The unchanged checkpoint is valid, as is a None rule version (velocity_step).

        Defects caught: the rule_version check removed; the pressure_cap_hits
        check removed. A bool, a float or a negative count would then reach a
        resume, which compares the version and carries the count on.
        """
        SteadyCheckpoint(**self._fields())
        SteadyCheckpoint(**self._fields(rule_version=None))
        with pytest.raises(error, match=match):
            SteadyCheckpoint(**self._fields(**change))

    @pytest.mark.parametrize(
        ("digest", "error"),
        [
            (True, TypeError),
            (b"0" * 64, TypeError),
            ("0" * 63, ValueError),
            ("A" * 64, ValueError),
            ("g" * 64, ValueError),
            (" " + "0" * 63, ValueError),
        ],
    )
    def test_an_eddy_viscosity_digest_that_is_not_one_is_refused(
        self, digest: object, error: type
    ) -> None:
        """None and 64 lowercase hexadecimal digits are digests; nothing else is.

        Defect caught: the digest check removed, so a digest read back from a
        damaged file as bytes or a short string reaches the resume's
        comparison, which then refuses every field with a misleading message.
        """
        SteadyCheckpoint(**self._fields(eddy_viscosity_sha256=None))
        SteadyCheckpoint(**self._fields(eddy_viscosity_sha256="0123456789abcdef" * 4))
        with pytest.raises(error, match="eddy_viscosity_sha256 must be"):
            SteadyCheckpoint(**self._fields(eddy_viscosity_sha256=digest))


@pytest.mark.integration
class TestTheSolveState:
    def test_a_solve_leaves_its_state(self, converged: dict) -> None:
        """flow_state is the faces, pressure and k, eps, nu_t the solver exposes.

        Defects caught: the flow state built from the pre-correction faces;
        the inlet-uniform k in place of the solve's; the outer count off by one.
        """
        for solver in converged.values():
            state = solver.flow_state
            assert state.u.tobytes() == solver.face_velocities.u.tobytes()
            assert state.v.tobytes() == solver.face_velocities.v.tobytes()
            assert state.stop_reason == solver.stop_reason
            assert state.outer == len(solver.residual_history)
            turbulence = solver.turbulence_state
            assert state.turbulent == (turbulence is not None)
            if turbulence is not None:
                for name in ("k", "eps", "nu_t"):
                    assert (
                        getattr(state, name).tobytes()
                        == getattr(turbulence, name).tobytes()
                    )
        assert _solver(_channel()).flow_state is None

    @pytest.mark.parametrize("rule", [ERROR_ESTIMATE, VELOCITY_STEP])
    @pytest.mark.parametrize("poisoned", ["u", "v", "p"])
    def test_a_solve_whose_fields_went_non_finite_keeps_no_state(
        self, rule: str, poisoned: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A diverged solve returns its fields and leaves no flow state or checkpoint.

        The corrector returns NaN in one field on the last outer iteration, as
        a run does that diverged before its cap. A finite solve ran first, so
        None afterwards is not an earlier solve's state, and a finite solve
        after it leaves both again. Defects caught: the finiteness guard
        removed (FlowState raises after the loop and the earlier state stays
        beside the new faces); p missing from the fields the guard reads.
        """
        config = _cavity(rule)
        solver = _solver(config)
        solver.solve_steady()
        assert solver.flow_state is not None and solver.steady_checkpoint is not None

        correct = solver._corrector.correct
        calls: list[int] = []

        def corrected_to_nan(
            prediction: MomentumPrediction, p: np.ndarray
        ) -> PressureCorrection:
            out = correct(prediction, p)
            calls.append(1)
            if len(calls) < config.max_simple_iter:
                return out
            return replace(
                out, **{poisoned: np.full_like(getattr(out, poisoned), np.nan)}
            )

        monkeypatch.setattr(solver._corrector, "correct", corrected_to_nan)
        u, v, p = solver.solve_steady()
        assert np.isnan(dict(u=u, v=v, p=p)[poisoned]).all()
        assert len(solver.residual_history) == config.max_simple_iter
        assert solver.flow_state is None and solver.steady_checkpoint is None

        monkeypatch.undo()
        solver.solve_steady()
        assert solver.flow_state is not None and solver.steady_checkpoint is not None


@pytest.mark.integration
class TestWarmStart:
    @pytest.mark.parametrize("name", list(FIXTURES))
    def test_the_start_from_rest_is_a_warm_start_from_rest(
        self, name: str, converged: dict
    ) -> None:
        """A warm start from rest and the inlets' k and eps is the rest solve, bitwise.

        So the warm path and initial=None run one loop. Defects caught: the
        warm path skipping apply_normal_velocity; the warm k and eps taken
        through anything but KEpsilonModel.state; the faces or p not copied.
        """
        solver = _solver(FIXTURES[name]())
        u, v, p = solver_module.allocate_fields(solver._mesh)
        turbulence = {}
        if solver._model is not None:
            start = solver._model.initial(*solver._walls.initial_values())
            turbulence = {
                key: np.array(getattr(start, key)) for key in ("k", "eps", "nu_t")
            }
        solver.solve_steady(initial=FlowState(u, v, p, **turbulence))
        _same_bits(solver, converged[name])

    def test_the_argument_is_the_probes_patched_start(
        self, converged: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """initial= gives warm59.py's patched start bitwise, on a changed boundary.

        The probe replaced allocate_fields and KEpsilonModel.initial by copies
        of a kept state's faces, pressure, k and eps (ADR-014 A, T1). Here the
        supply runs at 0.4 m/s from the 0.45 m/s state, 40 outer iterations.
        Defects caught: the state's nu_t read in place of C_mu k^2 / eps; the
        boundary's faces not written over the state's.
        """
        kept = converged["product"].flow_state
        argument = _solved(_product(cap=40, inlet=0.4), initial=kept)

        patched = _solver(_product(cap=40, inlet=0.4))
        model = patched._model

        def fields(_mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return np.array(kept.u), np.array(kept.v), np.array(kept.p)

        def turbulence(_k: float, _eps: float) -> TurbulenceState:
            return model.state(np.array(kept.k), np.array(kept.eps))

        monkeypatch.setattr(solver_module, "allocate_fields", fields)
        monkeypatch.setattr(model, "initial", turbulence)
        patched.solve_steady()
        _same_bits(argument, patched)
        assert argument.flow_state.u.tobytes() != kept.u.tobytes()

    @pytest.mark.parametrize("name", list(FIXTURES))
    def test_a_converged_state_stops_when_the_window_fills(
        self, name: str, converged: dict
    ) -> None:
        """Warm-started from its converged state, a solve stops at RATE_WINDOW outer.

        The rule starts fresh, so no stop can come before its window is full,
        and the state is already within the rule's tolerance, so the first
        full window stops it. The faces move by the iteration error the first
        solve left, which the rule bounded: under iteration_error_tol times
        the velocity scale. Defects caught: the rule carried over from the
        state (a stop at once); the warm start ignored (a solve from rest).
        """
        config = FIXTURES[name]()
        before = converged[name]
        after = _solved(config, initial=before.flow_state)
        assert after.converged
        assert len(after.residual_history) == RATE_WINDOW
        scale = StaggeredBoundary(Mesh(config), config).get_max_boundary_velocity()
        bound = config.iteration_error_tol * scale
        for face in ("u", "v"):
            moved = np.abs(
                getattr(after.flow_state, face) - getattr(before.flow_state, face)
            )
            assert 0.0 < float(moved.max()) < bound

    def test_a_state_of_another_grid_is_refused(self, converged: dict) -> None:
        """Defect caught: the shape check removed, leaving the failure to later code."""
        raw = yaml.safe_load(case_path("poiseuille").read_text(encoding="utf-8"))
        raw["domain"]["nx"], raw["domain"]["ny"] = 20, 10
        solver = _solver(SimConfig.from_dict(raw))
        with pytest.raises(ValueError, match="initial is a state of a 40x20 grid"):
            solver.solve_steady(initial=converged["channel"].flow_state)
        assert solver.flow_state is None and solver.residual_history == []

    def test_turbulence_present_or_absent_against_the_solve_is_refused(
        self, converged: dict
    ) -> None:
        """Defects caught: a missing k let through; a laminar solve ignoring k."""
        laminar = converged["channel"].flow_state
        coupled = converged["product"].flow_state
        with pytest.raises(
            ValueError, match="has no k and eps and this solve is coupled"
        ):
            _solver(_product()).solve_steady(
                initial=FlowState(coupled.u, coupled.v, coupled.p)
            )
        bare = FlowState(
            laminar.u, laminar.v, laminar.p, *(np.ones(laminar.p.shape),) * 3
        )
        with pytest.raises(
            ValueError, match="carries k, eps and nu_t and this solve is laminar"
        ):
            _solver(_channel()).solve_steady(initial=bare)

    def test_a_state_with_k_not_positive_is_refused(self, converged: dict) -> None:
        """Defect caught: the warm k unchecked, so a bad start fails mid-solve."""
        state = converged["product"].flow_state
        live = np.argwhere(state.k > 0.0)[0]
        k = np.array(state.k)
        k[tuple(live)] = 0.0
        bad = FlowState(state.u, state.v, state.p, k, state.eps, state.nu_t)
        with pytest.raises(ValueError, match="positive"):
            _solver(_product()).solve_steady(initial=bad)

    def test_what_is_not_a_state_is_refused(self, converged: dict) -> None:
        """Defects caught: a checkpoint, or a bare tuple, read as a warm start."""
        with pytest.raises(TypeError, match="initial must be a FlowState"):
            _solver(_channel()).solve_steady(
                initial=converged["channel"].steady_checkpoint
            )
        with pytest.raises(TypeError, match="resume must be a SteadyCheckpoint"):
            _solver(_channel()).solve_steady(resume=converged["channel"].flow_state)


@pytest.mark.integration
class TestResume:
    @pytest.mark.parametrize("name", list(FIXTURES))
    @pytest.mark.parametrize("cut", [1, 57, 150, -30])
    def test_a_capped_solve_resumed_is_the_uninterrupted_solve(
        self, name: str, cut: int, converged: dict
    ) -> None:
        """Cut at CUT outer and resumed, a solve reaches the uninterrupted one bitwise.

        The cuts fall before the rule's window fills (1, 57), after it (150)
        and 30 outer iterations before the uninterrupted stop (-30), where a
        window not restored cannot fill again in time; both solves stop by
        the rule, so the stop's iteration is pinned as well as the fields.
        Defects caught: nu_t rebuilt from k and eps on resume (the warm
        start's rule); the rule's window not restored (a stop 100 outer
        iterations after the cut, caught at -30 only, since the rule reads
        nothing older than its window); the residual history restarted; the
        outer count restarted, so the cap is not the total.
        """
        make = FIXTURES[name]
        if cut < 0:
            cut += len(converged[name].residual_history)
        cut_solve = _solved(make(cap=cut))
        assert cut_solve.stop_reason == CAP_STOP
        checkpoint = cut_solve.steady_checkpoint
        assert checkpoint.state.outer == cut
        resumed = _solved(make(), resume=checkpoint)
        _same_bits(resumed, converged[name])
        assert resumed.converged

    def test_a_resume_carries_the_cap_hits(self) -> None:
        """Corrections capped before the cut still count after it.

        The channel with max_pressure_iter 2 caps every correction, so the
        solve runs to max_simple_iter with a cap hit per outer iteration.
        Defect caught: pressure_cap_hits reset on resume, which the two
        fixtures, capped nowhere, cannot see.
        """

        def capped(cap: int) -> SimConfig:
            config = _channel(cap)
            object.__setattr__(config, "max_pressure_iter", 2)
            return config

        whole = _solved(capped(40))
        assert whole.pressure_cap_hits == 40
        resumed = _solved(capped(40), resume=_solved(capped(25)).steady_checkpoint)
        _same_bits(resumed, whole)

    def test_a_resume_past_the_cap_takes_no_iteration(self, converged: dict) -> None:
        """A checkpoint at the cap resumed under the same cap stops there, unchanged.

        Defect caught: the loop counted from zero, so a resume ran another cap's worth.
        """
        checkpoint = _solved(_channel(cap=30)).steady_checkpoint
        again = _solved(_channel(cap=30), resume=checkpoint)
        assert again.stop_reason == CAP_STOP
        assert again.flow_state.u.tobytes() == checkpoint.state.u.tobytes()
        assert again.residual_history == list(checkpoint.residual_history)

    def test_a_converged_checkpoint_is_refused(self, converged: dict) -> None:
        """Defect caught: a resume of a finished solve, which iterates past its stop."""
        with pytest.raises(ValueError, match="stopped converged"):
            _solver(_channel()).solve_steady(
                resume=converged["channel"].steady_checkpoint
            )

    def test_another_rule_version_is_refused(self) -> None:
        """Defect caught: a velocity_step checkpoint resumed under error_estimate."""
        raw = yaml.safe_load(case_path("poiseuille").read_text(encoding="utf-8"))
        raw["domain"]["nx"], raw["domain"]["ny"] = 40, 20
        raw["solver"]["stopping_rule"] = "velocity_step"
        raw["solver"]["max_simple_iter"] = 20
        checkpoint = _solved(SimConfig.from_dict(raw)).steady_checkpoint
        assert checkpoint.rule_version is None
        with pytest.raises(
            ValueError, match="rule version is None and this solve applies 3"
        ):
            _solver(_channel()).solve_steady(resume=checkpoint)

    def test_another_eddy_viscosity_is_refused(self) -> None:
        """A resume runs under the prescribed field its checkpoint's solve ran under.

        The same field (a copy) resumes to the uninterrupted solve under it;
        another field, none where there was one, and one where there was none
        are each refused before any iteration (ADR-014 D's second note).
        Defect caught: the digest comparison removed, so every one of the three
        resumes runs and continues a solve that was not this one.
        """
        mesh = Mesh(_channel())
        field = np.full(mesh.cell_type.shape, 2.0e-5)
        under = _solved(_channel(cap=20), eddy_viscosity=field).steady_checkpoint
        plain = _solved(_channel(cap=20)).steady_checkpoint
        assert plain.eddy_viscosity_sha256 is None
        assert under.eddy_viscosity_sha256 is not None
        resumed = _solved(_channel(cap=40), eddy_viscosity=field.copy(), resume=under)
        _same_bits(resumed, _solved(_channel(cap=40), eddy_viscosity=field))
        for checkpoint, given in [(under, 1.5 * field), (under, None), (plain, field)]:
            solver = _solver(_channel(cap=40))
            with pytest.raises(ValueError, match="under the eddy_viscosity it ran"):
                solver.solve_steady(eddy_viscosity=given, resume=checkpoint)
            assert solver.residual_history == []

    def test_initial_and_resume_together_are_refused(self, converged: dict) -> None:
        """Defect caught: one of the two silently preferred."""
        checkpoint = _solved(_channel(cap=5)).steady_checkpoint
        with pytest.raises(ValueError, match="initial and resume are exclusive"):
            _solver(_channel()).solve_steady(
                initial=checkpoint.state, resume=checkpoint
            )

    def test_a_checkpoint_whose_history_does_not_match_its_count_is_refused(
        self, converged: dict
    ) -> None:
        """Defects caught: a history cut short; a checkpoint with no FlowState."""
        checkpoint = _solved(_channel(cap=5)).steady_checkpoint
        with pytest.raises(ValueError, match="4 entries for 5 outer iterations"):
            SteadyCheckpoint(
                checkpoint.state,
                checkpoint.residual_history[:-1],
                checkpoint.velocity_steps,
                checkpoint.viscosity_steps,
                checkpoint.rule_version,
                checkpoint.pressure_cap_hits,
            )
        with pytest.raises(TypeError, match="state must be a FlowState"):
            SteadyCheckpoint((), (), (), (), None, 0)  # type: ignore[arg-type]

    def test_a_negative_nu_t_in_a_live_cell_is_refused(
        self, capped_checkpoint: SteadyCheckpoint
    ) -> None:
        """A checkpoint edited to a negative eddy viscosity is refused before the solve.

        Defect caught: the resume's own nu_t check removed. The first
        iteration's viscosity check refuses the same field, but only after the
        solve's history and cap hits were reset, with another message.
        """
        nu_t = np.array(capped_checkpoint.state.nu_t)
        nu_t[_live_cell()] = -1e-6
        bad = replace(
            capped_checkpoint, state=replace(capped_checkpoint.state, nu_t=nu_t)
        )
        solver = _solver(_product())
        with pytest.raises(ValueError, match="resume's nu_t is negative"):
            solver.solve_steady(resume=bad)
        assert solver.residual_history == [] and solver.flow_state is None

    @pytest.mark.parametrize("value", [0.0, -1.0])
    @pytest.mark.parametrize("name", ["k", "eps"])
    def test_k_or_eps_not_positive_is_refused_on_a_resume(
        self, name: str, value: float, capped_checkpoint: SteadyCheckpoint
    ) -> None:
        """A resume refuses k or eps not positive in a live cell, before any iteration.

        Defect caught: the resume's k and eps taken without KEpsilonModel.state,
        so the first k and eps step refuses them instead, after the solve has
        loaded the checkpoint's history.
        """
        field = np.array(getattr(capped_checkpoint.state, name))
        field[_live_cell()] = value
        bad = replace(
            capped_checkpoint, state=replace(capped_checkpoint.state, **{name: field})
        )
        solver = _solver(_product())
        with pytest.raises(ValueError, match="positive"):
            solver.solve_steady(resume=bad)
        assert solver.residual_history == []


def _same_state(handed: SteadyCheckpoint, capped: SteadyCheckpoint) -> None:
    """A periodic checkpoint against the checkpoint a solve capped there leaves.

    Everything bitwise but the stop reason: the periodic one's is None, the
    capped one's CAP_STOP.
    """
    for name in FIELDS:
        x, y = getattr(handed.state, name), getattr(capped.state, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert x.tobytes() == y.tobytes(), name
    assert handed.state.outer == capped.state.outer
    assert handed.state.stop_reason is None and capped.state.stop_reason == CAP_STOP
    for name in ("residual_history", "velocity_steps", "viscosity_steps"):
        assert getattr(handed, name) == getattr(capped, name), name
    assert (handed.rule_version, handed.pressure_cap_hits) == (
        capped.rule_version,
        capped.pressure_cap_hits,
    )
    assert handed.eddy_viscosity_sha256 == capped.eddy_viscosity_sha256


@pytest.mark.integration
class TestPeriodicCheckpoint:
    """``solve_steady(on_checkpoint=...)`` every checkpoint_interval outer (step 3)."""

    @pytest.mark.parametrize("name", list(FIXTURES))
    def test_each_call_holds_what_a_solve_capped_there_leaves(self, name: str) -> None:
        """Called at outer 10, 20 and 30 of a solve capped at 30, each the capped state.

        The solve with the writer is the solve without it, bitwise. Defects
        caught: the cadence off by one (``iteration % interval``, a call at
        outer 1, 11 and 21); the checkpoint built before the iteration's
        turbulence step or with the rule's window of the iteration before; a
        writer that changes the iterates.
        """
        make = FIXTURES[name]
        handed: list[SteadyCheckpoint] = []
        with_writer = _solved(make(cap=30, interval=10), on_checkpoint=handed.append)
        assert [c.state.outer for c in handed] == [10, 20, 30]
        for checkpoint in handed:
            capped = _solved(make(cap=checkpoint.state.outer)).steady_checkpoint
            _same_state(checkpoint, capped)
        _same_bits(with_writer, _solved(make(cap=30)))

    def test_no_call_at_the_converged_stop_or_without_the_key(
        self, converged: dict
    ) -> None:
        """An interval equal to the stop's iteration count is never called; nor is no key.

        Defects caught: a call on the converging iteration (a converged
        checkpoint, which a resume refuses, written over the last one that
        could be resumed); a default cadence without the key.
        """
        stop = len(converged["channel"].residual_history)
        handed: list[SteadyCheckpoint] = []
        _solved(_channel(interval=stop), on_checkpoint=handed.append)
        assert handed == []
        _solved(_channel(cap=30), on_checkpoint=handed.append)
        assert handed == []

    def test_a_resumed_solve_counts_from_the_checkpoints_outer(self) -> None:
        """Resumed from outer 25 under interval 10 and cap 40: calls at 30 and 40.

        Defect caught: the cadence counted from the resume's first
        iteration, which calls at 35 and so breaks the promise of at most an
        interval lost.
        """
        handed: list[SteadyCheckpoint] = []
        checkpoint = _solved(_channel(cap=25)).steady_checkpoint
        _solved(
            _channel(cap=40, interval=10),
            resume=checkpoint,
            on_checkpoint=handed.append,
        )
        assert [c.state.outer for c in handed] == [30, 40]

    def test_no_call_when_a_field_is_not_finite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A diverging iteration is skipped, not raised on: the solve still returns.

        The corrector returns NaN in p on the last outer iteration, as in
        ``TestTheSolveState``; the writer is called on every one before it.
        Defect caught: the finiteness check left out of the periodic call, so
        FlowState raises inside the loop and the diverged fields, which the
        probe runners classify, never come back.
        """
        config = _cavity(VELOCITY_STEP, cap=8, interval=1)
        solver = _solver(config)
        correct = solver._corrector.correct
        calls: list[int] = []

        def corrected_to_nan(
            prediction: MomentumPrediction, p: np.ndarray
        ) -> PressureCorrection:
            out = correct(prediction, p)
            calls.append(1)
            if len(calls) < config.max_simple_iter:
                return out
            return replace(out, p=np.full_like(out.p, np.nan))

        monkeypatch.setattr(solver._corrector, "correct", corrected_to_nan)
        handed: list[SteadyCheckpoint] = []
        solver.solve_steady(on_checkpoint=handed.append)
        assert [c.state.outer for c in handed] == list(range(1, 8))
        assert solver.flow_state is None
