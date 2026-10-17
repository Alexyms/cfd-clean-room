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
"""

import numpy as np
import pytest
import yaml

import src.solver_staggered as solver_module
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import Mesh
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


def _channel(cap: int = 5000) -> SimConfig:
    """VAL-001's channel on 40x20 under its own error_estimate rule."""
    raw = yaml.safe_load(case_path("poiseuille").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = 40, 20
    raw["solver"]["max_simple_iter"] = cap
    return SimConfig.from_dict(raw)


def _product(cap: int = 5000, inlet: float | None = None) -> SimConfig:
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
    return SimConfig.from_dict(raw)


FIXTURES = {"channel": _channel, "product": _product}


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
    @pytest.mark.parametrize("cut", [1, 57, 150])
    def test_a_capped_solve_resumed_is_the_uninterrupted_solve(
        self, name: str, cut: int, converged: dict
    ) -> None:
        """Cut at CUT outer and resumed, a solve reaches the uninterrupted one bitwise.

        The cuts fall before the rule's window fills (1, 57) and after (150),
        and both solves stop by the rule, so the stop's iteration is pinned
        as well as the fields. Defects caught: nu_t rebuilt from k and eps on
        resume (the warm start's rule); the rule's window not restored (a
        later stop); the residual history or the cap hits restarted; the
        outer count restarted, so the cap is not the total.
        """
        make = FIXTURES[name]
        cut_solve = _solved(make(cap=cut))
        assert cut_solve.stop_reason == CAP_STOP
        checkpoint = cut_solve.steady_checkpoint
        assert checkpoint.state.outer == cut
        resumed = _solved(make(), resume=checkpoint)
        _same_bits(resumed, converged[name])
        assert resumed.converged

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
