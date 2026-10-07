"""Tests for the staggered SIMPLE solver (ECR-001 step 6, REQ-S04, REQ-S07, REQ-S12).

The solver is a loop over the step 4 predictor and the step 5 corrector, so
most of what these tests check is the loop: what it hands the callback, what
it resets, that it keeps the closed-domain compatibility step 5 proved for
one correction, and that it never disturbs a Dirichlet face. The corrector's
output at every iteration is observed by wrapping PressureCorrector.correct,
which leaves the solver's code path untouched.
"""

import dataclasses
from collections.abc import Callable
from time import perf_counter

import numpy as np
import pytest
import yaml

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import Mesh
from src.momentum import MomentumPrediction, MomentumPredictor
from src.pressure import PressureCorrection, PressureCorrector
from src.solver_staggered import StaggeredSolver
from src.staggered import FaceVelocities, allocate_fields, to_cell_centers
from src.stopping import ErrorEstimateRule, ImbalanceSummary, IterationState
from validation.cases import (
    CASE_GRIDS,
    case_path,
    load_case,
    load_preset,
    with_velocity_step,
)

EPS = np.finfo(np.float64).eps


# _case and _channel stop by velocity_step whatever rule the case file names, so
# a test's rule does not change when a case file does. A test that wants
# error_estimate asks for it through _ruled.
def _case(name: str, n: int, **overrides: object) -> SimConfig:
    """A committed case on an n x n grid, with top-level sections overridden."""
    raw = yaml.safe_load(case_path(name).read_text(encoding="utf-8"))
    raw["domain"]["nx"] = raw["domain"]["ny"] = n
    raw.update(overrides)
    return with_velocity_step(SimConfig.from_dict(raw))


def _channel(nx: int = 12, ny: int = 6) -> SimConfig:
    """The VAL-001 channel on a small grid."""
    return with_velocity_step(load_case("poiseuille", grid=(nx, ny)))


def _ruled(
    name: str,
    grid: tuple[int, int],
    boundaries: dict | None = None,
    fluid: dict | None = None,
    domain: dict | None = None,
    **keys: object,
) -> SimConfig:
    """A committed case on a grid with solver keys, and optionally more, replaced."""
    raw = yaml.safe_load(case_path(name).read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = grid
    raw["solver"].update(keys)
    raw["fluid"].update(fluid or {})
    raw["domain"].update(domain or {})
    if boundaries is not None:
        raw["boundaries"] = boundaries
    return SimConfig.from_dict(raw)


# Spy cases in which no factor of the flux scale is 1: the channel at rho 1.2, and a
# closed 2.0 by 1.0 box at rho 1.2 under a 0.5 m/s lid.
_LID = dict(type="velocity_inlet", location="top", x_start=0, x_end=2, u_velocity=0.5)
_RHO_CHANNEL = {"name": "poiseuille", "grid": (12, 6), "fluid": {"density": 1.2}}
_BOX = _RHO_CHANNEL | {
    "name": "cavity",
    "domain": {"width": 2},
    "boundaries": {"lid": _LID},
}


@dataclasses.dataclass
class _RuleSeen:
    """What the error_estimate rules a solver built were given."""

    speeds: list[float] = dataclasses.field(default_factory=list)
    fluxes: list[float] = dataclasses.field(default_factory=list)
    steps: list[float] = dataclasses.field(default_factory=list)
    readings: list[ImbalanceSummary] = dataclasses.field(default_factory=list)


def _spy_rules(monkeypatch: pytest.MonkeyPatch) -> _RuleSeen:
    """Record each rule's scales, each step and each imbalance summary it receives."""
    seen = _RuleSeen()

    class Spy(ErrorEstimateRule):
        def __init__(
            self, velocity_scale: float, flux_scale: float, *tols: float
        ) -> None:
            seen.speeds.append(velocity_scale)
            seen.fluxes.append(flux_scale)
            super().__init__(velocity_scale, flux_scale, *tols)

        def update(
            self, step: float, imbalance: Callable[[], ImbalanceSummary]
        ) -> bool:
            seen.steps.append(step)

            def recorded() -> ImbalanceSummary:
                seen.readings.append(imbalance())
                return seen.readings[-1]

            return super().update(step, recorded)

    monkeypatch.setattr("src.solver_staggered.ErrorEstimateRule", Spy)
    return seen


def _build(config: SimConfig) -> tuple[Mesh, StaggeredBoundary, StaggeredSolver]:
    mesh = Mesh(config)
    boundary = StaggeredBoundary(mesh, config)
    return mesh, boundary, StaggeredSolver(mesh, config, boundary)


def _record_corrections(monkeypatch: pytest.MonkeyPatch) -> list[PressureCorrection]:
    """Wrap PressureCorrector.correct so every correction the solver makes is kept."""
    seen: list[PressureCorrection] = []
    original = PressureCorrector.correct

    def recording(
        self: PressureCorrector, prediction: MomentumPrediction, p: np.ndarray
    ) -> PressureCorrection:
        out = original(self, prediction, p)
        seen.append(out)
        return out

    monkeypatch.setattr(PressureCorrector, "correct", recording)
    return seen


def _absolute_face_flux(mesh: Mesh, rho: float, u: np.ndarray, v: np.ndarray) -> float:
    """Sum of |rho u A| over every face: the scale the telescoping sum cancels."""
    return float(
        rho
        * (
            np.abs(u * mesh.dy_cell[:, None]).sum()
            + np.abs(v * mesh.dx_cell[None, :]).sum()
        )
    )


@pytest.mark.integration
class TestContract:
    """The public shape the harness depends on (SYSTEM.md, section 4)."""

    def test_returns_three_cell_centered_float64_contiguous_arrays(self) -> None:
        """(u, v, p) are [ny, nx], float64 and C-contiguous, the array convention."""
        config = _case("cavity", 6)
        mesh, _bc, solver = _build(config)
        for field in solver.solve_steady():
            assert field.shape == mesh.cell_type.shape
            assert field.dtype == np.float64
            assert field.flags["C_CONTIGUOUS"]

    def test_callback_gets_cell_centered_fields_and_the_corrector_iterations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once per outer iteration, with the corrected faces averaged to centers.

        Comparing against the face arrays each correction returned shows the
        callback is handed the cell-centered conversion, not the faces.
        """
        corrections = _record_corrections(monkeypatch)
        config = _case("cavity", 6)
        mesh, _bc, solver = _build(config)
        states: list[IterationState] = []
        u, _v, _p = solver.solve_steady(on_iteration=states.append)

        assert len(states) == len(solver.residual_history) == len(corrections)
        assert [s.iteration for s in states] == list(range(len(states)))
        assert [s.pressure_iterations for s in states] == [
            c.iterations for c in corrections
        ]
        assert [s.residual for s in states] == solver.residual_history
        for state, corrected in zip(states, corrections, strict=True):
            u_c, v_c = to_cell_centers(corrected.u, corrected.v)
            assert state.u.shape == state.v.shape == mesh.cell_type.shape
            assert np.array_equal(state.u, u_c)
            assert np.array_equal(state.v, v_c)
            assert state.p is corrected.p
        assert np.array_equal(u, states[-1].u)
        assert solver.last_pressure_iterations == corrections[-1].iterations
        assert solver.pressure_cap_hits == 0

    def test_iteration_count_cap_hits_and_stage_timers_reset_at_the_start_of_each_solve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A solve that fails before its first correction reports none of the last one."""
        config = _ruled("cavity", (6, 6), max_pressure_iter=2, max_simple_iter=5)
        _mesh, _bc, solver = _build(config)
        solver.solve_steady()
        assert solver.last_pressure_iterations == 2
        assert solver.pressure_cap_hits == 5
        assert set(solver.stage_seconds) == {"momentum", "pressure", "correct"}
        assert solver.stage_seconds["pressure"] > 0.0

        def fail(
            self: MomentumPredictor, u: np.ndarray, v: np.ndarray, p: np.ndarray
        ) -> MomentumPrediction:
            raise RuntimeError("stop before the first correction")

        monkeypatch.setattr(MomentumPredictor, "predict", fail)
        with pytest.raises(RuntimeError, match="first correction"):
            solver.solve_steady()
        assert solver.last_pressure_iterations == 0
        assert solver.pressure_cap_hits == 0
        assert solver.stage_seconds == {
            "momentum": 0.0,
            "pressure": 0.0,
            "correct": 0.0,
        }
        assert solver.residual_history == []

    def test_error_estimate_flux_scale_is_the_correctors(self) -> None:
        """One F for the rounding floor and the rule: the solver reads the corrector's."""
        config = _ruled("poiseuille", (12, 6), stopping_rule="error_estimate")
        _mesh, _bc, solver = _build(config)
        assert solver.flux_scale == solver._corrector.flux_scale
        assert solver.flux_scale == pytest.approx(1.0 * 0.1 * 0.5, rel=1e-14)
        _mesh, _bc, plain = _build(_case("cavity", 6))
        assert plain.flux_scale is None
        assert plain._corrector.flux_scale == pytest.approx(1.0, rel=1e-14)

    def test_last_mass_imbalance_is_that_of_the_returned_faces(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Observability only, but it must describe the field that was returned."""
        corrections = _record_corrections(monkeypatch)
        config = _case("cavity", 6)
        mesh, bc, solver = _build(config)
        solver.solve_steady()
        last = corrections[-1]
        expected = PressureCorrector(mesh, config, bc).mass_imbalance(last.u, last.v)
        assert np.array_equal(solver.last_mass_imbalance, expected)


@pytest.mark.integration
class TestFaceVelocities:
    """The face-field contract of REQ-S13 (ADR-011 A, SYSTEM.md section 4).

    Each promise is checked bitwise, not to a tolerance: the faces exposed
    must be the faces the stopping rule judged, and an average or an
    imbalance that differed in the last bit would show a different array.
    """

    def test_none_before_the_first_solve_and_set_by_a_solve_that_hits_the_cap(
        self,
    ) -> None:
        """Set at the end of every solve, converged or not."""
        config = _ruled("cavity", (6, 6), convergence_tol=1e-300, max_simple_iter=20)
        _mesh, _bc, solver = _build(config)
        assert solver.face_velocities is None
        solver.solve_steady()
        assert solver.converged is False
        assert isinstance(solver.face_velocities, FaceVelocities)

    @pytest.mark.parametrize("case", ["cavity", "channel"])
    def test_two_face_averages_are_the_returned_cell_means_bitwise(
        self, case: str
    ) -> None:
        """A swap of u and v, or a copy from before the last correction, shows here."""
        config = _case("cavity", 6) if case == "cavity" else _channel()
        mesh, _bc, solver = _build(config)
        u, v, _p = solver.solve_steady()
        faces = solver.face_velocities
        assert faces.u.shape == (mesh.yc.shape[0], mesh.x.shape[0])
        assert faces.v.shape == (mesh.y.shape[0], mesh.xc.shape[0])
        u_c, v_c = to_cell_centers(faces.u, faces.v)
        assert np.array_equal(u_c, u)
        assert np.array_equal(v_c, v)
        assert not np.array_equal(faces.u[:, :-1], 0.0)

    def test_mass_imbalance_of_the_faces_is_last_mass_imbalance_bitwise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The faces are the last correction's, the ones the imbalance was read from."""
        corrections = _record_corrections(monkeypatch)
        config = _channel()
        mesh, bc, solver = _build(config)
        solver.solve_steady()
        faces = solver.face_velocities
        corrector = PressureCorrector(mesh, config, bc)
        assert np.array_equal(
            corrector.mass_imbalance(faces.u, faces.v), solver.last_mass_imbalance
        )
        assert np.array_equal(faces.u, corrections[-1].u)
        assert np.array_equal(faces.v, corrections[-1].v)
        assert len(corrections) > 1
        assert not np.array_equal(faces.u, corrections[-2].u)

    def test_arrays_are_owned_read_only_copies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Writing raises; the solver's working arrays are not shared; a second solve leaves the first object alone."""
        corrections = _record_corrections(monkeypatch)
        config = _case("cavity", 6)
        _mesh, _bc, solver = _build(config)
        solver.solve_steady()
        first = solver.face_velocities
        for arr in (first.u, first.v):
            assert arr.dtype == np.float64
            assert arr.flags["C_CONTIGUOUS"]
            with pytest.raises(ValueError, match="read-only"):
                arr[0, 0] = 1.0
        last = corrections[-1]
        assert not np.shares_memory(first.u, last.u)
        assert not np.shares_memory(first.v, last.v)
        u_snapshot, v_snapshot = first.u.copy(), first.v.copy()
        solver.solve_steady()
        assert solver.face_velocities is not first
        assert np.array_equal(first.u, u_snapshot)
        assert np.array_equal(first.v, v_snapshot)


@pytest.mark.validation
def test_val001_40x20_continuity_remeasured_from_the_exposed_faces() -> None:
    """REQ-S13's continuity promise, re-measured from face_velocities alone.

    The committed VAL-001 case at 40x20 under its own error_estimate rule.
    The imbalance recomputed from the exposed faces must meet REQ-S04's
    per-cell and domain-sum clauses at mass_imbalance_tol, and its absolute
    sum over the flux scale the summed bound: the faces exposed are the
    faces the rule judged. About 25 s.
    """
    config = load_case("poiseuille", grid=(40, 20))
    assert config.stopping_rule == "error_estimate"
    mesh, bc, solver = _build(config)
    start = perf_counter()
    solver.solve_steady()
    seconds = perf_counter() - start
    assert solver.stop_reason == "error_estimate_and_continuity"

    faces = solver.face_velocities
    imbalance = PressureCorrector(mesh, config, bc).mass_imbalance(faces.u, faces.v)
    worst = float(np.abs(imbalance).max())
    signed_sum = float(imbalance.sum())
    absolute_sum = float(np.abs(imbalance).sum())
    print(f"VAL-001 40x20 from face_velocities, {len(solver.residual_history)} outer")
    print(f"  in {seconds:.1f} s; worst cell {worst:.3e}; signed sum {signed_sum:.3e}")
    print(f"  absolute sum {absolute_sum:.3e} over flux scale {solver.flux_scale:.3e}")
    assert worst < config.mass_imbalance_tol
    assert abs(signed_sum) < config.mass_imbalance_tol
    assert absolute_sum / solver.flux_scale < config.iteration_error_tol
    assert np.array_equal(imbalance, solver.last_mass_imbalance)


@pytest.mark.integration
class TestReferenceVelocity:
    """The residual divides by reference_velocity, F_ref / (rho h) (SYSTEM.md, section 4)."""

    @pytest.mark.parametrize(
        "case_id", [c for c, (kind, _nx, _ny) in CASE_GRIDS.items() if kind == "cavity"]
    )
    def test_closed_domain_reference_is_the_largest_boundary_velocity(
        self, case_id: str
    ) -> None:
        """Equal with ==: F_ref is rho times that velocity times h, over rho h.

        The expected value is formed from the staggered layer's own quantities
        by the contract's formula, in the solver's order of operations.
        """
        config = load_preset(case_id)
        mesh, bc, solver = _build(config)
        nx, ny = mesh.xc.shape[0], mesh.yc.shape[0]
        h = max(float(mesh.x[nx]) / nx, float(mesh.y[ny]) / ny)
        assert bc.get_total_inlet_flux() == 0.0
        f_ref = config.rho * bc.get_max_boundary_velocity() * h
        assert solver.reference_velocity == f_ref / (config.rho * h)
        assert solver.reference_velocity == pytest.approx(
            bc.get_max_boundary_velocity(), rel=1e-14
        )

    def test_channel_reference_is_the_exact_inlet_flux_over_the_spacing(self) -> None:
        """F_ref is rho times the exact face sum of the inlet flux, over rho h.

        The staggered inlet spans the whole left edge, so the flux is the
        prescribed velocity times the height exactly. The retired collocated
        layer's flux was two corner cells short of it
        (docs/reports/inlet_flux_comparison.md).
        """
        config = load_preset("val001_80x40")
        mesh, bc, solver = _build(config)
        exact_flux = 0.1 * 0.5
        assert bc.get_total_inlet_flux() == pytest.approx(exact_flux, rel=1e-14)
        assert solver.reference_velocity == pytest.approx(
            exact_flux / mesh.dx, rel=1e-14
        )


@pytest.mark.integration
class TestPhysics:
    """Limits the loop must respect whatever the discretisation's accuracy."""

    def test_quiescent_closed_box_stays_at_rest_and_stops_at_once(self) -> None:
        """No moving wall: zero velocity exactly, and the first residual is zero."""
        config = _case("cavity", 6, boundaries={})
        _mesh, _bc, solver = _build(config)
        u, v, p = solver.solve_steady()
        assert np.all(u == 0.0)
        assert np.all(v == 0.0)
        assert np.all(p == 0.0)
        assert solver.residual_history == [0.0]

    def test_closed_domain_imbalance_sums_to_zero_at_every_iteration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Step 5 proved one correction compatible; the loop must keep every one so.

        Same bound as step 5: eight epsilon times the total absolute face
        flux of that iteration's corrected field.
        """
        corrections = _record_corrections(monkeypatch)
        config = _case("cavity", 8)
        mesh, bc, solver = _build(config)
        solver.solve_steady()
        assert len(corrections) == len(solver.residual_history) > 1
        pc = PressureCorrector(mesh, config, bc)
        for corrected in corrections:
            total = float(pc.mass_imbalance(corrected.u, corrected.v).sum())
            bound = (
                8.0
                * EPS
                * _absolute_face_flux(mesh, config.rho, corrected.u, corrected.v)
            )
            assert abs(total) <= bound

    @pytest.mark.parametrize("which", ["cavity", "channel"])
    def test_dirichlet_faces_hold_what_apply_normal_velocity_wrote(
        self, which: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every wall and inlet face, compared with ==; outlet faces carry the flow.

        apply_normal_velocity writes no outlet face, so those are excluded
        from the comparison and checked for the outflow they carry instead.
        """
        corrections = _record_corrections(monkeypatch)
        config = _case("cavity", 6) if which == "cavity" else _channel()
        mesh, bc, solver = _build(config)
        solver.solve_steady()
        final = corrections[-1]
        expected_u, expected_v, _p = allocate_fields(mesh)
        bc.apply_normal_velocity(expected_u, expected_v)
        outlets = bc.pressure_outlets()
        edges = {
            "left": (final.u[:, 0], expected_u[:, 0]),
            "right": (final.u[:, -1], expected_u[:, -1]),
            "bottom": (final.v[0, :], expected_v[0, :]),
            "top": (final.v[-1, :], expected_v[-1, :]),
        }
        for edge, (got, want) in edges.items():
            dirichlet = ~outlets[edge].is_outlet
            assert np.array_equal(got[dirichlet], want[dirichlet]), edge
        if which == "channel":
            assert outlets["right"].is_outlet.all()
            assert np.all(final.u[:, -1] > 0.0)

    def test_outlet_faces_are_extrapolated_before_every_prediction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The momentum contract leaves outlet faces to the caller: zero gradient.

        Every prediction must see each outlet face equal to the interior face
        beside it, including the first, where the allocated value is zero.
        """
        seen: list[np.ndarray] = []
        original = MomentumPredictor.predict

        def watching(
            self: MomentumPredictor, u: np.ndarray, v: np.ndarray, p: np.ndarray
        ) -> MomentumPrediction:
            seen.append(u[:, -1] - u[:, -2])
            return original(self, u, v, p)

        monkeypatch.setattr(MomentumPredictor, "predict", watching)
        _mesh, _bc, solver = _build(_channel())
        solver.solve_steady()
        assert len(seen) == len(solver.residual_history) > 1
        assert all(np.all(gap == 0.0) for gap in seen)

    def test_stretched_cavity_runs_and_returns_finite_fields(self) -> None:
        """Runs on a stretched mesh, which the retired collocated solver refused. No accuracy claim."""
        mesh_block = {"x": {"stretch_ratio": 1.2}, "y": {"stretch_ratio": 1.2}}
        config = _case("cavity", 8, mesh=mesh_block)
        mesh, _bc, solver = _build(config)
        assert not mesh.is_uniform
        u, v, p = solver.solve_steady()
        for field in (u, v, p):
            assert np.all(np.isfinite(field))
        assert np.abs(u).max() > 0.0
        assert solver.residual_history[-1] < config.convergence_tol


@pytest.mark.integration
class TestStoppingRule:
    """converged, stop_reason and the error_estimate rule's scale (src/stopping.py)."""

    def test_stop_is_reported_and_reset_at_the_start_of_each_solve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """None before a solve, set by one, and cleared by a solve that fails early."""
        _mesh, _bc, solver = _build(_case("cavity", 6))
        assert (solver.converged, solver.stop_reason) == (False, None)
        solver.solve_steady()
        assert solver.converged is True
        assert solver.stop_reason == "velocity_step_below_tol"

        def fail(
            self: MomentumPredictor, u: np.ndarray, v: np.ndarray, p: np.ndarray
        ) -> MomentumPrediction:
            raise RuntimeError("stop before the first correction")

        monkeypatch.setattr(MomentumPredictor, "predict", fail)
        with pytest.raises(RuntimeError, match="first correction"):
            solver.solve_steady()
        assert (solver.converged, solver.stop_reason) == (False, None)

    def test_error_estimate_stops_later_with_continuity_met(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The velocity_step rule would have stopped earlier on the same trajectory.

        Review 24b B1: the last summary the rule received is the returned
        field's worst, absolute-summed and signed-summed imbalance, exactly; a
        zero, signed or swapped absolute sum fails, and so does a signed sum
        replaced by the absolute one or by zero (here -1e-17 against 7.7e-10).
        Test 24b T1: each step is the residual in m/s, and dv exceeds du at 55
        of this solve's iterations, so du alone fails.
        """
        seen = _spy_rules(monkeypatch)
        config = _ruled("cavity", (6, 6), stopping_rule="error_estimate")
        _mesh, _bc, solver = _build(config)
        solver.solve_steady()
        assert solver.stop_reason == "error_estimate_and_continuity"
        assert solver.converged is True
        signed = solver.last_mass_imbalance
        cells = np.abs(signed)
        assert seen.readings[-1] == ImbalanceSummary(
            worst=cells.max(), absolute_sum=cells.sum(), signed_sum=signed.sum()
        )
        assert cells.max() < config.mass_imbalance_tol
        assert cells.sum() / solver.flux_scale < config.iteration_error_tol
        assert abs(signed.sum()) < config.mass_imbalance_tol
        expected = [r * solver.reference_velocity for r in solver.residual_history]
        assert seen.steps == pytest.approx(expected, rel=1e-12)
        assert min(solver.residual_history[:-1]) < config.convergence_tol

    # A step test at 10 passes at once; the cap is below the rule's window.
    @pytest.mark.parametrize(
        ("rule", "tol"), [("velocity_step", 1e-300), ("error_estimate", 10.0)]
    )
    def test_reaching_the_cap_is_reported_as_not_converged(
        self, rule: str, tol: float
    ) -> None:
        """Twenty iterations and no stop is not convergence, under either rule."""
        config = _ruled(
            "cavity",
            (6, 6),
            stopping_rule=rule,
            convergence_tol=tol,
            max_simple_iter=20,
        )
        _mesh, _bc, solver = _build(config)
        solver.solve_steady()
        assert len(solver.residual_history) == 20
        assert (solver.converged, solver.stop_reason) == (False, "max_simple_iter")

    @pytest.mark.parametrize(
        ("case", "speed", "flux", "reference"),
        [
            ({"name": "poiseuille", "grid": (12, 6)}, 0.1, 0.05, 0.6),
            ({"name": "cavity", "grid": (6, 6)}, 1.0, 1.0, 1.0),
            (_RHO_CHANNEL, 0.1, 1.2 * 0.05, 0.6),
            (_BOX, 0.5, 1.2 * 0.5 * 2.0, 0.5),
        ],
        ids=["channel", "cavity", "channel-rho-1.2", "box-2x1-lid-0.5"],
    )
    def test_error_estimate_scales_are_physical_and_the_step_is_in_m_per_s(
        self,
        monkeypatch: pytest.MonkeyPatch,
        case: dict,
        speed: float,
        flux: float,
        reference: float,
    ) -> None:
        """One rule at construction and one per solve, never on reference_velocity.

        The velocity scale is the inlet or lid speed. The flux scale is rho
        times the inflow (0.1 through 0.5), or closed, rho times the lid speed
        times the longer side; the last two cases have no factor of 1, so a
        dropped rho, a dropped speed or min for max fails. Test 24 T1: each
        update gets the step in m/s, the residual times reference_velocity.
        """
        seen = _spy_rules(monkeypatch)
        config = _ruled(**case, stopping_rule="error_estimate", max_simple_iter=2)
        _mesh, _bc, solver = _build(config)
        solver.solve_steady()
        assert solver.reference_velocity == pytest.approx(reference)
        assert seen.speeds == [speed, speed]
        assert seen.fluxes == pytest.approx([flux, flux], rel=1e-12)
        assert solver.flux_scale == pytest.approx(flux, rel=1e-12)
        expected = [r * solver.reference_velocity for r in solver.residual_history]
        assert seen.steps == pytest.approx(expected, rel=1e-12)

    def test_error_estimate_without_a_boundary_velocity_raises(self) -> None:
        """A closed box with no moving wall leaves the estimate without a scale."""
        config = _ruled("cavity", (6, 6), boundaries={}, stopping_rule="error_estimate")
        with pytest.raises(ValueError, match="stopping_rule error_estimate needs"):
            _build(config)


@pytest.mark.integration
class TestCappedCorrections:
    """A pressure correction that stops at max_pressure_iter is reported, not hidden (ADR-013 B).

    The report's probe room froze under the committed cap of 200 sweeps with
    47% of the supply unaccounted while the velocity-step rule called it
    converged (docs/reports/pressure_solver_ecr003.md, section 8.2). The
    solver counts capped corrections, warns once, and under velocity_step
    does not stop on an outer iteration whose correction reached the cap.
    """

    def test_capped_corrections_are_counted_and_warned_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With a cap of one every correction is capped: counted, and one warning."""
        corrections = _record_corrections(monkeypatch)
        config = _ruled(
            "cavity",
            (6, 6),
            stopping_rule="velocity_step",
            max_pressure_iter=1,
            max_simple_iter=12,
        )
        _mesh, _bc, solver = _build(config)
        with caplog.at_level("WARNING", logger="src.solver_staggered"):
            solver.solve_steady()
        assert all(c.reached_cap for c in corrections)
        assert solver.pressure_cap_hits == len(corrections) == 12
        warnings = [r for r in caplog.records if "max_pressure_iter" in r.getMessage()]
        assert len(warnings) == 1
        assert (solver.converged, solver.stop_reason) == (False, "max_simple_iter")

    def test_a_capped_outer_loop_that_converges_stops_on_an_uncapped_correction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One CG iteration per correction still converges the tiny cavity, and may stop.

        The outer loop converges on its own once b shrinks, and the last
        correction then meets the floor rather than the cap, so the refusal
        does not hold a converged solve hostage: the stop is allowed on that
        iteration, and the faces it returns do close.
        """
        corrections = _record_corrections(monkeypatch)
        config = _ruled(
            "cavity",
            (6, 6),
            stopping_rule="velocity_step",
            max_pressure_iter=1,
            max_simple_iter=2000,
        )
        _mesh, _bc, solver = _build(config)
        solver.solve_steady()
        assert (solver.converged, solver.stop_reason) == (
            True,
            "velocity_step_below_tol",
        )
        assert corrections[-1].reached_cap is False
        assert solver.pressure_cap_hits == len(corrections) - 1 > 100
        assert np.abs(solver.last_mass_imbalance).max() < 1e-12

    def test_velocity_step_does_not_stop_on_a_capped_correction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The frozen shape: a small velocity step with the faces unbalanced is not a stop.

        The limit of a cap is a correction truncated to nothing: the faces
        and the pressure returned as predicted, flagged capped. The momentum
        predictor alone then settles the tiny cavity into a steady field whose
        velocity step falls below convergence_tol while its imbalance stands
        at the supply's order, the shape of the report's section 8.2. The old
        rule called that converged; the new one runs to max_simple_iter. The
        second half isolates the refusal from the physics: the real,
        converging solve with every correction merely flagged capped must not
        stop either.
        """
        original = PressureCorrector.correct

        def truncated(
            self: PressureCorrector, prediction: MomentumPrediction, p: np.ndarray
        ) -> PressureCorrection:
            return PressureCorrection(
                u=np.ascontiguousarray(prediction.u_star),
                v=np.ascontiguousarray(prediction.v_star),
                p=p.copy(),
                p_prime=np.zeros_like(p),
                iterations=self._max_iter,
                reached_cap=True,
            )

        monkeypatch.setattr(PressureCorrector, "correct", truncated)
        config = _ruled(
            "cavity", (6, 6), stopping_rule="velocity_step", max_simple_iter=400
        )
        mesh, _bc, solver = _build(config)
        solver.solve_steady()
        assert (solver.converged, solver.stop_reason) == (False, "max_simple_iter")
        assert solver.pressure_cap_hits == 400
        assert min(solver.residual_history) < config.convergence_tol
        left = np.abs(solver.last_mass_imbalance).max()
        assert left > 1e-2 * config.rho * 1.0 * mesh.dy

        flagged: list[PressureCorrection] = []

        def capped(
            self: PressureCorrector, prediction: MomentumPrediction, p: np.ndarray
        ) -> PressureCorrection:
            out = dataclasses.replace(original(self, prediction, p), reached_cap=True)
            flagged.append(out)
            return out

        monkeypatch.setattr(PressureCorrector, "correct", capped)
        config = _ruled(
            "cavity", (6, 6), stopping_rule="velocity_step", max_simple_iter=300
        )
        _mesh, _bc, solver = _build(config)
        solver.solve_steady()
        assert len(flagged) == 300
        assert min(solver.residual_history) < config.convergence_tol
        assert (solver.converged, solver.stop_reason) == (False, "max_simple_iter")
        assert solver.pressure_cap_hits == 300

    def test_error_estimate_refuses_the_frozen_state_through_continuity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under error_estimate the per-cell condition refuses what a truncated correction leaves."""

        def truncated(
            self: PressureCorrector, prediction: MomentumPrediction, p: np.ndarray
        ) -> PressureCorrection:
            return PressureCorrection(
                u=np.ascontiguousarray(prediction.u_star),
                v=np.ascontiguousarray(prediction.v_star),
                p=p.copy(),
                p_prime=np.zeros_like(p),
                iterations=self._max_iter,
                reached_cap=True,
            )

        monkeypatch.setattr(PressureCorrector, "correct", truncated)
        config = _ruled(
            "cavity", (6, 6), stopping_rule="error_estimate", max_simple_iter=400
        )
        _mesh, _bc, solver = _build(config)
        solver.solve_steady()
        assert (solver.converged, solver.stop_reason) == (False, "max_simple_iter")
        assert min(solver.residual_history) < config.convergence_tol
        assert np.abs(solver.last_mass_imbalance).max() > config.mass_imbalance_tol
