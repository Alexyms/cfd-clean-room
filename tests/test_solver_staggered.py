"""Tests for the staggered SIMPLE solver (ECR-001 step 6, REQ-S04, REQ-S07, REQ-S12).

The solver is a loop over the step 4 predictor and the step 5 corrector, so
most of what these tests check is the loop: what it hands the callback, what
it resets, that it keeps the closed-domain compatibility step 5 proved for
one correction, and that it never disturbs a Dirichlet face. The corrector's
output at every iteration is observed by wrapping PressureCorrector.correct,
which leaves the solver's code path untouched.
"""

from collections.abc import Callable

import numpy as np
import pytest
import yaml

from src.boundary import BoundaryManager
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import Mesh
from src.momentum import MomentumPrediction, MomentumPredictor
from src.pressure import PressureCorrection, PressureCorrector
from src.solver_ns import IterationState, NavierStokesSolver
from src.solver_staggered import StaggeredSolver
from src.staggered import allocate_fields, to_cell_centers
from src.stopping import ErrorEstimateRule
from validation.cases import CASE_GRIDS, case_path, load_case

EPS = np.finfo(np.float64).eps


def _case(name: str, n: int, **overrides: object) -> SimConfig:
    """A committed case on an n x n grid, with top-level sections overridden."""
    raw = yaml.safe_load(case_path(name).read_text(encoding="utf-8"))
    raw["domain"]["nx"] = raw["domain"]["ny"] = n
    raw.update(overrides)
    return SimConfig.from_dict(raw)


def _channel(nx: int = 12, ny: int = 6) -> SimConfig:
    """The VAL-001 channel on a small grid."""
    return load_case("poiseuille", grid=(nx, ny))


def _ruled(
    name: str, grid: tuple[int, int], boundaries: dict | None = None, **keys: object
) -> SimConfig:
    """A committed case on a grid with solver keys, and optionally boundaries, replaced."""
    raw = yaml.safe_load(case_path(name).read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = grid
    raw["solver"].update(keys)
    if boundaries is not None:
        raw["boundaries"] = boundaries
    return SimConfig.from_dict(raw)


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
    """The collocated solver's public shape, which the harness depends on."""

    def test_returns_three_cell_centered_float64_contiguous_arrays(self) -> None:
        """(u, v, p) are [ny, nx], float64 and C-contiguous, like the collocated return."""
        config = _case("cavity", 6)
        mesh, _bc, solver = _build(config)
        for field in solver.solve_steady():
            assert field.shape == mesh.cell_type.shape
            assert field.dtype == np.float64
            assert field.flags["C_CONTIGUOUS"]

    def test_callback_gets_cell_centered_fields_and_the_corrector_sweeps(
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
        assert [s.pressure_sweeps for s in states] == [c.sweeps for c in corrections]
        assert [s.residual for s in states] == solver.residual_history
        for state, corrected in zip(states, corrections, strict=True):
            u_c, v_c = to_cell_centers(corrected.u, corrected.v)
            assert state.u.shape == state.v.shape == mesh.cell_type.shape
            assert np.array_equal(state.u, u_c)
            assert np.array_equal(state.v, v_c)
            assert state.p is corrected.p
        assert np.array_equal(u, states[-1].u)
        assert solver.last_pressure_sweeps == corrections[-1].sweeps

    def test_sweep_count_and_stage_timers_reset_at_the_start_of_each_solve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A solve that fails before its first correction reports none of the last one."""
        config = _case("cavity", 6)
        _mesh, _bc, solver = _build(config)
        solver.solve_steady()
        assert solver.last_pressure_sweeps > 0
        assert set(solver.stage_seconds) == {"momentum", "pressure", "correct"}
        assert solver.stage_seconds["pressure"] > 0.0

        def fail(
            self: MomentumPredictor, u: np.ndarray, v: np.ndarray, p: np.ndarray
        ) -> MomentumPrediction:
            raise RuntimeError("stop before the first correction")

        monkeypatch.setattr(MomentumPredictor, "predict", fail)
        with pytest.raises(RuntimeError, match="first correction"):
            solver.solve_steady()
        assert solver.last_pressure_sweeps == 0
        assert solver.stage_seconds == {
            "momentum": 0.0,
            "pressure": 0.0,
            "correct": 0.0,
        }
        assert solver.residual_history == []

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
class TestReferenceVelocity:
    """The stopping rule divides by the collocated solver's reference velocity."""

    @pytest.mark.parametrize(
        "case_id", [c for c, (kind, _nx, _ny) in CASE_GRIDS.items() if kind == "cavity"]
    )
    def test_closed_domain_reference_equals_the_collocated_one(
        self, case_id: str
    ) -> None:
        """Equal with ==: same lid speed, same h, same operations in the same order."""
        kind, nx, ny = CASE_GRIDS[case_id]
        config = load_case(kind, grid=(nx, ny))
        mesh, _bc, solver = _build(config)
        collocated = NavierStokesSolver(mesh, config, BoundaryManager(mesh, config))
        # The velocity NavierStokesSolver.solve_steady divides its residual by
        expected = collocated._F_ref / (config.rho * max(mesh.dx, mesh.dy))
        assert solver.reference_velocity == expected

    def test_channel_reference_differs_by_the_collocated_corner_cells(self) -> None:
        """The same formula on each layer's own inlet flux; the collocated one is 38/40.

        The collocated edge map gives the two corner cells of the inlet edge
        to the walls (docs/reports/inlet_flux_comparison.md), so its inlet
        flux, and with it its reference velocity, is 38/40 of the exact one.
        """
        kind, nx, ny = CASE_GRIDS["val001_80x40"]
        config = load_case(kind, grid=(nx, ny))
        mesh, bc, solver = _build(config)
        collocated = NavierStokesSolver(mesh, config, BoundaryManager(mesh, config))
        collocated_ref = collocated._F_ref / (config.rho * max(mesh.dx, mesh.dy))
        exact_flux = 0.1 * 0.5
        assert bc.get_total_inlet_flux() == pytest.approx(exact_flux, rel=1e-14)
        assert solver.reference_velocity == pytest.approx(
            exact_flux / mesh.dx, rel=1e-14
        )
        assert solver.reference_velocity / collocated_ref == pytest.approx(
            40.0 / 38.0, rel=1e-14
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
        """Runs on a stretched mesh, which the collocated solver refuses. No accuracy claim."""
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

    def test_error_estimate_stops_later_with_continuity_met(self) -> None:
        """The velocity_step rule would have stopped earlier on the same trajectory."""
        config = _ruled("cavity", (6, 6), stopping_rule="error_estimate")
        _mesh, _bc, solver = _build(config)
        solver.solve_steady()
        assert solver.stop_reason == "error_estimate_and_continuity"
        assert solver.converged is True
        assert np.abs(solver.last_mass_imbalance).max() < config.mass_imbalance_tol
        # Condition (c): the flux scale of a closed unit cavity is rho * 1 * 1.
        total = np.abs(solver.last_mass_imbalance).sum()
        assert total / config.rho < config.iteration_error_tol
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
        ("name", "grid", "speed", "flux", "reference"),
        [("poiseuille", (12, 6), 0.1, 0.05, 0.6), ("cavity", (6, 6), 1.0, 1.0, 1.0)],
    )
    def test_error_estimate_scales_are_physical_and_the_step_is_in_m_per_s(
        self,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        grid: tuple[int, int],
        speed: float,
        flux: float,
        reference: float,
    ) -> None:
        """One rule at construction and one per solve, never on reference_velocity.

        The velocity scale is the inlet or lid speed, and the flux scale is rho
        times the inflow (0.1 through 0.5) or, closed, the lid speed times the
        side. Test 24 T1: each update gets the step in m/s, the residual times
        reference_velocity (0.6 on the channel), not the residual itself.
        """
        speeds: list[float] = []
        fluxes: list[float] = []
        steps: list[float] = []

        class Spy(ErrorEstimateRule):
            def __init__(
                self, velocity_scale: float, flux_scale: float, *tols: float
            ) -> None:
                speeds.append(velocity_scale)
                fluxes.append(flux_scale)
                super().__init__(velocity_scale, flux_scale, *tols)

            def update(
                self, step: float, imbalance: Callable[[], tuple[float, float]]
            ) -> bool:
                steps.append(step)
                return super().update(step, imbalance)

        monkeypatch.setattr("src.solver_staggered.ErrorEstimateRule", Spy)
        config = _ruled(name, grid, stopping_rule="error_estimate", max_simple_iter=2)
        _mesh, _bc, solver = _build(config)
        solver.solve_steady()
        assert solver.reference_velocity == pytest.approx(reference)
        assert speeds == [speed, speed]
        assert fluxes == pytest.approx([flux, flux], rel=1e-12)
        expected = [r * solver.reference_velocity for r in solver.residual_history]
        assert steps == pytest.approx(expected, rel=1e-12)

    def test_error_estimate_without_a_boundary_velocity_raises(self) -> None:
        """A closed box with no moving wall leaves the estimate without a scale."""
        config = _ruled("cavity", (6, 6), boundaries={}, stopping_rule="error_estimate")
        with pytest.raises(ValueError, match="stopping_rule error_estimate needs"):
            _build(config)
