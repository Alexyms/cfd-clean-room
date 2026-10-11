"""Tests for the coupled k-epsilon solve in the staggered solver (ECR-002 step 6, item 3).

ADR-012 D's outer iteration: the momentum prediction with mu + rho nu_t and
the wall functions' viscosity on the wall faces, the pressure correction, one
k and eps step on the corrected faces, the under-relaxed eddy viscosity. Here
the first outer iteration is rebuilt by hand from the public pieces and
compared bit for bit with the solver's; the refusals, the positivity error's
iteration, and what the solver exposes. VAL-016 is in
tests/test_turbulent_channel.py. Each planted defect of prompt 45's list for
commit B fails a test here (docs/reports/probe45/plant45.py).
"""

from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pytest

from src import solver_staggered
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import FLUID, SOLID, Mesh
from src.momentum import MomentumPredictor
from src.pressure import PressureCorrector
from src.solver_staggered import StaggeredSolver, rule_version
from src.staggered import FaceVelocities, allocate_fields
from src.stopping import ErrorEstimateRule, ImbalanceSummary
from src.turbulence import (
    KEpsilonModel,
    PositivityError,
    TurbulenceBoundary,
    TurbulenceConditions,
    TurbulenceState,
)
from validation.transport_cases import AIR, PARTICLES, SOLVER_BLOCK

ALPHA_TURBULENCE = 0.6


def _channel(
    max_simple_iter: int = 1,
    turbulence: bool = True,
    inlet: bool = True,
    inlets: dict | None = None,
) -> SimConfig:
    """A short channel, 1.2 by 0.3 m on 12 by 6 cells, one obstacle on the floor.

    Inlet on the left at 2 m/s, pressure outlet on the right, walls at rest
    above and below. ``inlet`` False closes the left edge; ``inlets`` takes
    the left edge's segments in place of the one inlet.
    """
    boundaries = {
        "outlet": {
            "type": "pressure_outlet",
            "location": "right",
            "y_start": 0.0,
            "y_end": 0.3,
        }
    }
    if inlets is not None:
        boundaries |= inlets
    elif inlet:
        boundaries["inlet"] = {
            "type": "velocity_inlet",
            "location": "left",
            "y_start": 0.0,
            "y_end": 0.3,
            "velocity": 2.0,
            **(
                {"turbulence_intensity": 0.05, "dissipation_length": 0.03}
                if turbulence
                else {}
            ),
        }
    raw = {
        "domain": {"width": 1.2, "height": 0.3, "nx": 12, "ny": 6},
        "fluid": AIR,
        "particles": PARTICLES,
        "solver": {
            **SOLVER_BLOCK,
            "max_simple_iter": max_simple_iter,
            "stopping_rule": "error_estimate",
        },
        "boundaries": boundaries,
        "obstacles": [
            {
                "name": "step",
                "x_start": 0.5,
                "x_end": 0.7,
                "y_start": 0.0,
                "y_end": 0.05,
            }
        ],
        "sensors": [{"name": "centre", "x": 0.6, "y": 0.15}],
        "thresholds": {"5e-06": 100.0},
    }
    if turbulence:
        raw["turbulence"] = {
            "model": "k_epsilon",
            "wall_treatment": "scalable_wall_functions",
            "cfl_number": 0.5,
            "alpha_turbulence": ALPHA_TURBULENCE,
            "max_iter": 500,
            "tol": 1.0e-12,
        }
    return SimConfig.from_dict(raw)


# The left inlet split in two, 2 m/s below and 1 m/s above, with different k
# and eps, so the inflow-weighted start is not either inlet's.
TWO_INLETS = {
    "low": {
        "type": "velocity_inlet",
        "location": "left",
        "y_start": 0.0,
        "y_end": 0.15,
        "velocity": 2.0,
        "turbulence_intensity": 0.05,
        "dissipation_length": 0.03,
    },
    "high": {
        "type": "velocity_inlet",
        "location": "left",
        "y_start": 0.15,
        "y_end": 0.3,
        "velocity": 1.0,
        "turbulence_intensity": 0.2,
        "dissipation_length": 0.3,
    },
}


def _solver(config: SimConfig) -> tuple[Mesh, StaggeredBoundary, StaggeredSolver]:
    mesh = Mesh(config)
    boundary = StaggeredBoundary(mesh, config)
    return mesh, boundary, StaggeredSolver(mesh, config, boundary)


@pytest.mark.integration
class TestTheOuterIteration:
    """ADR-012 D's order, rebuilt by hand from the public pieces, bit for bit."""

    def test_the_first_outer_iteration(self) -> None:
        config = _channel(max_simple_iter=1)
        mesh, boundary, solver = _solver(config)
        solver.solve_steady()

        model = KEpsilonModel(mesh, config)
        walls = TurbulenceBoundary(mesh, config, boundary)
        predictor = MomentumPredictor(mesh, config, boundary)
        corrector = PressureCorrector(mesh, config, boundary)
        start = model.initial(*walls.initial_values())
        k_in = 1.5 * (0.05 * 2.0) ** 2
        np.testing.assert_array_equal(start.k[mesh.cell_type != 1], k_in)

        u, v, p = allocate_fields(mesh)
        boundary.apply_normal_velocity(u, v)
        u[:, -1] = u[:, -2]  # the pressure outlet's copy
        mu_eff = config.mu + config.rho * start.nu_t
        mu_eff[mesh.cell_type == 1] = config.mu
        wall_mu = walls.wall_viscosity(start.k, predictor.stencil_viscosity(mu_eff))
        prediction = predictor.predict(u, v, p, mu_eff=mu_eff, wall_mu=wall_mu)
        corrected = corrector.correct(prediction, p)
        faces = FaceVelocities.copy_of(corrected.u, corrected.v)
        stepped = model.step(start, faces, walls.conditions(start, faces))
        nu_t = (1.0 - ALPHA_TURBULENCE) * start.nu_t + ALPHA_TURBULENCE * stepped.nu_t

        assert solver.face_velocities is not None
        np.testing.assert_array_equal(solver.face_velocities.u, corrected.u)
        np.testing.assert_array_equal(solver.face_velocities.v, corrected.v)
        state = solver.turbulence_state
        assert state is not None
        np.testing.assert_array_equal(state.k, stepped.k)
        np.testing.assert_array_equal(state.eps, stepped.eps)
        np.testing.assert_array_equal(state.nu_t, nu_t)
        # The relaxation is visible: the step's own nu_t is not the state's.
        assert not np.array_equal(state.nu_t, stepped.nu_t)

    def test_the_wall_functions_reach_the_prediction(self) -> None:
        """The prediction with the wall viscosity differs from the one without."""
        config = _channel()
        mesh, boundary, _ = _solver(config)
        model = KEpsilonModel(mesh, config)
        walls = TurbulenceBoundary(mesh, config, boundary)
        predictor = MomentumPredictor(mesh, config, boundary)
        start = model.initial(*walls.initial_values())
        u, v, p = allocate_fields(mesh)
        boundary.apply_normal_velocity(u, v)
        mu_eff = config.mu + config.rho * start.nu_t
        default = predictor.stencil_viscosity(mu_eff)
        wall_mu = walls.wall_viscosity(start.k, default)
        changed = wall_mu["u"] != default["u"]
        np.testing.assert_array_equal(changed, walls.wall_faces["u"])
        with_walls = predictor.predict(u, v, p, mu_eff=mu_eff, wall_mu=wall_mu)
        without = predictor.predict(u, v, p, mu_eff=mu_eff)
        assert not np.array_equal(with_walls.a_p_u, without.a_p_u)


@pytest.mark.integration
class TestWhatTheSolverExposes:
    def test_the_state_is_read_only_and_the_stage_is_timed(self) -> None:
        _, _, solver = _solver(_channel(max_simple_iter=3))
        assert solver.turbulence_state is None
        solver.solve_steady()
        state = solver.turbulence_state
        assert isinstance(state, TurbulenceState)
        for array in (state.k, state.eps, state.nu_t):
            assert not array.flags.writeable
        assert set(solver.stage_seconds) == {
            "momentum",
            "pressure",
            "turbulence",
            "correct",
        }
        assert solver.stage_seconds["turbulence"] > 0.0

    def test_a_laminar_solve_has_no_turbulence_state(self) -> None:
        _, _, solver = _solver(_channel(max_simple_iter=3, turbulence=False))
        solver.solve_steady()
        assert solver.turbulence_state is None
        assert set(solver.stage_seconds) == {"momentum", "pressure", "correct"}

    def test_the_rule_version_is_recorded(self) -> None:
        """5 with condition (e), 3 without; the function agrees without a solver."""
        config, laminar_config = _channel(), _channel(turbulence=False)
        _, _, coupled = _solver(config)
        _, _, laminar = _solver(laminar_config)
        assert laminar.rule_version == rule_version(laminar_config) == 3
        assert coupled.rule_version == rule_version(config) == 5

    def test_condition_e_receives_the_largest_change_of_nu_t(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each outer iteration hands the rule max |nu_t_new - nu_t_old| over non-SOLID cells.

        Test 45 B1: the step (e) is fed was pinned by nothing. The eddy
        viscosity before and after each of four outer iterations is taken
        from solves capped at 1, 2, 3 and 4 iterations (the solve is
        deterministic) and from the start state, and the largest change is
        formed here; the rule records what the solver passed it.
        """
        passed, _, mesh, history = _four_outer_iterations(monkeypatch)
        live = mesh.cell_type != 1
        expected = [
            float(np.abs(after[live] - before[live]).max())
            for before, after in pairwise(history)
        ]
        assert passed == expected
        # Not every change is in one row of cells, so a step over part of the
        # field would show.
        rows = {
            int(np.argmax(np.abs(after - before).max(axis=1)))
            for before, after in pairwise(history)
        }
        assert rows != {0}

    def test_condition_e_scales_by_the_median_of_the_iterate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each outer iteration's scale is nu plus the median nu_t of its own FLUID cells.

        ADR-012 E, amended 2026-10-11 (rule version 5). The scale is formed
        here from the solves capped at 1 to 4 iterations, over the FLUID
        cells (the channel's obstacle and its ring are not FLUID), and it
        changes from one iteration to the next, so a scale formed once at
        the first iteration fails. Defects caught: the first iteration's
        median kept; the median over every non-SOLID cell; the inlet scale.
        """
        _, scales, mesh, history = _four_outer_iterations(monkeypatch)
        config = _channel()
        fluid = mesh.cell_type == FLUID
        assert not np.array_equal(fluid, mesh.cell_type != SOLID)
        nu = config.mu / config.rho
        expected = [nu + float(np.median(after[fluid])) for after in history[1:]]
        assert scales == expected
        assert len(set(scales)) == len(scales)
        every_live = [
            nu + float(np.median(after[mesh.cell_type != SOLID]))
            for after in history[1:]
        ]
        assert scales != every_live
        inlet = TurbulenceBoundary(
            mesh, config, StaggeredBoundary(mesh, config)
        ).largest_inlet_eddy_viscosity()
        assert all(scale != pytest.approx(nu + inlet) for scale in scales)


def _four_outer_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[float], list[float], Mesh, list[np.ndarray]]:
    """What the solver hands condition (e) over four outer iterations, and nu_t around them.

    Returns the viscosity steps and scales the rule received, the channel's
    mesh, and the eddy viscosity at the start and after each of the four
    iterations, from solves capped at 1, 2, 3 and 4 (the solve is
    deterministic).
    """
    steps: list[float] = []
    scales: list[float] = []

    class Recording(ErrorEstimateRule):
        def update(
            self,
            step: float,
            imbalance: Callable[[], ImbalanceSummary],
            viscosity_step: float | None = None,
            nu_scale: float | None = None,
        ) -> bool:
            assert viscosity_step is not None and nu_scale is not None
            steps.append(viscosity_step)
            scales.append(nu_scale)
            return super().update(
                step, imbalance, viscosity_step=viscosity_step, nu_scale=nu_scale
            )

    n = 4
    with monkeypatch.context() as patch:
        patch.setattr(solver_staggered, "ErrorEstimateRule", Recording)
        _, _, solver = _solver(_channel(max_simple_iter=n))
        solver.solve_steady()
    assert len(steps) == len(scales) == n

    config = _channel()
    mesh, boundary, _ = _solver(config)
    start = KEpsilonModel(mesh, config).initial(
        *TurbulenceBoundary(mesh, config, boundary).initial_values()
    )
    history = [np.array(start.nu_t)]
    for cap in range(1, n + 1):
        _, _, capped = _solver(_channel(max_simple_iter=cap))
        capped.solve_steady()
        assert capped.turbulence_state is not None
        history.append(np.array(capped.turbulence_state.nu_t))
    return steps, scales, mesh, history


@pytest.mark.unit
class TestRefusals:
    def test_a_prescribed_eddy_viscosity_is_refused_with_the_model_on(self) -> None:
        mesh, _, solver = _solver(_channel())
        with pytest.raises(ValueError, match="one source of nu_t"):
            solver.solve_steady(eddy_viscosity=np.zeros(mesh.cell_type.shape))
        assert solver.residual_history == []

    def test_a_coupled_solve_needs_an_inlet_that_admits_air(self) -> None:
        config = _channel(inlet=False)
        mesh = Mesh(config)
        with pytest.raises(ValueError, match="no velocity inlet admits air"):
            StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))

    def test_a_positivity_error_names_the_outer_iteration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, _, solver = _solver(_channel(max_simple_iter=5))
        calls = []
        real_step = KEpsilonModel.step

        def step(
            self: KEpsilonModel,
            state: TurbulenceState,
            faces: FaceVelocities,
            conditions: TurbulenceConditions,
            dt: float | None = None,
        ) -> TurbulenceState:
            calls.append(1)
            if len(calls) == 3:
                raise PositivityError("k is not positive and finite at 1 cell", -1.0)
            return real_step(self, state, faces, conditions, dt)

        monkeypatch.setattr(KEpsilonModel, "step", step)
        with pytest.raises(PositivityError, match="outer iteration 2: k is not") as err:
            solver.solve_steady()
        assert err.value.minimum == -1.0
