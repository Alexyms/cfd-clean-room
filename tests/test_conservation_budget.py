"""The conservation budget of the implicit transport solve (ADR-016, REQ-T05; prompt 78).

The implicit solve stops on its largest cell residual, and the part of the
change it has not made sums to ``dt sum_P r_P``, the leftover
(``docs/reports/budget_residual_mechanism.md``, section 2). Pinned here, each
test naming in its docstring the planted defect it was shown to catch
(``docs/reports/conservation_budget.md``, section 4):

- ``scalar_scheme.implicit_step`` returns the leftover of the field it
  returns, ``dt`` times the residual summed over the cells, recomputed here
  face by face; with ``leftover_bound`` the solve stops only when the
  leftover is within it as well, under Jacobi and conjugate gradients, at
  the start iterate and in the loop; a bound that cannot be met leaves the
  solve unconverged at its cap; a capped conjugate gradient iterate's
  leftover is formed; a per-cell step infinite in a SOLID cell leaves the
  leftover finite; a negative, NaN or bool bound is refused.
- Without the argument the solve is the one with an infinite bound, bit for
  bit, for every system the transport solver and the k-epsilon model hand
  it, fallbacks included.
- ``MassBudget`` books the leftover as its own term; a leftover unknown (a
  budget read from an older run file) leaves the residual as it was and
  refuses its own ratio and a restore.
- ``TransportSolver.solve_timestep`` books each step's leftover, gives each
  step the share ``leftover_share * conservation_limit * S * dt / t_end``,
  and raises ConservationBreachError when the accumulated leftover reaches
  ``conservation_limit`` times the supply, not the content: the share of the
  first release step carries that step's own source, and a room whose
  content is far below its supply is refused at ``L S`` and not before.

The 80x30 room's window spanning a release, the runner's exit code and the
older run file's restart are in tests/test_time_integration.py
(TestConservationBudget), on the kept flow state the suite already solves.
"""

import math

import numpy as np
import pytest

from src import scalar_scheme, solver_transport, turbulence
from src.boundary_concentration import SURFACE_FLOOR
from src.mesh import SOLID, Mesh
from src.scalar_scheme import CONJUGATE_GRADIENT, JACOBI, implicit_step
from src.solver_transport import (
    DEPOSIT_SURFACES,
    ConservationBreachError,
    MassBudget,
    TransportSolver,
)
from src.staggered import u_shape, v_shape
from tests.test_turbulence import _patchy_start, _random_case
from validation.transport_cases import (
    FixedConditions,
    ScalarPhysics,
    conditions_with,
    conservation_keys,
    random_face_field,
    transport_config,
    uniform_face_field,
)

LIMIT = conservation_keys()["conservation_limit"]
SHARE = conservation_keys()["leftover_share"]


def _system(
    seed: int, nx: int = 12, ny: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A transport-like system on a box with an obstacle: diffusion and a wall sink.

    Returns c_star, the volumes, g_u, g_v, the sink and the SOLID mask. The
    conductances vary face by face, so the weight ``(sum G) / a_P`` of the
    mechanism report is not uniform and the leftover is not zero.
    """
    config = transport_config(
        1.2,
        0.8,
        nx,
        ny,
        obstacles=[
            {"name": "b", "x_start": 0.5, "x_end": 0.7, "y_start": 0.0, "y_end": 0.3}
        ],
    )
    mesh = Mesh(config)
    rng = np.random.default_rng(seed)
    solid = mesh.cell_type == SOLID
    g_u = np.zeros(u_shape(mesh))
    g_v = np.zeros(v_shape(mesh))
    inner_u = ~solid[:, :-1] & ~solid[:, 1:]
    inner_v = ~solid[:-1, :] & ~solid[1:, :]
    g_u[:, 1:-1] = np.where(inner_u, rng.uniform(0.0, 0.05, inner_u.shape), 0.0)
    g_v[1:-1, :] = np.where(inner_v, rng.uniform(0.0, 0.05, inner_v.shape), 0.0)
    volume = np.outer(mesh.dy_cell, mesh.dx_cell)
    c_star = np.where(solid, 0.0, rng.uniform(0.0, 1.0e4, size=solid.shape))
    sink = np.zeros_like(volume)
    sink[0, :] = np.where(solid[0, :], 0.0, 0.01 * mesh.dx_cell)
    return c_star, volume, g_u, g_v, sink, solid


def _residual(
    c: np.ndarray,
    c_star: np.ndarray,
    volume: np.ndarray,
    dt: float | np.ndarray,
    g_u: np.ndarray,
    g_v: np.ndarray,
    sink: np.ndarray,
    fixed: np.ndarray,
) -> np.ndarray:
    """``b + sum G C_N - a_P C`` face by face, zero at the fixed cells."""
    over_dt = volume / dt
    r = over_dt * c_star - (over_dt + sink) * c
    r[:, 1:] += g_u[:, 1:-1] * (c[:, :-1] - c[:, 1:])
    r[:, :-1] += g_u[:, 1:-1] * (c[:, 1:] - c[:, :-1])
    r[1:, :] += g_v[1:-1, :] * (c[:-1, :] - c[1:, :])
    r[:-1, :] += g_v[1:-1, :] * (c[1:, :] - c[:-1, :])
    return np.where(fixed, 0.0, r)


def _same_result(
    a: scalar_scheme.ImplicitResult, b: scalar_scheme.ImplicitResult
) -> None:
    """Two results, every field bit for bit."""
    assert a.field.tobytes() == b.field.tobytes()
    assert (a.method, a.iterations, a.converged, a.fallbacks, a.fallback_sweeps) == (
        b.method,
        b.iterations,
        b.converged,
        b.fallbacks,
        b.fallback_sweeps,
    )
    assert np.float64(a.leftover).tobytes() == np.float64(b.leftover).tobytes()


# ---------------------------------------------------------------------------
# implicit_step: the leftover and its bound
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLeftover:
    @pytest.mark.parametrize("method", [JACOBI, CONJUGATE_GRADIENT])
    def test_the_leftover_is_dt_times_the_summed_residual_of_the_field(
        self, method: str
    ) -> None:
        """At a loose tolerance, recomputed face by face at the returned field.

        A scalar step and a per-cell one, with held cells under CG. Defect
        caught: the scalar step applied twice to the sum.
        """
        c_star, volume, g_u, g_v, sink, solid = _system(3)
        held = np.zeros_like(solid)
        held[-1, 2] = True
        for dt in (0.7, np.random.default_rng(4).uniform(0.2, 2.0, solid.shape)):
            result = implicit_step(
                c_star, volume, dt, g_u, g_v, sink, solid, 1e-3, 500,
                held=held, method=method,
            )  # fmt: skip
            assert result.iterations > 0
            r = _residual(
                result.field, c_star, volume, dt, g_u, g_v, sink, solid | held
            )
            expected = float(np.sum(np.where(solid, 0.0, dt) * r))
            scale = float(np.sum(volume * c_star))
            assert expected != 0.0
            assert abs(result.leftover - expected) <= 1e-12 * scale

    def test_the_leftover_is_the_change_of_total_the_sink_does_not_make(self) -> None:
        """The identity the booking rests on: total(C*) - total(C) = dt sum D C + leftover.

        Defect caught: a leftover of the wrong sign, which would double the
        budget's residual instead of closing it.
        """
        c_star, volume, g_u, g_v, sink, solid = _system(5)
        dt = 0.9
        result = implicit_step(c_star, volume, dt, g_u, g_v, sink, solid, 1e-3, 500)
        before = float(np.sum(volume * c_star))
        after = float(np.sum(volume * result.field))
        deposited = dt * float(np.sum(sink * result.field))
        assert abs(result.leftover) > 1e-6 * before
        assert abs((before - after - deposited) - result.leftover) <= 1e-12 * before

    @pytest.mark.parametrize("method", [JACOBI, CONJUGATE_GRADIENT])
    def test_a_bound_is_met_and_costs_iterations(self, method: str) -> None:
        """A bound a hundredth of the unbounded leftover: met, after more iterations.

        Defect caught: the bound ignored by the stop's second clause (the
        unbounded solve returned), under both methods.
        """
        c_star, volume, g_u, g_v, sink, solid = _system(7)
        args = (c_star, volume, 0.8, g_u, g_v, sink, solid, 1e-4, 2000)
        free = implicit_step(*args, method=method)
        bound = 0.01 * abs(free.leftover)
        held = implicit_step(*args, method=method, leftover_bound=bound)
        assert free.converged and held.converged
        assert abs(free.leftover) > bound
        assert abs(held.leftover) <= bound
        assert held.iterations > free.iterations
        assert held.fallbacks == 0

    def test_a_bound_that_cannot_be_met_stops_at_the_cap_unconverged(self) -> None:
        """A zero bound on a system whose leftover is not zero within five sweeps.

        The tolerance is one the max-norm test meets from the first iterate,
        so only the bound's clause keeps the solve going. Defect caught: a
        capped solve reported converged on the max-norm test alone.
        """
        c_star, volume, g_u, g_v, sink, solid = _system(9)
        args = (c_star, volume, 0.8, g_u, g_v, sink, solid, 1e6, 5)
        assert implicit_step(*args).iterations == 0
        result = implicit_step(*args, leftover_bound=0.0)
        assert result.iterations == 5 and not result.converged
        assert result.leftover != 0.0

    @pytest.mark.parametrize("bad", [-1e-9, math.nan])
    def test_a_negative_or_nan_bound_is_refused(self, bad: float) -> None:
        c_star, volume, g_u, g_v, sink, solid = _system(1)
        with pytest.raises(ValueError, match="leftover_bound"):
            implicit_step(
                c_star, volume, 0.8, g_u, g_v, sink, solid, 1e-6, 10,
                leftover_bound=bad,
            )  # fmt: skip

    @pytest.mark.parametrize("bad", [True, False, np.True_, np.False_])
    def test_a_bool_bound_is_refused(self, bad: bool) -> None:
        """True would read as the bound 1.0 and False as 0.0 (review 78 S4).

        Defect caught: the bool check removed, so both are accepted as numbers.
        """
        c_star, volume, g_u, g_v, sink, solid = _system(1)
        with pytest.raises(TypeError, match="leftover_bound"):
            implicit_step(
                c_star, volume, 0.8, g_u, g_v, sink, solid, 1e-6, 10,
                leftover_bound=bad,
            )  # fmt: skip

    @pytest.mark.parametrize("method", [JACOBI, CONJUGATE_GRADIENT])
    def test_a_step_infinite_in_a_solid_cell_leaves_the_leftover_finite(
        self, method: str
    ) -> None:
        """A per-cell step that is infinite in the SOLID cells, as the contract allows.

        The leftover and the field are bitwise those under a finite step there:
        the residual is zero in a SOLID cell and the step is not read. Defect
        caught: the per-cell step not masked in SOLID cells (infinity times the
        zero residual is NaN in the sum). The Jacobi sweep divides zero by the
        zero diagonal of an infinite-step SOLID cell and resets the cell, so
        its invalid-value warning is expected here.
        """
        c_star, volume, g_u, g_v, sink, solid = _system(3)
        assert solid.any()
        step = np.random.default_rng(4).uniform(0.2, 2.0, solid.shape)
        rest = (g_u, g_v, sink, solid, 1e-3, 500)
        finite = implicit_step(c_star, volume, step, *rest, method=method)
        with np.errstate(invalid="ignore"):
            infinite = implicit_step(
                c_star, volume, np.where(solid, np.inf, step), *rest, method=method
            )
        assert finite.leftover != 0.0 and math.isfinite(infinite.leftover)
        assert np.float64(infinite.leftover).tobytes() == (
            np.float64(finite.leftover).tobytes()
        )
        assert infinite.field.tobytes() == finite.field.tobytes()

    @pytest.mark.parametrize("method", [JACOBI, CONJUGATE_GRADIENT])
    def test_the_start_is_refused_when_its_leftover_is_over_the_bound(
        self, method: str
    ) -> None:
        """A tolerance the start meets, and a bound half the start's own leftover.

        Without the bound the solve takes no iteration and returns the start;
        with it the start is refused and the solve goes on until the leftover
        is within the bound. Defect caught: the stop's second clause ignored
        at the start iterate under conjugate gradients (the loop's own clause
        is the next test's).
        """
        c_star, volume, g_u, g_v, sink, solid = _system(9)
        dt = 0.8
        args = (c_star, volume, dt, g_u, g_v, sink, solid, 1e6, 2000)
        start = dt * float(
            np.sum(_residual(c_star, c_star, volume, dt, g_u, g_v, sink, solid))
        )
        bound = 0.5 * abs(start)
        free = implicit_step(*args, method=method)
        assert free.iterations == 0 and free.converged
        assert abs(free.leftover - start) <= 1e-12 * abs(start)
        held = implicit_step(*args, method=method, leftover_bound=bound)
        assert held.iterations > 0 and held.converged
        assert abs(held.leftover) <= bound

    def test_a_capped_conjugate_gradient_forms_the_leftover_of_its_iterate(
        self,
    ) -> None:
        """Three iterations of a tight solve: ``accept.leftover`` is the iterate's.

        Called on the helper, since ``implicit_step`` discards a capped CG
        iterate for the Jacobi fallback's and never reads this one. The tight
        stop is never met, so ``test`` is never called and only ``final``
        forms the leftover. Defect caught: ``final`` not called at the cap (the
        leftover then stays at the zero it started from).
        """
        c_star, volume, g_u, g_v, sink, solid = _system(11)
        dt = 0.8
        over_dt = volume / dt
        a_p = over_dt + g_u[:, :-1] + g_u[:, 1:] + g_v[:-1, :] + g_v[1:, :] + sink
        b = over_dt * c_star
        accept = scalar_scheme._Acceptance(dt, solid, 0.0)
        field, iterations, converged = scalar_scheme._conjugate_gradient(
            c_star.copy(), a_p, b, g_u, g_v, solid, 1e-14 * float(np.abs(b).max()), 3,
            accept,
        )  # fmt: skip
        assert iterations == 3 and not converged
        expected = dt * float(
            np.sum(_residual(field, c_star, volume, dt, g_u, g_v, sink, solid))
        )
        scale = float(np.sum(volume * c_star))
        assert expected != 0.0
        assert abs(accept.leftover - expected) <= 1e-12 * scale

    def test_nothing_to_solve_has_no_leftover(self) -> None:
        c_star, volume, g_u, g_v, sink, solid = _system(2)
        zero_u, zero_v = np.zeros_like(g_u), np.zeros_like(g_v)
        result = implicit_step(
            c_star, volume, 0.8, zero_u, zero_v, 0.0 * sink, solid, 1e-6, 10,
            leftover_bound=0.0,
        )  # fmt: skip
        assert result.field is c_star and result.leftover == 0.0 and result.converged


# ---------------------------------------------------------------------------
# REQ-S16's pattern: the argument absent is today's stop, bit for bit
# ---------------------------------------------------------------------------


def _captured(module: object, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Every call ``module`` makes to implicit_step, its arguments copied."""
    calls: list[tuple] = []
    original = scalar_scheme.implicit_step

    def capture(*args: object, **kwargs: object) -> scalar_scheme.ImplicitResult:
        kept = tuple(
            np.array(a, copy=True) if isinstance(a, np.ndarray) else a for a in args
        )
        calls.append((kept, dict(kwargs)))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "implicit_step", capture)
    return calls


@pytest.mark.unit
class TestWithoutTheBound:
    def test_transport_systems_without_it_equal_an_infinite_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ten steps of five classes, settling and deposition on: each system twice.

        Each system the transport solver handed the solve is solved without
        the argument and with an infinite bound, in this process. Defect
        caught: the path without the argument taking a bound of its own (a
        zero, which changes every field).
        """
        config = transport_config(1.0, 0.6, 14, 9, diffusion_tol=1e-8)
        mesh = Mesh(config)
        conditions = conditions_with(mesh, deposition_v=np.full(v_shape(mesh), 2e-3))
        solver = TransportSolver(
            mesh, config, ScalarPhysics(settling=1e-3, diffusion=5e-3),
            FixedConditions(conditions),
        )  # fmt: skip
        calls = _captured(solver_transport, monkeypatch)
        faces = random_face_field(mesh, 11, 0.05)
        c = np.random.default_rng(12).uniform(0.0, 1e5, mesh.cell_type.shape)
        for _ in range(10):
            c = solver.solve_timestep(c, faces, 0, 0.5 * solver.stable_dt(faces, 0))
        assert len(calls) == 10
        for args, kwargs in calls:
            assert kwargs.pop("leftover_bound") is not None
            plain = implicit_step(*args, **kwargs)
            infinite = implicit_step(*args, **kwargs, leftover_bound=math.inf)
            assert plain.iterations > 0
            _same_result(plain, infinite)

    def test_k_and_eps_systems_without_it_equal_an_infinite_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The random field from the patchy start, standard variant, 24 steps.

        From about step 17 the CG solves reach their cap and fall back to
        Jacobi (test 35 check 7), so both paths and the held eps cells are in
        the systems compared. Defect caught as above, on the CG path.
        """
        _, model, faces, conditions_for = _random_case("standard", "upwind")
        calls = _captured(turbulence, monkeypatch)
        state = model.state(*_patchy_start(model, model._volume.shape, seed=3))
        for _ in range(24):
            state = model.step(state, faces, conditions_for(state))
        assert len(calls) == 48
        fell_back = 0
        for args, kwargs in calls:
            assert "leftover_bound" not in kwargs
            plain = implicit_step(*args, **kwargs)
            infinite = implicit_step(*args, **kwargs, leftover_bound=math.inf)
            fell_back += plain.fallbacks
            _same_result(plain, infinite)
        assert fell_back > 0


# ---------------------------------------------------------------------------
# MassBudget's new term
# ---------------------------------------------------------------------------


def _budget(leftover: float | None) -> MassBudget:
    return MassBudget(
        initial=100.0,
        inflow=20.0,
        outflow=5.0,
        source=10.0,
        deposited=dict(zip(DEPOSIT_SURFACES, (1.0, 2.0, 3.0, 0.0), strict=True)),
        current=118.5,
        leftover=leftover,
    )


@pytest.mark.unit
class TestMassBudgetLeftover:
    def test_the_residual_books_the_leftover(self) -> None:
        """130 supplied, 11 out and deposited, 118.5 in the room, 0.5 left over.

        Defect caught: the leftover added instead of subtracted.
        """
        budget = _budget(0.5)
        assert budget.supplied() == 130.0
        assert budget.residual() == 0.0
        assert budget.relative() == 0.0
        assert budget.leftover_relative() == 0.5 / 130.0

    def test_an_unknown_leftover_leaves_the_residual_as_it_was(self) -> None:
        """A budget read from an older run file: the residual holds the leftover.

        Defect caught: an unknown leftover read as zero by the ratio, which
        would report an older run's leftover as zero.
        """
        budget = _budget(None)
        assert budget.residual() == 0.5
        assert budget.relative() == 0.5 / 130.0
        with pytest.raises(ValueError, match="leftover is unknown"):
            budget.leftover_relative()

    def test_before_a_step_and_without_supply_the_ratios_refuse(self) -> None:
        with pytest.raises(ValueError, match="no field yet"):
            MassBudget().supplied()
        empty = MassBudget(initial=0.0, current=0.0)
        with pytest.raises(ValueError, match="no particles"):
            empty.leftover_relative()

    def test_a_restore_with_an_unknown_leftover_is_refused(self) -> None:
        """Defect caught: an older checkpoint's budget restored, its leftover lost."""
        config = transport_config(1.0, 0.5, 6, 3)
        mesh = Mesh(config)
        solver = TransportSolver(
            mesh,
            config,
            ScalarPhysics(0.0, 1e-3),
            FixedConditions(conditions_with(mesh)),
        )
        n = len(config.particle_sizes)
        with pytest.raises(ValueError, match="leftover is unknown"):
            solver.restore_budgets([_budget(None)] + [_budget(0.0)] * (n - 1))
        solver.restore_budgets([_budget(0.25)] * n)
        assert [b.leftover for b in solver.budget] == [0.25] * n


# ---------------------------------------------------------------------------
# The solver: the booking, the share and the refusal
# ---------------------------------------------------------------------------


def _closed_box(
    max_diffusion_iter: int = 500, diffusion_tol: float = 1e-3, t_end: float = 1.0
) -> tuple[Mesh, TransportSolver]:
    """A closed box at rest with diffusion and a floor sink, five classes."""
    config = transport_config(
        1.0,
        0.5,
        12,
        6,
        max_diffusion_iter=max_diffusion_iter,
        diffusion_tol=diffusion_tol,
    )
    if t_end != config.t_end:
        raw = config.merged_mapping()
        raw["solver"]["t_end"] = t_end
        config = type(config).from_dict(raw)
    mesh = Mesh(config)
    deposition = np.zeros(v_shape(mesh))
    deposition[0, :] = 1e-3
    surface = np.zeros(v_shape(mesh), dtype=np.int32)
    surface[0, :] = SURFACE_FLOOR
    solver = TransportSolver(
        mesh,
        config,
        ScalarPhysics(settling=0.0, diffusion=2e-2),
        FixedConditions(
            conditions_with(mesh, deposition_v=deposition, surface_v=surface)
        ),
    )
    return mesh, solver


@pytest.mark.integration
class TestSolverBudget:
    def test_each_step_books_its_leftover_and_the_budget_closes(self) -> None:
        """Forty steps at a loose tolerance: closed to rounding with the term booked.

        Without it the residual is the accumulated leftover, many orders above
        rounding. Defect caught: the booking removed (the residual is then the
        leftover, prompt 77's T5).
        """
        mesh, solver = _closed_box()
        still = uniform_face_field(mesh, 0.0, 0.0)
        c = np.random.default_rng(21).uniform(0.0, 1e5, mesh.cell_type.shape)
        total = 0.0
        for _ in range(40):
            c = solver.solve_timestep(c, still, 0, 0.02)
            total += solver.last_leftover
        budget = solver.budget[0]
        assert budget.leftover == total
        assert abs(budget.leftover) > 1e-9 * budget.supplied()
        assert abs(budget.relative()) <= 1e-13
        assert abs(budget.relative() + budget.leftover_relative()) > 1e-9

    def test_each_step_s_leftover_is_within_its_share(self) -> None:
        """The share f L S dt / T, recomputed, at every step; T is solver.t_end.

        At diffusion_tol 1e-3 the max-norm stop alone leaves more than the
        share, so the share makes the solve sweep on. Defects caught: the share
        ignored in the stop; T fixed at 1 s instead of read from t_end; the
        share formed from the content instead of the supply.
        """
        for t_end in (1.0, 4.0):
            mesh, solver = _closed_box(t_end=t_end)
            still = uniform_face_field(mesh, 0.0, 0.0)
            c = np.random.default_rng(22).uniform(0.0, 1e5, mesh.cell_type.shape)
            dt = 0.02
            accumulated = 0.0
            for n in range(1, 41):
                c = solver.solve_timestep(c, still, 0, dt)
                supplied = solver.budget[0].supplied()
                share = SHARE * LIMIT * supplied * dt / t_end
                assert solver.last_leftover_bound == share
                assert abs(solver.last_leftover) <= share
                accumulated += solver.last_leftover
                assert abs(accumulated) <= SHARE * LIMIT * supplied * n * dt / t_end
            assert solver.diffusion_converged

    def test_the_share_made_the_solve_take_more_sweeps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same steps with the bound and without: more sweeps with it.

        Shows the share bites in the case above, so the test above can fail.
        """
        mesh, solver = _closed_box()
        calls = _captured(solver_transport, monkeypatch)
        still = uniform_face_field(mesh, 0.0, 0.0)
        c = np.random.default_rng(22).uniform(0.0, 1e5, mesh.cell_type.shape)
        for _ in range(5):
            c = solver.solve_timestep(c, still, 0, 0.02)
        more = 0
        for args, kwargs in calls:
            bounded = implicit_step(*args, **kwargs)
            kwargs.pop("leftover_bound")
            free = implicit_step(*args, **kwargs)
            more += bounded.iterations > free.iterations
        assert more == len(calls)

    def test_the_refusal_fires_when_the_accumulated_leftover_reaches_the_limit(
        self,
    ) -> None:
        """A one-sweep cap: the share cannot hold, and the accumulated leftover
        passes L S within a few steps; the step that takes it there raises.

        Defect caught: the check removed (the run goes on past REQ-T05).
        """
        mesh, solver = _closed_box(max_diffusion_iter=1, diffusion_tol=1e-12)
        still = uniform_face_field(mesh, 0.0, 0.0)
        c = np.random.default_rng(23).uniform(0.0, 1e5, mesh.cell_type.shape)
        before = 0.0
        with pytest.raises(ConservationBreachError) as caught:
            for _ in range(200):
                c = solver.solve_timestep(c, still, 0, 0.02)
                before = solver.budget[0].leftover
        budget = solver.budget[0]
        assert caught.value.size_class == 0
        assert caught.value.leftover == budget.leftover
        assert caught.value.supplied == budget.supplied()
        assert caught.value.limit == LIMIT
        assert abs(budget.leftover) >= LIMIT * budget.supplied()
        assert "conservation_limit" in str(caught.value)
        # The step that raised was the first to reach the limit, so a threshold
        # set too high would let the steps before it through and fail here.
        assert abs(before) < LIMIT * budget.supplied()

    def test_the_refusal_reads_the_supply_not_the_content(self) -> None:
        """A room whose content is a thousandth of its supply, as late in a clean-out.

        The accumulated leftover is set to a fraction of ``L S`` through
        ``restore_budgets`` and one more step is taken. At 0.5 and 0.99 of
        ``L S`` it is far above ``L`` times the content and below ``L S``: not
        refused. At 1.01 of ``L S`` it is refused. Defects caught: the check
        reading the content instead of the supply (a clean-out inside REQ-T05
        would be stopped); the threshold ten times the limit, which the
        runner's test alone caught before (test 78 B2, T-S4).
        """
        for fraction, refused in ((0.5, False), (0.99, False), (1.01, True)):
            mesh, solver = _closed_box()
            still = uniform_face_field(mesh, 0.0, 0.0)
            c = np.random.default_rng(24).uniform(0.0, 1e5, mesh.cell_type.shape)
            content = MassBudget.in_domain(c, mesh)
            supply = 1000.0 * content
            leftover = fraction * LIMIT * supply
            # Closed: what was supplied is in the room or has left, less the leftover.
            booked = MassBudget(
                initial=supply,
                outflow=supply - content - leftover,
                current=content,
                leftover=leftover,
            )
            solver.restore_budgets([booked, *(MassBudget() for _ in solver.budget[1:])])
            if refused:
                with pytest.raises(ConservationBreachError) as caught:
                    solver.solve_timestep(c, still, 0, 0.02)
                assert caught.value.supplied == supply
                assert abs(caught.value.leftover) >= LIMIT * supply
            else:
                solver.solve_timestep(c, still, 0, 0.02)
                budget = solver.budget[0]
                assert abs(budget.leftover) > 100.0 * LIMIT * budget.current
                assert abs(budget.leftover) < LIMIT * budget.supplied()

    def test_the_first_release_step_gets_a_share_with_its_own_source_in_the_supply(
        self,
    ) -> None:
        """From an empty room, the first step's supply is its own source, and so is its share.

        Before the step the budget holds nothing, so a share formed from the
        supply before the step's inflow and source are booked is zero for the
        whole first step. Defect caught: that share (test 78 T-S2).
        """
        t_end, dt = 2.0, 0.02
        mesh, solver = _closed_box(t_end=t_end)
        still = uniform_face_field(mesh, 0.0, 0.0)
        rate = np.zeros(mesh.cell_type.shape)
        rate[3, 5] = 2.0e3
        released = float(np.sum(rate * np.outer(mesh.dy_cell, mesh.dx_cell))) * dt
        solver.solve_timestep(
            np.zeros(mesh.cell_type.shape), still, 0, dt, sources=rate
        )
        budget = solver.budget[0]
        assert budget.initial == 0.0 and budget.inflow == 0.0
        assert budget.supplied() == pytest.approx(released, rel=1e-12)
        share = SHARE * LIMIT * budget.supplied() * dt / t_end
        assert share > 0.0
        assert solver.last_leftover_bound == share
        assert abs(solver.last_leftover) <= share

    def test_with_the_share_working_the_refusal_does_not_fire(self) -> None:
        """The same box and steps under the file's cap: the share holds, no breach."""
        mesh, solver = _closed_box(diffusion_tol=1e-12)
        still = uniform_face_field(mesh, 0.0, 0.0)
        c = np.random.default_rng(23).uniform(0.0, 1e5, mesh.cell_type.shape)
        for _ in range(200):
            c = solver.solve_timestep(c, still, 0, 0.02)
        budget = solver.budget[0]
        assert abs(budget.leftover) < SHARE * LIMIT * budget.supplied()
