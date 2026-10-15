"""Tests for the additive-correction multigrid preconditioner (src/multigrid.py, prompt 53).

Four claims, each against an independent reference rather than the module's
own transfer functions:

- every coarse operator is ``P^T A P`` formed densely, with P the
  piecewise-constant prolongation written out cell by cell, on systems with
  SOLID cells, an outlet, a stretched mesh and a closed domain, on grids
  with odd sides so the lone edge cells are exercised;
- the V-cycle is symmetric and positive on the range, the two properties
  conjugate gradients needs of a preconditioner;
- every level's smoother weight is the fixed 2/3, justified by the weak
  diagonal dominance of every level (Gershgorin: lambda_max(D^-1 A) <= 2),
  checked densely;
- MG-PCG reaches the same relative residual as Jacobi-PCG on the same
  systems, the closed cavity among them, and lands within the tolerance's
  worth of a dense solve;
- under the default key the correction is the Jacobi loop's, to the bit.
"""

from collections.abc import Callable

import numpy as np
import pytest
import yaml

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import Mesh
from src.momentum import MomentumPrediction, MomentumPredictor
from src.multigrid import (
    COARSEST_CELLS,
    SMOOTHER_FACTOR,
    Multigrid,
    add_prolonged,
    coarsen,
    restrict,
)
from src.pressure import (
    RESIDUAL_FLOOR,
    PressureCoefficients,
    PressureCorrection,
    PressureCorrector,
    apply_operator,
    conjugate_gradient,
    jacobi_preconditioner,
    multigrid_preconditioner,
)
from src.solver_staggered import StaggeredSolver
from src.staggered import allocate_fields
from validation.cases import case_path

# The tightest level the configuration accepts.
TIGHT_RTOL = 1e-10
# Odd on both sides, so the first coarsening leaves a lone column and row,
# and large enough for three levels: 27x19, 14x10, 7x5.
NX, NY = 27, 19
CHANNEL = {
    "inlet": {
        "type": "velocity_inlet",
        "location": "left",
        "y_start": 0.0,
        "y_end": 1.0,
        "velocity": 0.3,
    },
    "outlet": {
        "type": "pressure_outlet",
        "location": "right",
        "y_start": 0.0,
        "y_end": 1.0,
    },
}
CAVITY = {
    "lid": {
        "type": "velocity_inlet",
        "location": "top",
        "x_start": 0.0,
        "x_end": 2.0,
        "u_velocity": 1.0,
        "v_velocity": 0.0,
    }
}
BLOCK = {"name": "block", "x_start": 0.7, "x_end": 1.3, "y_start": 0.3, "y_end": 0.7}
STRETCHED = {"x": {"stretch_ratio": 1.15}, "y": {"stretch_ratio": 1.25}}
# name: (boundaries, obstacles, mesh block)
SYSTEMS = {
    "open": (CHANNEL, [BLOCK], None),
    "open_stretched": (CHANNEL, [BLOCK], STRETCHED),
    "closed": (CAVITY, [BLOCK], None),
}


def _config(
    boundaries: dict,
    obstacles: list[dict],
    mesh: dict | None,
    preconditioner: str | None = None,
) -> SimConfig:
    solver = {
        "dt": 0.01,
        "t_end": 1.0,
        "output_interval": 10,
        "convergence_tol": 1.0e-6,
        "max_simple_iter": 100,
        "alpha_velocity": 0.7,
        "alpha_pressure": 0.3,
        "max_pressure_iter": 5000,
        "pressure_rtol": TIGHT_RTOL,
    }
    if preconditioner is not None:
        solver["pressure_preconditioner"] = preconditioner
    raw = {
        "domain": {"width": 2.0, "height": 1.0, "nx": NX, "ny": NY},
        "fluid": {"density": 1.2, "viscosity": 0.05, "temperature": 293.0},
        "particles": {
            "density": 1000.0,
            "sizes": [0.1e-6],
            "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {"diameters": [0.1e-6], "efficiencies": [0.99999]},
        },
        "solver": solver,
        "boundaries": boundaries,
        "obstacles": obstacles,
        "sensors": [{"name": "center", "x": 1.0, "y": 0.5}],
        "thresholds": {"0.1e-6": 100.0},
    }
    if mesh is not None:
        raw["mesh"] = mesh
    return SimConfig.from_dict(raw)


def _system(
    name: str, preconditioner: str | None = None
) -> tuple[
    PressureCorrector,
    PressureCoefficients,
    np.ndarray,
    MomentumPrediction,
    np.ndarray,
]:
    """A corrector, its coefficients, b, the prediction and p after five predictor sweeps."""
    boundaries, obstacles, mesh_block = SYSTEMS[name]
    config = _config(boundaries, obstacles, mesh_block, preconditioner)
    mesh = Mesh(config)
    bc = StaggeredBoundary(mesh, config)
    predictor = MomentumPredictor(mesh, config, bc)
    pc = PressureCorrector(mesh, config, bc)
    u, v, p = allocate_fields(mesh)
    bc.apply_normal_velocity(u, v)
    for _ in range(5):
        if boundaries is CHANNEL:
            u[:, -1] = u[:, -2]
        pred = predictor.predict(u, v, p)
        u, v = pred.u_star, pred.v_star
    c = pc.coefficients(pred.a_p_u, pred.a_p_v)
    return pc, c, pc.mass_imbalance(pred.u_star, pred.v_star), pred, p


def _dense_operator(c: PressureCoefficients) -> np.ndarray:
    """A over every cell, column by column from apply_operator."""
    ny, nx = c.a_p.shape
    matrix = np.zeros((ny * nx, ny * nx))
    for k in range(ny * nx):
        e = np.zeros(ny * nx)
        e[k] = 1.0
        matrix[:, k] = apply_operator(c, e.reshape(ny, nx)).ravel()
    return matrix


def _prolongation(shape: tuple[int, ...]) -> np.ndarray:
    """P for 2x2 aggregation, written out cell by cell: cell (j, i) in group (j//2, i//2)."""
    ny, nx = shape
    coarse_nx = (nx + 1) // 2
    matrix = np.zeros((ny * nx, ((ny + 1) // 2) * coarse_nx))
    for j in range(ny):
        for i in range(nx):
            matrix[j * nx + i, (j // 2) * coarse_nx + i // 2] = 1.0
    return matrix


def _dense_cycle(
    cycle: Callable[[np.ndarray], np.ndarray], active: np.ndarray
) -> np.ndarray:
    """A preconditioner as a dense matrix over every cell; columns of inactive cells zero."""
    ny, nx = active.shape
    matrix = np.zeros((ny * nx, ny * nx))
    for k in np.flatnonzero(active.ravel()):
        e = np.zeros(ny * nx)
        e[k] = 1.0
        matrix[:, k] = cycle(e.reshape(ny, nx)).ravel()
    return matrix


def _dominant_system(
    ny: int, nx: int, extra: float, seed: int = 7
) -> PressureCoefficients:
    """Random positive couplings with a diagonal ``extra`` above their sum.

    ``extra`` is an outlet term in every cell: zero gives a closed domain's
    rows, a large value a diagonal far from the couplings, whose largest
    eigenvalue of ``D^-1 A`` is well below 2.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 2.0, size=(ny, nx))
    a_e, a_n = a.copy(), a.copy()
    a_e[:, -1] = 0.0
    a_n[-1, :] = 0.0
    a_w = np.zeros_like(a)
    a_w[:, 1:] = a_e[:, :-1]
    a_s = np.zeros_like(a)
    a_s[1:, :] = a_n[:-1, :]
    return PressureCoefficients(
        a_p=a_e + a_w + a_n + a_s + extra, a_e=a_e, a_w=a_w, a_n=a_n, a_s=a_s
    )


def _largest_eigenvalue(c: PressureCoefficients) -> float:
    """lambda_max of ``D^-1 A`` over the cells with an equation, densely."""
    cells = np.flatnonzero(c.a_p.ravel() > 0.0)
    scale = 1.0 / np.sqrt(c.a_p.ravel()[cells])
    matrix = _dense_operator(c)[np.ix_(cells, cells)]
    return float(np.linalg.eigvalsh(scale[:, None] * matrix * scale)[-1])


def _projected_rhs(
    pc: PressureCorrector, c: PressureCoefficients, b: np.ndarray
) -> np.ndarray:
    """f = -b, projected onto the range on a closed domain, as correct forms it."""
    active = c.a_p > 0.0
    f = -b.copy()
    if pc.needs_pin:
        f[active] -= f[active].mean()
    f[~active] = 0.0
    return f


def _range_basis(active: np.ndarray, closed: bool) -> np.ndarray:
    """Orthonormal columns spanning the cells with an equation, less the constant when closed."""
    cells = np.flatnonzero(active.ravel())
    basis = np.zeros((active.size, cells.size))
    basis[cells, np.arange(cells.size)] = 1.0
    if closed:
        ones = np.zeros(active.size)
        ones[cells] = 1.0 / np.sqrt(cells.size)
        basis -= np.outer(ones, ones @ basis)
        basis = np.linalg.qr(basis)[0][:, : cells.size - 1]
    return basis


def _dense_solution(
    pc: PressureCorrector, c: PressureCoefficients, f: np.ndarray
) -> tuple[np.ndarray, float]:
    """The minimum-norm solution of A x = f over the range, and the smallest eigenvalue there.

    On a closed domain the constant is removed from both, so x is the range
    part of every solution; open, x is the solution.
    """
    active = c.a_p > 0.0
    basis = _range_basis(active, pc.needs_pin)
    reduced = basis.T @ _dense_operator(c) @ basis
    x = basis @ np.linalg.solve(reduced, basis.T @ f.ravel())
    return x.reshape(f.shape), float(np.linalg.eigvalsh(reduced)[0])


def _range_part(pc: PressureCorrector, active: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x with its mean over the cells with an equation removed on a closed domain."""
    out = np.where(active, x, 0.0)
    if pc.needs_pin:
        out[active] -= out[active].mean()
    return out


@pytest.mark.unit
class TestCoarseOperator:
    """Each level's operator against the dense Galerkin product, and the transfers."""

    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_each_level_is_the_dense_galerkin_product(self, name: str) -> None:
        """A_l = P^T A_(l-1) P to rounding on every level, with SOLID cells, an outlet or none.

        The system has SOLID cells (zero rows), an outlet term in its
        diagonal (open) or the constants as null space (closed), and odd
        sides. Defect caught: a coupling inside a group not subtracted from
        the coarse diagonal, or subtracted once instead of twice.
        """
        pc, c, _b, _pred, _p = _system(name)
        assert (c.a_p == 0.0).any()
        assert pc.needs_pin is (name == "closed")
        hierarchy = Multigrid(c, apply_operator)
        shapes = [level.coefficients.a_p.shape for level in hierarchy.levels]
        assert shapes == [(19, 27), (10, 14), (5, 7)]
        fine, fine_shape = _dense_operator(c), c.a_p.shape
        for level in hierarchy.levels[1:]:
            prolongation = _prolongation(fine_shape)
            galerkin = prolongation.T @ fine @ prolongation
            coarse = _dense_operator(level.coefficients)
            assert np.abs(coarse - galerkin).max() <= 1e-13 * np.abs(galerkin).max()
            fine, fine_shape = coarse, level.coefficients.a_p.shape

    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_a_coarse_cell_has_an_equation_when_a_child_has_one(
        self, name: str
    ) -> None:
        """The coarse mask is ``P^T`` of the fine one, and SOLID-only groups have no equation."""
        _pc, c, _b, _pred, _p = _system(name)
        hierarchy = Multigrid(c, apply_operator)
        for fine, coarse in zip(hierarchy.levels, hierarchy.levels[1:], strict=False):
            children = _prolongation(fine.active.shape).T @ fine.active.ravel()
            assert np.array_equal(coarse.active.ravel(), children > 0)
        assert np.array_equal(hierarchy.levels[0].active, c.a_p > 0.0)
        assert not hierarchy.levels[1].active.all()

    def test_restriction_is_the_transpose_of_prolongation(self) -> None:
        """<P^T r, z> = <r, P z> on every parity of the sides, against the dense P."""
        rng = np.random.default_rng(53)
        for shape in ((6, 8), (7, 8), (6, 9), (7, 9), (1, 5), (5, 1)):
            r = rng.standard_normal(shape)
            dense = _prolongation(shape)
            coarse_shape = ((shape[0] + 1) // 2, (shape[1] + 1) // 2)
            z = rng.standard_normal(coarse_shape)
            np.testing.assert_allclose(
                restrict(r).ravel(), dense.T @ r.ravel(), rtol=1e-14
            )
            x = np.zeros(shape)
            add_prolonged(x, z)
            np.testing.assert_array_equal(x.ravel(), dense @ z.ravel())

    def test_coarsening_stops_at_the_coarsest_size_or_a_side_of_one(self) -> None:
        """At most COARSEST_CELLS cells ends it, and so does a side of one cell."""

        def chain(ny: int, nx: int) -> list[tuple[int, int]]:
            c = _dominant_system(ny, nx, 0.1)
            return [
                lv.coefficients.a_p.shape for lv in Multigrid(c, apply_operator).levels
            ]

        assert COARSEST_CELLS == 64
        assert chain(16, 16) == [(16, 16), (8, 8)]
        assert chain(17, 16) == [(17, 16), (9, 8), (5, 4)]
        assert chain(8, 8) == [(8, 8)]
        assert chain(2, 100) == [(2, 100), (1, 50)]
        assert chain(1, 100) == [(1, 100)]

    def test_coarsen_keeps_the_coefficients_type(self) -> None:
        """A coarse level is a PressureCoefficients, so apply_operator reads it as it reads the fine."""
        _pc, c, _b, _pred, _p = _system("open")
        coarse, active = coarsen(c, c.a_p > 0.0)
        assert type(coarse) is PressureCoefficients
        assert coarse.a_p.shape == active.shape == (10, 14)

    def test_shapes_that_differ_are_refused(self) -> None:
        """Five arrays of one two-dimensional shape, or a ValueError."""
        _pc, c, _b, _pred, _p = _system("open")
        bad = PressureCoefficients(
            a_p=c.a_p, a_e=c.a_e[:, :-1], a_w=c.a_w, a_n=c.a_n, a_s=c.a_s
        )
        with pytest.raises(ValueError, match="five two-dimensional"):
            Multigrid(bad, apply_operator)


@pytest.mark.unit
class TestCycle:
    """The V-cycle: its weights, symmetry and positivity, and what it leaves alone."""

    @pytest.mark.parametrize("name", [*sorted(SYSTEMS), "diagonal"])
    def test_every_level_has_the_fixed_weight(self, name: str) -> None:
        """omega is SMOOTHER_FACTOR / 2 on every smoothed level, whatever its coefficients.

        The open channels' levels have a largest eigenvalue of ``D^-1 A``
        just below 2, and the last system has a diagonal far above its
        couplings (1.4 and 1.2 on its two smoothed levels), so a weight
        taken from an estimate of that eigenvalue would differ from 2/3 on
        them. Defect caught: the power-iteration weight restored. The
        coarsest level, solved exactly, has none, and the smoother is the
        weight over the diagonal.
        """
        if name == "diagonal":
            c = _dominant_system(27, 19, extra=10.0)
        else:
            c = _system(name)[1]
        hierarchy = Multigrid(c, apply_operator)
        for level in hierarchy.levels[:-1]:
            assert level.weight == SMOOTHER_FACTOR / 2
            assert level.weight == pytest.approx(2.0 / 3.0, rel=1e-15)
            lc = level.coefficients
            has = lc.a_p > 0.0
            np.testing.assert_allclose(
                level.smoother[has], SMOOTHER_FACTOR / 2 / lc.a_p[has], rtol=1e-14
            )
            assert (level.smoother[~has] == 0.0).all()
        assert hierarchy.levels[-1].weight == 0.0
        assert (hierarchy.levels[-1].smoother == 0.0).all()

    @pytest.mark.parametrize("name", [*sorted(SYSTEMS), "diagonal", "closed_rows"])
    def test_every_level_is_dominant_and_lambda_max_is_at_most_two(
        self, name: str
    ) -> None:
        """Gershgorin on every level, rows and then the dense largest eigenvalue.

        Each row's diagonal is at least the sum of its couplings, so the
        eigenvalues of ``D^-1 A`` lie in [0, 2] and the fixed weight leaves
        ``omega lambda_max`` at most 4/3, inside the contraction limit 2.
        This is the premise of the fixed weight; the systems include one
        with no outlet term (a closed domain's rows) and one with a large
        one.
        """
        if name == "diagonal":
            c = _dominant_system(27, 19, extra=10.0)
        elif name == "closed_rows":
            c = _dominant_system(27, 19, extra=0.0)
        else:
            c = _system(name)[1]
        hierarchy = Multigrid(c, apply_operator)
        for level in hierarchy.levels:
            lc = level.coefficients
            couplings = (lc.a_e + lc.a_w) + (lc.a_n + lc.a_s)
            assert (lc.a_p * (1 + 1e-12) >= couplings).all()
        for level in hierarchy.levels[:-1]:
            largest = _largest_eigenvalue(level.coefficients)
            assert largest <= 2.0 * (1 + 1e-12)
            assert level.weight * largest <= SMOOTHER_FACTOR * (1 + 1e-12)

    def test_a_finest_system_that_is_not_dominant_is_refused(self) -> None:
        """The bound needs dominance: a diagonal below its couplings raises, equal does not."""
        c = _dominant_system(12, 12, extra=0.0)
        Multigrid(c, apply_operator)
        a_p = c.a_p.copy()
        a_p[5, 5] *= 0.9
        bad = PressureCoefficients(a_p=a_p, a_e=c.a_e, a_w=c.a_w, a_n=c.a_n, a_s=c.a_s)
        with pytest.raises(ValueError, match="weakly diagonally dominant"):
            Multigrid(bad, apply_operator)

    @pytest.mark.parametrize("sweeps", [2, 3])
    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_more_than_one_sweep_keeps_the_cycle_symmetric_and_positive(
        self, name: str, sweeps: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """nu above 1 runs the extra pre-smoothing sweeps, as many as the post-smoothing.

        The built value is 1, so the extra loop never runs in the solver, but
        the constant is a knob (prompt 53's choice of nu used it). With the
        same count both ways the cycle is symmetric and positive on the
        range; a pre-smoothing loop dropped or miscounted breaks the
        symmetry. The cycle also differs from V(1, 1)'s.
        """
        pc, c, _b, _pred, _p = _system(name)
        active = c.a_p > 0.0
        once = _dense_cycle(Multigrid(c, apply_operator).cycle, active)
        monkeypatch.setattr("src.multigrid.SMOOTHING_SWEEPS", sweeps)
        matrix = _dense_cycle(Multigrid(c, apply_operator).cycle, active)
        assert np.abs(matrix - matrix.T).max() <= 1e-12 * np.abs(matrix).max()
        assert np.abs(matrix - once).max() > 1e-3 * np.abs(once).max()
        basis = _range_basis(active, pc.needs_pin)
        reduced = basis.T @ matrix @ basis
        eigenvalues = np.linalg.eigvalsh(0.5 * (reduced + reduced.T))
        assert eigenvalues[0] > 1e-6 * eigenvalues[-1] > 0.0

    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_the_cycle_is_symmetric(self, name: str) -> None:
        """<x, M y> = <M x, y> on random vectors, and the dense M equals its transpose.

        Defect caught: post-smoothing omitted, which leaves the pre-smoothed
        cycle unbalanced and not symmetric.
        """
        _pc, c, _b, _pred, _p = _system(name)
        active = c.a_p > 0.0
        hierarchy = Multigrid(c, apply_operator)
        rng = np.random.default_rng(11)
        for _ in range(3):
            x, y = (
                np.where(active, rng.standard_normal(c.a_p.shape), 0.0) for _ in "xy"
            )
            left = float(np.vdot(x, hierarchy.cycle(y)))
            right = float(np.vdot(hierarchy.cycle(x), y))
            assert abs(left - right) <= 1e-12 * abs(left)
        matrix = _dense_cycle(hierarchy.cycle, active)
        assert np.abs(matrix - matrix.T).max() <= 1e-12 * np.abs(matrix).max()

    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_the_preconditioner_is_positive_on_the_range(self, name: str) -> None:
        """The smallest eigenvalue of the preconditioner over the range is positive.

        The range is the cells with an equation, less the constant on the
        closed domain, where ``multigrid_preconditioner`` also projects. A
        ratio of smallest to largest above 1e-6 leaves no doubt it is not
        rounding.
        """
        pc, c, _b, _pred, _p = _system(name)
        active = c.a_p > 0.0
        precondition = multigrid_preconditioner(c, pc.needs_pin)
        basis = _range_basis(active, pc.needs_pin)
        matrix = basis.T @ _dense_cycle(precondition, active) @ basis
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        assert eigenvalues[0] > 1e-6 * eigenvalues[-1] > 0.0

    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_cells_without_an_equation_get_zero(self, name: str) -> None:
        """SOLID cells take no value from the prolongation; the input is not modified."""
        _pc, c, _b, _pred, _p = _system(name)
        active = c.a_p > 0.0
        r = np.where(active, np.random.default_rng(3).standard_normal(c.a_p.shape), 0.0)
        kept = r.copy()
        z = Multigrid(c, apply_operator).cycle(r)
        assert (z[~active] == 0.0).all() and (z[active] != 0.0).any()
        np.testing.assert_array_equal(r, kept)

    def test_the_closed_domain_projection_removes_the_mean(self) -> None:
        """Closed: the preconditioned vector has zero mean over the cells with an equation.

        Open: it is the cycle unchanged.
        """
        rng = np.random.default_rng(5)
        for name in ("closed", "open"):
            pc, c, _b, _pred, _p = _system(name)
            active = c.a_p > 0.0
            r = np.where(active, rng.standard_normal(c.a_p.shape), 0.0)
            z = multigrid_preconditioner(c, pc.needs_pin)(r)
            cycled = Multigrid(c, apply_operator).cycle(r)
            if pc.needs_pin:
                assert abs(z[active].mean()) <= 1e-15 * np.abs(z).max()
                assert abs(cycled[active].mean()) > 1e-6 * np.abs(cycled).max()
            else:
                np.testing.assert_array_equal(z, cycled)


@pytest.mark.unit
class TestMultigridSolve:
    """MG-PCG against Jacobi-PCG and a dense solve, directly and through the corrector."""

    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_reaches_the_level_jacobi_reaches_and_the_dense_solution(
        self, name: str
    ) -> None:
        """Both preconditioners meet the same relative residual; MG within its worth of dense.

        The error of any x with residual r is at most ``||r|| / lambda_min``
        on the range, so that is the bound. Multigrid takes fewer iterations
        than the diagonal, the reason it exists.
        """
        pc, c, b, _pred, _p = _system(name)
        f = _projected_rhs(pc, c, b)
        active = c.a_p > 0.0
        dense, smallest = _dense_solution(pc, c, f)
        results = {
            label: conjugate_gradient(
                lambda x: apply_operator(c, x), precondition, f, TIGHT_RTOL, 0.0, 5000
            )
            for label, precondition in (
                ("jacobi", jacobi_preconditioner(c)),
                ("multigrid", multigrid_preconditioner(c, pc.needs_pin)),
            )
        }
        for result in results.values():
            assert result.reached_cap is False
            assert result.residual_norm <= TIGHT_RTOL * np.linalg.norm(f)
        mg = results["multigrid"]
        error = _range_part(pc, active, mg.x) - dense
        assert np.linalg.norm(error) <= mg.residual_norm / smallest
        assert np.abs(dense).max() > 0.0
        assert 2 * mg.iterations < results["jacobi"].iterations

    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    def test_the_corrector_under_multigrid_meets_its_stop(self, name: str) -> None:
        """correct() with the key set: the true residual at pressure_rtol, p' at the dense one.

        The residual of the p' equation is the corrected faces' imbalance; it
        is read here from the returned p' as the corrector's tests read it.
        """
        pc, c, b, pred, p = _system(name, preconditioner="multigrid")
        out = pc.correct(pred, p)
        f = _projected_rhs(pc, c, b)
        active = c.a_p > 0.0
        r = f - apply_operator(c, out.p_prime)
        r[~active] = 0.0
        assert out.reached_cap is False
        assert np.linalg.norm(r) <= TIGHT_RTOL * np.linalg.norm(f)
        dense, smallest = _dense_solution(pc, c, f)
        error = _range_part(pc, active, out.p_prime) - dense
        assert np.linalg.norm(error) <= np.linalg.norm(r) / smallest
        assert (out.p_prime[~active] == 0.0).all()
        jacobi = _system(name)[0].correct(pred, p)
        assert out.iterations < jacobi.iterations

    @pytest.mark.parametrize("when", ["first", "outer 100"])
    def test_the_val002_cavity_system_is_solved_under_multigrid(
        self, when: str
    ) -> None:
        """The closed cavity's own case file, at its first correction and at outer 100.

        The 32x32 cavity is run under the default to the outer iteration
        named and the prediction handed to the corrector there is kept; a
        multigrid corrector then solves it to the case's own level, or to the
        floor where the right-hand side has shrunk below it, as the Jacobi
        corrector does on the same system.
        """
        raw = yaml.safe_load(case_path("cavity").read_text(encoding="utf-8"))
        raw["domain"]["nx"] = raw["domain"]["ny"] = 32
        raw["solver"]["max_simple_iter"] = 1 if when == "first" else 101
        config = SimConfig.from_dict(raw)
        mesh = Mesh(config)
        bc = StaggeredBoundary(mesh, config)
        seen: list[tuple[MomentumPrediction, np.ndarray]] = []
        original = PressureCorrector.correct

        def keeping(
            self: PressureCorrector, prediction: MomentumPrediction, p: np.ndarray
        ) -> PressureCorrection:
            seen.append((prediction, p.copy()))
            return original(self, prediction, p)

        PressureCorrector.correct = keeping  # type: ignore[method-assign]
        try:
            StaggeredSolver(mesh, config, bc).solve_steady()
        finally:
            PressureCorrector.correct = original  # type: ignore[method-assign]
        pred, p = seen[-1]
        raw["solver"]["pressure_preconditioner"] = "multigrid"
        mg_config = SimConfig.from_dict(raw)
        pc = PressureCorrector(mesh, mg_config, bc)
        assert pc.needs_pin
        out = pc.correct(pred, p)
        jacobi = PressureCorrector(mesh, config, bc).correct(pred, p)
        c = pc.coefficients(pred.a_p_u, pred.a_p_v)
        f = _projected_rhs(pc, c, pc.mass_imbalance(pred.u_star, pred.v_star))
        active = c.a_p > 0.0
        r = f - apply_operator(c, out.p_prime)
        r[~active] = 0.0
        stop = max(
            config.pressure_rtol * np.linalg.norm(f), RESIDUAL_FLOOR * pc.flux_scale
        )
        assert out.reached_cap is False and jacobi.reached_cap is False
        assert np.linalg.norm(r) <= stop
        assert out.iterations < jacobi.iterations
        dense, smallest = _dense_solution(pc, c, f)
        error = _range_part(pc, active, out.p_prime) - dense
        assert np.linalg.norm(error) <= np.linalg.norm(r) / smallest


def _base_conjugate_gradient(
    apply: Callable[[np.ndarray], np.ndarray],
    inverse_diagonal: np.ndarray,
    f: np.ndarray,
    rtol: float,
    floor: float,
    max_iter: int,
) -> tuple[np.ndarray, int, int]:
    """The loop of src/pressure.py before prompt 53, with its diagonal as an array.

    Copied from 8ee82d7 with the result tuple in place of the dataclass:
    x, the iterations and the products.
    """
    x = np.zeros_like(f)
    r = f.copy()
    f_norm = float(np.sqrt(np.vdot(f, f)))
    stop = max(rtol * f_norm, floor)
    if f_norm == 0.0:
        return x, 0, 0
    z = inverse_diagonal * r
    p = z.copy()
    rz = float(np.vdot(r, z))
    k = 0
    checks = 0
    while k < max_iter:
        q = apply(p)
        alpha = rz / float(np.vdot(p, q))
        x += alpha * p
        r -= alpha * q
        k += 1
        r_norm = float(np.sqrt(np.vdot(r, r)))
        if r_norm <= stop:
            r = f - apply(x)
            checks += 1
            r_norm = float(np.sqrt(np.vdot(r, r)))
            if r_norm <= stop:
                return x, k, k + checks
            z = inverse_diagonal * r
            p = z.copy()
            rz = float(np.vdot(r, z))
            continue
        z = inverse_diagonal * r
        rz_new = float(np.vdot(r, z))
        p *= rz_new / rz
        p += z
        rz = rz_new
    return x, k, k + checks + 1


@pytest.mark.unit
class TestDefaultPath:
    """With the key absent the correction is the base's Jacobi loop, to the bit."""

    @pytest.mark.parametrize("name", sorted(SYSTEMS))
    @pytest.mark.parametrize("rtol", [TIGHT_RTOL, 1e-4])
    def test_the_default_correction_is_the_base_loop_bitwise(
        self, name: str, rtol: float
    ) -> None:
        """p', the faces, p, the iterations and the products equal the base loop's exactly.

        The base loop is copied from the commit before the preconditioner
        became a callable, run on the same system with the same stop; the
        corrector is built with no pressure_preconditioner key. Defect
        caught: the default changed to multigrid, or the diagonal's product
        reordered.
        """
        pc, c, b, pred, p = _system(name)
        assert pc._multigrid is False
        pc._rtol = rtol
        out = pc.correct(pred, p)
        f = _projected_rhs(pc, c, b)
        active = c.a_p > 0.0
        inverse_diagonal = np.where(active, 1.0 / np.where(active, c.a_p, 1.0), 0.0)
        x, iterations, products = _base_conjugate_gradient(
            lambda v: apply_operator(c, v),
            inverse_diagonal,
            f,
            rtol,
            RESIDUAL_FLOOR * pc.flux_scale,
            5000,
        )
        if pc.needs_pin:
            x[active] -= x[pc.pin_cell]
        np.testing.assert_array_equal(out.p_prime, x)
        assert (out.iterations, out.products) == (iterations, products)
        explicit = _system(name, preconditioner="jacobi")[0]
        explicit._rtol = rtol
        same = explicit.correct(pred, p)
        for field in ("u", "v", "p", "p_prime"):
            np.testing.assert_array_equal(getattr(same, field), getattr(out, field))
