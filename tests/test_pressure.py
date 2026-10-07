"""Tests for the staggered pressure correction (REQ-S04, REQ-S08 as amended 2026-10-06).

The headline is the compatibility of the closed-domain system: the
right-hand side summed over the cavity must vanish to rounding, because the
wall faces hold zero exactly. The bound is machine epsilon times the total
absolute face flux of the field, the quantity the telescoping sum cancels,
not a tolerance chosen to pass. The collocated solver measured 2.90e-2,
9.23e-3 and 2.35e-3 on the same three grids (docs/reports/pressure_solver_probe.md,
table E).

The second is the solve (ECR-003 step 1, ADR-013): conjugate gradients
preconditioned with the diagonal, against a dense solve on open and closed
systems; its stop on the relative residual, the rounding floor, the check on
the true residual at exit, the iteration cap and its flag; the identity
between the residual of the p' equation and the corrected faces' imbalance;
the projection of the right-hand side on a closed domain; and the refusal of
a closed domain in two components. Where a bound is a "tolerance's worth" it
is the residual's 2-norm over the smallest eigenvalue the solve sees, read
from the dense matrix, not a number chosen to pass.
"""

from collections.abc import Callable

import numpy as np
import pytest
import yaml

import src.pressure as pressure
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import FLUID, SOLID, Mesh
from src.momentum import MomentumPrediction, MomentumPredictor
from src.pressure import (
    PRESSURE_SOLVER_VERSION,
    RESIDUAL_FLOOR,
    STAGGERED_METHOD,
    STAGGERED_METHODS,
    ConjugateGradientResult,
    PressureCoefficients,
    PressureCorrection,
    PressureCorrector,
    apply_operator,
    conjugate_gradient,
)
from src.solver_staggered import StaggeredSolver
from src.staggered import allocate_fields
from validation.cases import case_path, load_case

EPS = np.finfo(np.float64).eps
STRETCHED = {"x": {"stretch_ratio": 1.15}, "y": {"stretch_ratio": 1.25}}
# The tightest level the configuration accepts: the dense comparisons below
# are read against it.
TIGHT_RTOL = 1e-10


def _inlet(location: str, u_velocity: float, v_velocity: float) -> dict:
    span = {"x_start": 0.0, "x_end": 2.0} if location in ("top", "bottom") else {}
    span = span or {"y_start": 0.0, "y_end": 1.0}
    return {
        "type": "velocity_inlet",
        "location": location,
        "u_velocity": u_velocity,
        "v_velocity": v_velocity,
        **span,
    }


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
CAVITY = {"lid": _inlet("top", 1.0, 0.0)}
BLOCK = {
    "name": "block",
    "x_start": 0.7,
    "x_end": 1.3,
    "y_start": 0.3,
    "y_end": 0.7,
}
# A wall the full height of the domain, splitting it in two.
PARTITION = {
    "name": "partition",
    "x_start": 0.8,
    "x_end": 1.2,
    "y_start": 0.0,
    "y_end": 1.0,
}
# A wall the full length, splitting it into a lower and an upper part.
SHELF = {
    "name": "shelf",
    "x_start": 0.0,
    "x_end": 2.0,
    "y_start": 0.4,
    "y_end": 0.6,
}


def _config(
    boundaries: dict,
    nx: int = 8,
    ny: int = 6,
    obstacles: list[dict] | None = None,
    mesh: dict | None = None,
    max_pressure_iter: int = 5000,
    pressure_rtol: float = TIGHT_RTOL,
) -> SimConfig:
    raw = {
        "domain": {"width": 2.0, "height": 1.0, "nx": nx, "ny": ny},
        "fluid": {"density": 1.2, "viscosity": 0.05, "temperature": 293.0},
        "particles": {
            "density": 1000.0,
            "sizes": [0.1e-6],
            "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {"diameters": [0.1e-6], "efficiencies": [0.99999]},
        },
        "solver": {
            "dt": 0.01,
            "t_end": 1.0,
            "output_interval": 10,
            "convergence_tol": 1.0e-6,
            "max_simple_iter": 100,
            "alpha_velocity": 0.7,
            "alpha_pressure": 0.3,
            "max_pressure_iter": max_pressure_iter,
            "pressure_rtol": pressure_rtol,
        },
        "boundaries": boundaries,
        "obstacles": obstacles or [],
        "sensors": [{"name": "center", "x": 1.0, "y": 0.5}],
        "thresholds": {"0.1e-6": 100.0},
    }
    if mesh is not None:
        raw["mesh"] = mesh
    return SimConfig.from_dict(raw)


def _build(
    config: SimConfig,
) -> tuple[Mesh, StaggeredBoundary, MomentumPredictor, PressureCorrector]:
    mesh = Mesh(config)
    bc = StaggeredBoundary(mesh, config)
    return (
        mesh,
        bc,
        MomentumPredictor(mesh, config, bc),
        PressureCorrector(mesh, config, bc),
    )


def _predicted(
    config: SimConfig, sweeps: int = 5, outlet_right: bool = False
) -> tuple[Mesh, StaggeredBoundary, PressureCorrector, MomentumPrediction, np.ndarray]:
    """A non-trivial u*, v* from a few predictor sweeps starting from rest."""
    mesh, bc, mp, pc = _build(config)
    u, v, p = allocate_fields(mesh)
    bc.apply_normal_velocity(u, v)
    if outlet_right:
        u[:, -1] = u[:, -2]
    pred = mp.predict(u, v, p)
    for _ in range(sweeps - 1):
        u, v = pred.u_star, pred.v_star
        if outlet_right:
            u[:, -1] = u[:, -2]
        pred = mp.predict(u, v, p)
    return mesh, bc, pc, pred, p


def _case_with(name: str, n: int, **solver_keys: object) -> SimConfig:
    """A committed case on an n x n grid with solver keys replaced."""
    raw = yaml.safe_load(case_path(name).read_text(encoding="utf-8"))
    raw["domain"]["nx"] = raw["domain"]["ny"] = n
    raw["solver"].update(solver_keys)
    return SimConfig.from_dict(raw)


def _absolute_face_flux(mesh: Mesh, rho: float, u: np.ndarray, v: np.ndarray) -> float:
    """Sum of |rho u A| over every face: the scale the telescoping sum cancels."""
    return float(
        rho
        * (
            np.abs(u * mesh.dy_cell[:, None]).sum()
            + np.abs(v * mesh.dx_cell[None, :]).sum()
        )
    )


def _dense(c: PressureCoefficients) -> tuple[np.ndarray, np.ndarray]:
    """A as a dense matrix over the cells with an equation, and their flat indices."""
    active = c.a_p > 0.0
    ny, nx = active.shape
    cells = np.flatnonzero(active.ravel())
    index = -np.ones(ny * nx, dtype=int)
    index[cells] = np.arange(cells.size)
    matrix = np.zeros((cells.size, cells.size))
    for k, flat in enumerate(cells):
        j, i = divmod(int(flat), nx)
        matrix[k, k] = c.a_p[j, i]
        for coef, dj, di in (
            (c.a_e, 0, 1),
            (c.a_w, 0, -1),
            (c.a_n, 1, 0),
            (c.a_s, -1, 0),
        ):
            if coef[j, i] != 0.0:
                matrix[k, index[(j + dj) * nx + (i + di)]] = -coef[j, i]
    return matrix, cells


def _projected_rhs(
    pc: PressureCorrector, c: PressureCoefficients, b: np.ndarray
) -> np.ndarray:
    """f = -b with the mean over the cells with an equation removed on a closed domain."""
    active = c.a_p > 0.0
    f = -b.copy()
    if pc.needs_pin:
        f[active] -= f[active].mean()
    f[~active] = 0.0
    return f


def _inverse_diagonal(c: PressureCoefficients) -> np.ndarray:
    active = c.a_p > 0.0
    return np.where(active, 1.0 / np.where(active, c.a_p, 1.0), 0.0)


def _dense_solution(
    pc: PressureCorrector, c: PressureCoefficients, b: np.ndarray
) -> tuple[np.ndarray, float]:
    """p' by a dense solve, pinned as the corrector pins, and the smallest eigenvalue CG sees.

    On an open domain the matrix is positive definite and the eigenvalue is
    its smallest. On a closed domain the constants are the null space: the
    pin cell's row is replaced by x_pin = 0, the right-hand side is the
    projected one, and the eigenvalue is the smallest nonzero one.
    """
    matrix, cells = _dense(c)
    f = _projected_rhs(pc, c, b).ravel()[cells]
    eigenvalues = np.linalg.eigvalsh(matrix)
    if pc.needs_pin:
        pinned = matrix.copy()
        k = int(np.flatnonzero(cells == np.ravel_multi_index(pc.pin_cell, b.shape))[0])
        pinned[k, :] = 0.0
        pinned[k, k] = 1.0
        f = f.copy()
        f[k] = 0.0
        x = np.linalg.solve(pinned, f)
        smallest = float(eigenvalues[1])
    else:
        x = np.linalg.solve(matrix, f)
        smallest = float(eigenvalues[0])
    dense = np.zeros(b.size)
    dense[cells] = x
    return dense.reshape(b.shape), smallest


def _residual(
    c: PressureCoefficients, b: np.ndarray, p_prime: np.ndarray
) -> np.ndarray:
    """b + A p' over the cells with an equation: the imbalance the correction leaves."""
    r = b + apply_operator(c, p_prime)
    r[c.a_p <= 0.0] = 0.0
    return r


# ---------------------------------------------------------------------------
# The right-hand side
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMassImbalance:
    """The discrete divergence of u*, formed directly from face velocities."""

    @pytest.mark.parametrize("n", [20, 40, 80])
    def test_closed_domain_right_hand_side_sums_to_zero(self, n: int) -> None:
        """ECR-001 acceptance criterion 6, measured directly on the cavity.

        The bound is 8 eps times the total absolute face flux, the
        magnitude of what the telescoping sum cancels. The collocated
        solver measured 2.90e-2, 9.23e-3 and 2.35e-3 here.
        """
        config = load_case("cavity", grid=(n, n))
        mesh, bc, mp, pc = _build(config)
        u, v, p = allocate_fields(mesh)
        bc.apply_normal_velocity(u, v)
        for _ in range(5):
            pred = mp.predict(u, v, p)
            u, v = pred.u_star, pred.v_star
        b = pc.mass_imbalance(u, v)
        assert np.abs(b).max() > 1e-4, "the field must be genuinely non-solenoidal"
        bound = 8.0 * EPS * _absolute_face_flux(mesh, config.rho, u, v)
        assert abs(float(b.sum())) <= bound

    def test_uniform_field_has_zero_imbalance_exactly(self) -> None:
        """A constant field has identical face fluxes, so every cell closes exactly."""
        mesh, bc, _mp, pc = _build(
            _config(
                {e: _inlet(e, 0.4, -0.2) for e in ("bottom", "top", "left", "right")}
            )
        )
        u, v, _p = allocate_fields(mesh)
        u.fill(0.4)
        v.fill(-0.2)
        bc.apply_normal_velocity(u, v)
        assert np.all(pc.mass_imbalance(u, v) == 0.0)

    def test_sum_is_the_net_boundary_flux_of_an_open_domain(self) -> None:
        """On the channel the sum telescopes to outflow minus inflow."""
        config = _config(CHANNEL)
        mesh, _bc, pc, pred, _p = _predicted(config, outlet_right=True)
        u, v = pred.u_star, pred.v_star
        expected = config.rho * float(
            ((u[:, -1] - u[:, 0]) * mesh.dy_cell).sum()
            + ((v[-1, :] - v[0, :]) * mesh.dx_cell).sum()
        )
        b = pc.mass_imbalance(u, v)
        assert float(b.sum()) == pytest.approx(
            expected, abs=8.0 * EPS * _absolute_face_flux(mesh, config.rho, u, v)
        )

    def test_solid_cells_carry_no_imbalance(self) -> None:
        """SOLID cells are outside the solve and report zero imbalance."""
        mesh, _bc, pc, pred, _p = _predicted(_config(CAVITY, obstacles=[BLOCK]))
        b = pc.mass_imbalance(pred.u_star, pred.v_star)
        assert np.all(b[mesh.cell_type == SOLID] == 0.0)


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCoefficients:
    """a_nb = rho d_face A_face with the mesh's own face lengths."""

    def test_stretched_mesh_uses_the_real_face_lengths(self) -> None:
        """Each coefficient is rho A_face^2 / a_P with the cell's own widths."""
        mesh, _bc, pc, pred, _p = _predicted(_config(CAVITY, mesh=STRETCHED))
        c = pc.coefficients(pred.a_p_u, pred.a_p_v)
        j, i = 2, 3
        rho = 1.2
        assert c.a_e[j, i] == pytest.approx(
            rho * mesh.dy_cell[j] ** 2 / pred.a_p_u[j, i + 1], rel=1e-14
        )
        assert c.a_w[j, i] == pytest.approx(
            rho * mesh.dy_cell[j] ** 2 / pred.a_p_u[j, i], rel=1e-14
        )
        assert c.a_n[j, i] == pytest.approx(
            rho * mesh.dx_cell[i] ** 2 / pred.a_p_v[j + 1, i], rel=1e-14
        )
        assert c.a_s[j, i] == pytest.approx(
            rho * mesh.dx_cell[i] ** 2 / pred.a_p_v[j, i], rel=1e-14
        )
        assert mesh.dy_cell[j] != mesh.dy
        assert c.a_p[j, i] == c.a_e[j, i] + c.a_w[j, i] + c.a_n[j, i] + c.a_s[j, i]

    def test_walls_inlets_and_solid_faces_contribute_nothing(self) -> None:
        """No coefficient across a fixed face: the Neumann condition by absence."""
        config = _config(CHANNEL, obstacles=[BLOCK])
        mesh, _bc, pc, pred, _p = _predicted(config, outlet_right=True)
        c = pc.coefficients(pred.a_p_u, pred.a_p_v)
        assert np.all(c.a_w[:, 0] == 0.0)
        assert np.all(c.a_s[0, :] == 0.0)
        assert np.all(c.a_n[-1, :] == 0.0)
        solid = mesh.cell_type == SOLID
        assert np.all(c.a_p[solid] == 0.0)
        east_of_solid = np.zeros_like(solid)
        east_of_solid[:, 1:] = solid[:, :-1]
        assert np.all(c.a_w[east_of_solid] == 0.0)
        assert np.all(c.a_p[~solid] > 0.0)

    def test_closed_domain_diagonal_is_the_neighbour_sum(self) -> None:
        """With no outlet every row is pure Neumann and the domain is pinned."""
        _mesh, _bc, pc, pred, _p = _predicted(_config(CAVITY))
        c = pc.coefficients(pred.a_p_u, pred.a_p_v)
        assert np.array_equal(c.a_p, c.a_e + c.a_w + c.a_n + c.a_s)
        assert pc.needs_pin
        assert pc.pin_cell == (1, 1)

    def test_outlet_cells_carry_the_outlet_term_in_the_diagonal(self) -> None:
        """p' = 0 at the outlet face: a coefficient in a_P with no neighbour."""
        config = _config(CHANNEL)
        mesh, _bc, pc, pred, _p = _predicted(config, outlet_right=True)
        c = pc.coefficients(pred.a_p_u, pred.a_p_v)
        neighbours = c.a_e + c.a_w + c.a_n + c.a_s
        outlet_term = 1.2 * mesh.dy_cell**2 / pred.a_p_u[:, -2]
        assert c.a_p[:, -1] == pytest.approx(neighbours[:, -1] + outlet_term, rel=1e-14)
        assert np.all(c.a_e[:, -1] == 0.0)
        assert np.array_equal(c.a_p[:, :-1], neighbours[:, :-1])
        assert not pc.needs_pin

    def test_apply_operator_is_the_dense_matrix(self) -> None:
        """A x from the shifted products equals the dense matrix's product, on both domains."""
        for boundaries, outlet in ((CAVITY, False), (CHANNEL, True)):
            _mesh, _bc, pc, pred, _p = _predicted(
                _config(boundaries, obstacles=[BLOCK]), outlet_right=outlet
            )
            c = pc.coefficients(pred.a_p_u, pred.a_p_v)
            matrix, cells = _dense(c)
            rng = np.random.default_rng(37)
            x = rng.standard_normal(c.a_p.shape)
            x[c.a_p <= 0.0] = 0.0
            product = apply_operator(c, x)
            assert np.all(product[c.a_p <= 0.0] == 0.0)
            expected = matrix @ x.ravel()[cells]
            assert product.ravel()[cells] == pytest.approx(
                expected, rel=1e-13, abs=1e-15
            )
            assert np.array_equal(matrix, matrix.T)


# ---------------------------------------------------------------------------
# The correction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCorrection:
    """p', the corrected velocities and the updated pressure."""

    def test_uniform_field_gives_zero_correction_exactly(self) -> None:
        """A field with zero imbalance is returned bit for bit, with no iteration."""
        config = _config(
            {e: _inlet(e, 0.4, -0.2) for e in ("bottom", "top", "left", "right")}
        )
        mesh, bc, mp, pc = _build(config)
        u, v, p = allocate_fields(mesh)
        u.fill(0.4)
        v.fill(-0.2)
        bc.apply_normal_velocity(u, v)
        p[:] = 3.0
        pred = mp.predict(u, v, p)
        # The predictor reproduces the uniform field only to rounding; the
        # property under test is the corrector's, so hand it the exact field.
        pred = MomentumPrediction(
            u_star=u, v_star=v, a_p_u=pred.a_p_u, a_p_v=pred.a_p_v
        )
        out = pc.correct(pred, p)
        assert np.all(out.p_prime == 0.0)
        assert np.array_equal(out.u, pred.u_star)
        assert np.array_equal(out.v, pred.v_star)
        assert np.array_equal(out.p, p - p[0, 0])
        assert (out.iterations, out.reached_cap) == (0, False)

    def test_divergence_free_field_is_left_alone_to_rounding(self) -> None:
        """A stream-function field is solenoidal to the rounding of its differences."""
        config = _config(CAVITY)
        mesh, _bc, mp, pc = _build(config)

        def psi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.sin(np.pi * x / 2.0) * np.sin(np.pi * y)

        x_f, y_f = np.meshgrid(mesh.x, mesh.y)
        u = (
            psi(x_f[1:, :], y_f[1:, :]) - psi(x_f[:-1, :], y_f[:-1, :])
        ) / mesh.dy_cell[:, None]
        v = (
            -(psi(x_f[:, 1:], y_f[:, 1:]) - psi(x_f[:, :-1], y_f[:, :-1]))
            / mesh.dx_cell[None, :]
        )
        pred = mp.predict(u, v, np.zeros(mesh.cell_type.shape))
        pred = MomentumPrediction(
            u_star=u, v_star=v, a_p_u=pred.a_p_u, a_p_v=pred.a_p_v
        )
        out = pc.correct(pred, np.zeros(mesh.cell_type.shape))
        assert np.abs(out.u - u).max() < 1e-13
        assert np.abs(out.v - v).max() < 1e-13

    def test_boundary_faces_are_never_touched(self) -> None:
        """Walls and inlets hold exactly what step 3 wrote, before and after."""
        for boundaries in (CAVITY, CHANNEL):
            config = _config(boundaries)
            mesh, bc, pc, pred, p = _predicted(
                config, outlet_right=boundaries is CHANNEL
            )
            expected_u, expected_v, _ = allocate_fields(mesh)
            bc.apply_normal_velocity(expected_u, expected_v)
            out = pc.correct(pred, p)
            assert np.array_equal(out.u[:, 0], expected_u[:, 0])
            assert np.array_equal(out.v[0, :], expected_v[0, :])
            assert np.array_equal(out.v[-1, :], expected_v[-1, :])
            if boundaries is CAVITY:
                assert np.array_equal(out.u[:, -1], expected_u[:, -1])

    @pytest.mark.parametrize("domain", ["open", "closed"])
    def test_corrected_field_satisfies_continuity(self, domain: str) -> None:
        """Every cell closes to the configured level, on the channel and the cavity.

        The cavity is the system on which plain Jacobi stalled at any cap
        (docs/reports/pressure_correction_step5.md); CG needs no weight.
        """
        config = _config(CHANNEL if domain == "open" else CAVITY)
        _mesh, _bc, pc, pred, p = _predicted(config, outlet_right=domain == "open")
        before = pc.mass_imbalance(pred.u_star, pred.v_star)
        assert np.abs(before).max() > 1e-4
        out = pc.correct(pred, p)
        after = pc.mass_imbalance(out.u, out.v)
        assert np.linalg.norm(after) <= TIGHT_RTOL * np.linalg.norm(before)
        assert out.iterations < config.max_pressure_iter
        assert out.reached_cap is False

    def test_two_cell_domain_with_an_outlet_is_solved_exactly(self) -> None:
        """A wall, one interior face carrying q, an extrapolated outlet.

        With p' = 0 at the outlet and the outlet face borrowing the interior
        diagonal, the analytic solution is p' = (-2q/d, -q/d) and both the
        interior and the outlet face return to zero, so nothing flows.
        """
        outlet = {
            "out": {
                "type": "pressure_outlet",
                "location": "right",
                "y_start": 0.0,
                "y_end": 1.0,
            }
        }
        config = _config(outlet, nx=2, ny=1)
        mesh = Mesh(config)
        bc = StaggeredBoundary(mesh, config)
        pc = PressureCorrector(mesh, config, bc)
        q = 0.25
        u = np.array([[0.0, q, q]])
        v = np.zeros((2, 2))
        a_p_u = np.array([[0.0, 4.0, 0.0]])
        pred = MomentumPrediction(
            u_star=u, v_star=v, a_p_u=a_p_u, a_p_v=np.zeros((2, 2))
        )
        out = pc.correct(pred, np.zeros((1, 2)))
        d = mesh.dy_cell[0] / 4.0
        assert not pc.needs_pin
        assert out.p_prime[0, 0] == pytest.approx(-2.0 * q / d, rel=1e-12)
        assert out.p_prime[0, 1] == pytest.approx(-q / d, rel=1e-12)
        assert out.u[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert out.u[0, 2] == pytest.approx(0.0, abs=1e-12)
        assert out.u[0, 0] == 0.0
        assert out.p[0, 1] == pytest.approx(-0.3 * q / d, rel=1e-12)
        # Two unknowns: CG is exact in at most two iterations.
        assert out.iterations <= 2

    def test_face_changes_follow_the_gradient_of_a_dense_solve(self) -> None:
        """A known imbalance at the centre of a 3x3 open domain: p' agrees with an
        independent dense solve and each of the centre cell's four faces moves by
        -d times the p' difference across it."""
        config = _config(CHANNEL, nx=3, ny=3)
        mesh = Mesh(config)
        bc = StaggeredBoundary(mesh, config)
        pc = PressureCorrector(mesh, config, bc)
        q = 0.1
        u, v, _p = allocate_fields(mesh)
        u[1, 1], u[1, 2] = -q, q
        v[1, 1], v[2, 1] = -q, q
        a_p_u = np.zeros(u.shape)
        a_p_u[:, 1:-1] = 2.0
        a_p_v = np.zeros(v.shape)
        a_p_v[1:-1, :] = 2.0
        pred = MomentumPrediction(u_star=u, v_star=v, a_p_u=a_p_u, a_p_v=a_p_v)
        out = pc.correct(pred, np.zeros((3, 3)))

        c = pc.coefficients(a_p_u, a_p_v)
        b = pc.mass_imbalance(u, v)
        assert b[1, 1] == pytest.approx(1.2 * 2 * q * (mesh.dy + mesh.dx), rel=1e-14)
        n = 9
        matrix = np.zeros((n, n))
        for j in range(3):
            for i in range(3):
                k = 3 * j + i
                matrix[k, k] = c.a_p[j, i]
                if i < 2:
                    matrix[k, k + 1] = -c.a_e[j, i]
                if i > 0:
                    matrix[k, k - 1] = -c.a_w[j, i]
                if j < 2:
                    matrix[k, k + 3] = -c.a_n[j, i]
                if j > 0:
                    matrix[k, k - 3] = -c.a_s[j, i]
        dense = np.linalg.solve(matrix, -b.ravel()).reshape(3, 3)
        assert out.p_prime == pytest.approx(dense, abs=1e-9)
        d_u = mesh.dy / 2.0
        d_v = mesh.dx / 2.0
        assert out.u[1, 1] == u[1, 1] - d_u * (out.p_prime[1, 1] - out.p_prime[1, 0])
        assert out.u[1, 2] == u[1, 2] - d_u * (out.p_prime[1, 2] - out.p_prime[1, 1])
        assert out.v[1, 1] == v[1, 1] - d_v * (out.p_prime[1, 1] - out.p_prime[0, 1])
        assert out.v[2, 1] == v[2, 1] - d_v * (out.p_prime[2, 1] - out.p_prime[1, 1])

    def test_open_domain_is_not_pinned_and_the_outlet_face_is_corrected(self) -> None:
        """An outlet fixes the level, so no pin, and its face moves against p' = 0."""
        config = _config(CHANNEL)
        mesh, _bc, pc, pred, p = _predicted(config, outlet_right=True)
        out = pc.correct(pred, p)
        assert not pc.needs_pin
        assert out.p_prime[0, 0] != 0.0
        assert not np.array_equal(out.u[:, -1], pred.u_star[:, -1])
        d_out = mesh.dy_cell / pred.a_p_u[:, -2]
        assert out.u[:, -1] == pytest.approx(
            pred.u_star[:, -1] + d_out * out.p_prime[:, -1], rel=1e-14
        )

    def test_closed_domain_pins_the_reference_cell(self) -> None:
        """Both p' and the updated p are zero at the reference cell."""
        config = _config(CAVITY)
        _mesh, _bc, pc, pred, p = _predicted(config)
        p[:] = 5.0
        out = pc.correct(pred, p)
        assert pc.needs_pin
        assert out.p_prime[pc.pin_cell] == 0.0
        assert out.p[pc.pin_cell] == 0.0
        assert np.abs(out.p_prime).max() > 0.0

    def test_pinned_cell_is_typed_fluid(self) -> None:
        """The reference cell is the first cell typed FLUID, as in the collocated solver."""
        mesh, _bc, pc, _pred, _p = _predicted(_config(CAVITY))
        assert pc.needs_pin
        assert mesh.cell_type[pc.pin_cell] == FLUID
        assert pc.pin_cell == tuple(np.argwhere(mesh.cell_type == FLUID)[0])

    def test_pin_removes_a_nonzero_constant_mode(self) -> None:
        """The pin acts after the solve: p' is zero at the reference cell and the
        whole field is the raw solution shifted by its value there.

        CG from zero on the projected system returns the solution orthogonal
        to the constants, whose value at the reference cell is not zero. The
        raw solve is taken by switching the pin off on the same corrector,
        which also switches the projection off; on this compatible system
        the projection removes a mean at rounding level, so the pinned field
        equals raw minus its reference value to 1e-12, not bit for bit.
        """
        _mesh, _bc, pc, pred, p = _predicted(_config(CAVITY))
        pc.needs_pin = False
        raw = pc.correct(pred, p).p_prime
        pc.needs_pin = True
        out = pc.correct(pred, p)
        assert raw[pc.pin_cell] != 0.0
        assert out.p_prime[pc.pin_cell] == 0.0
        assert out.p_prime == pytest.approx(
            raw - raw[pc.pin_cell], rel=1e-12, abs=1e-12
        )

    def test_iterations_are_capped_by_max_pressure_iter_and_the_cap_reported(
        self,
    ) -> None:
        """The CG loop stops at the configured cap and says so; below it, it does not."""
        config = _config(CAVITY, max_pressure_iter=7)
        _mesh, _bc, pc, pred, p = _predicted(config)
        out = pc.correct(pred, p)
        assert (out.iterations, out.reached_cap) == (7, True)
        after = pc.mass_imbalance(out.u, out.v)
        before = pc.mass_imbalance(pred.u_star, pred.v_star)
        assert np.linalg.norm(after) > TIGHT_RTOL * np.linalg.norm(before)
        _mesh, _bc, pc, pred, p = _predicted(_config(CAVITY))
        out = pc.correct(pred, p)
        assert out.reached_cap is False
        assert 7 < out.iterations < 5000

    def test_wrong_shapes_are_rejected(self) -> None:
        """Mismatched p, velocity and diagonal shapes raise before any arithmetic."""
        config = _config(CAVITY)
        _mesh, _bc, pc, pred, p = _predicted(config)
        with pytest.raises(ValueError, match="expected p"):
            pc.correct(pred, p.T.copy())
        with pytest.raises(ValueError, match="staggered shapes"):
            pc.mass_imbalance(pred.v_star, pred.u_star)
        with pytest.raises(ValueError, match="diagonals"):
            pc.coefficients(pred.a_p_v, pred.a_p_u)

    def test_result_fields_are_the_contract(self) -> None:
        """iterations and reached_cap, not sweeps: the harness and the solver read them."""
        names = [f.name for f in PressureCorrection.__dataclass_fields__.values()]
        assert names == ["u", "v", "p", "p_prime", "iterations", "reached_cap"]
        assert PRESSURE_SOLVER_VERSION == 2

    def test_each_solver_version_has_its_own_label(self) -> None:
        """The label the scripts file results under is the current version's, and unique.

        Review 37 S4. Defect caught: two versions sharing a label, so a saved
        field of one could be read as the other's, or the current label not
        the current version's.
        """
        assert STAGGERED_METHODS[PRESSURE_SOLVER_VERSION] == STAGGERED_METHOD
        assert STAGGERED_METHODS == {1: "staggered-jacobi", 2: "staggered-cg"}
        assert len(set(STAGGERED_METHODS.values())) == len(STAGGERED_METHODS)


# ---------------------------------------------------------------------------
# The conjugate gradient solve (REQ-S08 as amended, ADR-013 A and B)
# ---------------------------------------------------------------------------


def _reference_recursion_only_cg(
    apply: Callable[[np.ndarray], np.ndarray],
    inverse_diagonal: np.ndarray,
    f: np.ndarray,
    stop: float,
    max_iter: int,
) -> np.ndarray:
    """CG that trusts its recursive residual: the loop without the exit check.

    The planted defect of the true-residual test: what a solve returns when
    it believes the recursion.
    """
    x = np.zeros_like(f)
    r = f.copy()
    z = inverse_diagonal * r
    p = z.copy()
    rz = float(np.vdot(r, z))
    for _ in range(max_iter):
        q = apply(p)
        alpha = rz / float(np.vdot(p, q))
        x += alpha * p
        r -= alpha * q
        if float(np.sqrt(np.vdot(r, r))) <= stop:
            break
        z = inverse_diagonal * r
        rz_new = float(np.vdot(r, z))
        p *= rz_new / rz
        p += z
        rz = rz_new
    return x


@pytest.mark.unit
class TestConjugateGradient:
    """The solve against a dense one, its stop, the floor, the exit check and the cap."""

    def _system(
        self, boundaries: dict, obstacles: list[dict] | None = None, **keys: object
    ) -> tuple[
        PressureCorrector,
        PressureCoefficients,
        np.ndarray,
        MomentumPrediction,
        np.ndarray,
    ]:
        outlet = boundaries is CHANNEL
        _mesh, _bc, pc, pred, p = _predicted(
            _config(boundaries, obstacles=obstacles, **keys), outlet_right=outlet
        )
        c = pc.coefficients(pred.a_p_u, pred.a_p_v)
        return pc, c, pc.mass_imbalance(pred.u_star, pred.v_star), pred, p

    @pytest.mark.parametrize("domain", ["open", "closed"])
    def test_agrees_with_a_dense_solve_to_the_tolerances_worth(
        self, domain: str
    ) -> None:
        """p' is within ||r|| / lambda_min of the dense solution, open and closed, with an obstacle.

        The error of any solution with residual r is bounded by ||r||_2 over
        the smallest eigenvalue CG sees (the smallest nonzero one on the
        singular cavity), so that, not a chosen number, is the bound. The
        true relative residual at exit meets the configured level.
        """
        boundaries = CHANNEL if domain == "open" else CAVITY
        pc, c, b, pred, p = self._system(boundaries, obstacles=[BLOCK])
        out = pc.correct(pred, p)
        dense, smallest = _dense_solution(pc, c, b)
        f = _projected_rhs(pc, c, b)
        r = _residual(c, b, out.p_prime)
        if pc.needs_pin:
            r[c.a_p > 0.0] -= b[c.a_p > 0.0].mean()
        assert np.linalg.norm(r) <= TIGHT_RTOL * np.linalg.norm(f)
        assert np.linalg.norm(out.p_prime - dense) <= np.linalg.norm(r) / smallest
        assert np.abs(out.p_prime - dense).max() < 1e-7 * np.abs(dense).max()
        assert np.abs(dense).max() > 0.0

    @pytest.mark.parametrize("outer", [0, 300])
    def test_val002_cavity_system_agrees_with_a_dense_solve(self, outer: int) -> None:
        """The VAL-002 cavity's own system, from rest and at a late outer iteration.

        The 20x20 case file's solve is run to the given outer iteration under
        the committed level and cap, the prediction handed to the corrector
        there is kept, and the correction on it is compared with a dense
        solve of the same pinned system to the tolerance's worth. At the late
        iteration the right-hand side is small and the stop may be the floor.
        """
        config = _case_with("cavity", 20, max_simple_iter=outer + 1)
        mesh = Mesh(config)
        bc = StaggeredBoundary(mesh, config)
        solver = StaggeredSolver(mesh, config, bc)
        seen: list[tuple[MomentumPrediction, np.ndarray]] = []
        original = PressureCorrector.correct

        def keeping(
            self: PressureCorrector, prediction: MomentumPrediction, p: np.ndarray
        ) -> PressureCorrection:
            seen.append((prediction, p.copy()))
            return original(self, prediction, p)

        PressureCorrector.correct = keeping  # type: ignore[method-assign]
        try:
            solver.solve_steady()
        finally:
            PressureCorrector.correct = original  # type: ignore[method-assign]
        pred, p = seen[outer]
        pc = PressureCorrector(mesh, config, bc)
        c = pc.coefficients(pred.a_p_u, pred.a_p_v)
        b = pc.mass_imbalance(pred.u_star, pred.v_star)
        out = pc.correct(pred, p)
        dense, smallest = _dense_solution(pc, c, b)
        f = _projected_rhs(pc, c, b)
        r = _residual(c, b, out.p_prime)
        r[c.a_p > 0.0] -= b[c.a_p > 0.0].mean()
        stop = max(
            config.pressure_rtol * np.linalg.norm(f), RESIDUAL_FLOOR * pc.flux_scale
        )
        assert out.reached_cap is False
        assert np.linalg.norm(r) <= stop
        assert np.linalg.norm(out.p_prime - dense) <= np.linalg.norm(r) / smallest
        assert np.abs(dense).max() > 0.0

    def test_stops_at_the_first_iteration_meeting_the_relative_level(self) -> None:
        """One iteration fewer leaves the residual above rtol ||f||; tighter levels take more."""
        pc, c, b, _pred, _p = self._system(CHANNEL)
        f = _projected_rhs(pc, c, b)
        inv = _inverse_diagonal(c)

        def apply(x: np.ndarray) -> np.ndarray:
            return apply_operator(c, x)

        counts = []
        for rtol in (1e-4, 1e-8, 1e-12):
            res = conjugate_gradient(apply, inv, f, rtol, 0.0, 5000)
            stop = rtol * np.linalg.norm(f)
            assert res.reached_cap is False
            assert res.residual_norm <= stop
            assert np.linalg.norm(f - apply(res.x)) == res.residual_norm
            short = conjugate_gradient(apply, inv, f, rtol, 0.0, res.iterations - 1)
            assert short.reached_cap is True
            assert short.iterations == res.iterations - 1
            assert np.linalg.norm(f - apply(short.x)) > stop
            counts.append(res.iterations)
        assert counts[0] < counts[1] < counts[2]

    def test_the_floor_ends_a_correction_at_rounding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Below RESIDUAL_FLOOR times F the residual is rounding, and the solve stops there.

        With the floor raised to 1e-2 F the correction stops as soon as the
        residual's 2-norm is below it, above the relative level and in fewer
        iterations than the real floor allows; the residual reported is the
        true one. A floor of zero removes the second stop.
        """
        pc, c, b, pred, p = self._system(CAVITY)
        f = _projected_rhs(pc, c, b)
        inv = _inverse_diagonal(c)
        assert RESIDUAL_FLOOR == 1e-13
        assert pc.flux_scale == pytest.approx(1.2 * 1.0 * 2.0)
        full = pc.correct(pred, p)
        monkeypatch.setattr(pressure, "RESIDUAL_FLOOR", 1e-2)
        floored = pc.correct(pred, p)
        r = _residual(c, b, floored.p_prime)
        r[c.a_p > 0.0] -= b[c.a_p > 0.0].mean()
        assert floored.iterations < full.iterations
        assert floored.reached_cap is False
        assert np.linalg.norm(r) <= 1e-2 * pc.flux_scale
        assert np.linalg.norm(r) > TIGHT_RTOL * np.linalg.norm(f)
        direct = conjugate_gradient(
            lambda x: apply_operator(c, x),
            inv,
            f,
            TIGHT_RTOL,
            1e-2 * pc.flux_scale,
            5000,
        )
        assert direct.iterations == floored.iterations
        assert direct.residual_norm == pytest.approx(np.linalg.norm(r), rel=1e-12)
        bare = conjugate_gradient(
            lambda x: apply_operator(c, x), inv, f, TIGHT_RTOL, 0.0, 5000
        )
        assert bare.iterations == full.iterations

    def test_flux_scale_is_the_stopping_rules(self) -> None:
        """F is rho times the inflow, or closed, rho times the lid speed times the longer side."""
        _mesh, _bc, _mp, channel = _build(_config(CHANNEL))
        assert channel.flux_scale == pytest.approx(1.2 * 0.3 * 1.0, rel=1e-14)
        _mesh, _bc, _mp, cavity = _build(_config(CAVITY))
        assert cavity.flux_scale == pytest.approx(1.2 * 1.0 * 2.0, rel=1e-14)
        _mesh, _bc, _mp, still = _build(_config({}))
        assert still.flux_scale == 0.0

    def test_exit_check_reads_the_true_residual_not_the_recursion(self) -> None:
        """A recursion that lies is caught at exit and the solve goes on.

        The operator is wrapped to return a product perturbed in one cell on
        its third call, so the recursive residual and the true one part by a
        known amount. The reference loop that trusts its recursion returns a
        solution whose true residual misses the stop; the built solve ends
        with the true residual under it, and reports that residual.
        """
        pc, c, b, _pred, _p = self._system(CHANNEL)
        f = _projected_rhs(pc, c, b)
        inv = _inverse_diagonal(c)
        rtol = 1e-8
        stop = rtol * float(np.linalg.norm(f))
        calls = {"n": 0}

        def lying(x: np.ndarray) -> np.ndarray:
            calls["n"] += 1
            q = apply_operator(c, x)
            if calls["n"] == 3:
                q = q.copy()
                q[2, 3] += 1e-3 * float(np.linalg.norm(f))
            return q

        trusting = _reference_recursion_only_cg(lying, inv, f, stop, 5000)
        assert np.linalg.norm(f - apply_operator(c, trusting)) > 100.0 * stop
        calls["n"] = 0
        res = conjugate_gradient(lying, inv, f, rtol, 0.0, 5000)
        assert res.reached_cap is False
        assert res.residual_norm <= stop
        assert np.linalg.norm(f - apply_operator(c, res.x)) == pytest.approx(
            res.residual_norm, rel=1e-12
        )
        # The honest operator agrees with itself at exit, within rounding.
        honest = conjugate_gradient(
            lambda x: apply_operator(c, x), inv, f, rtol, 0.0, 5000
        )
        assert honest.residual_norm <= stop

    def test_cap_is_reported_with_the_true_residual(self) -> None:
        """Three iterations on a system that needs more: capped, and the residual is f - A x."""
        pc, c, b, _pred, _p = self._system(CAVITY)
        f = _projected_rhs(pc, c, b)
        res = conjugate_gradient(
            lambda x: apply_operator(c, x), _inverse_diagonal(c), f, TIGHT_RTOL, 0.0, 3
        )
        assert isinstance(res, ConjugateGradientResult)
        assert (res.iterations, res.reached_cap) == (3, True)
        assert res.residual_norm == pytest.approx(
            np.linalg.norm(f - apply_operator(c, res.x)), rel=1e-12
        )
        assert res.residual_norm > TIGHT_RTOL * np.linalg.norm(f)

    def test_a_zero_right_hand_side_returns_at_once(self) -> None:
        """No iteration, no cap, zero solution and zero residual."""
        _pc, c, _b, _pred, _p = self._system(CHANNEL)
        f = np.zeros(c.a_p.shape)
        res = conjugate_gradient(
            lambda x: apply_operator(c, x),
            _inverse_diagonal(c),
            f,
            TIGHT_RTOL,
            0.0,
            5000,
        )
        assert (res.iterations, res.reached_cap, res.residual_norm) == (0, False, 0.0)
        assert np.all(res.x == 0.0)

    @pytest.mark.parametrize(
        ("boundaries", "obstacles"),
        [(CHANNEL, None), (CAVITY, None), (CAVITY, [BLOCK]), (CHANNEL, [BLOCK])],
        ids=["channel", "cavity", "cavity-obstacle", "channel-obstacle"],
    )
    def test_residual_is_the_corrected_faces_imbalance(
        self, boundaries: dict, obstacles: list[dict] | None
    ) -> None:
        """b + A p' equals the mass imbalance of the corrected faces, cell by cell, to rounding.

        The identity REQ-S04's clarification rests on: the relative residual
        is the fraction of u*'s imbalance a correction leaves. The bound is
        64 eps times the largest face flux (the report found 6e-14 of it).
        """
        pc, c, b, pred, p = self._system(
            boundaries, obstacles=obstacles, max_pressure_iter=5
        )
        out = pc.correct(pred, p)
        assert out.reached_cap is True, "a loose solve, so the residual is not rounding"
        left = pc.mass_imbalance(out.u, out.v)
        r = _residual(c, b, out.p_prime)
        mesh = pc._mesh
        face_flux = 1.2 * max(
            np.abs(pred.u_star * mesh.dy_cell[:, None]).max(),
            np.abs(pred.v_star * mesh.dx_cell[None, :]).max(),
        )
        assert np.abs(left).max() > 1e-6 * face_flux
        assert np.abs(left - r).max() <= 64.0 * EPS * face_flux

    def test_closed_domain_right_hand_side_is_projected_onto_the_range(self) -> None:
        """An incompatible b (air entering through a lid face) still converges.

        Without the projection the constant part of b is outside the range,
        no p' removes it, and CG runs to its cap. With it the solve stops at
        the relative level on the projected residual, the mean imbalance is
        left as it must be, and everything else is removed.
        """
        pc, c, b_compatible, pred, p = self._system(CAVITY)
        v_leaky = pred.v_star.copy()
        v_leaky[-1, 2:4] = -0.1
        leaky = MomentumPrediction(
            u_star=pred.u_star, v_star=v_leaky, a_p_u=pred.a_p_u, a_p_v=pred.a_p_v
        )
        b = pc.mass_imbalance(pred.u_star, v_leaky)
        active = c.a_p > 0.0
        assert abs(b[active].mean()) > 1e-3 * np.abs(b).max()
        assert abs(b_compatible[active].mean()) < 1e-12 * np.abs(b_compatible).max()
        out = pc.correct(leaky, p)
        assert out.reached_cap is False
        left = pc.mass_imbalance(out.u, out.v)
        assert left[active].mean() == pytest.approx(b[active].mean(), rel=1e-10)
        f = _projected_rhs(pc, c, b)
        assert np.linalg.norm(
            left[active] - left[active].mean()
        ) <= TIGHT_RTOL * np.linalg.norm(f)

    def test_a_cell_without_an_equation_is_left_out_of_the_solve(self) -> None:
        """A sealed inlet cell (a_P = 0, b != 0) gets no p' and does not stall the solve.

        Two obstacles wall the channel's bottom-left cell off from the
        interior, leaving it the inlet face that pours air in: it is not
        SOLID, it has no correctable face, and its imbalance is the inlet
        flux. The operator's row there is zero, so a right-hand side left
        nonzero at that cell could never be reduced and CG would run to its
        cap. The solve zeroes f there, converges for every other cell, and
        leaves the sealed cell's imbalance as it must.
        """
        pocket = [
            {
                "name": "east",
                "x_start": 0.25,
                "x_end": 0.5,
                "y_start": 0.0,
                "y_end": 1 / 6,
            },
            {
                "name": "north",
                "x_start": 0.0,
                "x_end": 0.25,
                "y_start": 1 / 6,
                "y_end": 1 / 3,
            },
        ]
        mesh, _bc, pc, pred, p = _predicted(
            _config(CHANNEL, obstacles=pocket), outlet_right=True
        )
        c = pc.coefficients(pred.a_p_u, pred.a_p_v)
        b = pc.mass_imbalance(pred.u_star, pred.v_star)
        assert mesh.cell_type[0, 0] != SOLID
        assert c.a_p[0, 0] == 0.0
        assert b[0, 0] == pytest.approx(-1.2 * 0.3 * mesh.dy_cell[0], rel=1e-14)
        out = pc.correct(pred, p)
        assert out.reached_cap is False
        assert out.p_prime[0, 0] == 0.0
        left = pc.mass_imbalance(out.u, out.v)
        assert left[0, 0] == b[0, 0]
        active = c.a_p > 0.0
        assert np.linalg.norm(left[active]) <= TIGHT_RTOL * np.linalg.norm(b[active])

    def test_closed_domain_in_two_components_is_refused(self) -> None:
        """A wall the full height splits the cavity and is refused; one block does not."""
        with pytest.raises(ValueError, match="2 connected components"):
            _build(_config(CAVITY, obstacles=[PARTITION]))
        _mesh, _bc, _mp, one = _build(_config(CAVITY, obstacles=[BLOCK]))
        assert one.needs_pin

    def test_open_domain_with_a_component_no_outlet_reaches_is_refused(self) -> None:
        """The same wall across the channel leaves the inlet side no outlet: refused.

        Review 37 S7, confirmed by test 37: the left part's block of the
        operator has no p' = 0 row and its right-hand side carries the
        inflow, which no p' removes, so every correction ran to the cap and
        left the faces 189 times more unbalanced than u*. A shelf the full
        length splits the channel too, but each part reaches the outlet, so
        it is accepted and its correction converges. Defect caught: the check
        skipped on an open domain, or an open domain held to one component.
        """
        stranded = r"open domain: 1 of the 2 connected components .* no pressure outlet"
        with pytest.raises(ValueError, match=stranded):
            _build(_config(CHANNEL, obstacles=[PARTITION]))
        mesh, _bc, pc, pred, p = _predicted(
            _config(CHANNEL, obstacles=[SHELF]), outlet_right=True
        )
        assert (mesh.cell_type[2:4, :] == SOLID).all()
        assert not pc.needs_pin
        out = pc.correct(pred, p)
        assert out.reached_cap is False
        left = pc.mass_imbalance(out.u, out.v)
        b = pc.mass_imbalance(pred.u_star, pred.v_star)
        assert np.linalg.norm(left) <= TIGHT_RTOL * np.linalg.norm(b)

    @pytest.mark.parametrize(
        ("keyword", "value"),
        [
            ("rtol", True),
            ("rtol", -1e-8),
            ("rtol", 1.0),
            ("rtol", float("nan")),
            ("rtol", "1e-8"),
            ("floor", True),
            ("floor", -1e-13),
            ("floor", float("inf")),
            ("floor", float("nan")),
            ("max_iter", True),
            ("max_iter", 0),
            ("max_iter", 50.0),
        ],
    )
    def test_bad_scalar_arguments_are_refused(
        self, keyword: str, value: object
    ) -> None:
        """Each scalar is checked for type, bool and range, as the retired sweep's weight was.

        Review 37 S6: max_iter=True ran one iteration, a float cap was
        accepted, and a negative rtol silently left only the floor.
        """
        _pc, c, b, _pred, _p = self._system(CHANNEL)
        args = {"rtol": 1e-8, "floor": 0.0, "max_iter": 50} | {keyword: value}
        with pytest.raises(ValueError, match=f"^{keyword} must be"):
            conjugate_gradient(
                lambda x: apply_operator(c, x), _inverse_diagonal(c), -b, **args
            )

    def test_edge_arguments_are_accepted_and_a_wrong_shape_refused(self) -> None:
        """rtol 0 with floor 0 and one iteration is a legal, capped solve; shapes must match."""
        _pc, c, b, _pred, _p = self._system(CHANNEL)
        inv = _inverse_diagonal(c)
        res = conjugate_gradient(lambda x: apply_operator(c, x), inv, -b, 0.0, 0.0, 1)
        assert (res.iterations, res.reached_cap) == (1, True)
        with pytest.raises(ValueError, match="inverse_diagonal"):
            conjugate_gradient(
                lambda x: apply_operator(c, x), inv.T.copy(), -b, 1e-8, 0.0, 50
            )
