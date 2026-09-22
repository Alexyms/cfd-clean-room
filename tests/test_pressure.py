"""Tests for the staggered pressure correction (REQ-S04, REQ-S08).

The headline is the compatibility of the closed-domain system: the
right-hand side summed over the cavity must vanish to rounding, because the
wall faces hold zero exactly. The bound is machine epsilon times the total
absolute face flux of the field, the quantity the telescoping sum cancels,
not a tolerance chosen to pass. The collocated solver measured 2.90e-2,
9.23e-3 and 2.35e-3 on the same three grids (docs/reports/pressure_solver_probe.md,
table E).

The second is why the Jacobi update is weighted: on the closed system the
checkerboard is an exact eigenvector, with eigenvalue exactly -1 for the
plain update and exactly 1 - 2w for the weighted one.
"""

import numpy as np
import pytest
import yaml

import src.pressure as pressure
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import FLUID, SOLID, Mesh
from src.momentum import MomentumPrediction, MomentumPredictor
from src.pressure import JACOBI_WEIGHT, PressureCoefficients, PressureCorrector
from src.staggered import allocate_fields
from validation.cases import case_path, load_case

EPS = np.finfo(np.float64).eps
STRETCHED = {"x": {"stretch_ratio": 1.15}, "y": {"stretch_ratio": 1.25}}


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


def _config(
    boundaries: dict,
    nx: int = 8,
    ny: int = 6,
    obstacles: list[dict] | None = None,
    mesh: dict | None = None,
    max_pressure_iter: int = 5000,
    pressure_tol: float = 1e-12,
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
            "pressure_tol": pressure_tol,
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


def _case_with_cap(name: str, n: int, max_pressure_iter: int) -> SimConfig:
    """A committed case on an n x n grid with only the sweep cap raised."""
    raw = yaml.safe_load(case_path(name).read_text(encoding="utf-8"))
    raw["domain"]["nx"] = raw["domain"]["ny"] = n
    raw["solver"]["max_pressure_iter"] = max_pressure_iter
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


# ---------------------------------------------------------------------------
# The correction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCorrection:
    """p', the corrected velocities and the updated pressure."""

    def test_uniform_field_gives_zero_correction_exactly(self) -> None:
        """A field with zero imbalance is returned bit for bit, after one sweep."""
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
        assert out.sweeps == 1

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

    def test_corrected_open_domain_satisfies_continuity(self) -> None:
        """With an outlet the system is nonsingular and every cell closes."""
        # The weight slows the slowest mode by about 1/w = 1.5, more than the
        # default cap of 5000 allows for on this grid.
        config = _config(CHANNEL, max_pressure_iter=20_000)
        _mesh, _bc, pc, pred, p = _predicted(config, outlet_right=True)
        before = pc.mass_imbalance(pred.u_star, pred.v_star)
        assert np.abs(before).max() > 1e-4
        out = pc.correct(pred, p)
        after = pc.mass_imbalance(out.u, out.v)
        c = pc.coefficients(pred.a_p_u, pred.a_p_v)
        assert np.abs(after).max() < 1e-9 * c.a_p.max()
        assert out.sweeps < config.max_pressure_iter

    @pytest.mark.parametrize("cap", [999, 1000])
    def test_closed_domain_plain_jacobi_stalls_on_the_checkerboard_mode(
        self, cap: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What undamped Jacobi leaves behind on the closed system, recorded.

        With a_P equal to the neighbour sum on a bipartite grid the sign
        vector (-1)^(i+j) is an exact eigenvector of the Jacobi iteration
        with eigenvalue -1, so that component of the error never decays
        and flips sign every sweep. The residual after the solve is then
        exactly a checkerboard, r / a_P = +-constant, whichever way the
        cap falls. This is the evidence JACOBI_WEIGHT answers, so the test
        switches the weight off to keep it reproducible; the weighted solve
        on the same field converges (TestJacobiWeight). See
        docs/reports/pressure_correction_step5.md.
        """
        monkeypatch.setattr(pressure, "JACOBI_WEIGHT", 1.0)
        config = _config(CAVITY, max_pressure_iter=cap)
        _mesh, _bc, pc, pred, p = _predicted(config)
        out = pc.correct(pred, p)
        assert out.sweeps == cap
        residual = pc.mass_imbalance(out.u, out.v)
        c = pc.coefficients(pred.a_p_u, pred.a_p_v)
        j, i = np.indices(residual.shape)
        checkerboard = residual * (-1.0) ** (j + i) / c.a_p
        amplitude = checkerboard.mean()
        assert abs(amplitude) > 1e-4
        assert np.sign(amplitude) == (1.0 if cap % 2 else -1.0)
        # Rounding accumulated over the sweeps, relative to the amplitude
        assert np.abs(checkerboard - amplitude).max() < 1e-10 * abs(amplitude)

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
        # The weighted sweep contracts more slowly here than the plain one, so
        # the tolerance is tightened until it no longer limits rel=1e-12 below.
        config = _config(outlet, nx=2, ny=1, pressure_tol=1e-14)
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
        """The pin acts on the solve: p' at the reference cell is exactly zero and
        the whole field was shifted by the nonzero value Jacobi left there.

        The constant vector is an exact eigenvector of the closed system, so
        without the subtraction the level of p' is wherever the iteration
        left it. The raw solve is taken by switching the pin off on the same
        corrector; the pinned field must be that field minus its reference
        value, bit for bit, and that value must not be zero.
        """
        _mesh, _bc, pc, pred, p = _predicted(_config(CAVITY))
        pc.needs_pin = False
        raw = pc.correct(pred, p).p_prime
        pc.needs_pin = True
        out = pc.correct(pred, p)
        assert raw[pc.pin_cell] != 0.0
        assert out.p_prime[pc.pin_cell] == 0.0
        assert np.array_equal(out.p_prime, raw - raw[pc.pin_cell])
        assert out.p_prime.mean() == pytest.approx(
            raw.mean() - raw[pc.pin_cell], rel=1e-12
        )

    def test_sweeps_are_capped_by_max_pressure_iter(self) -> None:
        """The Jacobi loop stops at the configured cap and reports it."""
        config = _config(CAVITY, max_pressure_iter=7)
        _mesh, _bc, pc, pred, p = _predicted(config)
        out = pc.correct(pred, p)
        assert out.sweeps == 7

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


# ---------------------------------------------------------------------------
# The weighted sweep
# ---------------------------------------------------------------------------

CLOSED_GEOMETRIES: dict[str, dict] = {
    "uniform": {},
    "stretched": {"mesh": STRETCHED},
    "obstacle": {"obstacles": [BLOCK]},
}


@pytest.mark.unit
class TestJacobiWeight:
    """Why the Jacobi update is weighted, and that the weight does its job (REQ-S08).

    On a closed domain the checkerboard (-1)^(i+j) is an exact eigenvector
    of the plain update with eigenvalue -1. The eigenvalues are compared
    bit for bit, not with a tolerance: they are exact, so any tolerance
    only widens the set of wrong weights that would pass.
    """

    def _closed_system(
        self, geometry: str
    ) -> tuple[PressureCorrector, PressureCoefficients, np.ndarray]:
        """Closed-cavity coefficients and the checkerboard over the cells with an equation."""
        _mesh, _bc, pc, pred, _p = _predicted(
            _config(CAVITY, **CLOSED_GEOMETRIES[geometry])
        )
        c = pc.coefficients(pred.a_p_u, pred.a_p_v)
        j, i = np.indices(c.a_p.shape)
        s = np.where(c.a_p > 0.0, (-1.0) ** (j + i), 0.0)
        return pc, c, s

    @pytest.mark.parametrize("geometry", CLOSED_GEOMETRIES)
    def test_plain_sweep_has_eigenvalue_exactly_minus_one(self, geometry: str) -> None:
        """One plain sweep returns exactly -s, in every cell.

        Exact rather than approximate because a_P is the same four-term sum
        the sweep forms: negating every term is exact in floating point, so
        the neighbour sum is -a_P s_P to the bit and the quotient is -s_P.
        """
        pc, c, s = self._closed_system(geometry)
        assert np.array_equal(c.a_p, c.a_e + c.a_w + c.a_n + c.a_s)
        out = pc.sweep(s, c, np.zeros_like(s), weight=1.0)
        assert np.array_equal(out, -s)

    @pytest.mark.parametrize("geometry", CLOSED_GEOMETRIES)
    def test_weighted_sweep_has_eigenvalue_minus_one_third(self, geometry: str) -> None:
        """One weighted sweep multiplies every cell by exactly 1 - 2w, which is -1/3.

        Two thirds has no exact binary form, so the stored weight is the
        nearest double and the eigenvalue of the iteration as computed is
        1 - 2 fl(2/3), one ulp above fl(-1/3). The test pins both: every
        cell's ratio equals 1 - 2w bit for bit, and that value is within
        one ulp of -1/3. A weight of 0.7 would miss by 6.7e-2.
        """
        pc, c, s = self._closed_system(geometry)
        before = s.copy()
        out = pc.sweep(s, c, np.zeros_like(s), weight=JACOBI_WEIGHT)
        assert np.array_equal(s, before), "the sweep must not modify its input"
        active = c.a_p > 0.0
        ratio = out[active] / s[active]
        assert JACOBI_WEIGHT == 2.0 / 3.0
        assert np.all(ratio == 1.0 - 2.0 * JACOBI_WEIGHT)
        assert abs(ratio[0] + 1.0 / 3.0) <= np.spacing(1.0 / 3.0)
        assert np.all(out[~active] == 0.0)

    def test_correct_applies_the_weight(self) -> None:
        """One sweep of correct from p' = 0 is the weighted sweep, not the plain one.

        On an open domain nothing is pinned, so p' after a single sweep is
        exactly w times the plain Jacobi update of zero.
        """
        config = _config(CHANNEL, max_pressure_iter=1)
        _mesh, _bc, pc, pred, p = _predicted(config, outlet_right=True)
        c = pc.coefficients(pred.a_p_u, pred.a_p_v)
        b = pc.mass_imbalance(pred.u_star, pred.v_star)
        zero = np.zeros_like(p)
        plain = pc.sweep(zero, c, b, weight=1.0)
        out = pc.correct(pred, p)
        assert out.sweeps == 1
        assert np.array_equal(out.p_prime, JACOBI_WEIGHT * plain)
        assert np.abs(plain).max() > 0.0

    def test_corrected_closed_domain_satisfies_continuity(self) -> None:
        """The closed counterpart of the open-domain test, to the same bound.

        This is the field and configuration on which plain Jacobi stalls at
        any cap (test_closed_domain_plain_jacobi_stalls_on_the_checkerboard_mode).
        """
        config = _config(CAVITY, max_pressure_iter=20_000)
        _mesh, _bc, pc, pred, p = _predicted(config)
        before = pc.mass_imbalance(pred.u_star, pred.v_star)
        assert np.abs(before).max() > 1e-4
        out = pc.correct(pred, p)
        after = pc.mass_imbalance(out.u, out.v)
        c = pc.coefficients(pred.a_p_u, pred.a_p_v)
        assert out.sweeps < config.max_pressure_iter
        assert np.abs(after).max() < 1e-9 * c.a_p.max()

    @pytest.mark.parametrize("n", [20, 40])
    def test_seeded_cavity_converges_at_the_case_tolerance(self, n: int) -> None:
        """The cavity the step 5 report found stuck at every cap now converges.

        Five predictor sweeps from rest, then one correction at the case
        file's pressure_tol, as in the report. Only the sweep cap is raised:
        the case file's own cap is below what the solve needs, a step 6
        concern. Plain Jacobi left 3.5e-3 and 8.4e-4 of the initial
        imbalance after 200,000 sweeps; the weighted solve stops on its
        tolerance and leaves less than 1e-6 of it. Sweep counts are in
        docs/reports/pressure_correction_step5.md, section 5.
        """
        config = _case_with_cap("cavity", n, 50_000)
        _mesh, _bc, pc, pred, p = _predicted(config)
        before = np.abs(pc.mass_imbalance(pred.u_star, pred.v_star)).max()
        out = pc.correct(pred, p)
        after = np.abs(pc.mass_imbalance(out.u, out.v)).max()
        assert out.sweeps < config.max_pressure_iter
        assert after < 1e-6 * before

    def test_sweep_rejects_bad_weights_and_shapes(self) -> None:
        """The weight must be a number in (0, 1]; every array must match the mesh."""
        pc, c, s = self._closed_system("uniform")
        b = np.zeros_like(s)
        for bad in (0.0, -0.5, 1.5, float("nan"), True, "0.5", None):
            with pytest.raises(ValueError, match="weight"):
                pc.sweep(s, c, b, weight=bad)
        with pytest.raises(ValueError, match="expected p_prime"):
            pc.sweep(s.T.copy(), c, b, weight=JACOBI_WEIGHT)
        with pytest.raises(ValueError, match="expected p_prime"):
            pc.sweep(s, c, b[:-1], weight=JACOBI_WEIGHT)
        short = PressureCoefficients(
            a_p=c.a_p, a_e=c.a_e[:, :-1], a_w=c.a_w, a_n=c.a_n, a_s=c.a_s
        )
        with pytest.raises(ValueError, match="expected p_prime"):
            pc.sweep(s, short, b, weight=JACOBI_WEIGHT)
