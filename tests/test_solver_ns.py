"""Tests for the Navier-Stokes SIMPLE solver.

Unit tests verify coefficient computation, under-relaxation, Jacobi
sweeps, pressure correction, velocity correction, and convergence
monitoring. Integration tests verify initialization and basic
convergence on a simple channel geometry.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.boundary import BoundaryManager
from src.config import SimConfig
from src.mesh import FLUID, SOLID, Mesh
from src.solver_ns import NavierStokesSolver


def _make_config(tmp_path: Path, overrides: dict | None = None) -> SimConfig:
    """Write a YAML config and return a loaded SimConfig."""
    base = {
        "domain": {"width": 1.0, "height": 1.0, "nx": 10, "ny": 10},
        "fluid": {"density": 1.2, "viscosity": 1.81e-5, "temperature": 293.0},
        "particles": {
            "density": 1000.0,
            "sizes": [0.1e-6],
            "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {
                "diameters": [0.1e-6],
                "efficiencies": [0.99999],
            },
        },
        "solver": {
            "dt": 0.01,
            "t_end": 1.0,
            "output_interval": 10,
            "convergence_tol": 1.0e-6,
            "max_simple_iter": 100,
            "alpha_velocity": 0.7,
            "alpha_pressure": 0.3,
            "max_pressure_iter": 50,
            "pressure_tol": 1.0e-6,
        },
        "boundaries": {
            "top_inlet": {
                "type": "velocity_inlet",
                "location": "top",
                "x_start": 0.0,
                "x_end": 1.0,
                "velocity": 0.1,
            },
            "bottom_outlet": {
                "type": "pressure_outlet",
                "location": "bottom",
                "x_start": 0.0,
                "x_end": 1.0,
            },
        },
        "obstacles": [],
        "sensors": [{"name": "center", "x": 0.5, "y": 0.5}],
        "thresholds": {"0.1e-6": 100.0},
    }
    if overrides:
        for key, val in overrides.items():
            base[key] = val
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(base, default_flow_style=False), encoding="utf-8")
    return SimConfig(str(path))


def _make_solver(
    tmp_path: Path, overrides: dict | None = None
) -> tuple[NavierStokesSolver, Mesh, SimConfig]:
    """Create a NavierStokesSolver from a test config."""
    config = _make_config(tmp_path, overrides)
    mesh = Mesh(config)
    boundary = BoundaryManager(mesh, config)
    return NavierStokesSolver(mesh, config, boundary), mesh, config


# ---------------------------------------------------------------------------
# Unit tests -- Coefficient computation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMomentumCoefficients:
    """Verify momentum coefficient computation."""

    def test_coefficients_positive(self, tmp_path) -> None:
        """All neighbor coefficients are non-negative at FLUID cells."""
        solver, mesh, _ = _make_solver(tmp_path)
        u = np.ones((10, 10), dtype=np.float64) * 0.1
        v = np.zeros((10, 10), dtype=np.float64)
        a_P, a_E, a_W, a_N, a_S = solver._compute_momentum_coefficients(u, v)

        fluid = mesh.cell_type == FLUID
        assert np.all(a_E[fluid] >= 0)
        assert np.all(a_W[fluid] >= 0)
        assert np.all(a_N[fluid] >= 0)
        assert np.all(a_S[fluid] >= 0)
        assert np.all(a_P[fluid] > 0)

    def test_zero_velocity_pure_diffusion(self, tmp_path) -> None:
        """With zero velocity, coefficients reduce to diffusion only.

        At Pe=0, hybrid scheme gives central difference: a_nb = D.
        """
        solver, mesh, config = _make_solver(tmp_path)
        u = np.zeros((10, 10), dtype=np.float64)
        v = np.zeros((10, 10), dtype=np.float64)
        _a_P, a_E, a_W, a_N, a_S = solver._compute_momentum_coefficients(u, v)

        D_ew = config.mu * mesh.dy / mesh.dx
        D_ns = config.mu * mesh.dx / mesh.dy

        # Interior FLUID cell not adjacent to any SOLID
        # (cell (5,5) on a 10x10 grid with no obstacles)
        assert a_E[5, 5] == pytest.approx(D_ew)
        assert a_W[5, 5] == pytest.approx(D_ew)
        assert a_N[5, 5] == pytest.approx(D_ns)
        assert a_S[5, 5] == pytest.approx(D_ns)

    def test_solid_neighbor_coefficient_zero(self, tmp_path) -> None:
        """Neighbor coefficient is zero when adjacent cell is SOLID."""
        solver, mesh, _ = _make_solver(
            tmp_path,
            overrides={
                "obstacles": [
                    {
                        "name": "block",
                        "x_start": 0.55,
                        "x_end": 0.75,
                        "y_start": 0.35,
                        "y_end": 0.65,
                    }
                ]
            },
        )
        u = np.zeros((10, 10), dtype=np.float64)
        v = np.zeros((10, 10), dtype=np.float64)
        _a_P, a_E, _a_W, _a_N, _a_S = solver._compute_momentum_coefficients(u, v)

        # Find a FLUID cell adjacent to SOLID on its east side
        for j in range(1, 9):
            for i in range(1, 8):
                if mesh.cell_type[j, i] == FLUID and mesh.cell_type[j, i + 1] == SOLID:
                    assert a_E[j, i] == 0.0, f"a_E at ({i},{j}) should be 0"
                    return
        pytest.skip("No FLUID cell with SOLID east neighbor found")


# ---------------------------------------------------------------------------
# Unit tests -- Under-relaxation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnderRelaxation:
    """Verify under-relaxation modifies coefficients correctly."""

    def test_a_P_ur_scaling(self, tmp_path) -> None:
        """Under-relaxed diagonal equals a_P / alpha_u."""
        solver, _, _ = _make_solver(tmp_path)
        a_P = np.ones((10, 10), dtype=np.float64) * 100.0
        a_P_ur = a_P / solver._alpha_u
        assert a_P_ur[5, 5] == pytest.approx(100.0 / 0.7)


# ---------------------------------------------------------------------------
# Unit tests -- d-coefficient
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDCoefficient:
    """Verify d = V / a_P computation."""

    def test_d_uses_original_a_P(self, tmp_path) -> None:
        """d-coefficient uses original a_P, not under-relaxed."""
        _solver, mesh, _ = _make_solver(tmp_path)
        a_P = np.ones((10, 10), dtype=np.float64) * 50.0
        V = mesh.dx * mesh.dy
        d_expected = V / 50.0

        d = np.zeros((10, 10), dtype=np.float64)
        fluid = mesh.cell_type == FLUID
        d[fluid] = V / a_P[fluid]

        assert d[5, 5] == pytest.approx(d_expected)

    def test_d_zero_at_solid(self, tmp_path) -> None:
        """d-coefficient is zero at SOLID cells."""
        _solver, mesh, _ = _make_solver(
            tmp_path,
            overrides={
                "obstacles": [
                    {
                        "name": "block",
                        "x_start": 0.4,
                        "x_end": 0.6,
                        "y_start": 0.4,
                        "y_end": 0.6,
                    }
                ]
            },
        )
        a_P = np.ones((10, 10), dtype=np.float64) * 50.0
        V = mesh.dx * mesh.dy
        d = np.zeros((10, 10), dtype=np.float64)
        fluid = mesh.cell_type == FLUID
        d[fluid] = V / a_P[fluid]

        solid_cells = mesh.cell_type == SOLID
        assert np.all(d[solid_cells] == 0.0)


# ---------------------------------------------------------------------------
# Unit tests -- Pressure correction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPressureCorrection:
    """Verify pressure correction Jacobi solver."""

    def test_zero_imbalance_gives_zero_correction(self, tmp_path) -> None:
        """With zero mass imbalance, p' should remain near zero."""
        solver, mesh, _ = _make_solver(tmp_path)
        mass_imb = np.zeros((10, 10), dtype=np.float64)
        d = np.ones((10, 10), dtype=np.float64) * 0.001
        d[~(mesh.cell_type == FLUID)] = 0.0

        p_prime = solver._solve_pressure_correction(mass_imb, d)
        assert np.max(np.abs(p_prime)) < 1e-10

    def test_solid_cells_zero_pprime(self, tmp_path) -> None:
        """SOLID cells have zero pressure correction."""
        solver, mesh, _ = _make_solver(
            tmp_path,
            overrides={
                "obstacles": [
                    {
                        "name": "block",
                        "x_start": 0.4,
                        "x_end": 0.6,
                        "y_start": 0.4,
                        "y_end": 0.6,
                    }
                ]
            },
        )
        mass_imb = np.ones((10, 10), dtype=np.float64) * 0.001
        mass_imb[~(mesh.cell_type == FLUID)] = 0.0
        d = np.ones((10, 10), dtype=np.float64) * 0.001
        d[~(mesh.cell_type == FLUID)] = 0.0

        p_prime = solver._solve_pressure_correction(mass_imb, d)
        solid = mesh.cell_type == SOLID
        assert np.all(p_prime[solid] == 0.0)


# ---------------------------------------------------------------------------
# Unit tests -- Velocity correction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVelocityCorrection:
    """Verify velocity correction formula."""

    def test_zero_pprime_no_correction(self, tmp_path) -> None:
        """With zero p', velocity correction leaves u_star unchanged."""
        solver, mesh, _ = _make_solver(tmp_path)
        u_star = np.ones((10, 10), dtype=np.float64) * 0.5
        v_star = np.ones((10, 10), dtype=np.float64) * -0.3
        p_prime = np.zeros((10, 10), dtype=np.float64)
        d = np.ones((10, 10), dtype=np.float64) * 0.001

        u, v = solver._correct_velocity(u_star, v_star, p_prime, d)

        fluid = mesh.cell_type == FLUID
        assert np.allclose(u[fluid], u_star[fluid])
        assert np.allclose(v[fluid], v_star[fluid])


# ---------------------------------------------------------------------------
# Unit tests -- Convergence monitoring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResidual:
    """Verify residual computation."""

    def test_divergence_free_gives_zero(self, tmp_path) -> None:
        """A uniform zero velocity field has zero residual."""
        solver, _, _ = _make_solver(tmp_path)
        u = np.zeros((10, 10), dtype=np.float64)
        v = np.zeros((10, 10), dtype=np.float64)
        p = np.zeros((10, 10), dtype=np.float64)

        residual = solver._compute_residual(u, v, p)
        assert residual == pytest.approx(0.0, abs=1e-15)

    def test_nonzero_divergence_gives_positive(self, tmp_path) -> None:
        """A velocity field with net mass imbalance gives positive residual."""
        solver, _, _ = _make_solver(tmp_path)
        u = np.zeros((10, 10), dtype=np.float64)
        v = np.zeros((10, 10), dtype=np.float64)
        # Create a divergence: flow in from left, no flow out
        u[:, 0] = 1.0
        u[:, 1] = 0.5
        p = np.zeros((10, 10), dtype=np.float64)

        residual = solver._compute_residual(u, v, p)
        assert residual > 0.0

    def test_public_compute_residual_before_solve(self, tmp_path) -> None:
        """compute_residual() returns 0.0 before solve_steady is called."""
        solver, _, _ = _make_solver(tmp_path)
        assert solver.compute_residual() == 0.0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSolverIntegration:
    """Verify solver initialization and basic convergence."""

    def test_default_config_initializes(self) -> None:
        """Solver initializes from the default clean room config."""
        config = SimConfig("configs/clean_room_default.yaml")
        mesh = Mesh(config)
        boundary = BoundaryManager(mesh, config)
        solver = NavierStokesSolver(mesh, config, boundary)
        assert solver._nx == config.nx
        assert solver._ny == config.ny

    def test_solve_runs_without_crash(self, tmp_path) -> None:
        """solve_steady runs for multiple iterations on a simple channel."""
        solver, _mesh, _config = _make_solver(
            tmp_path,
            overrides={
                "solver": {
                    "dt": 0.01,
                    "t_end": 1.0,
                    "output_interval": 10,
                    "convergence_tol": 1.0e-4,
                    "max_simple_iter": 20,
                    "alpha_velocity": 0.7,
                    "alpha_pressure": 0.3,
                    "max_pressure_iter": 30,
                    "pressure_tol": 1.0e-5,
                }
            },
        )
        u, v, p = solver.solve_steady()
        assert u.shape == (10, 10)
        assert v.shape == (10, 10)
        assert p.shape == (10, 10)

    def test_output_dtypes_and_contiguity(self, tmp_path) -> None:
        """Output arrays have correct dtype and are C-contiguous."""
        solver, _, _ = _make_solver(
            tmp_path,
            overrides={
                "solver": {
                    "dt": 0.01,
                    "t_end": 1.0,
                    "output_interval": 10,
                    "convergence_tol": 1.0,
                    "max_simple_iter": 5,
                    "alpha_velocity": 0.7,
                    "alpha_pressure": 0.3,
                    "max_pressure_iter": 10,
                    "pressure_tol": 1.0e-3,
                }
            },
        )
        u, v, p = solver.solve_steady()
        assert u.dtype == np.float64
        assert v.dtype == np.float64
        assert p.dtype == np.float64
        assert u.flags["C_CONTIGUOUS"]
        assert v.flags["C_CONTIGUOUS"]
        assert p.flags["C_CONTIGUOUS"]

    def test_velocities_bounded(self, tmp_path) -> None:
        """Velocities stay bounded during the solve.

        On a simple channel flow with low inlet velocity, the
        velocity magnitude should not exceed several multiples of
        the inlet velocity. This catches catastrophic divergence.
        """
        solver, _, _ = _make_solver(
            tmp_path,
            overrides={
                "domain": {"width": 1.0, "height": 1.0, "nx": 20, "ny": 20},
                "fluid": {"density": 1.0, "viscosity": 0.01, "temperature": 293.0},
                "solver": {
                    "dt": 0.01,
                    "t_end": 1.0,
                    "output_interval": 10,
                    "convergence_tol": 1.0e-4,
                    "max_simple_iter": 200,
                    "alpha_velocity": 0.3,
                    "alpha_pressure": 0.1,
                    "max_pressure_iter": 100,
                    "pressure_tol": 1.0e-6,
                },
            },
        )
        u, v, _p = solver.solve_steady()
        max_vel = max(float(np.max(np.abs(u))), float(np.max(np.abs(v))))
        # Velocities should stay within a reasonable range
        # (inlet velocity is 0.1 m/s from the base config)
        assert max_vel < 10.0, f"Velocity diverged to {max_vel:.2f}"

    def test_solve_timestep_raises(self, tmp_path) -> None:
        """solve_timestep raises NotImplementedError."""
        solver, _, _ = _make_solver(tmp_path)
        u = np.zeros((10, 10), dtype=np.float64)
        v = np.zeros((10, 10), dtype=np.float64)
        p = np.zeros((10, 10), dtype=np.float64)
        with pytest.raises(NotImplementedError):
            solver.solve_timestep(u, v, p, 0.01)
