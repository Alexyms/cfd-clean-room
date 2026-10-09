"""Developed channel flow under a viscosity that varies across the channel (ECR-002 step 4).

The momentum predictor's viscosity field (ADR-012 D) is driven through
``StaggeredSolver.solve_steady(eddy_viscosity=...)`` with a prescribed
``mu(y) = mu0 + rho nu_t(y)``, smooth and asymmetric, from 1 to about 4.6
times mu0. In developed flow ``u = u(y)``, ``v = 0`` and the shear stress is
linear across the channel, ``mu du/dy = -G (y - y0)``. With ``u = 0`` on both
walls,

    u(y) = -G int_0^y (s - y0) / mu(s) ds,
    y0 = int_0^H s / mu ds / int_0^H 1 / mu ds,

and G follows from the inflow, ``int_0^H u dy = U H``, which by parts is
``-G int_0^H (H - s)(s - y0) / mu(s) ds``. The integrals are taken by
composite Gauss-Legendre quadrature, accurate to rounding, far below the
discretisation error. With a uniform mu the integral is Poiseuille's parabola.

The developed section. The channel is four heights long with a uniform
inflow; the column read is the u faces at x = 3H. Under this field the inlet
disturbance decays more slowly than under a uniform viscosity: on 16 rows the
column at x = 1.5H still differs from the discrete developed profile by
1.8e-3 relative, at 2.25H by 2e-5, at 3H by 1e-5 (a six-height channel,
measured for this test). The test asserts that the column at 3H and the one
at 3.25H agree to well under the discretisation error, so the read section is
developed by the test's own measure.
"""

from collections.abc import Callable

import numpy as np
import pytest

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import Mesh
from src.solver_staggered import StaggeredSolver

H = 0.5
U = 0.1
RHO = 1.0
MU0 = 0.01
LENGTHS = 4  # channel length in heights
READ_AT = 3  # the column read, in heights from the inlet
GRIDS = (8, 12, 16)


def _mu(y: np.ndarray) -> np.ndarray:
    """The prescribed dynamic viscosity across the channel, Pa s."""
    return MU0 * (1.0 + 3.0 * y / H + 2.0 * np.sin(np.pi * y / H) ** 2)


def _integral(
    f: Callable[[np.ndarray], np.ndarray], a: float, b: float, panels: int = 400
) -> float:
    """Composite eight-point Gauss-Legendre quadrature of f over [a, b]."""
    x, w = np.polynomial.legendre.leggauss(8)
    edges = np.linspace(a, b, panels + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])[:, None]
    half = 0.5 * (edges[1:] - edges[:-1])[:, None]
    return float(np.sum(f(mid + half * x[None, :]) * w[None, :] * half))


def _developed(
    y: np.ndarray, mu: Callable[[np.ndarray], np.ndarray] = _mu
) -> np.ndarray:
    """u(y) of developed flow at mean velocity U under the viscosity mu(y)."""
    y0 = _integral(lambda s: s / mu(s), 0.0, H) / _integral(
        lambda s: 1.0 / mu(s), 0.0, H
    )
    shape_flux = -_integral(lambda s: (H - s) * (s - y0) / mu(s), 0.0, H)
    g = U * H / shape_flux
    return np.array(
        [-g * _integral(lambda s: (s - y0) / mu(s), 0.0, float(yy)) for yy in y]
    )


def _channel(ny: int) -> SimConfig:
    nx = LENGTHS * 2 * ny
    raw = {
        "domain": {"width": LENGTHS * H, "height": H, "nx": nx, "ny": ny},
        "fluid": {"density": RHO, "viscosity": MU0, "temperature": 293.0},
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
            "max_simple_iter": 20000,
            "alpha_velocity": 0.7,
            "alpha_pressure": 0.3,
            "max_pressure_iter": 5000,
            "pressure_rtol": 1.0e-8,
            "stopping_rule": "error_estimate",
            "iteration_error_tol": 1.0e-6,
            "mass_imbalance_tol": 1.0e-10,
        },
        "boundaries": {
            "inlet": {
                "type": "velocity_inlet",
                "location": "left",
                "y_start": 0.0,
                "y_end": H,
                "velocity": U,
            },
            "outlet": {
                "type": "pressure_outlet",
                "location": "right",
                "y_start": 0.0,
                "y_end": H,
            },
        },
        "obstacles": [],
        "sensors": [{"name": "center", "x": H, "y": H / 2}],
        "thresholds": {"0.1e-6": 100.0},
    }
    return SimConfig.from_dict(raw)


def _relative_l2(mesh: Mesh, a: np.ndarray, b: np.ndarray) -> float:
    w = mesh.dy_cell
    return float(np.sqrt(np.sum(w * (a - b) ** 2) / np.sum(w * b**2)))


@pytest.mark.unit
def test_the_reference_reduces_to_poiseuille_under_a_uniform_viscosity() -> None:
    """With mu constant the integral is 6 U y (H - y) / H^2, to rounding."""
    y = np.linspace(0.0, H, 11)
    parabola = 6.0 * U * y * (H - y) / H**2
    developed = _developed(y, mu=lambda s: np.full_like(s, MU0))
    assert developed == pytest.approx(parabola, rel=1e-12, abs=1e-15)


@pytest.mark.validation
def test_varying_viscosity_channel_converges_at_second_order() -> None:
    """ECR-002 step 4: the developed column against the integral, observed order 1.7 to 2.3."""
    errors = []
    for ny in GRIDS:
        config = _channel(ny)
        mesh = Mesh(config)
        solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
        nu_t = np.tile(((_mu(mesh.yc) - MU0) / RHO)[:, None], (1, mesh.xc.size))
        solver.solve_steady(eddy_viscosity=nu_t)
        assert solver.stop_reason == "error_estimate_and_continuity"
        faces = solver.face_velocities
        assert faces is not None
        column = faces.u[:, READ_AT * 2 * ny]
        downstream = faces.u[:, READ_AT * 2 * ny + ny // 2]
        error = _relative_l2(mesh, column, _developed(mesh.yc))
        # Developed by the test's own measure: a quarter height downstream
        # the column is the same to well under its error.
        assert _relative_l2(mesh, column, downstream) < 0.02 * error
        errors.append(error)
    ratios = np.array(GRIDS[1:]) / np.array(GRIDS[:-1])
    orders = np.log(np.array(errors[:-1]) / np.array(errors[1:])) / np.log(ratios)
    assert np.all((orders > 1.7) & (orders < 2.3)), (errors, orders)
