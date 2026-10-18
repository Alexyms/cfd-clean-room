"""Lift, drag and the Strouhal number (src/forces.py, ADR-014 F, step 6b).

The force gauge reads the pressure of the live cell beside each face of a
named obstacle and the shear of the momentum predictor's wall stencil, so
it is checked three ways: by hand on fields whose answer follows from the
mesh arrays (a uniform and a linear pressure, a uniform velocity); on a
steady symmetric flow, where the lift vanishes to rounding; and on a steady
channel, where the drag is compared with a control-volume momentum balance.

The balance's tolerance has a measured reason. The gauge reads the pressure
half a cell from each face and the shear over half a cell, as the
predictor's own stencils do, and a square body's corners make the flow
singular there, so the gauge's drag converges to the balance's at about
half order: the gap was 5.6%, 4.1% and 2.9% on 60x20, 120x40 and 240x80
(results/builder68/forces_proto.py, the freestream case below). The test
holds the two coarser grids within 7% and 5% and requires the gap to
shrink between them.

The Strouhal instrument is checked on synthetic signals whose frequency is
known. The planted defects each test was shown to catch are named in its
docstring (docs/reports/val022/plant68.py).
"""

import math

import numpy as np
import pytest

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.forces import STROUHAL_PERIODS, ForceGauge, ObstacleForces, strouhal
from src.mesh import SOLID, Mesh
from src.solver_staggered import StaggeredSolver
from src.staggered import allocate_fields

BLOCK = {"name": "block", "x_start": 1.0, "x_end": 1.2, "y_start": 0.4, "y_end": 0.6}
MU = 0.05


def _raw(nx: int, ny: int, freestream: bool, mesh: dict | None = None) -> dict:
    """A 3 m by 1 m channel, inlet 1 m/s, a 0.2 m block centred on the axis."""
    boundaries = {
        "inlet": {
            "type": "velocity_inlet",
            "location": "left",
            "y_start": 0.0,
            "y_end": 1.0,
            "velocity": 1.0,
        },
        "outlet": {
            "type": "pressure_outlet",
            "location": "right",
            "y_start": 0.0,
            "y_end": 1.0,
        },
    }
    if freestream:
        for side in ("top", "bottom"):
            boundaries[side] = {
                "type": "velocity_inlet",
                "location": side,
                "x_start": 0.0,
                "x_end": 3.0,
                "u_velocity": 1.0,
                "v_velocity": 0.0,
            }
    raw = {
        "domain": {"width": 3.0, "height": 1.0, "nx": nx, "ny": ny},
        "fluid": {"density": 1.0, "viscosity": MU, "temperature": 293.0},
        "particles": {
            "density": 1000.0,
            "sizes": [0.1e-6],
            "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {"diameters": [0.1e-6], "efficiencies": [0.99999]},
        },
        "solver": {
            "dt": 0.1,
            "t_end": 1.0,
            "output_interval": 10,
            "convergence_tol": 1e-12,
            "max_simple_iter": 20000,
            "alpha_velocity": 0.7,
            "alpha_pressure": 0.3,
            "max_pressure_iter": 5000,
            "pressure_rtol": 1e-10,
            "stopping_rule": "velocity_step",
        },
        "boundaries": boundaries,
        "obstacles": [dict(BLOCK)],
        "sensors": [{"name": "downstream", "x": 2.5, "y": 0.5}],
        "thresholds": {"0.1e-6": 100.0},
    }
    if mesh is not None:
        raw["mesh"] = mesh
    return raw


def _gauge(raw: dict) -> tuple[Mesh, SimConfig, ForceGauge]:
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    return mesh, config, ForceGauge(mesh, config, "block", 1.0)


def _stretched() -> dict:
    """Breakpoints on both axes with the block's edges among them."""
    x = [(0.0, 0.2), (1.0, 0.05), (1.2, 0.05), (3.0, 0.2)]
    y = [(0.0, 0.1), (0.4, 0.04), (0.6, 0.04), (1.0, 0.1)]
    return {
        "x": {"breakpoints": [{"at": a, "spacing": s} for a, s in x]},
        "y": {"breakpoints": [{"at": a, "spacing": s} for a, s in y]},
    }


def _stretched_raw() -> dict:
    from src.config import Breakpoint, breakpoint_faces

    mesh = _stretched()
    counts = [
        len(
            breakpoint_faces(
                [Breakpoint(p["at"], p["spacing"]) for p in mesh[axis]["breakpoints"]]
            )
        )
        - 1
        for axis in ("x", "y")
    ]
    return _raw(counts[0], counts[1], freestream=True, mesh=mesh)


@pytest.mark.unit
class TestByHand:
    """Fields whose force follows from the mesh arrays, on a breakpoint mesh."""

    def test_a_uniform_pressure_puts_no_force_on_a_closed_body(self) -> None:
        mesh, _config, gauge = _gauge(_stretched_raw())
        u, v, p = allocate_fields(mesh)
        p[:] = 101325.0
        forces = gauge.forces(u, v, p)
        # The two sides cancel to the rounding of sums of p H: 1e-12 of it is far
        # above that and far below any force a gradient would put there.
        scale = 101325.0 * 0.2
        assert abs(forces.fx_pressure) < 1e-12 * scale
        assert abs(forces.fy_pressure) < 1e-12 * scale

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_a_linear_pressure_pushes_down_its_gradient_from_half_a_cell_out(
        self, axis: str
    ) -> None:
        """p = a s: the force is -a H (W + half the two outside cells) along s.

        The gauge reads each face's pressure at the centre of the cell beside
        it, half that cell's width out, so the effective length between the
        two faces is the body's plus half of each outside cell. Catches: a
        face's sign flipped, a face length from the wrong axis.
        """
        mesh, _config, gauge = _gauge(_stretched_raw())
        u, v, p = allocate_fields(mesh)
        a = 3.0
        if axis == "x":
            p[:] = a * mesh.xc[None, :]
            i0 = int(np.searchsorted(mesh.x, 1.0))
            i1 = int(np.searchsorted(mesh.x, 1.2))
            span = 0.2 + 0.5 * (mesh.dx_cell[i0 - 1] + mesh.dx_cell[i1])
            expected = -a * 0.2 * span
            got, other = (
                gauge.forces(u, v, p).fx_pressure,
                gauge.forces(u, v, p).fy_pressure,
            )
        else:
            p[:] = a * mesh.yc[:, None]
            j0 = int(np.searchsorted(mesh.y, 0.4))
            j1 = int(np.searchsorted(mesh.y, 0.6))
            span = 0.2 + 0.5 * (mesh.dy_cell[j0 - 1] + mesh.dy_cell[j1])
            expected = -a * 0.2 * span
            got, other = (
                gauge.forces(u, v, p).fy_pressure,
                gauge.forces(u, v, p).fx_pressure,
            )
        assert got == pytest.approx(expected, rel=1e-12)
        assert other == pytest.approx(0.0, abs=1e-12)

    def test_a_uniform_stream_shears_the_top_and_bottom_by_the_wall_stencil(
        self,
    ) -> None:
        """u = 1 on every unknown, v = 0: mu u / (dy / 2) over each face's width.

        The rows just above and below the block, the corner faces' whole
        widths included (the predictor's stencil takes a corner face as a
        wall over its whole span). Catches: the half distance taken as a
        whole cell, the width taken as the cell's not the control volume's,
        a corner face left out.
        """
        mesh, config, gauge = _gauge(_stretched_raw())
        u, v, p = allocate_fields(mesh)
        u[:, 1:-1] = 1.0
        i0 = int(np.searchsorted(mesh.x, 1.0))
        i1 = int(np.searchsorted(mesh.x, 1.2))
        j0 = int(np.searchsorted(mesh.y, 0.4))
        j1 = int(np.searchsorted(mesh.y, 0.6))
        widths = mesh.dx_face[i0 : i1 + 1]
        below = MU * widths.sum() / (0.5 * mesh.dy_cell[j0 - 1])
        above = MU * widths.sum() / (0.5 * mesh.dy_cell[j1])
        solid = mesh.cell_type == SOLID
        u[:, 1:-1][solid[:, :-1] | solid[:, 1:]] = 0.0
        forces = gauge.forces(u, v, p)
        assert forces.fx_viscous == pytest.approx(below + above, rel=1e-12)
        assert forces.fy_viscous == 0.0 and config.mu == MU

    def test_a_uniform_cross_stream_shears_the_sides(self) -> None:
        """v = 1 on every unknown: the columns left and right of the block."""
        mesh, _config, gauge = _gauge(_stretched_raw())
        u, v, p = allocate_fields(mesh)
        v[1:-1, :] = 1.0
        solid = mesh.cell_type == SOLID
        v[1:-1, :][solid[:-1, :] | solid[1:, :]] = 0.0
        i0 = int(np.searchsorted(mesh.x, 1.0))
        i1 = int(np.searchsorted(mesh.x, 1.2))
        j0 = int(np.searchsorted(mesh.y, 0.4))
        j1 = int(np.searchsorted(mesh.y, 0.6))
        heights = mesh.dy_face[j0 : j1 + 1]
        left = MU * heights.sum() / (0.5 * mesh.dx_cell[i0 - 1])
        right = MU * heights.sum() / (0.5 * mesh.dx_cell[i1])
        forces = gauge.forces(u, v, p)
        assert forces.fy_viscous == pytest.approx(left + right, rel=1e-12)
        assert forces.fx_viscous == 0.0

    def test_the_coefficients_divide_by_half_rho_u_squared_d(self) -> None:
        mesh, config, _unused = _gauge(_stretched_raw())
        gauge = ForceGauge(mesh, config, "block", 2.0)
        forces = ObstacleForces(0.3, 0.1, -0.05, 0.01)
        c_d, c_l = gauge.coefficients(forces)
        assert gauge.height == pytest.approx(0.2, rel=1e-15)
        assert c_d == pytest.approx(0.4 / (0.5 * 1.0 * 4.0 * 0.2), rel=1e-15)
        assert c_l == pytest.approx(-0.04 / (0.5 * 1.0 * 4.0 * 0.2), rel=1e-15)

    @pytest.mark.parametrize(
        ("name", "velocity", "error"),
        [
            ("missing", 1.0, ValueError),
            ("block", 0.0, ValueError),
            ("block", -1.0, ValueError),
            ("block", math.inf, ValueError),
            ("block", True, TypeError),
            ("block", "1", TypeError),
        ],
    )
    def test_a_gauge_that_cannot_be_is_refused(
        self, name: str, velocity: object, error: type
    ) -> None:
        config = SimConfig.from_dict(_stretched_raw())
        with pytest.raises(error):
            ForceGauge(Mesh(config), config, name, velocity)  # type: ignore[arg-type]


def _steady(raw: dict) -> tuple[Mesh, SimConfig, StaggeredSolver]:
    config = SimConfig.from_dict(raw)
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    solver.solve_steady()
    assert solver.stop_reason == "velocity_step_below_tol"
    return mesh, config, solver


def _balance(
    mesh: Mesh, config: SimConfig, u: np.ndarray, p: np.ndarray, wall_u: float
) -> float:
    """The x-force on the body from the momentum balance between x = 0.5 and 2.5.

    The momentum flux rho u^2, the pressure (the mean of the two cells
    beside the plane) and the normal viscous stress 2 mu du/dx (centred)
    through the two planes of u faces, less the shear of the lateral
    boundaries between them by the wall stencil, the boundary moving at
    ``wall_u``.
    """
    rho, mu, dy = config.rho, config.mu, mesh.dy_cell
    a = int(np.argmin(np.abs(mesh.x - 0.5)))
    b = int(np.argmin(np.abs(mesh.x - 2.5)))

    def plane(k: int) -> float:
        pk = 0.5 * (p[:, k - 1] + p[:, k])
        dudx = (u[:, k + 1] - u[:, k - 1]) / (mesh.x[k + 1] - mesh.x[k - 1])
        return float(np.sum((rho * u[:, k] ** 2 + pk - 2.0 * mu * dudx) * dy))

    widths = mesh.dx_face[a : b + 1].copy()
    widths[0] = 0.5 * mesh.dx_cell[a]
    widths[-1] = 0.5 * mesh.dx_cell[b - 1]
    walls = sum(
        float(np.sum(mu * (u[row, a : b + 1] - wall_u) / (0.5 * dy[row]) * widths))
        for row in (0, -1)
    )
    return plane(a) - plane(b) - walls


@pytest.mark.integration
class TestSteadyFlows:
    """The gauge on converged steady flows past the block at Re 4."""

    @pytest.mark.parametrize("freestream", [False, True])
    def test_the_lift_on_a_symmetric_flow_is_zero_to_rounding(
        self, freestream: bool
    ) -> None:
        """The block on the channel's axis; the drag sets the scale of rounding.

        Catches: the top or bottom face's sign flipped, a side's shear sign.
        """
        mesh, config, solver = _steady(_raw(60, 20, freestream))
        state = solver.flow_state
        forces = ForceGauge(mesh, config, "block", 1.0).forces(
            state.u, state.v, state.p
        )
        assert forces.drag > 0.5
        assert abs(forces.lift) < 1e-12 * forces.drag
        assert abs(forces.fy_pressure) < 1e-12 * forces.drag

    def test_the_drag_meets_the_momentum_balance_within_its_measured_gap(self) -> None:
        """Freestream laterals, 60x20 and 120x40: within 7% and 5%, and closing.

        Catches: the left or right face's sign flipped, the shear's width or
        distance wrong, a face left out.
        """
        gaps = []
        for nx, ny, tolerance in ((60, 20, 0.07), (120, 40, 0.05)):
            mesh, config, solver = _steady(_raw(nx, ny, freestream=True))
            state = solver.flow_state
            drag = (
                ForceGauge(mesh, config, "block", 1.0)
                .forces(state.u, state.v, state.p)
                .drag
            )
            gap = (
                abs(_balance(mesh, config, state.u, state.p, wall_u=1.0) - drag) / drag
            )
            assert gap < tolerance, (nx, gap)
            gaps.append(gap)
        assert gaps[1] < gaps[0]


def _sine(
    st: float, periods: float, dt: float = 0.05, phase: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """A lift of Strouhal number st at D = U = 1, PERIODS periods long."""
    t = np.arange(0.0, periods / st, dt)
    return t, 0.4 * np.sin(2.0 * math.pi * st * t + phase)


@pytest.mark.unit
class TestStrouhal:
    """The period from the last ten periods' zero crossings, the spectrum beside it."""

    def test_a_sine_reads_its_own_number(self) -> None:
        """Interpolated crossings place the period to well inside 1e-4.

        Catches: the crossings not interpolated (they would read to a step,
        about 0.4% of the period here), the period counted over nine.
        """
        t, lift = _sine(0.146, 30.0)
        reading = strouhal(t, lift, 1.0, 1.0)
        assert reading.strouhal == pytest.approx(0.146, rel=1e-4)
        assert reading.crossings.size == STROUHAL_PERIODS + 1
        assert reading.agrees
        assert abs(reading.spectral_strouhal - 0.146) <= reading.resolution

    def test_it_reads_the_last_ten_periods_only(self) -> None:
        """A signal whose frequency changes: the early part does not count."""
        t1, lift1 = _sine(0.12, 15.0)
        t2, lift2 = _sine(0.146, 15.0)
        t = np.concatenate([t1, t1[-1] + 0.05 + t2])
        lift = np.concatenate([lift1, lift2])
        reading = strouhal(t, lift, 1.0, 1.0)
        assert reading.strouhal == pytest.approx(0.146, rel=1e-3)

    def test_d_and_u_scale_it(self) -> None:
        t, lift = _sine(0.146, 30.0)
        assert strouhal(t, lift, 2.0, 4.0).strouhal == pytest.approx(0.073, rel=1e-4)

    def test_fewer_than_ten_periods_are_refused(self) -> None:
        t, lift = _sine(0.146, 9.5)
        with pytest.raises(ValueError, match="needs the last 10"):
            strouhal(t, lift, 1.0, 1.0)

    def test_a_steady_lift_is_refused(self) -> None:
        t = np.arange(0.0, 100.0, 0.05)
        with pytest.raises(ValueError, match="whole periods"):
            strouhal(t, np.full_like(t, 1e-6), 1.0, 1.0)

    @pytest.mark.parametrize(
        "label",
        ["shapes", "two-d", "nan", "uneven", "decreasing", "height", "velocity"],
    )
    def test_malformed_input_is_refused(self, label: str) -> None:
        t, lift = _sine(0.146, 30.0)
        args = {"t": t, "lift": lift, "height": 1.0, "velocity": 1.0}
        if label == "shapes":
            args["lift"] = lift[:-1]
        elif label == "two-d":
            args["t"], args["lift"] = t[None, :], lift[None, :]
        elif label == "nan":
            args["lift"] = np.where(t > 10.0, np.nan, lift)
        elif label == "uneven":
            args["t"] = t * (1.0 + 1e-3 * np.sin(t))
        elif label == "decreasing":
            args["t"] = t[::-1]
        elif label == "height":
            args["height"] = 0.0
        else:
            args["velocity"] = -1.0
        with pytest.raises(ValueError):
            strouhal(**args)
