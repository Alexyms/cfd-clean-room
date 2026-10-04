"""Unit tests for the transport solver (ADR-011 B, C, D, F).

Every test that guards a trap carries its control inside it where the control
is data: a face field with one flux sign flipped, a step above stable_dt, a
mutable array handed to FieldHistory. Where the trap is a line of the solver
(the ratio guard, the sum in stable_dt, the scheme selector) the test asserts
the behaviour that line produces, and the mutation log under
results/builder32/ records that removing the line fails the test.

Meshes are small and, where it matters, non-square and stretched, so an
index that works only on a square uniform grid shows.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from src.boundary_concentration import (
    SURFACE_CEILING,
    SURFACE_FLOOR,
    SURFACE_WALL,
    ConcentrationBoundary,
    ConcentrationFaces,
)
from src.boundary_registry import BoundaryRegistry
from src.config import SimConfig
from src.mesh import SOLID, Mesh
from src.particles import ParticlePhysics
from src.solver_transport import (
    FieldHistory,
    MassBudget,
    TransportSolver,
    limited_face_values,
)
from src.staggered import FaceVelocities, u_shape, v_shape
from validation.metrics import centroid
from validation.transport_cases import (
    FixedConditions,
    ScalarPhysics,
    conditions_with,
    gaussian_cell_averages,
    rotation_face_field,
    transport_config,
    uniform_face_field,
    zero_conditions,
)

INERT = ScalarPhysics(settling=0.0, diffusion=0.0)


def _solver(
    config: SimConfig, physics: ScalarPhysics = INERT, **conditions: np.ndarray
) -> tuple[Mesh, TransportSolver]:
    """A solver over a mesh with fixed conditions, zeros unless overridden."""
    mesh = Mesh(config)
    faces = conditions_with(mesh, **conditions)
    return mesh, TransportSolver(mesh, config, physics, FixedConditions(faces))


def _random_field(mesh: Mesh, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 1.5, size=(mesh.yc.shape[0], mesh.xc.shape[0]))


def _random_faces(mesh: Mesh, seed: int, scale: float = 0.3) -> FaceVelocities:
    rng = np.random.default_rng(seed)
    return FaceVelocities.copy_of(
        rng.uniform(-scale, scale, size=u_shape(mesh)),
        rng.uniform(-scale, scale, size=v_shape(mesh)),
    )


# ---------------------------------------------------------------------------
# The limiter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLimiter:
    def test_umist_is_quick_between_r_one_and_five_on_uniform_data(self) -> None:
        """QUICK's own psi is (3 + r) / 4; UMIST returns it where r in [1, 5]."""
        for r in (1.0, 2.0, 3.0, 5.0):
            c_c, c_d = 1.0, 2.0
            c_up = c_c - r * (c_d - c_c)
            quick = (6.0 * c_c + 3.0 * c_d - c_up) / 8.0
            face = limited_face_values(
                np.array([c_up]), np.array([c_c]), np.array([c_d]), np.array([quick])
            )
            assert face[0] == pytest.approx(quick, rel=1e-14)

    def test_an_extremum_falls_back_to_upwind(self) -> None:
        """r <= 0 at a local extremum: the face value is the upstream cell's."""
        for c_up in (1.0, 1.5):
            face = limited_face_values(
                np.array([c_up]), np.array([1.0]), np.array([2.0]), np.array([1.2])
            )
            assert face[0] == 1.0

    def test_a_steep_upstream_gradient_is_clamped_at_the_downstream_value(
        self,
    ) -> None:
        """psi caps at 2, so the face value never passes the downstream cell."""
        face = limited_face_values(
            np.array([-20.0]), np.array([1.0]), np.array([2.0]), np.array([5.0])
        )
        assert face[0] == 2.0

    def test_the_guard_returns_the_upstream_value_where_the_difference_is_zero(
        self,
    ) -> None:
        """c_d == c_c makes r undefined; the value is c_c, finite, exactly."""
        quick = np.array([0.9, 1.1, 1.0])
        face = limited_face_values(
            np.array([0.0, 2.0, 1.0]), np.ones(3), np.ones(3), quick
        )
        assert np.array_equal(face, np.ones(3))

    def test_the_face_value_stays_between_the_two_cells(self) -> None:
        rng = np.random.default_rng(3)
        c_up, c_c, c_d, quick = (rng.normal(size=1000) for _ in range(4))
        face = limited_face_values(c_up, c_c, c_d, quick)
        lo, hi = np.minimum(c_c, c_d), np.maximum(c_c, c_d)
        assert np.all(face >= lo - 1e-15) and np.all(face <= hi + 1e-15)


# ---------------------------------------------------------------------------
# Construction and argument checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstruction:
    def test_a_configuration_without_a_transport_section_is_refused(self) -> None:
        raw = {
            "domain": {"width": 1.0, "height": 0.5, "nx": 4, "ny": 2},
            "fluid": {"density": 1.2, "viscosity": 1.8e-5, "temperature": 293.0},
            "particles": {
                "density": 1000.0,
                "sizes": [5.0e-6],
                "mean_free_path": 67.0e-9,
                "boundary_layer_thickness": 1.0e-3,
                "hepa_reference": {"diameters": [5.0e-6], "efficiencies": [0.9]},
            },
            "solver": {
                "dt": 0.01,
                "t_end": 1.0,
                "output_interval": 1,
                "convergence_tol": 1e-6,
                "max_simple_iter": 10,
                "alpha_velocity": 0.7,
                "alpha_pressure": 0.3,
                "max_pressure_iter": 10,
                "pressure_tol": 1e-6,
            },
            "boundaries": {},
            "sensors": [],
            "thresholds": {},
        }
        config = SimConfig.from_dict(raw)
        assert config.transport is None
        mesh = Mesh(config)
        with pytest.raises(ValueError, match="transport section"):
            TransportSolver(mesh, config, INERT, FixedConditions(zero_conditions(mesh)))

    def test_conditions_of_the_wrong_shape_are_refused(self) -> None:
        config = transport_config(1.0, 0.5, 6, 3)
        other = Mesh(transport_config(1.0, 0.5, 5, 3))
        mesh = Mesh(config)
        with pytest.raises(ValueError, match="inflow_u must have shape"):
            TransportSolver(
                mesh, config, INERT, FixedConditions(zero_conditions(other))
            )

    def test_one_budget_per_class_and_none_written_before_a_step(self) -> None:
        _, solver = _solver(transport_config(1.0, 0.5, 6, 3))
        assert len(solver.budget) == 1
        budget = solver.budget[0]
        assert budget.initial is None and budget.current is None
        assert set(budget.deposited) == {"floor", "ceiling", "wall", "obstacle"}
        with pytest.raises(ValueError, match="no field yet"):
            budget.residual()
        with pytest.raises(ValueError, match="no field yet"):
            budget.relative()


@pytest.mark.unit
class TestArgumentChecks:
    def setup_method(self) -> None:
        self.config = transport_config(1.2, 0.6, 6, 3)
        self.mesh, self.solver = _solver(self.config)
        self.faces = uniform_face_field(self.mesh, 0.3, -0.1)
        self.c = _random_field(self.mesh, 1)

    def test_a_step_above_stable_dt_raises(self) -> None:
        dt = self.solver.stable_dt(self.faces, 0)
        self.solver.solve_timestep(self.c, self.faces, 0, dt)
        with pytest.raises(ValueError, match="exceeds stable_dt"):
            self.solver.solve_timestep(self.c, self.faces, 0, dt * (1.0 + 1e-9))

    def test_a_non_positive_step_raises(self) -> None:
        for dt in (0.0, -1e-3):
            with pytest.raises(ValueError, match="dt must be positive"):
                self.solver.solve_timestep(self.c, self.faces, 0, dt)

    def test_a_field_of_the_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="expected C_k of shape"):
            self.solver.solve_timestep(self.c.T.copy(), self.faces, 0, 1e-3)

    def test_faces_of_the_wrong_shape_raise(self) -> None:
        other = uniform_face_field(Mesh(transport_config(1.2, 0.6, 5, 3)), 0.3, 0.0)
        with pytest.raises(ValueError, match="faces must have shapes"):
            self.solver.stable_dt(other, 0)
        with pytest.raises(ValueError, match="faces must have shapes"):
            self.solver.solve_timestep(self.c, other, 0, 1e-3)

    def test_v_ext_of_the_wrong_shape_raises(self) -> None:
        other = uniform_face_field(Mesh(transport_config(1.2, 0.6, 5, 3)), 0.0, 0.0)
        with pytest.raises(ValueError, match="v_ext must have shapes"):
            self.solver.stable_dt(self.faces, 0, v_ext=other)

    @pytest.mark.parametrize("size_class", [True, 0.0, "0", None])
    def test_a_non_int_size_class_raises(self, size_class: object) -> None:
        with pytest.raises(TypeError, match="size_class must be an int"):
            self.solver.stable_dt(self.faces, size_class)
        with pytest.raises(TypeError, match="size_class must be an int"):
            self.solver.solve_timestep(self.c, self.faces, size_class, 1e-3)

    @pytest.mark.parametrize("size_class", [-1, 1])
    def test_a_size_class_outside_the_configuration_raises(
        self, size_class: int
    ) -> None:
        with pytest.raises(IndexError, match="out of range"):
            self.solver.stable_dt(self.faces, size_class)

    def test_a_nan_face_velocity_is_refused_before_any_arithmetic(self) -> None:
        """A diverged velocity solve can hand over NaN faces (face_velocities is
        set converged or not); stable_dt and the step both refuse them, and
        the budget stays untouched."""
        u = self.faces.u.copy()
        u[1, 2] = np.nan
        bad = FaceVelocities.copy_of(u, self.faces.v)
        with pytest.raises(ValueError, match="faces must be finite"):
            self.solver.stable_dt(bad, 0)
        with pytest.raises(ValueError, match="faces must be finite"):
            self.solver.solve_timestep(self.c, bad, 0, 1e-3)
        assert self.solver.budget[0].initial is None

    def test_a_nan_v_ext_field_or_source_is_refused(self) -> None:
        v = np.zeros(v_shape(self.mesh))
        v[1, 1] = np.inf
        with pytest.raises(ValueError, match="v_ext must be finite"):
            self.solver.stable_dt(
                self.faces,
                0,
                v_ext=FaceVelocities.copy_of(np.zeros(u_shape(self.mesh)), v),
            )
        field = self.c.copy()
        field[0, 0] = np.nan
        with pytest.raises(ValueError, match="C_k must be finite"):
            self.solver.solve_timestep(field, self.faces, 0, 1e-3)
        rate = np.zeros((3, 6))
        rate[1, 1] = np.nan
        with pytest.raises(ValueError, match="sources must be finite"):
            self.solver.solve_timestep(self.c, self.faces, 0, 1e-3, sources=rate)
        assert self.solver.budget[0].initial is None

    def test_a_bool_or_non_finite_dt_is_refused(self) -> None:
        with pytest.raises(TypeError, match="dt must be a number"):
            self.solver.solve_timestep(self.c, self.faces, 0, True)
        for dt in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="dt must be finite"):
                self.solver.solve_timestep(self.c, self.faces, 0, dt)

    def test_conditions_with_a_nan_or_the_wrong_dtype_are_refused(self) -> None:
        inflow_u = np.zeros(u_shape(self.mesh))
        inflow_u[0, 0] = np.nan
        with pytest.raises(ValueError, match="inflow_u must be finite"):
            _solver(self.config, inflow_u=inflow_u)
        deposition_v = np.zeros(v_shape(self.mesh))
        deposition_v[0, 0] = -1.0
        with pytest.raises(
            ValueError, match="deposition_v must be finite and non-negative"
        ):
            _solver(self.config, deposition_v=deposition_v)
        faces = conditions_with(self.mesh)
        wrong = ConcentrationFaces(
            inflow_u=faces.inflow_u.astype(bool),
            inflow_v=faces.inflow_v,
            deposition_u=faces.deposition_u,
            deposition_v=faces.deposition_v,
            surface_u=faces.surface_u,
            surface_v=faces.surface_v,
            settling_v=faces.settling_v,
        )
        with pytest.raises(ValueError, match="inflow_u must have dtype kind 'f'"):
            TransportSolver(self.mesh, self.config, INERT, FixedConditions(wrong))

    def test_the_input_field_is_not_modified_and_the_output_is_fresh(self) -> None:
        before = self.c.copy()
        dt = self.solver.stable_dt(self.faces, 0)
        out = self.solver.solve_timestep(self.c, self.faces, 0, dt)
        assert np.array_equal(self.c, before)
        assert out.dtype == np.float64 and out.flags["C_CONTIGUOUS"]
        assert not np.shares_memory(out, self.c)


# ---------------------------------------------------------------------------
# stable_dt
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStableDt:
    def test_it_is_the_sum_of_the_directional_rates_in_the_worst_cell(self) -> None:
        """ADR-011 B: the unsplit step is a convex combination only when the sum
        of the two Courant numbers is at most 1/2, so the bound is on the sum.
        On solid-body rotation the worst cell is a corner, where both rates
        are large; the max form would let its sum reach 2 cfl there."""
        config = transport_config(1.28, 0.96, 32, 24, cfl_number=0.5)
        mesh, solver = _solver(config)
        faces = rotation_face_field(mesh, 1.0, (0.64, 0.48))
        rate_x = np.maximum(np.abs(faces.u[:, :-1]), np.abs(faces.u[:, 1:])) / mesh.dx
        rate_y = np.maximum(np.abs(faces.v[:-1, :]), np.abs(faces.v[1:, :])) / mesh.dy
        dt = solver.stable_dt(faces, 0)
        assert dt == pytest.approx(0.5 / (rate_x + rate_y).max(), rel=1e-14)
        dt_max_form = 0.5 / np.maximum(rate_x, rate_y).max()
        assert dt_max_form > dt
        courant_sum = dt_max_form * (rate_x + rate_y)
        assert courant_sum.max() > 0.5
        corner = np.unravel_index(np.argmax(rate_x + rate_y), rate_x.shape)
        assert corner in {(0, 0), (0, 31), (23, 0), (23, 31)}

    def test_it_scales_with_cfl_number_and_is_infinite_at_rest(self) -> None:
        mesh, a = _solver(transport_config(1.0, 0.5, 8, 4, cfl_number=0.1))
        _, b = _solver(transport_config(1.0, 0.5, 8, 4, cfl_number=0.4))
        faces = uniform_face_field(mesh, 0.2, 0.0)
        assert b.stable_dt(faces, 0) == pytest.approx(4.0 * a.stable_dt(faces, 0))
        assert a.stable_dt(uniform_face_field(mesh, 0.0, 0.0), 0) == float("inf")

    def test_v_ext_enters_the_bound(self) -> None:
        mesh, solver = _solver(transport_config(1.0, 0.5, 8, 4, cfl_number=0.2))
        still = uniform_face_field(mesh, 0.0, 0.0)
        drift = uniform_face_field(mesh, 0.0, -0.05)
        # A uniform drift gives every cell the same rate, so this checks that
        # v_ext enters the bound; which faces carry it is TestVExt's.
        assert solver.stable_dt(still, 0, v_ext=drift) == pytest.approx(
            0.2 * mesh.dy / 0.05
        )

    def test_faces_of_solid_cells_do_not_set_the_bound(self) -> None:
        """A field with a large velocity on an obstacle face is masked there."""
        config = transport_config(
            1.0,
            0.5,
            10,
            5,
            obstacles=[
                {
                    "name": "b",
                    "x_start": 0.4,
                    "x_end": 0.6,
                    "y_start": 0.2,
                    "y_end": 0.3,
                }
            ],
        )
        mesh, solver = _solver(config)
        solid = mesh.cell_type == SOLID
        assert solid.sum() == 2
        u = np.full(u_shape(mesh), 0.1)
        j, i = np.argwhere(solid)[0]
        u[j, i] = 100.0  # the west face of a SOLID cell
        faces = FaceVelocities.copy_of(u, np.zeros(v_shape(mesh)))
        assert solver.stable_dt(faces, 0) == pytest.approx(0.1 * mesh.dx / 0.1)


# ---------------------------------------------------------------------------
# Advection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdvection:
    def test_a_uniform_field_on_a_divergence_free_field_stays_uniform(self) -> None:
        """Rotation with every inlet carrying the uniform value: exact over 100
        steps (REQ-T11's mechanism). The control flips one interior face's
        sign, which makes two cells sources and sinks, and the field drifts."""
        config = transport_config(1.28, 0.96, 32, 24)
        mesh = Mesh(config)
        ones_u, ones_v = np.ones(u_shape(mesh)), np.ones(v_shape(mesh))
        solver = TransportSolver(
            mesh,
            config,
            INERT,
            FixedConditions(conditions_with(mesh, inflow_u=ones_u, inflow_v=ones_v)),
        )
        faces = rotation_face_field(mesh, 1.0, (0.64, 0.48))
        dt = solver.stable_dt(faces, 0)
        c = np.ones((24, 32))
        for _ in range(100):
            c = solver.solve_timestep(c, faces, 0, dt)
        assert np.abs(c - 1.0).max() <= 1e-13

        u = faces.u.copy()
        u[12, 16] = -u[12, 16]
        flipped = FaceVelocities.copy_of(u, faces.v)
        c = np.ones((24, 32))
        for _ in range(100):
            c = solver.solve_timestep(c, flipped, 0, dt)
        assert np.abs(c - 1.0).max() > 1e-3

    def test_upwind_by_configuration_gives_the_upwind_face_value(self) -> None:
        """One step on a random field against the upwind update written out:
        the face value is the upstream cell's, a clean inflow carries zero,
        an outflow carries the edge cell. The limited scheme must differ on
        the same data, or the selector could be ignored unseen."""
        config_up = transport_config(1.2, 0.6, 6, 3, scheme="upwind")
        config_umist = transport_config(1.2, 0.6, 6, 3, scheme="umist")
        mesh, upwind = _solver(config_up)
        _, umist = _solver(config_umist)
        faces = _random_faces(mesh, 7)
        c = _random_field(mesh, 8)
        dt = upwind.stable_dt(faces, 0)

        flux_u = faces.u * mesh.dy
        flux_v = faces.v * mesh.dx
        cx = np.pad(c, ((0, 0), (1, 1)))  # zero beyond the domain: clean inflow
        cy = np.pad(c, ((1, 1), (0, 0)))
        face_u = np.where(flux_u > 0.0, cx[:, :-1], cx[:, 1:])
        face_v = np.where(flux_v > 0.0, cy[:-1, :], cy[1:, :])
        fx, fy = flux_u * face_u, flux_v * face_v
        expected = c - dt / (mesh.dx * mesh.dy) * (
            fx[:, 1:] - fx[:, :-1] + fy[1:, :] - fy[:-1, :]
        )
        got = upwind.solve_timestep(c, faces, 0, dt)
        assert np.allclose(got, expected, rtol=0.0, atol=1e-15)
        assert not np.allclose(
            umist.solve_timestep(c, faces, 0, dt), expected, atol=1e-6
        )

    def test_an_inlet_carries_its_concentration_and_the_budget_books_it(self) -> None:
        """Left inlet carrying 2.0 into a zero field: after one step the inlet
        cells hold exactly u dt / dx times 2, and inflow is u dy dt times 2
        per face. Nothing beyond the first column changes because the field
        there is uniform, whatever the far node holds; the far node itself
        is checked by test_the_inlet_value_is_the_far_node_of_the_first_interior_face."""
        config = transport_config(1.0, 0.5, 8, 4)
        mesh = Mesh(config)
        inflow_u = np.zeros(u_shape(mesh))
        inflow_u[:, 0] = 2.0
        mesh, solver = _solver(config, inflow_u=inflow_u)
        faces = uniform_face_field(mesh, 0.1, 0.0)
        dt = solver.stable_dt(faces, 0)
        c = solver.solve_timestep(np.zeros((4, 8)), faces, 0, dt)
        assert np.allclose(c[:, 0], 2.0 * 0.1 * dt / mesh.dx)
        assert np.all(c[:, 1:] == 0.0)
        budget = solver.budget[0]
        assert budget.inflow == pytest.approx(4 * 0.1 * mesh.dy * dt * 2.0)
        assert budget.outflow == 0.0
        assert budget.residual() == pytest.approx(0.0, abs=1e-18)

    def test_an_outflow_face_carries_the_upwind_cell_and_the_budget_books_it(
        self,
    ) -> None:
        config = transport_config(1.0, 0.5, 8, 4)
        mesh, solver = _solver(config)
        faces = uniform_face_field(mesh, 0.0, -0.1)
        c = _random_field(mesh, 2)
        dt = solver.stable_dt(faces, 0)
        out = solver.solve_timestep(c, faces, 0, dt)
        budget = solver.budget[0]
        assert budget.outflow == pytest.approx(0.1 * mesh.dx * dt * c[0, :].sum())
        assert budget.inflow == 0.0
        assert budget.current == pytest.approx(MassBudget.in_domain(out, mesh))
        assert abs(budget.relative()) < 1e-14

    def test_v_ext_drifts_a_pulse_at_its_velocity(self) -> None:
        """A downward v_ext on still air moves the centroid by w T (REQ-T06)."""
        config = transport_config(1.0, 1.0, 20, 40)
        mesh, solver = _solver(config)
        still = uniform_face_field(mesh, 0.0, 0.0)
        drift = uniform_face_field(mesh, 0.0, -0.05)
        c = gaussian_cell_averages(mesh.x, mesh.y, (0.5, 0.7), 0.08)
        x0, y0 = centroid(c, mesh)
        dt = solver.stable_dt(still, 0, v_ext=drift)
        steps = 40
        for _ in range(steps):
            c = solver.solve_timestep(c, still, 0, dt, v_ext=drift)
        x1, y1 = centroid(c, mesh)
        assert x1 == pytest.approx(x0, abs=1e-12)
        assert y1 - y0 == pytest.approx(-0.05 * steps * dt, rel=1e-3)
        assert solver.budget[0].relative() == pytest.approx(0.0, abs=1e-13)

    def test_the_inlet_value_is_the_far_node_of_the_first_interior_face(
        self,
    ) -> None:
        """Leonard's boundary form (ADR-011 B): the quadratic through the inlet
        value at the face and the first two cells. A linear field with the
        inlet carrying 0.8 puts r = 2 at the first interior face, where the
        limiter keeps part of the quadratic, so the face value depends on
        the far node: 1.0875 with the inlet value, 1.1 with a zero ghost
        (the prompt's named trap), 1.0 with the edge cell. The first cell's
        update is checked against the value written out by hand."""
        config = transport_config(1.0, 0.5, 8, 4)
        mesh = Mesh(config)
        inflow_u = np.zeros(u_shape(mesh))
        carried = 0.8
        inflow_u[:, 0] = carried
        mesh, solver = _solver(config, inflow_u=inflow_u)
        faces = uniform_face_field(mesh, 0.1, 0.0)
        c = np.ones((4, 1)) * (1.0 + 0.1 * np.arange(8))[None, :]
        dt = solver.stable_dt(faces, 0)
        courant = 0.1 * dt / mesh.dx
        c0, c1 = c[0, 0], c[0, 1]
        # The quadratic through (x[0], carried), (xc[0], c0), (xc[1], c1) at x[1]
        # on a uniform mesh is c0 + (c1 - carried) / 3; then the UMIST clamp.
        quick = c0 + (c1 - carried) / 3.0
        r = (c0 - carried) / (c1 - c0)
        psi = max(
            0.0,
            min(2.0 * r, (1.0 + 3.0 * r) / 4.0, 2.0 * (quick - c0) / (c1 - c0), 2.0),
        )
        face_1 = c0 + 0.5 * psi * (c1 - c0)
        assert face_1 == pytest.approx(1.0875)
        out = solver.solve_timestep(c, faces, 0, dt)
        expected = c0 - courant * (face_1 - carried)
        assert np.allclose(out[:, 0], expected, rtol=1e-12, atol=0.0)
        # A zero ghost or the edge cell as the far node moves the face value
        # by at least 0.0125, which the first cell shows at courant times that.
        assert (
            abs(out[0, 0] - (c0 - courant * (1.1 - carried))) > 0.5 * courant * 0.0125
        )
        assert (
            abs(out[0, 0] - (c0 - courant * (1.0 - carried))) > 0.5 * courant * 0.0125
        )

    def test_a_far_node_inside_an_obstacle_reads_as_the_upstream_value(self) -> None:
        """One cell past an obstacle in a uniform flow, on a field linear in x:
        the far node of the face east of that cell is the SOLID cell. Read as
        the upstream value it makes r = 0 and the face upwind, so the cell
        loses one Courant number of its own value; read as the obstacle's
        zero it makes r = 15, the clamp returns the downstream value, and
        the cell would lose the downstream value instead."""
        config = transport_config(
            1.0,
            0.5,
            10,
            5,
            obstacles=[
                {
                    "name": "b",
                    "x_start": 0.4,
                    "x_end": 0.5,
                    "y_start": 0.2,
                    "y_end": 0.3,
                }
            ],
        )
        mesh, solver = _solver(config)
        solid = mesh.cell_type == SOLID
        assert [tuple(cell) for cell in np.argwhere(solid)] == [(2, 4)]
        faces = uniform_face_field(mesh, 0.1, 0.0)
        c = np.ones((5, 1)) * (1.0 + 0.1 * np.arange(10))[None, :]
        dt = solver.stable_dt(faces, 0)
        courant = 0.1 * dt / mesh.dx
        out = solver.solve_timestep(c, faces, 0, dt)
        # West face of (2, 5) is the obstacle face: no flux. East face: upwind.
        assert out[2, 5] == pytest.approx(c[2, 5] * (1.0 - courant), rel=1e-12)
        assert abs(out[2, 5] - (c[2, 5] - courant * c[2, 6])) > 0.5 * courant * 0.1
        # The row above the obstacle has no SOLID far node and is the plain scheme.
        assert out[3, 5] != pytest.approx(c[3, 5] * (1.0 - courant), rel=1e-6)

    def test_a_domain_face_behind_an_edge_obstacle_carries_no_flux(self) -> None:
        """An obstacle on the bottom edge under an inlet whose faces all carry
        3.0, the SOLID column's face included: data the boundary layer never
        produces, so the solver's own mask is what is tested. The face behind
        the SOLID cell carries nothing: inflow is booked for the nine live
        faces, the SOLID cell and the cell above it stay empty, and the
        budget closes. With the mask removed the tenth face's mass is booked
        in and then zeroed with the SOLID cell, a tenth of the supply lost."""
        config = transport_config(
            1.0,
            0.5,
            10,
            5,
            obstacles=[
                {
                    "name": "b",
                    "x_start": 0.4,
                    "x_end": 0.5,
                    "y_start": 0.0,
                    "y_end": 0.1,
                }
            ],
        )
        mesh = Mesh(config)
        assert mesh.cell_type[0, 4] == SOLID
        inflow_v = np.zeros(v_shape(mesh))
        inflow_v[0, :] = 3.0
        mesh, solver = _solver(config, inflow_v=inflow_v)
        faces = uniform_face_field(mesh, 0.0, 0.2)
        dt = solver.stable_dt(faces, 0)
        courant = 0.2 * dt / mesh.dy
        out = solver.solve_timestep(np.zeros((5, 10)), faces, 0, dt)
        budget = solver.budget[0]
        assert budget.inflow == pytest.approx(9 * 0.2 * mesh.dx * dt * 3.0, rel=1e-13)
        assert out[0, 4] == 0.0 and out[1, 4] == 0.0
        live = [i for i in range(10) if i != 4]
        assert np.allclose(out[0, live], 3.0 * courant, rtol=1e-13, atol=0.0)
        assert abs(budget.relative()) < 1e-14


# ---------------------------------------------------------------------------
# v_ext (REQ-T06): both components, interior faces only
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVExt:
    def test_the_horizontal_component_drifts_a_pulse_at_its_velocity(self) -> None:
        """A rightward v_ext on still air moves the centroid by w T in x and
        nothing in y."""
        config = transport_config(1.0, 1.0, 40, 20)
        mesh, solver = _solver(config)
        still = uniform_face_field(mesh, 0.0, 0.0)
        drift = uniform_face_field(mesh, 0.05, 0.0)
        c = gaussian_cell_averages(mesh.x, mesh.y, (0.3, 0.5), 0.08)
        x0, y0 = centroid(c, mesh)
        dt = solver.stable_dt(still, 0, v_ext=drift)
        steps = 40
        for _ in range(steps):
            c = solver.solve_timestep(c, still, 0, dt, v_ext=drift)
        x1, y1 = centroid(c, mesh)
        assert x1 - x0 == pytest.approx(0.05 * steps * dt, rel=1e-3)
        assert y1 == pytest.approx(y0, abs=1e-12)

    def test_it_acts_on_interior_faces_only(self) -> None:
        """A drift toward the floor and the right wall on a field of ones in a
        closed box: mass piles into the floor row and the right column and
        none leaves, since the domain faces carry no drift. With a domain
        face carrying it, the floor or the right wall would book an outflow."""
        config = transport_config(1.0, 0.5, 8, 4)
        mesh, solver = _solver(config)
        still = uniform_face_field(mesh, 0.0, 0.0)
        drift = uniform_face_field(mesh, 0.03, -0.05)
        dt = solver.stable_dt(still, 0, v_ext=drift)
        c = np.ones((4, 8))
        for _ in range(5):
            c = solver.solve_timestep(c, still, 0, dt, v_ext=drift)
        budget = solver.budget[0]
        assert budget.outflow == 0.0 and budget.inflow == 0.0
        assert budget.current == pytest.approx(budget.initial, rel=1e-14)
        # The corner the drift points at collects from both faces, the
        # opposite corner loses through both, and the column and row sums
        # shift with each component.
        assert c[0, -1] > 1.0 and c[-1, 0] < 1.0
        assert c.sum(axis=0)[-1] > c.sum(axis=0)[0]
        assert c.sum(axis=1)[0] > c.sum(axis=1)[-1]
        assert c.min() >= 0.0


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDiffusion:
    def test_one_implicit_step_matches_a_direct_solve_on_a_stretched_mesh(
        self,
    ) -> None:
        """Backward Euler diffusion assembled densely in the test from the
        mesh's centre-to-centre distances, against the solver's Jacobi
        result: the conductances must use dx_face and dy_face, not dx."""
        config = transport_config(
            1.0,
            0.6,
            7,
            5,
            diffusion_tol=1e-14,
            mesh={"x": {"stretch_ratio": 1.3}, "y": {"stretch_ratio": 1.2}},
        )
        mesh = Mesh(config)
        assert not mesh.is_uniform
        physics = ScalarPhysics(settling=0.0, diffusion=2.0e-3)
        solver = TransportSolver(
            mesh, config, physics, FixedConditions(zero_conditions(mesh))
        )
        c = _random_field(mesh, 5)
        dt = 0.5
        got = solver.solve_timestep(c, uniform_face_field(mesh, 0.0, 0.0), 0, dt)
        assert solver.diffusion_converged

        ny, nx = c.shape
        n = ny * nx
        idx = np.arange(n).reshape(ny, nx)
        a = np.zeros((n, n))
        volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        a[idx, idx] = volume / dt
        for j in range(ny):
            for i in range(nx - 1):
                g = 2.0e-3 * mesh.dy_cell[j] / mesh.dx_face[i + 1]
                p, q = idx[j, i], idx[j, i + 1]
                a[p, p] += g
                a[q, q] += g
                a[p, q] -= g
                a[q, p] -= g
        for j in range(ny - 1):
            for i in range(nx):
                g = 2.0e-3 * mesh.dx_cell[i] / mesh.dy_face[j + 1]
                p, q = idx[j, i], idx[j + 1, i]
                a[p, p] += g
                a[q, q] += g
                a[p, q] -= g
                a[q, p] -= g
        expected = np.linalg.solve(a, (volume / dt * c).ravel()).reshape(ny, nx)
        assert np.allclose(got, expected, rtol=1e-12, atol=1e-14)
        assert abs(solver.budget[0].relative()) < 1e-13

    def test_the_sweep_cap_is_honoured_reported_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two sweeps cannot reach 1e-14 on a resolved coefficient: the counters
        say so and a warning names the cap (SYSTEM.md section 4)."""
        config = transport_config(
            1.0, 0.5, 8, 4, max_diffusion_iter=2, diffusion_tol=1e-14
        )
        mesh, solver = _solver(
            config, physics=ScalarPhysics(settling=0.0, diffusion=1.0e-2)
        )
        with caplog.at_level(logging.WARNING, logger="src.solver_transport"):
            solver.solve_timestep(
                _random_field(mesh, 9), uniform_face_field(mesh, 0.0, 0.0), 0, 1.0
            )
        assert solver.last_diffusion_sweeps == 2
        assert not solver.diffusion_converged
        assert "max_diffusion_iter 2" in caplog.text

    def test_no_diffusion_takes_no_sweeps(self) -> None:
        mesh, solver = _solver(transport_config(1.0, 0.5, 8, 4))
        solver.solve_timestep(
            _random_field(mesh, 9), uniform_face_field(mesh, 0.1, 0.0), 0, 1e-3
        )
        assert solver.last_diffusion_sweeps == 0
        assert solver.diffusion_converged


# ---------------------------------------------------------------------------
# MassBudget and FieldHistory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMassBudget:
    def test_in_domain_counts_the_boundary_ring_and_not_solid_cells(self) -> None:
        config = transport_config(
            1.0,
            0.5,
            10,
            5,
            obstacles=[
                {
                    "name": "b",
                    "x_start": 0.4,
                    "x_end": 0.6,
                    "y_start": 0.15,
                    "y_end": 0.35,
                }
            ],
        )
        mesh = Mesh(config)
        c = np.ones((5, 10))
        expected = (50 - 4) * mesh.dx * mesh.dy
        assert MassBudget.in_domain(c, mesh) == pytest.approx(expected)

    def test_relative_refuses_a_budget_that_was_supplied_nothing(self) -> None:
        budget = MassBudget(initial=0.0, current=0.0)
        assert budget.residual() == 0.0
        with pytest.raises(ValueError, match="no particles were supplied"):
            budget.relative()

    def test_relative_is_the_residual_over_what_was_supplied(self) -> None:
        budget = MassBudget(
            initial=10.0, inflow=2.0, outflow=1.0, source=0.5, current=11.0
        )
        budget.deposited["floor"] = 0.25
        assert budget.residual() == pytest.approx(10.0 + 2.0 + 0.5 - 1.0 - 0.25 - 11.0)
        assert budget.relative() == pytest.approx(budget.residual() / 12.5)


@pytest.mark.unit
class TestFieldHistory:
    def test_every_must_be_a_positive_int(self) -> None:
        for bad in (True, 1.0, "2"):
            with pytest.raises(TypeError, match="every must be an int"):
                FieldHistory(bad)
        for bad in (0, -3):
            with pytest.raises(ValueError, match="every must be positive"):
                FieldHistory(bad)

    def test_it_records_on_the_interval_only(self) -> None:
        history = FieldHistory(3)
        for step in range(7):
            history.record(step, 0.1 * step, {0: np.full((2, 2), float(step))})
        assert [f[0] for f in history.frames] == [0, 3, 6]
        assert [f[1] for f in history.frames] == pytest.approx([0.0, 0.3, 0.6])

    def test_it_keeps_copies_not_references(self) -> None:
        history = FieldHistory(1)
        c = np.ones((2, 3))
        history.record(0, 0.0, {0: c})
        c[0, 0] = 5.0
        assert history.frames[0][2][0][0, 0] == 1.0
        assert not np.shares_memory(history.frames[0][2][0], c)

    def test_save_and_reload_are_bitwise(self, tmp_path: Path) -> None:
        history = FieldHistory(2)
        rng = np.random.default_rng(11)
        for step in range(5):
            history.record(
                step,
                0.25 * step,
                {0: rng.normal(size=(3, 4)), 2: rng.normal(size=(3, 4))},
            )
        path = tmp_path / "frames.npz"
        history.save(path)
        with np.load(path) as data:
            assert list(data["steps"]) == [0, 2, 4]
            assert np.array_equal(data["times"], [0.0, 0.5, 1.0])
            for k in (0, 2):
                stacked = np.stack([f[2][k] for f in history.frames])
                assert np.array_equal(data[f"C_{k}"], stacked)
            assert "C_1" not in data

    def test_save_refuses_no_frames_and_inconsistent_classes(
        self, tmp_path: Path
    ) -> None:
        empty = FieldHistory(1)
        with pytest.raises(ValueError, match="no frames"):
            empty.save(tmp_path / "x.npz")
        history = FieldHistory(1)
        history.record(0, 0.0, {0: np.zeros((2, 2))})
        history.record(1, 1.0, {1: np.zeros((2, 2))})
        with pytest.raises(ValueError, match="other classes"):
            history.save(tmp_path / "y.npz")


# ---------------------------------------------------------------------------
# Settling and deposition (ADR-011 D)
# ---------------------------------------------------------------------------


def _column(
    ny: int, settling: float, floor: float, settle_floor_face: bool = False
) -> tuple[Mesh, TransportSolver]:
    """A one-column sealed box: settling inside, a floor deposition velocity.

    ``settle_floor_face`` plants section D's trap as data: the settling
    increment marked on the floor face as well.
    """
    config = transport_config(0.2, 0.1 * ny, 1, ny, cfl_number=0.4)
    mesh = Mesh(config)
    deposition_v = np.zeros(v_shape(mesh))
    deposition_v[0, :] = floor
    surface_v = np.zeros(v_shape(mesh), dtype=np.int32)
    surface_v[0, :] = SURFACE_FLOOR
    settling_v = np.zeros(v_shape(mesh), dtype=bool)
    settling_v[1:-1, :] = True
    settling_v[0, :] = settle_floor_face
    solver = TransportSolver(
        mesh,
        config,
        ScalarPhysics(settling=settling, diffusion=0.0),
        FixedConditions(
            conditions_with(
                mesh,
                deposition_v=deposition_v,
                surface_v=surface_v,
                settling_v=settling_v,
            )
        ),
    )
    return mesh, solver


@pytest.mark.unit
class TestSettlingAndDeposition:
    def test_stable_dt_carries_the_settling_increment(self) -> None:
        mesh, solver = _column(10, settling=1e-3, floor=1e-3)
        still = uniform_face_field(mesh, 0.0, 0.0)
        assert solver.stable_dt(still, 0) == pytest.approx(0.4 * mesh.dy / 1e-3)

    def test_the_floor_removes_once_and_the_floor_row_is_stationary(self) -> None:
        """Section D's composition on a column: settling feeds the floor row at
        v_s C_0 and the floor removes v_d C_P with v_d = v_s, so the row stays
        at C_0 and the deposit is v_s C_0 W T exactly. With the increment
        planted on the floor face as well (test 30 B1), the floor face carries
        an advective outflow beside the deposition, the row falls, and the
        removal over the first step is 1.7 times the single one."""
        v_s, c_0 = 1e-3, 1e6
        mesh, solver = _column(10, settling=v_s, floor=v_s)
        still = uniform_face_field(mesh, 0.0, 0.0)
        dt = solver.stable_dt(still, 0)
        c = np.full((10, 1), c_0)
        for _ in range(5):
            c = solver.solve_timestep(c, still, 0, dt)
        budget = solver.budget[0]
        assert c[0, 0] == pytest.approx(c_0, rel=1e-14)
        assert budget.deposited["floor"] == pytest.approx(
            v_s * c_0 * mesh.dx * 5 * dt, rel=1e-13
        )
        assert budget.outflow == 0.0
        assert abs(budget.relative()) < 1e-14

        _, planted = _column(10, settling=v_s, floor=v_s, settle_floor_face=True)
        c = planted.solve_timestep(np.full((10, 1), c_0), still, 0, dt)
        removal = planted.budget[0].deposited["floor"] + planted.budget[0].outflow
        single = v_s * c_0 * mesh.dx * dt
        assert c[0, 0] < c_0
        assert planted.budget[0].outflow > 0.0
        assert removal == pytest.approx(single * (1.0 + 1.0 / 1.4), rel=1e-12)

    def test_the_deposition_sink_is_implicit_so_a_thin_cell_cannot_go_negative(
        self,
    ) -> None:
        """v_d dt / dy = 5 on the floor cell: the implicit sink leaves C_0 / 6,
        where an explicit one would leave -4 C_0 (REQ-T12). The deposit is
        booked from the new value."""
        mesh, solver = _column(4, settling=0.0, floor=1.0)
        still = uniform_face_field(mesh, 0.0, 0.0)
        c_0, dt = 100.0, 0.5
        a = 1.0 * dt / mesh.dy
        assert a == pytest.approx(5.0)
        c = solver.solve_timestep(np.full((4, 1), c_0), still, 0, dt)
        assert c[0, 0] == pytest.approx(c_0 / (1.0 + a), rel=1e-14)
        assert c.min() >= 0.0
        assert np.all(c[1:, 0] == c_0)
        budget = solver.budget[0]
        assert budget.deposited["floor"] == pytest.approx(
            1.0 * mesh.dx * dt * c[0, 0], rel=1e-14
        )
        assert abs(budget.relative()) < 1e-14


# ---------------------------------------------------------------------------
# Sources (ADR-011 C, decision 5)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSources:
    def test_a_source_in_a_solid_cell_a_negative_source_or_a_wrong_shape_raises(
        self,
    ) -> None:
        config = transport_config(
            1.0,
            0.5,
            10,
            5,
            obstacles=[
                {
                    "name": "b",
                    "x_start": 0.4,
                    "x_end": 0.6,
                    "y_start": 0.2,
                    "y_end": 0.3,
                }
            ],
        )
        mesh, solver = _solver(config)
        still = uniform_face_field(mesh, 0.0, 0.0)
        c = np.zeros((5, 10))
        in_solid = np.zeros((5, 10))
        in_solid[2, 4] = 1.0
        with pytest.raises(ValueError, match="SOLID cell"):
            solver.solve_timestep(c, still, 0, 1.0, sources=in_solid)
        negative = np.zeros((5, 10))
        negative[0, 0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            solver.solve_timestep(c, still, 0, 1.0, sources=negative)
        with pytest.raises(ValueError, match="expected sources of shape"):
            solver.solve_timestep(c, still, 0, 1.0, sources=np.zeros((10, 5)))

    def test_a_source_is_added_over_the_step_and_booked(self) -> None:
        mesh, solver = _solver(transport_config(1.0, 0.5, 8, 4))
        still = uniform_face_field(mesh, 0.0, 0.0)
        rate = np.zeros((4, 8))
        rate[1, 3] = 250.0
        c = solver.solve_timestep(np.zeros((4, 8)), still, 0, 0.2, sources=rate)
        assert c[1, 3] == pytest.approx(250.0 * 0.2)
        assert c.sum() == pytest.approx(250.0 * 0.2)
        budget = solver.budget[0]
        assert budget.source == pytest.approx(250.0 * mesh.dx * mesh.dy * 0.2)
        assert abs(budget.relative()) < 1e-14

    def test_a_source_in_a_floor_cell_deposits_in_the_same_step(self) -> None:
        """Added after advection and before the implicit solve (ADR-011 C), so
        the floor sink acts on it at once: the deposit after one step from an
        empty field is v_d A dt times s dt / (1 + v_d dt / dy)."""
        mesh, solver = _column(4, settling=0.0, floor=0.02)
        still = uniform_face_field(mesh, 0.0, 0.0)
        rate = np.zeros((4, 1))
        rate[0, 0] = 1000.0
        dt = 0.5
        c = solver.solve_timestep(np.zeros((4, 1)), still, 0, dt, sources=rate)
        a = 0.02 * dt / mesh.dy
        assert c[0, 0] == pytest.approx(1000.0 * dt / (1.0 + a), rel=1e-14)
        assert solver.budget[0].deposited["floor"] == pytest.approx(
            0.02 * mesh.dx * dt * c[0, 0], rel=1e-14
        )
        assert abs(solver.budget[0].relative()) < 1e-14


# ---------------------------------------------------------------------------
# Through the real boundary layer (integration: ConcentrationBoundary,
# BoundaryRegistry and ParticlePhysics feed the solver)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWithTheRealBoundaryLayer:
    def test_solid_cells_are_zero_after_a_step_and_carry_no_flux(self) -> None:
        """An obstacle in a uniform flow with the real ConcentrationBoundary:
        the SOLID cells are zero after the step although the input held mass
        there, and the budget still closes, so the mass went nowhere."""
        config = transport_config(
            1.0,
            0.5,
            10,
            5,
            boundaries={
                "in": {
                    "type": "velocity_inlet",
                    "location": "left",
                    "y_start": 0.0,
                    "y_end": 0.5,
                    "velocity": 0.1,
                },
                "out": {
                    "type": "pressure_outlet",
                    "location": "right",
                    "y_start": 0.0,
                    "y_end": 0.5,
                },
            },
            obstacles=[
                {
                    "name": "b",
                    "x_start": 0.4,
                    "x_end": 0.6,
                    "y_start": 0.15,
                    "y_end": 0.35,
                }
            ],
        )
        mesh = Mesh(config)
        physics = ParticlePhysics(config)
        boundary = ConcentrationBoundary(
            mesh, config, physics, BoundaryRegistry(config)
        )
        solver = TransportSolver(mesh, config, INERT, boundary)
        solid = mesh.cell_type == SOLID
        assert solid.sum() == 4
        faces = uniform_face_field(mesh, 0.1, 0.0)
        c = np.ones((5, 10))
        dt = solver.stable_dt(faces, 0)
        out = solver.solve_timestep(c, faces, 0, dt)
        assert np.all(out[solid] == 0.0)
        assert solver.budget[0].initial == pytest.approx(MassBudget.in_domain(c, mesh))
        assert abs(solver.budget[0].relative()) < 1e-14
        # The inflow face carries zero into a field of ones, so the first
        # column falls. The handed field is uniform, so it points into the
        # obstacle; those faces carry nothing, so the cell west of it keeps
        # its west inflow and loses nothing east (one Courant number gained)
        # and the cell east of it loses its east outflow and gains nothing.
        assert np.all(out[:, 0] < 1.0)
        j, i = np.argwhere(solid)[0]
        courant = 0.1 * dt / mesh.dx
        assert out[j, i - 1] == pytest.approx(1.0 + courant)
        assert out[j, i + 2] == pytest.approx(1.0 - courant)

    def test_obstacle_faces_are_booked_as_obstacle_and_domain_faces_by_surface(
        self,
    ) -> None:
        """A sealed box with an obstacle under the real ConcentrationBoundary
        and the 5 um class: after one settling step every slot equals the sum
        of v_d A dt C_P over the faces that surface owns, read off the new
        field, and the budget closes."""
        config = transport_config(
            1.0,
            0.5,
            10,
            5,
            obstacles=[
                {
                    "name": "b",
                    "x_start": 0.4,
                    "x_end": 0.6,
                    "y_start": 0.2,
                    "y_end": 0.3,
                }
            ],
        )
        mesh = Mesh(config)
        physics = ParticlePhysics(config)
        boundary = ConcentrationBoundary(
            mesh, config, physics, BoundaryRegistry(config)
        )
        solver = TransportSolver(mesh, config, physics, boundary)
        solid = mesh.cell_type == SOLID
        assert [tuple(c) for c in np.argwhere(solid)] == [(2, 4), (2, 5)]
        faces = boundary.faces_for(0)
        assert faces.surface_v[3, 4] == SURFACE_FLOOR  # top of the obstacle
        assert faces.surface_v[2, 4] == SURFACE_CEILING  # its underside
        assert (
            faces.surface_u[2, 4] == SURFACE_WALL
            and faces.surface_u[2, 6] == SURFACE_WALL
        )

        still = uniform_face_field(mesh, 0.0, 0.0)
        dt = solver.stable_dt(still, 0)
        c = solver.solve_timestep(np.full((5, 10), 1e6), still, 0, dt)
        v_floor = physics.deposition_velocity(0, "floor")
        v_ceiling = physics.deposition_velocity(0, "ceiling")
        v_wall = physics.deposition_velocity(0, "wall")
        dx, dy = mesh.dx, mesh.dy
        budget = solver.budget[0]
        expected_obstacle = dt * (
            v_floor * dx * (c[3, 4] + c[3, 5])
            + v_ceiling * dx * (c[1, 4] + c[1, 5])
            + v_wall * dy * (c[2, 3] + c[2, 6])
        )
        assert budget.deposited["obstacle"] == pytest.approx(
            expected_obstacle, rel=1e-13
        )
        assert budget.deposited["floor"] == pytest.approx(
            v_floor * dx * dt * c[0, :].sum(), rel=1e-13
        )
        assert budget.deposited["ceiling"] == pytest.approx(
            v_ceiling * dx * dt * c[-1, :].sum(), rel=1e-13
        )
        assert budget.deposited["wall"] == pytest.approx(
            v_wall * dy * dt * (c[:, 0].sum() + c[:, -1].sum()), rel=1e-13
        )
        assert np.all(c[solid] == 0.0)
        assert abs(budget.relative()) < 1e-14
        # The cell above the obstacle lost through its floor-type face and
        # gained nothing from below; the cell below it lost its settling
        # inflow and deposits to its ceiling-type face.
        assert c[3, 4] < 1e6 and c[1, 4] < 1e6

    def test_the_budget_closes_with_everything_on_over_an_arbitrary_field(
        self,
    ) -> None:
        """Inlet, outlet, walls, an obstacle, settling, deposition, diffusion and a
        source on a seeded random face field that is not divergence-free: the
        telescoping of ADR-011 F holds for any face velocities, so the residual
        is rounding after 30 steps."""
        config = transport_config(
            1.6,
            0.9,
            16,
            9,
            boundaries={
                "in": {
                    "type": "velocity_inlet",
                    "location": "left",
                    "y_start": 0.2,
                    "y_end": 0.7,
                    "velocity": 0.2,
                    "concentration": [3.0e5],
                },
                "out": {
                    "type": "pressure_outlet",
                    "location": "right",
                    "y_start": 0.1,
                    "y_end": 0.8,
                },
            },
            obstacles=[
                {
                    "name": "b",
                    "x_start": 0.6,
                    "x_end": 0.9,
                    "y_start": 0.3,
                    "y_end": 0.5,
                }
            ],
            diffusion_tol=1e-14,
        )
        mesh = Mesh(config)
        physics = ParticlePhysics(config)
        boundary = ConcentrationBoundary(
            mesh, config, physics, BoundaryRegistry(config)
        )
        solver = TransportSolver(mesh, config, physics, boundary)
        faces = _random_faces(mesh, 21, scale=0.3)
        rate = np.zeros((9, 16))
        rate[4, 12] = 2.0e4
        c = _random_field(mesh, 22) * 1e5
        dt = solver.stable_dt(faces, 0)
        for _ in range(30):
            c = solver.solve_timestep(c, faces, 0, dt, sources=rate)
        budget = solver.budget[0]
        assert budget.inflow > 0.0 and budget.outflow > 0.0 and budget.source > 0.0
        assert all(
            budget.deposited[name] > 0.0
            for name in ("floor", "ceiling", "wall", "obstacle")
        )
        assert abs(budget.relative()) < 1e-12
