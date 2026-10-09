"""Tests for the fixed-flow outlet (REQ-S18; ADR-012 D as amended 2026-10-08).

A fixed-flow outlet holds an outward normal velocity on its faces: stated, or
the equal share of the discrete inflow the stated outlets leave. These cases
use a small channel so each rule can be read off by hand: the share is formed
over the faces the mesh gives each segment, not the configured lengths, and
the channel uses powers of two where a test needs a remainder of exactly
nothing. The shipped product configuration is tested in
test_fixed_flow_product.py.
"""

import numpy as np
import pytest

from src.boundary_concentration import ConcentrationBoundary, ConcentrationFaces
from src.boundary_registry import (
    FIXED_FLOW_OUTLET,
    BoundaryRegistry,
    condition_of,
    fixed_flow_condition,
)
from src.boundary_staggered import StaggeredBoundary
from src.config import BoundarySpec, SimConfig
from src.mesh import FLUID, Mesh
from src.particles import ParticlePhysics
from src.pressure import PressureCorrector
from src.solver_transport import TransportSolver
from src.staggered import allocate_fields
from validation.transport_cases import transport_config, uniform_face_field

INLET = {
    "type": "velocity_inlet",
    "location": "left",
    "y_start": 0.0,
    "y_end": 1.0,
    "velocity": 0.1,
}


def _outlet(y0: float, y1: float, velocity: float | None = None) -> dict:
    segment = {
        "type": "fixed_flow_outlet",
        "location": "right",
        "y_start": y0,
        "y_end": y1,
    }
    if velocity is not None:
        segment["velocity"] = velocity
    return segment


def _channel(boundaries: dict, ny: int = 5, mesh: dict | None = None) -> SimConfig:
    """A 2 m by 1 m channel on 8 by ny cells with a transport section."""
    return transport_config(2.0, 1.0, 8, ny, boundaries, mesh=mesh)


def _build(config: SimConfig) -> tuple[Mesh, StaggeredBoundary]:
    mesh = Mesh(config)
    return mesh, StaggeredBoundary(mesh, config)


def _right_outflow(config: SimConfig, mesh: Mesh, u: np.ndarray) -> dict[str, float]:
    """Volumetric outflow of each fixed-flow segment on the right edge, from the written faces.

    The segment's faces are the cell centers inside its configured range,
    derived here from the range and ``yc`` directly.
    """
    out = {}
    for name, spec in config.boundaries.items():
        if spec.type != FIXED_FLOW_OUTLET:
            continue
        mask = (mesh.yc >= spec.y_start) & (mesh.yc <= spec.y_end)
        out[name] = float(np.sum(u[mask, -1] * mesh.dy_cell[mask]))
    return out


@pytest.mark.unit
class TestConfiguration:
    """The segment keys and the decision 4 rules that need no mesh."""

    def test_a_fixed_flow_outlet_without_a_velocity_is_valid(self) -> None:
        config = _channel({"inlet": INLET, "drain": _outlet(0.0, 1.0)})
        spec = config.boundaries["drain"]
        assert spec.type == FIXED_FLOW_OUTLET
        assert spec.velocity is None

    def test_a_stated_velocity_is_read_as_a_float(self) -> None:
        config = _channel(
            {"inlet": INLET, "drain": _outlet(0.0, 0.5), "rest": _outlet(0.5, 1.0, 1)}
        )
        assert config.boundaries["rest"].velocity == 1.0
        assert isinstance(config.boundaries["rest"].velocity, float)

    def test_a_stated_velocity_is_valid_beside_a_pressure_outlet(self) -> None:
        pressure = {"type": "pressure_outlet", "location": "right"} | {
            "y_start": 0.5,
            "y_end": 1.0,
        }
        config = _channel(
            {"inlet": INLET, "fan": _outlet(0.0, 0.5, 0.05), "open": pressure}
        )
        assert config.boundaries["fan"].velocity == 0.05

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("u_velocity", 0.1),
            ("v_velocity", 0.1),
            ("concentration", [1.0]),
            ("hepa_filtered", True),
            ("hepa_filtered", False),
            ("deposition_surface", "none"),
        ],
    )
    def test_keys_that_describe_air_entering_are_refused(
        self, key: str, value: object
    ) -> None:
        drain = _outlet(0.0, 1.0) | {key: value}
        with pytest.raises(
            ValueError, match=rf"boundaries\.drain\.{key} is not valid on a fixed_flow"
        ):
            _channel({"inlet": INLET, "drain": drain})

    @pytest.mark.parametrize("velocity", [0.0, -0.5])
    def test_a_velocity_that_is_not_positive_is_refused(self, velocity: float) -> None:
        with pytest.raises(ValueError, match=r"boundaries\.rest\.velocity must be pos"):
            _channel(
                {
                    "inlet": INLET,
                    "a": _outlet(0.0, 0.5),
                    "rest": _outlet(0.5, 1.0, velocity),
                }
            )

    @pytest.mark.parametrize("velocity", [True, "0.5", float("nan"), float("inf")])
    def test_a_velocity_that_is_not_a_finite_number_is_refused(
        self, velocity: object
    ) -> None:
        with pytest.raises(
            (TypeError, ValueError), match=r"boundaries\.rest\.velocity"
        ):
            _channel(
                {
                    "inlet": INLET,
                    "a": _outlet(0.0, 0.5),
                    "rest": _outlet(0.5, 1.0, velocity),
                }
            )

    def test_every_outlet_stating_a_velocity_without_a_pressure_outlet_is_refused(
        self,
    ) -> None:
        """Nothing would balance the flow: the configuration over-determines it."""
        boundaries = {
            "inlet": INLET,
            "a": _outlet(0.0, 0.5, 0.05),
            "b": _outlet(0.5, 1.0, 0.05),
        }
        with pytest.raises(ValueError, match=r"no pressure_outlet.*leave the velocity"):
            _channel(boundaries)

    def test_a_fixed_flow_outlet_without_a_velocity_inlet_is_refused(self) -> None:
        """No inlet means make-up air would enter through a pressure outlet (#61)."""
        pressure = {"type": "pressure_outlet", "location": "right"} | {
            "y_start": 0.0,
            "y_end": 0.5,
        }
        boundaries = {"fan": _outlet(0.5, 1.0, 0.05), "open": pressure}
        with pytest.raises(ValueError, match=r"need a velocity_inlet.*issue 61"):
            _channel(boundaries)
        # The nearest valid form: the same room with an inlet to drive it.
        assert (
            _channel({"inlet": INLET} | boundaries).boundaries["fan"].velocity == 0.05
        )

    def test_velocity_on_a_pressure_outlet_is_refused(self) -> None:
        """It means an outward flow on a fixed_flow_outlet, so it is not ignored here."""
        pressure = {"type": "pressure_outlet", "location": "right"} | {
            "y_start": 0.0,
            "y_end": 1.0,
        }
        with pytest.raises(
            ValueError,
            match=r"boundaries\.drain\.velocity is not valid on a pressure_outlet",
        ):
            _channel({"inlet": INLET, "drain": pressure | {"velocity": 0.1}})
        assert _channel({"inlet": INLET, "drain": pressure}).boundaries["drain"]

    def test_a_blank_velocity_on_a_fixed_flow_outlet_is_refused(self) -> None:
        """Stating none means leaving the key out; a YAML null is a slip."""
        blank = _outlet(0.0, 1.0) | {"velocity": None}
        with pytest.raises(ValueError, match=r"boundaries\.drain\.velocity is blank"):
            _channel({"inlet": INLET, "drain": blank})
        omitted = _channel({"inlet": INLET, "drain": _outlet(0.0, 1.0)})
        assert omitted.boundaries["drain"].velocity is None

    def test_an_unstated_fixed_flow_outlet_beside_a_pressure_outlet_is_refused(
        self,
    ) -> None:
        pressure = {"type": "pressure_outlet", "location": "right"} | {
            "y_start": 0.5,
            "y_end": 1.0,
        }
        with pytest.raises(
            ValueError, match=r"\['fan'\].*no velocity.*pressure_outlet"
        ):
            _channel({"inlet": INLET, "fan": _outlet(0.0, 0.5), "open": pressure})


@pytest.mark.unit
class TestRegistryCondition:
    @pytest.mark.parametrize(
        ("edge", "expected"),
        [
            ("top", (0.0, 0.5)),
            ("bottom", (0.0, -0.5)),
            ("left", (-0.5, 0.0)),
            ("right", (0.5, 0.0)),
        ],
    )
    def test_the_normal_component_points_out_of_the_domain(
        self, edge: str, expected: tuple[float, float]
    ) -> None:
        condition = fixed_flow_condition(edge, 0.5)
        assert condition.bc_type == FIXED_FLOW_OUTLET
        assert (condition.u_prescribed, condition.v_prescribed) == expected

    def test_condition_of_carries_a_stated_velocity_and_zero_for_none(self) -> None:
        stated = BoundarySpec(
            FIXED_FLOW_OUTLET, "bottom", x_start=0.0, x_end=1.0, velocity=0.25
        )
        unstated = BoundarySpec(FIXED_FLOW_OUTLET, "bottom", x_start=0.0, x_end=1.0)
        assert condition_of(stated, "bottom").v_prescribed == -0.25
        assert condition_of(unstated, "bottom").v_prescribed == 0.0


@pytest.mark.unit
class TestResolvedVelocities:
    """The channel: inlet 0.1 m/s over 1 m, outlets on the right edge."""

    def test_the_remainder_is_shared_over_the_discrete_length(self) -> None:
        """a covers centers 0.1 and 0.3 (0.4 m of faces, 0.45 m configured)."""
        config = _channel(
            {"inlet": INLET, "a": _outlet(0.0, 0.45, 0.05), "b": _outlet(0.45, 1.0)}
        )
        _, bc = _build(config)
        resolved = bc.fixed_flow_velocities()
        assert resolved["a"] == 0.05
        # b covers centers 0.5, 0.7 and 0.9: 0.6 m of faces, 0.55 m configured.
        inflow = 0.1 * 5 * 0.2
        assert resolved["b"] == pytest.approx((inflow - 0.05 * 0.4) / 0.6, rel=1e-14)
        configured = (inflow - 0.05 * 0.45) / 0.55
        assert abs(resolved["b"] - configured) > 1e-3

    def test_outflow_equals_inflow_on_the_faces_actually_written(self) -> None:
        config = _channel(
            {"inlet": INLET, "a": _outlet(0.0, 0.45, 0.05), "b": _outlet(0.45, 1.0)}
        )
        mesh, bc = _build(config)
        u, v, _ = allocate_fields(mesh)
        bc.apply_normal_velocity(u, v)
        out = _right_outflow(config, mesh, u)
        inflow = float(np.sum(u[:, 0] * mesh.dy_cell))
        assert sum(out.values()) == pytest.approx(inflow, rel=1e-14)
        assert out["a"] == pytest.approx(0.05 * 0.4, rel=1e-14)

    def test_segments_that_state_none_share_one_velocity(self) -> None:
        config = _channel(
            {"inlet": INLET, "a": _outlet(0.0, 0.5), "b": _outlet(0.5, 1.0)}
        )
        mesh, bc = _build(config)
        u, v, _ = allocate_fields(mesh)
        bc.apply_normal_velocity(u, v)
        assert np.all(u[:, -1] == u[0, -1])
        resolved = bc.fixed_flow_velocities()
        assert resolved["a"] == resolved["b"]
        assert resolved["a"] == pytest.approx(0.1, rel=1e-14)

    def test_the_accessor_returns_a_copy(self) -> None:
        config = _channel({"inlet": INLET, "drain": _outlet(0.0, 1.0)})
        _, bc = _build(config)
        bc.fixed_flow_velocities()["drain"] = 99.0
        assert bc.fixed_flow_velocities()["drain"] == pytest.approx(0.1, rel=1e-14)

    def test_a_configuration_without_fixed_flow_outlets_resolves_none(self) -> None:
        pressure = {"type": "pressure_outlet", "location": "right"} | {
            "y_start": 0.0,
            "y_end": 1.0,
        }
        _, bc = _build(_channel({"inlet": INLET, "drain": pressure}))
        assert bc.fixed_flow_velocities() == {}

    def test_a_remainder_of_exactly_zero_is_refused_naming_the_numbers(self) -> None:
        """Inflow 0.5 * 1.0 and a stated outflow of 1.0 over 0.5 m: nothing left."""
        boundaries = {
            "inlet": INLET | {"velocity": 0.5},
            "fan": _outlet(0.0, 0.5, 1.0),
            "rest": _outlet(0.5, 1.0),
        }
        with pytest.raises(ValueError) as caught:
            _build(_channel(boundaries, ny=4))
        message = str(caught.value)
        assert "inflow 0.5" in message
        assert "stated outflow 0.5" in message
        assert "remainder 0" in message

    def test_a_negative_remainder_is_refused_naming_the_numbers(self) -> None:
        boundaries = {
            "inlet": INLET,
            "fan": _outlet(0.0, 0.5, 0.9),
            "rest": _outlet(0.5, 1.0),
        }
        with pytest.raises(
            ValueError, match=r"inflow 0\.1.*stated outflow.*remainder -"
        ):
            _build(_channel(boundaries))

    def test_an_unstated_segment_covering_no_face_is_refused(self) -> None:
        """A range between two cell centers holds no face to carry the remainder."""
        boundaries = {
            "inlet": INLET,
            "fan": _outlet(0.0, 0.2, 0.05),
            "thin": _outlet(0.31, 0.39),
        }
        with pytest.raises(ValueError, match=r"\['thin'\].*cover no face"):
            _build(_channel(boundaries))

    def test_a_pressure_outlet_balances_the_stated_fixed_flows(self) -> None:
        """With a pressure outlet nothing is shared: the stated value is held."""
        pressure = {"type": "pressure_outlet", "location": "right"} | {
            "y_start": 0.4,
            "y_end": 1.0,
        }
        config = _channel(
            {"inlet": INLET, "fan": _outlet(0.0, 0.4, 0.05), "open": pressure}
        )
        mesh, bc = _build(config)
        u, v, _ = allocate_fields(mesh)
        u[:, -1] = 7.0
        bc.apply_normal_velocity(u, v)
        assert bc.fixed_flow_velocities() == {"fan": 0.05}
        assert np.all(u[:2, -1] == 0.05)
        assert np.all(u[2:, -1] == 7.0)  # the pressure outlet's faces are not written
        assert bc.has_pressure_outlet()


@pytest.mark.unit
class TestStaggeredFaces:
    """The faces and flags the velocity layer writes for a fixed-flow outlet."""

    def _drain(self) -> tuple[SimConfig, Mesh, StaggeredBoundary]:
        config = _channel({"inlet": INLET, "drain": _outlet(0.0, 1.0)})
        return (config, *_build(config))

    def test_the_normal_face_holds_the_resolved_outward_velocity(self) -> None:
        _, mesh, bc = self._drain()
        u, v, _ = allocate_fields(mesh)
        u.fill(3.0)
        bc.apply_normal_velocity(u, v)
        assert np.all(u[:, -1] == bc.fixed_flow_velocities()["drain"])
        assert np.all(u[:, -1] > 0.0)

    def test_the_tangential_velocity_is_dirichlet_zero(self) -> None:
        """A pressure outlet leaves it free; the fixed flow holds it at zero."""
        _, _, bc = self._drain()
        right = bc.tangential_conditions()["right"]
        assert np.all(right.is_dirichlet)
        assert np.all(right.value == 0.0)

        pressure = {"type": "pressure_outlet", "location": "right"} | {
            "y_start": 0.0,
            "y_end": 1.0,
        }
        _, control = _build(_channel({"inlet": INLET, "drain": pressure}))
        free = control.tangential_conditions()["right"].is_dirichlet
        assert not np.all(free)

    def test_no_pressure_outlet_remains(self) -> None:
        _, _, bc = self._drain()
        assert not bc.has_pressure_outlet()
        for outlet in bc.pressure_outlets().values():
            assert not np.any(outlet.is_outlet)

    def test_the_outlet_is_not_counted_in_the_inlet_flux(self) -> None:
        """One outlet states 0.05 m/s, so an outlet counted as inflow shows as nonzero."""
        config = _channel(
            {"inlet": INLET, "a": _outlet(0.0, 0.4, 0.05), "b": _outlet(0.4, 1.0)}
        )
        _, bc = _build(config)
        assert bc.get_inlet_flux("a") == 0.0
        assert bc.get_inlet_flux("b") == 0.0
        assert bc.get_total_inlet_flux() == bc.get_inlet_flux("inlet")
        assert bc.get_total_inlet_flux() == pytest.approx(0.1, rel=1e-14)

    def test_the_outlet_does_not_set_the_velocity_scale(self) -> None:
        """A fan faster than the inlet would otherwise move the stopping rule's scale."""
        pressure = {"type": "pressure_outlet", "location": "right"} | {
            "y_start": 0.5,
            "y_end": 1.0,
        }
        config = _channel(
            {"inlet": INLET, "fan": _outlet(0.0, 0.5, 0.9), "open": pressure}
        )
        _, bc = _build(config)
        assert bc.get_max_boundary_velocity() == 0.1

    def test_the_closed_domain_path_engages_in_the_corrector(self) -> None:
        config, mesh, bc = self._drain()
        corrector = PressureCorrector(mesh, config, bc)
        assert corrector.needs_pin
        first = np.argwhere(mesh.cell_type == FLUID)[0]
        assert corrector.pin_cell == (int(first[0]), int(first[1]))
        # The inflow branch of the flux scale: the inflow is positive.
        assert corrector.flux_scale == pytest.approx(
            config.rho * bc.get_total_inlet_flux(), rel=1e-15
        )


@pytest.mark.integration
class TestConcentrationChannel:
    """The scalar layer books a fixed-flow outlet as it books a pressure outlet."""

    def test_the_faces_are_those_of_a_pressure_outlet_over_the_same_range(self) -> None:
        def faces_of(outlet_type: str) -> ConcentrationFaces:
            outlet = _outlet(0.0, 1.0)
            outlet["type"] = outlet_type
            config = _channel({"inlet": INLET, "drain": outlet})
            mesh = Mesh(config)
            boundary = ConcentrationBoundary(
                mesh, config, ParticlePhysics(config), BoundaryRegistry(config)
            )
            return boundary.faces_for(0)

        fixed, pressure = faces_of(FIXED_FLOW_OUTLET), faces_of("pressure_outlet")
        for name in (
            "inflow_u",
            "inflow_v",
            "deposition_u",
            "deposition_v",
            "surface_u",
            "surface_v",
            "settling_v",
        ):
            assert np.array_equal(getattr(fixed, name), getattr(pressure, name)), name

    def test_the_budget_books_the_outflow_through_a_fixed_flow_outlet(self) -> None:
        """Uniform concentration advected out at 0.1 m/s leaves at u C dy per face."""
        config = _channel({"inlet": INLET, "drain": _outlet(0.0, 1.0)})
        mesh = Mesh(config)
        boundary = ConcentrationBoundary(
            mesh, config, ParticlePhysics(config), BoundaryRegistry(config)
        )
        solver = TransportSolver(mesh, config, ParticlePhysics(config), boundary)
        faces = uniform_face_field(mesh, 0.1, 0.0)
        c = np.ones((mesh.yc.shape[0], mesh.xc.shape[0]))
        dt = solver.stable_dt(faces, 0)
        solver.solve_timestep(c, faces, 0, dt)
        budget = solver.budget[0]
        assert budget.outflow == pytest.approx(0.1 * 1.0 * dt, rel=1e-12)
        assert budget.inflow == 0.0
