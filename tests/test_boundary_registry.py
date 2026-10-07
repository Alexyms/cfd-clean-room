"""Tests for the shared boundary registry (REQ-S12.1).

The registry interprets configured boundary segments for the staggered
velocity layer and the concentration layer. These tests pin the rules both
layers rely on: inclusive coverage on the matching edge, first match in
configuration order, wall by default, the prescribed-velocity decomposition
per edge, and coverage_along, the one derivation of which faces a segment
covers (REQ-S12.1).
"""

import numpy as np
import pytest

from src.boundary_registry import (
    BEHIND_SOLID,
    NO_SLIP_WALL,
    UNCOVERED,
    BoundaryRegistry,
    EdgeCondition,
    EdgeCoverage,
    condition_of,
    covers,
)
from src.config import BoundarySpec, SimConfig


def _config(boundaries: dict) -> SimConfig:
    raw = {
        "domain": {"width": 1.0, "height": 1.0, "nx": 10, "ny": 10},
        "fluid": {"density": 1.2, "viscosity": 1.81e-5, "temperature": 293.0},
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
            "max_pressure_iter": 5000,
            "pressure_rtol": 1.0e-8,
        },
        "boundaries": boundaries,
        "obstacles": [],
        "sensors": [{"name": "center", "x": 0.5, "y": 0.5}],
        "thresholds": {"0.1e-6": 100.0},
    }
    return SimConfig.from_dict(raw)


TOP_INLET = BoundarySpec(
    type="velocity_inlet", location="top", x_start=0.3, x_end=0.7, velocity=0.45
)
LEFT_OUTLET = BoundarySpec(
    type="pressure_outlet", location="left", y_start=0.0, y_end=1.0
)


@pytest.mark.unit
class TestCoverage:
    """Which points of an edge a segment covers."""

    def test_point_inside_the_range_on_the_same_edge_is_covered(self) -> None:
        assert covers(TOP_INLET, "top", 0.5)

    def test_endpoints_are_inclusive(self) -> None:
        assert covers(TOP_INLET, "top", 0.3)
        assert covers(TOP_INLET, "top", 0.7)

    def test_point_outside_the_range_is_not_covered(self) -> None:
        assert not covers(TOP_INLET, "top", 0.2)
        assert not covers(TOP_INLET, "top", 0.8)

    def test_other_edges_are_not_covered_whatever_the_coordinate(self) -> None:
        for edge in ("bottom", "left", "right"):
            assert not covers(TOP_INLET, edge, 0.5)

    def test_left_right_segments_use_the_y_range(self) -> None:
        assert covers(LEFT_OUTLET, "left", 0.5)
        assert not covers(LEFT_OUTLET, "right", 0.5)

    def test_missing_range_never_covers(self) -> None:
        spec = BoundarySpec(type="wall", location="top")
        assert not covers(spec, "top", 0.5)


@pytest.mark.unit
class TestConditionOf:
    """The velocity a segment prescribes on its edge."""

    def test_wall_is_the_shared_no_slip_condition(self) -> None:
        spec = BoundarySpec(type="wall", location="top", x_start=0.0, x_end=1.0)
        assert condition_of(spec, "top") is NO_SLIP_WALL

    def test_outlet_prescribes_no_velocity(self) -> None:
        assert condition_of(LEFT_OUTLET, "left") == EdgeCondition(
            "pressure_outlet", 0.0, 0.0
        )

    @pytest.mark.parametrize(
        ("edge", "expected"),
        [
            ("top", (0.0, -0.45)),
            ("bottom", (0.0, 0.45)),
            ("left", (0.45, 0.0)),
            ("right", (-0.45, 0.0)),
        ],
    )
    def test_magnitude_is_decomposed_normal_and_inward(
        self, edge: str, expected: tuple[float, float]
    ) -> None:
        spec = BoundarySpec(type="velocity_inlet", location=edge, velocity=0.45)
        condition = condition_of(spec, edge)
        assert condition.bc_type == "velocity_inlet"
        assert (condition.u_prescribed, condition.v_prescribed) == expected

    def test_explicit_components_override_the_magnitude(self) -> None:
        spec = BoundarySpec(
            type="velocity_inlet",
            location="top",
            velocity=0.45,
            u_velocity=1.0,
            v_velocity=0.0,
        )
        condition = condition_of(spec, "top")
        assert (condition.u_prescribed, condition.v_prescribed) == (1.0, 0.0)

    def test_one_explicit_component_reads_the_other_as_zero(self) -> None:
        spec = BoundarySpec(type="velocity_inlet", location="top", u_velocity=1.0)
        condition = condition_of(spec, "top")
        assert (condition.u_prescribed, condition.v_prescribed) == (1.0, 0.0)

    def test_unknown_type_is_rejected(self) -> None:
        spec = BoundarySpec(type="periodic", location="top", x_start=0.0, x_end=1.0)
        with pytest.raises(ValueError, match="Unrecognized boundary type"):
            condition_of(spec, "top")


@pytest.mark.unit
class TestRegistryLookup:
    """The registry's answer at a point of an edge."""

    def test_uncovered_point_is_a_no_slip_wall(self) -> None:
        registry = BoundaryRegistry(
            _config(
                {
                    "inlet": {
                        "type": "velocity_inlet",
                        "location": "top",
                        "x_start": 0.3,
                        "x_end": 0.7,
                        "velocity": 0.45,
                    }
                }
            )
        )
        assert registry.condition_at("top", 0.1) is NO_SLIP_WALL
        assert registry.condition_at("bottom", 0.5) is NO_SLIP_WALL

    def test_covered_point_reports_the_segment(self) -> None:
        registry = BoundaryRegistry(
            _config(
                {
                    "inlet": {
                        "type": "velocity_inlet",
                        "location": "top",
                        "x_start": 0.3,
                        "x_end": 0.7,
                        "velocity": 0.45,
                    }
                }
            )
        )
        assert registry.condition_at("top", 0.5) == EdgeCondition(
            "velocity_inlet", 0.0, -0.45
        )

    def test_first_covering_segment_in_config_order_wins(self) -> None:
        registry = BoundaryRegistry(
            _config(
                {
                    "first": {
                        "type": "velocity_inlet",
                        "location": "top",
                        "x_start": 0.0,
                        "x_end": 0.5,
                        "velocity": 0.1,
                    },
                    "second": {
                        "type": "pressure_outlet",
                        "location": "top",
                        "x_start": 0.5,
                        "x_end": 1.0,
                    },
                }
            )
        )
        # 0.5 lies in both ranges; the segment listed first decides.
        assert registry.condition_at("top", 0.5).bc_type == "velocity_inlet"
        assert registry.condition_at("top", 0.75).bc_type == "pressure_outlet"

    def test_spec_by_name_and_unknown_name(self) -> None:
        registry = BoundaryRegistry(
            _config(
                {
                    "lid": {
                        "type": "velocity_inlet",
                        "location": "top",
                        "x_start": 0.0,
                        "x_end": 1.0,
                        "u_velocity": 1.0,
                    }
                }
            )
        )
        assert registry.spec("lid").u_velocity == 1.0
        with pytest.raises(KeyError, match="not_real"):
            registry.spec("not_real")
        assert list(registry.boundaries) == ["lid"]


COVERAGE_SEGMENTS = {
    "inlet": {
        "type": "velocity_inlet",
        "location": "top",
        "x_start": 0.25,
        "x_end": 0.5,
        "velocity": 0.3,
    },
    "later_inlet": {
        "type": "velocity_inlet",
        "location": "top",
        "x_start": 0.5,
        "x_end": 0.8,
        "velocity": 0.7,
    },
    "outlet": {
        "type": "pressure_outlet",
        "location": "right",
        "y_start": 0.0,
        "y_end": 0.4,
    },
}


@pytest.mark.unit
class TestCoverageAlong:
    """Coverage at a run of points, the derivation both boundary layers read."""

    def test_segment_at_names_the_first_covering_segment_or_none(self) -> None:
        registry = BoundaryRegistry(_config(COVERAGE_SEGMENTS))
        name, spec = registry.segment_at("top", 0.45)
        assert name == "inlet" and spec is registry.spec("inlet")
        # The two ranges meet at 0.5; the first in configuration order decides.
        assert registry.segment_at("top", 0.5)[0] == "inlet"
        assert registry.segment_at("top", 0.7)[0] == "later_inlet"
        assert registry.segment_at("top", 0.9) is None
        assert registry.segment_at("bottom", 0.3) is None

    def test_points_carry_their_segment_its_condition_and_the_solid_rule(self) -> None:
        """Centers of ten cells of 0.1 on the top edge; the fourth cell behind is SOLID."""
        registry = BoundaryRegistry(_config(COVERAGE_SEGMENTS))
        centers = np.arange(10) * 0.1 + 0.05
        solid = np.zeros(10, dtype=bool)
        solid[3] = True
        coverage = registry.coverage_along("top", centers, solid)
        assert len(coverage) == 10
        assert [c.name for c in coverage] == [
            None,
            None,
            "inlet",
            None,
            "inlet",
            "later_inlet",
            "later_inlet",
            "later_inlet",
            None,
            None,
        ]
        assert coverage[3] is BEHIND_SOLID
        assert coverage[0] is UNCOVERED and coverage[9] is UNCOVERED
        assert coverage[2] == EdgeCoverage(
            "inlet",
            registry.spec("inlet"),
            EdgeCondition("velocity_inlet", 0.0, -0.3),
            False,
        )
        assert coverage[5].condition == EdgeCondition("velocity_inlet", 0.0, -0.7)
        for point, center in zip(coverage, centers, strict=True):
            if not point.solid:
                assert point.condition == registry.condition_at("top", center)
            else:
                assert point.condition == NO_SLIP_WALL
            assert point.solid is bool(solid[list(centers).index(center)])

    def test_a_plain_sequence_and_an_outlet_edge_work_the_same(self) -> None:
        registry = BoundaryRegistry(_config(COVERAGE_SEGMENTS))
        coverage = registry.coverage_along(
            "right", [0.1, 0.3, 0.5], [False, False, False]
        )
        assert [c.name for c in coverage] == ["outlet", "outlet", None]
        assert coverage[0].condition.bc_type == "pressure_outlet"
        assert coverage[2] is UNCOVERED

    def test_mismatched_lengths_raise(self) -> None:
        registry = BoundaryRegistry(_config(COVERAGE_SEGMENTS))
        with pytest.raises(ValueError, match="one SOLID flag per point"):
            registry.coverage_along("top", [0.1, 0.2], [False])
