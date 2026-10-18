"""Tests for the YAML configuration loader.

Unit tests verify that SimConfig loads valid configurations correctly
and rejects invalid configurations with clear error messages at load
time (REQ-C02).
"""

import dataclasses
import math
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

from src.config import (
    CFL_NUMBER_BOUND,
    DEFAULT_PRESSURE_PRECONDITIONER,
    EVENT_TIME_RTOL,
    MULTIGRID,
    PRESSURE_JACOBI,
    PRESSURE_PRECONDITIONERS,
    PRESSURE_RTOL_BOUNDS,
    RETIRED_PRESSURE_TOL_KEY,
    SOLVER_KEYS,
    SimConfig,
    TransportSpec,
    TurbulenceSpec,
)
from validation.cases import CONFIG_DIR, load_case
from validation.transport_cases import SOLVER_BLOCK


def _write_config(tmp_path, overrides: dict | None = None) -> str:
    """Write a valid YAML config to tmp_path, optionally with overrides.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest tmp_path fixture directory.
    overrides : dict or None
        Keys to replace in the base config dict before writing.
        Each key replaces the entire top-level section (no nested
        merge). Set a section to None to remove it entirely.

    Returns
    -------
    str
        Path to the written YAML file.
    """
    base = {
        "domain": {"width": 4.0, "height": 3.0, "nx": 80, "ny": 60},
        "fluid": {"density": 1.2, "viscosity": 1.81e-5, "temperature": 293.0},
        "particles": {
            "density": 1000.0,
            "sizes": [0.1e-6, 0.3e-6, 0.5e-6, 1.0e-6, 5.0e-6],
            "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {
                "diameters": [0.1e-6, 0.3e-6, 0.5e-6, 1.0e-6, 5.0e-6],
                "efficiencies": [0.99999, 0.99970, 0.99990, 0.99999, 0.99999],
            },
        },
        "solver": {
            "dt": 0.01,
            "t_end": 60.0,
            "output_interval": 10,
            "convergence_tol": 1.0e-6,
            "max_simple_iter": 500,
            "alpha_velocity": 0.7,
            "alpha_pressure": 0.3,
            "max_pressure_iter": 5000,
            "pressure_rtol": 1.0e-8,
        },
        "boundaries": {
            "hepa_supply": {
                "type": "velocity_inlet",
                "location": "top",
                "x_start": 0.5,
                "x_end": 3.5,
                "velocity": 0.45,
            },
        },
        "obstacles": [
            {
                "name": "equipment",
                "x_start": 1.0,
                "x_end": 2.0,
                "y_start": 0.0,
                "y_end": 1.0,
            },
        ],
        "sensors": [{"name": "center", "x": 2.0, "y": 1.5}],
        "thresholds": {"0.5e-6": 3520.0},
    }

    if overrides:
        for key, val in overrides.items():
            if val is None:
                base.pop(key, None)
            else:
                base[key] = val

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(base, default_flow_style=False), encoding="utf-8")
    return str(config_path)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSimConfigValid:
    """Tests that a valid configuration loads correctly."""

    def test_loads_without_error(self, tmp_path) -> None:
        """SimConfig loads a valid YAML file without raising."""
        path = _write_config(tmp_path)
        config = SimConfig(path)
        assert config.room_width == 4.0
        assert config.room_height == 3.0

    def test_domain_attributes(self, tmp_path) -> None:
        """Domain section maps to correct typed attributes."""
        config = SimConfig(_write_config(tmp_path))
        assert config.nx == 80
        assert config.ny == 60
        assert isinstance(config.nx, int)
        assert isinstance(config.ny, int)
        assert isinstance(config.room_width, float)

    def test_fluid_attributes(self, tmp_path) -> None:
        """Fluid section maps to rho, mu, temperature attributes."""
        config = SimConfig(_write_config(tmp_path))
        assert config.rho == 1.2
        assert config.mu == pytest.approx(1.81e-5)
        assert config.temperature == 293.0

    def test_particle_attributes(self, tmp_path) -> None:
        """Particle section maps to density, sizes, mean_free_path."""
        config = SimConfig(_write_config(tmp_path))
        assert config.particle_density == 1000.0
        assert len(config.particle_sizes) == 5
        assert config.particle_sizes[0] == pytest.approx(0.1e-6)
        assert config.mean_free_path == pytest.approx(67.0e-9)

    def test_boundary_layer_thickness(self, tmp_path) -> None:
        """boundary_layer_thickness is loaded from the particles section."""
        config = SimConfig(_write_config(tmp_path))
        assert config.boundary_layer_thickness == pytest.approx(1.0e-3)

    def test_hepa_reference_data(self, tmp_path) -> None:
        """HEPA reference diameters and efficiencies are loaded."""
        config = SimConfig(_write_config(tmp_path))
        assert len(config.hepa_reference.diameters) == 5
        assert len(config.hepa_reference.efficiencies) == 5
        assert config.hepa_reference.efficiencies[1] == pytest.approx(0.99970)

    def test_solver_attributes(self, tmp_path) -> None:
        """Solver section maps to dt, t_end, output_interval, etc."""
        config = SimConfig(_write_config(tmp_path))
        assert config.dt == 0.01
        assert config.t_end == 60.0
        assert config.output_interval == 10
        assert config.convergence_tol == pytest.approx(1.0e-6)
        assert config.max_simple_iter == 500
        assert config.alpha_velocity == pytest.approx(0.7)
        assert config.alpha_pressure == pytest.approx(0.3)
        assert config.max_pressure_iter == 5000
        assert config.pressure_rtol == pytest.approx(1.0e-8)

    def test_boundaries_parsed(self, tmp_path) -> None:
        """Boundaries section produces BoundarySpec objects."""
        config = SimConfig(_write_config(tmp_path))
        assert "hepa_supply" in config.boundaries
        bc = config.boundaries["hepa_supply"]
        assert bc.type == "velocity_inlet"
        assert bc.location == "top"
        assert bc.velocity == 0.45

    def test_obstacles_parsed(self, tmp_path) -> None:
        """Obstacles section produces ObstacleSpec objects."""
        config = SimConfig(_write_config(tmp_path))
        assert len(config.obstacles) == 1
        assert config.obstacles[0].name == "equipment"

    def test_sensors_parsed(self, tmp_path) -> None:
        """Sensors section produces SensorSpec objects."""
        config = SimConfig(_write_config(tmp_path))
        assert len(config.sensors) == 1
        assert config.sensors[0].name == "center"
        assert config.sensors[0].x == 2.0

    def test_thresholds_parsed(self, tmp_path) -> None:
        """Thresholds section produces a string-keyed dict of floats."""
        config = SimConfig(_write_config(tmp_path))
        assert "0.5e-6" in config.thresholds
        assert config.thresholds["0.5e-6"] == 3520.0

    def test_default_yaml_loads(self) -> None:
        """The shipped clean_room_default.yaml loads on the product grid of step 8."""
        config = SimConfig("configs/clean_room_default.yaml")
        assert config.nx == 320
        assert config.ny == 120
        assert config.room_width == 8.0
        assert len(config.sensors) == 4
        assert len(config.obstacles) == 4

    def test_default_yaml_is_the_product_configuration_of_step_8(self) -> None:
        """The default file carries the settings prompts 46 to 48 measured the room under.

        Alex's decision of 2026-10-15 (prompt 56; ECR-002 section 8, step 8's
        note; ADR-013 decision 3 as amended): the k-epsilon section with the
        standard variant and upwind advection, the error_estimate rule with
        ADR-011 G's imbalance bound at the file's own cell volume, ten
        momentum sweeps, pressure_rtol 1e-4, the 15,000 cap (10,000 until Alex's
        decision 3 of 2026-10-15, prompt 58), alpha_velocity
        0.5, the multigrid preconditioner stated, and the supply's inlet
        turbulence at 5% and 0.1 m.
        """
        config = SimConfig(CONFIG_DIR / "clean_room_default.yaml")
        assert config.turbulence == TurbulenceSpec(
            model="k_epsilon",
            variant="standard",
            wall_treatment="scalable_wall_functions",
            cfl_number=0.25,
            advection_scheme="upwind",
            alpha_turbulence=0.7,
            max_iter=500,
            tol=1e-10,
        )
        assert config.stopping_rule == "error_estimate"
        assert config.iteration_error_tol == 1e-6
        v_min = (config.room_width / config.nx) * (config.room_height / config.ny)
        assert config.mass_imbalance_tol == pytest.approx(
            1e-4 * config.rho * v_min / config.t_end, rel=1e-12
        )
        assert config.momentum_sweeps == 10
        assert config.pressure_rtol == 1e-4
        assert config.max_simple_iter == 15000
        assert config.alpha_velocity == 0.5
        assert config.pressure_preconditioner == "multigrid"
        supply = config.boundaries["hepa_supply"]
        assert supply.turbulence_intensity == 0.05
        assert supply.dissipation_length == 0.1
        for name, spec in config.boundaries.items():
            if name != "hepa_supply":
                assert spec.turbulence_intensity is None
                assert spec.dissipation_length is None

    def test_default_yaml_carries_the_transport_section_and_a_filtered_supply(
        self,
    ) -> None:
        """The product case reads its transport block and marks hepa_supply filtered."""
        config = SimConfig(CONFIG_DIR / "clean_room_default.yaml")
        assert config.transport == TransportSpec(
            cfl_number=0.1,
            advection_scheme="umist",
            max_diffusion_iter=200,
            diffusion_tol=1e-8,
            turbulent_schmidt=0.7,
        )
        supply = config.boundaries["hepa_supply"]
        assert supply.hepa_filtered is True
        assert supply.concentration is None
        for name, spec in config.boundaries.items():
            if name != "hepa_supply":
                assert spec.hepa_filtered is False
            assert spec.concentration is None
            assert spec.deposition_surface is None


@pytest.mark.unit
class TestSimConfigMissingSections:
    """Tests that missing required sections raise ValueError."""

    @pytest.mark.parametrize(
        "section",
        [
            "domain",
            "fluid",
            "particles",
            "solver",
            "boundaries",
            "sensors",
            "thresholds",
        ],
    )
    def test_missing_section_raises(self, tmp_path, section: str) -> None:
        """Missing a required top-level section raises ValueError."""
        path = _write_config(tmp_path, overrides={section: None})
        with pytest.raises(
            ValueError, match=f"Missing required config section.*{section}"
        ):
            SimConfig(path)


@pytest.mark.unit
class TestSimConfigMissingKeys:
    """Tests that missing required keys within sections raise ValueError."""

    def test_missing_domain_width(self, tmp_path) -> None:
        """Missing domain.width raises ValueError."""
        path = _write_config(
            tmp_path, overrides={"domain": {"height": 3.0, "nx": 80, "ny": 60}}
        )
        with pytest.raises(ValueError, match=r"domain\.width"):
            SimConfig(path)

    def test_missing_fluid_viscosity(self, tmp_path) -> None:
        """Missing fluid.viscosity raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={"fluid": {"density": 1.2, "temperature": 293.0}},
        )
        with pytest.raises(ValueError, match=r"fluid\.viscosity"):
            SimConfig(path)

    def test_missing_particle_sizes(self, tmp_path) -> None:
        """Missing particles.sizes raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "particles": {
                    "density": 1000.0,
                    "mean_free_path": 67e-9,
                    "boundary_layer_thickness": 1e-3,
                }
            },
        )
        with pytest.raises(ValueError, match=r"particles\.sizes"):
            SimConfig(path)


@pytest.mark.unit
class TestSimConfigInvalidValues:
    """Tests that out-of-range and wrong-type values are rejected."""

    def test_negative_viscosity_raises(self, tmp_path) -> None:
        """Negative fluid viscosity raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "fluid": {"density": 1.2, "viscosity": -1.0, "temperature": 293.0}
            },
        )
        with pytest.raises(ValueError, match=r"fluid\.viscosity.*positive"):
            SimConfig(path)

    def test_zero_nx_raises(self, tmp_path) -> None:
        """Zero grid dimension raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={"domain": {"width": 4.0, "height": 3.0, "nx": 0, "ny": 60}},
        )
        with pytest.raises(ValueError, match=r"domain\.nx.*positive"):
            SimConfig(path)

    def test_negative_ny_raises(self, tmp_path) -> None:
        """Negative grid dimension raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={"domain": {"width": 4.0, "height": 3.0, "nx": 80, "ny": -5}},
        )
        with pytest.raises(ValueError, match=r"domain\.ny.*positive"):
            SimConfig(path)

    def test_string_nx_raises_type_error(self, tmp_path) -> None:
        """String where integer expected raises TypeError."""
        path = _write_config(
            tmp_path,
            overrides={
                "domain": {"width": 4.0, "height": 3.0, "nx": "eighty", "ny": 60}
            },
        )
        with pytest.raises(TypeError, match=r"domain\.nx.*integer"):
            SimConfig(path)

    def test_string_viscosity_raises_type_error(self, tmp_path) -> None:
        """String where float expected raises TypeError."""
        path = _write_config(
            tmp_path,
            overrides={
                "fluid": {"density": 1.2, "viscosity": "low", "temperature": 293.0}
            },
        )
        with pytest.raises(TypeError, match=r"fluid\.viscosity.*number"):
            SimConfig(path)

    def test_empty_particle_sizes_raises(self, tmp_path) -> None:
        """Empty particle sizes list raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "particles": {
                    "density": 1000.0,
                    "sizes": [],
                    "mean_free_path": 67e-9,
                    "boundary_layer_thickness": 1e-3,
                }
            },
        )
        with pytest.raises(ValueError, match=r"sizes.*must not be empty"):
            SimConfig(path)

    def test_negative_particle_size_raises(self, tmp_path) -> None:
        """Negative particle size raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "particles": {
                    "density": 1000.0,
                    "sizes": [-1e-6],
                    "mean_free_path": 67e-9,
                    "boundary_layer_thickness": 1e-3,
                }
            },
        )
        with pytest.raises(ValueError, match=r"sizes.*positive"):
            SimConfig(path)

    def test_hepa_efficiency_out_of_range_raises(self, tmp_path) -> None:
        """HEPA efficiency > 1.0 raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "particles": {
                    "density": 1000.0,
                    "sizes": [0.1e-6],
                    "mean_free_path": 67e-9,
                    "boundary_layer_thickness": 1e-3,
                    "hepa_reference": {
                        "diameters": [0.1e-6],
                        "efficiencies": [1.5],
                    },
                }
            },
        )
        with pytest.raises(ValueError, match=r"HEPA efficiency.*\[0, 1\]"):
            SimConfig(path)

    def test_hepa_length_mismatch_raises(self, tmp_path) -> None:
        """Mismatched HEPA diameters and efficiencies lengths raise ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "particles": {
                    "density": 1000.0,
                    "sizes": [0.1e-6],
                    "mean_free_path": 67e-9,
                    "boundary_layer_thickness": 1e-3,
                    "hepa_reference": {
                        "diameters": [0.1e-6, 0.3e-6],
                        "efficiencies": [0.99999],
                    },
                }
            },
        )
        with pytest.raises(ValueError, match="same length"):
            SimConfig(path)

    def test_boundary_invalid_type_raises(self, tmp_path) -> None:
        """Unrecognized boundary type raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "supersonic_inlet",
                        "location": "top",
                        "x_start": 0.5,
                        "x_end": 3.5,
                    }
                }
            },
        )
        with pytest.raises(ValueError, match=r"boundaries\.bad\.type"):
            SimConfig(path)

    def test_boundary_invalid_location_raises(self, tmp_path) -> None:
        """Unrecognized boundary location raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "wall",
                        "location": "diagonal",
                    }
                }
            },
        )
        with pytest.raises(ValueError, match=r"boundaries\.bad\.location"):
            SimConfig(path)

    def test_velocity_inlet_missing_velocity_raises(self, tmp_path) -> None:
        """velocity_inlet without velocity field raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "velocity_inlet",
                        "location": "top",
                        "x_start": 0.5,
                        "x_end": 3.5,
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="velocity_inlet requires"):
            SimConfig(path)

    def test_velocity_inlet_negative_velocity_raises(self, tmp_path) -> None:
        """velocity_inlet with negative velocity raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "velocity_inlet",
                        "location": "top",
                        "x_start": 0.5,
                        "x_end": 3.5,
                        "velocity": -0.5,
                    }
                }
            },
        )
        with pytest.raises(ValueError, match=r"velocity.*positive"):
            SimConfig(path)

    def test_velocity_inlet_bool_velocity_raises_type_error(self, tmp_path) -> None:
        """velocity_inlet with boolean velocity raises TypeError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "velocity_inlet",
                        "location": "top",
                        "x_start": 0.5,
                        "x_end": 3.5,
                        "velocity": True,
                    }
                }
            },
        )
        with pytest.raises(TypeError, match=r"velocity.*number"):
            SimConfig(path)

    def test_velocity_inlet_bool_u_velocity_raises_type_error(self, tmp_path) -> None:
        """velocity_inlet with boolean u_velocity raises TypeError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "velocity_inlet",
                        "location": "top",
                        "x_start": 0.5,
                        "x_end": 3.5,
                        "u_velocity": True,
                    }
                }
            },
        )
        with pytest.raises(TypeError, match=r"u_velocity.*number"):
            SimConfig(path)

    def test_velocity_inlet_bool_v_velocity_raises_type_error(self, tmp_path) -> None:
        """velocity_inlet with boolean v_velocity raises TypeError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "velocity_inlet",
                        "location": "top",
                        "x_start": 0.5,
                        "x_end": 3.5,
                        "v_velocity": False,
                    }
                }
            },
        )
        with pytest.raises(TypeError, match=r"v_velocity.*number"):
            SimConfig(path)

    def test_velocity_inlet_string_u_velocity_raises_type_error(self, tmp_path) -> None:
        """velocity_inlet with non-numeric string u_velocity raises TypeError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "velocity_inlet",
                        "location": "top",
                        "x_start": 0.5,
                        "x_end": 3.5,
                        "u_velocity": "fast",
                    }
                }
            },
        )
        with pytest.raises(TypeError, match=r"u_velocity.*number"):
            SimConfig(path)

    def test_velocity_inlet_only_u_velocity_accepted(self, tmp_path) -> None:
        """velocity_inlet with u_velocity only is accepted; v_velocity is None."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "lid": {
                        "type": "velocity_inlet",
                        "location": "top",
                        "x_start": 0.0,
                        "x_end": 4.0,
                        "u_velocity": 1.0,
                    }
                }
            },
        )
        config = SimConfig(path)
        bc = config.boundaries["lid"]
        assert bc.u_velocity == pytest.approx(1.0)
        assert bc.v_velocity is None
        assert bc.velocity is None

    def test_velocity_inlet_components_override_magnitude(self, tmp_path) -> None:
        """velocity_inlet with both velocity and components stores all three.

        The loader accepts the config; the boundary layer uses the explicit
        components and ignores the magnitude for decomposition.
        """
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "lid": {
                        "type": "velocity_inlet",
                        "location": "top",
                        "x_start": 0.0,
                        "x_end": 4.0,
                        "velocity": 2.0,
                        "u_velocity": 1.0,
                        "v_velocity": 0.0,
                    }
                }
            },
        )
        config = SimConfig(path)
        bc = config.boundaries["lid"]
        assert bc.velocity == pytest.approx(2.0)
        assert bc.u_velocity == pytest.approx(1.0)
        assert bc.v_velocity == pytest.approx(0.0)

    def test_boundaries_as_list_raises(self, tmp_path) -> None:
        """Boundaries section as a list instead of mapping raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={"boundaries": [{"type": "wall", "location": "top"}]},
        )
        with pytest.raises(ValueError, match="boundaries must be a mapping"):
            SimConfig(path)

    def test_boundary_top_missing_x_coords_raises(self, tmp_path) -> None:
        """Top boundary missing x_start raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "wall",
                        "location": "top",
                        "x_end": 3.0,
                    }
                }
            },
        )
        with pytest.raises(ValueError, match=r"boundaries\.bad\.x_start"):
            SimConfig(path)

    def test_boundary_left_missing_y_coords_raises(self, tmp_path) -> None:
        """Left boundary missing y_start raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "wall",
                        "location": "left",
                        "y_end": 2.0,
                    }
                }
            },
        )
        with pytest.raises(ValueError, match=r"boundaries\.bad\.y_start"):
            SimConfig(path)

    def test_boundary_top_inverted_x_raises(self, tmp_path) -> None:
        """Top boundary with x_start >= x_end raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "wall",
                        "location": "top",
                        "x_start": 3.0,
                        "x_end": 1.0,
                    }
                }
            },
        )
        with pytest.raises(ValueError, match=r"x_start.*less than.*x_end"):
            SimConfig(path)

    def test_boundary_left_inverted_y_raises(self, tmp_path) -> None:
        """Left boundary with y_start >= y_end raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "wall",
                        "location": "left",
                        "y_start": 2.0,
                        "y_end": 0.5,
                    }
                }
            },
        )
        with pytest.raises(ValueError, match=r"y_start.*less than.*y_end"):
            SimConfig(path)

    def test_boundary_top_x_outside_domain_raises(self, tmp_path) -> None:
        """Top boundary with x_end beyond room_width raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "wall",
                        "location": "top",
                        "x_start": 0.0,
                        "x_end": 10.0,
                    }
                }
            },
        )
        with pytest.raises(ValueError, match=r"outside domain"):
            SimConfig(path)

    def test_boundary_right_y_outside_domain_raises(self, tmp_path) -> None:
        """Right boundary with y_end beyond room_height raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "bad": {
                        "type": "wall",
                        "location": "right",
                        "y_start": 0.0,
                        "y_end": 10.0,
                    }
                }
            },
        )
        with pytest.raises(ValueError, match=r"outside domain"):
            SimConfig(path)

    def test_sensor_out_of_domain_x_raises(self, tmp_path) -> None:
        """Sensor x coordinate outside domain raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={"sensors": [{"name": "bad", "x": 10.0, "y": 1.0}]},
        )
        with pytest.raises(ValueError, match=r"sensors\[0\].*x=10"):
            SimConfig(path)

    def test_sensor_out_of_domain_y_raises(self, tmp_path) -> None:
        """Sensor y coordinate outside domain raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={"sensors": [{"name": "bad", "x": 1.0, "y": -1.0}]},
        )
        with pytest.raises(ValueError, match=r"sensors\[0\].*y=-1"):
            SimConfig(path)

    def test_obstacle_inverted_x_raises(self, tmp_path) -> None:
        """Obstacle with x_start >= x_end raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "obstacles": [
                    {
                        "name": "bad",
                        "x_start": 2.0,
                        "x_end": 1.0,
                        "y_start": 0.0,
                        "y_end": 1.0,
                    }
                ]
            },
        )
        with pytest.raises(ValueError, match=r"x_start.*less than.*x_end"):
            SimConfig(path)

    def test_obstacle_inverted_y_raises(self, tmp_path) -> None:
        """Obstacle with y_start >= y_end raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "obstacles": [
                    {
                        "name": "bad",
                        "x_start": 0.0,
                        "x_end": 1.0,
                        "y_start": 2.0,
                        "y_end": 1.0,
                    }
                ]
            },
        )
        with pytest.raises(ValueError, match=r"y_start.*less than.*y_end"):
            SimConfig(path)

    def test_obstacle_outside_domain_raises(self, tmp_path) -> None:
        """Obstacle extending beyond domain bounds raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "obstacles": [
                    {
                        "name": "bad",
                        "x_start": 0.0,
                        "x_end": 10.0,
                        "y_start": 0.0,
                        "y_end": 1.0,
                    }
                ]
            },
        )
        with pytest.raises(ValueError, match=r"outside domain"):
            SimConfig(path)

    def test_domain_not_mapping_raises(self, tmp_path) -> None:
        """Domain section as a scalar raises ValueError."""
        path = _write_config(tmp_path, overrides={"domain": 5})
        with pytest.raises(ValueError, match="domain must be a mapping"):
            SimConfig(path)

    def test_bool_domain_width_raises_type_error(self, tmp_path) -> None:
        """Boolean where float expected raises TypeError."""
        path = _write_config(
            tmp_path,
            overrides={"domain": {"width": True, "height": 3.0, "nx": 80, "ny": 60}},
        )
        with pytest.raises(TypeError, match=r"domain\.width.*number"):
            SimConfig(path)

    def test_bool_in_particle_sizes_raises_type_error(self, tmp_path) -> None:
        """Boolean in particle sizes list raises TypeError."""
        path = _write_config(
            tmp_path,
            overrides={
                "particles": {
                    "density": 1000.0,
                    "sizes": [True],
                    "mean_free_path": 67e-9,
                    "boundary_layer_thickness": 1e-3,
                }
            },
        )
        with pytest.raises(TypeError, match=r"particles\.sizes\[0\].*number"):
            SimConfig(path)

    def test_obstacles_not_list_raises(self, tmp_path) -> None:
        """Obstacles section as a mapping raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={"obstacles": {"bad": "value"}},
        )
        with pytest.raises(ValueError, match="obstacles must be a list"):
            SimConfig(path)

    def test_wall_boundary_ignores_velocity(self, tmp_path) -> None:
        """Wall boundary with invalid velocity value does not store it."""
        path = _write_config(
            tmp_path,
            overrides={
                "boundaries": {
                    "wall1": {
                        "type": "wall",
                        "location": "left",
                        "y_start": 0.0,
                        "y_end": 2.0,
                        "velocity": "fast",
                    }
                }
            },
        )
        config = SimConfig(path)
        assert config.boundaries["wall1"].velocity is None

    def test_hepa_diameters_unsorted_raises(self, tmp_path) -> None:
        """Unsorted HEPA reference diameters raise ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "particles": {
                    "density": 1000.0,
                    "sizes": [0.1e-6],
                    "mean_free_path": 67e-9,
                    "boundary_layer_thickness": 1e-3,
                    "hepa_reference": {
                        "diameters": [0.5e-6, 0.1e-6],
                        "efficiencies": [0.99990, 0.99999],
                    },
                }
            },
        )
        with pytest.raises(ValueError, match=r"sorted ascending"):
            SimConfig(path)

    def test_missing_hepa_reference_raises(self, tmp_path) -> None:
        """Missing hepa_reference section raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "particles": {
                    "density": 1000.0,
                    "sizes": [0.1e-6],
                    "mean_free_path": 67e-9,
                    "boundary_layer_thickness": 1e-3,
                }
            },
        )
        with pytest.raises(ValueError, match=r"particles\.hepa_reference"):
            SimConfig(path)

    def test_negative_threshold_raises(self, tmp_path) -> None:
        """Negative threshold value raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={"thresholds": {"0.5e-6": -100.0}},
        )
        with pytest.raises(ValueError, match=r"thresholds.*non-negative"):
            SimConfig(path)

    def test_bool_threshold_raises_type_error(self, tmp_path) -> None:
        """Boolean threshold value raises TypeError."""
        path = _write_config(
            tmp_path,
            overrides={"thresholds": {"0.5e-6": True}},
        )
        with pytest.raises(TypeError, match=r"thresholds.*number"):
            SimConfig(path)

    def test_alpha_velocity_zero_raises(self, tmp_path) -> None:
        """Alpha velocity of zero raises ValueError (causes division by zero)."""
        path = _write_config(
            tmp_path,
            overrides={
                "solver": {
                    "dt": 0.01,
                    "t_end": 60.0,
                    "output_interval": 10,
                    "convergence_tol": 1e-6,
                    "max_simple_iter": 500,
                    "alpha_velocity": 0.0,
                    "alpha_pressure": 0.3,
                    "max_pressure_iter": 5000,
                    "pressure_rtol": 1e-8,
                }
            },
        )
        with pytest.raises(ValueError, match=r"alpha_velocity.*\(0\.0, 1\.0\]"):
            SimConfig(path)

    def test_alpha_velocity_too_large_raises(self, tmp_path) -> None:
        """Alpha velocity above 1.0 raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "solver": {
                    "dt": 0.01,
                    "t_end": 60.0,
                    "output_interval": 10,
                    "convergence_tol": 1e-6,
                    "max_simple_iter": 500,
                    "alpha_velocity": 1.5,
                    "alpha_pressure": 0.3,
                    "max_pressure_iter": 5000,
                    "pressure_rtol": 1e-8,
                }
            },
        )
        with pytest.raises(ValueError, match=r"alpha_velocity.*\(0\.0, 1\.0\]"):
            SimConfig(path)

    def test_alpha_pressure_negative_raises(self, tmp_path) -> None:
        """Negative alpha pressure raises ValueError."""
        path = _write_config(
            tmp_path,
            overrides={
                "solver": {
                    "dt": 0.01,
                    "t_end": 60.0,
                    "output_interval": 10,
                    "convergence_tol": 1e-6,
                    "max_simple_iter": 500,
                    "alpha_velocity": 0.7,
                    "alpha_pressure": -0.1,
                    "max_pressure_iter": 5000,
                    "pressure_rtol": 1e-8,
                }
            },
        )
        with pytest.raises(ValueError, match=r"alpha_pressure.*\(0\.0, 1\.0\]"):
            SimConfig(path)

    def test_alpha_bool_raises_type_error(self, tmp_path) -> None:
        """Boolean alpha value raises TypeError."""
        path = _write_config(
            tmp_path,
            overrides={
                "solver": {
                    "dt": 0.01,
                    "t_end": 60.0,
                    "output_interval": 10,
                    "convergence_tol": 1e-6,
                    "max_simple_iter": 500,
                    "alpha_velocity": True,
                    "alpha_pressure": 0.3,
                    "max_pressure_iter": 5000,
                    "pressure_rtol": 1e-8,
                }
            },
        )
        with pytest.raises(TypeError, match=r"alpha_velocity.*number"):
            SimConfig(path)

    def test_alpha_velocity_at_one_accepted(self, tmp_path) -> None:
        """Alpha velocity of exactly 1.0 is valid (no under-relaxation)."""
        path = _write_config(
            tmp_path,
            overrides={
                "solver": {
                    "dt": 0.01,
                    "t_end": 60.0,
                    "output_interval": 10,
                    "convergence_tol": 1e-6,
                    "max_simple_iter": 500,
                    "alpha_velocity": 1.0,
                    "alpha_pressure": 0.3,
                    "max_pressure_iter": 5000,
                    "pressure_rtol": 1e-8,
                }
            },
        )
        config = SimConfig(path)
        assert config.alpha_velocity == 1.0

    def test_missing_file_raises(self) -> None:
        """Non-existent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SimConfig("nonexistent_config.yaml")

    def test_non_mapping_yaml_raises(self, tmp_path) -> None:
        """YAML file containing a list instead of mapping raises ValueError."""
        path = tmp_path / "config.yaml"
        path.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            SimConfig(str(path))


@pytest.mark.unit
class TestFromDict:
    """SimConfig.from_dict applies the file constructor's validation to a mapping."""

    def test_matches_the_file_constructor(self, tmp_path) -> None:
        """Loading a file and loading its parsed mapping give equal attributes."""
        path = _write_config(tmp_path)
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        from_file = SimConfig(path)
        from_mapping = SimConfig.from_dict(raw)
        for name in ("nx", "ny", "rho", "mu", "alpha_velocity", "max_pressure_iter"):
            assert getattr(from_mapping, name) == getattr(from_file, name)

    def test_rejects_non_mapping(self) -> None:
        with pytest.raises(ValueError, match="mapping"):
            SimConfig.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_validates_like_the_file_constructor(self, tmp_path) -> None:
        """An invalid value fails in from_dict exactly as it fails from a file."""
        path = _write_config(tmp_path)
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["fluid"]["viscosity"] = -1.0
        with pytest.raises(ValueError):
            SimConfig.from_dict(raw)


@pytest.mark.unit
class TestMeshStretching:
    """The optional mesh section defaults to uniform and validates each axis."""

    def _raw(self, tmp_path, mesh) -> dict:
        path = _write_config(tmp_path)
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if mesh is not None:
            raw["mesh"] = mesh
        return raw

    def test_absent_section_is_uniform(self, tmp_path) -> None:
        config = SimConfig(_write_config(tmp_path))
        assert config.stretch_x.ratio == 1.0 and config.stretch_x.min_spacing is None
        assert config.stretch_y.ratio == 1.0 and config.stretch_y.min_spacing is None

    def test_ratio_is_read_per_axis(self, tmp_path) -> None:
        config = SimConfig.from_dict(
            self._raw(
                tmp_path, {"x": {"stretch_ratio": 1.05}, "y": {"stretch_ratio": 1.2}}
            )
        )
        assert config.stretch_x.ratio == 1.05 and config.stretch_y.ratio == 1.2

    def test_min_wall_spacing_is_read(self, tmp_path) -> None:
        config = SimConfig.from_dict(
            self._raw(tmp_path, {"y": {"min_wall_spacing": 0.01}})
        )
        assert config.stretch_y.min_spacing == 0.01
        assert config.stretch_x.ratio == 1.0

    def test_ratio_below_one_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="stretch_ratio must be >= 1"):
            SimConfig.from_dict(self._raw(tmp_path, {"x": {"stretch_ratio": 0.9}}))

    def test_both_quantities_raise(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="not both"):
            SimConfig.from_dict(
                self._raw(
                    tmp_path, {"x": {"stretch_ratio": 1.1, "min_wall_spacing": 0.01}}
                )
            )

    def test_spacing_above_uniform_raises(self, tmp_path) -> None:
        # base config: width 4.0, nx 80, uniform 0.05
        with pytest.raises(ValueError, match="must not exceed the uniform spacing"):
            SimConfig.from_dict(self._raw(tmp_path, {"x": {"min_wall_spacing": 0.06}}))

    @pytest.mark.parametrize("bad", [0, -0.01, "0.01", True])
    def test_non_positive_or_non_numeric_spacing_raises(self, tmp_path, bad) -> None:
        with pytest.raises((TypeError, ValueError)):
            SimConfig.from_dict(self._raw(tmp_path, {"x": {"min_wall_spacing": bad}}))

    def test_unknown_axis_or_key_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="not a recognised axis"):
            SimConfig.from_dict(self._raw(tmp_path, {"z": {"stretch_ratio": 1.1}}))
        with pytest.raises(ValueError, match="not recognised"):
            SimConfig.from_dict(self._raw(tmp_path, {"x": {"ratio": 1.1}}))


@pytest.mark.unit
class TestMomentumSweepsKey:
    """solver.momentum_sweeps (ECR-002 step 4): optional, a positive int, default 1."""

    def _raw(self, tmp_path: Path, **keys: object) -> dict:
        with open(_write_config(tmp_path), encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["solver"].update(keys)
        return raw

    def test_absent_gives_one_and_a_given_count_is_read(self, tmp_path: Path) -> None:
        """No key is the one sweep every stored result used; ten is read as ten."""
        assert SimConfig(_write_config(tmp_path)).momentum_sweeps == 1
        given = SimConfig.from_dict(self._raw(tmp_path, momentum_sweeps=10))
        assert given.momentum_sweeps == 10

    def test_it_is_a_solver_key_after_the_required_ones(self) -> None:
        """The harness records it from SOLVER_KEYS, directly after pressure_rtol."""
        index = SOLVER_KEYS.index("momentum_sweeps")
        assert SOLVER_KEYS[index - 1] == "pressure_rtol"

    @pytest.mark.parametrize(
        ("bad", "error"),
        [
            (0, ValueError),
            (-3, ValueError),
            (True, TypeError),
            (2.0, TypeError),
            ("10", TypeError),
            (None, TypeError),
        ],
    )
    def test_bad_values_are_refused(
        self, tmp_path: Path, bad: object, error: type[Exception]
    ) -> None:
        """Not positive, a bool, a float, a string or null is refused, naming the key."""
        with pytest.raises(error, match=r"solver\.momentum_sweeps"):
            SimConfig.from_dict(self._raw(tmp_path, momentum_sweeps=bad))


@pytest.mark.unit
class TestCheckpointIntervalKey:
    """solver.checkpoint_interval (ADR-014 D, step 3): optional, a positive int."""

    def _raw(self, tmp_path: Path, **keys: object) -> dict:
        with open(_write_config(tmp_path), encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["solver"].update(keys)
        return raw

    def test_absent_gives_none_and_a_given_interval_is_read(
        self, tmp_path: Path
    ) -> None:
        """No key writes no periodic checkpoint; 25 is read as 25.

        Defect caught: a default interval in code, which would write
        checkpoints in every run that never asked for them.
        """
        assert SimConfig(_write_config(tmp_path)).checkpoint_interval is None
        given = SimConfig.from_dict(self._raw(tmp_path, checkpoint_interval=25))
        assert given.checkpoint_interval == 25

    def test_it_is_the_last_solver_key(self) -> None:
        """The harness records it from SOLVER_KEYS, after the earlier optional keys."""
        assert SOLVER_KEYS[-1] == "checkpoint_interval"

    @pytest.mark.parametrize(
        ("bad", "error"),
        [
            (0, ValueError),
            (-5, ValueError),
            (True, TypeError),
            (10.0, TypeError),
            ("10", TypeError),
            (None, TypeError),
        ],
    )
    def test_bad_values_are_refused(
        self, tmp_path: Path, bad: object, error: type[Exception]
    ) -> None:
        """Not positive, a bool, a float, a string or null is refused, naming the key.

        Defect caught: the key read with ``_optional_positive_int``'s
        default path for null, or without the bool guard.
        """
        with pytest.raises(error, match=r"solver\.checkpoint_interval"):
            SimConfig.from_dict(self._raw(tmp_path, checkpoint_interval=bad))


@pytest.mark.unit
class TestStoppingRuleKeys:
    """The optional stopping keys default and validate; unknown solver keys are refused."""

    def _raw(self, tmp_path: Path, **keys: object) -> dict:
        with open(_write_config(tmp_path), encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["solver"].update(keys)
        return raw

    def test_absent_keys_give_the_defaults_and_present_ones_are_read(
        self, tmp_path: Path
    ) -> None:
        """Absent keys give velocity_step, 1e-6 and 1e-10; given ones are read as given."""
        keys = ("stopping_rule", "iteration_error_tol", "mass_imbalance_tol")
        absent = SimConfig(_write_config(tmp_path))
        assert [getattr(absent, k) for k in keys] == ["velocity_step", 1e-6, 1e-10]
        given = ["error_estimate", 1e-7, 1e-12]
        present = SimConfig.from_dict(
            self._raw(tmp_path, **dict(zip(keys, given, strict=True)))
        )
        assert [getattr(present, k) for k in keys] == given

    @pytest.mark.parametrize(
        ("key", "bad", "error"),
        [
            ("stopping_rule", "residual", ValueError),
            ("stopping_rule", 1, TypeError),
            ("iteration_error_tol", 0.0, ValueError),
            ("iteration_error_tol", -1e-6, ValueError),
            ("iteration_error_tol", True, TypeError),
            ("mass_imbalance_tol", 0, ValueError),
            ("mass_imbalance_tol", -1e-10, ValueError),
            ("mass_imbalance_tol", "1e-10", TypeError),
        ],
    )
    def test_bad_values_are_rejected(
        self, tmp_path: Path, key: str, bad: object, error: type[Exception]
    ) -> None:
        """An unknown rule, a rule or tolerance of the wrong type, or one not positive is refused."""
        with pytest.raises(error, match=f"solver.{key}"):
            SimConfig.from_dict(self._raw(tmp_path, **{key: bad}))

    def test_an_unknown_solver_key_is_rejected(self, tmp_path: Path) -> None:
        """Review 24 B1: a misspelt optional key would otherwise run its default."""
        with pytest.raises(
            ValueError, match=r"solver\.stoping_rule is not a recognised"
        ):
            SimConfig.from_dict(self._raw(tmp_path, stoping_rule="error_estimate"))


@pytest.mark.unit
class TestPressureKeys:
    """The pressure correction's keys (REQ-S08 as amended 2026-10-06, ADR-013 D)."""

    def _raw(self, tmp_path: Path, **keys: object) -> dict:
        with open(_write_config(tmp_path), encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["solver"].update(keys)
        return raw

    def test_pressure_rtol_is_read_and_its_bounds_are_inclusive_exclusive(
        self, tmp_path: Path
    ) -> None:
        """A value in [1e-10, 1) is read as given; the lower bound is accepted."""
        for value in (1e-10, 1e-8, 0.5):
            config = SimConfig.from_dict(self._raw(tmp_path, pressure_rtol=value))
            assert config.pressure_rtol == value
        assert PRESSURE_RTOL_BOUNDS == (1e-10, 1.0)

    @pytest.mark.parametrize(
        ("bad", "error", "message"),
        [
            (1.0, ValueError, r"solver\.pressure_rtol must be in \[1e-10, 1\.0\)"),
            (2.0, ValueError, r"solver\.pressure_rtol must be in"),
            (9.9e-11, ValueError, r"solver\.pressure_rtol must be in"),
            (1e-12, ValueError, r"solver\.pressure_rtol must be in"),
            (0.0, ValueError, r"solver\.pressure_rtol must be positive"),
            (-1e-8, ValueError, r"solver\.pressure_rtol must be positive"),
            (float("nan"), ValueError, r"solver\.pressure_rtol must be finite"),
            (True, TypeError, r"solver\.pressure_rtol must be a number"),
            ("1e-8", TypeError, r"solver\.pressure_rtol must be a number"),
        ],
    )
    def test_bad_pressure_rtol_is_rejected(
        self, tmp_path: Path, bad: object, error: type[Exception], message: str
    ) -> None:
        """Out of range, non-positive, non-finite, bool or string: each refused by name."""
        with pytest.raises(error, match=message):
            SimConfig.from_dict(self._raw(tmp_path, pressure_rtol=bad))

    def test_missing_pressure_rtol_or_cap_is_refused(self, tmp_path: Path) -> None:
        """Both keys stay required; neither has a default (ADR-013 decision 4)."""
        for key in ("pressure_rtol", "max_pressure_iter"):
            raw = self._raw(tmp_path)
            del raw["solver"][key]
            with pytest.raises(
                ValueError, match=f"Missing required key: 'solver.{key}'"
            ):
                SimConfig.from_dict(raw)

    def test_the_retired_pressure_tol_is_refused_naming_the_new_key(
        self, tmp_path: Path
    ) -> None:
        """A Jacobi-era file cannot be read with its pascals as a relative residual.

        Refused whether or not pressure_rtol is also given, with a message
        that names pressure_rtol, distinct from the unknown-key refusal.
        """
        retired = r"solver\.pressure_tol was the weighted Jacobi.*pressure_rtol"
        with pytest.raises(ValueError, match=retired) as refused:
            SimConfig.from_dict(self._raw(tmp_path, pressure_tol=1e-6))
        # The range in the message is the one the range check enforces (review
        # 37 S6: it was written out by hand beside the constant).
        lower, upper = PRESSURE_RTOL_BOUNDS
        assert f"pressure_rtol, in [{lower}, {upper})," in str(refused.value)
        raw = self._raw(tmp_path, pressure_tol=1e-6)
        del raw["solver"]["pressure_rtol"]
        with pytest.raises(ValueError, match=retired):
            SimConfig.from_dict(raw)
        assert "pressure_tol" not in SOLVER_KEYS
        assert RETIRED_PRESSURE_TOL_KEY == "pressure_tol"

    def test_solver_keys_is_the_one_ordered_list(self) -> None:
        """Fifteen keys, nine required then six optional, pressure_rtol among them."""
        assert isinstance(SOLVER_KEYS, tuple)
        assert len(SOLVER_KEYS) == len(set(SOLVER_KEYS)) == 15
        assert "pressure_rtol" in SOLVER_KEYS
        assert SOLVER_KEYS[-5:] == (
            "stopping_rule",
            "iteration_error_tol",
            "mass_imbalance_tol",
            "pressure_preconditioner",
            "checkpoint_interval",
        )

    def test_pressure_preconditioner_defaults_to_multigrid_and_reads_either_choice(
        self, tmp_path: Path
    ) -> None:
        """Absent is multigrid; each named choice, the diagonal included, is read.

        Prompt 53 built the multigrid preconditioner as an option and prompt
        54 made it the default. Defect caught: the default left at jacobi.
        """
        assert PRESSURE_PRECONDITIONERS == (PRESSURE_JACOBI, MULTIGRID)
        assert (PRESSURE_JACOBI, MULTIGRID) == ("jacobi", "multigrid")
        assert DEFAULT_PRESSURE_PRECONDITIONER == "multigrid"
        assert SimConfig(_write_config(tmp_path)).pressure_preconditioner == "multigrid"
        for choice in PRESSURE_PRECONDITIONERS:
            raw = self._raw(tmp_path, pressure_preconditioner=choice)
            assert SimConfig.from_dict(raw).pressure_preconditioner == choice

    @pytest.mark.parametrize(
        ("bad", "error", "message"),
        [
            ("amg", ValueError, r"solver\.pressure_preconditioner must be one of"),
            (
                "Multigrid",
                ValueError,
                r"solver\.pressure_preconditioner must be one of",
            ),
            ("", ValueError, r"solver\.pressure_preconditioner must be one of"),
            (True, TypeError, r"solver\.pressure_preconditioner must be a string"),
            (1, TypeError, r"solver\.pressure_preconditioner must be a string"),
            (None, TypeError, r"solver\.pressure_preconditioner must be a string"),
        ],
    )
    def test_bad_pressure_preconditioner_is_rejected(
        self, tmp_path: Path, bad: object, error: type[Exception], message: str
    ) -> None:
        """An unknown name, a case variant, a bool, a number or null: each refused by name."""
        with pytest.raises(error, match=message):
            SimConfig.from_dict(self._raw(tmp_path, pressure_preconditioner=bad))

    def test_every_committed_configuration_names_the_new_key_and_cap(self) -> None:
        """Every YAML file and the transport stand-ins carry the key and the cap.

        The laminar cases carry 1e-8; the Annex 20 case (VAL-017, prompt 49),
        the backward-facing step (VAL-019, prompt 52), the impinging jet
        (VAL-020, prompt 55) and, since step 8 (prompt 56), the product file
        and its two layouts (tests/test_layouts.py), and since step 0b of ADR-014
        (prompt 61) the vented room, carry the turbulent
        product setting, 1e-4 (ADR-013 decision 3 as amended 2026-10-09;
        ECR-002 section 8, step 8).
        """
        turbulent = {
            "clean_room_default.yaml": 1e-4,
            "clean_room_l1_return3_closed.yaml": 1e-4,
            "clean_room_l2_left_supply.yaml": 1e-4,
            "clean_room_vents.yaml": 1e-4,
            "scenario_clean_out.yaml": 1e-4,
            "scenario_door_leak.yaml": 1e-4,
            "scenario_equipment_dust.yaml": 1e-4,
            "scenario_filter_breach.yaml": 1e-4,
            "validation_annex20.yaml": 1e-4,
            "validation_backstep.yaml": 1e-4,
            "validation_impinging_jet.yaml": 1e-4,
        }
        for path in sorted(CONFIG_DIR.glob("*.yaml")):
            config = SimConfig(path)
            assert config.pressure_rtol == turbulent.get(path.name, 1e-8), path.name
            assert config.max_pressure_iter == 5000, path.name
        assert SOLVER_BLOCK["pressure_rtol"] == 1e-8
        assert SOLVER_BLOCK["max_pressure_iter"] == 5000


@pytest.mark.unit
class TestTransportSection:
    """The optional transport section (ADR-011 I): defaults, ranges, types, unknown keys."""

    def _raw(self, tmp_path: Path, transport: object = None, **keys: object) -> dict:
        with open(_write_config(tmp_path), encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if transport is None:
            transport = {
                "cfl_number": 0.1,
                "max_diffusion_iter": 200,
                "diffusion_tol": 1e-8,
            } | keys
        raw["transport"] = transport
        return raw

    def test_absent_section_gives_none(self, tmp_path: Path) -> None:
        """The velocity-only validation cases carry no transport block."""
        assert SimConfig(_write_config(tmp_path)).transport is None
        assert load_case("poiseuille").transport is None
        assert load_case("cavity").transport is None

    def test_present_section_is_read_with_umist_the_default(
        self, tmp_path: Path
    ) -> None:
        spec = SimConfig.from_dict(self._raw(tmp_path)).transport
        assert spec == TransportSpec(
            cfl_number=0.1,
            advection_scheme="umist",
            max_diffusion_iter=200,
            diffusion_tol=1e-8,
        )
        given = SimConfig.from_dict(
            self._raw(tmp_path, advection_scheme="upwind", cfl_number=CFL_NUMBER_BOUND)
        ).transport
        assert (given.advection_scheme, given.cfl_number) == ("upwind", 0.5)

    @pytest.mark.parametrize(
        ("key", "bad", "error"),
        [
            ("cfl_number", 0.0, ValueError),
            ("cfl_number", -0.1, ValueError),
            ("cfl_number", 0.5000001, ValueError),
            ("cfl_number", 1.0, ValueError),
            ("cfl_number", True, TypeError),
            ("cfl_number", "0.1", TypeError),
            ("advection_scheme", "quick", ValueError),
            ("advection_scheme", 1, TypeError),
            ("max_diffusion_iter", 0, ValueError),
            ("max_diffusion_iter", -5, ValueError),
            ("max_diffusion_iter", 2.5, TypeError),
            ("max_diffusion_iter", True, TypeError),
            ("diffusion_tol", 0.0, ValueError),
            ("diffusion_tol", -1e-8, ValueError),
            ("diffusion_tol", False, TypeError),
        ],
    )
    def test_bad_values_are_rejected(
        self, tmp_path: Path, key: str, bad: object, error: type[Exception]
    ) -> None:
        """Out of range, the wrong type, or a bool where a number is expected."""
        with pytest.raises(error, match=f"transport.{key}"):
            SimConfig.from_dict(self._raw(tmp_path, **{key: bad}))

    @pytest.mark.parametrize(
        "key", ["cfl_number", "max_diffusion_iter", "diffusion_tol"]
    )
    def test_missing_required_key_raises(self, tmp_path: Path, key: str) -> None:
        raw = self._raw(tmp_path)
        del raw["transport"][key]
        with pytest.raises(ValueError, match=f"transport.{key}"):
            SimConfig.from_dict(raw)

    def test_unknown_key_and_non_mapping_are_rejected(self, tmp_path: Path) -> None:
        """A misspelt key would otherwise run its default, as review 24 B1 found for solver."""
        with pytest.raises(
            ValueError, match=r"transport\.cfl_numbre is not a recognised"
        ):
            SimConfig.from_dict(self._raw(tmp_path, cfl_numbre=0.1))
        with pytest.raises(ValueError, match="transport must be a mapping"):
            SimConfig.from_dict(self._raw(tmp_path, transport=[0.1]))

    def test_turbulent_schmidt_absent_is_none_with_no_default(
        self, tmp_path: Path
    ) -> None:
        """REQ-T13: the key is optional and nothing in code stands in for it."""
        spec = SimConfig.from_dict(self._raw(tmp_path)).transport
        assert spec.turbulent_schmidt is None

    @pytest.mark.parametrize(
        ("given", "expected"),
        [(0.7, 0.7), (1, 1.0), (0.2, 0.2), (1.3, 1.3), (50, 50.0)],
    )
    def test_turbulent_schmidt_valid_values_are_read_as_floats(
        self, tmp_path: Path, given: object, expected: float
    ) -> None:
        """Any positive finite number is read; the literature range is not enforced."""
        spec = SimConfig.from_dict(
            self._raw(tmp_path, turbulent_schmidt=given)
        ).transport
        assert spec.turbulent_schmidt == expected
        assert isinstance(spec.turbulent_schmidt, float)

    @pytest.mark.parametrize(
        ("bad", "error"),
        [
            (0.0, ValueError),
            (-0.7, ValueError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (True, TypeError),
            (False, TypeError),
            ("0.7", TypeError),
            (None, TypeError),
        ],
    )
    def test_turbulent_schmidt_bad_values_are_rejected(
        self, tmp_path: Path, bad: object, error: type[Exception]
    ) -> None:
        """Zero, negative, non-finite, bool, string and null are refused at load."""
        with pytest.raises(error, match=r"transport\.turbulent_schmidt"):
            SimConfig.from_dict(self._raw(tmp_path, turbulent_schmidt=bad))


# The keys a velocity inlet that admits air states with the model on.
INLET_TURBULENCE = {"turbulence_intensity": 0.05, "dissipation_length": 0.1}

TURBULENCE = {
    "model": "k_epsilon",
    "wall_treatment": "scalable_wall_functions",
    "cfl_number": 0.4,
    "alpha_turbulence": 0.7,
    "max_iter": 300,
    "tol": 1e-9,
}


@pytest.mark.unit
class TestTurbulenceSection:
    """The optional turbulence section (ADR-012 I, ECR-002 step 1): defaults, ranges, types, unknown keys."""

    def _raw(self, tmp_path: Path, turbulence: object = None, **keys: object) -> dict:
        """The base configuration with the section, as step 6 requires it.

        The model runs under the error_estimate rule only, and the supply,
        which admits air, states its turbulence keys.
        """
        with open(_write_config(tmp_path), encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["turbulence"] = TURBULENCE | keys if turbulence is None else turbulence
        raw["solver"]["stopping_rule"] = "error_estimate"
        raw["boundaries"]["hepa_supply"].update(INLET_TURBULENCE)
        return raw

    def test_absent_section_gives_none(self, tmp_path: Path) -> None:
        """Absent means the model is off: no laminar validation case carries one.

        The product configuration has carried the section since ECR-002 step
        8 (prompt 56); TestSimConfigValid pins what it says.
        """
        assert SimConfig(_write_config(tmp_path)).turbulence is None
        assert load_case("poiseuille").turbulence is None
        assert load_case("cavity").turbulence is None

    def test_present_section_is_read_with_standard_the_default(
        self, tmp_path: Path
    ) -> None:
        """A present section parses with the standard variant and upwind advection by default."""
        spec = SimConfig.from_dict(self._raw(tmp_path)).turbulence
        assert spec == TurbulenceSpec(
            model="k_epsilon",
            variant="standard",
            wall_treatment="scalable_wall_functions",
            cfl_number=0.4,
            advection_scheme="upwind",
            alpha_turbulence=0.7,
            max_iter=300,
            tol=1e-9,
        )
        given = SimConfig.from_dict(
            self._raw(
                tmp_path,
                variant="rng",
                cfl_number=CFL_NUMBER_BOUND,
                alpha_turbulence=1.0,
                max_iter=1,
            )
        ).turbulence
        assert (
            given.variant,
            given.cfl_number,
            given.alpha_turbulence,
            given.max_iter,
        ) == ("rng", 0.5, 1.0, 1)

    @pytest.mark.parametrize("scheme", ["upwind", "umist"])
    def test_the_advection_scheme_is_read_as_given(
        self, tmp_path: Path, scheme: str
    ) -> None:
        """Either value is read as stated (ADR-012 I, note of 2026-10-10)."""
        spec = SimConfig.from_dict(
            self._raw(tmp_path, advection_scheme=scheme)
        ).turbulence
        assert spec.advection_scheme == scheme

    def test_the_two_sections_advection_schemes_are_independent(
        self, tmp_path: Path
    ) -> None:
        """Absent from both sections, k and eps take upwind and the particles
        keep umist; a value in one section does not reach the other."""
        raw = self._raw(tmp_path)
        raw["transport"] = {
            "cfl_number": 0.1,
            "max_diffusion_iter": 200,
            "diffusion_tol": 1e-8,
        }
        config = SimConfig.from_dict(raw)
        assert (
            config.turbulence.advection_scheme,
            config.transport.advection_scheme,
        ) == (
            "upwind",
            "umist",
        )
        raw["turbulence"]["advection_scheme"] = "umist"
        raw["transport"]["advection_scheme"] = "upwind"
        config = SimConfig.from_dict(raw)
        assert (
            config.turbulence.advection_scheme,
            config.transport.advection_scheme,
        ) == (
            "umist",
            "upwind",
        )

    def test_the_section_changes_nothing_else_in_the_configuration(
        self, tmp_path: Path
    ) -> None:
        """Every other field is the same, but for the inlet keys the section requires.

        Changed in ECR-002 step 6 (prompt 45): with the section present the
        supply must state its turbulence keys, and without it may not, so the
        two configurations differ by those two fields of the supply alone.
        """
        raw = self._raw(tmp_path)
        with_section = vars(SimConfig.from_dict(raw))
        del raw["turbulence"]
        for key in INLET_TURBULENCE:
            del raw["boundaries"]["hepa_supply"][key]
        without = vars(SimConfig.from_dict(raw))
        assert with_section.pop("turbulence") is not None
        assert without.pop("turbulence") is None
        # The kept mapping is the mapping each loaded (prompt 64), so it
        # differs by the section and the supply's two keys by construction.
        assert "turbulence" in with_section.pop("_mapping")
        assert "turbulence" not in without.pop("_mapping")
        supply = with_section["boundaries"]["hepa_supply"]
        assert (supply.turbulence_intensity, supply.dissipation_length) == (0.05, 0.1)
        with_section["boundaries"] = {
            name: dataclasses.replace(
                spec, turbulence_intensity=None, dissipation_length=None
            )
            for name, spec in with_section["boundaries"].items()
        }
        assert with_section.keys() == without.keys()
        for key in without:
            assert with_section[key] == without[key], key

    @pytest.mark.parametrize(
        ("key", "bad", "error"),
        [
            ("model", "k_omega", ValueError),
            ("model", "K_EPSILON", ValueError),
            ("model", 1, TypeError),
            ("model", True, TypeError),
            ("variant", "realizable", ValueError),
            ("variant", None, TypeError),
            ("variant", True, TypeError),
            ("wall_treatment", "standard_wall_functions", ValueError),
            ("wall_treatment", 0, TypeError),
            ("cfl_number", 0.0, ValueError),
            ("cfl_number", -0.1, ValueError),
            ("cfl_number", 0.5000001, ValueError),
            ("cfl_number", 1.0, ValueError),
            ("cfl_number", True, TypeError),
            ("cfl_number", "0.4", TypeError),
            ("cfl_number", float("nan"), ValueError),
            ("advection_scheme", "quick", ValueError),
            ("advection_scheme", "UPWIND", ValueError),
            ("advection_scheme", "", ValueError),
            ("advection_scheme", 1, TypeError),
            ("advection_scheme", True, TypeError),
            ("advection_scheme", None, TypeError),
            ("alpha_turbulence", 0.0, ValueError),
            ("alpha_turbulence", -0.5, ValueError),
            ("alpha_turbulence", 1.0000001, ValueError),
            ("alpha_turbulence", False, TypeError),
            ("alpha_turbulence", float("inf"), ValueError),
            ("alpha_turbulence", float("nan"), ValueError),
            ("max_iter", 0, ValueError),
            ("max_iter", -3, ValueError),
            ("max_iter", 2.5, TypeError),
            ("max_iter", True, TypeError),
            ("max_iter", "300", TypeError),
            ("tol", 0.0, ValueError),
            ("tol", -1e-9, ValueError),
            ("tol", True, TypeError),
            ("tol", float("nan"), ValueError),
            ("tol", -float("inf"), ValueError),
        ],
    )
    def test_bad_values_are_rejected(
        self, tmp_path: Path, key: str, bad: object, error: type[Exception]
    ) -> None:
        """Out of range, the wrong type, a bool where a number is expected, or not finite."""
        with pytest.raises(error, match=f"turbulence.{key}"):
            SimConfig.from_dict(self._raw(tmp_path, **{key: bad}))

    @pytest.mark.parametrize(
        "key",
        [
            "model",
            "wall_treatment",
            "cfl_number",
            "alpha_turbulence",
            "max_iter",
            "tol",
        ],
    )
    def test_missing_required_key_raises(self, tmp_path: Path, key: str) -> None:
        raw = self._raw(tmp_path)
        del raw["turbulence"][key]
        with pytest.raises(ValueError, match=f"turbulence.{key}"):
            SimConfig.from_dict(raw)

    def test_unknown_key_and_non_mapping_are_rejected(self, tmp_path: Path) -> None:
        """A misspelt key would otherwise run its default; the model's constants are not keys."""
        with pytest.raises(
            ValueError, match=r"turbulence\.varient is not a recognised"
        ):
            SimConfig.from_dict(self._raw(tmp_path, varient="rng"))
        with pytest.raises(ValueError, match=r"turbulence\.c_mu is not a recognised"):
            SimConfig.from_dict(self._raw(tmp_path, c_mu=0.09))
        with pytest.raises(ValueError, match="turbulence must be a mapping"):
            SimConfig.from_dict(self._raw(tmp_path, turbulence=["k_epsilon"]))


@pytest.mark.unit
class TestInletTurbulenceKeys:
    """The inlet keys and the stopping rule with the model on (ECR-002 step 6, item 1)."""

    def _raw(self, tmp_path: Path, model: bool = True, **supply: object) -> dict:
        with open(_write_config(tmp_path), encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["solver"]["stopping_rule"] = "error_estimate"
        if model:
            raw["turbulence"] = dict(TURBULENCE)
        raw["boundaries"]["hepa_supply"].update(supply)
        return raw

    def test_an_inlet_that_admits_air_states_both(self, tmp_path: Path) -> None:
        config = SimConfig.from_dict(self._raw(tmp_path, **INLET_TURBULENCE))
        spec = config.boundaries["hepa_supply"]
        assert (spec.turbulence_intensity, spec.dissipation_length) == (0.05, 0.1)

    @pytest.mark.parametrize("key", sorted(INLET_TURBULENCE))
    def test_each_is_required_with_the_model_on(self, tmp_path: Path, key: str) -> None:
        given = {k: v for k, v in INLET_TURBULENCE.items() if k != key}
        with pytest.raises(ValueError, match=rf"hepa_supply\.{key}'.*admits air"):
            SimConfig.from_dict(self._raw(tmp_path, **given))

    @pytest.mark.parametrize("key", sorted(INLET_TURBULENCE))
    def test_each_is_refused_with_the_model_off(self, tmp_path: Path, key: str) -> None:
        raw = self._raw(tmp_path, model=False, **{key: INLET_TURBULENCE[key]})
        with pytest.raises(ValueError, match=rf"{key} is read only by the k-epsilon"):
            SimConfig.from_dict(raw)

    @pytest.mark.parametrize("key", sorted(INLET_TURBULENCE))
    def test_each_is_refused_on_an_inlet_with_zero_normal_velocity(
        self, tmp_path: Path, key: str
    ) -> None:
        """A moving wall admits no air; it needs neither key and takes neither."""
        raw = self._raw(tmp_path, **INLET_TURBULENCE)
        raw["boundaries"]["belt"] = {
            "type": "velocity_inlet",
            "location": "bottom",
            "x_start": 2.0,
            "x_end": 4.0,
            "u_velocity": 0.3,
            "v_velocity": 0.0,
        }
        assert SimConfig.from_dict(raw).boundaries["belt"].turbulence_intensity is None
        raw["boundaries"]["belt"][key] = INLET_TURBULENCE[key]
        with pytest.raises(ValueError, match=rf"belt\.{key} is not valid.*zero"):
            SimConfig.from_dict(raw)

    @pytest.mark.parametrize("key", sorted(INLET_TURBULENCE))
    @pytest.mark.parametrize(
        "segment_type", ["wall", "pressure_outlet", "fixed_flow_outlet"]
    )
    def test_each_is_refused_on_another_segment_type(
        self, tmp_path: Path, segment_type: str, key: str
    ) -> None:
        """Every segment type but the inlet refuses each key (review 45 S6)."""
        raw = self._raw(tmp_path, **INLET_TURBULENCE)
        raw["boundaries"]["side"] = {
            "type": segment_type,
            "location": "left",
            "y_start": 0.0,
            "y_end": 3.0,
            key: INLET_TURBULENCE[key],
        }
        with pytest.raises(
            ValueError, match=rf"side\.{key} is only valid on a velocity_inlet"
        ):
            SimConfig.from_dict(raw)

    @pytest.mark.parametrize(
        ("key", "bad", "error"),
        [
            ("turbulence_intensity", 0.0, ValueError),
            ("turbulence_intensity", 1.0, ValueError),
            ("turbulence_intensity", -0.05, ValueError),
            ("turbulence_intensity", 5.0, ValueError),
            ("turbulence_intensity", float("nan"), ValueError),
            ("turbulence_intensity", True, TypeError),
            ("turbulence_intensity", "0.05", TypeError),
            ("turbulence_intensity", None, TypeError),
            ("dissipation_length", 0.0, ValueError),
            ("dissipation_length", -0.1, ValueError),
            ("dissipation_length", float("inf"), ValueError),
            ("dissipation_length", False, TypeError),
            ("dissipation_length", None, TypeError),
        ],
    )
    def test_bad_values_are_rejected(
        self, tmp_path: Path, key: str, bad: object, error: type[Exception]
    ) -> None:
        keys = INLET_TURBULENCE | {key: bad}
        with pytest.raises(error, match=key):
            SimConfig.from_dict(self._raw(tmp_path, **keys))

    @pytest.mark.parametrize("rule", ["velocity_step", None])
    def test_the_velocity_step_rule_is_refused_with_the_model_on(
        self, tmp_path: Path, rule: str | None
    ) -> None:
        """Named or by default: the rule with condition (e) scores the turbulent cases."""
        raw = self._raw(tmp_path, **INLET_TURBULENCE)
        if rule is None:
            del raw["solver"]["stopping_rule"]
        else:
            raw["solver"]["stopping_rule"] = rule
        with pytest.raises(ValueError, match=r"stopping_rule must be 'error_estimate'"):
            SimConfig.from_dict(raw)


WALL_SEGMENT = {"type": "wall", "location": "left", "y_start": 0.0, "y_end": 3.0}
OUTLET_SEGMENT = {
    "type": "pressure_outlet",
    "location": "bottom",
    "x_start": 0.0,
    "x_end": 1.0,
}


@pytest.mark.unit
class TestSegmentKeys:
    """concentration, hepa_filtered and deposition_surface on a segment (ADR-011 E)."""

    def _raw(self, tmp_path: Path, segment: str, **keys: object) -> dict:
        with open(_write_config(tmp_path), encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["boundaries"]["door"] = dict(WALL_SEGMENT)
        raw["boundaries"]["floor_return"] = dict(OUTLET_SEGMENT)
        raw["boundaries"][segment].update(keys)
        return raw

    def test_absent_keys_give_the_defaults(self, tmp_path: Path) -> None:
        """No concentration (a clean supply), not filtered, the edge decides the surface."""
        config = SimConfig.from_dict(self._raw(tmp_path, "door"))
        for spec in config.boundaries.values():
            assert spec.concentration is None
            assert spec.hepa_filtered is False
            assert spec.deposition_surface is None

    def test_present_keys_are_read(self, tmp_path: Path) -> None:
        raw = self._raw(
            tmp_path,
            "hepa_supply",
            concentration=[1, 2.5, 0, 4, 5e3],
            hepa_filtered=True,
        )
        raw["boundaries"]["door"]["deposition_surface"] = "none"
        config = SimConfig.from_dict(raw)
        supply = config.boundaries["hepa_supply"]
        assert supply.concentration == (1.0, 2.5, 0.0, 4.0, 5000.0)
        assert isinstance(supply.concentration, tuple)
        assert all(isinstance(c, float) for c in supply.concentration)
        assert supply.hepa_filtered is True
        assert config.boundaries["door"].deposition_surface == "none"
        for surface in ("floor", "ceiling", "wall"):
            raw["boundaries"]["door"]["deposition_surface"] = surface
            assert (
                SimConfig.from_dict(raw).boundaries["door"].deposition_surface
                == surface
            )

    @pytest.mark.parametrize(
        ("segment", "key", "value"),
        [
            ("door", "concentration", [1, 2, 3, 4, 5]),
            ("floor_return", "concentration", [1, 2, 3, 4, 5]),
            ("door", "hepa_filtered", True),
            ("floor_return", "hepa_filtered", False),
            ("hepa_supply", "deposition_surface", "floor"),
            ("floor_return", "deposition_surface", "none"),
        ],
    )
    def test_a_key_on_the_wrong_segment_type_raises_naming_the_segment(
        self, tmp_path: Path, segment: str, key: str, value: object
    ) -> None:
        """Each key is valid on one segment type; elsewhere it is an error, not ignored."""
        with pytest.raises(
            ValueError, match=rf"boundaries\.{segment}\.{key} is only valid"
        ):
            SimConfig.from_dict(self._raw(tmp_path, segment, **{key: value}))

    @pytest.mark.parametrize(
        ("segment", "key", "bad", "error"),
        [
            ("hepa_supply", "concentration", [1, 2, 3, 4], ValueError),
            ("hepa_supply", "concentration", [1, 2, 3, 4, 5, 6], ValueError),
            ("hepa_supply", "concentration", [], ValueError),
            ("hepa_supply", "concentration", 5.0, TypeError),
            ("hepa_supply", "concentration", [1, 2, -3, 4, 5], ValueError),
            ("hepa_supply", "concentration", [1, 2, True, 4, 5], TypeError),
            ("hepa_supply", "concentration", [1, 2, "3", 4, 5], TypeError),
            ("hepa_supply", "hepa_filtered", 1, TypeError),
            ("hepa_supply", "hepa_filtered", "true", TypeError),
            ("door", "deposition_surface", "roof", ValueError),
            ("door", "deposition_surface", 3, TypeError),
            ("door", "deposition_surface", True, TypeError),
        ],
    )
    def test_bad_values_are_rejected(
        self,
        tmp_path: Path,
        segment: str,
        key: str,
        bad: object,
        error: type[Exception],
    ) -> None:
        """A list of the wrong length, a bool or string where a number is expected, a negative value, a non-bool flag, an unknown surface."""
        with pytest.raises(error, match=rf"boundaries\.{segment}\.{key}"):
            SimConfig.from_dict(self._raw(tmp_path, segment, **{key: bad}))

    def test_validation_case_files_load_unchanged(self) -> None:
        """Neither validation case carries a transport block or a segment key."""
        for name in ("poiseuille", "cavity"):
            config = load_case(name)
            assert config.transport is None
            for spec in config.boundaries.values():
                assert (
                    spec.concentration,
                    spec.hepa_filtered,
                    spec.deposition_surface,
                ) == (
                    None,
                    False,
                    None,
                )


RULE_SUPPLY = {
    "type": "velocity_inlet",
    "location": "top",
    "x_start": 0.5,
    "x_end": 3.5,
    "velocity": 0.45,
}
RULE_LID = {
    "type": "velocity_inlet",
    "location": "top",
    "x_start": 0.0,
    "x_end": 4.0,
    "u_velocity": 1.0,
    "v_velocity": 0.0,
}


@pytest.mark.unit
class TestSegmentRules:
    """Decisions 1 to 3 of 2026-10-03: unknown keys, zero-normal inlets, overlap."""

    def _raw(self, tmp_path: Path, boundaries: dict) -> dict:
        with open(_write_config(tmp_path), encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["boundaries"] = boundaries
        return raw

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("hepa_filterd", True),
            ("concentrations", [1, 2, 3, 4, 5]),
            ("deposition_surfce", "none"),
        ],
    )
    def test_an_unknown_segment_key_raises_naming_the_segment(
        self, tmp_path: Path, key: str, value: object
    ) -> None:
        """A misspelt key would otherwise change the physics with no message (REQ-C02)."""
        raw = self._raw(tmp_path, {"hepa_supply": {**RULE_SUPPLY, key: value}})
        with pytest.raises(
            ValueError, match=rf"boundaries\.hepa_supply\.{key} is not a recognised"
        ):
            SimConfig.from_dict(raw)

    @pytest.mark.parametrize(
        ("key", "value"), [("concentration", [1, 2, 3, 4, 5]), ("hepa_filtered", True)]
    )
    def test_a_concentration_key_on_a_zero_normal_inlet_raises(
        self, tmp_path: Path, key: str, value: object
    ) -> None:
        """A tangential lid admits no air; the scalar layer treats it as a wall."""
        raw = self._raw(tmp_path, {"lid": {**RULE_LID, key: value}})
        with pytest.raises(
            ValueError,
            match=rf"boundaries\.lid\.{key} is not valid on a velocity_inlet whose",
        ):
            SimConfig.from_dict(raw)
        # The same keys on an oblique inlet with a nonzero normal are fine.
        oblique = {**RULE_LID, "v_velocity": -0.2, key: value}
        SimConfig.from_dict(self._raw(tmp_path, {"lid": oblique}))
        # And a one-component inlet whose only component is tangential is a lid.
        tangential = {k: v for k, v in RULE_LID.items() if k != "v_velocity"} | {
            key: value
        }
        with pytest.raises(ValueError, match="normal velocity is zero"):
            SimConfig.from_dict(self._raw(tmp_path, {"lid": tangential}))

    def test_overlapping_segments_on_one_edge_raise_naming_both(
        self, tmp_path: Path
    ) -> None:
        """Two conditions for one face: the file would say two things to two readers."""
        a = {**RULE_SUPPLY, "x_start": 0.25, "x_end": 0.85}
        b = {**RULE_SUPPLY, "x_start": 0.55, "x_end": 1.25}
        with pytest.raises(
            ValueError, match=r"boundaries\.a and boundaries\.b overlap on the top edge"
        ):
            SimConfig.from_dict(self._raw(tmp_path, {"a": a, "b": b}))
        wall = {"type": "wall", "location": "top", "x_start": 0.8, "x_end": 1.0}
        with pytest.raises(
            ValueError, match=r"boundaries\.a and boundaries\.door overlap"
        ):
            SimConfig.from_dict(self._raw(tmp_path, {"a": a, "door": wall}))
        left = {"type": "wall", "location": "left", "y_start": 0.0, "y_end": 2.0}
        hood = {
            "type": "pressure_outlet",
            "location": "left",
            "y_start": 1.0,
            "y_end": 3.0,
        }
        with pytest.raises(
            ValueError,
            match=r"boundaries\.left and boundaries\.hood overlap on the left edge",
        ):
            SimConfig.from_dict(self._raw(tmp_path, {"left": left, "hood": hood}))

    def test_touching_segments_and_other_edges_load(self, tmp_path: Path) -> None:
        """Ranges that meet at one coordinate are not an overlap; the first decides there."""
        a = {**RULE_SUPPLY, "x_start": 0.25, "x_end": 0.85}
        b = {**RULE_SUPPLY, "x_start": 0.85, "x_end": 1.25}
        bottom = {**RULE_SUPPLY, "location": "bottom", "x_start": 0.5, "x_end": 1.0}
        config = SimConfig.from_dict(self._raw(tmp_path, {"a": a, "b": b, "c": bottom}))
        assert list(config.boundaries) == ["a", "b", "c"]

    def test_the_committed_configurations_still_load(self) -> None:
        for path in (CONFIG_DIR / "clean_room_default.yaml",):
            SimConfig(path)
        for name in ("poiseuille", "cavity"):
            load_case(name)


@pytest.mark.unit
class TestFiniteNumbers:
    """NaN and infinity are rejected wherever a number is expected (review 31 S7)."""

    def _raw(self, tmp_path: Path) -> dict:
        with open(_write_config(tmp_path), encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["transport"] = {
            "cfl_number": 0.1,
            "max_diffusion_iter": 200,
            "diffusion_tol": 1e-8,
        }
        raw["boundaries"]["hepa_supply"]["concentration"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        return raw

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    @pytest.mark.parametrize(
        ("section", "key", "helper"),
        [
            (("transport",), "cfl_number", "_require_positive_float"),
            (("transport",), "diffusion_tol", "_require_positive_float"),
            (("domain",), "width", "_require_positive_float"),
            (("boundaries", "hepa_supply"), "x_start", "_require_float"),
            (("solver",), "alpha_velocity", "_require_relaxation_factor"),
        ],
    )
    def test_scalar_helpers_reject_non_finite(
        self, tmp_path: Path, section: tuple, key: str, helper: str, bad: float
    ) -> None:
        raw = self._raw(tmp_path)
        target = raw
        for part in section:
            target = target[part]
        target[key] = bad
        with pytest.raises(
            ValueError, match=rf"{'.'.join(section)}\.{key} must be finite"
        ):
            SimConfig.from_dict(raw)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    @pytest.mark.parametrize(
        "key", ["velocity", "u_velocity", "v_velocity", "thresholds"]
    )
    def test_velocity_and_threshold_keys_reject_non_finite(
        self, tmp_path: Path, key: str, bad: float
    ) -> None:
        """Test 31b B2: the four keys with inline checks go through the same helper."""
        raw = self._raw(tmp_path)
        supply = raw["boundaries"]["hepa_supply"]
        if key == "thresholds":
            label = "thresholds.0.5e-6"
            raw["thresholds"]["0.5e-6"] = bad
        elif key == "velocity":
            label = "boundaries.hepa_supply.velocity"
            supply["velocity"] = bad
        else:
            label = f"boundaries.hepa_supply.{key}"
            del supply["velocity"]
            supply["u_velocity"], supply["v_velocity"] = 0.1, -0.45
            supply[key] = bad
        with pytest.raises(ValueError, match=rf"{label} must be finite"):
            SimConfig.from_dict(raw)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    @pytest.mark.parametrize(
        ("section", "key"),
        [
            (("boundaries", "hepa_supply"), "concentration"),
            (("particles", "hepa_reference"), "efficiencies"),
            (("particles",), "sizes"),
        ],
        ids=[
            "concentration-float-list",
            "efficiencies-float-list",
            "sizes-positive-float-list",
        ],
    )
    def test_list_helpers_reject_non_finite_items(
        self, tmp_path: Path, section: tuple, key: str, bad: float
    ) -> None:
        raw = self._raw(tmp_path)
        target = raw
        for part in section:
            target = target[part]
        target[key][1] = bad
        with pytest.raises(
            ValueError, match=rf"{'.'.join(section)}\.{key}\[1\] must be finite"
        ):
            SimConfig.from_dict(raw)


# A room for the vent rules (ADR-014 J): a rack on the floor, a tool standing
# clear of it with a free bottom face, the supply at the top and two returns
# that share the remainder.
VENT_OBSTACLES = [
    {"name": "rack", "x_start": 1.0, "x_end": 2.0, "y_start": 0.0, "y_end": 1.0},
    {"name": "tool", "x_start": 2.5, "x_end": 3.0, "y_start": 0.5, "y_end": 1.5},
]
VENT_RETURNS = {
    "return_left": {
        "type": "fixed_flow_outlet",
        "location": "bottom",
        "x_start": 0.25,
        "x_end": 0.75,
    },
    "return_right": {
        "type": "fixed_flow_outlet",
        "location": "bottom",
        "x_start": 3.25,
        "x_end": 3.75,
    },
}
DISCHARGE = {
    "type": "velocity_inlet",
    "location": "obstacle",
    "obstacle": "rack",
    "face": "top",
    "x_start": 1.25,
    "x_end": 1.75,
    "velocity": 1.0,
}
INTAKE = {
    "type": "fixed_flow_outlet",
    "location": "obstacle",
    "obstacle": "rack",
    "face": "left",
    "y_start": 0.25,
    "y_end": 0.75,
    "velocity": 1.0,
}
EXHAUST = {
    "type": "fixed_flow_outlet",
    "location": "obstacle",
    "obstacle": "tool",
    "face": "right",
    "y_start": 0.75,
    "y_end": 1.25,
    "velocity": 0.5,
}


@pytest.mark.unit
class TestVents:
    """Equipment vents on obstacle faces (ADR-014 J): one test per rule at load."""

    def _raw(self, tmp_path: Path, **vents: dict) -> dict:
        with open(_write_config(tmp_path), encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["obstacles"] = [dict(o) for o in VENT_OBSTACLES]
        raw["boundaries"] = {
            "hepa_supply": dict(raw["boundaries"]["hepa_supply"]),
            **{k: dict(v) for k, v in VENT_RETURNS.items()},
            **{k: dict(v) for k, v in vents.items()},
        }
        return raw

    def _refused(
        self, raw: dict, match: str, error: type | tuple[type, ...] = ValueError
    ) -> None:
        with pytest.raises(error, match=match):
            SimConfig.from_dict(raw)

    def test_a_discharge_an_intake_and_an_exhaust_load(self, tmp_path: Path) -> None:
        raw = self._raw(
            tmp_path,
            fan_out={**DISCHARGE, "recirculate_from": "fan_in", "hepa_filtered": True},
            fan_in=INTAKE,
            exhaust=EXHAUST,
        )
        config = SimConfig.from_dict(raw)
        out = config.boundaries["fan_out"]
        assert (out.location, out.obstacle, out.face) == ("obstacle", "rack", "top")
        assert (out.x_start, out.x_end, out.y_start, out.y_end) == (
            1.25,
            1.75,
            None,
            None,
        )
        assert out.velocity == 1.0
        assert out.recirculate_from == "fan_in"
        assert out.hepa_filtered is True
        exhaust = config.boundaries["exhaust"]
        assert (exhaust.face, exhaust.y_start, exhaust.y_end) == ("right", 0.75, 1.25)
        assert exhaust.recirculate_from is None
        # A domain-edge segment carries none of the vent keys.
        supply = config.boundaries["hepa_supply"]
        assert (supply.obstacle, supply.face, supply.recirculate_from) == (None,) * 3

    def test_a_vent_on_a_floating_obstacles_bottom_face_loads(
        self, tmp_path: Path
    ) -> None:
        bottom = {k: v for k, v in EXHAUST.items() if k not in ("y_start", "y_end")} | {
            "face": "bottom",
            "x_start": 2.6,
            "x_end": 2.9,
        }
        config = SimConfig.from_dict(self._raw(tmp_path, exhaust=bottom))
        assert config.boundaries["exhaust"].face == "bottom"

    def test_an_unknown_obstacle_is_refused(self, tmp_path: Path) -> None:
        raw = self._raw(tmp_path, exhaust={**EXHAUST, "obstacle": "furnace"})
        self._refused(raw, r"exhaust\.obstacle names 'furnace', which no obstacle")

    def test_an_obstacle_name_two_obstacles_share_is_refused(
        self, tmp_path: Path
    ) -> None:
        raw = self._raw(tmp_path, exhaust=EXHAUST)
        raw["obstacles"][0]["name"] = "tool"
        self._refused(raw, r"exhaust\.obstacle names 'tool', which 2 obstacles share")

    @pytest.mark.parametrize("key", ["obstacle", "face"])
    def test_obstacle_and_face_are_required_on_a_vent(
        self, tmp_path: Path, key: str
    ) -> None:
        vent = {k: v for k, v in EXHAUST.items() if k != key}
        self._refused(
            self._raw(tmp_path, exhaust=vent),
            rf"Missing required key: 'boundaries\.exhaust\.{key}'",
        )

    @pytest.mark.parametrize("key", ["obstacle", "face"])
    def test_obstacle_and_face_must_be_strings(self, tmp_path: Path, key: str) -> None:
        raw = self._raw(tmp_path, exhaust={**EXHAUST, key: 3})
        self._refused(raw, rf"exhaust\.{key} must be a string", TypeError)

    def test_a_face_that_is_not_one_of_the_four_is_refused(
        self, tmp_path: Path
    ) -> None:
        raw = self._raw(tmp_path, exhaust={**EXHAUST, "face": "front"})
        self._refused(raw, r"exhaust\.face must be one of \['top', 'bottom'")

    @pytest.mark.parametrize(
        ("y_start", "y_end"), [(0.25, 0.75), (1.25, 1.75), (0.4, 1.6)]
    )
    def test_an_extent_beyond_the_face_is_refused(
        self, tmp_path: Path, y_start: float, y_end: float
    ) -> None:
        """The tool's right face spans y 0.5 to 1.5."""
        raw = self._raw(
            tmp_path, exhaust={**EXHAUST, "y_start": y_start, "y_end": y_end}
        )
        self._refused(raw, r"does not lie within the right face of 'tool', \[0\.5, 1")

    @pytest.mark.parametrize(("y_start", "y_end"), [(1.0, 1.0), (1.2, 0.8)])
    def test_an_empty_extent_is_refused(
        self, tmp_path: Path, y_start: float, y_end: float
    ) -> None:
        raw = self._raw(
            tmp_path, exhaust={**EXHAUST, "y_start": y_start, "y_end": y_end}
        )
        self._refused(raw, r"exhaust: y_start .* must be less than y_end")

    def test_a_face_on_the_domain_edge_is_refused(self, tmp_path: Path) -> None:
        """The rack stands on the floor: it has no bottom face to vent from."""
        vent = {**DISCHARGE, "face": "bottom"}
        self._refused(
            self._raw(tmp_path, fan_out=vent),
            r"fan_out: the bottom face of 'rack' lies on the domain edge",
        )

    def test_a_face_against_another_obstacle_is_refused(self, tmp_path: Path) -> None:
        raw = self._raw(tmp_path, fan_out=DISCHARGE)
        shelf = {
            "name": "shelf",
            "x_start": 1.5,
            "x_end": 2.0,
            "y_start": 1.0,
            "y_end": 1.2,
        }
        raw["obstacles"].append(shelf)
        self._refused(raw, r"obstacle 'shelf' stands against the top face of 'rack'")
        # Beside the extent, the same shelf leaves the vent's faces free.
        shelf["x_start"] = 1.75
        SimConfig.from_dict(raw)

    def test_two_vents_overlapping_on_one_face_are_refused(
        self, tmp_path: Path
    ) -> None:
        second = {**EXHAUST, "y_start": 1.0, "y_end": 1.5}
        raw = self._raw(tmp_path, exhaust=EXHAUST, second=second)
        self._refused(
            raw,
            r"boundaries\.exhaust and boundaries\.second overlap on the right face "
            r"of 'tool'",
        )
        # Meeting at a point, and the same extent on another face, are not overlaps.
        touching = {**second, "y_start": 1.25}
        SimConfig.from_dict(self._raw(tmp_path, exhaust=EXHAUST, second=touching))
        other_face = {**EXHAUST, "face": "left"}
        SimConfig.from_dict(self._raw(tmp_path, exhaust=EXHAUST, second=other_face))

    @pytest.mark.parametrize("bc_type", ["wall", "pressure_outlet"])
    def test_a_type_other_than_inlet_or_fixed_flow_is_refused(
        self, tmp_path: Path, bc_type: str
    ) -> None:
        vent = {k: v for k, v in EXHAUST.items() if k != "velocity"} | {"type": bc_type}
        self._refused(
            self._raw(tmp_path, exhaust=vent),
            r"exhaust\.type must be one of \['velocity_inlet', 'fixed_flow_outlet'\] "
            "on an obstacle face",
        )

    @pytest.mark.parametrize("vent", [DISCHARGE, EXHAUST], ids=["inlet", "fixed_flow"])
    def test_a_vent_without_a_velocity_is_refused(
        self, tmp_path: Path, vent: dict
    ) -> None:
        stated = {k: v for k, v in vent.items() if k != "velocity"}
        self._refused(
            self._raw(tmp_path, vent=stated),
            r"Missing required key: 'boundaries\.vent\.velocity'\. A vent states",
        )

    @pytest.mark.parametrize("key", ["u_velocity", "v_velocity"])
    def test_velocity_components_on_a_vent_are_refused(
        self, tmp_path: Path, key: str
    ) -> None:
        raw = self._raw(tmp_path, fan_out={**DISCHARGE, key: 0.3})
        self._refused(raw, rf"fan_out\.{key} is not valid on a vent")

    @pytest.mark.parametrize("velocity", [True, "1.0", float("nan")])
    def test_a_velocity_that_is_not_a_finite_number_is_refused(
        self, tmp_path: Path, velocity: object
    ) -> None:
        raw = self._raw(tmp_path, fan_out={**DISCHARGE, "velocity": velocity})
        self._refused(raw, r"fan_out\.velocity must be", (TypeError, ValueError))

    @pytest.mark.parametrize("vent", [DISCHARGE, EXHAUST], ids=["inlet", "fixed_flow"])
    def test_a_velocity_that_is_not_positive_is_refused(
        self, tmp_path: Path, vent: dict
    ) -> None:
        raw = self._raw(tmp_path, vent={**vent, "velocity": -0.5})
        self._refused(raw, r"vent\.velocity must be positive")

    def test_an_inlet_vent_states_its_turbulence_with_the_model_on(
        self, tmp_path: Path
    ) -> None:
        raw = self._raw(tmp_path, fan_out=DISCHARGE)
        raw["solver"]["stopping_rule"] = "error_estimate"
        raw["turbulence"] = dict(TURBULENCE)
        raw["boundaries"]["hepa_supply"].update(INLET_TURBULENCE)
        self._refused(raw, r"fan_out\.turbulence_intensity'.*admits air")
        raw["boundaries"]["fan_out"].update(INLET_TURBULENCE)
        spec = SimConfig.from_dict(raw).boundaries["fan_out"]
        assert spec.turbulence_intensity == 0.05

    @pytest.mark.parametrize(
        ("key", "value"),
        [("obstacle", "rack"), ("face", "top"), ("recirculate_from", "fan_in")],
    )
    def test_vent_keys_on_a_domain_edge_segment_are_refused(
        self, tmp_path: Path, key: str, value: str
    ) -> None:
        raw = self._raw(tmp_path)
        raw["boundaries"]["hepa_supply"][key] = value
        self._refused(
            raw,
            rf"hepa_supply\.{key} is valid only on a vent \(location 'obstacle'\), "
            "not on the top edge",
        )

    def test_the_other_axis_extent_on_a_vent_is_refused(self, tmp_path: Path) -> None:
        raw = self._raw(tmp_path, fan_out={**DISCHARGE, "y_start": 1.0})
        self._refused(raw, r"fan_out\.y_start is not valid on a vent on the obstacle's")

    def test_an_unknown_key_on_a_vent_is_refused(self, tmp_path: Path) -> None:
        raw = self._raw(tmp_path, fan_out={**DISCHARGE, "recirculate": "fan_in"})
        self._refused(raw, r"fan_out\.recirculate is not a recognised boundary key")

    def test_a_location_that_is_neither_an_edge_nor_obstacle_is_refused(
        self, tmp_path: Path
    ) -> None:
        raw = self._raw(tmp_path, fan_out={**DISCHARGE, "location": "rack"})
        self._refused(raw, r"fan_out\.location must be one of .* or 'obstacle'")

    def test_recirculate_from_naming_no_segment_is_refused(
        self, tmp_path: Path
    ) -> None:
        raw = self._raw(tmp_path, fan_out={**DISCHARGE, "recirculate_from": "fan_in"})
        self._refused(raw, r"recirculate_from names 'fan_in', which is not a boundary")

    @pytest.mark.parametrize("source", ["return_left", "other_out"])
    def test_recirculate_from_naming_no_fixed_flow_vent_is_refused(
        self, tmp_path: Path, source: str
    ) -> None:
        """A domain-edge return is not a vent; a second discharge is not an intake."""
        other_out = {
            k: v for k, v in DISCHARGE.items() if k not in ("x_start", "x_end")
        } | {"face": "right", "y_start": 0.25, "y_end": 0.75}
        raw = self._raw(
            tmp_path,
            fan_out={**DISCHARGE, "recirculate_from": source},
            other_out=other_out,
        )
        self._refused(raw, rf"names '{source}', which is not a fixed_flow_outlet vent")

    def test_recirculate_from_another_obstacle_is_refused(self, tmp_path: Path) -> None:
        raw = self._raw(
            tmp_path,
            fan_out={**DISCHARGE, "recirculate_from": "exhaust"},
            exhaust=EXHAUST,
        )
        self._refused(raw, r"names 'exhaust', a vent on 'tool', not on 'rack'")

    def test_recirculate_from_with_a_concentration_is_refused(
        self, tmp_path: Path
    ) -> None:
        vent = {**DISCHARGE, "recirculate_from": "fan_in", "concentration": [1.0] * 5}
        raw = self._raw(tmp_path, fan_out=vent, fan_in=INTAKE)
        self._refused(raw, r"fan_out\.concentration is not valid with recirculate_from")

    def test_recirculate_from_on_an_exhaust_is_refused(self, tmp_path: Path) -> None:
        raw = self._raw(tmp_path, fan_in={**INTAKE, "recirculate_from": "exhaust"})
        self._refused(
            raw, r"fan_in\.recirculate_from is only valid on a velocity_inlet segment"
        )

    def test_recirculate_from_must_be_a_string(self, tmp_path: Path) -> None:
        raw = self._raw(tmp_path, fan_out={**DISCHARGE, "recirculate_from": 1})
        self._refused(raw, r"fan_out\.recirculate_from must be a string", TypeError)

    def test_a_closed_box_of_equal_stated_flows_loads_and_an_unequal_one_is_refused(
        self, tmp_path: Path
    ) -> None:
        """With no return to take a remainder the stated flows must balance."""
        raw = self._raw(tmp_path)
        raw["boundaries"] = {"fan_out": dict(DISCHARGE), "fan_in": dict(INTAKE)}
        config = SimConfig.from_dict(raw)
        assert list(config.boundaries) == ["fan_out", "fan_in"]
        raw["boundaries"]["fan_in"]["velocity"] = 0.9
        self._refused(
            raw,
            r"no pressure_outlet, so nothing balances the flow: the stated outflow "
            r"0\.45 m\^2/s differs from the inflow 0\.5 m\^2/s",
        )


@pytest.mark.unit
def test_the_vents_file_is_the_default_room_with_three_vents() -> None:
    """configs/clean_room_vents.yaml (prompt 61): every other key the default file's."""
    with open(CONFIG_DIR / "clean_room_default.yaml", encoding="utf-8") as handle:
        default = yaml.safe_load(handle)
    with open(CONFIG_DIR / "clean_room_vents.yaml", encoding="utf-8") as handle:
        vented = yaml.safe_load(handle)
    assert {k: v for k, v in vented.items() if k != "boundaries"} == {
        k: v for k, v in default.items() if k != "boundaries"
    }
    added = {
        k: v for k, v in vented["boundaries"].items() if k not in default["boundaries"]
    }
    assert {
        k: v for k, v in vented["boundaries"].items() if k in default["boundaries"]
    } == (default["boundaries"])
    assert sorted(added) == ["etch_exhaust", "rack_discharge", "rack_intake"]
    config = SimConfig(CONFIG_DIR / "clean_room_vents.yaml")
    discharge = config.boundaries["rack_discharge"]
    assert (discharge.obstacle, discharge.face) == ("server_rack", "top")
    assert discharge.recirculate_from == "rack_intake"
    # The rack's discharge returns what its intake draws: equal flows.
    intake = config.boundaries["rack_intake"]
    assert discharge.velocity * (discharge.x_end - discharge.x_start) == pytest.approx(
        intake.velocity * (intake.y_end - intake.y_start), rel=1e-12
    )
    assert config.boundaries["etch_exhaust"].face == "right"


@pytest.mark.unit
def test_a_recirculating_pair_of_unequal_flow_is_refused(tmp_path: Path) -> None:
    """Review 61 S6: the discharge would create particles at the intake's mean."""
    raw = TestVents()._raw(
        tmp_path,
        fan_out={**DISCHARGE, "recirculate_from": "fan_in"},
        fan_in={**INTAKE, "velocity": 0.9},
    )
    with pytest.raises(
        ValueError,
        match=r"fan_out\.recirculate_from: the discharge's flow 0\.5 m\^2/s differs "
        r"from its intake 'fan_in''s 0\.45 m\^2/s",
    ):
        SimConfig.from_dict(raw)


# ---------------------------------------------------------------------------
# The scenario section (ADR-014 B, decision 7; step 1)
# ---------------------------------------------------------------------------

# A small room for the scenario rules: a supply, a door on the left wall with
# a side wall above it, two returns that take the remainder, and a threshold
# for every class so threshold_factor resolves.
SCENARIO_BOUNDARIES = {
    "hepa_supply": {
        "type": "velocity_inlet",
        "location": "top",
        "x_start": 0.5,
        "x_end": 3.5,
        "velocity": 0.45,
        "hepa_filtered": True,
    },
    "door": {"type": "wall", "location": "left", "y_start": 0.0, "y_end": 2.1},
    "side_wall": {"type": "wall", "location": "left", "y_start": 2.1, "y_end": 3.0},
    "return_a": {
        "type": "fixed_flow_outlet",
        "location": "bottom",
        "x_start": 0.25,
        "x_end": 0.75,
    },
    "return_b": {
        "type": "fixed_flow_outlet",
        "location": "bottom",
        "x_start": 3.25,
        "x_end": 3.75,
    },
}
SCENARIO_THRESHOLDS = {
    "0.1e-6": 1.0e5,
    "0.3e-6": 1.02e4,
    "0.5e-6": 3.52e3,
    "1.0e-6": 8.32e2,
    "5.0e-6": 2.93e1,
}
SCENARIO_TRANSPORT = {
    "cfl_number": 0.1,
    "max_diffusion_iter": 200,
    "diffusion_tol": 1e-8,
}
DOOR_IN = {
    "name": "door_in",
    "type": "velocity_inlet",
    "location": "left",
    "y_start": 0.0,
    "y_end": 1.05,
    "velocity": 0.25,
    "concentration": [1.0] * 5,
}
DOOR_OUT = {
    "name": "door_out",
    "type": "fixed_flow_outlet",
    "location": "left",
    "y_start": 1.05,
    "y_end": 2.1,
    "velocity": 0.25,
}


def _door_open(start: float = 10.0, duration: float = 5.0, **extra: object) -> dict:
    """The door replaced by the two-way exchange (ADR-014 B) as an event."""
    return {
        "name": "door_open",
        "start": start,
        "duration": duration,
        "replace": {"segment": "door", "with": [dict(DOOR_IN), dict(DOOR_OUT)]},
        **extra,
    }


def _scenario_base(tmp_path: Path) -> dict:
    """The scenario tests' base room as a mapping."""
    with open(_write_config(tmp_path), encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    base["boundaries"] = {k: dict(v) for k, v in SCENARIO_BOUNDARIES.items()}
    base["transport"] = dict(SCENARIO_TRANSPORT)
    base["thresholds"] = dict(SCENARIO_THRESHOLDS)
    return base


def _scenario_raw(tmp_path: Path, events: list, base: dict | None = None) -> dict:
    """A scenario mapping whose base, base.yaml, is written into tmp_path."""
    if base is None:
        base = _scenario_base(tmp_path)
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(base, sort_keys=False), encoding="utf-8"
    )
    return {
        "scenario": {"base": "base.yaml", "events": events},
        "solver": {"t_end": 60.0, "output_interval": 10},
        "transport": {"cfl_number": 0.4},
    }


def _load_scenario(tmp_path: Path, raw: dict) -> SimConfig:
    return SimConfig.from_dict(raw, base_dir=tmp_path)


def _refused_scenario(
    tmp_path: Path,
    raw: dict,
    match: str,
    error: type | tuple[type, ...] = ValueError,
) -> None:
    with pytest.raises(error, match=match):
        _load_scenario(tmp_path, raw)


@pytest.mark.unit
class TestScenarioInheritance:
    """The scenario file over its base (decision 7, REQ-C03)."""

    def test_the_files_sections_merge_over_the_base(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [_door_open()])
        raw["solver"]["t_end"] = 30.0
        raw["sensors"] = [{"name": "door_side", "x": 0.5, "y": 1.0}]
        config = _load_scenario(tmp_path, raw)
        # The file's keys win; the base's other keys in the same section stay.
        assert config.t_end == 30.0
        assert config.max_simple_iter == 500
        assert config.transport is not None
        assert config.transport.cfl_number == 0.4
        assert config.transport.max_diffusion_iter == 200
        # A list replaces the base's whole.
        assert [s.name for s in config.sensors] == ["door_side"]
        assert config.boundaries["door"].type == "wall"
        spec = config.scenario
        assert spec is not None
        assert spec.base_path == (tmp_path / "base.yaml").resolve()
        assert spec.t_end == 30.0
        assert spec.base["solver"]["t_end"] == 30.0
        assert "scenario" not in spec.base

    def test_a_configuration_without_the_section_has_no_scenario(
        self, tmp_path: Path
    ) -> None:
        assert SimConfig(_write_config(tmp_path)).scenario is None

    def test_the_merged_base_is_read_only(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [_door_open()])
        spec = _load_scenario(tmp_path, raw).scenario
        with pytest.raises(TypeError):
            spec.base["solver"]["t_end"] = 1.0  # type: ignore[index]
        copy = spec.mapping(())
        copy["solver"]["t_end"] = 1.0
        assert spec.base["solver"]["t_end"] == 60.0

    def test_a_relative_base_is_read_from_the_files_directory(
        self, tmp_path: Path
    ) -> None:
        raw = _scenario_raw(tmp_path, [])
        path = tmp_path / "scenario.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        spec = SimConfig(path).scenario
        assert spec.base_path == (tmp_path / "base.yaml").resolve()
        # The scenario keeps its own file; a mapping has none (review 62 S4).
        assert spec.path == path.resolve()
        assert _load_scenario(tmp_path, raw).scenario.path is None

    def test_a_relative_base_with_no_directory_is_refused(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [])
        with pytest.raises(
            ValueError, match=r"scenario\.base 'base\.yaml' is relative"
        ):
            SimConfig.from_dict(raw)

    def test_a_missing_base_is_refused(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [])
        raw["scenario"]["base"] = "absent.yaml"
        _refused_scenario(tmp_path, raw, r"not found", FileNotFoundError)

    def test_a_base_that_is_a_scenario_is_refused(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [])
        (tmp_path / "middle.yaml").write_text(
            yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
        )
        raw["scenario"]["base"] = "middle.yaml"
        _refused_scenario(tmp_path, raw, r"'middle\.yaml' is itself a scenario")

    def test_the_section_must_be_a_mapping(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [])
        raw["scenario"] = ["base.yaml"]
        _refused_scenario(tmp_path, raw, r"scenario must be a mapping")

    def test_an_unknown_scenario_key_is_refused(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [])
        raw["scenario"]["event"] = []
        _refused_scenario(tmp_path, raw, r"scenario\.event is not a recognised")

    def test_the_base_must_be_a_string(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [])
        raw["scenario"]["base"] = 3
        _refused_scenario(tmp_path, raw, r"scenario\.base must be a string", TypeError)

    def test_events_are_required_and_a_list(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [])
        del raw["scenario"]["events"]
        _refused_scenario(tmp_path, raw, r"Missing required key: 'scenario\.events'")
        raw["scenario"]["events"] = {"door_open": _door_open()}
        _refused_scenario(tmp_path, raw, r"scenario\.events must be a list", TypeError)

    @pytest.mark.parametrize(
        ("section", "key"),
        [
            ("solver", "t_end"),
            ("solver", "output_interval"),
            ("transport", "cfl_number"),
        ],
    )
    def test_each_run_key_is_stated_in_the_scenario_itself(
        self, tmp_path: Path, section: str, key: str
    ) -> None:
        """The base states every one; inheriting it is still refused."""
        raw = _scenario_raw(tmp_path, [])
        del raw[section][key]
        _refused_scenario(
            tmp_path, raw, rf"Missing required key: '{section}\.{key}' in the scenario"
        )


@pytest.mark.unit
class TestScenarioEvents:
    """Every event's name, times and change (ADR-014 B), one test per rule."""

    def test_an_event_must_be_a_mapping(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, ["door_open"])
        _refused_scenario(tmp_path, raw, r"scenario\.events\[0\] must be a mapping")

    def test_an_unknown_event_key_is_refused(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [_door_open(end=15.0)])
        _refused_scenario(tmp_path, raw, r"events\[0\]\.end is not a recognised event")

    def test_the_name_is_a_non_empty_string(self, tmp_path: Path) -> None:
        event = _door_open()
        del event["name"]
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(
            tmp_path, raw, r"Missing required key: 'scenario\.events\[0\]\.name'"
        )
        raw = _scenario_raw(tmp_path, [_door_open(name=7)])
        _refused_scenario(
            tmp_path, raw, r"events\[0\]\.name must be a string", TypeError
        )
        raw = _scenario_raw(tmp_path, [_door_open(name="  ")])
        _refused_scenario(tmp_path, raw, r"events\[0\]\.name must not be empty")

    def test_two_events_of_one_name_are_refused(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [_door_open(), _door_open(start=30.0)])
        _refused_scenario(tmp_path, raw, r"two events are named 'door_open'")

    @pytest.mark.parametrize(
        ("start", "error", "match"),
        [
            (None, ValueError, r"Missing required key: .*\.start'"),
            (True, TypeError, r"\.start must be a number, got bool"),
            ("10", TypeError, r"\.start must be a number, got str"),
            (float("nan"), ValueError, r"\.start must be finite"),
            (float("inf"), ValueError, r"\.start must be finite"),
            (-1.0, ValueError, r"\.start must be non-negative"),
        ],
    )
    def test_the_start_is_a_finite_non_negative_number(
        self, tmp_path: Path, start: object, error: type, match: str
    ) -> None:
        event = _door_open()
        if start is None:
            del event["start"]
        else:
            event["start"] = start
        _refused_scenario(tmp_path, _scenario_raw(tmp_path, [event]), match, error)

    @pytest.mark.parametrize(
        ("duration", "error", "match"),
        [
            (None, ValueError, r"Missing required key: .*\.duration'"),
            (True, TypeError, r"\.duration must be a number, got bool"),
            (float("nan"), ValueError, r"\.duration must be finite"),
            (float("inf"), ValueError, r"\.duration must be finite"),
            (0.0, ValueError, r"\.duration must be positive"),
            (-5.0, ValueError, r"\.duration must be positive"),
        ],
    )
    def test_the_duration_is_a_finite_positive_number(
        self, tmp_path: Path, duration: object, error: type, match: str
    ) -> None:
        event = _door_open()
        if duration is None:
            del event["duration"]
        else:
            event["duration"] = duration
        _refused_scenario(tmp_path, _scenario_raw(tmp_path, [event]), match, error)

    def test_an_event_ends_by_t_end(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [_door_open(start=50.0, duration=10.5)])
        _refused_scenario(
            tmp_path, raw, r"ends at 60\.5 s, after solver\.t_end \(60\.0 s\)"
        )
        # Ending exactly at t_end loads.
        _load_scenario(tmp_path, _scenario_raw(tmp_path, [_door_open(50.0, 10.0)]))

    def test_an_event_states_exactly_one_change(self, tmp_path: Path) -> None:
        event = _door_open()
        del event["replace"]
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(tmp_path, raw, r"must state exactly one of .*got none")
        two = _door_open(set={"segment": "hepa_supply", "keys": {"velocity": 0.5}})
        raw = _scenario_raw(tmp_path, [two])
        _refused_scenario(tmp_path, raw, r"exactly one of .*\['replace', 'set'\]")

    def test_the_change_is_a_mapping(self, tmp_path: Path) -> None:
        event = _door_open()
        event["replace"] = ["door"]
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(tmp_path, raw, r"'door_open'\)\.replace must be a mapping")

    def test_two_events_on_one_segment_must_not_overlap(self, tmp_path: Path) -> None:
        leak = {
            "name": "seal_leak",
            "start": 12.0,
            "duration": 10.0,
            "set": {"segment": "door", "keys": {"deposition_surface": "none"}},
        }
        raw = _scenario_raw(tmp_path, [_door_open(), leak])
        _refused_scenario(
            tmp_path,
            raw,
            r"'door_open' \[10\.0, 15\.0\) and 'seal_leak' \[12\.0, 22\.0\) both change "
            r"segment 'door' and overlap",
        )
        # Meeting at an instant is not an overlap: the first has expired.
        leak["start"] = 15.0
        _load_scenario(tmp_path, _scenario_raw(tmp_path, [_door_open(), leak]))
        # Events on two segments may overlap.
        supply = {
            "name": "supply_change",
            "start": 12.0,
            "duration": 10.0,
            "set": {"segment": "hepa_supply", "keys": {"hepa_filtered": False}},
        }
        _load_scenario(tmp_path, _scenario_raw(tmp_path, [_door_open(), supply]))

    def _wall_event(self, name: str, start: float, duration: float) -> dict:
        return {
            "name": name,
            "start": start,
            "duration": duration,
            "set": {"segment": "door", "keys": {"deposition_surface": "none"}},
        }

    def test_back_to_back_events_meet_whatever_the_rounding(
        self, tmp_path: Path
    ) -> None:
        """Review 62 S5: 0.1 + 0.2 is above 0.3 in doubles."""
        assert 0.1 + 0.2 > 0.3
        first = self._wall_event("first", 0.1, 0.2)
        second = self._wall_event("second", 0.3, 0.2)
        spec = _load_scenario(
            tmp_path, _scenario_raw(tmp_path, [first, second])
        ).scenario
        # One instant at 0.3 s, no sliver interval, and one event active there.
        assert spec.times() == (0.0, 0.1, 0.3, 0.5, 60.0)
        assert [e.name for e in spec.active(0.3)] == ["second"]
        assert [e.name for e in spec.active(0.1)] == ["first"]

    def test_an_overlap_just_past_the_tolerance_is_refused(
        self, tmp_path: Path
    ) -> None:
        tolerance = EVENT_TIME_RTOL * 60.0
        second = self._wall_event("second", 0.3 - 2.0 * tolerance, 0.2)
        raw = _scenario_raw(tmp_path, [self._wall_event("first", 0.1, 0.2), second])
        _refused_scenario(
            tmp_path, raw, r"'first' .* and 'second' .* both change segment 'door'"
        )
        # Within the tolerance the two still meet.
        second["start"] = 0.3 - 0.5 * tolerance
        _load_scenario(tmp_path, raw)

    def test_events_on_two_segments_leave_no_sliver_interval(
        self, tmp_path: Path
    ) -> None:
        supply = {
            "name": "supply_breach",
            "start": 0.3,
            "duration": 5.0,
            "set": {"segment": "hepa_supply", "keys": {"hepa_filtered": False}},
        }
        events = [self._wall_event("door_seal", 0.1, 0.2), supply]
        spec = _load_scenario(tmp_path, _scenario_raw(tmp_path, events)).scenario
        assert spec.times() == (0.0, 0.1, 0.3, 5.3, 60.0)
        assert [e.name for e in spec.active(0.3)] == ["supply_breach"]

    def test_an_end_within_the_tolerance_of_t_end_loads(self, tmp_path: Path) -> None:
        """The end's rounding past t_end is not an end after it."""
        raw = _scenario_raw(tmp_path, [self._wall_event("late", 0.1, 0.2)])
        raw["solver"]["t_end"] = 0.3
        assert 0.1 + 0.2 > 0.3
        spec = _load_scenario(tmp_path, raw).scenario
        assert spec.times() == (0.0, 0.1, 0.3)
        assert all(b - a > spec.time_tolerance for a, b in pairwise(spec.times()))


@pytest.mark.unit
class TestScenarioReplace:
    """A replace: the named base segment and its replacements (ADR-014 B)."""

    def test_the_replacements_take_the_segments_place(self, tmp_path: Path) -> None:
        config = _load_scenario(tmp_path, _scenario_raw(tmp_path, [_door_open()]))
        spec = config.scenario
        change = spec.events[0].change
        assert change.segment == "door"
        assert [name for name, _ in change.segments] == ["door_in", "door_out"]
        mapping = spec.mapping(spec.active(10.0))
        assert list(mapping["boundaries"]) == [
            "hepa_supply",
            "door_in",
            "door_out",
            "side_wall",
            "return_a",
            "return_b",
        ]
        assert "name" not in mapping["boundaries"]["door_in"]
        assert mapping["boundaries"]["door_in"]["concentration"] == [1.0] * 5

    def test_an_empty_replacement_removes_the_segment(self, tmp_path: Path) -> None:
        """The amendment of 2026-10-17: the coverage rule's wall takes its faces."""
        event = _door_open()
        event["replace"]["with"] = []
        spec = _load_scenario(tmp_path, _scenario_raw(tmp_path, [event])).scenario
        assert spec.events[0].change.segments == ()
        assert "door" not in spec.mapping(spec.active(10.0))["boundaries"]
        assert "door" in spec.mapping(spec.active(15.0))["boundaries"]

    def test_an_unknown_replace_key_is_refused(self, tmp_path: Path) -> None:
        event = _door_open()
        event["replace"]["by"] = []
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(tmp_path, raw, r"replace\.by is not a recognised key")

    def test_the_segment_names_a_segment_of_the_base(self, tmp_path: Path) -> None:
        event = _door_open()
        event["replace"]["segment"] = "window"
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(
            tmp_path, raw, r"segment names 'window', which is not a segment"
        )
        del event["replace"]["segment"]
        _refused_scenario(tmp_path, raw, r"Missing required key: .*replace\.segment'")

    def test_with_is_required_and_a_list(self, tmp_path: Path) -> None:
        event = _door_open()
        del event["replace"]["with"]
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(tmp_path, raw, r"Missing required key: .*replace\.with'")
        event["replace"]["with"] = dict(DOOR_IN)
        _refused_scenario(tmp_path, raw, r"replace\.with must be a list", TypeError)

    def test_each_replacement_is_a_named_mapping(self, tmp_path: Path) -> None:
        event = _door_open()
        event["replace"]["with"] = ["door_in"]
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(tmp_path, raw, r"with\[0\] must be a mapping")
        event["replace"]["with"] = [{k: v for k, v in DOOR_IN.items() if k != "name"}]
        _refused_scenario(tmp_path, raw, r"Missing required key: .*with\[0\]\.name'")
        event["replace"]["with"] = [{**DOOR_IN, "name": ""}]
        _refused_scenario(tmp_path, raw, r"with\[0\]\.name must not be empty")

    def test_a_replacement_may_not_take_another_segments_name(
        self, tmp_path: Path
    ) -> None:
        event = _door_open()
        event["replace"]["with"][0]["name"] = "side_wall"
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(tmp_path, raw, r"'side_wall' is the name of another segment")
        event["replace"]["with"][0]["name"] = "door_out"
        _refused_scenario(tmp_path, raw, r"with\[1\]\.name 'door_out' is given twice")
        # The replaced segment's own name may be reused.
        event["replace"]["with"][0]["name"] = "door"
        _load_scenario(tmp_path, raw)

    def test_concurrent_replacements_of_one_name_are_refused(
        self, tmp_path: Path
    ) -> None:
        moved = {
            "name": "return_moved",
            "start": 12.0,
            "duration": 5.0,
            "replace": {
                "segment": "return_a",
                "with": [
                    {
                        "name": "door_in",
                        "type": "fixed_flow_outlet",
                        "location": "bottom",
                        "x_start": 0.25,
                        "x_end": 0.75,
                    }
                ],
            },
        }
        raw = _scenario_raw(tmp_path, [_door_open(), moved])
        _refused_scenario(
            tmp_path,
            raw,
            r"from 12\.0 s to 15\.0 s with \['door_open', 'return_moved'\] active does "
            r"not load: .*'return_moved' names a segment 'door_in'",
        )


@pytest.mark.unit
class TestScenarioSet:
    """A set: keys of a named base segment changed in place (ADR-014 B)."""

    def _breach(self, **keys: object) -> dict:
        return {
            "name": "breach",
            "start": 5.0,
            "duration": 20.0,
            "set": {"segment": "hepa_supply", "keys": keys},
        }

    def test_the_keys_merge_over_the_segments(self, tmp_path: Path) -> None:
        event = self._breach(hepa_filtered=False, concentration=[2.0] * 5)
        spec = _load_scenario(tmp_path, _scenario_raw(tmp_path, [event])).scenario
        supply = spec.mapping(spec.active(5.0))["boundaries"]["hepa_supply"]
        assert supply["hepa_filtered"] is False
        assert supply["concentration"] == [2.0] * 5
        assert supply["velocity"] == 0.45
        assert (
            spec.mapping(spec.active(25.0))["boundaries"]["hepa_supply"]
            == (SCENARIO_BOUNDARIES["hepa_supply"])
        )

    def test_an_unknown_set_key_is_refused(self, tmp_path: Path) -> None:
        event = self._breach(hepa_filtered=False)
        event["set"]["values"] = {}
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(tmp_path, raw, r"set\.values is not a recognised key")

    def test_the_segment_names_a_segment_of_the_base(self, tmp_path: Path) -> None:
        event = self._breach(hepa_filtered=False)
        event["set"]["segment"] = "supply"
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(tmp_path, raw, r"segment names 'supply', which is not")

    @pytest.mark.parametrize("keys", [None, {}, ["hepa_filtered"]])
    def test_the_keys_are_a_non_empty_mapping(
        self, tmp_path: Path, keys: object
    ) -> None:
        event = self._breach()
        if keys is None:
            del event["set"]["keys"]
            match = r"Missing required key: .*set\.keys'"
        else:
            event["set"]["keys"] = keys
            match = r"set\.keys must be a non-empty mapping"
        _refused_scenario(tmp_path, _scenario_raw(tmp_path, [event]), match)

    def test_an_unknown_segment_key_is_refused(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [self._breach(filtered=False)])
        _refused_scenario(
            tmp_path, raw, r"set\.keys\.filtered is not a recognised boundary"
        )


@pytest.mark.unit
class TestScenarioSource:
    """A source: a box and a rate or an emission per class (ADR-014 B)."""

    def _dust(self, **body: object) -> dict:
        source = {
            "x_start": 2.5,
            "x_end": 2.6,
            "y_start": 1.0,
            "y_end": 1.1,
            "emission": [1.0e4] * 5,
        }
        source.update(body)
        return {"name": "dust", "start": 10.0, "duration": 30.0, "source": source}

    def test_a_source_loads(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [self._dust()])
        spec = _load_scenario(tmp_path, raw).scenario
        source = spec.events[0].change
        assert (source.x_start, source.x_end, source.y_start, source.y_end) == (
            2.5,
            2.6,
            1.0,
            1.1,
        )
        assert source.emission == (1.0e4,) * 5
        assert source.rate is None
        # A source changes no boundary.
        assert spec.mapping(spec.active(10.0)) == spec.mapping(())

    def test_an_unknown_source_key_is_refused(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [self._dust(strength=1.0)])
        _refused_scenario(tmp_path, raw, r"source\.strength is not a recognised key")

    @pytest.mark.parametrize("key", ["x_start", "x_end", "y_start", "y_end"])
    def test_each_box_key_is_a_finite_number(self, tmp_path: Path, key: str) -> None:
        event = self._dust()
        del event["source"][key]
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(tmp_path, raw, rf"Missing required key: .*source\.{key}'")
        event["source"][key] = True
        _refused_scenario(tmp_path, raw, rf"source\.{key} must be a number", TypeError)
        event["source"][key] = float("nan")
        _refused_scenario(tmp_path, raw, rf"source\.{key} must be finite")

    @pytest.mark.parametrize(
        "box", [(2.6, 2.5, 1.0, 1.1), (2.5, 2.6, 1.1, 1.1)], ids=["x", "y"]
    )
    def test_an_empty_box_is_refused(
        self, tmp_path: Path, box: tuple[float, ...]
    ) -> None:
        body = dict(zip(("x_start", "x_end", "y_start", "y_end"), box, strict=True))
        raw = _scenario_raw(tmp_path, [self._dust(**body)])
        _refused_scenario(tmp_path, raw, r"the box .* is empty")

    @pytest.mark.parametrize(
        "body", [{"x_start": -0.1}, {"x_end": 4.1}, {"y_start": -0.1}, {"y_end": 3.1}]
    )
    def test_a_box_outside_the_domain_is_refused(
        self, tmp_path: Path, body: dict
    ) -> None:
        raw = _scenario_raw(tmp_path, [self._dust(**body)])
        _refused_scenario(
            tmp_path, raw, r"is outside the domain \[0, 4\.0\] x \[0, 3\.0\]"
        )

    def test_exactly_one_of_rate_and_emission(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [self._dust(rate=[1.0] * 5)])
        _refused_scenario(
            tmp_path, raw, r"exactly one of rate .*\['rate', 'emission'\]"
        )
        event = self._dust()
        del event["source"]["emission"]
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(tmp_path, raw, r"exactly one of rate .*got none")
        event["source"]["rate"] = [3.0] * 5
        rate = _load_scenario(tmp_path, raw).scenario.events[0].change.rate
        assert rate == (3.0,) * 5

    @pytest.mark.parametrize("key", ["rate", "emission"])
    @pytest.mark.parametrize(
        ("values", "error", "match"),
        [
            (1.0, TypeError, r" must be a list"),
            ([1.0] * 4, ValueError, r" must have one value per particle class \(5\)"),
            ([1.0] * 4 + [-1.0], ValueError, r"\[4\] must be non-negative"),
            ([1.0] * 4 + [True], TypeError, r"\[4\] must be a number, got bool"),
            ([1.0] * 4 + [float("nan")], ValueError, r"\[4\] must be finite"),
        ],
    )
    def test_the_per_class_array(
        self,
        tmp_path: Path,
        key: str,
        values: object,
        error: type,
        match: str,
    ) -> None:
        event = self._dust()
        del event["source"]["emission"]
        event["source"][key] = values
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(tmp_path, raw, rf"source\.{key}{match}", error)


@pytest.mark.unit
class TestScenarioInitial:
    """An initial: the field at t = 0, uniform per class (ADR-014 B)."""

    def _initial(self, **body: object) -> dict:
        return {"name": "start_dirty", "start": 0.0, "initial": body}

    def test_the_thresholds_times_a_factor(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [self._initial(threshold_factor=100.0)])
        initial = _load_scenario(tmp_path, raw).scenario.events[0].change
        assert initial.values == tuple(100.0 * v for v in SCENARIO_THRESHOLDS.values())
        assert initial.threshold_factor == 100.0

    def test_a_concentration_per_class(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [self._initial(concentration=[5.0] * 5)])
        initial = _load_scenario(tmp_path, raw).scenario.events[0].change
        assert initial.values == (5.0,) * 5
        assert initial.threshold_factor is None

    def test_an_initial_is_never_active(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [self._initial(threshold_factor=100.0)])
        spec = _load_scenario(tmp_path, raw).scenario
        assert spec.times() == (0.0, 60.0)
        assert spec.active(0.0) == ()

    def test_exactly_one_of_concentration_and_factor(self, tmp_path: Path) -> None:
        raw = _scenario_raw(
            tmp_path, [self._initial(concentration=[1.0] * 5, threshold_factor=2.0)]
        )
        _refused_scenario(tmp_path, raw, r"exactly one of concentration and threshold")
        raw = _scenario_raw(tmp_path, [self._initial()])
        _refused_scenario(tmp_path, raw, r"exactly one of concentration .*got none")

    def test_an_unknown_initial_key_is_refused(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [self._initial(factor=100.0)])
        _refused_scenario(tmp_path, raw, r"initial\.factor is not a recognised key")

    @pytest.mark.parametrize(
        ("factor", "error", "match"),
        [
            (0.0, ValueError, r"must be positive"),
            (-1.0, ValueError, r"must be positive"),
            (True, TypeError, r"must be a number, got bool"),
            (float("inf"), ValueError, r"must be finite"),
        ],
    )
    def test_the_factor_is_a_finite_positive_number(
        self, tmp_path: Path, factor: object, error: type, match: str
    ) -> None:
        raw = _scenario_raw(tmp_path, [self._initial(threshold_factor=factor)])
        _refused_scenario(tmp_path, raw, rf"threshold_factor {match}", error)

    def test_the_factor_needs_a_threshold_for_every_class(self, tmp_path: Path) -> None:
        base = _scenario_base(tmp_path)
        del base["thresholds"]["0.3e-6"]
        raw = _scenario_raw(tmp_path, [self._initial(threshold_factor=100.0)], base)
        _refused_scenario(
            tmp_path, raw, r"the class of diameter 3e-07 m has no threshold keys"
        )

    def test_the_concentration_per_class(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [self._initial(concentration=[1.0] * 6)])
        _refused_scenario(
            tmp_path, raw, r"concentration must have one value per particle"
        )
        raw = _scenario_raw(tmp_path, [self._initial(concentration=[1.0] * 4 + [-2.0])])
        _refused_scenario(tmp_path, raw, r"concentration\[4\] must be non-negative")

    @pytest.mark.parametrize(
        ("values", "error", "match"),
        [
            (1.0, TypeError, r" must be a list"),
            ([1.0] * 4 + [True], TypeError, r"\[4\] must be a number, got bool"),
            ([1.0] * 4 + [float("nan")], ValueError, r"\[4\] must be finite"),
        ],
    )
    def test_the_concentrations_type_and_entries(
        self, tmp_path: Path, values: object, error: type, match: str
    ) -> None:
        """Review 62 S7: the rate and emission rejections, on the third caller."""
        raw = _scenario_raw(tmp_path, [self._initial(concentration=values)])
        _refused_scenario(tmp_path, raw, rf"initial\.concentration{match}", error)

    def test_an_initial_starts_at_zero_with_no_duration(self, tmp_path: Path) -> None:
        event = self._initial(threshold_factor=100.0)
        event["start"] = 1.0
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(tmp_path, raw, r"start must be 0 on an initial event")
        event = self._initial(threshold_factor=100.0)
        event["duration"] = 5.0
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(tmp_path, raw, r"duration is not valid on an initial event")

    def test_two_initials_are_refused(self, tmp_path: Path) -> None:
        second = {**self._initial(concentration=[1.0] * 5), "name": "start_dusty"}
        raw = _scenario_raw(tmp_path, [self._initial(threshold_factor=100.0), second])
        _refused_scenario(tmp_path, raw, r"\['start_dirty', 'start_dusty'\] each set")


@pytest.mark.unit
class TestScenarioHoldTime:
    """scenario.hold_time, the clean-out's stated hold (prompt 64's decision 2)."""

    def _raw(self, tmp_path: Path, **section: object) -> dict:
        initial = {"name": "start_dirty", "start": 0.0, "initial": {}}
        initial["initial"]["threshold_factor"] = 100.0
        raw = _scenario_raw(tmp_path, [initial])
        raw["scenario"].update(section)
        return raw

    def test_absent_is_none_and_present_is_read(self, tmp_path: Path) -> None:
        assert _load_scenario(tmp_path, self._raw(tmp_path)).scenario.hold_time is None
        spec = _load_scenario(tmp_path, self._raw(tmp_path, hold_time=5.5)).scenario
        assert spec.hold_time == 5.5

    @pytest.mark.parametrize(
        ("value", "error", "match"),
        [
            (0.0, ValueError, r"must be positive"),
            (-1.0, ValueError, r"must be positive"),
            (True, TypeError, r"must be a number, got bool"),
            ("5", TypeError, r"must be a number, got str"),
            (None, TypeError, r"must be a number, got NoneType"),
            (float("nan"), ValueError, r"must be finite"),
        ],
    )
    def test_the_hold_is_a_finite_positive_number(
        self, tmp_path: Path, value: object, error: type, match: str
    ) -> None:
        raw = self._raw(tmp_path, hold_time=value)
        _refused_scenario(tmp_path, raw, rf"scenario\.hold_time {match}", error)

    def test_the_hold_is_below_t_end(self, tmp_path: Path) -> None:
        assert _load_scenario(tmp_path, self._raw(tmp_path, hold_time=59.9))
        for value in (60.0, 61.0):
            raw = self._raw(tmp_path, hold_time=value)
            _refused_scenario(tmp_path, raw, r"hold_time \(.* s\) must be below")

    def test_the_hold_needs_an_initial_event(self, tmp_path: Path) -> None:
        raw = _scenario_raw(tmp_path, [_door_open()])
        raw["scenario"]["hold_time"] = 5.0
        _refused_scenario(tmp_path, raw, r"hold_time needs an initial event")

    def test_the_hold_needs_a_start_above_the_thresholds(self, tmp_path: Path) -> None:
        """A start below every threshold is within spec at t = 0: refused.

        One class at its threshold is not within spec ("within" is below),
        so it is accepted. Defects caught: the start not compared (the hold
        ends the run at hold_time, review 64 S5); any class below taken for
        every class; a class exactly at its threshold taken as within spec.
        """
        raw = self._raw(tmp_path, hold_time=5.5)
        raw["scenario"]["events"][0]["initial"] = {"threshold_factor": 0.5}
        _refused_scenario(
            tmp_path, raw, r"'start_dirty' starts every class below its threshold"
        )
        below = [0.5 * t for t in SCENARIO_THRESHOLDS.values()]
        raw["scenario"]["events"][0]["initial"] = {"concentration": below}
        _refused_scenario(tmp_path, raw, r"the hold would end the run at 5.5 s")
        for level in (SCENARIO_THRESHOLDS["5.0e-6"], 2.0 * 2.93e1):
            raw["scenario"]["events"][0]["initial"] = {
                "concentration": [*below[:4], level]
            }
            assert _load_scenario(tmp_path, raw).scenario.hold_time == 5.5

    def test_the_reference_clean_out_runs_to_its_end(self) -> None:
        """No hold, to 240 s: past the coarse mean's 100:1 time, three well-mixed times.

        The well-mixed time ln(100) / N is from T6: the free area 17.52 m^2
        over the supply 3.15 m^2/s. Defect caught: the hold restored, which
        stops the run on the sensors before the ratio and the local field
        are defined (prompt 66).
        """
        config = SimConfig(CONFIG_DIR / "scenario_clean_out.yaml")
        assert config.scenario.hold_time is None
        free = config.room_width * config.room_height - sum(
            (o.x_end - o.x_start) * (o.y_end - o.y_start) for o in config.obstacles
        )
        supply = config.boundaries["hepa_supply"]
        flow = supply.velocity * (supply.x_end - supply.x_start)
        assert config.t_end == 240.0
        assert config.t_end > 3.0 * math.log(100.0) * free / flow


@pytest.mark.unit
class TestScenarioSchmidtNumber:
    """A turbulent scenario states its Schmidt number at load (review 64 S3)."""

    def test_a_turbulent_scenario_without_one_is_refused(self, tmp_path: Path) -> None:
        """The default room less its Schmidt number, under the reference clean-out.

        The base alone loads, since it is never marched; the scenario is
        refused at load, before any flow is solved; the base with the number
        restored gives a scenario that loads. Defect caught: the refusal
        left to the first transport step, after the flow solve.
        """
        base = yaml.safe_load(
            (CONFIG_DIR / "clean_room_default.yaml").read_text(encoding="utf-8")
        )
        schmidt = base["transport"].pop("turbulent_schmidt")
        assert "turbulence" in base
        path = tmp_path / "base.yaml"
        path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
        assert SimConfig(path).transport.turbulent_schmidt is None
        raw = yaml.safe_load(
            (CONFIG_DIR / "scenario_clean_out.yaml").read_text(encoding="utf-8")
        )
        raw["scenario"]["base"] = "base.yaml"
        _refused_scenario(tmp_path, raw, r"needs transport\.turbulent_schmidt")
        base["transport"]["turbulent_schmidt"] = schmidt
        path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
        assert _load_scenario(tmp_path, raw).transport.turbulent_schmidt == schmidt


@pytest.mark.unit
class TestMergedMapping:
    """SimConfig keeps the merged mapping it loaded (prompt 64's decision 3)."""

    def test_a_plain_file_is_kept_as_read(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path)
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        config = SimConfig(path)
        assert config.merged_mapping() == raw
        assert SimConfig.from_dict(raw).merged_mapping() == raw

    def test_a_copy_is_returned_and_the_argument_is_not_kept(
        self, tmp_path: Path
    ) -> None:
        with open(_write_config(tmp_path), encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        config = SimConfig.from_dict(raw)
        raw["domain"]["nx"] = 7
        copy = config.merged_mapping()
        assert copy["domain"]["nx"] == 80
        copy["domain"]["nx"] = 9
        assert config.merged_mapping()["domain"]["nx"] == 80

    def test_a_scenario_is_its_merged_base_and_its_section(
        self, tmp_path: Path
    ) -> None:
        raw = _scenario_raw(tmp_path, [_door_open()])
        raw["solver"]["t_end"] = 30.0
        config = _load_scenario(tmp_path, raw)
        mapping = config.merged_mapping()
        assert list(mapping)[-1] == "scenario"
        assert mapping["scenario"] == raw["scenario"]
        assert {k: v for k, v in mapping.items() if k != "scenario"} == dict(
            config.scenario.mapping(())
        )
        assert mapping["solver"]["t_end"] == 30.0
        again = _load_scenario(tmp_path, mapping)
        assert again.merged_mapping() == mapping
        assert again.boundaries == config.boundaries
        assert again.scenario.events == config.scenario.events


@pytest.mark.unit
class TestScenarioIntervalsLoad:
    """Every interval's merged mapping loads as a file (ADR-014 B, REQ-C02)."""

    def test_a_replacement_that_overlaps_a_neighbour_is_refused(
        self, tmp_path: Path
    ) -> None:
        event = _door_open()
        event["replace"]["with"][1]["y_end"] = 2.5
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(
            tmp_path,
            raw,
            r"from 10\.0 s to 15\.0 s with \['door_open'\] active does not load: "
            r"boundaries\.door_out and boundaries\.side_wall overlap on the left edge",
        )

    def test_a_replacement_that_omits_a_required_key_is_refused(
        self, tmp_path: Path
    ) -> None:
        event = _door_open()
        del event["replace"]["with"][0]["velocity"]
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(
            tmp_path,
            raw,
            r"does not load: boundaries\.door_in: velocity_inlet requires",
        )

    def test_the_loaders_error_type_is_kept(self, tmp_path: Path) -> None:
        event = _door_open()
        event["replace"]["with"][0]["velocity"] = "fast"
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(
            tmp_path, raw, r"door_in\.velocity must be a number, got str", TypeError
        )

    def test_a_set_that_breaks_a_segment_rule_is_refused(self, tmp_path: Path) -> None:
        event = {
            "name": "dusty_door",
            "start": 0.0,
            "duration": 5.0,
            "set": {"segment": "door", "keys": {"concentration": [1.0] * 5}},
        }
        raw = _scenario_raw(tmp_path, [event])
        _refused_scenario(
            tmp_path,
            raw,
            r"from 0\.0 s to 5\.0 s .*\.door\.concentration is only valid on a "
            "velocity_inlet",
        )

    def test_a_fans_intake_and_discharge_stop_together(self, tmp_path: Path) -> None:
        """A removed discharge leaves its intake an exhaust that no rule of a
        file refuses; the scenario layer refuses the half-stopped fan."""
        base = TestVents()._raw(
            tmp_path,
            fan_out={**DISCHARGE, "recirculate_from": "fan_in"},
            fan_in=INTAKE,
        )
        base["transport"] = dict(SCENARIO_TRANSPORT)

        def stop(segment: str, start: float = 10.0) -> dict:
            return {
                "name": f"{segment}_stops",
                "start": start,
                "duration": 5.0,
                "replace": {"segment": segment, "with": []},
            }

        raw = _scenario_raw(tmp_path, [stop("fan_out")], base)
        _refused_scenario(
            tmp_path,
            raw,
            r"with \['fan_out_stops'\] active: the intake 'fan_in' still draws while "
            r"no discharge returns it",
        )
        raw = _scenario_raw(tmp_path, [stop("fan_in")], base)
        _refused_scenario(
            tmp_path, raw, r"recirculate_from names 'fan_in', which is not a boundary"
        )
        raw = _scenario_raw(tmp_path, [stop("fan_out"), stop("fan_in")], base)
        _load_scenario(tmp_path, raw)
        # One a moment late leaves the half-stopped fan in an interval of its own.
        raw = _scenario_raw(tmp_path, [stop("fan_out"), stop("fan_in", 10.5)], base)
        _refused_scenario(tmp_path, raw, r"from 10\.0 s to 10\.5 s")


REFERENCE_SCENARIOS = {
    "scenario_door_leak.yaml": ("door_open", "replace"),
    "scenario_filter_breach.yaml": ("panel_breach", "replace"),
    "scenario_equipment_dust.yaml": ("litho_dust", "source"),
    "scenario_clean_out.yaml": ("contaminated_start", "initial"),
}


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(REFERENCE_SCENARIOS))
def test_the_reference_scenarios_extend_the_default_room(name: str) -> None:
    """ADR-014 B and Alex's decision of 2026-10-17: the room without vents,
    unchanged; a scenario's own sections are its run keys."""
    path = CONFIG_DIR / name
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["scenario"]["base"] == "clean_room_default.yaml"
    assert set(raw) == {"scenario", "solver", "transport"}
    assert set(raw["solver"]) == {"t_end", "output_interval"}
    assert raw["transport"] == {"cfl_number": 0.4}
    config = SimConfig(path)
    spec = config.scenario
    assert spec.base_path == (CONFIG_DIR / "clean_room_default.yaml").resolve()
    event_name, kind = REFERENCE_SCENARIOS[name]
    assert [e.name for e in spec.events] == [event_name]
    assert type(spec.events[0].change).__name__.lower() == kind
    default = SimConfig(CONFIG_DIR / "clean_room_default.yaml")
    assert (config.nx, config.ny) == (default.nx, default.ny)
    assert config.boundaries == default.boundaries
    assert config.transport.cfl_number == 0.4


@pytest.mark.unit
def test_the_reference_scenarios_state_their_sources_values() -> None:
    """The door's values are warm59.py's; the dust source is ECR-002 step 8's S2."""
    door = SimConfig(CONFIG_DIR / "scenario_door_leak.yaml").scenario
    door_in, door_out = (dict(s) for _, s in door.events[0].change.segments)
    assert (door_in["y_start"], door_in["y_end"], door_in["velocity"]) == (
        0.0,
        1.05,
        0.25,
    )
    assert (door_out["y_start"], door_out["y_end"], door_out["velocity"]) == (
        1.05,
        2.1,
        0.25,
    )
    assert (door_in["turbulence_intensity"], door_in["dissipation_length"]) == (
        0.05,
        0.1,
    )
    dust = SimConfig(CONFIG_DIR / "scenario_equipment_dust.yaml").scenario
    source = dust.events[0].change
    assert (source.x_start, source.x_end, source.y_start, source.y_end) == (
        3.80,
        3.90,
        2.00,
        2.10,
    )
    assert source.emission == (1.0e4,) * 5
    clean = SimConfig(CONFIG_DIR / "scenario_clean_out.yaml")
    hundredfold = tuple(100.0 * t for t in clean.class_thresholds())
    assert clean.scenario.events[0].change.values == hundredfold
    assert door_in["concentration"] == hundredfold
    breach = SimConfig(CONFIG_DIR / "scenario_filter_breach.yaml").scenario
    panel = dict(breach.events[0].change.segments[1][1])
    assert (panel["x_start"], panel["x_end"], panel["hepa_filtered"]) == (
        3.5,
        4.1,
        False,
    )
    assert panel["concentration"] == hundredfold
