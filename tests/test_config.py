"""Tests for the YAML configuration loader.

Unit tests verify that SimConfig loads valid configurations correctly
and rejects invalid configurations with clear error messages at load
time (REQ-C02).
"""

from pathlib import Path

import pytest
import yaml

from src.config import SimConfig


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
            "max_pressure_iter": 200,
            "pressure_tol": 1.0e-6,
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
        assert config.max_pressure_iter == 200
        assert config.pressure_tol == pytest.approx(1.0e-6)

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
        """The shipped clean_room_default.yaml loads without error."""
        config = SimConfig("configs/clean_room_default.yaml")
        assert config.nx == 200
        assert config.ny == 75
        assert config.room_width == 8.0
        assert len(config.sensors) == 4
        assert len(config.obstacles) == 4


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
                    "max_pressure_iter": 200,
                    "pressure_tol": 1e-6,
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
                    "max_pressure_iter": 200,
                    "pressure_tol": 1e-6,
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
                    "max_pressure_iter": 200,
                    "pressure_tol": 1e-6,
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
                    "max_pressure_iter": 200,
                    "pressure_tol": 1e-6,
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
                    "max_pressure_iter": 200,
                    "pressure_tol": 1e-6,
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
