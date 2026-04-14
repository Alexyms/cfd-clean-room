"""Tests for the YAML configuration loader.

Unit tests verify that SimConfig loads valid configurations correctly
and rejects invalid configurations with clear error messages at load
time (REQ-C02).
"""

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
        Keys to merge into (or replace within) the base config dict
        before writing. Use a nested dict to override specific sections.
        Set a section to None to remove it entirely.

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
        assert config.nx == 80
        assert len(config.sensors) == 4


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
