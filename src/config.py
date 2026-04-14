"""YAML configuration loader with validation for the CFD simulation.

Loads simulation parameters from a YAML file, validates all values at
load time, and provides typed attribute access. Satisfies REQ-C01
(single source of truth) and REQ-C02 (fail-fast validation).
"""

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BoundarySpec:
    """Specification for a single boundary condition.

    Parameters
    ----------
    type : str
        Boundary condition type: "velocity_inlet", "pressure_outlet",
        or "wall".
    location : str
        Domain edge: "top", "bottom", "left", or "right".
    x_start : float or None
        Start of boundary segment along x axis (for top/bottom).
    x_end : float or None
        End of boundary segment along x axis (for top/bottom).
    y_start : float or None
        Start of boundary segment along y axis (for left/right).
    y_end : float or None
        End of boundary segment along y axis (for left/right).
    velocity : float or None
        Prescribed velocity magnitude for velocity_inlet type.
    """

    type: str
    location: str
    x_start: float | None = None
    x_end: float | None = None
    y_start: float | None = None
    y_end: float | None = None
    velocity: float | None = None


@dataclass(frozen=True)
class ObstacleSpec:
    """Specification for an internal obstacle (solid region).

    Parameters
    ----------
    name : str
        Descriptive name for the obstacle.
    x_start : float
        Left edge x coordinate in meters.
    x_end : float
        Right edge x coordinate in meters.
    y_start : float
        Bottom edge y coordinate in meters.
    y_end : float
        Top edge y coordinate in meters.
    """

    name: str
    x_start: float
    x_end: float
    y_start: float
    y_end: float


@dataclass(frozen=True)
class SensorSpec:
    """Specification for a contamination sensor location.

    Parameters
    ----------
    name : str
        Sensor identifier.
    x : float
        Sensor x coordinate in meters.
    y : float
        Sensor y coordinate in meters.
    """

    name: str
    x: float
    y: float


@dataclass(frozen=True)
class HepaReference:
    """HEPA filter reference efficiency data for interpolation.

    Parameters
    ----------
    diameters : list[float]
        Reference particle diameters in meters, sorted ascending.
    efficiencies : list[float]
        Single-pass collection efficiency for each diameter.
    """

    diameters: list[float] = field(default_factory=list)
    efficiencies: list[float] = field(default_factory=list)


_VALID_BOUNDARY_TYPES: set[str] = {"velocity_inlet", "pressure_outlet", "wall"}
_VALID_BOUNDARY_LOCATIONS: set[str] = {"top", "bottom", "left", "right"}


class SimConfig:
    """Simulation configuration loaded and validated from a YAML file.

    All simulation parameters are validated at load time. Missing keys,
    out-of-range values, and type mismatches raise immediately with
    clear error messages (REQ-C02).

    Parameters
    ----------
    yaml_path : str or Path
        Path to the YAML configuration file.
    """

    def __init__(self, yaml_path: str | PathLike[str]) -> None:
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError("Configuration file must contain a YAML mapping")

        self._validate_and_load(raw)

    def _validate_and_load(self, raw: dict) -> None:
        """Validate all parameters and store as typed attributes."""
        # Domain
        domain = self._require_section(raw, "domain")
        if not isinstance(domain, dict):
            raise ValueError("domain must be a mapping")
        self.room_width: float = self._require_positive_float(domain, "width", "domain")
        self.room_height: float = self._require_positive_float(
            domain, "height", "domain"
        )
        self.nx: int = self._require_positive_int(domain, "nx", "domain")
        self.ny: int = self._require_positive_int(domain, "ny", "domain")

        # Fluid
        fluid = self._require_section(raw, "fluid")
        if not isinstance(fluid, dict):
            raise ValueError("fluid must be a mapping")
        self.rho: float = self._require_positive_float(fluid, "density", "fluid")
        self.mu: float = self._require_positive_float(fluid, "viscosity", "fluid")
        self.temperature: float = self._require_positive_float(
            fluid, "temperature", "fluid"
        )

        # Particles
        particles = self._require_section(raw, "particles")
        if not isinstance(particles, dict):
            raise ValueError("particles must be a mapping")
        self.particle_density: float = self._require_positive_float(
            particles, "density", "particles"
        )
        self.particle_sizes: list[float] = self._require_positive_float_list(
            particles, "sizes", "particles"
        )
        self.mean_free_path: float = self._require_positive_float(
            particles, "mean_free_path", "particles"
        )
        self.boundary_layer_thickness: float = self._require_positive_float(
            particles, "boundary_layer_thickness", "particles"
        )

        # HEPA reference data
        if "hepa_reference" not in particles:
            raise ValueError("Missing required key: 'particles.hepa_reference'")
        hepa_raw = particles["hepa_reference"]
        if not isinstance(hepa_raw, dict):
            raise ValueError("particles.hepa_reference must be a mapping")
        hepa_diameters = self._require_positive_float_list(
            hepa_raw, "diameters", "particles.hepa_reference"
        )
        hepa_efficiencies = self._require_float_list(
            hepa_raw, "efficiencies", "particles.hepa_reference"
        )
        if len(hepa_diameters) != len(hepa_efficiencies):
            raise ValueError(
                "particles.hepa_reference.diameters and efficiencies "
                "must have the same length"
            )
        for i in range(len(hepa_diameters) - 1):
            if hepa_diameters[i] >= hepa_diameters[i + 1]:
                raise ValueError(
                    "particles.hepa_reference.diameters must be sorted "
                    f"ascending, but index {i} ({hepa_diameters[i]}) >= "
                    f"index {i + 1} ({hepa_diameters[i + 1]})"
                )
        for eff in hepa_efficiencies:
            if not 0.0 <= eff <= 1.0:
                raise ValueError(f"HEPA efficiency must be in [0, 1], got {eff}")
        self.hepa_reference: HepaReference = HepaReference(
            diameters=hepa_diameters, efficiencies=hepa_efficiencies
        )

        # Solver
        solver = self._require_section(raw, "solver")
        if not isinstance(solver, dict):
            raise ValueError("solver must be a mapping")
        self.dt: float = self._require_positive_float(solver, "dt", "solver")
        self.t_end: float = self._require_positive_float(solver, "t_end", "solver")
        self.output_interval: int = self._require_positive_int(
            solver, "output_interval", "solver"
        )
        self.convergence_tol: float = self._require_positive_float(
            solver, "convergence_tol", "solver"
        )
        self.max_simple_iter: int = self._require_positive_int(
            solver, "max_simple_iter", "solver"
        )

        # Boundaries
        boundaries_raw = self._require_section(raw, "boundaries")
        if not isinstance(boundaries_raw, dict):
            raise ValueError("boundaries must be a mapping")
        self.boundaries: dict[str, BoundarySpec] = {}
        for name, spec in boundaries_raw.items():
            if not isinstance(spec, dict):
                raise ValueError(f"boundaries.{name} must be a mapping")
            ctx = f"boundaries.{name}"
            bc_type = self._require_string(spec, "type", ctx)
            if bc_type not in _VALID_BOUNDARY_TYPES:
                raise ValueError(
                    f"{ctx}.type must be one of "
                    f"{sorted(_VALID_BOUNDARY_TYPES)}, got '{bc_type}'"
                )
            bc_location = self._require_string(spec, "location", ctx)
            if bc_location not in _VALID_BOUNDARY_LOCATIONS:
                raise ValueError(
                    f"{ctx}.location must be one of "
                    f"{sorted(_VALID_BOUNDARY_LOCATIONS)}, got '{bc_location}'"
                )
            bc_velocity = None
            if bc_type == "velocity_inlet":
                raw_vel = spec.get("velocity")
                if raw_vel is None:
                    raise ValueError(
                        f"{ctx}: velocity_inlet requires a 'velocity' field"
                    )
                if isinstance(raw_vel, bool) or not isinstance(raw_vel, (int, float)):
                    raise TypeError(
                        f"{ctx}.velocity must be a number, got {type(raw_vel).__name__}"
                    )
                if raw_vel <= 0:
                    raise ValueError(f"{ctx}.velocity must be positive, got {raw_vel}")
                bc_velocity = float(raw_vel)

            # Coordinate validation based on boundary orientation
            bc_x_start = None
            bc_x_end = None
            bc_y_start = None
            bc_y_end = None
            if bc_location in ("top", "bottom"):
                bc_x_start = self._require_float(spec, "x_start", ctx)
                bc_x_end = self._require_float(spec, "x_end", ctx)
                if bc_x_start >= bc_x_end:
                    raise ValueError(
                        f"{ctx}: x_start ({bc_x_start}) must be less "
                        f"than x_end ({bc_x_end})"
                    )
                if bc_x_start < 0 or bc_x_end > self.room_width:
                    raise ValueError(
                        f"{ctx}: x range [{bc_x_start}, {bc_x_end}] "
                        f"outside domain [0, {self.room_width}]"
                    )
            else:  # left, right
                bc_y_start = self._require_float(spec, "y_start", ctx)
                bc_y_end = self._require_float(spec, "y_end", ctx)
                if bc_y_start >= bc_y_end:
                    raise ValueError(
                        f"{ctx}: y_start ({bc_y_start}) must be less "
                        f"than y_end ({bc_y_end})"
                    )
                if bc_y_start < 0 or bc_y_end > self.room_height:
                    raise ValueError(
                        f"{ctx}: y range [{bc_y_start}, {bc_y_end}] "
                        f"outside domain [0, {self.room_height}]"
                    )

            self.boundaries[name] = BoundarySpec(
                type=bc_type,
                location=bc_location,
                x_start=bc_x_start,
                x_end=bc_x_end,
                y_start=bc_y_start,
                y_end=bc_y_end,
                velocity=bc_velocity,
            )

        # Obstacles (optional)
        obstacles_raw = raw.get("obstacles", [])
        if not isinstance(obstacles_raw, list):
            raise ValueError("obstacles must be a list")
        self.obstacles: list[ObstacleSpec] = []
        for i, obs in enumerate(obstacles_raw):
            if not isinstance(obs, dict):
                raise ValueError(f"obstacles[{i}] must be a mapping")
            ctx = f"obstacles[{i}]"
            x0 = self._require_float(obs, "x_start", ctx)
            x1 = self._require_float(obs, "x_end", ctx)
            y0 = self._require_float(obs, "y_start", ctx)
            y1 = self._require_float(obs, "y_end", ctx)
            if x0 >= x1:
                raise ValueError(
                    f"{ctx}: x_start ({x0}) must be less than x_end ({x1})"
                )
            if y0 >= y1:
                raise ValueError(
                    f"{ctx}: y_start ({y0}) must be less than y_end ({y1})"
                )
            if x0 < 0 or x1 > self.room_width:
                raise ValueError(
                    f"{ctx}: x range [{x0}, {x1}] outside domain [0, {self.room_width}]"
                )
            if y0 < 0 or y1 > self.room_height:
                raise ValueError(
                    f"{ctx}: y range [{y0}, {y1}] outside domain "
                    f"[0, {self.room_height}]"
                )
            self.obstacles.append(
                ObstacleSpec(
                    name=self._require_string(obs, "name", ctx),
                    x_start=x0,
                    x_end=x1,
                    y_start=y0,
                    y_end=y1,
                )
            )

        # Sensors
        sensors_raw = self._require_section(raw, "sensors")
        if not isinstance(sensors_raw, list):
            raise ValueError("sensors must be a list")
        self.sensors: list[SensorSpec] = []
        for i, sensor in enumerate(sensors_raw):
            if not isinstance(sensor, dict):
                raise ValueError(f"sensors[{i}] must be a mapping")
            ctx = f"sensors[{i}]"
            sx = self._require_float(sensor, "x", ctx)
            sy = self._require_float(sensor, "y", ctx)
            if not 0 <= sx <= self.room_width:
                raise ValueError(f"{ctx}: x={sx} outside domain [0, {self.room_width}]")
            if not 0 <= sy <= self.room_height:
                raise ValueError(
                    f"{ctx}: y={sy} outside domain [0, {self.room_height}]"
                )
            self.sensors.append(
                SensorSpec(
                    name=self._require_string(sensor, "name", ctx),
                    x=sx,
                    y=sy,
                )
            )

        # Thresholds
        thresholds_raw = self._require_section(raw, "thresholds")
        if not isinstance(thresholds_raw, dict):
            raise ValueError("thresholds must be a mapping")
        self.thresholds: dict[str, float] = {}
        for key, val in thresholds_raw.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(
                    f"thresholds.{key} must be a number, got {type(val).__name__}"
                )
            if val < 0:
                raise ValueError(f"thresholds.{key} must be non-negative, got {val}")
            self.thresholds[str(key)] = float(val)

    # -- Validation helpers --------------------------------------------------

    @staticmethod
    def _require_section(raw: dict, key: str) -> dict | list:
        """Require a top-level section exists in the config."""
        if key not in raw:
            raise ValueError(f"Missing required config section: '{key}'")
        return raw[key]

    @staticmethod
    def _require_string(section: dict, key: str, context: str) -> str:
        """Require a string value in a config section."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = section[key]
        if not isinstance(val, str):
            raise TypeError(
                f"{context}.{key} must be a string, got {type(val).__name__}"
            )
        return val

    @staticmethod
    def _require_float(section: dict, key: str, context: str) -> float:
        """Require a numeric value, return as float."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = section[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(
                f"{context}.{key} must be a number, got {type(val).__name__}"
            )
        return float(val)

    @staticmethod
    def _require_positive_float(section: dict, key: str, context: str) -> float:
        """Require a positive numeric value, return as float."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = section[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(
                f"{context}.{key} must be a number, got {type(val).__name__}"
            )
        if val <= 0:
            raise ValueError(f"{context}.{key} must be positive, got {val}")
        return float(val)

    @staticmethod
    def _require_positive_int(section: dict, key: str, context: str) -> int:
        """Require a positive integer value."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = section[key]
        if not isinstance(val, int) or isinstance(val, bool):
            raise TypeError(
                f"{context}.{key} must be an integer, got {type(val).__name__}"
            )
        if val <= 0:
            raise ValueError(f"{context}.{key} must be positive, got {val}")
        return val

    @staticmethod
    def _require_float_list(section: dict, key: str, context: str) -> list[float]:
        """Require a list of numeric values, return as list of floats."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = section[key]
        if not isinstance(val, list):
            raise TypeError(f"{context}.{key} must be a list, got {type(val).__name__}")
        result = []
        for i, item in enumerate(val):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(
                    f"{context}.{key}[{i}] must be a number, got {type(item).__name__}"
                )
            result.append(float(item))
        return result

    @staticmethod
    def _require_positive_float_list(
        section: dict, key: str, context: str
    ) -> list[float]:
        """Require a list of positive numeric values, return as list of floats."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = section[key]
        if not isinstance(val, list):
            raise TypeError(f"{context}.{key} must be a list, got {type(val).__name__}")
        if len(val) == 0:
            raise ValueError(f"{context}.{key} must not be empty")
        result = []
        for i, item in enumerate(val):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(
                    f"{context}.{key}[{i}] must be a number, got {type(item).__name__}"
                )
            if item <= 0:
                raise ValueError(f"{context}.{key}[{i}] must be positive, got {item}")
            result.append(float(item))
        return result
