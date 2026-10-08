"""YAML configuration loader with validation for the CFD simulation.

Loads simulation parameters from a YAML file, validates all values at
load time, and provides typed attribute access. Satisfies REQ-C01
(single source of truth) and REQ-C02 (fail-fast validation).
"""

import math
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
        "fixed_flow_outlet" or "wall".
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
        Applied normal to the wall surface. On a fixed_flow_outlet, the
        outward normal velocity it holds, or None for an outlet that
        shares the inflow the stated outlets leave (ADR-012 D, amended
        2026-10-08).
    u_velocity : float or None
        Explicit u-component for velocity_inlet. When present,
        overrides the automatic normal decomposition of velocity.
    v_velocity : float or None
        Explicit v-component for velocity_inlet. When present,
        overrides the automatic normal decomposition of velocity.
    concentration : tuple[float, ...] or None
        Upstream concentration carried by a velocity_inlet, one
        non-negative value per configured particle class, particles per
        cubic meter. None means a clean supply (ADR-011 E). A tuple, so
        the loaded configuration cannot be edited in place.
    hepa_filtered : bool
        velocity_inlet only. When True the carried concentration is
        reduced by the class's HEPA efficiency. Default False.
    deposition_surface : str or None
        wall only: "floor", "ceiling", "wall" or "none", overriding the
        edge's default surface for deposition. None means the edge
        decides (bottom is floor, top is ceiling, left and right are
        walls).

    The three concentration keys are keyword-only. A velocity_inlet whose
    prescribed normal component is zero (a tangential lid) admits no air
    and is a wall to the scalar layer (ADR-011 E, amended 2026-10-03), so
    it may carry neither ``concentration`` nor ``hepa_filtered``.
    """

    type: str
    location: str
    x_start: float | None = None
    x_end: float | None = None
    y_start: float | None = None
    y_end: float | None = None
    velocity: float | None = None
    u_velocity: float | None = None
    v_velocity: float | None = None
    concentration: tuple[float, ...] | None = field(default=None, kw_only=True)
    hepa_filtered: bool = field(default=False, kw_only=True)
    deposition_surface: str | None = field(default=None, kw_only=True)


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


@dataclass(frozen=True)
class StretchSpec:
    """Geometric wall clustering for one mesh axis.

    Exactly one of the two quantities is specified; the mesh derives the
    other because the cell count fixes their relationship (see
    src/mesh.py). The default is a uniform axis.

    Parameters
    ----------
    ratio : float
        Geometric ratio between adjacent cell widths from each wall toward
        the center, >= 1. A ratio of 1 is a uniform axis.
    min_spacing : float or None
        Width of the wall-adjacent cell in meters. When given, the ratio
        is derived from it and ``ratio`` is ignored.
    """

    ratio: float = 1.0
    min_spacing: float | None = None


@dataclass(frozen=True)
class TransportSpec:
    """The transport section: what the scalar solver reads (ADR-011 I).

    Parameters
    ----------
    cfl_number : float
        Courant number the explicit advection step runs at, in
        (0, CFL_NUMBER_BOUND]. An accuracy choice below the bound: the
        forward Euler error grows with it.
    advection_scheme : str
        "umist" (QUICK bounded by the UMIST limiter, the default) or
        "upwind" (first order, the comparison scheme).
    max_diffusion_iter : int
        Cap on Jacobi sweeps of the implicit diffusion and deposition
        system per class per step.
    diffusion_tol : float
        Tolerance the implicit diffusion solve iterates to.
    turbulent_schmidt : float or None
        Sc_t of ``D_t = nu_t / Sc_t``, the turbulent particle diffusivity
        (REQ-T13, ADR-012 F and decision 8): positive and finite, with no
        default in code. None when the key is absent, and the solver then
        refuses an eddy viscosity field. A laminar configuration needs none.
    """

    cfl_number: float
    advection_scheme: str
    max_diffusion_iter: int
    diffusion_tol: float
    turbulent_schmidt: float | None = None


@dataclass(frozen=True)
class TurbulenceSpec:
    """The turbulence section: what the k-epsilon model reads (ADR-012 I).

    Present means the model is configured; absent, the model is off and
    every other result is unchanged. Until ECR-002 step 6 couples the model
    into the flow solver, nothing in the solver reads it either.

    Parameters
    ----------
    model : str
        "k_epsilon", the one model ECR-002 builds.
    variant : str
        "standard" (the default) or "rng" (ADR-012 A, decision 2).
    wall_treatment : str
        "scalable_wall_functions" (ADR-012 B, decision 3).
    cfl_number : float
        The pseudo-time Courant number of the k and eps step, in
        (0, CFL_NUMBER_BOUND] (ADR-012 C).
    alpha_turbulence : float
        Under-relaxation of the eddy viscosity in the coupled solve, in
        (0, 1].
    max_iter : int
        Cap on Jacobi sweeps of each implicit k and eps solve.
    tol : float
        Tolerance the implicit k and eps solves iterate to.
    """

    model: str
    variant: str
    wall_treatment: str
    cfl_number: float
    alpha_turbulence: float
    max_iter: int
    tol: float


_VALID_BOUNDARY_TYPES: set[str] = {
    "velocity_inlet",
    "pressure_outlet",
    "fixed_flow_outlet",
    "wall",
}
_VALID_BOUNDARY_LOCATIONS: set[str] = {"top", "bottom", "left", "right"}

# Stopping rules the solver block may name (src/stopping.py). velocity_step is
# the default, the rule every stored result was produced under.
VELOCITY_STEP = "velocity_step"
ERROR_ESTIMATE = "error_estimate"
STOPPING_RULES: tuple[str, ...] = (VELOCITY_STEP, ERROR_ESTIMATE)
DEFAULT_ITERATION_ERROR_TOL = 1.0e-6
# ECR-001 acceptance criterion 6: the per-cell imbalance and its signed domain
# sum each below 1e-10, absolute.
DEFAULT_MASS_IMBALANCE_TOL = 1.0e-10
# The relative residual the pressure correction solves to (REQ-S08 as amended
# 2026-10-06, ADR-013 D). The lower bound: on the product's first correction
# from rest the true residual cannot fall below about 1.3e-13 of the flux
# scale, so a level at or below about 1e-12 would run every such correction to
# its cap; 1e-10 keeps two orders above that and two below the default 1e-8.
# The upper bound excludes a level that asks for nothing.
PRESSURE_RTOL_BOUNDS: tuple[float, float] = (1.0e-10, 1.0)
# The key the weighted Jacobi solve read until ECR-003 step 1: pascals of
# change per sweep, which means nothing for the conjugate gradient solve. It is
# refused by name so a saved configuration cannot be read as the new key.
RETIRED_PRESSURE_TOL_KEY = "pressure_tol"
# Every key the solver block accepts, nine required and three optional, in the
# order the harness records them: it reads this tuple, so a key added here is
# in every harness row (GitHub issue 38). With optional keys a misspelt one
# would otherwise fall back to its default.
SOLVER_KEYS: tuple[str, ...] = (
    "dt",
    "t_end",
    "output_interval",
    "convergence_tol",
    "max_simple_iter",
    "alpha_velocity",
    "alpha_pressure",
    "max_pressure_iter",
    "pressure_rtol",
    "stopping_rule",
    "iteration_error_tol",
    "mass_imbalance_tol",
)

# Transport section (ADR-011 I), optional: the velocity-only validation cases
# do not carry it. The limited face scheme is bounded under forward Euler at
# a Courant number of at most 1/2 (ADR-011 B); the bound is a property of the
# scheme and lives here as a constant. cfl_number is the Courant number a run
# advects at, at most the bound.
CFL_NUMBER_BOUND = 0.5
UMIST = "umist"
UPWIND = "upwind"
ADVECTION_SCHEMES: tuple[str, ...] = (UMIST, UPWIND)
_TRANSPORT_KEYS: frozenset[str] = frozenset(
    {
        "cfl_number",
        "advection_scheme",
        "max_diffusion_iter",
        "diffusion_tol",
        "turbulent_schmidt",
    }
)

# Turbulence section (ADR-012 I), optional: absent means the model is off.
# The k and eps step uses the transport scheme in pseudo-time, so its Courant
# number has the scheme's bound, CFL_NUMBER_BOUND. The model constants are
# not configuration: they define the published variants and live in
# src/turbulence.py.
K_EPSILON = "k_epsilon"
TURBULENCE_MODELS: tuple[str, ...] = (K_EPSILON,)
STANDARD = "standard"
RNG = "rng"
TURBULENCE_VARIANTS: tuple[str, ...] = (STANDARD, RNG)
SCALABLE_WALL_FUNCTIONS = "scalable_wall_functions"
WALL_TREATMENTS: tuple[str, ...] = (SCALABLE_WALL_FUNCTIONS,)
_TURBULENCE_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "variant",
        "wall_treatment",
        "cfl_number",
        "alpha_turbulence",
        "max_iter",
        "tol",
    }
)

# Every key a boundary segment accepts. With optional keys a misspelt one
# would otherwise change the physics with no message (REQ-C02).
_SEGMENT_KEYS: frozenset[str] = frozenset(
    {
        "type",
        "location",
        "x_start",
        "x_end",
        "y_start",
        "y_end",
        "velocity",
        "u_velocity",
        "v_velocity",
        "concentration",
        "hepa_filtered",
        "deposition_surface",
    }
)

# Surfaces a wall segment may name for deposition (ADR-011 E). The first
# three are the orientations ParticlePhysics.deposition_velocity accepts;
# "none" switches deposition off on the segment.
DEPOSITION_FLOOR = "floor"
DEPOSITION_CEILING = "ceiling"
DEPOSITION_WALL = "wall"
DEPOSITION_NONE = "none"
DEPOSITION_SURFACES: tuple[str, ...] = (
    DEPOSITION_FLOOR,
    DEPOSITION_CEILING,
    DEPOSITION_WALL,
    DEPOSITION_NONE,
)


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

    @classmethod
    def from_dict(cls, raw: dict) -> "SimConfig":
        """Build a configuration from an already-parsed mapping.

        Applies exactly the validation the file constructor applies. Used
        where a committed configuration needs one field overridden before
        loading, such as the grid in a refinement study.

        Parameters
        ----------
        raw : dict
            Mapping with the same structure as a configuration file.

        Returns
        -------
        SimConfig
            Validated configuration.

        Raises
        ------
        ValueError
            If raw is not a mapping or any parameter fails validation.
        """
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a mapping")
        config = cls.__new__(cls)
        config._validate_and_load(raw)
        return config

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

        # Mesh stretching. Optional section; absent means uniform on both axes.
        mesh_raw = raw.get("mesh", {})
        if mesh_raw is None:
            mesh_raw = {}
        if not isinstance(mesh_raw, dict):
            raise ValueError("mesh must be a mapping")
        for key in mesh_raw:
            if key not in ("x", "y"):
                raise ValueError(f"mesh.{key} is not a recognised axis; use x or y")
        self.stretch_x: StretchSpec = self._parse_stretch(
            mesh_raw.get("x"), "mesh.x", self.room_width, self.nx
        )
        self.stretch_y: StretchSpec = self._parse_stretch(
            mesh_raw.get("y"), "mesh.y", self.room_height, self.ny
        )

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
        if RETIRED_PRESSURE_TOL_KEY in solver:
            lower, upper = PRESSURE_RTOL_BOUNDS
            raise ValueError(
                f"solver.{RETIRED_PRESSURE_TOL_KEY} was the weighted Jacobi solve's "
                "stop, pascals of change per sweep, and was retired with it "
                "(ECR-003, 2026-10-06); the conjugate gradient solve stops on the "
                f"relative residual solver.pressure_rtol, in [{lower}, {upper}), with "
                "max_pressure_iter its iteration cap"
            )
        for key in solver:
            if key not in SOLVER_KEYS:
                raise ValueError(
                    f"solver.{key} is not a recognised solver key; "
                    f"known: {sorted(SOLVER_KEYS)}"
                )
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
        self.alpha_velocity: float = self._require_relaxation_factor(
            solver, "alpha_velocity", "solver"
        )
        self.alpha_pressure: float = self._require_relaxation_factor(
            solver, "alpha_pressure", "solver"
        )
        self.max_pressure_iter: int = self._require_positive_int(
            solver, "max_pressure_iter", "solver"
        )
        self.pressure_rtol: float = self._require_positive_float(
            solver, "pressure_rtol", "solver"
        )
        lower, upper = PRESSURE_RTOL_BOUNDS
        if not lower <= self.pressure_rtol < upper:
            raise ValueError(
                f"solver.pressure_rtol must be in [{lower}, {upper}), the relative "
                f"residual the pressure correction solves to, got {self.pressure_rtol}"
            )
        # Optional: absent keys give the velocity-step rule and its behaviour.
        self.stopping_rule: str = VELOCITY_STEP
        if "stopping_rule" in solver:
            self.stopping_rule = self._require_string(solver, "stopping_rule", "solver")
            if self.stopping_rule not in STOPPING_RULES:
                raise ValueError(
                    f"solver.stopping_rule must be one of {list(STOPPING_RULES)}, "
                    f"got '{self.stopping_rule}'"
                )
        self.iteration_error_tol: float = self._optional_positive_float(
            solver, "iteration_error_tol", "solver", DEFAULT_ITERATION_ERROR_TOL
        )
        self.mass_imbalance_tol: float = self._optional_positive_float(
            solver, "mass_imbalance_tol", "solver", DEFAULT_MASS_IMBALANCE_TOL
        )

        # Transport (optional). Unknown keys raise as they do in the solver
        # block, so a misspelt key cannot fall back to a default.
        self.transport: TransportSpec | None = None
        if "transport" in raw:
            self.transport = self._parse_transport(raw["transport"])

        # Turbulence (optional, ADR-012 I). Absent means the model is off.
        self.turbulence: TurbulenceSpec | None = None
        if "turbulence" in raw:
            self.turbulence = self._parse_turbulence(raw["turbulence"])

        # Boundaries
        boundaries_raw = self._require_section(raw, "boundaries")
        if not isinstance(boundaries_raw, dict):
            raise ValueError("boundaries must be a mapping")
        self.boundaries: dict[str, BoundarySpec] = {}
        for name, spec in boundaries_raw.items():
            if not isinstance(spec, dict):
                raise ValueError(f"boundaries.{name} must be a mapping")
            ctx = f"boundaries.{name}"
            for key in spec:
                if key not in _SEGMENT_KEYS:
                    raise ValueError(
                        f"{ctx}.{key} is not a recognised boundary key; "
                        f"known: {sorted(_SEGMENT_KEYS)}"
                    )
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
            bc_u_velocity = None
            bc_v_velocity = None
            if bc_type == "velocity_inlet":
                # Support explicit u/v components or normal-direction magnitude
                raw_u = spec.get("u_velocity")
                raw_v = spec.get("v_velocity")
                raw_vel = spec.get("velocity")

                has_components = raw_u is not None or raw_v is not None
                if not has_components and raw_vel is None:
                    raise ValueError(
                        f"{ctx}: velocity_inlet requires 'velocity' or "
                        f"'u_velocity'/'v_velocity' fields"
                    )

                if raw_vel is not None:
                    bc_velocity = self._finite_number(raw_vel, f"{ctx}.velocity")
                    if bc_velocity <= 0:
                        raise ValueError(
                            f"{ctx}.velocity must be positive, got {raw_vel}"
                        )

                bc_u_velocity = (
                    self._finite_number(raw_u, f"{ctx}.u_velocity")
                    if raw_u is not None
                    else None
                )
                bc_v_velocity = (
                    self._finite_number(raw_v, f"{ctx}.v_velocity")
                    if raw_v is not None
                    else None
                )

                # A zero normal component (a tangential lid) admits no air: the
                # scalar layer treats the segment as a wall, so the inlet
                # concentration keys have nothing to describe.
                normal = (
                    bc_v_velocity if bc_location in ("top", "bottom") else bc_u_velocity
                )
                if has_components and not normal:
                    for key in ("concentration", "hepa_filtered"):
                        if key in spec:
                            raise ValueError(
                                f"{ctx}.{key} is not valid on a velocity_inlet whose "
                                "normal velocity is zero: no air crosses it and the "
                                "scalar layer treats it as a wall"
                            )

            if bc_type == "fixed_flow_outlet":
                # A fixed flow is a normal velocity and nothing else. The inlet's
                # component and concentration keys describe air entering, and
                # the outlet carries nothing in (ADR-011 E).
                for key in (
                    "u_velocity",
                    "v_velocity",
                    "concentration",
                    "hepa_filtered",
                    "deposition_surface",
                ):
                    if key in spec:
                        raise ValueError(
                            f"{ctx}.{key} is not valid on a fixed_flow_outlet; "
                            "it holds an outward normal 'velocity' only"
                        )
                if spec.get("velocity") is not None:
                    bc_velocity = self._finite_number(
                        spec["velocity"], f"{ctx}.velocity"
                    )
                    if bc_velocity <= 0:
                        raise ValueError(
                            f"{ctx}.velocity must be positive (outward), "
                            f"got {spec['velocity']}"
                        )

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

            # Concentration keys (ADR-011 E). Each is allowed on one segment
            # type only, so a key on the wrong type is an error, not ignored.
            bc_concentration = None
            if "concentration" in spec:
                self._require_segment_type(
                    spec, "concentration", ctx, bc_type, "velocity_inlet"
                )
                bc_concentration = tuple(
                    self._require_float_list(spec, "concentration", ctx)
                )
                n_classes = len(self.particle_sizes)
                if len(bc_concentration) != n_classes:
                    raise ValueError(
                        f"{ctx}.concentration must have one value per particle "
                        f"class ({n_classes}), got {len(bc_concentration)}"
                    )
                for i, value in enumerate(bc_concentration):
                    if value < 0:
                        raise ValueError(
                            f"{ctx}.concentration[{i}] must be non-negative, "
                            f"got {value}"
                        )
            bc_hepa_filtered = False
            if "hepa_filtered" in spec:
                self._require_segment_type(
                    spec, "hepa_filtered", ctx, bc_type, "velocity_inlet"
                )
                raw_flag = spec["hepa_filtered"]
                if not isinstance(raw_flag, bool):
                    raise TypeError(
                        f"{ctx}.hepa_filtered must be a bool, "
                        f"got {type(raw_flag).__name__}"
                    )
                bc_hepa_filtered = raw_flag
            bc_deposition_surface = None
            if "deposition_surface" in spec:
                self._require_segment_type(
                    spec, "deposition_surface", ctx, bc_type, "wall"
                )
                bc_deposition_surface = self._require_string(
                    spec, "deposition_surface", ctx
                )
                if bc_deposition_surface not in DEPOSITION_SURFACES:
                    raise ValueError(
                        f"{ctx}.deposition_surface must be one of "
                        f"{list(DEPOSITION_SURFACES)}, got '{bc_deposition_surface}'"
                    )

            self.boundaries[name] = BoundarySpec(
                type=bc_type,
                location=bc_location,
                x_start=bc_x_start,
                x_end=bc_x_end,
                y_start=bc_y_start,
                y_end=bc_y_end,
                velocity=bc_velocity,
                u_velocity=bc_u_velocity,
                v_velocity=bc_v_velocity,
                concentration=bc_concentration,
                hepa_filtered=bc_hepa_filtered,
                deposition_surface=bc_deposition_surface,
            )

        self._reject_overlapping_segments()
        self._check_fixed_flow_outlets()

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
            number = self._finite_number(val, f"thresholds.{key}")
            if number < 0:
                raise ValueError(f"thresholds.{key} must be non-negative, got {val}")
            self.thresholds[str(key)] = number

    def _reject_overlapping_segments(self) -> None:
        """Raise if two segments on one edge overlap.

        Where two ranges intersect in more than a point the file describes
        two conditions for one face; the registry would give the face to
        the first segment and a reader counting by segment would count it
        twice. Ranges that meet at one coordinate are allowed, and the
        first segment in configuration order decides there.
        """
        names = list(self.boundaries)
        for i, a_name in enumerate(names):
            a = self.boundaries[a_name]
            a0, a1 = self._segment_range(a)
            for b_name in names[i + 1 :]:
                b = self.boundaries[b_name]
                if b.location != a.location:
                    continue
                b0, b1 = self._segment_range(b)
                if a0 < b1 and b0 < a1:
                    raise ValueError(
                        f"boundaries.{a_name} and boundaries.{b_name} overlap on "
                        f"the {a.location} edge ([{a0}, {a1}] and [{b0}, {b1}]); "
                        "segments on one edge must not overlap"
                    )

    def _check_fixed_flow_outlets(self) -> None:
        """Raise unless the fixed-flow outlets leave the flow balance determined.

        Without a pressure outlet the outflow must equal the inflow, so
        at least one fixed-flow outlet has to state no velocity and take
        the remainder; if every one stated a velocity the configuration
        would over-determine the balance. With a pressure outlet, that
        outlet balances the flow and every fixed-flow outlet must state
        its velocity. What needs the mesh (a remainder that is not
        positive) is checked when the boundary is built.
        """
        fixed = {
            name: spec
            for name, spec in self.boundaries.items()
            if spec.type == "fixed_flow_outlet"
        }
        if not fixed:
            return
        unstated = [name for name, spec in fixed.items() if spec.velocity is None]
        has_pressure_outlet = any(
            spec.type == "pressure_outlet" for spec in self.boundaries.values()
        )
        if has_pressure_outlet and unstated:
            raise ValueError(
                f"boundaries {unstated} are fixed_flow_outlet segments with no "
                "velocity, but the configuration has a pressure_outlet that "
                "balances the flow; give each fixed_flow_outlet a velocity"
            )
        if not has_pressure_outlet and not unstated:
            raise ValueError(
                f"every fixed_flow_outlet {sorted(fixed)} states a velocity and the "
                "configuration has no pressure_outlet, so nothing balances the "
                "flow; leave the velocity off at least one of them"
            )

    @staticmethod
    def _segment_range(spec: BoundarySpec) -> tuple[float, float]:
        """The segment's range along its edge."""
        if spec.location in ("top", "bottom"):
            return spec.x_start, spec.x_end
        return spec.y_start, spec.y_end

    # -- Validation helpers --------------------------------------------------

    @classmethod
    def _parse_stretch(
        cls, section: dict | None, context: str, length: float, n: int
    ) -> StretchSpec:
        """Parse one axis of the mesh section into a StretchSpec.

        Accepts ``stretch_ratio`` (>= 1) or ``min_wall_spacing`` (in
        (0, length / n]) but not both. An absent or empty axis is uniform.
        """
        if section is None:
            return StretchSpec()
        if not isinstance(section, dict):
            raise ValueError(f"{context} must be a mapping")
        for key in section:
            if key not in ("stretch_ratio", "min_wall_spacing"):
                raise ValueError(
                    f"{context}.{key} is not recognised; use stretch_ratio or "
                    "min_wall_spacing"
                )
        has_ratio = "stretch_ratio" in section
        has_spacing = "min_wall_spacing" in section
        if has_ratio and has_spacing:
            raise ValueError(
                f"{context}: give stretch_ratio or min_wall_spacing, not both; the "
                "cell count fixes the other"
            )
        if has_spacing:
            spacing = cls._require_positive_float(section, "min_wall_spacing", context)
            uniform = length / n
            if spacing > uniform:
                raise ValueError(
                    f"{context}.min_wall_spacing must not exceed the uniform spacing "
                    f"{uniform} for {n} cells on {length}, got {spacing}"
                )
            return StretchSpec(ratio=1.0, min_spacing=spacing)
        if has_ratio:
            ratio = cls._require_positive_float(section, "stretch_ratio", context)
            if ratio < 1.0:
                raise ValueError(
                    f"{context}.stretch_ratio must be >= 1 (1 is uniform), got {ratio}"
                )
            return StretchSpec(ratio=ratio, min_spacing=None)
        return StretchSpec()

    @classmethod
    def _parse_transport(cls, section: object) -> TransportSpec:
        """Parse the transport section into a TransportSpec.

        ``cfl_number`` must lie in (0, CFL_NUMBER_BOUND]; ``advection_scheme``
        defaults to umist; ``turbulent_schmidt`` is optional, positive and
        finite when present, and None when absent; the other two keys are
        required. Any other key raises.
        """
        if not isinstance(section, dict):
            raise ValueError("transport must be a mapping")
        for key in section:
            if key not in _TRANSPORT_KEYS:
                raise ValueError(
                    f"transport.{key} is not a recognised transport key; "
                    f"known: {sorted(_TRANSPORT_KEYS)}"
                )
        cfl_number = cls._require_positive_float(section, "cfl_number", "transport")
        if cfl_number > CFL_NUMBER_BOUND:
            raise ValueError(
                f"transport.cfl_number must be in (0, {CFL_NUMBER_BOUND}], the "
                f"limited scheme's stability bound, got {cfl_number}"
            )
        scheme = UMIST
        if "advection_scheme" in section:
            scheme = cls._require_string(section, "advection_scheme", "transport")
            if scheme not in ADVECTION_SCHEMES:
                raise ValueError(
                    f"transport.advection_scheme must be one of "
                    f"{list(ADVECTION_SCHEMES)}, got '{scheme}'"
                )
        # No default: the key is needed exactly where an eddy viscosity field
        # is used, and a value assumed here would be a parameter defined in
        # code (REQ-C01).
        turbulent_schmidt = None
        if "turbulent_schmidt" in section:
            turbulent_schmidt = cls._require_positive_float(
                section, "turbulent_schmidt", "transport"
            )
        return TransportSpec(
            cfl_number=cfl_number,
            advection_scheme=scheme,
            max_diffusion_iter=cls._require_positive_int(
                section, "max_diffusion_iter", "transport"
            ),
            diffusion_tol=cls._require_positive_float(
                section, "diffusion_tol", "transport"
            ),
            turbulent_schmidt=turbulent_schmidt,
        )

    @classmethod
    def _parse_turbulence(cls, section: object) -> TurbulenceSpec:
        """Parse the turbulence section into a TurbulenceSpec.

        ``variant`` defaults to standard; every other key is required.
        ``cfl_number`` must lie in (0, CFL_NUMBER_BOUND] and
        ``alpha_turbulence`` in (0, 1]. Any other key raises.
        """
        if not isinstance(section, dict):
            raise ValueError("turbulence must be a mapping")
        for key in section:
            if key not in _TURBULENCE_KEYS:
                raise ValueError(
                    f"turbulence.{key} is not a recognised turbulence key; "
                    f"known: {sorted(_TURBULENCE_KEYS)}"
                )
        model = cls._require_choice(section, "model", "turbulence", TURBULENCE_MODELS)
        variant = STANDARD
        if "variant" in section:
            variant = cls._require_choice(
                section, "variant", "turbulence", TURBULENCE_VARIANTS
            )
        wall_treatment = cls._require_choice(
            section, "wall_treatment", "turbulence", WALL_TREATMENTS
        )
        cfl_number = cls._require_positive_float(section, "cfl_number", "turbulence")
        if cfl_number > CFL_NUMBER_BOUND:
            raise ValueError(
                f"turbulence.cfl_number must be in (0, {CFL_NUMBER_BOUND}], the "
                f"limited scheme's stability bound, got {cfl_number}"
            )
        return TurbulenceSpec(
            model=model,
            variant=variant,
            wall_treatment=wall_treatment,
            cfl_number=cfl_number,
            alpha_turbulence=cls._require_relaxation_factor(
                section, "alpha_turbulence", "turbulence"
            ),
            max_iter=cls._require_positive_int(section, "max_iter", "turbulence"),
            tol=cls._require_positive_float(section, "tol", "turbulence"),
        )

    @classmethod
    def _require_choice(
        cls, section: dict, key: str, context: str, choices: tuple[str, ...]
    ) -> str:
        """Require a string that is one of ``choices``."""
        value = cls._require_string(section, key, context)
        if value not in choices:
            raise ValueError(
                f"{context}.{key} must be one of {list(choices)}, got '{value}'"
            )
        return value

    @staticmethod
    def _require_segment_type(
        section: dict, key: str, context: str, bc_type: str, allowed: str
    ) -> None:
        """Raise unless a segment key sits on the one segment type it is valid for."""
        if key in section and bc_type != allowed:
            raise ValueError(
                f"{context}.{key} is only valid on a {allowed} segment, "
                f"got type '{bc_type}'"
            )

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
    def _finite_number(val: object, label: str) -> float:
        """A finite float from a non-bool number; TypeError or ValueError otherwise.

        Every numeric key passes through here (REQ-C02): a bool is an int in
        Python and NaN compares false against every bound, so neither is
        caught by a range check alone.
        """
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{label} must be a number, got {type(val).__name__}")
        if not math.isfinite(val):
            raise ValueError(f"{label} must be finite, got {val}")
        return float(val)

    @classmethod
    def _require_float(cls, section: dict, key: str, context: str) -> float:
        """Require a finite numeric value, return as float."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        return cls._finite_number(section[key], f"{context}.{key}")

    @classmethod
    def _require_positive_float(cls, section: dict, key: str, context: str) -> float:
        """Require a positive finite numeric value, return as float."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = cls._finite_number(section[key], f"{context}.{key}")
        if val <= 0:
            raise ValueError(f"{context}.{key} must be positive, got {val}")
        return val

    @classmethod
    def _optional_positive_float(
        cls, section: dict, key: str, context: str, default: float
    ) -> float:
        """A positive numeric value if the key is present, else the default."""
        if key not in section:
            return default
        return cls._require_positive_float(section, key, context)

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

    @classmethod
    def _require_float_list(cls, section: dict, key: str, context: str) -> list[float]:
        """Require a list of finite numeric values, return as list of floats."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = section[key]
        if not isinstance(val, list):
            raise TypeError(f"{context}.{key} must be a list, got {type(val).__name__}")
        return [
            cls._finite_number(item, f"{context}.{key}[{i}]")
            for i, item in enumerate(val)
        ]

    @classmethod
    def _require_positive_float_list(
        cls, section: dict, key: str, context: str
    ) -> list[float]:
        """Require a list of positive finite numeric values, return as list of floats."""
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = section[key]
        if not isinstance(val, list):
            raise TypeError(f"{context}.{key} must be a list, got {type(val).__name__}")
        if len(val) == 0:
            raise ValueError(f"{context}.{key} must not be empty")
        result = []
        for i, item in enumerate(val):
            number = cls._finite_number(item, f"{context}.{key}[{i}]")
            if number <= 0:
                raise ValueError(f"{context}.{key}[{i}] must be positive, got {item}")
            result.append(number)
        return result

    @classmethod
    def _require_relaxation_factor(cls, section: dict, key: str, context: str) -> float:
        """Require a relaxation factor in the range (0.0, 1.0].

        Zero is rejected because it causes division by zero in
        SIMPLE. Values above 1.0 cause divergence.
        """
        if key not in section:
            raise ValueError(f"Missing required key: '{context}.{key}'")
        val = cls._finite_number(section[key], f"{context}.{key}")
        if val <= 0.0 or val > 1.0:
            raise ValueError(f"{context}.{key} must be in (0.0, 1.0], got {val}")
        return val
