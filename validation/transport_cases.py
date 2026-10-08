"""The transport validation cases of ADR-011 section H, as data.

Each case is one function returning a TransportCase: the configuration, the
mesh, the face velocity field set directly on the staggered faces, the scalar
condition at every face, the particle properties the case fixes, the initial
field as exact cell averages and the analytical field it is scored against.
tests/test_diffusion.py, tests/test_advection.py, tests/test_smith_hutton.py,
tests/test_sealed_box.py and the item 0 reproduction under results/builder32/
read one instrument, so a threshold judged in a test and a figure reported
beside it come from the same inputs.

Three things here stand in for what a product run takes from elsewhere.
The face field is written on the faces by a formula instead of coming from
StaggeredSolver.face_velocities, because every case needs an exact field
(uniform, solid-body rotation, the Smith-Hutton streamfunction) whose discrete
divergence is known. The scalar conditions are built here instead of by
ConcentrationBoundary, because the cases switch deposition and settling off,
hand the solver a profile at an inlet, or set a floor deposition velocity
equal to the settling velocity exactly (ADR-011 H, VAL-014). And the particle
properties are a ScalarPhysics with the settling velocity and diffusion
coefficient the case names, because a physical class cannot have D = 1e-3
or D = 0. The transport solver reads exactly settling_velocity,
diffusion_coeff and faces_for of what it is handed, so these stand-ins
satisfy its contract (src/solver_transport.py).

Lives outside src/ on purpose, with validation/metrics.py: the simulation
never imports it.
"""

import math
from dataclasses import dataclass

import numpy as np

from src.boundary_concentration import SURFACE_FLOOR, ConcentrationFaces
from src.config import UMIST, SimConfig
from src.mesh import Mesh
from src.particles import ParticlePhysics
from src.staggered import FaceVelocities, u_shape, v_shape

# The particle class every case carries: 5 um, the settling-dominated class
# of ADR-011 G's table, so a case that reads ParticlePhysics gets the
# largest settling velocity. Fluid properties are air at the product's
# density of 1.2, as section H asks (non-unit, on purpose).
PARTICLE_SIZE = 5.0e-6
AIR = {"density": 1.2, "viscosity": 1.81e-5, "temperature": 293.0}
PARTICLES = {
    "density": 1000.0,
    "sizes": [PARTICLE_SIZE],
    "mean_free_path": 67.0e-9,
    "boundary_layer_thickness": 1.0e-3,
    "hepa_reference": {"diameters": [PARTICLE_SIZE], "efficiencies": [0.99999]},
}
# The velocity solver's block, present because SimConfig requires it; no case
# here runs the velocity solver.
SOLVER_BLOCK = {
    "dt": 0.01,
    "t_end": 1.0,
    "output_interval": 10,
    "convergence_tol": 1.0e-6,
    "max_simple_iter": 100,
    "alpha_velocity": 0.7,
    "alpha_pressure": 0.3,
    "max_pressure_iter": 5000,
    "pressure_rtol": 1.0e-8,
}

# ADR-011 H: the validation cases advect at the product case's Courant number.
CFL_NUMBER = 0.1
# The supply speed of the product case, which rows 1 of VAL-004 and VAL-013 use.
SUPPLY_SPEED = 0.45
# ADR-012 decision 8: the turbulent Schmidt number the gate rows with an eddy
# viscosity field run at, the value the default configuration carries.
TURBULENT_SCHMIDT = 0.7
# ADR-012 F: the room's core has cell Peclet numbers U dx / D_t of about 8 to
# 250 on the turbulent diffusivity; the prescribed fields below span them.
CORE_PECLET = (8.0, 250.0)


@dataclass(frozen=True)
class ScalarPhysics:
    """The two particle properties the transport solver reads, fixed by a case.

    Parameters
    ----------
    settling : float
        Settling velocity in m/s, positive downward, for every class.
    diffusion : float
        Diffusion coefficient in m^2/s for every class.
    n_classes : int
        Number of classes the case configures; one by default.
    """

    settling: float
    diffusion: float
    n_classes: int = 1

    def _check(self, size_class: int) -> None:
        if isinstance(size_class, bool) or not isinstance(size_class, int):
            raise TypeError(
                f"size_class must be an int, got {type(size_class).__name__}"
            )
        if not 0 <= size_class < self.n_classes:
            raise IndexError(
                f"size_class {size_class} out of range [0, {self.n_classes - 1}]"
            )

    def settling_velocity(self, size_class: int) -> float:
        """The fixed settling velocity.

        Parameters
        ----------
        size_class : int
            Index into the case's classes.

        Returns
        -------
        float
            ``settling``, m/s, positive downward.

        Raises
        ------
        TypeError
            If ``size_class`` is not an int (a bool is not one here).
        IndexError
            If ``size_class`` is outside the case's classes.
        """
        self._check(size_class)
        return self.settling

    def diffusion_coeff(self, size_class: int) -> float:
        """The fixed diffusion coefficient.

        Parameters
        ----------
        size_class : int
            Index into the case's classes.

        Returns
        -------
        float
            ``diffusion``, m^2/s.

        Raises
        ------
        TypeError
            If ``size_class`` is not an int (a bool is not one here).
        IndexError
            If ``size_class`` is outside the case's classes.
        """
        self._check(size_class)
        return self.diffusion


@dataclass(frozen=True)
class FixedConditions:
    """A ConcentrationBoundary stand-in that hands out one ConcentrationFaces.

    Parameters
    ----------
    faces : ConcentrationFaces
        The conditions every class receives.
    """

    faces: ConcentrationFaces

    def faces_for(self, size_class: int) -> ConcentrationFaces:
        """The fixed conditions, the same for every class.

        Parameters
        ----------
        size_class : int
            Index of the class; any non-negative int.

        Returns
        -------
        ConcentrationFaces
            ``faces``.

        Raises
        ------
        TypeError
            If ``size_class`` is not an int (a bool is not one here).
        IndexError
            If ``size_class`` is negative.
        """
        if isinstance(size_class, bool) or not isinstance(size_class, int):
            raise TypeError(
                f"size_class must be an int, got {type(size_class).__name__}"
            )
        if size_class < 0:
            raise IndexError(f"size_class {size_class} is negative")
        return self.faces


@dataclass(frozen=True)
class TransportCase:
    """One validation case: everything a test or a reproduction runs from.

    Parameters
    ----------
    name : str
        The case's name in ADR-011 H.
    config : SimConfig
        Carries the transport section the solver reads and the domain.
    mesh : Mesh
        The case's mesh.
    faces : FaceVelocities
        The face velocity field, set on the staggered faces.
    conditions : FixedConditions
        The scalar condition at every face, the same for every class.
    physics : ScalarPhysics
        The settling velocity and diffusion coefficient the case fixes.
    initial : np.ndarray
        The initial concentration field, [ny, nx], exact cell averages where
        the field is analytical.
    exact : np.ndarray or None
        The field the final state is scored against, [ny, nx], or None for a
        case scored on bounds or on the budget instead.
    t_end : float
        The simulated time the case runs to, seconds.
    """

    name: str
    config: SimConfig
    mesh: Mesh
    faces: FaceVelocities
    conditions: FixedConditions
    physics: ScalarPhysics
    initial: np.ndarray
    exact: np.ndarray | None
    t_end: float


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def transport_config(
    width: float,
    height: float,
    nx: int,
    ny: int,
    boundaries: dict | None = None,
    *,
    rho: float = AIR["density"],
    cfl_number: float = CFL_NUMBER,
    scheme: str = UMIST,
    max_diffusion_iter: int = 500,
    diffusion_tol: float = 1.0e-13,
    output_interval: int = SOLVER_BLOCK["output_interval"],
    obstacles: list[dict] | None = None,
    mesh: dict | None = None,
    turbulent_schmidt: float | None = None,
) -> SimConfig:
    """A validated configuration carrying a transport section.

    Parameters
    ----------
    width, height : float
        Domain size in metres.
    nx, ny : int
        Cell counts.
    boundaries : dict, optional
        Boundary segments in the file's form; an empty mapping is a closed
        box of default walls.
    rho : float
        Air density; 1.2 unless a case says otherwise.
    cfl_number : float
        The Courant number the case advects at.
    scheme : str
        "umist" or "upwind".
    max_diffusion_iter, diffusion_tol : int, float
        The implicit solve's cap and tolerance. The default tolerance is
        tight so the diffusion cases' budgets close to rounding.
    output_interval : int
        FieldHistory's recording interval for the case.
    obstacles : list[dict], optional
        Obstacles in the file's form.
    mesh : dict, optional
        The file's ``mesh`` section, for a stretched case; absent is uniform.
    turbulent_schmidt : float, optional
        Sc_t for a case that hands the solver an eddy viscosity field; None
        leaves the key out, as a laminar configuration does.

    Returns
    -------
    SimConfig
        Validated as a file would be.
    """
    transport = {
        "cfl_number": cfl_number,
        "advection_scheme": scheme,
        "max_diffusion_iter": max_diffusion_iter,
        "diffusion_tol": diffusion_tol,
    }
    if turbulent_schmidt is not None:
        transport["turbulent_schmidt"] = turbulent_schmidt
    raw = {
        "domain": {"width": width, "height": height, "nx": nx, "ny": ny},
        "mesh": mesh or {},
        "fluid": {**AIR, "density": rho},
        "particles": PARTICLES,
        "solver": {**SOLVER_BLOCK, "output_interval": output_interval},
        "transport": transport,
        "boundaries": boundaries or {},
        "obstacles": obstacles or [],
        "sensors": [{"name": "centre", "x": width / 2.0, "y": height / 2.0}],
        "thresholds": {str(PARTICLE_SIZE): 100.0},
    }
    return SimConfig.from_dict(raw)


def prescribed_eddy_viscosity(
    mesh: Mesh, speed: float, schmidt: float = TURBULENT_SCHMIDT
) -> np.ndarray:
    """A non-uniform eddy viscosity for the gate rows, scaled to the case.

    Parameters
    ----------
    mesh : Mesh
        A uniform mesh; ``mesh.dx`` is the length the Peclet number uses.
    speed : float
        The case's velocity scale in m/s.
    schmidt : float
        Sc_t the case runs at.

    Returns
    -------
    np.ndarray
        ``nu_t`` in m^2/s, [ny, nx].

    Notes
    -----
    ``nu_t = Sc_t U dx / Pe`` puts the cell Peclet number on the turbulent
    diffusivity ``D_t = nu_t / Sc_t`` at ``Pe``. The field runs from
    ``Pe = 250`` to ``Pe = 8`` (``CORE_PECLET``) log-linearly along the
    domain diagonal, ``s = (x / Lx + y / Ly) / 2``:
    ``nu_t = nu_lo (nu_hi / nu_lo)^s``. A band of cells, ``max(2, nx / 16)``
    columns wide from column ``nx / 3``, is zero over the full height, so the
    harmonic mean's zero branch is crossed on the faces on both sides of it.
    """
    pe_low, pe_high = CORE_PECLET
    nu_hi = schmidt * speed * mesh.dx / pe_low
    nu_lo = schmidt * speed * mesh.dx / pe_high
    s = 0.5 * (mesh.xc[None, :] / mesh.x[-1] + mesh.yc[:, None] / mesh.y[-1])
    field = nu_lo * (nu_hi / nu_lo) ** s
    first = mesh.xc.size // 3
    field[:, first : first + max(2, mesh.xc.size // 16)] = 0.0
    return np.ascontiguousarray(field)


def erf_values(z: np.ndarray) -> np.ndarray:
    """The error function over an array, since the project has no scipy dependency.

    Parameters
    ----------
    z : np.ndarray
        Any shape.

    Returns
    -------
    np.ndarray
        ``math.erf`` of every entry, in the same shape.
    """
    flat = np.asarray(z, dtype=np.float64).ravel()
    return np.array([math.erf(v) for v in flat]).reshape(np.shape(z))


def _axis_averages(faces: np.ndarray, centre: float, sigma: float) -> np.ndarray:
    """Exact cell averages of exp(-(s - centre)^2 / (2 sigma^2)) between faces."""
    z = (faces - centre) / (sigma * math.sqrt(2.0))
    integral = erf_values(z) * sigma * math.sqrt(math.pi / 2.0)
    return np.diff(integral) / np.diff(faces)


def gaussian_cell_averages(
    x_faces: np.ndarray,
    y_faces: np.ndarray,
    centre: tuple[float, float],
    sigma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Exact cell averages of a separable Gaussian, [ny, nx].

    Parameters
    ----------
    x_faces, y_faces : np.ndarray
        Face coordinates, ``mesh.x`` and ``mesh.y``.
    centre : tuple[float, float]
        (x, y) of the peak.
    sigma : float
        Standard deviation, the same on both axes.
    amplitude : float
        Peak value of the continuous Gaussian.

    Returns
    -------
    np.ndarray
        ``amplitude * gx[i] * gy[j]`` with each factor the error-function
        integral over the cell divided by its width, so the discrete content
        ``sum(C V)`` is the analytical ``2 pi sigma^2 amplitude`` to rounding
        when the tails lie inside the domain.
    """
    gx = _axis_averages(np.asarray(x_faces), centre[0], sigma)
    gy = _axis_averages(np.asarray(y_faces), centre[1], sigma)
    return amplitude * np.outer(gy, gx)


def gaussian_mass(sigma: float, amplitude: float = 1.0) -> float:
    """The integral of a two-dimensional Gaussian over the plane.

    Parameters
    ----------
    sigma : float
        Standard deviation on both axes.
    amplitude : float
        Peak value.

    Returns
    -------
    float
        ``2 pi sigma^2 amplitude``.
    """
    return 2.0 * math.pi * sigma**2 * amplitude


def heat_kernel_cell_averages(
    x_faces: np.ndarray,
    y_faces: np.ndarray,
    centre: tuple[float, float],
    sigma_0: float,
    diffusivity: float,
    t: float,
) -> np.ndarray:
    """Exact cell averages of a diffusing Gaussian at time t, [ny, nx].

    Parameters
    ----------
    x_faces, y_faces : np.ndarray
        Face coordinates.
    centre : tuple[float, float]
        The fixed centre.
    sigma_0 : float
        Standard deviation at t = 0, where the peak is one.
    diffusivity : float
        D in m^2/s.
    t : float
        Time in seconds.

    Returns
    -------
    np.ndarray
        The Gaussian of ``sigma^2 = sigma_0^2 + 2 D t`` and amplitude
        ``sigma_0^2 / sigma^2`` (Carslaw and Jaeger 1959, section 10.2),
        which keeps the content ``2 pi sigma_0^2`` for all t.
    """
    variance = sigma_0**2 + 2.0 * diffusivity * t
    return gaussian_cell_averages(
        x_faces, y_faces, centre, math.sqrt(variance), sigma_0**2 / variance
    )


def zero_conditions(mesh: Mesh) -> ConcentrationFaces:
    """Every face open and inert: inflow clean, no deposition, no settling.

    Parameters
    ----------
    mesh : Mesh
        Fixes the face shapes.

    Returns
    -------
    ConcentrationFaces
        Zeros, SURFACE_NONE and False everywhere, read-only. The condition
        VAL-004 row 2 hands the solver (ADR-011 H).
    """
    return conditions_with(mesh)


def conditions_with(mesh: Mesh, **arrays: np.ndarray) -> ConcentrationFaces:
    """zero_conditions with named arrays replaced.

    Parameters
    ----------
    mesh : Mesh
        Fixes the face shapes.
    **arrays : np.ndarray
        Any of the seven ConcentrationFaces fields, in its shape and dtype.

    Returns
    -------
    ConcentrationFaces
        Read-only copies.

    Raises
    ------
    ValueError
        On a name that is not a field or a shape that is not the face's.
    """
    us, vs = u_shape(mesh), v_shape(mesh)
    fields: dict[str, np.ndarray] = {
        "inflow_u": np.zeros(us),
        "inflow_v": np.zeros(vs),
        "deposition_u": np.zeros(us),
        "deposition_v": np.zeros(vs),
        "surface_u": np.zeros(us, dtype=np.int32),
        "surface_v": np.zeros(vs, dtype=np.int32),
        "settling_v": np.zeros(vs, dtype=bool),
    }
    for name, value in arrays.items():
        if name not in fields:
            raise ValueError(f"{name} is not a ConcentrationFaces field")
        given = np.array(value, dtype=fields[name].dtype, copy=True)
        if given.shape != fields[name].shape:
            raise ValueError(
                f"{name} must have shape {fields[name].shape}, got {given.shape}"
            )
        fields[name] = given
    for value in fields.values():
        value.flags.writeable = False
    return ConcentrationFaces(**fields)


def uniform_face_field(mesh: Mesh, u: float, v: float) -> FaceVelocities:
    """The same velocity on every face.

    Parameters
    ----------
    mesh : Mesh
        Fixes the face shapes.
    u, v : float
        The velocity components, m/s.

    Returns
    -------
    FaceVelocities
        ``u`` on every vertical face and ``v`` on every horizontal face.
    """
    return FaceVelocities.copy_of(np.full(u_shape(mesh), u), np.full(v_shape(mesh), v))


def rotation_face_field(
    mesh: Mesh, omega: float, centre: tuple[float, float]
) -> FaceVelocities:
    """Solid-body rotation set on the faces, ``u = -omega (y - y_c)``, ``v = omega (x - x_c)``.

    Parameters
    ----------
    mesh : Mesh
        Fixes the face positions and shapes.
    omega : float
        Angular velocity, rad/s, positive anticlockwise.
    centre : tuple[float, float]
        (x_c, y_c) of the axis.

    Returns
    -------
    FaceVelocities
        u varies only along y and v only along x, so the discrete divergence
        of every cell is zero exactly, not to rounding (ADR-011 H, row 2).
    """
    u = np.broadcast_to(-omega * (mesh.yc - centre[1])[:, None], u_shape(mesh))
    v = np.broadcast_to(omega * (mesh.xc - centre[0])[None, :], v_shape(mesh))
    return FaceVelocities.copy_of(u, v)


def smith_hutton_face_field(mesh: Mesh, speed: float) -> FaceVelocities:
    """The Smith and Hutton (1982) field on a 2 by 1 domain, scaled by ``speed``.

    Parameters
    ----------
    mesh : Mesh
        A 2.0 m by 1.0 m domain's mesh.
    speed : float
        U, m/s.

    Returns
    -------
    FaceVelocities
        The field below on the faces.

    Notes
    -----
    With ``x' = x - 1`` in [-1, 1] and ``y' = y`` in [0, 1]:
    ``u = 2 U y' (1 - x'^2)`` on the u faces and ``v = -2 U x' (1 - y'^2)``
    on the v faces. The normal velocity is zero exactly on the left, right
    and top edges; the bottom edge is inflow for x' < 0 and outflow for
    x' > 0. The x difference of u over a cell is ``-4 U x'_c y'_c dx`` and
    the y difference of v is its negative, so the discrete divergence is
    zero on uniform cells (ADR-011 H).
    """
    x_prime_faces = mesh.x - 1.0
    x_prime_centres = mesh.xc - 1.0
    u = 2.0 * speed * mesh.yc[:, None] * (1.0 - x_prime_faces[None, :] ** 2)
    v = -2.0 * speed * x_prime_centres[None, :] * (1.0 - mesh.y[:, None] ** 2)
    return FaceVelocities.copy_of(u, v)


def smith_hutton_inlet(x_prime: np.ndarray, alpha: float) -> np.ndarray:
    """The inlet profile ``1 + tanh(alpha (2 x' + 1))`` for x' in [-1, 0].

    Parameters
    ----------
    x_prime : np.ndarray
        Positions along the inlet, ``x - 1``.
    alpha : float
        The profile's steepness; 10 in ADR-011 H.

    Returns
    -------
    np.ndarray
        The profile at each position, from ``1 - tanh(alpha)`` to
        ``1 + tanh(alpha)``.
    """
    return 1.0 + np.tanh(alpha * (2.0 * np.asarray(x_prime) + 1.0))


def random_face_field(mesh: Mesh, seed: int, scale: float) -> FaceVelocities:
    """A seeded uniform random field on every face, not divergence-free.

    Parameters
    ----------
    mesh : Mesh
        Fixes the face shapes.
    seed : int
        The generator's seed, so the field is the same on every run.
    scale : float
        Every component is drawn from [-scale, scale), m/s.

    Returns
    -------
    FaceVelocities
        The drawn field.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(-scale, scale, size=u_shape(mesh))
    v = rng.uniform(-scale, scale, size=v_shape(mesh))
    return FaceVelocities.copy_of(u, v)


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------

# VAL-003: closed 2.0 m by 1.2 m box, 200x120 cells of 0.01 m, D = 1e-3, a
# Gaussian of five cells' standard deviation centred off every node, run until
# sigma has doubled.
DIFFUSION = {
    "width": 2.0,
    "height": 1.2,
    "nx": 200,
    "ny": 120,
    "diffusivity": 1.0e-3,
    "sigma_0": 0.05,
    "centre": (0.93, 0.61),
}


def diffusion_case() -> TransportCase:
    """VAL-003, pure diffusion (ADR-011 H; REQ-T07).

    Returns
    -------
    TransportCase
        A zero face field and inert conditions; ``exact`` is the heat kernel
        at ``t_end = 3 sigma_0^2 / (2 D)``, where sigma is twice sigma_0.
        The time step is the test's, from the diffusion number.
    """
    p = DIFFUSION
    config = transport_config(p["width"], p["height"], p["nx"], p["ny"])
    mesh = Mesh(config)
    t_end = 3.0 * p["sigma_0"] ** 2 / (2.0 * p["diffusivity"])
    return TransportCase(
        name="VAL-003 pure diffusion",
        config=config,
        mesh=mesh,
        faces=uniform_face_field(mesh, 0.0, 0.0),
        conditions=FixedConditions(zero_conditions(mesh)),
        physics=ScalarPhysics(settling=0.0, diffusion=p["diffusivity"]),
        initial=gaussian_cell_averages(mesh.x, mesh.y, p["centre"], p["sigma_0"]),
        exact=heat_kernel_cell_averages(
            mesh.x, mesh.y, p["centre"], p["sigma_0"], p["diffusivity"], t_end
        ),
        t_end=t_end,
    )


# VAL-004 row 1: 2.0 m by 1.2 m channel, 100x60 cells of 0.02 m, a uniform
# field at the supply speed 30 degrees above the x axis, a four-cell Gaussian
# carried 1.0 m. The start is 4.4 sigma from the floor and the end 4.4 sigma
# from the ceiling, so the tail the boundary cuts is below 1e-4 of the peak.
OBLIQUE_PULSE = {
    "width": 2.0,
    "height": 1.2,
    "nx": 100,
    "ny": 60,
    "angle_degrees": 30.0,
    "sigma": 0.08,
    "start": (0.31, 0.35),
    "travel": 1.0,
}


def oblique_pulse_case(scheme: str = UMIST) -> TransportCase:
    """VAL-004 row 1, the oblique channel pulse (ADR-011 H; REQ-T08, REQ-T12).

    Parameters
    ----------
    scheme : str
        "umist" (the gate) or "upwind" (the control).

    Returns
    -------
    TransportCase
        Every boundary face open with a clean inflow; ``exact`` is the
        initial pulse translated by the flow over ``t_end = travel / U``.
    """
    p = OBLIQUE_PULSE
    config = transport_config(p["width"], p["height"], p["nx"], p["ny"], scheme=scheme)
    mesh = Mesh(config)
    angle = math.radians(p["angle_degrees"])
    u, v = SUPPLY_SPEED * math.cos(angle), SUPPLY_SPEED * math.sin(angle)
    t_end = p["travel"] / SUPPLY_SPEED
    end = (p["start"][0] + u * t_end, p["start"][1] + v * t_end)
    return TransportCase(
        name="VAL-004 row 1, oblique channel pulse",
        config=config,
        mesh=mesh,
        faces=uniform_face_field(mesh, u, v),
        conditions=FixedConditions(zero_conditions(mesh)),
        physics=ScalarPhysics(settling=0.0, diffusion=0.0),
        initial=gaussian_cell_averages(mesh.x, mesh.y, p["start"], p["sigma"]),
        exact=gaussian_cell_averages(mesh.x, mesh.y, end, p["sigma"]),
        t_end=t_end,
    )


# VAL-004 row 2: a 1.28 m square, 64x64 cells of 0.02 m, solid-body rotation
# about the centre at 1 rad/s, a four-cell Gaussian 12 cells from the axis,
# one revolution.
ROTATING_PUFF = {
    "side": 1.28,
    "n": 64,
    "omega": 1.0,
    "sigma": 0.08,
    "radius_cells": 12,
    "output_interval": 40,
}


def rotating_puff_case(scheme: str = UMIST) -> TransportCase:
    """VAL-004 row 2, the rotating puff (ADR-011 H; REQ-T08, REQ-T12).

    Parameters
    ----------
    scheme : str
        "umist" (the gate) or "upwind" (the control).

    Returns
    -------
    TransportCase
        A ConcentrationFaces of zeros, as section H says; ``exact`` is the
        initial field, since one revolution returns the puff to its start.
        ``config.output_interval`` is the FieldHistory interval the test
        records the puff at.
    """
    p = ROTATING_PUFF
    config = transport_config(
        p["side"],
        p["side"],
        p["n"],
        p["n"],
        scheme=scheme,
        output_interval=p["output_interval"],
    )
    mesh = Mesh(config)
    centre = (p["side"] / 2.0, p["side"] / 2.0)
    dx = p["side"] / p["n"]
    puff_centre = (centre[0] + p["radius_cells"] * dx, centre[1])
    initial = gaussian_cell_averages(mesh.x, mesh.y, puff_centre, p["sigma"])
    return TransportCase(
        name="VAL-004 row 2, rotating puff",
        config=config,
        mesh=mesh,
        faces=rotation_face_field(mesh, p["omega"], centre),
        conditions=FixedConditions(zero_conditions(mesh)),
        physics=ScalarPhysics(settling=0.0, diffusion=0.0),
        initial=initial,
        exact=initial,
        t_end=2.0 * math.pi / p["omega"],
    )


# VAL-013: Smith and Hutton (1982) on 2.0 m by 1.0 m, 100x50 cells of 0.02 m.
SMITH_HUTTON = {
    "width": 2.0,
    "height": 1.0,
    "nx": 100,
    "ny": 50,
    "alpha": 10.0,
    "t_cap": 20.0,
}


def smith_hutton_case(turbulent_schmidt: float | None = None) -> TransportCase:
    """VAL-013, the bounded-advection case (ADR-011 H; REQ-T12).

    Parameters
    ----------
    turbulent_schmidt : float, optional
        Sc_t for the row that hands the solver an eddy viscosity field; None
        leaves it out of the configuration.

    Returns
    -------
    TransportCase
        The bottom faces with x' < 0 carry the tanh inlet profile at the
        face position; every wall face is inert (``deposition_surface:
        none`` in the configuration); the initial field is the inlet's
        minimum ``1 - tanh(alpha)``. ``exact`` is None: the case is scored on
        bounds at every step, and its outlet profile is reported against the
        inlet's mirror image unscored. ``t_end`` is the 20 s cap.
    """
    p = SMITH_HUTTON
    inert_wall = {"type": "wall", "deposition_surface": "none"}
    boundaries = {
        "left": {**inert_wall, "location": "left", "y_start": 0.0, "y_end": 1.0},
        "right": {**inert_wall, "location": "right", "y_start": 0.0, "y_end": 1.0},
        "top": {**inert_wall, "location": "top", "x_start": 0.0, "x_end": 2.0},
        "inlet": {
            "type": "velocity_inlet",
            "location": "bottom",
            "x_start": 0.0,
            "x_end": 1.0,
            "velocity": SUPPLY_SPEED,
        },
        "outlet": {
            "type": "pressure_outlet",
            "location": "bottom",
            "x_start": 1.0,
            "x_end": 2.0,
        },
    }
    config = transport_config(
        p["width"],
        p["height"],
        p["nx"],
        p["ny"],
        boundaries,
        turbulent_schmidt=turbulent_schmidt,
    )
    mesh = Mesh(config)
    x_prime = mesh.xc - 1.0
    inflow_v = np.zeros(v_shape(mesh))
    inlet = x_prime < 0.0
    inflow_v[0, inlet] = smith_hutton_inlet(x_prime[inlet], p["alpha"])
    initial = np.full((p["ny"], p["nx"]), 1.0 - math.tanh(p["alpha"]))
    return TransportCase(
        name="VAL-013 Smith-Hutton",
        config=config,
        mesh=mesh,
        faces=smith_hutton_face_field(mesh, SUPPLY_SPEED),
        conditions=FixedConditions(conditions_with(mesh, inflow_v=inflow_v)),
        physics=ScalarPhysics(settling=0.0, diffusion=0.0),
        initial=initial,
        exact=None,
        t_end=p["t_cap"],
    )


# VAL-014: a sealed 8.0 m by 3.0 m room on 40x30 cells of 0.2 m by 0.1 m, the
# 5 um class settling onto the floor at cfl_number 0.4.
SEALED_BOX = {
    "width": 8.0,
    "height": 3.0,
    "nx": 40,
    "ny": 30,
    "cfl_number": 0.4,
    "c_0": 1.0e6,
}


def sealed_box_case(
    settling: float | None = None,
    settle_floor_face: bool = False,
) -> TransportCase:
    """VAL-014, the sealed box (ADR-011 D and H).

    Parameters
    ----------
    settling : float, optional
        The class's settling velocity, m/s. None takes the Stokes velocity of
        the case's 5 um class from ParticlePhysics on its configuration
        (REQ-T03), 7.78e-4 m/s.
    settle_floor_face : bool
        False is the case. True plants section D's trap as data: the
        settling increment marked on the floor face as well, the double
        count test 30 B1 found.

    Returns
    -------
    TransportCase
        A zero face field; the settling increment on every interior
        horizontal face; the floor's ``deposition_v`` equal to ``settling``
        booked as floor; nothing else deposits.
        ``t_end`` is the time the front takes to reach the floor, H / v_s;
        the test runs a fraction of it.
    """
    p = SEALED_BOX
    config = transport_config(
        p["width"], p["height"], p["nx"], p["ny"], cfl_number=p["cfl_number"]
    )
    if settling is None:
        settling = ParticlePhysics(config).settling_velocity(0)
    mesh = Mesh(config)
    deposition_v = np.zeros(v_shape(mesh))
    deposition_v[0, :] = settling
    surface_v = np.zeros(v_shape(mesh), dtype=np.int32)
    surface_v[0, :] = SURFACE_FLOOR
    settling_v = np.zeros(v_shape(mesh), dtype=bool)
    settling_v[1:-1, :] = True
    settling_v[0, :] = settle_floor_face
    return TransportCase(
        name="VAL-014 sealed box",
        config=config,
        mesh=mesh,
        faces=uniform_face_field(mesh, 0.0, 0.0),
        conditions=FixedConditions(
            conditions_with(
                mesh,
                deposition_v=deposition_v,
                surface_v=surface_v,
                settling_v=settling_v,
            )
        ),
        physics=ScalarPhysics(settling=settling, diffusion=0.0),
        initial=np.full((p["ny"], p["nx"]), p["c_0"]),
        exact=None,
        t_end=p["height"] / settling,
    )
