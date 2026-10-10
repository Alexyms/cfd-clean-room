"""The k-epsilon model on a prescribed face velocity field (ADR-012 A, C; ECR-002 step 1).

Two quantities at the cell centres, beside the pressure: the turbulent kinetic
energy k (m^2/s^2) and its dissipation rate eps (m^2/s^3). The model is
written per unit density, which the incompressible solver holds constant, so
the eddy viscosity it returns is kinematic, ``nu_t = C_mu k^2 / eps`` in
m^2/s; REQ-S14's dynamic ``mu_t`` is ``rho nu_t``. Per unit density the
equations of ADR-012 A are

    dk/dt + div(u k)     = div((nu + nu_t / sigma_k) grad k)   + P - eps
    deps/dt + div(u eps) = div((nu + nu_t / sigma_e) grad eps) + C_1 (eps / k) P
                           - C_2 eps^2 / k - R
    P = nu_t S^2,  S^2 = 2 S_ij S_ij

with RNG's ``R = C_mu eta^3 (1 - eta / eta_0) / (1 + beta eta^3) eps^2 / k``,
``eta = S k / eps``, and R = 0 for the standard variant.

The step (ADR-012 C). One step of the transport scheme
(``src/scalar_scheme.py``) per quantity, in three parts:

1. advection with the UMIST-limited QUICK face value, forward Euler;
2. the explicit growth, ``dt P`` for k and ``dt (C_1 (eps / k) P - min(R, 0))``
   for eps, from the previous iterate;
3. one implicit solve per quantity for diffusion and decay together, the
   decay ``eps / k`` for k and ``C_2 eps / k + max(R, 0) / eps`` for eps,
   from the previous iterate, in the diagonal.

RNG's R splits by sign so that step 2 adds only non-negative terms and step
3's diagonal only non-negative ones: positive R sits in eps's diagonal,
negative R in its growth. The decay sits in the diagonal, never explicitly:
an explicit ``k - dt eps`` goes negative at a large step. Nothing is
clipped.

The two kinds of step. ``dt`` None is the local pseudo-time step
``dt_P = cfl / (max(|u_w|, |u_e|) / dx + max(|v_s|, |v_n|) / dy)``, capped
at the largest finite dt_P, which a cell at rest takes; pseudo-time is not
time accurate, and the steady state does not depend on it. On a field where
no cell moves no dt_P is finite and the step raises. A float ``dt`` is one
true-time step for every cell (VAL-015), refused above the advection's
stable step.

Gradients (ADR-012 C). du/dx and dv/dy at a cell centre are face differences.
du/dy and dv/dx are formed at the four corners of the cell and averaged to
its centre. At a corner on a domain edge the edge's own tangential velocity,
from the conditions, sits at the edge; a face with a SOLID cell on either
side lies on or inside an obstacle, where the velocity is zero, and a corner
reads that zero at the corner itself. So a no-slip wall gives the full shear
across the half cell beside it.

Boundaries (ADR-012 C) arrive as data, a TurbulenceConditions the caller
builds for every step. The inflow values are read where a domain face's flux
points into the room, as the transport solver reads a concentration. A
fixed_flow_outlet holds an outward velocity (ADR-012 D, amended 2026-10-08),
so its faces never read an inflow value. Faces beside a SOLID cell carry no
flux. Diffusion crosses only faces between two non-SOLID cells, so a wall,
an obstacle face and an outlet face carry nothing. In the cells the
conditions name, eps is held at the given value and the production is the
given one.

After every step k > 0 and eps > 0 at every non-SOLID cell, or the step
raises PositivityError (REQ-S15).

Wall functions (ADR-012 B, ECR-002 step 6). TurbulenceBoundary builds the
coupled solve's boundary data from the staggered boundary layer: the wall
viscosity of the momentum wall stencil (``wall_viscosity``, in
``MomentumPredictor.predict``'s ``wall_mu`` layout), the conditions of each
step (``conditions``: the inflow values, the edges' tangential velocities,
eps held and the production given in the wall cells) and the inlet values
the solve starts from. A wall is a face with zero normal velocity that is
not an outlet (``StaggeredBoundary.wall_faces``), at rest or moving, and
every obstacle face; an inlet that admits air and an outlet are not walls.
The constants are Launder and Spalding's, kappa 0.41 and E 9.793, and the
floor of the scalable form is computed from them (``log_law_floor``).
"""

import logging
import math
from dataclasses import dataclass

import numpy as np

from src.boundary_registry import VELOCITY_INLET, BoundaryRegistry
from src.boundary_staggered import StaggeredBoundary
from src.config import RNG, STANDARD, SimConfig
from src.mesh import SOLID, Mesh
from src.scalar_scheme import advective_flux, implicit_step, mesh_axes
from src.staggered import (
    FaceVelocities,
    edge_cell_inputs,
    p_shape,
    u_shape,
    v_shape,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantConstants:
    """The constants of one k-epsilon variant (ADR-012 A).

    Parameters
    ----------
    c_mu, c_1, c_2 : float
        The eddy-viscosity, production and destruction constants.
    sigma_k, sigma_e : float
        The turbulent Prandtl numbers of k and eps.
    eta_0, beta : float or None
        RNG's strain constants; None for a variant without the R term.
    """

    c_mu: float
    c_1: float
    c_2: float
    sigma_k: float
    sigma_e: float
    eta_0: float | None = None
    beta: float | None = None


# The published variants (ADR-012 A, decision 2). Module constants, not
# configuration: a configured C_mu would be a different model under the same
# name (ADR-012 I). Each value was checked against its source before the
# build; the record, with quotations and locations, is
# results/builder35/constants.md (prompt 35, item 0).
VARIANTS: dict[str, VariantConstants] = {
    # Launder and Spalding (1974), Table 2.1: 0.09, 1.44, 1.92, 1.0, 1.3.
    STANDARD: VariantConstants(c_mu=0.09, c_1=1.44, c_2=1.92, sigma_k=1.0, sigma_e=1.3),
    # Yakhot, Orszag, Thangam, Gatski and Speziale (1992), from its preprint,
    # ICASE Report 91-65 (the published paper was not reached): C_eps1 1.42,
    # C_eps2 1.68, alpha_K = alpha_eps = 1.39 (sigma = 1 / 1.39 = 0.7194),
    # eta_0 4.38, beta 0.012. C_mu is 0.0845, as OpenFOAM (which cites the
    # 1992 paper), PHOENICS and Fluent give it; the preprint prints
    # "C_mu ~ 0.085". Its own fixed-point relation, eta_0 = ((C_2 - 1) /
    # (C_mu (C_1 - 1)))^(1/2), gives the printed 4.38 at 0.0845 (4.377) and
    # not at 0.085 (4.364), so 0.0845 is the value the other constants imply
    # (Alex, 2026-10-05). Speziale and Thangam (ICASE 92-3) use another set,
    # C_mu 0.085, sigma 0.7179 and beta 0.015, which this is not.
    RNG: VariantConstants(
        c_mu=0.0845,
        c_1=1.42,
        c_2=1.68,
        sigma_k=0.7194,
        sigma_e=0.7194,
        eta_0=4.38,
        beta=0.012,
    ),
}


# The law of the wall, u+ = ln(E y+) / kappa (ADR-012 B): Launder and Spalding
# (1974). Model constants, as the variants' are, not configuration.
KAPPA = 0.41
E_WALL = 9.793


def log_law_floor(kappa: float, e: float) -> float:
    """The scalable form's floor y*_0, where u+ = y+ meets u+ = ln(E y+) / kappa.

    Parameters
    ----------
    kappa : float
        The von Karman constant, positive.
    e : float
        The log law's constant E, with ``ln(E / kappa) > 1`` so that the two
        laws meet.

    Returns
    -------
    float
        The root of ``kappa y = ln(E y)`` above ``1 / kappa``, by Newton's
        method from ``2 / kappa``. ``kappa y - ln(E y)`` is convex with its
        least value at ``1 / kappa``, so it has one root on each side; the
        lower lies below y+ of 1 and means nothing here. 11.53 for kappa
        0.41 and E 9.793 (ADR-012 B).

    Raises
    ------
    ValueError
        If the two laws do not meet, or Newton's method does not settle.
    """
    if not math.log(e / kappa) > 1.0:
        raise ValueError(f"u+ = y+ and u+ = ln({e} y+) / {kappa} do not meet")
    y = 2.0 / kappa
    for _ in range(100):
        step = (kappa * y - math.log(e * y)) / (kappa - 1.0 / y)
        y -= step
        if abs(step) <= 4.0 * math.ulp(y):
            return y
    raise ValueError("Newton's method did not settle on the log-law floor")


Y_STAR_FLOOR = log_law_floor(KAPPA, E_WALL)


def wall_viscosity(
    k_p: np.ndarray, y_p: np.ndarray, c_mu: float, rho: float, nu: float
) -> np.ndarray:
    """The scalable wall function as a viscosity on the wall face (ADR-012 B).

    Parameters
    ----------
    k_p : np.ndarray
        k at the first node, m^2/s^2, positive.
    y_p : np.ndarray
        The node's distance to the wall, m, positive; same shape.
    c_mu : float
        The variant's C_mu.
    rho, nu : float
        Density, kg/m^3, and molecular kinematic viscosity, m^2/s.

    Returns
    -------
    np.ndarray
        ``mu_w = rho C_mu^(1/4) k_p^(1/2) kappa y_p / ln(E max(y*, y*_0))``,
        Pa s, with ``y* = C_mu^(1/4) k_p^(1/2) y_p / nu``.

    Notes
    -----
    The wall stencil ``mu_w (u_P - u_wall) / y_p`` then carries the log
    law's shear ``tau_w = rho C_mu^(1/4) k_p^(1/2) kappa (u_P - u_wall) /
    ln(E y*)``, with k, not u_tau, as the velocity scale, so it stays finite
    at a stagnation point. At ``y* = y*_0``, ``kappa y*_0 = ln(E y*_0)``
    makes ``mu_w = rho nu = mu`` exactly. Above the floor ``mu_w > mu`` and
    grows with y*. Below it the log law is still evaluated at y*_0 (the
    scalable form, Grotjans and Menter 1998), so ``mu_w = mu y* / y*_0``,
    less than mu: a node inside the viscous sublayer is treated as at its
    edge, and the wall shear goes to zero with k, as at a stagnation point.
    """
    u_k = c_mu**0.25 * np.sqrt(k_p)
    y_star = u_k * y_p / nu
    return rho * u_k * KAPPA * y_p / np.log(E_WALL * np.maximum(y_star, Y_STAR_FLOOR))


def inflow_values(
    intensity: float, normal_speed: float, dissipation_length: float
) -> tuple[float, float]:
    """k and eps carried in by an inlet face (ADR-012 C).

    Parameters
    ----------
    intensity : float
        The segment's ``turbulence_intensity`` I, a fraction.
    normal_speed : float
        The face's normal velocity |u_n|, m/s.
    dissipation_length : float
        The segment's ``dissipation_length`` l_e, m.

    Returns
    -------
    tuple[float, float]
        ``k = 1.5 (I |u_n|)^2`` and ``eps = k^(3/2) / l_e``: the
        specification's convention, with no C_mu^(3/4) (ADR-012 C). A
        source writing ``eps = C_mu^(3/4) k^(3/2) / l`` maps to
        ``l_e = l / C_mu^(3/4)``.
    """
    k = 1.5 * (intensity * abs(normal_speed)) ** 2
    return k, k**1.5 / dissipation_length


class PositivityError(RuntimeError):
    """k or eps is not positive at a non-SOLID cell after a step (REQ-S15).

    Parameters
    ----------
    message : str
        Names the quantity, the count of cells and the first cell.
    minimum : float
        The least value of the offending quantity over the non-SOLID cells,
        NaN when one is not a number.
    """

    def __init__(self, message: str, minimum: float) -> None:
        super().__init__(message)
        self.minimum = minimum


@dataclass(frozen=True, eq=False)
class TurbulenceState:
    """k, eps and the eddy viscosity on the cell centres.

    Build one with ``KEpsilonModel.state`` or ``initial``, which check it.

    Parameters
    ----------
    k : np.ndarray
        Turbulent kinetic energy, m^2/s^2, [ny, nx], float64, read-only;
        positive in non-SOLID cells, zero in SOLID ones.
    eps : np.ndarray
        Its dissipation rate, m^2/s^3, likewise.
    nu_t : np.ndarray
        The kinematic eddy viscosity, m^2/s, [ny, nx], float64, read-only;
        non-negative and finite in non-SOLID cells, zero in SOLID ones.
        ``state`` builds it as ``C_mu k^2 / eps``; a caller may build a
        state with another such field, as step 6's under-relaxed one.
    """

    k: np.ndarray
    eps: np.ndarray
    nu_t: np.ndarray


@dataclass(frozen=True, eq=False)
class TurbulenceConditions:
    """The boundary values of k and eps for one step, as data the caller builds.

    Parameters
    ----------
    inflow_k_u, inflow_eps_u : np.ndarray
        k and eps carried in through the vertical domain faces, [ny, nx+1],
        float64; only columns 0 and nx are read, where the flux points into
        the room. Positive and finite on those columns.
    inflow_k_v, inflow_eps_v : np.ndarray
        The same on the horizontal domain faces, [ny+1, nx], rows 0 and ny.
    tangential_bottom, tangential_top : np.ndarray
        The edge's own tangential velocity u at the corners along the bottom
        and top edges, [nx+1], m/s: a wall's velocity (zero at rest), an
        inlet's tangential velocity. Step 6 fills them from the staggered
        boundary layer, which owns them.
    tangential_left, tangential_right : np.ndarray
        The tangential velocity v at the corners along the left and right
        edges, [ny+1].
    eps_held : np.ndarray
        Bool [ny, nx], the non-SOLID cells whose eps is held (wall cells).
    eps_wall : np.ndarray
        Float [ny, nx], the held eps, positive where ``eps_held``; read
        nowhere else.
    production_given : np.ndarray
        Bool [ny, nx], the non-SOLID cells whose production is given.
    production : np.ndarray
        Float [ny, nx], the given production per unit density, m^2/s^3,
        non-negative where ``production_given``; read nowhere else.
    """

    inflow_k_u: np.ndarray
    inflow_k_v: np.ndarray
    inflow_eps_u: np.ndarray
    inflow_eps_v: np.ndarray
    tangential_bottom: np.ndarray
    tangential_top: np.ndarray
    tangential_left: np.ndarray
    tangential_right: np.ndarray
    eps_held: np.ndarray
    eps_wall: np.ndarray
    production_given: np.ndarray
    production: np.ndarray

    @classmethod
    def uniform(cls, mesh: Mesh, k: float, eps: float) -> "TurbulenceConditions":
        """Conditions with one inflow k and eps everywhere, edges at rest, nothing held.

        Parameters
        ----------
        mesh : Mesh
            Fixes the shapes.
        k, eps : float
            The inflow values on every domain face.

        Returns
        -------
        TurbulenceConditions
            Stationary edges, no held eps and no given production.
        """
        ny, nx = p_shape(mesh)
        return cls(
            inflow_k_u=np.full(u_shape(mesh), float(k)),
            inflow_k_v=np.full(v_shape(mesh), float(k)),
            inflow_eps_u=np.full(u_shape(mesh), float(eps)),
            inflow_eps_v=np.full(v_shape(mesh), float(eps)),
            tangential_bottom=np.zeros(nx + 1),
            tangential_top=np.zeros(nx + 1),
            tangential_left=np.zeros(ny + 1),
            tangential_right=np.zeros(ny + 1),
            eps_held=np.zeros((ny, nx), dtype=bool),
            eps_wall=np.zeros((ny, nx)),
            production_given=np.zeros((ny, nx), dtype=bool),
            production=np.zeros((ny, nx)),
        )


@dataclass(frozen=True, eq=False)
class StepTerms:
    """Step 2's growth and step 3's decay per cell, from one iterate (ADR-012 C).

    All per unit density, zero in SOLID cells.

    Parameters
    ----------
    production : np.ndarray
        ``P = nu_t S^2``, m^2/s^3, or the given value where the conditions
        give one.
    rng_r : np.ndarray
        RNG's R, m^2/s^4, signed; zero for the standard variant.
    growth_k : np.ndarray
        Added to k times dt, m^2/s^3: ``P``.
    growth_eps : np.ndarray
        Added to eps times dt, m^2/s^4: ``C_1 (eps / k) P - min(R, 0)``.
    decay_k : np.ndarray
        k's diagonal coefficient, 1/s: ``eps / k``.
    decay_eps : np.ndarray
        eps's diagonal coefficient, 1/s: ``C_2 eps / k + max(R, 0) / eps``.
    """

    production: np.ndarray
    rng_r: np.ndarray
    growth_k: np.ndarray
    growth_eps: np.ndarray
    decay_k: np.ndarray
    decay_eps: np.ndarray


class KEpsilonModel:
    """Advance k and eps one step on a prescribed face velocity field.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh, uniform or stretched.
    config : SimConfig
        Must carry a ``turbulence`` section: variant, cfl_number, max_iter,
        tol. Supplies the density and molecular viscosity.

    Attributes
    ----------
    constants : VariantConstants
        The configured variant's constants.
    last_sweeps : tuple[int, int]
        Jacobi sweeps of the last step's k and eps solves.
    solves_converged : bool
        Whether both of the last step's solves met ``tol`` within the cap.

    Raises
    ------
    ValueError
        Without a turbulence section.
    """

    def __init__(self, mesh: Mesh, config: SimConfig) -> None:
        if config.turbulence is None:
            raise ValueError(
                "the k-epsilon model needs a turbulence section in the configuration"
            )
        spec = config.turbulence
        self.constants: VariantConstants = VARIANTS[spec.variant]
        self._mesh = mesh
        self._cfl = spec.cfl_number
        self._max_iter = spec.max_iter
        self._tol = spec.tol
        self._nu = config.mu / config.rho
        self._p_shape = p_shape(mesh)
        self._u_shape = u_shape(mesh)
        self._v_shape = v_shape(mesh)

        solid = mesh.cell_type == SOLID
        self._solid = solid
        self._live = ~solid
        self._volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        # Faces with an advective flux: both cells non-SOLID, or one
        # non-SOLID edge cell at a domain face (the transport solver's rule).
        live_u = np.ones(self._u_shape, dtype=bool)
        live_u[:, 0] = ~solid[:, 0]
        live_u[:, -1] = ~solid[:, -1]
        live_u[:, 1:-1] = ~solid[:, :-1] & ~solid[:, 1:]
        live_v = np.ones(self._v_shape, dtype=bool)
        live_v[0, :] = ~solid[0, :]
        live_v[-1, :] = ~solid[-1, :]
        live_v[1:-1, :] = ~solid[:-1, :] & ~solid[1:, :]
        self._live_u, self._live_v = live_u, live_v
        # Diffusion crosses interior faces between two non-SOLID cells only.
        self._inner_u = np.zeros(self._u_shape, dtype=bool)
        self._inner_u[:, 1:-1] = live_u[:, 1:-1]
        self._inner_v = np.zeros(self._v_shape, dtype=bool)
        self._inner_v[1:-1, :] = live_v[1:-1, :]
        self._area_u = np.broadcast_to(mesh.dy_cell[:, None], self._u_shape)
        self._area_v = np.broadcast_to(mesh.dx_cell[None, :], self._v_shape)
        # The far cell's share of each interior face's centre-to-centre
        # distance, for the harmonic mean of the two cells' diffusivities.
        self._w_east = (mesh.xc[1:] - mesh.x[1:-1]) / mesh.dx_face[1:-1]
        self._w_north = (mesh.yc[1:] - mesh.y[1:-1]) / mesh.dy_face[1:-1]
        self._axis_x, self._axis_y = mesh_axes(mesh)
        self.last_sweeps: tuple[int, int] = (0, 0)
        self.solves_converged: bool = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initial(self, k: float, eps: float) -> TurbulenceState:
        """The uniform state at k and eps, the inlet values in the coupled solve.

        Parameters
        ----------
        k : float
            m^2/s^2, positive and finite.
        eps : float
            m^2/s^3, positive and finite.

        Returns
        -------
        TurbulenceState
            k and eps uniform over the non-SOLID cells, zero in SOLID ones.

        Raises
        ------
        TypeError
            If either is a bool or not a number.
        ValueError
            If either is not positive and finite.
        """
        for name, value in (("k", k), ("eps", eps)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float)
            ):
                raise TypeError(f"{name} must be a number, got {type(value).__name__}")
        return self.state(np.full(self._p_shape, k), np.full(self._p_shape, eps))

    def state(self, k: np.ndarray, eps: np.ndarray) -> TurbulenceState:
        """A checked state from k and eps, with its eddy viscosity.

        Parameters
        ----------
        k, eps : np.ndarray
            [ny, nx]; positive and finite in every non-SOLID cell. SOLID
            cells are set to zero.

        Returns
        -------
        TurbulenceState
            Read-only copies and ``nu_t = C_mu k^2 / eps``.

        Raises
        ------
        ValueError
            If a shape does not fit the mesh, or k or eps is not positive and
            finite in a non-SOLID cell.
        """
        k = self._cell_field(k, "k")
        eps = self._cell_field(eps, "eps")
        nu_t = self.eddy_viscosity(k, eps)
        for array in (k, eps, nu_t):
            array.setflags(write=False)
        return TurbulenceState(k=k, eps=eps, nu_t=nu_t)

    def eddy_viscosity(self, k: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """The kinematic eddy viscosity ``C_mu k^2 / eps``, m^2/s.

        Parameters
        ----------
        k, eps : np.ndarray
            [ny, nx], positive in every non-SOLID cell.

        Returns
        -------
        np.ndarray
            [ny, nx], zero in SOLID cells. The dynamic ``mu_t`` of REQ-S14 is
            ``rho`` times it.
        """
        nu_t = np.zeros(self._p_shape)
        live = self._live
        nu_t[live] = self.constants.c_mu * k[live] ** 2 / eps[live]
        return nu_t

    def strain_squared(
        self, faces: FaceVelocities, conditions: TurbulenceConditions
    ) -> np.ndarray:
        """``S^2 = 2 S_ij S_ij`` at the cell centres, 1/s^2 (ADR-012 C).

        Parameters
        ----------
        faces : FaceVelocities
            The prescribed face velocities.
        conditions : TurbulenceConditions
            Supplies the edges' tangential velocities.

        Returns
        -------
        np.ndarray
            ``2 (du/dx)^2 + 2 (dv/dy)^2 + (du/dy + dv/dx)^2``, [ny, nx], zero
            in SOLID cells. du/dx and dv/dy are face differences; du/dy and
            dv/dx are formed at the four corners and averaged to the centre,
            with the edge's tangential velocity at a domain edge and zero at
            an obstacle face.
        """
        self._check_faces(faces)
        self._check_conditions(conditions)
        return self._strain_squared(faces, conditions)

    def terms(
        self,
        state: TurbulenceState,
        faces: FaceVelocities,
        conditions: TurbulenceConditions,
    ) -> StepTerms:
        """The growth and decay a step applies, from the iterate ``state``.

        Parameters
        ----------
        state : TurbulenceState
            The previous iterate.
        faces : FaceVelocities
            The prescribed face velocities.
        conditions : TurbulenceConditions
            Supplies the edges' tangential velocities and the given
            production.

        Returns
        -------
        StepTerms
            Per cell, per unit density, zero in SOLID cells.
        """
        self._check_state(state)
        self._check_faces(faces)
        self._check_conditions(conditions)
        return self._terms(state, faces, conditions)

    def pseudo_time_step(self, faces: FaceVelocities) -> np.ndarray:
        """The local pseudo-time step per cell, s (ADR-012 C).

        Parameters
        ----------
        faces : FaceVelocities
            The prescribed face velocities.

        Returns
        -------
        np.ndarray
            ``cfl / (max(|u_w|, |u_e|) / dx + max(|v_s|, |v_n|) / dy)`` per
            cell over the faces that carry a flux, [ny, nx]; a cell at rest,
            and every SOLID cell, takes the largest finite value, so that
            the explicit growth of a cell at rest stays bounded.

        Raises
        ------
        ValueError
            When no non-SOLID cell moves: no dt_P is finite, so the step has
            no value. Pass a true-time step instead.
        """
        self._check_faces(faces)
        rate = self._rate(faces)
        moving = self._live & (rate > 0.0)
        if not moving.any():
            raise ValueError(
                "no cell moves, so the pseudo-time step has no value; "
                "pass a true-time dt"
            )
        dt_cell = np.zeros(self._p_shape)
        dt_cell[moving] = self._cfl / rate[moving]
        dt_cell[~moving] = dt_cell[moving].max()
        return dt_cell

    def step(
        self,
        state: TurbulenceState,
        faces: FaceVelocities,
        conditions: TurbulenceConditions,
        dt: float | None = None,
    ) -> TurbulenceState:
        """Advance k and eps one step (ADR-012 C).

        Parameters
        ----------
        state : TurbulenceState
            The previous iterate.
        faces : FaceVelocities
            The prescribed face velocities.
        conditions : TurbulenceConditions
            This step's boundary values, checked against the mesh.
        dt : float, optional
            None takes the local pseudo-time step; a float is one true-time
            step, seconds, for every cell, at most the advection's stable
            step ``cfl_number`` over the largest cell rate.

        Returns
        -------
        TurbulenceState
            The new iterate, k > 0 and eps > 0 at every non-SOLID cell.

        Raises
        ------
        TypeError
            If ``dt`` is a bool or not a number.
        ValueError
            If ``dt`` is not finite and positive or exceeds the stable step,
            ``dt`` is None on a field where no cell moves, a shape or dtype
            does not fit the mesh, an input is not finite, or a condition is
            out of range. Every check runs before any arithmetic.
        PositivityError
            If k or eps is not positive and finite at a non-SOLID cell after
            the step.
        """
        self._check_state(state)
        self._check_faces(faces)
        self._check_conditions(conditions)
        dt_cell: float | np.ndarray
        if dt is None:
            dt_cell = self.pseudo_time_step(faces)
        else:
            dt_cell = self._true_time_step(faces, dt)
        terms = self._terms(state, faces, conditions)
        c = conditions
        u_adv = np.where(self._live_u, faces.u, 0.0)
        v_adv = np.where(self._live_v, faces.v, 0.0)
        flux_u = u_adv * self._area_u
        flux_v = v_adv * self._area_v

        # Steps 1 and 2: advection, then the explicit growth.
        k_star = self._advect(
            state.k, flux_u, flux_v, c.inflow_k_u, c.inflow_k_v, dt_cell
        )
        eps_star = self._advect(
            state.eps, flux_u, flux_v, c.inflow_eps_u, c.inflow_eps_v, dt_cell
        )
        k_star = np.where(self._solid, 0.0, k_star + dt_cell * terms.growth_k)
        eps_star = np.where(self._solid, 0.0, eps_star + dt_cell * terms.growth_eps)
        eps_star = np.where(c.eps_held, c.eps_wall, eps_star)

        # Step 3: diffusion and decay, implicit, eps held where given.
        const = self.constants
        g_u_k, g_v_k = self._conductances(state.nu_t, const.sigma_k)
        g_u_e, g_v_e = self._conductances(state.nu_t, const.sigma_e)
        k_result = implicit_step(
            k_star,
            self._volume,
            dt_cell,
            g_u_k,
            g_v_k,
            self._volume * terms.decay_k,
            self._solid,
            self._tol,
            self._max_iter,
        )
        eps_result = implicit_step(
            eps_star,
            self._volume,
            dt_cell,
            g_u_e,
            g_v_e,
            self._volume * terms.decay_eps,
            self._solid,
            self._tol,
            self._max_iter,
            held=c.eps_held,
        )
        self.last_sweeps = (k_result.sweeps, eps_result.sweeps)
        self.solves_converged = k_result.converged and eps_result.converged
        if not self.solves_converged:
            logger.warning(
                "implicit k or eps solve stopped at max_iter %d", self._max_iter
            )
        self._check_positive(k_result.field, "k")
        self._check_positive(eps_result.field, "eps")
        return self.state(k_result.field, eps_result.field)

    # ------------------------------------------------------------------
    # The step's parts
    # ------------------------------------------------------------------

    def _strain_squared(
        self, faces: FaceVelocities, c: TurbulenceConditions
    ) -> np.ndarray:
        mesh = self._mesh
        u, v = faces.u, faces.v
        # A face beside a SOLID cell carries no air, here as in the rate and
        # the advection: its stored velocity is read as zero.
        u_live = np.where(self._live_u, u, 0.0)
        v_live = np.where(self._live_v, v, 0.0)
        dudx = (u_live[:, 1:] - u_live[:, :-1]) / mesh.dx_cell[None, :]
        dvdy = (v_live[1:, :] - v_live[:-1, :]) / mesh.dy_cell[:, None]
        # du/dy at the corners [ny+1, nx+1]: along each u column the edge
        # values sit at the edges and the faces at the cell centres. A face
        # beside a SOLID cell is on or in an obstacle, zero at the corner.
        dudy = _corner_difference(
            np.vstack([c.tangential_bottom, u, c.tangential_top]),
            np.concatenate(([mesh.y[0]], mesh.yc, [mesh.y[-1]])),
            mesh.y,
            np.pad(~self._live_u, ((1, 1), (0, 0))),
        )
        # dv/dx likewise along each v row, worked in the transposed frame.
        dvdx = _corner_difference(
            np.vstack([c.tangential_left, v.T, c.tangential_right]),
            np.concatenate(([mesh.x[0]], mesh.xc, [mesh.x[-1]])),
            mesh.x,
            np.pad((~self._live_v).T, ((1, 1), (0, 0))),
        ).T
        shear = dudy + dvdx
        shear_centre = 0.25 * (
            shear[:-1, :-1] + shear[:-1, 1:] + shear[1:, :-1] + shear[1:, 1:]
        )
        s2 = 2.0 * dudx**2 + 2.0 * dvdy**2 + shear_centre**2
        return np.where(self._solid, 0.0, s2)

    def _terms(
        self,
        state: TurbulenceState,
        faces: FaceVelocities,
        c: TurbulenceConditions,
    ) -> StepTerms:
        const = self.constants
        live = self._live
        s2 = self._strain_squared(faces, c)
        production = np.where(c.production_given, c.production, state.nu_t * s2)
        production = np.where(live, production, 0.0)
        ratio = np.zeros(self._p_shape)
        ratio[live] = state.eps[live] / state.k[live]
        rng_r = np.zeros(self._p_shape)
        if const.eta_0 is not None and const.beta is not None:
            eta = np.zeros(self._p_shape)
            eta[live] = np.sqrt(s2[live]) * state.k[live] / state.eps[live]
            coefficient = (
                const.c_mu
                * eta**3
                * (1.0 - eta / const.eta_0)
                / (1.0 + const.beta * eta**3)
            )
            rng_r = np.where(live, coefficient * state.eps * ratio, 0.0)
        # R's sign split (ADR-012 C): negative R adds to eps's growth, positive
        # R sits in its diagonal, so neither step changes sign.
        growth_eps = const.c_1 * ratio * production - np.minimum(rng_r, 0.0)
        decay_eps = const.c_2 * ratio
        decay_eps[live] += np.maximum(rng_r[live], 0.0) / state.eps[live]
        return StepTerms(
            production=production,
            rng_r=rng_r,
            growth_k=production,
            growth_eps=growth_eps,
            decay_k=ratio,
            decay_eps=decay_eps,
        )

    def _rate(self, faces: FaceVelocities) -> np.ndarray:
        """``max(|u_w|, |u_e|) / dx + max(|v_s|, |v_n|) / dy`` over the faces with a flux."""
        mesh = self._mesh
        u = np.abs(np.where(self._live_u, faces.u, 0.0))
        v = np.abs(np.where(self._live_v, faces.v, 0.0))
        return (
            np.maximum(u[:, :-1], u[:, 1:]) / mesh.dx_cell[None, :]
            + np.maximum(v[:-1, :], v[1:, :]) / mesh.dy_cell[:, None]
        )

    def _true_time_step(self, faces: FaceVelocities, dt: float) -> float:
        """Check a true-time step against the advection's stable step."""
        if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, (int, float)):
            raise TypeError(f"dt must be a number, got {type(dt).__name__}")
        if not math.isfinite(dt):
            raise ValueError(f"dt must be finite, got {dt}")
        if not dt > 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        rate = self._rate(faces)
        worst = float(rate[self._live].max()) if self._live.any() else 0.0
        if worst > 0.0 and not dt <= self._cfl / worst:
            raise ValueError(
                f"dt {dt} exceeds the stable step {self._cfl / worst}; the explicit "
                "advection is a convex combination only below it"
            )
        return float(dt)

    def _advect(
        self,
        q: np.ndarray,
        flux_u: np.ndarray,
        flux_v: np.ndarray,
        inflow_u: np.ndarray,
        inflow_v: np.ndarray,
        dt_cell: float | np.ndarray,
    ) -> np.ndarray:
        """Step 1: forward Euler with the limited face value, the step per cell."""
        adv_u = advective_flux(q, flux_u, inflow_u, self._axis_x, False)
        adv_v = advective_flux(q.T, flux_v.T, inflow_v.T, self._axis_y, False).T
        divergence = adv_u[:, 1:] - adv_u[:, :-1] + adv_v[1:, :] - adv_v[:-1, :]
        return q - dt_cell * divergence / self._volume

    def _conductances(
        self, nu_t: np.ndarray, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(nu + nu_t / sigma)_f A_f / d_f`` on the faces between two non-SOLID cells.

        The face value is the distance-weighted harmonic mean of the two
        cells' diffusivities, ADR-012 F's rule for the transport diffusivity
        (Patankar's interface conductivity: the flux crosses the two half
        cells in series), so k, eps and the particles use one rule. It is
        written so that equal cell values return that value exactly.
        """
        mesh = self._mesh
        gamma = np.where(self._solid, 1.0, self._nu + nu_t / sigma)
        west, east = gamma[:, :-1], gamma[:, 1:]
        face_u = west / (1.0 + self._w_east[None, :] * (west / east - 1.0))
        south, north = gamma[:-1, :], gamma[1:, :]
        face_v = south / (1.0 + self._w_north[:, None] * (south / north - 1.0))
        g_u = np.zeros(self._u_shape)
        g_v = np.zeros(self._v_shape)
        g_u[:, 1:-1] = face_u * self._area_u[:, 1:-1] / mesh.dx_face[None, 1:-1]
        g_v[1:-1, :] = face_v * self._area_v[1:-1, :] / mesh.dy_face[1:-1, None]
        return (
            np.where(self._inner_u, g_u, 0.0),
            np.where(self._inner_v, g_v, 0.0),
        )

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _check_positive(self, q: np.ndarray, name: str) -> None:
        """Raise PositivityError unless q is positive and finite at every non-SOLID cell.

        REQ-S15's assertion. An infinite value passes ``q > 0`` and is refused
        too: the eddy viscosity is not defined there either.
        """
        bad = self._live & ~(np.isfinite(q) & (q > 0.0))
        if bad.any():
            j, i = np.argwhere(bad)[0]
            minimum = float(np.min(q[self._live]))
            raise PositivityError(
                f"{name} is not positive and finite at {int(bad.sum())} non-SOLID cell(s), "
                f"first at row {j}, column {i} ({q[j, i]!r}); least {minimum!r}",
                minimum,
            )

    def _cell_field(self, q: np.ndarray, name: str) -> np.ndarray:
        """A float64 copy of a cell field, zero in SOLID cells, positive elsewhere."""
        out = np.array(q, dtype=np.float64, order="C", copy=True)
        if out.shape != self._p_shape:
            raise ValueError(
                f"expected {name} of shape {self._p_shape}, got {out.shape}"
            )
        live = out[self._live]
        if not (np.isfinite(live).all() and (live > 0.0).all()):
            raise ValueError(
                f"{name} must be positive and finite in every non-SOLID cell"
            )
        out[self._solid] = 0.0
        return out

    def _check_state(self, state: TurbulenceState) -> None:
        """The state's contract, before any arithmetic.

        Shapes; k and eps positive and finite in every non-SOLID cell; nu_t
        non-negative and finite there, so every face conductance is
        non-negative (ADR-012 C's M-matrix); all three zero in SOLID cells.
        """
        for name in ("k", "eps", "nu_t"):
            array = getattr(state, name)
            if array.shape != self._p_shape:
                raise ValueError(
                    f"state.{name} must have shape {self._p_shape}, got {array.shape}"
                )
        for name in ("k", "eps"):
            live = getattr(state, name)[self._live]
            if not (np.isfinite(live).all() and (live > 0.0).all()):
                raise ValueError(
                    f"state.{name} must be positive and finite in every non-SOLID cell"
                )
        live = state.nu_t[self._live]
        if not (np.isfinite(live).all() and (live >= 0.0).all()):
            raise ValueError(
                "state.nu_t must be non-negative and finite in every non-SOLID cell"
            )
        for name in ("k", "eps", "nu_t"):
            if np.any(getattr(state, name)[self._solid] != 0.0):
                raise ValueError(f"state.{name} must be zero in SOLID cells")

    def _check_faces(self, faces: FaceVelocities) -> None:
        if faces.u.shape != self._u_shape or faces.v.shape != self._v_shape:
            raise ValueError(
                f"faces must have shapes u {self._u_shape} and v {self._v_shape}, "
                f"got u {faces.u.shape} and v {faces.v.shape}"
            )
        if not (np.isfinite(faces.u).all() and np.isfinite(faces.v).all()):
            raise ValueError("faces must be finite")

    def _check_conditions(self, c: TurbulenceConditions) -> None:
        """Shapes, dtypes, finiteness and ranges of the conditions, before any arithmetic."""
        ny, nx = self._p_shape
        floats = {
            "inflow_k_u": self._u_shape,
            "inflow_eps_u": self._u_shape,
            "inflow_k_v": self._v_shape,
            "inflow_eps_v": self._v_shape,
            "tangential_bottom": (nx + 1,),
            "tangential_top": (nx + 1,),
            "tangential_left": (ny + 1,),
            "tangential_right": (ny + 1,),
            "eps_wall": self._p_shape,
            "production": self._p_shape,
        }
        for name, shape in floats.items():
            array = getattr(c, name)
            if array.shape != shape or array.dtype != np.float64:
                raise ValueError(
                    f"conditions.{name} must be float64 of shape {shape}, "
                    f"got {array.dtype} {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"conditions.{name} must be finite")
        for name in ("eps_held", "production_given"):
            array = getattr(c, name)
            if array.shape != self._p_shape or array.dtype != np.bool_:
                raise ValueError(
                    f"conditions.{name} must be bool of shape {self._p_shape}"
                )
            if (array & self._solid).any():
                raise ValueError(f"conditions.{name} names a SOLID cell")
        edges = (
            c.inflow_k_u[:, [0, -1]],
            c.inflow_eps_u[:, [0, -1]],
            c.inflow_k_v[[0, -1], :],
            c.inflow_eps_v[[0, -1], :],
        )
        if not all((edge > 0.0).all() for edge in edges):
            raise ValueError(
                "conditions' inflow k and eps must be positive on the domain faces"
            )
        if not (c.eps_wall[c.eps_held] > 0.0).all():
            raise ValueError("conditions.eps_wall must be positive where eps is held")
        if not (c.production[c.production_given] >= 0.0).all():
            raise ValueError(
                "conditions.production must be non-negative where it is given"
            )


@dataclass(frozen=True, eq=False)
class _VelocityWalls:
    """The wall-function faces of one velocity component.

    ``mask`` is [ny+1, nx+1] in ``wall_mu``'s layout; ``rows`` and ``cols``
    index its True entries; ``frame_t`` and ``frame_s`` locate each face's
    unknown in the component's own frame, where it lies between cells
    ``[t, s-1]`` and ``[t, s]``; ``distance`` is the unknown's distance to
    the face.
    """

    mask: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    frame_t: np.ndarray
    frame_s: np.ndarray
    distance: np.ndarray


def _velocity_walls(
    solid: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    t_faces: np.ndarray,
    t_centers: np.ndarray,
    transpose: bool,
) -> _VelocityWalls:
    """Locate one component's wall-function faces, in its own frame.

    ``solid`` is [nt, ns] with the component's own axis last; ``low`` and
    ``high`` [ns+1] mark the wall corners of the two transverse domain
    edges. An unknown's transverse face is a wall where it lies on such an
    edge corner, or where the neighbour face beyond it bounds a SOLID cell
    on either side: the momentum stencil's obstacle faces, its corner rule
    included. ``transpose`` maps the frame to ``wall_mu``'s layout for v.
    """
    nt, ns = solid.shape
    bounds_solid = np.zeros((nt, ns + 1), dtype=bool)
    bounds_solid[:, 1:-1] = solid[:, :-1] | solid[:, 1:]
    unknown = np.zeros((nt, ns + 1), dtype=bool)
    unknown[:, 1:-1] = ~bounds_solid[:, 1:-1]
    south = np.zeros((nt, ns + 1), dtype=bool)
    north = np.zeros((nt, ns + 1), dtype=bool)
    south[0, :] = low
    south[1:, :] = bounds_solid[:-1, :]
    north[-1, :] = high
    north[:-1, :] = bounds_solid[1:, :]
    t_s, s_s = np.nonzero(south & unknown)
    t_n, s_n = np.nonzero(north & unknown)
    face_t = np.concatenate((t_s, t_n + 1))
    face_s = np.concatenate((s_s, s_n))
    distance = np.concatenate(
        (t_centers[t_s] - t_faces[t_s], t_faces[t_n + 1] - t_centers[t_n])
    )
    mask = np.zeros((nt + 1, ns + 1), dtype=bool)
    mask[face_t, face_s] = True
    rows, cols = face_t, face_s
    if transpose:
        mask, rows, cols = np.ascontiguousarray(mask.T), face_s, face_t
    mask.flags.writeable = False
    return _VelocityWalls(
        mask=mask,
        rows=rows,
        cols=cols,
        frame_t=np.concatenate((t_s, t_n)),
        frame_s=face_s,
        distance=distance,
    )


@dataclass(frozen=True)
class _InletFace:
    """One domain face that admits air: its inflow values and its inflow."""

    k: float
    eps: float
    inflow: float


class TurbulenceBoundary:
    """The coupled solve's boundary data for k, eps and the walls (ADR-012 B, C).

    Built once per solver from the staggered boundary layer; each outer
    iteration asks it for the wall viscosity of the momentum prediction and
    the conditions of the k and eps step.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh, uniform or stretched.
    config : SimConfig
        Must carry a ``turbulence`` section; supplies the variant's C_mu,
        rho, mu and each inlet's ``turbulence_intensity`` and
        ``dissipation_length``.
    boundary : StaggeredBoundary
        Supplies the wall faces and the edges' tangential conditions.

    Attributes
    ----------
    wall_faces : dict[str, np.ndarray]
        Keys "u" and "v", bool [ny+1, nx+1] in ``wall_mu``'s layout, read-only:
        the faces the wall function sets. Every domain-edge face of a wall
        (zero normal velocity, not an outlet) beside an unknown, and every
        obstacle face. Not an inlet that admits air, not an outlet.
    wall_cells : np.ndarray
        Bool [ny, nx], read-only: the non-SOLID cells with at least one wall
        face, where eps is held and the production given.

    Raises
    ------
    ValueError
        Without a turbulence section, or if a velocity inlet that admits air
        lacks its turbulence keys.

    Notes
    -----
    The wall cells (ADR-012 B, C). For each wall face of a cell, with y_P the
    distance from the cell centre to that face, k_P the cell's k and
    ``u_k = C_mu^(1/4) k_P^(1/2)``:

    - eps held at ``C_mu^(3/4) k_P^(3/2) / (kappa y_P) = u_k^3 / (kappa y_P)``;
    - the production per unit density
      ``P = (tau_w / rho) u_k / (kappa y_P)``, with
      ``tau_w / rho = u_k kappa |U_P - U_wall| / ln(E max(y*, y*_0))`` the
      wall function's shear (the same law ``wall_viscosity`` codes, at the
      cell centre) and ``u_k / (kappa y_P)`` the log law's velocity gradient
      at the node. That is ``P = u_k^2 |U_P - U_wall| / (y_P ln(E max(y*,
      y*_0)))``. Source: the standard wall-function production of Launder
      and Spalding (1974), as Versteeg and Malalasekera (2007, chapter 9)
      write it: ``P_k = tau_w (dU/dy)_P`` with ``(dU/dy)_P = u_k / (kappa
      y_P)``. In the log layer, where ``|U_P - U_wall| = (u_k / kappa) ln(E
      y*)``, it equals the held eps.

    U_P is the cell-centre velocity along the wall, the mean of the cell's
    two faces parallel to it (a face beside a SOLID cell reads as zero, as
    in the step), and U_wall the edge's tangential velocity at the cell, the
    mean of its two corners' values, or zero at an obstacle.

    A cell with more than one wall face (a corner) takes the arithmetic mean
    of its walls' eps and of their productions. On one wall it is that
    wall's rule; a reflection of the room maps a cell's set of walls onto
    the reflected cell's, so the mean is symmetric under reflection. It is
    the rule OpenFOAM's epsilonWallFunction applies through its corner
    weights, one over the number of wall faces of the cell.

    The inflow (ADR-012 C): ``inflow_values`` per inlet face. An outlet face
    the flux turns inward through takes the adjacent cell's value, and so
    does every other domain face, where no flux enters and nothing is read;
    beside a SOLID edge cell, where no flux crosses, the largest k or eps of
    the field stands in. A tangential velocity at a pressure outlet, zero
    gradient in the momentum layer, is the adjacent face's.

    The initial state (ECR-002 step 6): k and eps uniform at their means over
    the inlet faces weighted by each face's inflow, the values the supply
    carries in as a whole. Where the inlets agree, their common value.
    """

    def __init__(
        self, mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary
    ) -> None:
        if config.turbulence is None:
            raise ValueError(
                "the k-epsilon boundary needs a turbulence section in the configuration"
            )
        self._c_mu = VARIANTS[config.turbulence.variant].c_mu
        self._rho = config.rho
        self._nu = config.mu / config.rho
        self._p_shape = p_shape(mesh)
        solid = mesh.cell_type == SOLID
        self._solid = solid
        self._live = ~solid
        self._live_u = np.ones(u_shape(mesh), dtype=bool)
        self._live_u[:, 0] = ~solid[:, 0]
        self._live_u[:, -1] = ~solid[:, -1]
        self._live_u[:, 1:-1] = ~solid[:, :-1] & ~solid[:, 1:]
        self._live_v = np.ones(v_shape(mesh), dtype=bool)
        self._live_v[0, :] = ~solid[0, :]
        self._live_v[-1, :] = ~solid[-1, :]
        self._live_v[1:-1, :] = ~solid[:-1, :] & ~solid[1:, :]
        walls = boundary.wall_faces()
        self._tangential = boundary.tangential_conditions()

        self._velocity = {
            "u": _velocity_walls(
                solid,
                walls["bottom"].corners,
                walls["top"].corners,
                mesh.y,
                mesh.yc,
                transpose=False,
            ),
            "v": _velocity_walls(
                np.ascontiguousarray(solid.T),
                walls["left"].corners,
                walls["right"].corners,
                mesh.x,
                mesh.xc,
                transpose=True,
            ),
        }
        self.wall_faces: dict[str, np.ndarray] = {
            key: walls_.mask for key, walls_ in self._velocity.items()
        }

        # The wall cells: per side, which cells have a wall there, the
        # centre's distance to it and the wall's own velocity along it.
        ny, nx = self._p_shape
        live = self._live
        south = np.zeros((ny, nx), dtype=bool)
        north = np.zeros((ny, nx), dtype=bool)
        west = np.zeros((ny, nx), dtype=bool)
        east = np.zeros((ny, nx), dtype=bool)
        south[0, :] = walls["bottom"].cells
        south[1:, :] = solid[:-1, :]
        north[-1, :] = walls["top"].cells
        north[:-1, :] = solid[1:, :]
        west[:, 0] = walls["left"].cells
        west[:, 1:] = solid[:, :-1]
        east[:, -1] = walls["right"].cells
        east[:, :-1] = solid[:, 1:]
        mean = {
            edge: 0.5 * (cond.value[:-1] + cond.value[1:])
            for edge, cond in self._tangential.items()
        }
        moving = {side: np.zeros((ny, nx)) for side in ("s", "n", "w", "e")}
        moving["s"][0, :] = mean["bottom"]
        moving["n"][-1, :] = mean["top"]
        moving["w"][:, 0] = mean["left"]
        moving["e"][:, -1] = mean["right"]
        ones = np.ones((ny, nx))
        self._sides = (
            (south & live, (mesh.yc - mesh.y[:-1])[:, None] * ones, moving["s"], "u"),
            (north & live, (mesh.y[1:] - mesh.yc)[:, None] * ones, moving["n"], "u"),
            (west & live, (mesh.xc - mesh.x[:-1])[None, :] * ones, moving["w"], "v"),
            (east & live, (mesh.x[1:] - mesh.xc)[None, :] * ones, moving["e"], "v"),
        )
        count = sum(side[0].astype(np.int64) for side in self._sides)
        self._wall_count = np.maximum(count, 1)
        self.wall_cells: np.ndarray = count > 0
        self.wall_cells.flags.writeable = False

        self._inlets = self._inlet_faces(mesh, config)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def wall_viscosity(
        self, k: np.ndarray, elsewhere: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """``wall_mu`` for the momentum prediction: mu_w on every wall-function face.

        Parameters
        ----------
        k : np.ndarray
            The current k, [ny, nx], positive in every non-SOLID cell.
        elsewhere : dict[str, np.ndarray]
            Keys "u" and "v", [ny+1, nx+1]: the viscosity every other face
            of the wall stencil keeps (an inlet's or an outlet's face).

        Returns
        -------
        dict[str, np.ndarray]
            New arrays, ``elsewhere``'s values with ``wall_viscosity`` of
            the face's unknown in place on the wall-function faces: y_P the
            unknown's distance to the face (half its cell), k_P the mean of
            the two cells the unknown lies between.
        """
        out = {}
        for key, frame in (("u", k), ("v", k.T)):
            walls = self._velocity[key]
            field = np.array(elsewhere[key], dtype=np.float64, copy=True)
            k_p = 0.5 * (
                frame[walls.frame_t, walls.frame_s - 1]
                + frame[walls.frame_t, walls.frame_s]
            )
            field[walls.rows, walls.cols] = wall_viscosity(
                k_p, walls.distance, self._c_mu, self._rho, self._nu
            )
            out[key] = field
        return out

    def conditions(
        self, state: TurbulenceState, faces: FaceVelocities
    ) -> TurbulenceConditions:
        """The k and eps step's conditions from the current state and corrected faces.

        Parameters
        ----------
        state : TurbulenceState
            The iterate whose k the wall cells' values are formed from.
        faces : FaceVelocities
            The corrected faces of this outer iteration.

        Returns
        -------
        TurbulenceConditions
            The inflow values, the edges' tangential velocities, eps held and
            the production given in the wall cells (the class Notes).
        """
        k, eps = state.k, state.eps
        u = np.where(self._live_u, faces.u, 0.0)
        v = np.where(self._live_v, faces.v, 0.0)
        along = {
            "u": 0.5 * (u[:, :-1] + u[:, 1:]),
            "v": 0.5 * (v[:-1, :] + v[1:, :]),
        }
        eps_sum = np.zeros(self._p_shape)
        production_sum = np.zeros(self._p_shape)
        for mask, distance, wall_speed, component in self._sides:
            eps_w, production_w = self._wall_cell_values(
                k, distance, np.abs(along[component] - wall_speed)
            )
            eps_sum += np.where(mask, eps_w, 0.0)
            production_sum += np.where(mask, production_w, 0.0)
        held = np.array(self.wall_cells)
        adjacent = {
            "bottom": faces.u[0, :],
            "top": faces.u[-1, :],
            "left": faces.v[:, 0],
            "right": faces.v[:, -1],
        }
        tangential = {
            edge: np.where(cond.is_dirichlet, cond.value, adjacent[edge])
            for edge, cond in self._tangential.items()
        }
        inflow_k_u, inflow_k_v = self._inflow(k, "k")
        inflow_eps_u, inflow_eps_v = self._inflow(eps, "eps")
        return TurbulenceConditions(
            inflow_k_u=inflow_k_u,
            inflow_k_v=inflow_k_v,
            inflow_eps_u=inflow_eps_u,
            inflow_eps_v=inflow_eps_v,
            tangential_bottom=np.array(tangential["bottom"], dtype=np.float64),
            tangential_top=np.array(tangential["top"], dtype=np.float64),
            tangential_left=np.array(tangential["left"], dtype=np.float64),
            tangential_right=np.array(tangential["right"], dtype=np.float64),
            eps_held=held,
            eps_wall=np.where(held, eps_sum / self._wall_count, 0.0),
            production_given=held.copy(),
            production=np.where(held, production_sum / self._wall_count, 0.0),
        )

    def initial_values(self) -> tuple[float, float]:
        """The uniform k and eps the coupled solve starts from (class Notes).

        Returns
        -------
        tuple[float, float]
            k (m^2/s^2) and eps (m^2/s^3): their means over the inlet faces
            weighted by each face's inflow.

        Raises
        ------
        ValueError
            If no velocity inlet admits air: nothing sets the supply's
            turbulence.
        """
        inlets = self._require_inlets()
        total = sum(face.inflow for face in inlets)
        k = sum(face.k * face.inflow for face in inlets) / total
        eps = sum(face.eps * face.inflow for face in inlets) / total
        return k, eps

    def largest_inlet_eddy_viscosity(self) -> float:
        """The largest ``C_mu k_in^2 / eps_in`` over the inlet faces, m^2/s.

        Condition (e)'s scale takes it (ADR-012 E): boundary data, fixed per
        solve.

        Raises
        ------
        ValueError
            If no velocity inlet admits air.
        """
        return max(self._c_mu * face.k**2 / face.eps for face in self._require_inlets())

    # ------------------------------------------------------------------
    # Parts
    # ------------------------------------------------------------------

    def _wall_cell_values(
        self, k: np.ndarray, y_p: np.ndarray, slip: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """One wall's held eps and production per cell (class Notes)."""
        u_k = self._c_mu**0.25 * np.sqrt(k)
        y_star = u_k * y_p / self._nu
        log_term = np.log(E_WALL * np.maximum(y_star, Y_STAR_FLOOR))
        return u_k**3 / (KAPPA * y_p), u_k**2 * slip / (y_p * log_term)

    def _inflow(self, q: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        """The inflow arrays of one quantity: inlet values, else the adjacent cell's."""
        ny, nx = self._p_shape
        stand_in = float(q[self._live].max())
        edge_cells = {
            "bottom": q[0, :],
            "top": q[-1, :],
            "left": q[:, 0],
            "right": q[:, -1],
        }
        values = {}
        for edge, cells in edge_cells.items():
            row = np.where(cells > 0.0, cells, stand_in)
            for index, face in self._inlet_by_edge[edge].items():
                row[index] = getattr(face, name)
            values[edge] = row
        flow_u = np.zeros((ny, nx + 1))
        flow_v = np.zeros((ny + 1, nx))
        flow_u[:, 0], flow_u[:, -1] = values["left"], values["right"]
        flow_v[0, :], flow_v[-1, :] = values["bottom"], values["top"]
        return flow_u, flow_v

    def _inlet_faces(self, mesh: Mesh, config: SimConfig) -> list[_InletFace]:
        """Every domain face that admits air, from the shared coverage (REQ-S12.1)."""
        registry = BoundaryRegistry(config)
        widths = {
            "bottom": mesh.dx_cell,
            "top": mesh.dx_cell,
            "left": mesh.dy_cell,
            "right": mesh.dy_cell,
        }
        self._inlet_by_edge: dict[str, dict[int, _InletFace]] = {}
        faces: list[_InletFace] = []
        for edge, width in widths.items():
            by_index: dict[int, _InletFace] = {}
            coverage = registry.coverage_along(edge, *edge_cell_inputs(mesh, edge))
            for index, point in enumerate(coverage):
                if point.spec is None or point.condition.bc_type != VELOCITY_INLET:
                    continue
                c = point.condition
                normal = c.v_prescribed if edge in ("bottom", "top") else c.u_prescribed
                if normal == 0.0:
                    continue
                spec = point.spec
                if spec.turbulence_intensity is None or spec.dissipation_length is None:
                    raise ValueError(
                        f"boundary '{point.name}' admits air but states no "
                        "turbulence_intensity or dissipation_length"
                    )
                k_in, eps_in = inflow_values(
                    spec.turbulence_intensity, normal, spec.dissipation_length
                )
                face = _InletFace(
                    k=k_in, eps=eps_in, inflow=abs(normal) * float(width[index])
                )
                by_index[index] = face
                faces.append(face)
            self._inlet_by_edge[edge] = by_index
        return faces

    def _require_inlets(self) -> list[_InletFace]:
        if not self._inlets:
            raise ValueError(
                "no velocity inlet admits air, so nothing sets the supply's k and "
                "eps: the coupled solve takes its initial state and condition "
                "(e)'s scale from the inlets"
            )
        return self._inlets


def _corner_difference(
    values: np.ndarray, positions: np.ndarray, corners: np.ndarray, buried: np.ndarray
) -> np.ndarray:
    """The derivative along the first axis at the corners between rows of values.

    Parameters
    ----------
    values : np.ndarray
        [n+2, m]: the low edge's values, the n rows of face values at the
        cell centres, the high edge's values.
    positions : np.ndarray
        [n+2], the coordinate of each row: the low edge, the centres, the
        high edge.
    corners : np.ndarray
        [n+1], the corner coordinates along the axis.
    buried : np.ndarray
        [n+2, m], True where a row's face has a SOLID cell on either side; its
        value is replaced by zero at the corner itself, on the obstacle.

    Returns
    -------
    np.ndarray
        [n+1, m], ``(upper - lower) / (x_upper - x_lower)`` at each corner,
        zero where both sides are buried, so the two positions coincide.
    """
    lo_val, hi_val = values[:-1], values[1:]
    lo_pos = np.broadcast_to(positions[:-1, None], lo_val.shape)
    hi_pos = np.broadcast_to(positions[1:, None], hi_val.shape)
    corner = np.broadcast_to(corners[:, None], lo_val.shape)
    lo_b, hi_b = buried[:-1], buried[1:]
    lo_val = np.where(lo_b, 0.0, lo_val)
    hi_val = np.where(hi_b, 0.0, hi_val)
    lo_pos = np.where(lo_b, corner, lo_pos)
    hi_pos = np.where(hi_b, corner, hi_pos)
    span = hi_pos - lo_pos
    defined = span > 0.0
    return np.where(defined, (hi_val - lo_val) / np.where(defined, span, 1.0), 0.0)
