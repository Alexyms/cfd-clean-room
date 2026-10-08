"""Transport of one particle class on the staggered face velocities (ADR-011 B, C, D, F).

Phase 3 (REQ-T01, T03 to T08, T11, T12, N01). Concentration ``C_k`` sits at the
cell centres with the pressure, number of particles per cubic metre, and is
advected by the face velocities the velocity solver exposes (REQ-S13): the
flux through the face ``u[j, i]`` is ``u dy_cell[j]`` times a face
concentration, with no interpolation of the velocity, so a uniform field
drifts only at the rate the stopping rule's per-cell imbalance sets
(REQ-T11). One call advances one class by one explicit step.

The step (ADR-011 C). Advection is forward Euler with the face value below;
``stable_dt`` bounds the step so that the sum of the two directional Courant
numbers in the worst cell is ``cfl_number``, at most 1/2, where each update
is a convex combination of neighbouring values and the field stays within its
bounds (REQ-T12). Diffusion and the deposition sink are backward Euler, one
Jacobi solve to ``diffusion_tol`` within ``max_diffusion_iter`` sweeps, so a
step can never take a cell below zero. Sources are added after the advection
update and before the implicit solve.

The face value (ADR-011 B). QUICK's quadratic through the upstream,
downstream and next-upstream nodes, evaluated by ``quick_face_values`` with
Lagrange weights from the node positions, then clamped by the UMIST limiter,
``psi(r) = max(0, min(2r, (1 + 3r) / 4, psi_quick, 2))`` with
``r = (C_C - C_U) / (C_D - C_C)``. On a uniform mesh ``psi_quick`` is
``(3 + r) / 4`` and the limiter is Lien and Leschziner's; on a stretched mesh
the quadratic is the one the node positions give and the clamp is applied to
it. Where the downstream difference is zero the ratio is undefined and the
face value is ``C_C``; without that guard the scheme is not exact on a
uniform field. ``advection_scheme: upwind`` takes ``C_C`` everywhere, the
comparison scheme. The face value, the advective flux and the implicit
solve are ``src/scalar_scheme.py``'s, shared with the k-epsilon model; this
module passes them the class's fluxes, conductances and deposition sink.

Boundaries (ADR-011 E). The solver reads a ConcentrationFaces and imposes
nothing of its own. At a domain face the sign of the face flux decides:
inward, the face carries ``inflow_u`` or ``inflow_v``, placed at the face as
Leonard's boundary node so the first interior face's stencil reads the inlet
value at its physical location; outward, the face value is the upwind cell's
with no stencil. A wall face has zero flux, which the velocity layer
guarantees for the velocity solver's fields; a face between a non-SOLID and
a SOLID cell, and a domain face behind a SOLID edge cell, carries no
advective flux whatever velocity it holds, so no field can carry mass into a
SOLID cell. No diffusion crosses any of them. A far-upstream node that falls
in a SOLID cell is read as the upstream cell's value, the zero-gradient wall
rule the domain edges use. SOLID cells are zero after every step.

Settling (ADR-011 D). The class's settling velocity is subtracted from the
vertical face velocity on the faces ``settling_v`` marks, horizontal faces
between two non-SOLID cells, and nowhere else; every other horizontal face
removes through ``deposition_v`` once, the floor value including settling
(REQ-T09). ``v_ext`` is a per-class drift velocity in FaceVelocities shape,
added on the interior faces between two non-SOLID cells, both components;
None reads as zero (REQ-T06, ADR-007).

The budget (ADR-011 F). ``MassBudget``, one per class, is written by the
solver from the face fluxes it subtracts from the cells and the source array
it adds, in the same step, and by nothing else. ``FieldHistory`` is the
output contract for Phase 7's animation; the solver never calls it, since it
does not own the loop.

The solver reads ``settling_velocity`` and ``diffusion_coeff`` of ``physics``
and ``faces_for`` of ``boundary``, and nothing else of either; the protocols
``ParticleProperties`` and ``ScalarConditions`` say so in the signature, so a
validation case may hand it objects of its own that answer those calls
(validation/transport_cases.py). ParticlePhysics and ConcentrationBoundary
are the production types.
"""

import logging
import math
from dataclasses import dataclass, field
from os import PathLike
from typing import Protocol

import numpy as np

from src.boundary_concentration import (
    SURFACE_CEILING,
    SURFACE_FLOOR,
    SURFACE_WALL,
    ConcentrationBoundary,
    ConcentrationFaces,
)
from src.config import (
    DEPOSITION_CEILING,
    DEPOSITION_FLOOR,
    DEPOSITION_WALL,
    UPWIND,
    SimConfig,
)
from src.mesh import SOLID, Mesh
from src.particles import ParticlePhysics
from src.scalar_scheme import advective_flux, implicit_step, mesh_axes
from src.staggered import FaceVelocities, p_shape, u_shape, v_shape

logger = logging.getLogger(__name__)

# The budget's deposition slots. The first three are domain surfaces, booked
# by the face's surface code; OBSTACLE is every depositing face between a
# non-SOLID and a SOLID cell, whatever its orientation (ADR-011 F).
OBSTACLE = "obstacle"
DEPOSIT_SURFACES: tuple[str, ...] = (
    DEPOSITION_FLOOR,
    DEPOSITION_CEILING,
    DEPOSITION_WALL,
    OBSTACLE,
)
_DOMAIN_SURFACE: dict[int, str] = {
    SURFACE_FLOOR: DEPOSITION_FLOOR,
    SURFACE_CEILING: DEPOSITION_CEILING,
    SURFACE_WALL: DEPOSITION_WALL,
}


@dataclass
class MassBudget:
    """The particle count of one class, as the solver books it (ADR-011 F).

    Every term is a number of particles per metre of depth, since ``C`` is
    a number concentration and the depth is one metre. Written by
    ``TransportSolver.solve_timestep`` only; tests and Phase 4 read it.

    Parameters
    ----------
    initial : float or None
        ``in_domain`` of the field first handed to the solver; None before
        the first step.
    inflow : float
        Cumulative particles carried in through domain faces.
    outflow : float
        Cumulative particles carried out through domain faces.
    source : float
        Cumulative ``sum(sources V) dt``.
    deposited : dict[str, float]
        Cumulative deposition by surface: floor, ceiling, wall, obstacle.
    current : float or None
        ``in_domain`` of the field the last step returned; None before it.
    """

    initial: float | None = None
    inflow: float = 0.0
    outflow: float = 0.0
    source: float = 0.0
    deposited: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(DEPOSIT_SURFACES, 0.0)
    )
    current: float | None = None

    @staticmethod
    def in_domain(C_k: np.ndarray, mesh: Mesh) -> float:  # noqa: N803 -- the contract's name for the class field (SYSTEM.md section 4)
        """Particles in the domain: ``sum(C V)`` over non-SOLID cells.

        Parameters
        ----------
        C_k : np.ndarray
            Concentration, shape [ny, nx].
        mesh : Mesh
            Supplies the cell widths and ``cell_type``. The BOUNDARY ring
            holds concentration and carries every inlet and outlet, so it
            counts; a sum over FLUID alone would omit it (ADR-011 F).

        Returns
        -------
        float
            The content, particles per metre of depth.
        """
        volumes = np.outer(mesh.dy_cell, mesh.dx_cell)
        return float(np.sum((C_k * volumes)[mesh.cell_type != SOLID]))

    def residual(self) -> float:
        """``initial + inflow + source - outflow - deposited - current``.

        Raises
        ------
        ValueError
            Before the first step, when there is nothing to balance.
        """
        if self.initial is None or self.current is None:
            raise ValueError("the budget has no field yet; step one first")
        return (
            self.initial
            + self.inflow
            + self.source
            - self.outflow
            - sum(self.deposited.values())
            - self.current
        )

    def relative(self) -> float:
        """``residual()`` over ``initial + inflow + source``, the particles supplied.

        Raises
        ------
        ValueError
            Before the first step, or when nothing was supplied and the
            ratio is undefined.
        """
        residual = self.residual()
        supplied = self.initial + self.inflow + self.source
        if supplied == 0.0:
            raise ValueError("no particles were supplied; the ratio is undefined")
        return residual / supplied


class FieldHistory:
    """Concentration fields recorded every N steps, Phase 7's animation input.

    Parameters
    ----------
    every : int
        Record when ``step % every == 0``; ``config.output_interval``.

    Attributes
    ----------
    frames : list[tuple[int, float, dict[int, np.ndarray]]]
        ``(step, t, fields)`` per recorded step, the fields copied.

    Raises
    ------
    TypeError
        If ``every`` is not an int (a bool is not one here).
    ValueError
        If ``every`` is not positive.
    """

    def __init__(self, every: int) -> None:
        if isinstance(every, bool) or not isinstance(every, int):
            raise TypeError(f"every must be an int, got {type(every).__name__}")
        if every <= 0:
            raise ValueError(f"every must be positive, got {every}")
        self.every: int = every
        self.frames: list[tuple[int, float, dict[int, np.ndarray]]] = []

    def record(self, step: int, t: float, fields: dict[int, np.ndarray]) -> None:
        """Keep a copy of every class field when the step is on the interval.

        Parameters
        ----------
        step : int
            The time step index, zero at the initial field.
        t : float
            Simulated time in seconds.
        fields : dict[int, np.ndarray]
            Class index to its field, [ny, nx]. Copied, so a later step
            cannot change a recorded frame.
        """
        if step % self.every != 0:
            return
        copies = {
            int(k): np.array(C, dtype=np.float64, copy=True) for k, C in fields.items()
        }
        self.frames.append((step, float(t), copies))

    def save(self, path: str | PathLike[str]) -> None:
        """Write the frames as one npz file.

        Parameters
        ----------
        path : str or PathLike
            Destination; numpy appends ``.npz`` if it is missing.

        Notes
        -----
        Keys: ``steps`` [n_frames] int64, ``times`` [n_frames] float64, and
        ``C_<k>`` [n_frames, ny, nx] float64 for every class ``k`` present.
        Every frame must carry the same classes.

        Raises
        ------
        ValueError
            With no frames, or when the frames do not all carry the same
            classes.
        """
        if not self.frames:
            raise ValueError("no frames to save")
        classes = sorted(self.frames[0][2])
        for step, _, fields in self.frames:
            if sorted(fields) != classes:
                raise ValueError(f"frame at step {step} carries other classes")
        arrays: dict[str, np.ndarray] = {
            "steps": np.array([f[0] for f in self.frames], dtype=np.int64),
            "times": np.array([f[1] for f in self.frames], dtype=np.float64),
        }
        for k in classes:
            arrays[f"C_{k}"] = np.stack([f[2][k] for f in self.frames])
        np.savez(path, **arrays)


class ParticleProperties(Protocol):
    """What the solver reads of a particle model: ParticlePhysics satisfies it.

    A validation case may hand the solver a stand-in that fixes both values
    (validation.transport_cases.ScalarPhysics).
    """

    def settling_velocity(self, size_class: int) -> float:
        """Settling velocity of the class in m/s, positive downward."""
        ...

    def diffusion_coeff(self, size_class: int) -> float:
        """Diffusion coefficient of the class in m^2/s."""
        ...


class ScalarConditions(Protocol):
    """What the solver reads of a boundary layer: ConcentrationBoundary satisfies it.

    A validation case may hand the solver a stand-in that returns the faces
    it built (validation.transport_cases.FixedConditions).
    """

    def faces_for(self, size_class: int) -> ConcentrationFaces:
        """The scalar condition at every face for one class."""
        ...


class TransportSolver:
    """Advance one particle class one step on the staggered face velocities.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh, uniform or stretched.
    config : SimConfig
        Must carry a ``transport`` section: cfl_number, advection_scheme,
        max_diffusion_iter, diffusion_tol. Supplies the class count.
    physics : ParticlePhysics or ParticleProperties
        Supplies ``settling_velocity`` and ``diffusion_coeff`` per class;
        nothing else of it is read.
    boundary : ConcentrationBoundary or ScalarConditions
        Supplies ``faces_for(size_class)``, the scalar condition at every
        face; nothing else of it is read.

    Attributes
    ----------
    budget : list[MassBudget]
        One per class, written by ``solve_timestep`` only.
    last_diffusion_sweeps : int
        Jacobi sweeps the last implicit solve took.
    diffusion_converged : bool
        Whether the last implicit solve met ``diffusion_tol`` within the cap.

    Raises
    ------
    ValueError
        Without a transport section, or when a class's ConcentrationFaces
        is not shaped for the mesh, not of the dtype the contract names, or
        not finite.
    """

    def __init__(
        self,
        mesh: Mesh,
        config: SimConfig,
        physics: ParticlePhysics | ParticleProperties,
        boundary: ConcentrationBoundary | ScalarConditions,
    ) -> None:
        if config.transport is None:
            raise ValueError(
                "the transport solver needs a transport section in the configuration"
            )
        spec = config.transport
        self._mesh = mesh
        self._cfl = spec.cfl_number
        self._upwind = spec.advection_scheme == UPWIND
        self._max_sweeps = spec.max_diffusion_iter
        self._tol = spec.diffusion_tol
        self._u_shape = u_shape(mesh)
        self._v_shape = v_shape(mesh)
        self._p_shape = p_shape(mesh)
        self._n_classes = len(config.particle_sizes)

        solid = mesh.cell_type == SOLID
        self._solid = solid
        self._live = ~solid
        self._volume = np.outer(mesh.dy_cell, mesh.dx_cell)
        # Faces with an advective flux: both cells non-SOLID, or one non-SOLID
        # edge cell at a domain face.
        live_u = np.ones(self._u_shape, dtype=bool)
        live_u[:, 0] = ~solid[:, 0]
        live_u[:, -1] = ~solid[:, -1]
        live_u[:, 1:-1] = ~solid[:, :-1] & ~solid[:, 1:]
        live_v = np.ones(self._v_shape, dtype=bool)
        live_v[0, :] = ~solid[0, :]
        live_v[-1, :] = ~solid[-1, :]
        live_v[1:-1, :] = ~solid[:-1, :] & ~solid[1:, :]
        self._live_u, self._live_v = live_u, live_v
        # Interior faces between two non-SOLID cells: where diffusion crosses
        # and where v_ext acts.
        inner_u = np.zeros(self._u_shape, dtype=bool)
        inner_u[:, 1:-1] = live_u[:, 1:-1]
        inner_v = np.zeros(self._v_shape, dtype=bool)
        inner_v[1:-1, :] = live_v[1:-1, :]
        self._inner_u, self._inner_v = inner_u, inner_v
        self._area_u = np.broadcast_to(mesh.dy_cell[:, None], self._u_shape)
        self._area_v = np.broadcast_to(mesh.dx_cell[None, :], self._v_shape)
        # Diffusive conductance per unit D: A_f / d_face on the inner faces.
        self._conductance_u = np.where(
            inner_u, self._area_u / mesh.dx_face[None, :], 0.0
        )
        self._conductance_v = np.where(
            inner_v, self._area_v / mesh.dy_face[:, None], 0.0
        )

        self._axis_x, self._axis_y = mesh_axes(mesh)

        self._settling = [
            float(physics.settling_velocity(k)) for k in range(self._n_classes)
        ]
        self._diffusion = [
            float(physics.diffusion_coeff(k)) for k in range(self._n_classes)
        ]
        self._faces = [boundary.faces_for(k) for k in range(self._n_classes)]
        for k, faces in enumerate(self._faces):
            self._check_conditions(faces, k)
        # Deposition per class: v_d A_f on every depositing face, and the sum
        # over a cell's faces that sits in its implicit diagonal. Each
        # depositing face has one non-SOLID neighbour, so adding a face to
        # both its cells and zeroing SOLID cells books it once.
        self._deposit_u = [f.deposition_u * self._area_u for f in self._faces]
        self._deposit_v = [f.deposition_v * self._area_v for f in self._faces]
        self._deposit_cell = [
            np.where(
                solid,
                0.0,
                du[:, :-1] + du[:, 1:] + dv[:-1, :] + dv[1:, :],
            )
            for du, dv in zip(self._deposit_u, self._deposit_v, strict=True)
        ]
        self.budget: list[MassBudget] = [MassBudget() for _ in range(self._n_classes)]
        self.last_diffusion_sweeps: int = 0
        self.diffusion_converged: bool = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def stable_dt(
        self,
        faces: FaceVelocities,
        size_class: int,
        v_ext: FaceVelocities | None = None,
    ) -> float:
        """The largest step the explicit advection of this class is stable at.

        Parameters
        ----------
        faces : FaceVelocities
            The air's face velocities.
        size_class : int
            Index into the configured particle sizes.
        v_ext : FaceVelocities, optional
            Per-class drift increment on the faces; None is zero.

        Returns
        -------
        float
            ``cfl_number`` over the largest, across non-SOLID cells, of
            ``max(|u_w|, |u_e|) / dx_cell + max(|v_s|, |v_n|) / dy_cell``,
            the advecting velocities carrying the class's settling increment
            and ``v_ext``. The sum, not the larger, of the two directional
            rates: the unsplit update is a convex combination only when
            their sum is at most 1/2 (ADR-011 B). Infinite when nothing
            moves.

        Raises
        ------
        TypeError
            If ``size_class`` is not an int (a bool is not one here).
        IndexError
            If ``size_class`` is outside the configured classes.
        ValueError
            If ``faces`` or ``v_ext`` is not shaped for the mesh or holds a
            value that is not finite.
        """
        k = self._check_class(size_class)
        u_adv, v_adv = self._advecting_velocities(faces, k, v_ext)
        mesh = self._mesh
        rate = (
            np.maximum(np.abs(u_adv[:, :-1]), np.abs(u_adv[:, 1:]))
            / mesh.dx_cell[None, :]
            + np.maximum(np.abs(v_adv[:-1, :]), np.abs(v_adv[1:, :]))
            / mesh.dy_cell[:, None]
        )
        worst = float(rate[self._live].max()) if self._live.any() else 0.0
        if worst == 0.0:
            return float("inf")
        return self._cfl / worst

    def solve_timestep(
        self,
        C_k: np.ndarray,  # noqa: N803 -- the contract's name for the class field (SYSTEM.md section 4)
        faces: FaceVelocities,
        size_class: int,
        dt: float,
        v_ext: FaceVelocities | None = None,
        sources: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance one class by one step and book the budget.

        Parameters
        ----------
        C_k : np.ndarray
            Concentration of the class, shape [ny, nx], particles per cubic
            metre. Not modified.
        faces : FaceVelocities
            The air's face velocities.
        size_class : int
            Index into the configured particle sizes.
        dt : float
            The step, seconds, at most ``stable_dt``.
        v_ext : FaceVelocities, optional
            Per-class drift increment on the faces; None is zero (REQ-T06).
        sources : np.ndarray, optional
            Rate of the class, [ny, nx], particles per cubic metre per
            second, non-negative, zero in SOLID cells; None is zero. Added as
            ``sources dt`` and ``sum(sources V) dt`` is booked.

        Returns
        -------
        np.ndarray
            The new field, [ny, nx], float64, C-contiguous, zero in SOLID
            cells.

        Raises
        ------
        TypeError
            If ``size_class`` or ``dt`` is a bool (Python's, numpy's, or for
            ``dt`` a 0-d bool array), or ``size_class`` is not an int.
        IndexError
            If ``size_class`` is outside the configured classes.
        ValueError
            If ``dt`` is not finite and positive or exceeds ``stable_dt``, a
            shape does not fit the mesh, an array holds a value that is not
            finite, or a source is negative or sits in a SOLID cell. Every
            check runs before any arithmetic, so a bad input leaves the
            budget untouched.
        """
        k = self._check_class(size_class)
        c = np.array(C_k, dtype=np.float64, order="C", copy=True)
        if c.shape != self._p_shape:
            raise ValueError(f"expected C_k of shape {self._p_shape}, got {c.shape}")
        if not np.isfinite(c).all():
            raise ValueError("C_k must be finite")
        # A 0-d bool array is neither a bool nor an np.bool_ and would step
        # with dt = 1.0.
        if isinstance(dt, (bool, np.bool_)) or (
            isinstance(dt, np.ndarray) and dt.dtype == np.bool_
        ):
            raise TypeError("dt must be a number, not a bool")
        if not math.isfinite(dt):
            raise ValueError(f"dt must be finite, got {dt}")
        if not dt > 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        limit = self.stable_dt(faces, k, v_ext)
        # Written as "not dt <= limit" so that a limit that is not a number
        # could never let a step through; the finite checks on the faces make
        # such a limit impossible, and this form costs nothing.
        if not dt <= limit:
            raise ValueError(
                f"dt {dt} exceeds stable_dt {limit} for class {k}; the explicit "
                "advection step is a convex combination only below it"
            )
        rate = None
        if sources is not None:
            rate = np.asarray(sources, dtype=np.float64)
            if rate.shape != self._p_shape:
                raise ValueError(
                    f"expected sources of shape {self._p_shape}, got {rate.shape}"
                )
            if not np.isfinite(rate).all():
                raise ValueError("sources must be finite")
            if np.any(rate < 0.0):
                raise ValueError("sources must be non-negative (ADR-011 B)")
            if np.any(rate[self._solid] != 0.0):
                raise ValueError("a source sits in a SOLID cell")
        c[self._solid] = 0.0
        budget = self.budget[k]
        mesh = self._mesh
        if budget.initial is None:
            budget.initial = MassBudget.in_domain(c, mesh)

        conditions = self._faces[k]
        u_adv, v_adv = self._advecting_velocities(faces, k, v_ext)
        flux_u = u_adv * self._area_u
        flux_v = v_adv * self._area_v
        adv_u = advective_flux(
            c, flux_u, conditions.inflow_u, self._axis_x, self._upwind
        )
        adv_v = advective_flux(
            c.T, flux_v.T, conditions.inflow_v.T, self._axis_y, self._upwind
        ).T
        divergence = adv_u[:, 1:] - adv_u[:, :-1] + adv_v[1:, :] - adv_v[:-1, :]
        c_star = c - dt * divergence / self._volume
        if rate is not None:
            # After the advection update and before the implicit solve, so a
            # source in a floor cell deposits in the same step (ADR-011 C).
            c_star += dt * rate
            budget.source += float(np.sum(rate * self._volume)) * dt
        c_star[self._solid] = 0.0

        # Boundary bookkeeping from the fluxes just applied: a positive flux
        # points toward increasing coordinate, so it enters at the low face
        # and leaves at the high face.
        inflow = (
            np.sum(np.where(flux_u[:, 0] > 0.0, adv_u[:, 0], 0.0))
            - np.sum(np.where(flux_u[:, -1] < 0.0, adv_u[:, -1], 0.0))
            + np.sum(np.where(flux_v[0, :] > 0.0, adv_v[0, :], 0.0))
            - np.sum(np.where(flux_v[-1, :] < 0.0, adv_v[-1, :], 0.0))
        )
        outflow = (
            -np.sum(np.where(flux_u[:, 0] < 0.0, adv_u[:, 0], 0.0))
            + np.sum(np.where(flux_u[:, -1] > 0.0, adv_u[:, -1], 0.0))
            - np.sum(np.where(flux_v[0, :] < 0.0, adv_v[0, :], 0.0))
            + np.sum(np.where(flux_v[-1, :] > 0.0, adv_v[-1, :], 0.0))
        )
        budget.inflow += float(inflow) * dt
        budget.outflow += float(outflow) * dt

        c_new = self._implicit_step(c_star, dt, self._diffusion[k], k)
        self._book_deposition(c_new, dt, k, conditions)
        budget.current = MassBudget.in_domain(c_new, mesh)
        return np.ascontiguousarray(c_new)

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _check_class(self, size_class: int) -> int:
        if isinstance(size_class, bool) or not isinstance(size_class, int):
            raise TypeError(
                f"size_class must be an int, got {type(size_class).__name__}"
            )
        if not 0 <= size_class < self._n_classes:
            raise IndexError(
                f"size_class {size_class} out of range [0, {self._n_classes - 1}]"
            )
        return size_class

    def _check_pair(self, u: np.ndarray, v: np.ndarray, what: str) -> None:
        if u.shape != self._u_shape or v.shape != self._v_shape:
            raise ValueError(
                f"{what} must have shapes u {self._u_shape} and v {self._v_shape}, "
                f"got u {u.shape} and v {v.shape}"
            )
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            raise ValueError(f"{what} must be finite")

    def _check_conditions(self, faces: ConcentrationFaces, k: int) -> None:
        """Shapes, dtypes and finiteness of a class's faces, before any arithmetic.

        The dtypes are exactly the boundary_concentration contract's: float64
        for the carried and deposition velocities, int32 for the surface
        codes, bool for the settling mask.
        """
        dtypes = {
            "inflow": np.dtype(np.float64),
            "deposition": np.dtype(np.float64),
            "surface": np.dtype(np.int32),
            "settling": np.dtype(np.bool_),
        }
        for name in (
            "inflow_u",
            "inflow_v",
            "deposition_u",
            "deposition_v",
            "surface_u",
            "surface_v",
            "settling_v",
        ):
            array = getattr(faces, name)
            shape = self._u_shape if name.endswith("_u") else self._v_shape
            if array.shape != shape:
                raise ValueError(
                    f"class {k}: {name} must have shape {shape}, got {array.shape}"
                )
            expected = dtypes[name.rsplit("_", 1)[0]]
            if array.dtype != expected:
                raise ValueError(
                    f"class {k}: {name} must have dtype {expected.name}, got {array.dtype}"
                )
            if expected.kind == "f" and not (
                np.isfinite(array).all() and (array >= 0.0).all()
            ):
                raise ValueError(f"class {k}: {name} must be finite and non-negative")

    # ------------------------------------------------------------------
    # The step's parts
    # ------------------------------------------------------------------

    def _advecting_velocities(
        self, faces: FaceVelocities, k: int, v_ext: FaceVelocities | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The class's advecting face velocities: the air's, masked, with settling and v_ext."""
        self._check_pair(faces.u, faces.v, "faces")
        u_adv = np.where(self._live_u, faces.u, 0.0)
        v_adv = np.where(self._live_v, faces.v, 0.0)
        # Settling acts in -y on the faces the boundary layer marks, between
        # two non-SOLID cells; every other horizontal face removes through
        # deposition_v once (ADR-011 D).
        v_adv = v_adv - np.where(self._faces[k].settling_v, self._settling[k], 0.0)
        if v_ext is not None:
            self._check_pair(v_ext.u, v_ext.v, "v_ext")
            u_adv = u_adv + np.where(self._inner_u, v_ext.u, 0.0)
            v_adv = v_adv + np.where(self._inner_v, v_ext.v, 0.0)
        return u_adv, v_adv

    def _implicit_step(
        self, c_star: np.ndarray, dt: float, diffusivity: float, k: int
    ) -> np.ndarray:
        """Backward Euler diffusion and deposition, by ``scalar_scheme.implicit_step``.

        ``G_f = D A_f / d_f`` on the interior faces between non-SOLID cells,
        formed here before the solve, and the deposition sink of every wall
        face of P in the diagonal, so the step cannot take a cell below zero
        (ADR-011 C). Records the sweeps and whether the tolerance was met,
        and logs a warning at the cap.
        """
        g_u = diffusivity * self._conductance_u
        g_v = diffusivity * self._conductance_v
        result = implicit_step(
            c_star,
            self._volume,
            dt,
            g_u,
            g_v,
            self._deposit_cell[k],
            self._solid,
            self._tol,
            self._max_sweeps,
        )
        if not result.converged:
            logger.warning(
                "implicit diffusion solve stopped at max_diffusion_iter %d",
                self._max_sweeps,
            )
        self.last_diffusion_sweeps = result.sweeps
        self.diffusion_converged = result.converged
        return result.field

    def _book_deposition(
        self, c_new: np.ndarray, dt: float, k: int, conditions: ConcentrationFaces
    ) -> None:
        """Add ``v_d A_f C_P dt`` of every depositing face to its surface's slot.

        ``C_P`` is the depositing face's one non-SOLID neighbour, read as the
        sum of its two neighbours over a zero-padded field, since the other
        is SOLID or outside. Domain faces (index 0 and n along the normal)
        book to the surface their code names; interior faces are obstacle
        faces and book to ``obstacle`` whatever their code (ADR-011 F).
        """
        deposited = self.budget[k].deposited
        padded_x = np.pad(c_new, ((0, 0), (1, 1)))
        padded_y = np.pad(c_new, ((1, 1), (0, 0)))
        flux_u = self._deposit_u[k] * dt * (padded_x[:, :-1] + padded_x[:, 1:])
        flux_v = self._deposit_v[k] * dt * (padded_y[:-1, :] + padded_y[1:, :])
        deposited[OBSTACLE] += float(flux_u[:, 1:-1].sum() + flux_v[1:-1, :].sum())
        for code, name in _DOMAIN_SURFACE.items():
            deposited[name] += float(
                flux_u[:, 0][conditions.surface_u[:, 0] == code].sum()
                + flux_u[:, -1][conditions.surface_u[:, -1] == code].sum()
                + flux_v[0, :][conditions.surface_v[0, :] == code].sum()
                + flux_v[-1, :][conditions.surface_v[-1, :] == code].sum()
            )
