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
comparison scheme.

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
and ``faces_for`` of ``boundary``, and nothing else of either, so a
validation case may hand it objects of its own that answer those calls
(validation/transport_cases.py).
"""

import logging
from dataclasses import dataclass, field
from os import PathLike

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
from src.momentum import quick_face_values
from src.particles import ParticlePhysics
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
        """``residual()`` over ``initial + inflow + source``, the particles supplied."""
        supplied = self.initial + self.inflow + self.source
        if supplied == 0.0:
            raise ValueError("no particles were supplied; the ratio is undefined")
        return self.residual() / supplied


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


def limited_face_values(
    c_up: np.ndarray, c_c: np.ndarray, c_d: np.ndarray, quick: np.ndarray
) -> np.ndarray:
    """The UMIST clamp of the QUICK face value (ADR-011 B).

    Parameters
    ----------
    c_up : np.ndarray
        The node upstream of the upstream cell, per face.
    c_c : np.ndarray
        The upstream cell value.
    c_d : np.ndarray
        The downstream cell value.
    quick : np.ndarray
        The unlimited quadratic's face value, from ``quick_face_values``.

    Returns
    -------
    np.ndarray
        ``c_c + psi(r) (c_d - c_c) / 2`` with ``psi = max(0, min(2r,
        (1 + 3r) / 4, psi_quick, 2))``, ``r = (c_c - c_up) / (c_d - c_c)``
        and ``psi_quick = 2 (quick - c_c) / (c_d - c_c)``; ``c_c`` where the
        downstream difference is zero, so a uniform field is exact.
    """
    d_down = c_d - c_c
    defined = d_down != 0.0
    safe = np.where(defined, d_down, 1.0)
    r = (c_c - c_up) / safe
    psi_quick = 2.0 * (quick - c_c) / safe
    psi = np.maximum(
        0.0,
        np.minimum.reduce(
            [2.0 * r, (1.0 + 3.0 * r) / 4.0, psi_quick, np.full_like(r, 2.0)]
        ),
    )
    return np.where(defined, c_c + 0.5 * psi * d_down, c_c)


@dataclass(frozen=True)
class _Axis:
    """One advection direction with that axis last, for ``_advective_flux``.

    ``nodes`` are the boundary face, the cell centres and the other boundary
    face along the axis; ``faces`` the interior face coordinates; ``left``
    the index in ``nodes`` of the node on the low side of each interior
    face; ``solid_ext`` the SOLID mask padded with one False on each side
    in the transposed shape [nt, ns+2].
    """

    nodes: np.ndarray
    faces: np.ndarray
    left: np.ndarray
    solid_ext: np.ndarray


class TransportSolver:
    """Advance one particle class one step on the staggered face velocities.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh, uniform or stretched.
    config : SimConfig
        Must carry a ``transport`` section: cfl_number, advection_scheme,
        max_diffusion_iter, diffusion_tol. Supplies the class count.
    physics : ParticlePhysics
        Supplies ``settling_velocity`` and ``diffusion_coeff`` per class.
    boundary : ConcentrationBoundary
        Supplies ``faces_for(size_class)``, the scalar condition at every
        face.

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
        is not shaped for the mesh.
    """

    def __init__(
        self,
        mesh: Mesh,
        config: SimConfig,
        physics: ParticlePhysics,
        boundary: ConcentrationBoundary,
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

        ny, nx = self._p_shape
        self._axis_x = _Axis(
            nodes=np.concatenate(([mesh.x[0]], mesh.xc, [mesh.x[-1]])),
            faces=mesh.x[1:-1],
            left=np.arange(1, nx),
            solid_ext=np.pad(solid, ((0, 0), (1, 1))),
        )
        self._axis_y = _Axis(
            nodes=np.concatenate(([mesh.y[0]], mesh.yc, [mesh.y[-1]])),
            faces=mesh.y[1:-1],
            left=np.arange(1, ny),
            solid_ext=np.ascontiguousarray(np.pad(solid, ((1, 1), (0, 0))).T),
        )

        self._settling = [
            float(physics.settling_velocity(k)) for k in range(self._n_classes)
        ]
        self._diffusion = [
            float(physics.diffusion_coeff(k)) for k in range(self._n_classes)
        ]
        self._faces = [boundary.faces_for(k) for k in range(self._n_classes)]
        for k, faces in enumerate(self._faces):
            self._check_conditions(faces, k)
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
            If ``size_class`` is not an int.
        ValueError
            If ``dt`` is not positive or exceeds ``stable_dt``, a shape does
            not fit the mesh, or a source is negative or sits in a SOLID
            cell.
        """
        k = self._check_class(size_class)
        c = np.array(C_k, dtype=np.float64, order="C", copy=True)
        if c.shape != self._p_shape:
            raise ValueError(f"expected C_k of shape {self._p_shape}, got {c.shape}")
        if not dt > 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        limit = self.stable_dt(faces, k, v_ext)
        if dt > limit:
            raise ValueError(
                f"dt {dt} exceeds stable_dt {limit} for class {k}; the explicit "
                "advection step is a convex combination only below it"
            )
        if sources is not None:
            raise ValueError("sources are not applied by this commit")
        c[self._solid] = 0.0
        budget = self.budget[k]
        mesh = self._mesh
        if budget.initial is None:
            budget.initial = MassBudget.in_domain(c, mesh)

        conditions = self._faces[k]
        u_adv, v_adv = self._advecting_velocities(faces, k, v_ext)
        flux_u = u_adv * self._area_u
        flux_v = v_adv * self._area_v
        adv_u = self._advective_flux(c, flux_u, conditions.inflow_u, self._axis_x)
        adv_v = self._advective_flux(
            c.T, flux_v.T, conditions.inflow_v.T, self._axis_y
        ).T
        divergence = adv_u[:, 1:] - adv_u[:, :-1] + adv_v[1:, :] - adv_v[:-1, :]
        c_star = c - dt * divergence / self._volume
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

        c_new = self._implicit_step(c_star, dt, self._diffusion[k])
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

    def _check_conditions(self, faces: ConcentrationFaces, k: int) -> None:
        for name in ("inflow_u", "deposition_u", "surface_u"):
            if getattr(faces, name).shape != self._u_shape:
                raise ValueError(
                    f"class {k}: {name} must have shape {self._u_shape}, "
                    f"got {getattr(faces, name).shape}"
                )
        for name in ("inflow_v", "deposition_v", "surface_v", "settling_v"):
            if getattr(faces, name).shape != self._v_shape:
                raise ValueError(
                    f"class {k}: {name} must have shape {self._v_shape}, "
                    f"got {getattr(faces, name).shape}"
                )

    # ------------------------------------------------------------------
    # The step's parts
    # ------------------------------------------------------------------

    def _advecting_velocities(
        self, faces: FaceVelocities, k: int, v_ext: FaceVelocities | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The class's advecting face velocities, zero on faces that carry no flux."""
        self._check_pair(faces.u, faces.v, "faces")
        u_adv = np.where(self._live_u, faces.u, 0.0)
        v_adv = np.where(self._live_v, faces.v, 0.0)
        if v_ext is not None:
            self._check_pair(v_ext.u, v_ext.v, "v_ext")
            u_adv = u_adv + np.where(self._inner_u, v_ext.u, 0.0)
            v_adv = v_adv + np.where(self._inner_v, v_ext.v, 0.0)
        return u_adv, v_adv

    def _advective_flux(
        self,
        c: np.ndarray,
        flux: np.ndarray,
        inflow: np.ndarray,
        axis: _Axis,
    ) -> np.ndarray:
        """Volume flux times face concentration on every face along the last axis.

        ``c`` has shape [nt, ns], ``flux`` and ``inflow`` [nt, ns+1]. The two
        boundary nodes hold the carried concentration where the flux enters
        and the adjacent cell value otherwise, so an inflow face reads the
        inlet value at the face and every other boundary is zero gradient.
        """
        ns = c.shape[1]
        low_in = flux[:, 0] > 0.0
        high_in = flux[:, -1] < 0.0
        c_low = np.where(low_in, inflow[:, 0], c[:, 0])
        c_high = np.where(high_in, inflow[:, -1], c[:, -1])
        ext = np.concatenate([c_low[:, None], c, c_high[:, None]], axis=1)
        face = np.empty_like(flux)
        face[:, 0] = c_low
        face[:, -1] = c_high
        if ns >= 2:
            k = axis.left
            positive = flux[:, 1:-1] > 0.0
            c_c = np.where(positive, ext[:, k], ext[:, k + 1])
            if self._upwind:
                face[:, 1:-1] = c_c
            else:
                c_d = np.where(positive, ext[:, k + 1], ext[:, k])
                c_up = np.where(positive, ext[:, k - 1], ext[:, k + 2])
                far_solid = np.where(
                    positive, axis.solid_ext[:, k - 1], axis.solid_ext[:, k + 2]
                )
                quick = quick_face_values(ext, axis.nodes, k, axis.faces, positive)
                # A far node inside an obstacle is read as the upstream value,
                # the zero-gradient rule a domain wall gets.
                c_up = np.where(far_solid, c_c, c_up)
                quick = np.where(far_solid, c_c, quick)
                face[:, 1:-1] = limited_face_values(c_up, c_c, c_d, quick)
        return flux * face

    def _implicit_step(
        self, c_star: np.ndarray, dt: float, diffusivity: float
    ) -> np.ndarray:
        """Backward Euler diffusion by Jacobi; the explicit field when nothing diffuses.

        ``a_P C_P - sum_f G_f C_N = (V / dt) C*``, with ``G_f = D A_f / d_f``
        on the interior faces between non-SOLID cells. Each sweep adds the
        residual over the diagonal, so the stop reads the system's own
        residual: its largest entry below ``diffusion_tol`` times the largest
        right-hand side.
        """
        g_u = diffusivity * self._conductance_u
        g_v = diffusivity * self._conductance_v
        diag_extra = g_u[:, :-1] + g_u[:, 1:] + g_v[:-1, :] + g_v[1:, :]
        if not diag_extra.any():
            self.last_diffusion_sweeps = 0
            self.diffusion_converged = True
            return c_star
        over_dt = self._volume / dt
        a_p = over_dt + diag_extra
        b = over_dt * c_star
        scale = float(np.abs(b).max())
        c = c_star.copy()
        padded = np.pad(c, 1)
        self.diffusion_converged = False
        sweeps = 0
        while True:
            padded[1:-1, 1:-1] = c
            neighbours = (
                g_u[:, 1:] * padded[1:-1, 2:]
                + g_u[:, :-1] * padded[1:-1, :-2]
                + g_v[1:, :] * padded[2:, 1:-1]
                + g_v[:-1, :] * padded[:-2, 1:-1]
            )
            residual = b + neighbours - a_p * c
            residual[self._solid] = 0.0
            if float(np.abs(residual).max()) <= self._tol * scale:
                self.diffusion_converged = True
                break
            if sweeps >= self._max_sweeps:
                logger.warning(
                    "implicit diffusion solve stopped at max_diffusion_iter %d",
                    self._max_sweeps,
                )
                break
            c = c + residual / a_p
            c[self._solid] = 0.0
            sweeps += 1
        self.last_diffusion_sweeps = sweeps
        return c
