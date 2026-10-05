"""The cell-centred scalar scheme shared by transport and turbulence (ADR-011 B, C; ADR-012 C).

A scalar at the cell centres, advected by the staggered face velocities and
diffused implicitly, is a particle concentration in ``solver_transport`` and
the turbulence quantities k and eps in ``turbulence``. Both take their
scheme from this module, so the two cannot drift apart. It lives here, not
in ``solver_transport``, because turbulence is part of the flow layer and the
particle solver depends on the flow layer, not the reverse (prompt 35, a
departure from ADR-012 I, which named import from the transport solver).

The face value (ADR-011 B). QUICK's quadratic through the upstream,
downstream and next-upstream nodes, evaluated by ``quick_face_values`` with
Lagrange weights from the node positions, clamped by the UMIST limiter
``psi(r) = max(0, min(2r, (1 + 3r) / 4, psi_quick, 2))`` with
``r = (C_C - C_U) / (C_D - C_C)``. Where the downstream difference is zero
the face value is ``C_C``, which keeps the scheme exact on a uniform field.
With ``upwind`` the face value is ``C_C`` everywhere.

Boundaries (ADR-011 E). At a domain face the sign of the flux decides:
inward, the face carries the caller's inflow value, placed at the face as
Leonard's boundary node; outward, the upwind cell's value. A far-upstream
node inside a SOLID cell is read as the upstream cell's value. Which faces
carry a flux at all is the caller's: it passes zero flux on a wall face, a
face beside a SOLID cell, or anything it holds shut.

The implicit solve (ADR-011 C, ADR-012 C step 3). Backward Euler for
diffusion and a non-negative cell sink together:
``(V / dt + sum_f G_f + D_P) C_P - sum_f G_f C_N = (V / dt) C*``. Its
matrix has a positive diagonal and non-positive off-diagonals and is
strictly diagonally dominant, so a non-negative ``C*`` gives a non-negative
solution at any step. ``dt`` may be one number or one per cell, the local
pseudo-time step of ADR-012 C.
"""

from dataclasses import dataclass

import numpy as np

from src.mesh import SOLID, Mesh
from src.momentum import quick_face_values


@dataclass(frozen=True)
class Axis:
    """One advection direction with that axis last, for ``advective_flux``.

    Parameters
    ----------
    nodes : np.ndarray
        The low boundary face, the cell centres and the high boundary face
        along the axis, shape [ns+2].
    faces : np.ndarray
        The interior face coordinates, shape [ns-1].
    left : np.ndarray
        The index in ``nodes`` of the node on the low side of each interior
        face, shape [ns-1].
    solid_ext : np.ndarray
        The SOLID mask padded with one False on each side, in the axis-last
        shape [nt, ns+2].
    """

    nodes: np.ndarray
    faces: np.ndarray
    left: np.ndarray
    solid_ext: np.ndarray


@dataclass(frozen=True)
class ImplicitResult:
    """What one implicit solve returns.

    Parameters
    ----------
    field : np.ndarray
        The solution, [ny, nx]; ``C*`` itself when nothing diffuses and no
        cell has a sink.
    sweeps : int
        Jacobi sweeps taken.
    converged : bool
        Whether the residual met the tolerance within the sweep cap.
    """

    field: np.ndarray
    sweeps: int
    converged: bool


def mesh_axes(mesh: Mesh) -> tuple[Axis, Axis]:
    """The two advection axes of a mesh.

    Parameters
    ----------
    mesh : Mesh
        Uniform or stretched.

    Returns
    -------
    tuple[Axis, Axis]
        The x axis, for fields in their own [ny, nx] layout, and the y axis,
        for fields transposed to [nx, ny].
    """
    solid = mesh.cell_type == SOLID
    ny, nx = solid.shape
    axis_x = Axis(
        nodes=np.concatenate(([mesh.x[0]], mesh.xc, [mesh.x[-1]])),
        faces=mesh.x[1:-1],
        left=np.arange(1, nx),
        solid_ext=np.pad(solid, ((0, 0), (1, 1))),
    )
    axis_y = Axis(
        nodes=np.concatenate(([mesh.y[0]], mesh.yc, [mesh.y[-1]])),
        faces=mesh.y[1:-1],
        left=np.arange(1, ny),
        solid_ext=np.ascontiguousarray(np.pad(solid, ((1, 1), (0, 0))).T),
    )
    return axis_x, axis_y


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


def advective_flux(
    c: np.ndarray,
    flux: np.ndarray,
    inflow: np.ndarray,
    axis: Axis,
    upwind: bool,
) -> np.ndarray:
    """Volume flux times face value on every face along the last axis.

    Parameters
    ----------
    c : np.ndarray
        The cell field with the axis last, [nt, ns].
    flux : np.ndarray
        Volume flux per face, [nt, ns+1], positive toward increasing
        coordinate; zero on every face that carries nothing.
    inflow : np.ndarray
        The value carried in where the flux at a domain face points inward,
        [nt, ns+1]; only the two end columns are read.
    axis : Axis
        The axis, from ``mesh_axes``.
    upwind : bool
        First-order upwind instead of the limited QUICK value.

    Returns
    -------
    np.ndarray
        ``flux`` times the face value, [nt, ns+1]. The two boundary nodes
        hold the inflow value where the flux enters and the adjacent cell
        value otherwise, so an inflow face reads the inlet value at the face
        and every other boundary is zero gradient.
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
        if upwind:
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


def implicit_step(
    c_star: np.ndarray,
    volume: np.ndarray,
    dt: float | np.ndarray,
    g_u: np.ndarray,
    g_v: np.ndarray,
    diagonal: np.ndarray,
    solid: np.ndarray,
    tol: float,
    max_sweeps: int,
) -> ImplicitResult:
    """Backward Euler diffusion with a cell sink, solved by Jacobi.

    Parameters
    ----------
    c_star : np.ndarray
        The explicit field, [ny, nx], zero in SOLID cells.
    volume : np.ndarray
        Cell volumes per metre of depth, [ny, nx].
    dt : float or np.ndarray
        The step: one number, or one per cell [ny, nx], positive and finite
        in every non-SOLID cell.
    g_u, g_v : np.ndarray
        Diffusive conductance ``Gamma_f A_f / d_f`` per face, [ny, nx+1] and
        [ny+1, nx], zero on every face nothing diffuses across.
    diagonal : np.ndarray
        The non-negative sink per cell in the diagonal, [ny, nx], in the
        units of ``V / dt``: ``v_d A`` of the depositing faces for a
        concentration, ``V eps / k`` for k.
    solid : np.ndarray
        The SOLID mask, [ny, nx]; those cells are held at zero.
    tol : float
        Stop when the largest residual is at most ``tol`` times the largest
        right-hand side.
    max_sweeps : int
        The sweep cap.

    Returns
    -------
    ImplicitResult
        The field, the sweeps and whether the tolerance was met.

    Notes
    -----
    ``a_P C_P - sum_f G_f C_N = (V / dt) C*`` with ``a_P = V / dt + sum_f
    G_f + D_P``. Each sweep adds the residual over the diagonal, so the stop
    reads the system's own residual. The diagonal's terms are summed in a
    fixed order, west, east, south, north, then the sink: floating-point
    sums depend on order, and the transport gate's bits depend on this one.
    """
    diag_extra = g_u[:, :-1] + g_u[:, 1:] + g_v[:-1, :] + g_v[1:, :] + diagonal
    if not diag_extra.any():
        return ImplicitResult(field=c_star, sweeps=0, converged=True)
    over_dt = volume / dt
    a_p = over_dt + diag_extra
    b = over_dt * c_star
    scale = float(np.abs(b).max())
    c = c_star.copy()
    padded = np.pad(c, 1)
    converged = False
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
        residual[solid] = 0.0
        if float(np.abs(residual).max()) <= tol * scale:
            converged = True
            break
        if sweeps >= max_sweeps:
            break
        c = c + residual / a_p
        c[solid] = 0.0
        sweeps += 1
    return ImplicitResult(field=c, sweeps=sweeps, converged=converged)
