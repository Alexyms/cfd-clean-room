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
pseudo-time step of ADR-012 C. A cell may be held at its value of ``C*``, as
k-epsilon holds eps in the wall cells (ADR-012 C): its row reads
``C_P = C*_P`` exactly, the limit of a dominant diagonal, and its neighbours
see the held value.

Two methods solve it (prompt 51). Jacobi, the transport solver's: every
iterate of a non-negative ``C*`` is non-negative, which a concentration
spanning many decades down to zero needs. Diagonal-preconditioned conjugate
gradients, the k-epsilon model's: the matrix is symmetric positive definite,
and CG reaches the same residual in roughly the square root of Jacobi's
sweeps, but a truncated iterate carries no sign guarantee. Both stop on the
same max-norm residual, so ``tol`` means the same under either. A CG solve
that reaches its cap, or returns a value at or below zero in a cell it
solves for, falls back to Jacobi from ``C*`` (Alex, 2026-10-12): the result
is then the Jacobi path's, bit for bit, and keeps its sign guarantee. A
system conditioned beyond double precision is where CG needs it: the
prompt 51 probe found a truncated CG iterate of k at -3.5e16 where the
condition number was 2.4e16 (results/builder51/).
"""

from dataclasses import dataclass

import numpy as np
from threadpoolctl import ThreadpoolController

from src.mesh import SOLID, Mesh
from src.momentum import quick_face_values

# The two methods of ``implicit_step``.
JACOBI = "jacobi"
CONJUGATE_GRADIENT = "cg"
IMPLICIT_METHODS = (JACOBI, CONJUGATE_GRADIENT)

# BLAS threads the conjugate gradient solve may use. Its two dot products per
# iteration run over the cell field, and above about 10,000 elements OpenBLAS
# splits each across its pool at a fixed cost: on the 216x72 Annex 20 mesh a
# dot product took 272 us on 24 threads against 4 us on one (prompt 51). It
# is the measurement behind pressure.py's PRESSURE_BLAS_THREADS, on vectors of
# the same length (docs/reports/blas_threads.md), kept here rather than
# imported so the shared scheme does not depend on the pressure module.
SCALAR_BLAS_THREADS = 1

# Discovering the loaded BLAS libraries is the expensive part of threadpoolctl,
# so it happens once at import and each solve only enters and leaves the limit.
_BLAS_CONTROLLER = ThreadpoolController()


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
    method : str
        The method asked for, ``JACOBI`` or ``CONJUGATE_GRADIENT``.
    iterations : int
        The method's iterations: Jacobi sweeps, or CG iterations (one product
        with the matrix each; the products that confirm the stop on the true
        residual are not counted).
    converged : bool
        Whether the returned field's residual met the tolerance within the
        cap: the fallback's when there was one.
    fallbacks : int
        1 when a CG solve fell back to Jacobi from ``C*``, else 0.
    fallback_sweeps : int
        The fallback's Jacobi sweeps; 0 without one.
    """

    field: np.ndarray
    method: str
    iterations: int
    converged: bool
    fallbacks: int
    fallback_sweeps: int


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
    max_iter: int,
    held: np.ndarray | None = None,
    method: str = JACOBI,
) -> ImplicitResult:
    """Backward Euler diffusion with a cell sink, by Jacobi or conjugate gradients.

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
    max_iter : int
        The iteration cap: Jacobi sweeps, or CG iterations.
    held : np.ndarray, optional
        Bool [ny, nx], non-SOLID cells held at their value of ``c_star``:
        their residual is zero, so they keep it, and their neighbours read
        it. None holds nothing, and the arithmetic is the unheld path's.
    method : str, optional
        ``JACOBI`` (the default) or ``CONJUGATE_GRADIENT``, for a field
        positive in every cell it solves for, as k and eps are: CG falls back
        to Jacobi when it reaches the cap or returns a value at or below zero
        in such a cell.

    Returns
    -------
    ImplicitResult
        The field, the method, its iterations, whether the tolerance was met
        and the fallback, if any.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``IMPLICIT_METHODS``.

    Notes
    -----
    ``a_P C_P - sum_f G_f C_N = (V / dt) C*`` with ``a_P = V / dt + sum_f
    G_f + D_P``. The stop is the same under both methods: the largest
    absolute residual over the non-SOLID, unheld cells, at most ``tol``
    times the largest ``|b|`` over all cells. Each Jacobi sweep adds the
    residual over the diagonal, so the stop reads the system's own residual
    every sweep. The diagonal's terms are summed in a fixed order, west,
    east, south, north, then the sink: floating-point sums depend on order,
    and the transport gate's bits depend on this one.

    Under conjugate gradients the SOLID and held cells are not unknowns:
    their values stay fixed and their couplings act on the right-hand side,
    which is the system Jacobi solves. The iteration starts from ``C*``, as
    Jacobi does; ``_conjugate_gradient`` has the loop and its stop. When CG
    reaches the cap, or leaves a value at or below zero in a cell it solves
    for, its iterate is discarded and Jacobi solves from ``C*`` with the same
    tolerance and cap. Not from the CG iterate: a non-negative start is what
    makes every Jacobi iterate non-negative.
    """
    if method not in IMPLICIT_METHODS:
        raise ValueError(f"method must be one of {IMPLICIT_METHODS}, got {method!r}")
    diag_extra = g_u[:, :-1] + g_u[:, 1:] + g_v[:-1, :] + g_v[1:, :] + diagonal
    if not diag_extra.any():
        return ImplicitResult(
            field=c_star,
            method=method,
            iterations=0,
            converged=True,
            fallbacks=0,
            fallback_sweeps=0,
        )
    over_dt = volume / dt
    a_p = over_dt + diag_extra
    b = over_dt * c_star
    stop = tol * float(np.abs(b).max())
    if method == JACOBI:
        c, sweeps, converged = _jacobi(
            c_star, a_p, b, g_u, g_v, solid, held, stop, max_iter
        )
        return ImplicitResult(
            field=c,
            method=method,
            iterations=sweeps,
            converged=converged,
            fallbacks=0,
            fallback_sweeps=0,
        )
    fixed = solid if held is None else solid | held
    c = c_star.copy()
    c[solid] = 0.0
    with _BLAS_CONTROLLER.limit(limits=SCALAR_BLAS_THREADS, user_api="blas"):
        c, iterations, converged = _conjugate_gradient(
            c, a_p, b, g_u, g_v, fixed, stop, max_iter
        )
    if converged and bool(np.all(c[~fixed] > 0.0)):
        return ImplicitResult(
            field=c,
            method=method,
            iterations=iterations,
            converged=True,
            fallbacks=0,
            fallback_sweeps=0,
        )
    c, sweeps, converged = _jacobi(
        c_star, a_p, b, g_u, g_v, solid, held, stop, max_iter
    )
    return ImplicitResult(
        field=c,
        method=method,
        iterations=iterations,
        converged=converged,
        fallbacks=1,
        fallback_sweeps=sweeps,
    )


def _jacobi(
    c_star: np.ndarray,
    a_p: np.ndarray,
    b: np.ndarray,
    g_u: np.ndarray,
    g_v: np.ndarray,
    solid: np.ndarray,
    held: np.ndarray | None,
    stop: float,
    max_iter: int,
) -> tuple[np.ndarray, int, bool]:
    """Jacobi sweeps from ``C*`` on ``implicit_step``'s system.

    Returns the field, the sweeps and whether the largest residual over the
    non-SOLID, unheld cells met ``stop`` within ``max_iter`` sweeps. Each
    sweep is a non-negative combination of the last iterate and ``b``, so a
    non-negative ``C*`` gives non-negative iterates throughout.
    """
    c = c_star.copy()
    padded = np.pad(c, 1)
    converged = False
    sweeps = 0
    while True:
        padded[1:-1, 1:-1] = c
        residual = b + _neighbour_sum(padded, g_u, g_v) - a_p * c
        residual[solid] = 0.0
        if held is not None:
            residual[held] = 0.0
        if float(np.abs(residual).max()) <= stop:
            converged = True
            break
        if sweeps >= max_iter:
            break
        c = c + residual / a_p
        c[solid] = 0.0
        sweeps += 1
    return c, sweeps, converged


def _neighbour_sum(padded: np.ndarray, g_u: np.ndarray, g_v: np.ndarray) -> np.ndarray:
    """``sum_f G_f C_N`` per cell, from a field padded with one zero all round.

    The four terms are summed east, west, north, south; the Jacobi path's bits
    depend on this order as they do on the diagonal's.
    """
    return (
        g_u[:, 1:] * padded[1:-1, 2:]
        + g_u[:, :-1] * padded[1:-1, :-2]
        + g_v[1:, :] * padded[2:, 1:-1]
        + g_v[:-1, :] * padded[:-2, 1:-1]
    )


def _conjugate_gradient(
    c: np.ndarray,
    a_p: np.ndarray,
    b: np.ndarray,
    g_u: np.ndarray,
    g_v: np.ndarray,
    fixed: np.ndarray,
    stop: float,
    max_iter: int,
) -> tuple[np.ndarray, int, bool]:
    """Diagonal-preconditioned conjugate gradients on ``implicit_step``'s system.

    Parameters
    ----------
    c : np.ndarray
        The start, [ny, nx], with the fixed cells at their values; updated in
        place and returned.
    a_p, b : np.ndarray
        The diagonal and the right-hand side, [ny, nx].
    g_u, g_v : np.ndarray
        The face conductances, as ``implicit_step`` takes them.
    fixed : np.ndarray
        Bool [ny, nx], the cells that are not unknowns (SOLID and held).
    stop : float
        The largest absolute residual over the unknowns at which the solve
        ends.
    max_iter : int
        The iteration cap.

    Returns
    -------
    tuple[np.ndarray, int, bool]
        The field, the iterations and whether the stop held.

    Notes
    -----
    Hestenes and Stiefel's iteration with the residual recursion
    ``r <- r - alpha q``. The direction is zero at every fixed cell, so a
    fixed cell's value never changes, not even by rounding, and the product
    ``q = A p`` is the matrix restricted to the unknowns. The recursion
    drifts from the true residual, so when it meets the stop, or at the cap,
    one more product forms the true residual ``b + sum_f G_f C_N - a_P C``;
    the solve ends converged only if that meets the stop, and otherwise
    restarts from it, as pressure.py's loop does (ADR-013 B).

    That loop is not shared. It starts from zero and stops on a 2-norm with
    an absolute floor; making its start and its stop parameters would change
    the pressure module, which this solve leaves alone (prompt 51).
    """
    padded = np.pad(c, 1)
    inverse = np.divide(1.0, a_p, out=np.zeros_like(a_p), where=~fixed)

    def true_residual() -> np.ndarray:
        padded[1:-1, 1:-1] = c
        r = b + _neighbour_sum(padded, g_u, g_v) - a_p * c
        r[fixed] = 0.0
        return r

    def product(p: np.ndarray) -> np.ndarray:
        padded[1:-1, 1:-1] = p
        q = a_p * p - _neighbour_sum(padded, g_u, g_v)
        q[fixed] = 0.0
        return q

    r = true_residual()
    if float(np.abs(r).max()) <= stop:
        return c, 0, True
    z = inverse * r
    p = z.copy()
    rz = float(np.vdot(r, z))
    iterations = 0
    while iterations < max_iter:
        q = product(p)
        alpha = rz / float(np.vdot(p, q))
        c += alpha * p
        r -= alpha * q
        iterations += 1
        if float(np.abs(r).max()) <= stop or iterations == max_iter:
            r = true_residual()
            if float(np.abs(r).max()) <= stop:
                return c, iterations, True
            if iterations == max_iter:
                break
            z = inverse * r
            p = z.copy()
            rz = float(np.vdot(r, z))
            continue
        z = inverse * r
        rz_new = float(np.vdot(r, z))
        p *= rz_new / rz
        p += z
        rz = rz_new
    return c, iterations, False
