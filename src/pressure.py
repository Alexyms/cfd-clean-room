"""Pressure correction on the staggered grid: the p' equation, its CG solve, the correction.

ECR-001 step 5 (REQ-S04, REQ-S08); the solve replaced in ECR-003 step 1
(REQ-S08 as amended 2026-10-06, ADR-013). Given u*, v* and the un-relaxed
momentum diagonals from MomentumPrediction, assemble the pressure correction
equation, solve it by conjugate gradients preconditioned with its diagonal,
correct the face velocities so the corrected field satisfies discrete
continuity, and update p. StaggeredSolver runs the outer loop.

The equation. The corrected velocity is ``u = u* + u'`` with ``u' = -d
(p'_(s+) - p'_(s-))`` at every correctable face, ``d = A_face / a_P``, and
continuity of the corrected field in cell (j, i) gives

    a_P p'_P = a_E p'_E + a_W p'_W + a_N p'_N + a_S p'_S - b,
    a_nb = rho d_face A_face,   a_P = sum(a_nb) + outlet terms,
    b = rho [(u*_e - u*_w) dy_cell[j] + (v*_n - v*_s) dx_cell[i]].

In operator form that is ``A p' = -b`` with ``(A x)_P = a_P x_P - sum(a_nb
x_nb)``, which apply_operator forms. The right-hand side is the discrete
divergence of u* formed directly from the stored face velocities, with no
interpolation anywhere. Summed over a closed domain it telescopes to the
mass flux through the boundary faces, which the staggered layout holds at
zero exactly, so the Neumann system is compatible to rounding. This is the
property the rebuild exists for; the collocated ghost-cell walls leaked and
made the same system unsolvable.

Where the boundary conditions go. A face is correctable when the momentum
predictor gave it a diagonal (``a_p > 0``): the interior faces of FLUID
cells. Walls, velocity inlets and faces of SOLID cells carry a fixed
velocity, so they contribute no coefficient and their u* flux simply
stays in b. That absence is the homogeneous Neumann condition; no wall
pressure condition is written. A pressure outlet fixes ``p' = 0`` at the
outlet face: the adjacent cell gets a coefficient toward the face, with
no neighbour value, and the outlet face velocity is corrected against
``p' = 0`` so the outlet cell also closes. The outlet face has no momentum
diagonal of its own, so its ``d`` uses the diagonal of the nearest
interior face of the same component, the staggered form of the collocated
``face_d`` rule that gives a FLUID-BOUNDARY face the fluid cell's d.

Closed domains. With no outlet the system is singular, with the constants
as its null space, and p' is defined up to a constant. The right-hand side
is projected onto the range before the solve (its mean over the cells with
an equation removed), the stop reads that projected residual, and after
the solve the first FLUID cell is the reference: its value is subtracted
from p' and from p after the update, as the collocated solver did. One
mean and one pin handle one connected component, so the constructor
refuses a closed domain whose cells with an equation form more than one.

The solve. The matrix is symmetric and positive definite on an open domain
and positive semi-definite on a closed one (the report, section 6), so
conjugate gradients from p' = 0, preconditioned by the diagonal a_P, is the
solve: one five-point product and one diagonal scaling per cell, both one
thread per cell on the GPU path of Phase 6, plus three reductions per
iteration. A correction stops at the first of: the residual's 2-norm at
most pressure_rtol times the right-hand side's; the residual's 2-norm at
most RESIDUAL_FLOOR times the flux scale F, where the face arithmetic's
rounding lives; max_pressure_iter iterations, which the correction reports
as reached_cap. The first two are read on the residual CG updates by
recursion, which drifts from the true one, and confirmed on the true
residual formed once more at exit (ADR-013 B). Weighted Jacobi, the
previous solve, needed hundreds of thousands of sweeps per correction on
the product mesh (docs/reports/pressure_solver_ecr003.md, section 7); its
evidence is docs/reports/pressure_correction_step5.md.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import FLUID, SOLID, Mesh
from src.momentum import MomentumPrediction
from src.staggered import p_shape, u_shape, v_shape

# Which solve wrote a saved pressure correction. 1 was the weighted Jacobi
# sweep of ECR-001 step 5 (2026-09-22 to 2026-10-06); 2 is the conjugate
# gradient solve of ECR-003. The scripts that reuse a saved solve store it with
# the solve and solve again when it differs or is missing, so a field written by
# one solver is never served as the other's.
PRESSURE_SOLVER_VERSION = 2

# The rounding floor of the stop, as a fraction of the flux scale F (rho times
# the inflow, or on a closed domain rho times the largest prescribed boundary
# velocity times the longer side, the stopping rule's definition). The residual
# of the p' equation is the corrected faces' mass imbalance, and that identity
# holds to about 6e-14 of the largest face flux (the report, section 2.4), so
# below 1e-13 F the residual is the face arithmetic's rounding and no relative
# level can be asked for. On a converged closed domain the right-hand side
# itself shrinks toward rounding, and this floor, not pressure_rtol, ends the
# correction (section 12.2). It is a property of the arithmetic, not a tolerance
# a case chooses, so it is a constant here rather than a configuration key.
RESIDUAL_FLOOR = 1e-13

# Guard against a zero inflow when the flux scale is formed. StaggeredSolver
# uses the same value for its reference velocity, so the two agree on which
# inflow counts as zero.
ZERO_SCALE = 1e-30


@dataclass(frozen=True)
class PressureCoefficients:
    """The p' equation, shape [ny, nx] each.

    Parameters
    ----------
    a_p : np.ndarray
        Diagonal. The neighbour sum plus any outlet-face term; zero at
        SOLID cells and at any cell with no correctable face.
    a_e, a_w, a_n, a_s : np.ndarray
        Neighbour coefficients ``rho d_face A_face``; zero across a face
        that is not correctable.
    """

    a_p: np.ndarray
    a_e: np.ndarray
    a_w: np.ndarray
    a_n: np.ndarray
    a_s: np.ndarray


@dataclass(frozen=True)
class PressureCorrection:
    """Result of one pressure correction.

    Parameters
    ----------
    u : np.ndarray
        Corrected x-velocity, shape [ny, nx+1]. Correctable interior faces
        and outlet faces move; walls, inlets and SOLID faces are the u*
        values untouched.
    v : np.ndarray
        Corrected y-velocity, shape [ny+1, nx], likewise.
    p : np.ndarray
        Updated pressure ``p + alpha_pressure p'``, shape [ny, nx], pinned
        to zero at the reference cell in a closed domain.
    p_prime : np.ndarray
        The pressure correction, shape [ny, nx], zero at SOLID cells.
    iterations : int
        Conjugate gradient iterations performed; the harness records it.
    reached_cap : bool
        True when the solve stopped at max_pressure_iter rather than at
        its relative level or the floor. The solver counts such
        corrections and, under velocity_step, does not stop on one.
    """

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    p_prime: np.ndarray
    iterations: int
    reached_cap: bool


@dataclass(frozen=True)
class ConjugateGradientResult:
    """What one conjugate gradient solve returned.

    Parameters
    ----------
    x : np.ndarray
        The solution, the shape of the right-hand side.
    iterations : int
        Iterations performed, each one product with the operator.
    reached_cap : bool
        True when the iteration cap ended the solve before the stop held
        on the true residual.
    residual_norm : float
        2-norm of the true residual ``f - A x`` at exit.
    """

    x: np.ndarray
    iterations: int
    reached_cap: bool
    residual_norm: float


def apply_operator(coefficients: PressureCoefficients, x: np.ndarray) -> np.ndarray:
    """Apply the p' operator: ``(A x)_P = a_P x_P - sum(a_nb x_nb)``.

    Parameters
    ----------
    coefficients : PressureCoefficients
        The p' equation, from ``PressureCorrector.coefficients``.
    x : np.ndarray
        A cell field, shape [ny, nx]; not modified.

    Returns
    -------
    np.ndarray
        ``A x``, shape [ny, nx]. Zero at a cell with no equation, since
        every coefficient there is zero, and a cell with an equation never
        reads such a neighbour, since the coefficient across that face is
        zero by construction.

    Notes
    -----
    The edge coefficients ``a_w[:, 0]``, ``a_e[:, -1]``, ``a_s[0, :]`` and
    ``a_n[-1, :]`` are zero by construction, so the four shifted products
    below read every neighbour that exists and nothing outside the grid.
    The order of the five terms is fixed: it is the order the probe of
    docs/reports/pressure_solver_ecr003.md used, so the iteration counts
    measured there are the counts this module produces.
    """
    c = coefficients
    y = c.a_p * x
    y[:, :-1] -= c.a_e[:, :-1] * x[:, 1:]
    y[:, 1:] -= c.a_w[:, 1:] * x[:, :-1]
    y[:-1, :] -= c.a_n[:-1, :] * x[1:, :]
    y[1:, :] -= c.a_s[1:, :] * x[:-1, :]
    return y


def conjugate_gradient(
    apply: Callable[[np.ndarray], np.ndarray],
    inverse_diagonal: np.ndarray,
    f: np.ndarray,
    rtol: float,
    floor: float,
    max_iter: int,
) -> ConjugateGradientResult:
    """Preconditioned conjugate gradients from zero on ``A x = f``.

    Parameters
    ----------
    apply : Callable[[np.ndarray], np.ndarray]
        Forms ``A x`` for a field of f's shape; A symmetric positive
        semi-definite, with f in its range.
    inverse_diagonal : np.ndarray
        The preconditioner ``M^-1``, here ``1 / a_P`` where a_P > 0 and zero
        elsewhere, f's shape.
    f : np.ndarray
        Right-hand side; zero where there is no equation.
    rtol : float
        Relative level: the stop is ``||r||_2 <= rtol ||f||_2``.
    floor : float
        Absolute level below which the residual is rounding: the stop is
        also met when ``||r||_2 <= floor``. Zero disables it.
    max_iter : int
        Iteration cap.

    Returns
    -------
    ConjugateGradientResult
        x, the iterations, whether the cap ended the solve, and the true
        residual's 2-norm at exit. A zero right-hand side returns zero at
        once, with no iteration.

    Notes
    -----
    Hestenes and Stiefel's iteration with the standard recursion for the
    residual, ``r <- r - alpha q``. The stop is tested on that recursive
    residual every iteration; when it holds, one more product forms the
    true residual ``f - A x`` and the solve ends only if that meets the
    stop too. Otherwise the iteration restarts from the true residual and
    goes on, still counting toward the cap, so a recursion that has
    drifted cannot end a correction early (ADR-013 B). The three
    reductions per iteration are ``vdot`` in NumPy's order; a different
    order moved a correction's faces by at most 1.6e-12 m/s on the
    product mesh (the report, section 12.2).
    """
    x = np.zeros_like(f)
    r = f.copy()
    f_norm = float(np.sqrt(np.vdot(f, f)))
    stop = max(rtol * f_norm, floor)
    if f_norm == 0.0:
        return ConjugateGradientResult(
            x=x, iterations=0, reached_cap=False, residual_norm=0.0
        )
    z = inverse_diagonal * r
    p = z.copy()
    rz = float(np.vdot(r, z))
    k = 0
    while k < max_iter:
        q = apply(p)
        alpha = rz / float(np.vdot(p, q))
        x += alpha * p
        r -= alpha * q
        k += 1
        r_norm = float(np.sqrt(np.vdot(r, r)))
        if r_norm <= stop:
            # The recursion drifts from the truth; one more product checks it,
            # and a solve whose true residual still misses the stop restarts
            # from that residual rather than returning it.
            r = f - apply(x)
            r_norm = float(np.sqrt(np.vdot(r, r)))
            if r_norm <= stop:
                return ConjugateGradientResult(
                    x=x, iterations=k, reached_cap=False, residual_norm=r_norm
                )
            z = inverse_diagonal * r
            p = z.copy()
            rz = float(np.vdot(r, z))
            continue
        z = inverse_diagonal * r
        rz_new = float(np.vdot(r, z))
        p *= rz_new / rz
        p += z
        rz = rz_new
    r = f - apply(x)
    return ConjugateGradientResult(
        x=x, iterations=k, reached_cap=True, residual_norm=float(np.sqrt(np.vdot(r, r)))
    )


class PressureCorrector:
    """Assemble and solve the p' equation and correct the staggered velocities.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    config : SimConfig
        Supplies rho, alpha_pressure, max_pressure_iter and pressure_rtol.
    boundary : StaggeredBoundary
        Supplies the pressure outlets and the flux scale's inputs.

    Attributes
    ----------
    needs_pin : bool
        True when no edge face is a pressure outlet, so p' is pinned.
    pin_cell : tuple[int, int]
        (j, i) of the reference cell, the first cell typed FLUID, selected
        as the collocated solver selected its own; meaningful only when
        ``needs_pin``.
    flux_scale : float
        F, kg/s per unit depth: rho times the total inflow, or on a closed
        domain rho times the largest prescribed boundary velocity times
        the longer side. The stopping rule's definition, formed here so
        the floor is the same under either rule; StaggeredSolver reads it.

    Raises
    ------
    ValueError
        On a closed domain whose cells with a pressure equation form more
        than one connected component: the projection removes one mean and
        the pin fixes one cell, so a second component would be solved to
        the wrong level (ADR-013 D).
    """

    def __init__(
        self, mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary
    ) -> None:
        self._mesh = mesh
        self._rho = config.rho
        self._alpha_p = config.alpha_pressure
        self._max_iter = config.max_pressure_iter
        self._rtol = config.pressure_rtol
        self._u_shape = u_shape(mesh)
        self._v_shape = v_shape(mesh)
        self._p_shape = p_shape(mesh)
        self._fluid = mesh.cell_type == FLUID
        self._solid = mesh.cell_type == SOLID

        outlets = boundary.pressure_outlets()
        self._out_left = outlets["left"].is_outlet
        self._out_right = outlets["right"].is_outlet
        self._out_bottom = outlets["bottom"].is_outlet
        self._out_top = outlets["top"].is_outlet
        self.needs_pin: bool = not boundary.has_pressure_outlet()
        self.pin_cell: tuple[int, int] = (0, 0)
        if self.needs_pin:
            fluid_idx = np.argwhere(self._fluid)
            if fluid_idx.size > 0:
                self.pin_cell = (int(fluid_idx[0, 0]), int(fluid_idx[0, 1]))
            self._check_one_component()
        self.flux_scale: float = self._flux_scale_of(boundary)

    def _flux_scale_of(self, boundary: StaggeredBoundary) -> float:
        """F: rho times the inflow, or closed, rho times the velocity scale times the longer side."""
        mesh = self._mesh
        inflow = boundary.get_total_inlet_flux()
        if inflow <= ZERO_SCALE:
            longer = max(float(mesh.x[-1]), float(mesh.y[-1]))
            inflow = boundary.get_max_boundary_velocity() * longer
        return self._rho * inflow

    def _check_one_component(self) -> None:
        """Raise unless the cells with a pressure equation are one connected component.

        On a closed domain a cell has an equation when one of its faces is
        correctable, which means a face shared with another non-SOLID
        cell; a sealed single cell has none and is not counted. Components
        are grown by flood fill through 4-neighbours, a few array passes
        per cell of diameter, once at construction.
        """
        open_cell = ~self._solid
        has_equation = open_cell & _dilate(open_cell)
        remaining = has_equation.copy()
        components = 0
        while remaining.any():
            components += 1
            reached = np.zeros_like(remaining)
            reached[tuple(np.argwhere(remaining)[0])] = True
            while True:
                grown = (reached | _dilate(reached)) & remaining
                if np.array_equal(grown, reached):
                    break
                reached = grown
            remaining &= ~reached
        if components > 1:
            raise ValueError(
                f"closed domain: the cells with a pressure equation form {components} "
                "connected components; the pressure correction projects one mean and "
                "pins one cell, so it solves one component only (ADR-013 D)"
            )

    # ------------------------------------------------------------------
    # The right-hand side
    # ------------------------------------------------------------------

    def mass_imbalance(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Discrete mass imbalance of every cell, directly from the face velocities.

        Parameters
        ----------
        u : np.ndarray
            x-velocity on vertical faces, shape [ny, nx+1].
        v : np.ndarray
            y-velocity on horizontal faces, shape [ny+1, nx].

        Returns
        -------
        np.ndarray
            ``rho [(u_e - u_w) dy_cell + (v_n - v_s) dx_cell]``, shape
            [ny, nx], zero at SOLID cells. Its sum over a closed domain is
            the net mass flux through the boundary faces.
        """
        self._check_shapes(u, v)
        mesh = self._mesh
        b = self._rho * (
            (u[:, 1:] - u[:, :-1]) * mesh.dy_cell[:, None]
            + (v[1:, :] - v[:-1, :]) * mesh.dx_cell[None, :]
        )
        b[self._solid] = 0.0
        return b

    # ------------------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------------------

    def _face_d(
        self, a_p_u: np.ndarray, a_p_v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """d = A_face / a_P on every face; zero where the face is not correctable.

        Outlet faces take the diagonal of the nearest interior face of the
        same component in their row or column.
        """
        mesh = self._mesh
        d_u = np.zeros(self._u_shape, dtype=np.float64)
        d_v = np.zeros(self._v_shape, dtype=np.float64)
        ok_u = a_p_u > 0.0
        ok_v = a_p_v > 0.0
        d_u[ok_u] = (mesh.dy_cell[:, None] * np.ones(self._u_shape))[ok_u] / a_p_u[ok_u]
        d_v[ok_v] = (mesh.dx_cell[None, :] * np.ones(self._v_shape))[ok_v] / a_p_v[ok_v]

        def borrow(
            d: np.ndarray,
            a_p: np.ndarray,
            edge: np.ndarray,
            own: np.ndarray,
            near: np.ndarray,
            length: np.ndarray,
        ) -> None:
            usable = edge & (a_p[near] > 0.0)
            d[own][usable] = length[usable] / a_p[near][usable]

        left, right = (slice(None), 0), (slice(None), -1)
        bottom, top = (0, slice(None)), (-1, slice(None))
        borrow(d_u, a_p_u, self._out_left, left, (slice(None), 1), mesh.dy_cell)
        borrow(d_u, a_p_u, self._out_right, right, (slice(None), -2), mesh.dy_cell)
        borrow(d_v, a_p_v, self._out_bottom, bottom, (1, slice(None)), mesh.dx_cell)
        borrow(d_v, a_p_v, self._out_top, top, (-2, slice(None)), mesh.dx_cell)
        return d_u, d_v

    def coefficients(
        self, a_p_u: np.ndarray, a_p_v: np.ndarray
    ) -> PressureCoefficients:
        """Assemble the p' equation from the momentum diagonals.

        Parameters
        ----------
        a_p_u : np.ndarray
            Un-relaxed u diagonals, shape [ny, nx+1], positive at unknowns.
        a_p_v : np.ndarray
            Un-relaxed v diagonals, shape [ny+1, nx].

        Returns
        -------
        PressureCoefficients
            Neighbour coefficients ``rho d_face A_face`` and the diagonal.
            The diagonal equals the neighbour sum except in cells with an
            outlet face, where the outlet term is added with no neighbour,
            which is the Dirichlet ``p' = 0`` at that face.
        """
        if a_p_u.shape != self._u_shape or a_p_v.shape != self._v_shape:
            raise ValueError(
                f"expected diagonals of shapes {self._u_shape} and {self._v_shape}, "
                f"got {a_p_u.shape} and {a_p_v.shape}"
            )
        mesh = self._mesh
        rho = self._rho
        d_u, d_v = self._face_d(a_p_u, a_p_v)
        coef_u = rho * d_u * mesh.dy_cell[:, None]
        coef_v = rho * d_v * mesh.dx_cell[None, :]

        a_e = coef_u[:, 1:].copy()
        a_w = coef_u[:, :-1].copy()
        a_n = coef_v[1:, :].copy()
        a_s = coef_v[:-1, :].copy()
        a_p = a_e + a_w + a_n + a_s

        # A domain-edge face with a coefficient is an outlet: it stays in the
        # diagonal but has no neighbour cell, so p' = 0 there.
        a_w[:, 0] = 0.0
        a_e[:, -1] = 0.0
        a_s[0, :] = 0.0
        a_n[-1, :] = 0.0
        for arr in (a_p, a_e, a_w, a_n, a_s):
            arr[self._solid] = 0.0
        return PressureCoefficients(a_p=a_p, a_e=a_e, a_w=a_w, a_n=a_n, a_s=a_s)

    # ------------------------------------------------------------------
    # Solve and correct
    # ------------------------------------------------------------------

    def correct(
        self, prediction: MomentumPrediction, p: np.ndarray
    ) -> PressureCorrection:
        """Solve for p', correct the velocities and update the pressure.

        Parameters
        ----------
        prediction : MomentumPrediction
            u*, v* and the un-relaxed diagonals from the momentum predictor.
        p : np.ndarray
            Current pressure, shape [ny, nx]; not modified.

        Returns
        -------
        PressureCorrection
            Corrected u and v, updated p, p', the iteration count and
            whether the cap ended the solve.

        Notes
        -----
        The solve is ``conjugate_gradient`` on ``A p' = -b`` from p' = 0,
        preconditioned by a_P, to a residual 2-norm at most pressure_rtol
        times the right-hand side's or at most RESIDUAL_FLOOR times
        flux_scale, confirmed on the true residual, or to max_pressure_iter
        iterations. The residual ``b + A p'`` is the mass imbalance the
        corrected faces leave in each cell, so the relative level is the
        fraction of u*'s imbalance a correction leaves (ADR-013 B). In a
        closed domain the right-hand side is projected onto the range
        first, and the reference cell's value is subtracted from p' after
        the solve and from p after the update, matching the collocated
        solver's pin at the end of each outer iteration.
        """
        u_star, v_star = prediction.u_star, prediction.v_star
        self._check_shapes(u_star, v_star)
        if p.shape != self._p_shape:
            raise ValueError(f"expected p of shape {self._p_shape}, got {p.shape}")

        c = self.coefficients(prediction.a_p_u, prediction.a_p_v)
        b = self.mass_imbalance(u_star, v_star)
        active = c.a_p > 0.0
        pin_j, pin_i = self.pin_cell

        f = -b
        if self.needs_pin:
            f[active] -= f[active].mean()
        f[~active] = 0.0
        inverse_diagonal = np.where(active, 1.0 / np.where(active, c.a_p, 1.0), 0.0)
        solved = conjugate_gradient(
            lambda x: apply_operator(c, x),
            inverse_diagonal,
            f,
            self._rtol,
            RESIDUAL_FLOOR * self.flux_scale,
            self._max_iter,
        )
        p_prime = solved.x
        if self.needs_pin:
            p_prime[active] -= p_prime[pin_j, pin_i]

        d_u, d_v = self._face_d(prediction.a_p_u, prediction.a_p_v)
        padded = np.zeros(
            (self._p_shape[0] + 2, self._p_shape[1] + 2), dtype=np.float64
        )
        padded[1:-1, 1:-1] = p_prime
        u = u_star - d_u * (padded[1:-1, 1:] - padded[1:-1, :-1])
        v = v_star - d_v * (padded[1:, 1:-1] - padded[:-1, 1:-1])

        p_next = p.copy()
        p_next[active] += self._alpha_p * p_prime[active]
        if self.needs_pin:
            p_next[active] -= p_next[pin_j, pin_i]
        return PressureCorrection(
            u=np.ascontiguousarray(u),
            v=np.ascontiguousarray(v),
            p=p_next,
            p_prime=p_prime,
            iterations=solved.iterations,
            reached_cap=solved.reached_cap,
        )

    def _check_shapes(self, u: np.ndarray, v: np.ndarray) -> None:
        if u.shape != self._u_shape or v.shape != self._v_shape:
            raise ValueError(
                f"expected staggered shapes u {self._u_shape} and v {self._v_shape}, "
                f"got u {u.shape} and v {v.shape}"
            )


def _dilate(mask: np.ndarray) -> np.ndarray:
    """Cells with a 4-neighbour in the mask."""
    out = np.zeros_like(mask)
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    return out
