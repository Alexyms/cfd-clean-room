"""Pressure correction on the staggered grid: the p' equation, its Jacobi solve, the correction.

ECR-001 step 5 (REQ-S04, REQ-S08). Given u*, v* and the un-relaxed momentum
diagonals from MomentumPrediction, assemble the pressure correction
equation, solve it with Jacobi iteration, correct the face velocities so
the corrected field satisfies discrete continuity, and update p. Nothing
here integrates into solve_steady; step 6 does that.

The equation. The corrected velocity is ``u = u* + u'`` with ``u' = -d
(p'_(s+) - p'_(s-))`` at every correctable face, ``d = A_face / a_P``, and
continuity of the corrected field in cell (j, i) gives

    a_P p'_P = a_E p'_E + a_W p'_W + a_N p'_N + a_S p'_S - b,
    a_nb = rho d_face A_face,   a_P = sum(a_nb) + outlet terms,
    b = rho [(u*_e - u*_w) dy_cell[j] + (v*_n - v*_s) dx_cell[i]].

The right-hand side is the discrete divergence of u* formed directly from
the stored face velocities, with no interpolation anywhere. Summed over a
closed domain it telescopes to the mass flux through the boundary faces,
which the staggered layout holds at zero exactly, so the Neumann system is
compatible to rounding. This is the property the rebuild exists for; the
collocated ghost-cell walls leaked and made the same system unsolvable.

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

Closed domains. With no outlet the system is singular and p' is defined
up to a constant. As the collocated solver does, the first FLUID cell is
the reference and its value is subtracted from p after the update; p' is
anchored there too so the returned correction has a definite level.

Jacobi is kept as REQ-S08 requires. Its cost is now real because the
system is solvable; the sweep count is returned so it can be measured.
"""

from dataclasses import dataclass

import numpy as np

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import FLUID, SOLID, Mesh
from src.momentum import MomentumPrediction
from src.staggered import p_shape, u_shape, v_shape


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
    sweeps : int
        Jacobi sweeps performed; the harness records it.
    """

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    p_prime: np.ndarray
    sweeps: int


class PressureCorrector:
    """Assemble and solve the p' equation and correct the staggered velocities.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    config : SimConfig
        Supplies rho, alpha_pressure, max_pressure_iter and pressure_tol.
    boundary : StaggeredBoundary
        Supplies the pressure outlets.

    Attributes
    ----------
    needs_pin : bool
        True when no edge face is a pressure outlet, so p' is pinned.
    pin_cell : tuple[int, int]
        (j, i) of the reference cell, the first cell typed FLUID, selected
        as the collocated solver selects its own; meaningful only when
        ``needs_pin``.
    """

    def __init__(
        self, mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary
    ) -> None:
        self._mesh = mesh
        self._rho = config.rho
        self._alpha_p = config.alpha_pressure
        self._max_iter = config.max_pressure_iter
        self._tol = config.pressure_tol
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
            Corrected u and v, updated p, p' and the sweep count.

        Notes
        -----
        Jacobi stops when the largest change of p' over the cells with an
        equation falls below pressure_tol or after max_pressure_iter
        sweeps. In a closed domain the reference cell's value is subtracted
        after the solve, and from p after the update, matching the
        collocated solver's pin at the end of each outer iteration.
        """
        u_star, v_star = prediction.u_star, prediction.v_star
        self._check_shapes(u_star, v_star)
        if p.shape != self._p_shape:
            raise ValueError(f"expected p of shape {self._p_shape}, got {p.shape}")

        c = self.coefficients(prediction.a_p_u, prediction.a_p_v)
        b = self.mass_imbalance(u_star, v_star)
        active = c.a_p > 0.0
        a_p_safe = np.where(active, c.a_p, 1.0)
        pin_j, pin_i = self.pin_cell

        p_prime = np.zeros(self._p_shape, dtype=np.float64)
        padded = np.zeros(
            (self._p_shape[0] + 2, self._p_shape[1] + 2), dtype=np.float64
        )
        sweeps = 0
        for _ in range(self._max_iter):
            padded[1:-1, 1:-1] = p_prime
            p_new = (
                c.a_e * padded[1:-1, 2:]
                + c.a_w * padded[1:-1, :-2]
                + c.a_n * padded[2:, 1:-1]
                + c.a_s * padded[:-2, 1:-1]
                - b
            ) / a_p_safe
            p_new[~active] = 0.0
            diff = (
                float(np.max(np.abs(p_new[active] - p_prime[active])))
                if active.any()
                else 0.0
            )
            p_prime = p_new
            sweeps += 1
            if diff < self._tol:
                break
        if self.needs_pin:
            p_prime[active] -= p_prime[pin_j, pin_i]

        d_u, d_v = self._face_d(prediction.a_p_u, prediction.a_p_v)
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
            sweeps=sweeps,
        )

    def _check_shapes(self, u: np.ndarray, v: np.ndarray) -> None:
        if u.shape != self._u_shape or v.shape != self._v_shape:
            raise ValueError(
                f"expected staggered shapes u {self._u_shape} and v {self._v_shape}, "
                f"got u {u.shape} and v {v.shape}"
            )
