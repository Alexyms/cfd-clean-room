"""Navier-Stokes solver using the SIMPLE algorithm on a collocated grid.

Solves the steady-state incompressible Navier-Stokes equations with
Rhie-Chow interpolation for face velocities, hybrid advection scheme
(Spalding 1972), and Jacobi iteration for both momentum and pressure
correction.
"""

import logging

import numpy as np

from src.boundary import BoundaryManager
from src.config import SimConfig
from src.mesh import FLUID, SOLID, Mesh

logger = logging.getLogger(__name__)


class NavierStokesSolver:
    """SIMPLE algorithm solver for steady incompressible flow.

    Parameters
    ----------
    mesh : Mesh
        Computational mesh with cell classification.
    config : SimConfig
        Simulation configuration with fluid properties and solver
        parameters (rho, mu, alpha_velocity, alpha_pressure,
        convergence_tol, max_simple_iter, max_pressure_iter,
        pressure_tol).
    boundary : BoundaryManager
        Boundary condition handler for velocity and pressure fields.
    """

    def __init__(
        self, mesh: Mesh, config: SimConfig, boundary: BoundaryManager
    ) -> None:
        self._mesh = mesh
        self._boundary = boundary

        self._nx = mesh.cell_type.shape[1]
        self._ny = mesh.cell_type.shape[0]
        self._dx = mesh.dx
        self._dy = mesh.dy

        self._rho = config.rho
        self._mu = config.mu
        self._alpha_u = config.alpha_velocity
        self._alpha_p = config.alpha_pressure
        self._convergence_tol = config.convergence_tol
        self._max_simple_iter = config.max_simple_iter
        self._max_pressure_iter = config.max_pressure_iter
        self._pressure_tol = config.pressure_tol

        self._cell_volume = self._dx * self._dy

        # Pre-compute masks
        self._fluid = mesh.cell_type == FLUID
        self._solid = mesh.cell_type == SOLID

        # Diffusion conductances (constant on uniform grid)
        self._D_ew = self._mu * self._dy / self._dx
        self._D_ns = self._mu * self._dx / self._dy

        # Reference mass flux for residual scaling
        self._F_ref = self._compute_reference_flux()

        # Detect if pressure pinning is needed (no pressure outlets)
        self._pin_pressure = self._needs_pressure_pin()
        self._pin_j: int = 0
        self._pin_i: int = 0
        if self._pin_pressure:
            # Find the first FLUID cell as the reference point
            fluid_idx = np.argwhere(self._fluid)
            if fluid_idx.size > 0:
                self._pin_j = int(fluid_idx[0, 0])
                self._pin_i = int(fluid_idx[0, 1])

        # Residual history for convergence monitoring
        self.residual_history: list[float] = []

    def _needs_pressure_pin(self) -> bool:
        """Check if any pressure outlet boundary exists.

        Without a Dirichlet pressure BC, the pressure correction is
        determined only up to a constant. Returns True if no pressure
        outlets exist and pinning is required.
        """
        for entry in self._boundary._entries:
            if entry.bc_type == "pressure_outlet":
                return False
        return True

    def _compute_reference_flux(self) -> float:
        """Compute total inlet mass flux for residual scaling.

        For closed domains (lid-driven cavity) where no volumetric flux
        crosses boundaries, falls back to a reference based on the
        maximum prescribed boundary velocity and a characteristic
        cell face area.
        """
        vol_flux = self._boundary.get_total_inlet_flux()
        if vol_flux > 0:
            return self._rho * vol_flux

        # Fallback for closed domains: use max boundary velocity * face area
        max_vel = 0.0
        for entry in self._boundary._entries:
            max_vel = max(max_vel, abs(entry.u_prescribed), abs(entry.v_prescribed))
        face_area = max(self._dx, self._dy)
        return max(self._rho * max_vel * face_area, 1e-30)

    def solve_steady(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve for steady-state velocity and pressure fields.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (u, v, p) velocity and pressure fields, each shape
            [ny, nx], dtype float64, C-contiguous.
        """
        ny, nx = self._ny, self._nx

        u = np.zeros((ny, nx), dtype=np.float64)
        v = np.zeros((ny, nx), dtype=np.float64)
        p = np.zeros((ny, nx), dtype=np.float64)

        self._boundary.apply_velocity_bc(u, v)
        self._boundary.apply_pressure_bc(p)

        self.residual_history = []

        for iteration in range(self._max_simple_iter):
            # Step 1: Momentum coefficients
            a_P, a_E, a_W, a_N, a_S = self._compute_momentum_coefficients(u, v)

            # Step 2: Pressure source terms
            b_u = self._compute_pressure_source_u(p)
            b_v = self._compute_pressure_source_v(p)

            # Step 3: Under-relaxation
            a_P_ur = a_P / self._alpha_u
            ur_boost = (1.0 - self._alpha_u) / self._alpha_u * a_P
            b_u_ur = b_u + ur_boost * u
            b_v_ur = b_v + ur_boost * v

            # Step 4: Momentum Jacobi sweep (single sweep per SIMPLE iter)
            u_star = self._jacobi_momentum_sweep(a_P_ur, a_E, a_W, a_N, a_S, b_u_ur, u)
            v_star = self._jacobi_momentum_sweep(a_P_ur, a_E, a_W, a_N, a_S, b_v_ur, v)
            self._boundary.apply_velocity_bc(u_star, v_star)

            # Step 5: d-coefficient (uses ORIGINAL a_P, not under-relaxed)
            d = np.zeros((ny, nx), dtype=np.float64)
            d[self._fluid] = self._cell_volume / a_P[self._fluid]

            # Step 6-7: Rhie-Chow face mass fluxes and mass imbalance
            F_e, F_w, F_n, F_s = self._compute_face_fluxes(u_star, v_star, p, d)
            mass_imbalance = F_e - F_w + F_n - F_s
            mass_imbalance[~self._fluid] = 0.0

            # Step 8: Pressure correction (Jacobi)
            p_prime = self._solve_pressure_correction(mass_imbalance, d)

            # Step 9: Correct velocity
            u, v = self._correct_velocity(u_star, v_star, p_prime, d)

            # Step 10: Update pressure
            p[self._fluid] += self._alpha_p * p_prime[self._fluid]
            if self._pin_pressure:
                p[self._fluid] -= p[self._pin_j, self._pin_i]
            self._boundary.apply_velocity_bc(u, v)
            self._boundary.apply_pressure_bc(p)

            # Step 11: Convergence check
            residual = self._compute_residual(u, v, p)
            self.residual_history.append(residual)

            if iteration % 50 == 0 or residual < self._convergence_tol:
                logger.info("SIMPLE iter %4d: residual = %.6e", iteration, residual)

            if residual < self._convergence_tol:
                logger.info("Converged at iteration %d", iteration)
                break

        return (
            np.ascontiguousarray(u),
            np.ascontiguousarray(v),
            np.ascontiguousarray(p),
        )

    def compute_residual(self) -> float:
        """Return the most recent scaled mass imbalance residual.

        Returns the last residual computed during solve_steady().
        Returns 0.0 if solve_steady() has not been called.

        Returns
        -------
        float
            Scaled residual from the last SIMPLE iteration.
        """
        if not self.residual_history:
            return 0.0
        return self.residual_history[-1]

    def solve_timestep(
        self,
        u: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance velocity and pressure fields by one timestep.

        Raises
        ------
        NotImplementedError
            Always. Time-stepping is a Phase 4 deliverable.
        """
        raise NotImplementedError("solve_timestep is deferred to Phase 4")

    def _compute_residual(self, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> float:
        """Compute scaled mass imbalance residual.

        Uses simple face interpolation to compute the mass flux through
        each face. The maximum absolute mass imbalance across all FLUID
        cells is divided by the reference inlet flux.

        Parameters
        ----------
        u : np.ndarray
            Horizontal velocity field [ny, nx].
        v : np.ndarray
            Vertical velocity field [ny, nx].
        p : np.ndarray
            Pressure field [ny, nx] (unused, kept for interface).

        Returns
        -------
        float
            Scaled residual. Zero means perfectly divergence-free.
        """
        F_e, F_w, F_n, F_s = self._simple_face_fluxes(u, v)
        div = F_e - F_w + F_n - F_s

        fluid_div = np.abs(div[self._fluid])
        if fluid_div.size == 0:
            return 0.0

        return float(np.max(fluid_div)) / self._F_ref

    # ------------------------------------------------------------------
    # Face flux computation
    # ------------------------------------------------------------------

    def _simple_face_fluxes(
        self, u: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute face mass fluxes using simple interpolation.

        Used for residual monitoring only.

        Parameters
        ----------
        u : np.ndarray
            Horizontal velocity field [ny, nx].
        v : np.ndarray
            Vertical velocity field [ny, nx].

        Returns
        -------
        tuple of 4 np.ndarray
            (F_e, F_w, F_n, F_s) per-cell flux arrays [ny, nx].
        """
        ny, nx = self._ny, self._nx
        rho = self._rho
        dx, dy = self._dx, self._dy
        ct = self._mesh.cell_type

        u_xface = 0.5 * (u[:, :-1] + u[:, 1:])
        Fx = rho * u_xface * dy
        Fx[(ct[:, :-1] == SOLID) | (ct[:, 1:] == SOLID)] = 0.0

        v_yface = 0.5 * (v[:-1, :] + v[1:, :])
        Fy = rho * v_yface * dx
        Fy[(ct[:-1, :] == SOLID) | (ct[1:, :] == SOLID)] = 0.0

        F_e = np.zeros((ny, nx), dtype=np.float64)
        F_w = np.zeros((ny, nx), dtype=np.float64)
        F_n = np.zeros((ny, nx), dtype=np.float64)
        F_s = np.zeros((ny, nx), dtype=np.float64)

        F_e[:, :-1] = Fx
        F_w[:, 1:] = Fx
        F_n[:-1, :] = Fy
        F_s[1:, :] = Fy

        return F_e, F_w, F_n, F_s

    def _compute_face_fluxes(
        self,
        u: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute face mass fluxes with Rhie-Chow interpolation.

        Adds a pressure correction to the simple face velocity that
        couples the collocated pressure and velocity fields and
        prevents checkerboard oscillation.

        Parameters
        ----------
        u : np.ndarray
            Horizontal velocity field [ny, nx].
        v : np.ndarray
            Vertical velocity field [ny, nx].
        p : np.ndarray
            Pressure field [ny, nx].
        d : np.ndarray
            d-coefficient V/a_P [ny, nx].

        Returns
        -------
        tuple of 4 np.ndarray
            (F_e, F_w, F_n, F_s) per-cell flux arrays [ny, nx].
        """
        ny, nx = self._ny, self._nx
        rho = self._rho
        dx, dy = self._dx, self._dy
        ct = self._mesh.cell_type

        # --- x-direction faces ---
        # Fx[j, i] = mass flux through face between columns i and i+1
        ct_L = ct[:, :-1]
        ct_R = ct[:, 1:]

        u_face = 0.5 * (u[:, :-1] + u[:, 1:])
        both_fluid_x = (ct_L == FLUID) & (ct_R == FLUID)
        # d at face: average for FLUID-FLUID, FLUID cell's d for FLUID-BOUNDARY
        d_face_x = np.where(
            both_fluid_x,
            0.5 * (d[:, :-1] + d[:, 1:]),
            np.where(ct_L == FLUID, d[:, :-1], d[:, 1:]),
        )

        # Rhie-Chow needs the 4-point stencil (i-1, i, i+1, i+2)
        can_rc_x = both_fluid_x.copy()
        can_rc_x[:, 0] = False
        if nx > 2:
            can_rc_x[:, -1] = False
        if nx > 3:
            can_rc_x[:, 1:-1] &= (ct[:, :-3] != SOLID) & (ct[:, 3:] != SOLID)

        rc_x = np.zeros_like(u_face)
        if nx > 3:
            # dpdx at left cell: (p[i+1] - p[i-1]) / (2*dx)
            dpdx_L = (p[:, 2 : nx - 1] - p[:, 0 : nx - 3]) / (2 * dx)
            # dpdx at right cell: (p[i+2] - p[i]) / (2*dx)
            dpdx_R = (p[:, 3:nx] - p[:, 1 : nx - 2]) / (2 * dx)
            grad_avg = 0.5 * (dpdx_L + dpdx_R)
            # Compact face gradient
            grad_face = (p[:, 2 : nx - 1] - p[:, 1 : nx - 2]) / dx
            rc_x[:, 1:-1] = d_face_x[:, 1:-1] * (grad_avg - grad_face)

        u_face_rc = u_face + np.where(can_rc_x, rc_x, 0.0)
        Fx = rho * u_face_rc * dy
        Fx[(ct_L == SOLID) | (ct_R == SOLID)] = 0.0

        # --- y-direction faces ---
        ct_B = ct[:-1, :]
        ct_T = ct[1:, :]

        v_face = 0.5 * (v[:-1, :] + v[1:, :])
        both_fluid_y = (ct_B == FLUID) & (ct_T == FLUID)
        d_face_y = np.where(
            both_fluid_y,
            0.5 * (d[:-1, :] + d[1:, :]),
            np.where(ct_B == FLUID, d[:-1, :], d[1:, :]),
        )

        can_rc_y = both_fluid_y.copy()
        can_rc_y[0, :] = False
        if ny > 2:
            can_rc_y[-1, :] = False
        if ny > 3:
            can_rc_y[1:-1, :] &= (ct[:-3, :] != SOLID) & (ct[3:, :] != SOLID)

        rc_y = np.zeros_like(v_face)
        if ny > 3:
            dpdy_B = (p[2 : ny - 1, :] - p[0 : ny - 3, :]) / (2 * dy)
            dpdy_T = (p[3:ny, :] - p[1 : ny - 2, :]) / (2 * dy)
            grad_avg_y = 0.5 * (dpdy_B + dpdy_T)
            grad_face_y = (p[2 : ny - 1, :] - p[1 : ny - 2, :]) / dy
            rc_y[1:-1, :] = d_face_y[1:-1, :] * (grad_avg_y - grad_face_y)

        v_face_rc = v_face + np.where(can_rc_y, rc_y, 0.0)
        Fy = rho * v_face_rc * dx
        Fy[(ct_B == SOLID) | (ct_T == SOLID)] = 0.0

        # --- Map to per-cell arrays ---
        F_e = np.zeros((ny, nx), dtype=np.float64)
        F_w = np.zeros((ny, nx), dtype=np.float64)
        F_n = np.zeros((ny, nx), dtype=np.float64)
        F_s = np.zeros((ny, nx), dtype=np.float64)

        F_e[:, :-1] = Fx
        F_w[:, 1:] = Fx
        F_n[:-1, :] = Fy
        F_s[1:, :] = Fy

        return F_e, F_w, F_n, F_s

    # ------------------------------------------------------------------
    # Momentum coefficients
    # ------------------------------------------------------------------

    def _compute_momentum_coefficients(
        self, u: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute momentum equation coefficients using the hybrid scheme.

        Parameters
        ----------
        u : np.ndarray
            Horizontal velocity field [ny, nx].
        v : np.ndarray
            Vertical velocity field [ny, nx].

        Returns
        -------
        tuple of 5 np.ndarray
            (a_P, a_E, a_W, a_N, a_S) coefficient arrays [ny, nx].
        """
        ny, nx = self._ny, self._nx
        rho = self._rho
        dx, dy = self._dx, self._dy
        D_ew = self._D_ew
        D_ns = self._D_ns
        ct = self._mesh.cell_type

        a_E = np.zeros((ny, nx), dtype=np.float64)
        a_W = np.zeros((ny, nx), dtype=np.float64)
        a_N = np.zeros((ny, nx), dtype=np.float64)
        a_S = np.zeros((ny, nx), dtype=np.float64)
        a_P = np.zeros((ny, nx), dtype=np.float64)

        sl = (slice(1, ny - 1), slice(1, nx - 1))

        ct_E = ct[1 : ny - 1, 2:nx]
        ct_W = ct[1 : ny - 1, 0 : nx - 2]
        ct_N = ct[2:ny, 1 : nx - 1]
        ct_S = ct[0 : ny - 2, 1 : nx - 1]

        # Face mass fluxes from current velocity
        F_e = rho * 0.5 * (u[1 : ny - 1, 1 : nx - 1] + u[1 : ny - 1, 2:nx]) * dy
        F_w = rho * 0.5 * (u[1 : ny - 1, 0 : nx - 2] + u[1 : ny - 1, 1 : nx - 1]) * dy
        F_n = rho * 0.5 * (v[1 : ny - 1, 1 : nx - 1] + v[2:ny, 1 : nx - 1]) * dx
        F_s = rho * 0.5 * (v[0 : ny - 2, 1 : nx - 1] + v[1 : ny - 1, 1 : nx - 1]) * dx

        # Hybrid scheme coefficients
        a_E_tmp = np.maximum(np.maximum(-F_e, D_ew - F_e / 2), 0.0)
        a_W_tmp = np.maximum(np.maximum(F_w, D_ew + F_w / 2), 0.0)
        a_N_tmp = np.maximum(np.maximum(-F_n, D_ns - F_n / 2), 0.0)
        a_S_tmp = np.maximum(np.maximum(F_s, D_ns + F_s / 2), 0.0)

        # SOLID neighbor handling
        D_wall_ew = 2.0 * D_ew
        D_wall_ns = 2.0 * D_ns

        solid_E = ct_E == SOLID
        solid_W = ct_W == SOLID
        solid_N = ct_N == SOLID
        solid_S = ct_S == SOLID

        a_E_tmp[solid_E] = 0.0
        a_W_tmp[solid_W] = 0.0
        a_N_tmp[solid_N] = 0.0
        a_S_tmp[solid_S] = 0.0

        # Zero fluxes at solid faces for net flux term
        F_e_net = np.where(solid_E, 0.0, F_e)
        F_w_net = np.where(solid_W, 0.0, F_w)
        F_n_net = np.where(solid_N, 0.0, F_n)
        F_s_net = np.where(solid_S, 0.0, F_s)

        a_E[sl] = a_E_tmp
        a_W[sl] = a_W_tmp
        a_N[sl] = a_N_tmp
        a_S[sl] = a_S_tmp

        # Diagonal
        a_P[sl] = (
            a_E_tmp
            + a_W_tmp
            + a_N_tmp
            + a_S_tmp
            + (F_e_net - F_w_net + F_n_net - F_s_net)
        )

        # Wall diffusion for SOLID neighbors
        a_P[sl] += np.where(solid_E, D_wall_ew, 0.0)
        a_P[sl] += np.where(solid_W, D_wall_ew, 0.0)
        a_P[sl] += np.where(solid_N, D_wall_ns, 0.0)
        a_P[sl] += np.where(solid_S, D_wall_ns, 0.0)

        # Clamp a_P to avoid division by zero at non-FLUID cells
        a_P = np.where(a_P > 0, a_P, 1.0)

        return a_P, a_E, a_W, a_N, a_S

    # ------------------------------------------------------------------
    # Pressure source terms
    # ------------------------------------------------------------------

    def _compute_pressure_source_u(self, p: np.ndarray) -> np.ndarray:
        """Compute pressure gradient source for u-momentum.

        Parameters
        ----------
        p : np.ndarray
            Pressure field [ny, nx].

        Returns
        -------
        np.ndarray
            Pressure source for u equation [ny, nx].
        """
        ny, nx = self._ny, self._nx
        dy = self._dy
        ct = self._mesh.cell_type
        b_u = np.zeros((ny, nx), dtype=np.float64)
        sl = (slice(1, ny - 1), slice(1, nx - 1))

        p_E = p[1 : ny - 1, 2:nx]
        p_W = p[1 : ny - 1, 0 : nx - 2]
        p_P = p[1 : ny - 1, 1 : nx - 1]

        ct_E = ct[1 : ny - 1, 2:nx]
        ct_W = ct[1 : ny - 1, 0 : nx - 2]

        p_east = np.where(ct_E == SOLID, p_P, p_E)
        p_west = np.where(ct_W == SOLID, p_P, p_W)

        b_u[sl] = -0.5 * dy * (p_east - p_west)
        return b_u

    def _compute_pressure_source_v(self, p: np.ndarray) -> np.ndarray:
        """Compute pressure gradient source for v-momentum.

        Parameters
        ----------
        p : np.ndarray
            Pressure field [ny, nx].

        Returns
        -------
        np.ndarray
            Pressure source for v equation [ny, nx].
        """
        ny, nx = self._ny, self._nx
        dx = self._dx
        ct = self._mesh.cell_type
        b_v = np.zeros((ny, nx), dtype=np.float64)
        sl = (slice(1, ny - 1), slice(1, nx - 1))

        p_N = p[2:ny, 1 : nx - 1]
        p_S = p[0 : ny - 2, 1 : nx - 1]
        p_P = p[1 : ny - 1, 1 : nx - 1]

        ct_N = ct[2:ny, 1 : nx - 1]
        ct_S = ct[0 : ny - 2, 1 : nx - 1]

        p_north = np.where(ct_N == SOLID, p_P, p_N)
        p_south = np.where(ct_S == SOLID, p_P, p_S)

        b_v[sl] = -0.5 * dx * (p_north - p_south)
        return b_v

    # ------------------------------------------------------------------
    # Jacobi sweeps
    # ------------------------------------------------------------------

    def _jacobi_momentum_sweep(
        self,
        a_P: np.ndarray,
        a_E: np.ndarray,
        a_W: np.ndarray,
        a_N: np.ndarray,
        a_S: np.ndarray,
        b: np.ndarray,
        phi_old: np.ndarray,
    ) -> np.ndarray:
        """One Jacobi sweep for the momentum equation.

        Parameters
        ----------
        a_P : np.ndarray
            Diagonal coefficient (under-relaxed) [ny, nx].
        a_E, a_W, a_N, a_S : np.ndarray
            Neighbor coefficients [ny, nx].
        b : np.ndarray
            Source term (under-relaxed) [ny, nx].
        phi_old : np.ndarray
            Previous iteration field [ny, nx].

        Returns
        -------
        np.ndarray
            Updated field after one Jacobi sweep.
        """
        ny, nx = self._ny, self._nx
        phi_new = phi_old.copy()

        sl = (slice(1, ny - 1), slice(1, nx - 1))
        phi_new[sl] = (
            a_E[sl] * phi_old[1 : ny - 1, 2:nx]
            + a_W[sl] * phi_old[1 : ny - 1, 0 : nx - 2]
            + a_N[sl] * phi_old[2:ny, 1 : nx - 1]
            + a_S[sl] * phi_old[0 : ny - 2, 1 : nx - 1]
            + b[sl]
        ) / a_P[sl]

        phi_new[self._solid] = 0.0
        return phi_new

    # ------------------------------------------------------------------
    # Pressure correction
    # ------------------------------------------------------------------

    def _solve_pressure_correction(
        self, mass_imbalance: np.ndarray, d: np.ndarray
    ) -> np.ndarray:
        """Solve the pressure correction equation using Jacobi iteration.

        Parameters
        ----------
        mass_imbalance : np.ndarray
            Mass source at each cell [ny, nx].
        d : np.ndarray
            d-coefficient V/a_P [ny, nx].

        Returns
        -------
        np.ndarray
            Pressure correction field p' [ny, nx].
        """
        ny, nx = self._ny, self._nx
        rho = self._rho
        dx, dy = self._dx, self._dy
        ct = self._mesh.cell_type

        # Pressure correction coefficients
        a_E_pp = np.zeros((ny, nx), dtype=np.float64)
        a_W_pp = np.zeros((ny, nx), dtype=np.float64)
        a_N_pp = np.zeros((ny, nx), dtype=np.float64)
        a_S_pp = np.zeros((ny, nx), dtype=np.float64)

        sl = (slice(1, ny - 1), slice(1, nx - 1))

        ct_E = ct[1 : ny - 1, 2:nx]
        ct_W = ct[1 : ny - 1, 0 : nx - 2]
        ct_N = ct[2:ny, 1 : nx - 1]
        ct_S = ct[0 : ny - 2, 1 : nx - 1]

        d_P = d[sl]
        d_E = d[1 : ny - 1, 2:nx]
        d_W = d[1 : ny - 1, 0 : nx - 2]
        d_N = d[2:ny, 1 : nx - 1]
        d_S = d[0 : ny - 2, 1 : nx - 1]

        def face_d(d_p, d_nb, ct_nb):
            return np.where(
                ct_nb == SOLID,
                0.0,
                np.where(ct_nb == FLUID, 0.5 * (d_p + d_nb), d_p),
            )

        d_e = face_d(d_P, d_E, ct_E)
        d_w = face_d(d_P, d_W, ct_W)
        d_n = face_d(d_P, d_N, ct_N)
        d_s = face_d(d_P, d_S, ct_S)

        a_E_pp[sl] = np.where(ct_E == SOLID, 0.0, rho * d_e * dy / dx)
        a_W_pp[sl] = np.where(ct_W == SOLID, 0.0, rho * d_w * dy / dx)
        a_N_pp[sl] = np.where(ct_N == SOLID, 0.0, rho * d_n * dx / dy)
        a_S_pp[sl] = np.where(ct_S == SOLID, 0.0, rho * d_s * dx / dy)

        a_P_pp = a_E_pp + a_W_pp + a_N_pp + a_S_pp
        a_P_pp = np.where(a_P_pp > 0, a_P_pp, 1.0)

        p_prime = np.zeros((ny, nx), dtype=np.float64)

        for _ in range(self._max_pressure_iter):
            self._boundary.apply_pressure_bc(p_prime)

            p_prime_new = np.zeros((ny, nx), dtype=np.float64)
            p_prime_new[sl] = (
                a_E_pp[sl] * p_prime[1 : ny - 1, 2:nx]
                + a_W_pp[sl] * p_prime[1 : ny - 1, 0 : nx - 2]
                + a_N_pp[sl] * p_prime[2:ny, 1 : nx - 1]
                + a_S_pp[sl] * p_prime[0 : ny - 2, 1 : nx - 1]
                - mass_imbalance[sl]
            ) / a_P_pp[sl]

            p_prime_new[~self._fluid] = 0.0

            fluid_vals = self._fluid[sl]
            diff = (
                float(
                    np.max(
                        np.abs(p_prime_new[sl][fluid_vals] - p_prime[sl][fluid_vals])
                    )
                )
                if np.any(fluid_vals)
                else 0.0
            )
            p_prime = p_prime_new

            if diff < self._pressure_tol:
                break

        # Set BOUNDARY ghost values so the velocity correction reads
        # correct p' gradients at cells adjacent to boundaries
        self._boundary.apply_pressure_bc(p_prime)
        return p_prime

    # ------------------------------------------------------------------
    # Velocity correction
    # ------------------------------------------------------------------

    def _correct_velocity(
        self,
        u_star: np.ndarray,
        v_star: np.ndarray,
        p_prime: np.ndarray,
        d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct cell-center velocities using pressure correction gradient.

        Parameters
        ----------
        u_star : np.ndarray
            Momentum-predicted u velocity [ny, nx].
        v_star : np.ndarray
            Momentum-predicted v velocity [ny, nx].
        p_prime : np.ndarray
            Pressure correction field [ny, nx].
        d : np.ndarray
            d-coefficient V/a_P [ny, nx].

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Corrected (u, v) fields.
        """
        ny, nx = self._ny, self._nx
        dx, dy = self._dx, self._dy
        ct = self._mesh.cell_type

        u = u_star.copy()
        v = v_star.copy()

        sl = (slice(1, ny - 1), slice(1, nx - 1))

        pp_E = p_prime[1 : ny - 1, 2:nx]
        pp_W = p_prime[1 : ny - 1, 0 : nx - 2]
        pp_N = p_prime[2:ny, 1 : nx - 1]
        pp_S = p_prime[0 : ny - 2, 1 : nx - 1]
        pp_P = p_prime[sl]

        ct_E = ct[1 : ny - 1, 2:nx]
        ct_W = ct[1 : ny - 1, 0 : nx - 2]
        ct_N = ct[2:ny, 1 : nx - 1]
        ct_S = ct[0 : ny - 2, 1 : nx - 1]

        pp_east = np.where(ct_E == SOLID, pp_P, pp_E)
        pp_west = np.where(ct_W == SOLID, pp_P, pp_W)
        pp_north = np.where(ct_N == SOLID, pp_P, pp_N)
        pp_south = np.where(ct_S == SOLID, pp_P, pp_S)

        u[sl] = u_star[sl] - d[sl] * (pp_east - pp_west) / (2 * dx)
        v[sl] = v_star[sl] - d[sl] * (pp_north - pp_south) / (2 * dy)

        u[self._solid] = 0.0
        v[self._solid] = 0.0

        return u, v
