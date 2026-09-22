"""Boundary condition application for the collocated grid solver.

Maps BOUNDARY cells to their BC type (wall, velocity_inlet,
pressure_outlet) and applies ghost cell values using interpolation
formulas that place the physical condition at the domain face, not
at the cell center. This is required for second-order accuracy on
the collocated grid.

Which condition applies where is decided by src/boundary_registry.py,
shared with the staggered layer in src/boundary_staggered.py. This module
owns only what is specific to the collocated layout: the ring of BOUNDARY
cells, the interior neighbor each ghost value reads from, and the ghost
formulas themselves. It is retired with the collocated solver at ECR-001
step 6.
"""

from dataclasses import dataclass

import numpy as np

from src.boundary_registry import (
    PRESSURE_OUTLET,
    VELOCITY_INLET,
    BoundaryRegistry,
    covers,
)
from src.config import SimConfig
from src.mesh import BOUNDARY, FLUID, Mesh


@dataclass
class _BCEntry:
    """Internal record for a single boundary cell's BC assignment.

    Parameters
    ----------
    i : int
        Cell x-index (column).
    j : int
        Cell y-index (row).
    ni : int
        Interior neighbor x-index.
    nj : int
        Interior neighbor y-index.
    bc_type : str
        One of "wall", "velocity_inlet", "pressure_outlet".
    u_prescribed : float
        Prescribed u-velocity at the domain face (0 for wall/outlet).
    v_prescribed : float
        Prescribed v-velocity at the domain face (0 for wall/outlet).
    edge : str
        Which domain edge: "top", "bottom", "left", "right".
    """

    i: int
    j: int
    ni: int
    nj: int
    bc_type: str
    u_prescribed: float
    v_prescribed: float
    edge: str


class BoundaryManager:
    """Apply boundary conditions to velocity and pressure fields.

    During construction, maps each BOUNDARY cell to its BC type and
    prescribed values by matching cell positions against the named
    boundary segments in the config. Unmatched edge cells default to
    no-slip wall.

    Ghost cell formulas place the physical condition at the domain
    face (distance dy/2 or dx/2 from the boundary cell center),
    not at the cell center itself.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh with cell_type classification.
    config : SimConfig
        Simulation configuration with boundary definitions.
    """

    def __init__(self, mesh: Mesh, config: SimConfig) -> None:
        self._mesh = mesh
        self._nx = mesh.cell_type.shape[1]
        self._ny = mesh.cell_type.shape[0]
        self._dx = mesh.dx
        self._dy = mesh.dy
        self._registry = BoundaryRegistry(config)
        self._boundaries = self._registry.boundaries
        self._entries: list[_BCEntry] = []

        self._build_bc_map(mesh)

    def _build_bc_map(self, mesh: Mesh) -> None:
        """Map each BOUNDARY cell to its BC type and interior neighbor."""
        nx = self._nx
        ny = self._ny

        for j in range(ny):
            for i in range(nx):
                if mesh.cell_type[j, i] != BOUNDARY:
                    continue

                edge = self._identify_edge(i, j, nx, ny)
                ni, nj = self._find_interior_neighbor(
                    i, j, edge, nx, ny, mesh.cell_type
                )

                condition = self._registry.condition_at(
                    edge, self._coordinate_along_edge(i, j, edge)
                )

                self._entries.append(
                    _BCEntry(
                        i=i,
                        j=j,
                        ni=ni,
                        nj=nj,
                        bc_type=condition.bc_type,
                        u_prescribed=condition.u_prescribed,
                        v_prescribed=condition.v_prescribed,
                        edge=edge,
                    )
                )

    def _coordinate_along_edge(self, i: int, j: int, edge: str) -> float:
        """Cell-center coordinate along an edge: x on top/bottom, y on left/right."""
        if edge in ("top", "bottom"):
            return float(self._mesh.xc[i])
        return float(self._mesh.yc[j])

    @staticmethod
    def _identify_edge(i: int, j: int, nx: int, ny: int) -> str:
        """Determine which domain edge a boundary cell sits on.

        For corner cells, bottom/top takes priority over left/right.

        Parameters
        ----------
        i : int
            Cell x-index.
        j : int
            Cell y-index.
        nx : int
            Grid width in cells.
        ny : int
            Grid height in cells.

        Returns
        -------
        str
            One of "bottom", "top", "left", "right".
        """
        if j == 0:
            return "bottom"
        if j == ny - 1:
            return "top"
        if i == 0:
            return "left"
        return "right"

    @staticmethod
    def _find_interior_neighbor(
        i: int,
        j: int,
        edge: str,
        nx: int,
        ny: int,
        cell_type: np.ndarray,
    ) -> tuple[int, int]:
        """Find the nearest FLUID cell toward the domain interior.

        If no FLUID cell exists along the inward direction (e.g., at
        corners where the adjacent cell is SOLID), falls back to the
        immediately adjacent interior cell regardless of type.

        Parameters
        ----------
        i : int
            Boundary cell x-index.
        j : int
            Boundary cell y-index.
        edge : str
            Which domain edge the cell sits on.
        nx : int
            Grid width in cells.
        ny : int
            Grid height in cells.
        cell_type : np.ndarray
            Cell classification array [ny, nx].

        Returns
        -------
        tuple[int, int]
            (ni, nj) indices of the interior neighbor.
        """
        if edge == "bottom":
            for jj in range(1, ny):
                if cell_type[jj, i] == FLUID:
                    return (i, jj)
            return (i, 1)
        if edge == "top":
            for jj in range(ny - 2, -1, -1):
                if cell_type[jj, i] == FLUID:
                    return (i, jj)
            return (i, ny - 2)
        if edge == "left":
            for ii in range(1, nx):
                if cell_type[j, ii] == FLUID:
                    return (ii, j)
            return (1, j)
        # right
        for ii in range(nx - 2, -1, -1):
            if cell_type[j, ii] == FLUID:
                return (ii, j)
        return (nx - 2, j)

    def apply_velocity_bc(self, u: np.ndarray, v: np.ndarray) -> None:
        """Set BOUNDARY cell velocities using ghost cell interpolation.

        Reads the current interior neighbor values and computes ghost
        cell values that place the physical BC at the domain face.

        Parameters
        ----------
        u : np.ndarray
            Horizontal velocity field [ny, nx], modified in-place.
        v : np.ndarray
            Vertical velocity field [ny, nx], modified in-place.

        Notes
        -----
        Dirichlet (wall, velocity_inlet):
            u_bnd = (2 * V_face + u_interior) / 3

        Neumann (pressure_outlet, zero gradient):
            u_bnd = u_interior
        """
        for e in self._entries:
            u_int = u[e.nj, e.ni]
            v_int = v[e.nj, e.ni]

            if e.bc_type == PRESSURE_OUTLET:
                u[e.j, e.i] = u_int
                v[e.j, e.i] = v_int
            else:
                # Dirichlet: wall (V=0) or velocity_inlet (V=prescribed)
                u[e.j, e.i] = (2.0 * e.u_prescribed + u_int) / 3.0
                v[e.j, e.i] = (2.0 * e.v_prescribed + v_int) / 3.0

    def apply_pressure_bc(self, p: np.ndarray) -> None:
        """Set BOUNDARY cell pressures using ghost cell interpolation.

        Parameters
        ----------
        p : np.ndarray
            Pressure field [ny, nx], modified in-place.

        Notes
        -----
        Wall, velocity_inlet: zero gradient (Neumann).
            p_bnd = p_interior

        Pressure_outlet: Dirichlet p=0 at domain face.
            p_bnd = (2*0 + p_interior) / 3 = p_interior / 3
        """
        for e in self._entries:
            p_int = p[e.nj, e.ni]

            if e.bc_type == PRESSURE_OUTLET:
                p[e.j, e.i] = p_int / 3.0
            else:
                p[e.j, e.i] = p_int

    def apply_concentration_bc(self, c: np.ndarray, size_class: int) -> None:
        """Apply concentration boundary conditions (Phase 3 placeholder).

        Parameters
        ----------
        c : np.ndarray
            Concentration field [ny, nx].
        size_class : int
            Particle size class index.

        Raises
        ------
        NotImplementedError
            Always. Concentration BCs are implemented in Phase 3.
        """
        raise NotImplementedError("Concentration BCs are deferred to Phase 3")

    def get_inlet_flux(self, boundary_name: str) -> float:
        """Compute total volumetric flux through a named inlet boundary.

        For a 2D simulation, flux = sum of (face_velocity * face_length)
        across all cells belonging to the named boundary.

        Parameters
        ----------
        boundary_name : str
            Name of a boundary defined in the config (e.g., "hepa_supply").

        Returns
        -------
        float
            Total volumetric flux in m^2/s (per unit depth).

        Raises
        ------
        KeyError
            If boundary_name is not found in the config.
        """
        spec = self._registry.spec(boundary_name)
        flux = 0.0

        for e in self._entries:
            if e.bc_type != VELOCITY_INLET:
                continue
            # Check if this entry belongs to the named boundary
            if not covers(spec, e.edge, self._coordinate_along_edge(e.i, e.j, e.edge)):
                continue

            if e.edge in ("top", "bottom"):
                face_len = self._dx
                flux += abs(e.v_prescribed) * face_len
            else:
                face_len = self._dy
                flux += abs(e.u_prescribed) * face_len

        return flux

    def get_total_inlet_flux(self) -> float:
        """Compute total volumetric flux across all velocity inlet boundaries.

        Returns the sum of get_inlet_flux(name) for all boundaries
        of type velocity_inlet. Used by the solver for residual scaling.

        Returns
        -------
        float
            Total volumetric flux in m^2/s (per unit depth).
        """
        total = 0.0
        for name, spec in self._boundaries.items():
            if spec.type == VELOCITY_INLET:
                total += self.get_inlet_flux(name)
        return total

    def has_pressure_outlet(self) -> bool:
        """Return True if any BOUNDARY cell is configured as a pressure_outlet BC.

        Used by the solver to determine whether pressure pinning is required
        for closed domains.

        Returns
        -------
        bool
            True if at least one entry has bc_type == "pressure_outlet".
        """
        return any(e.bc_type == PRESSURE_OUTLET for e in self._entries)

    def get_max_boundary_velocity(self) -> float:
        """Return the maximum absolute prescribed velocity across all BOUNDARY cells.

        Used by the solver as a reference scale for residual normalization
        in closed-domain cases where total volumetric flux is zero.

        Returns
        -------
        float
            Maximum of |u_prescribed| and |v_prescribed| over all entries.
            Returns 0.0 if there are no entries.
        """
        max_vel = 0.0
        for e in self._entries:
            max_vel = max(max_vel, abs(e.u_prescribed), abs(e.v_prescribed))
        return max_vel
