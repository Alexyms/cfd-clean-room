"""Boundary condition application for the collocated grid solver.

Maps BOUNDARY cells to their BC type (wall, velocity_inlet,
pressure_outlet) and applies ghost cell values using interpolation
formulas that place the physical condition at the domain face, not
at the cell center. This is required for second-order accuracy on
the collocated grid.
"""

from dataclasses import dataclass

import numpy as np

from src.config import BoundarySpec, SimConfig
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
        self._boundaries = config.boundaries
        self._entries: list[_BCEntry] = []

        self._build_bc_map(mesh, config)

    def _build_bc_map(self, mesh: Mesh, config: SimConfig) -> None:
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

                bc_type, u_face, v_face = self._match_boundary(i, j, edge, mesh, config)

                self._entries.append(
                    _BCEntry(
                        i=i,
                        j=j,
                        ni=ni,
                        nj=nj,
                        bc_type=bc_type,
                        u_prescribed=u_face,
                        v_prescribed=v_face,
                        edge=edge,
                    )
                )

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

    def _match_boundary(
        self,
        i: int,
        j: int,
        edge: str,
        mesh: Mesh,
        config: SimConfig,
    ) -> tuple[str, float, float]:
        """Match a boundary cell to a named BC or default to wall.

        Parameters
        ----------
        i : int
            Cell x-index.
        j : int
            Cell y-index.
        edge : str
            Which domain edge.
        mesh : Mesh
            Mesh for coordinate lookup.
        config : SimConfig
            Config with boundary definitions.

        Returns
        -------
        tuple[str, float, float]
            (bc_type, u_prescribed, v_prescribed) at the domain face.
        """
        xc = mesh.xc[i]
        yc = mesh.yc[j]

        for spec in config.boundaries.values():
            if spec.location != edge:
                continue

            if (
                edge in ("top", "bottom")
                and spec.x_start is not None
                and spec.x_end is not None
                and spec.x_start <= xc <= spec.x_end
            ):
                return self._bc_values(spec, edge)
            if (
                edge in ("left", "right")
                and spec.y_start is not None
                and spec.y_end is not None
                and spec.y_start <= yc <= spec.y_end
            ):
                return self._bc_values(spec, edge)

        # Default: no-slip wall
        return ("wall", 0.0, 0.0)

    @staticmethod
    def _bc_values(spec: BoundarySpec, edge: str) -> tuple[str, float, float]:
        """Extract face velocity components from a BoundarySpec.

        Parameters
        ----------
        spec : BoundarySpec
            The matched boundary specification.
        edge : str
            Which domain edge ("top", "bottom", "left", "right").

        Returns
        -------
        tuple[str, float, float]
            (bc_type, u_face, v_face).
        """
        if spec.type == "wall":
            return ("wall", 0.0, 0.0)

        if spec.type == "pressure_outlet":
            return ("pressure_outlet", 0.0, 0.0)

        if spec.type != "velocity_inlet":
            raise ValueError(f"Unrecognized boundary type: {spec.type}")

        # velocity_inlet: decompose velocity into u, v based on edge normal
        vel = spec.velocity if spec.velocity is not None else 0.0
        if edge == "top":
            # Flow enters downward (negative v)
            return ("velocity_inlet", 0.0, -vel)
        if edge == "bottom":
            # Flow enters upward (positive v)
            return ("velocity_inlet", 0.0, vel)
        if edge == "left":
            # Flow enters rightward (positive u)
            return ("velocity_inlet", vel, 0.0)
        # right: flow enters leftward (negative u)
        return ("velocity_inlet", -vel, 0.0)

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

            if e.bc_type == "pressure_outlet":
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

            if e.bc_type == "pressure_outlet":
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
        if boundary_name not in self._boundaries:
            raise KeyError(f"Boundary '{boundary_name}' not found in config")

        spec = self._boundaries[boundary_name]
        flux = 0.0

        for e in self._entries:
            if e.bc_type != "velocity_inlet":
                continue
            # Check if this entry belongs to the named boundary
            if not self._entry_matches_spec(e, spec):
                continue

            if e.edge in ("top", "bottom"):
                face_len = self._dx
                flux += abs(e.v_prescribed) * face_len
            else:
                face_len = self._dy
                flux += abs(e.u_prescribed) * face_len

        return flux

    def _entry_matches_spec(self, entry: _BCEntry, spec: BoundarySpec) -> bool:
        """Check if a BC entry belongs to a specific named boundary.

        Parameters
        ----------
        entry : _BCEntry
            The boundary cell entry.
        spec : BoundarySpec
            The boundary specification to match against.

        Returns
        -------
        bool
            True if the entry's position and edge match the spec.
        """
        if entry.edge != spec.location:
            return False
        xc = self._mesh.xc[entry.i]
        yc = self._mesh.yc[entry.j]
        if entry.edge in ("top", "bottom"):
            return (
                spec.x_start is not None
                and spec.x_end is not None
                and spec.x_start <= xc <= spec.x_end
            )
        return (
            spec.y_start is not None
            and spec.y_end is not None
            and spec.y_start <= yc <= spec.y_end
        )
