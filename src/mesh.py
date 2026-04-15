"""Structured rectangular grid with cell classification for the CFD domain.

Generates face coordinates, cell center coordinates, and a cell type
array that classifies each cell as FLUID, SOLID, or BOUNDARY based on
the domain geometry and obstacle positions from the simulation config.
"""

import numpy as np

from src.config import SimConfig

# Cell type constants used as integer values in the cell_type array.
# Not an Enum because these are stored in numpy arrays and compared
# with standard integer operations.
FLUID: int = 0
SOLID: int = 1
BOUNDARY: int = 2


class Mesh:
    """Structured rectangular grid for the simulation domain.

    Builds a uniform grid from the domain dimensions in the config,
    then classifies each cell as FLUID, SOLID, or BOUNDARY based on
    obstacle positions and domain edges.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration containing domain dimensions
        (room_width, room_height, nx, ny) and obstacle definitions.
    """

    def __init__(self, config: SimConfig) -> None:
        nx = config.nx
        ny = config.ny
        width = config.room_width
        height = config.room_height

        self.dx: float = width / nx
        self.dy: float = height / ny

        # Face coordinates (cell edges)
        self.x: np.ndarray = np.linspace(0.0, width, nx + 1)
        self.y: np.ndarray = np.linspace(0.0, height, ny + 1)

        # Cell center coordinates
        self.xc: np.ndarray = self.x[:-1] + self.dx / 2.0
        self.yc: np.ndarray = self.y[:-1] + self.dy / 2.0

        # Cell classification
        self.cell_type: np.ndarray = self._classify_cells(nx, ny, config.obstacles)

        self._nx = nx
        self._ny = ny

    def _classify_cells(
        self,
        nx: int,
        ny: int,
        obstacles: list,
    ) -> np.ndarray:
        """Classify each cell as FLUID, SOLID, or BOUNDARY.

        Parameters
        ----------
        nx : int
            Number of cells in the x direction.
        ny : int
            Number of cells in the y direction.
        obstacles : list
            List of ObstacleSpec objects from the config.

        Returns
        -------
        np.ndarray
            Cell type array with shape (ny, nx), dtype int32,
            C-contiguous.
        """
        cell_type = np.full((ny, nx), FLUID, dtype=np.int32)

        # Mark obstacle cells (center inside obstacle bounding box)
        for obs in obstacles:
            for j in range(ny):
                for i in range(nx):
                    if (
                        obs.x_start <= self.xc[i] <= obs.x_end
                        and obs.y_start <= self.yc[j] <= obs.y_end
                    ):
                        cell_type[j, i] = SOLID

        # Mark boundary cells (domain edges that are not SOLID)
        for i in range(nx):
            if cell_type[0, i] != SOLID:
                cell_type[0, i] = BOUNDARY
            if cell_type[ny - 1, i] != SOLID:
                cell_type[ny - 1, i] = BOUNDARY
        for j in range(ny):
            if cell_type[j, 0] != SOLID:
                cell_type[j, 0] = BOUNDARY
            if cell_type[j, nx - 1] != SOLID:
                cell_type[j, nx - 1] = BOUNDARY

        return np.ascontiguousarray(cell_type)

    def is_fluid(self, i: int, j: int) -> bool:
        """Check whether the cell at grid position (i, j) is a fluid cell.

        Parameters
        ----------
        i : int
            Cell x-index (column).
        j : int
            Cell y-index (row).

        Returns
        -------
        bool
            True if cell_type[j, i] == FLUID.

        Raises
        ------
        IndexError
            If i or j is outside the valid grid range.
        """
        if i < 0 or i >= self._nx or j < 0 or j >= self._ny:
            raise IndexError(
                f"Cell index ({i}, {j}) out of range "
                f"[0..{self._nx - 1}, 0..{self._ny - 1}]"
            )
        return int(self.cell_type[j, i]) == FLUID

    def get_neighbors(self, i: int, j: int) -> list[tuple[int, int]]:
        """Return the 4-connected neighbors of the cell at (i, j).

        Returns all neighbors within domain bounds regardless of cell
        type. The caller decides how to handle SOLID or BOUNDARY
        neighbors.

        Parameters
        ----------
        i : int
            Cell x-index (column).
        j : int
            Cell y-index (row).

        Returns
        -------
        list[tuple[int, int]]
            List of (ni, nj) neighbor positions. Contains 2 to 4
            entries depending on whether the cell is at a corner,
            edge, or interior.

        Raises
        ------
        IndexError
            If i or j is outside the valid grid range.
        """
        if i < 0 or i >= self._nx or j < 0 or j >= self._ny:
            raise IndexError(
                f"Cell index ({i}, {j}) out of range "
                f"[0..{self._nx - 1}, 0..{self._ny - 1}]"
            )
        neighbors = []
        if j + 1 < self._ny:
            neighbors.append((i, j + 1))  # North
        if j - 1 >= 0:
            neighbors.append((i, j - 1))  # South
        if i + 1 < self._nx:
            neighbors.append((i + 1, j))  # East
        if i - 1 >= 0:
            neighbors.append((i - 1, j))  # West
        return neighbors
