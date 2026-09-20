"""Structured rectangular grid with cell classification for the CFD domain.

Generates face coordinates, cell center coordinates, per-cell widths, the
center-to-center distances a face-based diffusion stencil needs, and a
cell type array that classifies each cell as FLUID, SOLID, or BOUNDARY
based on the domain geometry and obstacle positions from the simulation
config.

Coordinate conventions
----------------------
For an axis with n cells on [0, L]:

``x``        faces, shape (n+1,). ``x[0] == 0.0`` and ``x[n] == L`` exactly.
``xc``       cell centers, shape (n,). Each center is the midpoint of its
             two bounding faces, ``xc[i] = 0.5 * (x[i] + x[i+1])``, so a
             face value averaged from its two neighbouring centers, or a
             center value averaged from its two faces, is second-order
             accurate on any mesh this module builds.
``dx_cell``  cell widths, shape (n,). ``dx_cell[i] = x[i+1] - x[i]``.
``dx_face``  center-to-center distances at faces, shape (n+1,). Interior
             faces carry ``xc[i] - xc[i-1]``; the two boundary faces carry
             the distance from the wall to the first center,
             ``dx_face[0] = xc[0] - x[0]`` and ``dx_face[n] = x[n] - xc[n-1]``,
             which is what a diffusive flux through a wall face divides by.
``dx``       the uniform spacing ``L / n``. On a stretched mesh it is the
             mean spacing and is not a valid stencil length; consumers that
             need a length use ``dx_cell`` or ``dx_face``.

Geometric wall clustering
-------------------------
Each axis may be clustered toward both walls with cell widths that grow by
a constant geometric ratio ``r`` from each wall to the center, mirrored
about the midpoint. The cell count ``n`` is fixed by the configuration,
so the ratio and the wall-adjacent width ``h`` are not independent: for
``n`` even, ``h = (L / 2) (r - 1) / (r^(n/2) - 1)``, and for ``n`` odd the
center cell of width ``h r^((n-1)/2)`` is shared between the two halves.
The configuration names one of the two and this module derives the other.
When the ratio is given, ``h`` follows from the closed form above. When
the wall spacing is given, ``r`` is found by bisection and reported. Either
way the faces are built from both walls toward the middle so that
``x[n] == L`` holds exactly rather than to within accumulated rounding.

A ratio of exactly 1 takes the uniform code path, which reproduces the
arrays the uniform mesh has always produced bit for bit.

Cell classification
-------------------
``cell_type`` marks the outermost ring of cells BOUNDARY and any cell whose
center lies inside an obstacle SOLID. This is the collocated solver's ghost
cell convention: a BOUNDARY cell is a storage slot for the ghost value that
places the physical condition at the domain face. ECR-001 step 3 replaces
that convention and will revisit what a BOUNDARY cell means; nothing here
assumes either reading beyond what the current solver requires.
"""

import numpy as np

from src.config import ObstacleSpec, SimConfig, StretchSpec

# Cell type constants used as integer values in the cell_type array.
# Not an Enum because these are stored in numpy arrays and compared
# with standard integer operations.
FLUID: int = 0
SOLID: int = 1
BOUNDARY: int = 2

# Bisection tolerance on the geometric ratio when the wall spacing is the
# specified quantity. Relative to the ratio itself, so it holds at any
# domain scale.
_RATIO_TOLERANCE: float = 1e-14


def wall_spacing_for_ratio(length: float, n: int, ratio: float) -> float:
    """Wall-adjacent cell width of a symmetric geometric distribution.

    Parameters
    ----------
    length : float
        Axis length L.
    n : int
        Number of cells.
    ratio : float
        Geometric ratio between adjacent cell widths, >= 1.

    Returns
    -------
    float
        Width h of the cell next to either wall.

    Notes
    -----
    Two halves of m cells each with widths h, h r, ..., h r^(m-1) must
    span L. For n even, m = n / 2 and h = (L / 2)(r - 1) / (r^m - 1). For
    n odd, m = (n - 1) / 2 and a center cell of width h r^m sits between
    the halves. A ratio of 1 gives L / n.
    """
    if ratio == 1.0:
        return length / n
    m = n // 2
    half_sum = (ratio**m - 1.0) / (ratio - 1.0)
    if n % 2 == 0:
        return length / (2.0 * half_sum)
    return length / (2.0 * half_sum + ratio**m)


def ratio_for_wall_spacing(length: float, n: int, spacing: float) -> float:
    """Geometric ratio whose symmetric distribution has the given wall width.

    Parameters
    ----------
    length : float
        Axis length L.
    n : int
        Number of cells.
    spacing : float
        Target width of the wall-adjacent cell, 0 < spacing <= L / n.

    Returns
    -------
    float
        The ratio r >= 1, found by bisection to a relative tolerance of
        1e-14. Exactly 1.0 when spacing equals L / n.

    Notes
    -----
    wall_spacing_for_ratio is strictly decreasing in r, so bisection on
    [1, r_hi] converges once r_hi brackets the target.
    """
    uniform = length / n
    if spacing == uniform:
        return 1.0
    if not 0.0 < spacing < uniform:
        raise ValueError(
            f"wall spacing must lie in (0, {uniform}] for {n} cells on {length}, "
            f"got {spacing}"
        )
    if n < 3:
        raise ValueError("wall clustering needs at least 3 cells per axis")
    lo, hi = 1.0, 2.0
    while wall_spacing_for_ratio(length, n, hi) > spacing:
        hi *= 2.0
    while hi - lo > _RATIO_TOLERANCE * hi:
        mid = 0.5 * (lo + hi)
        if wall_spacing_for_ratio(length, n, mid) > spacing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def geometric_faces(length: float, n: int, ratio: float) -> np.ndarray:
    """Face coordinates for n cells clustered symmetrically at both walls.

    Parameters
    ----------
    length : float
        Axis length L.
    n : int
        Number of cells, >= 1.
    ratio : float
        Geometric ratio between adjacent cell widths, > 1. Use the
        uniform path for a ratio of exactly 1.

    Returns
    -------
    np.ndarray
        Faces, shape (n+1,), with ``faces[0] == 0.0`` and
        ``faces[n] == length`` exactly.

    Notes
    -----
    The left half is accumulated from the wall; the right half is the
    mirror image ``length - faces[k]``. For n even the middle face is set
    to ``length / 2`` directly. Closure is therefore exact by construction
    and the distribution is symmetric to within one rounding of the mirror.
    """
    h = wall_spacing_for_ratio(length, n, ratio)
    m = n // 2
    widths = h * ratio ** np.arange(m, dtype=np.float64)
    left = np.concatenate(([0.0], np.cumsum(widths)))
    faces = np.empty(n + 1, dtype=np.float64)
    faces[: m + 1] = left
    if n % 2 == 0:
        faces[m] = 0.5 * length
    faces[n - m :] = length - left[::-1]
    faces[0] = 0.0
    faces[n] = length
    return np.ascontiguousarray(faces)


def _axis(length: float, n: int, spec: StretchSpec) -> tuple[np.ndarray, float, float]:
    """Faces, effective ratio and effective wall spacing for one axis.

    The uniform path is the original linspace construction so that a
    ratio of 1 reproduces the historical arrays bit for bit.
    """
    ratio = spec.ratio
    if spec.min_spacing is not None:
        ratio = ratio_for_wall_spacing(length, n, spec.min_spacing)
    if ratio == 1.0:
        return np.linspace(0.0, length, n + 1), 1.0, length / n
    faces = geometric_faces(length, n, ratio)
    return faces, ratio, float(faces[1] - faces[0])


class Mesh:
    """Structured rectangular grid for the simulation domain.

    Builds a uniform or wall-clustered grid from the domain dimensions and
    stretching specification in the config, then classifies each cell as
    FLUID, SOLID, or BOUNDARY based on obstacle positions and domain edges.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration containing domain dimensions
        (room_width, room_height, nx, ny), the per-axis stretching
        specification (stretch_x, stretch_y) and obstacle definitions.

    Attributes
    ----------
    x, y : np.ndarray
        Face coordinates, shapes (nx+1,) and (ny+1,).
    xc, yc : np.ndarray
        Cell centers, shapes (nx,) and (ny,), midpoints of their faces.
    dx, dy : float
        Uniform spacing L / n; the mean spacing on a stretched mesh.
    dx_cell, dy_cell : np.ndarray
        Cell widths, shapes (nx,) and (ny,).
    dx_face, dy_face : np.ndarray
        Center-to-center distances at faces, shapes (nx+1,) and (ny+1,),
        wall-to-first-center at the two boundary faces.
    stretch_ratio_x, stretch_ratio_y : float
        Effective geometric ratio on each axis; 1.0 when uniform.
    min_spacing_x, min_spacing_y : float
        Effective wall-adjacent cell width on each axis.
    is_uniform : bool
        True when both axes have ratio 1.0.
    cell_type : np.ndarray
        Cell classification, shape (ny, nx), dtype int32.
    """

    def __init__(self, config: SimConfig) -> None:
        nx = config.nx
        ny = config.ny
        width = config.room_width
        height = config.room_height

        self.dx: float = width / nx
        self.dy: float = height / ny

        # Face coordinates (cell edges), uniform or clustered per axis
        self.x, self.stretch_ratio_x, self.min_spacing_x = _axis(
            width, nx, config.stretch_x
        )
        self.y, self.stretch_ratio_y, self.min_spacing_y = _axis(
            height, ny, config.stretch_y
        )
        self.is_uniform: bool = self.stretch_ratio_x == 1.0 and (
            self.stretch_ratio_y == 1.0
        )

        # Cell center coordinates. The uniform expressions are the historical
        # ones and are kept so the uniform arrays do not change by an ulp.
        if self.stretch_ratio_x == 1.0:
            self.xc: np.ndarray = self.x[:-1] + self.dx / 2.0
        else:
            self.xc = 0.5 * (self.x[:-1] + self.x[1:])
        if self.stretch_ratio_y == 1.0:
            self.yc: np.ndarray = self.y[:-1] + self.dy / 2.0
        else:
            self.yc = 0.5 * (self.y[:-1] + self.y[1:])

        # Per-cell widths and center-to-center distances at faces
        self.dx_cell: np.ndarray = np.diff(self.x)
        self.dy_cell: np.ndarray = np.diff(self.y)
        self.dx_face: np.ndarray = self._face_distances(self.x, self.xc)
        self.dy_face: np.ndarray = self._face_distances(self.y, self.yc)

        # Cell classification
        self.cell_type: np.ndarray = self._classify_cells(nx, ny, config.obstacles)

        self._nx = nx
        self._ny = ny

    @staticmethod
    def _face_distances(faces: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Center-to-center distance at every face, wall-to-center at the ends."""
        out = np.empty(faces.shape[0], dtype=np.float64)
        out[0] = centers[0] - faces[0]
        out[1:-1] = np.diff(centers)
        out[-1] = faces[-1] - centers[-1]
        return out

    def _classify_cells(
        self,
        nx: int,
        ny: int,
        obstacles: list[ObstacleSpec],
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
