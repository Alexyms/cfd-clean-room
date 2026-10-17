"""Staggered (MAC) field layout: shapes, allocation and face-to-center averaging.

Layout (REQ-S07). For a mesh with nx by ny cells:

``u``  x-velocity on the vertical (east-west) faces, shape [ny, nx+1].
       ``u[j, i]`` sits at ``(x[i], yc[j])``; ``u[j, 0]`` and ``u[j, nx]``
       are on the left and right domain boundaries.
``v``  y-velocity on the horizontal (north-south) faces, shape [ny+1, nx].
       ``v[j, i]`` sits at ``(xc[i], y[j])``; ``v[0, i]`` and ``v[ny, i]``
       are on the bottom and top domain boundaries.
``p``  pressure at cell centers, shape [ny, nx], at ``(xc[i], yc[j])``.

The public ``solve_steady`` contract returns cell-centered fields of shape
[ny, nx], so the solver averages u and v to cell centers with
``to_cell_centers`` before returning. Because every cell center is the
midpoint of its two bounding faces (see src/mesh.py), that average is
second-order accurate on uniform and stretched meshes alike and reproduces
a field that is linear in the coordinate exactly.

The layout was internal to the velocity solver until ADR-011 (REQ-S13).
Continuity is enforced on the faces and the cell means do not carry it, so
the solver now also exposes the faces of its last solve as a
``FaceVelocities``, the transport solver's input type. The dataclass is
defined here because this module owns the layout; every instance holds
read-only float64 copies, so a reader cannot disturb the solver and the
solver cannot disturb a reader.

The equipment vents of ADR-014 J sit on obstacle faces, which are interior
entries of the face arrays. ``obstacle_face_inputs`` finds them, as
``edge_cell_inputs`` finds the domain edges' faces, so every layer that reads
a vent reads the same faces.

This module holds no discretization. It computes no flux, no gradient and
no coefficient; ECR-001 steps 3 to 5 own those.
"""

from dataclasses import dataclass

import numpy as np

from src.config import ObstacleSpec
from src.mesh import SOLID, Mesh


def u_shape(mesh: Mesh) -> tuple[int, int]:
    """Shape of the u-velocity array on vertical faces, (ny, nx+1)."""
    return mesh.yc.shape[0], mesh.x.shape[0]


def v_shape(mesh: Mesh) -> tuple[int, int]:
    """Shape of the v-velocity array on horizontal faces, (ny+1, nx)."""
    return mesh.y.shape[0], mesh.xc.shape[0]


def p_shape(mesh: Mesh) -> tuple[int, int]:
    """Shape of the pressure array at cell centers, (ny, nx)."""
    return mesh.yc.shape[0], mesh.xc.shape[0]


def allocate_fields(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allocate zeroed staggered fields for a mesh.

    Parameters
    ----------
    mesh : Mesh
        Mesh that fixes the shapes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (u, v, p) with shapes [ny, nx+1], [ny+1, nx] and [ny, nx],
        dtype float64, C-contiguous, all zero.
    """
    u = np.zeros(u_shape(mesh), dtype=np.float64)
    v = np.zeros(v_shape(mesh), dtype=np.float64)
    p = np.zeros(p_shape(mesh), dtype=np.float64)
    return u, v, p


def u_face_coordinates(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of every u storage location as broadcastable 2D arrays.

    Parameters
    ----------
    mesh : Mesh
        Mesh the field lives on.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, Y), each shape [ny, nx+1], with ``X[j, i] = x[i]`` and
        ``Y[j, i] = yc[j]``.
    """
    return np.meshgrid(mesh.x, mesh.yc)


def v_face_coordinates(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of every v storage location as broadcastable 2D arrays.

    Parameters
    ----------
    mesh : Mesh
        Mesh the field lives on.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, Y), each shape [ny+1, nx], with ``X[j, i] = xc[i]`` and
        ``Y[j, i] = y[j]``.
    """
    return np.meshgrid(mesh.xc, mesh.y)


def cell_center_coordinates(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of every pressure storage location as 2D arrays.

    Parameters
    ----------
    mesh : Mesh
        Mesh the field lives on.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, Y), each shape [ny, nx], with ``X[j, i] = xc[i]`` and
        ``Y[j, i] = yc[j]``.
    """
    return np.meshgrid(mesh.xc, mesh.yc)


def edge_cells(cell_type: np.ndarray, edge: str) -> np.ndarray:
    """The cells along a domain edge, in the storage order of that edge's faces.

    Parameters
    ----------
    cell_type : np.ndarray
        ``mesh.cell_type``, shape [ny, nx].
    edge : str
        One of "bottom", "top", "left", "right".

    Returns
    -------
    np.ndarray
        A view: row 0 for the bottom edge, row ny-1 for the top, column 0
        for the left and column nx-1 for the right, so entry k sits behind
        the k-th face of that edge (``v[0, k]``, ``v[ny, k]``, ``u[k, 0]``,
        ``u[k, nx]``). Both boundary layers read the SOLID cells along an
        edge through this one function.

    Raises
    ------
    ValueError
        On an unknown edge name.
    """
    if edge == "bottom":
        return cell_type[0, :]
    if edge == "top":
        return cell_type[-1, :]
    if edge == "left":
        return cell_type[:, 0]
    if edge == "right":
        return cell_type[:, -1]
    raise ValueError(f"unknown edge '{edge}'; use bottom, top, left or right")


def edge_cell_inputs(mesh: Mesh, edge: str) -> tuple[np.ndarray, np.ndarray]:
    """The cell coordinates along an edge and the SOLID mask behind its faces.

    The two inputs ``BoundaryRegistry.coverage_along`` takes, derived here
    once so that the velocity layer and the concentration layer ask the same
    question of the registry and cannot disagree about which faces an edge
    has or which of them sit behind an obstacle (REQ-S12.1).

    Parameters
    ----------
    mesh : Mesh
        Supplies ``xc``, ``yc`` and ``cell_type``.
    edge : str
        One of "bottom", "top", "left", "right".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(coordinates, solid)``: the cell-center coordinates along the edge
        in face order (``xc`` on the bottom and top edges, ``yc`` on the left
        and right) and, for each, whether the cell behind the face is SOLID.

    Raises
    ------
    ValueError
        On an unknown edge name.
    """
    solid = edge_cells(mesh.cell_type, edge) == SOLID
    coordinates = mesh.xc if edge in ("bottom", "top") else mesh.yc
    return coordinates, solid


@dataclass(frozen=True, eq=False)
class ObstacleFace:
    """The mesh faces on one face of one obstacle that border the room (ADR-014 J).

    Parameters
    ----------
    component : str
        "v" on a top or bottom face (horizontal faces), "u" on a side.
    rows, cols : np.ndarray
        int64, the index of each face in its component's array, in order
        along the face. Faces between the obstacle's cells and a
        non-SOLID cell on the named side only; never a domain-edge face.
    coordinates : np.ndarray
        The position of each face along the obstacle face: ``xc`` on a top
        or bottom face, ``yc`` on a side.
    widths : np.ndarray
        Each face's width: ``dx_cell`` or ``dy_cell``.
    corner_rows, corner_cols : np.ndarray
        int64, in ``MomentumPredictor.predict``'s ``wall_mu`` layout
        [ny+1, nx+1]: the faces of the tangential component's wall stencil
        that lie over the obstacle face between two of these faces, the
        horizontal faces through ``(x[i], y[j])`` on a top or bottom face,
        the vertical ones on a side. The two ends of a run, at the
        obstacle's corners, are walls over half their span and are not
        listed (the momentum stencil's corner rule).
    corner_coordinates : np.ndarray
        The position of each such face along the obstacle face: ``x`` or
        ``y``.
    """

    component: str
    rows: np.ndarray
    cols: np.ndarray
    coordinates: np.ndarray
    widths: np.ndarray
    corner_rows: np.ndarray
    corner_cols: np.ndarray
    corner_coordinates: np.ndarray


def obstacle_face_inputs(mesh: Mesh, obstacle: ObstacleSpec, face: str) -> ObstacleFace:
    """The faces of one obstacle face that border the room, for the vent layers.

    The obstacle-face counterpart of ``edge_cell_inputs``: every layer that
    reads an equipment vent (ADR-014 J) derives its faces here and asks the
    registry which segment covers each, so the velocity, turbulence and
    concentration layers cannot disagree about a vent's faces (REQ-S12.1).

    Parameters
    ----------
    mesh : Mesh
        Supplies the coordinates, the widths, ``cell_type`` and the
        obstacle's cells (``Mesh.obstacle_mask``).
    obstacle : ObstacleSpec
        The obstacle.
    face : str
        One of "top", "bottom", "left", "right".

    Returns
    -------
    ObstacleFace
        Empty arrays when no face of the obstacle on that side borders a
        non-SOLID cell (the face is on the domain edge or covered).

    Raises
    ------
    ValueError
        On an unknown face name.
    """
    cells = mesh.obstacle_mask(obstacle)
    open_cell = mesh.cell_type != SOLID
    if face == "top":
        rows, cols = np.nonzero(cells[:-1, :] & open_cell[1:, :])
        rows = rows + 1
    elif face == "bottom":
        rows, cols = np.nonzero(cells[1:, :] & open_cell[:-1, :])
        rows = rows + 1
    elif face == "left":
        rows, cols = np.nonzero(cells[:, 1:] & open_cell[:, :-1])
        cols = cols + 1
    elif face == "right":
        rows, cols = np.nonzero(cells[:, :-1] & open_cell[:, 1:])
        cols = cols + 1
    else:
        raise ValueError(f"unknown face '{face}'; use top, bottom, left or right")
    horizontal = face in ("top", "bottom")
    # Along the face: by column on a top or bottom face, by row on a side.
    order = np.lexsort((cols, rows) if horizontal else (rows, cols))
    rows, cols = rows[order].astype(np.int64), cols[order].astype(np.int64)
    # Two neighbouring faces of the run bound one tangential stencil face
    # between them, at the corner they share.
    if horizontal:
        shared = (rows[1:] == rows[:-1]) & (cols[1:] == cols[:-1] + 1)
        corner_rows, corner_cols = rows[1:][shared], cols[1:][shared]
        coordinates, widths = mesh.xc[cols], mesh.dx_cell[cols]
        corner_coordinates = mesh.x[corner_cols]
    else:
        shared = (cols[1:] == cols[:-1]) & (rows[1:] == rows[:-1] + 1)
        corner_rows, corner_cols = rows[1:][shared], cols[1:][shared]
        coordinates, widths = mesh.yc[rows], mesh.dy_cell[rows]
        corner_coordinates = mesh.y[corner_rows]
    return ObstacleFace(
        component="v" if horizontal else "u",
        rows=rows,
        cols=cols,
        coordinates=np.array(coordinates, dtype=np.float64),
        widths=np.array(widths, dtype=np.float64),
        corner_rows=corner_rows,
        corner_cols=corner_cols,
        corner_coordinates=np.array(corner_coordinates, dtype=np.float64),
    )


def check_staggered_pair(u: np.ndarray, v: np.ndarray) -> None:
    """Raise unless u and v are a consistent staggered pair.

    Parameters
    ----------
    u : np.ndarray
        x-velocity on vertical faces, shape [ny, nx+1].
    v : np.ndarray
        y-velocity on horizontal faces, shape [ny+1, nx].

    Raises
    ------
    ValueError
        If either array is not 2D, or the shapes do not describe one mesh.
    """
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("u and v must be 2D arrays")
    ny, nxp1 = u.shape
    nyp1, nx = v.shape
    if nxp1 != nx + 1 or nyp1 != ny + 1:
        raise ValueError(
            f"inconsistent staggered shapes: u {u.shape} expects v ({ny + 1}, "
            f"{nxp1 - 1}), got v {v.shape}"
        )


@dataclass(frozen=True, eq=False)
class FaceVelocities:
    """Face velocities on their staggered storage locations (REQ-S13, ADR-011 A).

    The velocity solver sets one at the end of every solve, converged or
    not, and the transport solver advects with it: the fluxes continuity
    was enforced on, not a reconstruction from the cell means. Build one
    with ``copy_of``, which copies the arrays and clears their writeable
    flag; the constructor refuses anything that does not already meet the
    contract below, so every instance meets it.

    Parameters
    ----------
    u : np.ndarray
        x-velocity on vertical faces, shape [ny, nx+1], float64,
        C-contiguous, read-only.
    v : np.ndarray
        y-velocity on horizontal faces, shape [ny+1, nx], float64,
        C-contiguous, read-only.

    Raises
    ------
    ValueError
        If the shapes are not a consistent staggered pair, or either array
        is not float64, C-contiguous, read-only and the owner of its data.
        A read-only view of a writeable array is refused: its owner could
        still change the instance.
    """

    u: np.ndarray
    v: np.ndarray

    def __post_init__(self) -> None:
        check_staggered_pair(self.u, self.v)
        for name, arr in (("u", self.u), ("v", self.v)):
            if arr.dtype != np.float64:
                raise ValueError(
                    f"FaceVelocities.{name} must be float64, got {arr.dtype}"
                )
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    f"FaceVelocities.{name} must be C-contiguous "
                    f"(C_CONTIGUOUS is {arr.flags['C_CONTIGUOUS']})"
                )
            if arr.flags.writeable:
                raise ValueError(
                    f"FaceVelocities.{name} must be read-only; build with copy_of"
                )
            if not arr.flags.owndata:
                raise ValueError(
                    f"FaceVelocities.{name} must own its data, not view another "
                    "array's; build with copy_of"
                )

    @classmethod
    def copy_of(cls, u: np.ndarray, v: np.ndarray) -> "FaceVelocities":
        """Read-only float64, C-contiguous copies of a staggered pair.

        Parameters
        ----------
        u : np.ndarray
            x-velocity on vertical faces, shape [ny, nx+1].
        v : np.ndarray
            y-velocity on horizontal faces, shape [ny+1, nx].

        Returns
        -------
        FaceVelocities
            Owning copies; the arguments are not modified and later writes
            to them do not reach the copies.
        """
        u_copy = np.array(u, dtype=np.float64, order="C", copy=True)
        v_copy = np.array(v, dtype=np.float64, order="C", copy=True)
        u_copy.flags.writeable = False
        v_copy.flags.writeable = False
        return cls(u_copy, v_copy)


def to_cell_centers(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average face velocities to cell centers.

    Parameters
    ----------
    u : np.ndarray
        x-velocity on vertical faces, shape [ny, nx+1].
    v : np.ndarray
        y-velocity on horizontal faces, shape [ny+1, nx].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (u_c, v_c), each shape [ny, nx], dtype float64, C-contiguous.

    Notes
    -----
    ``u_c[j, i] = 0.5 (u[j, i] + u[j, i+1])`` and
    ``v_c[j, i] = 0.5 (v[j, i] + v[j+1, i])``. Cell centers are face
    midpoints, so the plain average is the second-order interpolant on any
    mesh from src/mesh.py; no width weighting is needed or wanted.
    """
    check_staggered_pair(u, v)
    u_c = 0.5 * (u[:, :-1] + u[:, 1:])
    v_c = 0.5 * (v[:-1, :] + v[1:, :])
    return (
        np.ascontiguousarray(u_c, dtype=np.float64),
        np.ascontiguousarray(v_c, dtype=np.float64),
    )
