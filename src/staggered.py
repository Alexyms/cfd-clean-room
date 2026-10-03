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

This module holds no discretization. It computes no flux, no gradient and
no coefficient; ECR-001 steps 3 to 5 own those.
"""

from dataclasses import dataclass

import numpy as np

from src.mesh import Mesh


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


@dataclass(frozen=True)
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
        is not float64, C-contiguous and read-only.
    """

    u: np.ndarray
    v: np.ndarray

    def __post_init__(self) -> None:
        check_staggered_pair(self.u, self.v)
        for name, arr in (("u", self.u), ("v", self.v)):
            if arr.dtype != np.float64 or not arr.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    f"FaceVelocities.{name} must be float64 and C-contiguous, "
                    f"got {arr.dtype} with flags {arr.flags['C_CONTIGUOUS']}"
                )
            if arr.flags.writeable:
                raise ValueError(
                    f"FaceVelocities.{name} must be read-only; build with copy_of"
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
