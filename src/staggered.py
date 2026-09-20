"""Staggered (MAC) field layout: shapes, allocation and face-to-center averaging.

Layout (REQ-S07). For a mesh with nx by ny cells:

``u``  x-velocity on the vertical (east-west) faces, shape [ny, nx+1].
       ``u[j, i]`` sits at ``(x[i], yc[j])``; ``u[j, 0]`` and ``u[j, nx]``
       are on the left and right domain boundaries.
``v``  y-velocity on the horizontal (north-south) faces, shape [ny+1, nx].
       ``v[j, i]`` sits at ``(xc[i], y[j])``; ``v[0, i]`` and ``v[ny, i]``
       are on the bottom and top domain boundaries.
``p``  pressure at cell centers, shape [ny, nx], at ``(xc[i], yc[j])``.

The layout is internal to the solver. The public ``solve_steady`` contract
returns cell-centered fields of shape [ny, nx], so the solver averages u
and v to cell centers with ``to_cell_centers`` before returning. Because
every cell center is the midpoint of its two bounding faces (see
src/mesh.py), that average is second-order accurate on uniform and
stretched meshes alike and reproduces a field that is linear in the
coordinate exactly.

This module holds no discretization. It computes no flux, no gradient and
no coefficient; ECR-001 steps 3 to 5 own those.
"""

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
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("u and v must be 2D arrays")
    ny, nxp1 = u.shape
    nyp1, nx = v.shape
    if nxp1 != nx + 1 or nyp1 != ny + 1:
        raise ValueError(
            f"inconsistent staggered shapes: u {u.shape} expects v ({ny + 1}, "
            f"{nxp1 - 1}), got v {v.shape}"
        )
    u_c = 0.5 * (u[:, :-1] + u[:, 1:])
    v_c = 0.5 * (v[:-1, :] + v[1:, :])
    return (
        np.ascontiguousarray(u_c, dtype=np.float64),
        np.ascontiguousarray(v_c, dtype=np.float64),
    )
