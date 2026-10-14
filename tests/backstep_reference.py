"""A constructed solution of the VAL-019 case and every number the comparison reads from it.

``validation.backstep.evaluate`` turns a solved field into the report's tables
and the harness row. A test that checks its keys and counts cannot tell a
profile compared with the wrong column, or a coefficient scaled by the wrong
speed, from the right one. This module builds one field on the case's medium
grid whose every comparison number differs from its neighbours' (a centreline
speed off U_ref, a pressure that varies with the row, a turbulence field that
varies along the wall), and computes each number again with NumPy alone:
separable ``np.interp`` for the sampling, loops and sums for the thicknesses
and the reattachment, nothing from ``validation.backstep`` except the stored
data it compares with. tests/test_backward_step.py and tests/test_benchmark.py
assert ``evaluate`` and the harness row against it.

The geometry is the case's by construction (step height H, the step face at
114 H, the data's frame x/H = 0 at the face), stated here and not read back
from the code under test.
"""

import numpy as np

from src.config import SimConfig
from src.mesh import Mesh
from src.turbulence import VARIANTS, wall_viscosity
from validation import backstep

H = 0.0127
STEP_FACE_H = 114.0
# The channel's centreline speed in the constructed field. Off U_ref (44.2) by
# 10% so a normalisation by the constant shows.
CENTRE = 40.0
# The layer's e-folding height, in H: u = CENTRE (1 - exp(-d / (0.4 H))) at
# the height d above the wall, so delta_99 = 0.4 ln(100) H = 1.84 H and the
# momentum thickness is 0.4 H / 2 = 0.2 H.
LAYER = 0.4
ZERO_CROSSING_H = 5.1
PRESSURE_SCALE = 300.0
DATUM_H = 37.5


def fields(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The constructed u on the vertical faces, p and k on the cells.

    Parameters
    ----------
    mesh : Mesh
        The case's medium grid (face columns on 0.25 H, a face row on H).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``u`` [ny, nx+1]: a layer growing from the step's top upstream and
        from the floor downstream, zero over the block, the floor row
        downstream reversed up to x/H = 5.1 and positive past it (so the
        wall shear changes sign inside a face pair); ``p`` [ny, nx]: a
        wave in x with a ramp in y, so the bottom wall's two rows and the
        ceiling differ; ``k`` [ny, nx]: positive, varying along and across
        the wall rows, so the median of y* over a stretch of floor depends
        on where the stretch begins.
    """
    x_face = (mesh.x - STEP_FACE_H * H) / H
    x_cell = (mesh.xc - STEP_FACE_H * H) / H
    ny = mesh.yc.size
    height = np.broadcast_to(mesh.yc[:, None], (ny, mesh.x.size)).copy()
    left = x_face <= 0.0
    height[:, left] -= H
    height = np.maximum(height, 0.0)
    u = CENTRE * (1.0 - np.exp(-height / (LAYER * H)))
    u[:, ~left] *= 1.0 + 0.003 * x_face[~left]
    u[np.ix_(mesh.yc < H, left)] = 0.0
    u[0, ~left] = 0.6 * (x_face[~left] - ZERO_CROSSING_H)
    p = PRESSURE_SCALE * np.sin(x_cell / 9.0)[None, :] + 40.0 * (mesh.yc / H)[:, None]
    k = (3.0 + 2.0 * np.sin(x_cell / 7.0))[None, :] * (
        1.0 + 0.2 * np.cos(np.arange(ny))
    )[:, None]
    return u, p, k


def bilinear(mesh: Mesh, u: np.ndarray, x_over_h: float, y_over_h: float) -> float:
    """u at one point: np.interp along x on every row, then along y.

    The rows are the cell-centre rows with a zero wall row added on the floor
    and on the ceiling, the lattice the report samples on.
    """
    x = STEP_FACE_H * H + x_over_h * H
    rows = np.vstack((np.zeros(mesh.x.size), u, np.zeros(mesh.x.size)))
    heights = np.concatenate(([mesh.y[0]], mesh.yc, [mesh.y[-1]]))
    column = np.array([np.interp(x, mesh.x, row) for row in rows])
    return float(np.interp(y_over_h * H, heights, column))


def differences(values: np.ndarray, reference: tuple[float, ...]) -> dict:
    """The largest and RMS difference, and the count, as the report stores them."""
    diff = np.asarray(values, dtype=np.float64) - np.asarray(reference)
    return {
        "max_abs": float(np.abs(diff).max()),
        "rms": float(np.sqrt(np.mean(diff**2))),
        "stations": int(diff.size),
    }


def expected(
    mesh: Mesh,
    config: SimConfig,
    u: np.ndarray,
    p: np.ndarray,
    k: np.ndarray,
) -> dict:
    """The report ``backstep.evaluate`` must return for these fields, computed again.

    Parameters
    ----------
    mesh : Mesh
        The case's medium grid.
    config : SimConfig
        The case, for the air's density and viscosity.
    u, p, k : np.ndarray
        The fields of ``fields``.

    Returns
    -------
    dict
        The same nested dictionary as ``evaluate``, leaf for leaf.
    """
    rho, nu = config.rho, config.mu / config.rho
    c_mu = VARIANTS["standard"].c_mu
    j_up = int(np.argmin(np.abs(mesh.y - H)))
    i_step = int(np.argmin(np.abs(mesh.x - STEP_FACE_H * H)))
    x_cell = (mesh.xc - STEP_FACE_H * H) / H

    # Inflow: the centreline at x/H = -4, mid-height of the 8 H channel.
    u_ref = bilinear(mesh, u, -4.0, 5.0)
    i_ref = int(np.argmin(np.abs(mesh.x - (STEP_FACE_H - 4.0) * H)))
    rows = [j for j in range(j_up, mesh.yc.size) if mesh.yc[j] <= 5.0 * H]
    height = np.concatenate(([0.0], [mesh.yc[j] - H for j in rows]))
    speed = np.concatenate(([0.0], [u[j, i_ref] for j in rows])) / u_ref
    delta_99 = float(np.interp(0.99, speed, height))
    integrand = speed * (1.0 - speed)
    theta = float(
        sum(
            0.5 * (integrand[n] + integrand[n + 1]) * (height[n + 1] - height[n])
            for n in range(height.size - 1)
        )
    )

    # Wall shear on the two rows, every interior face but the step's own.
    x_shear, tau = [], []
    for j, faces in ((j_up, range(1, i_step)), (0, range(i_step + 1, mesh.x.size - 1))):
        y_p = mesh.yc[j] - mesh.y[j]
        for i in faces:
            k_p = 0.5 * (k[j, i - 1] + k[j, i])
            mu_w = wall_viscosity(np.array([k_p]), np.array([y_p]), c_mu, rho, nu)[0]
            x_shear.append((mesh.x[i] - STEP_FACE_H * H) / H)
            tau.append(mu_w * u[j, i] / y_p)
    x_shear, tau = np.array(x_shear), np.array(tau)
    crossings = [n for n in range(tau.size - 1) if tau[n] < 0.0 <= tau[n + 1]]
    n = crossings[-1]
    x_r = float(
        x_shear[n] + (x_shear[n + 1] - x_shear[n]) * tau[n] / (tau[n] - tau[n + 1])
    )

    cf = tau / (0.5 * rho * u_ref**2)
    wall_p = np.concatenate((p[j_up, :i_step], p[0, i_step:]))
    datum = np.interp(DATUM_H, x_cell, wall_p)
    cp = (wall_p - datum) / (0.5 * rho * u_ref**2)

    profiles = {}
    for station, profile in backstep.PROFILES.items():
        values = [bilinear(mesh, u, station, y) / u_ref for y in profile.y_over_h]
        profiles[str(station)] = differences(np.array(values), profile.u_over_uref)

    def y_star(j: int, cols: np.ndarray) -> np.ndarray:
        if j == mesh.yc.size - 1:
            y_p = mesh.y[-1] - mesh.yc[-1]
        else:
            y_p = mesh.yc[j] - mesh.y[j]
        return c_mu**0.25 * np.sqrt(k[j, cols]) * y_p / nu

    top = mesh.yc.size - 1
    near_ref = int(np.argmin(np.abs(x_cell + 4.0)))
    near_ten = int(np.argmin(np.abs(x_cell - 10.0)))
    return {
        "u_ref": {"computed": u_ref, "over_measured": u_ref / 44.2},
        "inflow": {
            "delta_99_over_h": delta_99 / H,
            "re_theta": u_ref * theta / nu,
            "delta_99_measured": backstep.DELTA_99_MEASURED,
            "re_theta_measured": backstep.RE_THETA_MEASURED,
        },
        "x_r_over_h": x_r,
        "x_r_measured": 6.26,
        "x_r_uncertainty": 0.10,
        "profiles": profiles,
        "cf": differences(
            np.interp(backstep.CF_STATIONS, x_shear, cf), backstep.CF_MEASURED
        ),
        "cp": differences(
            np.interp(backstep.CP_BOTTOM_STATIONS, x_cell, cp),
            backstep.CP_BOTTOM_MEASURED,
        ),
        "y_star": {
            "floor_downstream": float(
                np.median(y_star(0, np.nonzero(x_cell > x_r)[0]))
            ),
            "floor_at_station": float(y_star(0, np.array([near_ten]))[0]),
            "step_top": float(np.median(y_star(j_up, np.arange(i_step)))),
            "step_top_at_station": float(y_star(j_up, np.array([near_ref]))[0]),
            "ceiling_upstream": float(
                np.median(y_star(top, np.nonzero(x_cell < 0.0)[0]))
            ),
            "ceiling_at_station": float(y_star(top, np.array([near_ref]))[0]),
            "floor": backstep.Y_STAR_FLOOR,
        },
    }


def assert_matches(got: object, want: object, path: str = "report") -> None:
    """Assert two nested reports agree leaf for leaf, floats to 1e-10 relative."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), path
        for key, value in want.items():
            assert_matches(got[key], value, f"{path}[{key!r}]")
    elif isinstance(want, float):
        assert isinstance(got, float), f"{path}: {got!r}"
        assert abs(got - want) <= 1e-10 * max(abs(want), 1e-300) + 1e-14, (
            f"{path}: {got!r} against {want!r}"
        )
    else:
        assert got == want, f"{path}: {got!r} against {want!r}"
