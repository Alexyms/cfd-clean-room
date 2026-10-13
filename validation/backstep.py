"""The backward-facing step's data, geometry and comparison (VAL-019).

The case is ADR-012 G (ii): Driver and Seegmiller's (1985) step as the NASA
Turbulence Modeling Resource poses it. Step height H = 0.0127 m; an
upstream channel 8 H tall whose floor is the step's top, a downstream
channel 9 H tall (expansion ratio 1.125), both walls straight; U_ref 44.2
m/s at the channel centre at x/H = -4; Re_H 36,000, so nu = U_ref H /
36,000; the boundary layer before the step about 1.5 H thick at Re_theta
5,000; reattachment at x/H = 6.26 +/- 0.10 by laser oil-flow
interferometry. The profiles are at x/H = -4, 1, 4, 6 and 10, the skin
friction and the pressure coefficient along the bottom wall (and the
pressure along the top wall too).

Frames. The data's x is x/H from the step's face and its y is y/H from the
downstream floor, so the upstream floor (the step's top) is y/H = 1. The
solver's frame has its origin at the floor of the inlet wall, so the data's
x/H = 0 is the step obstacle's ``x_end`` (L_up, the upstream length the
precursor set) and y is the same. ``case_x`` and ``data_x`` convert.

Normalisation. Every comparison divides by the computed centreline speed
at x/H = -4 (``centreline_speed``), as the data divide by U_ref measured
there; the inlet's uniform speed is set so that this speed is U_ref within
1%, and ``inflow_check`` reports the ratio with the layer's delta_99 and
Re_theta at the same station.

Provenance of the data: the three files the Resource publishes for the
case, fetched unchanged on 2026-10-12 and stored with their addresses and
SHA-256 checksums in validation/data/driver_seegmiller_1985/PROVENANCE.md.
The tuples below were printed from those files by
docs/reports/probe52/data52.py; nothing was typed. PROFILE_ROWS holds, per
station, (y/H, u/U_ref, 1000 u'v'/U_ref^2); CF_ROWS (x/H, Cf, its error);
the CP rows (x/H, Cp shifted so it is zero near x/H = 40, Cp referenced to
x/H = -4). The published k-epsilon results beside them (PUBLISHED_X_R) are
the NPARC Alliance Validation Archive's (https://www.grc.nasa.gov/www/wind/
valid/backstep/backstep01/backstep01.html, read 2026-10-12): Chien
k-epsilon in NPARC and in WIND, WIND's variable-C_mu option and Menter SST,
on a 238 x 185 grid clustered to y+ = 1 from x/H = -105 to +50. The Chien
model is a low-Reynolds-number k-epsilon integrated to the wall, not the
standard model with wall functions this solver runs; every table that
cites those numbers says so.

No threshold is set here: the allowance is Alex's, OPEN until the first
coupled results (ADR-012 G (ii), decision 6). ``evaluate`` reports; it
passes or fails nothing.
"""

from dataclasses import dataclass

import numpy as np

from src.config import ObstacleSpec, SimConfig
from src.mesh import Mesh
from src.turbulence import VARIANTS, Y_STAR_FLOOR, wall_viscosity

# The case (ADR-012 G (ii); the Resource's page, PROVENANCE.md).
STEP_HEIGHT = 0.0127
U_REF = 44.2
RE_H = 36000.0
NU = U_REF * STEP_HEIGHT / RE_H
UPSTREAM_HEIGHTS = 8.0
DOWNSTREAM_HEIGHTS = 9.0
OUTLET_STATION = 50.0
REFERENCE_STATION = -4.0
PROFILE_STATIONS: tuple[float, ...] = (1.0, 4.0, 6.0, 10.0)
X_R_MEASURED = 6.26
X_R_UNCERTAINTY = 0.10
DELTA_99_MEASURED = 1.5
RE_THETA_MEASURED = 5000.0
# The data's shifted Cp is exactly zero at its last bottom-wall station.
CP_ZERO_STATION = 37.5
# The fraction of the centreline speed that defines the layer's edge.
EDGE_FRACTION = 0.99
# Half the width of the band about x/H = -4 the upstream y+ is read in.
STATION_HALF_WIDTH = 0.5

# Published reattachment lengths, x_r / H (the NPARC archive; the caveat in
# the module docstring: Chien's is a low-Reynolds k-epsilon, not the standard
# model with wall functions).
PUBLISHED_X_R: dict[str, float] = {
    "NPARC Chien k-epsilon (low-Re, to the wall)": 5.31,
    "WIND Chien k-epsilon (low-Re, to the wall)": 5.30,
    "WIND Chien k-epsilon, variable C_mu (low-Re)": 5.55,
    "WIND Menter SST": 6.43,
}

PROFILE_ROWS: dict[float, tuple[tuple[float, float, float], ...]] = {
    -4.0: (
        (1.1, 0.657, -1.02),
        (1.15, 0.696, -1.01),
        (1.2, 0.719, -1.07),
        (1.3, 0.76, -1.07),
        (1.4, 0.79, -0.97),
        (1.5, 0.818, -0.92),
        (1.7, 0.87, -0.71),
        (2.0, 0.926, -0.44),
        (2.4, 0.982, -0.1),
        (2.8, 1.003, 0.02),
        (3.2, 1.003, 0.03),
        (3.6, 1.0, 0.04),
        (4.0, 1.001, 0.03),
        (5.0, 1.001, 0.03),
        (6.0, 1.0, 0.04),
        (7.5, 1.001, 0.04),
        (8.2, 0.943, 0.46),
    ),
    1.0: (
        (0.1, -0.008, -0.72),
        (0.15, -0.017, 0.46),
        (0.2, -0.027, 0.21),
        (0.3, -0.035, -0.14),
        (0.4, -0.035, -0.43),
        (0.5, -0.035, -0.65),
        (0.6, -0.027, -0.74),
        (0.7, -0.005, -0.92),
        (0.8, 0.081, -3.71),
        (0.9, 0.292, -7.05),
        (1.0, 0.519, -5.85),
        (1.1, 0.704, -1.54),
        (1.15, 0.74, -1.25),
        (1.2, 0.762, -1.05),
        (1.3, 0.793, -0.98),
        (1.4, 0.827, -0.82),
        (1.5, 0.851, -0.81),
        (1.7, 0.891, -0.64),
        (2.0, 0.942, -0.44),
        (2.4, 0.99, -0.12),
        (2.8, 1.009, 0.01),
        (3.2, 1.003, 0.04),
        (3.6, 1.002, 0.03),
        (4.0, 1.0, 0.03),
        (5.0, 0.997, 0.01),
        (6.0, 0.994, -0.01),
        (7.5, 0.993, 0.01),
        (8.2, 0.919, 0.59),
    ),
    4.0: (
        (0.1, -0.171, -2.64),
        (0.15, -0.154, -4.45),
        (0.2, -0.134, -6.3),
        (0.3, -0.057, -9.94),
        (0.4, 0.052, -12.02),
        (0.5, 0.169, -11.11),
        (0.6, 0.279, -8.99),
        (0.7, 0.381, -6.93),
        (0.8, 0.473, -5.98),
        (0.9, 0.568, -4.52),
        (1.0, 0.656, -3.48),
        (1.1, 0.73, -2.1),
        (1.15, 0.754, -1.65),
        (1.2, 0.778, -1.19),
        (1.3, 0.811, -0.87),
        (1.4, 0.835, -0.67),
        (1.5, 0.856, -0.56),
        (1.7, 0.895, -0.45),
        (2.0, 0.94, -0.23),
        (2.4, 0.978, 0.0),
        (2.8, 0.986, 0.06),
        (3.2, 0.984, 0.05),
        (3.6, 0.985, 0.07),
        (4.0, 0.985, 0.06),
        (5.0, 0.983, 0.06),
        (6.0, 0.98, 0.06),
        (7.5, 0.98, 0.07),
        (8.2, 0.892, 0.65),
    ),
    6.0: (
        (0.1, 0.147, -4.1),
        (0.15, 0.169, -5.48),
        (0.2, 0.188, -6.79),
        (0.3, 0.248, -7.86),
        (0.4, 0.306, -8.23),
        (0.5, 0.369, -8.24),
        (0.6, 0.435, -7.36),
        (0.7, 0.501, -6.57),
        (0.8, 0.569, -5.06),
        (0.9, 0.632, -3.79),
        (1.0, 0.686, -2.8),
        (1.1, 0.73, -1.76),
        (1.15, 0.747, -1.51),
        (1.2, 0.763, -1.25),
        (1.3, 0.794, -0.8),
        (1.4, 0.815, -0.65),
        (1.5, 0.834, -0.61),
        (1.7, 0.869, -0.43),
        (2.0, 0.915, -0.21),
        (2.4, 0.949, -0.04),
        (2.8, 0.954, 0.03),
        (3.2, 0.956, 0.03),
        (3.6, 0.959, 0.03),
        (4.0, 0.961, 0.02),
        (5.0, 0.963, 0.02),
        (6.0, 0.965, 0.02),
        (7.5, 0.969, 0.03),
        (8.2, 0.87, 0.76),
    ),
    10.0: (
        (0.1, 0.418, -1.43),
        (0.15, 0.433, -1.84),
        (0.2, 0.447, -2.35),
        (0.3, 0.471, -3.43),
        (0.4, 0.496, -4.19),
        (0.5, 0.526, -4.51),
        (0.6, 0.56, -4.61),
        (0.7, 0.596, -4.61),
        (0.8, 0.638, -4.3),
        (0.9, 0.676, -3.68),
        (1.0, 0.712, -2.94),
        (1.1, 0.745, -2.33),
        (1.15, 0.758, -2.02),
        (1.2, 0.772, -1.71),
        (1.3, 0.798, -1.36),
        (1.4, 0.819, -0.96),
        (1.5, 0.839, -0.72),
        (1.7, 0.871, -0.35),
        (2.0, 0.908, -0.14),
        (2.4, 0.929, 0.02),
        (2.8, 0.933, 0.04),
        (3.2, 0.937, 0.04),
        (3.6, 0.939, 0.04),
        (4.0, 0.94, 0.02),
        (5.0, 0.944, 0.03),
        (6.0, 0.947, 0.03),
        (7.5, 0.949, 0.05),
        (8.2, 0.827, 0.95),
    ),
}

CF_ROWS: tuple[tuple[float, float, float], ...] = (
    (-3.956, 0.00288, 0.000202),
    (-1.804, 0.00285, 0.000171),
    (-0.804, 0.00311, 0.000311),
    (0.484, 9e-05, 1.4e-05),
    (1.804, -0.00011, 1.9e-05),
    (2.804, -0.00052, 0.000109),
    (3.804, -0.00102, 0.000163),
    (4.804, -0.00082, 0.000123),
    (5.882, -0.00022, 6.2e-05),
    (7.09, 0.00045, 0.000167),
    (7.978, 0.00102, 0.000122),
    (9.196, 0.00118, 0.000106),
    (11.196, 0.00154, 0.000123),
    (13.196, 0.00173, 0.000121),
    (16.196, 0.00191, 0.000134),
    (20.196, 0.00202, 0.000141),
    (24.136, 0.00204, 0.000143),
    (28.13, 0.00205, 0.000205),
    (32.196, 0.00212, 0.000148),
    (35.994, 0.0023, 0.000184),
)

CP_BOTTOM_ROWS: tuple[tuple[float, float, float], ...] = (
    (-8.5, -0.1102, 0.0039),
    (-6.5, -0.1141, 0.0),
    (-4.5, -0.1189, -0.0048),
    (-2.5, -0.1372, -0.0231),
    (-0.5, -0.1613, -0.0472),
    (0.5, -0.1777, -0.0636),
    (1.0, -0.1806, -0.0665),
    (2.0, -0.1883, -0.0742),
    (3.0, -0.1903, -0.0762),
    (3.5, -0.1806, -0.0665),
    (4.0, -0.1565, -0.0424),
    (4.5, -0.1064, 0.0077),
    (5.0, -0.0659, 0.0482),
    (5.5, -0.0359, 0.0782),
    (6.0, -0.0012, 0.1129),
    (6.5, 0.0162, 0.1303),
    (7.0, 0.0248, 0.1389),
    (8.0, 0.0374, 0.1515),
    (9.0, 0.0394, 0.1535),
    (11.0, 0.0336, 0.1477),
    (13.0, 0.0268, 0.1409),
    (15.0, 0.0201, 0.1342),
    (17.0, 0.0162, 0.1303),
    (19.5, 0.0144, 0.1285),
    (21.5, 0.0105, 0.1246),
    (23.5, 0.0086, 0.1227),
    (25.5, 0.0086, 0.1227),
    (27.5, 0.0077, 0.1218),
    (29.5, 0.0077, 0.1218),
    (31.5, 0.0058, 0.1199),
    (33.5, 0.0019, 0.116),
    (35.5, 0.0048, 0.1189),
    (37.5, 0.0, 0.1141),
)
CP_TOP_ROWS: tuple[tuple[float, float, float], ...] = (
    (-5.0, -0.1054, 0.0087),
    (3.0, -0.0727, 0.0414),
    (5.0, -0.063, 0.0511),
    (7.0, -0.0543, 0.0598),
    (9.0, -0.0331, 0.081),
    (11.0, -0.0148, 0.0993),
    (13.0, -0.0023, 0.1118),
    (15.0, -0.0003, 0.1138),
    (17.0, 0.0065, 0.1206),
    (19.0, 0.0075, 0.1216),
    (21.0, 0.0075, 0.1216),
    (23.0, 0.0066, 0.1207),
    (25.0, 0.0095, 0.1236),
    (27.0, 0.0076, 0.1217),
    (29.0, 0.0067, 0.1208),
    (31.0, 0.0048, 0.1189),
    (33.0, 0.0019, 0.116),
    (35.0, 0.0038, 0.1179),
    (37.0, -0.001, 0.1131),
)


@dataclass(frozen=True)
class Profile:
    """One measured profile: the data's rows at one station.

    Parameters
    ----------
    station : float
        x/H of the profile.
    y_over_h : tuple[float, ...]
        y/H from the downstream floor, in the file's order (upward).
    u_over_uref : tuple[float, ...]
        Measured u / U_ref at each height.
    uv : tuple[float, ...]
        Measured 1000 u'v' / U_ref^2 at each height.
    """

    station: float
    y_over_h: tuple[float, ...]
    u_over_uref: tuple[float, ...]
    uv: tuple[float, ...]


PROFILES: dict[float, Profile] = {
    station: Profile(
        station=station,
        y_over_h=tuple(r[0] for r in rows),
        u_over_uref=tuple(r[1] for r in rows),
        uv=tuple(r[2] for r in rows),
    )
    for station, rows in PROFILE_ROWS.items()
}

CF_STATIONS: tuple[float, ...] = tuple(r[0] for r in CF_ROWS)
CF_MEASURED: tuple[float, ...] = tuple(r[1] for r in CF_ROWS)
CF_ERROR: tuple[float, ...] = tuple(r[2] for r in CF_ROWS)
CP_BOTTOM_STATIONS: tuple[float, ...] = tuple(r[0] for r in CP_BOTTOM_ROWS)
CP_BOTTOM_MEASURED: tuple[float, ...] = tuple(r[1] for r in CP_BOTTOM_ROWS)
CP_TOP_STATIONS: tuple[float, ...] = tuple(r[0] for r in CP_TOP_ROWS)
CP_TOP_MEASURED: tuple[float, ...] = tuple(r[1] for r in CP_TOP_ROWS)


@dataclass(frozen=True)
class Errors:
    """The difference of a computed line from a measured one at its stations.

    Parameters
    ----------
    max_abs : float
        The largest absolute difference.
    rms : float
        The root mean square of the differences.
    stations : int
        How many stations the two sets hold.
    """

    max_abs: float
    rms: float
    stations: int


def errors(values: np.ndarray, reference: np.ndarray) -> Errors:
    """The largest and the root-mean-square difference of two sets of station values.

    Parameters
    ----------
    values : np.ndarray
        Computed values, one per station.
    reference : np.ndarray
        Measured values at the same stations.

    Returns
    -------
    Errors
        ``values - reference``.

    Raises
    ------
    ValueError
        If the two sets differ in length, are empty, or hold a non-finite
        value: a NaN would make the RMS NaN and say nothing.
    """
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if values.shape != reference.shape or values.ndim != 1:
        raise ValueError(f"{values.shape} values against {reference.shape} stations")
    if values.size == 0:
        raise ValueError("no stations")
    if not (np.isfinite(values).all() and np.isfinite(reference).all()):
        raise ValueError("a station value is not finite")
    diff = values - reference
    return Errors(
        max_abs=float(np.abs(diff).max()),
        rms=float(np.sqrt(np.mean(diff**2))),
        stations=int(values.size),
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def step(config: SimConfig) -> ObstacleSpec:
    """The case's step obstacle, the one named "step".

    Parameters
    ----------
    config : SimConfig
        A loaded step case.

    Returns
    -------
    ObstacleSpec
        The block from the inlet wall to the step's face.

    Raises
    ------
    ValueError
        If the case has no obstacle of that name, or more than one.
    """
    found = [o for o in config.obstacles if o.name == "step"]
    if len(found) != 1:
        raise ValueError(
            f"the case needs one obstacle named 'step', found {len(found)}"
        )
    return found[0]


def upstream_length(config: SimConfig) -> float:
    """L_up: the step face's x in the solver's frame, m."""
    return float(step(config).x_end)


def case_x(config: SimConfig, x_over_h: np.ndarray | float) -> np.ndarray:
    """The solver-frame x of a data station x/H."""
    return (
        upstream_length(config)
        + np.asarray(x_over_h, dtype=np.float64) * step(config).y_end
    )


def data_x(config: SimConfig, x: np.ndarray | float) -> np.ndarray:
    """The data-frame x/H of a solver-frame x."""
    return (np.asarray(x, dtype=np.float64) - upstream_length(config)) / step(
        config
    ).y_end


def step_row(mesh: Mesh, config: SimConfig) -> int:
    """The index of the first cell row above the step's top.

    Parameters
    ----------
    mesh : Mesh
        The case's grid.
    config : SimConfig
        The case.

    Returns
    -------
    int
        The row whose lower face is the step height; its cells over the
        block are the step top's wall cells.

    Raises
    ------
    ValueError
        If no face lies on the step height within 1e-9 of a cell height,
        the corner being off the grid lines.
    """
    height = step(config).y_end
    j = int(np.argmin(np.abs(mesh.y - height)))
    if abs(mesh.y[j] - height) > 1e-9 * mesh.dy_cell.min():
        raise ValueError(f"no y face on the step height {height} m")
    return j


def step_column(mesh: Mesh, config: SimConfig) -> int:
    """The index of the x face on the step's face.

    Raises
    ------
    ValueError
        If no face lies on it within 1e-9 of a cell width.
    """
    length = upstream_length(config)
    i = int(np.argmin(np.abs(mesh.x - length)))
    if abs(mesh.x[i] - length) > 1e-9 * mesh.dx_cell.min():
        raise ValueError(f"no x face on the step's face at {length} m")
    return i


# ---------------------------------------------------------------------------
# Sampling the solution
# ---------------------------------------------------------------------------


def sample_u(
    mesh: Mesh, u_faces: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """u at points of the domain, bilinear in the staggered u storage.

    Parameters
    ----------
    mesh : Mesh
        The grid the faces belong to.
    u_faces : np.ndarray
        The x-velocity on the vertical faces, shape [ny, nx+1], ``u[j, i]``
        at ``(x[i], yc[j])``.
    x, y : np.ndarray
        The points, m, each of the same shape.

    Returns
    -------
    np.ndarray
        u at each point, m/s. The storage is extended by a row of zeros on
        the floor and on the ceiling, the no-slip walls, so a point between
        the outermost row of centres and a wall is interpolated to the wall.
        A face of a SOLID cell holds zero, so a point between the step's top
        and the first row of centres above it is interpolated toward a zero
        half a cell below the wall, not at it; no data station lies there.

    Notes
    -----
    Bilinear interpolation on the ``(x, yc)`` lattice, as VAL-017's sampler:
    exact for a field linear in x and y, and a station on a face or on a
    row of centres reads that face's or row's own value.
    """
    xs = np.asarray(mesh.x, dtype=np.float64)
    ys = np.concatenate(([float(mesh.y[0])], np.asarray(mesh.yc), [float(mesh.y[-1])]))
    field = np.zeros((ys.size, xs.size), dtype=np.float64)
    field[1:-1, :] = u_faces
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, xs.size - 2)
    j = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, ys.size - 2)
    tx = (x - xs[i]) / (xs[i + 1] - xs[i])
    ty = (y - ys[j]) / (ys[j + 1] - ys[j])
    return (
        (1.0 - tx) * (1.0 - ty) * field[j, i]
        + tx * (1.0 - ty) * field[j, i + 1]
        + (1.0 - tx) * ty * field[j + 1, i]
        + tx * ty * field[j + 1, i + 1]
    )


def centreline_speed(mesh: Mesh, config: SimConfig, u_faces: np.ndarray) -> float:
    """u at the upstream channel's centre at x/H = -4, the normalisation of every comparison.

    Parameters
    ----------
    mesh : Mesh
        The case's grid.
    config : SimConfig
        The case.
    u_faces : np.ndarray
        The x-velocity on the vertical faces, [ny, nx+1].

    Returns
    -------
    float
        u at (x(-4), the mid-height of the upstream channel), m/s.
    """
    h = step(config).y_end
    y = h + 0.5 * UPSTREAM_HEIGHTS * h
    x = case_x(config, REFERENCE_STATION)
    return float(sample_u(mesh, u_faces, np.array([x]), np.array([y]))[0])


@dataclass(frozen=True)
class InflowCheck:
    """The computed inflow at x/H = -4 against what the data describe.

    Parameters
    ----------
    u_centre : float
        The computed centreline speed, m/s.
    u_centre_over_uref : float
        Its ratio to U_ref, 44.2 m/s; the inlet speed is set so it is 1
        within 1%.
    delta_99_over_h : float
        The height above the step's top at which u first reaches 0.99 of the
        centreline speed, over H; the data describe about 1.5.
    re_theta : float
        The momentum-thickness Reynolds number ``u_centre theta / nu`` of the
        bottom wall's layer, theta integrated from the wall to mid-channel;
        the data describe 5,000.
    """

    u_centre: float
    u_centre_over_uref: float
    delta_99_over_h: float
    re_theta: float


def layer_thicknesses(
    y_from_wall: np.ndarray, u: np.ndarray, u_edge: float
) -> tuple[float, float]:
    """delta_99 and the momentum thickness of a wall layer from its profile.

    Parameters
    ----------
    y_from_wall : np.ndarray
        Heights above the wall, m, increasing, the first above zero.
    u : np.ndarray
        u at each height, m/s. The wall itself (zero, at zero height) is
        added here.
    u_edge : float
        The speed the layer tends to, m/s.

    Returns
    -------
    tuple[float, float]
        (delta_99, theta), m: the height at which u first reaches
        ``EDGE_FRACTION u_edge``, linear between the points; and the
        trapezoid integral of ``(u / u_edge)(1 - u / u_edge)`` from the wall
        to the last height.

    Raises
    ------
    ValueError
        If the profile never reaches the edge fraction.
    """
    yy = np.concatenate(([0.0], np.asarray(y_from_wall, dtype=np.float64)))
    uu = np.concatenate(([0.0], np.asarray(u, dtype=np.float64))) / u_edge
    above = np.nonzero(uu >= EDGE_FRACTION)[0]
    if above.size == 0 or above[0] == 0:
        raise ValueError("the profile does not reach the layer's edge")
    j = int(above[0])
    delta = yy[j - 1] + (yy[j] - yy[j - 1]) * (EDGE_FRACTION - uu[j - 1]) / (
        uu[j] - uu[j - 1]
    )
    theta = float(np.trapezoid(uu * (1.0 - uu), yy))
    return float(delta), theta


def inflow_check(mesh: Mesh, config: SimConfig, u_faces: np.ndarray) -> InflowCheck:
    """The inflow at x/H = -4: the centreline speed, delta_99 and Re_theta.

    The profile is read at the cell rows above the step's top on the face
    column nearest x(-4) (the nearest face, so each row's own value is
    read), from the first row to mid-channel; the wall itself is zero.
    """
    h = step(config).y_end
    j0 = step_row(mesh, config)
    x = float(case_x(config, REFERENCE_STATION))
    i = int(np.argmin(np.abs(mesh.x - x)))
    top = h + 0.5 * UPSTREAM_HEIGHTS * h
    rows = np.nonzero(mesh.yc[j0:] <= top)[0] + j0
    y = np.asarray(mesh.yc[rows]) - h
    u = np.asarray(u_faces[rows, i], dtype=np.float64)
    u_centre = centreline_speed(mesh, config, u_faces)
    delta, theta = layer_thicknesses(y, u, u_centre)
    nu = config.mu / config.rho
    return InflowCheck(
        u_centre=u_centre,
        u_centre_over_uref=u_centre / U_REF,
        delta_99_over_h=delta / h,
        re_theta=u_centre * theta / nu,
    )


def profile_values(
    mesh: Mesh, config: SimConfig, u_faces: np.ndarray, u_ref: float
) -> dict[float, np.ndarray]:
    """u / u_ref at every measured height of every profile station.

    Parameters
    ----------
    mesh : Mesh
        The case's grid.
    config : SimConfig
        The case.
    u_faces : np.ndarray
        The x-velocity on the vertical faces, [ny, nx+1].
    u_ref : float
        The computed centreline speed at x/H = -4, m/s.

    Returns
    -------
    dict[float, np.ndarray]
        For each station of PROFILES, u / u_ref at its heights, in order.
    """
    h = step(config).y_end
    out = {}
    for station, profile in PROFILES.items():
        y = np.array(profile.y_over_h) * h
        x = np.full_like(y, float(case_x(config, station)))
        out[station] = sample_u(mesh, u_faces, x, y) / u_ref
    return out


# ---------------------------------------------------------------------------
# The walls: shear, reattachment, pressure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WallLine:
    """A quantity along the bottom wall, at the x positions it is read at.

    Parameters
    ----------
    x_over_h : np.ndarray
        x/H of each reading, increasing; the step's top (x/H < 0) and the
        downstream floor (x/H > 0), the face on the step's own vertical
        wall left out.
    values : np.ndarray
        The quantity at each x.
    """

    x_over_h: np.ndarray
    values: np.ndarray


def bottom_wall_shear(
    mesh: Mesh, config: SimConfig, u_faces: np.ndarray, k: np.ndarray
) -> WallLine:
    """The wall function's shear on the bottom wall at every u face, Pa.

    Parameters
    ----------
    mesh : Mesh
        The case's grid.
    config : SimConfig
        The case, with its turbulence section (the variant's C_mu).
    u_faces : np.ndarray
        The x-velocity on the vertical faces, [ny, nx+1].
    k : np.ndarray
        Turbulent kinetic energy on the cells, [ny, nx].

    Returns
    -------
    WallLine
        ``tau_w = mu_w u_P / y_P`` at each u face of the wall row: the row
        above the step's top for faces left of the step, the floor row for
        faces right of it; ``mu_w`` the scalable wall function
        (``src.turbulence.wall_viscosity``) with k_P the mean of the two
        cells the face lies between and y_P the row's half height, as
        ``TurbulenceBoundary.wall_viscosity`` forms it; the sign is u_P's.
        The face on the step's vertical wall (x/H = 0) is left out.
    """
    spec = config.turbulence
    if spec is None:
        raise ValueError("the shear needs the case's turbulence section")
    c_mu = VARIANTS[spec.variant].c_mu
    nu = config.mu / config.rho
    j_up, i_step = step_row(mesh, config), step_column(mesh, config)
    # Interior faces only: the inlet face carries no wall cell on its left,
    # and the outlet face is the copied neighbour.
    i_left = np.arange(1, i_step)
    i_right = np.arange(i_step + 1, mesh.x.size - 1)
    xs, taus = [], []
    for j, cols in ((j_up, i_left), (0, i_right)):
        y_p = float(mesh.yc[j] - mesh.y[j])
        k_p = 0.5 * (k[j, cols - 1] + k[j, cols])
        mu_w = wall_viscosity(k_p, np.full(cols.size, y_p), c_mu, config.rho, nu)
        xs.append(mesh.x[cols])
        taus.append(mu_w * u_faces[j, cols] / y_p)
    x = np.concatenate(xs)
    return WallLine(x_over_h=np.asarray(data_x(config, x)), values=np.concatenate(taus))


def reattachment(x_over_h: np.ndarray, tau: np.ndarray) -> float:
    """x_r / H: where the wall shear last turns from negative to positive.

    Parameters
    ----------
    x_over_h : np.ndarray
        x/H of each shear reading, increasing.
    tau : np.ndarray
        The wall shear at each x (any units; the sign and the ratio matter).

    Returns
    -------
    float
        Between the last pair of neighbouring readings with ``tau[i] < 0 <=
        tau[i+1]``, the zero of the straight line through them:
        ``x[i] + (x[i+1] - x[i]) tau[i] / (tau[i] - tau[i+1])``. The last
        pair, because a secondary eddy in the step's corner can put an
        earlier crossing near x/H = 0; a reading exactly zero counts as
        attached.

    Raises
    ------
    ValueError
        If the shear never turns from negative to positive, or the arrays
        differ in length.
    """
    x = np.asarray(x_over_h, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    if x.shape != t.shape or x.ndim != 1:
        raise ValueError(f"{x.shape} positions against {t.shape} shear readings")
    rising = np.nonzero((t[:-1] < 0.0) & (t[1:] >= 0.0))[0]
    if rising.size == 0:
        raise ValueError("the wall shear never turns from negative to positive")
    i = int(rising[-1])
    return float(x[i] + (x[i + 1] - x[i]) * t[i] / (t[i] - t[i + 1]))


def friction_coefficient(shear: WallLine, u_ref: float, rho: float) -> WallLine:
    """Cf = tau_w / (rho u_ref^2 / 2) along the wall."""
    return WallLine(shear.x_over_h, shear.values / (0.5 * rho * u_ref**2))


def at_stations(line: WallLine, stations: np.ndarray | tuple[float, ...]) -> np.ndarray:
    """A wall line read at the data's stations, linear between its readings.

    Raises
    ------
    ValueError
        If a station lies outside the line's extent.
    """
    s = np.asarray(stations, dtype=np.float64)
    if s.min() < line.x_over_h[0] or s.max() > line.x_over_h[-1]:
        raise ValueError(
            f"stations {s.min()} to {s.max()} outside the line's "
            f"{line.x_over_h[0]} to {line.x_over_h[-1]}"
        )
    return np.interp(s, line.x_over_h, line.values)


def wall_pressure(
    mesh: Mesh, config: SimConfig, p: np.ndarray, wall: str = "bottom"
) -> WallLine:
    """The pressure of the wall cells along the bottom or the top wall, Pa, at their centres.

    Parameters
    ----------
    mesh : Mesh
        The case's grid.
    config : SimConfig
        The case.
    p : np.ndarray
        The pressure on the cells, [ny, nx].
    wall : str
        "bottom": the row above the step's top for cells over the block, the
        floor row downstream, the cell row over the step's vertical face
        included once (its floor-row cell, downstream of the face). "top":
        the ceiling row.

    Returns
    -------
    WallLine
        The pressure at each wall cell's centre. A wall cell's pressure
        stands for the wall's at second order, the normal gradient being
        zero there.
    """
    if wall == "top":
        return WallLine(np.asarray(data_x(config, mesh.xc)), np.asarray(p[-1, :]))
    if wall != "bottom":
        raise ValueError(f"wall must be 'bottom' or 'top', got {wall!r}")
    j_up, i_step = step_row(mesh, config), step_column(mesh, config)
    values = np.concatenate((p[j_up, :i_step], p[0, i_step:]))
    return WallLine(np.asarray(data_x(config, mesh.xc)), values)


def pressure_coefficient(line: WallLine, u_ref: float, rho: float) -> WallLine:
    """Cp along a wall, shifted to zero at x/H = 37.5 as the data's shifted column is.

    The data's `cp` column is `cp_orig` less one constant chosen so Cp is
    zero near x/H = 40, and its last bottom-wall station, x/H = 37.5, reads
    exactly zero. The computed line is shifted the same way: its value at
    x/H = 37.5, linear between the wall cells, is subtracted, so the two
    lines share the datum the data chose.
    """
    datum = float(np.interp(CP_ZERO_STATION, line.x_over_h, line.values))
    return WallLine(line.x_over_h, (line.values - datum) / (0.5 * rho * u_ref**2))


# ---------------------------------------------------------------------------
# y+ at the wall nodes
# ---------------------------------------------------------------------------


def wall_y_star(
    mesh: Mesh, config: SimConfig, k: np.ndarray, x_r_over_h: float | None
) -> dict[str, float | None]:
    """y* of the first node on the three walls the case reports, against the floor.

    Parameters
    ----------
    mesh : Mesh
        The case's grid.
    config : SimConfig
        The case.
    k : np.ndarray
        Turbulent kinetic energy on the cells, [ny, nx].
    x_r_over_h : float or None
        The reattachment length, so the floor is read downstream of it;
        None reads the floor over the whole downstream length.

    Returns
    -------
    dict[str, float | None]
        ``y* = C_mu^(1/4) k_P^(1/2) y_P / nu`` at the wall cells, the
        quantity the wall functions evaluate (ADR-012 B): the median on
        the floor downstream of reattachment (``floor_downstream``), on the
        step's top over the upstream length (``step_top``) and on the
        ceiling upstream of the step (``ceiling_upstream``); each wall's
        value in the cell nearest x/H = -4 or, for the floor, nearest the
        last profile station x/H = 10 (``*_at_station``); and ``floor``, the
        scalable floor 11.53 they are read against.
    """
    spec = config.turbulence
    if spec is None:
        raise ValueError("y* needs the case's turbulence section")
    c_mu = VARIANTS[spec.variant].c_mu
    nu = config.mu / config.rho
    j_up, i_step = step_row(mesh, config), step_column(mesh, config)
    x = np.asarray(data_x(config, mesh.xc))

    def y_star(j: int, cols: np.ndarray) -> np.ndarray:
        y_p = (
            float(mesh.yc[j] - mesh.y[j])
            if j < mesh.yc.size - 1
            else float(mesh.y[-1] - mesh.yc[-1])
        )
        return c_mu**0.25 * np.sqrt(k[j, cols]) * y_p / nu

    start = x_r_over_h if x_r_over_h is not None else 0.0
    floor_cols = np.nonzero(x > start)[0]
    up_cols = np.arange(0, i_step)
    top_cols = np.nonzero(x < 0.0)[0]
    near_ref = int(np.argmin(np.abs(x - REFERENCE_STATION)))
    near_ten = int(np.argmin(np.abs(x - PROFILE_STATIONS[-1])))
    floor = y_star(0, floor_cols)
    step_top = y_star(j_up, up_cols)
    ceiling = y_star(mesh.yc.size - 1, top_cols)
    return {
        "floor_downstream": float(np.median(floor)) if floor.size else None,
        "floor_at_station": float(y_star(0, np.array([near_ten]))[0]),
        "step_top": float(np.median(step_top)),
        "step_top_at_station": float(y_star(j_up, np.array([near_ref]))[0]),
        "ceiling_upstream": float(np.median(ceiling)),
        "ceiling_at_station": float(y_star(mesh.yc.size - 1, np.array([near_ref]))[0]),
        "floor": Y_STAR_FLOOR,
    }


# ---------------------------------------------------------------------------
# The whole comparison
# ---------------------------------------------------------------------------


def evaluate(
    mesh: Mesh,
    config: SimConfig,
    u_faces: np.ndarray,
    p: np.ndarray,
    k: np.ndarray,
) -> dict:
    """Every comparison of one solve with the data, reported, nothing scored.

    Parameters
    ----------
    mesh : Mesh
        The case's grid.
    config : SimConfig
        The case.
    u_faces : np.ndarray
        The final x-velocity on the vertical faces, [ny, nx+1].
    p : np.ndarray
        The final pressure on the cells, [ny, nx].
    k : np.ndarray
        The final turbulent kinetic energy on the cells, [ny, nx].

    Returns
    -------
    dict
        ``u_ref``: the computed centreline speed at x/H = -4 and its ratio
        to 44.2; ``inflow``: delta_99 / H and Re_theta there; ``x_r_over_h``:
        the reattachment length, or None when the floor shear never turns
        positive; ``profiles``: per station, the largest and RMS difference
        of u / u_ref from the data over its heights; ``cf`` and ``cp``: the
        same for the bottom wall's coefficients at the data's stations;
        ``y_star``: the first-node values of ``wall_y_star``.

    Notes
    -----
    Every number is in the data's normalisation, by the computed centreline
    speed, so a 1% miss on that speed moves every profile by 1%. The
    dictionary is JSON-serialisable (floats, ints, None and strings), as the
    harness row stores it.
    """
    check = inflow_check(mesh, config, u_faces)
    u_ref = check.u_centre
    rho = config.rho
    shear = bottom_wall_shear(mesh, config, u_faces, k)
    try:
        x_r: float | None = reattachment(shear.x_over_h, shear.values)
    except ValueError:
        x_r = None
    cf = friction_coefficient(shear, u_ref, rho)
    cp = pressure_coefficient(wall_pressure(mesh, config, p), u_ref, rho)
    profiles = {
        str(station): errors(values, np.array(PROFILES[station].u_over_uref)).__dict__
        for station, values in profile_values(mesh, config, u_faces, u_ref).items()
    }
    return {
        "u_ref": {"computed": u_ref, "over_measured": check.u_centre_over_uref},
        "inflow": {
            "delta_99_over_h": check.delta_99_over_h,
            "re_theta": check.re_theta,
            "delta_99_measured": DELTA_99_MEASURED,
            "re_theta_measured": RE_THETA_MEASURED,
        },
        "x_r_over_h": x_r,
        "x_r_measured": X_R_MEASURED,
        "x_r_uncertainty": X_R_UNCERTAINTY,
        "profiles": profiles,
        "cf": errors(at_stations(cf, CF_STATIONS), np.array(CF_MEASURED)).__dict__,
        "cp": errors(
            at_stations(cp, CP_BOTTOM_STATIONS), np.array(CP_BOTTOM_MEASURED)
        ).__dict__,
        "y_star": wall_y_star(mesh, config, k, x_r),
    }
