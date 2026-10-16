"""The plane impinging jet's data, geometry and comparison (VAL-020).

The case is ADR-012 G (vi) and ECR-002 criterion 13: case I of Khayrullina,
van Hooff, Blocken and van Heijst (2017), "PIV measurements of isothermal
plane turbulent impinging jets at moderate Reynolds numbers", Exp Fluids
58(4):31, DOI 10.1007/s00348-017-2315-0 (CC BY). A nozzle w_jet = 16 mm
wide, flush with the top plate of a water channel, issues downward onto a
plate h_jet = 360 mm below (h_jet / w_jet 22.5) at Re = V0 w_jet / nu =
8,000, V0 = 0.5 m/s with nu = 1.0e-6 m^2/s; 2D PIV in the vertical
centreplane gives the mean velocity and I_uv = sqrt(u'^2 + v'^2) / V0.

Frames. The paper's y runs down from the nozzle and its x across the jet
from the axis; the solver's origin is the floor of the left wall with y
up, so the paper's y / h_jet is ``1 - y / H_JET`` here and its x is the
distance from the nozzle's centre (``axis_x``). Every station below is
stored in the paper's coordinates and the readers convert.

Provenance of the data. Every value was read from the paper's raster
figures (the file results/builder55/sources/khayrullina_2017.pdf, SHA-256
c9d08bb01f1bc8ddff6dee2b5d437a3f0184dc77347b457bd29e8538ac05421d, fetched
2026-10-15 from https://link.springer.com/content/pdf/10.1007/s00348-017-
2315-0.pdf) by docs/reports/probe55/digitize55.py, which calibrates each
panel on its gridlines and axis box and checks the result against the
numbers the paper states in its text (the potential core 4.8 w_jet, the
half-velocity point near 0.85 h_jet, the I_uv peak 0.21 at 0.55 h_jet,
the spreading rate 0.10). The tuples were printed by data55.py; nothing
was typed, and tests/test_impinging_jet.py pins them by a hash. The record
with every intermediate number is results/builder55/khayrullina_2017_
digitized.json; docs/reports/ecr002_val020_impinging_jet.md, section 2.2,
describes each reading and its uncertainty.

- CENTRELINE_ROWS, figure 4(a): (y / h_jet, V_0,y / V0, half-extent of the
  symbol in V / V0). The three 16 mm series (Re 8,000, 10,000 and 13,000)
  are plotted on top of one another and the paper calls them identical,
  so each symbol cluster is one station and its centroid the value; one
  pixel is 0.0022 V0 and 0.0013 h_jet.
- INTENSITY_ROWS, figure 6: (y / h_jet, I_uv, True when the Re 8,000 open
  circle's own centre was read, False when the centroid of the three
  series' symbols stands in, which differ by up to 0.02 there); one pixel
  is 0.0006 I_uv and 0.0010 h_jet.
- HALF_WIDTH_ROWS, figure 8(a): (y / w_jet, delta_0.5 / w_jet), half the
  distance between the white dots the paper draws at 2 delta_0.5 on each
  side; one pixel is 0.028 w_jet.
- LATERAL_V_48_ROWS and LATERAL_I_48_ROWS, figures 5(a) and 7(a): (x /
  delta_0.5, V / V_0,y) and (x / delta_0.5, I_uv) at y' / w_jet = 4.8 from
  the virtual origin y0 = -0.8 w_jet, so at y = 4.0 w_jet, y / h_jet
  0.178; the green filled circles. The other seven heights of those
  figures (plain lines and open markers with the curves drawn through
  them) could not be digitised; for them the paper's own Gaussian fit,
  V / V_0,y = exp(-0.693 (x / delta_0.5)^2), stated to hold for 5.9 <=
  y' / w_jet <= 20.1, stands in (GAUSSIAN_HEIGHTS_W), and every number
  that rests on it is labelled "gaussian".

What the 2D PIV gives for k. I_uv^2 V0^2 is the sum of two of the three
normal stresses; the model's k is half the sum of three. With the
out-of-plane stress taken as the mean of the two measured, k = 0.75
I_uv^2 V0^2 (K_ISOTROPIC); the in-plane half-sum 0.5 I_uv^2 V0^2
(K_IN_PLANE) is a lower bound. Both are reported.

No threshold is set: the allowance is Alex's (ECR-002 criterion 13,
OPEN). ``evaluate`` reports every comparison and scores nothing.
"""

from dataclasses import dataclass

import numpy as np

from src.config import BoundarySpec, SimConfig
from src.mesh import Mesh
from src.turbulence import VARIANTS, Y_STAR_FLOOR

# The case (table 1 and section 2 of the paper).
W_JET = 0.016
H_JET = 0.360
RE = 8000.0
NU = 1.0e-6
V0 = RE * NU / W_JET
# The kinematic virtual origin for the 16 mm nozzle, y0 / w_jet (section 4.1).
VIRTUAL_ORIGIN_W = -0.8
# The Gaussian the paper fits to the self-similar lateral profiles.
GAUSS_A = -0.693
# The potential core ends where V_0,y / V0 falls below this (Deo et al. 2007a).
CORE_FRACTION = 0.98
# The paper's self-similar range for the 16 mm nozzle, in y / h_jet and in
# y / w_jet (section 4.1), over which the decay and spreading rates are read.
SELF_SIMILAR_H = (0.35, 0.75)
SELF_SIMILAR_W = (7.5, 17.0)
# The jet behaves as a free jet until this y / h_jet from the nozzle (section
# 5), the range prediction (c) is scored on; NARROW_FREE_JET_H is the prompt's
# narrower reading, the 30% nearest the nozzle, reported beside it.
FREE_JET_H = 0.70
NARROW_FREE_JET_H = 0.30
# The impingement region, the last 0.12 h_jet above the plate (section 4.1).
IMPINGEMENT_H = 0.88
# The plate's first-node y* is read as a median within this many nozzle
# widths of the axis, the wall jet's reach in the paper's figure 8.
PLATE_MEDIAN_HALF_WIDTH_W = 10.0
# The lateral profile heights, y' / w_jet from the virtual origin.
DIGITISED_HEIGHT_W = 4.8
GAUSSIAN_HEIGHTS_W = (5.9, 7.9, 12.1, 14.9, 20.1)
# Where the Gaussian is compared: x / delta_0.5 over the figure's range.
GAUSSIAN_STATIONS = tuple(np.round(np.linspace(-2.0, 2.0, 17), 4).tolist())
# k from I_uv: the out-of-plane stress as the mean of the two measured, and
# the in-plane half-sum alone.
K_ISOTROPIC = 0.75
K_IN_PLANE = 0.5
# The numbers the paper states in its text, for the 16 mm nozzle.
STATED = {
    "potential_core_w": 4.8,
    "half_velocity_h": 0.85,
    "impingement_h": 0.12,
    "decay_rate": (0.14, 0.15),
    "spreading_rate": 0.10,
    "i_uv_peak": (0.21, 0.55),
    "i_uv_before_impingement": 0.16,
    "i_uv_plate_peak": (0.21, 0.99),
    "core_i_uv_at_2w": 0.05,
}

# The case's inlet segment in the configuration, the nozzle in the top wall.
INLET = "nozzle"

CENTRELINE_ROWS: tuple[tuple[float, float, float], ...] = (
    (0.0619, 0.9878, 0.0097),
    (0.078, 0.9912, 0.0097),
    (0.0916, 0.9872, 0.0086),
    (0.1073, 0.9987, 0.0075),
    (0.1227, 0.9981, 0.0086),
    (0.1383, 0.9977, 0.0097),
    (0.1538, 0.9983, 0.0086),
    (0.1696, 0.9962, 0.0086),
    (0.1843, 0.9918, 0.0086),
    (0.2004, 0.9863, 0.0097),
    (0.2153, 0.9796, 0.0075),
    (0.2308, 0.9693, 0.0086),
    (0.2459, 0.9592, 0.0086),
    (0.2603, 0.9468, 0.0086),
    (0.2763, 0.9334, 0.0086),
    (0.2912, 0.9171, 0.0075),
    (0.3071, 0.9041, 0.0097),
    (0.3218, 0.8882, 0.0086),
    (0.3377, 0.8714, 0.0097),
    (0.3527, 0.8572, 0.0108),
    (0.3682, 0.8415, 0.0097),
    (0.3832, 0.8264, 0.0097),
    (0.3991, 0.8106, 0.0118),
    (0.4137, 0.7963, 0.0097),
    (0.4299, 0.7828, 0.0097),
    (0.4446, 0.7695, 0.0108),
    (0.4607, 0.7576, 0.0108),
    (0.4754, 0.7442, 0.0097),
    (0.491, 0.7314, 0.0108),
    (0.5054, 0.7197, 0.0097),
    (0.5294, 0.6997, 0.0086),
    (0.5485, 0.6839, 0.0086),
    (0.5674, 0.6695, 0.0108),
    (0.5869, 0.6569, 0.0108),
    (0.6063, 0.6468, 0.0086),
    (0.6252, 0.6351, 0.0097),
    (0.6444, 0.6275, 0.0086),
    (0.6621, 0.6105, 0.0118),
    (0.6813, 0.5999, 0.0097),
    (0.7046, 0.5904, 0.0086),
    (0.7279, 0.5753, 0.0075),
    (0.7514, 0.5622, 0.0075),
    (0.7733, 0.5473, 0.0108),
    (0.7972, 0.5276, 0.0118),
    (0.8201, 0.5007, 0.0097),
    (0.8422, 0.4713, 0.0086),
    (0.8648, 0.4345, 0.0086),
    (0.915, 0.3363, 0.0075),
    (0.9273, 0.2872, 0.0129),
    (0.9524, 0.2116, 0.0075),
    (0.994, 0.0579, 0.0075),
)
INTENSITY_ROWS: tuple[tuple[float, float, bool], ...] = (
    (0.1052, 0.0561, False),
    (0.1089, 0.0344, False),
    (0.133, 0.0535, True),
    (0.1589, 0.0638, True),
    (0.1837, 0.0849, True),
    (0.2099, 0.0967, True),
    (0.2348, 0.1052, False),
    (0.2606, 0.1246, True),
    (0.2869, 0.1357, True),
    (0.3117, 0.154, True),
    (0.3339, 0.166, False),
    (0.3375, 0.1519, False),
    (0.3397, 0.1611, False),
    (0.3649, 0.1737, False),
    (0.3649, 0.1837, False),
    (0.3897, 0.1837, False),
    (0.4136, 0.1949, True),
    (0.4395, 0.1974, True),
    (0.4645, 0.2013, True),
    (0.4904, 0.2035, True),
    (0.5155, 0.2016, True),
    (0.5447, 0.2055, True),
    (0.5872, 0.1997, True),
    (0.625, 0.1936, True),
    (0.651, 0.181, True),
    (0.6736, 0.1773, True),
    (0.697, 0.1758, True),
    (0.7198, 0.1769, True),
    (0.7431, 0.1727, True),
    (0.7658, 0.1735, True),
    (0.7884, 0.169, True),
    (0.8119, 0.1659, True),
    (0.8346, 0.1653, True),
    (0.858, 0.1648, True),
    (0.8807, 0.1631, True),
    (0.9033, 0.1632, True),
    (0.9266, 0.167, True),
    (0.9493, 0.1827, True),
    (0.9731, 0.1986, True),
    (0.9945, 0.1857, False),
)
HALF_WIDTH_ROWS: tuple[tuple[float, float], ...] = (
    (2.607, 0.4695),
    (4.105, 0.5528),
    (5.691, 0.6194),
    (7.196, 0.7296),
    (8.694, 0.8528),
    (10.199, 0.9889),
    (11.671, 1.1252),
    (14.675, 1.4387),
    (16.178, 1.5892),
    (17.652, 1.7936),
    (19.263, 2.0666),
)
LATERAL_V_48_ROWS: tuple[tuple[float, float], ...] = (
    (-1.9565, 0.0181),
    (-1.8073, 0.0494),
    (-1.6628, 0.0921),
    (-1.515, 0.1622),
    (-1.3653, 0.248),
    (-1.2186, 0.3531),
    (-0.9282, 0.6249),
    (-0.6313, 0.8744),
    (-0.4822, 0.9445),
    (-0.3362, 0.9808),
    (0.2576, 0.9826),
    (0.4002, 0.9461),
    (0.5497, 0.8628),
    (1.1392, 0.3201),
    (1.2898, 0.2072),
    (1.4366, 0.1242),
    (1.5835, 0.0645),
    (1.7312, 0.0246),
    (1.8789, 0.0073),
)
LATERAL_I_48_ROWS: tuple[tuple[float, float], ...] = (
    (-1.9556, 0.0855),
    (-1.6654, 0.15),
    (-1.3646, 0.2348),
    (-1.219, 0.2691),
    (-1.0709, 0.2996),
    (-0.9245, 0.2935),
    (-0.7766, 0.2684),
    (-0.6324, 0.2082),
    (-0.4827, 0.1465),
    (-0.3344, 0.1086),
    (-0.1868, 0.0887),
    (-0.0409, 0.0797),
    (0.1071, 0.0831),
    (0.2545, 0.0994),
    (0.4, 0.1377),
    (0.5485, 0.2017),
    (0.6962, 0.262),
    (0.8458, 0.2907),
    (0.9903, 0.276),
    (1.142, 0.248),
    (1.5827, 0.12),
    (1.7285, 0.0856),
    (1.8756, 0.0656),
)

# The stations as arrays, in the paper's coordinates.
CENTRELINE_Y = np.array([r[0] for r in CENTRELINE_ROWS])
CENTRELINE_V = np.array([r[1] for r in CENTRELINE_ROWS])
INTENSITY_Y = np.array([r[0] for r in INTENSITY_ROWS])
INTENSITY_I = np.array([r[1] for r in INTENSITY_ROWS])
# True where the Re 8,000 circle's own centre was read; False where the
# centroid of the three series' symbols stands in, up to 0.02 above it.
INTENSITY_RINGS = np.array([r[2] for r in INTENSITY_ROWS])
HALF_WIDTH_Y = np.array([r[0] for r in HALF_WIDTH_ROWS])
HALF_WIDTH_DELTA = np.array([r[1] for r in HALF_WIDTH_ROWS])


# ---------------------------------------------------------------------------
# Frames and sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Errors:
    """The difference of a set of computed values from a reference.

    Parameters
    ----------
    max_abs : float
        The largest absolute difference.
    rms : float
        The root mean square of the differences.
    mean : float
        The mean signed difference, computed minus reference.
    stations : int
        How many stations the two sets hold.
    """

    max_abs: float
    rms: float
    mean: float
    stations: int


def errors(values: np.ndarray, reference: np.ndarray) -> Errors:
    """``values - reference`` summarised.

    Parameters
    ----------
    values, reference : np.ndarray
        One value per station each, the same length.

    Returns
    -------
    Errors
        The largest, RMS and mean difference and the station count.

    Raises
    ------
    ValueError
        If the lengths differ, the set is empty or a value is not finite:
        a NaN would make every summary NaN without saying why.
    """
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if values.shape != reference.shape or values.ndim != 1:
        raise ValueError(f"{values.shape} values against {reference.shape} reference")
    if values.size == 0:
        raise ValueError("no stations")
    if not (np.isfinite(values).all() and np.isfinite(reference).all()):
        raise ValueError("a value is not finite")
    diff = values - reference
    return Errors(
        max_abs=float(np.abs(diff).max()),
        rms=float(np.sqrt(np.mean(diff**2))),
        mean=float(diff.mean()),
        stations=int(values.size),
    )


def nozzle(config: SimConfig) -> BoundarySpec:
    """The nozzle's inlet segment.

    Raises
    ------
    ValueError
        If the configuration has no velocity inlet named INLET on the top wall.
    """
    spec = config.boundaries.get(INLET)
    if spec is None or spec.type != "velocity_inlet" or spec.location != "top":
        raise ValueError(
            f"the case needs a velocity inlet named {INLET!r} on the top wall"
        )
    return spec


def axis_x(config: SimConfig) -> float:
    """The jet axis, the nozzle's centre, in the solver's x (m)."""
    spec = nozzle(config)
    assert spec.x_start is not None and spec.x_end is not None
    return 0.5 * (spec.x_start + spec.x_end)


def solver_y(y_over_h: np.ndarray | float) -> np.ndarray:
    """The paper's y / h_jet (down from the nozzle) as the solver's y (m, up from the plate)."""
    return H_JET * (1.0 - np.asarray(y_over_h, dtype=np.float64))


def paper_y(y: np.ndarray | float) -> np.ndarray:
    """The solver's y (m) as the paper's y / h_jet."""
    return 1.0 - np.asarray(y, dtype=np.float64) / H_JET


def _bilinear(
    xs: np.ndarray, ys: np.ndarray, field: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Bilinear interpolation on a lattice, points outside it clamped to its edge."""
    x = np.clip(np.asarray(x, dtype=np.float64), xs[0], xs[-1])
    y = np.clip(np.asarray(y, dtype=np.float64), ys[0], ys[-1])
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


def sample_v(
    mesh: Mesh, v_faces: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """v at points of the domain, bilinear in the staggered v storage.

    Parameters
    ----------
    mesh : Mesh
        The grid the faces belong to.
    v_faces : np.ndarray
        The y-velocity on the horizontal faces, shape [ny+1, nx],
        ``v[j, i]`` at ``(xc[i], y[j])``, positive up.
    x, y : np.ndarray
        The points, m, each of the same shape.

    Returns
    -------
    np.ndarray
        v at each point, m/s. A point outside the lattice of centres in x
        reads the edge column; y covers the whole height, the plate and the
        top wall being faces.

    Notes
    -----
    Exact for a field linear in x and y, and a point on a face row and a
    centre column reads that face's own value.
    """
    return _bilinear(np.asarray(mesh.xc), np.asarray(mesh.y), v_faces, x, y)


def sample_cells(
    mesh: Mesh, field: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """A cell field at points of the domain, bilinear in the cell centres.

    Parameters
    ----------
    mesh : Mesh
        The grid.
    field : np.ndarray
        A cell field, shape [ny, nx].
    x, y : np.ndarray
        The points, m, each of the same shape.

    Returns
    -------
    np.ndarray
        The field at each point; a point between the outermost centres and
        a wall reads the outermost centre (no wall value is assumed).
    """
    return _bilinear(np.asarray(mesh.xc), np.asarray(mesh.yc), field, x, y)


# ---------------------------------------------------------------------------
# The centreline
# ---------------------------------------------------------------------------

# How many centre columns on each side of the axis the axis fit uses.
AXIS_COLUMNS = 2


def axis_columns(mesh: Mesh, config: SimConfig) -> np.ndarray:
    """The indices of the AXIS_COLUMNS centre columns nearest the axis on each side."""
    x = np.asarray(mesh.xc) - axis_x(config)
    left = np.nonzero(x < 0.0)[0]
    right = np.nonzero(x >= 0.0)[0]
    if left.size < AXIS_COLUMNS or right.size < AXIS_COLUMNS:
        raise ValueError(
            f"fewer than {AXIS_COLUMNS} centre columns on a side of the axis"
        )
    return np.concatenate((left[-AXIS_COLUMNS:], right[:AXIS_COLUMNS]))


def axis_fit(x_from_axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The value on the axis from the columns beside it, by a symmetric parabola.

    The axis is a face between two columns of centres, so no stored value
    sits on it and the nearest centre is half a cell off, which on a
    four-cell nozzle reads a Gaussian jet 4% low. A least-squares fit of
    ``a + b x^2`` to the columns' values gives the axis value ``a``; for
    two symmetric pairs at d and 3d it is ``(9 V_d - V_3d) / 8`` with each
    pair averaged, exact for any parabola.

    Parameters
    ----------
    x_from_axis : np.ndarray
        The columns' distances from the axis, m, shape [columns].
    values : np.ndarray
        The field at those columns, shape [..., columns]; the fit is taken
        along the last axis.

    Returns
    -------
    np.ndarray
        ``a`` for each leading index, shape [...].
    """
    x2 = np.asarray(x_from_axis, dtype=np.float64) ** 2
    design = np.stack((np.ones_like(x2), x2), axis=1)
    coefficients, *_ = np.linalg.lstsq(
        design, np.asarray(values, dtype=np.float64).T, rcond=None
    )
    return coefficients[0]


def centreline(
    mesh: Mesh, config: SimConfig, v_faces: np.ndarray, y_over_h: np.ndarray
) -> np.ndarray:
    """V_0,y / V0 on the axis at the paper's stations.

    Parameters
    ----------
    mesh, config
        The grid and the case.
    v_faces : np.ndarray
        The y-velocity on the horizontal faces, [ny+1, nx].
    y_over_h : np.ndarray
        Stations, y / h_jet down from the nozzle.

    Returns
    -------
    np.ndarray
        The downward speed on the axis over V0, one per station: linear in
        y between the face rows, then ``axis_fit`` across the four columns
        nearest the axis.
    """
    stations = np.asarray(y_over_h, dtype=np.float64)
    cols = axis_columns(mesh, config)
    xc = np.asarray(mesh.xc)[cols]
    y = solver_y(stations)
    beside = np.stack(
        [-sample_v(mesh, v_faces, np.full_like(y, xi), y) for xi in xc], axis=-1
    )
    return axis_fit(xc - axis_x(config), beside) / V0


def first_crossing(y_over_h: np.ndarray, v_over_v0: np.ndarray, level: float) -> float:
    """Where a centreline first falls below a level, by linear interpolation.

    Parameters
    ----------
    y_over_h : np.ndarray
        Stations in increasing order, y / h_jet.
    v_over_v0 : np.ndarray
        V_0,y / V0 at each.
    level : float
        The level crossed.

    Returns
    -------
    float
        y / h_jet of the crossing: the station itself when it is the first
        and already below, else interpolated from the station before it.

    Raises
    ------
    ValueError
        If no station lies below the level, or the stations are not increasing.
    """
    y = np.asarray(y_over_h, dtype=np.float64)
    v = np.asarray(v_over_v0, dtype=np.float64)
    if y.size < 1 or np.any(np.diff(y) <= 0.0):
        raise ValueError("stations must increase")
    below = np.nonzero(v < level)[0]
    if below.size == 0:
        raise ValueError(f"the centreline never falls below {level}")
    k = int(below[0])
    if k == 0:
        return float(y[0])
    t = (v[k - 1] - level) / (v[k - 1] - v[k])
    return float(y[k - 1] + t * (y[k] - y[k - 1]))


def potential_core_w(y_over_h: np.ndarray, v_over_v0: np.ndarray) -> float:
    """The potential core length in nozzle widths: where V_0,y / V0 first falls below 0.98."""
    return first_crossing(y_over_h, v_over_v0, CORE_FRACTION) * H_JET / W_JET


def decay_rate(y_over_h: np.ndarray, v_over_v0: np.ndarray) -> float:
    """K_u, the slope of (V0 / V_0,y)^2 against (y - y0) / w_jet over the self-similar range.

    Parameters
    ----------
    y_over_h, v_over_v0 : np.ndarray
        The centreline stations and values.

    Returns
    -------
    float
        The least-squares slope over the stations with SELF_SIMILAR_H[0] <=
        y / h_jet <= SELF_SIMILAR_H[1].

    Raises
    ------
    ValueError
        If fewer than two stations lie in the range, or a value there is zero.
    """
    y = np.asarray(y_over_h, dtype=np.float64)
    v = np.asarray(v_over_v0, dtype=np.float64)
    inside = (y >= SELF_SIMILAR_H[0]) & (y <= SELF_SIMILAR_H[1])
    if inside.sum() < 2:
        raise ValueError("fewer than two stations in the self-similar range")
    if np.any(v[inside] == 0.0):
        raise ValueError("a centreline value in the self-similar range is zero")
    distance = y[inside] * H_JET / W_JET - VIRTUAL_ORIGIN_W
    inverse = (1.0 / v[inside]) ** 2
    slope, _ = np.polyfit(distance, inverse, 1)
    return float(slope)


# ---------------------------------------------------------------------------
# Lateral profiles and the half-width
# ---------------------------------------------------------------------------


def row_speed(
    mesh: Mesh, config: SimConfig, v_faces: np.ndarray, y_over_h: float
) -> tuple[np.ndarray, np.ndarray]:
    """The downward speed across the jet at one height, at every centre column.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x from the axis (m) at each centre column, and -v there (m/s).
    """
    xc = np.asarray(mesh.xc)
    y = np.full_like(xc, float(solver_y(y_over_h)))
    return xc - axis_x(config), -sample_v(mesh, v_faces, xc, y)


def half_width_w(
    mesh: Mesh, config: SimConfig, v_faces: np.ndarray, y_over_h: float
) -> float:
    """delta_0.5 / w_jet at one height: where the speed first halves on each side, averaged.

    Parameters
    ----------
    mesh, config
        The grid and the case.
    v_faces : np.ndarray
        The y-velocity on the horizontal faces, [ny+1, nx].
    y_over_h : float
        The height, y / h_jet down from the nozzle.

    Returns
    -------
    float
        The mean of the two sides' half-widths in nozzle widths. On each
        side the speed is read at the centre columns outward from the axis
        and the first fall below half the axis speed is interpolated
        linearly; the axis speed is the mean of the two columns beside it.

    Raises
    ------
    ValueError
        If the axis speed is not positive, or a side never halves.
    """
    x, speed = row_speed(mesh, config, v_faces, y_over_h)
    centre = float(centreline(mesh, config, v_faces, np.array([y_over_h]))[0] * V0)
    if centre <= 0.0:
        raise ValueError(
            f"the axis speed at y / h_jet {y_over_h} is {centre}, not positive"
        )
    half = 0.5 * centre

    def side(order: np.ndarray) -> float:
        xs, vs = np.abs(x[order]), speed[order]
        below = np.nonzero(vs < half)[0]
        if below.size == 0:
            raise ValueError(f"the speed never halves at y / h_jet {y_over_h}")
        k = int(below[0])
        if k == 0:
            return float(xs[0])
        t = (vs[k - 1] - half) / (vs[k - 1] - vs[k])
        return float(xs[k - 1] + t * (xs[k] - xs[k - 1]))

    right = np.nonzero(x > 0.0)[0]
    left = np.nonzero(x < 0.0)[0][::-1]
    return 0.5 * (side(right) + side(left)) / W_JET


def spreading_rate(y_over_w: np.ndarray, delta_over_w: np.ndarray) -> float:
    """K_y, the slope of delta_0.5 against y over the self-similar range, both in w_jet.

    Raises
    ------
    ValueError
        If fewer than two heights lie in SELF_SIMILAR_W.
    """
    y = np.asarray(y_over_w, dtype=np.float64)
    d = np.asarray(delta_over_w, dtype=np.float64)
    inside = (y >= SELF_SIMILAR_W[0]) & (y <= SELF_SIMILAR_W[1])
    if inside.sum() < 2:
        raise ValueError("fewer than two heights in the self-similar range")
    slope, _ = np.polyfit(y[inside], d[inside], 1)
    return float(slope)


def self_similar(
    mesh: Mesh,
    config: SimConfig,
    v_faces: np.ndarray,
    y_over_h: float,
    x_over_delta: np.ndarray,
) -> tuple[np.ndarray, float]:
    """V / V_0,y at x = xi delta_0.5 for the given xi, with the computed delta_0.5.

    Returns
    -------
    tuple[np.ndarray, float]
        The profile at the stations and delta_0.5 / w_jet, both the
        computed solution's own.
    """
    delta = half_width_w(mesh, config, v_faces, y_over_h)
    xi = np.asarray(x_over_delta, dtype=np.float64)
    x = axis_x(config) + xi * delta * W_JET
    y = np.full_like(xi, float(solver_y(y_over_h)))
    centre = centreline(mesh, config, v_faces, np.array([y_over_h]))[0] * V0
    return -sample_v(mesh, v_faces, x, y) / centre, delta


def gaussian(x_over_delta: np.ndarray) -> np.ndarray:
    """The paper's fit, exp(-0.693 (x / delta_0.5)^2)."""
    xi = np.asarray(x_over_delta, dtype=np.float64)
    return np.exp(GAUSS_A * xi**2)


def height_over_h(y_prime_over_w: float) -> float:
    """A lateral profile's height, y' / w_jet from the virtual origin, as y / h_jet."""
    return (y_prime_over_w + VIRTUAL_ORIGIN_W) * W_JET / H_JET


# ---------------------------------------------------------------------------
# Turbulence
# ---------------------------------------------------------------------------


def k_on_axis(
    mesh: Mesh, config: SimConfig, k: np.ndarray, y_over_h: np.ndarray
) -> np.ndarray:
    """k / V0^2 on the axis at the paper's stations.

    Linear in y between the rows of centres, then ``axis_fit`` across the
    four columns nearest the axis, as ``centreline`` reads v.
    """
    stations = np.asarray(y_over_h, dtype=np.float64)
    cols = axis_columns(mesh, config)
    xc = np.asarray(mesh.xc)[cols]
    y = solver_y(stations)
    beside = np.stack(
        [sample_cells(mesh, k, np.full_like(y, xi), y) for xi in xc], axis=-1
    )
    return axis_fit(xc - axis_x(config), beside) / V0**2


def intensity_from_k(k_over_v0sq: np.ndarray) -> np.ndarray:
    """The model's I_uv estimate, sqrt(4 k / 3) / V0: the two in-plane stresses of an isotropic k."""
    return np.sqrt(np.asarray(k_over_v0sq, dtype=np.float64) / K_ISOTROPIC)


def stagnation(mesh: Mesh, config: SimConfig, k: np.ndarray) -> dict[str, float]:
    """k / V0^2 at the stagnation point, three readings.

    Returns
    -------
    dict[str, float]
        ``first_cell``: on the axis (``axis_fit``) in the first row of
        cells above the plate; ``impingement_max``: the largest value on
        the axis over the rows with y / h_jet > IMPINGEMENT_H;
        ``impingement_max_y_over_h``: where.
    """
    yc = np.asarray(mesh.yc)
    cols = axis_columns(mesh, config)
    on_axis = axis_fit(np.asarray(mesh.xc)[cols] - axis_x(config), k[:, cols]) / V0**2
    rows = np.nonzero(paper_y(yc) > IMPINGEMENT_H)[0]
    if rows.size == 0:
        raise ValueError("no row of centres lies in the impingement region")
    peak = rows[int(np.argmax(on_axis[rows]))]
    return {
        "first_cell": float(on_axis[0]),
        "impingement_max": float(on_axis[peak]),
        "impingement_max_y_over_h": float(paper_y(yc[peak])),
    }


def plate_y_star(mesh: Mesh, config: SimConfig, k: np.ndarray) -> dict[str, float]:
    """y* of the first node on the plate, against the scalable floor.

    Returns
    -------
    dict[str, float]
        ``y* = C_mu^(1/4) k_P^(1/2) y_P / nu`` in the plate's row of cells
        (ADR-012 B): ``median`` within PLATE_MEDIAN_HALF_WIDTH_W nozzle
        widths of the axis, ``at_axis`` (the mean of the two cells beside
        the axis), ``largest`` over the whole plate, and ``floor``, 11.53.
    """
    spec = config.turbulence
    if spec is None:
        raise ValueError("y* needs the case's turbulence section")
    c_mu = VARIANTS[spec.variant].c_mu
    nu = config.mu / config.rho
    y_p = float(mesh.yc[0] - mesh.y[0])
    row = c_mu**0.25 * np.sqrt(np.maximum(k[0, :], 0.0)) * y_p / nu
    x = np.asarray(mesh.xc) - axis_x(config)
    near = np.abs(x) <= PLATE_MEDIAN_HALF_WIDTH_W * W_JET
    cols = axis_columns(mesh, config)
    return {
        "median": float(np.median(row[near])),
        "at_axis": float(axis_fit(x[cols], row[cols])),
        "largest": float(row.max()),
        "floor": Y_STAR_FLOOR,
    }


# ---------------------------------------------------------------------------
# The whole comparison
# ---------------------------------------------------------------------------


def _errors_dict(e: Errors) -> dict:
    return {"max_abs": e.max_abs, "rms": e.rms, "mean": e.mean, "stations": e.stations}


def _range_errors(
    y: np.ndarray, computed: np.ndarray, data: np.ndarray, keep: np.ndarray
) -> dict:
    return _errors_dict(errors(computed[keep], data[keep]))


def evaluate(mesh: Mesh, config: SimConfig, v_faces: np.ndarray, k: np.ndarray) -> dict:
    """Every comparison of one solve with the data, reported, nothing scored.

    Parameters
    ----------
    mesh : Mesh
        The case's grid.
    config : SimConfig
        The case.
    v_faces : np.ndarray
        The final y-velocity on the horizontal faces, [ny+1, nx].
    k : np.ndarray
        The final turbulent kinetic energy on the cells, [ny, nx].

    Returns
    -------
    dict
        ``centreline``: V_0,y / V0 at the 51 stations and its errors over
        the whole line, the free-jet part (y / h_jet <= FREE_JET_H), the
        prompt's narrower reading (<= NARROW_FREE_JET_H) and the
        impingement region (> IMPINGEMENT_H); ``potential_core_w``,
        ``half_velocity_h`` and ``decay_rate``, each computed, from the
        data and as the paper states it; ``half_width``: delta_0.5 / w_jet
        at the 11 dot heights, its errors and the spreading rates;
        ``lateral``: per height, V / V_0,y against the digitised 4.8 profile
        or the Gaussian (``source`` says which), with the computed delta;
        ``lateral_intensity_48``: the model's I_uv estimate against the
        digitised profile; ``k_axis``: k / V0^2 at the 40 I_uv stations
        against the isotropic and the in-plane estimates, the errors over
        the 30 stations that hold the Re 8,000 circle's own value and,
        with the suffix ``_all``, over every station; ``stagnation``: the
        three readings, the ratio to the data's estimate at the data's
        nearest station to the plate (a centroid station) and at the
        nearest ring station (the ``ring_`` keys); ``y_star``: the plate's first
        node. JSON-serialisable (floats, ints, lists, strings, None). A
        reader that cannot be taken (a centreline that never halves, a
        profile that never halves) stores None for its value, so one
        failed reading does not hide the others.
    """
    data_y, data_v = CENTRELINE_Y, CENTRELINE_V
    computed_v = centreline(mesh, config, v_faces, data_y)
    free = data_y <= FREE_JET_H
    narrow = data_y <= NARROW_FREE_JET_H
    impinging = data_y > IMPINGEMENT_H
    report: dict = {
        "centreline": {
            "y_over_h": data_y.tolist(),
            "computed": computed_v.tolist(),
            "data": data_v.tolist(),
            "all": _range_errors(data_y, computed_v, data_v, np.ones_like(free)),
            "free_jet": _range_errors(data_y, computed_v, data_v, free),
            "free_jet_narrow": _range_errors(data_y, computed_v, data_v, narrow),
            "impingement": _range_errors(data_y, computed_v, data_v, impinging),
        }
    }

    def guarded(func, *args) -> float | None:
        try:
            return float(func(*args))
        except ValueError:
            return None

    report["potential_core_w"] = {
        "computed": guarded(potential_core_w, data_y, computed_v),
        "data": potential_core_w(data_y, data_v),
        "stated": STATED["potential_core_w"],
    }
    report["half_velocity_h"] = {
        "computed": guarded(first_crossing, data_y, computed_v, 0.5),
        "data": first_crossing(data_y, data_v, 0.5),
        "stated": STATED["half_velocity_h"],
    }
    report["decay_rate"] = {
        "computed": guarded(decay_rate, data_y, computed_v),
        "data": decay_rate(data_y, data_v),
        "stated": list(STATED["decay_rate"]),
    }

    heights_h = HALF_WIDTH_Y * W_JET / H_JET
    deltas = [guarded(half_width_w, mesh, config, v_faces, float(h)) for h in heights_h]
    held = np.array([d is not None for d in deltas])
    delta_arr = np.array([0.0 if d is None else d for d in deltas])
    report["half_width"] = {
        "y_over_w": HALF_WIDTH_Y.tolist(),
        "computed": deltas,
        "data": HALF_WIDTH_DELTA.tolist(),
        "errors": _errors_dict(errors(delta_arr[held], HALF_WIDTH_DELTA[held]))
        if held.any()
        else None,
        "spreading_rate": {
            "computed": guarded(spreading_rate, HALF_WIDTH_Y[held], delta_arr[held]),
            "data": spreading_rate(HALF_WIDTH_Y, HALF_WIDTH_DELTA),
            "stated": STATED["spreading_rate"],
        },
    }

    lateral: dict = {}
    xi_48 = np.array([r[0] for r in LATERAL_V_48_ROWS])
    v_48 = np.array([r[1] for r in LATERAL_V_48_ROWS])
    heights = [(DIGITISED_HEIGHT_W, xi_48, v_48, "digitised")] + [
        (
            h,
            np.array(GAUSSIAN_STATIONS),
            gaussian(np.array(GAUSSIAN_STATIONS)),
            "gaussian",
        )
        for h in GAUSSIAN_HEIGHTS_W
    ]
    for y_prime, xi, reference, source in heights:
        y_h = height_over_h(y_prime)
        try:
            profile, delta = self_similar(mesh, config, v_faces, y_h, xi)
            entry = {
                "errors": _errors_dict(errors(profile, reference)),
                "delta_over_w": delta,
                "computed": profile.tolist(),
            }
        except ValueError as err:
            entry = {
                "errors": None,
                "delta_over_w": None,
                "computed": None,
                "why": str(err),
            }
        lateral[str(y_prime)] = {
            "y_prime_over_w": y_prime,
            "y_over_h": y_h,
            "source": source,
            "x_over_delta": xi.tolist(),
            "reference": reference.tolist(),
            **entry,
        }
    report["lateral"] = lateral

    y_48 = height_over_h(DIGITISED_HEIGHT_W)
    xi_i = np.array([r[0] for r in LATERAL_I_48_ROWS])
    i_48 = np.array([r[1] for r in LATERAL_I_48_ROWS])
    try:
        delta_48 = half_width_w(mesh, config, v_faces, y_48)
        x = axis_x(config) + xi_i * delta_48 * W_JET
        y = np.full_like(xi_i, float(solver_y(y_48)))
        model_i = intensity_from_k(sample_cells(mesh, k, x, y) / V0**2)
        report["lateral_intensity_48"] = {
            "y_over_h": y_48,
            "x_over_delta": xi_i.tolist(),
            "computed": model_i.tolist(),
            "data": i_48.tolist(),
            "errors": _errors_dict(errors(model_i, i_48)),
        }
    except ValueError as err:
        report["lateral_intensity_48"] = {
            "y_over_h": y_48,
            "errors": None,
            "why": str(err),
        }

    k_axis = k_on_axis(mesh, config, k, INTENSITY_Y)
    iso = K_ISOTROPIC * INTENSITY_I**2
    plane = K_IN_PLANE * INTENSITY_I**2
    rings = INTENSITY_RINGS
    report["k_axis"] = {
        "y_over_h": INTENSITY_Y.tolist(),
        "ring": rings.tolist(),
        "computed": k_axis.tolist(),
        "data_isotropic": iso.tolist(),
        "data_in_plane": plane.tolist(),
        "errors_isotropic": _errors_dict(errors(k_axis[rings], iso[rings])),
        "errors_in_plane": _errors_dict(errors(k_axis[rings], plane[rings])),
        "errors_isotropic_all": _errors_dict(errors(k_axis, iso)),
        "errors_in_plane_all": _errors_dict(errors(k_axis, plane)),
        "ratio_isotropic_median": float(np.median(k_axis[rings] / iso[rings])),
        "ratio_isotropic_median_all": float(np.median(k_axis / iso)),
    }
    last = int(np.argmax(INTENSITY_Y))
    last_ring = int(np.argmax(np.where(rings, INTENSITY_Y, -1.0)))
    readings = stagnation(mesh, config, k)
    report["stagnation"] = {
        **readings,
        "data_station_y_over_h": float(INTENSITY_Y[last]),
        "data_station_ring": bool(rings[last]),
        "computed_at_data_station": float(k_axis[last]),
        "data_isotropic": float(iso[last]),
        "data_in_plane": float(plane[last]),
        "ratio_isotropic": float(k_axis[last] / iso[last]),
        "ratio_in_plane": float(k_axis[last] / plane[last]),
        "ring_station_y_over_h": float(INTENSITY_Y[last_ring]),
        "computed_at_ring_station": float(k_axis[last_ring]),
        "ring_isotropic": float(iso[last_ring]),
        "ring_in_plane": float(plane[last_ring]),
        "ring_ratio_isotropic": float(k_axis[last_ring] / iso[last_ring]),
        "ring_ratio_in_plane": float(k_axis[last_ring] / plane[last_ring]),
    }
    report["y_star"] = plate_y_star(mesh, config, k)
    return report
