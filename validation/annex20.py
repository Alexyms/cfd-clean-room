"""The Annex 20 room's measurements, the published prediction and the comparison (VAL-017).

The case is ADR-012 G (iii): Nielsen's (1990) two-dimensional isothermal room,
L 9.0 m by H 3.0 m, a slot h = 0.168 m under the ceiling on the left wall,
u0 = 0.455 m/s, Re 5,000 on h. The specification names four comparison lines,
x = H, x = 2H, y = h/2 and y = H - h/2, with y measured down from the ceiling
(its figure 1). This module stores them in the solver's frame, origin at the
floor of the inlet wall and y up, so the specification's y = h/2 is the
ceiling-jet line at y = 2.916 m here and its y = H - h/2 the floor line at
y = 0.084 m, and a vertical line's station y_spec / H becomes H (1 - y_spec / H).

Provenance of the measurements (MEASURED in every row below): the Aalborg
benchmark workbook "Laser-doppler measurements of the isothermal two
dimensional test case", https://www.aaudxp-cms.aau.dk/media/udffy5eg/
592302_2d_-benchmark_test_measurements_new.xls, SHA-256 e6dc80485371e3d076cf0
677202958df07d07e9207b787053ea68313bc745a8e, fetched 2026-10-04 for prompt 33
(docs/reports/product_case_reynolds.md, section 6.1): the symmetry plane
(z/W = 0) of Restivo's (1979) model at Re 5,000, digitized in 2005 by Arnkell
J. Petersen from the specification's figures 5 and 7, the mean u / u0 column
of each line (the rms column is not stored). The values were read from the
workbook by docs/reports/probe49/digitize49.py and printed by data49.py;
nothing was typed. Section 6.3 of the report above checked the workbook
against its own digitization of figure 5 (within 0.008 u0) and found the
x = 2H profile carrying 0.60 to 0.64 of the inlet flux, the room being
three-dimensional there. The workbook itself is not committed (its portal
terms allow one private copy and forbid redistribution); the values are.

Provenance of the published prediction (PUBLISHED, the third entry of each
row): Rong and Nielsen (2008), "Simulation with different turbulence models in
an annex 20 room benchmark test using Ansys CFX 11.0", DCE Technical Report
46, Aalborg University, https://homes.civil.aau.dk/pvn/cfd-benchmarks/
two_d_literature/2005_2010/Rong_l_and_P_V_Nielsen_2008.pdf, SHA-256
07d2ceba2fdefffce5fd02ba97a1da479c885cac3785de17fbddead3274722d4, fetched
2026-10-04 for prompt 33b (ADR-012 source 10). Its standard k-epsilon run
(Ansys CFX 11.0, scalable wall functions, 4,736 cells after a three-grid
study) is the red line of its figures 5(a), 5(b), 9(a) and 9(b), raster
images of 265 x 404, 274 x 408, 705 x 294 and 454 x 202 pixels embedded in
the PDF. digitize49.py extracts each at its native size, calibrates the axes
from the tick marks on the axis box (fit residuals at most 0.011 m and 0.005
u/u0), traces the red line by continuity so the legend's segment is refused,
and resamples it at the measurement stations by linear interpolation; one
pixel is 0.008 m and 0.008 u/u0 on the vertical lines, 0.014 to 0.023 m and
0.004 to 0.012 u/u0 on the horizontal ones. The check: the figures' own
measurement circles, mapped through the same calibration, agree with the
workbook within 0.007 u/u0 wherever they match a station (17 of 28 circles on
each horizontal line, within 0.024 and 0.064 m in x); on the vertical lines
the circles sit a consistent 0.03 to 0.04 m from the workbook's stations in
y (their plotting, not the calibration, whose tick residual there is below
0.005 m), so the published values near the ceiling jet, where du/dy is
largest, carry an uncertainty of about 0.05 u/u0 from that offset. A station
the traced line does not reach (the first station of each horizontal line,
at the inlet wall, and the top station of x = 2H, where the line turns to
the ceiling within a pixel row) holds None. The record with every
intermediate number is results/builder49/rong_nielsen_2008_digitized.json.

The allowance (Alex, 2026-10-11; ADR-012 decision 6, amended): on each line
of SCORED_LINES, x = H and the ceiling jet, the RMS of u / u0 against the
measurements, over the stations the line holds, is no larger than the
published prediction's RMS against the same measurements on that line
(``published_rms``: 0.0703 at x = H over its 25 stations, 0.1356 on the jet
over the 27 the published line reaches, the 0.070 and 0.136 the decision
cites). The allowance is computed from the stored values, so it has no
second copy. x = 2H and the floor are reported, not scored: the measured
room is three-dimensional at x = 2H, and the floor line is the model's own
departure, which the published run shares. ``score_lines`` applies it.
"""

from dataclasses import dataclass

import numpy as np

from src.mesh import Mesh

# The specification's room (Nielsen 1990, equations (2) to (4)).
U0 = 0.455
HEIGHT = 3.0
LENGTH = 9.0
SLOT = 0.168

# One row per measurement station: (coordinate along the line in metres,
# measured u / u0, published standard k-epsilon u / u0 or None).
X3_ROWS: tuple[tuple[float, float, float | None], ...] = (
    (2.9292, 0.76790, 0.8498),
    (2.8695, 0.78180, 0.7533),
    (2.8113, 0.75280, 0.6324),
    (2.7717, 0.68860, 0.5503),
    (2.7348, 0.62830, 0.4790),
    (2.6721, 0.50350, 0.3734),
    (2.5599, 0.34190, 0.2193),
    (2.4390, 0.23050, 0.1312),
    (2.3406, 0.16200, 0.0940),
    (2.1159, 0.07500, 0.0610),
    (1.9119, 0.03080, 0.0405),
    (1.7280, -0.00670, 0.0216),
    (1.4778, -0.04290, -0.0048),
    (1.2522, -0.07380, -0.0295),
    (1.0410, -0.09800, -0.0568),
    (0.8388, -0.10590, -0.0830),
    (0.6213, -0.11270, -0.1094),
    (0.5028, -0.11820, -0.1200),
    (0.4170, -0.11670, -0.1283),
    (0.3450, -0.12130, -0.1365),
    (0.2742, -0.12020, -0.1365),
    (0.2058, -0.11600, -0.1440),
    (0.1431, -0.11400, -0.1529),
    (0.0903, -0.11080, -0.1529),
    (0.0402, -0.09930, -0.1447),
)
X6_ROWS: tuple[tuple[float, float, float | None], ...] = (
    (2.9697, 0.59070, None),
    (2.9385, 0.62850, 0.6504),
    (2.8929, 0.63510, 0.6359),
    (2.8428, 0.63410, 0.6049),
    (2.7990, 0.61460, 0.5811),
    (2.7081, 0.54970, 0.5154),
    (2.6286, 0.49590, 0.4561),
    (2.5020, 0.40290, 0.3618),
    (2.3640, 0.30210, 0.2805),
    (2.1549, 0.17870, 0.1967),
    (1.9362, 0.09080, 0.1267),
    (1.7349, 0.04070, 0.0792),
    (1.5051, -0.03170, 0.0249),
    (1.2879, -0.08550, -0.0285),
    (1.0752, -0.15760, -0.0772),
    (0.8790, -0.21400, -0.1311),
    (0.6723, -0.24770, -0.1849),
    (0.5454, -0.26660, -0.1810),
    (0.4470, -0.27720, -0.1708),
    (0.3651, -0.28640, -0.1624),
    (0.2976, -0.29590, -0.1554),
    (0.2307, -0.30640, -0.1485),
    (0.1677, -0.31690, -0.1420),
    (0.1200, -0.32210, -0.1371),
    (0.0768, -0.30930, -0.1327),
)
Y2916_ROWS: tuple[tuple[float, float, float | None], ...] = (
    (0.0120, 1.08140, None),
    (0.5355, 1.06550, 1.0205),
    (1.0209, 1.03565, 1.0205),
    (1.4742, 0.97635, 1.0082),
    (1.9866, 0.91730, 0.9517),
    (2.5038, 0.86670, 0.8848),
    (3.0042, 0.82460, 0.8354),
    (3.4788, 0.78765, 0.7983),
    (3.9750, 0.75030, 0.7491),
    (4.5174, 0.71550, 0.7181),
    (5.0577, 0.69260, 0.6872),
    (5.4966, 0.66595, 0.6625),
    (6.0066, 0.64395, 0.6379),
    (6.5583, 0.61700, 0.6132),
    (6.8991, 0.59410, 0.5885),
    (7.2054, 0.56470, 0.5568),
    (7.4781, 0.50930, 0.5242),
    (7.8753, 0.45350, 0.4427),
    (8.0343, 0.38835, 0.3928),
    (8.1078, 0.31110, 0.3716),
    (8.2056, 0.24310, 0.3361),
    (8.2995, 0.14825, 0.2971),
    (8.3892, 0.07090, 0.2603),
    (8.4774, -0.02275, 0.2153),
    (8.4993, -0.13325, 0.2058),
    (8.5956, -0.20005, 0.1660),
    (8.7141, -0.18935, 0.0950),
    (8.8365, -0.14760, 0.0367),
)
Y0084_ROWS: tuple[tuple[float, float, float | None], ...] = (
    (0.1521, 0.04090, None),
    (0.3588, 0.05220, -0.0063),
    (0.5337, 0.04735, -0.0148),
    (0.6759, 0.03695, -0.0200),
    (0.8286, 0.03040, -0.0274),
    (0.9891, 0.01740, -0.0359),
    (1.1628, 0.00775, -0.0443),
    (1.3980, -0.00820, -0.0570),
    (1.5252, -0.01840, -0.0667),
    (1.6776, -0.01040, -0.0743),
    (1.8405, -0.03035, -0.0865),
    (2.0313, -0.04280, -0.0992),
    (2.3298, -0.06140, -0.1160),
    (2.6919, -0.07160, -0.1371),
    (3.0348, -0.09385, -0.1582),
    (3.5628, -0.13195, -0.1920),
    (4.0383, -0.16930, -0.2215),
    (4.5366, -0.20160, -0.2510),
    (5.0052, -0.25210, -0.2827),
    (5.5617, -0.30105, -0.3101),
    (6.0597, -0.32025, -0.3354),
    (6.5649, -0.32090, -0.3502),
    (7.0776, -0.29860, -0.3439),
    (7.5609, -0.25165, -0.3132),
    (8.0808, -0.16210, -0.2170),
    (8.4114, -0.06660, -0.0847),
    (8.7357, 0.06790, 0.1179),
    (8.9397, 0.22450, 0.2950),
)


@dataclass(frozen=True)
class Line:
    """One of the specification's four comparison lines, in the solver's frame.

    Parameters
    ----------
    key : str
        A short identifier: "x3", "x6", "y2916" or "y0084".
    label : str
        The specification's name for the line.
    vertical : bool
        True for a line of constant x (stations along y), False for a line
        of constant y (stations along x).
    position : float
        The constant coordinate, m.
    stations : tuple[float, ...]
        The measurement stations along the line, m, in the workbook's order.
    measured : tuple[float, ...]
        Measured u / u0 at each station.
    published : tuple[float | None, ...]
        Rong and Nielsen's standard k-epsilon u / u0 at each station, None
        where the digitized line does not reach the station.
    """

    key: str
    label: str
    vertical: bool
    position: float
    stations: tuple[float, ...]
    measured: tuple[float, ...]
    published: tuple[float | None, ...]


def _line(
    key: str,
    label: str,
    vertical: bool,
    position: float,
    rows: tuple[tuple[float, float, float | None], ...],
) -> Line:
    return Line(
        key=key,
        label=label,
        vertical=vertical,
        position=position,
        stations=tuple(r[0] for r in rows),
        measured=tuple(r[1] for r in rows),
        published=tuple(r[2] for r in rows),
    )


LINES: dict[str, Line] = {
    "x3": _line("x3", "x = H (3.0 m)", True, 3.0, X3_ROWS),
    "x6": _line("x6", "x = 2H (6.0 m)", True, 6.0, X6_ROWS),
    "y2916": _line(
        "y2916", "y = H - h/2 (2.916 m), the ceiling jet", False, 2.916, Y2916_ROWS
    ),
    "y0084": _line("y0084", "y = h/2 (0.084 m), the floor", False, 0.084, Y0084_ROWS),
}


@dataclass(frozen=True)
class LineErrors:
    """The difference of two sets of line values, over the stations both hold.

    Parameters
    ----------
    max_abs : float
        The largest absolute difference, in u / u0.
    rms : float
        The root mean square of the differences, in u / u0.
    stations : int
        How many stations the two sets both hold.
    """

    max_abs: float
    rms: float
    stations: int


def sample_u(
    mesh: Mesh, u_faces: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """u at points of the room, bilinear in the staggered u storage.

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

    Notes
    -----
    Bilinear interpolation on the ``(x, yc)`` lattice: exact for a field
    linear in x and y, and the value a station on a face or on a row of
    centres reads is that face's or row's own.
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


def sample_lines(mesh: Mesh, u_faces: np.ndarray) -> dict[str, np.ndarray]:
    """u / u0 at every measurement station of the four lines.

    Parameters
    ----------
    mesh : Mesh
        The grid the faces belong to.
    u_faces : np.ndarray
        The x-velocity on the vertical faces, shape [ny, nx+1].

    Returns
    -------
    dict[str, np.ndarray]
        For each key of LINES, u / u0 at that line's stations, in order.
    """
    out = {}
    for key, line in LINES.items():
        stations = np.array(line.stations)
        if line.vertical:
            x, y = np.full_like(stations, line.position), stations
        else:
            x, y = stations, np.full_like(stations, line.position)
        out[key] = sample_u(mesh, u_faces, x, y) / U0
    return out


def line_errors(
    values: np.ndarray, reference: tuple[float | None, ...] | np.ndarray
) -> LineErrors:
    """The largest and the root-mean-square difference of two sets of line values.

    Parameters
    ----------
    values : np.ndarray
        u / u0 at a line's stations, one value per station.
    reference : tuple[float | None, ...] or np.ndarray
        u / u0 at the same stations; a None entry is a station the reference
        does not hold, left out of both numbers.

    Returns
    -------
    LineErrors
        ``values - reference`` over the stations both hold.

    Raises
    ------
    ValueError
        If the two sets differ in length, or no station is held by both.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(reference) != values.size:
        raise ValueError(
            f"{values.size} values against {len(reference)} reference stations"
        )
    held = np.array([r is not None for r in reference])
    if not held.any():
        raise ValueError("the reference holds no station")
    ref = np.array([0.0 if r is None else float(r) for r in reference])
    diff = (values - ref)[held]
    return LineErrors(
        max_abs=float(np.abs(diff).max()),
        rms=float(np.sqrt(np.mean(diff**2))),
        stations=int(held.sum()),
    )


# The lines decision 6 scores, in the order the harness records them.
SCORED_LINES: tuple[str, ...] = ("x3", "y2916")


@dataclass(frozen=True)
class LineScore:
    """One line's RMS against the measurements, beside the published RMS.

    Parameters
    ----------
    key : str
        The line's key in LINES.
    rms : float
        RMS of u / u0 minus the measurements over every station of the line.
    stations : int
        How many stations the RMS is taken over: all the line holds.
    allowance : float
        The published prediction's RMS against the measurements on the same
        line (``published_rms``).
    scored : bool
        Whether the line is one of SCORED_LINES.
    passed : bool or None
        ``rms <= allowance`` on a scored line; None on an unscored one.
    """

    key: str
    rms: float
    stations: int
    allowance: float
    scored: bool
    passed: bool | None


def published_rms(key: str) -> float:
    """The published prediction's RMS against the measurements on one line.

    Parameters
    ----------
    key : str
        A key of LINES.

    Returns
    -------
    float
        RMS of the published u / u0 minus the measured, over the stations the
        published line reaches.
    """
    line = LINES[key]
    return line_errors(np.array(line.measured), line.published).rms


def score_lines(values: dict[str, np.ndarray]) -> dict[str, LineScore]:
    """Each line's RMS against the measurements, and on a scored line pass or fail.

    Parameters
    ----------
    values : dict[str, np.ndarray]
        u / u0 at each line's stations, as ``sample_lines`` returns them; a
        value for every key of LINES.

    Returns
    -------
    dict[str, LineScore]
        One score per key of LINES, in LINES' order.

    Raises
    ------
    ValueError
        If a line is missing, its values are not one per station, or a value
        is not finite: a NaN would make the RMS NaN, and ``NaN <= allowance``
        is False, so the line would fail without saying why.
    """
    missing = [key for key in LINES if key not in values]
    if missing:
        raise ValueError(f"no values for the lines {missing}")
    scores = {}
    for key, line in LINES.items():
        sampled = np.asarray(values[key], dtype=np.float64)
        if sampled.shape != (len(line.stations),):
            raise ValueError(
                f"{key}: {sampled.shape} values against {len(line.stations)} stations"
            )
        if not np.isfinite(sampled).all():
            raise ValueError(f"{key}: a sampled value is not finite")
        errors = line_errors(sampled, line.measured)
        allowance = published_rms(key)
        scored = key in SCORED_LINES
        scores[key] = LineScore(
            key=key,
            rms=errors.rms,
            stations=errors.stations,
            allowance=allowance,
            scored=scored,
            passed=bool(errors.rms <= allowance) if scored else None,
        )
    return scores


def vertical_line_flux(line: Line) -> float:
    """The net flux a vertical line's measured profile carries, over u0 h.

    The strips between the outermost stations and the walls are closed with
    u falling linearly to zero at the wall (the no-slip closure of
    docs/reports/product_case_reynolds.md, section 6.3). In a steady
    two-dimensional room the ratio is 1; the measured room gives about 1.17
    at x = H and 0.62 at x = 2H.

    Parameters
    ----------
    line : Line
        A vertical line.

    Returns
    -------
    float
        The integral of u / u0 over y from the floor to the ceiling, over h.

    Raises
    ------
    ValueError
        If the line is not vertical.
    """
    if not line.vertical:
        raise ValueError(f"{line.key} is not a vertical line")
    order = np.argsort(line.stations)
    y = np.array(line.stations)[order]
    u = np.array(line.measured)[order]
    yy = np.concatenate(([0.0], y, [HEIGHT]))
    uu = np.concatenate(([0.0], u, [0.0]))
    return float(np.trapezoid(uu, yy) / SLOT)
