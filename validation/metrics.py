"""Error metrics for the validation cases, with their reference data.

Each metric names what it measures and what it measures against, so a
later change of reference or formula is visible in every record that
carries it rather than silent.
"""

from dataclasses import dataclass, field

import numpy as np

from src.config import SimConfig
from src.mesh import FLUID, Mesh

# Ghia, Ghia and Shin (1982), Re = 100: Table I (u) and Table II (v).
#
# Revision r2, 2026-09-22: the v table is replaced. The one that entered the
# repository with the VAL-002 test on 2026-04-16 (d589b9f) was not Table II:
# its stations were Table I's y stations, and six of its sixteen values appear
# nowhere in Table II. It also failed a check that needs no source: the net
# vertical flux through y = 0.5 of a closed cavity is zero, and that table
# integrated to -0.095 (tests/test_validation.py, TestGhiaTables). The u table
# was right and is unchanged. The reference name moves to ghia_1982_re100_r2
# instead of keeping ghia_1982_re100, because every stored row carrying the
# old name was scored against the old table; reusing the name would silently
# change what those rows claim. See the ECR-001 erratum.
GHIA_REFERENCE = "ghia_1982_re100_r2"

# u-velocity along the vertical centerline (x = 0.5), sampled at these y.
GHIA_U_Y: tuple[float, ...] = (
    1.0000,
    0.9766,
    0.9688,
    0.9609,
    0.9531,
    0.8516,
    0.7344,
    0.6172,
    0.5000,
    0.4531,
    0.2813,
    0.1719,
    0.1016,
    0.0703,
    0.0625,
    0.0547,
    0.0000,
)
GHIA_U_VAL: tuple[float, ...] = (
    1.00000,
    0.84123,
    0.78871,
    0.73722,
    0.68717,
    0.23151,
    0.00332,
    -0.13641,
    -0.20581,
    -0.21090,
    -0.15662,
    -0.10150,
    -0.06434,
    -0.04775,
    -0.04192,
    -0.03717,
    0.00000,
)

# v-velocity along the horizontal centerline (y = 0.5), sampled at these x.
GHIA_V_X: tuple[float, ...] = (
    1.0000,
    0.9688,
    0.9609,
    0.9531,
    0.9453,
    0.9063,
    0.8594,
    0.8047,
    0.5000,
    0.2344,
    0.2266,
    0.1563,
    0.0938,
    0.0781,
    0.0703,
    0.0625,
    0.0000,
)
GHIA_V_VAL: tuple[float, ...] = (
    0.00000,
    -0.05906,
    -0.07391,
    -0.08864,
    -0.10313,
    -0.16914,
    -0.22445,
    -0.24533,
    0.05454,
    0.17527,
    0.17507,
    0.16077,
    0.12317,
    0.10890,
    0.10091,
    0.09233,
    0.00000,
)


# Marchi, Suero and Araki (2009), "The lid-driven square cavity flow: numerical
# solution with a 1024 x 1024 grid", J. Braz. Soc. Mech. Sci. & Eng. 31(3), DOI
# 10.1590/S1678-58782009000300004. Second-order finite volumes on 1024 x 1024 with
# multiple Richardson extrapolations, iterated to round-off. The Re = 100 column of
# Table 6 (solution) and Table 7 (estimated discretization error U). Each profile
# value is the mean of the two faces either side of its station, so it sits on the
# true centerline, and every velocity is over the lid speed.
#
# Extracted 2026-09-23 from the PDF's text layer; nothing was typed. pdftotext
# -layout (xpdf 4.00) and pypdf 6.19.0 agree digit for digit on every value in all
# five Re columns of both tables. The -layout text prints each Table 7 label one
# line below its own values. The glyph positions put every label within 1.5 pt of
# its row's baseline, with rows 9.7 pt apart, and that pairing is the one used.
# No metric or criterion uses this table: VAL-002 and the harness stay on
# ghia_1982_re100_r2. tests/test_validation.py, TestMarchiTable, checks it.
MARCHI_REFERENCE = "marchi_2009_re100"

# (y, u on x = 0.5, U), one row per table row.
MARCHI_U_ROWS: tuple[tuple[float, float, float], ...] = (
    (0.0625, -4.1974991e-2, 4.5e-8),
    (0.125, -7.7125399e-2, 7.2e-8),
    (0.1875, -1.09816214e-1, 8.6e-8),
    (0.25, -1.41930064e-1, 8.6e-8),
    (0.3125, -1.72712391e-1, 7.3e-8),
    (0.375, -1.98470859e-1, 5.0e-8),
    (0.4375, -2.12962392e-1, 2.0e-8),
    (0.5, -2.091491418e-1, 8.6e-9),
    (0.5625, -1.82080595e-1, 2.8e-8),
    (0.625, -1.31256301e-1, 3.5e-8),
    (0.6875, -6.0245594e-2, 3.7e-8),
    (0.75, 2.7874448e-2, 4.6e-8),
    (0.8125, 1.40425325e-1, 7.1e-8),
    (0.875, 3.1055709e-1, 1.1e-7),
    (0.9375, 5.97466694e-1, 9.5e-8),
)

# (x, v on y = 0.5, U), one row per table row.
MARCHI_V_ROWS: tuple[tuple[float, float, float], ...] = (
    (0.0625, 9.4807616e-2, 7.2e-8),
    (0.125, 1.4924300e-1, 1.0e-7),
    (0.1875, 1.74342933e-1, 9.7e-8),
    (0.25, 1.79243328e-1, 7.9e-8),
    (0.3125, 1.69132064e-1, 5.5e-8),
    (0.375, 1.45730201e-1, 2.9e-8),
    (0.4375, 1.087758646e-1, 3.7e-9),
    (0.5, 5.7536559e-2, 2.0e-8),
    (0.5625, -7.748504e-3, 4.8e-8),
    (0.625, -8.4066715e-2, 5.3e-8),
    (0.6875, -1.63010143e-1, 5.0e-8),
    (0.75, -2.27827313e-1, 5.2e-8),
    (0.8125, -2.53768577e-1, 7.3e-8),
    (0.875, -2.18690812e-1, 8.7e-8),
    (0.9375, -1.23318170e-1, 5.8e-8),
)

# Mass flow rate through y = 0.5 between x = 0 and 0.5, the paper's Eq. (4), and U.
MARCHI_M = 6.6547335e-2
MARCHI_M_ERR = 2.7e-8


@dataclass(frozen=True)
class ErrorMetric:
    """One accuracy measurement against a named reference.

    Parameters
    ----------
    metric : str
        Name of the quantity measured.
    value : float
        The measurement. Lower is better.
    reference : str
        What the solution was compared against.
    components : dict[str, float]
        Sub-measurements the value was taken from, when there are several.
    """

    metric: str
    value: float
    reference: str
    components: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Return a JSON-serialisable mapping of the metric."""
        out: dict = {
            "metric": self.metric,
            "value": self.value,
            "reference": self.reference,
        }
        if self.components:
            out["components"] = dict(self.components)
        return out


def _inlet_velocity(config: SimConfig) -> float:
    """Prescribed magnitude of the single velocity inlet in the case."""
    inlets = [
        spec for spec in config.boundaries.values() if spec.type == "velocity_inlet"
    ]
    if len(inlets) != 1 or inlets[0].velocity is None:
        raise ValueError("Poiseuille metric needs exactly one velocity inlet")
    return float(inlets[0].velocity)


def _lid_velocity(config: SimConfig) -> float:
    """Tangential speed of the single moving lid in the cavity case."""
    lids = [
        spec for spec in config.boundaries.values() if spec.type == "velocity_inlet"
    ]
    if len(lids) != 1 or lids[0].u_velocity is None:
        raise ValueError("cavity metric needs exactly one lid with u_velocity")
    return float(lids[0].u_velocity)


def poiseuille_profiles(
    config: SimConfig, mesh: Mesh, u: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mid-channel profile: (y, computed u, analytical u) over FLUID cells.

    Parameters
    ----------
    config : SimConfig
        Case configuration; supplies the channel height and inlet speed.
    mesh : Mesh
        Mesh the solution was computed on.
    u : np.ndarray
        Horizontal velocity field [ny, nx].

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Cell-centre heights, computed u and analytical u at x = L/2.
    """
    i_mid = config.nx // 2
    fluid = mesh.cell_type[:, i_mid] == FLUID
    y = np.asarray(mesh.yc)[fluid]
    u_num = u[fluid, i_mid]
    height = config.room_height
    u_max = 1.5 * _inlet_velocity(config)
    u_ref = u_max * 4.0 * y * (height - y) / height**2
    return y, u_num, u_ref


def cavity_centerline_profiles(
    config: SimConfig, mesh: Mesh, u: np.ndarray, v: np.ndarray
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Centerline profiles with wall values appended, ready for interpolation.

    Parameters
    ----------
    config : SimConfig
        Case configuration; supplies the grid size and lid speed.
    mesh : Mesh
        Mesh the solution was computed on.
    u, v : np.ndarray
        Velocity fields [ny, nx].

    Returns
    -------
    tuple[list[float], list[float], list[float], list[float]]
        (y, u along x = 0.5, x, v along y = 0.5). The floor and lid values
        bound the u profile; both side walls bound the v profile.
    """
    u_lid = _lid_velocity(config)
    i_mid = config.nx // 2
    fluid_col = mesh.cell_type[:, i_mid] == FLUID
    y_profile = [0.0, *np.asarray(mesh.yc)[fluid_col], 1.0]
    u_profile = [0.0, *u[fluid_col, i_mid], u_lid]

    j_mid = config.ny // 2
    fluid_row = mesh.cell_type[j_mid, :] == FLUID
    x_profile = [0.0, *np.asarray(mesh.xc)[fluid_row], 1.0]
    v_profile = [0.0, *v[j_mid, fluid_row], 0.0]
    return y_profile, u_profile, x_profile, v_profile


def _bracket(centers: np.ndarray, target: float) -> tuple[int, float]:
    """Index i and weight w with target = (1 - w) centers[i] + w centers[i + 1].

    A target that coincides with a center gets that center and w = 0, so an
    odd grid reads its middle column exactly.
    """
    i = int(np.searchsorted(centers, target, side="right")) - 1
    i = min(max(i, 0), len(centers) - 2)
    return i, float((target - centers[i]) / (centers[i + 1] - centers[i]))


def cavity_true_centerline_profiles(
    config: SimConfig, mesh: Mesh, u: np.ndarray, v: np.ndarray
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Profiles on the true centerlines of the cavity, walls appended.

    Parameters
    ----------
    config : SimConfig
        Case configuration; supplies the lid speed.
    mesh : Mesh
        Mesh the solution was computed on.
    u, v : np.ndarray
        Cell-centered velocity fields [ny, nx].

    Returns
    -------
    tuple[list[float], list[float], list[float], list[float]]
        (y, u along the vertical midline, x, v along the horizontal
        midline), in the layout of cavity_centerline_profiles.

    Notes
    -----
    cavity_centerline_profiles reads column nx // 2, whose center lies at
    0.5 + h/2 on an even grid, a first-order error in position. Here each
    profile is interpolated linearly between the two columns (rows) whose
    centers bracket the midline, which is second order on any mesh: the mean
    of the two middle columns on an even uniform grid, the middle column on an
    odd one. A row is kept when both of its bracketing cells are FLUID.

    Every position comes from the mesh: the midlines are halfway between its
    first and last faces, and the wall values are appended at those faces. On
    the unit cavity that is x = 0.5 and y = 0.5 with walls at 0 and 1, where
    Ghia's stations lie.
    """
    u_lid = _lid_velocity(config)
    xc, yc = np.asarray(mesh.xc), np.asarray(mesh.yc)
    i, wx = _bracket(xc, 0.5 * (mesh.x[0] + mesh.x[-1]))
    col = (mesh.cell_type[:, i] == FLUID) & (mesh.cell_type[:, i + 1] == FLUID)
    u_line = (1.0 - wx) * u[:, i] + wx * u[:, i + 1]
    y_profile = [mesh.y[0], *yc[col], mesh.y[-1]]
    u_profile = [0.0, *u_line[col], u_lid]

    j, wy = _bracket(yc, 0.5 * (mesh.y[0] + mesh.y[-1]))
    row = (mesh.cell_type[j, :] == FLUID) & (mesh.cell_type[j + 1, :] == FLUID)
    v_line = (1.0 - wy) * v[j, :] + wy * v[j + 1, :]
    x_profile = [mesh.x[0], *xc[row], mesh.x[-1]]
    v_profile = [0.0, *v_line[row], 0.0]
    return y_profile, u_profile, x_profile, v_profile


def poiseuille_l2_error(config: SimConfig, mesh: Mesh, u: np.ndarray) -> ErrorMetric:
    """L2 relative error of the mid-channel u profile against the parabola.

    Parameters
    ----------
    config : SimConfig
        Case configuration; supplies the channel height and inlet speed.
    mesh : Mesh
        Mesh the solution was computed on.
    u : np.ndarray
        Horizontal velocity field [ny, nx].

    Returns
    -------
    ErrorMetric
        Relative L2 error over the FLUID cells of the column at x = L/2.

    Notes
    -----
    For plane Poiseuille flow u(y) = u_max * 4 y (H - y) / H^2 with
    u_max = 1.5 u_mean, and a uniform inlet gives u_mean equal to the
    inlet speed.
    """
    _y, u_num, u_ref = poiseuille_profiles(config, mesh, u)
    value = float(np.sqrt(np.sum((u_num - u_ref) ** 2) / np.sum(u_ref**2)))
    return ErrorMetric(
        metric="l2_relative_error_u_midchannel",
        value=value,
        reference="analytical_poiseuille",
    )


def cavity_centerline_errors(
    config: SimConfig, mesh: Mesh, u: np.ndarray, v: np.ndarray
) -> ErrorMetric:
    """Max normalized centerline errors against Ghia et al. (1982) at Re = 100.

    Parameters
    ----------
    config : SimConfig
        Case configuration; supplies the grid size.
    mesh : Mesh
        Mesh the solution was computed on.
    u, v : np.ndarray
        Velocity fields [ny, nx].

    Returns
    -------
    ErrorMetric
        Value is the larger of the u and v errors; both are in components.

    Notes
    -----
    The solver profile is linearly interpolated onto Ghia's sample points
    with the wall values (u = 0 at the floor, u = u_lid at the lid, v = 0
    at both side walls) appended. Errors are normalized by the lid speed.
    """
    y_profile, u_profile, x_profile, v_profile = cavity_centerline_profiles(
        config, mesh, u, v
    )
    u_lid = _lid_velocity(config)
    u_err = float(
        np.max(np.abs(np.interp(GHIA_U_Y, y_profile, u_profile) / u_lid - GHIA_U_VAL))
    )
    v_err = float(
        np.max(np.abs(np.interp(GHIA_V_X, x_profile, v_profile) / u_lid - GHIA_V_VAL))
    )
    return ErrorMetric(
        metric="max_normalized_centerline_error",
        value=max(u_err, v_err),
        reference=GHIA_REFERENCE,
        components={"u": u_err, "v": v_err},
    )


def cavity_true_centerline_errors(
    config: SimConfig, mesh: Mesh, u: np.ndarray, v: np.ndarray
) -> ErrorMetric:
    """Max normalized errors against Ghia et al. (1982), sampled on the true centerlines.

    Parameters
    ----------
    config : SimConfig
        Case configuration; supplies the lid speed.
    mesh : Mesh
        Mesh the solution was computed on.
    u, v : np.ndarray
        Cell-centered velocity fields [ny, nx].

    Returns
    -------
    ErrorMetric
        Value is the larger of the u and v errors; both are in components.

    Notes
    -----
    The same comparison as cavity_centerline_errors, taken on the profiles
    of cavity_true_centerline_profiles. The metric carries its own name
    because every row stored under the old one was sampled half a cell off
    the centerlines on an even grid.
    """
    y_profile, u_profile, x_profile, v_profile = cavity_true_centerline_profiles(
        config, mesh, u, v
    )
    u_lid = _lid_velocity(config)
    u_err = float(
        np.max(np.abs(np.interp(GHIA_U_Y, y_profile, u_profile) / u_lid - GHIA_U_VAL))
    )
    v_err = float(
        np.max(np.abs(np.interp(GHIA_V_X, x_profile, v_profile) / u_lid - GHIA_V_VAL))
    )
    return ErrorMetric(
        metric="max_normalized_centerline_error_r2",
        value=max(u_err, v_err),
        reference=GHIA_REFERENCE,
        components={"u": u_err, "v": v_err},
    )
