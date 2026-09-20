"""Error metrics for the validation cases, with their reference data.

Each metric names what it measures and what it measures against, so a
later change of reference or formula is visible in every record that
carries it rather than silent.
"""

from dataclasses import dataclass, field

import numpy as np

from src.config import SimConfig
from src.mesh import FLUID, Mesh

# Ghia, Ghia and Shin (1982), Re = 100.
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
GHIA_V_VAL: tuple[float, ...] = (
    0.00000,
    -0.05906,
    -0.07391,
    -0.08864,
    -0.24533,
    -0.22445,
    -0.16914,
    -0.11477,
    -0.10313,
    -0.04272,
    0.02135,
    0.07156,
    0.09515,
    0.10091,
    0.10643,
    0.00000,
)


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
        reference="ghia_1982_re100",
        components={"u": u_err, "v": v_err},
    )
