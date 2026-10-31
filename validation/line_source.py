"""A line source in decaying grid turbulence: the data, the readers and the comparison (VAL-024).

The case is ADR-015 A: Warhaft (1984), "The interrogation of turbulent flows by
a heated wire" (J. Fluid Mech. 144, 297-338), a heated wire across a wind
tunnel behind a biplane grid. The grid mesh is M = 0.025 m (his section 2),
the mean speed U = 7 m/s, the wire at x_o / M = 52 from the grid. The mean
temperature profile behind the wire is Gaussian and he reports its half-width
at half the centreline value, sigma = 0.849 times it (his section 3.2).

The model is tested on two things in one case (ADR-015 A). The decay of k and
eps in a uniform stream with no production: standard k-epsilon's solution is
k = k_o (1 + t / t_v)^(-n) with n = 1 / (C_e2 - 1) = 1.087 and t_v = k_o / ((C_e2 -
1) eps_o). And the plume's width against the measured one, where gradient
diffusion (D_t = nu_t / Sc_t, REQ-T13) is expected wider than the measured
plume in the near field, where a real plume grows as sigma_v t, and close to it
once t is of order the Lagrangian time scale.

Frames. The solver's x runs along the stream from the inlet, the source is
SOURCE_COLUMN cells in, and a station is x' / M mesh lengths downstream of the
source cell's centre. The width at a station is read from the cell-centred
widths by linear interpolation in x.

Provenance of the data. The 52 M source's points are the rows of table T8 of
docs/reports/probe70/digitised70c.md, which digitise70c.py generates from the
marker centres read off Warhaft's figures 4(a), 4(b) and 5 (prompt 70c;
the script's docstring gives the method and its precision, about 0.3% to 0.5%
of the 80 M half-width). They are stored at the table's printed precision, x' /
M to two decimals and the half-width over M to four, and
tests/test_line_source.py parses the table and compares every row. The 20 M
source's four points and the integral scales of Warhaft's table 1 serve one
estimate at 100 M, where the 52 M data end. The reference at a station is the
52 M source's points interpolated linearly in log x and log sigma between their
stations (``reference_sigma``), as T9 computes it; the Anand and Pope curve
(``reference_curve``, Viswanathan and Pope 2008 equations 11 to 15) is not the
interpolant and checks the data.

The criterion is Alex's (ADR-015 A, 2026-10-30): the decay exponent within
DECAY_ALLOWANCE of the model's own, and the width within WIDTH_ALLOWANCE of the
measured one at SCORED_STATIONS_M. The other stations are reported and not
scored (ADR-015 A: 5 and 10 M are the free fall, which never reaches the
sensors; 80 M is a carried limit, the model's plume about a fifth narrower than
measured there; 100 M has no measurement behind it).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.boundary_concentration import ConcentrationBoundary
from src.boundary_registry import BoundaryRegistry
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import Mesh
from src.particles import ParticlePhysics
from src.solver_staggered import StaggeredSolver
from src.solver_transport import TransportSolver
from src.staggered import FaceVelocities
from src.turbulence import STANDARD, VARIANTS

# Warhaft (1984) as Viswanathan and Pope (2008, Phys. Fluids 20, table I)
# tabulate it, at his own grid mesh.
MESH_LENGTH = 0.025  # m, Warhaft section 2
SPEED = 7.0  # m/s
SOURCE_STATION = 52.0  # x_o / M of the wire
SIGMA_U_AT_M = 2.44  # m/s, at one mesh length from the grid
SIGMA_V_AT_M = 2.27  # m/s, sigma_v = sigma_w
DECAY_EXPONENT_M = 1.4  # the velocity variance's exponent, Sawford's refit
LANGEVIN_C0 = 2.1  # the Langevin constant of the reference curve
MOLECULAR_DIFFUSIVITY = 2.1e-5  # m^2/s, heat in air
WIRE_SIZE = 2.5e-5  # m, the wire's effective source size
HW_TO_SIGMA = 0.849  # Warhaft section 3.2: sigma of a Gaussian over its half-width

# The 52 M source's points, T8 of digitised70c.md: (wire, x' / M, half-width / M).
# The fine wire (0.025 mm) is plotted to 8 M and the thick one (0.127 mm) from 8 M.
FINE, THICK = "0.025 mm", "0.127 mm"
WARHAFT_52M: tuple[tuple[str, float, float], ...] = (
    (FINE, 0.22, 0.0123),
    (FINE, 0.55, 0.0173),
    (FINE, 0.89, 0.0279),
    (FINE, 3.59, 0.0883),
    (FINE, 3.67, 0.0840),
    (FINE, 5.20, 0.1413),
    (FINE, 5.69, 0.1416),
    (THICK, 7.97, 0.1703),
    (FINE, 7.99, 0.2098),
    (THICK, 8.09, 0.1955),
    (FINE, 8.13, 0.2064),
    (THICK, 8.31, 0.1749),
    (THICK, 29.90, 0.5558),
    (THICK, 30.16, 0.5610),
    (THICK, 59.99, 0.8306),
    (THICK, 60.24, 0.8448),
    (THICK, 80.04, 1.0301),
    (THICK, 80.45, 1.0161),
)
# The 20 M source's points (T8), (x' / M, half-width / M), and the integral
# scale u'^3 / eps at the two source stations (Warhaft's table 1), m.
WARHAFT_20M: tuple[tuple[float, float], ...] = (
    (7.71, 0.2402),
    (32.93, 0.7314),
    (91.93, 1.3401),
    (132.58, 1.8514),
)
INTEGRAL_SCALE = {20.0: 1.02e-2, 52.0: 1.35e-2}
# Points within this many mesh lengths of one another, by round(x / BIN), are one
# station of the composite (the figures repeat 30, 60 and 80 M; the near-source
# points of the fine wire fall in the first bin).
STATION_BIN = 2.0
# Where the two wires' readings are averaged into one station, and the limits of
# the fine wire's range and the thick wire's.
JOIN_STATION = 8.1
FINE_LAST, THICK_FIRST = 7.5, 9.0

# The model's decay, the criterion and the protocol.
STATIONS_M = (5.0, 10.0, 20.0, 50.0, 80.0, 100.0)
SCORED_STATIONS_M = (20.0, 50.0)
DECAY_ALLOWANCE = 0.1
WIDTH_ALLOWANCE = 0.20
SOURCE_COLUMN = 2  # the source cell's column; the inlet is the left edge
SOURCE_RATE = 1.0  # per cell, the unit source
FIT_RANGE = (0.1, 0.9)  # the decay is read over this fraction of the domain
STEADY_RTOL = 1.0e-6  # the march's relative change at which the field is steady
TIME_CAP = 3.0  # s, the march's cap
NEAR_SOURCE = (
    0.5  # columns nearer than this many mesh lengths to the source hold the source cell
)

# The inlet segment and the report's metric.
INLET = "stream"
# The inlet's keys are the experiment's constants written at full precision;
# this is the tolerance ``evaluate`` holds them to.
INLET_RTOL = 1.0e-9
METRIC = "largest_scored_width_departure"
REFERENCE = (
    "warhaft_1984_52M_digitised_T8_loglog; criterion Alex 2026-10-30 (ADR-015 A): "
    "decay exponent within 0.1 of the model's own, width within 20% at 20 and 50 M"
)


@dataclass(frozen=True)
class SourceTurbulence:
    """The turbulence at the source station, the inlet's and the reference's.

    Parameters
    ----------
    sigma_u, sigma_v : float
        The velocity standard deviations at the station, m/s.
    k, eps : float
        Turbulent kinetic energy, m^2/s^2, and its dissipation rate, m^2/s^3.
    intensity, dissipation_length : float
        The inlet keys ``turbulence_intensity`` and ``dissipation_length``
        that give k and eps (ADR-012 C), the second in m.
    k_over_eps, lagrangian_time, flight_time : float
        k / eps, s; the Lagrangian time scale (4 / (3 C_0)) k / eps, s; and the
        flight time x_o / U from the grid to the wire, s.
    virtual_time : float
        The model's virtual origin t_v = k_o / ((C_e2 - 1) eps_o), s.
    """

    sigma_u: float
    sigma_v: float
    k: float
    eps: float
    intensity: float
    dissipation_length: float
    k_over_eps: float
    lagrangian_time: float
    flight_time: float
    virtual_time: float


def model_decay_exponent() -> float:
    """Return standard k-epsilon's decay exponent 1 / (C_e2 - 1) (ADR-012 A).

    Returns
    -------
    float
        n of k = k_o (1 + t / t_v)^(-n) in a uniform stream with no production,
        1.087 for C_e2 = 1.92.
    """
    return 1.0 / (VARIANTS[STANDARD].c_2 - 1.0)


def source_turbulence(mesh_length: float = MESH_LENGTH) -> SourceTurbulence:
    """Return k, eps and the inlet keys at the wire's station.

    Parameters
    ----------
    mesh_length : float
        The grid mesh M in metres; sets the flight time x_o M / U, so eps =
        m k_o U / (x_o M).

    Returns
    -------
    SourceTurbulence
        The velocity standard deviations decay as (x / M)^(-m / 2), k = (sigma_u^2
        + 2 sigma_v^2) / 2, and eps = m k / t_o, the exact decay rate of that
        power law at the station.
    """
    decay = SOURCE_STATION ** (-DECAY_EXPONENT_M / 2.0)
    sigma_u, sigma_v = SIGMA_U_AT_M * decay, SIGMA_V_AT_M * decay
    k = 0.5 * (sigma_u**2 + 2.0 * sigma_v**2)
    flight = SOURCE_STATION * mesh_length / SPEED
    eps = DECAY_EXPONENT_M * k / flight
    k_over_eps = k / eps
    return SourceTurbulence(
        sigma_u=sigma_u,
        sigma_v=sigma_v,
        k=k,
        eps=eps,
        intensity=float(np.sqrt(k / 1.5) / SPEED),
        dissipation_length=float(k**1.5 / eps),
        k_over_eps=k_over_eps,
        lagrangian_time=4.0 / (3.0 * LANGEVIN_C0) * k_over_eps,
        flight_time=flight,
        virtual_time=k / ((VARIANTS[STANDARD].c_2 - 1.0) * eps),
    )


def reference_curve(x_over_m: np.ndarray | float) -> np.ndarray:
    """Return the Anand and Pope plume width for the 52 M source, in metres.

    Parameters
    ----------
    x_over_m : np.ndarray or float
        Distance downstream of the source in mesh lengths, x' / M.

    Returns
    -------
    np.ndarray
        sqrt(sigma_o^2 + 2 kappa t + Delta^2(t)), the displacement variance of
        the Langevin model with C_0 = 2.1 in the decaying turbulence
        (Viswanathan and Pope 2008, equations 11 to 15), t = x' M / U.

    Notes
    -----
    Delta^2 = 2 sigma_v^2(t_o) t_o^2 [(1 + t / t_o)^(r - s) / (r (r - s)) +
    (1 + t / t_o)^(-s) / (r s) - 1 / (s (r - s))], with r = (m / 2)(3 C_0 / 2 -
    1) + 1 and s = (m / 2)(3 C_0 / 2 + 1) - 1. The curve is the check on the
    digitised data, which it matches within 5% from 30 to 80 M (ADR-015 A).
    """
    turbulence = source_turbulence()
    t = np.asarray(x_over_m, dtype=float) * MESH_LENGTH / SPEED
    r = 0.5 * DECAY_EXPONENT_M * (1.5 * LANGEVIN_C0 - 1.0) + 1.0
    s = 0.5 * DECAY_EXPONENT_M * (1.5 * LANGEVIN_C0 + 1.0) - 1.0
    tau = t / turbulence.flight_time
    bracket = (
        (1.0 + tau) ** (r - s) / (r * (r - s))
        + (1.0 + tau) ** (-s) / (r * s)
        - 1.0 / (s * (r - s))
    )
    delta2 = 2.0 * turbulence.sigma_v**2 * turbulence.flight_time**2 * bracket
    return np.sqrt(WIRE_SIZE**2 + 2.0 * MOLECULAR_DIFFUSIVITY * t + delta2)


def loglog_interp(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """Interpolate y at x linearly in log x and log y between the points.

    Parameters
    ----------
    x : float
        The abscissa, positive and within [xs[0], xs[-1]].
    xs, ys : np.ndarray
        Positive points, ``xs`` increasing.

    Returns
    -------
    float
        exp of the linear interpolant of ln y against ln x at ln x.
    """
    return float(np.exp(np.interp(np.log(x), np.log(xs), np.log(ys))))


def composite_52m() -> tuple[np.ndarray, np.ndarray]:
    """Return the 52 M source as Warhaft plots it: one sigma per station.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x' / M increasing and sigma in metres. The fine wire is used to
        FINE_LAST, the thick wire from THICK_FIRST, and the points of both
        between them are averaged into one at JOIN_STATION; points in the same
        STATION_BIN are averaged into one station. sigma = HW_TO_SIGMA x the
        half-width x M.
    """
    fine = sorted((x, hw) for wire, x, hw in WARHAFT_52M if wire == FINE)
    thick = sorted((x, hw) for wire, x, hw in WARHAFT_52M if wire == THICK)

    def sigma(hw: float) -> float:
        return HW_TO_SIGMA * hw * MESH_LENGTH

    joined = [sigma(hw) for x, hw in fine if x > FINE_LAST] + [
        sigma(hw) for x, hw in thick if x < THICK_FIRST
    ]
    points = [(x, sigma(hw)) for x, hw in fine if x <= FINE_LAST]
    points.append((JOIN_STATION, float(np.mean(joined))))
    points += [(x, sigma(hw)) for x, hw in thick if x >= THICK_FIRST]
    merged: dict[int, list[tuple[float, float]]] = {}
    for x, s in points:
        merged.setdefault(round(x / STATION_BIN), []).append((x, s))
    xs = np.array([np.mean([x for x, _ in group]) for group in merged.values()])
    ys = np.array([np.mean([s for _, s in group]) for group in merged.values()])
    order = np.argsort(xs)
    return xs[order], ys[order]


def reference_sigma(station: float) -> float:
    """Return the measured plume width at a station, in metres.

    Parameters
    ----------
    station : float
        x' / M, within the 52 M data's range (the first to the last station of
        ``composite_52m``).

    Returns
    -------
    float
        The composite's points interpolated in log-log (T9's reference).

    Raises
    ------
    ValueError
        If the station lies outside the data; ``extrapolated_sigma`` and
        ``collapse_sigma`` serve past its end.
    """
    xs, ys = composite_52m()
    if not xs[0] <= station <= xs[-1]:
        raise ValueError(
            f"station {station} M is outside the 52 M data, {xs[0]:.2f} to "
            f"{xs[-1]:.2f} M"
        )
    return loglog_interp(station, xs, ys)


def extrapolated_sigma(station: float) -> float:
    """Return the width past the data's end at the slope of its last two stations.

    Parameters
    ----------
    station : float
        x' / M beyond the last station of ``composite_52m``.

    Returns
    -------
    float
        sigma in metres, the last point carried on along the log-log slope of the
        last two (0.69 for the 60 to 80 M pair).
    """
    xs, ys = composite_52m()
    slope = np.log(ys[-1] / ys[-2]) / np.log(xs[-1] / xs[-2])
    return float(ys[-1] * (station / xs[-1]) ** slope)


def collapse_sigma(station: float) -> float:
    """Return the width past the data's end read through the 20 M source.

    Parameters
    ----------
    station : float
        x' / M for the 52 M source.

    Returns
    -------
    float
        sigma in metres: the 20 M source's half-width at the same x' / x_o
        (log-log between its points), scaled by the ratio of the two stations'
        integral scales (Warhaft's figure 10 collapses half-widths over the
        integral scale), times HW_TO_SIGMA and M.
    """
    xs = np.array([x for x, _ in WARHAFT_20M])
    half_widths = np.array([hw for _, hw in WARHAFT_20M])
    at_20 = loglog_interp(station / SOURCE_STATION * 20.0, xs, half_widths)
    scale = INTEGRAL_SCALE[SOURCE_STATION] / INTEGRAL_SCALE[20.0]
    return HW_TO_SIGMA * at_20 * scale * MESH_LENGTH


def decay_exponent(mesh: Mesh, k_line: np.ndarray, source: SourceTurbulence) -> float:
    """Return the decay exponent n of k along the centreline row.

    Parameters
    ----------
    mesh : Mesh
        The case's mesh.
    k_line : np.ndarray
        k along the row, [nx], with x the stream's axis.
    source : SourceTurbulence
        Gives the model's virtual origin t_v.

    Returns
    -------
    float
        Minus the slope of ln k against ln(t_v + x / U) over the middle of the
        domain, FIT_RANGE of its length. The model's own solution k = k_o (1 + t
        / t_v)^(-n) is a straight line of slope -n on those axes.
    """
    xc = np.asarray(mesh.xc)
    length = float(mesh.x[-1])
    chosen = (xc >= FIT_RANGE[0] * length) & (xc <= FIT_RANGE[1] * length)
    time_axis = source.virtual_time + xc / SPEED
    slope, _intercept = np.polyfit(np.log(time_axis[chosen]), np.log(k_line[chosen]), 1)
    return -float(slope)


def plume_widths(mesh: Mesh, field: np.ndarray) -> np.ndarray:
    """Return the second moment of each column's profile about its own mean.

    Parameters
    ----------
    mesh : Mesh
        The case's mesh.
    field : np.ndarray
        The concentration, [ny, nx].

    Returns
    -------
    np.ndarray
        [nx], metres: sqrt(sum C (y - y_mean)^2 / sum C), which for a Gaussian
        is its standard deviation (the experiment fits Gaussians). NaN where a
        column holds no positive concentration.
    """
    yc = np.asarray(mesh.yc)
    out = np.full(field.shape[1], np.nan)
    for i in range(field.shape[1]):
        column = field[:, i]
        total = column.sum()
        if total <= 0.0:
            continue
        mean = (column * yc).sum() / total
        out[i] = float(np.sqrt((column * (yc - mean) ** 2).sum() / total))
    return out


def width_at(mesh: Mesh, widths: np.ndarray, source_x: float, station: float) -> float:
    """Return the plume width at a station, in metres.

    Parameters
    ----------
    mesh : Mesh
        The case's mesh.
    widths : np.ndarray
        ``plume_widths`` of the field, [nx].
    source_x : float
        The source cell's centre, m.
    station : float
        x' / M downstream of the source.

    Returns
    -------
    float
        The width, linear in x between the cell centres more than NEAR_SOURCE
        mesh lengths downstream of the source (the columns nearer hold the
        source cell itself).
    """
    xc = np.asarray(mesh.xc)
    x_prime = (xc - source_x) / MESH_LENGTH
    ok = x_prime > NEAR_SOURCE
    usable = x_prime[ok]
    if not usable[0] <= station <= usable[-1]:
        # np.interp would return the end column's width for a station past it.
        raise ValueError(
            f"station {station:g} M lies outside the columns read, "
            f"{usable[0]:.2f} to {usable[-1]:.2f} M downstream of the source"
        )
    return float(np.interp(station, usable, widths[ok]))


@dataclass(frozen=True)
class Solution:
    """What a solve of the case leaves: the flow's k, the plume and the costs.

    Parameters
    ----------
    mesh : Mesh
        The case's mesh.
    config : SimConfig
        The case.
    k : np.ndarray
        The final turbulent kinetic energy, [ny, nx].
    field : np.ndarray
        The plume's concentration at the steady state, [ny, nx].
    flow : dict[str, object]
        ``stop_reason``, ``outer_iterations`` and ``wall_seconds`` of the flow solve.
    march : dict[str, object]
        ``dt``, ``steps``, ``t_end``, ``stopped_by`` ("steady" or "cap"),
        ``last_relative_change`` and ``wall_seconds`` of the scalar march.
    """

    mesh: Mesh
    config: SimConfig
    k: np.ndarray
    field: np.ndarray
    flow: dict[str, object]
    march: dict[str, object]


def source_row(mesh: Mesh) -> int:
    """Return the row of the source cell: the first above the centreline.

    Parameters
    ----------
    mesh : Mesh
        The case's mesh, [ny, nx] cells.

    Returns
    -------
    int
        The row index ny // 2, the lower of the two rows beside the
        centreline on an even count.
    """
    return mesh.cell_type.shape[0] // 2


def march_scalar(
    mesh: Mesh,
    config: SimConfig,
    nu_t: np.ndarray,
    faces: FaceVelocities,
) -> tuple[np.ndarray, dict[str, object]]:
    """March one passive class from a unit source to its steady state.

    Parameters
    ----------
    mesh : Mesh
        The case's mesh.
    config : SimConfig
        The case, with a transport section and one particle class.
    nu_t : np.ndarray
        The converged eddy viscosity, [ny, nx], m^2/s.
    faces : FaceVelocities
        The converged face velocities (``StaggeredSolver.face_velocities``).

    Returns
    -------
    tuple[np.ndarray, dict[str, object]]
        The concentration, [ny, nx], and the march's record (``dt``, ``steps``,
        ``t_end``, ``stopped_by``, ``last_relative_change``, ``wall_seconds``).
        The source is SOURCE_RATE in the cell SOURCE_COLUMN along the row
        ``source_row``; the march stops when the relative change of a step falls
        to STEADY_RTOL or at TIME_CAP.
    """
    physics = ParticlePhysics(config)
    transport = TransportSolver(
        mesh,
        config,
        physics,
        ConcentrationBoundary(mesh, config, physics, BoundaryRegistry(config)),
    )
    viscosity = np.ascontiguousarray(nu_t, dtype=np.float64)
    dt = transport.stable_dt(faces, 0)
    ny, nx = mesh.cell_type.shape
    sources = np.zeros((ny, nx))
    sources[source_row(mesh), SOURCE_COLUMN] = SOURCE_RATE
    field = np.zeros((ny, nx))
    started = time.perf_counter()
    t, step, change = 0.0, 0, float("inf")
    while t < TIME_CAP and change > STEADY_RTOL:
        step += 1
        new = transport.solve_timestep(
            field,
            faces,
            0,
            dt,
            sources=sources,
            eddy_viscosity=viscosity,
        )
        change = float(np.abs(new - field).max() / max(new.max(), 1e-300))
        field = new
        t = step * dt
    return field, {
        "dt": dt,
        "steps": step,
        "t_end": t,
        "stopped_by": "steady" if change <= STEADY_RTOL else "cap",
        "last_relative_change": change,
        "wall_seconds": time.perf_counter() - started,
    }


def solve(config: SimConfig) -> Solution:
    """Solve the case: the flow to its stop, then the plume to steady.

    Parameters
    ----------
    config : SimConfig
        The case from ``validation.cases.load_line_source``.

    Returns
    -------
    Solution
        The final k, the plume and the costs.

    Raises
    ------
    RuntimeError
        If the solver holds no flow state or face velocities after
        ``solve_steady`` returns. The solver sets both before it returns and
        raises otherwise, so this is a guard on its contract for the type
        checker, not a path a case reaches.
    """
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    started = time.perf_counter()
    solver.solve_steady()
    flow_seconds = time.perf_counter() - started
    state, faces = solver.flow_state, solver.face_velocities
    if state is None or faces is None:
        raise RuntimeError("the flow solve left no state to march the plume on")
    field, march = march_scalar(mesh, config, state.nu_t, faces)
    return Solution(
        mesh=mesh,
        config=config,
        k=np.array(state.k),
        field=field,
        flow={
            "stop_reason": solver.stop_reason,
            "outer_iterations": len(solver.residual_history),
            "wall_seconds": flow_seconds,
        },
        march=march,
    )


def evaluate(mesh: Mesh, config: SimConfig, k: np.ndarray, field: np.ndarray) -> dict:
    """Compare the solve with the measurements and score the criterion.

    Parameters
    ----------
    mesh : Mesh
        The case's mesh.
    config : SimConfig
        The case. Its inlet must carry the experiment's speed and the source
        station's turbulence keys (``source_turbulence``), which set the
        reference's parameters and the decay's virtual origin.
    k : np.ndarray
        The final turbulent kinetic energy, [ny, nx].
    field : np.ndarray
        The plume's concentration, [ny, nx].

    Returns
    -------
    dict
        ``decay`` (the exponent ``n``, the model's own ``n_model``, the
        ``departure`` between them, the ``allowance`` and ``passed``);
        ``source_column_sigma_m``, the plume's width at the source column
        itself (the cell's size on the grid, reported beside the near-field
        ratios); ``stations`` (per station of STATIONS_M: ``model_sigma_m``,
        the data (``data_sigma_m``, a list of one estimate or two at 100 M),
        ``ratio`` the model over each estimate, ``reference_curve_sigma_m``,
        ``scored`` and, for a scored station, ``departure`` and ``passed``);
        ``value`` the largest departure of a scored ratio from 1; and
        ``passed`` the decay and every scored station together. Nothing
        outside the criterion changes ``passed``.

    Raises
    ------
    ValueError
        If the inlet's speed or turbulence keys are not the experiment's, so
        that the reference and the virtual origin would not be the case's.
    """
    turbulence = source_turbulence()
    inlet = config.boundaries[INLET]
    given = (inlet.velocity, inlet.turbulence_intensity, inlet.dissipation_length)
    wanted = (SPEED, turbulence.intensity, turbulence.dissipation_length)
    if not np.allclose(given, wanted, rtol=INLET_RTOL, atol=0.0):
        raise ValueError(
            f"the inlet {INLET!r} carries velocity, intensity and dissipation length "
            f"{given}; the reference assumes {wanted}"
        )
    n = decay_exponent(mesh, np.asarray(k)[source_row(mesh), :], turbulence)
    n_model = model_decay_exponent()
    decay = {
        "n": n,
        "n_model": n_model,
        "departure": abs(n - n_model),
        "allowance": DECAY_ALLOWANCE,
        "passed": abs(n - n_model) <= DECAY_ALLOWANCE,
    }
    widths = plume_widths(mesh, field)
    source_x = float(np.asarray(mesh.xc)[SOURCE_COLUMN])
    xs, _ys = composite_52m()
    stations = []
    for station in STATIONS_M:
        model = width_at(mesh, widths, source_x, station)
        if station <= xs[-1]:
            data = [reference_sigma(station)]
        else:
            data = [extrapolated_sigma(station), collapse_sigma(station)]
        row: dict[str, object] = {
            "station": station,
            "model_sigma_m": model,
            "data_sigma_m": data,
            "ratio": [model / d for d in data],
            "reference_curve_sigma_m": float(reference_curve(station)),
            "scored": station in SCORED_STATIONS_M,
        }
        if row["scored"]:
            departure = abs(model / data[0] - 1.0)
            row["departure"] = departure
            row["passed"] = departure <= WIDTH_ALLOWANCE
        stations.append(row)
    scored = [row for row in stations if row["scored"]]
    return {
        "decay": decay,
        "source_column_sigma_m": float(widths[SOURCE_COLUMN]),
        "stations": stations,
        "value": max(float(row["departure"]) for row in scored),  # type: ignore[arg-type]
        "passed": bool(decay["passed"] and all(row["passed"] for row in scored)),
    }
