"""Lift and drag on a named obstacle, and the Strouhal number of its lift (ADR-014 F).

ADR-014 step 6b. The force the fluid exerts on an obstacle, per unit depth,
is the pressure on its faces and the shear on them, read off the staggered
fields where the momentum predictor's own stencils put them:

- Pressure. Each face of the obstacle takes the pressure of the non-SOLID
  cell beside it, the cell centre half a cell from the face, times the
  face's length: on the left face it pushes in +x, on the right in -x, on
  the bottom in +y, on the top in -y.
- Shear. The obstacle's top and bottom carry the x-shear of the u unknowns
  beside them, and its sides the y-shear of the v unknowns, each the
  momentum predictor's wall stencil at an obstacle face (src/momentum.py,
  ADR-012 B): ``mu (phi_P - 0) / d`` over the unknown's control-volume
  width, ``d`` half the unknown's cell. A face whose neighbour bounds the
  obstacle's cell at a corner is a wall over its whole span, as the stencil
  treats it. The normal viscous stress on a rigid no-slip face is zero
  (continuity makes the normal derivative of the normal velocity vanish
  there), so the sides carry no x-shear and the top and bottom no y-shear.

The coefficients divide by ``0.5 rho U^2 D`` with U the configured reference
velocity and D the obstacle's height, as Sohankar, Norberg and Davidson
(1998) define them for the square cylinder.

The Strouhal number (``strouhal``) is read from the lift: the period is the
mean spacing of the last eleven upward zero crossings, each placed by linear
interpolation between the two samples around it, so ten periods; the
spectrum's peak over the same window is the cross-check. Fewer than ten
periods is refused, never extrapolated.
"""

import math
from dataclasses import dataclass

import numpy as np

from src.config import SimConfig
from src.mesh import SOLID, Mesh

# The periods the Strouhal number is read over (ADR-014 F: the last ten).
STROUHAL_PERIODS = 10


@dataclass(frozen=True)
class ObstacleForces:
    """The force on one obstacle, per unit depth, N/m, split by kind.

    Parameters
    ----------
    fx_pressure, fx_viscous : float
        The x-force from the pressure on the sides and the shear on the top
        and bottom; positive downstream (+x).
    fy_pressure, fy_viscous : float
        The y-force from the pressure on the top and bottom and the shear on
        the sides; positive up (+y).
    """

    fx_pressure: float
    fx_viscous: float
    fy_pressure: float
    fy_viscous: float

    @property
    def drag(self) -> float:
        """The x-force, N/m."""
        return self.fx_pressure + self.fx_viscous

    @property
    def lift(self) -> float:
        """The y-force, N/m."""
        return self.fy_pressure + self.fy_viscous


class ForceGauge:
    """Integrate the pressure and shear on one named obstacle's faces.

    Parameters
    ----------
    mesh : Mesh
        The mesh the fields live on.
    config : SimConfig
        Supplies rho, mu and the obstacle.
    obstacle : str
        The obstacle's name in ``config.obstacles``.
    reference_velocity : float
        U of the coefficients, m/s, finite and positive.

    Attributes
    ----------
    height : float
        D of the coefficients: the obstacle's extent in y, m.

    Raises
    ------
    ValueError
        If no obstacle has the name, it covers no cell, or the reference
        velocity is not finite and positive.
    TypeError
        If the reference velocity is not a real number (a bool is refused).
    """

    def __init__(
        self, mesh: Mesh, config: SimConfig, obstacle: str, reference_velocity: float
    ) -> None:
        named = [o for o in config.obstacles if o.name == obstacle]
        if not named:
            raise ValueError(
                f"no obstacle is named {obstacle!r}; the configuration has "
                f"{[o.name for o in config.obstacles]}"
            )
        if isinstance(reference_velocity, bool) or not isinstance(
            reference_velocity, (int, float)
        ):
            raise TypeError(
                "reference_velocity must be a real number, got "
                f"{type(reference_velocity).__name__}"
            )
        if not math.isfinite(reference_velocity) or reference_velocity <= 0.0:
            raise ValueError(
                f"reference_velocity must be finite and positive, got {reference_velocity}"
            )
        spec = named[0]
        body = mesh.obstacle_mask(spec)
        if not body.any():
            raise ValueError(f"obstacle {obstacle!r} covers no cell of this mesh")
        self.height: float = spec.y_end - spec.y_start
        self._dynamic = 0.5 * config.rho * float(reference_velocity) ** 2 * self.height
        self._mu = config.mu
        live = mesh.cell_type != SOLID
        ny, nx = body.shape

        # Pressure: the live cell beside each face of the body, with the face's
        # length signed by the direction the pressure there pushes the body:
        # the cell to the west (+x), east (-x), south (+y) and north (-y).
        rows, cols = np.nonzero(body)
        x_faces, y_faces = [], []
        for dj, di, sign in ((0, -1, 1.0), (0, 1, -1.0), (-1, 0, 1.0), (1, 0, -1.0)):
            jn, in_ = rows + dj, cols + di
            on_mesh = (jn >= 0) & (jn < ny) & (in_ >= 0) & (in_ < nx)
            jn, in_ = jn[on_mesh], in_[on_mesh]
            j, i = rows[on_mesh], cols[on_mesh]
            beside = live[jn, in_]
            jn, in_, j, i = jn[beside], in_[beside], j[beside], i[beside]
            if di:
                x_faces.append((jn, in_, sign * mesh.dy_cell[j]))
            else:
                y_faces.append((jn, in_, sign * mesh.dx_cell[i]))
        self._pressure_x = _stack(x_faces)
        self._pressure_y = _stack(y_faces)

        # Shear: the u unknowns above and below the body, the v unknowns
        # left and right of it, each beside a face that bounds a body cell.
        solid = mesh.cell_type == SOLID
        u_face_body = np.zeros((ny, nx + 1), dtype=bool)
        u_face_body[:, 1:-1] = body[:, :-1] | body[:, 1:]
        u_face_body[:, 0] = body[:, 0]
        u_face_body[:, -1] = body[:, -1]
        u_unknown = np.zeros((ny, nx + 1), dtype=bool)
        u_unknown[:, 1:-1] = ~(solid[:, :-1] | solid[:, 1:])
        v_face_body = np.zeros((ny + 1, nx), dtype=bool)
        v_face_body[1:-1, :] = body[:-1, :] | body[1:, :]
        v_face_body[0, :] = body[0, :]
        v_face_body[-1, :] = body[-1, :]
        v_unknown = np.zeros((ny + 1, nx), dtype=bool)
        v_unknown[1:-1, :] = ~(solid[:-1, :] | solid[1:, :])

        # u at (yc[j], x[i]); the wall below is at y[j], the wall above at y[j+1].
        below = np.zeros_like(u_unknown)
        below[1:, :] = u_unknown[1:, :] & u_face_body[:-1, :]
        above = np.zeros_like(u_unknown)
        above[:-1, :] = u_unknown[:-1, :] & u_face_body[1:, :]
        half_y = 0.5 * mesh.dy_cell[:, None]
        width_x = mesh.dx_face[None, :]
        self._shear_u = (below | above) * self._mu * width_x / half_y

        # v at (y[j], xc[i]); the wall left is at x[i], the wall right at x[i+1].
        left = np.zeros_like(v_unknown)
        left[:, 1:] = v_unknown[:, 1:] & v_face_body[:, :-1]
        right = np.zeros_like(v_unknown)
        right[:, :-1] = v_unknown[:, :-1] & v_face_body[:, 1:]
        half_x = 0.5 * mesh.dx_cell[None, :]
        width_y = mesh.dy_face[:, None]
        self._shear_v = (left | right) * self._mu * width_y / half_x

    def forces(self, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> ObstacleForces:
        """The force the fields put on the obstacle, per unit depth.

        Parameters
        ----------
        u : np.ndarray
            x-velocity on the vertical faces, m/s, [ny, nx+1].
        v : np.ndarray
            y-velocity on the horizontal faces, m/s, [ny+1, nx].
        p : np.ndarray
            Pressure at the cell centres, Pa, [ny, nx].

        Returns
        -------
        ObstacleForces
            The four parts, N/m.
        """
        jx, ix, wx = self._pressure_x
        jy, iy, wy = self._pressure_y
        return ObstacleForces(
            fx_pressure=float(np.sum(wx * p[jx, ix])),
            fx_viscous=float(np.sum(self._shear_u * u)),
            fy_pressure=float(np.sum(wy * p[jy, iy])),
            fy_viscous=float(np.sum(self._shear_v * v)),
        )

    def coefficients(self, forces: ObstacleForces) -> tuple[float, float]:
        """C_D and C_L: the drag and the lift over 0.5 rho U^2 D.

        Parameters
        ----------
        forces : ObstacleForces
            From ``forces``.

        Returns
        -------
        tuple[float, float]
            (C_D, C_L), dimensionless.
        """
        return forces.drag / self._dynamic, forces.lift / self._dynamic


def _stack(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rows, columns and signed lengths of every face of one kind, joined."""
    return (
        np.concatenate([part[0] for part in parts]),
        np.concatenate([part[1] for part in parts]),
        np.concatenate([part[2] for part in parts]),
    )


@dataclass(frozen=True)
class StrouhalReading:
    """The Strouhal number read from a lift signal.

    Parameters
    ----------
    strouhal : float
        ``f D / U`` with f one over the mean period of the last ten.
    period : float
        That mean period, s.
    crossings : np.ndarray
        The times of the eleven upward zero crossings the periods span, s.
    spectral_strouhal : float
        The Strouhal number of the spectrum's peak over the same window.
    resolution : float
        The spectrum's Strouhal resolution, ``D / (U T)`` with T the window.
    agrees : bool
        Whether the two readings are within one resolution of each other.
    """

    strouhal: float
    period: float
    crossings: np.ndarray
    spectral_strouhal: float
    resolution: float
    agrees: bool


def strouhal(
    t: np.ndarray, lift: np.ndarray, height: float, velocity: float
) -> StrouhalReading:
    """The Strouhal number of the last ten periods of a lift signal.

    Parameters
    ----------
    t : np.ndarray
        Sample times, s, strictly increasing, evenly spaced.
    lift : np.ndarray
        The lift (or its coefficient) at those times, finite.
    height : float
        D, m, finite and positive.
    velocity : float
        U, m/s, finite and positive.

    Returns
    -------
    StrouhalReading
        The period from the zero crossings and the spectrum's cross-check.

    Raises
    ------
    TypeError
        If ``height`` or ``velocity`` is not a real number (a bool is refused).
    ValueError
        If the arrays differ in shape, are not 1-D, not finite, the times are
        not strictly increasing and evenly spaced, D or U is not finite and
        positive, or the signal holds fewer than ten periods (eleven upward
        crossings).

    Notes
    -----
    An upward crossing lies between samples k and k+1 where ``lift[k] < 0 <=
    lift[k+1]``, at ``t[k] - lift[k] (t[k+1] - t[k]) / (lift[k+1] - lift[k])``.
    The spectrum is the magnitude of the real FFT of the window from the
    eleventh-last crossing to the end of the signal less its mean; its peak
    bin, the zero frequency excluded, is the cross-check.
    """
    t = np.asarray(t, dtype=np.float64)
    lift = np.asarray(lift, dtype=np.float64)
    if t.ndim != 1 or t.shape != lift.shape:
        raise ValueError(
            f"t and lift must be 1-D of one shape, got {t.shape} and {lift.shape}"
        )
    if not (np.isfinite(t).all() and np.isfinite(lift).all()):
        raise ValueError("t and lift must be finite")
    for name, value in (("height", height), ("velocity", velocity)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
    steps = np.diff(t)
    if (
        steps.size == 0
        or np.any(steps <= 0.0)
        or not np.allclose(steps, steps[0], rtol=1e-9)
    ):
        raise ValueError("t must be strictly increasing and evenly spaced")
    up = np.nonzero((lift[:-1] < 0.0) & (lift[1:] >= 0.0))[0]
    if up.size < STROUHAL_PERIODS + 1:
        raise ValueError(
            f"the lift holds {max(up.size - 1, 0)} whole periods and the reading "
            f"needs the last {STROUHAL_PERIODS}; run longer"
        )
    k = up[-(STROUHAL_PERIODS + 1) :]
    crossings = t[k] - lift[k] * (t[k + 1] - t[k]) / (lift[k + 1] - lift[k])
    period = float((crossings[-1] - crossings[0]) / STROUHAL_PERIODS)
    st = height / (velocity * period)

    window = lift[k[0] :]
    span = float(steps[0] * window.size)
    spectrum = np.abs(np.fft.rfft(window - window.mean()))
    frequencies = np.fft.rfftfreq(window.size, d=float(steps[0]))
    peak = int(np.argmax(spectrum[1:])) + 1
    spectral = float(frequencies[peak]) * height / velocity
    resolution = height / (velocity * span)
    return StrouhalReading(
        strouhal=st,
        period=period,
        crossings=crossings,
        spectral_strouhal=spectral,
        resolution=resolution,
        agrees=abs(spectral - st) <= resolution,
    )
