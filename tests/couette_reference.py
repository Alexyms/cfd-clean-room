"""The one-dimensional Couette and channel reference of VAL-016 (ECR-002 step 6).

A steady solve of the k-epsilon equations with the scalable wall treatment
across a 0.3 m gap, in NumPy alone: nothing from src/, so it cannot inherit a
defect of the code it checks. The model constants are copied from their
published values (ADR-012 A and B); tests/test_turbulent_channel.py checks
that they equal src/turbulence.py's. The case's numbers (gap, air, wall
speed, inlet turbulence) live here too, so the test and the demonstration
scripts under docs/reports/probe45/ run one case.

The grid has the 2D grid's wall cells (height H / NY, so the same y_P, eps
held and the production given there, the wall viscosity on the momentum's
wall faces) and splits each interior 2D cell into REFINE cells. With REFINE 1
it is the 2D stencil's own x-invariant limit: face diffusivities the
distance-weighted harmonic mean, the strain at a centre the mean of the
gradients at its two faces, as in 2D. With REFINE odd every 2D cell centre is
a reference cell centre. k and eps step in a local pseudo-time, the
momentum is solved directly each step, and a channel's pressure gradient is
set so its bulk velocity is U_M.
"""

import math

import numpy as np

GAP = 0.3
RHO = 1.2
MU = 1.81e-5
NU = MU / RHO
# U_W for Re_tau about 3,000 on the half gap (ADR-012 G (ii)); off a round
# number so nothing aligns with the grid.
U_W = 15.7
# The channel's bulk velocity, the Couette inlet's.
U_M = U_W / 2.0
# The inlet's turbulence: 6% of the inflow speed and a dissipation length of a
# third of the gap, in ADR-012 C's convention, which put the inflow k and nu_t
# near the developed core's (k about 0.33 against 0.28 m^2/s^2).
INTENSITY = 0.06
DISSIPATION_LENGTH = 0.1
KAPPA, E_WALL = 0.41, 9.793
# Published constants (ADR-012 A), copied; tests/test_turbulent_channel.py
# checks them against src/turbulence.py.
VARIANTS = {
    "standard": {"c_mu": 0.09, "c_1": 1.44, "c_2": 1.92, "sigma_k": 1.0,
                 "sigma_e": 1.3, "eta_0": None, "beta": None},
    "rng": {"c_mu": 0.0845, "c_1": 1.42, "c_2": 1.68, "sigma_k": 0.7194,
            "sigma_e": 0.7194, "eta_0": 4.38, "beta": 0.012},
}  # fmt: skip


def log_law_floor() -> float:
    """y*_0 for KAPPA and E_WALL, by Newton's method from 2 / KAPPA."""
    y = 2.0 / KAPPA
    for _ in range(100):
        step = (KAPPA * y - math.log(E_WALL * y)) / (KAPPA - 1.0 / y)
        y -= step
        if abs(step) <= 4.0 * math.ulp(y):
            break
    return y


Y_FLOOR = log_law_floor()


# ---------------------------------------------------------------------------
# The one-dimensional reference
# ---------------------------------------------------------------------------


def _thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = b.size
    cp, dp = np.empty(n), np.empty(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _conductance(gamma: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Harmonic face value over the centre-to-centre distance, interior faces."""
    da, db = half[:-1], half[1:]
    return 1.0 / (da / gamma[:-1] + db / gamma[1:])


def reference(
    variant: str,
    ny: int,
    refine: int,
    kind: str = "couette",
    tol: float = 1e-10,
    full: bool = False,
) -> dict:
    """The steady 1D solve (module docstring); k and eps by a local pseudo-time step.

    The relative change of k and eps per step stops it at ``tol``; it falls to
    rounding, about 1e-12, so a tighter one runs to the cap. With ``full`` the
    record carries every reference cell's yc, u and k as well as the 2D
    grid's centres.
    """
    const = VARIANTS[variant]
    h = GAP / ny
    widths = np.concatenate(([h], np.full((ny - 2) * refine, h / refine), [h]))
    faces = np.concatenate(([0.0], np.cumsum(widths)))
    yc = 0.5 * (faces[:-1] + faces[1:])
    half = 0.5 * widths
    n = widths.size
    c_mu = const["c_mu"]
    top = U_W if kind == "couette" else 0.0
    k = np.full(n, 0.3)
    eps = np.full(n, 1.0)
    y_p = 0.5 * h
    # A channel is driven by a pressure gradient set so the bulk velocity is U_M.
    dpdx = 0.0
    u = np.full(n, U_M)
    for iteration in range(200000):  # noqa: B007 -- read after the loop as the count
        nu_t = c_mu * k**2 / eps
        g = _conductance(RHO * (NU + nu_t), half)
        u_k = c_mu**0.25 * np.sqrt(k[[0, -1]])
        y_star = u_k * y_p / NU
        log_term = np.log(E_WALL * np.maximum(y_star, Y_FLOOR))
        g_wall = RHO * u_k * KAPPA / log_term  # mu_w / y_p
        a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)
        a[1:], c[:-1] = -g, -g
        b[1:] += g
        b[:-1] += g
        b[0] += g_wall[0]
        b[-1] += g_wall[1]
        d = np.zeros(n)
        d[-1] += g_wall[1] * top
        if kind == "channel":
            # Solve for the unit-gradient response and scale to the bulk flux.
            unit = _thomas(a, b, c, widths.copy())
            dpdx = -U_M * GAP / float(unit @ widths)
            u = -dpdx * unit
        else:
            u = _thomas(a, b, c, d)
        gradient = np.diff(u) / (half[:-1] + half[1:])
        low = np.concatenate(([u[0] / y_p], gradient))
        high = np.concatenate((gradient, [(top - u[-1]) / y_p]))
        s = np.abs(0.5 * (low + high))
        production = nu_t * s**2
        slip = np.abs(np.array([u[0], top - u[-1]]))
        production[[0, -1]] = u_k**2 * slip / (y_p * log_term)
        eps_wall = u_k**3 / (KAPPA * y_p)
        ratio = eps / k
        r_term = np.zeros(n)
        if const["eta_0"] is not None:
            eta = s * k / eps
            r_term = (
                c_mu * eta**3 * (1.0 - eta / const["eta_0"])
                / (1.0 + const["beta"] * eta**3) * eps * ratio
            )  # fmt: skip
        step = 1.0 / ratio
        gk = _conductance(NU + nu_t / const["sigma_k"], half)
        a, c = np.zeros(n), np.zeros(n)
        b = widths / step + ratio * widths
        a[1:], c[:-1] = -gk, -gk
        b[1:] += gk
        b[:-1] += gk
        k_new = _thomas(a, b, c, widths / step * k + production * widths)
        ge = _conductance(NU + nu_t / const["sigma_e"], half)
        a, c = np.zeros(n), np.zeros(n)
        b = (
            widths / step
            + (const["c_2"] * ratio + np.maximum(r_term, 0.0) / eps) * widths
        )
        a[1:], c[:-1] = -ge, -ge
        b[1:] += ge
        b[:-1] += ge
        d = (
            widths / step * eps
            + (const["c_1"] * ratio * production - np.minimum(r_term, 0.0)) * widths
        )
        for idx, value in ((0, eps_wall[0]), (n - 1, eps_wall[1])):
            a[idx], c[idx], b[idx], d[idx] = 0.0, 0.0, 1.0, value
        d[1] -= a[1] * eps_wall[0]
        a[1] = 0.0
        d[-2] -= c[-2] * eps_wall[1]
        c[-2] = 0.0
        eps_new = _thomas(a, b, c, d)
        change = max(
            np.max(np.abs(k_new / k - 1.0)), np.max(np.abs(eps_new / eps - 1.0))
        )
        k, eps = k_new, eps_new
        if change < tol:
            break
    nu_t = c_mu * k**2 / eps
    centres = (
        [0] + [1 + (j - 1) * refine + refine // 2 for j in range(1, ny - 1)] + [n - 1]
    )
    m = int(np.searchsorted(faces, GAP / 2.0)) - 1
    face_nu = (half[m] + half[m + 1]) / (
        half[m] / (NU + nu_t[m]) + half[m + 1] / (NU + nu_t[m + 1])
    )
    stress = face_nu * (u[m + 1] - u[m]) / (yc[m + 1] - yc[m])
    return {
        "variant": variant, "ny": ny, "refine": refine, "kind": kind,
        "iterations": iteration + 1, "change": float(change),
        "yc": yc[centres].tolist(), "u": u[centres].tolist(), "k": k[centres].tolist(),
        "u_tau_mid": math.sqrt(abs(stress)), "dpdx": dpdx,
        "tau_wall": [float(g_wall[0] * u[0]), float(g_wall[1] * abs(top - u[-1]))],
    } | ({"profile": {"yc": yc.tolist(), "u": u.tolist(), "k": k.tolist()}} if full else {})  # fmt: skip


def case_raw(
    kind: str,
    variant: str,
    ny: int,
    length: float,
    dx: float,
    cfl: float,
    tol: float = 1e-6,
    max_pressure_iter: int = 5000,
) -> dict:
    """The 2D case as a configuration mapping, for SimConfig.from_dict.

    A uniform inlet on the left (U_W / 2 for Couette, U_M for the channel)
    with INTENSITY and DISSIPATION_LENGTH, a pressure outlet on the right,
    the lower wall at rest and, for Couette, the upper wall a velocity inlet
    with zero normal velocity moving at U_W. Ten momentum sweeps, the
    error_estimate rule at ``tol``, the pseudo-time Courant number ``cfl``,
    ``max_pressure_iter`` conjugate-gradient iterations at most per correction.
    """
    inlet_speed = U_W / 2.0 if kind == "couette" else U_M
    boundaries = {
        "inlet": {
            "type": "velocity_inlet", "location": "left", "y_start": 0.0, "y_end": GAP,
            "velocity": inlet_speed, "turbulence_intensity": INTENSITY,
            "dissipation_length": DISSIPATION_LENGTH,
        },
        "outlet": {"type": "pressure_outlet", "location": "right", "y_start": 0.0, "y_end": GAP},
    }  # fmt: skip
    if kind == "couette":
        boundaries["belt"] = {
            "type": "velocity_inlet", "location": "top", "x_start": 0.0, "x_end": length,
            "u_velocity": U_W, "v_velocity": 0.0,
        }  # fmt: skip
    return {
        "domain": {"width": length, "height": GAP, "nx": round(length / dx), "ny": ny},
        "fluid": {"density": RHO, "viscosity": MU, "temperature": 293.0},
        "particles": {
            "density": 1000.0, "sizes": [5.0e-6], "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {"diameters": [5.0e-6], "efficiencies": [0.99999]},
        },
        "solver": {
            "dt": 0.01, "t_end": 1.0, "output_interval": 10, "convergence_tol": 1e-6,
            "max_simple_iter": 20000, "alpha_velocity": 0.7, "alpha_pressure": 0.3,
            "max_pressure_iter": max_pressure_iter, "pressure_rtol": 1e-8,
            "stopping_rule": "error_estimate", "momentum_sweeps": 10,
            "iteration_error_tol": tol,
        },
        "turbulence": {
            "model": "k_epsilon", "variant": variant,
            "wall_treatment": "scalable_wall_functions", "cfl_number": cfl,
            "alpha_turbulence": 0.7, "max_iter": 500, "tol": 1e-10,
        },
        "boundaries": boundaries, "obstacles": [],
        "sensors": [{"name": "c", "x": length / 2.0, "y": GAP / 2.0}],
        "thresholds": {"5e-06": 100.0},
    }  # fmt: skip
