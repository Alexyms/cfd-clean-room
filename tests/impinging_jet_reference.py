"""A constructed impinging-jet field and a second computation of every number ``evaluate`` reports.

The lesson of test 52: every number the comparison returns is checked
against an independent computation, not just its keys. The field here is
built so that each reader's wrong form gives a visibly different number:
the centreline is linear in the paper's y with an offset above V0 at the
nozzle (so the potential core is a real crossing and a reader that takes
the nearest column instead of the axis fit reads another core), the
profile is a Gaussian whose half-width differs between the two sides (so
a reader that takes one side alone differs from the mean), the speed
collapses linearly to zero in the impingement region, and k varies in
both x and y with a maximum on the axis inside the impingement region
that is not in the first cell.

``expected`` computes every entry of ``validation.impinging_jet.evaluate``
again here from the same arrays, by loops and ``np.interp`` rather than
the module's vectorised bilinear read and least-squares fit, and
``assert_matches`` compares the two dictionaries number by number.
"""

import math

import numpy as np

from src.config import SimConfig
from src.mesh import Mesh
from src.turbulence import VARIANTS, Y_STAR_FLOOR
from validation import impinging_jet as ij

# The constructed centreline: V_c / V0 = CENTRE_TOP - CENTRE_SLOPE s, s the
# paper's y / h_jet, down to the impingement region where it falls linearly
# to zero at the plate.
CENTRE_TOP = 1.02
CENTRE_SLOPE = 0.6
# The constructed half-width in nozzle widths: WIDTH_AT_NOZZLE + WIDTH_SLOPE
# times y / w_jet, the right side wider by WIDTH_SKEW.
WIDTH_AT_NOZZLE = 0.55
WIDTH_SLOPE = 0.09
WIDTH_SKEW = 1.1
# k / V0^2: K_BASE (1 + K_Y s) (1 + K_X (x / w_jet)^2), plus a bump on the axis
# in the impingement region peaking at s = K_PEAK_S.
K_BASE = 0.01
K_Y = 0.5
K_X = 0.02
K_BUMP = 0.02
K_PEAK_S = 0.95
K_PEAK_WIDTH = 0.03


def centre_speed(s: np.ndarray) -> np.ndarray:
    """V_c / V0 at the paper's y / h_jet."""
    free = CENTRE_TOP - CENTRE_SLOPE * s
    collapse = np.clip((1.0 - s) / (1.0 - ij.IMPINGEMENT_H), 0.0, 1.0)
    return free * collapse


def half_width(s: np.ndarray, side: np.ndarray) -> np.ndarray:
    """delta_0.5 in metres at y / h_jet, per side (+1 right, -1 left)."""
    base = ij.W_JET * (WIDTH_AT_NOZZLE + WIDTH_SLOPE * s * ij.H_JET / ij.W_JET)
    return base * np.where(side > 0, WIDTH_SKEW, 1.0)


def fields(mesh: Mesh, config: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """v on the horizontal faces [ny+1, nx] and k on the cells [ny, nx]."""
    axis = ij.axis_x(config)
    xc, yf = np.meshgrid(np.asarray(mesh.xc), np.asarray(mesh.y))
    s = 1.0 - yf / ij.H_JET
    x = xc - axis
    delta = half_width(s, np.sign(x))
    v = -ij.V0 * centre_speed(s) * np.exp(ij.GAUSS_A * (x / delta) ** 2)
    xc2, yc = np.meshgrid(np.asarray(mesh.xc), np.asarray(mesh.yc))
    sc = 1.0 - yc / ij.H_JET
    bump = K_BUMP * np.exp(-(((sc - K_PEAK_S) / K_PEAK_WIDTH) ** 2))
    bump = bump * np.exp(-(((xc2 - axis) / ij.W_JET) ** 2))
    k = ij.V0**2 * (
        K_BASE * (1.0 + K_Y * sc) * (1.0 + K_X * ((xc2 - axis) / ij.W_JET) ** 2) + bump
    )
    return v, k


# ---------------------------------------------------------------------------
# The second computation
# ---------------------------------------------------------------------------


def row_at(ys: np.ndarray, field: np.ndarray, y: float) -> np.ndarray:
    """A field's row at height y, linear between the two rows around it, clamped."""
    ys = np.asarray(ys)
    y = min(max(float(y), float(ys[0])), float(ys[-1]))
    j = int(np.searchsorted(ys, y, side="right") - 1)
    j = min(max(j, 0), ys.size - 2)
    t = (y - ys[j]) / (ys[j + 1] - ys[j])
    return (1.0 - t) * field[j] + t * field[j + 1]


def axis_value(x_from_axis: np.ndarray, row: np.ndarray) -> float:
    """(9 V_d - V_3d) / 8 from the two column pairs nearest the axis, each pair averaged."""
    order = np.argsort(np.abs(x_from_axis))
    near, far = order[:2], order[2:4]
    d1 = np.abs(x_from_axis[near]).mean()
    d3 = np.abs(x_from_axis[far]).mean()
    assert math.isclose(d3, 3.0 * d1, rel_tol=1e-9), (
        "the x grid is uniform about the axis"
    )
    return float((9.0 * row[near].mean() - row[far].mean()) / 8.0)


def centreline(
    mesh: Mesh, config: SimConfig, v: np.ndarray, stations: np.ndarray
) -> list[float]:
    x = np.asarray(mesh.xc) - ij.axis_x(config)
    out = []
    for s in stations:
        row = -row_at(mesh.y, v, ij.H_JET * (1.0 - s))
        out.append(axis_value(x, row) / ij.V0)
    return out


def crossing(stations: np.ndarray, values: list[float], level: float) -> float | None:
    for i, (s, val) in enumerate(zip(stations, values, strict=True)):
        if val < level:
            if i == 0:
                return float(s)
            s0, v0 = stations[i - 1], values[i - 1]
            return float(s0 + (v0 - level) / (v0 - val) * (s - s0))
    return None


def slope(x: list[float], y: list[float]) -> float:
    """The least-squares slope by the textbook formula."""
    xm, ym = sum(x) / len(x), sum(y) / len(y)
    num = sum((a - xm) * (b - ym) for a, b in zip(x, y, strict=True))
    den = sum((a - xm) ** 2 for a in x)
    return num / den


def decay_rate(stations: np.ndarray, values: list[float]) -> float | None:
    xs, ys = [], []
    for s, val in zip(stations, values, strict=True):
        if ij.SELF_SIMILAR_H[0] <= s <= ij.SELF_SIMILAR_H[1]:
            xs.append(s * ij.H_JET / ij.W_JET - ij.VIRTUAL_ORIGIN_W)
            ys.append(1.0 / val**2)
    return slope(xs, ys) if len(xs) >= 2 else None


def half_width_w(
    mesh: Mesh, config: SimConfig, v: np.ndarray, s: float
) -> float | None:
    x = np.asarray(mesh.xc) - ij.axis_x(config)
    row = -row_at(mesh.y, v, ij.H_JET * (1.0 - s))
    centre = axis_value(x, row)
    if centre <= 0.0:
        return None
    half = 0.5 * centre
    found = []
    for sign in (1.0, -1.0):
        idx = [i for i in np.argsort(np.abs(x)) if np.sign(x[i]) == sign]
        hit = None
        for n, i in enumerate(idx):
            if row[i] < half:
                if n == 0:
                    hit = abs(x[i])
                else:
                    p = idx[n - 1]
                    hit = abs(x[p]) + (row[p] - half) / (row[p] - row[i]) * (
                        abs(x[i]) - abs(x[p])
                    )
                break
        if hit is None:
            return None
        found.append(hit)
    return 0.5 * sum(found) / ij.W_JET


def profile_at(
    mesh: Mesh, config: SimConfig, v: np.ndarray, s: float, x_abs: np.ndarray
) -> np.ndarray:
    """-v at absolute x positions on the row at s, linear in x between the centres."""
    row = -row_at(mesh.y, v, ij.H_JET * (1.0 - s))
    return np.interp(x_abs, np.asarray(mesh.xc), row)


def k_axis(
    mesh: Mesh, config: SimConfig, k: np.ndarray, stations: np.ndarray
) -> list[float]:
    x = np.asarray(mesh.xc) - ij.axis_x(config)
    return [
        axis_value(x, row_at(mesh.yc, k, ij.H_JET * (1.0 - s))) / ij.V0**2
        for s in stations
    ]


def errors(values: list[float], reference: list[float]) -> dict:
    diffs = [a - b for a, b in zip(values, reference, strict=True)]
    return {
        "max_abs": max(abs(d) for d in diffs),
        "rms": math.sqrt(sum(d * d for d in diffs) / len(diffs)),
        "mean": sum(diffs) / len(diffs),
        "stations": len(diffs),
    }


def expected(mesh: Mesh, config: SimConfig, v: np.ndarray, k: np.ndarray) -> dict:
    """Every entry of ``evaluate`` computed again from the same arrays."""
    axis = ij.axis_x(config)
    y_c = ij.CENTRELINE_Y
    computed = centreline(mesh, config, v, y_c)
    data = ij.CENTRELINE_V.tolist()

    def ranged(keep: list[bool]) -> dict:
        return errors(
            [c for c, ok in zip(computed, keep, strict=True) if ok],
            [d for d, ok in zip(data, keep, strict=True) if ok],
        )

    out: dict = {
        "centreline": {
            "y_over_h": y_c.tolist(),
            "computed": computed,
            "data": data,
            "all": ranged([True] * len(computed)),
            "free_jet": ranged([s <= ij.FREE_JET_H for s in y_c]),
            "free_jet_narrow": ranged([s <= ij.NARROW_FREE_JET_H for s in y_c]),
            "impingement": ranged([s > ij.IMPINGEMENT_H for s in y_c]),
        },
        "potential_core_w": {
            "computed": None
            if crossing(y_c, computed, ij.CORE_FRACTION) is None
            else crossing(y_c, computed, ij.CORE_FRACTION) * ij.H_JET / ij.W_JET,
            "data": crossing(y_c, data, ij.CORE_FRACTION) * ij.H_JET / ij.W_JET,
            "stated": 4.8,
        },
        "half_velocity_h": {
            "computed": crossing(y_c, computed, 0.5),
            "data": crossing(y_c, data, 0.5),
            "stated": 0.85,
        },
        "decay_rate": {
            "computed": decay_rate(y_c, computed),
            "data": decay_rate(y_c, data),
            "stated": [0.14, 0.15],
        },
    }
    heights = ij.HALF_WIDTH_Y
    deltas = [
        half_width_w(mesh, config, v, float(h * ij.W_JET / ij.H_JET)) for h in heights
    ]
    held = [
        (h, d, r)
        for h, d, r in zip(heights, deltas, ij.HALF_WIDTH_DELTA, strict=True)
        if d is not None
    ]
    out["half_width"] = {
        "y_over_w": heights.tolist(),
        "computed": deltas,
        "data": ij.HALF_WIDTH_DELTA.tolist(),
        "errors": errors([d for _, d, _ in held], [r for _, _, r in held])
        if held
        else None,
        "spreading_rate": {
            "computed": slope(
                [
                    h
                    for h, _, _ in held
                    if ij.SELF_SIMILAR_W[0] <= h <= ij.SELF_SIMILAR_W[1]
                ],
                [
                    d
                    for h, d, _ in held
                    if ij.SELF_SIMILAR_W[0] <= h <= ij.SELF_SIMILAR_W[1]
                ],
            ),
            "data": slope(
                [
                    h
                    for h in heights
                    if ij.SELF_SIMILAR_W[0] <= h <= ij.SELF_SIMILAR_W[1]
                ],
                [
                    d
                    for h, d in zip(heights, ij.HALF_WIDTH_DELTA, strict=True)
                    if ij.SELF_SIMILAR_W[0] <= h <= ij.SELF_SIMILAR_W[1]
                ],
            ),
            "stated": 0.10,
        },
    }
    lateral = {}
    sets = [
        (
            ij.DIGITISED_HEIGHT_W,
            [r[0] for r in ij.LATERAL_V_48_ROWS],
            [r[1] for r in ij.LATERAL_V_48_ROWS],
            "digitised",
        )
    ]
    for h in ij.GAUSSIAN_HEIGHTS_W:
        xi = list(ij.GAUSSIAN_STATIONS)
        sets.append((h, xi, [math.exp(ij.GAUSS_A * x * x) for x in xi], "gaussian"))
    for y_prime, xi, reference, source in sets:
        s = (y_prime + ij.VIRTUAL_ORIGIN_W) * ij.W_JET / ij.H_JET
        delta = half_width_w(mesh, config, v, s)
        entry: dict = {
            "y_prime_over_w": y_prime,
            "y_over_h": s,
            "source": source,
            "x_over_delta": list(xi),
            "reference": list(reference),
        }
        if delta is None:
            entry.update({"errors": None, "delta_over_w": None, "computed": None})
        else:
            x_abs = axis + np.array(xi) * delta * ij.W_JET
            centre = axis_value(
                np.asarray(mesh.xc) - axis, -row_at(mesh.y, v, ij.H_JET * (1.0 - s))
            )
            prof = (profile_at(mesh, config, v, s, x_abs) / centre).tolist()
            entry.update(
                {
                    "errors": errors(prof, reference),
                    "delta_over_w": delta,
                    "computed": prof,
                }
            )
        lateral[str(y_prime)] = entry
    out["lateral"] = lateral

    s48 = (ij.DIGITISED_HEIGHT_W + ij.VIRTUAL_ORIGIN_W) * ij.W_JET / ij.H_JET
    d48 = half_width_w(mesh, config, v, s48)
    xi_i = [r[0] for r in ij.LATERAL_I_48_ROWS]
    i_48 = [r[1] for r in ij.LATERAL_I_48_ROWS]
    if d48 is None:
        out["lateral_intensity_48"] = {"y_over_h": s48, "errors": None}
    else:
        x_abs = axis + np.array(xi_i) * d48 * ij.W_JET
        k_row = row_at(mesh.yc, k, ij.H_JET * (1.0 - s48))
        k_at = np.interp(x_abs, np.asarray(mesh.xc), k_row) / ij.V0**2
        model_i = [math.sqrt(val / ij.K_ISOTROPIC) for val in k_at]
        out["lateral_intensity_48"] = {
            "y_over_h": s48,
            "x_over_delta": xi_i,
            "computed": model_i,
            "data": i_48,
            "errors": errors(model_i, i_48),
        }

    y_i = ij.INTENSITY_Y
    k_c = k_axis(mesh, config, k, y_i)
    iso = [ij.K_ISOTROPIC * i * i for i in ij.INTENSITY_I]
    plane = [ij.K_IN_PLANE * i * i for i in ij.INTENSITY_I]
    rings = [bool(r[2]) for r in ij.INTENSITY_ROWS]

    def only(values: list[float]) -> list[float]:
        return [v for v, ring in zip(values, rings, strict=True) if ring]

    out["k_axis"] = {
        "y_over_h": y_i.tolist(),
        "ring": rings,
        "computed": k_c,
        "data_isotropic": iso,
        "data_in_plane": plane,
        "errors_isotropic": errors(only(k_c), only(iso)),
        "errors_in_plane": errors(only(k_c), only(plane)),
        "errors_isotropic_all": errors(k_c, iso),
        "errors_in_plane_all": errors(k_c, plane),
        "ratio_isotropic_median": float(
            np.median([a / b for a, b in zip(only(k_c), only(iso), strict=True)])
        ),
        "ratio_isotropic_median_all": float(
            np.median([a / b for a, b in zip(k_c, iso, strict=True)])
        ),
    }
    x = np.asarray(mesh.xc) - axis
    on_axis = [axis_value(x, k[j]) / ij.V0**2 for j in range(k.shape[0])]
    rows = [j for j, yc in enumerate(mesh.yc) if 1.0 - yc / ij.H_JET > ij.IMPINGEMENT_H]
    peak = max(rows, key=lambda j: on_axis[j])
    last = max(range(len(y_i)), key=lambda i: y_i[i])
    last_ring = max((i for i in range(len(y_i)) if rings[i]), key=lambda i: y_i[i])
    out["stagnation"] = {
        "first_cell": on_axis[0],
        "impingement_max": on_axis[peak],
        "impingement_max_y_over_h": 1.0 - float(mesh.yc[peak]) / ij.H_JET,
        "data_station_y_over_h": float(y_i[last]),
        "data_station_ring": rings[last],
        "computed_at_data_station": k_c[last],
        "data_isotropic": iso[last],
        "data_in_plane": plane[last],
        "ratio_isotropic": k_c[last] / iso[last],
        "ratio_in_plane": k_c[last] / plane[last],
        "ring_station_y_over_h": float(y_i[last_ring]),
        "computed_at_ring_station": k_c[last_ring],
        "ring_isotropic": iso[last_ring],
        "ring_in_plane": plane[last_ring],
        "ring_ratio_isotropic": k_c[last_ring] / iso[last_ring],
        "ring_ratio_in_plane": k_c[last_ring] / plane[last_ring],
    }
    c_mu = VARIANTS[config.turbulence.variant].c_mu
    nu = config.mu / config.rho
    y_p = float(mesh.yc[0] - mesh.y[0])
    row = [c_mu**0.25 * math.sqrt(max(val, 0.0)) * y_p / nu for val in k[0]]
    near = [
        val
        for val, xi in zip(row, x, strict=True)
        if abs(xi) <= ij.PLATE_MEDIAN_HALF_WIDTH_W * ij.W_JET
    ]
    out["y_star"] = {
        "median": float(np.median(near)),
        "at_axis": axis_value(x, np.array(row)),
        "largest": max(row),
        "floor": Y_STAR_FLOOR,
    }
    return out


def assert_matches(got: object, want: object, path: str = "report") -> None:
    """Compare two reports number by number, 1e-9 relative, 1e-12 absolute."""
    if isinstance(want, dict):
        assert isinstance(got, dict), path
        assert set(got) == set(want), f"{path}: keys {sorted(got)} != {sorted(want)}"
        for key in want:
            assert_matches(got[key], want[key], f"{path}.{key}")
    elif isinstance(want, list | tuple):
        assert isinstance(got, list | tuple), path
        assert len(got) == len(want), f"{path}: {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            assert_matches(g, w, f"{path}[{i}]")
    elif want is None or isinstance(want, str | bool):
        assert got == want, f"{path}: {got!r} != {want!r}"
    else:
        assert got is not None, f"{path}: None for {want}"
        assert math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-12), (
            f"{path}: {got} != {want}"
        )
