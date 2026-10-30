"""Prompt 70c: the measured plume widths of Warhaft (1984) and Stapountzis et al. (1986).

The two primary papers sit under results/papers/ (gitignored). This script holds
the digitised points and regenerates digitised70c.md from them; it does not need
the papers to run. With --detect it re-reads the figures from the PDFs (PyMuPDF
and SciPy needed) and prints the marker centres it finds, for comparison with
the DETECTIONS recorded below. With --check it regenerates the table and
compares it with digitised70c.md.

Method. Each figure page was rendered at 400 dpi and the plot cropped. The axis
frame and its tick marks were found as long dark runs; the calibration below maps
pixel to data through the labelled major ticks (linear axes: figures 4 and 5 of
Warhaft; logarithmic: figure 7 of Stapountzis, whose tick row is t / t_L(0) and
whose lower label row is x_w / M). Markers were located as the centroids of dark
blobs surviving an erosion of radius 6 px (filled circles), of enclosed white
regions of 40 to 900 px with a near-square bounding box (open circles), and of
regions where the dark-pixel density in a 13 px window exceeds 0.5 (crosses,
where two strokes meet). Every centre was checked on an overlay by eye; the
curves' own crossings and the legend's symbols were dropped. A centroid is good
to about 2 px, so a width is good to about 0.003 M on figure 4(b) (0.3% of the
80 M half-width, 1.03 M), 0.005 M on figure 5 (0.5% of it), 0.0005 M on
figure 4(a), and 1% on figure 7's logarithmic axes; the calibration adds about
1 px. The two figures' readings of the same 80 M point differ by 1.4%, so the
scatter between figures is larger than the centroid precision. Figure 7's
x_w / M labels are not at the drawn ticks; the tick row is t / t_L(0), and
x_w / M is placed from it through
U t_L(0) / M = 6.07 (t_L(0) = L_w / sigma_w at the source, equations 2.2 and
2.3), which puts x_w / M = 1 within 5 px of the printed label's centre.

Warhaft's mean profiles are Gaussian and he reports the half-width at half the
centreline value; sigma = 0.849 times the half-width (his section 3.2). His grid
mesh is M = 0.025 m (section 2), U = 7 m/s. Viswanathan and Pope tabulate
2.54 cm, which the first probe used. The model is rerun at his 0.025 m
(line70.py --mesh 0.025, records line_<grid>_m25.json), so the model, the data
and the reference curve share one mesh length and widths compare in metres;
T9a puts the first probe's widths in mesh lengths beside the rerun's, which is
what a rescale of the first probe would have given. Stapountzis et al.
fit Gaussians and report sigma_m directly, in metres; M = 2.54 cm, U = 4.35 m/s,
the wire at x_o = 19.3 M.

    python docs/reports/probe70/digitise70c.py            # regenerate the table
    python docs/reports/probe70/digitise70c.py --check    # regenerate and compare
    python docs/reports/probe70/digitise70c.py --detect   # re-read the figures
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import line70  # noqa: E402  (the probe's constants and reference curve)

PAPERS = ROOT / "results" / "papers"
RECORDS = ROOT / "results" / "builder70"
OUT = HERE / "digitised70c.md"
HW_TO_SIGMA = 0.849  # Warhaft, section 3.2: sigma of a Gaussian over its half-width
M_WARHAFT = line70.M_WARHAFT  # m, Warhaft section 2
U_WARHAFT = 7.0
# Warhaft table 1: the integral scale l = u^3 / eps at the source stations, m.
L_INTEGRAL = {20: 1.02e-2, 52: 1.35e-2, 60: 1.43e-2}
# Stapountzis et al.: section 2.1 and equations 2.2, 2.3, 2.5; section 4.1.
M_STAP, U_STAP, XO_STAP = 2.54e-2, 4.35, 19.3
SIGMA_W_COEF, M_EXP_STAP = 0.15, 1.43  # sigma_w^2 / U^2 = 0.15 (x / M)^-1.43
SIGMA_O_STAP = 0.92 * 1.52e-4  # m, 0.92 d_w
KAPPA_STAP = 2.2e-5  # m^2/s, heat in air at 30 C (the paper does not state its value)

# Crops: (pdf, page, x0, y0) at 400 dpi; the marker centres below are crop pixels.
CROPS = {
    "w_fig4a": ("the interface of thermal fields from line sources.pdf", 7, 640, 480),
    "w_fig4b": ("the interface of thermal fields from line sources.pdf", 7, 640, 1640),
    "w_fig5": ("the interface of thermal fields from line sources.pdf", 8, 740, 480),
    "s_fig7": ("structure of the temperature field downwind.pdf", 11, 640, 240),
}
# Calibrations from the detected tick marks (crop pixels).
CAL = {
    # x: majors at 1..8 M, 159.43 px apart, 1 M at 193.5; y: 0.1 at 697.0, 0.2 at 294.5
    "w_fig4a": dict(
        x=lambda px: (px - (193.5 - 159.43)) / 159.43,
        y=lambda py: 0.1 + (697.0 - py) * (0.1 / 402.5),
    ),
    # x: 20 M at 360.5, 80 M at 1317.5; y: 1.0 at 103.0, 0.5 at 501.5
    "w_fig4b": dict(
        x=lambda px: 20.0 + (px - 360.5) / ((1317.5 - 360.5) / 60.0),
        y=lambda py: 1.0 - (py - 103.0) / ((501.5 - 103.0) / 0.5),
    ),
    # x: 50 M at 479.5, 100 M at 880.5; y: 1 at 489.5, 402.5 px per unit
    "w_fig5": dict(
        x=lambda px: 50.0 + (px - 479.5) / ((880.5 - 479.5) / 50.0),
        y=lambda py: 1.0 - (py - 489.5) / 402.5,
    ),
    # x: the tick row is t / t_L(0) (1 at 897.5, a decade 373.0 px); x_w / M = (U t_L(0) / M)
    # t / t_L(0) with U t_L(0) / M = 6.07 from equations 2.2 and 2.3, so x_w / M = 1 at 605.5;
    # the lower label row's glyph centres (229, 601, 974.5, 1345.5 for 1e-1 to 1e2) agree within 5 px.
    # y: sigma 1e-2 m at 318.5, 1e-3 at 691.5
    "s_fig7": dict(
        x=lambda px: 10.0 ** ((px - 605.5) / 373.0),
        y=lambda py: 10.0 ** (-2.0 - (py - 318.5) / 373.0),
    ),
}
# (figure, source x_o / M, wire, marker, crop pixel centre); the mean half-width y' / M
# against x' / M for Warhaft, sigma_m (m) against x_w / M for Stapountzis.
DETECTIONS = [
    ("w_fig4a", 52, "0.025 mm", "filled circle", (69.0, 1050.1)),
    ("w_fig4a", 52, "0.025 mm", "filled circle", (121.2, 1030.0)),
    ("w_fig4a", 52, "0.025 mm", "filled circle", (175.7, 987.1)),
    ("w_fig4a", 52, "0.025 mm", "filled circle", (618.6, 761.5)),
    ("w_fig4a", 52, "0.025 mm", "filled circle", (941.8, 529.4)),
    ("w_fig4a", 52, "0.025 mm", "filled circle", (1308.2, 255.2)),
    ("w_fig4a", 52, "0.127 mm", "cross", (1304.1, 413.9)),
    ("w_fig4b", 52, "0.025 mm", "filled circle", (98.7, 829.6)),
    ("w_fig4b", 52, "0.025 mm", "filled circle", (124.4, 787.4)),
    ("w_fig4b", 52, "0.025 mm", "filled circle", (171.2, 735.5)),
    ("w_fig4b", 52, "0.127 mm", "cross", (174.0, 760.6)),
    ("w_fig4b", 52, "0.127 mm", "cross", (522.6, 452.9)),
    ("w_fig4b", 52, "0.127 mm", "cross", (1002.3, 226.7)),
    ("w_fig4b", 52, "0.127 mm", "cross", (1318.1, 79.0)),
    ("w_fig5", 52, "0.127 mm", "open circle", (143.4, 813.3)),
    ("w_fig5", 52, "0.127 mm", "open circle", (318.3, 668.3)),
    ("w_fig5", 52, "0.127 mm", "open circle", (559.6, 557.7)),
    ("w_fig5", 52, "0.127 mm", "open circle", (723.7, 483.0)),
    ("w_fig5", 20, "0.127 mm", "filled circle", (140.3, 795.3)),
    ("w_fig5", 20, "0.127 mm", "filled circle", (342.6, 597.6)),
    ("w_fig5", 20, "0.127 mm", "filled circle", (815.8, 352.6)),
    ("w_fig5", 20, "0.127 mm", "filled circle", (1141.8, 146.8)),
    ("w_fig5", 60, "0.127 mm", "cross", (262.1, 731.1)),
    ("w_fig5", 60, "0.127 mm", "cross", (340.3, 675.9)),
    ("w_fig5", 60, "0.127 mm", "cross", (661.7, 561.5)),
    ("s_fig7", 19.3, "0.152 mm", "open circle", (78.8, 940.5)),
    ("s_fig7", 19.3, "0.152 mm", "open circle", (255.8, 917.1)),
    ("s_fig7", 19.3, "0.152 mm", "open circle", (338.4, 851.6)),
    ("s_fig7", 19.3, "0.152 mm", "open circle", (414.7, 800.7)),
    ("s_fig7", 19.3, "0.152 mm", "open circle", (711.1, 548.4)),
    ("s_fig7", 19.3, "0.152 mm", "open circle", (823.2, 452.7)),
    ("s_fig7", 19.3, "0.152 mm", "open circle", (968.6, 374.1)),
    ("s_fig7", 19.3, "0.152 mm", "open circle", (1045.3, 276.3)),
]
STATIONS = (5.0, 10.0, 20.0, 50.0, 80.0, 100.0)
# The row stations the spill puts the gap sensors' nearest contour cells at (spill70c.py).
SPILL_STATIONS = (14.0, 16.0, 20.0, 30.0, 40.0, 50.0, 60.0, 71.0)
RERUN = "_m25"  # record suffix of line70.py --mesh 0.025
FIGURE_NAME = {
    "w_fig4a": "Warhaft figure 4(a)",
    "w_fig4b": "Warhaft figure 4(b)",
    "w_fig5": "Warhaft figure 5",
    "s_fig7": "Stapountzis et al. figure 7",
}


def warhaft_points() -> list[dict]:
    """Every Warhaft point in data units: x' / M, half-width / M, sigma (m)."""
    rows = []
    for fig, xo, wire, marker, (px, py) in DETECTIONS:
        if not fig.startswith("w_"):
            continue
        x_over_m = CAL[fig]["x"](px)
        hw = CAL[fig]["y"](py)
        rows.append(
            {
                "figure": fig,
                "xo": xo,
                "wire": wire,
                "marker": marker,
                "x_over_M": x_over_m,
                "hw_over_M": hw,
                "sigma_m": HW_TO_SIGMA * hw * M_WARHAFT,
            }
        )
    return rows


def stapountzis_points() -> list[dict]:
    rows = []
    for fig, xo, wire, marker, (px, py) in DETECTIONS:
        if fig != "s_fig7":
            continue
        rows.append(
            {
                "figure": fig,
                "xo": xo,
                "wire": wire,
                "marker": marker,
                "x_over_M": CAL[fig]["x"](px),
                "sigma_m": CAL[fig]["y"](py),
            }
        )
    return rows


def reference_warhaft(
    x_over_m: np.ndarray, xo: float = line70.X_O_OVER_M
) -> np.ndarray:
    """The probe's reference curve (line70.reference_sigma) at Warhaft's mesh length.

    Parameters
    ----------
    x_over_m : np.ndarray
        Distance downstream of the source in mesh lengths, x' / M.
    xo : float
        The source's station in mesh lengths; sets sigma_v and t_o there.

    Returns
    -------
    np.ndarray
        The plume width in metres, for sigma_v at one mesh length decaying as
        (x / M)^(-m / 2) and M = 0.025 m.
    """
    sigma_v = line70.SIGMA_V_AT_M * xo ** (-line70.M_EXP / 2)
    t_o = xo * M_WARHAFT / line70.U
    t = np.asarray(x_over_m) * M_WARHAFT / line70.U
    return line70.reference_sigma(t, sigma_v, t_o)


def reference_stapountzis(x_over_m: np.ndarray) -> np.ndarray:
    """Equations 11 to 15 of Viswanathan and Pope with Stapountzis's parameters."""
    sigma_w = U_STAP * np.sqrt(SIGMA_W_COEF * XO_STAP ** (-M_EXP_STAP))
    t_o = XO_STAP * M_STAP / U_STAP
    t = np.asarray(x_over_m) * M_STAP / U_STAP
    r = 0.5 * M_EXP_STAP * (1.5 * line70.C_0 - 1.0) + 1.0
    s = 0.5 * M_EXP_STAP * (1.5 * line70.C_0 + 1.0) - 1.0
    tau = t / t_o
    bracket = (
        (1.0 + tau) ** (r - s) / (r * (r - s))
        + (1.0 + tau) ** (-s) / (r * s)
        - 1.0 / (s * (r - s))
    )
    delta2 = 2.0 * sigma_w**2 * t_o**2 * bracket
    return np.sqrt(SIGMA_O_STAP**2 + 2.0 * KAPPA_STAP * t + delta2)


def model_sigma(grid: str, x_over_m: np.ndarray, suffix: str = RERUN) -> np.ndarray:
    """The probe's plume width at x' / M downstream of its source, interpolated.

    Parameters
    ----------
    grid : str
        The record's grid tag, "280x50" or "560x100".
    x_over_m : np.ndarray
        Stations in mesh lengths of the record's own mesh length.
    suffix : str
        "" for the first probe (M = 2.54 cm), RERUN for M = 2.5 cm.

    Returns
    -------
    np.ndarray
        The width in metres at the stations, linear between cell centres.
    """
    path = RECORDS / f"line_{grid}{suffix}.json"
    r = json.loads(path.read_text(encoding="utf-8"))
    xc = np.asarray(r["xc"])
    sigma = np.asarray(r["sigma"])
    x_prime = (xc - r["source"]["x_m"]) / r["experiment"]["M_m"]
    ok = x_prime > 0.5
    return np.interp(x_over_m, x_prime[ok], sigma[ok])


def loglog_interp(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """Interpolate y at x linearly in log x and log y between the points (xs, ys)."""
    return float(np.exp(np.interp(np.log(x), np.log(xs), np.log(ys))))


def composite_52(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Warhaft's 52 M source as he plots it: the fine wire to 8 M, the thick wire
    from 8 M on (both wires at 8 M averaged), sorted by x' / M."""
    fine = [
        (p["x_over_M"], p["sigma_m"])
        for p in points
        if p["xo"] == 52 and p["wire"] == "0.025 mm"
    ]
    thick = [
        (p["x_over_M"], p["sigma_m"])
        for p in points
        if p["xo"] == 52 and p["wire"] == "0.127 mm"
    ]
    fine = sorted(fine)
    thick = sorted(thick)
    near_8 = [s for x, s in fine if x > 7.5] + [s for x, s in thick if x < 9]
    pts = [(x, s) for x, s in fine if x <= 7.5]
    pts.append((8.1, float(np.mean(near_8))))
    pts += [(x, s) for x, s in thick if x >= 9]
    # duplicate stations (figures 4(b) and 5 repeat 30, 60 and 80 M): average them
    merged: dict[int, list] = {}
    for x, s in pts:
        merged.setdefault(round(x / 2.0), []).append((x, s))
    xs = np.array([np.mean([x for x, _ in v]) for v in merged.values()])
    ys = np.array([np.mean([s for _, s in v]) for v in merged.values()])
    order = np.argsort(xs)
    return xs[order], ys[order]


def nearest_datum(station: float, xs: np.ndarray) -> str:
    """The composite 52 M station nearest in distance, with its distance factor."""
    if station > xs.max():
        return "none, extrapolated"
    k = int(np.argmin(np.abs(np.log(station / xs))))
    factor = max(station / xs[k], xs[k] / station)
    return f"{xs[k]:.1f} (x{factor:.1f})"


def fmt(v: float, nd: int = 3) -> str:
    """Format a number with nd decimals."""
    return f"{v:.{nd}f}"


def spread_at_80(points: list[dict]) -> float:
    """Relative difference between the figures' readings of the 52 M source at 80 M."""
    hw = [
        p["hw_over_M"] for p in points if p["xo"] == 52 and 79.0 < p["x_over_M"] < 81.0
    ]
    return max(hw) / min(hw) - 1.0


def build_text() -> str:
    """Render digitised70c.md from the recorded marker centres and the probe records.

    Returns
    -------
    str
        The whole file: tables T8, T9, T9a, T12 and T10.

    Raises
    ------
    FileNotFoundError
        If a probe record (results/builder70/line_<grid>[_m25].json) is missing.
    """
    w = warhaft_points()
    s = stapountzis_points()
    lines = [
        "# Prompt 70c: the measured plume widths behind a line source in grid turbulence",
        "",
        "Generated by `digitise70c.py` from the marker centres recorded in it, which were read "
        "from Warhaft (1984, JFM 144) figures 4(a), 4(b) and 5 and Stapountzis, Sawford, Hunt "
        "and Britter (1986, JFM 165) figure 7 as the script's docstring describes. The papers "
        "are under `results/papers/` (gitignored). sigma = 0.849 x half-width (Warhaft, "
        "section 3.2). The reference curve is the probe's (`line70.reference_sigma`: Anand and "
        "Pope's displacement variance for the Langevin model with C_0 = 2.1 in the decaying "
        "turbulence, Viswanathan and Pope 2008 equations 11 to 15, with Warhaft's parameters "
        "as that paper tabulates them), at Warhaft's mesh length M = 0.025 m (prompt 70d).",
        "",
        "### T8. Warhaft (1984): the mean temperature profile's half-width behind a single "
        "heated wire at x_o / M = 20, 52 and 60, grid M = 0.025 m, U = 7 m/s, from the "
        "figures named; sigma = 0.849 x half-width; the reference curve at the same "
        "x' / M with the source station's sigma_v and t_o (the probe's constants, at "
        "M = 0.025 m); t / T_L with the 52 M station's T_L",
        "",
        "| Source x_o / M | Wire | Figure, marker | x' / M | Half-width / M | sigma (mm) | "
        "sigma / M | t / T_L | Reference sigma (mm) | Data / reference |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    turb = line70.source_turbulence(M_WARHAFT)
    t_l = 4.0 / (3.0 * line70.C_0) * turb["k_over_eps"]
    for p in sorted(w, key=lambda q: (q["xo"], q["x_over_M"])):
        t = p["x_over_M"] * M_WARHAFT / line70.U
        ref = float(reference_warhaft(np.array([p["x_over_M"]]), p["xo"])[0])
        ref_s, ratio_s = fmt(ref * 1e3, 2), fmt(p["sigma_m"] / ref, 2)
        ttl = fmt(t / t_l, 2) if p["xo"] == 52 else "-"
        lines.append(
            f"| {p['xo']} | {p['wire']} | {FIGURE_NAME[p['figure']]}, {p['marker']} | "
            f"{fmt(p['x_over_M'], 2)} | {fmt(p['hw_over_M'], 4)} | {fmt(p['sigma_m'] * 1e3, 2)} | "
            f"{fmt(HW_TO_SIGMA * p['hw_over_M'], 4)} | {ttl} | {ref_s} | {ratio_s} |"
        )
    xs, ys = composite_52(w)
    stations = np.array(STATIONS)
    m280 = model_sigma("280x50", stations)
    m560 = model_sigma("560x100", stations)
    ref_st = reference_warhaft(stations)
    lines += [
        "",
        "### T9. Prediction (d) rescored against the data: the model's width over Warhaft's "
        "at the stations. The model is the probe rerun at his mesh length (`line70.py --mesh "
        "0.025`, records `line_<grid>_m25.json`, 280x50 and 560x100), so the model, the data "
        "and the reference curve share M = 0.025 m and widths compare in metres. The data at "
        "a station are the 52 M source's points interpolated in log-log between their "
        "stations (the fine wire to 8 M, the thick wire from 8 M, both averaged at 8 M, "
        "repeated stations averaged); that interpolation is the row's reference, and the "
        "Anand and Pope curve is shown beside it, not used as the interpolant. The nearest "
        "datum column gives the 52 M station nearest in distance and the factor in distance "
        "to it. The 52 M data end at 80 M; the 100 M row gives two estimates, the 60 to 80 M "
        "slope extrapolated and the 20 M source read at the same x' / x_o through figure "
        "10's collapse (half-width over the source station's integral scale l, table 1: "
        "1.02 and 1.35 cm), and the model's ratio against each",
        "",
        "| Station x' / M | t / T_L | Nearest 52 M datum (M) | Data sigma (mm) | "
        "Reference sigma (mm) | Data / reference | Model sigma 280x50, 560x100 (mm) | "
        "Model / data | Model / reference |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    p20 = sorted([(p["x_over_M"], p["hw_over_M"]) for p in w if p["xo"] == 20])
    x20 = np.array([x for x, _ in p20])
    h20 = np.array([h for _, h in p20])
    for k, st in enumerate(STATIONS):
        t = st * M_WARHAFT / line70.U
        if st <= xs.max():
            data = [loglog_interp(st, xs, ys)]
            how = ""
        else:
            slope = np.log(ys[-1] / ys[-2]) / np.log(xs[-1] / xs[-2])
            extrap = float(ys[-1] * (st / xs[-1]) ** slope)
            hw20 = loglog_interp(st / 52.0 * 20.0, x20, h20)
            collapse = HW_TO_SIGMA * hw20 * L_INTEGRAL[52] / L_INTEGRAL[20] * M_WARHAFT
            data = [extrap, collapse]
            how = f" (extrapolated at slope {slope:.2f}; by the collapse)"
        data_s = ", ".join(fmt(d * 1e3, 1) for d in data) + how
        ratio_ref = ", ".join(fmt(d / ref_st[k], 2) for d in data)
        model_over = ", ".join(f"{m280[k] / d:.3f} and {m560[k] / d:.3f}" for d in data)
        lines.append(
            f"| {st:g} | {fmt(t / t_l, 2)} | {nearest_datum(st, xs)} | {data_s} | "
            f"{fmt(ref_st[k] * 1e3, 1)} | {ratio_ref} | "
            f"{fmt(m280[k] * 1e3, 2)}, {fmt(m560[k] * 1e3, 2)} | {model_over} | "
            f"{m280[k] / ref_st[k]:.3f}, {m560[k] / ref_st[k]:.3f} |"
        )
    lines += [
        "",
        "Digitising precision: a centroid is good to about 2 px, which is 0.3% (figure 4(b)) "
        "to 0.5% (figure 5) of the 80 M half-width; the two figures' readings of the 80 M "
        f"point differ by {100.0 * spread_at_80(w):.1f}%.",
        "",
        "### T9a. What a rescale of the first probe would have given: the model's width in "
        "mesh lengths, sigma / M, at the stations, for the first probe (M = 2.54 cm, "
        "records `line_<grid>.json`) and the rerun (M = 2.5 cm), and the ratio of the rerun "
        "to the first probe; the model is self-similar in x / M apart from the molecular "
        "terms and the cell size, which do not scale with M",
        "",
        "| Station x' / M | First probe 280x50, 560x100 | Rerun 280x50, 560x100 | "
        "Rerun / first probe |",
        "|---|---|---|---|",
    ]
    first = [model_sigma(g, stations, "") / line70.M for g in ("280x50", "560x100")]
    rerun = [model_sigma(g, stations) / M_WARHAFT for g in ("280x50", "560x100")]
    for k, st in enumerate(STATIONS):
        lines.append(
            f"| {st:g} | {first[0][k]:.4f}, {first[1][k]:.4f} | "
            f"{rerun[0][k]:.4f}, {rerun[1][k]:.4f} | "
            f"{rerun[0][k] / first[0][k]:.4f}, {rerun[1][k] / first[1][k]:.4f} |"
        )
    lines += [
        "",
        "### T10. Stapountzis et al. (1986) figure 7: sigma_m of the mean temperature profile "
        "behind the wire at x_o / M = 19.3, M = 2.54 cm, U = 4.35 m/s, sigma_w / U = 4.65% at "
        "the source (equation 2.2), against the same Langevin curve with that paper's "
        "parameters (m = 1.43, sigma_o = 0.92 d_w = 0.14 mm, kappa 2.2e-5 m^2/s, C_0 = 2.1); "
        "t_L(0) = L_w / sigma_w = 0.0355 s from equations 2.2 and 2.3",
        "",
        "| x_w / M | t / t_L(0) | sigma_m (mm) | sigma_m / M | Curve sigma (mm) | Data / curve |",
        "|---|---|---|---|---|---|",
    ]
    sigma_w = U_STAP * np.sqrt(SIGMA_W_COEF * XO_STAP ** (-M_EXP_STAP))
    l_w = 7.03e-2 * XO_STAP**0.47 * M_STAP
    t_l0 = l_w / sigma_w
    for p in sorted(s, key=lambda q: q["x_over_M"]):
        t = p["x_over_M"] * M_STAP / U_STAP
        ref = float(reference_stapountzis(np.array([p["x_over_M"]]))[0])
        lines.append(
            f"| {fmt(p['x_over_M'], 3)} | {fmt(t / t_l0, 3)} | {fmt(p['sigma_m'] * 1e3, 3)} | "
            f"{fmt(p['sigma_m'] / M_STAP, 4)} | {fmt(ref * 1e3, 3)} | {fmt(p['sigma_m'] / ref, 2)} |"
        )
    spill = np.array(SPILL_STATIONS)
    s280 = model_sigma("280x50", spill)
    s560 = model_sigma("560x100", spill)
    lines += [
        "",
        "### T12. The model's width over the measured across the stations the spill puts the "
        "gap sensors' nearest contour cells at (T11, 14 to 71 M on the row's axis): the "
        "52 M source's points interpolated in log-log as in T9, the rerun's widths on both "
        "grids",
        "",
        "| Station x' / M | t / T_L | Nearest 52 M datum (M) | Data sigma (mm) | "
        "Model sigma 280x50, 560x100 (mm) | Model / data 280x50, 560x100 |",
        "|---|---|---|---|---|---|",
    ]
    for k, st in enumerate(SPILL_STATIONS):
        t = st * M_WARHAFT / line70.U
        data = loglog_interp(st, xs, ys)
        lines.append(
            f"| {st:g} | {fmt(t / t_l, 2)} | {nearest_datum(st, xs)} | {fmt(data * 1e3, 1)} | "
            f"{fmt(s280[k] * 1e3, 2)}, {fmt(s560[k] * 1e3, 2)} | "
            f"{s280[k] / data:.3f}, {s560[k] / data:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Regenerate digitised70c.md, compare it with --check, or re-read the figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detect", action="store_true", help="re-read the figures")
    parser.add_argument("--check", action="store_true", help="regenerate and compare")
    args = parser.parse_args()
    if args.detect:
        return detect()
    try:
        text = build_text()
    except FileNotFoundError as err:
        print(f"probe record missing: {err.filename}")
        return 1
    if args.check:
        if OUT.read_text(encoding="utf-8") != text:
            print("digitised70c.md differs from the records")
            return 1
        print("digitised70c.md matches the records")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0


def detect() -> int:
    """Re-read the figures: frame, ticks and marker candidates per crop."""
    import fitz  # PyMuPDF
    from scipy import ndimage as ndi

    for tag, (pdf, page, x0, y0) in CROPS.items():
        path = PAPERS / pdf
        if not path.exists():
            print(f"{tag}: {path} missing")
            continue
        pix = fitz.open(path)[page - 1].get_pixmap(dpi=400)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        gray = arr[y0:, x0:, :3].mean(axis=2)
        dark = gray < 110
        er = ndi.binary_erosion(dark, structure=np.ones((13, 13), bool))
        lab, _ = ndi.label(er)
        centres = ndi.center_of_mass(er, lab, range(1, lab.max() + 1))
        print(f"{tag}: {len(centres)} filled blobs after erosion; first ten (x, y):")
        for cy, cx in centres[:10]:
            print(f"  ({cx:.1f}, {cy:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
