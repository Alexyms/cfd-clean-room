"""Digitise the case I (Re 8,000, w_jet 16 mm) curves of Khayrullina et al. (2017).

Usage:
    python digitize55.py inspect    the frame, gridlines and calibration of each panel
    python digitize55.py curves     every curve, the checks, the record

Khayrullina, van Hooff, Blocken and van Heijst (2017), "PIV measurements of
isothermal plane turbulent impinging jets at moderate Reynolds numbers",
Exp Fluids 58(4):31, DOI 10.1007/s00348-017-2315-0 (CC BY). The file is
results/builder55/sources/khayrullina_2017.pdf, SHA-256
c9d08bb01f1bc8ddff6dee2b5d437a3f0184dc77347b457bd29e8538ac05421d, fetched
2026-10-15 from https://link.springer.com/content/pdf/10.1007/s00348-017-2315-0.pdf
(the TU/e address the prompt names answered 403). The figures are raster
images embedded in the PDF, extracted at their native size with no
resampling (results/builder55/figures/, by the xref each PANELS entry names).

The panels and what is read from each (the frame is the solver's: y down
from the nozzle in the paper, V the streamwise speed, V0 the exit speed):

- figure 4(a), xref 216 (1772 x 862, JPEG): the centreline V_0,y / V0
  against y / h_jet. The three 16 mm series (Re 8,000 open circles, 10,000
  filled squares, 13,000 crosses, all cyan) are plotted on top of one
  another and the paper says the distributions are identical for one
  w_jet, so each cyan blob is one station of the 16 mm curve; its centroid
  is the value, its extent the uncertainty.
- figure 6, xref 234 (969 x 1163): I_uv = sqrt(u_rms^2 + v_rms^2) / V0
  along the centreline against y / h_jet, the same cyan blobs.
- figure 5(a), xref 232 (1772 x 617): V / V_0,y against x / delta_0.5 at
  the heights y' / w_jet the legend names (y' from the virtual origin at
  y0 = -0.8 w_jet), each curve a line with its own marker. The markers are
  found by matching the legend's own symbol (cut from the legend) over the
  panel; the curve is the markers, in x order.
- figure 7(a), xref 266 (1780 x 634): I_uv against x / delta_0.5 at the
  same heights, the same markers.
- figure 3(a) and (b), xref 144 (1772 x 1929): |V| / V0 and I_uv across
  the jet at y = 2 w_jet, the cyan open circles joined by a line (Re
  8,000, AR 18.75); the circles are matched from the panel's own legend
  symbol.
- figure 8(a), xref 298 (1500 x 1914): the half-width delta_0.5 (the white
  dashed lines) and 2 delta_0.5 (the white dots) over the V / V0 contours,
  against y / w_jet.

Calibration: each panel's axis box is the long grey or dark line on each
side; the gridlines inside it are the printed values (the lists in PANELS,
read off the figure), and a linear map per axis is fitted to the gridlines
and the box edges by least squares; the residual of that fit and the pixel
size in data units are printed and kept. The checks: the Gaussian the paper
fits to figure 5 (exp(-0.693 (x / delta_0.5)^2)) against the digitised
curves in the self-similar range; the paper's stated numbers against the
digitised ones (potential core 4.8 w_jet, the half-velocity point at 0.85
h_jet, the impingement region 0.12 h_jet, the centreline I_uv peak 0.21 at
y / h_jet 0.55, the core I_uv 0.05 at y = 2 w_jet, the spreading rate 0.10).

Output: results/builder55/khayrullina_2017_digitized.json and the overlay
images beside it. Needs opencv-python-headless (results/builder49/venv),
not a project dependency.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIGURES = ROOT / "results" / "builder55" / "figures"
OUT = ROOT / "results" / "builder55" / "khayrullina_2017_digitized.json"
PDF_SHA256 = "c9d08bb01f1bc8ddff6dee2b5d437a3f0184dc77347b457bd29e8538ac05421d"

# The case (table 1 and section 2 of the paper).
W_JET = 0.016
H_JET = 0.360
RE = 8000.0
NU = 1.0e-6  # Re 7,200 at 8 mm is V0 0.9 m/s (section 3): nu = 1.0e-6 m^2/s
V0 = RE * NU / W_JET
VIRTUAL_ORIGIN_W = -0.8  # y0 / w_jet for the 16 mm nozzle (section 4.1)
GAUSS_A = -0.693
# The paper's self-similar range for the 16 mm nozzle, 0.35 to 0.75 h_jet,
# in nozzle widths (its section 4.1: 7.5 <= y / w_jet <= 17).
SELF_SIMILAR_W = (7.5, 17.0)

# Panels: the file, the crop that holds the panel alone (rows, cols), the
# axis values of the gridlines and box edges in pixel order (left to right,
# top to bottom), and what the axes carry.
PANELS: dict[str, dict] = {
    "fig4a": {
        "file": "fig4_x216.jpeg",
        "crop": ((0, 862), (0, 700)),
        "grey": 200,
        # The cyan annotations (rows, cols of the crop): the "4.8 w_jet"
        # text and its arrow, the "0.12 h_jet" text and its arrow.
        "exclude": [
            ((8, 178), (588, 640)),
            ((85, 120), (360, 475)),
            ((640, 760), (330, 700)),
        ],
        "x_values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        "y_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "x_name": "v_over_v0",
        "y_name": "y_over_h",
    },
    "fig6": {
        "file": "fig6_x234.png",
        "crop": ((0, 1163), (0, 969)),
        "grey": 200,
        "legend": ((15, 410), (624, 941)),
        "x_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "y_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "x_name": "i_uv",
        "y_name": "y_over_h",
    },
    "fig5a": {
        "file": "fig5_x232.png",
        "crop": ((0, 560), (0, 900)),
        "grey": 200,
        "x_values": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "y_values": [1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
        "x_name": "x_over_half_width",
        "y_name": "v_over_v0y",
    },
    "fig7a": {
        "file": "fig7_x266.png",
        "crop": ((0, 575), (0, 900)),
        "grey": 200,
        "x_values": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "y_values": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        "x_name": "x_over_half_width",
        "y_name": "i_uv",
    },
    "fig3a": {
        "file": "fig3_x144.png",
        "crop": ((0, 500), (0, 880)),
        "grey": 200,
        "x_values": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "y_values": [1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
        "x_name": "x_over_w",
        "y_name": "v_over_v0",
    },
    "fig3b": {
        "file": "fig3_x144.png",
        "crop": ((0, 500), (900, 1772)),
        "grey": 200,
        "x_values": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "y_values": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        "x_name": "x_over_w",
        "y_name": "i_uv",
    },
    "fig8a": {
        "file": "fig8_x298.png",
        "crop": ((0, 960), (0, 700)),
        "grey": 110,
        "box_only": True,
        "x_values": [-6.0, 6.0],
        "y_values": [0.0, 22.5],
        "x_name": "x_over_w",
        "y_name": "y_over_w",
    },
}

# The heights of figures 5(a) and 7(a), as the legend names them, and the
# legend symbol of each: the legend strip (rows, cols) in the figure, and
# where each symbol sits in it (col). Symbols are cut from the strip and
# matched over the panel. 2.8 and 18.4 are plain lines without a marker
# and 35.1 is only in panel (b); 30.8 is only in panel (b) too.
LEGEND_HEIGHTS = {
    "fig5a": {
        "filled": (4.8, "green"),
        "not_digitised": [2.8, 5.9, 7.9, 12.1, 14.9, 18.4, 20.1],
    },
    "fig7a": {
        "filled": (4.8, "green"),
        "not_digitised": [2.8, 5.9, 7.9, 12.1, 14.9, 18.4, 20.1],
    },
}


def rgb(name: str) -> np.ndarray:
    """One figure as RGB at its native size."""
    bgr = cv2.imread(str(FIGURES / name), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"missing {FIGURES / name}")
    return bgr[:, :, ::-1]


def masks(img: np.ndarray, grey: int) -> dict[str, np.ndarray]:
    """Grey (axes and gridlines), cyan (the 16 mm series), green, black, white."""
    r, g, b = (img[:, :, i].astype(int) for i in range(3))
    neutral = (np.abs(r - g) < 28) & (np.abs(g - b) < 28)
    return {
        "grey": neutral & (r <= grey) & (r >= 40),
        "dark": neutral & (r <= 110),
        "cyan": (g >= 100) & (b >= 100) & (r <= 120) & (g - r >= 40),
        "green": (g >= 110) & (r <= 120) & (b <= 120) & (g - r >= 50) & (g - b >= 50),
        "white": (r >= 235) & (g >= 235) & (b >= 235),
    }


def groups(positions: np.ndarray) -> list[float]:
    """Centres of runs of adjacent indices."""
    out: list[list[int]] = []
    for p in positions:
        if out and p - out[-1][-1] <= 2:
            out[-1].append(int(p))
        else:
            out.append([int(p)])
    return [float(np.mean(g)) for g in out]


def frame(grey: np.ndarray, dark: np.ndarray) -> dict[str, int]:
    """The axis box: the outermost rows and columns that are grey or dark along most of their length."""
    lines = grey | dark
    rows, cols = lines.shape
    long_rows = np.where(lines.sum(axis=1) > 0.6 * cols)[0]
    long_cols = np.where(lines.sum(axis=0) > 0.6 * rows)[0]
    return {
        "top": int(long_rows.min()),
        "bottom": int(long_rows.max()),
        "left": int(long_cols.min()),
        "right": int(long_cols.max()),
    }


def gridlines(grey: np.ndarray, box: dict[str, int]) -> dict[str, list[float]]:
    """Columns and rows inside the box that are grey along most of the box: the gridlines."""
    inner = grey[box["top"] + 3 : box["bottom"] - 2, box["left"] + 3 : box["right"] - 2]
    h, w = inner.shape
    cols = np.where(inner.sum(axis=0) > 0.35 * h)[0] + box["left"] + 3
    rows = np.where(inner.sum(axis=1) > 0.35 * w)[0] + box["top"] + 3
    return {"cols": groups(cols), "rows": groups(rows)}


def assign(
    lines: list[float], lo: int, hi: int, values: list[float]
) -> tuple[list[float], list[float], list[float]]:
    """Match the detected gridlines to the printed values between the box edges.

    The edges carry the first and last value, which fixes a provisional
    map; each detected line is given the nearest value if it lies within
    a quarter of the spacing of it, and a line that matches no value, or a
    value matched twice, is left out. Returns the pixels, their values and
    the unassigned lines.
    """
    span = (values[-1] - values[0]) / (hi - lo)
    step = (
        abs(values[1] - values[0]) if len(values) > 1 else abs(values[-1] - values[0])
    )
    px, vals, lost = [float(lo)], [values[0]], []
    taken = {0, len(values) - 1}
    for line in lines:
        if abs(line - lo) < 5 or abs(line - hi) < 5:
            continue
        v = values[0] + span * (line - lo)
        k = int(np.argmin(np.abs(np.array(values) - v)))
        if abs(values[k] - v) > 0.25 * step or k in taken:
            lost.append(line)
            continue
        taken.add(k)
        px.append(float(line))
        vals.append(values[k])
    px.append(float(hi))
    vals.append(values[-1])
    order = np.argsort(px)
    return [px[i] for i in order], [vals[i] for i in order], lost


def linear(px: list[float], values: list[float]) -> tuple[float, float, float]:
    """Slope, intercept and largest residual of value against pixel."""
    a, b = np.polyfit(px, values, 1)
    res = float(np.abs(np.polyval((a, b), px) - np.array(values)).max())
    return float(a), float(b), res


def calibrate(key: str, img: np.ndarray) -> dict:
    """Frame, gridlines and the two linear maps of one panel."""
    spec = PANELS[key]
    (r0, r1), (c0, c1) = spec["crop"]
    sub = img[r0:r1, c0:c1]
    m = masks(sub, spec["grey"])
    if spec.get("legend"):
        (lr0, lr1), (lc0, lc1) = spec["legend"]
        for name in m:
            m[name][lr0:lr1, lc0:lc1] = False
    box = frame(m["grey"], m["dark"])
    if spec.get("box_only"):
        cols, rows = [], []
    else:
        found = gridlines(m["grey"], box)
        cols, rows = found["cols"], found["rows"]
    cols_px, x_vals, lost_cols = assign(
        cols, box["left"], box["right"], spec["x_values"]
    )
    rows_px, y_vals, lost_rows = assign(
        rows, box["top"], box["bottom"], spec["y_values"]
    )
    report = {
        "file": spec["file"],
        "crop": spec["crop"],
        "frame": box,
        "gridline_cols": cols,
        "gridline_rows": rows,
        "gridlines_unassigned": {"cols": lost_cols, "rows": lost_rows},
        "gridlines_used": {"cols": len(cols_px) - 2, "rows": len(rows_px) - 2},
    }
    spec = {**spec, "x_values": x_vals, "y_values": y_vals}
    ax, bx, rx = linear(cols_px, spec["x_values"])
    ay, by, ry = linear(rows_px, spec["y_values"])
    report["calibration"] = {
        "x_per_col": ax,
        "x_intercept": bx,
        "x_fit_residual": rx,
        "y_per_row": ay,
        "y_intercept": by,
        "y_fit_residual": ry,
        "pixel_in_x": abs(ax),
        "pixel_in_y": abs(ay),
    }
    return report


def to_data(
    cal: dict, cols: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel columns and rows to the panel's axis values."""
    return (
        cal["x_per_col"] * np.asarray(cols, float) + cal["x_intercept"],
        cal["y_per_row"] * np.asarray(rows, float) + cal["y_intercept"],
    )


def inside(box: dict[str, int], mask: np.ndarray, margin: int = 2) -> np.ndarray:
    """The mask with everything on or outside the axis box cleared."""
    out = np.zeros_like(mask)
    out[
        box["top"] + margin : box["bottom"] - margin + 1,
        box["left"] + margin : box["right"] - margin + 1,
    ] = mask[
        box["top"] + margin : box["bottom"] - margin + 1,
        box["left"] + margin : box["right"] - margin + 1,
    ]
    return out


def blobs(mask: np.ndarray, min_size: int = 4, max_size: int = 40) -> list[dict]:
    """Connected components of a mask, with centroid and extent."""
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    out = []
    for i in range(1, n):
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if min_size <= w <= max_size and min_size <= h <= max_size:
            out.append(
                {
                    "col": float(centroids[i][0]),
                    "row": float(centroids[i][1]),
                    "width": w,
                    "height": h,
                    "area": int(stats[i, cv2.CC_STAT_AREA]),
                }
            )
    return out


def ring_centres(
    mask: np.ndarray, found: list[dict]
) -> list[tuple[float, float] | None]:
    """The centre of the open circle inside each blob, by its hole, or None.

    An open circle is a ring: filling the blob and subtracting it leaves
    the ring's hole, whose centroid is the circle's centre. A filled square
    or a cross leaves no hole; a blob with no hole gets None.
    """
    out = []
    for b in found:
        c0, r0 = int(b["col"] - b["width"]), int(b["row"] - b["height"])
        c1, r1 = c0 + 2 * b["width"] + 1, r0 + 2 * b["height"] + 1
        c0, r0 = max(c0, 0), max(r0, 0)
        window = mask[r0:r1, c0:c1].astype(np.uint8)
        padded = np.pad(window, 1)
        flood = padded.copy()
        h, w = flood.shape
        cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
        holes = (flood == 0)[1:-1, 1:-1]
        if holes.sum() < 2:
            out.append(None)
            continue
        rows, cols = np.nonzero(holes)
        out.append((float(cols.mean()) + c0, float(rows.mean()) + r0))
    return out


def cyan_series(key: str, img: np.ndarray, cal_report: dict) -> dict:
    """The cyan blobs of a panel, in y order: the 16 mm series' stations.

    Each blob holds the three Re series' symbols where they overlap; the
    blob's centroid is their mean and its extent their spread, and where
    the open circle's hole is found its centre is Re 8,000's own value.
    """
    spec = PANELS[key]
    (r0, r1), (c0, c1) = spec["crop"]
    sub = img[r0:r1, c0:c1]
    m = masks(sub, spec["grey"])
    cyan = inside(cal_report["frame"], m["cyan"])
    if spec.get("legend"):
        (lr0, lr1), (lc0, lc1) = spec["legend"]
        cyan[lr0:lr1, lc0:lc1] = False
    for (er0, er1), (ec0, ec1) in spec.get("exclude", []):
        cyan[er0:er1, ec0:ec1] = False
    found = blobs(cyan)
    rings = ring_centres(cyan, found)
    cal = cal_report["calibration"]
    xs, ys = to_data(cal, [b["col"] for b in found], [b["row"] for b in found])
    order = np.argsort(ys)
    px_x, px_y = cal["pixel_in_x"], cal["pixel_in_y"]
    stations = []
    for i in order:
        ring = rings[i]
        entry = {
            spec["y_name"]: float(ys[i]),
            spec["x_name"]: float(xs[i]),
            "half_extent_x": 0.5 * found[i]["width"] * px_x,
            "half_extent_y": 0.5 * found[i]["height"] * px_y,
            "blob": found[i],
        }
        if ring is not None:
            rx, ry = to_data(cal, [ring[0]], [ring[1]])
            entry["ring"] = {spec["y_name"]: float(ry[0]), spec["x_name"]: float(rx[0])}
        stations.append(entry)
    return {
        "stations": stations,
        "count": len(stations),
        "rings": sum(r is not None for r in rings),
    }


def filled_circles(
    ink: np.ndarray, box: dict[str, int], erode: int = 2, min_area: int = 4
) -> list[dict]:
    """The filled markers of a colour mask inside the axis box, by erosion.

    Eroding the mask by two pixels removes every curve (one or two pixels
    wide) and the outlines of the open markers, and leaves the cores of
    the filled circles (about nine pixels across), whose centroids on the
    uneroded mask are the markers' centres.
    """
    sub = ink[box["top"] : box["bottom"] + 1, box["left"] : box["right"] + 1]
    kernel = np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)
    cores = cv2.erode(sub.astype(np.uint8), kernel)
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cores, connectivity=8
    )
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        core = labels == i
        grown = cv2.dilate(core.astype(np.uint8), kernel).astype(bool) & sub.astype(
            bool
        )
        rows, cols = np.nonzero(grown)
        out.append(
            {
                "col": float(cols.mean()) + box["left"],
                "row": float(rows.mean()) + box["top"],
                "core_area": int(stats[i, cv2.CC_STAT_AREA]),
                "width": int(cols.max() - cols.min() + 1),
                "height": int(rows.max() - rows.min() + 1),
            }
        )
    out.sort(key=lambda m: m["col"])
    return out


def marker_curves(key: str, img: np.ndarray, cal_report: dict) -> dict:
    """Figures 5(a) and 7(a): the one height whose markers can be read, the filled circles.

    The legend names eight heights. The curves at 2.8 and 18.4 w_jet are
    plain lines, and the open markers (triangles at 5.9 and 7.9, circles
    at 12.1, squares at 20.1) and the crosses at 14.9 are seven pixels
    across with the curves of the same colour drawn through them and the
    other curves crossing them, so neither a template nor the hole a
    marker encloses finds them reliably (prompt 55; the attempts are in
    the record's history). The filled circles at 4.8 w_jet are solid and
    survive an erosion that removes everything else; that height is
    digitised. For the other heights the paper's own Gaussian fit stands
    in, with the caveat stated beside every number that rests on it.
    """
    spec = PANELS[key]
    legend = LEGEND_HEIGHTS[key]
    (r0, r1), (c0, c1) = spec["crop"]
    m = masks(img[r0:r1, c0:c1], spec["grey"])
    cal = cal_report["calibration"]
    box = cal_report["frame"]
    height, colour = legend["filled"]
    found = filled_circles(m[colour], box)
    xs, ys = to_data(cal, [f["col"] for f in found], [f["row"] for f in found])
    return {
        str(height): {
            "y_prime_over_w": height,
            "y_over_w": height + VIRTUAL_ORIGIN_W,
            "y_over_h": (height + VIRTUAL_ORIGIN_W) * W_JET / H_JET,
            "symbol": f"{colour} filled circle",
            "markers": len(found),
            "x": xs.tolist(),
            "value": ys.tolist(),
            "pixels": [(f["col"] + c0, f["row"] + r0) for f in found],
            "marker_sizes": [(f["width"], f["height"]) for f in found],
        },
        "not_digitised": legend["not_digitised"],
    }


def half_width(img: np.ndarray, cal_report: dict) -> dict:
    """Figure 8(a): the white dashed delta_0.5 lines and the white dots at 2 delta_0.5.

    White pixels inside the box, the blank strip at the top and the plate
    row left out; the dots are the compact components, the dashes the
    rest. Each is mapped to (x / w, y / w), and the half-width at a height
    is half the distance between the left and right branches, so an
    offset of the jet axis from x = 0 cancels.
    """
    box = cal_report["frame"]
    (r0, r1), (c0, c1) = PANELS["fig8a"]["crop"]
    sub = img[r0:r1, c0:c1]
    m = masks(sub, PANELS["fig8a"]["grey"])
    white = inside(box, m["white"], margin=4)
    cal = cal_report["calibration"]
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        white.astype(np.uint8), connectivity=8
    )
    dots, dashes = [], []
    for i in range(1, n):
        w, h, area = int(stats[i, 2]), int(stats[i, 3]), int(stats[i, 4])
        x, y = to_data(cal, [centroids[i][0]], [centroids[i][1]])
        if w > 60 or h > 60 or y[0] < 1.0:
            continue
        entry = {
            "x_over_w": float(x[0]),
            "y_over_w": float(y[0]),
            "width": w,
            "height": h,
            "area": area,
        }
        (dots if max(w, h) <= 9 and area >= 8 else dashes).append(entry)

    def pair(points: list[dict]) -> list[dict]:
        left = sorted(
            [p for p in points if p["x_over_w"] < 0], key=lambda p: p["y_over_w"]
        )
        right = sorted(
            [p for p in points if p["x_over_w"] > 0], key=lambda p: p["y_over_w"]
        )
        out = []
        for p in left:
            q = min(
                right, key=lambda q: abs(q["y_over_w"] - p["y_over_w"]), default=None
            )
            if q is None or abs(q["y_over_w"] - p["y_over_w"]) > 0.6:
                continue
            out.append(
                {
                    "y_over_w": 0.5 * (p["y_over_w"] + q["y_over_w"]),
                    "half_distance": 0.5 * (q["x_over_w"] - p["x_over_w"]),
                    "axis_offset": 0.5 * (q["x_over_w"] + p["x_over_w"]),
                }
            )
        return out

    return {
        "dots_2delta": pair(dots),
        "dashes_delta": pair(dashes),
        "dot_count": len(dots),
        "dash_count": len(dashes),
        "dots": dots,
        "dashes": dashes,
        "pixel_in_w": cal["pixel_in_x"],
    }


def checks(cals: dict) -> dict:
    """The digitised curves against the numbers the paper states in its text.

    Potential core 4.8 w_jet (V_0,y / V0 >= 0.98), the half-velocity point
    at 0.85 h_jet, the centreline I_uv peak 0.21 at y / h_jet 0.55, the
    spreading rate 0.10 and delta_0.5 = 0.5 w_jet at the nozzle, the
    Gaussian fit against the digitised profile at 4.8 w_jet.
    """
    out: dict = {}
    centre = cals["fig4a"]["series"]["stations"]
    y = np.array([s["y_over_h"] for s in centre])
    v = np.array([s["v_over_v0"] for s in centre])
    below = np.nonzero(v < 0.98)[0]
    out["potential_core_w"] = {
        "digitised": float(y[below[0]] * H_JET / W_JET) if below.size else None,
        "stated": 4.8,
    }
    order = np.argsort(y)
    out["half_velocity_y_over_h"] = {
        "digitised": float(np.interp(0.5, v[order][::-1], y[order][::-1])),
        "stated": 0.85,
    }
    rings = [s for s in cals["fig6"]["series"]["stations"] if "ring" in s]
    i_uv = np.array([s["ring"]["i_uv"] for s in rings])
    y_i = np.array([s["ring"]["y_over_h"] for s in rings])
    peak = int(np.argmax(i_uv[y_i < 0.8]))
    out["centreline_i_uv_peak"] = {
        "digitised": [float(i_uv[peak]), float(y_i[peak])],
        "stated": [0.21, 0.55],
    }
    dots = cals["fig8a"]["half_width"]["dots_2delta"]
    yy = np.array([d["y_over_w"] for d in dots])
    delta = 0.5 * np.array([d["half_distance"] for d in dots])
    similar = (yy >= SELF_SIMILAR_W[0]) & (yy <= SELF_SIMILAR_W[1])
    slope, _intercept = np.polyfit(yy[similar], delta[similar], 1)
    out["spreading_rate"] = {
        "digitised": float(slope),
        "over_y_over_w": list(SELF_SIMILAR_W),
        "pairs_used": int(similar.sum()),
        "at_first_pair": [float(yy[0]), float(delta[0])],
        "stated": 0.10,
        "nozzle_half_width": 0.5,
    }
    curve = cals["fig5a"]["curves"]["4.8"]
    x = np.array(curve["x"])
    val = np.array(curve["value"])
    gauss = np.exp(GAUSS_A * x**2)
    out["gaussian_vs_4_8"] = {
        "max_abs": float(np.abs(val - gauss).max()),
        "rms": float(np.sqrt(np.mean((val - gauss) ** 2))),
        "note": "4.8 w_jet lies below the self-similar range the paper fits (5.9 to 20.1)",
    }
    return out


def overlay(
    name: str, img: np.ndarray, points: list[tuple[float, float]], colour=(255, 0, 255)
) -> None:
    """The figure with the digitised points marked, three times enlarged."""
    out = img.copy()
    for c, r in points:
        cv2.circle(out, (round(c), round(r)), 2, colour, -1)
    cv2.imwrite(
        str(OUT.parent / f"overlay_{name}.png"),
        cv2.resize(out[:, :, ::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
    )


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "curves"
    images = {spec["file"]: rgb(spec["file"]) for spec in PANELS.values()}
    cals = {key: calibrate(key, images[PANELS[key]["file"]]) for key in PANELS}
    if mode == "inspect":
        for key, rep in cals.items():
            print(key, json.dumps(rep))
        return
    record: dict = {
        "source": {
            "pdf": "results/builder55/sources/khayrullina_2017.pdf",
            "sha256": PDF_SHA256,
            "case": {"re": RE, "w_jet": W_JET, "h_jet": H_JET, "nu": NU, "v0": V0},
        },
        "panels": cals,
    }
    for key in ("fig4a", "fig6"):
        rep = cals[key]
        series = cyan_series(key, images[PANELS[key]["file"]], rep)
        rep["series"] = series
        (r0, _), (c0, _) = PANELS[key]["crop"]
        overlay(
            key,
            images[PANELS[key]["file"]],
            [
                (s["blob"]["col"] + c0, s["blob"]["row"] + r0)
                for s in series["stations"]
            ],
        )
        cal = rep["calibration"]
        print(
            key,
            series["count"],
            "stations,",
            series["rings"],
            "rings; pixel",
            cal["pixel_in_x"],
            cal["pixel_in_y"],
        )
        for st in series["stations"]:
            ring = st.get("ring")
            xn, yn = PANELS[key]["x_name"], PANELS[key]["y_name"]
            ring_text = "" if ring is None else f" ring {ring[yn]:.4f} {ring[xn]:.4f}"
            print(
                f"  {st[yn]:.4f} {st[xn]:.4f} +/- {st['half_extent_x']:.4f} "
                f"(blob {st['blob']['width']}x{st['blob']['height']}){ring_text}"
            )
    for key in ("fig5a", "fig7a"):
        rep = cals[key]
        curves = marker_curves(key, images[PANELS[key]["file"]], rep)
        rep["curves"] = curves
        pts = [
            p for h, c in curves.items() if h != "not_digitised" for p in c["pixels"]
        ]
        overlay(key, images[PANELS[key]["file"]], pts)
        for h, c in curves.items():
            if h == "not_digitised":
                continue
            span = (min(c["x"]), max(c["x"])) if c["x"] else None
            print(
                key,
                h,
                c["markers"],
                "markers, x range",
                span,
                "sizes",
                sorted(set(c["marker_sizes"])),
            )
    rep = cals["fig8a"]
    rep["half_width"] = half_width(images[PANELS["fig8a"]["file"]], rep)
    hw = rep["half_width"]
    print(
        "fig8a dots",
        hw["dot_count"],
        "dashes",
        hw["dash_count"],
        "paired dots",
        len(hw["dots_2delta"]),
        "paired dashes",
        len(hw["dashes_delta"]),
    )
    for p in hw["dots_2delta"]:
        print(
            f"  2delta: y/w {p['y_over_w']:.2f} half-distance {p['half_distance']:.3f} offset {p['axis_offset']:.3f}"
        )
    record["checks"] = checks(cals)
    for name, value in record["checks"].items():
        print("check", name, value)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1))
    print("written", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
