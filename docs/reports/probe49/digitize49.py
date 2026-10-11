"""Digitize the standard k-epsilon curves of Rong and Nielsen (2008), figures 5 and 9.

Usage:
    python digitize49.py inspect    the frame and tick positions of each panel
    python digitize49.py curves     the curves, the circle check and the stations

Rong and Nielsen (2008), DCE Technical Report 46, Aalborg University (ADR-012
source 10; the file at results/builder33b/sources/rong_nielsen_2008.pdf,
SHA-256 07d2ceba2fdefffce5fd02ba97a1da479c885cac3785de17fbddead3274722d4,
fetched for prompt 33b). Its figure 5 (a, b) holds the standard k-epsilon
prediction along the vertical lines x = 3.0 and 6.0 m, y in metres up from
the floor, and figure 9 (a, b) along the horizontal lines y = 0.084 and 2.916
m, x in metres from the inlet wall; the prediction is a red line and the
measurements green circles. The four panels are raster images embedded in the
PDF (265 x 404, 274 x 408, 705 x 294 and 454 x 202 pixels), extracted here at
their native size with no resampling.

Calibration. The axis box is the long dark horizontal and vertical lines; the
ticks are the short dark segments standing on the box's bottom and left
edges, assigned the values printed beside them (CALIBRATION, read off the
`inspect` output), and a linear map per axis is fitted to them. The check: the
green circles are found as the connected components of the green mask, mapped
through the same calibration and matched to the workbook's measurements
(results/builder33/sources/annex20_2d_measurements.xls, the data the
validation case stores), each circle to its nearest station; the residuals
are printed and kept.

The curve. The red mask, with the legend's red segment refused by continuity:
walking along the independent axis one pixel at a time, the run of red pixels
nearest to the last position is the curve. Where a green circle hides the
line the gap is left and the curve interpolated across it. The curve is
resampled at the measurement stations of the workbook by linear
interpolation, and the departure from the measurement is the published
prediction minus the measurement at each station.

Output: results/builder49/rong_nielsen_2008_digitized.json. Needs pymupdf and
opencv-python-headless (results/builder49/venv), neither a project dependency.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import xlrd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PDF = ROOT / "results" / "builder33b" / "sources" / "rong_nielsen_2008.pdf"
WORKBOOK = ROOT / "results" / "builder33" / "sources" / "annex20_2d_measurements.xls"
OUT = ROOT / "results" / "builder49" / "rong_nielsen_2008_digitized.json"
H = 3.0
U0 = 0.455

# The four panels: the image's xref in the PDF, the line, and which image axis
# carries the independent variable.
PANELS = {
    "x3": {"xref": 32, "figure": "5(a)", "line": "x = 3.0 m", "vertical": True},
    "x6": {"xref": 34, "figure": "5(b)", "line": "x = 6.0 m", "vertical": True},
    "y0084": {"xref": 53, "figure": "9(a)", "line": "y = 0.084 m", "vertical": False},
    "y2916": {"xref": 55, "figure": "9(b)", "line": "y = 2.916 m", "vertical": False},
}
# Tick values in pixel order (left to right along the bottom edge, top to
# bottom along the left edge), read off the figures and checked against the
# `inspect` counts. A tick that falls on the axis box's own edge (figure 5's
# -0.5, figure 9's 0 and 9 m) is not found and not listed; figure 9(a) has one
# tick on its left edge, so that axis is read from the box's edges, the
# printed -0.5 and 0.5, as well.
CALIBRATION = {
    "x3": {"bottom": [0.0, 0.5, 1.0], "left": [2.5, 2.0, 1.5, 1.0, 0.5]},
    "x6": {"bottom": [0.0, 0.5, 1.0], "left": [2.5, 2.0, 1.5, 1.0, 0.5]},
    "y0084": {"bottom": [1, 2, 3, 4, 5, 6, 7, 8], "left": [0.0]},
    "y2916": {"bottom": [1, 2, 3, 4, 5, 6, 7, 8], "left": [1.0, 0.5, 0.0]},
}
# The workbook's lines in the panel's coordinates: the vertical lines are
# stored as y / H down from the ceiling and u / u0; the horizontal ones as
# x / H and u / u0. The panel coordinate is metres, y up from the floor.
WORKBOOK_LINES = {
    "x3": ("vertical", 1, 2),
    "x6": ("vertical", 5, 6),
    "y2916": ("horizontal", 1, 2),
    "y0084": ("horizontal", 5, 6),
}


def workbook() -> dict[str, dict[str, list[float]]]:
    """The four measured lines, in the panels' coordinates."""
    sheet = xlrd.open_workbook(WORKBOOK).sheet_by_index(0)
    rows = {"vertical": range(11, 36), "horizontal": range(42, 70)}
    out = {}
    for key, (kind, ccol, ucol) in WORKBOOK_LINES.items():
        c = np.array([float(sheet.cell_value(r, ccol)) for r in rows[kind]])
        u = np.array([float(sheet.cell_value(r, ucol)) for r in rows[kind]])
        coord = H * (1.0 - c) if kind == "vertical" else H * c
        out[key] = {"coordinate": coord.tolist(), "u_over_u0": u.tolist()}
    return out


def image(xref: int) -> np.ndarray:
    """One embedded image as RGB, at its native size."""
    info = pymupdf.open(PDF).extract_image(xref)
    raw = np.frombuffer(info["image"], dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return bgr[:, :, ::-1]


def masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Red (the curve), green (the circles) and dark (the axes and text)."""
    r, g, b = (rgb[:, :, i].astype(int) for i in range(3))
    return {
        "red": (r >= 140) & (g <= 110) & (b <= 110),
        "green": (g >= 130) & (r <= 140) & (b <= 140) & (g - r >= 40),
        "dark": (r <= 110) & (g <= 110) & (b <= 110),
    }


def frame(dark: np.ndarray) -> dict[str, int]:
    """The axis box: the outermost long dark rows and columns."""
    rows, cols = dark.shape
    long_rows = np.where(dark.sum(axis=1) > 0.5 * cols)[0]
    long_cols = np.where(dark.sum(axis=0) > 0.5 * rows)[0]
    return {
        "top": int(long_rows.min()),
        "bottom": int(long_rows.max()),
        "left": int(long_cols.min()),
        "right": int(long_cols.max()),
    }


def ticks(dark: np.ndarray, box: dict[str, int]) -> dict[str, list[float]]:
    """Tick centres along the bottom edge (columns) and the left edge (rows).

    A tick is a short dark segment standing on the inside of the edge, two
    to six pixels long; the band just inside the edge is scanned for columns
    (rows) dark through the band, and adjacent ones are merged.
    """

    def merge(positions: np.ndarray) -> list[float]:
        groups: list[list[int]] = []
        for p in positions:
            if groups and p - groups[-1][-1] <= 1:
                groups[-1].append(int(p))
            else:
                groups.append([int(p)])
        return [float(np.mean(g)) for g in groups]

    inside = 3
    band = dark[box["bottom"] - inside - 1 : box["bottom"] - 1, :]
    bottom = np.where(band.all(axis=0))[0]
    bottom = bottom[(bottom > box["left"] + 2) & (bottom < box["right"] - 2)]
    band = dark[:, box["left"] + 1 : box["left"] + inside + 1]
    left = np.where(band.all(axis=1))[0]
    left = left[(left > box["top"] + 2) & (left < box["bottom"] - 2)]
    return {"bottom": merge(bottom), "left": merge(left)}


def linear(px: list[float], values: list[float]) -> tuple[float, float]:
    """Slope and intercept of value against pixel, by least squares."""
    a, b = np.polyfit(px, values, 1)
    return float(a), float(b)


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """The (start, end) index pairs of the True runs of a 1-D mask."""
    out = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


MAX_JUMP = 8.0  # px: a run further than this from the last position is not the curve
MATCH_PX = 3.0  # px: a circle within this of a station, on both axes, matches it
LOOSE_PX = 8.0  # px: the same check loosened, to see an offset in the figure's plotting


def trace(red: np.ndarray, box: dict[str, int], vertical: bool) -> np.ndarray:
    """The curve as (independent pixel, dependent pixel) pairs, legend refused.

    Walks the independent axis from the end where the curve is alone (the
    floor for a vertical line, the inlet wall for a horizontal one) and at
    each step keeps the red run nearest the last position, if it lies within
    MAX_JUMP pixels of it. Rows or columns with no red pixel, or none within
    reach (the curve hidden under a circle while the legend's segment shows),
    are skipped and interpolated across later.
    """
    inner = red[box["top"] + 1 : box["bottom"], box["left"] + 1 : box["right"]]
    if vertical:
        lines = [inner[i, :] for i in range(inner.shape[0] - 1, -1, -1)]
        index = list(range(inner.shape[0] - 1, -1, -1))
    else:
        lines = [inner[:, j] for j in range(inner.shape[1])]
        index = list(range(inner.shape[1]))
    found = []
    last = None
    for ind, line in zip(index, lines, strict=True):
        rr = runs(line)
        if not rr:
            continue
        centres = [0.5 * (a + b) for a, b in rr]
        if last is None:
            if len(rr) > 1:
                continue
            choice = centres[0]
        else:
            choice = min(centres, key=lambda c: abs(c - last))
            if abs(choice - last) > MAX_JUMP:
                continue
        last = choice
        found.append((ind, choice))
    out = np.array(found, dtype=float)
    if vertical:
        out[:, 0] += box["top"] + 1
        out[:, 1] += box["left"] + 1
    else:
        out[:, 0] += box["left"] + 1
        out[:, 1] += box["top"] + 1
    return out


def circles(green: np.ndarray, box: dict[str, int]) -> np.ndarray:
    """Centres (col, row) of the green circles inside the box, by components."""
    inner = np.zeros_like(green)
    inner[box["top"] + 1 : box["bottom"], box["left"] + 1 : box["right"]] = green[
        box["top"] + 1 : box["bottom"], box["left"] + 1 : box["right"]
    ]
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        inner.astype(np.uint8), connectivity=8
    )
    out = []
    for i in range(1, n):
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if 5 <= w <= 14 and 5 <= h <= 14:
            out.append(centroids[i])
    return np.array(out, dtype=float)


def panel(key: str, inspect_only: bool, measured: dict) -> dict:
    """Calibrate, trace and check one panel."""
    spec = PANELS[key]
    rgb = image(spec["xref"])
    m = masks(rgb)
    box = frame(m["dark"])
    tk = ticks(m["dark"], box)
    report: dict = {
        "figure": spec["figure"],
        "line": spec["line"],
        "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "frame": box,
        "ticks_px": tk,
    }
    if inspect_only:
        return report
    cal = CALIBRATION[key]
    for edge in ("bottom", "left"):
        if len(tk[edge]) != len(cal[edge]):
            raise SystemExit(
                f"{key}: {len(tk[edge])} ticks found on the {edge} edge, "
                f"{len(cal[edge])} values given: {tk[edge]}"
            )
    if len(cal["left"]) == 1:
        # One tick on the left edge (figure 9(a)): the box's top and bottom
        # edges are read as the printed extremes of the axis, +/-0.5.
        rows_px = [float(box["bottom"]), tk["left"][0], float(box["top"])]
        rows_val = [-0.5, cal["left"][0], 0.5]
    else:
        rows_px, rows_val = tk["left"], cal["left"]
    col_map = linear(tk["bottom"], cal["bottom"])
    row_map = linear(rows_px, rows_val)
    report["calibration"] = {
        "value_per_column": col_map[0],
        "column_intercept": col_map[1],
        "value_per_row": row_map[0],
        "row_intercept": row_map[1],
        "left_edge_fit_px": rows_px,
        "left_edge_fit_values": rows_val,
        "tick_fit_residual_bottom": float(
            np.abs(np.polyval(col_map, tk["bottom"]) - np.array(cal["bottom"])).max()
        ),
        "tick_fit_residual_left": float(
            np.abs(np.polyval(row_map, rows_px) - np.array(rows_val)).max()
        ),
    }

    def to_data(cols: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.polyval(col_map, cols), np.polyval(row_map, rows)

    traced = trace(m["red"], box, spec["vertical"])
    if spec["vertical"]:
        u, coord = to_data(traced[:, 1], traced[:, 0])
    else:
        coord, u = to_data(traced[:, 0], traced[:, 1])
    order = np.argsort(coord)
    coord, u = coord[order], u[order]
    report["curve"] = {
        "points": int(coord.size),
        "coordinate": coord.tolist(),
        "u_over_u0": u.tolist(),
        "pixel_in_coordinate": abs(row_map[0] if spec["vertical"] else col_map[0]),
        "pixel_in_u_over_u0": abs(col_map[0] if spec["vertical"] else row_map[0]),
    }

    # The circle check against the workbook.
    cc = circles(m["green"], box)
    if spec["vertical"]:
        c_u, c_coord = to_data(cc[:, 0], cc[:, 1])
    else:
        c_coord, c_u = to_data(cc[:, 0], cc[:, 1])
    meas = measured[key]
    m_coord, m_u = np.array(meas["coordinate"]), np.array(meas["u_over_u0"])
    px_coord = abs(row_map[0] if spec["vertical"] else col_map[0])
    px_u = abs(col_map[0] if spec["vertical"] else row_map[0])
    matched, loose, unmatched = [], [], []
    for ci, (c0, u0) in enumerate(zip(c_coord, c_u, strict=True)):
        j = int(np.argmin(np.hypot((m_coord - c0) / px_coord, (m_u - u0) / px_u)))
        entry = {
            "circle": ci,
            "station": j,
            "d_coordinate": float(c0 - m_coord[j]),
            "d_u_over_u0": float(u0 - m_u[j]),
        }
        # A circle that sits more than MATCH_PX from every station on either
        # axis is not a measurement (the legend's circle) or a merged pair.
        within = (
            abs(entry["d_coordinate"]) <= MATCH_PX * px_coord
            and abs(entry["d_u_over_u0"]) <= MATCH_PX * px_u
        )
        loosely = (
            abs(entry["d_coordinate"]) <= LOOSE_PX * px_coord
            and abs(entry["d_u_over_u0"]) <= LOOSE_PX * px_u
        )
        (matched if within else unmatched).append(entry)
        if loosely:
            loose.append(entry)
    d_c = np.array([x["d_coordinate"] for x in matched])
    d_u = np.array([x["d_u_over_u0"] for x in matched])
    l_c = np.array([x["d_coordinate"] for x in loose])
    l_u = np.array([x["d_u_over_u0"] for x in loose])
    report["circle_check"] = {
        "circles_found": int(cc.shape[0]),
        "stations": int(m_coord.size),
        "matched": len(matched),
        "distinct_stations_matched": len({x["station"] for x in matched}),
        "unmatched": len(unmatched),
        "match_px": MATCH_PX,
        "d_coordinate_mean": float(d_c.mean()),
        "d_coordinate_max_abs": float(np.abs(d_c).max()),
        "d_u_mean": float(d_u.mean()),
        "d_u_max_abs": float(np.abs(d_u).max()),
        "matches": matched,
        "unmatched_circles": unmatched,
        "loose_px": LOOSE_PX,
        "loose_matched": len(loose),
        "loose_d_coordinate_mean": float(l_c.mean()),
        "loose_d_coordinate_max_abs": float(np.abs(l_c).max()),
        "loose_d_u_mean": float(l_u.mean()),
        "loose_d_u_max_abs": float(np.abs(l_u).max()),
    }
    overlay = rgb.copy()
    for ind, dep in traced:
        rr, cc_ = (
            (round(ind), round(dep)) if spec["vertical"] else (round(dep), round(ind))
        )
        overlay[rr, cc_] = (0, 0, 255)
    for c in cc:
        cv2.circle(overlay, (round(c[0]), round(c[1])), 1, (255, 0, 255), -1)
    cv2.imwrite(
        str(OUT.parent / f"overlay_{key}.png"),
        cv2.resize(
            overlay[:, :, ::-1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST
        ),
    )

    # The prediction at the stations and its departure from the measurement.
    inside = (m_coord >= coord.min()) & (m_coord <= coord.max())
    at = np.interp(m_coord, coord, u)
    report["at_stations"] = {
        "coordinate": m_coord.tolist(),
        "measured_u_over_u0": m_u.tolist(),
        "published_u_over_u0": [
            float(a) if ok else None for a, ok in zip(at, inside, strict=True)
        ],
        "departure": [
            float(a - b) if ok else None
            for a, b, ok in zip(at, m_u, inside, strict=True)
        ],
        "stations_outside_curve": int((~inside).sum()),
    }
    dep = np.array([d for d in report["at_stations"]["departure"] if d is not None])
    report["departure_summary"] = {
        "max_abs": float(np.abs(dep).max()),
        "rms": float(np.sqrt(np.mean(dep**2))),
        "mean": float(dep.mean()),
    }
    return report


def main() -> None:
    """Dispatch the two modes of the module docstring."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "curves"
    inspect_only = mode == "inspect"
    measured = workbook()
    out = {
        "source": {
            "pdf": str(PDF.relative_to(ROOT)),
            "sha256": "07d2ceba2fdefffce5fd02ba97a1da479c885cac3785de17fbddead3274722d4",
            "workbook": str(WORKBOOK.relative_to(ROOT)),
            "workbook_sha256": "e6dc80485371e3d076cf0677202958df07d07e9207b787053ea68313bc745a8e",
        },
        "measured": measured,
        "panels": {key: panel(key, inspect_only, measured) for key in PANELS},
    }
    if inspect_only:
        for key, rep in out["panels"].items():
            print(key, json.dumps(rep))
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    for key, rep in out["panels"].items():
        cal, chk, dep = (
            rep["calibration"],
            rep["circle_check"],
            rep["departure_summary"],
        )
        print(
            f"{key}: frame {rep['frame']} ticks {rep['ticks_px']}\n"
            f"  pixel = {rep['curve']['pixel_in_coordinate']:.4f} m, "
            f"{rep['curve']['pixel_in_u_over_u0']:.4f} u/u0; tick residuals "
            f"{cal['tick_fit_residual_bottom']:.4f}, {cal['tick_fit_residual_left']:.4f}\n"
            f"  circles {chk['circles_found']} found, {chk['matched']} matched to "
            f"{chk['distinct_stations_matched']} of {chk['stations']} stations, "
            f"{chk['unmatched']} unmatched; d_coord mean "
            f"{chk['d_coordinate_mean']:+.4f} max {chk['d_coordinate_max_abs']:.4f} m; "
            f"d_u mean {chk['d_u_mean']:+.4f} max {chk['d_u_max_abs']:.4f}; within "
            f"{LOOSE_PX:.0f} px {chk['loose_matched']}: d_coord mean "
            f"{chk['loose_d_coordinate_mean']:+.4f} max {chk['loose_d_coordinate_max_abs']:.4f} m, "
            f"d_u mean {chk['loose_d_u_mean']:+.4f} max {chk['loose_d_u_max_abs']:.4f}\n"
            f"  curve {rep['curve']['points']} points over "
            f"[{rep['curve']['coordinate'][0]:.3f}, {rep['curve']['coordinate'][-1]:.3f}]; "
            f"stations outside {rep['at_stations']['stations_outside_curve']}; "
            f"departure max {dep['max_abs']:.4f} rms {dep['rms']:.4f} mean {dep['mean']:+.4f}"
        )
        st = rep["at_stations"]
        print("  station, measured, published, departure:")
        for c, a, b, d in zip(
            st["coordinate"],
            st["measured_u_over_u0"],
            st["published_u_over_u0"],
            st["departure"],
            strict=True,
        ):
            print(
                f"    {c:7.3f} {a:+.4f} {b if b is not None else '-':>8.4} {d if d is not None else '-':>8.4}"
            )
    print(f"written {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
