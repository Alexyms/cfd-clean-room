"""ECR-002 step 5 (prompt 44): the comparisons and the report's tables from conv44.py's records.

Usage:
    python tables44.py matrix          measurement 1: every product run, classified
    python tables44.py base            the base fields and the Z scales
    python tables44.py pairs           measurement 2: one against ten sweeps where both converged
    python tables44.py grids           measurement 3: 40x15 and 80x30 against 200x75
    python tables44.py corner          measurement 4: the corner-free runs against the committed rule
    python tables44.py rtol            measurement 5: pressure_rtol 1e-4 and 1e-2 against 1e-8
    python tables44.py cavity          measurement 6: the cavity's stops and centreline extremes
    python tables44.py all             everything above

Reads results/builder44/*.json and *.npz, prints markdown tables, and keeps
every number it printed in results/builder44/tables_MODE.json. Each run is
classified by the report's section 2.1 rules, in order: diverged (the run
stopped past 100 m/s or non-finite), converged (the solver's own stop),
growing (at the cap with the largest speed above 5 m/s), bounded and not
converged (at the cap under 5 m/s), with step 0's sub-classes falling,
stalled or neither over the last 500 iterations.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "builder44"
GRIDS = ("40x15", "80x30", "200x75")
FIELDS = ("U1", "U2", "U3", "Z2", "Z3", "Z4", "L")
NAME = re.compile(
    r"^(?P<field>U1|U2|U3|Z2|Z3|Z4|L)_(?P<grid>\d+x\d+)_s(?P<sweeps>\d+)(?P<rest>.*)$"
)
GROWING_SPEED = 5.0
WINDOW = 500
TOLERANCE_MS = 1e-6 * 0.45
FAR = 0.2
LINE_STEP = 0.02
EDGE_MARGIN = 0.1
VERTICAL_X = 2.7
HORIZONTAL_Y = 1.2
RECALLED = {
    "u_min": (-0.38857, 0.1717),
    "v_max": (0.37694, 0.1578),
    "v_min": (-0.52708, 0.9092),
}


def load(name: str) -> dict:
    """A record by name."""
    return json.loads((OUT / f"{name}.json").read_text())


def fields(name: str) -> dict:
    """A record's kept arrays."""
    return dict(np.load(OUT / f"{name}.npz"))


def records() -> dict[str, dict]:
    """Every product record, keyed by name, with its parsed name."""
    out = {}
    for path in sorted(OUT.glob("*.json")):
        m = NAME.match(path.stem)
        if not m or path.stem.endswith("_time"):
            continue
        rec = load(path.stem)
        rest = m["rest"]
        rec["_field"], rec["_grid"], rec["_sweeps"] = (
            m["field"],
            m["grid"],
            int(m["sweeps"]),
        )
        rtol = re.search(r"_r([0-9e.-]+)", rest)
        rec["_rtol"] = rtol.group(1) if rtol else "1e-8"
        rec["_rtol_set"] = rtol is not None
        rec["_cf"] = "_cf" in rest
        out[path.stem] = rec
    return out


def classify(rec: dict) -> tuple[str, str]:
    """The class and, for a bounded run, step 0's sub-class."""
    if rec["stop"] == "diverged":
        return "diverged", ""
    if rec["stop"] == "error_estimate_and_continuity":
        return "converged", ""
    if rec["max_speed_end"] > GROWING_SPEED:
        return "growing", ""
    res = np.array(rec["residual"])
    w = res[-WINDOW:]
    if w[-1] <= 1.1 * w.min() and w[-1] < 0.5 * w[0]:
        sub = "falling"
    elif w.max() / w.min() < 2.0:
        sub = "stalled"
    else:
        sub = "neither"
    return "bounded", sub


def fmt(x: float | None, digits: int = 3) -> str:
    """A number for a table cell."""
    if x is None:
        return "-"
    if isinstance(x, int) or (float(x).is_integer() and abs(x) >= 1000):
        return f"{int(x):,}"
    return f"{x:.{digits}g}"


def row_of(rec: dict) -> dict:
    """The matrix table's columns for one record."""
    cls, sub = classify(rec)
    return {
        "name": rec["name"],
        "field": rec["_field"],
        "grid": rec["_grid"],
        "sweeps": rec["_sweeps"],
        "rtol": rec["_rtol"],
        "corner_free": rec["_cf"],
        "class": cls + (f" ({sub})" if sub else ""),
        "stop": rec["stop"],
        "outer": rec["outer"],
        "velocity_step_outer": rec["velocity_step_outer"],
        "residual_min": rec["residual_min"],
        "residual_min_at": rec["residual_min_at"],
        "residual_end": rec["residual_end"],
        "estimate_end": rec.get("estimate_end"),
        "max_speed_end": rec["max_speed_end"],
        "at_end": rec["at_end"],
        "max_speed_peak": rec["max_speed_peak"],
        "inner_mean": rec["inner_mean"],
        "inner_max": rec["inner_max"],
        "cap_hits": rec["cap_hits"],
        "seconds": rec["seconds"],
        "seconds_per_outer": rec["seconds_per_outer"],
        "face_hash": rec["face_hash"],
        "commit": rec["commit"],
        "started": rec["started"],
    }


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """A markdown table."""
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


def keep(mode: str, data: object) -> None:
    """The printed numbers, as JSON beside the records."""
    (OUT / f"tables_{mode}.json").write_text(json.dumps(data, indent=1, default=float))


def matrix(_args: argparse.Namespace) -> None:
    """Measurement 1 and the counterfactual rows, one line per run."""
    recs = records()
    order = {f: i for i, f in enumerate(FIELDS)}
    rows = sorted(
        (row_of(r) for r in recs.values()),
        key=lambda r: (
            r["corner_free"],
            r["rtol"] != "1e-8",
            order[r["field"]],
            GRIDS.index(r["grid"]),
            r["sweeps"],
        ),
    )
    headers = [
        "Run",
        "Class",
        "Stop",
        "Outer",
        "velocity_step at",
        "Residual: least (at), end",
        "Estimate at end",
        "Largest speed at end (m/s), cell",
        "Peak speed",
        "CG per correction: mean, largest",
        "Cap hits",
        "Wall (s), per outer",
        "Face hash",
    ]
    table = [
        [
            r["name"],
            r["class"],
            r["stop"],
            fmt(r["outer"]),
            fmt(r["velocity_step_outer"]),
            f"{r['residual_min']:.2e} ({r['residual_min_at']:,}), {r['residual_end']:.2e}",
            fmt(r["estimate_end"]),
            f"{r['max_speed_end']:.3g}, {tuple(r['at_end'])}",
            f"{r['max_speed_peak']:.3g}",
            f"{r['inner_mean']:.0f}, {r['inner_max']}",
            fmt(r["cap_hits"]),
            f"{r['seconds']:.0f}, {r['seconds_per_outer']:.3f}",
            r["face_hash"][:16],
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("matrix", rows)


def base(_args: argparse.Namespace) -> None:
    """The base fields: stop, count, nu_t0 percentiles and the Z scales."""
    rows = []
    for grid in GRIDS:
        path = OUT / f"base_{grid}.json"
        if not path.exists():
            continue
        rec = load(f"base_{grid}")
        rows.append(
            {
                "grid": grid,
                "stop": rec["stop"],
                "outer": rec["outer"],
                "velocity_step_outer": rec["velocity_step_outer"],
                "seconds": rec["seconds"],
                "fluid_cells": rec.get("fluid_cells"),
                "core_cells": rec.get("core_cells"),
                "nut0_core": rec.get("nut0_core"),
                "nut0_all": rec.get("nut0_all"),
                "nut0_max": rec.get("nut0_max"),
                "core_median_over_nu_air": rec.get("core_median_over_nu_air"),
                "scales": rec.get("scales"),
                "max_speed_end": rec["max_speed_end"],
                "at_end": rec["at_end"],
                "face_hash": rec["face_hash"],
            }
        )
    headers = [
        "Grid",
        "Stop",
        "Outer",
        "velocity_step at",
        "Fluid, core cells",
        "nu_t0 core: 5th, 25th, median, 75th, 95th (m^2/s)",
        "Largest",
        "Core median / nu_air",
        "s for Z2, Z3, Z4",
        "Largest speed, cell",
    ]
    table = [
        [
            r["grid"],
            r["stop"],
            fmt(r["outer"]),
            fmt(r["velocity_step_outer"]),
            f"{r['fluid_cells']}, {r['core_cells']}",
            ", ".join(f"{x:.3g}" for x in r["nut0_core"]) if r["nut0_core"] else "-",
            fmt(r["nut0_max"]),
            fmt(r["core_median_over_nu_air"], 4),
            ", ".join(f"{r['scales'][k]:.4g}" for k in ("Z2", "Z3", "Z4"))
            if r["scales"]
            else "-",
            f"{r['max_speed_end']:.3g}, {tuple(r['at_end'])}",
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("base", rows)


def field_difference(a: dict, b: dict) -> dict:
    """Largest and RMS cell-centred velocity difference over non-SOLID cells, and the faces'."""
    solid = a["cell_type"] == 2
    du = np.where(solid, 0.0, b["u_c"] - a["u_c"])
    dv = np.where(solid, 0.0, b["v_c"] - a["v_c"])
    speed = np.hypot(du, dv)
    k = int(np.argmax(speed))
    j, i = np.unravel_index(k, speed.shape)
    return {
        "max_abs_du": float(np.abs(du).max()),
        "max_abs_dv": float(np.abs(dv).max()),
        "max_speed_difference": float(speed.max()),
        "at_xy": [float(a["xc"][i]), float(a["yc"][j])],
        "rms_speed_difference": float(np.sqrt(np.mean(speed[~solid] ** 2))),
        "median_speed_difference": float(np.median(speed[~solid])),
        "faces_max_abs_du": float(np.abs(b["u_faces"] - a["u_faces"]).max()),
        "faces_max_abs_dv": float(np.abs(b["v_faces"] - a["v_faces"]).max()),
    }


def converged(rec: dict) -> bool:
    """True for the solver's own stop."""
    return rec["stop"] == "error_estimate_and_continuity"


def pairs(_args: argparse.Namespace) -> None:
    """Measurement 2: one sweep against ten where both converged."""
    recs = records()
    rows = []
    for field in FIELDS:
        for grid in GRIDS:
            one, ten = f"{field}_{grid}_s1", f"{field}_{grid}_s10"
            if one not in recs or ten not in recs:
                continue
            a, b = recs[one], recs[ten]
            if not (converged(a) and converged(b)):
                continue
            d = field_difference(fields(one), fields(ten))
            rows.append(
                {
                    "field": field,
                    "grid": grid,
                    "outer_one": a["outer"],
                    "outer_ten": b["outer"],
                    "seconds_one": a["seconds"],
                    "seconds_ten": b["seconds"],
                    **d,
                    "max_over_tolerance": d["max_speed_difference"] / TOLERANCE_MS,
                }
            )
    headers = [
        "Field",
        "Grid",
        "Outer: one, ten",
        "Wall (s): one, ten",
        "max |du|, |dv| (m/s)",
        "Largest speed difference, at",
        "RMS, median",
        "Largest / tolerance (4.5e-7 m/s)",
    ]
    table = [
        [
            r["field"],
            r["grid"],
            f"{r['outer_one']:,}, {r['outer_ten']:,}",
            f"{r['seconds_one']:.0f}, {r['seconds_ten']:.0f}",
            f"{r['max_abs_du']:.2e}, {r['max_abs_dv']:.2e}",
            f"{r['max_speed_difference']:.2e}, ({r['at_xy'][0]:.3g}, {r['at_xy'][1]:.3g})",
            f"{r['rms_speed_difference']:.2e}, {r['median_speed_difference']:.2e}",
            f"{r['max_over_tolerance']:.3g}",
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("pairs", rows)


def bilinear(
    xc: np.ndarray, yc: np.ndarray, f: np.ndarray, px: np.ndarray, py: np.ndarray
) -> np.ndarray:
    """Bilinear interpolation from a uniform cell-centre lattice."""
    dx, dy = float(xc[1] - xc[0]), float(yc[1] - yc[0])
    fx, fy = (px - xc[0]) / dx, (py - yc[0]) / dy
    i = np.clip(np.floor(fx).astype(int), 0, xc.size - 2)
    j = np.clip(np.floor(fy).astype(int), 0, yc.size - 2)
    tx, ty = np.clip(fx - i, 0.0, 1.0), np.clip(fy - j, 0.0, 1.0)
    return (
        (1 - tx) * (1 - ty) * f[j, i]
        + tx * (1 - ty) * f[j, i + 1]
        + (1 - tx) * ty * f[j + 1, i]
        + tx * ty * f[j + 1, i + 1]
    )


def sample_points() -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """The sensors and the two lines, points inside obstacle rectangles excluded."""
    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    boxes = [
        (o["x_start"], o["x_end"], o["y_start"], o["y_end"]) for o in raw["obstacles"]
    ]
    width, height = raw["domain"]["width"], raw["domain"]["height"]

    def outside(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        inside = np.zeros_like(px, dtype=bool)
        for x0, x1, y0, y1 in boxes:
            inside |= (px > x0) & (px < x1) & (py > y0) & (py < y1)
        return ~inside

    sx = np.array([s["x"] for s in raw["sensors"]], dtype=float)
    sy = np.array([s["y"] for s in raw["sensors"]], dtype=float)
    names = [s["name"] for s in raw["sensors"]]
    ys = np.arange(EDGE_MARGIN, height - EDGE_MARGIN + 1e-9, LINE_STEP)
    vx, vy = np.full_like(ys, VERTICAL_X), ys
    xs = np.arange(EDGE_MARGIN, width - EDGE_MARGIN + 1e-9, LINE_STEP)
    hx, hy = xs, np.full_like(xs, HORIZONTAL_Y)
    out = {}
    for key, px, py, labels in (
        ("sensors", sx, sy, names),
        (f"vertical x={VERTICAL_X}", vx, vy, None),
        (f"horizontal y={HORIZONTAL_Y}", hx, hy, None),
    ):
        m = outside(px, py)
        out[key] = (
            px[m],
            py[m],
            [labels[k] for k in np.flatnonzero(m)] if labels else None,
        )
    return out


def far_from_obstacles(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """True where a point is more than FAR from every obstacle rectangle."""
    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    far = np.ones_like(px, dtype=bool)
    for o in raw["obstacles"]:
        dx = np.maximum.reduce([o["x_start"] - px, np.zeros_like(px), px - o["x_end"]])
        dy = np.maximum.reduce([o["y_start"] - py, np.zeros_like(py), py - o["y_end"]])
        far &= np.hypot(dx, dy) > FAR
    return far


def interpolated(
    name: str, px: np.ndarray, py: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """u and v of a record at points, SOLID cells at zero."""
    f = fields(name)
    solid = f["cell_type"] == 2
    u = np.where(solid, 0.0, f["u_c"])
    v = np.where(solid, 0.0, f["v_c"])
    return bilinear(f["xc"], f["yc"], u, px, py), bilinear(f["xc"], f["yc"], v, px, py)


def grids(_args: argparse.Namespace) -> None:
    """Measurement 3: each field converged on all three grids, coarse against 200x75."""
    recs = records()
    points = sample_points()
    result = {"lines": {}, "sensors": {}}
    rows = []
    sensor_rows = []
    for field in FIELDS:
        chosen = {}
        for grid in GRIDS:
            for sweeps in (1, 10):
                name = f"{field}_{grid}_s{sweeps}"
                if name in recs and converged(recs[name]):
                    chosen[grid] = name
                    break
        if len(chosen) < 3:
            continue
        fine = chosen["200x75"]
        for key, (px, py, labels) in points.items():
            uf, vf = interpolated(fine, px, py)
            if key == "sensors":
                for k, label in enumerate(labels):
                    sensor_rows.append(
                        {
                            "field": field,
                            "sensor": label,
                            "x": float(px[k]),
                            "y": float(py[k]),
                            **{
                                f"{g}_u": float(
                                    interpolated(
                                        chosen[g], px[k : k + 1], py[k : k + 1]
                                    )[0][0]
                                )
                                for g in GRIDS
                            },
                            **{
                                f"{g}_v": float(
                                    interpolated(
                                        chosen[g], px[k : k + 1], py[k : k + 1]
                                    )[1][0]
                                )
                                for g in GRIDS
                            },
                        }
                    )
            far = far_from_obstacles(px, py)
            for coarse in ("40x15", "80x30"):
                uc, vc = interpolated(chosen[coarse], px, py)
                du, dv = uc - uf, vc - vf
                for comp, d in (("u", du), ("v", dv)):
                    k = int(np.argmax(np.abs(d)))
                    kf = int(np.argmax(np.abs(d[far]))) if far.any() else 0
                    rows.append(
                        {
                            "field": field,
                            "points": key,
                            "coarse": coarse,
                            "runs": [chosen[coarse], fine],
                            "component": comp,
                            "n_points": int(px.size),
                            "n_far": int(far.sum()),
                            "max_abs": float(np.abs(d).max()),
                            "at": [float(px[k]), float(py[k])],
                            "rms": float(np.sqrt(np.mean(d * d))),
                            "max_abs_far": float(np.abs(d[far]).max())
                            if far.any()
                            else None,
                            "at_far": [float(px[far][kf]), float(py[far][kf])]
                            if far.any()
                            else None,
                            "rms_far": float(np.sqrt(np.mean(d[far] ** 2)))
                            if far.any()
                            else None,
                            "scale_fine_max_abs": float(
                                np.abs(uf if comp == "u" else vf).max()
                            ),
                        }
                    )
    headers = [
        "Field",
        "Points",
        "Coarse grid",
        "Comp.",
        "Points (far)",
        "Largest |diff| (m/s), at",
        "RMS",
        "Largest, far from obstacles, at",
        "RMS far",
        "Scale: fine max |comp.|",
    ]
    table = [
        [
            r["field"],
            r["points"],
            r["coarse"],
            r["component"],
            f"{r['n_points']} ({r['n_far']})",
            f"{r['max_abs']:.3g}, ({r['at'][0]:.2f}, {r['at'][1]:.2f})",
            f"{r['rms']:.3g}",
            f"{r['max_abs_far']:.3g}, ({r['at_far'][0]:.2f}, {r['at_far'][1]:.2f})"
            if r["max_abs_far"] is not None
            else "-",
            fmt(r["rms_far"]),
            f"{r['scale_fine_max_abs']:.3g}",
        ]
        for r in rows
    ]
    print_table(headers, table)
    headers = [
        "Field",
        "Sensor (x, y)",
        "u: 40x15, 80x30, 200x75 (m/s)",
        "v: 40x15, 80x30, 200x75 (m/s)",
    ]
    table = [
        [
            r["field"],
            f"{r['sensor']} ({r['x']:.1f}, {r['y']:.1f})",
            ", ".join(f"{r[f'{g}_u']:+.4f}" for g in GRIDS),
            ", ".join(f"{r[f'{g}_v']:+.4f}" for g in GRIDS),
        ]
        for r in sensor_rows
    ]
    print_table(headers, table)
    result["lines"], result["sensors"] = rows, sensor_rows
    keep("grids", result)


def counterfactual(suffix_key: str, label: str, mode: str) -> None:
    """Measurements 4 and 5: each counterfactual record against its committed-rule record."""
    recs = records()
    rows = []
    for name, rec in sorted(recs.items()):
        if not rec[suffix_key]:
            continue
        base_name = f"{rec['_field']}_{rec['_grid']}_s{rec['_sweeps']}"
        if base_name not in recs:
            continue
        ref = recs[base_name]
        d = (
            field_difference(fields(base_name), fields(name))
            if (converged(ref) and converged(rec))
            else None
        )
        rows.append(
            {
                "name": name,
                "reference": base_name,
                "setting": rec["_rtol"] if mode == "rtol" else "corner-free",
                "class": " ".join(classify(rec)).strip(),
                "class_reference": " ".join(classify(ref)).strip(),
                "outer": rec["outer"],
                "outer_reference": ref["outer"],
                "outer_ratio": rec["outer"] / ref["outer"],
                "velocity_step_outer": rec["velocity_step_outer"],
                "velocity_step_reference": ref["velocity_step_outer"],
                "inner_mean": rec["inner_mean"],
                "inner_max": rec["inner_max"],
                "inner_mean_reference": ref["inner_mean"],
                "inner_max_reference": ref["inner_max"],
                "seconds": rec["seconds"],
                "seconds_reference": ref["seconds"],
                "cap_hits": rec["cap_hits"],
                "difference": d,
            }
        )
    headers = [
        "Run",
        label,
        "Class: this, committed",
        "Outer: this, committed (ratio)",
        "velocity_step at: this, committed",
        "CG mean, largest: this; committed",
        "Wall (s): this, committed",
        "Faces: max |du|, |dv| (m/s)",
        "Cells: largest speed difference, at; RMS",
    ]
    table = [
        [
            r["name"],
            r["setting"],
            f"{r['class']}, {r['class_reference']}",
            f"{r['outer']:,}, {r['outer_reference']:,} ({r['outer_ratio']:.3f})",
            f"{fmt(r['velocity_step_outer'])}, {fmt(r['velocity_step_reference'])}",
            f"{r['inner_mean']:.0f}, {r['inner_max']}; {r['inner_mean_reference']:.0f}, {r['inner_max_reference']}",
            f"{r['seconds']:.0f}, {r['seconds_reference']:.0f}",
            f"{r['difference']['faces_max_abs_du']:.2e}, {r['difference']['faces_max_abs_dv']:.2e}"
            if r["difference"]
            else "-",
            f"{r['difference']['max_speed_difference']:.2e}, ({r['difference']['at_xy'][0]:.3g}, {r['difference']['at_xy'][1]:.3g}); {r['difference']['rms_speed_difference']:.2e}"
            if r["difference"]
            else "-",
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep(mode, rows)


def corner(_args: argparse.Namespace) -> None:
    """Measurement 4."""
    counterfactual("_cf", "Setting", "corner")


def rtol(_args: argparse.Namespace) -> None:
    """Measurement 5."""
    counterfactual("_rtol_set", "pressure_rtol", "rtol")


def cavity(_args: argparse.Namespace) -> None:
    """Measurement 6: the cavity's stop, count and centreline extremes."""
    rows = []
    for n in (40, 80):
        path = OUT / f"cavity_{n}x{n}.json"
        if not path.exists():
            continue
        rec = load(f"cavity_{n}x{n}")
        c = rec["centerline"]
        u_min = c["u_on_vertical"]["min"]
        v_max = c["v_on_horizontal"]["max"]
        v_min = c["v_on_horizontal"]["min"]
        rows.append(
            {
                "grid": f"{n}x{n}",
                "stop": rec["stop"],
                "outer": rec["outer"],
                "velocity_step_outer": rec["velocity_step_outer"],
                "seconds": rec["seconds"],
                "inner_mean": rec["inner_mean"],
                "inner_max": rec["inner_max"],
                "cap_hits": rec["cap_hits"],
                "u_min": u_min,
                "v_max": v_max,
                "v_min": v_min,
                "against_recalled": {
                    k: {
                        "sample_relative": (row["value"] - RECALLED[k][0])
                        / abs(RECALLED[k][0]),
                        "parabola_relative": (row["parabola"][1] - RECALLED[k][0])
                        / abs(RECALLED[k][0]),
                        "position_difference": row["parabola"][0] - RECALLED[k][1],
                    }
                    for k, row in (("u_min", u_min), ("v_max", v_max), ("v_min", v_min))
                },
                "face_hash": rec["face_hash"],
            }
        )
    headers = [
        "Grid",
        "Stop",
        "Outer",
        "velocity_step at",
        "Wall (s)",
        "CG mean, largest",
        "u_min (sample at y; parabola at y)",
        "v_max (sample at x; parabola at x)",
        "v_min (sample at x; parabola at x)",
        "Parabola against recalled: u_min, v_max, v_min",
    ]
    table = [
        [
            r["grid"],
            r["stop"],
            fmt(r["outer"]),
            fmt(r["velocity_step_outer"]),
            f"{r['seconds']:.0f}",
            f"{r['inner_mean']:.0f}, {r['inner_max']}",
            f"{r['u_min']['value']:.5f} at {r['u_min']['at']:.4f}; {r['u_min']['parabola'][1]:.5f} at {r['u_min']['parabola'][0]:.4f}",
            f"{r['v_max']['value']:.5f} at {r['v_max']['at']:.4f}; {r['v_max']['parabola'][1]:.5f} at {r['v_max']['parabola'][0]:.4f}",
            f"{r['v_min']['value']:.5f} at {r['v_min']['at']:.4f}; {r['v_min']['parabola'][1]:.5f} at {r['v_min']['parabola'][0]:.4f}",
            ", ".join(
                f"{100 * r['against_recalled'][k]['parabola_relative']:+.1f}%"
                for k in ("u_min", "v_max", "v_min")
            ),
        ]
        for r in rows
    ]
    print_table(headers, table)
    keep("cavity", rows)


def everything(args: argparse.Namespace) -> None:
    """Every table."""
    for name, func in (
        ("matrix", matrix),
        ("base", base),
        ("pairs", pairs),
        ("grids", grids),
        ("corner", corner),
        ("rtol", rtol),
        ("cavity", cavity),
    ):
        print(f"### {name}\n")
        func(args)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, func in (
        ("matrix", matrix),
        ("base", base),
        ("pairs", pairs),
        ("grids", grids),
        ("corner", corner),
        ("rtol", rtol),
        ("cavity", cavity),
        ("all", everything),
    ):
        sub.add_parser(name).set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
