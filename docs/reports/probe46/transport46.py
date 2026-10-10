"""Measurement 4 of prompt 46: where particles go on a converged coupled flow.

Usage:
    python transport46.py run NAME [--cap-seconds S] [--window W]

NAME is a converged record of coupled46.py under results/builder46/. The
committed TransportSolver steps two classes, 0.5 and 5 micrometres (indices
2 and 4 of the configuration's five), on the record's frozen face velocities
with its eddy viscosity passed as `eddy_viscosity` and the configuration's
turbulent_schmidt, from a zero field, with one continuous source (the
report's section 2.2, measurement 4): the cells whose centres lie in the
0.2 m square [2.6, 2.8] x [0.9, 1.1] m, the rate spread by volume so the
emission totals Q particles per second per metre of depth. dt is the smaller
of the two classes' stable_dt, rounded down so that a window of W seconds is
a whole number of steps. The steady rule is the report's: over every window
the in-domain count's relative change, each sensor's change relative to the
largest sensor reading, and the removal rate against Q; steady when the first
two are below RULE_TOL and the removal is within REMOVAL_TOL of Q; cap at
--cap-seconds. The record (transport_NAME.json and .npz) carries the sensors,
the deposition rate per named surface and per 0.2 m segment, the five largest
segments per class, the budget, and the fields.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.boundary_concentration import (  # noqa: E402
    SURFACE_FLOOR,
    SURFACE_WALL,
    ConcentrationBoundary,
)
from src.boundary_registry import BoundaryRegistry  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.particles import ParticlePhysics  # noqa: E402
from src.solver_transport import TransportSolver  # noqa: E402
from src.staggered import FaceVelocities  # noqa: E402

OUT = ROOT / "results" / "builder46"
CLASSES = (2, 4)
SOURCE_BOX = (2.6, 2.8, 0.9, 1.1)
EMISSION = 1.0e4
WINDOW = 1.0
CAP_SECONDS = 600.0
RULE_TOL = 1e-4
REMOVAL_TOL = 1e-3
SEGMENT = 0.2
HOTSPOTS = 5
EDGE_TOL = 1e-9


def commit() -> str:
    """The checked-out commit."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Measuring devices
# ---------------------------------------------------------------------------


def source_rate(mesh: Mesh, emission: float) -> tuple[np.ndarray, dict]:
    """The source array, particles per cubic metre per second, and what it covers."""
    x0, x1, y0, y1 = SOURCE_BOX
    xc, yc = np.meshgrid(mesh.xc, mesh.yc)
    inside = (
        (xc >= x0 - EDGE_TOL)
        & (xc <= x1 + EDGE_TOL)
        & (yc >= y0 - EDGE_TOL)
        & (yc <= y1 + EDGE_TOL)
        & (mesh.cell_type != SOLID)
    )
    volume = np.outer(mesh.dy_cell, mesh.dx_cell)
    covered = float(volume[inside].sum())
    if covered == 0.0:
        raise ValueError("no cell centre lies in the source box")
    rate = np.where(inside, emission / covered, 0.0)
    return rate, {
        "box": SOURCE_BOX,
        "cells": int(inside.sum()),
        "area": covered,
        "emission": emission,
        "rate_per_cell": emission / covered,
    }


def interpolated(
    field: np.ndarray, mesh: Mesh, points: list[tuple[float, float]]
) -> list[float]:
    """Bilinear interpolation from the cell centres, SOLID cells at zero.

    Step 5's measurement 3 rule (`tables44.interpolated`), on the uniform
    product mesh: a point between four centres takes their bilinear
    combination, with a SOLID centre's value read as zero.
    """
    values = np.where(mesh.cell_type == SOLID, 0.0, field)
    xc, yc = mesh.xc, mesh.yc
    out = []
    for x, y in points:
        i = int(np.clip(np.searchsorted(xc, x) - 1, 0, xc.size - 2))
        j = int(np.clip(np.searchsorted(yc, y) - 1, 0, yc.size - 2))
        tx = (x - xc[i]) / (xc[i + 1] - xc[i])
        ty = (y - yc[j]) / (yc[j + 1] - yc[j])
        tx, ty = float(np.clip(tx, 0.0, 1.0)), float(np.clip(ty, 0.0, 1.0))
        out.append(
            float(
                (1 - tx) * (1 - ty) * values[j, i]
                + tx * (1 - ty) * values[j, i + 1]
                + (1 - tx) * ty * values[j + 1, i]
                + tx * ty * values[j + 1, i + 1]
            )
        )
    return out


def face_deposition(
    c: np.ndarray, mesh: Mesh, faces: object
) -> tuple[np.ndarray, np.ndarray]:
    """The deposition rate of every face, particles per second per metre of depth.

    `_book_deposition`'s rule: ``v_d A_f C_P`` with C_P the depositing face's
    one non-SOLID neighbour, read as the sum of its two neighbours over a
    zero-padded field.
    """
    c = np.where(mesh.cell_type == SOLID, 0.0, c)
    padded_x = np.pad(c, ((0, 0), (1, 1)))
    padded_y = np.pad(c, ((1, 1), (0, 0)))
    rate_u = (
        faces.deposition_u  # type: ignore[attr-defined]
        * mesh.dy_cell[:, None]
        * (padded_x[:, :-1] + padded_x[:, 1:])
    )
    rate_v = (
        faces.deposition_v  # type: ignore[attr-defined]
        * mesh.dx_cell[None, :]
        * (padded_y[:-1, :] + padded_y[1:, :])
    )
    return rate_u, rate_v


def surface_names(
    mesh: Mesh, faces: object, raw: dict
) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], str]]:
    """A name for every depositing face: u faces and v faces, by index.

    Floor pieces are contiguous runs of depositing faces along the domain's
    bottom row, named by their x interval. An interior v face with the
    floor code is an obstacle top, named by the obstacle whose x range
    holds it and whose top lies within half a cell of the face; an interior
    u face with the wall code is an obstacle side, named by the obstacle
    whose west or east edge lies within half a cell. The domain's two side
    walls and the ceiling are one surface each.
    """
    ny, nx = mesh.cell_type.shape
    dep_u = faces.deposition_u  # type: ignore[attr-defined]
    dep_v = faces.deposition_v  # type: ignore[attr-defined]
    sur_u = faces.surface_u  # type: ignore[attr-defined]
    sur_v = faces.surface_v  # type: ignore[attr-defined]
    obstacles = raw["obstacles"]
    half_x, half_y = 0.5 * float(mesh.dx), 0.5 * float(mesh.dy)
    names_v: dict[tuple[int, int], str] = {}
    names_u: dict[tuple[int, int], str] = {}
    # The floor, in pieces.
    start = None
    for i in range(nx + 1):
        depositing = i < nx and dep_v[0, i] > 0.0 and sur_v[0, i] == SURFACE_FLOOR
        if depositing and start is None:
            start = i
        if not depositing and start is not None:
            label = f"floor {mesh.x[start]:.2f}-{mesh.x[i]:.2f}"
            for ii in range(start, i):
                names_v[(0, ii)] = label
            start = None
    for i in range(nx):
        if dep_v[ny, i] > 0.0:
            names_v[(ny, i)] = "ceiling"
    for j in range(1, ny):
        for i in range(nx):
            if dep_v[j, i] <= 0.0:
                continue
            x, y = float(mesh.xc[i]), float(mesh.y[j])
            which = "obstacle top" if sur_v[j, i] == SURFACE_FLOOR else "underside"
            for o in obstacles:
                edge = o["y_end"] if which == "obstacle top" else o["y_start"]
                if (
                    o["x_start"] - half_x <= x <= o["x_end"] + half_x
                    and abs(y - edge) <= half_y + EDGE_TOL
                ):
                    names_v[(j, i)] = (
                        f"{o['name']} {'top' if which == 'obstacle top' else 'underside'}"
                    )
                    break
            else:
                names_v[(j, i)] = f"{which} unmatched ({x:.2f}, {y:.2f})"
    for j in range(ny):
        for i in (0, nx):
            if dep_u[j, i] > 0.0:
                names_u[(j, i)] = "left wall" if i == 0 else "right wall"
        for i in range(1, nx):
            if dep_u[j, i] <= 0.0 or sur_u[j, i] != SURFACE_WALL:
                continue
            x, y = float(mesh.x[i]), float(mesh.yc[j])
            for o in obstacles:
                if not o["y_start"] - half_y <= y <= o["y_end"] + half_y:
                    continue
                if abs(x - o["x_start"]) <= half_x + EDGE_TOL:
                    names_u[(j, i)] = f"{o['name']} west"
                    break
                if abs(x - o["x_end"]) <= half_x + EDGE_TOL:
                    names_u[(j, i)] = f"{o['name']} east"
                    break
            else:
                names_u[(j, i)] = f"side unmatched ({x:.2f}, {y:.2f})"
    return names_u, names_v


def by_surface(
    rate_u: np.ndarray,
    rate_v: np.ndarray,
    names_u: dict,
    names_v: dict,
    mesh: Mesh,
) -> tuple[dict[str, float], dict[str, float]]:
    """The deposition rate per named surface and per 0.2 m segment.

    A segment is named by its surface and the start of its 0.2 m bin along
    the surface: x for floors, tops and the ceiling, y for walls and sides.
    """
    surfaces: dict[str, float] = {}
    segments: dict[str, float] = {}
    for (j, i), label in names_v.items():
        value = float(rate_v[j, i])
        surfaces[label] = surfaces.get(label, 0.0) + value
        x = float(mesh.xc[i])
        key = f"{label} | x {math.floor(x / SEGMENT + EDGE_TOL) * SEGMENT:.1f}"
        segments[key] = segments.get(key, 0.0) + value
    for (j, i), label in names_u.items():
        value = float(rate_u[j, i])
        surfaces[label] = surfaces.get(label, 0.0) + value
        y = float(mesh.yc[j])
        key = f"{label} | y {math.floor(y / SEGMENT + EDGE_TOL) * SEGMENT:.1f}"
        segments[key] = segments.get(key, 0.0) + value
    return surfaces, segments


# ---------------------------------------------------------------------------
# The march
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """March both classes to the steady rule or the cap on one converged record."""
    name = args.name
    rec = json.loads((OUT / f"{name}.json").read_text())
    if rec["stop"] != "error_estimate_and_continuity":
        raise ValueError(f"{name} did not converge ({rec['stop']})")
    kept = np.load(OUT / f"{name}.npz")
    raw = rec["raw"]
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    physics = ParticlePhysics(cfg)
    boundary = ConcentrationBoundary(mesh, cfg, physics, BoundaryRegistry(cfg))
    solver = TransportSolver(mesh, cfg, physics, boundary)
    faces = FaceVelocities.copy_of(kept["u_faces"], kept["v_faces"])
    nu_t = np.ascontiguousarray(kept["nu_t"], dtype=np.float64)
    rate, source = source_rate(mesh, args.emission)
    window = float(args.window)
    limit = min(solver.stable_dt(faces, k) for k in CLASSES)
    per_window = math.ceil(window / limit)
    dt = window / per_window
    sensors = [(s.name, (s.x, s.y)) for s in cfg.sensors]
    points = [p for _, p in sensors]
    tree = commit()
    started = datetime.now().isoformat(timespec="seconds")
    print(
        f"transport_{name} start {started} commit {tree} dt {dt:.4e} "
        f"({per_window} steps per {window} s window) source {json.dumps(source)}",
        flush=True,
    )

    fields = {k: np.zeros(mesh.cell_type.shape) for k in CLASSES}
    volume = np.outer(mesh.dy_cell, mesh.dx_cell)
    history: dict[int, list[dict]] = {k: [] for k in CLASSES}
    stop: dict[int, str] = {}
    previous: dict[int, dict] = {}
    t0 = time.perf_counter()
    t = 0.0
    windows = 0
    cap_windows = round(args.cap_seconds / window)
    live = {k: True for k in CLASSES}
    while any(live.values()) and windows < cap_windows:
        for _ in range(per_window):
            for k in CLASSES:
                if live[k]:
                    fields[k] = solver.solve_timestep(
                        fields[k], faces, k, dt, sources=rate, eddy_viscosity=nu_t
                    )
        t += window
        windows += 1
        for k in CLASSES:
            if not live[k]:
                continue
            budget = solver.budget[k]
            total = float(np.sum(fields[k] * volume))
            readings = interpolated(fields[k], mesh, points)
            removed = budget.outflow + sum(budget.deposited.values())
            deposited = sum(budget.deposited.values())
            now = {
                "t": t,
                "total": total,
                "sensors": readings,
                "removed": removed,
                "deposited": deposited,
                "outflow": budget.outflow,
            }
            entry = {"t": t, "total": total, "sensors": readings}
            if k in previous:
                was = previous[k]
                scale = max(max(readings), 1e-300)
                entry["total_change"] = abs(total - was["total"]) / max(total, 1e-300)
                entry["sensor_change"] = max(
                    abs(a - b) / scale
                    for a, b in zip(readings, was["sensors"], strict=True)
                )
                entry["removal_over_source"] = (removed - was["removed"]) / (
                    args.emission * window
                )
                entry["deposition_rate"] = (deposited - was["deposited"]) / window
                entry["outflow_rate"] = (budget.outflow - was["outflow"]) / window
                steady = (
                    entry["total_change"] < RULE_TOL
                    and entry["sensor_change"] < RULE_TOL
                    and abs(entry["removal_over_source"] - 1.0) < REMOVAL_TOL
                )
                if steady:
                    live[k] = False
                    stop[k] = "steady"
            history[k].append(entry)
            previous[k] = now
        if windows % 10 == 0 or not any(live.values()):
            line = " ".join(
                f"class {k}: total {history[k][-1]['total']:.4g} "
                f"dtot {history[k][-1].get('total_change', float('nan')):.2e} "
                f"dsens {history[k][-1].get('sensor_change', float('nan')):.2e} "
                f"rem {history[k][-1].get('removal_over_source', float('nan')):.5f}"
                for k in CLASSES
            )
            print(
                f"transport_{name} t {t:6.1f}s {line} wall {time.perf_counter() - t0:7.1f}s",
                flush=True,
            )
    for k in CLASSES:
        stop.setdefault(k, "cap")
    seconds = time.perf_counter() - t0

    out: dict = {
        "name": f"transport_{name}",
        "flow_record": name,
        "flow_commit": rec["commit"],
        "flow_face_hash": rec["face_hash"],
        "commit": tree,
        "started": started,
        "seconds": seconds,
        "grid": rec["grid"],
        "variant": rec["variant"],
        "classes": {str(k): cfg.particle_sizes[k] for k in CLASSES},
        "turbulent_schmidt": cfg.transport.turbulent_schmidt if cfg.transport else None,
        "cfl_number": cfg.transport.cfl_number if cfg.transport else None,
        "dt": dt,
        "steps_per_window": per_window,
        "window": window,
        "cap_seconds": args.cap_seconds,
        "rule_tol": RULE_TOL,
        "removal_tol": REMOVAL_TOL,
        "source": source,
        "segment": SEGMENT,
        "sensors": [n for n, _ in sensors],
        "sensor_points": points,
        "per_class": {},
    }
    arrays: dict[str, np.ndarray] = {
        "xc": mesh.xc,
        "yc": mesh.yc,
        "cell_type": mesh.cell_type,
        "source_rate": rate,
    }
    for k in CLASSES:
        c = fields[k]
        cond = boundary.faces_for(k)
        rate_u, rate_v = face_deposition(c, mesh, cond)
        names_u, names_v = surface_names(mesh, cond, raw)
        surfaces, segments = by_surface(rate_u, rate_v, names_u, names_v, mesh)
        ranked = sorted(segments.items(), key=lambda kv: -kv[1])
        budget = solver.budget[k]
        last = history[k][-1]
        readings = last["sensors"]
        order = sorted(range(len(readings)), key=lambda i: -readings[i])
        total_faces = float(rate_u.sum() + rate_v.sum())
        out["per_class"][str(k)] = {
            "diameter": cfg.particle_sizes[k],
            "settling_velocity": physics.settling_velocity(k),
            "diffusion_coeff": physics.diffusion_coeff(k),
            "deposition_velocity": {
                s: physics.deposition_velocity(k, s)
                for s in ("floor", "ceiling", "wall")
            },
            "stop": stop[k],
            "t_end": last["t"],
            "windows": len(history[k]),
            "last_window": {
                key: last.get(key)
                for key in ("total_change", "sensor_change", "removal_over_source")
            },
            "total_in_domain": last["total"],
            "sensors": dict(zip(out["sensors"], readings, strict=True)),
            "sensor_order": [out["sensors"][i] for i in order],
            "budget": {
                "source": budget.source,
                "inflow": budget.inflow,
                "outflow": budget.outflow,
                "deposited": dict(budget.deposited),
                "current": budget.current,
                "relative_residual": budget.relative(),
            },
            "deposition_rate_total": total_faces,
            "deposition_rate_over_source": total_faces / args.emission,
            "outflow_rate_over_source": last.get("outflow_rate", 0.0) / args.emission,
            "budget_deposition_rate_over_source": (
                last.get("deposition_rate", 0.0) / args.emission
            ),
            "by_surface": dict(sorted(surfaces.items(), key=lambda kv: -kv[1])),
            "by_surface_share": {
                s: v / total_faces if total_faces > 0.0 else None
                for s, v in sorted(surfaces.items(), key=lambda kv: -kv[1])
            },
            "segments": dict(ranked),
            "hotspots": [s for s, _ in ranked[:HOTSPOTS]],
            "hotspot_rates": [v for _, v in ranked[:HOTSPOTS]],
            "history": history[k],
        }
        arrays[f"C_{k}"] = c
        arrays[f"deposition_u_{k}"] = rate_u
        arrays[f"deposition_v_{k}"] = rate_v
    (OUT / f"transport_{name}.json").write_text(json.dumps(out, default=float))
    np.savez(OUT / f"transport_{name}.npz", **arrays)
    summary = {
        k: {
            key: v[key]
            for key in (
                "stop",
                "t_end",
                "last_window",
                "sensors",
                "sensor_order",
                "hotspots",
                "deposition_rate_over_source",
            )
        }
        for k, v in out["per_class"].items()
    }
    print(
        f"transport_{name} done {datetime.now().isoformat(timespec='seconds')} {seconds:.0f}s "
        f"{json.dumps(summary, default=float)}",
        flush=True,
    )


def main() -> None:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    p.add_argument("name")
    p.add_argument("--cap-seconds", type=float, default=CAP_SECONDS)
    p.add_argument("--window", type=float, default=WINDOW)
    p.add_argument("--emission", type=float, default=EMISSION)
    p.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
