"""Measurement 3 of prompt 47: three sources on a converged coupled flow.

Usage:
    python transport47.py check NAME SOURCE [--cfl C] [--cap-seconds S]
    python transport47.py run NAME SOURCE [--cfl C] [--cap-seconds S] [--tag T]
    python transport47.py sources

NAME is a converged flow record of coupled47.py under results/builder47/;
SOURCE is S1, S2 or S3 (the report's section 2.5), or a moved source
S1m, S2m, S3m once `check` has written the move. `check` marches the source
on NAME, scores the discrimination check (the report's section 2.5) and,
when it fails, applies the stated move rule and writes the moved square to
results/builder47/sources.json; `run` is the march alone, recorded as
transport_NAME_SOURCE[_TAG]. `sources` prints the squares in force.

The march is prompt 46's (transport46.py) with the source square a
parameter and the Courant number of the transport step set here (0.4 by
default, the report's section 2.5; --cfl 0.1 for the check row). The
committed TransportSolver steps the two classes 0.5 and 5 micrometres on
the record's frozen face velocities with its eddy viscosity passed as
`eddy_viscosity` and the configuration's turbulent_schmidt, from a zero
field. The scoring (sensors, surfaces, 0.2 m segments, hotspots) is
transport46's, imported.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))

import transport46  # noqa: E402
from transport46 import (  # noqa: E402
    CLASSES,
    EDGE_TOL,
    EMISSION,
    REMOVAL_TOL,
    RULE_TOL,
    WINDOW,
    commit,
    face_deposition,
    interpolated,
    score,
    surface_names,
)

from src.boundary_concentration import ConcentrationBoundary  # noqa: E402
from src.boundary_registry import BoundaryRegistry  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.particles import ParticlePhysics  # noqa: E402
from src.solver_transport import TransportSolver  # noqa: E402
from src.staggered import FaceVelocities  # noqa: E402

OUT = ROOT / "results" / "builder47"
SOURCES_FILE = OUT / "sources.json"
# The prompt's three sources as 0.1 m closed squares (x0, x1, y0, y1).
SOURCES = {
    "S1": (0.75, 0.85, 1.15, 1.25),
    "S2": (3.80, 3.90, 2.00, 2.10),
    "S3": (6.15, 6.25, 1.15, 1.25),
}
WHERE = {
    "S1": "an operator in the recirculation roll beside the door wall",
    "S2": "a process emission just above the litho tool's top",
    "S3": "an operator at the hood bench",
}
MARCH_CFL = 0.4
CAP_SECONDS = 600.0
FLOOR_RATIO = 1e-6
NEEDED = 2
MOVE = 0.5


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def sources_in_force() -> dict[str, dict]:
    """The three sources and any moved ones the check has written."""
    out = {
        name: {"box": list(box), "where": WHERE[name], "moved_from": None}
        for name, box in SOURCES.items()
    }
    if SOURCES_FILE.exists():
        out.update(json.loads(SOURCES_FILE.read_text()))
    return out


def source_rate(mesh: Mesh, box: tuple, emission: float) -> tuple[np.ndarray, dict]:
    """The source array, particles per cubic metre per second, and what it covers.

    transport46.source_rate with the square a parameter: the cells whose
    centres lie in the closed square, the rate spread by volume so the
    emission totals Q per second per metre of depth.
    """
    x0, x1, y0, y1 = box
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
        raise ValueError(f"no cell centre lies in the source square {box}")
    rate = np.where(inside, emission / covered, 0.0)
    return rate, {
        "box": list(box),
        "cells": int(inside.sum()),
        "area": covered,
        "emission": emission,
        "rate_per_cell": emission / covered,
    }


def touches(box: tuple, x0: float, x1: float, y0: float, y1: float) -> bool:
    """Whether the closed square shares an edge or a corner with the rectangle."""
    bx0, bx1, by0, by1 = box
    return (
        bx0 <= x1 + EDGE_TOL
        and bx1 >= x0 - EDGE_TOL
        and by0 <= y1 + EDGE_TOL
        and by1 >= y0 - EDGE_TOL
    )


def surfaces_touched(box: tuple, raw: dict) -> list[str]:
    """The named surfaces whose extent the source square touches.

    Obstacle tops and sides by the obstacle's rectangle; the floor, the
    ceiling and the two walls by the domain's edges. A surface the source
    touches does not count in the discrimination check.
    """
    width, height = raw["domain"]["width"], raw["domain"]["height"]
    x0, x1, y0, y1 = box
    out: list[str] = []
    for o in raw["obstacles"]:
        if not touches(box, o["x_start"], o["x_end"], o["y_start"], o["y_end"]):
            continue
        if abs(y0 - o["y_end"]) <= EDGE_TOL:
            out.append(f"{o['name']} top")
        if abs(y1 - o["y_start"]) <= EDGE_TOL:
            out.append(f"{o['name']} underside")
        if abs(x1 - o["x_start"]) <= EDGE_TOL:
            out.append(f"{o['name']} west")
        if abs(x0 - o["x_end"]) <= EDGE_TOL:
            out.append(f"{o['name']} east")
    if y0 <= EDGE_TOL:
        out.append("floor")
    if y1 >= height - EDGE_TOL:
        out.append("ceiling")
    if x0 <= EDGE_TOL:
        out.append("left wall")
    if x1 >= width - EDGE_TOL:
        out.append("right wall")
    return out


def sensors_inside(box: tuple, cfg: SimConfig) -> list[str]:
    """Sensors whose point lies in the closed square."""
    x0, x1, y0, y1 = box
    return [
        s.name
        for s in cfg.sensors
        if x0 - EDGE_TOL <= s.x <= x1 + EDGE_TOL
        and y0 - EDGE_TOL <= s.y <= y1 + EDGE_TOL
    ]


def moved_box(box: tuple, raw: dict) -> tuple[tuple, str]:
    """The report's move rule: 0.5 m up, else 0.5 m in x away from the nearest return."""
    x0, x1, y0, y1 = box
    height = raw["domain"]["height"]
    up = (x0, x1, y0 + MOVE, y1 + MOVE)
    blocked = up[3] > height + EDGE_TOL or any(
        touches(up, o["x_start"], o["x_end"], o["y_start"], o["y_end"])
        and not (abs(up[2] - o["y_end"]) <= EDGE_TOL)
        for o in raw["obstacles"]
    )
    if not blocked:
        return up, "0.5 m upward"
    xm = 0.5 * (x0 + x1)
    returns = [
        0.5 * (seg["x_start"] + seg["x_end"])
        for seg in raw["boundaries"].values()
        if seg["type"] == "fixed_flow_outlet" and seg["location"] == "bottom"
    ]
    nearest = min(returns, key=lambda c: abs(c - xm))
    sign = 1.0 if xm >= nearest else -1.0
    return (x0 + sign * MOVE, x1 + sign * MOVE, y0, y1), (
        f"0.5 m in x away from the return at x {nearest:.3g}"
    )


# ---------------------------------------------------------------------------
# The march
# ---------------------------------------------------------------------------


def surface_concentrations(
    c: np.ndarray, mesh: Mesh, cond: object, raw: dict
) -> dict[str, float]:
    """The largest C_P over each named surface's depositing faces."""
    names_u, names_v = surface_names(mesh, cond, raw)
    values = np.where(mesh.cell_type == SOLID, 0.0, c)
    padded_x = np.pad(values, ((0, 0), (1, 1)))
    padded_y = np.pad(values, ((1, 1), (0, 0)))
    out: dict[str, float] = {}
    for (j, i), label in names_v.items():
        kind = "floor" if label.startswith("floor ") else label
        cp = float(padded_y[j, i] + padded_y[j + 1, i])
        out[kind] = max(out.get(kind, 0.0), cp)
    for (j, i), label in names_u.items():
        cp = float(padded_x[j, i] + padded_x[j, i + 1])
        out[label] = max(out.get(label, 0.0), cp)
    return out


def march(
    name: str,
    source_name: str,
    box: tuple,
    cfl: float,
    cap_seconds: float,
    emission: float,
    tag: str = "",
) -> dict:
    """March both classes on the record NAME from the source square to the steady rule or the cap."""
    rec = json.loads((OUT / f"{name}.json").read_text())
    if rec["stop"] != "error_estimate_and_continuity":
        raise ValueError(f"{name} did not converge ({rec['stop']})")
    kept = np.load(OUT / f"{name}.npz")
    raw = json.loads(json.dumps(rec["raw"]))
    raw["transport"]["cfl_number"] = cfl
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    physics = ParticlePhysics(cfg)
    boundary = ConcentrationBoundary(mesh, cfg, physics, BoundaryRegistry(cfg))
    solver = TransportSolver(mesh, cfg, physics, boundary)
    faces = FaceVelocities.copy_of(kept["u_faces"], kept["v_faces"])
    nu_t = np.ascontiguousarray(kept["nu_t"], dtype=np.float64)
    rate, source = source_rate(mesh, box, emission)
    source["name"] = source_name
    source["where"] = WHERE.get(source_name[:2], "")
    window = WINDOW
    limit = min(solver.stable_dt(faces, k) for k in CLASSES)
    per_window = math.ceil(window / limit)
    dt = window / per_window
    sensors = [(s.name, (s.x, s.y)) for s in cfg.sensors]
    points = [p for _, p in sensors]
    inside = sensors_inside(box, cfg)
    touched = surfaces_touched(box, raw)
    tree = commit()
    label = f"transport_{name}_{source_name}" + (f"_{tag}" if tag else "")
    started = datetime.now().isoformat(timespec="seconds")
    print(
        f"{label} start {started} commit {tree} cfl {cfl} dt {dt:.4e} "
        f"({per_window} steps per {window} s window) source {json.dumps(source)} "
        f"sensors inside {inside} surfaces touched {touched}",
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
    cap_windows = round(cap_seconds / window)
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
                    emission * window
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
                f"{label} t {t:6.1f}s {line} wall {time.perf_counter() - t0:7.1f}s",
                flush=True,
            )
    for k in CLASSES:
        stop.setdefault(k, "cap")
    seconds = time.perf_counter() - t0

    out: dict = {
        "name": label,
        "flow_record": name,
        "flow_commit": rec["commit"],
        "flow_face_hash": rec["face_hash"],
        "commit": tree,
        "started": started,
        "seconds": seconds,
        "grid": rec["grid"],
        "variant": rec["variant"],
        "source_name": source_name,
        "classes": {str(k): cfg.particle_sizes[k] for k in CLASSES},
        "turbulent_schmidt": cfg.transport.turbulent_schmidt if cfg.transport else None,
        "cfl_number": cfl,
        "dt": dt,
        "steps_per_window": per_window,
        "window": window,
        "cap_seconds": cap_seconds,
        "rule_tol": RULE_TOL,
        "removal_tol": REMOVAL_TOL,
        "floor_ratio": FLOOR_RATIO,
        "source": source,
        "sensors_inside_source": inside,
        "surfaces_touched_by_source": touched,
        "segment": transport46.SEGMENT,
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
    source_cells = rate > 0.0
    for k in CLASSES:
        c = fields[k]
        cond = boundary.faces_for(k)
        rate_u, rate_v = face_deposition(c, mesh, cond)
        scored = score(rate_u, rate_v, mesh, cond, raw, emission)
        budget = solver.budget[k]
        last = history[k][-1]
        readings = dict(zip(out["sensors"], last["sensors"], strict=True))
        source_c = float(c[source_cells].max())
        floor = FLOOR_RATIO * source_c
        above = {s: v for s, v in readings.items() if v > floor and s not in inside}
        order = sorted(above, key=lambda s: -above[s])
        surface_c = surface_concentrations(c, mesh, cond, raw)
        surfaces_above = {
            s: v for s, v in surface_c.items() if v > floor and s not in touched
        }
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
            "source_cell_concentration": source_c,
            "floor": floor,
            "sensors": readings,
            "sensors_above_floor": above,
            "sensor_order_above_floor": order,
            "surface_concentrations": surface_c,
            "surfaces_above_floor": dict(
                sorted(surfaces_above.items(), key=lambda kv: -kv[1])
            ),
            "discriminates": len(above) >= NEEDED or len(surfaces_above) >= NEEDED,
            "budget": {
                "source": budget.source,
                "inflow": budget.inflow,
                "outflow": budget.outflow,
                "deposited": dict(budget.deposited),
                "current": budget.current,
                "relative_residual": budget.relative(),
            },
            "outflow_rate_over_source": last.get("outflow_rate", 0.0) / emission,
            "budget_deposition_rate_over_source": (
                last.get("deposition_rate", 0.0) / emission
            ),
            **scored,
            "history": history[k],
        }
        arrays[f"C_{k}"] = c
        arrays[f"deposition_u_{k}"] = rate_u
        arrays[f"deposition_v_{k}"] = rate_v
    (OUT / f"{label}.json").write_text(json.dumps(out, default=float))
    np.savez(OUT / f"{label}.npz", **arrays)
    summary = {
        k: {
            key: v[key]
            for key in (
                "stop",
                "t_end",
                "last_window",
                "floor",
                "sensors_above_floor",
                "sensor_order_above_floor",
                "surfaces_above_floor",
                "discriminates",
                "hotspots",
                "deposition_rate_over_source",
            )
        }
        for k, v in out["per_class"].items()
    }
    print(
        f"{label} done {datetime.now().isoformat(timespec='seconds')} {seconds:.0f}s "
        f"{json.dumps(summary, default=float)}",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def check(args: argparse.Namespace) -> None:
    """The discrimination check of one source on NAME; move it once if it fails."""
    sources = sources_in_force()
    entry = sources[args.source]
    box = tuple(entry["box"])
    out = march(
        args.name, args.source, box, float(args.cfl), args.cap_seconds, args.emission
    )
    passes = {k: v["discriminates"] for k, v in out["per_class"].items()}
    verdict = all(passes.values())
    raw = json.loads((OUT / f"{args.name}.json").read_text())["raw"]
    result = {
        "source": args.source,
        "box": list(box),
        "flow_record": args.name,
        "passes_per_class": passes,
        "passes": verdict,
        "moved": None,
    }
    if not verdict and args.source in SOURCES:
        new_box, rule = moved_box(box, raw)
        moved_name = f"{args.source}m"
        sources[moved_name] = {
            "box": list(new_box),
            "where": f"{WHERE[args.source]}, moved {rule}",
            "moved_from": args.source,
            "rule": rule,
        }
        SOURCES_FILE.write_text(
            json.dumps({k: v for k, v in sources.items() if k not in SOURCES}, indent=1)
        )
        result["moved"] = {"name": moved_name, "box": list(new_box), "rule": rule}
        print(f"check {args.source}: FAILS; moved to {moved_name} {new_box} ({rule})")
    else:
        print(
            f"check {args.source}: {'PASSES' if verdict else 'FAILS (already moved)'}"
        )
    checks = {}
    if (OUT / "checks.json").exists():
        checks = json.loads((OUT / "checks.json").read_text())
    checks[f"{args.name}/{args.source}"] = result
    (OUT / "checks.json").write_text(json.dumps(checks, indent=1))


def run(args: argparse.Namespace) -> None:
    """The march of one source on NAME."""
    entry = sources_in_force()[args.source]
    march(
        args.name,
        args.source,
        tuple(entry["box"]),
        float(args.cfl),
        args.cap_seconds,
        args.emission,
        tag=args.tag,
    )


def sources(_args: argparse.Namespace) -> None:
    """The squares in force."""
    print(json.dumps(sources_in_force(), indent=1))


def main() -> None:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode, func in (("check", check), ("run", run)):
        p = sub.add_parser(mode)
        p.add_argument("name")
        p.add_argument("source")
        p.add_argument("--cfl", default=str(MARCH_CFL))
        p.add_argument("--cap-seconds", type=float, default=CAP_SECONDS)
        p.add_argument("--emission", type=float, default=EMISSION)
        if mode == "run":
            p.add_argument("--tag", default="")
        p.set_defaults(func=func)
    sub.add_parser("sources").set_defaults(func=sources)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
