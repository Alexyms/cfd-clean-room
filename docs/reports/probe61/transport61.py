"""Prompt 61: the Phase 3 marches on the vented room and the default room.

Usage:
    python transport61.py run NAME SOURCE --root TREE [--concurrent N]

NAME is a flow record of flow61.py under results/builder61/ (vents_320x120,
default_320x120); SOURCE is S1, S2 or S3, prompt 47's squares. The march is
prompt 56's (docs/reports/probe56/transport56.py, itself prompt 47's march with
the Schmidt number a parameter), copied with the differences marked
`# prompt 61:` below:

- the records are results/builder61/'s and the src/ is TREE's (--root), the
  commit the flows ran;
- removal is net of what the vents bring in: ``outflow - inflow`` plus the
  deposition. The default room's supply is clean, so its inflow is zero and
  the number is step 8's. In the vented room the rack's intake books the
  room air it draws as outflow and the discharge returns it as inflow, so
  the gross outflow counts recirculated particles as removed; the steady
  rule's removal clause and M3 read the net figure. The record keeps both
  (``outflow_rate_over_source`` gross, ``net_removal_over_source`` net);
- the record keeps the budget's inflow rate per window.

Everything else is prompt 56's at Sc_t 0.7: the two classes, the Courant
number 0.4, the frozen faces and eddy viscosity, the steady rule and the
600 s cap, the scoring (transport46.score).

"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import importlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
# prompt 61: TREE's src/ first, the commit the flows ran (--root).
_TREE = (
    Path(sys.argv[sys.argv.index("--root") + 1]).resolve()
    if "--root" in sys.argv
    else ROOT
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_TREE))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe47"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))
# prompt 61: the tree's src imported before the probe modules, which put the
# main checkout on the path; every later import of src reuses these.
importlib.import_module("src.boundary_concentration")
importlib.import_module("src.solver_transport")

import transport46  # noqa: E402
import transport47  # noqa: E402
from transport46 import (  # noqa: E402
    CLASSES,
    EMISSION,
    REMOVAL_TOL,
    RULE_TOL,
    WINDOW,
    commit,
    face_deposition,
    interpolated,
    score,
)
from transport47 import (  # noqa: E402
    CAP_SECONDS,
    FLOOR_RATIO,
    MARCH_CFL,
    NEEDED,
    SOURCES,
    WHERE,
    sensors_inside,
    source_rate,
    surface_concentrations,
    surfaces_touched,
)

from src.boundary_concentration import ConcentrationBoundary  # noqa: E402
from src.boundary_registry import BoundaryRegistry  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.particles import ParticlePhysics  # noqa: E402
from src.solver_transport import TransportSolver  # noqa: E402
from src.staggered import FaceVelocities  # noqa: E402

OUT = ROOT / "results" / "builder61"  # prompt 61
transport46.OUT = OUT
transport47.OUT = OUT
DEFAULT_SCHMIDT = 0.7
SCHMIDT_RANGE = (0.2, 0.7, 1.3)


def march(
    name: str,
    source_name: str,
    box: tuple,
    cfl: float,
    cap_seconds: float,
    emission: float,
    schmidt: float,
    concurrent: int,
    tag: str = "",
    allow_cap: bool = False,
) -> dict:
    """March both classes on the record NAME from the source square to the steady rule or the cap.

    transport47.march with the Schmidt number a parameter; the three
    differences are marked.
    """
    rec = json.loads((OUT / f"{name}.json").read_text())
    at_cap = rec["stop"] == "max_simple_iter"  # prompt 56: the labelled supplement
    if at_cap and not allow_cap:
        raise ValueError(
            f"{name} did not converge ({rec['stop']}); --at-cap marches the field at "
            "the cap as a labelled supplement"
        )
    if not at_cap and rec["stop"] != "error_estimate_and_continuity":
        raise ValueError(f"{name} did not converge ({rec['stop']})")
    import src.solver_transport as solver_module  # prompt 61: the tree's src

    if Path(solver_module.__file__).resolve().parents[1] != _TREE:
        raise SystemExit(f"src was imported from {solver_module.__file__}, not {_TREE}")
    kept = np.load(OUT / f"{name}.npz")
    raw = json.loads(json.dumps(rec["raw"]))
    raw["transport"]["cfl_number"] = cfl
    raw["transport"]["turbulent_schmidt"] = schmidt  # prompt 56: Sc_t a parameter
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
    # prompt 56: the label carries the Schmidt number.
    label = (
        f"transport_{name}_{source_name}_sc{schmidt:g}"
        + ("_cap" if at_cap else "")
        + (f"_{tag}" if tag else "")
    )
    started = datetime.now().isoformat(timespec="seconds")
    print(
        f"{label} start {started} commit {tree} cfl {cfl} Sc_t {schmidt:g} dt {dt:.4e} "
        f"({per_window} steps per {window} s window) source {json.dumps(source)} "
        f"sensors inside {inside} surfaces touched {touched} concurrent {concurrent}",
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
            # prompt 61: net of what the vents bring in.
            removed = budget.outflow - budget.inflow + sum(budget.deposited.values())
            deposited = sum(budget.deposited.values())
            now = {
                "t": t,
                "total": total,
                "sensors": readings,
                "removed": removed,
                "deposited": deposited,
                "outflow": budget.outflow,
                "inflow": budget.inflow,  # prompt 61
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
                entry["inflow_rate"] = (
                    budget.inflow - was["inflow"]
                ) / window  # prompt 61
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
        "flow_stop": rec["stop"],  # prompt 56
        "at_cap": at_cap,  # prompt 56: a supplement, not a converged room
        "commit": tree,
        "started": started,
        "seconds": seconds,
        "concurrent_processes": concurrent,  # prompt 56
        "grid": rec["grid"],
        "layout": rec.get("layout"),  # prompt 56
        "variant": rec["raw"]["turbulence"]["variant"],  # prompt 61
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
            # prompt 61: the net removal through the openings, M3's figure here.
            "inflow_rate_over_source": last.get("inflow_rate", 0.0) / emission,
            "net_removal_over_source": (
                last.get("outflow_rate", 0.0) - last.get("inflow_rate", 0.0)
            )
            / emission,
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


def run(args: argparse.Namespace) -> None:
    """The march of one source on NAME at one Schmidt number."""
    march(
        args.name,
        args.source,
        SOURCES[args.source],
        float(args.cfl),
        args.cap_seconds,
        args.emission,
        float(args.schmidt),
        args.concurrent,
        tag=args.tag,
        allow_cap=args.at_cap,
    )


def main() -> None:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    p.add_argument("name")
    p.add_argument("source", choices=sorted(SOURCES))
    p.add_argument("--schmidt", default=str(DEFAULT_SCHMIDT))
    p.add_argument("--cfl", default=str(MARCH_CFL))
    p.add_argument("--cap-seconds", type=float, default=CAP_SECONDS)
    p.add_argument("--emission", type=float, default=EMISSION)
    p.add_argument("--tag", default="")
    p.add_argument("--at-cap", action="store_true")
    p.add_argument("--concurrent", type=int, default=1)
    p.add_argument("--root", default=str(ROOT))  # prompt 61: read at import
    p.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
