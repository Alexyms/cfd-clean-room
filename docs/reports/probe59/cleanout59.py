"""Prompt 59b probe E: the clean-out march at Courant 0.1 and 0.4, the time error itself.

ADR-014 decision 2 (review 59, B1). On the product room at 160x60 the converged
L0 flow of ECR-002 step 8 (results/builder56/L0_160x60.npz, commit b9e8191)
carries two classes, 0.5 and 5 um (prompt 56's CLASSES), with Sc_t 0.7: the
room uniformly contaminated at t = 0 (C0 = 1 in every non-SOLID cell), the
supply clean, marched at one step for both classes (the smaller stable_dt)
until every sensor of both classes is below 1e-3 of C0, or T_CAP. Two arms:

    python cleanout59.py 0.1 [--concurrent N]
    python cleanout59.py 0.4 [--concurrent N]

Each record (results/builder59/cleanout_cfl<c>.json) carries, per class and
sensor, the first time the sensor's cell falls below 1e-2 and 1e-3 of C0 (and
stays below to the end), the free-area mean of the field every step with the
well-mixed exp(-N t) beside it, N the supply flow over the free area, the
budget's relative residual at the end, and the wall time. tables59.py's T7
sets the two arms side by side. One BLAS thread.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.boundary_concentration import ConcentrationBoundary  # noqa: E402
from src.boundary_registry import BoundaryRegistry  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.particles import ParticlePhysics  # noqa: E402
from src.solver_transport import TransportSolver  # noqa: E402
from src.staggered import FaceVelocities  # noqa: E402

OUT = ROOT / "results" / "builder59"
L0 = ROOT / "results" / "builder56" / "L0_160x60"
PRODUCT = ROOT / "configs" / "clean_room_default.yaml"
GRID = (160, 60)
CLASSES = (2, 4)
SCHMIDT = 0.7
C0 = 1.0
LEVELS = (1e-2, 1e-3)
T_CAP = 600.0


def commit() -> str:
    """The short commit of the tree the probe runs in."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.strip()


def room_raw(cfl: float) -> dict:
    """The product file's mapping on GRID with the arm's Courant number and Sc_t."""
    raw = yaml.safe_load(PRODUCT.read_text(encoding="utf-8"))
    nx, ny = GRID
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    domain, block = raw["domain"], raw["solver"]
    v_min = (domain["width"] / nx) * (domain["height"] / ny)
    block["mass_imbalance_tol"] = (
        1e-4 * raw["fluid"]["density"] * v_min / block["t_end"]
    )
    raw["transport"]["cfl_number"] = cfl
    raw["transport"]["turbulent_schmidt"] = SCHMIDT
    return raw


def sensor_cells(cfg: SimConfig, mesh: Mesh) -> list[tuple[str, int, int]]:
    """Each sensor's name and the (j, i) of the cell whose centre is nearest it."""
    cells = []
    for s in cfg.sensors:
        i = int(np.argmin(np.abs(mesh.xc - s.x)))
        j = int(np.argmin(np.abs(mesh.yc - s.y)))
        if mesh.cell_type[j, i] == SOLID:
            raise ValueError(f"sensor {s.name} sits in a SOLID cell")
        cells.append((s.name, j, i))
    return cells


def last_time_above(
    times: np.ndarray, values: np.ndarray, level: float
) -> float | None:
    """The first time after which the reading stays below LEVEL times C0; None if never."""
    above = np.flatnonzero(values >= level * C0)
    if above.size == 0:
        return 0.0
    if above[-1] == values.size - 1:
        return None
    return float(times[above[-1] + 1])


def march(cfl: float, concurrent: int, cap: float = T_CAP) -> dict:
    """The clean-out at one Courant number, recorded."""
    raw = room_raw(cfl)
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    physics = ParticlePhysics(cfg)
    boundary = ConcentrationBoundary(mesh, cfg, physics, BoundaryRegistry(cfg))
    solver = TransportSolver(mesh, cfg, physics, boundary)
    kept = np.load(f"{L0}.npz")
    l0 = json.loads(Path(f"{L0}.json").read_text(encoding="utf-8"))
    if l0["stop"] != "error_estimate_and_continuity":
        raise ValueError(f"L0_160x60 did not converge: {l0['stop']}")
    faces = FaceVelocities.copy_of(kept["u_faces"], kept["v_faces"])
    nu_t = np.ascontiguousarray(kept["nu_t"], dtype=np.float64)
    live = mesh.cell_type != SOLID
    area = (mesh.dx_cell[None, :] * mesh.dy_cell[:, None]) * live
    free_area = float(area.sum())
    supply = cfg.boundaries["hepa_supply"]
    rate = (supply.x_end - supply.x_start) * supply.velocity / free_area
    dt = min(solver.stable_dt(faces, k) for k in CLASSES)
    sensors = sensor_cells(cfg, mesh)
    fields = {k: np.where(live, C0, 0.0) for k in CLASSES}
    times = [0.0]
    means = {k: [C0] for k in CLASSES}
    readings = {k: {name: [C0] for name, _, _ in sensors} for k in CLASSES}
    started = datetime.now().isoformat(timespec="seconds")
    tree = commit()
    print(
        f"cleanout cfl {cfl}: start {started} commit {tree} dt {dt:.4e} "
        f"({1.0 / dt:.0f} steps per s) N {rate:.4f} per s concurrent {concurrent}",
        flush=True,
    )
    t0 = perf_counter()
    t, step = 0.0, 0
    while t < cap:
        step += 1
        t = step * dt
        for k in CLASSES:
            fields[k] = solver.solve_timestep(
                fields[k], faces, k, dt, eddy_viscosity=nu_t
            )
        times.append(t)
        for k in CLASSES:
            means[k].append(float((fields[k] * area).sum() / free_area))
            for name, j, i in sensors:
                readings[k][name].append(float(fields[k][j, i]))
        if step % 2000 == 0:
            worst = max(readings[k][n][-1] for k in CLASSES for n, _, _ in sensors)
            print(
                f"cleanout cfl {cfl}: t {t:.1f} s, worst sensor {worst:.3e}, "
                f"mean {means[CLASSES[0]][-1]:.3e}, {perf_counter() - t0:.0f} s",
                flush=True,
            )
        if all(
            readings[k][n][-1] < LEVELS[-1] * C0 for k in CLASSES for n, _, _ in sensors
        ):
            break
    wall = perf_counter() - t0
    t_arr = np.array(times)
    record = {
        "cfl": cfl,
        "grid": list(GRID),
        "classes": list(CLASSES),
        "sizes_m": [cfg.particle_sizes[k] for k in CLASSES],
        "turbulent_schmidt": SCHMIDT,
        "flow": {
            "record": "results/builder56/L0_160x60",
            "commit": l0["commit"],
            "face_hash": l0["face_hash"],
        },
        "c0": C0,
        "dt": dt,
        "steps": step,
        "t_end": float(t),
        "stopped_by": "sensors below 1e-3" if t < cap else "cap",
        "air_change_rate_per_s": rate,
        "free_area_m2": free_area,
        "started": started,
        "commit": tree,
        "concurrent_processes": concurrent,
        "wall_seconds": wall,
        "seconds_per_step": wall / step,
        "wall_seconds_per_simulated_second": wall / t,
        "budget_relative": {
            str(k): float(solver.budget[k].relative()) for k in CLASSES
        },
        "recovery": {
            str(k): {
                name: {
                    f"{level:g}": last_time_above(
                        t_arr, np.array(readings[k][name]), level
                    )
                    for level in LEVELS
                }
                for name, _, _ in sensors
            }
            for k in CLASSES
        },
        "mean_recovery": {
            str(k): {
                f"{level:g}": last_time_above(t_arr, np.array(means[k]), level)
                for level in LEVELS
            }
            for k in CLASSES
        },
        "times": times,
        "means": {str(k): means[k] for k in CLASSES},
        "well_mixed": [float(C0 * np.exp(-rate * tt)) for tt in times],
        "readings": {str(k): readings[k] for k in CLASSES},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"cleanout_cfl{cfl:g}.json").write_text(json.dumps(record), encoding="utf-8")
    print(
        f"cleanout cfl {cfl}: {record['stopped_by']} at t {t:.2f} s after {step} steps, "
        f"{wall:.0f} s wall; recovery {json.dumps(record['recovery'])}",
        flush=True,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt 59b probe E, the clean-out march"
    )
    parser.add_argument("cfl", type=float)
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument("--cap", type=float, default=T_CAP)
    args = parser.parse_args()
    march(args.cfl, args.concurrent, args.cap)


if __name__ == "__main__":
    main()
