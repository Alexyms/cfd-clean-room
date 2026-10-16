"""Prompt 59 probe D: the product room's time scales and the transport march's cost, from the configuration.

ADR-014 sections A and C. Everything here is arithmetic on
configs/clean_room_default.yaml and one measurement on the converged L0 flow
of ECR-002 step 8 (results/builder56/L0_320x120.npz): the free area of the
slice, the supply, hood and return flows, the air change rate N (supply flow
over free area, per metre of depth), the turnover 1 / N, the well-mixed
100:1 recovery time ln(100) / N, the settling time of each class over the
room height, and the transport step the explicit scheme can take on the L0
faces at the file's cfl_number, with the wall time of one step per class,
which sets the cost of a simulated second. Output: a markdown table on
stdout and results/builder59/scales59.json.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import math
import sys
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
from src.mesh import Mesh  # noqa: E402
from src.particles import ParticlePhysics  # noqa: E402
from src.solver_transport import TransportSolver  # noqa: E402
from src.staggered import FaceVelocities  # noqa: E402

OUT = ROOT / "results" / "builder59"
PRODUCT = ROOT / "configs" / "clean_room_default.yaml"
L0 = ROOT / "results" / "builder56" / "L0_320x120.npz"
TIMED_STEPS = 20


def main() -> None:
    raw = yaml.safe_load(PRODUCT.read_text(encoding="utf-8"))
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    physics = ParticlePhysics(cfg)
    width, height = cfg.room_width, cfg.room_height
    obstacle_area = sum(
        (o.x_end - o.x_start) * (o.y_end - o.y_start) for o in cfg.obstacles
    )
    free_area = width * height - obstacle_area
    supply = cfg.boundaries["hepa_supply"]
    supply_flow = (supply.x_end - supply.x_start) * supply.velocity
    hood = cfg.boundaries["hood_exhaust"]
    hood_flow = (hood.y_end - hood.y_start) * hood.velocity
    returns = [b for n, b in cfg.boundaries.items() if n.startswith("floor_return")]
    return_width = sum(b.x_end - b.x_start for b in returns)
    return_flow = supply_flow - hood_flow
    rate = supply_flow / free_area
    scales = {
        "free_area_m2": free_area,
        "obstacle_area_m2": obstacle_area,
        "supply_flow_m2_per_s": supply_flow,
        "hood_flow_m2_per_s": hood_flow,
        "return_flow_m2_per_s": return_flow,
        "return_face_velocity_m_per_s": return_flow / return_width,
        "air_change_rate_per_s": rate,
        "air_changes_per_hour": rate * 3600.0,
        "turnover_s": 1.0 / rate,
        "recovery_100_to_1_well_mixed_s": math.log(100.0) / rate,
        "length_over_supply_speed_s": width / supply.velocity,
        "height_over_supply_speed_s": height / supply.velocity,
        "classes": [],
    }
    kept = np.load(L0)
    faces = FaceVelocities.copy_of(kept["u_faces"], kept["v_faces"])
    nu_t = np.ascontiguousarray(kept["nu_t"], dtype=np.float64)
    boundary = ConcentrationBoundary(mesh, cfg, physics, BoundaryRegistry(cfg))
    solver = TransportSolver(mesh, cfg, physics, boundary)
    c0 = np.where(mesh.cell_type != 2, 1.0e5, 0.0)
    for k, size in enumerate(cfg.particle_sizes):
        v_s = physics.settling_velocity(k)
        dt = solver.stable_dt(faces, k)
        c = c0.copy()
        t0 = perf_counter()
        for _ in range(TIMED_STEPS):
            c = solver.solve_timestep(c, faces, k, dt, eddy_viscosity=nu_t)
        seconds = (perf_counter() - t0) / TIMED_STEPS
        scales["classes"].append(
            {
                "size_m": size,
                "settling_velocity_m_per_s": v_s,
                "settling_time_over_height_s": height / v_s if v_s > 0 else math.inf,
                "stable_dt_s": dt,
                "steps_per_simulated_second": 1.0 / dt,
                "seconds_per_step": seconds,
                "wall_seconds_per_simulated_second": seconds / dt,
            }
        )
    scales["wall_seconds_per_simulated_second_all_classes"] = sum(
        c["wall_seconds_per_simulated_second"] for c in scales["classes"]
    )
    scales["cfl_number"] = cfg.transport.cfl_number
    scales["grid"] = [cfg.nx, cfg.ny]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scales59.json").write_text(json.dumps(scales, indent=1), encoding="utf-8")
    print("| Quantity | Value |")
    print("|---|---|")
    for key, value in scales.items():
        if key in ("classes", "grid"):
            continue
        print(
            f"| {key} | {value:.4g} |"
            if isinstance(value, float)
            else f"| {key} | {value} |"
        )
    print()
    print(
        "| Class (um) | Settling velocity (m/s) | Height over settling (s) | Stable dt at cfl (s) | Steps per simulated s | Wall s per step | Wall s per simulated s |"
    )
    print("|---|---|---|---|---|---|---|")
    for c in scales["classes"]:
        print(
            f"| {c['size_m'] * 1e6:g} | {c['settling_velocity_m_per_s']:.3e} | "
            f"{c['settling_time_over_height_s']:.4g} | {c['stable_dt_s']:.4g} | "
            f"{c['steps_per_simulated_second']:.4g} | {c['seconds_per_step']:.4g} | "
            f"{c['wall_seconds_per_simulated_second']:.4g} |"
        )


if __name__ == "__main__":
    main()
