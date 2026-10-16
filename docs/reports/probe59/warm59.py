"""Prompt 59 probe A: re-solving the product room to a changed door state, warm against rest.

ADR-014 section A, tier 1. The product room (configs/clean_room_default.yaml,
320x120) is solved to the "door open" state three ways, and each run's outer
count and wall time are recorded:

    python warm59.py rest [--concurrent N]
        door open, from rest: the control, the cost of a solve that does not
        warm start.
    python warm59.py warm [--concurrent N]
        door open, warm-started from the converged door-closed state L0 of
        ECR-002 step 8 (results/builder56/L0_320x120.npz, commit b9e8191,
        harness row val018_320x120 at b9e8191): the cost of the event's start.
    python warm59.py back [--concurrent N]
        door closed again, warm-started from the warm run's converged state
        (results/builder59/warm.npz): the cost of the event's end.
    python warm59.py smoke
        the warm arm on 160x60 from L0_160x60 for five outer iterations, to
        check the plumbing; the record is labelled smoke.

The door open state is the model ADR-014 section B recommends: the door
segment (a wall, x = 0, y 0 to 2.1 m) becomes an exchange across the open
doorway, corridor air entering over the lower half of the opening at
EXCHANGE_VELOCITY and room air leaving over the upper half at the same
velocity, so the net flow through the doorway is zero and every return keeps
the share it has with the door closed. The inflow carries the supply's
turbulence keys (5%, 0.1 m), stated here because the probe chooses them.

The warm start replaces the solver's fields at rest with the saved faces,
pressure, k and eps, by patching the two calls solve_steady makes to allocate
them; nothing under src/ changes. The saved nu_t is the solve's under-relaxed
value, and the model rebuilds C_mu k^2 / eps from k and eps, which at a
converged stop differ by less than condition (e)'s tolerance.

One BLAS thread, as prompts 46 to 58: OPENBLAS_NUM_THREADS=1 before NumPy
loads. Records go to results/builder59/<arm>.json and .npz.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import hashlib
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

import src.solver_staggered as solver_module  # noqa: E402
from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.solver_staggered import StaggeredSolver  # noqa: E402
from src.stopping import IterationState  # noqa: E402

OUT = ROOT / "results" / "builder59"
L0_DIR = ROOT / "results" / "builder56"
PRODUCT = ROOT / "configs" / "clean_room_default.yaml"
EXCHANGE_VELOCITY = 0.25  # m/s, each way across the open doorway (ADR-014 B)
DOOR_SPLIT = 1.05  # m, the height below which corridor air enters
SAMPLE_EVERY = 10


def commit() -> str:
    """The short commit of the tree the probe runs in."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.strip()


def face_hash(u: np.ndarray, v: np.ndarray) -> str:
    """SHA-256 of the two face arrays' bytes, u then v."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(u).tobytes())
    digest.update(np.ascontiguousarray(v).tobytes())
    return digest.hexdigest()


def product_raw(grid: tuple[int, int] | None = None, cap: int | None = None) -> dict:
    """The product file's mapping, the grid and cap overridden where asked."""
    raw = yaml.safe_load(PRODUCT.read_text(encoding="utf-8"))
    if grid is not None:
        nx, ny = grid
        raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
        domain, block = raw["domain"], raw["solver"]
        v_min = (domain["width"] / nx) * (domain["height"] / ny)
        block["mass_imbalance_tol"] = (
            1e-4 * raw["fluid"]["density"] * v_min / block["t_end"]
        )
    if cap is not None:
        raw["solver"]["max_simple_iter"] = cap
    return raw


def door_open(raw: dict) -> dict:
    """The mapping with the door segment replaced by the open doorway's exchange."""
    boundaries = raw["boundaries"]
    door = boundaries.pop("door")
    if door["type"] != "wall" or door["location"] != "left":
        raise ValueError(f"the door segment is not the left wall expected: {door}")
    top = door["y_end"]
    supply = boundaries["hepa_supply"]
    ordered = {}
    for name, segment in boundaries.items():
        ordered[name] = segment
        if name == "hepa_supply":
            ordered["door_in"] = {
                "type": "velocity_inlet",
                "location": "left",
                "y_start": door["y_start"],
                "y_end": DOOR_SPLIT,
                "velocity": EXCHANGE_VELOCITY,
                "turbulence_intensity": supply["turbulence_intensity"],
                "dissipation_length": supply["dissipation_length"],
            }
            ordered["door_out"] = {
                "type": "fixed_flow_outlet",
                "location": "left",
                "y_start": DOOR_SPLIT,
                "y_end": top,
                "velocity": EXCHANGE_VELOCITY,
            }
    raw["boundaries"] = ordered
    return raw


def saved_state(path: Path) -> dict[str, np.ndarray]:
    """The faces, pressure, k and eps of a kept record, as fresh float64 arrays."""
    kept = np.load(path)
    return {
        name: np.ascontiguousarray(kept[name], dtype=np.float64)
        for name in ("u_faces", "v_faces", "p", "k", "eps")
    }


def solve(
    name: str,
    raw: dict,
    start: dict[str, np.ndarray] | None,
    start_label: str,
    concurrent: int,
) -> dict:
    """One solve, from rest or from a saved state, recorded under NAME."""
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    solver = StaggeredSolver(mesh, cfg, boundary)
    if start is not None:
        u0, v0, p0 = start["u_faces"], start["v_faces"], start["p"]
        k0, eps0 = start["k"], start["eps"]
        model = solver._model
        if model is None:
            raise ValueError("the warm start needs the turbulence section")

        def fields_from_start(_mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return u0.copy(), v0.copy(), p0.copy()

        def state_from_start(_k: float, _eps: float):
            return model.state(k0.copy(), eps0.copy())

        solver_module.allocate_fields = fields_from_start
        model.initial = state_from_start

    trajectory: list[dict] = []
    t_start = perf_counter()

    def observe(state: IterationState) -> None:
        if state.iteration % SAMPLE_EVERY == 0:
            trajectory.append(
                {
                    "outer": state.iteration,
                    "seconds": perf_counter() - t_start,
                    "residual": state.residual,
                    "pressure_iterations": state.pressure_iterations,
                }
            )

    started = datetime.now().isoformat(timespec="seconds")
    tree = commit()
    print(
        f"{name}: start {started} commit {tree} from {start_label} "
        f"concurrent {concurrent}",
        flush=True,
    )
    u_c, v_c, p = solver.solve_steady(on_iteration=observe)
    wall = perf_counter() - t_start
    faces = solver.face_velocities
    state = solver.turbulence_state
    assert faces is not None and state is not None
    imbalance = solver.last_mass_imbalance
    live = mesh.cell_type != 2
    speed = np.hypot(u_c, v_c)
    record = {
        "name": name,
        "commit": tree,
        "started": started,
        "start": start_label,
        "concurrent_processes": concurrent,
        "grid": [cfg.nx, cfg.ny],
        "door_model": {
            "exchange_velocity": EXCHANGE_VELOCITY,
            "split_height": DOOR_SPLIT,
            "segments": {
                k: v for k, v in raw["boundaries"].items() if k.startswith("door")
            },
        },
        "max_simple_iter": cfg.max_simple_iter,
        "stop": solver.stop_reason,
        "converged": bool(solver.converged),
        "outer": len(solver.residual_history),
        "wall_seconds": wall,
        "seconds_per_outer": wall / max(len(solver.residual_history), 1),
        "stage_seconds": dict(solver.stage_seconds),
        "cap_hits": int(solver.pressure_cap_hits),
        "rule_version": solver.rule_version,
        "residual_first": float(solver.residual_history[0]),
        "residual_min": float(min(solver.residual_history)),
        "residual_end": float(solver.residual_history[-1]),
        "worst_imbalance_end": float(np.max(np.abs(imbalance[live]))),
        "signed_imbalance_end": float(np.sum(imbalance[live])),
        "max_speed_end": float(np.max(speed[live])),
        "face_hash": face_hash(faces.u, faces.v),
        "trajectory": trajectory,
        "residual_history": [float(r) for r in solver.residual_history],
        "raw": raw,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    np.savez(
        OUT / f"{name}.npz",
        u_faces=np.array(faces.u),
        v_faces=np.array(faces.v),
        p=p,
        u_c=u_c,
        v_c=v_c,
        k=np.array(state.k),
        eps=np.array(state.eps),
        nu_t=np.array(state.nu_t),
        cell_type=mesh.cell_type,
    )
    print(
        f"{name}: {record['stop']} after {record['outer']} outer, {wall:.0f} s "
        f"({record['seconds_per_outer']:.3f} s per outer), cap hits "
        f"{record['cap_hits']}, worst imbalance {record['worst_imbalance_end']:.2e}",
        flush=True,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 59 probe A, the warm start")
    parser.add_argument("arm", choices=("rest", "warm", "back", "smoke"))
    parser.add_argument("--concurrent", type=int, default=1)
    args = parser.parse_args()
    if args.arm == "rest":
        solve("rest", door_open(product_raw()), None, "rest", args.concurrent)
    elif args.arm == "warm":
        l0 = json.loads((L0_DIR / "L0_320x120.json").read_text(encoding="utf-8"))
        label = (
            f"results/builder56/L0_320x120.npz (commit {l0['commit']}, stop "
            f"{l0['stop']} at outer {l0['outer']}, face hash {l0['face_hash'][:12]})"
        )
        solve(
            "warm",
            door_open(product_raw()),
            saved_state(L0_DIR / "L0_320x120.npz"),
            label,
            args.concurrent,
        )
    elif args.arm == "back":
        warm = json.loads((OUT / "warm.json").read_text(encoding="utf-8"))
        label = (
            f"results/builder59/warm.npz (commit {warm['commit']}, stop "
            f"{warm['stop']} at outer {warm['outer']}, face hash "
            f"{warm['face_hash'][:12]})"
        )
        solve(
            "back", product_raw(), saved_state(OUT / "warm.npz"), label, args.concurrent
        )
    else:
        raw = door_open(product_raw(grid=(160, 60), cap=5))
        solve(
            "smoke",
            raw,
            saved_state(L0_DIR / "L0_160x60.npz"),
            "results/builder56/L0_160x60.npz",
            args.concurrent,
        )


if __name__ == "__main__":
    main()
