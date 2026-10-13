"""Item 6 of prompt 51: prompt 48's standard upwind product rows on 80x30 and 160x60.

Usage:
    python product51.py run GRID LABEL [--root TREE]    GRID 80x30 or 160x60
    python product51.py compare GRID BASE BUILT         two records, and prompt 50's

``run`` builds prompt 48's row (docs/reports/probe48/coupled48.py: prompt
47's ``control`` arm, the product configuration regridded with prompt 46's
settings, standard variant, the advection key absent so k and eps are
advected by the default upwind) with ``coupled47.product_raw`` imported from
TREE, and solves it with TREE's committed StaggeredSolver and its committed
rule. Prompt 46 to 50's runners read the model's ``last_sweeps``, which this
branch renames, so this driver keeps its own trace: every outer iteration's
k and eps iteration counts (``last_iterations`` or, on main,
``last_sweeps``), the fallbacks (zero on main, which has none) and whether
both solves converged. A PositivityError is recorded with its message and
ends the run. The record is results/builder51/product_GRID_LABEL.json of the
main checkout, the final faces product_GRID_LABEL.npz. One BLAS thread.

``compare`` writes compare_product_GRID.json: the stop and outer count of
BASE, BUILT and prompt 50's record (results/builder50/standard_GRID_upwind),
seconds per outer iteration of BASE and BUILT, and the largest face
difference of BUILT from BASE and from prompt 50's faces over the velocity
scale, the largest prescribed boundary velocity
(``get_max_boundary_velocity``).
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[3]
MAIN = Path(
    subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
).parent
OUT = MAIN / "results" / "builder51"
BASE50 = MAIN / "results" / "builder50"
CAP = 10000


def face_hash(u: np.ndarray, v: np.ndarray) -> str:
    """SHA-256 over u's bytes then v's (docs/reports/probe41/outlet41.py)."""
    return hashlib.sha256(
        np.ascontiguousarray(u, dtype="<f8").tobytes()
        + np.ascontiguousarray(v, dtype="<f8").tobytes()
    ).hexdigest()


def run(args: argparse.Namespace) -> None:
    """One product row from TREE, traced, to its stop, its cap or a positivity error."""
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "docs" / "reports" / "probe47"))
    coupled47 = importlib.import_module("coupled47")
    if Path(coupled47.__file__).resolve().parents[3] != root:
        raise SystemExit(
            f"coupled47 was imported from {coupled47.__file__}, not {root}"
        )
    config_module = importlib.import_module("src.config")
    mesh_module = importlib.import_module("src.mesh")
    boundary_module = importlib.import_module("src.boundary_staggered")
    solver_module = importlib.import_module("src.solver_staggered")
    turbulence = importlib.import_module("src.turbulence")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    raw = coupled47.product_raw(args.grid, "standard", "control", CAP)
    config = config_module.SimConfig.from_dict(raw)
    mesh = mesh_module.Mesh(config)
    boundary = boundary_module.StaggeredBoundary(mesh, config)
    solver = solver_module.StaggeredSolver(mesh, config, boundary)
    model = solver._model
    trace: dict[str, list] = {
        "k_iterations": [],
        "eps_iterations": [],
        "k_fallbacks": [],
        "eps_fallbacks": [],
        "scalar_converged": [],
    }

    def callback(_state: object) -> None:
        counts = getattr(model, "last_iterations", None)
        if counts is None:
            counts = model.last_sweeps
        fallbacks = getattr(model, "last_fallbacks", (0, 0))
        trace["k_iterations"].append(int(counts[0]))
        trace["eps_iterations"].append(int(counts[1]))
        trace["k_fallbacks"].append(int(fallbacks[0]))
        trace["eps_fallbacks"].append(int(fallbacks[1]))
        trace["scalar_converged"].append(bool(model.solves_converged))

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    positivity = None
    try:
        solver.solve_steady(on_iteration=callback)
    except turbulence.PositivityError as err:
        positivity = str(err)
    wall = time.perf_counter() - t0
    outer = len(trace["k_iterations"])
    faces = solver.face_velocities
    record = {
        "grid": args.grid,
        "label": args.label,
        "commit": commit,
        "root": str(root),
        "started": started,
        "stop": solver.stop_reason if positivity is None else "positivity_error",
        "positivity_error": positivity,
        "outer": outer,
        "wall_seconds": wall,
        "seconds_per_outer": wall / outer,
        "k_iterations_mean": float(np.mean(trace["k_iterations"])),
        "eps_iterations_mean": float(np.mean(trace["eps_iterations"])),
        "fallbacks": int(sum(trace["k_fallbacks"]) + sum(trace["eps_fallbacks"])),
        "scalar_cap_hits": int(outer - sum(trace["scalar_converged"])),
        "velocity_scale": boundary.get_max_boundary_velocity(),
        "face_hash": None if faces is None else face_hash(faces.u, faces.v),
        "trace": trace,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"product_{args.grid}_{args.label}"
    (OUT / f"{name}.json").write_text(json.dumps(record))
    if faces is not None:
        np.savez_compressed(
            OUT / f"{name}.npz", u_faces=np.array(faces.u), v_faces=np.array(faces.v)
        )
    print(
        f"{name}: {record['stop']} at {outer}, {wall:.0f} s, "
        f"k/eps mean {record['k_iterations_mean']:.1f}/{record['eps_iterations_mean']:.1f}, "
        f"fallbacks {record['fallbacks']}",
        flush=True,
    )


def compare(args: argparse.Namespace) -> None:
    """BASE and BUILT side by side, with prompt 50's record of the row."""
    recs = {
        lab: json.loads((OUT / f"product_{args.grid}_{lab}.json").read_text())
        for lab in (args.base, args.built)
    }
    faces = {
        lab: np.load(OUT / f"product_{args.grid}_{lab}.npz")
        for lab in (args.base, args.built)
    }
    name50 = f"standard_{args.grid}_upwind"
    rec50 = json.loads((BASE50 / f"{name50}.json").read_text())
    faces50 = np.load(BASE50 / f"{name50}.npz")
    scale = recs[args.built]["velocity_scale"]

    def largest(a: np.lib.npyio.NpzFile, b: np.lib.npyio.NpzFile) -> float:
        return max(
            float(np.abs(a["u_faces"] - b["u_faces"]).max()),
            float(np.abs(a["v_faces"] - b["v_faces"]).max()),
        )

    out = {
        "grid": args.grid,
        "prompt50": {"stop": rec50["stop"], "outer": rec50["outer"]},
        "base_equals_prompt50_faces": largest(faces[args.base], faces50) == 0.0,
        "velocity_scale": scale,
        "face_difference_over_scale_built_vs_base": largest(
            faces[args.built], faces[args.base]
        )
        / scale,
        "face_difference_over_scale_built_vs_prompt50": largest(
            faces[args.built], faces50
        )
        / scale,
    }
    for lab, rec in recs.items():
        out[lab] = {
            key: rec[key]
            for key in (
                "commit",
                "stop",
                "outer",
                "wall_seconds",
                "seconds_per_outer",
                "k_iterations_mean",
                "eps_iterations_mean",
                "fallbacks",
                "scalar_cap_hits",
                "face_hash",
            )
        }
    out["outer_change_vs_prompt50"] = (
        recs[args.built]["outer"] - rec50["outer"]
    ) / rec50["outer"]
    out["seconds_per_outer_change"] = (
        recs[args.built]["seconds_per_outer"] - recs[args.base]["seconds_per_outer"]
    ) / recs[args.base]["seconds_per_outer"]
    text = json.dumps(out, indent=1)
    (OUT / f"compare_product_{args.grid}.json").write_text(text)
    print(text)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    p.add_argument("grid", choices=("80x30", "160x60"))
    p.add_argument("label")
    p.add_argument("--root", default=str(HERE))
    p.set_defaults(func=run)
    p = sub.add_parser("compare")
    p.add_argument("grid", choices=("80x30", "160x60"))
    p.add_argument("base")
    p.add_argument("built")
    p.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
