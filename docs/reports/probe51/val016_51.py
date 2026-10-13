"""Item 6 of prompt 51: VAL-016's split criteria under the CG solve, against main.

Usage:
    python val016_51.py LABEL [--root TREE]

Runs tests/test_turbulent_channel.py's measurements from TREE (default this
checkout), with that file's own constants and fixture body: the standard
model's 2D Couette solve on 12 rows, its stop and outer count, the k and eps
iteration counts and fallbacks per outer iteration (read as annex51.py reads
them), criterion (a)'s two differences from the same-grid 1D reference and
its u_tau ratio, and criterion (b)'s core-k ratio range for both variants.
(b) runs the 1D reference alone (tests/couette_reference.py imports nothing
from src/), so it cannot move; it is recorded so the table is whole. The
record is results/builder51/val016_LABEL.json of the main checkout.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import importlib
import json
import math
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


def main() -> None:
    """Measure VAL-016 on TREE and write the record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--root", default=str(HERE))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    channel = importlib.import_module("tests.test_turbulent_channel")
    if Path(channel.__file__).resolve().parents[1] != root:
        raise SystemExit(f"tests were imported from {channel.__file__}, not {root}")
    ref = channel.ref
    config_module = importlib.import_module("src.config")
    mesh_module = importlib.import_module("src.mesh")
    boundary_module = importlib.import_module("src.boundary_staggered")
    solver_module = importlib.import_module("src.solver_staggered")

    config = config_module.SimConfig.from_dict(
        ref.case_raw(
            "couette", "standard", channel.ROWS, channel.LENGTH, channel.DX, channel.CFL
        )
    )
    mesh = mesh_module.Mesh(config)
    solver = solver_module.StaggeredSolver(
        mesh, config, boundary_module.StaggeredBoundary(mesh, config)
    )
    model = solver._model
    counts: dict[str, list[int]] = {"k": [], "eps": [], "fallbacks": []}

    def callback(_state: object) -> None:
        its = getattr(model, "last_iterations", None)
        if its is None:
            its = model.last_sweeps
        counts["k"].append(int(its[0]))
        counts["eps"].append(int(its[1]))
        counts["fallbacks"].append(int(sum(getattr(model, "last_fallbacks", (0, 0)))))

    t0 = time.perf_counter()
    solver.solve_steady(on_iteration=callback)
    wall = time.perf_counter() - t0
    faces, state = solver.face_velocities, solver.turbulence_state
    # The fixture's body (tests/test_turbulent_channel.py, couette_2d).
    i = int(channel.STATION * mesh.xc.size)
    u = np.array(faces.u[:, i])
    k = 0.5 * (state.k[:, i - 1] + state.k[:, i])
    nu_t = 0.5 * (state.nu_t[:, i - 1] + state.nu_t[:, i])
    j = channel.ROWS // 2
    a, b = ref.NU + nu_t[j - 1], ref.NU + nu_t[j]
    stress = 2.0 * a * b / (a + b) * (u[j] - u[j - 1]) / (mesh.yc[j] - mesh.yc[j - 1])
    u_tau = math.sqrt(stress)
    one_d = ref.reference("standard", channel.ROWS, 1)
    du = float(np.abs(u - np.array(one_d["u"])).max() / ref.U_W)
    dk = float(
        np.abs(k / u_tau**2 - np.array(one_d["k"]) / one_d["u_tau_mid"] ** 2).max()
    )
    core = {}
    for variant in ("standard", "rng"):
        refined = ref.reference(variant, channel.ROWS, channel.REFINE)
        ratio = channel._core(
            refined["yc"],
            refined["k"],
            refined["u_tau_mid"],
            ref.VARIANTS[variant]["c_mu"],
        )
        core[variant] = [float(ratio.min()), float(ratio.max())]
    record = {
        "label": args.label,
        "root": str(root),
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "stop": solver.stop_reason,
        "rule_version": solver.rule_version,
        "outer": len(counts["k"]),
        "wall_seconds": wall,
        "k_iterations_mean": float(np.mean(counts["k"])),
        "eps_iterations_mean": float(np.mean(counts["eps"])),
        "fallbacks": int(sum(counts["fallbacks"])),
        "k_min": float(state.k[state.k > 0.0].min()),
        "a_du_over_uw": du,
        "a_dk_over_utau2": dk,
        "a_u_tau_ratio": u_tau / one_d["u_tau_mid"],
        "a_tol": channel.IMPLEMENTATION_TOL,
        "b_core_k_ratio": core,
        "b_tol": channel.MODEL_TOL,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"val016_{args.label}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
