"""Item 6 of prompt 53: capture the pressure systems the corrector is handed.

Usage:
    python capture53.py CASE [--root TREE] [--late N]

CASE is one of CASES below. The case is built as the harness builds it
(``load_preset``, then Mesh, StaggeredBoundary and StaggeredSolver) or, for
the product room, as prompt 48's standard upwind row (prompt 47's
``coupled47.product_raw`` with the control arm), from TREE (default this
checkout), and solved from rest under its committed settings, the default
Jacobi preconditioner included. ``PressureCorrector.correct`` is wrapped
so that, before the committed correction runs, the p' system of outer
iteration 50 and of outer iteration N (default 2,000) is saved, built by the
corrector's own public ``coefficients`` and ``mass_imbalance``: the five
coefficient arrays, b, the pin, the flux scale, ``pressure_rtol`` and the
cap. The run ends after the late capture. A run that stops before N saves
the system of its last outer iteration instead, the converged state. The
records are results/builder53/systems/CASE_early.npz and CASE_late.npz of
the main checkout. One BLAS thread.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
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
OUT = MAIN / "results" / "builder53" / "systems"
EARLY = 50
PRODUCT_CAP = 10000
# Case name: ("preset", id) or ("product", grid).
CASES: dict[str, tuple[str, str]] = {
    "val001": ("preset", "val001_80x40"),
    "cavity": ("preset", "val002_80x80"),
    "product160": ("product", "160x60"),
    "product320": ("product", "320x120"),
    "annex": ("preset", "val017_216x72"),
    "step": ("preset", "val019_656x60"),
    # Development systems, outside the measurement set: nu and the weight
    # were chosen on these (the report, section 2).
    "product80": ("product", "80x30"),
    "step410": ("preset", "val019_410x40"),
}


class CaptureDoneError(Exception):
    """Raised after the late capture to end the run."""


def commit(root: Path) -> str:
    """The commit checked out at TREE."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def build_config(root: Path, case: str) -> object:
    """The case's committed configuration, from TREE."""
    kind, name = CASES[case]
    if kind == "preset":
        cases = importlib.import_module("validation.cases")
        return cases.load_preset(name)
    sys.path.insert(0, str(root / "docs" / "reports" / "probe47"))
    coupled47 = importlib.import_module("coupled47")
    if Path(coupled47.__file__).resolve().parents[3] != root:
        raise SystemExit(f"coupled47 was imported from {coupled47.__file__}")
    config_module = importlib.import_module("src.config")
    raw = coupled47.product_raw(name, "standard", "control", PRODUCT_CAP)
    return config_module.SimConfig.from_dict(raw)


def save(corrector: object, prediction: object, path: Path, meta: dict) -> None:
    """The p' system of one correction, from the corrector's public methods."""
    c = corrector.coefficients(prediction.a_p_u, prediction.a_p_v)
    b = corrector.mass_imbalance(prediction.u_star, prediction.v_star)
    np.savez_compressed(
        path,
        a_p=c.a_p,
        a_e=c.a_e,
        a_w=c.a_w,
        a_n=c.a_n,
        a_s=c.a_s,
        b=b,
        needs_pin=corrector.needs_pin,
        pin_cell=np.array(corrector.pin_cell),
        flux_scale=corrector.flux_scale,
        rtol=corrector._rtol,
        max_iter=corrector._max_iter,
        meta=json.dumps(meta),
    )


def main() -> None:
    """Run CASE from rest and save its two systems."""
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--root", default=str(HERE))
    parser.add_argument("--late", type=int, default=2000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    pressure = importlib.import_module("src.pressure")
    if Path(pressure.__file__).resolve().parents[1] != root:
        raise SystemExit(f"src was imported from {pressure.__file__}, not {root}")
    mesh_module = importlib.import_module("src.mesh")
    boundary_module = importlib.import_module("src.boundary_staggered")
    solver_module = importlib.import_module("src.solver_staggered")

    config = build_config(root, args.case)
    mesh = mesh_module.Mesh(config)
    boundary = boundary_module.StaggeredBoundary(mesh, config)
    solver = solver_module.StaggeredSolver(mesh, config, boundary)
    OUT.mkdir(parents=True, exist_ok=True)
    tree = commit(root)
    state: dict = {"index": 0, "last": None, "iterations": []}
    original = pressure.PressureCorrector.correct

    def meta(index: int, tag: str) -> dict:
        return {
            "case": args.case,
            "outer": index,
            "tag": tag,
            "commit": tree,
            "grid": [int(mesh.cell_type.shape[1]), int(mesh.cell_type.shape[0])],
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def correct(self: object, prediction: object, p: np.ndarray) -> object:
        index = state["index"]
        state["index"] += 1
        state["last"] = (self, prediction, index)
        if index == EARLY:
            save(self, prediction, OUT / f"{args.case}_early.npz", meta(index, "early"))
        if index == args.late:
            save(self, prediction, OUT / f"{args.case}_late.npz", meta(index, "late"))
            raise CaptureDoneError
        result = original(self, prediction, p)
        state["iterations"].append(int(result.iterations))
        return result

    pressure.PressureCorrector.correct = correct
    t0 = time.perf_counter()
    try:
        solver.solve_steady()
        corrector, prediction, index = state["last"]
        save(corrector, prediction, OUT / f"{args.case}_late.npz", meta(index, "last"))
        ending = f"stopped ({solver.stop_reason}) at {index}"
    except CaptureDoneError:
        ending = f"captured at {args.late}"
    iterations = state["iterations"]
    print(
        f"{args.case}: {ending}, {time.perf_counter() - t0:.0f} s, Jacobi-PCG "
        f"iterations at outer {EARLY} {iterations[EARLY] if len(iterations) > EARLY else None}, "
        f"last {iterations[-1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
