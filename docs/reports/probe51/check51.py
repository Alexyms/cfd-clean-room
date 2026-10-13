"""Unchanged-by-construction checks of prompt 51 (item 4): the laminar hashes and the Jacobi path's bits.

Usage:
    python check51.py [--root TREE] val CASE --label L    a VAL preset's stop, count and face hash
    python check51.py [--root TREE] jacobi --label L     the Jacobi battery's hash
    python check51.py report --base B --built L          each record of L against B's

``val`` is prompt 50's (docs/reports/probe50/check50.py): step 4's measuring,
docs/reports/probe43/field43.py's ``val`` (the committed solver on the
preset, SHA-256 over u's bytes then v's), imported from TREE so that a
worktree of another commit measures that commit.

``jacobi`` runs ``scalar_scheme.implicit_step`` from TREE, with no method
named, on a fixed battery of 240 systems on the stretched 11 by 7 obstacle
mesh of tests/test_scalar_scheme.py: conductances from 1e-3 to 1e3, steps
from 1e-6 to 1e6 s, one step or one per cell, with and without a sink, with
and without held cells, at tolerances from 1e-6 to 1e-14 and caps from 1 to
5,000 sweeps, so capped and converged solves both appear. It hashes every
returned field's bytes in order, with the sweep count and the convergence
flag of each, read under the name either tree gives them. The transport
solver and k-epsilon's Jacobi solves run this code; the transport gate tests
pass besides (the commit's own run).

The base is measured in a worktree of origin/main at the branch point
(results/builder51/base), the built tree in the branch's checkout at commit
A, on one machine. Records go to results/builder51/ of the main checkout.
"""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
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
CASES = ("val001_80x40", "val001_80x40_stretched", "val002_80x80")


def load_field43(root: Path) -> object:
    """field43 from TREE, its records redirected to OUT."""
    sys.path.insert(0, str(root / "docs" / "reports" / "probe43"))
    field43 = importlib.import_module("field43")
    if Path(field43.__file__).resolve().parents[3] != root:
        raise SystemExit(f"field43 was imported from {field43.__file__}, not {root}")
    field43.OUT = OUT
    field43.probe.OUT = OUT
    return field43


def jacobi(root: Path, label: str) -> None:
    """The battery through TREE's implicit_step; one hash over every result."""
    sys.path.insert(0, str(root))
    scalar_scheme = importlib.import_module("src.scalar_scheme")
    if Path(scalar_scheme.__file__).resolve().parents[1] != root:
        raise SystemExit(f"src was imported from {scalar_scheme.__file__}, not {root}")
    mesh_module = importlib.import_module("src.mesh")
    staggered = importlib.import_module("src.staggered")
    transport_cases = importlib.import_module("validation.transport_cases")
    config = transport_cases.transport_config(
        1.5,
        0.9,
        11,
        7,
        obstacles=[
            {
                "name": "block",
                "x_start": 0.6,
                "x_end": 0.9,
                "y_start": 0.0,
                "y_end": 0.4,
            }
        ],
        mesh={"x": {"stretch_ratio": 1.25}, "y": {"stretch_ratio": 1.15}},
    )
    mesh = mesh_module.Mesh(config)
    solid = mesh.cell_type == mesh_module.SOLID
    volume = np.outer(mesh.dy_cell, mesh.dx_cell)
    inner_u = ~solid[:, :-1] & ~solid[:, 1:]
    inner_v = ~solid[:-1, :] & ~solid[1:, :]
    digest = hashlib.sha256()
    counts = []
    converged = []
    for seed in range(240):
        rng = np.random.default_rng(seed)
        high = 10.0 ** rng.uniform(-3.0, 3.0)
        g_u = np.zeros(staggered.u_shape(mesh))
        g_v = np.zeros(staggered.v_shape(mesh))
        g_u[:, 1:-1] = np.where(inner_u, rng.uniform(0.0, high, inner_u.shape), 0.0)
        g_v[1:-1, :] = np.where(inner_v, rng.uniform(0.0, high, inner_v.shape), 0.0)
        c_star = np.where(solid, 0.0, rng.uniform(0.0, 1.0, size=solid.shape))
        dt = 10.0 ** rng.uniform(-6.0, 6.0)
        if seed % 2:
            dt = dt * rng.uniform(0.1, 10.0, size=solid.shape)
        diagonal = np.zeros(solid.shape)
        if seed % 3:
            diagonal = np.where(solid, 0.0, rng.uniform(0.0, high, size=solid.shape))
        held = None
        if seed % 4 == 0:
            held = np.zeros_like(solid)
            held[-1, :] = True
            held[2, 1] = True
            c_star[held] = rng.uniform(1.0, 4.0)
        tol = 10.0 ** -rng.uniform(6.0, 14.0)
        cap = int(rng.choice([1, 7, 50, 500, 5000]))
        result = scalar_scheme.implicit_step(
            c_star, volume, dt, g_u, g_v, diagonal, solid, tol, cap, held=held
        )
        digest.update(np.ascontiguousarray(result.field, dtype="<f8").tobytes())
        count = getattr(result, "iterations", None)
        counts.append(int(result.sweeps if count is None else count))
        converged.append(bool(result.converged))
    digest.update(json.dumps([counts, converged]).encode())
    record = {
        "label": label,
        "root": str(root),
        "systems": len(counts),
        "converged": sum(converged),
        "capped": len(converged) - sum(converged),
        "sweeps_total": sum(counts),
        "hash": digest.hexdigest(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"jacobi_{label}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


def report(base: str, built: str) -> None:
    """Each laminar case's stop, count and face hash, and the battery's hash, BUILT against BASE."""
    summary = []
    for case in CASES:
        paths = {lab: OUT / f"val_{case}_{lab}.json" for lab in (base, built)}
        if not all(p.exists() for p in paths.values()):
            summary.append({"case": case, "present": False})
            continue
        a, b = (json.loads(paths[lab].read_text()) for lab in (base, built))
        summary.append(
            {
                "case": case,
                "stop": (a["stop"], b["stop"]),
                "outer": (a["outer"], b["outer"]),
                "face_hash": (a["face_hash"][:16], b["face_hash"][:16]),
                "equal": a["stop"] == b["stop"]
                and a["outer"] == b["outer"]
                and a["face_hash"] == b["face_hash"],
            }
        )
    a, b = (
        json.loads((OUT / f"jacobi_{lab}.json").read_text()) for lab in (base, built)
    )
    summary.append(
        {
            "case": "jacobi_battery",
            "systems": (a["systems"], b["systems"]),
            "capped": (a["capped"], b["capped"]),
            "sweeps_total": (a["sweeps_total"], b["sweeps_total"]),
            "hash": (a["hash"][:16], b["hash"][:16]),
            "equal": a["hash"] == b["hash"],
        }
    )
    text = json.dumps(summary, indent=1)
    (OUT / f"report_{base}_{built}.json").write_text(text)
    print(text)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(HERE))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("val")
    p.add_argument("case", choices=CASES)
    p.add_argument("--label", required=True)
    p = sub.add_parser("jacobi")
    p.add_argument("--label", required=True)
    p = sub.add_parser("report")
    p.add_argument("--base", required=True)
    p.add_argument("--built", required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "report":
        report(args.base, args.built)
        return
    root = Path(args.root).resolve()
    if args.command == "jacobi":
        jacobi(root, args.label)
        return
    field43 = load_field43(root)
    field43.val(argparse.Namespace(case=args.case, zero=False, label=args.label))


if __name__ == "__main__":
    main()
