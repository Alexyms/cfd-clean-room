"""Item 7 of prompt 54: prompt 51's product and VAL-016 drivers under the default or the diagonal.

Usage:
    python inject54.py --root TREE [--preconditioner P] product GRID LABEL
    python inject54.py --root TREE [--preconditioner P] val016 LABEL

GRID is 80x30 or 160x60; P is ``default`` (nothing injected: the keys the
drivers build carry no preconditioner, so they run the default, multigrid
since prompt 54) or ``jacobi``. For ``jacobi`` it imports TREE's
``src.config`` and wraps ``SimConfig._validate_and_load`` so that every
configuration built in this process, from a file or a mapping, carries
``solver.pressure_preconditioner: jacobi`` and passes the committed
validation with it; then runs prompt 51's driver
(docs/reports/probe51/product51.py ``run``, or val016_51.py) from this
checkout unchanged, its records redirected to results/builder54/ of the main
checkout, which also gain the preconditioner, the count of configurations
built under it, and every solver's pressure cap hits and stage seconds (the
committed solver class is wrapped to keep its instances). The drivers build the product row of prompt 48 (standard upwind)
and VAL-016's measurement exactly as they did for prompt 51, so the records
compare with prompt 51's ``cg`` records, the Jacobi solve on the same code, and
with prompt 53's Jacobi reruns.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

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
OUT = MAIN / "results" / "builder54"
PROBE51 = HERE / "docs" / "reports" / "probe51"


INJECTED = [0]
SOLVERS: list = []


def track(root: Path) -> None:
    """Keep the committed solver's instances without touching any configuration."""
    sys.path.insert(0, str(root))
    config_module = importlib.import_module("src.config")
    if Path(config_module.__file__).resolve().parents[1] != root:
        raise SystemExit(f"src was imported from {config_module.__file__}, not {root}")
    solver_module = importlib.import_module("src.solver_staggered")
    committed = solver_module.StaggeredSolver

    class KeptSolver(committed):  # type: ignore[misc, valid-type]
        """The committed solver, kept so its cap count and stage times can be read."""

        def __init__(self, *a: object, **k: object) -> None:
            super().__init__(*a, **k)
            SOLVERS.append(self)

    solver_module.StaggeredSolver = KeptSolver


def inject(root: Path, preconditioner: str) -> None:
    """Every SimConfig built in this process gets the preconditioner key; INJECTED counts them."""
    sys.path.insert(0, str(root))
    config_module = importlib.import_module("src.config")
    if Path(config_module.__file__).resolve().parents[1] != root:
        raise SystemExit(f"src was imported from {config_module.__file__}, not {root}")
    original = config_module.SimConfig._validate_and_load

    def with_key(self: object, raw: dict) -> None:
        raw = dict(raw)
        raw["solver"] = {**raw["solver"], "pressure_preconditioner": preconditioner}
        original(self, raw)
        INJECTED[0] += 1

    config_module.SimConfig._validate_and_load = with_key
    solver_module = importlib.import_module("src.solver_staggered")
    committed = solver_module.StaggeredSolver

    class KeptSolver(committed):  # type: ignore[misc, valid-type]
        """The committed solver, kept so its cap count and stage times can be read."""

        def __init__(self, *a: object, **k: object) -> None:
            super().__init__(*a, **k)
            SOLVERS.append(self)

    solver_module.StaggeredSolver = KeptSolver


def main() -> None:
    """Inject, then run the named driver."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--preconditioner", choices=("default", "jacobi"), default="default"
    )
    sub = parser.add_subparsers(dest="driver", required=True)
    p = sub.add_parser("product")
    p.add_argument("grid", choices=("80x30", "160x60"))
    p.add_argument("label")
    p = sub.add_parser("val016")
    p.add_argument("label")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.preconditioner == "default":
        track(root)
    else:
        inject(root, args.preconditioner)
    sys.path.insert(0, str(PROBE51))
    if args.driver == "product":
        product51 = importlib.import_module("product51")
        product51.OUT = OUT
        product51.run(
            argparse.Namespace(grid=args.grid, label=args.label, root=str(root))
        )
    else:
        val016_51 = importlib.import_module("val016_51")
        val016_51.OUT = OUT
        sys.argv = ["val016_51.py", args.label, "--root", str(root)]
        val016_51.main()
    name = (
        f"product_{args.grid}_{args.label}"
        if args.driver == "product"
        else f"val016_{args.label}"
    )
    path = OUT / f"{name}.json"
    record = json.loads(path.read_text())
    record["preconditioner"] = args.preconditioner
    record["configs_injected"] = INJECTED[0]
    record["solvers"] = len(SOLVERS)
    record["pressure_cap_hits"] = sum(solver.pressure_cap_hits for solver in SOLVERS)
    record["stage_seconds"] = [dict(solver.stage_seconds) for solver in SOLVERS]
    path.write_text(json.dumps(record))
    print(
        f"{name}: {INJECTED[0]} configurations under {args.preconditioner}", flush=True
    )


if __name__ == "__main__":
    main()
