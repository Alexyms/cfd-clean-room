"""Item 7 of prompt 50: prompt 48's standard upwind product rows under rule version 5.

Usage:
    python product50.py run GRID           GRID 80x30 or 160x60
    python product50.py compare GRID       the stopped faces against prompt 48's

``run`` is prompt 48's row (docs/reports/probe48/coupled48.py: prompt 47's
``control`` arm, the committed StaggeredSolver on the product configuration
regridded with prompt 46's settings, the key absent so k and eps are advected
by the default upwind) with its own settings, under the committed rule. The
one substitution is the recording rule, rule50.RecordingRule50 in place of
prompt 46's RecordingRule, which takes the per-step scale rule version 5
passes; its decision is the committed rule's. The record is
standard_GRID_upwind under results/builder50/. ``compare`` reads it beside
prompt 48's record of the same name (results/builder48/) and writes
compare_GRID.json: the stops, the outer counts, the scale at the stop
against the inlet scale prompt 48's row ran under, and the largest
difference of the stopped face velocities over the velocity scale (the
largest prescribed boundary velocity).
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe48"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe47"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))

import coupled46  # noqa: E402
import coupled47  # noqa: E402
import coupled48  # noqa: E402
import rule50  # noqa: E402

OUT = ROOT / "results" / "builder50"
BASE = ROOT / "results" / "builder48"
GRIDS = ("80x30", "160x60")


def run(args: argparse.Namespace) -> None:
    """Prompt 48's row on GRID, the recording rule taking version 5's scale."""
    for module in (coupled46, coupled47, coupled48):
        module.OUT = OUT
    coupled47.RecordingRule = rule50.RecordingRule50
    coupled48.run(
        argparse.Namespace(
            grid=args.grid, variant="standard", scheme="", cap=0, log_every=100
        )
    )


def compare(args: argparse.Namespace) -> None:
    """The new row against prompt 48's: stop, count, scale and face difference."""
    name = f"standard_{args.grid}_upwind"
    new, old = (json.loads((d / f"{name}.json").read_text()) for d in (OUT, BASE))
    a, b = (np.load(d / f"{name}.npz") for d in (OUT, BASE))
    scale = new["velocity_scale"]
    assert scale == old["velocity_scale"]
    du = float(np.abs(a["u_faces"] - b["u_faces"]).max())
    dv = float(np.abs(a["v_faces"] - b["v_faces"]).max())
    nu = new["nu_air"]
    summary = {
        "grid": args.grid,
        "commit": new["commit"],
        "stop": (old["stop"], new["stop"]),
        "outer": (old["outer"], new["outer"]),
        "outer_change": (new["outer"] - old["outer"]) / old["outer"],
        "rule_version": (old["rule_version"], new["rule_version"]),
        "nu_scale_over_nu": (old["nu_scale"] / nu, new["nu_scale"] / nu),
        "readings_at_stop": (old["readings_at_stop"], new["readings_at_stop"]),
        "from_when": (old["from_when"], new["from_when"]),
        "face_hash": (old["face_hash"][:16], new["face_hash"][:16]),
        "velocity_scale": scale,
        "max_face_difference_over_scale": max(du, dv) / scale,
        "cap_hits": (old["cap_hits"], new["cap_hits"]),
        "positivity": (old["positivity"], new["positivity"]),
    }
    text = json.dumps(summary, indent=1)
    (OUT / f"compare_{args.grid}.json").write_text(text)
    print(text)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode, func in (("run", run), ("compare", compare)):
        p = sub.add_parser(mode)
        p.add_argument("grid", choices=GRIDS)
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
