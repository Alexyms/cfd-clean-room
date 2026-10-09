"""Prediction (e) of prompt 43, characterised (prompt 43b item 4): the 40x15 ladder at Re 895.

Usage:
    python tail43.py history          residual histories at A, B1 and B2, and B2's tail
    python tail43.py diff             the B1 and B2 converged fields compared
    python tail43.py corner RUNG      the ladder with only the corner QUICK zeroing removed

`history` and `diff` read the records field43.py wrote at commits A, B1 and B2
(results/builder43/ladder_{A,B1,B2}_895.json and .npz). `corner` runs the
committed solver on the committed product configuration, the ladder's keys,
with one change made here and not under src/: the predictor is a subclass
whose `_deferred_correction` is the committed method's text with the two
lines that zero the QUICK correction at obstacle faces removed. At a full
wall face the mass flux is zero and those lines change nothing; at an
obstacle corner they carry the corner rule's QUICK half. The rest of B2 (the
boundary form's far nodes) stays. Records go to results/builder43/.
"""

import argparse
import inspect
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import field43  # noqa: E402

from src import momentum  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402

OUT = field43.OUT
CORNER_LINES = "        if obstacles:\n            dq_t[o.walls.face] = 0.0\n"
TAIL_MARKS = (1, 25, 50, 100, 200, 300, 375, 391, 450, 500, 550, 600, 650, 700, 728)


def load(label: str) -> dict:
    """A ladder record at Re 895 from field43.py."""
    return json.loads((OUT / f"ladder_{label}_895.json").read_text())


def history(_args: argparse.Namespace) -> None:
    """The three residual histories side by side, and the shape of B2's tail."""
    runs = {label: np.array(load(label)["residual"]) for label in ("A", "B1", "B2")}
    table = []
    for n in TAIL_MARKS:
        row = {"outer": n}
        for label, res in runs.items():
            row[label] = float(res[n - 1]) if n <= res.size else None
        table.append(row)
    b2 = runs["B2"]
    tol = 1e-6
    tail = np.log10(b2[390:])
    steps = np.diff(tail)
    # A least-squares slope of log10(residual) per outer iteration over the
    # tail, and the number of reversals of its direction.
    slope = float(np.polyfit(np.arange(tail.size), tail, 1)[0])
    reversals = int(np.sum(np.sign(steps[1:]) != np.sign(steps[:-1])))
    first_below = {
        label: int(np.argmax(res < tol)) + 1 if np.any(res < tol) else None
        for label, res in runs.items()
    }
    crossings = {
        f"first outer below {level:g}": {
            label: int(np.argmax(res < level)) + 1 if np.any(res < level) else None
            for label, res in runs.items()
        }
        for level in (1e-3, 1e-4, 1e-5, 2e-6)
    }
    record = {
        "table": table,
        "stops": {label: int(res.size) for label, res in runs.items()},
        "first_below_tol": first_below,
        "crossings": crossings,
        "b2_at_a_stop": float(b2[runs["A"].size - 1]),
        "b2_at_b1_stop": float(b2[runs["B1"].size - 1]),
        "b2_tail_from_391": {
            "log10_slope_per_outer": slope,
            "outers_per_decade": float(-1.0 / slope) if slope < 0 else None,
            "direction_reversals": reversals,
            "steps": int(steps.size),
            "max_over_tail": float(b2[390:].max()),
            "min_over_tail": float(b2[390:].min()),
        },
    }
    field43.write("tail_history", record)


def beside_obstacle(mesh: Mesh) -> np.ndarray:
    """Non-SOLID cells with a SOLID cell across one of their four faces."""
    solid = mesh.cell_type == SOLID
    near = np.zeros_like(solid)
    near[1:, :] |= solid[:-1, :]
    near[:-1, :] |= solid[1:, :]
    near[:, 1:] |= solid[:, :-1]
    near[:, :-1] |= solid[:, 1:]
    return near & ~solid


def diff(_args: argparse.Namespace) -> None:
    """Where the B1 and B2 converged fields differ, beside obstacles against the rest."""
    a = np.load(OUT / "ladder_B1_895.npz")
    b = np.load(OUT / "ladder_B2_895.npz")
    raw = field43.probe.product_raw(
        40, 15, field43.probe.LADDER_FACTORS["895"], 1, 1e-8, None
    )
    mesh = Mesh(field43.SimConfig.from_dict(raw))
    solid = mesh.cell_type == SOLID
    speed = np.hypot(b["u_c"] - a["u_c"], b["v_c"] - a["v_c"])
    speed[solid] = 0.0
    near = beside_obstacle(mesh)
    rest = ~near & ~solid
    j, i = np.unravel_index(int(np.argmax(speed)), speed.shape)
    squares = speed**2
    record = {
        "largest_velocity_difference": float(speed.max()),
        "at_cell": [int(j), int(i)],
        "at_xy": [float(mesh.xc[i]), float(mesh.yc[j])],
        "that_cell_beside_an_obstacle": bool(near[j, i]),
        "largest_beside_obstacles": float(speed[near].max()),
        "largest_elsewhere": float(speed[rest].max()),
        "rms_beside_obstacles": float(np.sqrt(squares[near].mean())),
        "rms_elsewhere": float(np.sqrt(squares[rest].mean())),
        "share_of_squared_difference_beside_obstacles": float(
            squares[near].sum() / squares[~solid].sum()
        ),
        "cells_beside_obstacles": int(near.sum()),
        "cells_elsewhere": int(rest.sum()),
        "largest_speed_b2": float(np.hypot(b["u_c"], b["v_c"])[~solid].max()),
    }
    field43.write("tail_diff", record)


def corner_free_class() -> type:
    """MomentumPredictor with the obstacle faces' QUICK zeroing taken out, nothing else."""
    source = inspect.getsource(momentum.MomentumPredictor._deferred_correction)
    if source.count(CORNER_LINES) != 1:
        raise ValueError(
            "the committed _deferred_correction no longer has the corner lines"
        )
    changed = textwrap.dedent(source.replace(CORNER_LINES, "", 1))
    namespace = dict(vars(momentum))
    exec(compile(changed, "<corner-free _deferred_correction>", "exec"), namespace)
    return type(
        "CornerFreePredictor",
        (momentum.MomentumPredictor,),
        {"_deferred_correction": namespace["_deferred_correction"]},
    )


def corner(args: argparse.Namespace) -> None:
    """One ladder rung with the corner zeroing removed."""
    factor = field43.probe.LADDER_FACTORS[args.rung]
    room = field43.built_room(
        f"ladder_cornerfree_{args.rung}", 40, 15, factor, 3000, 1e-8, 1, None
    )
    cls = corner_free_class()
    room.solver._predictor = cls(room.mesh, room.cfg, room.boundary)
    room.meta["arm"] = "built43_corner_quick_zeroing_removed"
    field43.probe.run(room, log_every=50)


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("history").set_defaults(func=history)
    sub.add_parser("diff").set_defaults(func=diff)
    p = sub.add_parser("corner")
    p.add_argument("rung", choices=sorted(field43.probe.LADDER_FACTORS))
    p.set_defaults(func=corner)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
