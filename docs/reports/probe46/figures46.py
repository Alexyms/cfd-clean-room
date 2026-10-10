"""The report's figures from the step 6 product records (prompt 46).

Usage:
    python figures46.py fields NAME           three panels: streamlines coloured by speed,
                                              nu_t / nu (log scale), k
    python figures46.py concentration NAME    one panel per class of transport_NAME
    python figures46.py all                   every converged flow record and every
                                              transport record

Figures go beside the report as docs/reports/ecr002_step6_NAME.png and
docs/reports/ecr002_step6_NAME_concentration.png, small PNGs (90 dpi).
The streamlines are drawn on the cell-centred velocity with SOLID cells at
rest; the obstacles are drawn over them. The source square and the sensors
are marked on the concentration panels.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.mesh import SOLID  # noqa: E402

OUT = ROOT / "results" / "builder46"
FIGURES = ROOT / "docs" / "reports"
DPI = 90


def load(name: str) -> tuple[dict, dict]:
    """A record and its arrays."""
    rec = json.loads((OUT / f"{name}.json").read_text())
    return rec, dict(np.load(OUT / f"{name}.npz"))


def grey_bad(name: str) -> matplotlib.colors.Colormap:
    """A colour map whose masked (SOLID) cells draw in the obstacles' grey."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("0.75")
    return cmap


def draw_room(ax: plt.Axes, raw: dict) -> None:
    """The obstacles, the openings and the frame."""
    for o in raw["obstacles"]:
        ax.add_patch(
            Rectangle(
                (o["x_start"], o["y_start"]),
                o["x_end"] - o["x_start"],
                o["y_end"] - o["y_start"],
                facecolor="0.75",
                edgecolor="0.3",
                linewidth=0.6,
                zorder=5,
            )
        )
    for seg in raw["boundaries"].values():
        if seg["type"] == "wall":
            continue
        if seg["location"] in ("bottom", "top"):
            y = 0.0 if seg["location"] == "bottom" else raw["domain"]["height"]
            ax.plot([seg["x_start"], seg["x_end"]], [y, y], color="k", lw=2.5, zorder=6)
        else:
            x = 0.0 if seg["location"] == "left" else raw["domain"]["width"]
            ax.plot([x, x], [seg["y_start"], seg["y_end"]], color="k", lw=2.5, zorder=6)
    ax.set_xlim(0.0, raw["domain"]["width"])
    ax.set_ylim(0.0, raw["domain"]["height"])
    ax.set_aspect("equal")


def fields(args: argparse.Namespace) -> None:
    """The three-panel figure of a converged flow record."""
    rec, a = load(args.name)
    raw = rec["raw"]
    solid = a["cell_type"] == SOLID
    xc, yc = a["xc"], a["yc"]
    u = np.where(solid, 0.0, a["u_c"])
    v = np.where(solid, 0.0, a["v_c"])
    speed = np.hypot(u, v)
    nu = rec["nu_air"]
    ratio = np.ma.masked_where(solid, a["nu_t"] / nu)
    k = np.ma.masked_where(solid, a["k"])
    x_edges = np.concatenate(
        ([0.0], 0.5 * (xc[1:] + xc[:-1]), [raw["domain"]["width"]])
    )
    y_edges = np.concatenate(
        ([0.0], 0.5 * (yc[1:] + yc[:-1]), [raw["domain"]["height"]])
    )

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.5), constrained_layout=True)
    ax = axes[0]
    strm = ax.streamplot(
        xc,
        yc,
        u,
        v,
        color=speed,
        cmap="viridis",
        density=(2.0, 1.0),
        linewidth=0.7,
        arrowsize=0.7,
    )
    fig.colorbar(strm.lines, ax=ax, label="speed (m/s)", shrink=0.9)
    draw_room(ax, raw)
    ax.set_title(
        f"{rec['name']}: streamlines coloured by speed ({rec['outer']:,} outer iterations)"
    )
    ax = axes[1]
    pm = ax.pcolormesh(
        x_edges,
        y_edges,
        ratio,
        norm=LogNorm(vmin=max(float(ratio.min()), 1e-1), vmax=float(ratio.max())),
        cmap=grey_bad("magma"),
        shading="flat",
    )
    fig.colorbar(pm, ax=ax, label="nu_t / nu", shrink=0.9)
    draw_room(ax, raw)
    core = rec["core"]
    ax.set_title(
        f"nu_t / nu (core median {core['nu_t_over_nu_median']:.3g}, "
        f"95th {core['nu_t_over_nu_p95']:.3g})"
    )
    ax = axes[2]
    pm = ax.pcolormesh(x_edges, y_edges, k, cmap=grey_bad("cividis"), shading="flat")
    fig.colorbar(pm, ax=ax, label="k (m^2/s^2)", shrink=0.9)
    draw_room(ax, raw)
    y = rec["y_star"]
    ax.set_title(
        f"k (y+ at the wall nodes: median {y['median']:.3g}, "
        f"{y['share_below_floor']:.0%} below 11.53)"
    )
    for ax in axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    path = FIGURES / f"ecr002_step6_{rec['name']}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def concentration(args: argparse.Namespace) -> None:
    """One panel per class of a transport record, on a log colour scale."""
    rec = json.loads((OUT / f"transport_{args.name}.json").read_text())
    a = dict(np.load(OUT / f"transport_{args.name}.npz"))
    flow, _ = load(args.name)
    raw = flow["raw"]
    solid = a["cell_type"] == SOLID
    xc, yc = a["xc"], a["yc"]
    x_edges = np.concatenate(
        ([0.0], 0.5 * (xc[1:] + xc[:-1]), [raw["domain"]["width"]])
    )
    y_edges = np.concatenate(
        ([0.0], 0.5 * (yc[1:] + yc[:-1]), [raw["domain"]["height"]])
    )
    classes = list(rec["per_class"].items())
    fig, axes = plt.subplots(
        len(classes), 1, figsize=(9.0, 3.6 * len(classes)), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    box = rec["source"]["box"]
    for ax, (k, c) in zip(axes, classes, strict=True):
        field = np.ma.masked_where(solid, a[f"C_{k}"])
        top = float(field.max())
        pm = ax.pcolormesh(
            x_edges,
            y_edges,
            field,
            norm=LogNorm(vmin=top * 1e-4, vmax=top),
            cmap=grey_bad("inferno"),
            shading="flat",
        )
        fig.colorbar(pm, ax=ax, label="C (per m^3)", shrink=0.9)
        draw_room(ax, raw)
        ax.add_patch(
            Rectangle(
                (box[0], box[2]),
                box[1] - box[0],
                box[3] - box[2],
                fill=False,
                edgecolor="cyan",
                linewidth=1.2,
                zorder=7,
            )
        )
        for name, (x, y) in zip(rec["sensors"], rec["sensor_points"], strict=True):
            ax.plot(x, y, marker="o", color="white", markersize=5, zorder=8)
            ax.annotate(
                f"{name}: {c['sensors'][name]:.3g}",
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                color="white",
                fontsize=7,
                zorder=8,
            )
        ax.set_title(
            f"{rec['name']}: {c['diameter'] * 1e6:g} um, {c['stop']} at {c['t_end']:.0f} s; "
            f"largest deposition {c['hotspots'][0]}"
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    path = FIGURES / f"ecr002_step6_{args.name}_concentration.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def everything(_args: argparse.Namespace) -> None:
    """Every converged flow record's fields and every transport record's concentration."""
    for path in sorted(OUT.glob("*.json")):
        stem = path.stem
        if stem.startswith(("transport_", "tables_", "bounded_")) or stem.endswith(
            ("_time", "_smoke")
        ):
            continue
        rec = json.loads(path.read_text())
        if rec.get("stop") == "error_estimate_and_continuity" and "core" in rec:
            fields(argparse.Namespace(name=stem))
    for path in sorted(OUT.glob("transport_*.json")):
        concentration(argparse.Namespace(name=path.stem[len("transport_") :]))


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("fields")
    p.add_argument("name")
    p.set_defaults(func=fields)
    p = sub.add_parser("concentration")
    p.add_argument("name")
    p.set_defaults(func=concentration)
    sub.add_parser("all").set_defaults(func=everything)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
