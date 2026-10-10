"""The report's figures from the prompt 47 records, with the room's boundaries marked.

Usage:
    python figures47.py fields NAME           three panels: streamlines coloured by speed,
                                              nu_t / nu (log scale), k
    python figures47.py concentration LABEL   the two classes of transport record LABEL
                                              (transport_NAME_SOURCE) and a deposition panel
    python figures47.py all                   every converged flow record and every
                                              transport record

Figures go beside the report as docs/reports/ecr002_step6_<record>.png, small
PNGs (90 dpi). Every panel marks the supply (a blue bar along the ceiling
with downward arrows), the returns (red bars on the floor), the hood (an
orange bar on the right wall with an outward arrow), the door (a bar on the
left wall over its height), the obstacles (grey), and the sensors (labelled);
the concentration panels mark the source square too. The deposition panel
draws each 0.2 m segment's deposition as a bar standing on its surface, one
colour per class, its length proportional to the segment's rate, with the
scale in the title.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.mesh import SOLID  # noqa: E402

OUT = ROOT / "results" / "builder47"
FIGURES = ROOT / "docs" / "reports"
DPI = 90
SUPPLY_COLOUR = "tab:blue"
RETURN_COLOUR = "tab:red"
HOOD_COLOUR = "tab:orange"
DOOR_COLOUR = "0.2"
CLASS_COLOURS = ("tab:cyan", "gold")
BAR_LENGTH = 0.5  # m, the bar of the largest segment
SEGMENT_KEY = re.compile(r"^(?P<surface>.+) \| (?P<axis>[xy]) (?P<start>[-\d.]+)$")


def load(name: str) -> tuple[dict, dict]:
    """A record and its arrays."""
    rec = json.loads((OUT / f"{name}.json").read_text())
    return rec, dict(np.load(OUT / f"{name}.npz"))


def grey_bad(name: str) -> matplotlib.colors.Colormap:
    """A colour map whose masked (SOLID) cells draw in the obstacles' grey."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("0.75")
    return cmap


def edges(xc: np.ndarray, yc: np.ndarray, raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cell edges from the centres, for pcolormesh."""
    x_edges = np.concatenate(
        ([0.0], 0.5 * (xc[1:] + xc[:-1]), [raw["domain"]["width"]])
    )
    y_edges = np.concatenate(
        ([0.0], 0.5 * (yc[1:] + yc[:-1]), [raw["domain"]["height"]])
    )
    return x_edges, y_edges


def draw_room(
    ax: plt.Axes, raw: dict, sensors: bool = True, source: tuple | None = None
) -> None:
    """The obstacles, the openings with their markers, the door, the sensors, the source."""
    width, height = raw["domain"]["width"], raw["domain"]["height"]
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
        ax.annotate(
            o["name"],
            (0.5 * (o["x_start"] + o["x_end"]), 0.5 * (o["y_start"] + o["y_end"])),
            ha="center",
            va="center",
            fontsize=6,
            color="0.25",
            zorder=6,
        )
    bar = 0.06 * height
    for name, seg in raw["boundaries"].items():
        if seg["type"] == "velocity_inlet" and seg["location"] == "top":
            x0, x1 = seg["x_start"], seg["x_end"]
            ax.add_patch(
                Rectangle(
                    (x0, height - bar), x1 - x0, bar, color=SUPPLY_COLOUR, zorder=6
                )
            )
            for x in np.linspace(x0, x1, 9)[1:-1]:
                ax.add_patch(
                    FancyArrowPatch(
                        (x, height - bar),
                        (x, height - 3.5 * bar),
                        arrowstyle="-|>",
                        mutation_scale=7,
                        color=SUPPLY_COLOUR,
                        linewidth=0.8,
                        zorder=6,
                    )
                )
            ax.annotate(
                "supply",
                (0.5 * (x0 + x1), height - 0.5 * bar),
                ha="center",
                va="center",
                fontsize=6,
                color="white",
                zorder=7,
            )
        elif seg["type"] == "fixed_flow_outlet" and seg["location"] == "bottom":
            x0, x1 = seg["x_start"], seg["x_end"]
            ax.add_patch(
                Rectangle((x0, 0.0), x1 - x0, bar, color=RETURN_COLOUR, zorder=6)
            )
            ax.annotate(
                f"R{name[-1]}",
                (0.5 * (x0 + x1), 0.5 * bar),
                ha="center",
                va="center",
                fontsize=6,
                color="white",
                zorder=7,
            )
        elif seg["type"] == "fixed_flow_outlet" and seg["location"] == "right":
            y0, y1 = seg["y_start"], seg["y_end"]
            ax.add_patch(
                Rectangle((width - bar, y0), bar, y1 - y0, color=HOOD_COLOUR, zorder=6)
            )
            ax.add_patch(
                FancyArrowPatch(
                    (width - 3.5 * bar, 0.5 * (y0 + y1)),
                    (width - bar, 0.5 * (y0 + y1)),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    color=HOOD_COLOUR,
                    linewidth=1.2,
                    zorder=6,
                )
            )
            ax.annotate(
                "hood",
                (width - 3.6 * bar, 0.5 * (y0 + y1)),
                ha="right",
                va="center",
                fontsize=6,
                color=HOOD_COLOUR,
                zorder=7,
            )
        elif seg["type"] == "wall" and seg["location"] == "left":
            y0, y1 = seg["y_start"], seg["y_end"]
            ax.plot([0.0, 0.0], [y0, y1], color=DOOR_COLOUR, lw=4, zorder=6)
            ax.annotate(
                name,
                (0.3 * bar, 0.5 * (y0 + y1)),
                rotation=90,
                ha="left",
                va="center",
                fontsize=6,
                color=DOOR_COLOUR,
                zorder=7,
            )
    if sensors:
        for s in raw["sensors"]:
            ax.plot(
                s["x"],
                s["y"],
                marker="o",
                color="white",
                markeredgecolor="k",
                markersize=5,
                zorder=8,
            )
            ax.annotate(
                s["name"],
                (s["x"], s["y"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
                color="k",
                zorder=8,
                bbox={"facecolor": "white", "alpha": 0.6, "linewidth": 0, "pad": 1},
            )
    if source is not None:
        x0, x1, y0, y1 = source
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="lime",
                linewidth=1.5,
                zorder=9,
            )
        )
        ax.annotate(
            "source",
            (x1, y1),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=6,
            color="lime",
            zorder=9,
        )
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


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
    x_edges, y_edges = edges(xc, yc, raw)

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
    path = FIGURES / f"ecr002_step6_{rec['name']}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def segment_bar(
    key: str, length: float, raw: dict, mesh_x: np.ndarray, segment: float
) -> tuple[float, float, float, float] | None:
    """The rectangle (x, y, w, h) of one segment's bar standing on its surface.

    Floors and obstacle tops: the bar rises from the surface; the ceiling:
    it hangs; walls and obstacle sides: it points into the room. A floor
    segment stands on y = 0 over its x bin; an obstacle top or side on the
    obstacle the key names.
    """
    m = SEGMENT_KEY.match(key)
    if m is None:
        return None
    surface, start = m["surface"], float(m["start"])
    width, height = raw["domain"]["width"], raw["domain"]["height"]
    obstacles = {o["name"]: o for o in raw["obstacles"]}
    if surface == "floor":
        return (start, 0.0, segment, length)
    if surface == "ceiling":
        return (start, height - length, segment, length)
    if surface == "left wall":
        return (0.0, start, length, segment)
    if surface == "right wall":
        return (width - length, start, length, segment)
    for name, o in obstacles.items():
        if surface == f"{name} top":
            return (start, o["y_end"], segment, length)
        if surface == f"{name} underside":
            return (start, o["y_start"] - length, segment, length)
        if surface == f"{name} west":
            return (o["x_start"] - length, start, length, segment)
        if surface == f"{name} east":
            return (o["x_end"], start, length, segment)
    return None


def concentration(args: argparse.Namespace) -> None:
    """One panel per class on a log colour scale, and the deposition panel."""
    rec = json.loads((OUT / f"{args.label}.json").read_text())
    a = dict(np.load(OUT / f"{args.label}.npz"))
    flow, _ = load(rec["flow_record"])
    raw = flow["raw"]
    solid = a["cell_type"] == SOLID
    xc, yc = a["xc"], a["yc"]
    x_edges, y_edges = edges(xc, yc, raw)
    classes = list(rec["per_class"].items())
    box = tuple(rec["source"]["box"])
    fig, axes = plt.subplots(
        len(classes) + 1,
        1,
        figsize=(9.0, 3.6 * (len(classes) + 1)),
        constrained_layout=True,
    )
    for ax, (k, c) in zip(axes[:-1], classes, strict=True):
        field = np.ma.masked_where(solid, a[f"C_{k}"])
        top = float(field.max())
        pm = ax.pcolormesh(
            x_edges,
            y_edges,
            field,
            norm=LogNorm(vmin=top * 1e-6, vmax=top),
            cmap=grey_bad("inferno"),
            shading="flat",
        )
        fig.colorbar(pm, ax=ax, label="C (per m^3)", shrink=0.9)
        draw_room(ax, raw, source=box)
        for name, (x, y) in zip(rec["sensors"], rec["sensor_points"], strict=True):
            value = c["sensors"][name]
            text = f"{value:.2g}" if name in c["sensors_above_floor"] else "below floor"
            ax.annotate(
                text,
                (x, y),
                textcoords="offset points",
                xytext=(4, -9),
                fontsize=6,
                color="white",
                zorder=8,
            )
        ax.set_title(
            f"{c['diameter'] * 1e6:g} um: {c['stop']} at {c['t_end']:.0f} s; colour floor "
            f"1e-6 of the peak; largest deposition {c['hotspots'][0]}",
            fontsize=9,
        )
    ax = axes[-1]
    draw_room(ax, raw, source=box)
    largest = max(max(c["segments"].values()) for _, c in classes if c["segments"])
    handles: list[Rectangle] = []
    for (_k, c), colour, shift in zip(
        classes, CLASS_COLOURS, (0.0, 0.5 * rec["segment"]), strict=True
    ):
        for key, value in c["segments"].items():
            # Bars under a thousandth of the largest would draw as hairlines.
            if value <= 1e-3 * largest:
                continue
            geom = segment_bar(
                key, BAR_LENGTH * value / largest, raw, xc, rec["segment"]
            )
            if geom is None:
                continue
            x, y, w, h = geom
            # The two classes share each bin; each takes half its width.
            if w == rec["segment"]:
                ax.add_patch(
                    Rectangle(
                        (x + shift, y), 0.5 * w, h, color=colour, alpha=0.85, zorder=10
                    )
                )
            else:
                ax.add_patch(
                    Rectangle(
                        (x, y + shift), w, 0.5 * h, color=colour, alpha=0.85, zorder=10
                    )
                )
        handles.append(
            Rectangle((0, 0), 1, 1, color=colour, label=f"{c['diameter'] * 1e6:g} um")
        )
    # Above the supply bar, which would otherwise cover the legend's text.
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9).set_zorder(
        20
    )
    ax.set_title(
        f"deposition per 0.2 m segment, as bars standing on their surfaces\n"
        f"scale: a {BAR_LENGTH} m bar is {largest:.3g} particles per s per m depth "
        f"(the largest segment); bars below 1e-3 of it omitted",
        fontsize=8,
    )
    fig.suptitle(rec["name"], fontsize=10)
    path = (
        FIGURES / f"ecr002_step6_{rec['name'][len('transport_') :]}_concentration.png"
    )
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
        if stem in ("rooms", "sources", "checks"):
            continue
        rec = json.loads(path.read_text())
        if rec.get("stop") == "error_estimate_and_continuity" and "core" in rec:
            fields(argparse.Namespace(name=stem))
    for path in sorted(OUT.glob("transport_*.json")):
        # A tagged record (the Courant check row) is compared in a table,
        # not drawn.
        if path.stem.endswith("_smoke") or re.search(r"_S\d+m?_", path.stem):
            continue
        concentration(argparse.Namespace(label=path.stem))


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("fields")
    p.add_argument("name")
    p.set_defaults(func=fields)
    p = sub.add_parser("concentration")
    p.add_argument("label")
    p.set_defaults(func=concentration)
    sub.add_parser("all").set_defaults(func=everything)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
