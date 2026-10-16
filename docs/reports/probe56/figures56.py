"""The report's figures from the prompt 56 records.

Usage:
    python figures56.py fields NAME           prompt 47's three panels of a converged flow
                                              record, as docs/reports/ecr002_step8_NAME.png
    python figures56.py concentration LABEL   prompt 47's concentration figure of a transport
                                              record, under results/builder56/figures/
    python figures56.py compare CLASS         the comparison figure of one class (2 or 4):
                                              the three layouts on 320x120 at Sc_t 0.7
    python figures56.py all                   every converged flow record, every transport
                                              record, both comparison figures

The flow and concentration figures are prompt 47's drawing code
(docs/reports/probe47/figures47.py `fields` and `concentration`, imported),
drawn into results/builder56/figures/ and the flow figures copied beside
the report under this step's names; the concentration figures of the 36
marches stay under results, since the comparison figures are the ones the
report reads. The comparison figure (the report's section 2.7) is this
file's: three columns, one per layout; three rows, the flow (streamlines
coloured by speed, one colour scale), the litho-top source's plume (log
colour scale shared across the layouts), and the deposition per 0.2 m
segment summed over the three sources as bars standing on their surfaces
(one bar scale, stated in the title), each deposition panel titled with
the layout's score.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe47"))

import figures47  # noqa: E402
import tables56  # noqa: E402
from figures47 import (  # noqa: E402
    BAR_LENGTH,
    DPI,
    draw_room,
    edges,
    grey_bad,
    segment_bar,
)

from src.mesh import SOLID  # noqa: E402

OUT = ROOT / "results" / "builder56"
SCRATCH = OUT / "figures"
FIGURES = ROOT / "docs" / "reports"
figures47.OUT = OUT
figures47.FIGURES = SCRATCH
LAYOUTS = ("L0", "L1", "L2")
CHANGES = {
    "L0": "the room as it is",
    "L1": "return 3 closed",
    "L2": "the supply over the left half",
}
SOURCES = ("S1", "S2", "S3")
PLUME_SOURCE = "S2"
PRODUCT_GRID = "320x120"
BASE_SC = "0.7"
PROCESS_SURFACES = ("litho_tool top", "etch_chamber top")
CLASS_COLOUR = {"2": "tab:cyan", "4": "gold"}
COLOUR_FLOOR = 1e-6
BAR_FLOOR = 1e-3


def load(name: str) -> tuple[dict, dict]:
    """A record and its arrays."""
    rec = json.loads((OUT / f"{name}.json").read_text())
    return rec, dict(np.load(OUT / f"{name}.npz"))


def fields(args: argparse.Namespace) -> None:
    """Prompt 47's three-panel figure, drawn under results and copied beside the report."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    figures47.fields(argparse.Namespace(name=args.name))
    drawn = SCRATCH / f"ecr002_step6_{args.name}.png"
    path = FIGURES / f"ecr002_step8_{args.name}.png"
    shutil.copyfile(drawn, path)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def concentration(args: argparse.Namespace) -> None:
    """Prompt 47's concentration figure of one march, kept under results."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    figures47.concentration(argparse.Namespace(label=args.label))
    drawn = (
        SCRATCH / f"ecr002_step6_{args.label[len('transport_') :]}_concentration.png"
    )
    path = SCRATCH / f"ecr002_step8_{args.label[len('transport_') :]}_concentration.png"
    drawn.replace(path)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def summed_segments(marches: list[dict], k: str) -> dict[str, float]:
    """Every 0.2 m segment's deposition of one class, summed over the sources."""
    out: dict[str, float] = {}
    for rec in marches:
        for key, value in rec["per_class"][k]["segments"].items():
            out[key] = out.get(key, 0.0) + value
    return out


def process_deposition(marches: list[dict], k: str) -> dict[str, float]:
    """The deposition on each process surface summed over the sources, and the score."""
    out = {s: 0.0 for s in PROCESS_SURFACES}
    for rec in marches:
        for s in PROCESS_SURFACES:
            out[s] += rec["per_class"][k]["by_surface"].get(s, 0.0)
    out["score"] = sum(out[s] for s in PROCESS_SURFACES)
    return out


def march_label(layout: str, source: str) -> str:
    """The march record of a layout's source on the product grid at Sc_t 0.7, the supplement included."""
    return tables56.transport_key(layout, PRODUCT_GRID, source, float(BASE_SC))


def compare(args: argparse.Namespace) -> None:
    """The comparison figure of one class: the three layouts on 320x120 at Sc_t 0.7."""
    k = args.size_class
    flows = {
        layout: load(tables56.flow_name(layout, PRODUCT_GRID)) for layout in LAYOUTS
    }
    marches = {
        layout: [
            json.loads((OUT / f"{march_label(layout, source)}.json").read_text())
            for source in SOURCES
        ]
        for layout in LAYOUTS
    }
    plumes = {
        layout: dict(np.load(OUT / f"{march_label(layout, PLUME_SOURCE)}.npz"))
        for layout in LAYOUTS
    }
    diameter = marches["L0"][0]["per_class"][k]["diameter"]
    um = f"{diameter * 1e6:g}"
    # One scale per row across the layouts.
    speeds = {}
    for layout, (_rec, a) in flows.items():
        solid = a["cell_type"] == SOLID
        u = np.where(solid, 0.0, a["u_c"])
        v = np.where(solid, 0.0, a["v_c"])
        speeds[layout] = (u, v, np.hypot(u, v))
    speed_norm = Normalize(0.0, max(s[2].max() for s in speeds.values()))
    peak = max(
        float(
            np.ma.masked_where(
                plumes[layout]["cell_type"] == SOLID, plumes[layout][f"C_{k}"]
            ).max()
        )
        for layout in LAYOUTS
    )
    segments = {layout: summed_segments(marches[layout], k) for layout in LAYOUTS}
    largest = max(max(seg.values()) for seg in segments.values())
    scores = {layout: process_deposition(marches[layout], k) for layout in LAYOUTS}

    fig, axes = plt.subplots(3, 3, figsize=(19.5, 8.6), constrained_layout=True)
    for col, layout in enumerate(LAYOUTS):
        rec, a = flows[layout]
        raw = rec["raw"]
        xc, yc = a["xc"], a["yc"]
        x_edges, y_edges = edges(xc, yc, raw)
        u, v, speed = speeds[layout]
        ax = axes[0, col]
        strm = ax.streamplot(
            xc,
            yc,
            u,
            v,
            color=speed,
            cmap="viridis",
            norm=speed_norm,
            density=(1.6, 0.8),
            linewidth=0.6,
            arrowsize=0.6,
        )
        if col == len(LAYOUTS) - 1:
            fig.colorbar(
                strm.lines, ax=axes[0, :].tolist(), label="speed (m/s)", shrink=0.9
            )
        draw_room(ax, raw)
        name = tables56.label_of(layout, PRODUCT_GRID)
        ax.set_title(
            f"{name}, {CHANGES[layout]}: streamlines coloured by speed\n"
            f"{rec['outer']:,} outer iterations, {rec['stop']}",
            fontsize=8,
        )

        ax = axes[1, col]
        plume = plumes[layout]
        solid = plume["cell_type"] == SOLID
        field = np.ma.masked_where(solid, plume[f"C_{k}"])
        pm = ax.pcolormesh(
            x_edges,
            y_edges,
            field,
            norm=LogNorm(vmin=peak * COLOUR_FLOOR, vmax=peak),
            cmap=grey_bad("inferno"),
            shading="flat",
        )
        if col == len(LAYOUTS) - 1:
            fig.colorbar(pm, ax=axes[1, :].tolist(), label="C (per m^3)", shrink=0.9)
        plume_rec = marches[layout][SOURCES.index(PLUME_SOURCE)]
        draw_room(ax, raw, source=tuple(plume_rec["source"]["box"]))
        c = plume_rec["per_class"][k]
        ax.set_title(
            f"{PLUME_SOURCE} plume, {um} um, {c['stop']} at {c['t_end']:.0f} s\n"
            f"litho top {c['by_surface'].get(PROCESS_SURFACES[0], 0.0):.3g}, "
            f"etch top {c['by_surface'].get(PROCESS_SURFACES[1], 0.0):.3g} per s per m depth",
            fontsize=8,
        )

        ax = axes[2, col]
        draw_room(ax, raw)
        for key, value in segments[layout].items():
            # Bars under a thousandth of the largest would draw as hairlines.
            if value <= BAR_FLOOR * largest:
                continue
            geom = segment_bar(
                key, BAR_LENGTH * value / largest, raw, xc, plume_rec["segment"]
            )
            if geom is None:
                continue
            x, y, w, h = geom
            ax.add_patch(
                Rectangle((x, y), w, h, color=CLASS_COLOUR[k], alpha=0.85, zorder=10)
            )
        s = scores[layout]
        ax.set_title(
            f"{name} score {s['score']:.3g} per s per m depth "
            f"(litho top {s[PROCESS_SURFACES[0]]:.3g}, etch top {s[PROCESS_SURFACES[1]]:.3g})\n"
            f"deposition per 0.2 m segment, S1 to S3 summed",
            fontsize=8,
        )
    fig.suptitle(
        f"The three layouts on {PRODUCT_GRID} at Sc_t {BASE_SC}, {um} um: the flow, the "
        f"litho-top source's plume (colour floor {COLOUR_FLOOR:g} of the largest peak), and "
        f"the deposition bars (a {BAR_LENGTH} m bar is {largest:.3g} particles per s per m "
        f"depth, the largest segment; bars below {BAR_FLOOR:g} of it omitted)",
        fontsize=10,
    )
    path = FIGURES / f"ecr002_step8_compare_{um}um.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def everything(_args: argparse.Namespace) -> None:
    """Every converged flow record, every march, both comparison figures."""
    for path in sorted(OUT.glob("L?_*.json")):
        if path.stem.endswith("_time"):
            continue
        rec = json.loads(path.read_text())
        if rec.get("stop") == "error_estimate_and_continuity" and "core" in rec:
            fields(argparse.Namespace(name=path.stem))
    for path in sorted(OUT.glob("transport_*.json")):
        if tables56.TRANSPORT.match(path.stem):
            concentration(argparse.Namespace(label=path.stem))
    for k in ("2", "4"):
        if all(
            (OUT / f"{march_label(layout, source)}.json").exists()
            for layout in LAYOUTS
            for source in SOURCES
        ):
            compare(argparse.Namespace(size_class=k))


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
    p = sub.add_parser("compare")
    p.add_argument("size_class", choices=("2", "4"))
    p.set_defaults(func=compare)
    sub.add_parser("all").set_defaults(func=everything)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
