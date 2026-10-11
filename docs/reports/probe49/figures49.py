"""The report's figures from the prompt 49 records.

Usage:
    python figures49.py fields NAME    three panels: streamlines coloured by speed,
                                       nu_t / nu (log scale), k; the slot, the outlet
                                       and the four comparison lines marked
    python figures49.py lines          one figure per comparison line: the measured
                                       points, every converged run, the published line
    python figures49.py all            the fields of every converged record, and the lines

Figures go beside the report as docs/reports/ecr002_val017_<record>.png and
ecr002_val017_line_<line>.png, small PNGs (90 dpi). The published line is
drawn at the digitizer's full resolution from
results/builder49/rong_nielsen_2008_digitized.json when that record exists,
and from the module's station values otherwise.
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
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))

from tables46 import classify  # noqa: E402

from validation import annex20  # noqa: E402

OUT = ROOT / "results" / "builder49"
FIGURES = ROOT / "docs" / "reports"
DIGITIZED = OUT / "rong_nielsen_2008_digitized.json"
DPI = 90
NAME = re.compile(r"^(standard|rng)_\d+x\d+_(upwind|umist)(_.*)?$")
CONVERGED = "error_estimate_and_continuity"
AT_CAP_COMPARED = "bounded"
INLET_COLOUR = "tab:blue"
OUTLET_COLOUR = "tab:orange"
LINE_COLOUR = "0.3"
BAR = 0.12  # m, the thickness of the opening bars
RUN_STYLES = {
    "standard_216x72_upwind": ("tab:blue", "-"),
    "standard_216x72_umist": ("tab:blue", "--"),
    "standard_375x125_upwind": ("tab:green", "-"),
    "standard_375x125_umist": ("tab:green", "--"),
    "rng_216x72_upwind": ("tab:purple", "-"),
}


def load(name: str) -> tuple[dict, dict]:
    """A record and its arrays."""
    rec = json.loads((OUT / f"{name}.json").read_text())
    return rec, dict(np.load(OUT / f"{name}.npz"))


def record_class(rec: dict) -> str:
    """tables46's class, with its sub-class."""
    cls, sub = classify(rec)
    return cls + (f" ({sub})" if sub else "")


def compared(rec: dict) -> bool:
    """Converged, or bounded at the cap: the records whose fields are drawn."""
    return rec["stop"] == CONVERGED or record_class(rec).startswith(AT_CAP_COMPARED)


def edges(xc: np.ndarray, yc: np.ndarray, raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cell edges from the centres, the domain ends closing them."""
    x = np.concatenate(([0.0], 0.5 * (xc[1:] + xc[:-1]), [raw["domain"]["width"]]))
    y = np.concatenate(([0.0], 0.5 * (yc[1:] + yc[:-1]), [raw["domain"]["height"]]))
    return x, y


def draw_room(ax: plt.Axes, raw: dict, lines: bool = True) -> None:
    """The slot (blue, arrow in), the outlet (orange, arrow out) and the four lines."""
    width, height = raw["domain"]["width"], raw["domain"]["height"]
    inlet, outlet = raw["boundaries"]["inlet"], raw["boundaries"]["outlet"]
    ax.add_patch(
        Rectangle(
            (0.0, inlet["y_start"]),
            BAR,
            inlet["y_end"] - inlet["y_start"],
            color=INLET_COLOUR,
            zorder=5,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (-0.4, 0.5 * (inlet["y_start"] + inlet["y_end"])),
            (BAR + 0.3, 0.5 * (inlet["y_start"] + inlet["y_end"])),
            arrowstyle="-|>",
            mutation_scale=12,
            color=INLET_COLOUR,
            zorder=6,
            clip_on=False,
        )
    )
    ax.add_patch(
        Rectangle(
            (width - BAR, outlet["y_start"]),
            BAR,
            outlet["y_end"] - outlet["y_start"],
            color=OUTLET_COLOUR,
            zorder=5,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (width - BAR - 0.3, 0.5 * (outlet["y_start"] + outlet["y_end"])),
            (width + 0.4, 0.5 * (outlet["y_start"] + outlet["y_end"])),
            arrowstyle="-|>",
            mutation_scale=12,
            color=OUTLET_COLOUR,
            zorder=6,
            clip_on=False,
        )
    )
    if lines:
        for line in annex20.LINES.values():
            if line.vertical:
                ax.plot(
                    [line.position] * 2, [0.0, height], ":", color=LINE_COLOUR, lw=0.8
                )
            else:
                ax.plot(
                    [0.0, width], [line.position] * 2, ":", color=LINE_COLOUR, lw=0.8
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
    xc, yc = a["xc"], a["yc"]
    u, v = a["u_c"], a["v_c"]
    speed = np.hypot(u, v)
    nu = rec["nu_air"]
    ratio = a["nu_t"] / nu
    x_edges, y_edges = edges(xc, yc, raw)

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 10.5), constrained_layout=True)
    ax = axes[0]
    strm = ax.streamplot(
        xc,
        yc,
        u,
        v,
        color=speed,
        cmap="viridis",
        density=(2.4, 0.8),
        linewidth=0.7,
        arrowsize=0.7,
    )
    fig.colorbar(strm.lines, ax=ax, label="speed (m/s)", shrink=0.9)
    draw_room(ax, raw)
    ax.set_title(
        f"{rec['name']}: streamlines coloured by speed ({rec['outer']:,} outer "
        f"iterations, {record_class(rec)})"
    )
    ax = axes[1]
    pm = ax.pcolormesh(
        x_edges,
        y_edges,
        ratio,
        norm=LogNorm(vmin=max(float(ratio.min()), 1e-1), vmax=float(ratio.max())),
        cmap="magma",
        shading="flat",
    )
    fig.colorbar(pm, ax=ax, label="nu_t / nu", shrink=0.9)
    draw_room(ax, raw)
    core = rec["core"]
    ax.set_title(
        f"nu_t / nu (room median {core['nu_t_over_nu_median']:.3g}, "
        f"95th {core['nu_t_over_nu_p95']:.3g})"
    )
    ax = axes[2]
    pm = ax.pcolormesh(x_edges, y_edges, a["k"], cmap="cividis", shading="flat")
    fig.colorbar(pm, ax=ax, label="k (m^2/s^2)", shrink=0.9)
    draw_room(ax, raw)
    y = rec["y_star"]
    ax.set_title(
        f"k (y+ at the wall nodes: median {y['median']:.3g}, "
        f"{y['share_below_floor']:.0%} below 11.53)"
    )
    path = FIGURES / f"ecr002_val017_{rec['name']}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def published_curve(key: str) -> tuple[np.ndarray, np.ndarray, str]:
    """The published line at full resolution if the digitizer's record exists."""
    if DIGITIZED.exists():
        panel = json.loads(DIGITIZED.read_text())["panels"][key]["curve"]
        return (
            np.array(panel["coordinate"]),
            np.array(panel["u_over_u0"]),
            "digitized line",
        )
    line = annex20.LINES[key]
    held = [i for i, p in enumerate(line.published) if p is not None]
    return (
        np.array([line.stations[i] for i in held]),
        np.array([line.published[i] for i in held]),
        "at the stations",
    )


def converged_lines() -> dict[str, dict]:
    """The line samples of every compared record (converged, or falling at the cap)."""
    out = {}
    for path in sorted(OUT.glob("lines_*.json")):
        rec = json.loads(path.read_text())
        flow = json.loads((OUT / f"{rec['name']}.json").read_text())
        if compared(flow) and NAME.match(rec["name"]):
            rec["_class"] = record_class(flow)
            out[rec["name"]] = rec
    return out


def lines(_args: argparse.Namespace) -> None:
    """One figure per comparison line."""
    runs = converged_lines()
    for key, line in annex20.LINES.items():
        fig, ax = plt.subplots(figsize=(5.0, 7.0) if line.vertical else (9.0, 4.2))
        stations = np.array(line.stations)
        meas = np.array(line.measured)
        pc, pu, how = published_curve(key)
        if line.vertical:
            ax.plot(
                pu,
                pc,
                "-",
                color="tab:red",
                lw=1.0,
                label=f"Rong and Nielsen 2008, standard k-eps ({how})",
            )
            for name, rec in runs.items():
                colour, style = RUN_STYLES.get(name, ("0.5", "-"))
                ax.plot(
                    rec["lines"][key]["u_over_u0"],
                    stations,
                    style,
                    color=colour,
                    lw=1.2,
                    label=f"{name} ({rec['_class']})",
                )
            ax.plot(
                meas,
                stations,
                "o",
                mfc="none",
                color="black",
                ms=5,
                label="measured (Nielsen 1990)",
            )
            ax.set_xlabel("u / u0")
            ax.set_ylabel("y (m)")
            ax.set_ylim(0.0, annex20.HEIGHT)
            ax.axvline(0.0, color="0.7", lw=0.6)
        else:
            ax.plot(
                pc,
                pu,
                "-",
                color="tab:red",
                lw=1.0,
                label=f"Rong and Nielsen 2008, standard k-eps ({how})",
            )
            for name, rec in runs.items():
                colour, style = RUN_STYLES.get(name, ("0.5", "-"))
                ax.plot(
                    stations,
                    rec["lines"][key]["u_over_u0"],
                    style,
                    color=colour,
                    lw=1.2,
                    label=f"{name} ({rec['_class']})",
                )
            ax.plot(
                stations,
                meas,
                "o",
                mfc="none",
                color="black",
                ms=5,
                label="measured (Nielsen 1990)",
            )
            ax.set_xlabel("x (m)")
            ax.set_ylabel("u / u0")
            ax.set_xlim(0.0, annex20.LENGTH)
            ax.axhline(0.0, color="0.7", lw=0.6)
        ax.set_title(f"VAL-017, {line.label}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, lw=0.3, alpha=0.5)
        fig.tight_layout()
        path = FIGURES / f"ecr002_val017_line_{key}.png"
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def everything(_args: argparse.Namespace) -> None:
    """The fields of every converged record, then the lines."""
    for path in sorted(OUT.glob("*.json")):
        if not NAME.match(path.stem):
            continue
        rec = json.loads(path.read_text())
        if "stop" in rec and compared(rec) and "core" in rec:
            fields(argparse.Namespace(name=path.stem))
    lines(argparse.Namespace())


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("fields")
    p.add_argument("name")
    p.set_defaults(func=fields)
    sub.add_parser("lines").set_defaults(func=lines)
    sub.add_parser("all").set_defaults(func=everything)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
