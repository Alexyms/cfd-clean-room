"""The report's figures from the prompt 55 records.

Usage:
    python figures55.py fields NAME    the marked figure of one run: streamlines,
                                       nu_t / nu and k over the jet, the readings in the titles
    python figures55.py centreline     V_0,y / V0 and k / V0^2 on the axis, every compared run
                                       against the data
    python figures55.py halfwidth      delta_0.5 / w against the dots, every compared run
    python figures55.py lateral        the self-similar profiles at the six heights
    python figures55.py all            every figure

Reads results/builder55/NAME.json and NAME.npz (impinging55.py) and writes
docs/reports/ecr002_val020_*.png. The numbers in the titles are
tables55's, recomputed from the same records.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import tables55  # noqa: E402

from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from validation import impinging_jet as ij  # noqa: E402

OUT = tables55.OUT
FIGURES = ROOT / "docs" / "reports"
DPI = 90
W = ij.W_JET
# The jet region drawn in the field figures, in nozzle widths from the axis
# and in y / h_jet.
ZOOM_W = 12.0
RUN_STYLES = {
    "standard_272x90_upwind": ("tab:blue", "-"),
    "standard_408x135_upwind": ("tab:red", "-"),
    "standard_408x135_umist": ("tab:green", "--"),
    "rng_408x135_upwind": ("tab:purple", "-."),
    "standard_272x90_upwind_w100": ("tab:orange", ":"),
}


def load(name: str) -> tuple[dict, dict]:
    rec = tables55.records()[name]
    return rec, dict(np.load(OUT / f"{name}.npz"))


def uniform_rows(
    yc: np.ndarray, fields: tuple[np.ndarray, ...], count: int = 181
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Cell-centre fields resampled on equally spaced rows, linear in y per column.

    matplotlib's streamplot wants an equally spaced lattice; the mesh is
    clustered in y. The streamlines are drawn from the resampled fields;
    the colour maps keep the mesh's own cells.
    """
    y = np.linspace(float(yc[0]), float(yc[-1]), count)
    out = []
    for field in fields:
        out.append(
            np.stack(
                [np.interp(y, yc, field[:, i]) for i in range(field.shape[1])], axis=1
            )
        )
    return y, out


def fields(args: argparse.Namespace) -> None:
    rec, a = load(args.name)
    config = SimConfig.from_dict(rec["raw"])
    mesh = Mesh(config)
    ev = tables55.evaluation(args.name, rec)
    axis = ij.axis_x(config)
    xc = (np.asarray(mesh.xc) - axis) / W
    yc = np.asarray(mesh.yc) / ij.H_JET
    x_edges = (np.asarray(mesh.x) - axis) / W
    y_edges = np.asarray(mesh.y) / ij.H_JET
    u, v = a["u_c"], a["v_c"]
    speed = np.hypot(u, v)
    ratio = a["nu_t"] / rec["nu_air"]
    cls_text = rec["_class"]
    st, ys = ev["stagnation"], ev["y_star"]

    fig = plt.figure(figsize=(9.0, 14.0), constrained_layout=True)
    grid = fig.add_gridspec(4, 1, height_ratios=[1.0, 2.0, 2.0, 2.0])
    y_u, (u_u, v_u, speed_u) = uniform_rows(yc, (u, v, speed))
    ax = fig.add_subplot(grid[0])
    ax.streamplot(
        xc, y_u, u_u, v_u, color="0.4", density=(3.0, 0.6), linewidth=0.5, arrowsize=0.5
    )
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect(ij.H_JET / W)
    ax.set_title(
        f"{rec['name']}: the whole domain, {rec['width_w']:.0f} w wide\n"
        f"({rec['outer']:,} outer iterations, {cls_text}); potential core "
        f"{tables55.fmt(ev['potential_core_w']['computed'], 3)} w (data {ev['potential_core_w']['data']:.2f})",
        fontsize=9,
    )
    sel = (xc >= -ZOOM_W - 1.0) & (xc <= ZOOM_W + 1.0)
    ax = fig.add_subplot(grid[1])
    strm = ax.streamplot(
        xc[sel],
        y_u,
        u_u[:, sel],
        v_u[:, sel],
        color=speed_u[:, sel] / ij.V0,
        cmap="viridis",
        density=(1.6, 1.2),
        linewidth=0.7,
        arrowsize=0.7,
    )
    fig.colorbar(strm.lines, ax=ax, label="speed / V0", shrink=0.9)
    ax.set_xlim(-ZOOM_W, ZOOM_W)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect(ij.H_JET / W)
    ax.set_title(
        "streamlines coloured by speed\n(x / w across, y / h up from the plate)",
        fontsize=9,
    )
    ax = fig.add_subplot(grid[2])
    pm = ax.pcolormesh(
        x_edges,
        y_edges,
        ratio,
        norm=LogNorm(
            vmin=max(float(ratio[ratio > 0].min()), 1e-1), vmax=float(ratio.max())
        ),
        cmap="magma",
        shading="flat",
    )
    fig.colorbar(pm, ax=ax, label="nu_t / nu", shrink=0.9)
    ax.set_xlim(-ZOOM_W, ZOOM_W)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect(ij.H_JET / W)
    core = rec["core"]
    ax.set_title(
        f"nu_t / nu (median {core['nu_t_over_nu_median']:.3g}, largest {core['nu_t_over_nu_all_max']:.3g})",
        fontsize=9,
    )
    ax = fig.add_subplot(grid[3])
    pm = ax.pcolormesh(
        x_edges, y_edges, a["k"] / ij.V0**2, cmap="cividis", shading="flat"
    )
    fig.colorbar(pm, ax=ax, label="k / V0^2", shrink=0.9)
    ax.set_xlim(-ZOOM_W, ZOOM_W)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect(ij.H_JET / W)
    ax.set_title(
        f"k / V0^2: at the last ring station {st['computed_at_ring_station']:.4f} vs the PIV estimate "
        f"{st['ring_isotropic']:.4f} (ratio {st['ring_ratio_isotropic']:.2f})\n"
        f"plate y+ median {ys['median']:.1f}, at the axis {ys['at_axis']:.1f}, floor {ys['floor']:.2f}",
        fontsize=9,
    )
    path = FIGURES / f"ecr002_val020_{rec['name']}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def centreline(_args: argparse.Namespace) -> None:
    recs = tables55.compared_records()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 6.5))
    ax = axes[0]
    for name, (rec, ev) in recs.items():
        colour, style = RUN_STYLES.get(name, ("0.5", "-"))
        ax.plot(
            ev["centreline"]["computed"],
            ev["centreline"]["y_over_h"],
            style,
            color=colour,
            lw=1.2,
            label=f"{name} ({rec['_class']})",
        )
    ax.errorbar(
        ij.CENTRELINE_V,
        ij.CENTRELINE_Y,
        xerr=[r[2] for r in ij.CENTRELINE_ROWS],
        fmt="o",
        mfc="none",
        color="black",
        ms=4,
        lw=0.6,
        label="PIV, Khayrullina et al. (2017) figure 4(a), 16 mm",
    )
    ax.axhline(ij.FREE_JET_H, color="0.6", lw=0.6, ls=":")
    ax.axhline(ij.IMPINGEMENT_H, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("V_0,y / V0")
    ax.set_ylabel("y / h_jet (down from the nozzle)")
    ax.set_xlim(0.0, 1.2)
    ax.set_ylim(1.0, 0.0)
    ax.set_title("the centreline")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax = axes[1]
    for name, (_rec, ev) in recs.items():
        colour, style = RUN_STYLES.get(name, ("0.5", "-"))
        ax.plot(
            ev["k_axis"]["computed"],
            ev["k_axis"]["y_over_h"],
            style,
            color=colour,
            lw=1.2,
            label=name,
        )
    rings = np.array([r[2] for r in ij.INTENSITY_ROWS])
    ax.plot(
        ij.K_ISOTROPIC * ij.INTENSITY_I[rings] ** 2,
        ij.INTENSITY_Y[rings],
        "o",
        mfc="none",
        color="black",
        ms=4,
        label="0.75 I_uv^2, Re 8,000 circles (figure 6)",
    )
    ax.plot(
        ij.K_ISOTROPIC * ij.INTENSITY_I[~rings] ** 2,
        ij.INTENSITY_Y[~rings],
        "s",
        mfc="none",
        color="0.4",
        ms=4,
        label="0.75 I_uv^2, the three series' centroid",
    )
    ax.plot(
        ij.K_IN_PLANE * ij.INTENSITY_I**2,
        ij.INTENSITY_Y,
        ".",
        color="0.6",
        ms=3,
        label="0.5 I_uv^2 (in-plane lower bound)",
    )
    ax.set_xlabel("k / V0^2")
    ax.set_ylim(1.0, 0.0)
    ax.set_title("k on the axis")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    path = FIGURES / "ecr002_val020_centreline.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def halfwidth(_args: argparse.Namespace) -> None:
    recs = tables55.compared_records()
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for name, (_rec, ev) in recs.items():
        colour, style = RUN_STYLES.get(name, ("0.5", "-"))
        hw = ev["half_width"]
        held = [
            (y, d)
            for y, d in zip(hw["y_over_w"], hw["computed"], strict=True)
            if d is not None
        ]
        if held:
            ax.plot(
                [y for y, _ in held],
                [d for _, d in held],
                style,
                marker=".",
                color=colour,
                lw=1.2,
                label=f"{name} (K_y {tables55.fmt(hw['spreading_rate']['computed'], 3)})",
            )
    ax.plot(
        ij.HALF_WIDTH_Y,
        ij.HALF_WIDTH_DELTA,
        "o",
        mfc="none",
        color="black",
        ms=5,
        label=f"PIV dots, figure 8(a) (K_y {ij.spreading_rate(ij.HALF_WIDTH_Y, ij.HALF_WIDTH_DELTA):.3f})",
    )
    ax.axvspan(*ij.SELF_SIMILAR_W, color="0.9", zorder=0)
    ax.set_xlabel("y / w_jet (down from the nozzle)")
    ax.set_ylabel("delta_0.5 / w_jet")
    ax.set_title("the half-width (shaded: the self-similar range)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    path = FIGURES / "ecr002_val020_halfwidth.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def lateral(_args: argparse.Namespace) -> None:
    recs = tables55.compared_records()
    heights = tables55.LATERAL_HEIGHTS
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), sharex=True, sharey=True)
    xi = np.linspace(-2.0, 2.0, 81)
    for ax, h in zip(axes.flat, heights, strict=True):
        for name, (_rec, ev) in recs.items():
            entry = ev["lateral"][h]
            if entry["computed"] is None:
                continue
            colour, style = RUN_STYLES.get(name, ("0.5", "-"))
            ax.plot(
                entry["x_over_delta"],
                entry["computed"],
                style,
                marker=".",
                ms=3,
                color=colour,
                lw=1.0,
                label=f"{name} (delta {entry['delta_over_w']:.2f} w)",
            )
        ax.plot(
            xi,
            ij.gaussian(xi),
            "--",
            color="tab:orange",
            lw=1.2,
            label="Gaussian fit (paper)",
        )
        if h == "4.8":
            ax.plot(
                [r[0] for r in ij.LATERAL_V_48_ROWS],
                [r[1] for r in ij.LATERAL_V_48_ROWS],
                "o",
                mfc="none",
                color="black",
                ms=4,
                label="PIV, figure 5(a)",
            )
        ax.set_title(
            f"y'/w = {h} (y/h {ij.height_over_h(float(h)):.3f}), {ev['lateral'][h]['source'] if recs else ''}"
        )
        ax.grid(True, lw=0.3, alpha=0.5)
    axes[0, 0].legend(fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("x / delta_0.5")
    for ax in axes[:, 0]:
        ax.set_ylabel("V / V_0,y")
    axes[0, 0].set_xlim(-2.0, 2.0)
    axes[0, 0].set_ylim(0.0, 1.1)
    fig.tight_layout()
    path = FIGURES / "ecr002_val020_lateral.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def everything(args: argparse.Namespace) -> None:
    for name in tables55.compared_records():
        fields(argparse.Namespace(name=name))
    centreline(args)
    halfwidth(args)
    lateral(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("fields")
    p.add_argument("name")
    p.set_defaults(func=fields)
    for name, func in (
        ("centreline", centreline),
        ("halfwidth", halfwidth),
        ("lateral", lateral),
        ("all", everything),
    ):
        p = sub.add_parser(name)
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
