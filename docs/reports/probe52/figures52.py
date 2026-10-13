"""The report's figures from the prompt 52 records.

Usage:
    python figures52.py fields NAME    the marked figure of one converged run: the whole
                                       domain's streamlines, then x/H -5 to 25 in three
                                       panels (streamlines coloured by speed, nu_t / nu on
                                       a log scale, k); the step, the inlet, the outlet and
                                       the reattachment point marked
    python figures52.py profiles       one figure per station (x/H = -4, 1, 4, 6, 10): the
                                       measured points and every compared run
    python figures52.py walls          Cf and Cp along the bottom wall, one figure each
    python figures52.py precursor      the precursor's layer along the channel
    python figures52.py all            everything above

Figures go beside the report as docs/reports/ecr002_val019_<record>.png,
ecr002_val019_profile_x<station>.png, ecr002_val019_cf.png,
ecr002_val019_cp.png and ecr002_val019_precursor.png, small PNGs (90 dpi).
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
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import tables52  # noqa: E402

from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from validation import backstep  # noqa: E402

OUT = ROOT / "results" / "builder52"
FIGURES = ROOT / "docs" / "reports"
DPI = 90
H = backstep.STEP_HEIGHT
ZOOM = (-5.0, 25.0)
RUN_STYLES = {
    "standard_410x40_upwind": ("tab:orange", "-"),
    "standard_656x60_upwind": ("tab:blue", "-"),
    "standard_656x60_umist": ("tab:blue", "--"),
    "standard_984x90_upwind": ("tab:green", "-"),
    "rng_656x60_upwind": ("tab:purple", "-"),
}
STEP_COLOUR = "0.35"
INLET_COLOUR = "tab:blue"
OUTLET_COLOUR = "tab:orange"
X_R_COLOUR = "tab:red"


def load(name: str) -> tuple[dict, dict]:
    rec = json.loads((OUT / f"{name}.json").read_text())
    return rec, dict(np.load(OUT / f"{name}.npz"))


def edges(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(mesh.x) / H, np.asarray(mesh.y) / H


def draw_geometry(
    ax: plt.Axes, config: SimConfig, x_r: float | None, zoom: bool
) -> None:
    """The step (grey block), the inlet (blue arrow), the outlet (orange arrow), x_r."""
    l_up = backstep.upstream_length(config) / H
    ax.add_patch(Rectangle((-l_up, 0.0), l_up, 1.0, color=STEP_COLOUR, zorder=4))
    if not zoom:
        ax.add_patch(
            FancyArrowPatch(
                (-l_up - 6.0, 5.0),
                (-l_up + 2.0, 5.0),
                arrowstyle="-|>",
                mutation_scale=12,
                color=INLET_COLOUR,
                zorder=6,
                clip_on=False,
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (48.0, 4.5),
                (56.0, 4.5),
                arrowstyle="-|>",
                mutation_scale=12,
                color=OUTLET_COLOUR,
                zorder=6,
                clip_on=False,
            )
        )
        ax.text(
            -l_up - 1.0, 8.4, f"inlet x/H = {-l_up:.0f}", color=INLET_COLOUR, fontsize=7
        )
        ax.text(
            44.0, 8.4, "outlet x/H = 50", color=OUTLET_COLOUR, fontsize=7, ha="right"
        )
    else:
        ax.annotate(
            f"inlet at x/H = {-l_up:.0f}",
            xy=(ZOOM[0], 5.0),
            xytext=(ZOOM[0] + 1.0, 7.8),
            color=INLET_COLOUR,
            fontsize=7,
            arrowprops={"arrowstyle": "->", "color": INLET_COLOUR},
        )
        ax.annotate(
            "outlet at x/H = 50",
            xy=(ZOOM[1], 4.5),
            xytext=(ZOOM[1] - 6.0, 7.8),
            color=OUTLET_COLOUR,
            fontsize=7,
            arrowprops={"arrowstyle": "->", "color": OUTLET_COLOUR},
        )
    if x_r is not None:
        ax.plot(
            [x_r], [0.0], marker="^", color=X_R_COLOUR, ms=8, zorder=7, clip_on=False
        )
        ax.annotate(
            f"x_r / H = {x_r:.2f}",
            xy=(x_r, 0.0),
            xytext=(x_r + 0.8, 1.6),
            color=X_R_COLOUR,
            fontsize=8,
            arrowprops={"arrowstyle": "->", "color": X_R_COLOUR},
        )
    ax.set_xlabel("x / H")
    ax.set_ylabel("y / H")


def fields(args: argparse.Namespace) -> None:
    rec, a = load(args.name)
    config = SimConfig.from_dict(rec["raw"])
    mesh = Mesh(config)
    ev = tables52.evaluation(args.name, rec)
    x_r = ev["x_r_over_h"]
    xc = np.asarray(backstep.data_x(config, mesh.xc))
    yc = np.asarray(mesh.yc) / H
    x_edges, y_edges = edges(mesh)
    x_edges = x_edges - backstep.upstream_length(config) / H
    u, v = a["u_c"], a["v_c"]
    speed = np.hypot(u, v)
    ratio = a["nu_t"] / rec["nu_air"]
    cls = tables52.classify(rec)
    cls_text = cls[0] + (f" ({cls[1]})" if cls[1] else "")
    l_up = backstep.upstream_length(config) / H

    fig = plt.figure(figsize=(11.0, 12.5), constrained_layout=True)
    grid = fig.add_gridspec(4, 1, height_ratios=[1.0, 2.2, 2.2, 2.2])
    ax = fig.add_subplot(grid[0])
    ax.streamplot(
        xc, yc, u, v, color="0.4", density=(3.0, 0.5), linewidth=0.5, arrowsize=0.5
    )
    draw_geometry(ax, config, x_r, zoom=False)
    ax.set_xlim(-l_up, 50.0)
    ax.set_ylim(0.0, 9.0)
    ax.set_aspect(2.0)
    ax.set_title(
        f"{rec['name']}: the whole domain ({rec['outer']:,} outer iterations, {cls_text}); "
        f"U_centre at x/H = -4 is {ev['u_ref']['over_measured']:.4f} U_ref"
    )
    sel = (xc >= ZOOM[0] - 1.0) & (xc <= ZOOM[1] + 1.0)
    ax = fig.add_subplot(grid[1])
    strm = ax.streamplot(
        xc[sel],
        yc,
        u[:, sel],
        v[:, sel],
        color=speed[:, sel],
        cmap="viridis",
        density=(2.0, 1.2),
        linewidth=0.7,
        arrowsize=0.7,
    )
    fig.colorbar(strm.lines, ax=ax, label="speed (m/s)", shrink=0.9)
    draw_geometry(ax, config, x_r, zoom=True)
    ax.set_xlim(*ZOOM)
    ax.set_ylim(0.0, 9.0)
    ax.set_aspect("equal")
    ax.set_title("streamlines coloured by speed")
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
    draw_geometry(ax, config, x_r, zoom=True)
    ax.set_xlim(*ZOOM)
    ax.set_ylim(0.0, 9.0)
    ax.set_aspect("equal")
    core = rec["core"]
    ax.set_title(
        f"nu_t / nu (median {core['nu_t_over_nu_median']:.3g}, largest {core['nu_t_over_nu_all_max']:.3g})"
    )
    ax = fig.add_subplot(grid[3])
    pm = ax.pcolormesh(x_edges, y_edges, a["k"], cmap="cividis", shading="flat")
    fig.colorbar(pm, ax=ax, label="k (m^2/s^2)", shrink=0.9)
    draw_geometry(ax, config, x_r, zoom=True)
    ax.set_xlim(*ZOOM)
    ax.set_ylim(0.0, 9.0)
    ax.set_aspect("equal")
    y = ev["y_star"]
    ax.set_title(
        f"k (first-node y+: floor downstream of x_r {y['floor_downstream']:.0f}, step top "
        f"{y['step_top']:.0f}, ceiling {y['ceiling_upstream']:.0f}; floor {y['floor']:.2f})"
    )
    path = FIGURES / f"ecr002_val019_{rec['name']}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def profiles(_args: argparse.Namespace) -> None:
    recs = tables52.compared_records()
    values = {
        name: tables52.station_values(name, rec) for name, (rec, _ev) in recs.items()
    }
    for station, profile in backstep.PROFILES.items():
        fig, ax = plt.subplots(figsize=(5.0, 6.5))
        y = np.array(profile.y_over_h)
        for name, (rec, _ev) in recs.items():
            colour, style = RUN_STYLES.get(name, ("0.5", "-"))
            ax.plot(
                values[name][station],
                y,
                style,
                color=colour,
                lw=1.2,
                label=f"{name} ({rec['_class']})",
            )
        ax.plot(
            profile.u_over_uref,
            y,
            "o",
            mfc="none",
            color="black",
            ms=5,
            label="measured (Driver and Seegmiller 1985)",
        )
        ax.axvline(0.0, color="0.7", lw=0.6)
        if station > 0:
            ax.axhline(1.0, color=STEP_COLOUR, lw=0.6, ls=":")
        ax.set_xlabel("u / U_ref")
        ax.set_ylabel("y / H")
        ax.set_ylim(0.0, 9.0)
        ax.set_title(f"VAL-019, x/H = {station:g}")
        ax.legend(fontsize=6.5, loc="lower right")
        ax.grid(True, lw=0.3, alpha=0.5)
        fig.tight_layout()
        path = FIGURES / f"ecr002_val019_profile_x{station:g}.png"
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def walls(_args: argparse.Namespace) -> None:
    recs = tables52.compared_records()
    lines: dict[str, dict[str, backstep.WallLine]] = {}
    for name, (rec, _ev) in recs.items():
        arrays = np.load(OUT / f"{name}.npz")
        config = SimConfig.from_dict(rec["raw"])
        mesh = Mesh(config)
        u_ref = backstep.centreline_speed(mesh, config, arrays["u_faces"])
        shear = backstep.bottom_wall_shear(mesh, config, arrays["u_faces"], arrays["k"])
        lines[name] = {
            "cf": backstep.friction_coefficient(shear, u_ref, config.rho),
            "cp": backstep.pressure_coefficient(
                backstep.wall_pressure(mesh, config, arrays["p"]), u_ref, config.rho
            ),
            "cp_top": backstep.pressure_coefficient(
                backstep.wall_pressure(mesh, config, arrays["p"], "top"),
                u_ref,
                config.rho,
            ),
        }
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for name, (rec, _ev) in recs.items():
        colour, style = RUN_STYLES.get(name, ("0.5", "-"))
        cf = lines[name]["cf"]
        ax.plot(
            cf.x_over_h,
            1000 * cf.values,
            style,
            color=colour,
            lw=1.2,
            label=f"{name} ({rec['_class']})",
        )
    ax.errorbar(
        backstep.CF_STATIONS,
        1000 * np.array(backstep.CF_MEASURED),
        yerr=1000 * np.array(backstep.CF_ERROR),
        fmt="o",
        mfc="none",
        color="black",
        ms=5,
        lw=0.8,
        label="measured",
    )
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.axvline(0.0, color=STEP_COLOUR, lw=0.6, ls=":")
    ax.set_xlim(-10.0, 40.0)
    ax.set_xlabel("x / H")
    ax.set_ylabel("1000 Cf")
    ax.set_title("VAL-019, skin friction along the bottom wall")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    path = FIGURES / "ecr002_val019_cf.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for name, (rec, _ev) in recs.items():
        colour, style = RUN_STYLES.get(name, ("0.5", "-"))
        cp = lines[name]["cp"]
        ax.plot(
            cp.x_over_h,
            cp.values,
            style,
            color=colour,
            lw=1.2,
            label=f"{name} ({rec['_class']})",
        )
        top = lines[name]["cp_top"]
        ax.plot(top.x_over_h, top.values, style, color=colour, lw=0.6, alpha=0.5)
    ax.plot(
        backstep.CP_BOTTOM_STATIONS,
        backstep.CP_BOTTOM_MEASURED,
        "o",
        mfc="none",
        color="black",
        ms=5,
        label="measured, bottom wall",
    )
    ax.plot(
        backstep.CP_TOP_STATIONS,
        backstep.CP_TOP_MEASURED,
        "s",
        mfc="none",
        color="0.5",
        ms=4,
        label="measured, top wall (thin lines: computed)",
    )
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.axvline(0.0, color=STEP_COLOUR, lw=0.6, ls=":")
    ax.set_xlim(-10.0, 40.0)
    ax.set_xlabel("x / H")
    ax.set_ylabel("Cp, shifted to zero at x/H = 37.5")
    ax.set_title("VAL-019, pressure coefficient along the walls")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    path = FIGURES / "ecr002_val019_cp.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def precursor(_args: argparse.Namespace) -> None:
    pre = json.loads((OUT / "precursor_precursor.json").read_text())
    x = [r["x_h"] for r in pre["rows"]]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(
        x,
        [r["delta_99_h"] for r in pre["rows"]],
        "o-",
        color="tab:blue",
        ms=3,
        label="delta_99 / H",
    )
    ax.axhline(backstep.DELTA_99_MEASURED, color="tab:blue", lw=0.6, ls=":")
    ax.set_xlabel("x / H from the inlet")
    ax.set_ylabel("delta_99 / H", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(
        x,
        [r["re_theta"] for r in pre["rows"]],
        "s-",
        color="tab:red",
        ms=3,
        label="Re_theta",
    )
    ax2.axhline(backstep.RE_THETA_MEASURED, color="tab:red", lw=0.6, ls=":")
    ax2.set_ylabel("Re_theta", color="tab:red")
    for key, colour in (
        ("x_delta_99_reaches_1_5h", "tab:blue"),
        ("x_re_theta_reaches_5000", "tab:red"),
    ):
        if pre.get(key) is not None:
            ax.axvline(pre[key], color=colour, lw=0.8, ls="--")
    ax.set_title(
        "The precursor channel: the bottom wall's layer along x (dashed: where each reaches the data's value)"
    )
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    path = FIGURES / "ecr002_val019_precursor.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path.relative_to(ROOT), f"{path.stat().st_size / 1024:.0f} KB")


def everything(args: argparse.Namespace) -> None:
    for name, (rec, _ev) in tables52.compared_records().items():
        if "core" in rec:
            fields(argparse.Namespace(name=name))
    profiles(args)
    walls(args)
    precursor(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("fields")
    p.add_argument("name")
    p.set_defaults(func=fields)
    for fn in (profiles, walls, precursor):
        sub.add_parser(fn.__name__).set_defaults(func=fn)
    sub.add_parser("all").set_defaults(func=everything)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
