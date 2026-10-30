"""Prompt 70's figures: the breach's arrival maps and the jet's half-widths.

    python docs/reports/probe70/fig70.py

Writes docs/reports/probe70/arrival70.png (the 0.1 um arrival map at frame
resolution for the control arm and the largest-viscosity arm, the file's
sensors drawn, cells never reached left blank) and, when the revised jet
records exist, jet70.png (the velocity and scalar half-widths against x
with the fits and the literature band). Reads results/builder70/.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.config import SimConfig  # noqa: E402
from src.io_manager import RunFile  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402

RESULTS = ROOT / "results" / "builder70"
ARMS = (("i05", "5%, 0.1 m (the file's values)"), ("i10_l30", "10%, 0.3 m"))
EVENT_START = 10.0
JET_TAGS = (
    "240x200_h5_c1",
    "120x100_h5_c1",
    "120x200_h10_c0.5",
    "120x100_h5_c1_kumist",
)
MEASURED_BAND = (0.100, 0.110)


def arrival_map(arm: str) -> tuple[Mesh, SimConfig, np.ndarray]:
    mapping = yaml.safe_load(
        (RESULTS / "configs" / f"scenario_filter_breach_{arm}.yaml").read_text(
            encoding="utf-8"
        )
    )
    base = yaml.safe_load(Path(mapping["scenario"]["base"]).read_text(encoding="utf-8"))
    config = SimConfig.from_dict(base)
    mesh = Mesh(config)
    frames = RunFile(RESULTS / f"breach_{arm}.h5").read_frames()
    threshold = config.class_thresholds()[0]
    live = mesh.cell_type != SOLID
    reached = (frames.fields[0] >= threshold) & live[None, :, :]
    ever = reached.any(axis=0)
    first = np.where(
        ever, frames.times[np.clip(reached.argmax(axis=0), 0, None)], np.nan
    )
    return mesh, config, first - EVENT_START


def draw_arrivals() -> Path:
    fig, axes = plt.subplots(len(ARMS), 1, figsize=(12, 4.2 * len(ARMS)))
    for ax, (arm, label) in zip(axes, ARMS, strict=True):
        mesh, config, latency = arrival_map(arm)
        x, y = np.asarray(mesh.x), np.asarray(mesh.y)
        shown = np.ma.masked_invalid(latency)
        pcm = ax.pcolormesh(x, y, shown, cmap="viridis", shading="flat")
        solid = np.ma.masked_where(mesh.cell_type != SOLID, np.ones_like(latency))
        ax.pcolormesh(x, y, solid, cmap="Greys", vmin=0, vmax=1.5, shading="flat")
        for sensor in config.sensors:
            ax.plot(sensor.x, sensor.y, "r^", markersize=9)
            ax.annotate(
                sensor.name,
                (sensor.x, sensor.y),
                xytext=(4, 4),
                textcoords="offset points",
                color="r",
            )
        fig.colorbar(pcm, ax=ax, label="latency from the breach's start (s)")
        ax.set_aspect("equal")
        ax.set_title(
            f"0.1 um arrival map, filter breach on 160x60, supply intensity {label}; blank: never reached"
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.tight_layout()
    out = HERE / "arrival70.png"
    fig.savefig(out, dpi=110)
    return out


def draw_jet() -> Path | None:
    records = [(tag, RESULTS / f"jet_{tag}.json") for tag in JET_TAGS]
    records = [(tag, p) for tag, p in records if p.exists()]
    if not records:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for tag, path in records:
        r = json.loads(path.read_text(encoding="utf-8"))
        b = r["slot_width_m"]
        xb = np.asarray(r["xc"]) / b
        for ax, key, fit_key in (
            (axes[0], "widths_u", "velocity"),
            (axes[1], "widths_c", "scalar"),
        ):
            w = np.asarray(r[key]) / b
            line = ax.plot(
                xb, w, ".", markersize=3, label=f"{tag}: K1 {r[fit_key]['k1']:.4f}"
            )
            k1, k2 = r[fit_key]["k1"], r[fit_key]["k2_slot_widths"]
            lo, hi = r["fit_range_slot_widths"]
            xs = np.array([lo, hi])
            ax.plot(xs, k1 * xs + k2, "-", color=line[0].get_color())
    for ax, title in zip(
        axes, ("velocity half-width", "scalar half-width (Sc_t 0.7)"), strict=True
    ):
        xs = np.array([0.0, 60.0])
        ax.fill_between(
            xs,
            MEASURED_BAND[0] * xs,
            MEASURED_BAND[1] * xs,
            color="0.85",
            label="measured 0.100 to 0.110 (velocity)",
        )
        ax.set_xlabel("x / b")
        ax.set_ylabel("y_1/2 / b")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = HERE / "jet70.png"
    fig.savefig(out, dpi=110)
    return out


LINE_TAGS = ("560x100", "280x50")


def draw_line() -> Path | None:
    """line70.png: the plume's width against distance, model and reference, both grids."""
    records = [(tag, RESULTS / f"line_{tag}.json") for tag in LINE_TAGS]
    records = [(tag, p) for tag, p in records if p.exists()]
    if not records:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, path in records:
        r = json.loads(path.read_text(encoding="utf-8"))
        big_m = r["experiment"]["M_m"]
        x_s = r["source"]["x_m"]
        xm = (np.asarray(r["xc"]) - x_s) / big_m
        sel = xm > 0
        ax.plot(
            xm[sel],
            np.asarray(r["sigma"])[sel] / big_m,
            ".",
            markersize=3,
            label=f"model, {tag}",
        )
        if tag == records[0][0]:
            ax.plot(
                xm[sel],
                np.asarray(r["sigma_ref"])[sel] / big_m,
                "k-",
                label="reference: Anand and Pope's curve with Warhaft's parameters",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("(x - x_s) / M")
    ax.set_ylabel("sigma / M")
    ax.set_title("Line source in decaying grid turbulence: plume width")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = HERE / "line70.png"
    fig.savefig(out, dpi=110)
    return out


def main() -> int:
    print(f"wrote {draw_arrivals()}")
    jet = draw_jet()
    print(f"wrote {jet}" if jet else "no revised jet record yet")
    line = draw_line()
    print(f"wrote {line}" if line else "no line-source record yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
