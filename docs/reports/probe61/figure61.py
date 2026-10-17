"""The step 0b figure: the steady velocity near the rack and the etch chamber, both rooms.

Usage:
    python figure61.py

Reads results/builder61/default_320x120.npz and vents_320x120.npz (flow61.py)
and writes docs/reports/adr014_step0b_vents.png: four panels, the default room
above and the vented room below, the server rack's neighbourhood on the left
and the etch chamber's on the right. Speed at the cell centres is the colour,
one sequential hue on one scale for all four panels; arrows show the
direction on every fourth cell. Obstacles are grey and each vent is drawn as a
thick line on its face with its name.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hash61 import main_results  # noqa: E402

FIGURE = HERE.parent / "adr014_step0b_vents.png"
WINDOWS = (
    ("server rack", (0.6, 3.2), (0.0, 3.0)),
    ("etch chamber", (4.4, 7.0), (0.0, 3.0)),
)
ROOMS = (("default_320x120", "default room"), ("vents_320x120", "vented room"))
SOLID = 1
EVERY = 4
SPEED_MAX = 0.6


def centres(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre velocity from the faces, SOLID cells masked."""
    u, v = data["u_faces"], data["v_faces"]
    uc = 0.5 * (u[:, :-1] + u[:, 1:])
    vc = 0.5 * (v[:-1, :] + v[1:, :])
    solid = data["cell_type"] == SOLID
    return np.ma.masked_where(solid, uc), np.ma.masked_where(solid, vc)


def main() -> None:
    """Draw and save the four panels."""
    results = main_results()
    raw = yaml.safe_load(
        (HERE.parents[2] / "configs" / "clean_room_vents.yaml").read_text(
            encoding="utf-8"
        )
    )
    obstacles = {o["name"]: o for o in raw["obstacles"]}
    vents = {
        name: spec
        for name, spec in raw["boundaries"].items()
        if spec.get("location") == "obstacle"
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    mesh_image = None
    for row, (name, label) in enumerate(ROOMS):
        data = np.load(results / f"{name}.npz")
        xc, yc = data["xc"], data["yc"]
        uc, vc = centres(data)
        speed = np.ma.sqrt(uc**2 + vc**2)
        for col, (title, (x0, x1), (y0, y1)) in enumerate(WINDOWS):
            ax = axes[row, col]
            ix = (xc >= x0) & (xc <= x1)
            iy = (yc >= y0) & (yc <= y1)
            mesh_image = ax.pcolormesh(
                xc[ix],
                yc[iy],
                speed[np.ix_(iy, ix)],
                cmap="Blues",
                vmin=0.0,
                vmax=SPEED_MAX,
                shading="nearest",
            )
            sx = np.where(ix)[0][::EVERY]
            sy = np.where(iy)[0][::EVERY]
            ax.quiver(
                xc[sx],
                yc[sy],
                uc[np.ix_(sy, sx)],
                vc[np.ix_(sy, sx)],
                color="#333333",
                scale=8.0,
                width=0.0025,
                headwidth=4,
            )
            for obs in obstacles.values():
                ax.add_patch(
                    plt.Rectangle(
                        (obs["x_start"], obs["y_start"]),
                        obs["x_end"] - obs["x_start"],
                        obs["y_end"] - obs["y_start"],
                        facecolor="#b8b8b8",
                        edgecolor="#6b6b6b",
                        linewidth=1.0,
                    )
                )
            if name == "vents_320x120":
                for vent_name, spec in vents.items():
                    obs = obstacles[spec["obstacle"]]
                    face = spec["face"]
                    if face in ("top", "bottom"):
                        y = obs["y_end"] if face == "top" else obs["y_start"]
                        xs, ys = (spec["x_start"], spec["x_end"]), (y, y)
                    else:
                        x = obs["x_start"] if face == "left" else obs["x_end"]
                        xs, ys = (x, x), (spec["y_start"], spec["y_end"])
                    if not (x0 <= xs[0] <= x1):
                        continue
                    ax.plot(
                        xs, ys, color="#111111", linewidth=4.0, solid_capstyle="butt"
                    )
                    ax.annotate(
                        vent_name,
                        (np.mean(xs), np.mean(ys)),
                        xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=9,
                        color="#111111",
                        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"},
                    )
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal")
            ax.set_title(f"{label}: near the {title}", fontsize=11)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
    bar = fig.colorbar(mesh_image, ax=axes, shrink=0.7, extend="max")
    bar.set_label("speed at the cell centre (m/s)")
    fig.suptitle(
        "Steady velocity on 320x120, the default room against the room with three vents",
        fontsize=12,
    )
    fig.savefig(FIGURE, dpi=130)
    print(f"written {FIGURE}")


if __name__ == "__main__":
    main()
