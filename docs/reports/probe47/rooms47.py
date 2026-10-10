"""Which room positions each grid represents exactly (prompt 47, section 1's claim).

Usage:
    python rooms47.py [GRID ...]      default: 80x30 160x60 200x75 320x120

Reads configs/clean_room_default.yaml and lists every x and y position it
states: the ends of each boundary segment, each obstacle's edges and heights,
and the sensors' coordinates. For each grid it says which of those lie on a
cell face (a sensor on a cell centre is also exact for the bilinear read)
and prints the count of positions the grid rounds, with the rounded ones
named. Prints markdown and keeps the numbers in
results/builder47/rooms.json.
"""

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "results" / "builder47"
GRIDS = {
    "80x30": (80, 30),
    "160x60": (160, 60),
    "200x75": (200, 75),
    "320x120": (320, 120),
}
TOL = 1e-9


def positions(raw: dict) -> list[tuple[str, str, float]]:
    """Every stated position as (what, axis, value)."""
    out: list[tuple[str, str, float]] = []
    for name, seg in raw["boundaries"].items():
        for key in ("x_start", "x_end", "y_start", "y_end"):
            if key in seg:
                out.append((f"{name}.{key}", key[0], float(seg[key])))
    for o in raw["obstacles"]:
        for key in ("x_start", "x_end", "y_start", "y_end"):
            out.append((f"{o['name']}.{key}", key[0], float(o[key])))
    for s in raw["sensors"]:
        out.append((f"sensor {s['name']}.x", "x", float(s["x"])))
        out.append((f"sensor {s['name']}.y", "y", float(s["y"])))
    return out


def on_face(value: float, spacing: float) -> bool:
    """Whether value is a whole number of spacings from zero."""
    q = value / spacing
    return abs(q - round(q)) < TOL


def main() -> None:
    grids = sys.argv[1:] or list(GRIDS)
    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    width, height = raw["domain"]["width"], raw["domain"]["height"]
    pos = positions(raw)
    kept: dict = {"positions": [list(p) for p in pos], "grids": {}}
    print(
        "| Grid | Cell dx, dy (m) | Positions on faces | Rounded | Rounded positions |"
    )
    print("|---|---|---|---|---|")
    for grid in grids:
        nx, ny = GRIDS[grid]
        dx, dy = width / nx, height / ny
        rounded = []
        for what, axis, value in pos:
            spacing = dx if axis == "x" else dy
            # A sensor is read bilinearly from the centres, so a centre is
            # exact for it as well as a face.
            exact = on_face(value, spacing) or (
                what.startswith("sensor") and on_face(value - 0.5 * spacing, spacing)
            )
            if not exact:
                rounded.append(f"{what} {value:g}")
        kept["grids"][grid] = {
            "dx": dx,
            "dy": dy,
            "positions": len(pos),
            "rounded": rounded,
        }
        print(
            f"| {grid} | {dx:.4g}, {dy:.4g} | {len(pos) - len(rounded)} of {len(pos)} | "
            f"{len(rounded)} | {'; '.join(rounded) or '-'} |"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rooms.json").write_text(json.dumps(kept, indent=1))


if __name__ == "__main__":
    main()
