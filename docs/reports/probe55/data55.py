"""Print the digitised data of Khayrullina et al. (2017) as the tuples validation/impinging_jet.py stores.

Usage:
    python data55.py

Reads results/builder55/khayrullina_2017_digitized.json (digitize55.py)
and prints Python source: CENTRELINE_ROWS (y / h_jet, V_0,y / V0, the
half-extent of the blob in V / V0), INTENSITY_ROWS (y / h_jet, I_uv, True
when the open circle's own centre was read and False when the blob's
centroid stands in), HALF_WIDTH_ROWS (y / w_jet, delta_0.5 / w_jet from the
2 delta_0.5 dots), LATERAL_V_48_ROWS and LATERAL_I_48_ROWS (x / delta_0.5,
the value) at y' / w_jet = 4.8. Nothing is typed into the module; the
test hashes the module's tuples against the values printed here.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECORD = ROOT / "results" / "builder55" / "khayrullina_2017_digitized.json"
# A blob narrower than this many pixels in either direction is a fragment
# of a symbol under another, not a station.
MIN_BLOB = 6
# Stations closer than this in y / h_jet are the same PIV row.
SAME_STATION = 0.006


def rows(name: str, items: list[tuple], fmt: str) -> None:
    print(f"{name}: tuple[tuple[{fmt}], ...] = (")
    for item in items:
        print("    " + repr(item) + ",")
    print(")")


def main() -> None:
    rec = json.loads(RECORD.read_text())
    centre = rec["panels"]["fig4a"]["series"]["stations"]
    rows(
        "CENTRELINE_ROWS",
        [
            (
                round(s["y_over_h"], 4),
                round(s["v_over_v0"], 4),
                round(s["half_extent_x"], 4),
            )
            for s in centre
        ],
        "float, float, float",
    )
    stations = rec["panels"]["fig6"]["series"]["stations"]
    rings: dict[tuple[float, float], None] = {}
    for st in stations:
        if "ring" in st:
            rings[(round(st["ring"]["y_over_h"], 4), round(st["ring"]["i_uv"], 4))] = (
                None
            )
    ring_y = [key[0] for key in rings]
    intensity = [(y, i, True) for y, i in rings]
    for st in stations:
        if "ring" in st or min(st["blob"]["width"], st["blob"]["height"]) < MIN_BLOB:
            continue
        y = round(st["y_over_h"], 4)
        # A blob at a height a ring was read at is the other two Re's
        # symbols beside the circle, not a station of their own.
        if any(abs(y - ry) <= SAME_STATION for ry in ring_y):
            continue
        intensity.append((y, round(st["i_uv"], 4), False))
    intensity.sort()
    rows("INTENSITY_ROWS", intensity, "float, float, bool")
    dots = rec["panels"]["fig8a"]["half_width"]["dots_2delta"]
    rows(
        "HALF_WIDTH_ROWS",
        [(round(d["y_over_w"], 3), round(0.5 * d["half_distance"], 4)) for d in dots],
        "float, float",
    )
    for panel, name in (("fig5a", "LATERAL_V_48_ROWS"), ("fig7a", "LATERAL_I_48_ROWS")):
        curve = rec["panels"][panel]["curves"]["4.8"]
        rows(
            name,
            [
                (round(x, 4), round(v, 4))
                for x, v in zip(curve["x"], curve["value"], strict=True)
            ],
            "float, float",
        )
    cal = {
        k: rec["panels"][k]["calibration"]
        for k in ("fig4a", "fig6", "fig5a", "fig7a", "fig8a")
    }
    for k, c in cal.items():
        print(
            f"# {k}: pixel {c['pixel_in_x']:.4g} x, {c['pixel_in_y']:.4g} y; fit residual {c['x_fit_residual']:.4g}, {c['y_fit_residual']:.4g}"
        )
    print("# checks:", json.dumps(rec["checks"]))


if __name__ == "__main__":
    main()
