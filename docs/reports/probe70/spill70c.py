"""Prompt 70c: where and when the contour cell nearest each sensor is set (review 70 B1).

For each breach arm of probe B (tables70.ARMS: the 160x60 arms and the gate's 320x120
breach), class 0.1 um: the non-SOLID cell ever at or above threshold that lies nearest
each sensor's cell (the argmin of T6's centre-to-centre distance), the first stored frame
at or above threshold in that cell, the bound that frame puts on the time after the
onset at 10 s (one frame spacing wide), and that bound over the supply's Lagrangian time
scale T_L = (4 / (3 C_0)) k / eps at the arm's inlet values, with C_0 as the reference
curve of line70.py has it. The cell's offset from the nearest obstacle (beyond its
edge, above its top) shows where in the room it lies. The row station is t / T_L times
the line-source row's T_L U / M, which is the distance along the row's axis that has the
same t / T_L (prompt 70d: the figures in the ADR's prose come from the ranges printed
under the table).

    python docs/reports/probe70/spill70c.py            # write spill70c.md
    python docs/reports/probe70/spill70c.py --check    # regenerate and compare
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import line70  # noqa: E402
from tables70 import ARMS, EVENT_START, arm_inputs  # noqa: E402

from src.io_manager import RunFile  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.time_integration import sensor_cells  # noqa: E402

OUT = HERE / "spill70c.md"
# T_L U / M of the line-source row: k / eps at its source is t_o / m, so this is
# (4 / (3 C_0)) x_o / m mesh lengths per Lagrangian time, whatever M is.
ROW_M_PER_T_L = 4.0 / (3.0 * line70.C_0) * line70.X_O_OVER_M / line70.M_EXP
# The fall from the ceiling to the floor returns, s: 3 m of room at the supply's 0.45 m/s
# is 6.7 s, which the ADR rounds to 7.
RETURN_FALL_S = 7.0


def offsets(config, x: float, y: float) -> tuple[str, float, float]:
    """The nearest obstacle to a point, how far beyond its edge, how far above its top.

    Parameters
    ----------
    config : SimConfig
        Supplies the obstacles' rectangles.
    x, y : float
        The point in metres.

    Returns
    -------
    tuple[str, float, float]
        The obstacle's name, the horizontal distance outside its x range (zero over
        it) and the height above its top (negative below it).
    """
    best = None
    for obs in config.obstacles:
        dx = max(obs.x_start - x, 0.0, x - obs.x_end)
        dy = max(obs.y_start - y, 0.0, y - obs.y_end)
        if best is None or np.hypot(dx, dy) < best[0]:
            best = (float(np.hypot(dx, dy)), obs.name, dx, y - obs.y_end)
    return best[1], best[2], best[3]


def read_arm(arm: str, scenario: Path, path: Path) -> dict:
    """Read one arm's run file: the nearest reached cell of each sensor and its timing.

    Parameters
    ----------
    arm : str
        The arm's name (tables70.ARMS).
    scenario : Path
        The scenario file holding the arm's inputs.
    path : Path
        The arm's run file.

    Returns
    -------
    dict
        The arm's grid, supply T_L, frame spacing, one row per sensor and the
        row just above the tool tops.
    """
    config, base, intensity, length, _schmidt = arm_inputs(scenario)
    mesh = Mesh(config)
    live = mesh.cell_type != SOLID
    xc, yc = np.asarray(mesh.xc), np.asarray(mesh.yc)
    run = RunFile(path)
    frames = run.read_frames()
    names = run.read_sensor_readings().names
    cells = sensor_cells(config, mesh)
    threshold = config.class_thresholds()[0]
    reached = (frames.fields[0] >= threshold) & live[None]
    ever = reached.any(axis=0)
    first = np.where(ever, reached.argmax(axis=0), -1)
    spacing = float(np.median(np.diff(frames.times)))
    u_in = base["boundaries"]["hepa_supply"]["velocity"]
    k_in = 1.5 * (intensity * u_in) ** 2
    eps_in = k_in**1.5 / length
    t_l = 4.0 / (3.0 * line70.C_0) * k_in / eps_in
    reached_j, reached_i = np.nonzero(ever)
    rows = []
    for s, (j, i) in enumerate(cells):
        d2 = (xc[reached_i] - xc[i]) ** 2 + (yc[reached_j] - yc[j]) ** 2
        m = int(np.argmin(d2))
        jj, ii = reached_j[m], reached_i[m]
        t_first = float(frames.times[first[jj, ii]])
        hi = t_first - EVENT_START
        lo = hi - spacing
        obstacle, beyond, above = offsets(config, float(xc[ii]), float(yc[jj]))
        rows.append(
            {
                "sensor": names[s],
                "sensor_xy": (float(xc[i]), float(yc[j])),
                "cell_xy": (float(xc[ii]), float(yc[jj])),
                "distance": float(np.sqrt(d2[m])),
                "obstacle": obstacle,
                "beyond": beyond,
                "above": above,
                "t_first": t_first,
                "lo": lo,
                "hi": hi,
                "tl_lo": lo / t_l,
                "tl_hi": hi / t_l,
                "row_lo": lo / t_l * ROW_M_PER_T_L,
                "row_hi": hi / t_l * ROW_M_PER_T_L,
            }
        )
    top = None
    top_rows = np.flatnonzero((yc > 2.0) & (yc < 2.1))
    if top_rows.size:
        jt = top_rows[0]
        hit = np.flatnonzero(first[jt] >= 0)
        if hit.size:
            top = (
                float(yc[jt]),
                float(xc[hit[0]]),
                float(xc[hit[-1]]),
                float(frames.times[first[jt][hit].min()]),
            )
    return {
        "arm": arm,
        "grid": f"{config.nx}x{config.ny}",
        "t_l": t_l,
        "spacing": spacing,
        "rows": rows,
        "top": top,
    }


def span(values: list[float], nd: int = 2) -> str:
    """The range of the values as 'a to b' with nd decimals."""
    return f"{min(values):.{nd}f} to {max(values):.{nd}f}"


def summarise(label: str, rows: list[dict]) -> str:
    """One line of ranges over a group of rows (the figures the ADR's prose quotes)."""
    return (
        f"- {label} ({len(rows)} rows): {span([r['beyond'] for r in rows])} m beyond the "
        f"obstacle's edge, {span([r['above'] for r in rows])} m above its top (negative: below it); "
        f"{min(r['lo'] for r in rows):.2f} to {max(r['hi'] for r in rows):.2f} s after "
        f"onset; t / T_L {span([r['tl_lo'] for r in rows] + [r['tl_hi'] for r in rows])}; "
        f"row station {min(r['row_lo'] for r in rows):.0f} to "
        f"{max(r['row_hi'] for r in rows):.0f} M."
    )


def build_text() -> str:
    """Render spill70c.md from the arms' run files.

    Returns
    -------
    str
        T11 (one row per arm and sensor), the ranges over its groups, the fall to
        the floor returns at each arm's T_L and the row above the tool tops.
    """
    arms = []
    for arm, scenario, path in ARMS:
        if not path.exists():
            raise FileNotFoundError(2, "run file missing", str(path))
        arms.append(read_arm(arm, scenario, path))
    lines = [
        "# Prompt 70c: where the contour cell nearest each sensor is set",
        "",
        "Generated by `spill70c.py` from the breach arms' run files (`results/builder70/`, "
        "untracked, this machine; `--check` regenerates and compares). The reading is in the "
        "script's docstring.",
        "",
        "### T11. For each breach arm and sensor, class 0.1 um: the cell nearest the sensor "
        "that is ever at or above threshold (the cell T6's distance is measured to), its "
        "offset from the nearest obstacle, the first stored frame at or above threshold "
        "there, the bound that frame puts on the time after the 10 s onset (one frame "
        "spacing wide), that bound over the arm's supply T_L, and the row station with the "
        f"same t / T_L (T_L U / M = {ROW_M_PER_T_L:.1f} M per Lagrangian time)",
        "",
        "| Arm (grid; T_L at its supply) | Sensor (cell centre, m) | Nearest contour cell (m) "
        "| Distance (m) | Beyond the nearest obstacle's edge, above its top (m) | First frame "
        "at or above (s) | After onset (s) | t / T_L | Row station (M) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        for r in a["rows"]:
            sx, sy = r["sensor_xy"]
            cx, cy = r["cell_xy"]
            lines.append(
                f"| {a['arm']} ({a['grid']}; {a['t_l']:.2f} s) | {r['sensor']} "
                f"({sx:.3f}, {sy:.3f}) | ({cx:.3f}, {cy:.3f}) | {r['distance']:.3f} | "
                f"{r['beyond']:.2f}, {r['above']:.2f} ({r['obstacle']}) | {r['t_first']:.2f} | "
                f"{r['lo']:.2f} to {r['hi']:.2f} | {r['tl_lo']:.2f} to {r['tl_hi']:.2f} | "
                f"{r['row_lo']:.0f} to {r['row_hi']:.0f} |"
            )
    every = [r for a in arms for r in a["rows"]]
    gap = [r for r in every if r["sensor"].startswith("above_gap")]
    other = [r for r in every if not r["sensor"].startswith("above_gap")]
    file_inputs = [
        r for a in arms if a["arm"] in ("i05", "gate_320x120") for r in a["rows"]
    ]
    file_gap = [r for r in file_inputs if r["sensor"].startswith("above_gap")]
    lines += [
        "",
        "Ranges over the table's rows (frame bounds, so each time is one frame spacing wide: "
        + ", ".join(
            f"{a['spacing']:.2f} s on {a['grid']}" for a in arms[:1] + arms[-1:]
        )
        + "):",
        "",
        summarise("The gap sensors, every arm", gap),
        summarise("The gap sensors, the file's inputs (i05 and the gate)", file_gap),
        summarise("The door and hood sensors, every arm", other),
        summarise("All sensors, every arm", every),
        "",
        f"The fall to the floor returns ({RETURN_FALL_S:g} s) at each arm's supply T_L, "
        "t / T_L and the row station with the same t / T_L: "
        + "; ".join(
            f"{a['arm']} {RETURN_FALL_S / a['t_l']:.1f} ({RETURN_FALL_S / a['t_l'] * ROW_M_PER_T_L:.0f} M)"
            for a in arms
        )
        + ".",
        "",
        "The row of cells just above the tool tops: the x range reached and the earliest frame "
        "at or above threshold on it:",
        "",
    ]
    for a in arms:
        if a["top"] is not None:
            y, x0, x1, t = a["top"]
            lines.append(
                f"- {a['arm']} (y {y:.3f}): x {x0:.2f} to {x1:.2f} m; earliest frame {t:.2f} s."
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write spill70c.md, or compare it with the records under --check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="regenerate and compare")
    args = parser.parse_args()
    try:
        text = build_text()
    except FileNotFoundError as err:
        print(f"probe record missing: {err.filename}")
        return 1
    if args.check:
        if OUT.read_text(encoding="utf-8") != text:
            print("spill70c.md differs from the records")
            return 1
        print("spill70c.md matches the records")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
