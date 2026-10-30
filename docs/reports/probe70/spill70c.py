"""Prompt 70c: where and when the contour cell nearest each sensor is set (review 70 B1).

For each breach arm of probe B (tables70.ARMS: the 160x60 arms and the gate's 320x120
breach), class 0.1 um: the non-SOLID cell ever at or above threshold that lies nearest
each sensor's cell (the argmin of T6's centre-to-centre distance), the first stored frame
at or above threshold in that cell, the bound that frame puts on the time after the
onset at 10 s (one frame spacing wide), and that bound over the supply's Lagrangian time
scale T_L = (4 / (3 C_0)) k / eps at the arm's inlet values, with C_0 = 2.1 as the
reference curve of line70.py has it. The tool-top row names where the plume lands.

    python docs/reports/probe70/spill70c.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from tables70 import ARMS, EVENT_START, arm_inputs  # noqa: E402

from src.io_manager import RunFile  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.time_integration import sensor_cells  # noqa: E402

C_0 = 2.1
ROW_M_PER_T_L = 0.08557 * 7.0 / 0.0254  # the line-source row: T_L U / M, mesh lengths


def main() -> int:
    for arm, scenario, path in ARMS:
        if not path.exists():
            print(f"{arm}: {path} missing")
            continue
        config, base, intensity, length, _schmidt = arm_inputs(scenario)
        mesh = Mesh(config)
        live = mesh.cell_type != SOLID
        xc, yc = np.asarray(mesh.xc), np.asarray(mesh.yc)
        run = RunFile(path)
        frames = run.read_frames()
        sensors = run.read_sensor_readings()
        cells = sensor_cells(config, mesh)
        threshold = config.class_thresholds()[0]
        reached = (frames.fields[0] >= threshold) & live[None]
        ever = reached.any(axis=0)
        first = np.where(ever, reached.argmax(axis=0), -1)
        spacing = float(np.median(np.diff(frames.times)))
        u_in = base["boundaries"]["hepa_supply"]["velocity"]
        k_in = 1.5 * (intensity * u_in) ** 2
        eps_in = k_in**1.5 / length
        t_l = 4.0 / (3.0 * C_0) * k_in / eps_in
        print(
            f"\n{arm} {config.nx}x{config.ny}: I {intensity} l_e {length} m, k / eps "
            f"{k_in / eps_in:.2f} s, T_L {t_l:.2f} s; frame spacing {spacing:.3f} s"
        )
        reached_j, reached_i = np.nonzero(ever)
        for s, (j, i) in enumerate(cells):
            d2 = (xc[reached_i] - xc[i]) ** 2 + (yc[reached_j] - yc[j]) ** 2
            m = int(np.argmin(d2))
            jj, ii = reached_j[m], reached_i[m]
            t_first = float(frames.times[first[jj, ii]])
            hi = t_first - EVENT_START
            lo = hi - spacing
            print(
                f"  {sensors.names[s]:12s} sensor ({xc[i]:.3f}, {yc[j]:.3f}) nearest reached "
                f"({xc[ii]:.3f}, {yc[jj]:.3f}) at {np.sqrt(d2[m]):.3f} m; first frame "
                f"{t_first:.2f} s; after onset {lo:.2f} to {hi:.2f} s; t / T_L {lo / t_l:.2f} "
                f"to {hi / t_l:.2f}; row station {lo / t_l * ROW_M_PER_T_L:.0f} to "
                f"{hi / t_l * ROW_M_PER_T_L:.0f} M"
            )
        top_rows = np.flatnonzero((yc > 2.0) & (yc < 2.1))
        if top_rows.size:
            jt = top_rows[0]
            hit = np.flatnonzero(first[jt] >= 0)
            if hit.size:
                print(
                    f"  row just above the tool tops (y {yc[jt]:.3f}): reached x "
                    f"{xc[hit[0]]:.2f} to {xc[hit[-1]]:.2f} m; earliest frame "
                    f"{frames.times[first[jt][hit].min()]:.2f} s"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
