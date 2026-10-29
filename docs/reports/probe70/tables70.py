"""Generate tables70.md from prompt 70's probe records (results/builder70/).

    python docs/reports/probe70/tables70.py            # write tables70.md
    python docs/reports/probe70/tables70.py --check    # regenerate and compare

Probe A (jet70.py): jet_<grid>.json. Probe B (run70.sh, configs70.py): the
breach arms' run files breach_<arm>.h5 and their configurations. Every
number in the tables is read from those records; the readings:

- T1: the jet's fits as jet70.py wrote them, beside the literature's values
  (ADR-015 A).
- T2: each arm's inputs, the supply's inlet eddy viscosity C_mu sqrt(k) l_e
  with k = 1.5 (I U)^2 (ADR-012 C's convention), and its flow solve's outer
  count, stop and wall time from /flow/<fingerprint>/.
- T3: each sensor's peak reading over its threshold and its first time at
  or above, from /transport/sensors/ (every step).
- T4: the plume's width on the row of cells holding each sensor height (the
  cell below a face, the sensor rule of ADR-014 C): the number of non-SOLID
  cells of the row at or above the threshold (1% of the breached panel's
  concentration) and at or above ten times it, times dx, at the stored
  frame nearest the breach's end (70 s) and the largest over the frames;
  the x extent of the threshold cells at that frame.
- T5: the arrival map at frame resolution: per class and non-SOLID cell,
  the first stored frame's time at or above the threshold; the fraction of
  non-SOLID cells reached by the end, the earliest arrival (cell and time,
  its latency from the breach's start at 10 s), and the map's value at the
  file's four sensor cells.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.config import SimConfig  # noqa: E402
from src.io_manager import RunFile  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.time_integration import sensor_cells  # noqa: E402

RESULTS = ROOT / "results" / "builder70"
TABLES = HERE / "tables70.md"
# (record tag, label): the confined first version and the revised domain
JET_CASES = (
    ("240x120", "30 b high, coflow 0.05 U_j (confined), 240x120"),
    ("120x60", "30 b high, coflow 0.05 U_j (confined), 120x60"),
    ("240x200_h5_c1", "50 b high, coflow 0.1 U_j, 240x200"),
    ("120x100_h5_c1", "50 b high, coflow 0.1 U_j, 120x100"),
    ("120x200_h10_c0.5", "100 b high, coflow 0.05 U_j, 120x200 (the coflow's share)"),
    (
        "120x100_h5_c1_kumist",
        "50 b high, coflow 0.1 U_j, 120x100, k and eps by UMIST (the scheme's share)",
    ),
)
ARMS = ("i05", "i10", "i10_l30", "i02_l05", "sc02")
ROWS_M = (
    2.5,
    2.05,
    1.5,
    1.2,
)  # the sensor heights and the first row over the tool tops
EVENT_START, EVENT_END = 10.0, 70.0
C_MU = 0.09
LITERATURE = [
    (
        "Measured, plane jets into still air (Gutmark and Wygnanski 1976; Browne et al. 1983; "
        "Thomas and Chu 1989; Thomas and Prakash 1991; Ramaprian and Chandrasekhara 1985)",
        "0.100 to 0.110",
        "-",
    ),
    (
        "Standard k-epsilon, steady RANS similarity solution (Wilcox, as tabulated by Premnath "
        "and Abraham 2005)",
        "0.108",
        "-",
    ),
    ("Standard k-epsilon, unsteady RANS (Magi et al., same table)", "0.103", "-"),
    (
        "Standard k-epsilon, parabolic march, simplified / full production term (Chen and Kim "
        "1987)",
        "0.110 / 0.117",
        "-",
    ),
    (
        "Measured scalar (temperature) to velocity, K1_scalar / K1_u (Stanley and Sarkar 1999, "
        "table 7: Ramaprian and Chandrasekhara, Browne et al., Jenkins and Goldschmidt, Davies "
        "et al.)",
        "-",
        "1.15 to 1.52",
    ),
    (
        "DNS, Re_h 3,000, scalar to velocity (Stanley and Sarkar 1999)",
        "0.093 (velocity)",
        "1.51",
    ),
]


def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "never" if value == np.inf else "none"
    return f"{value:.{digits}f}"


def jet_table() -> list[str]:
    lines = [
        "### T1. The plane jet's spreading rate (probe A, jet70.py), standard k-epsilon on "
        "the product's model and solver keys",
        "",
        "| Case | Cells across the slot | Outer iterations | Stop | Flow wall (s) | "
        "K1 velocity | Virtual origin (slot widths) | Fit RMS residual (slot widths) | "
        "K1 scalar (Sc_t 0.7) | Scalar / velocity | Scalar march: steps, stop, wall (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tag, grid in JET_CASES:
        path = RESULTS / f"jet_{tag}.json"
        if not path.exists():
            lines.append(f"| {grid} | not run | | | | | | | | | |")
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        m = r["scalar_march"]
        lines.append(
            f"| {grid} | {r['cells_across_slot']} | {r['flow']['outer_iterations']} | "
            f"{r['flow']['stop_reason']} | {r['flow']['wall_seconds']:.0f} | "
            f"{r['velocity']['k1']:.4f} | {r['velocity']['virtual_origin_slot_widths']:.2f} | "
            f"{r['velocity']['rms_residual_slot_widths']:.3f} | {r['scalar']['k1']:.4f} | "
            f"{r['scalar_to_velocity_ratio']:.3f} | {m['steps']}, {m['stopped_by']}, "
            f"{m['wall_seconds']:.0f} |"
        )
    lines += [
        "",
        "| Source | K1 velocity | Scalar / velocity |",
        "|---|---|---|",
    ]
    for source, k1, ratio in LITERATURE:
        lines.append(f"| {source} | {k1} | {ratio} |")
    return lines


def arm_inputs(arm: str) -> tuple[SimConfig, dict, float, float, float]:
    mapping = yaml.safe_load(
        (RESULTS / "configs" / f"scenario_filter_breach_{arm}.yaml").read_text(
            encoding="utf-8"
        )
    )
    base = yaml.safe_load(Path(mapping["scenario"]["base"]).read_text(encoding="utf-8"))
    supply = base["boundaries"]["hepa_supply"]
    config = SimConfig.from_dict(base)
    return (
        config,
        base,
        supply["turbulence_intensity"],
        supply["dissipation_length"],
        base["transport"]["turbulent_schmidt"],
    )


def flow_summary(path: Path) -> tuple[int, str, float]:
    with h5py.File(path, "r") as run_file:
        group = run_file["flow"]
        names = list(group)
        state = group[names[0]]
        return (
            int(state.attrs["outer"]),
            str(state.attrs.get("stop_reason", "")),
            float(state.attrs["wall_seconds"]),
        )


def row_index(faces: np.ndarray, position: float) -> int:
    """The cell holding the position, the one below it when it sits on a face."""
    return int(np.searchsorted(faces, position - 1e-9 * faces[-1], side="right") - 1)


def arm_tables() -> list[str]:
    t2 = [
        "### T2. The breach arms on 160x60 (probe B, run70.sh): inputs and the flow solve",
        "",
        "| Arm | Intensity I | Dissipation length l_e (m) | Sc_t | Supply nu_t = C_mu sqrt(k) "
        "l_e (m^2/s) | nu_t / nu | Outer iterations | Stop | Flow wall (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    t3 = [
        "### T3. The sensors: peak reading over threshold and first time at or above, per arm "
        "and class",
        "",
        "| Arm | Class (um) | Sensor | Peak over threshold | First time at or above (s) |",
        "|---|---|---|---|---|",
    ]
    t4 = [
        "### T4. The plume's width on the sensor rows: cells at or above the threshold (1% of "
        "the panel's air) and ten times it, times dx; at the frame nearest 70 s and the "
        "largest over the frames",
        "",
        "| Arm | Class (um) | Row y (m) | Width at 1% at 70 s (m) | Largest width at 1% (m) "
        "| Width at 10% at 70 s (m) | Largest width at 10% (m) | Extent of the 1% cells at "
        "70 s (m) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    t5 = [
        "### T5. The arrival map at frame resolution (every 100 steps): cells reached, the "
        "earliest arrival, and the map at the file's sensors",
        "",
        "| Arm | Class (um) | Non-SOLID cells reached by the end | Earliest arrival: cell "
        "centre (x, y) m, time (s), latency from 10 s | Map at near_door, above_gap_1, "
        "above_gap_2, hood_entry (s) |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        path = RESULTS / f"breach_{arm}.h5"
        if not path.exists():
            t2.append(f"| {arm} | not run | | | | | | | |")
            continue
        config, base, intensity, length, schmidt = arm_inputs(arm)
        mesh = Mesh(config)
        u_in = base["boundaries"]["hepa_supply"]["velocity"]
        k_in = 1.5 * (intensity * u_in) ** 2
        nu_t_in = C_MU * np.sqrt(k_in) * length
        nu = config.mu / config.rho
        outer, stop, wall = flow_summary(path)
        t2.append(
            f"| {arm} | {intensity:.2f} | {length:.2f} | {schmidt:.1f} | {nu_t_in:.2e} | "
            f"{nu_t_in / nu:.0f} | {outer} | {stop} | {wall:.0f} |"
        )
        run = RunFile(path)
        frames = run.read_frames()
        sensors = run.read_sensor_readings()
        thresholds = config.class_thresholds()
        sizes = config.particle_sizes
        cells = sensor_cells(config, mesh)
        live = mesh.cell_type != SOLID
        dx = float(mesh.dx_cell[0])
        xc, yc = np.asarray(mesh.xc), np.asarray(mesh.yc)
        at_end = int(np.argmin(np.abs(frames.times - EVENT_END)))
        for k, threshold in enumerate(thresholds):
            size_um = sizes[k] * 1e6
            for s, name in enumerate(sensors.names):
                series = sensors.readings[:, k, s]
                above = np.flatnonzero(series >= threshold)
                first = float(sensors.times[above[0]]) if above.size else None
                t3.append(
                    f"| {arm} | {size_um:g} | {name} | {series.max() / threshold:.3e} | "
                    f"{fmt(first) if first is not None else 'never'} |"
                )
            field = frames.fields[k]
            for y in ROWS_M:
                j = row_index(np.asarray(mesh.y), y)
                row = field[:, j, :] * live[j][None, :]
                n1 = (row >= threshold).sum(axis=1)
                n10 = (row >= 10 * threshold).sum(axis=1)
                hit = np.flatnonzero(row[at_end] >= threshold)
                extent = (
                    f"{xc[hit[0]] - dx / 2:.2f} to {xc[hit[-1]] + dx / 2:.2f}"
                    if hit.size
                    else "none"
                )
                t4.append(
                    f"| {arm} | {size_um:g} | {y:.2f} | {n1[at_end] * dx:.2f} | "
                    f"{n1.max() * dx:.2f} | {n10[at_end] * dx:.2f} | {n10.max() * dx:.2f} | "
                    f"{extent} |"
                )
            reached = (field >= threshold) & live[None, :, :]
            ever = reached.any(axis=0)
            first_index = np.where(ever, reached.argmax(axis=0), -1)
            arrival = np.where(
                ever, frames.times[np.clip(first_index, 0, None)], np.inf
            )
            fraction = ever[live].mean()
            if ever.any():
                jj, ii = np.unravel_index(
                    np.argmin(np.where(live, arrival, np.inf)), arrival.shape
                )
                earliest = (
                    f"({xc[ii]:.3f}, {yc[jj]:.3f}), {arrival[jj, ii]:.2f}, "
                    f"{arrival[jj, ii] - EVENT_START:.2f}"
                )
            else:
                earliest = "none"
            at_sensors = ", ".join(
                "never" if not np.isfinite(arrival[j, i]) else f"{arrival[j, i]:.2f}"
                for j, i in cells
            )
            t5.append(
                f"| {arm} | {size_um:g} | {fraction * 100:.2f}% | {earliest} | {at_sensors} |"
            )
    return [*t2, "", *t3, "", *t4, "", *t5]


def generate() -> str:
    lines = [
        "# Prompt 70 probe tables",
        "",
        "Generated by `tables70.py` from `results/builder70/` (untracked, this machine); "
        "`--check` regenerates and compares. The readings are in the script's docstring.",
        "",
    ]
    lines += [*jet_table(), "", *arm_tables()]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = generate()
    if args.check:
        current = TABLES.read_text(encoding="utf-8")
        if current != text:
            print("tables70.md differs from the records")
            return 1
        print("tables70.md matches the records")
        return 0
    TABLES.write_text(text, encoding="utf-8")
    print(f"wrote {TABLES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
