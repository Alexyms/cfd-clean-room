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
# (label, scenario mapping, run file): the five 160x60 arms, then the gate's 320x120 breach
# (prompt 69, the file's inputs) read the same way for the grid's effect (review 70 B3).
B69 = ROOT / "results" / "builder69"
ARMS = (
    *(
        (
            name,
            RESULTS / "configs" / f"scenario_filter_breach_{name}.yaml",
            RESULTS / f"breach_{name}.h5",
        )
        for name in ("i05", "i10", "i10_l30", "i02_l05", "sc02")
    ),
    (
        "gate_320x120",
        B69 / "configs" / "scenario_filter_breach.yaml",
        B69 / "filter_breach.h5",
    ),
)
REUSED = {"sc02": "i05"}  # arms whose flow state --flows-from copied (review 70 S10)
# A sensor reading below this fraction of its threshold is the numerical tail of a sharp
# front, set by the grid more than by any input (review 70 B3), and is reported as
# "below the resolved plume", not as a number.
RESOLVED_FLOOR = 1e-2
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
        "DNS, Re_h 3,000, scalar case, scalar to velocity (Stanley and Sarkar 1999, "
        "table 7: K1u 0.105, K1s 0.158)",
        "0.105 (velocity, the scalar case; 0.093 in the main case)",
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


def arm_inputs(scenario: Path) -> tuple[SimConfig, dict, float, float, float]:
    mapping = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    base_path = Path(mapping["scenario"]["base"])
    if not base_path.is_absolute():
        base_path = scenario.parent / base_path
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
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
        "### T2. The breach arms on 160x60 (probe B, run70.sh) and the gate's 320x120 breach "
        "(prompt 69): inputs and the flow solve",
        "",
        "| Arm | Grid | Intensity I | Dissipation length l_e (m) | Sc_t | Supply nu_t = C_mu "
        "sqrt(k) l_e (m^2/s) | nu_t / nu | Outer iterations | Stop | Flow wall (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    t3 = [
        "### T3. The sensors: peak reading over threshold and first time at or above, per arm "
        f"and class; a peak below {RESOLVED_FLOOR:g} of threshold is the front's numerical "
        "tail and is reported as below the resolved plume",
        "",
        "| Arm | Class (um) | Sensor | Peak over threshold | First time at or above (s) |",
        "|---|---|---|---|---|",
    ]
    t6 = [
        "### T6. The plume's edge against the sensors: the distance from each sensor's cell to "
        "the nearest non-SOLID cell ever at or above the threshold (centre to centre), in metres "
        "and in cells, per arm and class",
        "",
        "| Arm | Grid | Class (um) | Sensor | Distance to the threshold contour (m) | In cells | "
        "Sensor cell reached |",
        "|---|---|---|---|---|---|---|",
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
        "### T5. The arrival map at frame resolution: cells reached, the earliest arrival, and "
        "the map at the file's sensors. A frame is stored every 100 steps, so every time here "
        "is the first stored frame at or after the crossing, an upper bound by up to one frame "
        "spacing (the spacing is in the column); section D's in-march maps remove that bound",
        "",
        "| Arm | Class (um) | Non-SOLID cells reached by the end | Earliest arrival: cell "
        "centre (x, y) m, first frame at or above (s), its bound on the latency from 10 s; "
        "frame spacing (s) | Map at near_door, above_gap_1, above_gap_2, hood_entry (s, first "
        "frame) |",
        "|---|---|---|---|---|",
    ]
    for arm, scenario, path in ARMS:
        if not path.exists():
            t2.append(f"| {arm} | not run | | | | | | | | |")
            continue
        config, base, intensity, length, schmidt = arm_inputs(scenario)
        mesh = Mesh(config)
        grid = f"{config.nx}x{config.ny}"
        u_in = base["boundaries"]["hepa_supply"]["velocity"]
        k_in = 1.5 * (intensity * u_in) ** 2
        nu_t_in = C_MU * np.sqrt(k_in) * length
        nu = config.mu / config.rho
        outer, stop, wall = flow_summary(path)
        solve = f"reused from {REUSED[arm]}: {outer}" if arm in REUSED else f"{outer}"
        t2.append(
            f"| {arm} | {grid} | {intensity:.2f} | {length:.2f} | {schmidt:.1f} | "
            f"{nu_t_in:.2e} | {nu_t_in / nu:.0f} | {solve} | {stop} | {wall:.0f} |"
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
        spacing = float(np.median(np.diff(frames.times)))
        for k, threshold in enumerate(thresholds):
            size_um = sizes[k] * 1e6
            for s, name in enumerate(sensors.names):
                series = sensors.readings[:, k, s]
                above = np.flatnonzero(series >= threshold)
                first = float(sensors.times[above[0]]) if above.size else None
                margin = series.max() / threshold
                peak = (
                    f"{margin:.3e}"
                    if margin >= RESOLVED_FLOOR
                    else f"below the resolved plume (< {RESOLVED_FLOOR:g})"
                )
                t3.append(
                    f"| {arm} | {size_um:g} | {name} | {peak} | "
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
                    f"at most {arrival[jj, ii] - EVENT_START:.2f}; {spacing:.3f}"
                )
                reached_j, reached_i = np.nonzero(ever & live)
                for s, (j, i) in enumerate(cells):
                    d2 = (xc[reached_i] - xc[i]) ** 2 + (yc[reached_j] - yc[j]) ** 2
                    distance = float(np.sqrt(d2.min()))
                    t6.append(
                        f"| {arm} | {grid} | {size_um:g} | {sensors.names[s]} | "
                        f"{distance:.3f} | {distance / dx:.1f} | "
                        f"{'yes' if ever[j, i] else 'no'} |"
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
    return [*t2, "", *t3, "", *t4, "", *t5, "", *t6]


LINE_GRIDS = ("280x50", "560x100")


def line_table() -> list[str]:
    """T7: the line source in grid turbulence (line70.py), both grids."""
    lines = [
        "### T7. The line source in decaying grid turbulence (prompt 70b, line70.py, the first probe at "
        "M = 2.54 cm as Viswanathan and Pope tabulate it; the rerun at Warhaft's 2.5 cm is T9's): the "
        "model's decay exponent and the plume's width against the reference curve "
        "(Anand and Pope's displacement variance with Warhaft's parameters, Viswanathan and "
        "Pope 2008 equations 11 to 15) at five stations downstream of the source",
        "",
        "| Grid | Cell (mm) | Plume width at the source column (mm) | Outer iterations | Stop | Flow wall "
        "(s) | n (model's own 1.087) | Scalar march: steps, stop, wall (s) | sigma / sigma_ref "
        "at 5 M (t / T_L 0.2) | 10 M (0.4) | 20 M (0.9) | 50 M (2.1) | 100 M (4.2) | sigma "
        "model, ref at 100 M (mm) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for grid in LINE_GRIDS:
        path = RESULTS / f"line_{grid}.json"
        if not path.exists():
            lines.append(f"| {grid} | not run | | | | | | | | | | | | |")
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        st, m = r["stations"], r["scalar_march"]
        ratios = " | ".join(
            f"{st[k]['ratio']:.3f}" for k in ("5", "10", "20", "50", "100")
        )
        lines.append(
            f"| {grid} | {r['dx_m'] * 1e3:.0f} | {r['sigma'][r['source']['column']] * 1e3:.2f} | "
            f"{r['flow']['outer_iterations']} | {r['flow']['stop_reason']} | "
            f"{r['flow']['wall_seconds']:.0f} | {r['decay']['n_model']:.4f} | {m['steps']}, "
            f"{m['stopped_by']}, {m['wall_seconds']:.0f} | {ratios} | "
            f"{st['100']['sigma_model_m'] * 1e3:.1f}, {st['100']['sigma_ref_m'] * 1e3:.1f} |"
        )
    return lines


def generate() -> str:
    lines = [
        "# Prompt 70 probe tables",
        "",
        "Generated by `tables70.py` from `results/builder70/` (untracked, this machine); "
        "`--check` regenerates and compares. The readings are in the script's docstring.",
        "",
    ]
    lines += [*jet_table(), "", *arm_tables(), "", *line_table()]
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
