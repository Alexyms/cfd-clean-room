"""Prompt 80: the cumulative counts' tables, from the run files (no new physics).

Reads prompt 79's five 320x120 runs (results/builder79/product/), its three 80x30
placement runs (results/builder79/placement/), this pass's probe reruns of them
(results/builder80/placement/, made by run80.sh) and the 80x30 clean-out of prompt 79's
sweep (its configuration only, for the candidates). All untracked and on this machine.
Writes tables80.md and the copies in docs/reports/cumulative_counts.md.

    python docs/reports/probe80/tables80.py            # write both
    python docs/reports/probe80/tables80.py --check    # regenerate and compare

The reading of the classes (the report states it): each class stands for the particles
nearest its size, so channel k, a counter's count at or above class k's size, is the sum
of class k and every larger class, compared with class k's threshold. The per-class
reading is class k alone against the same threshold. Both are made by probe80.channel.

On 320x120 both readings' maps are rebuilt from the stored frames (probe80.frame_maps),
so the two are compared at one resolution; the layouts are scored and searched on those
maps with placement.score and placement.search, and the layouts of record (searched on the
march's /arrival/ maps, as prompt 79 re-scored them) are scored on them too. The bracket
on a cumulative latency is the last frame before its first frame at or above (or the
onset) below, and above it the earlier of that frame and class k's own step-resolved
arrival. On 80x30 both readings are step-resolved (probe80's run).
"""

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import probe80  # noqa: E402 -- follows sys.path.insert
from probe80 import COUNTINGS, CUMULATIVE, PER_CLASS  # noqa: E402

from src.config import (  # noqa: E402
    MEAN,
    MOUNT_FREE,
    MOUNT_STAND,
    WORST_CASE,
    SimConfig,
)
from src.io_manager import RunFile  # noqa: E402
from src.mesh import SOLID, Mesh  # noqa: E402
from src.monitor import contaminating_events, sensor_cells  # noqa: E402
from src.placement import (  # noqa: E402
    FRAME_RESOLVED,
    STEP_RESOLVED,
    ClearScore,
    ClearUnits,
    Layout,
    RunArrival,
    Units,
    arrival_from_frames,
    arrival_from_run,
    candidates,
    clear_score,
    clear_times,
    detection_units,
    given_layout,
    rank,
    scenario_end,
    search,
)

PRODUCT_RUNS = ROOT / "results" / "builder79" / "product"
PLACEMENT_79 = ROOT / "results" / "builder79" / "placement"
PLACEMENT_80 = ROOT / "results" / "builder80" / "placement"
COARSE_CLEAN_OUT = ROOT / "results" / "builder79" / "sweep" / "clean_out_80x30.h5"
PRODUCT = ROOT / "configs" / "clean_room_default.yaml"
TABLES = HERE / "tables80.md"
REPORT = ROOT / "docs" / "reports" / "cumulative_counts.md"
RESCORED76 = ROOT / "docs" / "reports" / "probe79" / "rescored76.md"
TEST_XML = ROOT / "results" / "builder80" / "test80.xml"
TEST_PLANT_XML = ROOT / "results" / "builder80" / "test80_plant.xml"

EVENT_NAMES = ("door_leak", "filter_breach", "equipment_dust", "seal_leak")
FINE_NAMES = (*EVENT_NAMES, "clean_out")
COARSE_STEMS = {
    "door_leak": "door",
    "filter_breach": "breach",
    "equipment_dust": "dust",
}
TITLES = {
    "door_leak": "door",
    "filter_breach": "breach",
    "equipment_dust": "dust",
    "seal_leak": "seal leak",
    "clean_out": "clean-out",
}
WITHIN = 10.0  # the coverage rule's time, the latency report's (prompt 76)
FILE = "the file's four"
SURF = "surface search"
STAND = "product layout"
FREE = "free optimum"
LAYOUT_NAMES = (FILE, SURF, STAND, FREE)
READING = {PER_CLASS: "per-class", CUMULATIVE: "cumulative"}
RULES = (WORST_CASE, MEAN)
# The record's layouts, prompt 79's re-score of prompt 76c (rescored76.md L1), to
# confirm the step-resolved searches here reproduce them.
RECORD_PLACES = {
    SURF: "(1.463, 0.013), (1.488, 0.013), (3.188, 2.038), (3.738, 2.038)",
    STAND: "(0.138, 0.113), (0.062, 0.562), (3.188, 2.038), (3.738, 2.038)",
    FREE: "(0.062, 0.013), (0.062, 0.562), (3.738, 2.038), (3.513, 2.938)",
}
FRAME_PREDICTION_B = 0.36  # (b)'s one frame, seconds, as the prompt states it
FALL_D = 0.10  # (d)'s 10%
# The standoffs prompt 76 read the door at (tables76.DOOR_STANDOFFS).
DOOR_STANDOFFS = (0.05, 0.1, 0.2, 0.5, 1.0)
C2_PRODUCT = "**the product search, four with stands**"


# -- Text ---------------------------------------------------------------------


def where(mesh: Mesh, cell: tuple[int, int]) -> tuple[float, float]:
    """A cell's centre (x, y), metres."""
    j, i = cell
    return float(mesh.xc[i]), float(mesh.yc[j])


def places(mesh: Mesh, cells: tuple[tuple[int, int], ...]) -> str:
    """Cell centres as text."""
    return ", ".join("({:.3f}, {:.3f})".format(*where(mesh, cell)) for cell in cells)


def t3(value: float) -> str:
    """A time to the millisecond, or inf."""
    return "inf" if math.isinf(value) else f"{value:.3f}"


def label_text(sizes: list[float], label: tuple[str, str, int]) -> str:
    """A unit as text: scenario, channel."""
    name, _, k = label
    return f"{TITLES[name]}, {sizes[k] * 1e6:g} um"


def order_text(layouts: dict[str, Layout]) -> str:
    """The layouts by the worst case alone, best first, ties said."""
    ranked = sorted(layouts, key=lambda n: layouts[n].score.worst)
    out = ranked[0]
    for before, name in pairwise(ranked):
        same = layouts[before].score.worst == layouts[name].score.worst
        out += (" = " if same else " < ") + name
    return out


def rank_text(layouts: dict[str, Layout]) -> str:
    """Section D's ranking: the worst case, ties by the mean."""
    names = {id(layout): name for name, layout in layouts.items()}
    return ", ".join(names[id(layout)] for layout in rank(list(layouts.values())))


# -- The grids' data ------------------------------------------------------------


class Grid:
    """One grid's runs, both readings' units, the candidates and the searches.

    Parameters
    ----------
    title : str
        The grid, as text.
    paths : dict[str, Path]
        Scenario to the run file read.
    configs : list[SimConfig]
        Every configuration whose openings and sources exclude candidates.
    arrivals : dict[str, dict[str, RunArrival]]
        Counting to scenario to the maps scored.
    record : dict[str, RunArrival] or None
        The march's per-class maps (/arrival/), the layouts of record's basis,
        when the maps scored are frame-resolved.
    """

    def __init__(
        self,
        title: str,
        paths: dict[str, Path],
        configs: list[SimConfig],
        arrivals: dict[str, dict[str, RunArrival]],
        record: dict[str, RunArrival] | None,
    ) -> None:
        self.title = title
        self.paths = paths
        self.config = configs[0]
        self.mesh = Mesh(self.config)
        self.sizes = list(self.config.particle_sizes)
        self.spec = SimConfig(PRODUCT).placement
        assert self.spec is not None and self.spec.all_clear == 0
        spec = self.spec
        surfaces = tuple(m for m in spec.mount if m != MOUNT_STAND)
        self.pools = {
            SURF: candidates(self.mesh, configs, surfaces, spec.standoff),
            STAND: candidates(
                self.mesh,
                configs,
                spec.mount,
                spec.standoff,
                stand_height=spec.stand_height,
            ),
            FREE: candidates(self.mesh, configs, (MOUNT_FREE,), spec.standoff),
        }
        self.sensors = sensor_cells(self.config, self.mesh)
        self.arrivals = arrivals
        self.units = {c: detection_units(arrivals[c]) for c in COUNTINGS}
        assert self.units[PER_CLASS].labels == self.units[CUMULATIVE].labels
        self.labels = self.units[PER_CLASS].labels
        # found[(basis, rule)][name]: the layout searched on BASIS's maps.
        self.found: dict[tuple[str, str], dict[str, Layout]] = {}
        bases: dict[str, Units] = dict(self.units)
        if record is not None:
            self.record_units = detection_units(record)
            bases["record"] = self.record_units
        for basis, units in bases.items():
            for rule in RULES:
                self.found[(basis, rule)] = {
                    name: search(units, pool, spec.count, rule, within=WITHIN)
                    for name, pool in self.pools.items()
                }

    def scored(self, counting: str, cells: tuple[tuple[int, int], ...]) -> Layout:
        """A layout scored on one reading's maps over every unit."""
        return given_layout(self.units[counting], cells, within=WITHIN)

    def layouts(self, basis: str, rule: str) -> dict[str, tuple[tuple[int, int], ...]]:
        """The four layouts' cells, searched on BASIS under RULE."""
        found = self.found[(basis, rule)]
        return {
            FILE: self.sensors,
            SURF: found[SURF].cells,
            STAND: found[STAND].cells,
            FREE: found[FREE].cells,
        }


def events_of(config: SimConfig) -> tuple[tuple[str, ...], np.ndarray]:
    """The plan's contaminating events: names and onsets."""
    events = contaminating_events(config.scenario)
    return tuple(n for n, _ in events), np.array([o for _, o in events])


def config_of(path: Path) -> SimConfig:
    """The merged configuration a run file records."""
    return SimConfig.from_dict(RunFile(path).read_config().mapping)


def clear_units_of(
    paths: dict[str, Path],
    configs: dict[str, SimConfig],
    mesh: Mesh,
    thresholds: np.ndarray,
) -> dict[str, ClearUnits]:
    """Both readings' all-clear units, as ``placement.clear_units`` builds them."""
    live = mesh.cell_type != SOLID
    areas = np.outer(mesh.dy_cell, mesh.dx_cell)
    keys = ("clear", "ends", "times", "fractions", "gap", "labels")
    parts: dict[str, dict[str, list]] = {
        c: {key: [] for key in keys} for c in COUNTINGS
    }
    for name, path in paths.items():
        end = scenario_end(configs[name])
        if end is None:
            continue
        clear, stamps, share = probe80.frame_clear(path, thresholds, live, areas, end)
        gap = float(np.diff(stamps).max()) if stamps.size > 1 else 0.0
        for c in COUNTINGS:
            for k in range(thresholds.size):
                parts[c]["clear"].append(clear[c][k])
                parts[c]["ends"].append(end)
                parts[c]["times"].append(stamps)
                parts[c]["fractions"].append(share[c][k])
                parts[c]["gap"].append(gap)
                parts[c]["labels"].append((name, k))
    return {
        c: ClearUnits(
            np.stack(q["clear"]),
            np.array(q["ends"]),
            tuple(q["times"]),
            tuple(q["fractions"]),
            np.array(q["gap"]),
            tuple(q["labels"]),
        )
        for c, q in parts.items()
    }


class Fine:
    """320x120: both readings' frame maps, the march's maps and the sensors' steps."""

    def __init__(self) -> None:
        self.paths = {name: PRODUCT_RUNS / f"{name}.h5" for name in FINE_NAMES}
        self.configs = {name: config_of(p) for name, p in self.paths.items()}
        first = self.configs["door_leak"]
        self.mesh = Mesh(first)
        self.live = self.mesh.cell_type != SOLID
        self.thresholds = np.array(first.class_thresholds("prompt 80"))
        self.frames: dict[str, dict[str, RunArrival]] = {c: {} for c in COUNTINGS}
        self.record: dict[str, RunArrival] = {}
        self.times: dict[str, np.ndarray] = {}
        self.steps: dict[str, np.ndarray] = {}
        self.low: dict[str, float] = {}
        self.sensor_lat: dict[str, dict[str, np.ndarray]] = {}
        self.sensor_check: dict[str, bool] = {}
        for name, path in self.paths.items():
            config = self.configs[name]
            assert np.array_equal(
                np.array(config.class_thresholds("prompt 80")), self.thresholds
            )
            names, onsets = events_of(config)
            maps, times, low = probe80.frame_maps(
                path, self.thresholds, self.live, onsets
            )
            self.times[name], self.low[name] = times, low
            for c in COUNTINGS:
                self.frames[c][name] = RunArrival(
                    maps[c], onsets, FRAME_RESOLVED, names, float(times[-1])
                )
            step = arrival_from_run(path)
            assert step.resolution == STEP_RESOLVED
            assert np.array_equal(step.onsets, onsets) and step.event_names == names
            self.record[name] = step
            readings = RunFile(path).read_sensor_readings()
            self.steps[name] = readings.times
            self.sensor_lat[name] = {
                c: probe80.reading_latencies(
                    readings.readings, readings.times, self.thresholds, onsets, c
                )
                for c in COUNTINGS
            }
            cells = sensor_cells(config, self.mesh)
            rows, cols = zip(*cells, strict=True)
            at_cells = step.maps[:, :, list(rows), list(cols)] - onsets[:, None, None]
            self.sensor_check[name] = bool(
                np.array_equal(at_cells, self.sensor_lat[name][PER_CLASS])
            )
        self.clear = clear_units_of(
            self.paths, self.configs, self.mesh, self.thresholds
        )
        detection = {c: {n: self.frames[c][n] for n in EVENT_NAMES} for c in COUNTINGS}
        self.grid = Grid(
            "320x120",
            {n: self.paths[n] for n in EVENT_NAMES},
            [self.configs[n] for n in FINE_NAMES],
            detection,
            {n: self.record[n] for n in EVENT_NAMES},
        )


class Coarse:
    """80x30: both readings at every step from the probe's runs, and their frames."""

    def __init__(self) -> None:
        self.paths = {n: PLACEMENT_80 / f"{s}.h5" for n, s in COARSE_STEMS.items()}
        self.old = {n: PLACEMENT_79 / f"{s}.h5" for n, s in COARSE_STEMS.items()}
        self.configs = {n: config_of(p) for n, p in self.paths.items()}
        first = self.configs["door_leak"]
        self.mesh = Mesh(first)
        self.live = self.mesh.cell_type != SOLID
        self.thresholds = np.array(first.class_thresholds("prompt 80"))
        self.steps: dict[str, dict[str, RunArrival]] = {c: {} for c in COUNTINGS}
        self.frames: dict[str, dict[str, RunArrival]] = {c: {} for c in COUNTINGS}
        self.times: dict[str, np.ndarray] = {}
        self.step_times: dict[str, np.ndarray] = {}
        self.calibration: dict[str, dict[str, bool]] = {}
        self.low: dict[str, float] = {}
        for name, path in self.paths.items():
            config = self.configs[name]
            names, onsets = events_of(config)
            saved = np.load(path.with_name(f"{path.stem}.channels.npz"))
            assert str(saved["plant"]) == "", "a probe run was made under the plant"
            t_final = float(saved["t_final"])
            for c in COUNTINGS:
                maps = np.where(np.isnan(saved[c]), np.inf, saved[c])
                maps[:, :, ~self.live] = np.nan
                self.steps[c][name] = RunArrival(
                    maps, onsets, STEP_RESOLVED, names, t_final
                )
            fmaps, times, low = probe80.frame_maps(
                path, self.thresholds, self.live, onsets
            )
            self.times[name] = times
            self.low[name] = min(low, float(saved["low"]))
            for c in COUNTINGS:
                self.frames[c][name] = RunArrival(
                    fmaps[c], onsets, FRAME_RESOLVED, names, float(times[-1])
                )
            run, old = RunFile(path), RunFile(self.old[name])
            self.step_times[name] = run.read_sensor_readings().times
            self.calibration[name] = calibrate(run, old, saved, self.live)
            frames = run.read_frames()
            reference = arrival_from_frames(frames, self.thresholds, self.live, onsets)
            self.calibration[name]["per-class frame map = arrival_from_frames"] = (
                reference.tobytes() == fmaps[PER_CLASS].tobytes()
            )
        self.clear_check = {}
        areas = np.outer(self.mesh.dy_cell, self.mesh.dx_cell)
        for name, path in self.paths.items():
            end = scenario_end(self.configs[name])
            mine, stamps, share = probe80.frame_clear(
                path, self.thresholds, self.live, areas, end
            )
            theirs, kept, fractions = clear_times(
                RunFile(path).read_frames(), self.thresholds, self.live, areas, end
            )
            self.clear_check[name] = (
                mine[PER_CLASS].tobytes() == theirs.tobytes()
                and stamps.tobytes() == kept.tobytes(),
                float(np.abs(share[PER_CLASS] - fractions).max()),
            )
        configs = [*self.configs.values(), config_of(COARSE_CLEAN_OUT)]
        self.grid = Grid("80x30", self.paths, configs, self.steps, None)


def calibrate(
    run: RunFile, old: RunFile, saved: np.lib.npyio.NpzFile, live: np.ndarray
) -> dict[str, bool]:
    """The probe run against prompt 79's, bit for bit, and the probe against its tracker."""
    a, b = run.read_frames(), old.read_frames()
    frames = (
        a.steps.tobytes() == b.steps.tobytes()
        and a.times.tobytes() == b.times.tobytes()
        and all(a.fields[k].tobytes() == b.fields[k].tobytes() for k in b.fields)
        and sorted(a.fields) == sorted(b.fields)
    )
    with h5py.File(run.path, "r") as fa, h5py.File(old.path, "r") as fb:
        budgets = (
            fa["transport/budgets"][()].tobytes()
            == fb["transport/budgets"][()].tobytes()
        )
    ra, rb = run.read_sensor_readings(), old.read_sensor_readings()
    readings = (
        ra.readings.tobytes() == rb.readings.tobytes()
        and ra.times.tobytes() == rb.times.tobytes()
        and ra.steps.tobytes() == rb.steps.tobytes()
    )
    xa, xb = run.read_arrival(), old.read_arrival()
    arrival = (
        xa.first_above.tobytes() == xb.first_above.tobytes()
        and xa.onsets.tobytes() == xb.onsets.tobytes()
        and xa.t_final == xb.t_final
    )
    own = np.where(np.isnan(saved["own"]), np.inf, saved["own"])
    own[:, :, ~live] = np.nan
    return {
        "frames": frames,
        "budgets": budgets,
        "readings": readings,
        "/arrival/": arrival,
        "probe's per-class = its tracker's": saved["per_class"].tobytes()
        == saved["own"].tobytes(),
        "tracker's = /arrival/": own.tobytes() == xa.first_above.tobytes(),
    }


# -- Structural checks ------------------------------------------------------------


def structure_row(title: str, resolution: str, arrival: dict[str, RunArrival]) -> str:
    """One run's cumulative map against its per-class one, every cell."""
    p, q = arrival[PER_CLASS].maps, arrival[CUMULATIVE].maps
    live = ~np.isnan(p)
    assert np.array_equal(live, ~np.isnan(q))
    later = int((q[live] > p[live]).sum())
    earlier = int((q[live] < p[live]).sum())
    equal = int((q[live] == p[live]).sum())
    gained = int((np.isfinite(q) & np.isinf(p)).sum())
    largest = p[:, -1].tobytes() == q[:, -1].tobytes()
    return (
        f"| {title} | {resolution} | {int(live.sum()):,} | {earlier:,} | {gained:,} | "
        f"{equal:,} | {later} | {'yes' if largest else 'no'} |"
    )


def structure_table(fine: Fine, coarse: Coarse) -> list[str]:
    """S1: never later anywhere, the largest class identical."""
    lines = [
        "### S1. The cumulative maps against the per-class maps, every event, channel and non-SOLID cell (the stop condition and section 3's check)",
        "",
        "| Run | Grid, resolution | Entries | Cumulative earlier | of which reached only cumulatively | Equal | Cumulative later | 5 um channel identical |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in FINE_NAMES:
        arrival = {c: fine.frames[c][name] for c in COUNTINGS}
        lines.append(structure_row(name, "320x120, frames", arrival))
    for name in COARSE_STEMS:
        arrival = {c: coarse.steps[c][name] for c in COUNTINGS}
        lines.append(structure_row(name, "80x30, steps", arrival))
        arrival = {c: coarse.frames[c][name] for c in COUNTINGS}
        lines.append(structure_row(name, "80x30, frames", arrival))
    low = min([*fine.low.values(), *coarse.low.values()])
    lines += [
        "",
        f"The least concentration of any class in any live cell, over every frame of these runs "
        f"and every step of the 80x30 probe runs: {low:g} per cubic metre. A sum of "
        "non-negative terms is never below one of them in floating point, which is why the "
        "cumulative reading cannot be later.",
    ]
    return lines


def resolution_table(fine: Fine) -> list[str]:
    """S2: 320x120's per-class frame maps against the march's /arrival/."""
    lines = [
        "### S2. On 320x120, the per-class frame maps against the march's step maps (/arrival/), the limits ADR-015 D states",
        "",
        "| Run | Entries | Frame map before the step map | Reached at frames, never at steps | Reached at steps, never at frames | Largest lag where both reach (s) | Largest frame spacing (s) | File's four: monitor's readings = /arrival/ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in FINE_NAMES:
        f = fine.frames[PER_CLASS][name].maps
        s = fine.record[name].maps
        live = ~np.isnan(s)
        both = live & np.isfinite(f) & np.isfinite(s)
        lag = float((f[both] - s[both]).max()) if both.any() else 0.0
        lines.append(
            f"| {name} | {int(live.sum()):,} | {int((f[live] < s[live]).sum())} | "
            f"{int((np.isfinite(f) & np.isinf(s)).sum())} | "
            f"{int((np.isinf(f) & np.isfinite(s)).sum())} | {lag:.6f} | "
            f"{float(np.diff(fine.times[name]).max()):.6f} | "
            f"{'yes' if fine.sensor_check[name] else 'no'} |"
        )
    return lines


def calibration_table(coarse: Coarse) -> list[str]:
    """K0: the 80x30 probe runs against prompt 79's."""
    keys = list(next(iter(coarse.calibration.values())))
    lines = [
        "### K0. The 80x30 probe runs against prompt 79's runs, bit for bit (section 2's calibration)",
        "",
        "| Run | " + " | ".join(keys) + " |",
        "|---|" + "---|" * len(keys),
    ]
    for name, checks in coarse.calibration.items():
        lines.append(
            f"| {COARSE_STEMS[name]} | "
            + " | ".join("yes" if checks[k] else "no" for k in keys)
            + " |"
        )
    return lines


def reading_table(fine: Fine) -> list[str]:
    """C0: the channels, what each sums and its threshold."""
    sizes = fine.grid.sizes
    n = len(sizes)
    lines = [
        "### C0. The channels: what each sums under each reading, and the threshold it is compared with (the product file's ISO Class 5 values)",
        "",
        "| Channel | Per-class reading | Cumulative reading | Threshold (per cubic metre) |",
        "|---|---|---|---|",
    ]
    for k in range(n):
        cum = ", ".join(
            f"{sizes[j] * 1e6:g}" for j in sorted(probe80.members(k, n, CUMULATIVE))
        )
        lines.append(
            f"| >= {sizes[k] * 1e6:g} um | {sizes[k] * 1e6:g} um alone | the classes {cum} um | "
            f"{fine.thresholds[k]:,.6g} |"
        )
    return lines


# -- Reach and shift ---------------------------------------------------------------


def shift_table(fine: Fine, coarse: Coarse) -> list[str]:
    """D1: per run and channel, the cells reached and how much earlier the sum reaches them."""
    lines = [
        "### D1. Per run and channel, the cells each reading reaches and how much earlier the cumulative reading reaches the cells both reach (320x120 at frames, 80x30 at steps)",
        "",
        "| Grid | Run | Channel (um) | Cells reached, per-class | Cells reached, cumulative | Both reach, cumulative earlier | Median shift there (s) | Largest shift (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for grid, arrivals, names in (
        ("320x120", fine.frames, EVENT_NAMES),
        ("80x30", coarse.steps, COARSE_STEMS),
    ):
        sizes = fine.grid.sizes
        for name in names:
            p, q = arrivals[PER_CLASS][name].maps, arrivals[CUMULATIVE][name].maps
            for e in range(p.shape[0]):
                for k in range(p.shape[1]):
                    a, b = p[e, k], q[e, k]
                    both = np.isfinite(a) & np.isfinite(b)
                    shift = a[both] - b[both]
                    moved = shift[shift > 0]
                    lines.append(
                        f"| {grid} | {name} | {sizes[k] * 1e6:g} | {int(np.isfinite(a).sum()):,} | "
                        f"{int(np.isfinite(b).sum()):,} | {moved.size:,} | "
                        f"{float(np.median(moved)) if moved.size else 0.0:.3f} | "
                        f"{float(moved.max()) if moved.size else 0.0:.3f} |"
                    )
    return lines


# -- Scores ----------------------------------------------------------------------------


SCORE_HEAD = (
    "| Layout | Searched on | Rule | Cell centres (x, y), m | Per-class: worst (s), mean (s), "
    "undetected | Cumulative: worst (s), mean (s), undetected | Worst case earlier by (s) | "
    "Coverage within 10 s, per-class / cumulative |"
)
SCORE_RULE = "|---|---|---|---|---|---|---|---|"


def score_text(layout: Layout) -> str:
    """Worst, mean, undetected."""
    s = layout.score
    return f"{t3(s.worst)}, {s.mean:.3f}, {s.undetected}"


def scores_table(grid: Grid, bases: tuple[str, ...], tag: str, title: str) -> list[str]:
    """L1/K1: every layout scored on both readings' maps, over every unit."""
    lines = [f"### {tag}. {title}", "", SCORE_HEAD, SCORE_RULE]
    seen: set[tuple[str, tuple]] = set()
    basis_text = {
        "record": "the march's per-class maps (the layouts of record)",
        PER_CLASS: "the per-class maps scored here",
        CUMULATIVE: "the cumulative maps",
    }
    for basis in bases:
        for rule in RULES:
            for name, cells in grid.layouts(basis, rule).items():
                if name == FILE and (name, cells) in seen:
                    continue
                seen.add((name, cells))
                p, q = grid.scored(PER_CLASS, cells), grid.scored(CUMULATIVE, cells)
                gain = p.score.worst - q.score.worst
                gain_text = (
                    "-"
                    if math.isinf(p.score.worst) and math.isinf(q.score.worst)
                    else t3(gain)
                )
                lines.append(
                    f"| {name} | {'-' if name == FILE else basis_text[basis]} | "
                    f"{'-' if name == FILE else rule} | {places(grid.mesh, cells)} | "
                    f"{score_text(p)} | {score_text(q)} | {gain_text} | "
                    f"{p.score.coverage:.3f} / {q.score.coverage:.3f} |"
                )
    return lines


def search_table(grid: Grid, bases: tuple[str, ...], tag: str, title: str) -> list[str]:
    """L2/K2: the searches themselves, each over the units its candidates reach."""
    lines = [
        f"### {tag}. {title}",
        "",
        "| Pool | Candidates | Rule | Searched on | Cell centres (x, y), m | Method | Exact | Uncoverable units | Worst (s) | Mean (s) | Same cells as the per-class search |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in (SURF, STAND, FREE):
        for rule in RULES:
            reference = grid.found[(bases[0], rule)][name].cells
            for basis in bases:
                found = grid.found[(basis, rule)][name]
                gaps = (
                    ", ".join(sorted({TITLES[label[0]] for label in found.uncovered}))
                    or "none"
                )
                lines.append(
                    f"| {name} | {len(grid.pools[name]):,} | {rule} | {READING.get(basis, 'record')} | "
                    f"{places(grid.mesh, found.cells)} | {found.method} | "
                    f"{'yes' if found.exact else 'no'} | {gaps} | {t3(found.score.worst)} | "
                    f"{found.score.mean:.3f} | {'yes' if found.cells == reference else 'no'} |"
                )
    return lines


def door_table(fine: Fine) -> list[str]:
    """D2: surface candidates reaching a door unit at prompt 76's standoffs, both readings."""
    g = fine.grid
    spec = g.spec
    surfaces = tuple(m for m in spec.mount if m != MOUNT_STAND)
    configs = [fine.configs[n] for n in FINE_NAMES]
    rows = [u for u, label in enumerate(g.labels) if label[0] == "door_leak"]
    lines = [
        "### D2. On 320x120, the surface candidates (no stands) that reach a door unit at prompt 76's standoffs, under each reading's frame maps, and the seal leak's at the product's",
        "",
        "| Standoff (m) | Surface candidates | Reaching a door unit, per-class | Reaching a door unit, cumulative | Reaching a seal-leak unit, per-class | cumulative |",
        "|---|---|---|---|---|---|",
    ]
    seal = [u for u, label in enumerate(g.labels) if label[0] == "seal_leak"]
    for standoff in DOOR_STANDOFFS:
        pool = candidates(g.mesh, configs, surfaces, standoff)
        counts = []
        for which in (rows, seal):
            for c in COUNTINGS:
                reach = np.isfinite(g.units[c].latencies[which]).any(axis=0)
                counts.append(str(int(reach[pool[:, 0], pool[:, 1]].sum())))
        lines.append(f"| {standoff:g} | {len(pool):,} | " + " | ".join(counts) + " |")
    return lines


def orders(grid: Grid, basis_pc: str, basis_cum: str) -> dict[str, tuple[str, str]]:
    """Per rule: the four layouts' order under each reading (each searched on its own maps)."""
    out = {}
    for rule in RULES:
        p = {
            n: grid.scored(PER_CLASS, cells)
            for n, cells in grid.layouts(basis_pc, rule).items()
        }
        q = {
            n: grid.scored(CUMULATIVE, cells)
            for n, cells in grid.layouts(basis_cum, rule).items()
        }
        out[rule] = (order_text(p), order_text(q), rank_text(p), rank_text(q))
    return out


def order_table(fine: Fine, coarse: Coarse) -> list[str]:
    """O1: the four layouts' order, each reading's layouts on its own maps."""
    lines = [
        "### O1. The order of the four layouts, each searched and scored on one reading's maps, over every unit (prediction (c))",
        "",
        "| Grid | Search rule | Per-class, by the worst case | Cumulative, by the worst case | Same order | Per-class, section D's ranking (ties by the mean) | Cumulative, section D's ranking | Same ranking |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for grid, pc, cum in (
        (fine.grid, PER_CLASS, CUMULATIVE),
        (fine.grid, "record", CUMULATIVE),
        (coarse.grid, PER_CLASS, CUMULATIVE),
    ):
        for rule, (a, b, c, d) in orders(grid, pc, cum).items():
            title = grid.title + (
                "" if pc != "record" else ", per-class side the layouts of record"
            )
            lines.append(
                f"| {title} | {rule} | {a} | {b} | {'yes' if a == b else 'no'} | {c} | {d} | "
                f"{'yes' if c == d else 'no'} |"
            )
    return lines


# -- Brackets (320x120) ---------------------------------------------------------------


OPEN = "open"
EARLIER = "earlier"
SAME = "same"
NEVER = "never at a frame"
EXACT = "same, exact (5 um)"
READINGS_SAME = "same, exact (every-step readings)"
READINGS_EARLIER = "earlier, exact (every-step readings)"
VERDICTS = (EARLIER, SAME, EXACT, OPEN, NEVER, READINGS_SAME, READINGS_EARLIER)


class Bracket:
    """A cumulative latency's bracket at one cell, or the least over a layout's cells.

    ``low`` < T <= ``high`` (T >= ``low`` when ``closed``), seconds after the onset.
    ``crossed`` marks a cell where class k alone was at or above at a step and the sum
    below at a later frame before its first frame at or above, so the frames' lower
    edge is not a bound there and the onset is used.
    """

    def __init__(self, low: float, high: float, closed: bool, crossed: bool) -> None:
        self.low, self.high, self.closed, self.crossed = low, high, closed, crossed


def cell_bracket(
    frame_first: float, step_class: float, onset: float, times: np.ndarray
) -> Bracket:
    """The bracket on the cumulative arrival at one cell (absolute times in, latencies out)."""
    high = min(frame_first, step_class)
    before = times[times < frame_first] if math.isfinite(frame_first) else times
    crossed = False
    if math.isfinite(frame_first):
        prev = float(before.max()) if before.size else -math.inf
    else:
        prev = float(times.max())
    if prev < onset:
        low, closed = onset, True
    else:
        low, closed = prev, False
    if math.isfinite(step_class) and step_class <= low and not closed:
        crossed = True
        low, closed = onset, True
    return Bracket(low - onset, high - onset, closed, crossed)


def layout_bracket(parts: list[Bracket]) -> Bracket:
    """The least over a layout's cells of a latency each cell brackets."""
    low = min(b.low for b in parts)
    closed = any(b.closed and b.low == low for b in parts)
    return Bracket(
        low, min(b.high for b in parts), closed, any(b.crossed for b in parts)
    )


def verdict(b: Bracket, per_class: float, steps: np.ndarray) -> str:
    """Whether the bracket decides the cumulative latency against the per-class one."""
    if b.closed and b.low == b.high:
        return EXACT
    if math.isinf(b.high):
        return NEVER
    if b.high < per_class:
        return EARLIER
    inside = steps[(steps >= b.low) if b.closed else (steps > b.low)]
    inside = inside[inside <= b.high]
    return SAME if inside.size <= 1 else OPEN


def interval_text(b: Bracket) -> str:
    """A bracket as text."""
    if b.closed and b.low == b.high:
        return f"= {t3(b.low)}"
    if math.isinf(b.high):
        return f"never at a frame (after {b.low:.3f})"
    return f"{'[' if b.closed else '('}{b.low:.3f}, {b.high:.3f}]" + (
        " *" if b.crossed else ""
    )


class Brackets:
    """Per layout of record and unit (and per sensor), the cumulative latency's bracket."""

    def __init__(self, fine: Fine) -> None:
        g = fine.grid
        self.fine = fine
        self.layouts = g.layouts("record", WORST_CASE)
        self.cells: dict[str, list] = {}
        self.units: dict[str, list] = {}
        for name, cells in self.layouts.items():
            per_unit, per_cell = [], []
            for u, label in enumerate(g.labels):
                scenario, event, k = label
                e = fine.record[scenario].event_names.index(event)
                onset = float(fine.record[scenario].onsets[e])
                times = fine.times[scenario]
                parts = []
                for j, i in cells:
                    p = float(g.record_units.latencies[u, j, i])
                    if k == len(g.sizes) - 1:
                        # The largest channel is its class alone: the march's map is
                        # its cumulative map, exact at every step.
                        b = Bracket(p, p, True, False)
                    else:
                        b = cell_bracket(
                            float(fine.frames[CUMULATIVE][scenario].maps[e, k, j, i]),
                            float(fine.record[scenario].maps[e, k, j, i]),
                            onset,
                            times,
                        )
                    parts.append(b)
                    per_cell.append((label, (j, i), p, b))
                b = layout_bracket(parts)
                p = float(min(g.record_units.latencies[u, j, i] for j, i in cells))
                f = float(min(g.units[CUMULATIVE].latencies[u, j, i] for j, i in cells))
                pf = float(min(g.units[PER_CLASS].latencies[u, j, i] for j, i in cells))
                exact = None
                if name == FILE:
                    lat = fine.sensor_lat[scenario][CUMULATIVE][e, k]
                    exact = float(lat.min())
                v = verdict(b, p, fine.steps[scenario])
                if exact is not None:
                    v = READINGS_SAME if exact == p else READINGS_EARLIER
                per_unit.append((label, p, pf, f, b, v, exact))
            self.units[name] = per_unit
            self.cells[name] = per_cell

    def worst(self, name: str) -> tuple[float, float, float]:
        """The layout's worst case: per-class (steps), and the cumulative bracket's ends."""
        rows = self.units[name]
        return (
            max(r[1] for r in rows),
            max(r[4].low for r in rows),
            max(r[4].high for r in rows),
        )

    def worst_text(self, name: str) -> str:
        """The cumulative worst case's bracket as text, closed below where an exact unit sets it."""
        _, low, high = self.worst(name)
        if math.isinf(high):
            return "inf"
        closed = any(r[4].closed and r[4].low == low for r in self.units[name])
        return f"{'[' if closed else '('}{low:.3f}, {high:.3f}]"


def bracket_unit_table(br: Brackets) -> list[str]:
    """B1: per layout of record and unit."""
    sizes = br.fine.grid.sizes
    lines = [
        "### B1. On 320x120, per layout of record and unit, the cumulative latency's bracket against the march's per-class latency (s after the onset). The bracket is (last frame below, earlier of the first frame at or above and class k's own step arrival]; [ marks the onset as its lower end; * marks a cell where class k alone was at or above at a step before a frame at which the sum was below, so the onset is the lower end there. The 5 um channel is class 5 alone, so its cumulative latency is the march's (=). The file's four have every-step readings, so their cumulative latency is exact",
        "",
        "| Layout | Unit | Per-class, steps | Per-class, frames | Cumulative, frames | Cumulative bracket | Decided | Cumulative, steps (the file's four) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, rows in br.units.items():
        for label, p, pf, f, b, v, exact in rows:
            lines.append(
                f"| {name} | {label_text(sizes, label)} | {t3(p)} | {t3(pf)} | {t3(f)} | "
                f"{interval_text(b)} | {v} | {'-' if exact is None else t3(exact)} |"
            )
    return lines


def bracket_cell_table(br: Brackets) -> list[str]:
    """B3: per layout of record, sensor and unit."""
    sizes = br.fine.grid.sizes
    mesh = br.fine.mesh
    lines = [
        "### B3. On 320x120, per layout of record, sensor and unit, the cumulative latency's bracket at the sensor's cell (s after the onset), as B1",
        "",
        "| Layout | Sensor (x, y), m | Unit | Per-class, steps | Cumulative bracket |",
        "|---|---|---|---|---|",
    ]
    for name, rows in br.cells.items():
        for label, cell, p, b in sorted(
            rows, key=lambda r: (r[1], br.fine.grid.labels.index(r[0]))
        ):
            lines.append(
                f"| {name} | ({where(mesh, cell)[0]:.3f}, {where(mesh, cell)[1]:.3f}) | "
                f"{label_text(sizes, label)} | {t3(p)} | {interval_text(b)} |"
            )
    return lines


def bracket_summary_table(br: Brackets) -> list[str]:
    """B2: per layout of record, what the brackets decide."""
    lines = [
        "### B2. On 320x120, what the brackets decide, per layout of record: units whose cumulative latency is decided earlier than the per-class one (the bracket's top below it), decided the same (one step in the bracket), exact at 5 um (the channel is class 5 alone), left open, or never at a frame; for the file's four, exact from their every-step readings; and the worst case, the largest lower and upper ends over the units",
        "",
        "| Layout | Units | "
        + " | ".join(v[0].upper() + v[1:] for v in VERDICTS)
        + " | Cells marked * | Per-class worst, steps (s) | Cumulative worst, bracket (s) |",
        "|---|---|" + "---|" * len(VERDICTS) + "---|---|---|",
    ]
    for name, rows in br.units.items():
        counts = {v: sum(r[5] == v for r in rows) for v in VERDICTS}
        crossed = sum(
            b.crossed for _, _, b in [(c[0], c[1], c[3]) for c in br.cells[name]]
        )
        p, _, _ = br.worst(name)
        worst = br.worst_text(name)
        lines.append(
            f"| {name} | {len(rows)} | "
            + " | ".join(str(counts[v]) for v in VERDICTS)
            + f" | {crossed} | {t3(p)} | {worst} |"
        )
    open_units = [
        (name, label_text(br.fine.grid.sizes, r[0]), r[4].high - r[4].low)
        for name, rows in br.units.items()
        for r in rows
        if r[5] == OPEN
    ]
    if open_units:
        widest = max(open_units, key=lambda x: x[2])
        lines += [
            "",
            f"The open units' brackets are at most {widest[2]:.3f} s wide ({widest[0]}, "
            f"{widest[1]}).",
        ]
    return lines


# -- The bracket tested on 80x30 -------------------------------------------------------


def bracket_test_table(coarse: Coarse) -> list[str]:
    """K4: the 320x120 bracket method, applied to 80x30's frames, against its step truth."""
    lines = [
        "### K4. The bracket's method tested on 80x30, where the cumulative arrival is known at every step: each event, channel and non-SOLID cell reached at steps, the bracket built from the frames and class k's step map as B1 builds it",
        "",
        "| Run | Cells reached cumulatively at steps | Truth inside the bracket | Truth below its lower end (an excursion between frames) | Truth above its top | Marked * | Never at a frame, reached at steps |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in COARSE_STEMS:
        truth = coarse.steps[CUMULATIVE][name]
        frames = coarse.frames[CUMULATIVE][name].maps
        per_class = coarse.steps[PER_CLASS][name].maps
        times = coarse.times[name]
        inside = below = above = crossed = never = reached = 0
        for e, onset in enumerate(truth.onsets):
            t = truth.maps[e]
            idx = np.argwhere(np.isfinite(t))
            for k, j, i in idx:
                reached += 1
                b = cell_bracket(
                    float(frames[e, k, j, i]),
                    float(per_class[e, k, j, i]),
                    float(onset),
                    times,
                )
                lat = float(t[k, j, i]) - float(onset)
                crossed += b.crossed
                if math.isinf(b.high) and math.isinf(float(per_class[e, k, j, i])):
                    never += 1
                    continue
                lower_ok = lat >= b.low if b.closed else lat > b.low
                if lat > b.high:
                    above += 1
                elif not lower_ok:
                    below += 1
                else:
                    inside += 1
        lines.append(
            f"| {COARSE_STEMS[name]} | {reached:,} | {inside:,} | {below:,} | {above:,} | "
            f"{crossed:,} | {never:,} |"
        )
    return lines


# -- 80x30 per unit (prediction (d)) ------------------------------------------------------


def unit_fall_table(coarse: Coarse) -> list[str]:
    """K3: per unit at the searched layouts, per-class against cumulative, at steps."""
    g = coarse.grid
    lines = [
        "### K3. On 80x30, at steps, per unit: the latency at the product layout and the free optimum (each searched on the per-class maps, the worst-case rule) under each reading, and the fall (prediction (d))",
        "",
        "| Unit | Product layout: per-class (s) | cumulative (s) | fall | Free optimum: per-class (s) | cumulative (s) | fall |",
        "|---|---|---|---|---|---|---|",
    ]
    layouts = g.layouts(PER_CLASS, WORST_CASE)
    for u, label in enumerate(g.labels):
        cols = []
        for name in (STAND, FREE):
            cells = layouts[name]
            p = float(min(g.units[PER_CLASS].latencies[u, j, i] for j, i in cells))
            q = float(min(g.units[CUMULATIVE].latencies[u, j, i] for j, i in cells))
            fall = "-" if math.isinf(p) or p == 0 else f"{(p - q) / p:.1%}"
            cols += [t3(p), t3(q), fall]
        lines.append(f"| {label_text(g.sizes, label)} | " + " | ".join(cols) + " |")
    return lines


def largest_fall(coarse: Coarse, name: str) -> tuple[float, str]:
    """The largest relative fall over the units at a searched layout (per-class, worst case)."""
    g = coarse.grid
    cells = g.layouts(PER_CLASS, WORST_CASE)[name]
    best, which = 0.0, "none"
    for u, label in enumerate(g.labels):
        p = float(min(g.units[PER_CLASS].latencies[u, j, i] for j, i in cells))
        q = float(min(g.units[CUMULATIVE].latencies[u, j, i] for j, i in cells))
        if math.isfinite(p) and p > 0 and (p - q) / p > best:
            best, which = (p - q) / p, label_text(g.sizes, label)
    return best, which


# -- Predictions -----------------------------------------------------------------------


def predictions_table(fine: Fine, coarse: Coarse, br: Brackets) -> list[str]:
    """P1: the four predictions against the measurements."""
    rows = []
    # (a)
    detected = {}
    for title, grid in (("320x120", fine.grid), ("80x30", coarse.grid)):
        layout = grid.scored(CUMULATIVE, grid.sensors)
        detected[title] = len(grid.labels) - layout.score.undetected
    exact = sum(
        int(np.isfinite(fine.sensor_lat[n][CUMULATIVE]).any(axis=-1).sum())
        for n in EVENT_NAMES
    )
    holds_a = all(v == 0 for v in detected.values()) and exact == 0
    rows.append(
        (
            "(a)",
            "The file's four detect no unit on either grid",
            f"cumulative maps: {detected['320x120']} of {len(fine.grid.labels)} units on 320x120 "
            f"(frames), {detected['80x30']} of {len(coarse.grid.labels)} on 80x30 (steps); the "
            f"320x120 sensors' every-step readings summed: {exact} units",
            "holds" if holds_a else "fails",
        )
    )
    # (b)
    p_worst, low, high = br.worst(STAND)
    cells = br.layouts[STAND]
    pf = fine.grid.scored(PER_CLASS, cells).score.worst
    qf = fine.grid.scored(CUMULATIVE, cells).score.worst
    if low >= p_worst - FRAME_PREDICTION_B:
        b_text = "holds"
    elif high < p_worst - FRAME_PREDICTION_B:
        b_text = "fails"
    else:
        b_text = "open (the bracket straddles the limit)"
    rows.append(
        (
            "(b)",
            "On 320x120 the product layout's worst case is unchanged or earlier by at most one frame (0.36 s)",
            f"per-class at steps {t3(p_worst)} s; cumulative bracket {br.worst_text(STAND)} s; "
            f"at frames per-class {t3(pf)} s, cumulative {t3(qf)} s",
            b_text,
        )
    )
    # (c)
    parts, same = [], True
    for title, grid, pc in (
        ("320x120", fine.grid, PER_CLASS),
        ("320x120 against the record", fine.grid, "record"),
        ("80x30", coarse.grid, PER_CLASS),
    ):
        o = orders(grid, pc, CUMULATIVE)[WORST_CASE]
        parts.append(f"{title}: {'same' if o[0] == o[1] else 'differs'}")
        same &= o[0] == o[1]
    rows.append(
        (
            "(c)",
            "The order of the four layouts by worst case is unchanged on both grids",
            "; ".join(parts) + " (O1, the worst-case rule's searches)",
            "holds" if same else "fails",
        )
    )
    # (d)
    fall, which = largest_fall(coarse, STAND)
    rows.append(
        (
            "(d)",
            "On 80x30, at steps, at least one unit's latency at the searched layout falls by more than 10%",
            f"largest fall at the product layout {fall:.1%} ({which}); K3",
            "holds" if fall > FALL_D else "fails",
        )
    )
    lines = [
        "### P1. The predictions against the measurements",
        "",
        "| Prediction | As committed | Measured | Verdict |",
        "|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    return lines


def record_table(fine: Fine) -> list[str]:
    """R1: the step-resolved searches here against prompt 79's layouts of record."""
    g = fine.grid
    lines = [
        "### R1. The layouts of record: the searches on the march's per-class maps (the worst-case rule) against prompt 79's re-score (rescored76.md L1)",
        "",
        "| Layout | Searched here | Prompt 79 | Same |",
        "|---|---|---|---|",
    ]
    for name, text in RECORD_PLACES.items():
        here = places(g.mesh, g.found[("record", WORST_CASE)][name].cells)
        lines.append(
            f"| {name} | {here} | {text} | {'yes' if here == text else 'no'} |"
        )
    return lines


def clear_check_table(coarse: Coarse) -> list[str]:
    """A0: the streamed clear times against placement.clear_times on 80x30's frames."""
    lines = [
        "### A0. The streamed clear times (probe80.frame_clear, per-class) against placement.clear_times on the 80x30 probe runs' frames",
        "",
        "| Run | Clear times and frame times bitwise equal | Largest difference in the area fractions |",
        "|---|---|---|",
    ]
    for name, (same, diff) in coarse.clear_check.items():
        lines.append(
            f"| {COARSE_STEMS[name]} | {'yes' if same else 'no'} | {diff:.1e} |"
        )
    return lines


def clear_text(value: float) -> str:
    """A clear time, or not clear."""
    return "not clear" if math.isinf(value) else f"{value:.2f}"


def interval_of(value: float) -> str:
    """A premature interval, or a dash where either side is not clear."""
    return "-" if math.isnan(value) else f"{value:.2f}"


def early_count(r: ClearScore) -> int:
    """Units a layout declares clean before the room is (or while the room never is)."""
    cleared = np.isfinite(r.layout)
    gap = r.room - np.where(cleared, r.layout, 0.0)
    return int((cleared & (np.isinf(r.room) | (gap > 0))).sum())


def c2_line(units: ClearUnits, p: ClearScore, sizes: list[float]) -> str:
    """The per-class columns of A1 against prompt 79's C2 rows for the product layout."""
    text = RESCORED76.read_text(encoding="utf-8")
    block = text.split(C2_PRODUCT)[-1].split("**the three-plus-one")[0]
    rows = {}
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 6 and cells[0] in FINE_NAMES:
            rows[(cells[0], cells[1])] = cells[2:5]
    same = 0
    for u, (name, k) in enumerate(units.labels):
        interval = interval_of(p.interval[u])
        if interval == "-" and math.isinf(p.room[u]):
            interval = "room not clear"
        mine = [clear_text(p.layout[u]), clear_text(p.room[u]), interval]
        size = f"{sizes[k] * 1e6:g}"
        same += rows.get((name, size)) == mine
    return (
        f"The per-class columns against rescored76.md C2's rows for the product layout (the "
        f"layout's clear time, the room's and the interval, as printed there): {same} of "
        f"{len(units.labels)} units equal."
    )


def clear_table(fine: Fine) -> list[str]:
    """A1/A2: the all-clear reading of the layouts of record under both readings."""
    g = fine.grid
    layouts = g.layouts("record", WORST_CASE)
    scores: dict[str, dict[str, ClearScore]] = {
        name: {c: clear_score(fine.clear[c], cells) for c in COUNTINGS}
        for name, cells in layouts.items()
    }
    units = fine.clear[PER_CLASS]
    lines = [
        "### A1. On 320x120, per all-clear unit (scenario, channel): the room's clear time and the product layout's under each reading, and the premature interval (the room's less the layout's), frame-resolved (s; not clear is still at or above at the run's end)",
        "",
        "| Unit | t_e (s) | Room, per-class | Room, cumulative | Product layout, per-class | Product layout, cumulative | Interval, per-class | Interval, cumulative |",
        "|---|---|---|---|---|---|---|---|",
    ]
    p, q = scores[STAND][PER_CLASS], scores[STAND][CUMULATIVE]
    for u, (name, k) in enumerate(units.labels):
        lines.append(
            f"| {TITLES[name]}, {g.sizes[k] * 1e6:g} um | {units.ends[u]:g} | "
            f"{clear_text(p.room[u])} | {clear_text(q.room[u])} | {clear_text(p.layout[u])} | "
            f"{clear_text(q.layout[u])} | {interval_of(p.interval[u])} | "
            f"{interval_of(q.interval[u])} |"
        )
    lines += ["", c2_line(units, p, g.sizes)]
    lines += [
        "",
        "### A2. On 320x120, each layout of record's all-clear reading under each reading: the units it declares clean before the room is (interval above zero, or the room not clear), the worst and mean ranking keys and the largest live-area fraction at or above when it declares clean",
        "",
        "| Layout | Reading | Units declared clean early | Worst key | Mean key | Largest area fraction |",
        "|---|---|---|---|---|---|",
    ]
    for name, by in scores.items():
        for c in COUNTINGS:
            r = by[c]
            worst = "unbounded" if math.isinf(r.worst) else f"{r.worst:.2f}"
            mean = "unbounded" if math.isinf(r.mean) else f"{r.mean:.2f}"
            lines.append(
                f"| {name} | {READING[c]} | {early_count(r)} of {len(r.layout)} | {worst} | "
                f"{mean} | {r.worst_fraction:.2%} |"
            )
    return lines


def test_table() -> list[str]:
    """T1: section 3's check, as recorded by pytest, real and planted."""
    runs = {}
    for title, path in (("real", TEST_XML), ("planted", TEST_PLANT_XML)):
        cases = {}
        for case in ET.parse(path).getroot().iter("testcase"):
            if case.find("failure") is not None:
                cases[case.get("name")] = "fails"
            elif case.find("skipped") is not None:
                cases[case.get("name")] = "skipped"
            else:
                cases[case.get("name")] = "passes"
        runs[title] = cases
    lines = [
        "### T1. Section 3's check (docs/reports/probe80/test_probe80.py), as pytest recorded it in results/builder80/test80.xml and, under PROBE80_PLANT=smaller (the channel summed over class k and the smaller classes), test80_plant.xml",
        "",
        "| Case | On the real maps | Under the plant |",
        "|---|---|---|",
    ]
    for name, outcome in runs["real"].items():
        lines.append(f"| {name} | {outcome} | {runs['planted'].get(name, 'absent')} |")
    caught = sum(v == "fails" for v in runs["planted"].values())
    lines += [
        "",
        f"Under the plant {caught} of {len(runs['planted'])} cases fail.",
    ]
    return lines


# -- Render, splice, check ---------------------------------------------------------------


def render(fine: Fine, coarse: Coarse) -> dict[str, list[str]]:
    """Every table, by marker name."""
    br = Brackets(fine)
    return {
        "channels": reading_table(fine),
        "calibration": calibration_table(coarse),
        "structure": structure_table(fine, coarse),
        "resolution": resolution_table(fine),
        "record": record_table(fine),
        "shift": shift_table(fine, coarse),
        "fine_scores": scores_table(
            fine.grid,
            ("record", PER_CLASS, CUMULATIVE),
            "L1",
            "On 320x120, every layout scored on both readings' frame-resolved maps over the twenty units (four scenarios, five channels); the means capped at each unit's remaining run time",
        ),
        "fine_search": search_table(
            fine.grid,
            (PER_CLASS, CUMULATIVE, "record"),
            "L2",
            "On 320x120, the searches on each reading's frame-resolved maps and on the march's per-class maps (record), each over the units its candidates reach",
        ),
        "orders": order_table(fine, coarse),
        "door": door_table(fine),
        "bracket_summary": bracket_summary_table(br),
        "bracket_units": bracket_unit_table(br),
        "bracket_cells": bracket_cell_table(br),
        "coarse_scores": scores_table(
            coarse.grid,
            (PER_CLASS, CUMULATIVE),
            "K1",
            "On 80x30, every layout scored on both readings' step-resolved maps over the fifteen units (three scenarios, five channels)",
        ),
        "coarse_search": search_table(
            coarse.grid,
            (PER_CLASS, CUMULATIVE),
            "K2",
            "On 80x30, the searches on each reading's step-resolved maps, each over the units its candidates reach",
        ),
        "coarse_units": unit_fall_table(coarse),
        "bracket_test": bracket_test_table(coarse),
        "predictions": predictions_table(fine, coarse, br),
        "test": test_table(),
        "clear_check": clear_check_table(coarse),
        "clear": clear_table(fine),
    }


REPORT_MARKERS = (
    "channels",
    "calibration",
    "structure",
    "record",
    "fine_scores",
    "orders",
    "door",
    "bracket_summary",
    "coarse_scores",
    "coarse_units",
    "bracket_test",
    "predictions",
    "test",
    "clear_check",
    "clear",
)


def tables_text(tables: dict[str, list[str]]) -> str:
    """tables80.md."""
    head = [
        "# Prompt 80's tables: cumulative counts",
        "",
        "Generated by `docs/reports/probe80/tables80.py` from the run files under "
        "`results/builder79/` and `results/builder80/`; `--check` regenerates and compares, "
        "the copies in `docs/reports/cumulative_counts.md` included. Do not edit by hand.",
        "",
    ]
    body = []
    for lines in tables.values():
        body += [*lines, ""]
    return "\n".join(head + body)


def splice(text: str, tables: dict[str, list[str]]) -> str:
    """The report with each marked region replaced by its table."""
    for name in REPORT_MARKERS:
        begin, end = f"<!-- tables80 {name} begin -->", f"<!-- tables80 {name} end -->"
        if begin not in text:
            continue
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = head + begin + "\n" + "\n".join(tables[name]) + "\n" + end + tail
    return text


def main() -> int:
    """Write or check tables80.md and the report's copies."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="regenerate and compare")
    args = parser.parse_args()
    fine, coarse = Fine(), Coarse()
    tables = render(fine, coarse)
    text = tables_text(tables)
    report = REPORT.read_text(encoding="utf-8")
    spliced = splice(report, tables)
    if args.check:
        ok = TABLES.read_text(encoding="utf-8") == text and report == spliced
        print("tables80: " + ("match" if ok else "DIFFER"))
        return 0 if ok else 1
    TABLES.write_text(text, encoding="utf-8", newline="\n")
    REPORT.write_text(spliced, encoding="utf-8", newline="\n")
    print(f"wrote {TABLES.relative_to(ROOT)} and spliced {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
